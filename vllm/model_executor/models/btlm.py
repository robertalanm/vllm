# coding=utf-8
# Copyright 2023 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2023 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BTLM model."""

import math
import os
import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_btlm import BTLMConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "cerebras/btlm-3b-8k-base"
_CONFIG_FOR_DOC = "BTLMConfig"

BTLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "cerebras/btlm-3b-8k-base",
    # See all BTLM models at https://huggingface.co/models?filter=btlm
]


class SwiGLUActivation(nn.Module):
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 * nn.functional.silu(x2)

class BTLMAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.total_num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.total_num_heads

        self.qkv_proj = ColumnParallelLinear(config.hidden_size,
                                             3 * config.hidden_size,
                                             bias=False,
                                             gather_output=False,
                                             perform_initialization=False)
        self.out_proj = RowParallelLinear(config.hidden_size,
                                          config.hidden_size,
                                          bias=False,
                                          input_is_parallel=True,
                                          perform_initialization=False)

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        scaling = self.head_size**-0.5
        assert getattr(config, "rotary", True)
        assert config.rotary_dim % 2 == 0
        self.attn = PagedAttentionWithALiBi(self.num_heads, self.head_size,
                                           scaling, config.rotary_dim)
        self.warmup = False

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(position_ids, q, k, v, k_cache, v_cache,
                                input_metadata, cache_event)
        attn_output, _ = self.out_proj(attn_output)
        return attn_output



class BTLMMLP(nn.Module):

    def __init__(self, intermediate_size: int, config: BTLMConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.fc_in = ColumnParallelLinear(hidden_size,
                                          intermediate_size,
                                          gather_output=False,
                                          perform_initialization=False)
        self.fc_out = RowParallelLinear(intermediate_size,
                                        hidden_size,
                                        input_is_parallel=True,
                                        perform_initialization=False)
        self.act = get_act_fn(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.fc_out(hidden_states)
        return hidden_states


class BTLMBlock(nn.Module):

    def __init__(self, config: BTLMConfig):
        super().__init__()
        if config.n_inner is None:
            inner_dim = 4 * config.hidden_size
        else:
            inner_dim = config.n_inner
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = BTLMAttention(config)
        self.mlp = BTLMMLP(inner_dim, config)

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            input_metadata=input_metadata,
            cache_event=cache_event,
        )
        mlp_output = self.mlp(hidden_states)
        hidden_states = attn_output + mlp_output + residual
        return hidden_states

class BTLMModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = VocabParallelEmbedding(config.vocab_size, self.embed_dim, perform_initialization=False)
        self.h = nn.ModuleList([BTLMBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.h)):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]
            layer = self.h[i]
            hidden_states = layer(
                position_ids,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
            )
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

class BTLMForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = BTLMModel(config)
        self.lm_head = ColumnParallelLinear(config.n_embd,
                                            config.vocab_size,
                                            gather_output=False,
                                            perform_initialization=False)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.transformer(input_ids, positions, kv_caches,
                                        input_metadata, cache_events)
        next_tokens = self.sampler(self.lm_head.weight, hidden_states,
                                input_metadata, self.lm_head.bias)
        return next_tokens

    _column_parallel_weights = [
        "wte.weight", "fc_in.weight", "fc_in.bias", "lm_head.weight",
        "lm_head.bias"
    ]
    _row_parallel_weights = ["out_proj.weight", "fc_out.weight"]

    def load_weights(self,
                    model_name_or_path: str,
                    cache_dir: Optional[str] = None,
                    use_np_cache: bool = False):
        tp_rank = get_tensor_model_parallel_rank()
        state_dict = self.state_dict()
        for name, loaded_weight in hf_model_weights_iterator(
                model_name_or_path, cache_dir, use_np_cache):
            if "attn.bias" in name or "attn.masked_bias" in name:
                continue

            is_attention_weight = False
            for stride_id, att_weight_name in enumerate(
                ["q_proj", "k_proj", "v_proj"]):
                if att_weight_name not in name:
                    continue
                param = state_dict[name.replace(att_weight_name, "qkv_proj")]
                shard_size = param.shape[1]
                loaded_weight = loaded_weight[shard_size * tp_rank:shard_size *
                                            (tp_rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size *
                                        (stride_id + 1)]
                assert param_slice.shape == loaded_weight.shape
                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break
            if is_attention_weight:
                continue

            param = state_dict[name]
            load_tensor_parallel_weights(param, loaded_weight, name,
                                        self._column_parallel_weights,
                                        self._row_parallel_weights, tp_rank)