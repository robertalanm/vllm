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
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.model_executor.config.configuration_btlm import BTLMConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs



KVCache = Tuple[torch.Tensor, torch.Tensor]

class SwiGLUActivation(nn.Module):
    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 * nn.functional.silu(x2)


class AlibiPositionEmbeddingLayer(nn.Module):
    def __init__(self, num_heads):
        super(AlibiPositionEmbeddingLayer, self).__init__()

        self.num_heads = num_heads
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        rank = get_tensor_model_parallel_rank()

        slopes = torch.tensor(AlibiPositionEmbeddingLayer._get_alibi_slopes(num_heads)).unsqueeze(-1)
        
        # Shard the slopes tensor across the tensor model parallel group
        my_num_heads = (num_heads // tensor_model_parallel_world_size) + int(rank < (num_heads % tensor_model_parallel_world_size))
        self.slopes = VocabParallelEmbedding(my_num_heads, 1, init_tensor=slopes)

    def forward(
        self,
        seq_length,
        key_length,
        cached_qk_len,
    ):
        context_position = torch.arange(
            cached_qk_len, cached_qk_len + seq_length, device=self.slopes.device
        )[:, None]
        memory_position = torch.arange(
            key_length + cached_qk_len, device=self.slopes.device
        )[None, :]
        relative_position = memory_position - context_position
        relative_position = torch.abs(relative_position).unsqueeze(0).expand(self.num_heads, -1, -1)
        alibi = (self.slopes * -1.0).unsqueeze(1) * relative_position
        return alibi

    @staticmethod
    def _get_alibi_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + AlibiPositionEmbeddingLayer._get_alibi_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

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
        # self.relative_pe = AlibiPositionEmbeddingLayer(config.num_attention_heads)
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

        if self.relative_pe is not None:
            length = input_ids.shape[1]
            cached_kv_length = 0
            cached_kv = past_key_values[0]
            if cached_kv is not None:
                cached_kv_length = cached_kv[0].shape[-2]
            position_bias = self.relative_pe(length, length, cached_kv_length)
        else:
            position_bias = None

        past_length = 0
        past_key_values = tuple([None] * len(self.h))

        # for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

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