# Copyright 2026 Google LLC
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
"""Qwen4Exp (Qwen3.8 Flash-Next) for the TPU vLLM/torchax path.

First version: correct, not fast. Text only, no MTP.

* GDN layers reuse vLLM's QwenGatedDeltaNetAttention (TPU override in
  gdn_attention_op.py). Their conv state carries extra rows; the PLE layer
  keeps its short-conv state in those rows of its own decoder layer, since the
  TPU KV cache manager supports one mamba state shape per model.
* MoE reuses Qwen3NextSparseMoeBlock (as Qwen3.5-MoE does on TPU).
* QSA layers own a regular paged attention cache with one extra KV head that
  holds the raw indexer keys; selection and sparse GQA run in plain JAX
  (custom_ops/experimental/qwen4_exp/qsa.py).
* Hyper-connections and the PLE gate are plain torch ops.
* PLE: the runner hashes n-gram ids from each request's token history and
  gathers the rows from a host copy of the n-gram table before each step
  (MambaMetadata.ple_embeddings); the table never goes to HBM.

Math follows transformers' modeling_qwen4_exp.py (Apache 2.0, The Qwen Team
and The HuggingFace Inc. team).
"""

import math
import os
import types
from collections.abc import Iterable

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchax.interop import jax_view, torch_view
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import \
    QwenGatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator, MambaStateDtypeCalculator,
    MambaStateShapeCalculator)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.models.interfaces import (HasInnerState, IsHybrid,
                                                   SupportsMRoPE)
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
from vllm.model_executor.models.utils import (AutoWeightsLoader, WeightsMapper,
                                              extract_layer_index,
                                              make_layers, maybe_prefix)
from vllm.sequence import IntermediateTensors

from tpu_inference.layers.vllm.custom_ops.experimental.qwen4_exp.qsa import (
    ple_short_conv, qsa_attention)
from tpu_inference.models.vllm.vllm_model_wrapper_context import \
    get_vllm_model_wrapper_context

# Host copies of the PLE n-gram tables, in PLE layer order. The runner reads
# them (same process) to gather embeddings before each step.
HOST_NGRAM_TABLES: list = []

# Steps with at most this many (padded) tokens use the sparse-gather QSA path
# (default: all steps); larger steps use the dense masked reference path.
_QSA_SPARSE_MAX_TOKENS = int(os.environ.get("QWEN4EXP_QSA_SPARSE_MAX_TOKENS",
                                            str(1 << 30)))

# HF configs name QSA layers "full_attention" (raw) or "indexed_attention".
_QSA_LAYER_TYPES = ("full_attention", "indexed_attention",
                    "qwen_sparse_attention")


def ple_extra_conv_rows(text_config) -> int:
    """Rows the PLE short-conv state needs inside a GDN conv state slot."""
    if not getattr(text_config, "ple_layer_ids", None):
        return 0
    hc_hidden = text_config.hidden_size * text_config.hc_count
    state_len = (text_config.ple_conv_kernel_size - 1) * text_config.ngram_size
    conv_dim = (text_config.linear_key_head_dim *
                text_config.linear_num_key_heads * 2 +
                text_config.linear_value_head_dim *
                text_config.linear_num_value_heads)
    return -(-state_len * hc_hidden // conv_dim)


def _attn_metadata_for(layer_name: str):
    md = get_forward_context().attn_metadata
    if isinstance(md, dict):
        return md.get(layer_name, next(iter(md.values())))
    return md


class Qwen4ExpRMSNorm(nn.Module):
    """(1 + w) RMSNorm, optionally per group of `group_size` features."""

    def __init__(self, dim: int, eps: float, group_size: int | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        if self.group_size is not None:
            xf = xf.reshape(*xf.shape[:-1], -1, self.group_size)
        out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            out = out.flatten(-2)
        return (out * (1.0 + self.weight.float())).type_as(x)


class Qwen4ExpGatedResidual(nn.Module):
    """4-stream hyper-connection: mix streams into one block input, then
    inject the block output back into every stream."""

    def __init__(self, config, use_combine: bool = True):
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = Qwen4ExpRMSNorm(hc_hidden,
                                       config.rms_norm_eps,
                                       group_size=self.hidden_size)
        self.input_mix_weight_down = nn.Linear(hc_hidden,
                                               config.hc_lowrank,
                                               bias=False)
        self.input_mix_weight_up = nn.Linear(config.hc_lowrank,
                                             hc_hidden,
                                             bias=False)
        self.block_inject_weight = (nn.Linear(
            hc_hidden, self.hc_count, bias=False) if use_combine else None)

    def mix(self, x: torch.Tensor):
        xn = self.hc_norm(x)
        w = F.silu(self.input_mix_weight_down(xn) / self.hc_count)
        w = torch.sigmoid(self.input_mix_weight_up(w))
        w = w.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (w * xn.unflatten(-1, (self.hc_count, self.hidden_size))).mean(
            dim=-2)
        if self.block_inject_weight is None:
            return mixed, None
        inject = 2 * torch.sigmoid(
            self.block_inject_weight(xn) / self.hc_count)
        return mixed, inject

    @staticmethod
    def combine(x: torch.Tensor, block_out: torch.Tensor,
                inject: torch.Tensor) -> torch.Tensor:
        return x + (block_out.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)


class Qwen4ExpNGramTable(nn.Module):
    """The PLE n-gram table, kept in host memory.

    The full table (about 51B params for Flash-Next) does not belong in HBM.
    Weights load into a host numpy array that the runner gathers rows from
    before each step (as vLLM GPU does from pinned host memory); the device
    sees only the gathered (tokens, ple_embed_dim) embeddings. Checkpoints
    split the table into `split_ngram_parts` row shards and carry the hash
    layout as buffers, which must match the layout the runner hashes with.
    """

    def __init__(self, layout, dim: int, split_parts: int):
        super().__init__()
        self.layout = layout
        self.split_parts = split_parts
        self.dim = dim
        # Allocated on the first loaded shard, in the checkpoint's dtype
        # (bf16, or fp8 e4m3 with a per-tensor scale for ModelOpt FP8).
        self.host_table: np.ndarray | None = None
        self.scale = 1.0
        HOST_NGRAM_TABLES.append(self)

    @staticmethod
    def _to_host(w: torch.Tensor) -> np.ndarray:
        import ml_dtypes
        w = w.detach().to("cpu").contiguous()
        if w.dtype == torch.float8_e4m3fn:
            return w.view(torch.uint8).numpy().view(ml_dtypes.float8_e4m3fn)
        w = w.to(torch.bfloat16)
        return w.view(torch.int16).numpy().view(ml_dtypes.bfloat16)

    def _table_for(self, w: np.ndarray) -> np.ndarray:
        if self.host_table is None:
            self.host_table = np.empty(
                (self.layout.padded_vocab_size, self.dim), dtype=w.dtype)
        return self.host_table

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """(n, heads) int32 rows -> (n, heads * dim) bf16 embeddings."""
        import ml_dtypes
        rows = self.host_table[ids].reshape(ids.shape[0], -1)
        if rows.dtype != ml_dtypes.bfloat16 or self.scale != 1.0:
            rows = (rows.astype(np.float32) * self.scale).astype(
                ml_dtypes.bfloat16)
        return rows

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        expected = {
            "layer_multipliers": self.layout.multipliers,
            "ngram_heads_offsets": self.layout.head_offsets,
            "ngram_heads_vocab_sizes": self.layout.head_vocab_sizes,
        }
        rows = self.layout.padded_vocab_size
        shard_rows = -(-rows // self.split_parts)
        for name, w in weights:
            if name in expected:
                if not np.array_equal(w.cpu().numpy().astype(np.int64),
                                      expected[name]):
                    raise ValueError(
                        f"PLE hash buffer {name} differs from the host layout")
                continue
            if name == "ngram_embedding.weight_scale":
                if w.numel() != 1:
                    raise NotImplementedError(
                        "PLE table scale must be per-tensor")
                self.scale = float(w.float().item())
                continue
            prefix = "ngram_embedding.shard_"
            if name.startswith(prefix) and name.endswith(".weight"):
                start = int(name[len(prefix):-len(".weight")]) * shard_rows
                h = self._to_host(w)
                self._table_for(h)[start:start + h.shape[0]] = h
            elif name == "ngram_embedding.weight":
                h = self._to_host(w)
                self._table_for(h)[:] = h
            else:
                raise ValueError(f"unexpected PLE table weight {name}")
        # Nothing is registered as a torch parameter; report nothing loaded so
        # the loader does not look for one.
        return set()


class Qwen4ExpPLE(nn.Module):
    """Per-layer n-gram embedding injected into every HC stream."""

    def __init__(self, config, gdn_layer_name: str, ple_layer_index: int):
        super().__init__()
        from tpu_inference.layers.vllm.custom_ops.experimental.qwen4_exp.ngram import \
            build_ngram_layout
        self.layout = build_ngram_layout(config, ple_layer_index)
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        hc_hidden = self.hidden_size * self.hc_count
        ple_dim = config.ple_embed_dim
        self.ple_embedding = Qwen4ExpNGramTable(
            self.layout, ple_dim // self.layout.num_heads,
            getattr(config, "split_ngram_parts", 1) or 1)
        self.key_proj = nn.Linear(ple_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(ple_dim, self.hidden_size, bias=False)
        eps = config.rms_norm_eps
        self.norm_key = Qwen4ExpRMSNorm(hc_hidden, eps, self.hidden_size)
        self.norm_query = Qwen4ExpRMSNorm(hc_hidden, eps, self.hidden_size)
        self.norm_conv = Qwen4ExpRMSNorm(hc_hidden, eps, self.hidden_size)
        self.conv1d = nn.Conv1d(hc_hidden,
                                hc_hidden,
                                kernel_size=config.ple_conv_kernel_size,
                                groups=hc_hidden,
                                dilation=config.ngram_size,
                                bias=False)
        self.dilation = config.ngram_size
        self.gdn_layer_name = gdn_layer_name
        self.gdn_conv_rows = config.linear_conv_kernel_dim - 1

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        md = _attn_metadata_for(self.gdn_layer_name)
        if md.mamba is None or getattr(md.mamba, "ple_embeddings",
                                       None) is None:
            raise RuntimeError(
                "Qwen4Exp PLE needs MambaMetadata.ple_embeddings from the "
                "runner")
        num_tokens = hidden_states.shape[0]
        emb = torch_view(md.mamba.ple_embeddings[:num_tokens])
        key = self.norm_key(self.key_proj(emb)).unflatten(
            -1, (self.hc_count, self.hidden_size))
        value = self.value_proj(emb)
        query = self.norm_query(hidden_states).unflatten(
            -1, (self.hc_count, self.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(
            self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_normed = self.norm_conv(gated.flatten(-2))
        gated = gated.flatten(-2)
        conv = self._short_conv(gated_normed, md)
        return hidden_states + gated + conv

    def _short_conv(self, normed: torch.Tensor, md) -> torch.Tensor:
        ctx = get_vllm_model_wrapper_context()
        idx = ctx.layer_name_to_kvcache_index[self.gdn_layer_name]
        conv_state, ssm_state = ctx.kv_caches[idx]
        num_reqs = md.request_distribution[2]
        out, new_conv_state = ple_short_conv(
            jax_view(normed),
            jax_view(self.conv1d.weight),
            conv_state,
            md.mamba.state_indices,
            md.query_start_loc,
            md.seq_lens,
            num_reqs,
            row_offset=self.gdn_conv_rows,
            dilation=self.dilation,
            max_reqs=md.seq_lens.shape[0],
            padded_num_reqs=md.padded_num_reqs,
        )
        ctx.kv_caches[idx] = (new_conv_state, ssm_state)
        return torch_view(out)


class Qwen4ExpQSAIndexer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_heads = config.indexer_n_heads
        self.kv_heads = config.indexer_kv_heads
        self.head_dim = config.indexer_head_dim
        if self.kv_heads != 1:
            raise NotImplementedError("QSA indexer supports one key head")
        self.index_qk_proj = nn.Linear(config.hidden_size,
                                       (self.n_heads + self.kv_heads) *
                                       self.head_dim,
                                       bias=False)
        self.q_layernorm = Qwen4ExpRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_layernorm = Qwen4ExpRMSNorm(self.head_dim, config.rms_norm_eps)


class Qwen4ExpQSAAttention(nn.Module):
    """Gated GQA over indexer-selected tokens."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.eps = config.rms_norm_eps
        self.budget = config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        if config.indexer_head_dim > self.head_dim:
            raise NotImplementedError("indexer head dim above head dim")
        rope = config.rope_parameters
        rot = int(self.head_dim * rope.get("partial_rotary_factor", 1.0))
        base = float(rope["rope_theta"])
        self.inv_freq_np = (1.0 / (base**(np.arange(0, rot, 2, dtype=np.float32)
                                         / rot))).astype(np.float32)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.num_heads * 2,  # Q and the output gate
            self.num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(self.num_heads * self.head_dim,
                                        self.hidden_size,
                                        bias=False,
                                        quant_config=vllm_config.quant_config,
                                        prefix=f"{prefix}.o_proj")
        self.q_norm = Qwen4ExpRMSNorm(self.head_dim, self.eps)
        self.k_norm = Qwen4ExpRMSNorm(self.head_dim, self.eps)
        self.indexer = Qwen4ExpQSAIndexer(config)
        # Owns the paged cache only; forward never calls it. One extra KV
        # head stores the raw indexer keys. The head count is rounded up to a
        # power of two so the TPU cache can shard (or replicate) it over TP.
        cache_heads = 1 << (self.num_kv_heads).bit_length()
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=cache_heads,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=f"{prefix}.attn",
        )
        self.attn_layer_name = f"{prefix}.attn"

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        qkv, _ = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size],
                                 dim=-1)
        q, gate = torch.chunk(q_gate.view(T, self.num_heads,
                                          self.head_dim * 2),
                              2,
                              dim=-1)
        gate = gate.reshape(T, -1)
        q = self.q_norm(q)
        k = self.k_norm(k.view(T, self.num_kv_heads, self.head_dim))
        v = v.view(T, self.num_kv_heads, self.head_dim)

        ix = self.indexer
        iqk = ix.index_qk_proj(hidden_states)
        iq, ik = iqk.split([ix.n_heads * ix.head_dim, ix.head_dim], dim=-1)
        iq = ix.q_layernorm(iq.view(T, ix.n_heads, ix.head_dim))

        md = _attn_metadata_for(self.attn_layer_name)
        ctx = get_vllm_model_wrapper_context()
        cache_idx = ctx.layer_name_to_kvcache_index[self.attn_layer_name]
        kv_cache = ctx.kv_caches[cache_idx]
        pos = positions[0] if positions.dim() == 2 else positions
        out, new_cache = qsa_attention(
            jax_view(q),
            jax_view(k),
            jax_view(v),
            jax_view(iq),
            jax_view(ik),
            kv_cache,
            jax_view(pos),
            md.block_tables,
            md.query_start_loc,
            md.request_distribution[2],
            jax_view(ix.k_layernorm.weight),
            jnp.asarray(self.inv_freq_np),
            max_reqs=md.seq_lens.shape[0],
            padded_num_reqs=md.padded_num_reqs,
            eps=self.eps,
            compress_ratio=self.compress_ratio,
            budget=self.budget,
            scale=self.scaling,
            sparse_max_tokens=_QSA_SPARSE_MAX_TOKENS,
        )
        ctx.kv_caches[cache_idx] = new_cache
        out = torch_view(out).reshape(T, -1) * torch.sigmoid(gate)
        out, _ = self.o_proj(out)
        return out


def _extend_gdn_conv_state(layer: nn.Module, extra_rows: int) -> None:
    """Give a GDN layer's conv state `extra_rows` more rows (PLE state)."""
    if extra_rows == 0:
        return
    orig = layer.get_state_shape

    def get_state_shape(self):
        conv_shape, *rest = orig()
        conv_shape = tuple(conv_shape)
        # Conv state is (state_len, dim) on TPU; the TPU GDN op keeps every
        # row past kernel_size - 1 untouched.
        conv_shape = (conv_shape[0] + extra_rows, conv_shape[1])
        return (conv_shape, *rest)

    layer.get_state_shape = types.MethodType(get_state_shape, layer)


class Qwen4ExpDecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.layer_idx = extract_layer_index(prefix)
        self.layer_type = layer_type
        self.ple = None
        if layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=False,
            )
            _extend_gdn_conv_state(self.linear_attn,
                                   ple_extra_conv_rows(config))
            ple_ids = sorted(set(getattr(config, "ple_layer_ids", []) or []))
            if (self.layer_idx + 1) in ple_ids:
                self.ple = Qwen4ExpPLE(config, f"{prefix}.linear_attn",
                                       ple_ids.index(self.layer_idx + 1))
        elif layer_type in _QSA_LAYER_TYPES:
            if getattr(config, "indexer_n_heads", None) is None:
                raise NotImplementedError(
                    "dense (non-QSA) Qwen4Exp attention layers")
            self.self_attn = Qwen4ExpQSAAttention(config, vllm_config,
                                                  f"{prefix}.self_attn")
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")
        self.mlp = Qwen3NextSparseMoeBlock(vllm_config=vllm_config,
                                           prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = Qwen4ExpGatedResidual(config)
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(config)

    def forward(self, hidden_states: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            hidden_states = self.ple(hidden_states)
        block_in, inject = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            block_out = self.linear_attn(hidden_states=block_in)
        else:
            block_out = self.self_attn(block_in, positions)
        hidden_states = Qwen4ExpGatedResidual.combine(hidden_states,
                                                      block_out, inject)
        block_in, inject = self.mlp_hyper_connection.mix(hidden_states)
        block_out = self.mlp(block_in)
        return Qwen4ExpGatedResidual.combine(hidden_states, block_out, inject)


class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        if get_pp_group().world_size > 1:
            raise NotImplementedError("Qwen4Exp TPU: pipeline parallel")
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size,
                                                   config.hidden_size)
        HOST_NGRAM_TABLES.clear()  # one live model per process

        def get_layer(prefix: str):
            return Qwen4ExpDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(config,
                                                            use_combine=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = inputs_embeds.repeat(1, self.config.hc_count)
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states = layer(hidden_states, positions)
        mixed, _ = self.hyper_connection_mixer.mix(hidden_states)
        return mixed

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={
                # The final mixer has no injection; some checkpoints carry it.
                "hyper_connection_mixer.block_inject_weight": None,
            })
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)


class Qwen4ExpForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "model."})
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vllm_config = vllm_config
        self.model = Qwen4ExpModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size,
                                      config.hidden_size,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self,
                input_ids: torch.Tensor | None,
                positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors,
                          inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(self, input_tokens, mm_features):
        # Text only: the three M-RoPE axes carry the same position.
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_prefix={
            "mtp.": None,
            "model.visual.": None,
            "visual.": None,
        })
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=mapper)

    # ---- IsHybrid ----
    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        hf = vllm_config.model_config.hf_text_config
        conv_shape, ssm_shape = (
            MambaStateShapeCalculator.gated_delta_net_state_shape(
                vllm_config.parallel_config.tensor_parallel_size,
                hf.linear_num_key_heads, hf.linear_num_value_heads,
                hf.linear_key_head_dim, hf.linear_value_head_dim,
                hf.linear_conv_kernel_dim, 0))
        extra = ple_extra_conv_rows(hf)
        return (conv_shape[0] + extra, conv_shape[1]), ssm_shape

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


class Qwen4ExpForConditionalGeneration(Qwen4ExpForCausalLM):
    """Text-only stand-in for the HF vision-language architecture: vision
    tower weights are skipped."""
