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
"""Host-side n-gram hashing for the Qwen4Exp (Flash-Next) PLE layer.

The hash follows transformers' `Qwen4ExpTextNGramEmbedding` (Apache 2.0,
The Qwen Team and The HuggingFace Inc. team): SplitMix64 layer multipliers,
one prime-sized table per n-gram head, ids XOR-mixed in wrapping int64 and
reduced with a floor modulo. The runner computes the ids on the host from
each request's token history, so the device needs no int64 math and no
per-request n-gram context state.
"""

import math
from dataclasses import dataclass

import numpy as np

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _layer_multipliers(unigram_vocab_size: int, ngram_size: int,
                       ple_layer_index: int, seed: int) -> list[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return out


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


@dataclass
class NGramLayout:
    """Static hashing layout of one PLE layer."""
    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int
    multipliers: np.ndarray  # (ngram_size,) int64
    head_vocab_sizes: np.ndarray  # (num_heads,) int64
    head_offsets: np.ndarray  # (num_heads,) int64
    padded_vocab_size: int

    @property
    def num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram


def build_ngram_layout(text_config, ple_layer_index: int) -> NGramLayout:
    ngram_size = int(text_config.ngram_size)
    heads_per_ngram = int(text_config.heads_per_ngram)
    num_heads = (ngram_size - 1) * heads_per_ngram
    eos = text_config.eos_token_id
    eos = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
    seed = int(getattr(text_config, "seed", 1234))
    sizes, offsets, total = [], [], 0
    for head_idx in range(num_heads):
        global_head_idx = ple_layer_index * num_heads + head_idx
        size = _nth_prime_after(int(text_config.ngram_vocab_size_base) - 1,
                                global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    divisor = int(text_config.make_ngram_vocab_size_divisible_by)
    padded = math.ceil(total / divisor) * divisor
    mult = _layer_multipliers(int(text_config.vocab_size), ngram_size,
                              ple_layer_index, seed)
    # Multipliers are < 2**63, so they fit in int64 without wrapping.
    return NGramLayout(ngram_size=ngram_size,
                       heads_per_ngram=heads_per_ngram,
                       eos_token_id=eos,
                       multipliers=np.array(mult, dtype=np.int64),
                       head_vocab_sizes=np.array(sizes, dtype=np.int64),
                       head_offsets=np.array(offsets, dtype=np.int64),
                       padded_vocab_size=padded)


def shifted_tokens(history: np.ndarray, positions: np.ndarray,
                   ngram_size: int, eos: int) -> np.ndarray:
    """Tokens at shift 0..ngram_size-1 for each position, EOS-reset.

    history: (max_len,) token ids of one request; positions: (n,) int.
    Before position 0 the context is EOS, and a shift never crosses an EOS
    (transformers `_shift_right_ignore_eos`): once a shifted token is EOS,
    every larger shift is EOS too.
    """
    out = np.empty((positions.shape[0], ngram_size), dtype=np.int64)
    out[:, 0] = history[positions]
    hit_eos = np.zeros(positions.shape[0], dtype=bool)
    for s in range(1, ngram_size):
        src = positions - s
        tok = np.where(src >= 0, history[np.maximum(src, 0)], eos)
        tok = np.where(hit_eos, eos, tok)
        out[:, s] = tok
        hit_eos |= tok == eos
    return out


def ngram_ids_from_shifted(shifted: np.ndarray,
                           layout: NGramLayout) -> np.ndarray:
    """(n, ngram_size) shifted tokens -> (n, num_heads) int32 table rows."""
    with np.errstate(over="ignore"):
        prods = shifted.astype(np.int64) * layout.multipliers[None, :]
    blocks = []
    for ngram in range(2, layout.ngram_size + 1):
        start = (ngram - 2) * layout.heads_per_ngram
        end = start + layout.heads_per_ngram
        mixed = prods[:, 0].copy()
        for position in range(1, ngram):
            mixed = np.bitwise_xor(mixed, prods[:, position])
        sizes = layout.head_vocab_sizes[start:end]
        ids = np.mod(mixed[:, None], sizes[None, :])  # floor mod, >= 0
        blocks.append(ids + layout.head_offsets[None, start:end])
    return np.concatenate(blocks, axis=-1).astype(np.int32)


def batch_ngram_ids(token_ids: np.ndarray, req_indices: np.ndarray,
                    positions: np.ndarray, layout: NGramLayout) -> np.ndarray:
    """N-gram table rows for scheduled tokens.

    token_ids: (max_reqs, max_model_len) token history (the runner's
    `input_batch.token_ids_cpu`); req_indices / positions: (n,) per token.
    """
    n = positions.shape[0]
    width = token_ids.shape[1]
    flat = token_ids.reshape(-1)
    base = req_indices.astype(np.int64) * width
    shifted = np.empty((n, layout.ngram_size), dtype=np.int64)
    shifted[:, 0] = flat[base + positions]
    hit_eos = np.zeros(n, dtype=bool)
    for s in range(1, layout.ngram_size):
        src = positions.astype(np.int64) - s
        tok = np.where(src >= 0, flat[base + np.maximum(src, 0)],
                       layout.eos_token_id)
        tok = np.where(hit_eos, layout.eos_token_id, tok)
        shifted[:, s] = tok
        hit_eos |= tok == layout.eos_token_id
    return ngram_ids_from_shifted(shifted, layout)
