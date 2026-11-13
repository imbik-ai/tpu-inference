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
"""Reference (correct, not fast) JAX QSA for Qwen4Exp on TPU.

QSA = indexer-selected sparse GQA. Selection math follows transformers'
`Qwen4ExpTextQSAIndexer` (Apache 2.0): every complete group of
`compress_ratio` visible tokens is pooled (mean of raw index keys), RMS-normed
and RoPE'd at the group's first position; a query scores each group with
sum_h relu(q_h . k) / sqrt(d); the top `budget / compress_ratio` groups plus
the incomplete tail are the keys the query attends to.

Cache layout. The layer owns one regular paged attention cache (allocated by
the TPU KV cache manager from an `Attention` spec with `num_kv_heads + 1`
heads, rounded up to a power of two so TP can shard or replicate it).
Combined rows (k0, v0, k1, v1, ..., idx, unused...) per token: row
2h / 2h+1 hold K / V of KV head h, row 2*num_kv_heads holds the raw index key
(zero padded to the head dim). Pooled keys are recomputed from the raw keys
on every step instead of being cached.

Compute. For each (padded) request the op gathers the request's whole paged
history and runs a dense masked attention for the batch's tokens of that
request. Cost is O(tokens x max_model_len) per request; a sparse gather
kernel is the follow-up.
"""

import math

import jax
import jax.numpy as jnp


def rope_neox(x: jax.Array, positions: jax.Array,
              inv_freq: jax.Array) -> jax.Array:
    """Partial neox RoPE on the first 2*len(inv_freq) dims.

    x: (T, H, D); positions: (T,). cos/sin are rounded to x.dtype first, as in
    transformers.
    """
    rot = 2 * inv_freq.shape[0]
    freqs = positions.astype(jnp.float32)[:, None] * inv_freq[None, :]
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb).astype(x.dtype)[:, None, :]
    sin = jnp.sin(emb).astype(x.dtype)[:, None, :]
    x_rot, x_pass = x[..., :rot], x[..., rot:]
    x1, x2 = x_rot[..., :rot // 2], x_rot[..., rot // 2:]
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    out = x_rot * cos + rotated * sin
    return jnp.concatenate([out.astype(x.dtype), x_pass], axis=-1)


def gemma_rms_norm(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    xf = x.astype(jnp.float32)
    out = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (out * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def token_request_map(query_start_loc: jax.Array, num_reqs: jax.Array,
                      num_tokens: int, max_reqs: int):
    """Request index and validity of each (padded) token."""
    t = jnp.arange(num_tokens, dtype=jnp.int32)
    ends = query_start_loc[1:max_reqs + 1]
    req = jnp.searchsorted(ends, t, side="right").astype(jnp.int32)
    req = jnp.minimum(req, max_reqs - 1)
    total = query_start_loc[num_reqs]
    return req, t < total


def qsa_attention(
    q: jax.Array,  # (T, Hq, D), q_norm applied, no rope
    k: jax.Array,  # (T, Hkv, D), k_norm applied, no rope
    v: jax.Array,  # (T, Hkv, D)
    idx_q: jax.Array,  # (T, Hi, Di), q_layernorm applied, no rope
    idx_k: jax.Array,  # (T, Di) raw index keys
    kv_cache: jax.Array,  # (N, bs, A, P, Dc)
    positions: jax.Array,  # (T,)
    block_tables: jax.Array,  # (max_reqs * pages_per_req,)
    query_start_loc: jax.Array,  # (max_reqs + 1,)
    num_reqs: jax.Array,  # scalar int32
    idx_k_norm_weight: jax.Array,  # (Di,)
    inv_freq: jax.Array,  # (rot/2,) float32
    *,
    max_reqs: int,
    padded_num_reqs: int,
    eps: float,
    compress_ratio: int,
    budget: int,
    scale: float,
):
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    Di = idx_k.shape[-1]
    N, bs, A, P, Dc = kv_cache.shape
    C = A * P
    assert C >= 2 * Hkv + 1 and Dc >= max(D, Di)
    dtype = kv_cache.dtype

    positions = positions.astype(jnp.int32)
    req, valid = token_request_map(query_start_loc, num_reqs, T, max_reqs)
    bt = block_tables.reshape(max_reqs, -1)
    pages_per_req = bt.shape[1]

    # RoPE for the query/key paths at token positions.
    q = rope_neox(q, positions, inv_freq)
    k = rope_neox(k, positions, inv_freq)
    idx_q = rope_neox(idx_q, positions, inv_freq)

    # 1) Write this step's K, V and raw index keys into the paged cache.
    def pad_d(x):
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, Dc - x.shape[-1])])

    kv = jnp.stack([k, v], axis=2).reshape(T, 2 * Hkv, D)
    rows = jnp.concatenate([
        pad_d(kv).astype(dtype),
        pad_d(idx_k)[:, None, :].astype(dtype),
        jnp.zeros((T, C - 2 * Hkv - 1, Dc), dtype),
    ],
                           axis=1)
    # Index the cache in its native 5-D shape: reshaping the whole cache
    # changes its TPU tiling and costs full copies.
    page = bt[req, jnp.clip(positions // bs, 0, pages_per_req - 1)]
    page = jnp.where(valid, page, N)  # out of range -> dropped
    kv_cache = kv_cache.at[page, positions % bs].set(rows.reshape(
        T, A, P, Dc),
                                                     mode="drop")

    # 2) Per request: selection + dense masked attention.
    L = pages_per_req * bs
    NB = L // compress_ratio
    block_topk = min(budget // compress_ratio, NB)
    j = jnp.arange(L, dtype=jnp.int32)
    blk_of_j = jnp.minimum(j // compress_ratio, NB - 1)
    group = Hq // Hkv
    qg = q.reshape(T, Hkv, group, D)
    nb_t = (positions + 1) // compress_ratio  # complete visible groups
    bvalid = jnp.arange(NB, dtype=jnp.int32)[None, :] < nb_t[:, None]
    tail = (j[None, :] >= nb_t[:, None] * compress_ratio) & (
        j[None, :] <= positions[:, None])
    group_pos = jnp.arange(NB, dtype=jnp.int32) * compress_ratio
    out = jnp.zeros((T, Hq, D), dtype=q.dtype)

    for r in range(padded_num_reqs):
        hist = kv_cache[bt[r]].reshape(L, C, Dc)  # (L, C, Dc)
        k_r = hist[:, 0:2 * Hkv:2, :D]  # (L, Hkv, D)
        v_r = hist[:, 1:2 * Hkv:2, :D]
        ik_r = hist[:, 2 * Hkv, :Di]  # (L, Di)

        pooled = ik_r[:NB * compress_ratio].reshape(
            NB, compress_ratio, Di).astype(jnp.float32).mean(axis=1)
        pooled = gemma_rms_norm(pooled.astype(dtype), idx_k_norm_weight, eps)
        kc = rope_neox(pooled[:, None, :], group_pos, inv_freq)[:, 0, :]

        scores = jnp.einsum("thd,bd->thb",
                            idx_q.astype(jnp.float32),
                            kc.astype(jnp.float32),
                            preferred_element_type=jnp.float32)
        scores = jnp.maximum(scores, 0.0).sum(axis=1) / math.sqrt(Di)
        scores = jnp.where(bvalid, scores, -jnp.inf)
        _, top_idx = jax.lax.top_k(scores, block_topk)
        sel = jnp.zeros((T, NB), dtype=bool).at[jnp.arange(T)[:, None],
                                                top_idx].set(True)
        sel = sel & bvalid
        mask = jnp.take_along_axis(
            sel, jnp.broadcast_to(blk_of_j[None, :], (T, L)), axis=1)
        mask = (mask & (j[None, :] < nb_t[:, None] * compress_ratio)) | tail

        logits = jnp.einsum("tkgd,lkd->tkgl",
                            qg,
                            k_r,
                            preferred_element_type=jnp.float32) * scale
        logits = jnp.where(mask[:, None, None, :], logits, -jnp.inf)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        probs = jnp.exp(logits)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        o_r = jnp.einsum("tkgl,lkd->tkgd",
                         probs.astype(dtype),
                         v_r,
                         preferred_element_type=jnp.float32)
        o_r = o_r.reshape(T, Hq, D).astype(q.dtype)
        in_r = valid & (req == r)
        out = jnp.where(in_r[:, None, None], o_r, out)

    return out, kv_cache


def ple_short_conv(
    normed: jax.Array,  # (T, Cp) gated, normed PLE values
    conv_weight: jax.Array,  # (Cp, 1, K)
    conv_state: jax.Array,  # (S_slots, rows, conv_dim) GDN conv state
    state_indices: jax.Array,  # (max_reqs,)
    query_start_loc: jax.Array,  # (max_reqs + 1,)
    seq_lens: jax.Array,  # (max_reqs,)
    num_reqs: jax.Array,
    *,
    row_offset: int,
    dilation: int,
    max_reqs: int,
    padded_num_reqs: int,
):
    """Causal dilated depthwise conv with per-request state; returns
    (silu(conv), new_conv_state).

    The state (the last (K-1)*dilation inputs, oldest first) lives in rows
    [row_offset, row_offset + n_rows) of the same layer's GDN conv state
    slot, flattened over the conv_dim axis.
    """
    T, Cp = normed.shape
    K = conv_weight.shape[-1]
    S = (K - 1) * dilation
    n_slots, n_rows, conv_dim = conv_state.shape
    rows_needed = -(-S * Cp // conv_dim)
    assert row_offset + rows_needed <= n_rows

    req, valid = token_request_map(query_start_loc, num_reqs, T, max_reqs)
    qsl = query_start_loc
    qlen = qsl[1:max_reqs + 1] - qsl[:max_reqs]
    computed = seq_lens - qlen  # tokens before this step
    has_init = computed > 0
    slots = state_indices.astype(jnp.int32)

    def read_state(slot_ids):
        flat = conv_state[slot_ids, row_offset:row_offset + rows_needed, :]
        flat = flat.reshape(slot_ids.shape[0], rows_needed * conv_dim)
        return flat[:, :S * Cp].reshape(slot_ids.shape[0], S, Cp)

    state = read_state(slots)  # (max_reqs, S, Cp)
    state = jnp.where(has_init[:, None, None], state, 0).astype(normed.dtype)

    t = jnp.arange(T, dtype=jnp.int32)
    off = t - qsl[req]  # offset within this step's chunk
    acc = jnp.zeros((T, Cp), jnp.float32)
    w = conv_weight[:, 0, :].astype(jnp.float32)  # (Cp, K)
    for kk in range(K):
        lag = (K - 1 - kk) * dilation
        if lag == 0:
            x = normed
        else:
            from_chunk = off >= lag
            x_chunk = normed[jnp.maximum(t - lag, 0)]
            x_state = state[req, jnp.clip(S - lag + off, 0, S - 1)]
            x = jnp.where(from_chunk[:, None], x_chunk, x_state)
        acc = acc + x.astype(jnp.float32) * w[None, :, kk]
    conv = acc.astype(normed.dtype)
    out = jax.nn.silu(conv.astype(jnp.float32)).astype(normed.dtype)
    out = jnp.where(valid[:, None], out, 0)

    # New state: inputs at positions seq_len - S .. seq_len - 1.
    r = jnp.arange(padded_num_reqs, dtype=jnp.int32)
    jj = jnp.arange(S, dtype=jnp.int32)
    pos = seq_lens[:padded_num_reqs, None] - S + jj[None, :]
    c0 = computed[:padded_num_reqs, None]
    in_chunk = pos >= c0
    tok = jnp.clip(qsl[:padded_num_reqs, None] + (pos - c0), 0, T - 1)
    st_row = jnp.clip(S - (c0 - pos), 0, S - 1)
    new_state = jnp.where(in_chunk[..., None], normed[tok],
                          state[r[:, None], st_row])
    new_state = jnp.where((pos >= 0)[..., None], new_state, 0)
    flat = new_state.reshape(padded_num_reqs, S * Cp)
    flat = jnp.pad(flat, ((0, 0), (0, rows_needed * conv_dim - S * Cp)))
    flat = flat.reshape(padded_num_reqs, rows_needed,
                        conv_dim).astype(conv_state.dtype)
    write_slot = jnp.where(r < num_reqs, slots[:padded_num_reqs], n_slots)
    new_conv_state = conv_state.at[write_slot, row_offset:row_offset +
                                   rows_needed, :].set(flat, mode="drop")
    return out, new_conv_state
