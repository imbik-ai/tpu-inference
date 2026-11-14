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
(zero padded to the head dim). Compressed keys (pooled, normed, RoPE'd) are
written when a group completes, compactly into row 2*num_kv_heads+1 of the
first block_size/ratio slots of the group's page; there is no separate
compressor ring.

Compute. Small steps (decode) gather, per query, the compressed keys of its
request, pick the top groups, and attend over the selected tokens plus the
tail only. Large steps (prefill) run a dense masked attention per request
over its whole paged history, O(tokens x max_model_len). Plain JAX gathers;
a Pallas kernel is the follow-up.
"""

import functools
import math
import os

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.vllm.custom_ops.experimental.qwen4_exp.paged_gather import \
    paged_gather


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


def _qsa_write_and_compress(q, k, v, idx_q, idx_k, kv_cache, positions, bt,
                            req, valid, idx_k_norm_weight, inv_freq, *, eps,
                            compress_ratio, gather):
    """RoPE, then write K/V/raw index keys and the compressed key of every
    group this step completes."""
    T, _, D = q.shape
    Hkv = k.shape[1]
    Di = idx_k.shape[-1]
    N, bs, A, P, Dc = kv_cache.shape
    C = A * P
    R = compress_ratio
    assert C >= 2 * Hkv + 2 and Dc >= max(D, Di)
    dtype = kv_cache.dtype
    pages_per_req = bt.shape[1]

    q = rope_neox(q, positions, inv_freq)
    k = rope_neox(k, positions, inv_freq)
    idx_q = rope_neox(idx_q, positions, inv_freq)

    def pad_d(x):
        return jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, Dc - x.shape[-1])])

    assert P == 2, "QSA cache layout assumes a 16-bit cache (K, V pairs)"
    # Index the cache in its native 5-D shape: reshaping the whole cache
    # changes its TPU tiling and costs full copies.
    page = bt[req, jnp.clip(positions // bs, 0, pages_per_req - 1)]
    page = jnp.where(valid, page, N)  # out of range -> dropped
    off = positions % bs
    # Writes move whole token rows (every head): the cache is sharded over
    # the head axis under TP, and indexing one head makes the partitioner
    # gather the whole cache. Reads use dynamic_slice in a fori_loop: an XLA
    # gather from this layout first copies the whole cache on TPU.
    kv = pad_d(jnp.stack([k, v], axis=2)).astype(dtype)  # (T, Hkv, 2, Dc)
    page_c = jnp.minimum(page, N - 1)
    # A token's slot is always written before the compressed key stored in
    # the same slot (see below), so token rows can zero that row.
    row = jnp.zeros((T, A, P, Dc), dtype).at[:, :Hkv].set(kv)
    row = row.at[:, Hkv, 0].set(pad_d(idx_k).astype(dtype))
    kv_cache = kv_cache.at[page, off].set(row, mode="drop")

    # Compressed keys: mean of the group's raw keys, k-norm, RoPE at the
    # group's first position. Group g of a page is stored compactly in row
    # (Hkv, 1) of the page's slot g % (bs // R), so a request's compressed
    # keys are the first bs // R slots of each of its pages. That slot's
    # token (g % (bs // R) <= R * g + R - 1 within the page) is written
    # first, in an earlier step or earlier in this one.
    assert bs % R == 0
    done = valid & ((positions + 1) % R == 0)
    slot = (positions // R) % (bs // R)
    g0 = jnp.maximum(off - (R - 1), 0)

    raws = gather(kv_cache, page_c, g0, length=R)  # (T, R, A, P, Dc)
    crow = gather(kv_cache, page_c, slot, length=1)[:, 0]  # (T, A, P, Dc)
    raw = raws[:, :, Hkv, 0, :Di]  # (T, R, Di)
    pooled = raw.astype(jnp.float32).mean(axis=1).astype(dtype)
    pooled = gemma_rms_norm(pooled, idx_k_norm_weight, eps)
    ck = rope_neox(pooled[:, None, :], positions - (R - 1), inv_freq)[:, 0, :]
    a_c, p_c = Hkv, 1
    crow = crow.at[:, a_c, p_c].set(pad_d(ck).astype(dtype))
    kv_cache = kv_cache.at[jnp.where(done, page, N), slot].set(crow,
                                                               mode="drop")
    return q, idx_q, kv_cache, (a_c, p_c)


def topk_mask(scores: jax.Array, k: int) -> jax.Array:
    """Membership mask of jax.lax.top_k(scores, k) per row, restricted to
    entries > -inf (same set, including top_k's lower-index-first rule for
    ties). scores (rows, n) float32, finite values >= 0 or -inf.

    lax.top_k over the indexer's groups (k = 512 of up to max_model_len / 4)
    was the largest single op of a long prefill; this finds each row's k-th
    largest value by bisection on the float bits (31 vectorized
    compare-and-count passes) and needs no sort or scatter.
    """
    n = scores.shape[-1]
    if k >= n:
        return scores > -jnp.inf
    valid = scores > -jnp.inf
    # Non-negative float32 values order like their int32 bit patterns.
    keys = jnp.where(valid, jax.lax.bitcast_convert_type(
        jnp.maximum(scores, 0.0), jnp.int32), -1)

    def step(_, lohi):
        lo, hi = lohi
        mid = lo + (hi - lo) // 2
        ok = (keys >= mid[:, None]).sum(-1) >= k
        return jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)

    rows = scores.shape[:-1]
    lo = jnp.zeros(rows, jnp.int32)
    hi = jnp.full(rows, jnp.iinfo(jnp.int32).max, jnp.int32)
    thr, _ = jax.lax.fori_loop(0, 31, step, (lo, hi))  # k-th largest key
    above = keys > thr[:, None]
    tie = keys == thr[:, None]
    need = k - above.sum(-1, keepdims=True)
    take = tie & (jnp.cumsum(tie, axis=-1) <= need)
    return (above | take) & valid


_QSA_HEAD_LOCAL = os.environ.get("QWEN4EXP_QSA_HEAD_LOCAL", "1") == "1"


def _head_local_attend(kv_cache, qg, mask, pages, offs, *, length, D, scale,
                       dtype, mesh, head_axis, cache_spec):
    """Union-path attention with one cache head slot per device.

    qg (c, Hkv, G, D) replicated; mask (c, S) with S = len(pages) * length.
    Device h gathers the runs of its own head slot and, when h < Hkv,
    attends with KV head h; other devices (index / spare slots) return
    zeros. Returns (c, Hkv, G, D).
    """
    c, Hkv, G, _ = qg.shape

    def local(cache_l, qg_r, mask_r, pages_r, offs_r):
        h = jax.lax.axis_index(head_axis)
        kv = paged_gather(cache_l, pages_r, offs_r, length=length)
        kv = kv[:, :, 0]  # (U, R, P, Dc): this device's head slot
        keys = kv[..., 0, :D].reshape(-1, D)
        vals = kv[..., 1, :D].reshape(-1, D)
        q_h = jax.lax.dynamic_index_in_dim(qg_r, jnp.minimum(h, Hkv - 1),
                                           axis=1, keepdims=False)
        logits = jnp.einsum("tgd,sd->tgs", q_h, keys,
                            preferred_element_type=jnp.float32) * scale
        logits = jnp.where(mask_r[:, None, :], logits, -jnp.inf)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        probs = jnp.exp(logits)
        probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
        out = jnp.einsum("tgs,sd->tgd", probs.astype(dtype), vals,
                         preferred_element_type=jnp.float32)
        out = jnp.where(h < Hkv, out, 0.0)
        return out[:, None]  # (c, 1, G, D)

    out = jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(cache_spec or P(None, None, head_axis), P(), P(), P(), P()),
        out_specs=P(None, head_axis),
        check_vma=False,
    )(kv_cache, qg, mask, pages.astype(jnp.int32), offs.astype(jnp.int32))
    return out[:, :Hkv]


def _masked_softmax_attend(qg, keys, vals, mask, scale, dtype):
    """qg (T,Hkv,G,D); keys/vals (T or 1, S, Hkv, D); mask (T,S)."""
    logits = jnp.einsum("tkgd,tskd->tkgs",
                        qg,
                        keys,
                        preferred_element_type=jnp.float32) * scale
    logits = jnp.where(mask[:, None, None, :], logits, -jnp.inf)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    probs = jnp.exp(logits)
    probs = probs / jnp.sum(probs, axis=-1, keepdims=True)
    return jnp.einsum("tkgs,tskd->tkgd",
                      probs.astype(dtype),
                      vals,
                      preferred_element_type=jnp.float32)


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
    sparse_max_tokens: int = 1 << 30,
    union_caps: tuple = (2048, 8192),
    mesh=None,
    head_axis=None,
    cache_spec=None,
):
    """Steps with at most `sparse_max_tokens` (padded) tokens gather only
    each query's selected keys (in chunks of 64 queries); larger steps run a
    dense masked attention per request. Both read the cached compressed
    keys and give the same result; the sparse path is the default."""
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    Di = idx_k.shape[-1]
    N, bs, A, P, Dc = kv_cache.shape
    C = A * P
    R = compress_ratio
    dtype = kv_cache.dtype

    positions = positions.astype(jnp.int32)
    req, valid = token_request_map(query_start_loc, num_reqs, T, max_reqs)
    bt = block_tables.reshape(max_reqs, -1)
    pages_per_req = bt.shape[1]
    gather = functools.partial(paged_gather,
                               mesh=mesh,
                               head_axis=head_axis,
                               cache_spec=cache_spec)
    q, idx_q, kv_cache, (a_c, p_c) = _qsa_write_and_compress(
        q, k, v, idx_q, idx_k, kv_cache, positions, bt, req, valid,
        idx_k_norm_weight, inv_freq, eps=eps, compress_ratio=R,
        gather=gather)

    L = pages_per_req * bs
    NB = L // R
    block_topk = min(budget // R, NB)
    group = Hq // Hkv
    qg = q.reshape(T, Hkv, group, D)
    nb_t = (positions + 1) // R  # complete visible groups
    bvalid = jnp.arange(NB, dtype=jnp.int32)[None, :] < nb_t[:, None]

    def select(iq, ckeys, bv):
        """iq (t,Hi,Di), ckeys (t or 1, NB, Di), bv (t, NB) -> top groups."""
        scores = jnp.einsum("thd,tbd->thb",
                            iq.astype(jnp.float32),
                            ckeys.astype(jnp.float32),
                            preferred_element_type=jnp.float32)
        scores = jnp.maximum(scores, 0.0).sum(axis=1) / math.sqrt(Di)
        scores = jnp.where(bv, scores, -jnp.inf)
        return topk_mask(scores, block_topk)

    r_pad = padded_num_reqs

    if T <= sparse_max_tokens:
        # Sparse: per query, score its request's compressed keys, then fetch
        # K/V of the selected groups plus the tail group only, a group (R
        # contiguous tokens, never across pages) per DMA. Queries run in
        # chunks of `chunk` to bound the gathered K/V.
        # Compressed keys: the first bs // R slots of every page.
        ck_pages = gather(kv_cache,
                          bt[:r_pad].reshape(-1),
                          jnp.zeros((r_pad * pages_per_req, ), jnp.int32),
                          length=bs // R)
        ck_all = ck_pages[:, :, a_c, p_c, :Di].reshape(r_pad, NB, Di)
        chunk = min(T, 64)
        pad = (-T) % chunk

        def per_query(qg_c, req_c, pos_c, nb_c, sel):
            """Each query gathers its own selected groups (no sharing)."""
            top_idx, top_ok = jax.vmap(lambda m: (
                jnp.nonzero(m, size=block_topk, fill_value=0)[0],
                jnp.arange(block_topk) < m.sum()))(sel)
            groups = jnp.concatenate([top_idx, nb_c[:, None]], axis=1)
            counts = jnp.concatenate(
                [jnp.where(top_ok, R, 0), (pos_c + 1 - nb_c * R)[:, None]],
                axis=1)
            gtok = groups * R
            c, S = groups.shape
            gpage = bt[req_c[:, None], jnp.clip(gtok // bs, 0,
                                                pages_per_req - 1)]
            kv = gather(kv_cache, gpage.reshape(-1), (gtok % bs).reshape(-1),
                        length=R)  # (c*S, R, A, P, Dc)
            kv = kv.reshape(c, S, R, A, P, Dc)[:, :, :, :Hkv]
            keys = kv[..., 0, :D].reshape(c, S * R, Hkv, D)
            vals = kv[..., 1, :D].reshape(c, S * R, Hkv, D)
            ok = (jnp.arange(R, dtype=jnp.int32)[None, None, :] <
                  counts[:, :, None]).reshape(c, S * R)
            return _masked_softmax_attend(qg_c, keys, vals, ok, scale,
                                          dtype).astype(q.dtype)

        n_gid = r_pad * NB  # union ids: request * NB + group
        # Under TP the cache's head slots are spread one per device; then each
        # device attends with the KV head it holds (shard_map), so the
        # gathered keys never cross devices (only q and the output do).
        head_local = (_QSA_HEAD_LOCAL and mesh is not None
                      and head_axis is not None
                      and mesh.shape[head_axis] == A and A >= Hkv)
        # A chunk's union holds at most chunk * (top-k + tail) groups, so a
        # decode step (few queries) always fits one exact-size bucket.
        max_union = min(chunk * (block_topk + 1), n_gid)
        caps = sorted({min(c, max_union) for c in union_caps if c > 0})

        def request_bits(sel, req_c, valid_c):
            """(c, NB) per-query masks -> (c, r_pad * NB) in union-id space
            (request * NB + group); invalid queries select nothing."""
            sel = sel & valid_c[:, None]
            onehot = req_c[:, None] == jnp.arange(r_pad)[None, :]
            return (onehot[:, :, None] & sel[:, None, :]).reshape(
                sel.shape[0], n_gid)

        def shared(union_cap, qg_c, req_c, pos_c, nb_c, sel, valid_c):
            """The chunk's queries share one gather of the union of their
            groups (selected + tails, at most `union_cap`); each query masks
            to its own."""
            c = qg_c.shape[0]
            cnt = pos_c + 1 - nb_c * R
            tail_gid = jnp.where(valid_c & (cnt > 0), req_c * NB + nb_c,
                                 n_gid)
            qbits = request_bits(sel, req_c, valid_c)  # (c, n_gid)
            ubits = qbits.any(axis=0).at[tail_gid].set(True, mode="drop")
            uidx = jnp.nonzero(ubits, size=union_cap, fill_value=0)[0]
            u_ok = jnp.arange(union_cap) < ubits.sum()
            u_req, u_grp = uidx // NB, uidx % NB
            in_sel = qbits[:, uidx] & u_ok[None, :]
            is_tail = (uidx[None, :] == tail_gid[:, None]) & u_ok[None, :]
            ok = in_sel[:, :, None] | (is_tail[:, :, None] & (
                jnp.arange(R, dtype=jnp.int32)[None, None, :] <
                cnt[:, None, None]))  # (c, U, R)
            utok = u_grp * R
            upage = bt[u_req, jnp.clip(utok // bs, 0, pages_per_req - 1)]
            mask = ok.reshape(c, union_cap * R)
            if head_local:
                return _head_local_attend(kv_cache, qg_c, mask, upage,
                                          utok % bs, length=R, D=D,
                                          scale=scale, dtype=dtype, mesh=mesh,
                                          head_axis=head_axis,
                                          cache_spec=cache_spec).astype(
                                              q.dtype)
            kv = gather(kv_cache, upage, utok % bs, length=R)
            kv = kv[:, :, :Hkv]  # (U, R, Hkv, P, Dc)
            keys = kv[..., 0, :D].reshape(1, union_cap * R, Hkv, D)
            vals = kv[..., 1, :D].reshape(1, union_cap * R, Hkv, D)
            return _masked_softmax_attend(qg_c, keys, vals,
                                          ok.reshape(c, union_cap * R), scale,
                                          dtype).astype(q.dtype)

        def run_chunk(args):
            qg_c, iq_c, req_c, pos_c, valid_c = args
            req_c = jnp.minimum(req_c, r_pad - 1)
            nb_c = (pos_c + 1) // R
            bv = jnp.arange(NB, dtype=jnp.int32)[None, :] < nb_c[:, None]
            sel = select(iq_c, ck_all[req_c], bv)  # (c, NB) top-k mask
            if not caps:
                return per_query(qg_c, req_c, pos_c, nb_c, sel)
            # Union size, computed the same way as in `shared`.
            cnt = pos_c + 1 - nb_c * R
            bits = request_bits(sel, req_c, valid_c).any(axis=0)
            bits = bits.at[jnp.where(valid_c & (cnt > 0), req_c * NB + nb_c,
                                     n_gid)].set(True, mode="drop")
            n_union = bits.sum()
            # Smallest bucket that holds the union; past the last bucket,
            # per-query gathers.
            branch = (n_union > jnp.asarray(caps, jnp.int32)).sum()
            args = (qg_c, req_c, pos_c, nb_c, sel)
            branches = [
                functools.partial(shared, cap, *args, valid_c) for cap in caps
            ]
            if caps[-1] < max_union:
                branches.append(functools.partial(per_query, *args))
            if len(branches) == 1:
                return branches[0]()
            return jax.lax.switch(jnp.minimum(branch, len(branches) - 1),
                                  branches)

        def chunks(x):
            x = jnp.pad(x, [(0, pad)] + [(0, 0)] * (x.ndim - 1))
            return x.reshape(-1, chunk, *x.shape[1:])

        out = jax.lax.map(
            run_chunk, (chunks(qg), chunks(idx_q), chunks(req),
                        chunks(positions), chunks(valid)))
        out = out.reshape(-1, Hq, D)[:T]
        return jnp.where(valid[:, None, None], out, 0), kv_cache

    # Dense reference path: read each request's pages that hold this step's
    # visible tokens (dynamic loop bound), then masked attention over them.
    last_pos = jnp.full((r_pad, ), -1, jnp.int32).at[jnp.where(
        valid, req, r_pad)].max(positions, mode="drop")
    n_pages = (last_pos + bs) // bs  # ceil((last_pos + 1) / bs)
    pages_buf = jnp.zeros((r_pad, pages_per_req, bs, A, P, Dc), dtype)
    for r in range(r_pad):

        def read_page(i, buf, r=r):
            page = jax.lax.dynamic_slice_in_dim(kv_cache, bt[r, i], 1, 0)
            return jax.lax.dynamic_update_slice(buf, page[None],
                                                (r, i, 0, 0, 0, 0))

        pages_buf = jax.lax.fori_loop(0, n_pages[r], read_page, pages_buf)
    ck_all = pages_buf[:, :, :bs // R, a_c, p_c, :Di].reshape(r_pad, NB, Di)

    j = jnp.arange(L, dtype=jnp.int32)
    blk_of_j = jnp.minimum(j // R, NB - 1)
    tail = (j[None, :] >= nb_t[:, None] * R) & (j[None, :] <= positions[:, None])
    out = jnp.zeros((T, Hq, D), dtype=q.dtype)
    for r in range(padded_num_reqs):
        hist = pages_buf[r].reshape(L, C, Dc)  # (L, C, Dc)
        k_r = hist[:, 0:2 * Hkv:2, :D]  # (L, Hkv, D)
        v_r = hist[:, 1:2 * Hkv:2, :D]
        sel = select(idx_q, ck_all[r][None], bvalid) & bvalid
        mask = jnp.take_along_axis(
            sel, jnp.broadcast_to(blk_of_j[None, :], (T, L)), axis=1)
        mask = (mask & (j[None, :] < nb_t[:, None] * R)) | tail
        o_r = _masked_softmax_attend(qg, k_r[None], v_r[None], mask, scale,
                                     dtype)
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
    read_offsets: jax.Array | None = None,  # (max_reqs,) with spec decode
    num_spec: int = 0,
    num_windowed: jax.Array | None = None,  # requests [0, n) are windows
    read_state_indices: jax.Array | None = None,  # (max_reqs,) read slots
):
    """Causal dilated depthwise conv with per-request state; returns
    (silu(conv), new_conv_state).

    The state (the last (K-1)*dilation inputs, oldest first) lives in rows
    [row_offset, row_offset + n_rows) of the same layer's GDN conv state
    slot, flattened over the conv_dim axis.

    Speculative decoding follows the GDN slot convention: each request owns
    num_spec + 1 consecutive slots from state_indices[r]; a verify window
    (requests [0, num_windowed)) reads its state from slot
    state_indices[r] + read_offsets[r] and writes the state after window
    token t to slot state_indices[r] + t.
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
    # Align-mode prefix caching reads from another slot (the block holding
    # the last computed token) than it writes; default: the same slot.
    read_base = (slots if read_state_indices is None else
                 read_state_indices.astype(jnp.int32))

    def read_state(slot_ids):
        flat = conv_state[slot_ids, row_offset:row_offset + rows_needed, :]
        flat = flat.reshape(slot_ids.shape[0], rows_needed * conv_dim)
        return flat[:, :S * Cp].reshape(slot_ids.shape[0], S, Cp)

    windowed = jnp.zeros((max_reqs, ), bool)
    if num_spec > 0:
        windowed = jnp.arange(max_reqs) < num_windowed
        read_slots = read_base + jnp.where(windowed, read_offsets, 0)
    else:
        read_slots = read_base
    state = read_state(read_slots)  # (max_reqs, S, Cp)
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

    r = jnp.arange(padded_num_reqs, dtype=jnp.int32)
    jj = jnp.arange(S, dtype=jnp.int32)
    c0 = computed[:padded_num_reqs, None]

    def state_ending_at(end):  # (R,) exclusive end position -> (R, S, Cp)
        pos = end[:, None] - S + jj[None, :]
        in_chunk = pos >= c0
        tok = jnp.clip(qsl[:padded_num_reqs, None] + (pos - c0), 0, T - 1)
        st_row = jnp.clip(S - (c0 - pos), 0, S - 1)
        st = jnp.where(in_chunk[..., None], normed[tok], state[r[:, None],
                                                                st_row])
        st = jnp.where((pos >= 0)[..., None], st, 0)
        flat = st.reshape(padded_num_reqs, S * Cp)
        flat = jnp.pad(flat, ((0, 0), (0, rows_needed * conv_dim - S * Cp)))
        return flat.reshape(padded_num_reqs, rows_needed,
                            conv_dim).astype(conv_state.dtype)

    def write(cs, slot, rows):
        return cs.at[slot, row_offset:row_offset + rows_needed, :].set(
            rows, mode="drop")

    live = r < num_reqs
    base = slots[:padded_num_reqs]
    win = windowed[:padded_num_reqs]
    # Final state (positions seq_len - S .. seq_len - 1) for whole requests.
    new_conv_state = write(conv_state, jnp.where(live & ~win, base, n_slots),
                           state_ending_at(seq_lens[:padded_num_reqs]))
    for t in range(num_spec + 1 if num_spec > 0 else 0):
        # Verify windows: checkpoint after window token t at base + t.
        ok = live & win & (t < qlen[:padded_num_reqs])
        new_conv_state = write(new_conv_state,
                               jnp.where(ok, base + t, n_slots),
                               state_ending_at(c0[:, 0] + t + 1))
    return out, new_conv_state
