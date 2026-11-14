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
"""Gather fixed-length token runs from the paged KV cache with DMAs.

out[i] = cache[page[i], off[i] : off[i] + length] for i in range(n).

XLA gathers from the (pages, page_size, heads, 2, dim) cache layout first
relayout the whole cache on TPU, and dynamic-slice loops pay per-iteration
overhead; this kernel issues one HBM->HBM DMA per run instead. Under TP the
cache is sharded on the head axis, so it runs per shard (shard_map).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

_MAX_PER_CALL = 16384  # runs per kernel call (index arrays live in SMEM)


def _kernel(pages_ref, offs_ref, cache_hbm, out_hbm, sem, *, n, length):

    def start(i, c):
        pltpu.make_async_copy(
            cache_hbm.at[pages_ref[i], pl.ds(offs_ref[i], length)],
            out_hbm.at[i], sem.at[0]).start()
        return c

    def wait(i, c):
        pltpu.make_async_copy(cache_hbm.at[0, pl.ds(0, length)],
                              out_hbm.at[i], sem.at[0]).wait()
        return c

    jax.lax.fori_loop(0, n, start, 0)
    jax.lax.fori_loop(0, n, wait, 0)


def _gather_local(cache, pages, offs, *, length):
    n = pages.shape[0]
    out_shape = jax.ShapeDtypeStruct((n, length, *cache.shape[2:]),
                                     cache.dtype)
    return pl.pallas_call(
        functools.partial(_kernel, n=n, length=length),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, ),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA((1, ))],
        ),
    )(pages, offs, cache)


def paged_gather(cache,
                 pages,
                 offs,
                 *,
                 length,
                 mesh=None,
                 head_axis=None,
                 cache_spec=None):
    """cache (N, bs, A, P, D); pages/offs (n,) int32 -> (n, length, A, P, D).

    Runs [off, off + length) must lie inside one page. With a mesh, the
    cache's head axis (dim 2) is sharded over `head_axis`.
    """
    pages = pages.astype(jnp.int32)
    offs = offs.astype(jnp.int32)
    if jax.default_backend() != "tpu":  # reference path (CPU tests)
        return cache[pages[:, None],
                     offs[:, None] + jnp.arange(length, dtype=jnp.int32)]
    n = pages.shape[0]
    chunks = []
    for s in range(0, n, _MAX_PER_CALL):
        p, o = pages[s:s + _MAX_PER_CALL], offs[s:s + _MAX_PER_CALL]
        fn = functools.partial(_gather_local, length=length)
        if mesh is not None and head_axis is not None:
            fn = jax.shard_map(
                fn,
                mesh=mesh,
                in_specs=(cache_spec or P(None, None, head_axis), P(), P()),
                out_specs=P(None, None, head_axis),
                check_vma=False,
            )
        chunks.append(fn(cache, p, o))
    return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)
