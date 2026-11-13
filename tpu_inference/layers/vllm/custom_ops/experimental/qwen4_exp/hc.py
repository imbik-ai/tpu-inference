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
"""Qwen4Exp 4-stream hyper-connection mix/combine in JAX, fp32 inside.

Same math as transformers' Qwen4ExpTextGatedResidual (Apache 2.0). Matmul
inputs are bf16 (as in the checkpoint), but the normed streams, matmul
outputs, gates and the stream mean stay fp32 and are rounded once. Written
with torch ops, the mix rounded at every step and landed 2x further from an
fp32 reference than transformers' own bf16 run (0.006 vs 0.003 relative).
"""

import jax
import jax.numpy as jnp


def hc_mix(x, norm_w, down_w, up_w, inject_w, *, hc_count: int, hidden: int,
           eps: float):
    """x (T, C*H). Returns (mixed (T, H) in x.dtype, inject (T, C) fp32 or
    None)."""
    T = x.shape[0]
    xf = x.astype(jnp.float32).reshape(T, hc_count, hidden)
    xn = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + eps)
    xn = xn.reshape(T, -1) * (1.0 + norm_w.astype(jnp.float32))
    xb = xn.astype(down_w.dtype)
    d = jnp.dot(xb, down_w.T, preferred_element_type=jnp.float32)
    a = jax.nn.silu(d / hc_count)
    u = jnp.dot(a.astype(up_w.dtype), up_w.T,
                preferred_element_type=jnp.float32)
    w = jax.nn.sigmoid(u)
    mixed = (w.reshape(T, hc_count, hidden) *
             xn.reshape(T, hc_count, hidden)).mean(axis=1).astype(x.dtype)
    if inject_w is None:
        return mixed, None
    inj = 2.0 * jax.nn.sigmoid(
        jnp.dot(xb, inject_w.T, preferred_element_type=jnp.float32) /
        hc_count)
    return mixed, inj


def hc_combine(x, block_out, inject):
    """x (T, C*H) + block_out (T, H) injected into each stream; one rounding."""
    T = x.shape[0]
    C = inject.shape[-1]
    y = x.astype(jnp.float32).reshape(T, C, -1) + (
        block_out.astype(jnp.float32)[:, None, :] * inject[:, :, None])
    return y.reshape(T, -1).astype(x.dtype)
