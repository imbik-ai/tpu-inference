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
"""topk_mask must select exactly jax.lax.top_k's set (QSA indexer)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.layers.vllm.custom_ops.experimental.qwen4_exp.qsa import \
    topk_mask


def _ref(scores, k):
    vals, idx = jax.lax.top_k(scores, k)
    m = np.zeros(scores.shape, bool)
    np.put_along_axis(m, np.asarray(idx), np.asarray(vals > -jnp.inf), -1)
    return m


@pytest.mark.parametrize("n,k", [(1000, 64), (4096, 512), (300, 300), (513, 512)])
def test_matches_lax_top_k(n, k):
    rng = np.random.default_rng(n)
    s = rng.gamma(2.0, 1.0, (24, n)).astype(np.float32)
    s[:4] = np.round(s[:4] * 4) / 4  # many exact ties
    s[4:8] = 0.0  # all tied at zero (ReLU output)
    s[8] = np.float32(1e-40)  # subnormals
    valid = np.arange(n)[None, :] < rng.integers(0, n + 1, (24, 1))
    s = jnp.asarray(np.where(valid, s, -np.inf))
    np.testing.assert_array_equal(np.asarray(topk_mask(s, k)), _ref(s, k))
