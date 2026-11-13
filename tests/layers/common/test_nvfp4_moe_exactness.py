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
"""NVFP4 MoE weight processing must not change the checkpoint's values.

NVFP4 stores e2m1 values with one scale per 16 inputs. The generic MoE
weight processing re-quantizes them to e2m1 with one scale per whole row,
which moved every expert's output by about 15% on Qwen3.8 Flash-Next (found
by a layer-by-layer bisect against transformers). These tests reconstruct the
dense weights after processing and compare them with the checkpoint's, and
check that the NVFP4 MoE method keeps the checkpoint blocks by default.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from tpu_inference.layers.common.moe import MoEBackend
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, _process_quantized_moe_weights_impl)

E2M1_VALUES = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3,
                        -4, -6], np.float32)


def _nvfp4(rng, e, n, k):
    """Random e2m1 weights [e, n, k] with per-16 fp32 block scales."""
    codes = rng.integers(0, 16, size=(e, n, k))
    w = jnp.asarray(E2M1_VALUES[codes], jnp.float4_e2m1fn)
    scale = rng.lognormal(-6, 1, size=(e, n, k // 16)).astype(np.float32)
    return w, jnp.asarray(scale)


def _dense_ref(w, scale):
    """[e, n, k] -> dense [e, k, n] (the GMM layout: contraction first)."""
    e, n, k = w.shape
    x = np.asarray(w.astype(jnp.float32)).reshape(e, n, k // 16, 16)
    x = x * np.asarray(scale)[..., None]
    return x.reshape(e, n, k).transpose(0, 2, 1)


def _dense_processed(w, scale):
    """Processed weight [e, k, n] with scale [e, blocks, 1, n] -> dense."""
    e, k, n = w.shape
    blocks = scale.shape[1]
    x = np.asarray(w.astype(jnp.float32)).reshape(e, blocks, k // blocks, n)
    x = x * np.asarray(scale.astype(jnp.float32))
    return x.reshape(e, k, n)


class TestNvfp4MoeExactness(unittest.TestCase):

    def _run(self, disable_requant):
        rng = np.random.default_rng(0)
        e, hidden, inter = 4, 256, 128
        w13, s13 = _nvfp4(rng, e, 2 * inter, hidden)
        w2, s2 = _nvfp4(rng, e, hidden, inter)
        mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1),
                    ("data", "model"))
        out = _process_quantized_moe_weights_impl(
            FusedMoEWeights(w13_weight=w13,
                            w13_weight_scale=s13,
                            w13_bias=None,
                            w2_weight=w2,
                            w2_weight_scale=s2,
                            w2_bias=None),
            moe_backend=MoEBackend.GMM_TP,
            mesh=mesh,
            activation="silu",
            weight_block_size=(1, 16),
            desired_quant_dtype=jnp.float4_e2m1fn,
            requant_block_size=None,
            disable_weight_requantization=disable_requant,
        )
        errs = []
        for ref_w, ref_s, got_w, got_s in ((w13, s13, out.w13_weight,
                                            out.w13_weight_scale),
                                           (w2, s2, out.w2_weight,
                                            out.w2_weight_scale)):
            ref = _dense_ref(ref_w, ref_s)
            got = _dense_processed(got_w, got_s)
            errs.append(np.linalg.norm(got - ref) / np.linalg.norm(ref))
        return max(errs)

    def test_nvfp4_without_requantization_is_exact(self):
        self.assertLess(self._run(disable_requant=True), 1e-6)

    def test_generic_requantization_is_lossy(self):
        # Re-quantizing to one e2m1 scale per row: ~13% weight error here.
        self.assertGreater(self._run(disable_requant=False), 0.05)

    def test_nvfp4_method_keeps_checkpoint_blocks_by_default(self):
        from unittest import mock

        from tpu_inference.layers.vllm.quantization import nvfp4
        with mock.patch.object(nvfp4.envs, "DISABLE_WEIGHT_REQUANTIZATION",
                               False), \
                mock.patch.object(nvfp4.envs, "MOE_REQUANTIZE_WEIGHT_DTYPE",
                                  ""), \
                mock.patch.object(nvfp4.envs, "MOE_REQUANTIZE_BLOCK_SIZE",
                                  None):
            self.assertTrue(nvfp4.nvfp4_moe_keeps_checkpoint_blocks())


if __name__ == "__main__":
    unittest.main()
