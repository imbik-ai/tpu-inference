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
"""quantize_nvfp4 (torch) against the ModelOpt NVFP4 checkpoint layout."""

import torch

from tpu_inference.layers.vllm.custom_ops.experimental.qwen4_exp.nvfp4_quant import \
    quantize_nvfp4

_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _dequant(packed, scale, g):
    """Checkpoint dequantization: low nibble first, sign in bit 3."""
    nib = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F],
                      dim=-1).reshape(packed.shape[0], -1)
    val = _GRID[(nib & 7).long()] * torch.where((nib & 8) > 0, -1.0, 1.0)
    sc = scale.float() * g
    return (val.reshape(val.shape[0], -1, 16) * sc[..., None]).reshape(
        val.shape[0], -1)


def test_values_on_the_grid_round_trip_exactly():
    g = torch.tensor(0.25)
    # Blocks whose max is 6 * g * s for an e4m3-exact s.
    vals = _GRID[torch.randint(0, 8, (4, 64))] * torch.where(
        torch.rand(4, 64) < 0.5, -1.0, 1.0)
    vals[:, ::16] = 6.0  # every block reaches the grid maximum
    w = vals * 2.0 * g  # block scale 2.0 (exact in e4m3)
    packed, scale = quantize_nvfp4(w, g)
    assert packed.dtype == torch.uint8 and packed.shape == (4, 32)
    assert scale.dtype == torch.float8_e4m3fn and scale.shape == (4, 4)
    assert torch.equal(_dequant(packed, scale, g), w)


def test_random_weights_error_is_nvfp4_sized():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    g = w.abs().max() / (6.0 * 448.0)
    packed, scale = quantize_nvfp4(w, g)
    err = (_dequant(packed, scale, g) - w).norm() / w.norm()
    assert err < 0.12, err  # e2m1 with per-16 scales: about 0.1 relative
    assert err > 0.01, err  # and it is a real 4-bit quantization


def test_zero_block_stays_zero():
    w = torch.zeros(2, 32)
    w[1, 16:] = 1.0
    packed, scale = quantize_nvfp4(w, torch.tensor(1.0 / 6.0))
    out = _dequant(packed, scale, torch.tensor(1.0 / 6.0))
    assert torch.equal(out[0], torch.zeros(32))
    assert torch.allclose(out[1, 16:], torch.ones(16))
