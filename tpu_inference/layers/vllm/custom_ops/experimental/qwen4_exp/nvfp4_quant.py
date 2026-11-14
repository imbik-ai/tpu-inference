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
"""NVFP4 quantization in torch (for weights a checkpoint stores in another
format): e2m1 values packed two per byte (low nibble first, sign in bit 3)
with one e4m3 scale per 16 inputs and a caller-chosen fp32 global scale,
matching ModelOpt NVFP4 checkpoints."""

import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_nvfp4(w: torch.Tensor, global_scale: torch.Tensor):
    """(O, K) float -> packed e2m1 (O, K/2) uint8 (low nibble first) and
    per-16 e4m3 scales (O, K/16), for a given fp32 global scale."""
    O, K = w.shape
    wb = w.float().reshape(O, K // 16, 16)
    bmax = wb.abs().amax(-1)
    s = (bmax / 6.0 / global_scale).clamp(max=448.0).to(torch.float8_e4m3fn)
    scale = s.float() * global_scale
    q = wb / torch.where(scale > 0, scale, 1.0)[..., None]
    mid = (E2M1_GRID[1:] + E2M1_GRID[:-1]) / 2
    idx = torch.bucketize(q.abs().clamp(max=6.0), mid).to(torch.uint8)
    nib = (idx | torch.where(q < 0, 8, 0).to(torch.uint8)).reshape(O, K)
    packed = nib[:, 0::2] | (nib[:, 1::2] << 4)
    return packed, s


