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
"""ModelOpt MIXED_PRECISION for TPU.

The per-layer algorithm comes from the checkpoint's `quantized_layers`
(parsed by upstream). Supported on TPU: NVFP4 routed experts, kept 4-bit in
HBM through the NVFP4 MoE method (W4A16, dequant in VMEM); everything not
listed is unquantized. Other per-layer algorithms raise.

Example: nvidia/Qwen3.8-Flash-Next-NVFP4 (NVFP4 experts; its FP8 n-gram table
is not a vLLM layer and is handled by the model).
"""

from typing import Optional

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import \
    ModelOptMixedPrecisionConfig
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding

from tpu_inference.layers.common.quant_methods import MODELOPT_MIXED
from tpu_inference.layers.vllm.quantization.configs import VllmQuantConfig
from tpu_inference.layers.vllm.quantization.nvfp4 import VllmNvfp4MoEMethod
from tpu_inference.layers.vllm.quantization.unquantized import (
    VllmUnquantizedEmbeddingMethod, VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod)

_NVFP4_MOE_ALGOS = ("NVFP4", "W4A16_NVFP4")


@register_quantization_config(MODELOPT_MIXED)
class VllmModelOptMixedConfig(ModelOptMixedPrecisionConfig, VllmQuantConfig):

    @classmethod
    def get_name(cls):
        return MODELOPT_MIXED

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, Attention):
            return None
        algo = (None if self.is_layer_excluded(prefix) else
                self._resolve_quant_algo(prefix))
        if isinstance(layer, LinearBase):
            if algo is None:
                return VllmUnquantizedLinearMethod(
                    self.get_linear_config(layer))
            raise NotImplementedError(
                f"modelopt_mixed on TPU: {algo} linear layers ({prefix})")
        if isinstance(layer, RoutedExperts):
            if algo is None:
                return VllmUnquantizedFusedMoEMethod(self.get_moe_config(layer),
                                                     self.mesh)
            if algo.upper() in _NVFP4_MOE_ALGOS:
                layer.moe_config = self.get_moe_config(layer)
                return VllmNvfp4MoEMethod(self.nvfp4_config, layer, self.mesh)
            raise NotImplementedError(
                f"modelopt_mixed on TPU: {algo} MoE layers ({prefix})")
        if isinstance(layer, VocabParallelEmbedding):
            return VllmUnquantizedEmbeddingMethod(self.mesh)
        return None
