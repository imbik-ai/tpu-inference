# Copyright 2025 Google LLC
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

from typing import TYPE_CHECKING, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput

    from tpu_inference.runner.tpu_runner import TPUModelRunner


class StructuredDecodingManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

    @jax.jit(static_argnums=(0, ))
    def structured_decode_fn(self, require_struct_decoding: jax.Array,
                             grammar_bitmask: jax.Array, logits: jax.Array,
                             arange: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.any(require_struct_decoding),
            lambda: self._apply_grammar_bitmask_kernel(
                logits, grammar_bitmask, require_struct_decoding, arange),
            lambda: logits)

    @jax.jit(static_argnums=(0, ))
    def _apply_grammar_bitmask_kernel(self, logits: jax.Array,
                                      grammar_bitmask: jax.Array,
                                      require_struct_decoding: jax.Array,
                                      arange: jax.Array) -> jax.Array:

        # Unpack the bitmask for the entire batch at once.
        # grammar_bitmask: (B, N) where B=num_reqs, N=cdiv(vocab_size, 32)
        # arange: (32,)
        # (B, N, 1) and (1, 1, 32) broadcast to (B, N, 32)
        unpacked_bitmask = jnp.right_shift(grammar_bitmask[:, :, None],
                                           arange[None, None, :]) & 1 == 0

        # Reshape to (B, vocab_size) and apply to logits.
        # (B, N * 32) -> (B, vocab_size)
        unpacked_bitmask = unpacked_bitmask.reshape(
            logits.shape[0], -1)[:, :self.runner.vocab_size]

        masked_logits = jnp.where(unpacked_bitmask, -jnp.inf, logits)

        return jnp.where(require_struct_decoding, masked_logits, logits)

    def prepare_structured_decoding_input(
        self,
        logits: jax.Array,
        grammar_output: "GrammarOutput",
        scheduled_spec_decode_tokens: Optional[dict[str, list[int]]] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        if scheduled_spec_decode_tokens:
            return self._prepare_spec_decode_input(
                logits, grammar_output, scheduled_spec_decode_tokens)
        grammar_bitmask = grammar_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_reqs, _ = logits.shape

        # Reset pre-allocated tensors
        self.runner.grammar_bitmask_cpu.fill(0)
        self.runner.require_structured_out_cpu.fill(0)

        sorted_struct_requests = sorted(
            grammar_output.structured_output_request_ids)

        cumulative_mask_idx = 0
        for req_id in sorted_struct_requests:
            if req_id not in self.runner.input_batch.req_id_to_index:
                continue
            batch_index = self.runner.input_batch.req_id_to_index[req_id]
            self.runner.grammar_bitmask_cpu[batch_index] = grammar_bitmask[
                cumulative_mask_idx]
            # It's not guaranteed that all requests in this batch require
            # structured output, so create a bool tensor to represent
            # the requests that need structured output.
            self.runner.require_structured_out_cpu[batch_index] = True
            cumulative_mask_idx += 1

        (require_structured_out_cpu,
         grammar_bitmask_cpu, structured_decode_arange) = device_array(
             self.runner.mesh,
             (self.runner.require_structured_out_cpu[:num_reqs],
              self.runner.grammar_bitmask_cpu[:num_reqs],
              self.runner.structured_decode_arange))

        return (require_structured_out_cpu, grammar_bitmask_cpu,
                structured_decode_arange)

    def _prepare_spec_decode_input(
        self, logits: jax.Array, grammar_output: "GrammarOutput",
        scheduled_spec_decode_tokens: dict[str, list[int]]
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Bitmask rows for spec decode, laid out like the logits.

        With spec decode the logits hold 1 + num_draft rows per request, in
        batch order (SpeculativeDecodingManager.get_spec_decode_metadata), and
        the grammar bitmask holds `mask_stride` rows per structured request, in
        `structured_output_request_ids` order. Same mapping as vLLM's
        `apply_grammar_bitmask`.
        """
        grammar_bitmask = grammar_output.grammar_bitmask
        assert grammar_bitmask is not None
        assert self.runner.dp_size == 1, (
            "structured decoding with spec decode needs dp_size 1")
        num_rows, _ = logits.shape
        struct_req_ids = list(grammar_output.structured_output_request_ids)

        logit_index: dict[str, int] = {}
        offset = 0
        for batch_index, req_id in enumerate(
                self.runner.input_batch.req_ids[:self.runner.input_batch.
                                                num_reqs]):
            logit_index[req_id] = batch_index + offset
            offset += len(scheduled_spec_decode_tokens.get(req_id, ()))

        bitmask_rows = np.zeros((num_rows, grammar_bitmask.shape[1]),
                                dtype=np.int32)
        require_rows = np.zeros((num_rows, 1), dtype=np.bool_)
        mask_stride = grammar_bitmask.shape[0] // len(struct_req_ids)
        for i, req_id in enumerate(struct_req_ids):
            start = logit_index.get(req_id)
            if start is None:
                continue
            n = 1 + len(scheduled_spec_decode_tokens.get(req_id, ()))
            src = i * mask_stride
            bitmask_rows[start:start + n] = grammar_bitmask[src:src + n]
            require_rows[start:start + n] = True

        return device_array(
            self.runner.mesh,
            (require_rows, bitmask_rows, self.runner.structured_decode_arange))
