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
"""Async scheduling + spec decode: a request whose sampled token was discarded
(here the recompute of a preempted request, past its prompt but short of its
known output tokens) must not get a placeholder, or the next step treats its
recompute chunk as a verify window."""
from types import SimpleNamespace

import numpy as np

from tpu_inference.runner.tpu_runner import TPUModelRunner


def _runner():
    input_batch = SimpleNamespace(
        req_ids=["decode", "recompute"],
        req_id_to_index={"decode": 0, "recompute": 1},
        num_tokens_no_spec=np.array([100, 0], dtype=np.int64),
        num_tokens=np.array([100, 0], dtype=np.int64),
        # "recompute": prompt 50 tokens, 30 output tokens known, 60 computed.
        num_computed_tokens_cpu=np.array([99, 60], dtype=np.int64),
        num_prompt_tokens=np.array([90, 50], dtype=np.int64),
    )
    return SimpleNamespace(
        input_batch=input_batch,
        max_model_len=1024,
        dp_size=1,
        max_num_reqs=4,
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )


def test_discarded_request_gets_no_placeholder():
    runner = _runner()
    decode_state = SimpleNamespace(req_id="decode", output_token_ids=[])
    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={"decode": [1, 2, 3]})
    spec_decode_metadata = SimpleNamespace(
        req_ids_dp={0: ["decode", "recompute"]})

    placeholders = TPUModelRunner._update_placeholder(
        runner,
        discard_sampled_tokens_req_indices=[1],
        request_seq_lens=[(0, decode_state, 104)],
        scheduler_output=scheduler_output,
        spec_decode_metadata=spec_decode_metadata)

    assert placeholders == {"decode": 0}

    # Next step: the recompute request schedules its remaining 20 known
    # tokens; only the decode request's verify window is substituted.
    runner._pre_async_results = SimpleNamespace(
        placeholder_req_id_to_index=placeholders)
    cur, pre = TPUModelRunner._prepare_async_token_substitution_indices(
        runner, {0: ["decode", "recompute"]}, {0: [4, 20]}, 32, 1)
    assert cur[0] == [0, 1, 2, 3]
    assert pre[0] == [0, 1, 2, 3]
