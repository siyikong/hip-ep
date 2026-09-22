#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the MIT License.
#

"""Gemma4 standard ONNX Attention -> decomposed HIP GQA correctness."""

import numpy as np
import pytest
from onnx import TensorProto, helper

from framework.comparator import compare_outputs
from framework.onnx_utils import make_model_from_nodes


Q_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 256
Q_HIDDEN = Q_HEADS * HEAD_DIM
KV_HIDDEN = KV_HEADS * HEAD_DIM


def make_gemma4_attention(seq_len, past_len, left_window_size=-1):
    query = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT16, [1, "sequence", Q_HIDDEN]
    )
    key = helper.make_tensor_value_info(
        "key", TensorProto.FLOAT16, [1, "sequence", KV_HIDDEN]
    )
    value = helper.make_tensor_value_info(
        "value", TensorProto.FLOAT16, [1, "sequence", KV_HIDDEN]
    )
    mask = helper.make_tensor_value_info(
        "attn_mask",
        TensorProto.FLOAT16,
        [1, 1, "sequence", "total_sequence"],
    )
    past_key = helper.make_tensor_value_info(
        "past_key",
        TensorProto.FLOAT16,
        [1, KV_HEADS, "past_sequence", HEAD_DIM],
    )
    past_value = helper.make_tensor_value_info(
        "past_value",
        TensorProto.FLOAT16,
        [1, KV_HEADS, "past_sequence", HEAD_DIM],
    )

    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT16, [1, "sequence", Q_HIDDEN]
    )
    present_key = helper.make_tensor_value_info(
        "present_key",
        TensorProto.FLOAT16,
        [1, KV_HEADS, "total_sequence", HEAD_DIM],
    )
    present_value = helper.make_tensor_value_info(
        "present_value",
        TensorProto.FLOAT16,
        [1, KV_HEADS, "total_sequence", HEAD_DIM],
    )

    node = helper.make_node(
        "Attention",
        ["query", "key", "value", "attn_mask", "past_key", "past_value"],
        ["output", "present_key", "present_value"],
        q_num_heads=Q_HEADS,
        kv_num_heads=KV_HEADS,
        is_causal=0,
        scale=1.0 / np.sqrt(HEAD_DIM),
        softcap=0.0,
    )
    return make_model_from_nodes(
        [node],
        [query, key, value, mask, past_key, past_value],
        [output, present_key, present_value],
        opset=25,
    )


@pytest.mark.parametrize(
    "seq_len,past_len,left_window_size",
    [
        (274, 0, -1),
        (274, 0, 1024),
        (1, 274, -1),
        (1, 274, 1024),
        (1, 380, -1),
        (1, 380, 1024),
        (1, 1200, 1024),
    ],
)
def test_onnx_attention_gemma4_26b(
    model_runner,
    seq_len,
    past_len,
    left_window_size,
):
    model = make_gemma4_attention(seq_len, past_len, left_window_size)
    rng = np.random.default_rng(42)
    query = rng.uniform(
        -0.5, 0.5, [1, seq_len, Q_HIDDEN]
    ).astype(np.float16)
    key = rng.uniform(
        -0.5, 0.5, [1, seq_len, KV_HIDDEN]
    ).astype(np.float16)
    value = rng.uniform(
        -0.5, 0.5, [1, seq_len, KV_HIDDEN]
    ).astype(np.float16)
    total = past_len + seq_len
    mask = np.zeros([1, 1, seq_len, total], dtype=np.float16)
    if left_window_size > 0:
        for query_index in range(seq_len):
            absolute_query = past_len + query_index
            lower = max(0, absolute_query - left_window_size + 1)
            mask[:, :, query_index, :lower] = np.float16(-65504.0)
    past_key = rng.uniform(
        -0.5,
        0.5,
        [1, KV_HEADS, past_len, HEAD_DIM],
    ).astype(np.float16)
    past_value = rng.uniform(
        -0.5,
        0.5,
        [1, KV_HEADS, past_len, HEAD_DIM],
    ).astype(np.float16)

    actual, expected = model_runner.run_sample(
        model,
        [query, key, value, mask, past_key, past_value],
    )
    for output in actual:
        assert np.isfinite(output).all()
    compare_outputs(
        actual,
        expected,
        atol=2e-2,
        rtol=1e-2,
        cos_threshold=0.999,
    )
