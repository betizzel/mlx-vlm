"""Inkling2Loop (Inkling 2 Mini) contracts: loop schedule, caches, DSpark drafting."""

import json
from copy import deepcopy
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_vlm.generate.ar import _make_cache, generate_step
from mlx_vlm.tests.test_models import model_case

DATA = json.loads(Path(__file__).with_name("model_cases.json").read_text())[
    "speculative"
]


def loop_config():
    config = deepcopy(model_case("inkling2")["config"])
    text = config["text_config"]
    text.update(
        num_hidden_layers=6,
        local_layer_ids=[0, 1, 3, 4],
        loop_config={"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 4},
    )
    config["model_type"] = "inkling2_loop"
    text["model_type"] = "inkling2_loop_text"
    mtp = deepcopy(DATA["inkling2_dspark"])
    mtp["target_layer_ids"] = [1, 5, 10]
    config["mtp_config"] = mtp
    return config


def loop_case(dtype=mx.float32):
    from mlx_vlm.models.inkling2_loop import Model, ModelConfig
    from mlx_vlm.speculative.drafters.inkling2_dspark import Model as Draft
    from mlx_vlm.speculative.drafters.inkling2_dspark import ModelConfig as DraftConfig

    mx.random.seed(5)
    config = loop_config()
    target = Model(ModelConfig.from_dict(config))
    draft = Draft(
        DraftConfig(
            text_config={**config["text_config"], "mtp_config": config["mtp_config"]}
        )
    )
    lm = target.language_model
    lm.model.message_input.query = 0.5 * mx.random.normal(
        lm.model.message_input.query.shape
    )
    for layer in lm.model.layers:
        layer.mlp.e_score_correction_bias = mx.arange(4, dtype=mx.float32)
    target.set_dtype(dtype)
    draft.set_dtype(dtype)
    target.eval()
    draft.eval()
    return target, draft


def test_loop_effective_depth_and_captures():
    target, draft = loop_case()
    lm = target.language_model
    assert lm.config.total_num_hidden_layers == 10
    assert len(lm.make_cache()) == 6 + 2 * 2
    ids = mx.array([[1, 2, 3, 4, 5]])
    out = lm(ids, capture_layer_ids=[0, 5, 10])
    full = lm(ids)
    assert len(out.hidden_states) == 3
    np.testing.assert_allclose(
        np.array(lm._logits_from_norm(out.hidden_states[-1])),
        np.array(full.logits),
        atol=1e-5,
    )
    for invalid in ([11], [-1], [0, 0]):
        with pytest.raises(ValueError):
            lm(ids, capture_layer_ids=invalid)


def test_loop_cached_decode_matches_full():
    target, _ = loop_case()
    lm = target.language_model
    ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    full = lm(ids).logits
    cache = lm.make_cache()
    parts = [lm(ids[:, :6], cache=cache).logits]
    for i in range(6, ids.shape[1]):
        parts.append(lm(ids[:, i : i + 1], cache=cache).logits)
    np.testing.assert_allclose(
        np.array(mx.concatenate(parts, axis=1)), np.array(full), atol=2e-5
    )


def test_head_logits_are_fp32_for_leading_candidates(monkeypatch):
    from mlx_vlm.models.inkling2 import language

    monkeypatch.setattr(language, "_HEAD_EXACT", 8)
    target, _ = loop_case(mx.bfloat16)
    lm = target.language_model
    mx.random.seed(3)
    h = mx.random.normal((1, 3, lm.config.hidden_size)).astype(mx.bfloat16)
    actual = np.array(lm._logits_from_norm(h))
    reference = h.astype(mx.float32) @ lm.lm_head.weight.astype(mx.float32).T
    cap = lm.config.final_logit_softcapping
    if cap:
        reference = mx.tanh(reference / cap) * cap
    reference = np.array(reference[..., : lm.config.unpadded_vocab_size])
    leading = np.argsort(-reference, axis=-1)[..., :4]
    np.testing.assert_allclose(
        np.take_along_axis(actual, leading, -1),
        np.take_along_axis(reference, leading, -1),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(actual.argmax(-1), reference.argmax(-1))
    # The tail keeps BF16-rounded scores: close, but not the FP32 values.
    np.testing.assert_allclose(actual, reference, rtol=1e-2, atol=1e-2)
    assert not np.allclose(actual, reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("retained", [[1, 3], [4, 2], [0, 0]])
def test_loop_speculative_rollback(retained):
    target, draft = loop_case()
    lm = target.language_model
    cache = _make_cache(lm, [0, 0])
    prefix = mx.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mx.eval(lm(prefix, cache=cache).logits)
    controls = [[c.extract(row) for c in cache] for row in range(2)]
    block = mx.array([[11, 12, 13, 14], [15, 16, 17, 18]])
    out = lm(
        block,
        cache=cache,
        capture_layer_ids=draft.config.target_layer_ids,
        speculative_verify=True,
    )
    mx.eval(out.logits)
    if retained == [0, 0]:
        out.gdn_states.abort()
    else:
        out.gdn_states.commit(retained)
    for row, length in enumerate(retained):
        for i in range(length):
            mx.eval(lm(block[row : row + 1, i : i + 1], cache=controls[row]).logits)
        actual = [c.extract(row) for c in cache]
        np.testing.assert_allclose(
            np.array(lm(mx.array([[19]]), cache=actual).logits),
            np.array(lm(mx.array([[19]]), cache=controls[row]).logits),
            atol=3e-5,
            rtol=3e-5,
        )


def generated(target, drafter=None):
    options = {} if drafter is None else dict(draft_model=drafter, draft_kind="dflash")
    return [
        int(token.item()) if hasattr(token, "item") else int(token)
        for token, _ in generate_step(
            mx.array([[1, 2, 3, 4]]), target, None, None, max_tokens=16,
            temperature=0, **options,
        )
    ]


def test_loop_dspark_greedy_is_lossless():
    target, draft = loop_case()
    expected = generated(target)
    assert generated(target, draft) == expected
    assert draft.draft_lens and sum(draft.draft_lens) > 0


def test_inkling_tool_parser():
    from mlx_vlm.tools.extractor import process_tool_calls
    from mlx_vlm.tools.parsers import inkling
    from mlx_vlm.tools.registry import _infer_tool_parser

    assert _infer_tool_parser("{{ '<|content_invoke_tool_json|>' }}") == "inkling"
    out = (
        "Checking.<|message_model|>get_weather<|content_invoke_tool_json|>"
        '{"args": {"city": "Athens"}, "name": "get_weather"}<|end_message|>'
    )
    result = process_tool_calls(out, inkling, tools=[])
    assert result.calls[0]["function"]["name"] == "get_weather"
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"city": "Athens"}
    assert result.remaining_text == "Checking."
    plain = out.replace("<|message_model|>", "")
    assert process_tool_calls(plain, inkling, tools=[]).remaining_text == "Checking."
