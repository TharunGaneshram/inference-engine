import ctypes as C
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
from engine import EngineLibrary, load_config, model_path_from_env


ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "artifacts" / "reference.npz"


@pytest.fixture(scope="session")
def model_engine():
    model_path = model_path_from_env()
    api = EngineLibrary()
    handle = api.create(load_config(model_path))
    try:
        api.upload_model(handle, model_path)
        yield api, handle
    finally:
        api.lib.engine_destroy(handle)


@pytest.fixture(scope="session")
def goldens():
    assert REFERENCE.exists(), "run scripts/make_reference.py first"
    return np.load(REFERENCE)


def token_buffer(ids):
    return (C.c_int * len(ids))(*[int(v) for v in ids])


def residual(api, handle, hidden):
    out = torch.empty(hidden, dtype=torch.bfloat16)
    api.check(api.lib.engine_read_residual(handle, out.data_ptr(), hidden))
    return out.float().numpy()


def error_metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.abs(delta).max())
    scale = max(float(np.abs(expected).max()), 1e-12)
    cosine = float(np.dot(actual.ravel(), expected.ravel()) /
                    (np.linalg.norm(actual) * np.linalg.norm(expected) + 1e-30))
    return {"max_abs": max_abs, "relative_to_reference_max": max_abs / scale, "cosine": cosine}


@pytest.mark.e2e
def test_level2_layer_residual_diff(model_engine, goldens):
    api, handle = model_engine
    ids = goldens["single_ids"]
    input_ids = token_buffer(ids)
    api.check(api.lib.engine_reset(handle))
    config = load_config(model_path_from_env())
    metrics = []
    for layer_count in range(config["num_hidden_layers"] + 1):
        api.lib.engine_forward(handle, input_ids, len(ids), layer_count)
        actual = residual(api, handle, config["hidden_size"])
        expected = (
            goldens["single_pre_final"][0]
            if layer_count == config["num_hidden_layers"]
            else goldens[f"single_hidden_{layer_count}"][0]
        )
        row = error_metrics(actual, expected)
        metrics.append(row)
    (ROOT / "artifacts" / "level2_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    for layer_count, row in enumerate(metrics):
        threshold = 6.1e-2 if layer_count == config["num_hidden_layers"] else 3e-2
        assert row["relative_to_reference_max"] <= threshold, (
            f"residual over threshold at layer_count={layer_count}: {row} (limit {threshold})"
        )


@pytest.mark.e2e
@pytest.mark.parametrize(
    "prompt_name",
    ["single", "short", "medium_500", "repetitive_128", "boundary_2049"],
)
def test_level3_prefill_logits(model_engine, goldens, prompt_name):
    api, handle = model_engine
    ids = goldens[f"{prompt_name}_ids"]
    output = (C.c_int * 1)()
    input_ids = token_buffer(ids)
    count = api.lib.engine_generate(handle, input_ids, len(ids), 1, output)
    if count < 0:
        raise RuntimeError(api.error())
    logits = np.empty(128256, dtype=np.float32)
    api.check(api.lib.engine_last_logits(handle, logits.ctypes.data_as(C.POINTER(C.c_float))))
    expected = goldens[f"{prompt_name}_last_logits"]
    row = error_metrics(logits, expected)
    top1_match = int(int(np.argmax(logits)) == int(goldens[f"{prompt_name}_top1"][-1]))
    row["top1_match"] = top1_match
    (ROOT / "artifacts" / f"level3_{prompt_name}.json").write_text(
        json.dumps(row, indent=2) + "\n"
    )
    assert row["cosine"] > 0.999, f"{prompt_name} logits: {row}"
    assert top1_match == 1, f"{prompt_name} top-1 mismatch: {row}"


@pytest.mark.e2e
def test_level4_teacher_forced_decode(model_engine, goldens):
    api, handle = model_engine
    prompt = goldens["short_ids"]
    generated_ids = goldens["short_greedy_ids"]
    reference_logits = goldens["short_greedy_logits"]
    input_ids = token_buffer(prompt)
    first = api.lib.engine_generate(handle, input_ids, len(prompt), 1, (C.c_int * 1)())
    if first < 0:
        raise RuntimeError(api.error())

    logits = np.empty(128256, dtype=np.float32)
    metrics = []
    for step, expected in enumerate(reference_logits):
        api.check(api.lib.engine_last_logits(handle, logits.ctypes.data_as(C.POINTER(C.c_float))))
        row = error_metrics(logits, expected)
        row["top1_match"] = int(int(np.argmax(logits)) == int(np.argmax(expected)))
        metrics.append(row)
        if step + 1 < len(reference_logits):
            token = api.lib.engine_decode(handle, int(generated_ids[step]))
            if token < 0:
                raise RuntimeError(api.error())
    max_cosine_error = max(1.0 - row["cosine"] for row in metrics)
    agreement = sum(row["top1_match"] for row in metrics) / len(metrics)
    summary = {
        "steps": len(metrics),
        "max_abs": max(row["max_abs"] for row in metrics),
        "min_cosine": min(row["cosine"] for row in metrics),
        "top1_agreement": agreement,
        "first_divergence": next((i for i, row in enumerate(metrics) if not row["top1_match"]), None),
    }
    (ROOT / "artifacts" / "level4_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    assert max_cosine_error < 1e-3, summary
    assert agreement >= 0.99, summary


@pytest.mark.e2e
def test_level5_greedy_generation(model_engine, goldens):
    api, handle = model_engine
    prompt = [int(v) for v in goldens["short_ids"]]
    expected = [int(v) for v in goldens["short_greedy_ids"]]
    actual = api.generate(handle, prompt, 64)
    first_divergence = next(
        (i for i, (got, want) in enumerate(zip(actual, expected)) if got != want), None
    )
    summary = {
        "expected_tokens": len(expected),
        "generated_tokens": len(actual),
        "first_divergence": first_divergence,
        "matching_prefix_tokens": first_divergence if first_divergence is not None else min(len(actual), len(expected)),
    }
    (ROOT / "artifacts" / "level5_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    assert len(actual) == min(64, len(expected)), summary
    assert actual[0] == expected[0], summary


def read_logits(api, handle):
    logits = np.empty(128256, dtype=np.float32)
    api.check(api.lib.engine_last_logits(handle, logits.ctypes.data_as(C.POINTER(C.c_float))))
    return logits


@pytest.mark.e2e
def test_edge_one_token_and_bos_only(model_engine, goldens):
    api, handle = model_engine
    config = load_config(model_path_from_env())
    bos = [config["bos_token_id"]]
    for ids in ([int(goldens["single_ids"][0])], bos):
        generated = api.generate(handle, ids, 1)
        assert len(generated) == 1


@pytest.mark.e2e
def test_edge_exact_prefill_boundary_lengths(model_engine, goldens):
    api, handle = model_engine
    boundary = [int(v) for v in goldens["boundary_2049_ids"]]
    for length in (2048, 2049):
        generated = api.generate(handle, boundary[:length], 1)
        assert len(generated) == 1


@pytest.mark.e2e
def test_prefill_decode_and_chunked_agreement(model_engine, goldens):
    api, handle = model_engine
    ids = [int(v) for v in goldens["boundary_2049_ids"][:12]]
    full = token_buffer(ids)
    api.check(api.lib.engine_reset(handle))
    if api.lib.engine_prefill(handle, full, len(ids)) < 0:
        raise RuntimeError(api.error())
    full_logits = read_logits(api, handle)

    api.check(api.lib.engine_reset(handle))
    prefix = token_buffer(ids[:11])
    if api.lib.engine_prefill(handle, prefix, 11) < 0:
        raise RuntimeError(api.error())
    token = api.lib.engine_decode(handle, ids[11])
    if token < 0:
        raise RuntimeError(api.error())
    decode_logits = read_logits(api, handle)
    decode_metrics = error_metrics(decode_logits, full_logits)
    assert decode_metrics["cosine"] > 0.999
    assert int(np.argmax(decode_logits)) == int(np.argmax(full_logits))

    api.check(api.lib.engine_reset(handle))
    first_chunk = token_buffer(ids[:5])
    second_chunk = token_buffer(ids[5:])
    api.lib.engine_prefill(handle, first_chunk, 5)
    api.lib.engine_prefill(handle, second_chunk, 7)
    chunked_logits = read_logits(api, handle)
    chunk_metrics = error_metrics(chunked_logits, full_logits)
    assert chunk_metrics["cosine"] > 0.999
    assert int(np.argmax(chunked_logits)) == int(np.argmax(full_logits))
    (ROOT / "artifacts" / "prefill_agreement.json").write_text(
        json.dumps({"decode": decode_metrics, "chunked": chunk_metrics}, indent=2) + "\n"
    )


@pytest.mark.e2e
def test_max_seq_stop_and_long_prompt_rejection(model_engine, goldens):
    api, handle = model_engine
    seed = [int(v) for v in goldens["boundary_2049_ids"]]
    full_prompt = (seed * (4096 // len(seed) + 1))[:4096]
    generated = api.generate(handle, full_prompt, 4)
    assert len(generated) == 1
    too_long = token_buffer(full_prompt + [full_prompt[-1]])
    out = (C.c_int * 1)()
    status = api.lib.engine_generate(handle, too_long, len(too_long), 1, out)
    assert status < 0
    assert "prompt exceeds max_seq" in api.error()


@pytest.mark.e2e
def test_generation_repeatability_after_reset(model_engine, goldens):
    api, handle = model_engine
    prompt = [int(v) for v in goldens["short_ids"]]
    first = api.generate(handle, prompt, 8)
    first_logits = read_logits(api, handle).copy()
    second = api.generate(handle, prompt, 8)
    second_logits = read_logits(api, handle)
    assert first == second
    assert np.array_equal(first_logits, second_logits)
