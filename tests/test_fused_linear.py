import ctypes as C
import os
from pathlib import Path

import pytest
import torch


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)


def launcher(name, args):
    fn = getattr(lib, name)
    fn.argtypes = args
    fn.restype = C.c_int
    return fn


k_cache = launcher(
    "engine_project_rope_cache_bf16",
    [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
     C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p],
)
f32_k_cache = launcher(
    "engine_project_rope_cache_f32",
    [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
     C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p],
)
gemm = launcher("engine_gemm_bf16", [C.c_void_p, C.c_void_p, C.c_void_p,
                                       C.c_int, C.c_int, C.c_int, C.c_void_p])
gemm_f32 = launcher("engine_gemm_f32", [C.c_void_p, C.c_void_p, C.c_void_p,
                                          C.c_int, C.c_int, C.c_int, C.c_void_p])
gemv = launcher("engine_gemv_bf16", [C.c_void_p, C.c_void_p, C.c_void_p,
                                       C.c_int, C.c_int, C.c_void_p])
gemv_f32 = launcher("engine_gemv_f32", [C.c_void_p, C.c_void_p, C.c_void_p,
                                          C.c_int, C.c_int, C.c_void_p])
rope = launcher("engine_rope_bf16", [C.c_void_p, C.c_void_p, C.c_void_p,
                                       C.c_void_p, C.c_int, C.c_int,
                                       C.c_int, C.c_int, C.c_int, C.c_void_p])
rope_f32 = launcher("engine_rope_f32", [C.c_void_p, C.c_void_p, C.c_void_p,
                                          C.c_void_p, C.c_int, C.c_int,
                                          C.c_int, C.c_int, C.c_int, C.c_void_p])
fused_swiglu = launcher(
    "engine_linear_swiglu_bf16",
    [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
     C.c_int, C.c_int, C.c_int, C.c_void_p],
)
fused_swiglu_f32 = launcher(
    "engine_linear_swiglu_f32",
    [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
     C.c_int, C.c_int, C.c_int, C.c_void_p],
)
swiglu = launcher("engine_swiglu_bf16", [C.c_void_p, C.c_void_p,
                                          C.c_void_p, C.c_size_t, C.c_void_p])
swiglu_f32 = launcher("engine_swiglu_f32", [C.c_void_p, C.c_void_p,
                                              C.c_void_p, C.c_size_t, C.c_void_p])


@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("start_pos", [0, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_k_rope_cache_matches_separate_path(rows, start_pos, dtype):
    n_kv_heads, head_dim, hidden = 8, 64, 2048
    output_size = n_kv_heads * head_dim
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(output_size, hidden, device="cuda", dtype=dtype)
    actual = torch.empty(rows, output_size, device="cuda", dtype=dtype)
    frequencies = torch.arange(head_dim // 2, device="cuda").float()[None, :]
    inv_freq = torch.pow(10000.0, -2 * frequencies / head_dim)
    # The launcher consumes [position, half_dim] RoPE tables.
    cos_table = torch.zeros(start_pos + rows, head_dim // 2, device="cuda")
    sin_table = torch.zeros_like(cos_table)
    positions = torch.arange(start_pos, start_pos + rows, device="cuda").float()[:, None]
    theta = positions * inv_freq
    cos_table[start_pos:] = torch.cos(theta)
    sin_table[start_pos:] = torch.sin(theta)
    projected = torch.empty(rows, output_size, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    gemm_fn, gemv_fn = (gemm_f32, gemv_f32) if dtype == torch.float32 else (gemm, gemv)
    rope_fn = rope_f32 if dtype == torch.float32 else rope
    cache_fn = f32_k_cache if dtype == torch.float32 else k_cache
    if rows == 1:
        assert gemv_fn(x.data_ptr(), weight.data_ptr(), projected.data_ptr(),
                    output_size, hidden, stream) == 0
    else:
        assert gemm_fn(x.data_ptr(), weight.data_ptr(), projected.data_ptr(),
                    rows, output_size, hidden, stream) == 0
    torch.cuda.synchronize()
    expected = projected.clone()
    q = torch.empty(rows, head_dim, device="cuda", dtype=dtype)
    assert rope_fn(q.data_ptr(), expected.data_ptr(), cos_table.data_ptr(),
                sin_table.data_ptr(), start_pos, rows, 1, n_kv_heads,
                head_dim, stream) == 0
    torch.cuda.synchronize()
    assert cache_fn(x.data_ptr(), weight.data_ptr(), actual.data_ptr(),
                   cos_table.data_ptr(), sin_table.data_ptr(), start_pos, rows,
                   n_kv_heads, head_dim, hidden, stream) == 0
    torch.cuda.synchronize()
    if dtype == torch.bfloat16:
        error = (actual.float() - expected.float()).abs()
        scale = expected.float().abs().max().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert error.max() / scale < 1e-2
        assert cosine > 0.99999
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_gate_up_swiglu_matches_separate_path(rows, dtype):
    torch.manual_seed(1234)
    hidden, intermediate = 2048, 8192
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    gate_weight = torch.randn(intermediate, hidden, device="cuda", dtype=dtype)
    up_weight = torch.randn_like(gate_weight)
    gate = torch.empty(rows, intermediate, device="cuda", dtype=dtype)
    up = torch.empty_like(gate)
    actual = torch.empty_like(gate)
    expected = torch.empty_like(gate)
    stream = torch.cuda.current_stream().cuda_stream
    gemm_fn = gemm_f32 if dtype == torch.float32 else gemm
    swiglu_fn = swiglu_f32 if dtype == torch.float32 else swiglu
    fused_fn = fused_swiglu_f32 if dtype == torch.float32 else fused_swiglu
    if rows == 1:
        gemv_fn = gemv_f32 if dtype == torch.float32 else gemv
        assert gemv_fn(x.data_ptr(), gate_weight.data_ptr(), gate.data_ptr(),
                       intermediate, hidden, stream) == 0
        assert gemv_fn(x.data_ptr(), up_weight.data_ptr(), up.data_ptr(),
                       intermediate, hidden, stream) == 0
    else:
        assert gemm_fn(x.data_ptr(), gate_weight.data_ptr(), gate.data_ptr(),
                       rows, intermediate, hidden, stream) == 0
        assert gemm_fn(x.data_ptr(), up_weight.data_ptr(), up.data_ptr(),
                       rows, intermediate, hidden, stream) == 0
    assert swiglu_fn(gate.data_ptr(), up.data_ptr(), expected.data_ptr(),
                  rows * intermediate, stream) == 0
    assert fused_fn(x.data_ptr(), gate_weight.data_ptr(), up_weight.data_ptr(),
                        actual.data_ptr(), rows, intermediate, hidden, stream) == 0
    torch.cuda.synchronize()
    if dtype == torch.bfloat16:
        error = (actual.float() - expected.float()).abs()
        scale = expected.float().abs().max().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert error.max() / scale < 1e-2
        assert cosine > 0.99999
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_linear_awkward_edges(dtype):
    rows, hidden, n_kv, head_dim = 3, 33, 2, 64
    n = n_kv * head_dim
    x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    weight = torch.randn(n, hidden, device="cuda", dtype=dtype)
    actual_k = torch.empty(rows, n, device="cuda", dtype=dtype)
    projected = torch.empty_like(actual_k)
    cos = torch.randn(12, head_dim // 2, device="cuda")
    sin = torch.randn_like(cos)
    q = torch.empty(rows, head_dim, device="cuda", dtype=dtype)
    gemm_fn, rope_fn = (gemm_f32, rope_f32) if dtype == torch.float32 else (gemm, rope)
    k_fn = f32_k_cache if dtype == torch.float32 else k_cache
    stream = torch.cuda.current_stream().cuda_stream
    assert gemm_fn(x.data_ptr(), weight.data_ptr(), projected.data_ptr(),
                   rows, n, hidden, stream) == 0
    assert rope_fn(q.data_ptr(), projected.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                   7, rows, 1, n_kv, head_dim, stream) == 0
    assert k_fn(x.data_ptr(), weight.data_ptr(), actual_k.data_ptr(), cos.data_ptr(),
                sin.data_ptr(), 7, rows, n_kv, head_dim, hidden, stream) == 0

    intermediate = 70
    gate_weight = torch.randn(intermediate, hidden, device="cuda", dtype=dtype)
    up_weight = torch.randn_like(gate_weight)
    gate = torch.empty(rows, intermediate, device="cuda", dtype=dtype)
    up = torch.empty_like(gate)
    expected = torch.empty_like(gate)
    actual = torch.empty_like(gate)
    fused_fn = fused_swiglu_f32 if dtype == torch.float32 else fused_swiglu
    swiglu_fn = swiglu_f32 if dtype == torch.float32 else swiglu
    assert gemm_fn(x.data_ptr(), gate_weight.data_ptr(), gate.data_ptr(),
                   rows, intermediate, hidden, stream) == 0
    assert gemm_fn(x.data_ptr(), up_weight.data_ptr(), up.data_ptr(),
                   rows, intermediate, hidden, stream) == 0
    assert swiglu_fn(gate.data_ptr(), up.data_ptr(), expected.data_ptr(),
                     rows * intermediate, stream) == 0
    assert fused_fn(x.data_ptr(), gate_weight.data_ptr(), up_weight.data_ptr(),
                    actual.data_ptr(), rows, intermediate, hidden, stream) == 0
    torch.cuda.synchronize()
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual_k.float(), projected.float(), rtol=1e-2, atol=1e-3)
        delta = (actual.float() - expected.float()).abs()
        scale = expected.float().abs().max().clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        )
        assert delta.max() / scale < 1e-2
        assert cosine > 0.99999
    else:
        torch.testing.assert_close(actual_k, projected, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
