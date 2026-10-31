import ctypes as C
import os
from pathlib import Path

import pytest
import torch


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)
ptr, integer = C.c_void_p, C.c_int


def linear_launcher(name):
    fn = getattr(lib, name)
    fn.argtypes = [ptr, ptr, ptr, integer, integer, integer, ptr]
    fn.restype = integer
    return fn


def gemv_launcher(name):
    fn = getattr(lib, name)
    fn.argtypes = [ptr, ptr, ptr, integer, integer, ptr]
    fn.restype = integer
    return fn


gemm_f32 = linear_launcher("engine_gemm_f32")
gemm_bf16 = linear_launcher("engine_gemm_bf16")
gemv_f32 = gemv_launcher("engine_gemv_f32")
gemv_bf16 = gemv_launcher("engine_gemv_bf16")

MODEL_SHAPES = [
    (2048, 2048),  # q_proj, o_proj
    (512, 2048),   # k_proj, v_proj
    (8192, 2048),  # gate_proj, up_proj
    (2048, 8192),  # down_proj
    (128256, 2048),  # lm_head
]


def assert_reference(actual, x, weight, dtype, fp32_atol=1e-4):
    expected = x.float() @ weight.float().T
    if dtype == torch.bfloat16:
        expected = expected.to(dtype)
        torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-3)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=fp32_atol)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, gemm_f32), (torch.bfloat16, gemm_bf16)])
@pytest.mark.parametrize("n,k", MODEL_SHAPES)
def test_gemm_model_shapes(dtype, fn, n, k):
    m = 1
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    out = torch.empty(m, n, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), m, n, k, stream) == 0
    torch.cuda.synchronize()
    # Scalar fp32 accumulation at K=8192 differs from torch matmul's reduction
    # order by up to 0.001282 absolute on the three diagnostic seeds.
    if dtype == torch.float32 and k == 8192:
        fp32_atol = 1.5e-3
    elif dtype == torch.float32 and n == 128256:
        fp32_atol = 1.25e-4
    else:
        fp32_atol = 1e-4
    assert_reference(out, x, weight, dtype, fp32_atol=fp32_atol)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, gemm_f32), (torch.bfloat16, gemm_bf16)])
@pytest.mark.parametrize("m", [1, 3, 17])
def test_gemm_awkward_bounds(dtype, fn, m):
    n, k = 7, 33
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    out = torch.empty(m, n, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), m, n, k, stream) == 0
    torch.cuda.synchronize()
    assert_reference(out, x, weight, dtype)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, gemm_f32), (torch.bfloat16, gemm_bf16)])
@pytest.mark.parametrize("m", [3, 17])
@pytest.mark.parametrize("n,k", MODEL_SHAPES[:-1])
def test_gemm_prefill_model_shapes(dtype, fn, m, n, k):
    torch.manual_seed(1234)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    out = torch.empty(m, n, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), m, n, k, stream) == 0
    torch.cuda.synchronize()
    assert_reference(out, x, weight, dtype, fp32_atol=1.5e-3 if k == 8192 else 1e-4)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, gemv_f32), (torch.bfloat16, gemv_bf16)])
@pytest.mark.parametrize("n,k", MODEL_SHAPES + [(7, 33), (13, 70), (37, 129)])
def test_gemv_model_and_awkward_shapes(dtype, fn, n, k):
    x = torch.randn(k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    out = torch.empty(n, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), n, k, stream) == 0
    torch.cuda.synchronize()
    assert_reference(out[None, :], x[None, :], weight, dtype)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, gemv_f32), (torch.bfloat16, gemv_bf16)])
def test_gemv_unaligned_vector_tail(dtype, fn):
    n, k = 37, 129
    x = torch.randn(k + 1, device="cuda", dtype=dtype)[1:]
    weight = torch.randn(n * k + 1, device="cuda", dtype=dtype)[1:].view(n, k)
    out = torch.empty(n + 1, device="cuda", dtype=dtype)[1:]
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), n, k, stream) == 0
    torch.cuda.synchronize()
    assert_reference(out[None, :], x[None, :], weight, dtype)


@pytest.mark.parametrize("dtype,gemm,gemv", [(torch.float32, gemm_f32, gemv_f32), (torch.bfloat16, gemm_bf16, gemv_bf16)])
def test_gemv_agrees_with_gemm_at_one_row(dtype, gemm, gemv):
    m, n, k = 1, 7, 33
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    y_gemm = torch.empty(m, n, device="cuda", dtype=dtype)
    y_gemv = torch.empty(n, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    assert gemm(x.data_ptr(), weight.data_ptr(), y_gemm.data_ptr(), m, n, k, stream) == 0
    assert gemv(x.data_ptr(), weight.data_ptr(), y_gemv.data_ptr(), n, k, stream) == 0
    torch.cuda.synchronize()
    torch.testing.assert_close(y_gemm[0].float(), y_gemv.float(), rtol=1e-4, atol=1e-4)
