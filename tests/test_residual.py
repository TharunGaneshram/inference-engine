import ctypes as C
import os
from pathlib import Path

import pytest
import torch


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)


def _launcher(name):
    fn = getattr(lib, name)
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_void_p]
    fn.restype = C.c_int
    return fn


def _fused_launcher(name):
    fn = getattr(lib, name)
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
                   C.c_void_p, C.c_int, C.c_int, C.c_float, C.c_void_p]
    fn.restype = C.c_int
    return fn


@pytest.mark.parametrize(
    "dtype,launcher,exact",
    [
        (torch.float32, _launcher("engine_residual_f32"), False),
        (torch.bfloat16, _launcher("engine_residual_bf16"), True),
    ],
)
@pytest.mark.parametrize("size", [1, 257, 4099])
def test_residual_add_on_gpu(dtype, launcher, exact, size):
    a = torch.randn(size, device="cuda:0", dtype=dtype)
    b = torch.randn_like(a)
    out = torch.empty_like(a)
    stream = torch.cuda.current_stream().cuda_stream
    assert launcher(a.data_ptr(), b.data_ptr(), out.data_ptr(), size, stream) == 0
    torch.cuda.synchronize()
    if exact:
        assert torch.equal(out, a + b)
    else:
        torch.testing.assert_close(out, a + b, rtol=0, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 3, 17])
@pytest.mark.parametrize("alias", [False, True])
def test_fused_residual_rmsnorm(dtype, rows, alias):
    launcher = _fused_launcher(
        "engine_residual_rmsnorm_" + ("f32" if dtype == torch.float32 else "bf16")
    )
    hidden = 2048
    x = torch.randn(rows, hidden, device="cuda:0", dtype=dtype)
    proj = torch.randn_like(x)
    weight = torch.randn(hidden, device="cuda:0", dtype=dtype)
    residual = x if alias else torch.empty_like(x)
    normalized = torch.empty_like(x)
    expected_residual = x + proj
    if dtype == torch.bfloat16:
        expected_residual = expected_residual.to(dtype)
        expected_normalized = (
            (expected_residual.float() * torch.rsqrt(
                expected_residual.float().square().mean(dim=-1, keepdim=True) + 1e-5
            )).to(dtype).float() * weight.float()
        ).to(dtype)
    else:
        expected_normalized = (
            expected_residual * torch.rsqrt(
                expected_residual.square().mean(dim=-1, keepdim=True) + 1e-5
            )
        ) * weight
    stream = torch.cuda.current_stream().cuda_stream
    assert launcher(x.data_ptr(), proj.data_ptr(), weight.data_ptr(),
                    residual.data_ptr(), normalized.data_ptr(), rows, hidden,
                    1e-5, stream) == 0
    torch.cuda.synchronize()
    torch.testing.assert_close(residual, expected_residual, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        assert torch.equal(normalized, expected_normalized)
    else:
        torch.testing.assert_close(normalized, expected_normalized, rtol=1e-5, atol=1e-6)
