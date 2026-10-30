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
