import ctypes as C
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)


def launcher(name, argument_types):
    fn = getattr(lib, name)
    fn.argtypes = argument_types + [C.c_void_p]
    fn.restype = C.c_int
    return fn


ptr = C.c_void_p
size = C.c_size_t
integer = C.c_int
f32 = C.c_float
embed_f32 = launcher("engine_embedding_f32", [ptr, ptr, ptr, integer, integer])
embed_bf16 = launcher("engine_embedding_bf16", [ptr, ptr, ptr, integer, integer])
norm_f32 = launcher("engine_rmsnorm_f32", [ptr, ptr, ptr, integer, integer, f32])
norm_bf16 = launcher("engine_rmsnorm_bf16", [ptr, ptr, ptr, integer, integer, f32])
swiglu_f32 = launcher("engine_swiglu_f32", [ptr, ptr, ptr, size])
swiglu_bf16 = launcher("engine_swiglu_bf16", [ptr, ptr, ptr, size])


@pytest.mark.parametrize("dtype,fn", [(torch.float32, embed_f32), (torch.bfloat16, embed_bf16)])
@pytest.mark.parametrize("count", [1, 100])
def test_embedding_gather(dtype, fn, count):
    table = torch.randn(128, 64, device="cuda", dtype=dtype)
    ids = torch.randint(1, 127, (count,), device="cuda", dtype=torch.int32)
    ids[0] = 0
    if count > 1:
        ids[-1] = 127
    out = torch.empty(count, 64, device="cuda", dtype=dtype)
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(table.data_ptr(), ids.data_ptr(), out.data_ptr(), count, 64, stream) == 0
    torch.cuda.synchronize()
    torch.testing.assert_close(out, F.embedding(ids.long(), table), rtol=0, atol=0)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, norm_f32), (torch.bfloat16, norm_bf16)])
@pytest.mark.parametrize("rows,hidden", [(1, 2048), (7, 2048), (7, 4096)])
def test_rmsnorm(dtype, fn, rows, hidden):
    x = (torch.randn(rows, hidden, device="cuda", dtype=dtype) * 100).contiguous()
    weight = torch.randn(hidden, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    eps = 1e-5
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(x.data_ptr(), weight.data_ptr(), out.data_ptr(), rows, hidden, eps, stream) == 0
    torch.cuda.synchronize()
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    if dtype == torch.bfloat16:
        expected = (x.float() * torch.rsqrt(variance + eps)).to(dtype) * weight
        assert torch.equal(out, expected)
    else:
        expected = x * torch.rsqrt(variance + eps) * weight
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype,fn", [(torch.float32, swiglu_f32), (torch.bfloat16, swiglu_bf16)])
@pytest.mark.parametrize("count", [1, 257, 4099])
def test_swiglu(dtype, fn, count):
    gate = torch.randn(count, device="cuda", dtype=dtype) * 10
    up = torch.randn_like(gate) * 10
    out = torch.empty_like(gate)
    stream = torch.cuda.current_stream().cuda_stream
    assert fn(gate.data_ptr(), up.data_ptr(), out.data_ptr(), count, stream) == 0
    torch.cuda.synchronize()
    expected = F.silu(gate) * up
    if dtype == torch.bfloat16:
        assert torch.equal(out, expected)
    else:
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
