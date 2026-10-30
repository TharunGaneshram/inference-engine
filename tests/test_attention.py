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
lib.engine_attention_f32.argtypes = [
    C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
    C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_float, C.c_void_p,
]
lib.engine_attention_f32.restype = C.c_int
lib.engine_attention_bf16.argtypes = list(lib.engine_attention_f32.argtypes)
lib.engine_attention_bf16.restype = C.c_int


@pytest.mark.parametrize("dtype,launcher", [(torch.float32, lib.engine_attention_f32), (torch.bfloat16, lib.engine_attention_bf16)])
@pytest.mark.parametrize("history,rows,start_pos", [(1, 1, 0), (2, 1, 1), (1000, 1, 999), (13, 3, 10)])
def test_gqa_causal_attention(dtype, launcher, history, rows, start_pos):
    n_heads, n_kv_heads, head_dim = 8, 2, 64
    full_len = start_pos + rows
    kcache = torch.randn(full_len, n_kv_heads, head_dim, device="cuda", dtype=dtype)
    vcache = torch.randn_like(kcache)
    q = torch.randn(rows, n_heads, head_dim, device="cuda", dtype=dtype)
    out = torch.empty_like(q)
    stream = torch.cuda.current_stream().cuda_stream
    status = launcher(
        q.data_ptr(), kcache.data_ptr(), vcache.data_ptr(), out.data_ptr(),
        start_pos, rows, n_heads, n_kv_heads, head_dim, 1.0 / (head_dim ** 0.5), stream,
    )
    assert status == 0
    torch.cuda.synchronize()

    repeat = n_heads // n_kv_heads
    k = kcache.permute(1, 0, 2).repeat_interleave(repeat, dim=0)[None, ...]
    v = vcache.permute(1, 0, 2).repeat_interleave(repeat, dim=0)[None, ...]
    q_ref = q.permute(1, 0, 2)[None, ...]
    positions = torch.arange(full_len, device="cuda")
    query_positions = start_pos + torch.arange(rows, device="cuda")
    mask = positions[None, :] <= query_positions[:, None]
    expected = F.scaled_dot_product_attention(
        q_ref, k, v, attn_mask=mask[None, None, :, :], is_causal=False
    )[0].permute(1, 0, 2)
    if dtype == torch.float32:
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)
    else:
        torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)
