import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb


sys_path = Path(__file__).parents[1] / "python"
import sys
sys.path.insert(0, str(sys_path))
from engine import model_path_from_env


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)
lib.engine_build_rope_tables.argtypes = [
    C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_float,
    C.c_float, C.c_float, C.c_float, C.c_int,
]
lib.engine_build_rope_tables.restype = C.c_int
lib.engine_rope_f32.argtypes = [
    C.c_void_p, C.c_void_p, C.c_void_p, C.c_void_p,
    C.c_int, C.c_int, C.c_int, C.c_int, C.c_int, C.c_void_p,
]
lib.engine_rope_f32.restype = C.c_int
lib.engine_rope_bf16.argtypes = list(lib.engine_rope_f32.argtypes)
lib.engine_rope_bf16.restype = C.c_int


def rope_table(max_seq, head_dim):
    model_path = model_path_from_env()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    with open(Path(model_path) / "config.json") as f:
        raw = json.load(f)
    scaling = raw["rope_scaling"]
    cos = np.empty((max_seq, head_dim // 2), dtype=np.float32)
    sin = np.empty_like(cos)
    status = lib.engine_build_rope_tables(
        cos.ctypes.data, sin.ctypes.data, max_seq, head_dim,
        raw["rope_theta"], scaling["factor"], scaling["low_freq_factor"],
        scaling["high_freq_factor"], scaling["original_max_position_embeddings"],
    )
    assert status == 0
    return config, cos, sin


def test_rope_tables_match_transformers_at_boundaries():
    config, cos, sin = rope_table(4096, 64)
    rotary = LlamaRotaryEmbedding(config).cuda()
    x = torch.zeros(1, 3, 64, device="cuda", dtype=torch.float32)
    positions = torch.tensor([[0, 1, 4095]], device="cuda")
    ref_cos, ref_sin = rotary(x, positions)
    expected_cos = ref_cos[0, :, :32].cpu().numpy()
    expected_sin = ref_sin[0, :, :32].cpu().numpy()
    # The host table is intentionally computed in fp64; Transformers computes
    # its inverse frequencies and angles in fp32. The observed max absolute
    # difference at these positions is 1.972e-4.
    np.testing.assert_allclose(cos[[0, 1, 4095]], expected_cos, rtol=1e-4, atol=2.1e-4)
    np.testing.assert_allclose(sin[[0, 1, 4095]], expected_sin, rtol=1e-4, atol=2.1e-4)


@pytest.mark.parametrize("dtype,launcher", [(torch.float32, lib.engine_rope_f32), (torch.bfloat16, lib.engine_rope_bf16)])
@pytest.mark.parametrize("start_pos,rows", [(0, 1), (10, 5)])
def test_rope_kernel(dtype, launcher, start_pos, rows):
    head_dim, n_heads, n_kv_heads = 64, 8, 2
    config, cos, sin = rope_table(start_pos + rows + 1, head_dim)
    cos_gpu = torch.from_numpy(cos.copy()).cuda()
    sin_gpu = torch.from_numpy(sin.copy()).cuda()
    q = torch.randn(rows, n_heads, head_dim, device="cuda", dtype=dtype)
    k = torch.randn(rows, n_kv_heads, head_dim, device="cuda", dtype=dtype)
    q_ref = q.clone().permute(1, 0, 2)[None, ...]
    k_ref = k.clone().permute(1, 0, 2)[None, ...]
    stream = torch.cuda.current_stream().cuda_stream
    status = launcher(
        q.data_ptr(), k.data_ptr(), cos_gpu.data_ptr(), sin_gpu.data_ptr(),
        start_pos, rows, n_heads, n_kv_heads, head_dim, stream,
    )
    assert status == 0
    torch.cuda.synchronize()

    rotary = LlamaRotaryEmbedding(config).cuda()
    positions = torch.arange(start_pos, start_pos + rows, device="cuda")[None, :]
    x = torch.zeros(1, rows, head_dim, device="cuda", dtype=dtype)
    ref_cos, ref_sin = rotary(x, positions)
    q_expected, k_expected = apply_rotary_pos_emb(q_ref, k_ref, ref_cos, ref_sin)
    q_expected = q_expected[0].permute(1, 0, 2)
    k_expected = k_expected[0].permute(1, 0, 2)
    if dtype == torch.float32:
        torch.testing.assert_close(q, q_expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k, k_expected, rtol=1e-5, atol=1e-5)
    else:
        torch.testing.assert_close(q.float(), q_expected.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(k.float(), k_expected.float(), rtol=1e-2, atol=1e-2)
