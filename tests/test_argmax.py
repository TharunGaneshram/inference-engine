import ctypes as C
import os
from pathlib import Path

import pytest
import torch


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)
lib.engine_argmax_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_void_p]
lib.engine_argmax_f32.restype = C.c_int


@pytest.mark.parametrize("size", [17, 128256])
def test_argmax_matches_torch(size):
    values = torch.randn(size, device="cuda", dtype=torch.float32)
    values[-1] = 1000.0
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.current_stream().cuda_stream
    assert lib.engine_argmax_f32(values.data_ptr(), output.data_ptr(), size, stream) == 0
    torch.cuda.synchronize()
    assert output.item() == torch.argmax(values).item()


def test_argmax_ties_choose_first_index():
    values = torch.zeros(128256, device="cuda", dtype=torch.float32)
    values[3] = 10.0
    values[127] = 10.0
    output = torch.empty(1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.current_stream().cuda_stream
    assert lib.engine_argmax_f32(values.data_ptr(), output.data_ptr(), values.numel(), stream) == 0
    torch.cuda.synchronize()
    assert output.item() == 3
