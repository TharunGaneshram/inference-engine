import ctypes as C
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))
from engine import EngineLibrary, load_config, model_path_from_env


def test_real_model_upload_and_validation():
    model_path = model_path_from_env()
    api = EngineLibrary()
    handle = api.create(load_config(model_path))
    try:
        status = api.lib.engine_upload_tensor(
            handle, b"not.a.real.tensor", C.c_void_p(1), 2
        )
        assert status < 0
        assert "unknown tensor 'not.a.real.tensor'" in api.error()

        status = api.lib.engine_upload_tensor(
            handle, b"model.embed_tokens.weight", C.c_void_p(1), 2
        )
        assert status < 0
        assert "model.embed_tokens.weight" in api.error()
        assert "expected" in api.error()

        status = api.lib.engine_finalize(handle)
        assert status < 0
        assert "engine_finalize: missing tensor" in api.error()

        uploaded = api.upload_model(handle, model_path)
        assert uploaded == 146
    finally:
        api.lib.engine_destroy(handle)
