import ctypes as C
import json
import os
from pathlib import Path

from safetensors import safe_open


class EngineConfig(C.Structure):
    _fields_ = [
        ("n_layers", C.c_int),
        ("hidden", C.c_int),
        ("n_heads", C.c_int),
        ("n_kv_heads", C.c_int),
        ("head_dim", C.c_int),
        ("intermediate", C.c_int),
        ("vocab", C.c_int),
        ("rms_eps", C.c_float),
        ("rope_theta", C.c_float),
        ("rope_factor", C.c_float),
        ("rope_low_freq_factor", C.c_float),
        ("rope_high_freq_factor", C.c_float),
        ("rope_original_max_position_embeddings", C.c_int),
        ("tie_embeddings", C.c_bool),
        ("max_seq", C.c_int),
        ("eos_token_id", C.c_int),
    ]


class EngineLibrary:
    def __init__(self, library=None):
        library = library or os.environ.get(
            "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
        )
        self.lib = C.CDLL(library)
        self.lib.engine_create.argtypes = [C.POINTER(EngineConfig)]
        self.lib.engine_create.restype = C.c_void_p
        self.lib.engine_upload_tensor.argtypes = [
            C.c_void_p, C.c_char_p, C.c_void_p, C.c_size_t
        ]
        self.lib.engine_upload_tensor.restype = C.c_int
        self.lib.engine_finalize.argtypes = [C.c_void_p]
        self.lib.engine_finalize.restype = C.c_int
        self.lib.engine_prefill.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int]
        self.lib.engine_prefill.restype = C.c_int
        self.lib.engine_decode.argtypes = [C.c_void_p, C.c_int]
        self.lib.engine_decode.restype = C.c_int
        self.lib.engine_generate.argtypes = [
            C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int, C.POINTER(C.c_int)
        ]
        self.lib.engine_generate.restype = C.c_int
        self.lib.engine_reset.argtypes = [C.c_void_p]
        self.lib.engine_reset.restype = C.c_int
        self.lib.engine_forward.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.c_int]
        self.lib.engine_forward.restype = C.c_int
        self.lib.engine_read_residual.argtypes = [C.c_void_p, C.c_void_p, C.c_int]
        self.lib.engine_read_residual.restype = C.c_int
        self.lib.engine_last_logits.argtypes = [C.c_void_p, C.POINTER(C.c_float)]
        self.lib.engine_last_logits.restype = C.c_int
        self.lib.engine_last_error.argtypes = []
        self.lib.engine_last_error.restype = C.c_char_p
        self.lib.engine_destroy.argtypes = [C.c_void_p]
        self.lib.engine_destroy.restype = None

    def error(self):
        return self.lib.engine_last_error().decode("utf-8", errors="replace")

    def check(self, status):
        if status < 0:
            raise RuntimeError(self.error())

    def create(self, model_config, max_seq=4096):
        scaling = model_config.get("rope_scaling") or {}
        cfg = EngineConfig(
            model_config["num_hidden_layers"],
            model_config["hidden_size"],
            model_config["num_attention_heads"],
            model_config["num_key_value_heads"],
            model_config["head_dim"],
            model_config["intermediate_size"],
            model_config["vocab_size"],
            model_config["rms_norm_eps"],
            model_config["rope_theta"],
            scaling.get("factor", 1.0),
            scaling.get("low_freq_factor", 1.0),
            scaling.get("high_freq_factor", 1.0),
            scaling.get("original_max_position_embeddings", 0),
            model_config.get("tie_word_embeddings", False),
            max_seq,
            model_config.get("eos_token_id", -1),
        )
        handle = self.lib.engine_create(C.byref(cfg))
        if not handle:
            raise RuntimeError(self.error())
        return handle

    def upload_model(self, handle, model_path):
        model_path = Path(model_path)
        uploaded = 0
        for shard in sorted(model_path.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name).contiguous()
                    status = self.lib.engine_upload_tensor(
                        handle,
                        name.encode(),
                        C.c_void_p(tensor.data_ptr()),
                        tensor.numel() * tensor.element_size(),
                    )
                    self.check(status)
                    uploaded += 1
                    del tensor
        self.check(self.lib.engine_finalize(handle))
        return uploaded

    def generate(self, handle, token_ids, max_new):
        prompt = (C.c_int * len(token_ids))(*token_ids)
        output = (C.c_int * max_new)() if max_new else None
        count = self.lib.engine_generate(
            handle, prompt, len(token_ids), max_new, output
        )
        if count < 0:
            raise RuntimeError(self.error())
        return list(output[:count]) if count else []


def load_config(model_path):
    with (Path(model_path) / "config.json").open() as f:
        return json.load(f)


def model_path_from_env():
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("Set MODEL_PATH to the local model snapshot directory")
    return Path(model_path)
