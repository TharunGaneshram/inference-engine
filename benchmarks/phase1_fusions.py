"""Compare Phase 1 fused launchers with their Phase 0 launch sequences."""

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import statistics

import torch


LIBRARY = os.environ.get(
    "ENGINE_LIB", str(Path(__file__).parents[1] / "build" / "libinference.so")
)
lib = C.CDLL(LIBRARY)
PTR, INT, FLOAT = C.c_void_p, C.c_int, C.c_float


def launcher(name, argtypes):
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    fn.restype = INT
    return fn


residual = launcher("engine_residual_bf16", [PTR, PTR, PTR, C.c_size_t, PTR])
rmsnorm = launcher("engine_rmsnorm_bf16", [PTR, PTR, PTR, INT, INT, FLOAT, PTR])
fused_residual = launcher(
    "engine_residual_rmsnorm_bf16", [PTR, PTR, PTR, PTR, PTR, INT, INT, FLOAT, PTR]
)
gemv = launcher("engine_gemv_bf16", [PTR, PTR, PTR, INT, INT, PTR])
gemm = launcher("engine_gemm_bf16", [PTR, PTR, PTR, INT, INT, INT, PTR])
rope = launcher("engine_rope_bf16", [PTR, PTR, PTR, PTR, INT, INT, INT, INT, INT, PTR])
fused_k = launcher(
    "engine_project_rope_cache_bf16",
    [PTR, PTR, PTR, PTR, PTR, INT, INT, INT, INT, INT, PTR],
)
swiglu = launcher("engine_swiglu_bf16", [PTR, PTR, PTR, C.c_size_t, PTR])
fused_swiglu = launcher(
    "engine_linear_swiglu_bf16", [PTR, PTR, PTR, PTR, INT, INT, INT, PTR]
)


def checked(status):
    if status:
        raise RuntimeError(f"CUDA launcher returned {status}")


def median_ms(fn, stream, warmup=10, repeats=7, iterations=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def compare(fn0, fn1, stream):
    return {"phase0_sequence": median_ms(fn0, stream), "phase1_fused": median_ms(fn1, stream)}


def run():
    device = torch.device("cuda:0")
    stream = torch.cuda.current_stream(device).cuda_stream
    results = {}

    for rows in (1, 3):
        hidden = 2048
        x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
        proj = torch.randn_like(x)
        weight = torch.randn(hidden, device=device, dtype=torch.bfloat16)
        work = torch.empty_like(x)
        norm = torch.empty_like(x)

        def residual_sequence():
            work.copy_(x)
            checked(residual(work.data_ptr(), proj.data_ptr(), work.data_ptr(), x.numel(), stream))
            checked(rmsnorm(work.data_ptr(), weight.data_ptr(), norm.data_ptr(), rows,
                            hidden, 1e-5, stream))

        def residual_fused():
            work.copy_(x)
            checked(fused_residual(x.data_ptr(), proj.data_ptr(), weight.data_ptr(),
                                   work.data_ptr(), norm.data_ptr(), rows, hidden,
                                   1e-5, stream))

        results[f"residual_norm_m{rows}"] = compare(
            residual_sequence, residual_fused, stream
        )

        n_kv, head_dim = 8, 64
        n = n_kv * head_dim
        kw = torch.randn(n, hidden, device=device, dtype=torch.bfloat16)
        k = torch.empty(rows, n, device=device, dtype=torch.bfloat16)
        q = torch.empty(rows, head_dim, device=device, dtype=torch.bfloat16)
        cos = torch.randn(16, head_dim // 2, device=device)
        sin = torch.randn_like(cos)

        def k_sequence():
            if rows == 1:
                checked(gemv(x.data_ptr(), kw.data_ptr(), k.data_ptr(), n, hidden, stream))
            else:
                checked(gemm(x.data_ptr(), kw.data_ptr(), k.data_ptr(), rows, n, hidden, stream))
            checked(rope(q.data_ptr(), k.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                         7, rows, 1, n_kv, head_dim, stream))

        def k_fused():
            checked(fused_k(x.data_ptr(), kw.data_ptr(), k.data_ptr(), cos.data_ptr(),
                             sin.data_ptr(), 7, rows, n_kv, head_dim, hidden, stream))

        results[f"k_rope_cache_m{rows}"] = compare(k_sequence, k_fused, stream)

        intermediate = 8192
        gw = torch.randn(intermediate, hidden, device=device, dtype=torch.bfloat16)
        uw = torch.randn_like(gw)
        gate = torch.empty(rows, intermediate, device=device, dtype=torch.bfloat16)
        up = torch.empty_like(gate)
        activated = torch.empty_like(gate)

        def swiglu_sequence():
            if rows == 1:
                checked(gemv(x.data_ptr(), gw.data_ptr(), gate.data_ptr(), intermediate, hidden, stream))
                checked(gemv(x.data_ptr(), uw.data_ptr(), up.data_ptr(), intermediate, hidden, stream))
            else:
                checked(gemm(x.data_ptr(), gw.data_ptr(), gate.data_ptr(), rows, intermediate, hidden, stream))
                checked(gemm(x.data_ptr(), uw.data_ptr(), up.data_ptr(), rows, intermediate, hidden, stream))
            checked(swiglu(gate.data_ptr(), up.data_ptr(), activated.data_ptr(),
                           rows * intermediate, stream))

        def swiglu_fused():
            checked(fused_swiglu(x.data_ptr(), gw.data_ptr(), uw.data_ptr(),
                                 activated.data_ptr(), rows, intermediate, hidden, stream))

        results[f"swiglu_m{rows}"] = compare(swiglu_sequence, swiglu_fused, stream)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/phase1_fusions.json"))
    args = parser.parse_args()
    result = run()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
