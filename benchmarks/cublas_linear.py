"""Compare engine GEMV/GEMM launchers directly with cuBLAS on identical inputs."""

import argparse
import ctypes as C
import json
import statistics
from pathlib import Path

import torch


BLAS_PATH = "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so"
blas = C.CDLL(BLAS_PATH)
engine = C.CDLL(str(Path(__file__).parents[1] / "build" / "libinference.so"))
PTR, INT = C.c_void_p, C.c_int


def bind(name, argtypes):
    fn = getattr(engine, name)
    fn.argtypes = argtypes
    fn.restype = INT
    return fn


gemv_f32 = bind("engine_gemv_f32", [PTR, PTR, PTR, INT, INT, PTR])
gemv_bf16 = bind("engine_gemv_bf16", [PTR, PTR, PTR, INT, INT, PTR])
gemv_bf16_f32 = bind("engine_gemv_bf16_f32", [PTR, PTR, PTR, INT, INT, PTR])
gemm_f32 = bind("engine_gemm_f32", [PTR, PTR, PTR, INT, INT, INT, PTR])
gemm_bf16 = bind("engine_gemm_bf16", [PTR, PTR, PTR, INT, INT, INT, PTR])

blas.cublasCreate_v2.argtypes = [C.POINTER(PTR)]
blas.cublasCreate_v2.restype = INT
blas.cublasDestroy_v2.argtypes = [PTR]
blas.cublasDestroy_v2.restype = INT
blas.cublasSetStream_v2.argtypes = [PTR, PTR]
blas.cublasSetStream_v2.restype = INT
blas.cublasGemmEx.argtypes = [
    PTR, INT, INT, INT, INT, INT, PTR, PTR, INT, INT, PTR, INT, INT,
    PTR, PTR, INT, INT, INT, INT,
]
blas.cublasGemmEx.restype = INT
blas.cublasSgemv_v2.argtypes = [
    PTR, INT, INT, INT, PTR, PTR, INT, PTR, INT, PTR, PTR, INT,
]
blas.cublasSgemv_v2.restype = INT


def check(status):
    if status:
        raise RuntimeError(f"cuBLAS/engine status {status}")


def samples_ms(fn, repeats=7, iterations=20):
    for _ in range(10):
        check(fn())
    torch.cuda.synchronize()
    results = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            check(fn())
        end.record()
        end.synchronize()
        results.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(results), "samples_ms": results}


def run(output_path, dtype_filter=None):
    handle = PTR()
    check(blas.cublasCreate_v2(C.byref(handle)))
    results = {}
    try:
        stream = torch.cuda.current_stream().cuda_stream
        check(blas.cublasSetStream_v2(handle, PTR(stream)))
        shapes = [(2048, 2048), (512, 2048), (8192, 2048),
                  (2048, 8192), (128256, 2048)]
        dtypes = (torch.float32, torch.bfloat16) if dtype_filter is None else (
            torch.float32 if dtype_filter == "float32" else torch.bfloat16,
        )
        for dtype in dtypes:
            for n, k in shapes:
                x = torch.randn(k, device="cuda", dtype=dtype)
                weight = torch.randn(n, k, device="cuda", dtype=dtype)
                out_dtype = torch.float32 if n == 128256 else dtype
                actual, reference = (torch.empty(n, device="cuda", dtype=out_dtype)
                                     for _ in range(2))
                scalar = C.c_float(1.0)
                zero = C.c_float(0.0)
                if dtype == torch.float32:
                    engine_call = lambda: gemv_f32(x.data_ptr(), weight.data_ptr(),
                                                   actual.data_ptr(), n, k, stream)
                    if n == 128256:
                        def cublas_call():
                            return blas.cublasGemmEx(
                                handle, 1, 0, n, 1, k, C.byref(scalar),
                                weight.data_ptr(), 0, k, x.data_ptr(), 0, k,
                                C.byref(zero), reference.data_ptr(), 0, n, 68, -1,
                            )
                        cublas_api = "cublasGemmEx (M=1)"
                    else:
                        def cublas_call():
                            return blas.cublasSgemv_v2(
                                handle, 1, k, n, C.byref(scalar), weight.data_ptr(), k,
                                x.data_ptr(), 1, C.byref(zero), reference.data_ptr(), 1,
                            )
                        cublas_api = "cublasSgemv_v2"
                else:
                    fn = gemv_bf16_f32 if out_dtype == torch.float32 else gemv_bf16
                    engine_call = lambda fn=fn: fn(x.data_ptr(), weight.data_ptr(),
                                                   actual.data_ptr(), n, k, stream)
                    type_bf16, type_f32 = 14, 0
                    out_type = type_f32 if out_dtype == torch.float32 else type_bf16

                    def cublas_call():
                        return blas.cublasGemmEx(
                            handle, 1, 0, n, 1, k, C.byref(scalar), weight.data_ptr(),
                            type_bf16, k, x.data_ptr(), type_bf16, k, C.byref(zero),
                            reference.data_ptr(), out_type, n, 68, -1,
                        )
                check(engine_call())
                key = f"gemv_{str(dtype).split('.')[-1]}_n{n}_k{k}"
                print(f"benchmarking {key}: engine", flush=True)
                engine_time = samples_ms(engine_call)
                bytes_read = (n * k + k) * torch.tensor([], dtype=dtype).element_size()
                engine_time["effective_gb_s"] = bytes_read / (engine_time["median_ms"] * 1e6)
                engine_time["percent_of_565_392_gb_s"] = engine_time["effective_gb_s"] / 565.392 * 100
                if n == 128256:
                    torch.cuda.synchronize()
                    results[key] = {
                        "engine": engine_time,
                        "cublas": {"status": "not timed: direct cuBLAS calls repeatedly aborted the benchmark process for this shape"},
                    }
                    continue
                check(cublas_call())
                torch.cuda.synchronize()
                delta = (actual.float() - reference.float()).abs()
                scale = reference.float().abs().max().clamp_min(1e-12)
                cosine = torch.nn.functional.cosine_similarity(
                    actual.float().flatten(), reference.float().flatten(), dim=0
                )
                if cosine < 0.999:
                    raise RuntimeError(f"cuBLAS comparison cosine too low: {cosine.item()}")
                print(f"benchmarking {key}: cuBLAS", flush=True)
                blas_time = samples_ms(cublas_call)
                results[key] = {"engine": engine_time, "cublas": blas_time,
                                "cublas_api": cublas_api if dtype == torch.float32 else "cublasGemmEx (M=1)",
                                "numerical": {"max_abs": delta.max().item(),
                                              "max_abs_over_tensor_scale": (delta.max() / scale).item(),
                                              "cosine": cosine.item()}}

        for dtype in dtypes:
            for m, n, k in ((3, 8192, 2048), (17, 2048, 2048), (17, 2048, 8192)):
                x = torch.randn(m, k, device="cuda", dtype=dtype)
                weight = torch.randn(n, k, device="cuda", dtype=dtype)
                out = torch.empty(m, n, device="cuda", dtype=dtype)
                ref = torch.empty_like(out)
                type_a = 0 if dtype == torch.float32 else 14
                compute = 68
                engine_fn = gemm_f32 if dtype == torch.float32 else gemm_bf16
                engine_call = lambda: engine_fn(x.data_ptr(), weight.data_ptr(),
                                                out.data_ptr(), m, n, k, stream)
                alpha, beta = C.c_float(1.0), C.c_float(0.0)

                def cublas_call():
                    return blas.cublasGemmEx(handle, 1, 0, n, m, k, C.byref(alpha),
                                             weight.data_ptr(), type_a, k,
                                             x.data_ptr(), type_a, k, C.byref(beta),
                                             ref.data_ptr(), type_a, n, compute, -1)
                check(engine_call())
                check(cublas_call())
                torch.cuda.synchronize()
                delta = (out.float() - ref.float()).abs()
                scale = ref.float().abs().max().clamp_min(1e-12)
                cosine = torch.nn.functional.cosine_similarity(
                    out.float().flatten(), ref.float().flatten(), dim=0
                )
                if cosine < 0.999:
                    raise RuntimeError(f"cuBLAS comparison cosine too low: {cosine.item()}")
                key = f"gemm_{str(dtype).split('.')[-1]}_m{m}_n{n}_k{k}"
                print(f"benchmarking {key}", flush=True)
                results[key] = {"engine": samples_ms(engine_call),
                                "cublas": samples_ms(cublas_call),
                                "cublas_api": "cublasGemmEx",
                                "numerical": {"max_abs": delta.max().item(),
                                              "max_abs_over_tensor_scale": (delta.max() / scale).item(),
                                              "cosine": cosine.item()}}
        output_path.write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))
    finally:
        # Explicit cublasDestroy_v2 aborts during process teardown in this CUDA
        # runtime after successful benchmark calls. Process exit releases the handle.
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/cublas_linear.json"))
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    run(args.output, args.dtype)
