# Phase 2 Report: Vectorized GEMV and Tiled GEMM

Hardware: RTX A4500 (`sm_86`), CUDA 12.3, existing `.venv`, cached Llama-3.2-1B. Phase 1 commit: `f38f679`. Before every GPU run, `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv` showed no competing compute processes. Runs were sequential and bounded.

## Implementation

- GEMV uses aligned `float4` FP32 and packed `uint2` BF16 loads when K is divisible by four, with scalar fallback for misalignment and tails. Accumulation remains FP32 and outputs retain the caller's dtype.
- Prefill GEMM uses a shared-memory SIMT tile with compensated FP32 accumulation for FP32, and a masked BF16 Tensor Core tile with FP32 accumulators on `sm_86`.
- Fused BF16 K projection/RoPE/cache-store and dual gate/up/SwiGLU use Tensor Core tiles for prefill rows of at least 16. Smaller batches retain the warp-parallel Phase 1 path. Fused FP32 prefill callers use tiled SIMT paths; decode uses vectorized warp reductions.
- Tests cover model dimensions, M=1/3/17, awkward dimensions, misaligned addresses, tails, both dtypes, and fused callers. No existing reference tolerance was relaxed.
- Added direct cuBLAS and matched steady-state decode benchmarks: `benchmarks/cublas_linear.py` and `benchmarks/phase2_decode.py`.

## Correctness and safety gates

- Full project suite, including model and end-to-end cases: `136 passed in 14.37s`.
- Complete Phase 2 kernel suite: `123 passed in 8.22s` under memcheck (`ERROR SUMMARY: 0 errors`) and `123 passed in 270.80s` under racecheck (`0 hazards displayed (0 errors, 0 warnings)`).
- Eight-token real-model generation under memcheck and racecheck produced `Paris. It is the most populous city`; memcheck reported zero errors and racecheck zero hazards. `CUDA_LAUNCH_BLOCKING=1` produced the same continuation. The full suite's repeatability case passed bitwise after reset.
- Final model metrics: Level 2 final residual max absolute error `5.6071`, relative max `5.990%`, cosine `0.999625`; Level 3 top-1 matched on all five prompts, minimum cosine `0.999925`, maximum absolute error `0.316742`; Level 4 teacher-forced decode matched top-1 `64/64`, minimum cosine `0.999780`, max absolute error `0.244964`, no first divergence; Level 5 greedy generation matched `64/64`, no first divergence. Chunked prefill matched full prefill bitwise; prefill/decode agreement cosine was `0.999992`.

Each verification command below was preceded by the `nvidia-smi` query above:

- `timeout 900s env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" .venv/bin/pytest -q -x` — 136 passed.
- `timeout 1200s compute-sanitizer --tool memcheck --target-processes all env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" .venv/bin/pytest -q -x tests/test_residual.py tests/test_upload.py tests/test_elementwise_kernels.py tests/test_linear_kernels.py tests/test_rope.py tests/test_attention.py tests/test_argmax.py tests/test_fused_linear.py` — 123 passed, zero errors.
- Same kernel command with `--tool racecheck` and `timeout 1800s` — 123 passed, zero hazards.
- `timeout 900s compute-sanitizer --tool memcheck --target-processes all env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" PYTHONPATH="$PWD/python" .venv/bin/python scripts/generate.py 'The capital of France is' --max-new 8` — zero errors.
- Same generation command with `--tool racecheck` — zero hazards; with `CUDA_LAUNCH_BLOCKING=1` (without compute-sanitizer) it produced the same continuation.

## Performance

Matched Phase 1/Phase 2 decode used the same model and prompt, four untimed decode steps, eight timed steps, and three repetitions. Timings overlap within sample spread:

| Cache length | Phase 1 median (range) | Phase 2 median (range) | Change |
|---:|---:|---:|---:|
| 128 | 5.645 ms/token (5.635–5.658) | 5.555 ms/token (5.549–5.558) | −1.6% |
| 1,024 | 9.807 ms/token (9.803–9.835) | 9.836 ms/token (9.822–9.863) | +0.3% |

The initial one-shot harness showed a larger cache-1,024 difference because Phase 1's longer prefill warmed the GPU for several seconds, while Phase 2 prefill finished sooner. Explicit matched warmup removes that timing mismatch.

Matched end-to-end prefill with the boundary prompt:

| Prefill length | Phase 1 | Phase 2 | Change |
|---:|---:|---:|---:|
| 128 | 158.129 tok/s | 2,280.285 tok/s | 14.4× |
| 512 | 159.259 tok/s | 2,419.168 tok/s | 15.2× |
| 2,048 | 159.435 tok/s | 1,715.235 tok/s | 10.8× |

BF16 kernel timings from `benchmarks/linear.py`, measured on both commits:

| Operation | Phase 1 | Phase 2 | Change |
|---|---:|---:|---:|
| GEMV q/o, N=2,048 K=2,048 | 0.0201 ms | 0.0186 ms | −7.5% |
| GEMV k/v, N=512 K=2,048 | 0.0070 ms | 0.0059 ms | −15.7% |
| GEMV gate/up, N=8,192 K=2,048 | 0.0614 ms | 0.0606 ms | −1.3% |
| GEMV down, N=2,048 K=8,192 | 0.0623 ms | 0.0604 ms | −3.0% |
| GEMV lm_head, N=128,256 K=2,048 | 0.8909 ms | 0.8886 ms | −0.3% |
| GEMM gate/up, M=3 N=8,192 K=2,048 | 0.6536 ms | 0.2434 ms | −62.8% |

### Direct cuBLAS comparison

CUDA-event medians use seven samples of twenty calls after warmup. GEMV effective bandwidth counts weight and input bytes and uses the measured 565.392 GB/s copy peak as reference; cache effects can put effective bandwidth above that figure. Phase 2 and direct cuBLAS used the same stream, dtype, shape, and data. LM-head direct cuBLAS timing is omitted because calls for that shape reproducibly aborted the benchmark process.

| GEMV dtype | N × K | Engine ms | cuBLAS ms | Effective GB/s (% peak) |
|---|---:|---:|---:|---:|
| FP32 | 2,048 × 2,048 | 0.03205 | 0.03261 | 523.7 (92.6%) |
| FP32 | 512 × 2,048 | 0.00630 | 0.00645 | 667.3 (118.0%) |
| FP32 | 8,192 × 2,048 | 0.11612 | 0.11837 | 578.0 (102.2%) |
| FP32 | 2,048 × 8,192 | 0.12160 | 0.11776 | 552.2 (97.7%) |
| FP32 | 128,256 × 2,048 | 1.77572 | not timed | 591.7 (104.7%) |
| BF16 | 2,048 × 2,048 | 0.01715 | 0.02268 | 489.3 (86.5%) |
| BF16 | 512 × 2,048 | 0.00502 | 0.00676 | 418.8 (74.1%) |
| BF16 | 8,192 × 2,048 | 0.06001 | 0.06313 | 559.2 (98.9%) |
| BF16 | 2,048 × 8,192 | 0.05960 | 0.06938 | 563.3 (99.6%) |
| BF16 | 128,256 × 2,048 | 0.89062 | not timed | 589.9 (104.3%) |

| GEMM dtype | M × N × K | Engine ms | cuBLAS ms | Engine / cuBLAS |
|---|---:|---:|---:|---:|
| FP32 | 3 × 8,192 × 2,048 | 0.23368 | 0.15114 | 1.55× slower |
| FP32 | 17 × 2,048 × 2,048 | 0.30930 | 0.04209 | 7.35× slower |
| FP32 | 17 × 2,048 × 8,192 | 1.19730 | 0.13706 | 8.74× slower |
| BF16 | 3 × 8,192 × 2,048 | 0.24663 | 0.06282 | 3.93× slower |
| BF16 | 17 × 2,048 × 2,048 | 0.16328 | 0.02006 | 8.14× slower |
| BF16 | 17 × 2,048 × 8,192 | 0.62367 | 0.06881 | 9.06× slower |

The tiled GEMM improves over the Phase 1 naive path and produces the observed prefill gains, but remains substantially slower than cuBLAS, especially at M=17. This is an open optimization gap; no cuBLAS parity or speedup is claimed. Direct cuBLAS comparison cosine was at least `0.999996` for BF16 GEMV, at least `1.0` within float display precision for BF16 GEMM, and at least `0.9999999` for FP32. Max absolute error divided by reference tensor scale stayed below `0.32%` for BF16 and `3.4e-7` for FP32. Independent PyTorch and model checks passed as well.

Benchmarks used `benchmarks/linear.py`, `benchmarks/cublas_linear.py --dtype float32|bfloat16`, `benchmarks/e2e.py`, and `benchmarks/phase2_decode.py`. `P1_ENGINE_LIB` pointed to `/tmp/inference_phase1/build/libinference.so`, built from commit `f38f679` in a detached worktree for matched comparison.

## Gate result and remaining issue

Phase 2 correctness, sanitizer, steady-state decode, and prefill gates passed. GEMV and representative GEMM timings improved over Phase 1. Direct cuBLAS comparison exposed a substantial small-batch GEMM gap, recorded above for follow-up rather than presented as parity. No reboot, GPU reset, machine reconfiguration, or push was performed.
