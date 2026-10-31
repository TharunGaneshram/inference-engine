# Phase 1 Report: Fusion

Hardware: RTX A4500 (sm_86), GPU 0; existing `.venv`; Llama-3.2-1B snapshot recorded in `PHASE0_REPORT.md`. Before GPU runs, `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv` showed no competing compute processes.

## Implementation

- Fused residual add and RMSNorm, preserving the in-place residual stream and BF16 intermediate rounding. The engine carries normalized values into following operations and retains the unnormalized residual for diagnostics.
- Fused K projection, half-split RoPE, and direct position-major KV-cache write. Q RoPE remains a separate launch; no temporary K tensor is allocated.
- Fused gate/up projections and SwiGLU, retaining BF16 projection and activation rounding points.
- Internal generation consumes the device argmax output directly on the next embedding step. Public `engine_decode(handle, token)` behavior is unchanged.

The BF16 fused linear reductions use a warp per output to retain sufficient parallelism on decode and prefill. Reduction order differs from Phase 0 GEMM for multi-row inputs; fused BF16 tests compare tensor-scale max error and cosine. FP32 fused tests compare with the existing Phase 0 sequence.

## Correctness and safety gates

- Fused launcher tests: 30 passed across residual+RMSNorm, fused K/RoPE/cache, and fused SwiGLU; FP32/BF16; M=1/3/17 for residual and M=1/3 for linear paths. Existing kernel coverage includes awkward dimensions and argmax tie/endpoint cases.
- Full Phase 0 + Phase 1 kernel suite, including real model upload: `93 passed in 3.32s`.
- Same suite under compute-sanitizer memcheck: `93 passed in 9.14s`; `ERROR SUMMARY: 0 errors`.
- Same suite under racecheck: `93 passed in 49.35s`; `0 hazards displayed (0 errors, 0 warnings)`.
- Real-weight model suite, levels 2–6 and boundary/generation/reset cases: `13 passed in 74.53s`.
- Eight-token generation under memcheck: `Paris. It is the most populous city`; `ERROR SUMMARY: 0 errors`.
- Eight-token generation under racecheck: same tokens; `0 hazards displayed (0 errors, 0 warnings)`.
- Eight-token `CUDA_LAUNCH_BLOCKING=1` generation: same continuation.
- Model metrics: Level 2 final residual relative max error `5.99%`, cosine `0.999625` (existing Phase 0 allowance); Level 3 top-1 matched; Level 4 teacher-forced top-1 `64/64`, min cosine `0.999841`, no first divergence; Level 5 generated `64/64`, no first divergence. Reset generation remained bitwise repeatable.

Commands (each GPU invocation was preceded by the `nvidia-smi` process query above):

```bash
timeout 600s env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" .venv/bin/pytest -q -x tests/test_residual.py tests/test_upload.py tests/test_elementwise_kernels.py tests/test_linear_kernels.py tests/test_rope.py tests/test_attention.py tests/test_argmax.py tests/test_fused_linear.py
timeout 900s compute-sanitizer --tool memcheck --target-processes all env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" .venv/bin/pytest -q -x tests/test_residual.py tests/test_upload.py tests/test_elementwise_kernels.py tests/test_linear_kernels.py tests/test_rope.py tests/test_attention.py tests/test_argmax.py tests/test_fused_linear.py
timeout 900s compute-sanitizer --tool racecheck --target-processes all env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" .venv/bin/pytest -q -x tests/test_residual.py tests/test_upload.py tests/test_elementwise_kernels.py tests/test_linear_kernels.py tests/test_rope.py tests/test_attention.py tests/test_argmax.py tests/test_fused_linear.py
timeout 900s env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" .venv/bin/pytest -q -x -m e2e tests/test_model_e2e.py
timeout 900s compute-sanitizer --tool memcheck --target-processes all env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" PYTHONPATH="$PWD/python" .venv/bin/python scripts/generate.py 'The capital of France is' --max-new 8
timeout 900s compute-sanitizer --tool racecheck --target-processes all env ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" PYTHONPATH="$PWD/python" .venv/bin/python scripts/generate.py 'The capital of France is' --max-new 8
timeout 300s env CUDA_LAUNCH_BLOCKING=1 ENGINE_LIB="$PWD/build/libinference.so" MODEL_PATH="$MODEL_PATH" PYTHONPATH="$PWD/python" .venv/bin/python scripts/generate.py 'The capital of France is' --max-new 8
```

## Fused chain timings

CUDA-event medians, same inputs and GPU, 7 samples × 20 iterations after warmup. Phase 0 launcher sequence versus Phase 1 fused launcher, from `benchmarks/phase1_fusions.py`:

| Chain | M | Phase 0 (ms) | Phase 1 (ms) | Change |
|---|---:|---:|---:|---:|
| Residual + RMSNorm | 1 | 0.01092 | 0.00808 | −26.0% |
| Residual + RMSNorm | 3 | 0.01075 | 0.00808 | −24.8% |
| K projection + RoPE + cache write | 1 | 0.00835 | 0.00583 | −30.1% |
| K projection + RoPE + cache write | 3 | 0.15227 | 0.00788 | −94.8% |
| Gate + up + SwiGLU | 1 | 0.12370 | 0.11935 | −3.5% |
| Gate + up + SwiGLU | 3 | 1.30668 | 0.34365 | −73.7% |

Benchmark command: `timeout 900s env ENGINE_LIB="$PWD/build/libinference.so" .venv/bin/python benchmarks/phase1_fusions.py --output /tmp/phase1_fusions.json`.

## End-to-end performance

Matched model, GPU, prompt construction, and workload versus Phase 0 in `PHASE0_REPORT.md`:

| Measurement | Phase 0 | Phase 1 | Change |
|---|---:|---:|---:|
| Decode, cache 128 | 5.653 ms/token | 5.609 ms/token | −0.8% |
| Decode, cache 1,024 | 9.841 ms/token | 9.777 ms/token | −0.6% |
| Prefill 128 | 101.598 tokens/s | 158.129 tokens/s | +55.6% |
| Prefill 512 | 103.026 tokens/s | 159.259 tokens/s | +54.6% |
| Prefill 2,048 | 100.862 tokens/s | 159.435 tokens/s | +58.1% |

The end-to-end run used the cached 2,049-token boundary prompt repeated/truncated to each requested length, four decode steps, and `time.perf_counter`. Phase 0 and Phase 1 decode measurements are within expected run-to-run spread; prefill improves materially.

## Gate result

Phase 1 correctness, sanitizer, and targeted performance gates pass. The first fused implementation was serial per thread and regressed heavily; it was replaced by the warp-parallel naive reductions above, then the full model and sanitizer gates were rerun. No Phase 2 tiling or vectorized load optimization is included.
