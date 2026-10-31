# Phase 1 and 2 Implementation Plan

## Phase 1: fusion

Implement and GPU-validate one operation chain at a time: residual plus RMSNorm; K projection plus RoPE and KV-cache store; dual gate/up projection plus SwiGLU; and device-side argmax token handoff. Add focused FP32/BF16 and boundary tests first, then run each smallest case, sanitizer checks, and affected model paths before proceeding. Run the full Phase 0 kernel/model/boundary/generation gates and baseline-comparable benchmarks. Record evidence in `PHASE1_REPORT.md` and commit the complete passing phase once.

## Phase 2: linear kernels

Add FP32/BF16 coverage for model and awkward shapes, then optimize GEMV before tiled GEMM, including fused Phase 1 projection paths. Validate each kernel and caller on GPU and with memcheck/racecheck before moving forward. Add a benchmark-only direct cuBLAS comparison. Rerun the full Phase 0 and Phase 1 functionality gates plus Phase 2 tests, sanitizer cases, repeatability, blocking, and matched benchmarks. Record evidence in `PHASE2_REPORT.md` and commit the complete passing phase once, on top of Phase 1.

## Constraints

Use the existing environment and model. Inspect GPU/process state before every GPU test or benchmark, run sequentially with timeouts, preserve the host and connection, do not install software or push, and keep only the two implementation commits.
