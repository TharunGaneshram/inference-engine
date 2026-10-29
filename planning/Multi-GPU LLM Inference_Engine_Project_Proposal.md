# Multi-GPU LLM Inference Engine: Project Proposal

Oct 14, 2026 · @Tharun Ganeshram

## Overview

I will build a multi-GPU LLM inference engine from scratch in CUDA C++ and layer optimizations onto it one at a time, measuring each against a PyTorch/Hugging Face baseline. The goal is a first-principles understanding of why decode is memory-bandwidth-bound and how fusion, tiling and parallelism address it. Total effort is about 40 hours.

## Approach

The engine targets a Llama-architecture model (GQA, RoPE, RMSNorm, SwiGLU): a \~1B variant for development and an \~8B model (\~16 GB in bf16) for multi-GPU runs. Every kernel is unit-tested against a PyTorch fp32 reference, and end-to-end logits are checked against Hugging Face Transformers.

- **Kernel modules:** linear layers (GEMV for decode, tiled GEMM for prefill), attention, fused elementwise ops, embedding and sampling, KV-cache ops
- **Runtime:** C++ engine with weight upload and a multi-GPU launcher, with NCCL for all-reduce
- **Harness:** Python scripts for tokenization, reading safetensors files, correctness checks and benchmarking

## Core scope (34 hours)

| Phase | What gets built | Hours |
| --- | --- | --- |
| Setup | Toolchain, Nsight, measured peak memory bandwidth | 3 |
| 0. Baseline | Weight loader, naive kernels, KV cache, prefill/decode split, greedy sampling; logits match HF | 6 |
| 1. Fusion | RMSNorm+residual, RoPE+KV write, SwiGLU; argmax writes the next token into the input buffer | 3 |
| 2. GEMV/GEMM | Vectorized bandwidth-bound GEMV, tiled GEMM, compared to cuBLAS | 5 |
| 3. GQA | Grouped-query support in the attention kernel | 1 |
| 4. Flash attention | Tiled fused attention with online softmax | 6 |
| 5. Tensor parallelism | Column/row-split layers, head-split attention, NCCL all-reduce across GPUs | 6 |
| 6. Benchmarks | Roofline plots, scaling results, write-up | 4 |

## Stretch goals

**1. Paged KV cache (\~5 hours):** fixed-size KV blocks stored non-contiguously, a block table, and a paged attention kernel, validated against the contiguous cache and compared on memory use.

**2. Multi-sequence inference with continuous batching (\~5 hours):** several sequences share each decode step, with per-token sequence slot and position inputs and a scheduler that admits and retires sequences.

They use any remaining time, in this order, and are cut first if core work overruns.

## Hardware and validation

Development and tensor-parallel testing run on a UW engineering Linux machine with two NVIDIA A4500 GPUs (20 GB each, linked by NVLink), which supports 2-way tensor parallelism with the 8B model split across both cards. After the engine is complete, a few hours on a rented 4-8 GPU machine will run one scripted sweep to extend the scaling results beyond two GPUs.

## Deliverables

- A documented, tested repository with the engine and all kernels
- Roofline plots showing achieved memory bandwidth as a fraction of peak for decode kernels
- Tokens/s before and after each optimization, and a tensor-parallel scaling curve
- A short write-up of bottlenecks found and what they imply for other hardware architectures
