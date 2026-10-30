# Phase 0 CUDA inference engine

Single GPU, batch size 1 Llama inference in CUDA C++ with a Python test harness. The implementation targets Llama 3.2 1B, using bf16 weights and activations, fp32 accumulation and logits, and intentionally naive kernels.

## Requirements

- Linux with an NVIDIA GPU and a working NVIDIA driver
- CUDA Toolkit with `nvcc`, CMake 3.24+, and Python 3.10+
- A PyTorch build that can use the installed NVIDIA GPU
- A local Llama 3.2 1B model snapshot (Hugging Face access may require accepting the model license)

Create a virtual environment in the project root and install the Python prerequisites:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set `MODEL_PATH` to the model snapshot directory containing `config.json` and the safetensors weights. For example, use the snapshot path from your Hugging Face cache:

```bash
export MODEL_PATH=/path/to/model
```

## Build

Detect the GPU compute capability and build for it:

```bash
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
cmake --build build -j2
```

The shared library is `build/libinference.so`.

## Test

Before GPU tests or generation, check for other GPU compute processes:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Run the kernel and loader tests:

```bash
ENGINE_LIB="$PWD/build/libinference.so" .venv/bin/pytest -q -x tests/test_residual.py tests/test_upload.py tests/test_elementwise_kernels.py tests/test_linear_kernels.py tests/test_rope.py tests/test_attention.py tests/test_argmax.py
```

For model-level tests, first generate fp32 Hugging Face reference data, then run the end-to-end suite:

```bash
.venv/bin/python scripts/make_reference.py
ENGINE_LIB="$PWD/build/libinference.so" .venv/bin/pytest -q -x -m e2e tests/test_model_e2e.py
```

## Generate

```bash
ENGINE_LIB="$PWD/build/libinference.so" PYTHONPATH="$PWD/python" .venv/bin/python scripts/generate.py 'The capital of France is' --max-new 8
```

The C API is in `include/inference/engine.h`; kernel launchers are in `include/inference/kernels.h`. `python/engine.py` loads the model configuration and safetensors weights, then uploads them through the C API.
