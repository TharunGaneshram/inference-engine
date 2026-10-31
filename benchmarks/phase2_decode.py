"""Steady-state decode timing for matched Phase 1/Phase 2 builds."""

import argparse
import ctypes as C
import statistics
import time
from pathlib import Path

import numpy as np

from engine import EngineLibrary, load_config, model_path_from_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).parents[1]
    reference = np.load(root / "artifacts" / "reference.npz")
    boundary_ids = [int(value) for value in reference["boundary_2049_ids"]]
    model_path = model_path_from_env()
    api = EngineLibrary()
    handle = api.create(load_config(model_path))
    try:
        api.upload_model(handle, model_path)
        for cache_len in (128, 1024):
            samples = []
            prompt = (boundary_ids * (cache_len // len(boundary_ids) + 1))[:cache_len]
            token_ids = (C.c_int * len(prompt))(*prompt)
            for _ in range(args.repeats):
                api.check(api.lib.engine_reset(handle))
                token = api.lib.engine_prefill(handle, token_ids, len(prompt))
                if token < 0:
                    raise RuntimeError(api.error())
                for _ in range(args.warmup_steps):
                    token = api.lib.engine_decode(handle, token)
                    if token < 0:
                        raise RuntimeError(api.error())
                start = time.perf_counter()
                for _ in range(args.timed_steps):
                    token = api.lib.engine_decode(handle, token)
                    if token < 0:
                        raise RuntimeError(api.error())
                samples.append((time.perf_counter() - start) * 1000 / args.timed_steps)
            print(
                f"cache={cache_len} samples_ms_per_token={samples} "
                f"median_ms_per_token={statistics.median(samples):.6f}",
                flush=True,
            )
    finally:
        api.lib.engine_destroy(handle)


if __name__ == "__main__":
    main()
