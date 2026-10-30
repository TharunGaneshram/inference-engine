#!/usr/bin/env python3
"""Create fp32 Hugging Face reference arrays for the Phase 0 model tests."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from python.engine import model_path_from_env


def build_prompts(tokenizer):
    one = tokenizer("Hello", add_special_tokens=False).input_ids[:1]
    short = tokenizer("The capital of France is", add_special_tokens=False).input_ids
    medium_seed = tokenizer(
        "A quiet river moves beneath the old stone bridge.", add_special_tokens=False
    ).input_ids
    repeat_seed = tokenizer("ha ha ha! ", add_special_tokens=False).input_ids
    return {
        "single": one,
        "short": short,
        "medium_500": (medium_seed * (500 // len(medium_seed) + 1))[:500],
        "repetitive_128": (repeat_seed * (128 // len(repeat_seed) + 1))[:128],
        "boundary_2049": (medium_seed * (2049 // len(medium_seed) + 1))[:2049],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "reference.npz"))
    parser.add_argument("--max-new", type=int, default=64)
    args = parser.parse_args()

    model_path = model_path_from_env()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    ).to("cuda:0").eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    prompts = build_prompts(tokenizer)
    arrays = {}

    with torch.inference_mode():
        for name, ids in prompts.items():
            input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
            captured = {}
            hook = None
            if name in {"single", "short"}:
                def capture_last_layer(_module, _inputs, output):
                    captured["residual"] = (output[0] if isinstance(output, tuple) else output)[0].float().cpu().numpy()
                hook = model.model.layers[-1].register_forward_hook(capture_last_layer)
            output = model(
                input_ids,
                use_cache=False,
                output_hidden_states=name in {"single", "short"},
            )
            if hook is not None:
                hook.remove()
            arrays[f"{name}_ids"] = np.asarray(ids, dtype=np.int32)
            arrays[f"{name}_top1"] = output.logits.argmax(dim=-1)[0].cpu().numpy().astype(np.int32)
            arrays[f"{name}_last_logits"] = output.logits[0, -1].float().cpu().numpy()
            if name in {"single", "short"}:
                for i, hidden in enumerate(output.hidden_states):
                    arrays[f"{name}_hidden_{i}"] = hidden[0].float().cpu().numpy()
                arrays[f"{name}_pre_final"] = captured["residual"]
            del output, input_ids
            torch.cuda.empty_cache()

            input_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
            generated = model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=args.max_new,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
            )
            arrays[f"{name}_greedy_ids"] = generated.sequences[0, len(ids):].cpu().numpy().astype(np.int32)
            if generated.logits:
                arrays[f"{name}_greedy_logits"] = torch.stack(
                    [step[0].float() for step in generated.logits]
                ).cpu().numpy()
            del generated, input_ids
            torch.cuda.empty_cache()
            print(f"referenced {name}: prompt={len(ids)} generated={len(arrays[f'{name}_greedy_ids'])}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    metadata = {
        "model_path": str(model_path),
        "dtype": "float32",
        "max_new_tokens": args.max_new,
        "prompts": {name: len(ids) for name, ids in prompts.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"saved {len(arrays)} arrays to {output}")


if __name__ == "__main__":
    main()
