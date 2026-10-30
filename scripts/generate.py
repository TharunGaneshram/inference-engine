import argparse
import time

from transformers import AutoTokenizer

from engine import EngineLibrary, load_config, model_path_from_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="The capital of France is")
    parser.add_argument("--max-new", type=int, default=8)
    args = parser.parse_args()

    model_path = model_path_from_env()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompt_ids = tokenizer(args.prompt, return_tensors=None)["input_ids"]
    api = EngineLibrary()
    handle = api.create(load_config(model_path))
    try:
        api.upload_model(handle, model_path)
        start = time.perf_counter()
        generated = api.generate(handle, prompt_ids, args.max_new)
        elapsed = time.perf_counter() - start
        print(f"prompt: {args.prompt}")
        print(f"continuation: {tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)}")
        print(f"generated {len(generated)} tokens in {elapsed:.3f}s")
    finally:
        api.lib.engine_destroy(handle)


if __name__ == "__main__":
    main()
