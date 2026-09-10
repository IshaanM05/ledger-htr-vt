"""Extract just the LoRA delta weights from a full InternVL fine-tune
checkpoint. HF Trainer's checkpoint save for this custom trust_remote_code
model writes the entire merged model (~16GB: frozen base weights + LoRA
layers all in one state_dict), not a small peft-style adapter file -- but
only the `lora_A`/`lora_B` tensors (~68M params for rank 16, a few hundred
MB) were actually trained. The frozen base weights are just OpenGVLab/
InternVL3-8B unchanged and don't need to be duplicated or transferred
anywhere; extracting only the trained delta makes the checkpoint small
enough to actually move off a remote sandbox."""

import argparse
import json
import os

from safetensors.torch import load_file, save_file


def extract_lora_weights(checkpoint_dir: str, out_path: str, extra_trained_prefixes: tuple[str, ...] = ("mlp1.",)) -> dict:
    """`extra_trained_prefixes` covers parameters that were fully fine-tuned
    rather than LoRA-wrapped (e.g. this project's runs use --freeze_mlp False,
    so the mlp1 vision-to-language projector is directly trained, not just
    the LoRA_A/B deltas) -- without these, the extracted file would silently
    be missing part of what actually changed from the base model."""
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]
    lora_keys = [
        k
        for k in weight_map
        if "lora_A" in k or "lora_B" in k or any(k.startswith(p) for p in extra_trained_prefixes)
    ]
    if not lora_keys:
        raise ValueError(f"no lora_A/lora_B tensors found in {index_path} -- wrong checkpoint, or not a LoRA run?")

    shard_to_keys: dict[str, list[str]] = {}
    for k in lora_keys:
        shard_to_keys.setdefault(weight_map[k], []).append(k)

    result = {}
    for shard_file, keys in shard_to_keys.items():
        tensors = load_file(os.path.join(checkpoint_dir, shard_file))
        for k in keys:
            result[k] = tensors[k]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    save_file(result, out_path)
    total_params = sum(t.numel() for t in result.values())
    print(f"extracted {len(result)} LoRA tensors, {total_params:,} params -> {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--out-path", required=True)
    args = parser.parse_args()
    extract_lora_weights(args.checkpoint_dir, args.out_path)
