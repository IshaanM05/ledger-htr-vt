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


def load_lora_weights_into_base_model(
    adapter_path: str,
    base_model_id: str = "OpenGVLab/InternVL3-8B",
    lora_rank: int = 16,
):
    """The other half of extract_lora_weights: rebuild a usable fine-tuned
    model from just the small extracted delta file, without ever needing
    the original 16GB checkpoint again. Loads the base model fresh, wraps
    it with the SAME LoRA config the training run used (rank must match --
    this project's runs used --use_llm_lora 16), then loads the extracted
    tensors with strict=False since the file only contains this subset of
    keys (lora_A/B + mlp1.*) -- everything else stays at its base value,
    exactly like loading a peft adapter onto a base model, just via
    InternVL's own wrap_llm_lora() instead of peft.PeftModel."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    )
    model.wrap_llm_lora(r=lora_rank, lora_alpha=2 * lora_rank)
    adapter_weights = load_file(adapter_path)
    missing, unexpected = model.load_state_dict(adapter_weights, strict=False)
    if unexpected:
        raise ValueError(f"adapter file has keys that don't exist in a freshly-LoRA-wrapped base model: {unexpected[:5]}")
    print(f"loaded {len(adapter_weights)} tensors from {adapter_path} onto a fresh {base_model_id} + LoRA(r={lora_rank})")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", help="full checkpoint dir to extract the LoRA delta from")
    parser.add_argument("--out-path", help="where to write (extract mode) or read from (--load-into mode) the small adapter file")
    parser.add_argument("--load-into", action="store_true", help="reload mode: rebuild the model from --out-path instead of extracting")
    args = parser.parse_args()

    if args.load_into:
        load_lora_weights_into_base_model(args.out_path)
    else:
        extract_lora_weights(args.checkpoint_dir, args.out_path)
