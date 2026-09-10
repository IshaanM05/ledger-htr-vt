"""Generate transcriptions from a fine-tuned (or base) InternVL3 checkpoint
and either score against our verified metric (val split, Target known) or
write a Test.csv-format submission (Target unknown).

Image preprocessing below is copied from InternVL3-8B's own Hugging Face
model card "Quick Start" example verbatim (dynamic tiling into 448x448
patches, ImageNet normalization) -- not re-derived, since getting this
subtly wrong would silently degrade a model that was trained on exactly
this preprocessing.
"""

import argparse
import os
import shutil

import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from ledger_htr.metrics.scorer import final_score_from_dicts

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TRANSCRIBE_PROMPT = (
    "<image>\nTranscribe the handwritten text in this image exactly as written, "
    "preserving original spelling, punctuation, and abbreviations. Output only "
    "the transcription, nothing else."
)


def build_transform(input_size: int):
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_path: str, input_size: int = 448, max_num: int = 12) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])


_CUSTOM_CODE_FILES = [
    "configuration_intern_vit.py",
    "configuration_internvl_chat.py",
    "conversation.py",
    "modeling_intern_vit.py",
    "modeling_internvl_chat.py",
]


def ensure_custom_code_files(checkpoint_dir: str, base_model_id: str = "OpenGVLab/InternVL3-8B") -> None:
    """InternVL is a trust_remote_code model: its config's `auto_map` points at
    custom .py files that must sit alongside the weights to load. The
    fine-tuning script's checkpoint saves (HF Trainer's default save path)
    don't copy these over, so a checkpoint directory loads the weights fine
    but fails on `AutoModel.from_pretrained` with a missing-file error. Copy
    them from the base model's HF cache if they're not already present --
    idempotent, so safe to call before every load."""
    if all(os.path.exists(os.path.join(checkpoint_dir, f)) for f in _CUSTOM_CODE_FILES):
        return
    from huggingface_hub import snapshot_download

    base_snapshot = snapshot_download(base_model_id, allow_patterns=["*.py"])
    for fname in _CUSTOM_CODE_FILES:
        src = os.path.join(base_snapshot, fname)
        dst = os.path.join(checkpoint_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def load_partial_predictions(out_csv: str) -> dict[str, str]:
    """Predictions already written by a prior (possibly interrupted) run of
    this same --out-csv path, keyed by ID -- lets a rerun skip work already
    done instead of starting over from image 1."""
    if not os.path.exists(out_csv):
        return {}
    partial = pd.read_csv(out_csv, dtype={"ID": str})
    return dict(zip(partial["ID"], partial["Target"].astype(str)))


def run_inference(
    model_path: str,
    df: pd.DataFrame,
    image_dir: str,
    num_beams: int = 1,
    max_new_tokens: int = 128,
    max_num_tiles: int = 12,
    out_csv: str | None = None,
    resume: bool = True,
) -> dict[str, str]:
    """Runs inference image-by-image, writing each new prediction to
    `out_csv` immediately (append, flushed) rather than only at the end --
    a run that takes hours (beam search on the full test set) should not
    lose all its progress if the process is killed or the sandbox is
    interrupted partway through. If `out_csv` already has predictions from
    an earlier attempt at the same path, those IDs are skipped (resume)."""
    preds: dict[str, str] = {}
    if out_csv and resume:
        preds = load_partial_predictions(out_csv)
        if preds:
            print(f"resuming: {len(preds)} predictions already in {out_csv}, skipping those IDs")

    remaining = df[~df["ID"].astype(str).isin(preds.keys())]
    if remaining.empty:
        print("nothing left to do -- all IDs already predicted in the existing output file")
        return preds

    if os.path.isdir(model_path):
        ensure_custom_code_files(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = (
        AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        .eval()
        .cuda()
    )
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False, num_beams=num_beams)

    write_header = not (out_csv and os.path.exists(out_csv) and preds)
    csv_file = open(out_csv, "a", newline="") if out_csv else None
    try:
        for i, (_, row) in enumerate(remaining.iterrows()):
            image_path = os.path.join(image_dir, f"{row['ID']}.jpg")
            pixel_values = load_image(image_path, max_num=max_num_tiles).to(torch.bfloat16).cuda()
            response = model.chat(tokenizer, pixel_values, TRANSCRIBE_PROMPT, generation_config)
            text = response.strip()
            preds[str(row["ID"])] = text
            if csv_file is not None:
                if write_header:
                    csv_file.write("ID,Target\n")
                    write_header = False
                escaped = text.replace('"', '""')
                csv_file.write(f'"{row["ID"]}","{escaped}"\n')
                csv_file.flush()
                os.fsync(csv_file.fileno())
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(remaining)} done this run ({len(preds)}/{len(df)} total)")
    finally:
        if csv_file is not None:
            csv_file.close()
    return preds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF model id or local checkpoint dir")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--image-dir", default=None, help="defaults to <data-dir>/images")
    parser.add_argument("--mode", choices=["val", "test"], required=True)
    parser.add_argument("--folds-csv", default="data/processed/folds.csv")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore any existing --out-csv content and start over instead of skipping already-done IDs",
    )
    args = parser.parse_args()

    image_dir = args.image_dir or os.path.join(args.data_dir, "images")

    if args.mode == "val":
        train_df = pd.read_csv(os.path.join(args.data_dir, "Train.csv"))
        folds_df = pd.read_csv(args.folds_csv)
        merged = train_df.merge(folds_df, on="ID")
        df = merged[merged["fold"] == args.val_fold].drop(columns="fold").reset_index(drop=True)
    else:
        df = pd.read_csv(os.path.join(args.data_dir, "Test.csv")).reset_index(drop=True)

    if args.max_samples:
        df = df.head(args.max_samples)

    out_csv = args.out_csv or f"submissions/vlm_{args.mode}_predictions.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if args.no_resume and os.path.exists(out_csv):
        os.remove(out_csv)

    print(f"running inference on {len(df)} images (mode={args.mode}, num_beams={args.num_beams})")
    preds = run_inference(
        args.model_path,
        df,
        image_dir,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        out_csv=out_csv,
        resume=not args.no_resume,
    )

    # predictions are already written incrementally by run_inference (see
    # its docstring) -- no final bulk write needed, just verify completeness.
    missing = set(df["ID"].astype(str)) - set(preds.keys())
    if missing:
        raise RuntimeError(f"{len(missing)} IDs never got a prediction (e.g. {list(missing)[:5]}) -- {out_csv} is incomplete")
    print(f"wrote {len(preds)} predictions -> {out_csv}")

    if args.mode == "val":
        refs = dict(zip(df["ID"].astype(str), df["Target"].astype(str)))
        result = final_score_from_dicts(preds, refs)
        print(f"val_cer={result['cer']:.4f} val_wer={result['wer']:.4f} val_final={result['final']:.4f}")


if __name__ == "__main__":
    main()
