import json
import os

import pandas as pd

from ledger_htr.data.known_issues import CORRUPTED_TRAIN_IDS

TRANSCRIBE_PROMPT = (
    "<image>\nTranscribe the handwritten text in this image exactly as written, "
    "preserving original spelling, punctuation, and abbreviations. Output only "
    "the transcription, nothing else."
)


def build_conversations(target_text: str) -> list[dict]:
    """ShareGPT/LLaVA-style conversation turns -- the format InternVL's own
    fine-tuning dataset loader (internvl/train/dataset.py) expects: each turn
    is {"from": "human"|"gpt", "value": "..."}, confirmed directly against
    their source rather than assumed from a generic LLaVA convention."""
    return [
        {"from": "human", "value": TRANSCRIBE_PROMPT},
        {"from": "gpt", "value": target_text},
    ]


def write_internvl_jsonl(
    df: pd.DataFrame, image_subdir: str, out_path: str, exclude_corrupted: bool = True
) -> int:
    """Writes one JSON object per line: {"id", "image", "conversations"}.
    `image` is relative to whatever `root` the caller's meta.json points at
    (this project uses `data/raw/`, so `image_subdir` is just "images").
    Returns the number of rows written."""
    if exclude_corrupted:
        df = df[~df["ID"].isin(CORRUPTED_TRAIN_IDS)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            entry = {
                "id": n,
                "image": os.path.join(image_subdir, f"{row['ID']}.jpg"),
                "conversations": build_conversations(str(row["Target"])),
            }
            f.write(json.dumps(entry) + "\n")
            n += 1
    return n


def write_meta_json(dataset_name: str, jsonl_path: str, image_root: str, length: int, out_path: str) -> None:
    meta = {
        dataset_name: {
            "root": image_root,
            "annotation": jsonl_path,
            "data_augment": False,
            "repeat_time": 1,
            "length": length,
        }
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--folds-csv", default="data/processed/folds.csv")
    parser.add_argument("--val-fold", type=int, default=0)
    parser.add_argument("--out-dir", default="data/vlm_finetune")
    args = parser.parse_args()

    train_df = pd.read_csv(os.path.join(args.data_dir, "Train.csv"))
    folds_df = pd.read_csv(args.folds_csv)
    merged = train_df.merge(folds_df, on="ID")
    train_split = merged[merged["fold"] != args.val_fold].drop(columns="fold")
    val_split = merged[merged["fold"] == args.val_fold].drop(columns="fold")

    image_root = os.path.abspath(os.path.join(args.data_dir, "images"))
    train_jsonl = os.path.join(args.out_dir, "train.jsonl")
    val_jsonl = os.path.join(args.out_dir, "val.jsonl")

    n_train = write_internvl_jsonl(train_split, ".", train_jsonl)
    n_val = write_internvl_jsonl(val_split, ".", val_jsonl, exclude_corrupted=False)
    print(f"wrote {n_train} train rows -> {train_jsonl}")
    print(f"wrote {n_val} val rows -> {val_jsonl} (fold {args.val_fold}, corrupted rows kept -- this is for our own eval, not training)")

    write_meta_json(
        "ledger_htr_train",
        os.path.abspath(train_jsonl),
        image_root,
        n_train,
        os.path.join(args.out_dir, "meta.json"),
    )
    print(f"wrote meta.json -> {os.path.join(args.out_dir, 'meta.json')}")


if __name__ == "__main__":
    main()
