import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ledger_htr.data.dataset import LedgerSeq2SeqDataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class InferenceDataset(LedgerSeq2SeqDataset):
    """Same image preprocessing as training, but no labels (test set has none)."""

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['ID']}.jpg")
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return {"pixel_values": pixel_values, "ID": row["ID"]}


@torch.no_grad()
def predict(checkpoint_dir: str, csv_path: str, image_dir: str, batch_size: int = 8) -> dict[str, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir).to(device)
    model.eval()
    model.config.use_cache = True

    df = pd.read_csv(csv_path)
    ds = InferenceDataset(df, image_dir, processor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=_collate)

    predictions: dict[str, str] = {}
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        generated_ids = model.generate(pixel_values=pixel_values, max_length=model.generation_config.max_length)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for id_, text in zip(batch["ID"], texts):
            predictions[str(id_)] = text
    return predictions


def _collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "ID": [b["ID"] for b in batch],
    }


def write_submission(predictions: dict[str, str], sample_submission_csv: str, out_path: str) -> None:
    """Writes a submission.csv covering every ID in SampleSubmission.csv, in its
    order, filling anything missing with an empty string (so a completeness gap
    is visible rather than silently dropped — the plan flags empty preds as
    scored flat-out wrong regardless, but a missing ID crashing submission is worse)."""
    sample = pd.read_csv(sample_submission_csv)
    missing = [i for i in sample["ID"].astype(str) if i not in predictions]
    if missing:
        print(f"WARNING: {len(missing)} IDs from SampleSubmission.csv have no prediction: {missing[:5]}...")

    sample["Target"] = sample["ID"].astype(str).map(lambda i: predictions.get(i, ""))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sample.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(sample)} rows, {len(missing)} missing)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-csv", default=os.path.join(REPO_ROOT, "data", "raw", "Test.csv"))
    parser.add_argument("--image-dir", default=os.path.join(REPO_ROOT, "data", "raw", "images"))
    parser.add_argument("--sample-submission", default=os.path.join(REPO_ROOT, "data", "raw", "SampleSubmission.csv"))
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "submissions", "submission.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
        tmp_csv = "/tmp/_test_subset.csv"
        test_df.to_csv(tmp_csv, index=False)
        args.test_csv = tmp_csv

    preds = predict(args.checkpoint, args.test_csv, args.image_dir, batch_size=args.batch_size)
    write_submission(preds, args.sample_submission, args.out)
