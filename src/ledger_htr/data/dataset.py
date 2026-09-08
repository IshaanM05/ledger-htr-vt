import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class LedgerSeq2SeqDataset(Dataset):
    """Wraps (image, transcription) pairs for a seq2seq HTR model (TrOCR-style).
    Uses the model's own processor for image preprocessing, so this defers to
    whatever resizing that processor's pretrained backbone expects.

    `transform` (an albumentations.Compose, e.g. from augment.build_elastic_transform)
    is applied before the processor, train-split only — pass None for validation/
    inference so scoring always happens against clean images."""

    def __init__(self, df: pd.DataFrame, image_dir: str, processor, max_target_length: int = 140, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['ID']}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            from ledger_htr.augment.augment import apply_transform

            image = apply_transform(image, self.transform)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            row["Target"],
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
        ).input_ids
        labels = [
            label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels
        ]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def load_fold_split(train_csv: str, folds_csv: str, val_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split Train.csv into train/val DataFrames using the precomputed fold assignments."""
    train_df = pd.read_csv(train_csv)
    folds_df = pd.read_csv(folds_csv)
    merged = train_df.merge(folds_df, on="ID")
    train_split = merged[merged["fold"] != val_fold].drop(columns="fold").reset_index(drop=True)
    val_split = merged[merged["fold"] == val_fold].drop(columns="fold").reset_index(drop=True)
    return train_split, val_split
