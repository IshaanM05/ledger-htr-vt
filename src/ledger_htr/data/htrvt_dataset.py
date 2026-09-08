import os

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def build_char_list(train_csv: str, folds_csv: str) -> list[str]:
    """Alphabet for the CTC head, built from every non-corrupted training row
    (train+val across all folds) so it's stable regardless of which fold is
    held out. folds_csv already excludes the organizer-acknowledged corrupted
    rows (see data/known_issues.py, data/cv_split.py), so the inner merge
    here drops them automatically."""
    train_df = pd.read_csv(train_csv)
    folds_df = pd.read_csv(folds_csv)
    merged = train_df.merge(folds_df, on="ID")
    chars = set("".join(merged["Target"].astype(str)))
    return sorted(chars)


def _resize_and_pad(arr: np.ndarray, target_height: int, max_width: int) -> np.ndarray:
    """Aspect-preserving resize to target_height, then pad (or clamp) width to
    max_width with white — mirrors HTR-VT's own npThum/get_images preprocessing,
    which is tuned for exactly this kind of variable-width line crop."""
    orig_h, orig_w = arr.shape[:2]
    new_w = min(round(orig_w * target_height / orig_h), max_width)
    new_w = max(new_w, 1)
    resized = cv2.resize(arr, (new_w, target_height), interpolation=cv2.INTER_AREA)
    if new_w < max_width:
        pad = np.full((target_height, max_width - new_w), 255, dtype=resized.dtype)
        resized = np.concatenate([resized, pad], axis=1)
    return resized


class HTRVTLedgerDataset(Dataset):
    """Line-crop dataset for HTR-VT's CTC training: grayscale, resized to a
    fixed height with aspect ratio preserved, then padded/clamped to a fixed
    width. Returns (image_tensor[1,H,W] in [0,1], target_text) — the caller
    (train_htrvt.py) is responsible for CTC label encoding via
    HTR-VT's own CTCLabelConverter, matching the upstream repo's convention.

    `transform` (an albumentations.Compose, e.g. augment.build_elastic_transform_raw)
    is applied to the raw grayscale crop before resize/pad, train-split only."""

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        target_height: int = 64,
        max_width: int = 1024,
        transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.target_height = target_height
        self.max_width = max_width
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_dir, f"{row['ID']}.jpg")
        image = Image.open(image_path).convert("L")
        arr = np.array(image)
        if self.transform is not None:
            from ledger_htr.augment.augment import apply_transform

            arr = apply_transform(Image.fromarray(arr), self.transform)
        arr = _resize_and_pad(arr, self.target_height, self.max_width)
        tensor = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0)
        return tensor, str(row["Target"])


def htrvt_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    texts = [item[1] for item in batch]
    return images, texts
