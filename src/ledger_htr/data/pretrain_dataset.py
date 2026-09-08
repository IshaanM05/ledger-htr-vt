import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ledger_htr.data.htrvt_dataset import _resize_and_pad


def all_image_ids(data_dir: str) -> list[str]:
    """Every image ID across Train.csv + Test.csv, no labels -- self-supervised
    pretraining is explicitly allowed on "train and test pixels, no labels
    touched" (docs/plan.md), unlike supervised training which is restricted
    to Train.csv only. Includes the organizer-flagged corrupted rows too:
    that flag is about label quality, and pretraining never looks at labels."""
    train_df = pd.read_csv(os.path.join(data_dir, "Train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    return pd.concat([train_df["ID"], test_df["ID"]]).astype(str).tolist()


class MaskedImageDataset(Dataset):
    """Unlabeled line-crop images for masked-image-modeling pretraining --
    same resize/pad preprocessing as HTRVTLedgerDataset so the pretrained
    encoder sees the exact same input distribution it will be fine-tuned on."""

    def __init__(self, image_ids: list[str], image_dir: str, target_height: int, max_width: int):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.target_height = target_height
        self.max_width = max_width

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("L")
        arr = _resize_and_pad(np.array(image), self.target_height, self.max_width)
        return torch.from_numpy(arr).float().div_(255.0).unsqueeze(0)


def mask_image_strips(
    images: torch.Tensor, mask_ratio: float, num_strips: int, fill_value: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask random contiguous vertical strips per-sample (width is the
    sequence/time axis for this line-image architecture, so masking vertical
    strips forces the model to infer missing content from horizontal
    context -- the image-domain analogue of the span-masking already used
    for supervised regularization in HTRViT). `fill_value=1.0` (white)
    matches this archive's page background, i.e. masking looks like erased
    ink rather than an out-of-distribution solid block.

    Returns (masked_images, mask) where mask is 1.0 at masked pixels, 0.0
    elsewhere -- used to restrict the reconstruction loss to only the
    positions actually held out, per MAE's core principle."""
    b, c, h, w = images.shape
    masked = images.clone()
    mask = torch.zeros(b, 1, h, w, device=images.device, dtype=images.dtype)
    strip_width = max(int(w * mask_ratio / max(num_strips, 1)), 1)
    for i in range(b):
        for _ in range(num_strips):
            start = int(torch.randint(0, max(w - strip_width, 1), (1,)).item())
            masked[i, :, :, start : start + strip_width] = fill_value
            mask[i, :, :, start : start + strip_width] = 1.0
    return masked, mask
