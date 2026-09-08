import albumentations as A
import cv2
import numpy as np
from PIL import Image


def build_elastic_transform(target_size: tuple[int, int] = (384, 384), p: float = 0.5) -> A.Compose:
    """Elastic distortion only — the plan's Must-tier augmentation validates one
    technique at a time against CV, so this stays deliberately single-purpose.
    Resizes to target_size first so alpha/sigma are tuned against a consistent
    scale rather than this dataset's wildly varying raw crop sizes (267-5746px
    wide per EDA); target_size matches TrOCR's own processor resolution so the
    processor's internal resize becomes a no-op. alpha/sigma follow typical
    HTR-augmentation literature values (mild enough to keep letterforms legible).
    White fill for the constant border keeps out-of-bounds pixels consistent
    with this archive's paper background rather than a black default.
    """
    return A.Compose(
        [
            A.Resize(height=target_size[0], width=target_size[1]),
            A.ElasticTransform(
                alpha=40,
                sigma=6,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(255, 255, 255),
                p=p,
            ),
        ]
    )


def apply_transform(image: Image.Image, transform: A.Compose) -> np.ndarray:
    return transform(image=np.array(image))["image"]
