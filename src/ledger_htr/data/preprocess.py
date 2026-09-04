import os

import cv2
import numpy as np


def resize_fixed_height(image: np.ndarray, target_height: int = 64) -> np.ndarray:
    """Resize to a fixed height, preserving aspect ratio (width follows)."""
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, target_height), interpolation=interp)


def pad_to_width(image: np.ndarray, target_width: int, pad_value: int = 255) -> tuple[np.ndarray, int]:
    """Right-pad width to target_width. Returns (padded_image, valid_width) —
    valid_width is the original (pre-pad) width, for building an attention/CTC
    padding mask downstream. Raises if the image is already wider than target."""
    h, w = image.shape[:2]
    if w > target_width:
        raise ValueError(f"image width {w} exceeds target_width {target_width}; widen the bucket")
    pad_width = target_width - w
    if image.ndim == 3:
        pad_shape = (h, pad_width, image.shape[2])
    else:
        pad_shape = (h, pad_width)
    pad = np.full(pad_shape, pad_value, dtype=image.dtype)
    return np.hstack([image, pad]), w


def compute_width_buckets(widths: list[int], n_buckets: int = 10, round_to: int = 32) -> list[int]:
    """Quantile-based bucket boundaries over a width distribution, rounded up
    to `round_to` (conv-stride-friendly). Assign a sample's width to the
    smallest boundary >= it via `assign_bucket`, so batches share one padded
    width instead of every sample padding to the global max."""
    quantiles = np.quantile(widths, np.linspace(0, 1, n_buckets + 1)[1:])
    boundaries = sorted({int(np.ceil(q / round_to) * round_to) for q in quantiles})
    return boundaries


def assign_bucket(width: int, boundaries: list[int]) -> int:
    """Smallest boundary >= width; falls back to the widest bucket if width exceeds all of them."""
    for b in boundaries:
        if width <= b:
            return b
    return boundaries[-1]


def otsu_binarize(image: np.ndarray) -> np.ndarray:
    """Grayscale + mild blur + global Otsu threshold. Returns a binary (0/255) image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binarized = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def otsu_binarize_grid(image_paths: list[str], max_width: int = 900) -> np.ndarray:
    """Stack original/binarized pairs (each resized to max_width) into one grid image."""
    rows = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        if w > max_width:
            scale = max_width / w
            img = cv2.resize(img, (max_width, int(h * scale)))
        binarized = otsu_binarize(img)
        binarized_bgr = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
        pad = np.full((4, img.shape[1], 3), 255, dtype=np.uint8)
        rows.append(np.vstack([img, pad, binarized_bgr]))

    max_row_width = max(r.shape[1] for r in rows)
    padded_rows = []
    sep = np.full((6, max_row_width, 3), 200, dtype=np.uint8)
    for i, r in enumerate(rows):
        if r.shape[1] < max_row_width:
            pad = np.full((r.shape[0], max_row_width - r.shape[1], 3), 255, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded_rows.append(r)
        if i != len(rows) - 1:
            padded_rows.append(sep)

    return np.vstack(padded_rows)


def _demo() -> None:
    """Sanity check resize+bucket+pad against real data. Run: python -m src.ledger_htr.data.preprocess"""
    import random

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    image_dir = os.path.join(repo_root, "data", "raw", "images")
    files = os.listdir(image_dir)
    sample = random.Random(42).sample(files, min(500, len(files)))

    target_height = 64
    resized_widths = []
    for f in sample:
        img = cv2.imread(os.path.join(image_dir, f))
        resized = resize_fixed_height(img, target_height)
        resized_widths.append(resized.shape[1])

    boundaries = compute_width_buckets(resized_widths, n_buckets=10)
    print(f"Sampled {len(sample)} images, resized to height={target_height}")
    print(f"Resized width range: [{min(resized_widths)}, {max(resized_widths)}]")
    print(f"Bucket boundaries: {boundaries}")

    counts = {b: 0 for b in boundaries}
    for w in resized_widths:
        counts[assign_bucket(w, boundaries)] += 1
    for b in boundaries:
        print(f"  bucket <= {b}: {counts[b]} samples")

    # verify pad_to_width round-trips a real sample without error
    img = cv2.imread(os.path.join(image_dir, sample[0]))
    resized = resize_fixed_height(img, target_height)
    bucket = assign_bucket(resized.shape[1], boundaries)
    padded, valid_width = pad_to_width(resized, bucket)
    assert padded.shape[1] == bucket
    assert valid_width == resized.shape[1]
    print(f"\npad_to_width sanity check OK: {resized.shape} -> {padded.shape}, valid_width={valid_width}")


if __name__ == "__main__":
    _demo()
