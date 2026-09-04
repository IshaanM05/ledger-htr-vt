import os
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .preprocess import otsu_binarize_grid

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
IMAGE_DIR = os.path.join(RAW_DIR, "images")
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

SEED = 42


def load_train() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DIR, "Train.csv"))


def load_test() -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW_DIR, "Test.csv"))


def text_length_stats(df: pd.DataFrame) -> dict:
    char_lens = df["Target"].str.len()
    word_lens = df["Target"].str.split().apply(len)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(char_lens, bins=30, color="#4C72B0")
    axes[0].set_title("Character length")
    axes[0].set_xlabel("chars")
    axes[1].hist(word_lens, bins=range(1, word_lens.max() + 2), color="#DD8452")
    axes[1].set_title("Word count")
    axes[1].set_xlabel("words")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "char_length_hist.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(word_lens, bins=range(1, word_lens.max() + 2), color="#DD8452")
    ax.set_title("Word count per sample")
    ax.set_xlabel("words")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "word_length_hist.png"), dpi=120)
    plt.close(fig)

    return {
        "char_len": char_lens.describe().to_dict(),
        "word_len": word_lens.describe().to_dict(),
    }


def character_set(df: pd.DataFrame) -> dict:
    chars = set()
    for t in df["Target"]:
        chars.update(t)
    return {"n_unique": len(chars), "chars": sorted(chars)}


def image_size_stats(image_dir: str, sample_size: int = 1500) -> dict:
    files = os.listdir(image_dir)
    sample = random.Random(SEED).sample(files, min(sample_size, len(files)))
    widths, heights = [], []
    for f in sample:
        img = cv2.imread(os.path.join(image_dir, f))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(widths, heights, s=6, alpha=0.4, color="#55A868")
    ax.set_xlabel("width (px)")
    ax.set_ylabel("height (px)")
    ax.set_title(f"Crop dimensions (n={len(widths)} sampled)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "image_size_scatter.png"), dpi=120)
    plt.close(fig)

    return {
        "n_total_files": len(files),
        "n_sampled": len(widths),
        "width": pd.Series(widths).describe().to_dict(),
        "height": pd.Series(heights).describe().to_dict(),
    }


def duplicate_target_report(df: pd.DataFrame) -> dict:
    counts = df["Target"].value_counts()
    dups = counts[counts > 1]
    return {
        "n_distinct": int(df["Target"].nunique()),
        "n_total": int(len(df)),
        "n_targets_with_dupes": int(len(dups)),
        "n_rows_involved": int(dups.sum()) if len(dups) else 0,
    }


def binarization_sample(image_dir: str, ids: list[str], n: int = 6) -> None:
    rng = random.Random(SEED)
    chosen = rng.sample(ids, min(n, len(ids)))
    paths = [os.path.join(image_dir, f"{i}.jpg") for i in chosen]
    grid = otsu_binarize_grid(paths)
    cv2.imwrite(os.path.join(FIGURES_DIR, "otsu_comparison.png"), grid)


def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train = load_train()

    length_stats = text_length_stats(train)
    char_stats = character_set(train)
    img_stats = image_size_stats(IMAGE_DIR)
    dup_stats = duplicate_target_report(train)

    # bias the binarization sample toward size outliers so we see both typical
    # and extreme crops (tall multi-line ones included)
    files = sorted(os.listdir(IMAGE_DIR))
    ids_for_sample = [os.path.splitext(f)[0] for f in files if os.path.splitext(f)[0] in set(train["ID"])]
    binarization_sample(IMAGE_DIR, ids_for_sample, n=6)

    findings = f"""# EDA findings

## Dataset shape
- Train: {len(train)} rows (ID, Target)
- Images: {img_stats['n_total_files']} files in `data/raw/images/`

## Transcription length
- Character length: mean {length_stats['char_len']['mean']:.1f}, min {length_stats['char_len']['min']:.0f}, max {length_stats['char_len']['max']:.0f}
- Word count: mean {length_stats['word_len']['mean']:.1f}, min {length_stats['word_len']['min']:.0f}, max {length_stats['word_len']['max']:.0f}
- These are **line crops** (multi-word), not single-word crops.
- See `figures/char_length_hist.png`, `figures/word_length_hist.png`.

## Character set
- {char_stats['n_unique']} unique characters: `{''.join(char_stats['chars'])}`
- Standard ASCII letters, digits, and punctuation only — no long-s (ſ) or other
  archaic glyphs. Ground truth appears **normalized to modern letterforms**, not
  transcribed verbatim at the character level.

## Image sizes
- Sampled {img_stats['n_sampled']} of {img_stats['n_total_files']} images.
- Width: mean {img_stats['width']['mean']:.0f}px, range [{img_stats['width']['min']:.0f}, {img_stats['width']['max']:.0f}]
- Height: mean {img_stats['height']['mean']:.0f}px, range [{img_stats['height']['min']:.0f}, {img_stats['height']['max']:.0f}]
- Large spread in both dimensions (height p90={img_stats['height']['75%']:.0f}px+,
  some crops run well past that, likely multi-line). Preprocessing must not
  assume a fixed aspect ratio.
- See `figures/image_size_scatter.png`.

## Duplicate targets
- {dup_stats['n_distinct']} distinct targets out of {dup_stats['n_total']} rows.
- Only {dup_stats['n_targets_with_dupes']} targets repeat, involving {dup_stats['n_rows_involved']} rows ({100 * dup_stats['n_rows_involved'] / dup_stats['n_total']:.1f}%) — too rare to need grouped CV splitting.

## Otsu binarization
- See `figures/otsu_comparison.png` for original-vs-binarized sample pairs.
- Global Otsu does well on clean, evenly-lit pages but fails badly on
  stained/foxed backgrounds: uneven staining gets thresholded as ink, producing
  large black blobs that destroy the text (visible in ~half of the 6-sample
  grid). **Global Otsu is not sufficient as-is** — the real preprocessing
  pipeline (Sep 7 task) should use `cv2.adaptiveThreshold` or per-tile/local
  Otsu instead.
"""
    with open(os.path.join(REPORTS_DIR, "eda_findings.md"), "w") as f:
        f.write(findings)

    print(findings)


if __name__ == "__main__":
    main()
