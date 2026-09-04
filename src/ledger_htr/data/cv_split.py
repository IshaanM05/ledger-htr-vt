import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ledger_htr.data.known_issues import CORRUPTED_TRAIN_IDS

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")

SEED = 42


def make_folds(train_df: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> pd.DataFrame:
    char_lens = train_df["Target"].str.len()
    length_bins = pd.qcut(char_lens, q=n_splits, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.empty(len(train_df), dtype=int)
    for fold, (_, val_idx) in enumerate(skf.split(train_df, length_bins)):
        folds[val_idx] = fold

    return pd.DataFrame({"ID": train_df["ID"], "fold": folds})


def main() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train = pd.read_csv(os.path.join(RAW_DIR, "Train.csv"))
    n_before = len(train)
    train = train[~train["ID"].isin(CORRUPTED_TRAIN_IDS)].reset_index(drop=True)
    print(f"Excluded {n_before - len(train)} known-corrupted rows (see data/known_issues.py)")

    folds_df = make_folds(train)

    merged = train.merge(folds_df, on="ID")
    merged["char_len"] = merged["Target"].str.len()

    print(f"Total rows: {len(folds_df)}")
    for fold in sorted(folds_df["fold"].unique()):
        subset = merged[merged["fold"] == fold]
        print(
            f"  fold {fold}: n={len(subset)}, "
            f"char_len mean={subset['char_len'].mean():.2f}, std={subset['char_len'].std():.2f}"
        )

    out_path = os.path.join(PROCESSED_DIR, "folds.csv")
    folds_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
