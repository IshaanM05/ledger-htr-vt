# ledger-htr-vt

Solo/team entry into the **R.O.A.D. Barbados Historic Handwriting Challenge** (Zindi) — transcribing 18th–19th century Barbadian legal handwriting (deeds, wills, estate inventories) from pre-cropped line/word images.

Full plan, reading list, and day-by-day schedule: [docs/plan.md](docs/plan.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Layout

```
data/raw/            Train.csv, Test.csv, images/ — as downloaded from Zindi (gitignored)
data/processed/       preprocessed crops, CV fold assignments (gitignored)
external/             reference clones (HTR-VT, One-DM, ...) (gitignored)
checkpoints/          model weights (gitignored)
notebooks/            EDA and experiment notebooks
src/ledger_htr/
  data/                dataset loading, CV splitting, preprocessing
  models/               HTR-VT / TrOCR / VLM wrappers
  metrics/              weighted WER/CER scorer
  augment/               augmentation pipeline
  decode/                 beam search / lexicon rescoring
docs/plan.md           the full project plan
```

## Data

Not included — download `Train.csv`, `Test.csv`, `images.zip`, `Starters.zip` from the Zindi competition page and unpack into `data/raw/`.

## Metric

`0.5 × weighted_WER + 0.5 × weighted_CER`, length-weighted by `sqrt(reference_length)` per sample — see [docs/plan.md](docs/plan.md#metric--replicate-it-exactly-dont-approximate) for the exact formula. Do not rely on plain `jiwer.wer()`/`jiwer.cer()` for model selection.
