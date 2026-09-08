# HTR-VT — handoff for the molab/RTX 6000 session

Everything needed to run HTR-VT (this track's anchor model, per the current
plan — see `docs/plan.md`) on the RTX 6000 session, independent of the
local 12GB laptop run.

## 0. Status as of this handoff

- Fully **independent implementation** — not a port of the reference repo.
  `github.com/YutingLi0606/HTR-VT` ships with no LICENSE file, which the
  current plan flags as a real compliance risk for a competition with a
  code-review requirement (see `docs/plan.md`'s License audit). An earlier
  version of this code directly imported their model/optimizer/CTC-utility
  code; it was rewritten from the architecture description instead — see
  `src/ledger_htr/models/htr_vt.py` (CNN stem + ViT + CTC head, span
  masking), `src/ledger_htr/optim/sam.py` (SAM optimizer), and
  `src/ledger_htr/decode/ctc_codec.py` (CTC encode/decode). No dependency
  on `external/HTR-VT` remains — don't clone it for this.
- Smoke-tested locally (20-iteration run on 40 samples, no shape/dtype
  errors, 39.4M params). Not yet run for real — genuinely new, no prior
  HTR-VT numbers exist on this dataset yet.
- Verified/optimized before handoff: bf16 autocast on the forward pass
  (`use_bf16` in the config — no GradScaler needed, so it composes cleanly
  with SAM's two-step update, unlike fp16), configurable `num_workers`/
  `pin_memory`/`persistent_workers` on both DataLoaders, and a resize-
  interpolation fix (`INTER_LINEAR` when upscaling short crops, not
  `INTER_AREA` for every crop regardless of direction — EDA found crops as
  short as 34px tall, well under the 64px target). 13 unit tests pass
  (`tests/test_htrvt.py` covers the CTC codec's encode/decode logic and the
  model's forward shape/masking; `tests/test_scorer.py` is unrelated,
  pre-existing).
- Per the current plan, this is the **anchor model** for this track (not
  Should-tier/optional) — HTR-VT, PARSeq, and later InternVL3 are this
  side's models; TrOCR/Qwen3-VL/PyLaia/PP-OCRv6 belong to the teammate's
  track and shouldn't be duplicated here.

## 1. Get the repo

```bash
git clone https://github.com/IshaanM05/ledger-htr-vt.git
cd ledger-htr-vt
```
That's it — no second clone needed (see status note above).

## 2. Get the data

`data/` is gitignored (433MB, mostly images — not meant to live in git).
It's already been uploaded to a **public Google Drive folder** for this
exact purpose:

```
https://drive.google.com/drive/folders/1m_Xxg_tbmxBeBCUTxPa_5XugEMIE_IYr?usp=sharing
```

That folder ("ZindiOCRDataset") contains exactly four items — pull all four,
nothing else is needed:
- `images/` — subfolder, 5472 JPEGs, one `<ID>.jpg` per row across train+test
- `Train.csv` — 4098 rows, columns `ID,Target`
- `Test.csv` — 1374 rows, column `ID`
- `SampleSubmission.csv`

Use whatever Drive access this environment already has (the `gdrive-fsspec`
package if it's preinstalled, `gdown` against the public folder/file links,
or a direct browser download in the notebook UI — any method that ends
with real files on disk is fine, there's no required tool here). Place them
at exactly this path before continuing:
```
data/raw/Train.csv
data/raw/Test.csv
data/raw/SampleSubmission.csv
data/raw/images/*.jpg
```

**Verify before moving on** (don't trust the transfer silently succeeded):
```bash
wc -l data/raw/Train.csv data/raw/Test.csv data/raw/SampleSubmission.csv
# expect: 4099 / 1375 / 1375 lines (each count is rows+1 for the header)
find data/raw/images -name '*.jpg' | wc -l
# expect: 5472
```
If any of these don't match, the transfer is incomplete or corrupted —
don't proceed to fold-splitting or training until they do. If the public
link ever stops working (permissions changed, folder moved), ask the user
rather than falling back to a guess — the Zindi-redownload and local-rsync
routes below still work as a backup if needed.

<details>
<summary>Backup options if the Drive link doesn't work</summary>

- **Re-download from Zindi** directly (competition: "R.O.A.D. Barbados
  Historic Handwriting Challenge") if this environment has internet access
  and Zindi credentials/API access.
- **Transfer from the local machine** (`rsync`/`scp` from
  `/home/ishaan/Desktop/ledger-htr-vt/data/raw/`) — ask the user for a
  transfer path/method since this session can't see the local filesystem.
</details>

## 3. Regenerate the fold split

Deterministic from `Train.csv` alone (seed=42, length-stratified 5-fold) —
don't hand-copy `data/processed/folds.csv` from anywhere, just regenerate it
so it's guaranteed consistent with the code in this checkout:

```bash
python -m src.ledger_htr.data.cv_split
```

Sanity check the printed output: 5 folds, ~815-820 rows each (4077 total —
21 organizer-acknowledged corrupted rows are excluded, see
`src/ledger_htr/data/known_issues.py`), per-fold mean char-length within a
point or two of the global mean (~62).

## 4. Environment

Everything needed is already in `requirements.txt`
(`torch`, `torchvision`, `opencv-python`, `pandas`, `numpy`, `Pillow`,
`albumentations`, `scikit-learn` — `timm` is no longer required now that
the model doesn't import the reference repo). Verify
`torch.cuda.is_available()` is `True` after setup, and check the actual
driver's CUDA ceiling before letting any installer freely resolve a `torch`
build — an unpinned resolve silently breaking CUDA has bitten this project
before on a different machine.

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 5. Smoke-test first (under a minute, catches setup mistakes cheaply)

```bash
PYTHONPATH=src python -m ledger_htr.train_htrvt \
  --config configs/htrvt_smoketest.yaml \
  --max-train-samples 40 --max-val-samples 20
```
Expect it to run all 20 iterations and print a `val_final` line without
crashing (the number itself is near-meaningless at this scale — this is
purely a plumbing check: data loads, model forward/backward runs, SAM's
double step works, checkpoint saves). If this fails, fix the setup issue
first (missing data, wrong image paths, CUDA not available) before moving
to the real run. Clean up after: `rm -rf checkpoints/htrvt_smoketest runs/htrvt_smoketest_*`.

## 6. The real run

```bash
PYTHONPATH=src nohup python -m ledger_htr.train_htrvt \
  --config configs/htrvt_fold0_rtx6000.yaml \
  > logs/htrvt_fold0_rtx6000_$(date +%Y%m%d-%H%M%S).log 2>&1 &
```

`configs/htrvt_fold0_rtx6000.yaml` is sized for a 48GB card: batch size 128,
image width 2048 (256 CTC timesteps' worth of margin over this dataset's
up-to-120-character transcriptions), `num_workers: 8` (bump further if the
molab sandbox has more CPU cores free — data loading, not the model, is
the more likely bottleneck at this batch size), `use_bf16: true`. `total_iters: 4000` is a starting
point, not a hard target — the script always keeps the best checkpoint by
`val_final` (this project's verified corpus-level metric — see
`docs/plan.md`'s Metric section for why the generic
`sqrt(length)`-weighted formula elsewhere in that doc is *not* what's
implemented here), so an early plateau just means later iterations aren't
helping, not that anything's broken.

Wall-clock is unverified on this hardware — watch the
`iter N/4000 ... (Xs)` print lines (every 25 iters) for actual throughput
once it starts, and extrapolate from there. The model is small (39M params)
relative to a VLM or TrOCR-large, so this should be a fast run in absolute
terms even with SAM's double forward/backward per step.

## 7. Watch for

- **`loss=inf` or `nan` on individual print lines**: expected occasionally,
  not a bug. CTC loss can be infeasible for a specific sample if the
  network's output sequence length is too short relative to a target with
  many adjacent repeated characters (needs a blank between them). The
  criterion is built with `zero_infinity=True` specifically for this — it's
  handled, not silently wrong.
- **`val_cer`/`val_wer` stuck at exactly 1.0 for a while early on**: normal
  for the first several hundred iterations — CTC models often output
  near-empty predictions until the loss drops enough for real characters to
  emerge. Don't conclude it's broken before ~500-1000 iterations in.
- Checkpoints save to `checkpoints/htrvt_fold0_rtx6000/best/best.pth`
  (contains both `model` and `ema` state dicts — evaluation and "best"
  selection both use the EMA weights).

## 8. Multi-dataset pretraining (per the current plan, not yet built)

The current plan calls for supervised-pretraining this HTR-VT
reimplementation on combined public HTR benchmarks (IAM, RIMES,
READ2016/Bentham, Washington, Saint Gall) before fine-tuning on the archive
crops — this offsets not having an author checkpoint to start from. That
pretraining pipeline doesn't exist yet in this repo; the run above is
fine-tuning from random init in the meantime, which is a real (documented)
risk of landing below what the architecture is capable of, not a mistake —
see `docs/plan.md`'s License audit note on this. Worth building next if
this fold-0 random-init run's numbers look capacity-limited rather than
data-limited.

## 9. Reporting back

This session and the paired molab session are **not** directly connected —
relay results through the user, not by trying to message this session
directly. At minimum, report back:
- Final best `val_cer` / `val_wer` / `val_final` and at which iteration
- The full `metrics.csv` from `runs/htrvt_fold0_rtx6000_<timestamp>/` (small
  text file, easy to paste or attach)
- Anything that deviated from this doc (different data source, config
  tweaks, errors hit and how they were resolved) — this doc will get
  corrected based on what actually happened, per this project's
  keep-plan-and-logs-in-lockstep-with-reality convention.

The trained checkpoint itself (`best.pth`) doesn't need to come back
immediately — it's not needed until ensembling with the rest of this
track's models actually happens. If that becomes relevant before a better
transfer method exists, generating predictions for `Test.csv` on molab and
sending back just the resulting small CSV is far cheaper than moving model
weights (no `predict_htrvt.py` exists yet — would need to be written,
mirroring the TrOCR submission flow).
