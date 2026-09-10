# HTR-VT self-supervised pretraining + fold-0 fine-tune — molab RTX 6000, 2026-09-10

Direct follow-up to `docs/HTRVT_FOLD0_RUN_REPORT.md` (the random-init fold-0
run, `val_final=0.1207`). This time: masked-image-modeling pretraining on
all 5472 crops first, then fine-tune fold 0 from that encoder instead of
random init, to test whether it fixes the earlier run's overfitting-past-
iter-1700 pattern and closes the gap to TrOCR.

## TL;DR

- **Pretraining**: `val_recon_loss=0.00808` best (iter 3000/3000, still
  slowly improving at the end — no overfitting signal).
- **Fine-tune, fold 0, from the pretrained encoder**: **`val_cer=0.3342`,
  `val_wer=0.7880`, `val_final=0.4389`** at the final iteration (2000/2000)
  — **still climbing when the run ended**, unlike the random-init run.
- **This beats every prior local CV number in the project**, TrOCR
  included:

| Model | val_final | Notes |
|---|---:|---|
| HTR-VT, random-init (2026-09-08) | 0.1207 | peaked iter 1700/4000, degraded after |
| **HTR-VT, SSL-pretrained (this run)** | **0.4389** | still improving at final iteration |
| TrOCR baseline, corrected folds.csv | 0.4002 | teammate's track now, not being pushed further |
| TrOCR + elastic augmentation | 0.4037 | |

- Convergence was also much faster in wall-clock/iteration terms: this run
  reached `val_cer=0.83` by iter 800, versus iter ~1100-1200 for the
  random-init run to reach a comparable point.
- Total wall-clock: pretraining ~13 min (3000 iters), fine-tune ~30 min
  (2000 iters) — both on the RTX 6000.

## Why this run happened (context for future reference)

The original plan called for multi-dataset *supervised* pretraining
(IAM/RIMES/READ2016/Washington/Saint Gall) to offset HTR-VT's missing
author checkpoint. Verified directly against the live Zindi rules page
that this is disallowed ("You may use only the datasets provided for this
challenge") — training on external raw data, not just a licensing
nuance. Masked-image-modeling self-supervised pretraining on the
competition's *own* 6000 crops (train+test pixels, no labels) is the
compliant substitute already in the plan as a separate item — that's what
this run actually does. See `docs/HTRVT_TRACK2_HANDOFF.md`'s "Next up"
section (now superseded by this result) and the commit that added
`src/ledger_htr/pretrain_htrvt.py`.

## What actually ran

Connected directly to a fresh molab notebook via the `marimo-pair` skill
(no separate Claude Code terminal needed as a go-between this time) --
cloned the repo, pulled data from the shared Drive folder's `images.zip`
(found the file's Drive ID by scraping the folder's HTML page for
`data-id` attributes near the filename, then `gdown <id>` directly — ~11s
for 442MB, much faster than the per-file `gdown --folder` approach from
the first run), verified all counts, regenerated the fold split, ran both
smoke tests, then launched for real.

**Autopilot handoff, and one real gap it exposed.** A watcher script
(`watch_and_finetune.sh`) was launched detached inside the molab sandbox
itself (`subprocess.Popen(..., start_new_session=True)`, not tied to this
session or the user's laptop) to poll the pretraining PID and `exec` into
the fine-tune command the moment it exited — done specifically because the
user was about to close their laptop. When checked immediately after
reconnecting, `ps aux` showed *no* related processes running at all and
the fine-tune log didn't exist yet, which looked like the watcher had died
mid-poll (plausible cause: the molab sandbox itself pausing during
extended idle time, not just the individual process). Re-checked moments
later and the fine-tune was in fact running — `exec` replaces a shell
process's image while keeping its PID, so the watcher's PID (4430) showing
up as the `train_htrvt` process in `ps aux` was actually the expected,
correct outcome, not a leftover stale process. The apparent failure was
just a race in the first check, not an actual autopilot failure. Worth
remembering when it happens again: `exec` in a watcher script means
"process not found under its original name" is not evidence the handoff
failed.

Note on background-launch mechanics (also now in memory): plain
`nohup cmd &` submitted via a `bash -c` string through the marimo
scratchpad reliably failed (exit code 1, no output at all) — switching to
`subprocess.Popen([...], stdout=<real file handle>, start_new_session=True)`
from directly within the scratchpad's Python is what actually worked, both
for the training process itself and the watcher script.

## Outputs

| artifact | location | retrieved locally? |
|---|---|---|
| Pretraining `metrics.csv` | `runs/htrvt_pretrain_molab/metrics.csv` | ✅ |
| Pretraining best checkpoint | `checkpoints/htrvt_pretrain/best/best.pth` (molab only) | not yet — not needed locally, same reasoning as the fold-0 checkpoint below |
| Fine-tune `metrics.csv` + `resolved_config.yaml` | `runs/htrvt_fold0_rtx6000_pretrained_molab/` | ✅ |
| Fine-tune best checkpoint | `checkpoints/htrvt_fold0_rtx6000_pretrained/best/best.pth` (molab only) | not yet |

## Decisions for you

1. **Extend `total_iters`?** The fine-tune was still improving at iteration
   2000/2000 (`val_final` climbing every eval step: 0.32→0.38→0.42→0.43→0.44)
   — `total_iters=2000` was carried over from the random-init run's
   overfitting finding, which doesn't apply here. A longer run (or a
   second phase starting from this checkpoint) would likely score higher
   still; there's no evidence yet of where it plateaus.
2. **Run folds 1-4?** This is now a real, competitive result (beats TrOCR
   locally) rather than a "confirm it trains" sanity check — the case for
   spending the full 5-fold budget per `docs/plan.md`'s policy is much
   stronger than it was after the random-init run.
3. **Generate a submission?** Unlike the random-init run, this one is
   plausibly worth a submission slot — it's ahead of what's currently on
   the leaderboard from this track. No `predict_htrvt.py` exists yet to
   turn a checkpoint into `Test.csv` predictions; would need to be written
   first.
4. **Retrieve the checkpoints?** Neither `best.pth` has been pulled back
   from molab yet. Not blocking anything above (the molab sandbox is
   reproducible from checked-in configs + seed=42), but needed eventually
   for ensembling with TrOCR or generating a submission.
