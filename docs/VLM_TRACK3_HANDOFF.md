# InternVL3-8B LoRA fine-tune — Track 3 handoff for a new molab/RTX 6000 session

Everything needed to fine-tune InternVL3-8B (this track's second differentiator
model, per `docs/plan.md` — a different VLM lineage from the teammate's
Qwen, for genuine ensemble diversity) on a fresh RTX 6000 session,
independent of Track 2's HTR-VT notebook.

## 0. Why this exists, and what changed since the plan was written

- **Context**: the teammate already fine-tuned Qwen (LoRA) and got a public
  score of **~0.8** — then tried GRPO, beam search, pseudolabelling, and
  upsampling on top of it, and the score stayed flat. That's strong
  evidence 0.8 is close to a real ceiling for "VLM + LoRA + generation
  tricks" on this dataset, not a waypoint to 0.9 with more of the same.
  Realistic target for this track: land a working LoRA fine-tune in
  similar territory (would already be ~2x our current best of 0.44), for
  genuine architectural diversity in the eventual merged ensemble — not a
  solo shot at beating 0.9.
- **Corrected from an earlier plan in this doc's own drafting**: the
  general plan document says 48-96GB "makes a full fine-tune feasible" —
  that's true in aggregate across the *8 GPUs* InternVL's own docs assume
  for full fine-tuning. Redone the actual memory math for a **single**
  GPU: full fine-tune of an 8B model (AdamW, no ZeRO offload — nothing to
  shard across on 1 GPU) needs ~96GB just for weights+gradients+optimizer
  state, leaving ~2GB for activations on a 98GB card. Not viable. **LoRA
  is the correct primary path here** (~18GB total), not full fine-tune —
  use `--use_llm_lora <rank>`, not the `_full.sh` script as originally
  drafted.
- **License**: Apache-2.0, per the model card tag and consistent with the
  underlying Qwen2.5-7B (not Qwen2.5-VL — a different, permissively
  licensed model family) component. No LICENSE file exists in the HF model
  repo itself (checked directly, 404) — a genuine minor gap versus the
  ideal of an actual license file, but the *code* repo
  (`github.com/OpenGVLab/InternVL`) has a real, confirmed MIT LICENSE
  file, and the Apache-2.0 tag is consistent and undisputed across HF —
  a materially different, less risky situation than HTR-VT (zero license
  anywhere) or Qwen2.5-VL (an actual contradicting restrictive LICENSE
  file was found). Treat as sufficiently verified; flag if anything
  surfaces that contradicts it.

## 1. Get the repo and the data

```bash
git clone https://github.com/IshaanM05/ledger-htr-vt.git
git clone --depth 1 https://github.com/OpenGVLab/InternVL.git
```
The second clone is their actual fine-tuning code (MIT-licensed) — used
directly rather than reimplemented, since (unlike HTR-VT's reference repo)
there's no licensing reason to avoid it, and reimplementing their
image-token-injection/label-masking logic from scratch is a real, easy
place to introduce a silent training bug.

Data: same public Drive folder as Track 2, same fast path (find the file's
Drive ID by scraping the folder's HTML for `data-id` near the filename,
then `gdown <id>` directly — do NOT use `gdown --folder`, it's ~500x
slower per file):
```
https://drive.google.com/drive/folders/1m_Xxg_tbmxBeBCUTxPa_5XugEMIE_IYr?usp=sharing
```
Pull `Train.csv`, `Test.csv`, `SampleSubmission.csv`, and `images.zip`
(unzip into `data/raw/images/`). Verify counts exactly as Track 2's doc
specifies (4098/1374/1374/5472) before continuing.

Regenerate the fold split:
```bash
cd ledger-htr-vt && python3 -m ledger_htr.data.cv_split
```

## 2. Convert our data to InternVL's training format

InternVL's fine-tuning loader expects ShareGPT-style JSONL (confirmed by
reading `internvl_chat/internvl/train/dataset.py` directly, not assumed):
one JSON object per line, `{"id": int, "image": "relative/path.jpg",
"conversations": [{"from": "human", "value": "<image>\n..."},
{"from": "gpt", "value": "..."}]}`, plus a `meta.json` pointing at it.

```bash
cd ledger-htr-vt
python3 -m ledger_htr.data.vlm_finetune_format --val-fold 0
```
Writes `data/vlm_finetune/train.jsonl` (3261 rows, 21 corrupted rows
excluded — same list as every other model in this project,
`known_issues.py`), `data/vlm_finetune/val.jsonl` (816 rows, corrupted
rows *kept* — this is for our own scoring, not training, so it should
reflect the real distribution), and `meta.json`. 4 unit tests cover this
(`tests/test_vlm_finetune_format.py`) — already passing locally.

The prompt used (`TRANSCRIBE_PROMPT` in
`src/ledger_htr/data/vlm_finetune_format.py`): *"Transcribe the
handwritten text in this image exactly as written, preserving original
spelling, punctuation, and abbreviations. Output only the transcription,
nothing else."* No historical/paleographic context added yet — that's a
worthwhile experiment (see the "further ideas" section) but not the
starting point, to keep the first real run a clean baseline.

## 3. Environment — a real risk, read before installing

InternVL's `internvl_chat` code is pinned to `transformers==4.37.2`,
`tokenizers==0.15.1`, `accelerate<1`, `einops==0.6.1`, `numpy==1.26.4`.
A fresh molab sandbox will likely have something much newer (Track 2's
sandbox had `transformers==5.14.1`, `torch==2.11.0+cu130`) — a multi-year
version gap. This is a real compatibility risk, not a formality:

```bash
cd InternVL/internvl_chat
pip3 install -r requirements.txt
```
Do this in whatever sandbox is dedicated to this task (a fresh notebook,
per the plan) rather than one already running something else, since this
will downgrade `transformers` significantly. After installing, verify
immediately, don't assume:
```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "from transformers import AutoModel; print('ok')"
```
If `torch.cuda.is_available()` goes `False` after this install, the
requirements file pulled in a `torch` reinstall that broke CUDA — same
failure mode this project hit once before with `uv pip install` on the
local machine. Pin torch back to what was already working
(`pip3 install torch==<version> --no-deps` won't touch its CUDA build) if
that happens.

`flash-attn` was already present in Track 2's sandbox
(`flash-attn-4==4.0.0b15+sm120patch1`) — try `use_flash_attn=True` first
when loading the model; if it errors against the older transformers/newer
torch combination, fall back to `use_flash_attn=False` (slower, still
correct) rather than fighting a flash-attn version match.

`peft` is needed for LoRA and is **not** in their requirements.txt as
pinned above (odd, but confirmed by grep) — install it explicitly:
```bash
pip3 install peft==0.10.0
```

## 4. Smoke test before the real run

Before launching a real LoRA fine-tune, verify the model loads and can
run a single forward/backward step. This project's established pattern
(every prior track) is smoke-test first, real run second — don't skip it
here just because the setup is more involved:

```bash
python3 -c "
import torch
from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained('OpenGVLab/InternVL3-8B', trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained('OpenGVLab/InternVL3-8B', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).eval().cuda()
print('loaded ok, params:', sum(p.numel() for p in model.parameters()))
"
```
This alone downloads ~16GB of weights and confirms the trust_remote_code
path actually works against this sandbox's transformers version — a real
failure point given the version gap above, worth isolating before
involving the training script, LoRA wrapping, and our data all at once.

Then run their actual fine-tuning script against a tiny slice (create a
`data/vlm_finetune/train_smoke.jsonl` with ~20 lines via
`head -20 data/vlm_finetune/train.jsonl`, point a smoke meta.json at it,
run for a handful of steps) before committing to the full run.

## 5. The real run — LoRA, single GPU

No `internvl3.0/2nd_finetune/*_8b*lora*.sh` script exists for this
specific version (only `_full.sh` — checked directly, only older
versions like 2.0/2.5 have a matching `_lora.sh`). Adapt the full script's
flags rather than searching for a LoRA script that isn't there:

```bash
cd InternVL/internvl_chat
GPUS=1 PER_DEVICE_BATCH_SIZE=4 torchrun \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 --master_port=34229 \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "OpenGVLab/InternVL3-8B" \
  --conv_style "internvl2_5" \
  --use_fast_tokenizer False \
  --output_dir work_dirs/internvl3_8b_lora_fold0 \
  --meta_path "/absolute/path/to/ledger-htr-vt/data/vlm_finetune/meta.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 2 \
  --learning_rate 2e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee work_dirs/internvl3_8b_lora_fold0/training_log.txt
```

Deliberate changes from their `_full.sh` template: `--use_llm_lora 16`
(the actual LoRA switch — rank 16 is a reasonable starting point, not
tuned yet), `--freeze_backbone True` kept (freeze the vision tower, LoRA
only the language model — standard for this kind of fine-tune, matches
what "LoRA" usually means in this ecosystem), no `--deepspeed` flag
(single GPU has nothing to shard — dropping it avoids an extra dependency
and a config file that assumes multi-GPU), `--max_seq_length 4096` instead
of their `16384` (our transcriptions are short — mean 62 characters — no
reason to pay for a 16k context budget), `learning_rate 2e-4` (typical
LoRA rate, an order of magnitude higher than full-fine-tune's `2e-5`),
`num_train_epochs 3` as a starting point (not tuned — this project's
established pattern is to treat first-run hyperparameters as a baseline
to correct from actual `runs/`/log evidence, not a final answer).

`PER_DEVICE_BATCH_SIZE=4` with `gradient_accumulation_steps=8` gives an
effective batch of 32 — a guess, not measured against this sandbox's
actual memory headroom. Watch `nvidia-smi` after the first few steps; if
it's nowhere near the 98GB ceiling, this can go up; if it OOMs, drop batch
size before dropping `max_dynamic_patch` (fewer tiles per image would
lose real resolution on faint/small handwriting, a worse trade).

## 6. Evaluate against our verified metric

Their training script does not evaluate during training
(`--evaluation_strategy "no"`) — same as this project's standing practice
of never trusting a third-party pipeline's internal eval, always scoring
with our own verified corpus-level formula. After training (or against
any saved intermediate checkpoint in `work_dirs/.../checkpoint-N/`):

```bash
PYTHONPATH=src python3 -m ledger_htr.predict_vlm \
  --model-path work_dirs/internvl3_8b_lora_fold0/checkpoint-<N> \
  --mode val --val-fold 0 --num-beams 1
```
Prints `val_cer`/`val_wer`/`val_final` directly. Try `--num-beams 4` or
`5` once a checkpoint looks decent — beam search is a cheap addition on
top of a trained model, worth comparing against greedy (`num_beams=1`)
rather than assuming it helps by default.

`src/ledger_htr/predict_vlm.py`'s image preprocessing
(`build_transform`/`dynamic_preprocess`/`load_image`) is copied verbatim
from InternVL3-8B's own model card — not re-derived — specifically
because a subtly-wrong preprocessing function would silently make a
correctly-trained model look bad at eval time, which would be a
confusing, hard-to-diagnose failure mode.

## 7. Generate a submission

Once a checkpoint's `val_final` looks competitive (this project's bar:
CV-based decision, not leaderboard-chasing — see `docs/plan.md`'s
Pitfalls section):
```bash
PYTHONPATH=src python3 -m ledger_htr.predict_vlm \
  --model-path work_dirs/internvl3_8b_lora_fold0/checkpoint-<N> \
  --mode test --num-beams 4 --out-csv submissions/internvl3_lora_submission.csv
```
Run the completeness check against `SampleSubmission.csv` before actually
submitting (every ID present, no missing/empty predictions — the rules
page scores those as flat-out wrong).

## 8. Further ideas, not yet built (from the ideation pass that led here)

Roughly in order of expected value, given the teammate's plateau at 0.8
despite GRPO/beam-search/pseudolabelling/upsampling — see the reasoning
in this project's conversation log, not repeated in full here:
- **Ask what reward function the teammate's GRPO run actually used.** If
  it wasn't the exact corpus-level CER/WER formula this project verified
  against the real leaderboard (`ledger_htr/metrics/scorer.py`), that
  mismatch alone could explain a flat GRPO result — worth knowing before
  concluding GRPO itself is a dead end here.
- **Domain-aware prompting** — add historical/paleographic context to
  `TRANSCRIBE_PROMPT` (period, document type, common 18th-19th century
  abbreviation conventions) rather than a generic "transcribe this"
  instruction, to try to activate whatever period-English knowledge the
  base model's pretraining already contains.
- **A broader corrupted-label audit** beyond the 21 organizer-acknowledged
  rows — the teammate independently found label-quality issues; our
  current exclusion list came from one Zindi chat thread, not a
  systematic pass (e.g. flagging rows where multiple independently-trained
  models agree with each other but disagree sharply with the label).
- **PARSeq** (Apache-2.0, real pretrained checkpoint, fast) — a third
  architecturally distinct model (permutation-autoregressive, unlike
  HTR-VT's CTC or this VLM's causal LM decoding) for the eventual
  ensemble, cheap enough to build in parallel without competing for this
  track's GPU time.
- **Ensembling across the teammate's Qwen, this InternVL3, and HTR-VT**
  via MBR/ROVER (`docs/plan.md`'s Distillation section already specs the
  method) — the plan's actual thesis for getting past a single model's
  ceiling, worth building once 2-3 real model outputs exist to combine,
  not before.

## 9. Reporting back

Same convention as Track 2: this session and the paired molab session
aren't directly connected unless re-paired via the same `marimo-pair`
flow with a fresh URL/token — relay results through the user. Report:
`val_cer`/`val_wer`/`val_final` at whichever checkpoint scored best,
the actual `PER_DEVICE_BATCH_SIZE`/memory headroom found empirically,
anything in section 3's environment setup that didn't work as described
(this section will get corrected based on what actually happened, same
as every other doc in this project), and whether `use_flash_attn=True`
worked or needed the fallback.
