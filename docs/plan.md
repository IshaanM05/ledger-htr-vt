# Reading the Ledger — R.O.A.D. Barbados Historic Handwriting Challenge

A day-by-day plan from September 3rd to close, pairing what to build with what to actually read first — pretrained transfer, targeted augmentation, and ensembling, in that order of leverage.

## Snapshot

| | |
|---|---|
| **Starts** | 03 Sep 2026 |
| **Closes** | 05 Oct 2026 |
| **Runway** | 33 days |
| **Data** | 6,000 line/word crops, 18th–19th century Barbadian legal handwriting (deeds, wills, estate inventories) |
| **Compute** | RTX 4080 Laptop, 12GB GDDR6 (not the desktop card's 16GB) |
| **Metric** | 0.5 × weighted WER + 0.5 × weighted CER |

---

## The core bet

This is **not** a full-page layout problem — the images are already cropped to individual words or short lines, so segmentation is a non-issue. That reframes it as a small-data transfer-learning problem: ~6,000 crops of 18th–19th century Barbadian legal handwriting is nowhere near enough to train a recognizer from scratch.

Every point of leaderboard movement will come from:

1. **How well you transfer** a model already fluent in handwriting.
2. **How well you augment** the faded ink and irregular penmanship this specific archive has.
3. **How well you combine** a few complementary models rather than betting everything on one architecture.

---

## The differentiator stack

What most entrants won't try, pulled from papers published in the last 12–18 months and chosen because they actually fit a 6,000-sample archive and one 12GB laptop GPU, not because they're trendy.

**HTR-VT as the anchor model.** An encoder-only ViT with a CNN stem, CTC head, SAM optimizer, and span-mask regularization (Li et al., 2024). Its entire premise is matching CNN-RNN-CTC and TrOCR-scale models *without* needing large external pretraining — a near-perfect fit for this dataset's size, and a build most competitors reaching straight for a Hugging Face checkpoint won't attempt.

**Domain self-supervised pretraining.** Masked-image-modeling on all 6,000 provided crops — train *and* test pixels, no labels touched — before supervised fine-tuning. Warms the encoder to this exact paper, ink, and penmanship before it ever sees a transcription.

**Style-conditioned diffusion synthesis.** One-DM generates realistic handwriting from a single style exemplar via a diffusion model conditioned on high-frequency stroke detail. Condition it on your own training crops to synthesize more examples of the rare letterforms and ligatures early EDA turns up, instead of generic font-rendered synthetic text.

**LLM post-OCR correction — done carefully.** 2025 research on exactly this setup (English, historical documents) found real gains from open LLMs, but only with alignment-based post-processing to strip hallucinated additions, and only once you know whether your ground truth normalizes archaic spelling. Settle that in Phase 1 before betting on this.

**Distillation: ensemble → one lean model.** HTR-JAND (2024) reports a 62% relative CER cut from temperature-scaled soft-label distillation in its ablation. Distill your Phase 4 ensemble into a single compact HTR-VT student — most of the ensemble's accuracy, in one fast, easily-reproducible model. Full recipe below.

---

## Learning curriculum

Everything referenced anywhere in this plan, grouped by theme rather than dumped as one list. The day-by-day schedule links back into this — read a day or two before you need to implement, not after.

### Sequence recognition fundamentals

- Graves et al., *Connectionist Temporal Classification* (2006) — the original CTC paper; skim it for the forward-backward algorithm, since that's exactly what HTR-VT's and PyLaia's loss computes. *(ICML 2006 — searchable by title)*
- Awni Hannun, *Sequence Modeling with CTC* (Distill, 2017) — the visual explainer; read this **before** the original paper, not after. `distill.pub/2017/ctc`
- Jay Alammar, *The Illustrated Transformer* — the attention/decoder intuition behind TrOCR's decoder and every VLM's language backbone here. `jalammar.github.io/illustrated-transformer`
- Vaswani et al., *Attention Is All You Need* (2017) — read once Illustrated Transformer makes sense; the actual architecture TrOCR's decoder and the VLMs' backbones are built from. `arxiv.org/abs/1706.03762`

### Architectures used in this plan

- Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, 2020) — the encoder backbone behind TrOCR, HTR-VT, and every VLM here; worth a skim even with a CV background, for the patch-embedding-vs-CNN-stem distinction HTR-VT changes. `arxiv.org/abs/2010.11929`
- Li et al., *HTR-VT* (Pattern Recognition, 2024) — this plan's anchor model, in full. `arxiv.org/abs/2409.08573` · `github.com/YutingLi0606/HTR-VT`
- Foret et al., *Sharpness-Aware Minimization* (SAM, 2020) — the optimizer HTR-VT relies on; understand what "flat minima" buys you before tuning its rho. `arxiv.org/abs/2010.01412`
- He et al., *Masked Autoencoders Are Scalable Vision Learners* (MAE, 2021) — the masked-image-modeling recipe behind the self-supervised pretraining step. `arxiv.org/abs/2111.06377`
- Microsoft's *TrOCR* model card and Hugging Face fine-tuning walkthrough — the ensemble's second member. `huggingface.co/microsoft/trocr-large-handwritten`
- PyLaia — the CTC recognizer behind most Transkribus/READ pipelines, for the optional third sanity-check model. `github.com/jpuigcerver/PyLaia`
- Garrido-Muñoz et al., *On the Generalization of Handwritten Text Recognition Models* (CVPR 2025) — background on why HTR models trained on one archive transfer unevenly to another; directly relevant to how much to trust IAM-pretrained weights here. *(openaccess.thecvf.com/content/CVPR2025 — search the title)*

### Generative augmentation & diffusion

- Lilian Weng, *What are Diffusion Models?* — read before touching One-DM's code; the clearest single explainer of the forward/reverse process. `lilianweng.github.io/posts/2021-07-11-diffusion-models`
- Ho, Jain & Abbeel, *Denoising Diffusion Probabilistic Models* (2020) — the original formulation. `arxiv.org/abs/2006.11239`
- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models* (2022) — One-DM operates in this latent space, not pixel space; skim for why that matters for compute. `arxiv.org/abs/2112.10752`
- Dai et al., *One-DM* (ECCV 2024) — style-conditioned handwriting generation, released checkpoints. `arxiv.org/abs/2409.04004` · `github.com/dailenson/One-DM`
- WordStylist — a lighter latent-diffusion alternative if One-DM's footprint proves unfriendly to a single GPU. `github.com/koninik/WordStylist`
- TextRecognitionDataGenerator — for the baseline font-rendered synthetic set, before reaching for diffusion. `github.com/Belval/TextRecognitionDataGenerator`
- *Handwritten Text Recognition of Historical Manuscripts Using Transformer-Based Models* (2025) — the augmentation/ensembling case study the Phase 3 numbers are drawn from. `arxiv.org/abs/2508.11499`

### Efficient fine-tuning — the VLM on 12GB

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021). `arxiv.org/abs/2106.09685`
- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs* (2023) — specifically what makes a 3–4B VLM fit in 12GB. `arxiv.org/abs/2305.14314`
- Sebastian Raschka, *Practical Tips for Finetuning LLMs Using LoRA* — the most concrete, least hand-wavy walkthrough of rank/alpha/target-module choices. `magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms`
- Hugging Face PEFT docs — the library you'll actually call. `huggingface.co/docs/peft`
- *Finetuning Vision-Language Models as OCR Systems for Low-Resource Scripts* (Manchu case study, 2025) — the VLM-vs-CRNN generalization comparison this plan's VLM step is betting on. `arxiv.org/abs/2507.06761`

### Knowledge distillation

- Hinton, Vinyals & Dean, *Distilling the Knowledge in a Neural Network* (2015) — the temperature-scaled softmax trick everything else here builds on; short, and worth reading in full. `arxiv.org/abs/1503.02531`
- HTR-JAND (2024) — the teacher/student loss recipe behind the Distillation section. `arxiv.org/abs/2412.18524`
- LightOnOCR-1B — a live example of distilling a large VLM's OCR ability into a compact model via filtered pseudo-transcriptions. `huggingface.co/blog/lightonai/lightonocr`

### Post-processing & language modeling

- pyctcdecode — beam search + KenLM rescoring for the CTC models' output, ready to use rather than hand-rolled. `github.com/kensho-technologies/pyctcdecode`
- KenLM — for building the n-gram language model over your period-vocabulary lexicon. `github.com/kpu/kenlm`
- *OCR Error Post-Correction with LLMs in Historical Documents: No Free Lunches* (2025) — read before building the correction pass; the normalization pitfall alone is worth it. `arxiv.org/abs/2502.01205`

### The domain itself — reading 18th–19th century English handwriting

- The National Archives (UK), *Palaeography: reading old handwriting 1500–1800* — a genuinely practical, free tutorial with real manuscript exercises. Worth doing yourself early — you'll be judging model errors against your own reading of the same letterforms. `nationalarchives.gov.uk/palaeography`
- Folgerpedia's list of early modern English paleography resources — for specific letterforms (long s, secretary hand) if the archive leans that way. `folgerpedia.folger.edu/List_of_online_resources_for_early_modern_English_paleography`
- Period vocabulary for the lexicon/synthetic-text steps: Project Gutenberg's public-domain 18th–19th century English texts, the Oxford Text Archive, and The National Archives' guide to historical wills for probate-specific vocabulary. No single ready-made corpus covers this — expect to assemble it from a few sources, which is itself useful signal for what the lexicon needs to cover. `gutenberg.org` · `ota.bodleian.ox.ac.uk` · `nationalarchives.gov.uk/help-with-your-research/research-guides/wills-1384-1858`

### Metric & tooling reference

- Zindi's guide to weighted WER/CER — read before writing your local scorer. `zindi.world/learn/evaluating-language-generation-on-zindi-a-guide-to-weighted-wer-and-cer`
- jiwer — a quick unweighted WER/CER sanity check before your weighted implementation is trusted. `github.com/jitsi/jiwer`
- Albumentations docs — for the elastic/affine/perspective augmentation pipeline. `albumentations.ai/docs`

---

## Day-by-day, Sep 3 → Oct 5

Most days carry one **Learn** line and one **Build** line. Weight the split however your schedule actually allows — the order across days matters more than the hour count on any one of them.

### Phase 1 — Foundations + baseline (Sep 3 – Sep 6, 4 days)

> Understand the data and the metric cold, and get a mediocre model on the leaderboard by day 4. A bad submission on day 1 beats a great one on day 20 — it tells you the pipeline works.

**Thu · Sep 3**
- [ ] **Learn** — CTC fundamentals: Graves' original paper plus Hannun's Distill explainer; skim the Zindi weighted-metric guide too
- [ ] **Build** — join, download `Train.csv`, `Test.csv`, `images.zip`, `Starters.zip`; read the starter notebook end to end; first EDA pass (image sizes, crop types, transcription-length histogram, character set)

**Fri · Sep 4**
- [ ] **Learn** — The Illustrated Transformer, enough attention background to read TrOCR's architecture doc
- [ ] **Build** — resolve whether ground truth normalizes archaic spelling or preserves it verbatim; build a stratified 5-fold local CV split (stratify on transcription length)

**Sat · Sep 5**
- [ ] **Learn** — the National Archives palaeography tutorial; work through a few real exercises yourself, not just read about it
- [ ] **Build** — implement and unit-test the exact weighted WER/CER scorer (formula below) against a couple of hand-computed examples

**Sun · Sep 6**
- [ ] **Learn** — skim the TrOCR paper itself before fine-tuning it
- [ ] **Build** — fine-tune `microsoft/trocr-base-handwritten` near-default, submit, confirm local CV and public LB move together

**Checkpoint —** a working end-to-end pipeline and a real leaderboard position, however low.

### Phase 2 — Two model families, led by the unusual one (Sep 7 – Sep 13)

> HTR-VT and TrOCR fail differently on small data and come from different lineages — CTC vs. seq2seq. Build both, so later ensembling has something real to combine, and so you've implemented a paper instead of just calling `.from_pretrained()`.

**Mon · Sep 7**
- [ ] **Learn** — ViT paper, focused on the patch-embedding step HTR-VT replaces
- [ ] **Build** — preprocessing pass: binarize/normalize contrast, deskew, pad to consistent height, matched to what your pretrained backbones actually saw

**Tue · Sep 8**
- [ ] **Learn** — SAM optimizer paper — what "flat minima" buys you before you tune its rho
- [ ] **Build** — clone [HTR-VT](https://github.com/YutingLi0606/HTR-VT), get the authors' IAM config running end to end on a toy subset first

**Wed · Sep 9**
- [ ] **Learn** — re-read HTR-VT's span-masking section closely — it's the part you're about to adapt
- [ ] **Build** — adapt patch size and sequence length to this dataset's crop dimensions; get a first real training run going

**Thu · Sep 10**
- [ ] **Build** — debug and tune HTR-VT training: loss curves, CTC decoding sanity checks, confirm span masking and SAM are actually engaged

**Fri · Sep 11**
- [ ] **Build** — push TrOCR properly: `trocr-large-handwritten`, tune LR/warmup, freeze-then-unfreeze the encoder, label smoothing

**Sat · Sep 12**
- [ ] **Build** — log and compare per-sample errors for HTR-VT vs. TrOCR side by side — this is where you actually learn what the archive looks like

**Sun · Sep 13**
- [ ] **Build** — submit best single model of the week; write up error-pattern notes to target in Phase 3

**Checkpoint —** HTR-VT and TrOCR both trained on identical folds, with a shared local eval script — and you can explain why HTR-VT's optimizer and masking choices exist.

### Phase 3 — Make the data behave like the archive (Sep 14 – Sep 20)

> With this little real data, pretraining and augmentation are where most of the score is actually won. A published result on a comparably small historical-manuscript set (~4k lines) took CER from 1.93 to 1.60 through augmentation and ensembling alone — treat that as the floor, not the ceiling.

**Mon · Sep 14**
- [ ] **Learn** — MAE paper — the masked-image-modeling recipe you're about to run
- [ ] **Build** — implement masked-image-modeling self-supervised warm-up on all 6,000 provided crops (train + test pixels, no labels)

**Tue · Sep 15**
- [ ] **Build** — finish the SSL pretraining run; load those weights as HTR-VT's encoder init; confirm it trains stably from there

**Wed · Sep 16**
- [ ] **Learn** — Lilian Weng's diffusion-models explainer, ahead of Saturday's One-DM work
- [ ] **Build** — add elastic distortion as the primary augmentation (highest single-technique gain in the reference study)

**Thu · Sep 17**
- [ ] **Build** — layer in random affine, perspective warp, contrast/brightness jitter, and cutout patches, on-the-fly at ~50% per sample

**Fri · Sep 18**
- [ ] **Build** — baseline synthetic set: period vocabulary rendered in historical-style cursive fonts with TextRecognitionDataGenerator

**Sat · Sep 19**
- [ ] **Learn** — re-read One-DM's method section with implementation in mind
- [ ] **Build** — run One-DM's released checkpoint in inference mode; generate style-conditioned samples targeting the rare letterforms Phase 2's error analysis flagged

**Sun · Sep 20**
- [ ] **Build** — re-run HTR-VT + TrOCR on SSL-pretrained + augmented + synthetic data; compare fold-by-fold against Phase 2; submit

**Checkpoint —** a measurable CV improvement attributable to pretraining and augmentation, not just more training time.

### Phase 4 — A third opinion, then combine (Sep 21 – Sep 27)

> This is the differentiation week — most entrants will stop at a single fine-tuned TrOCR. A VLM fine-tune, careful LLM correction, a proper ensemble, and distilling that ensemble back down are what separate a good score from a winning one.

**Mon · Sep 21**
- [ ] **Learn** — the LoRA paper and Raschka's practical-tips post on rank/alpha/target-module choices
- [ ] **Build** — set up QLoRA fine-tuning for Qwen2.5-VL-3B or InternVL2.5-4B

**Tue · Sep 22**
- [ ] **Learn** — the QLoRA paper — specifically why 4-bit + LoRA is what makes this fit in 12GB
- [ ] **Build** — run and tune the VLM fine-tune: rank, target modules, batch size for 12GB

**Wed · Sep 23**
- [ ] **Build** — build the lexicon/LM rescoring pass: period-vocabulary word list, plus beam-search rescoring or edit-distance snapping to the nearest known word

**Thu · Sep 24**
- [ ] **Learn** — the "No Free Lunches" LLM post-correction paper, in full
- [ ] **Build** — implement LLM post-OCR correction with alignment-based hallucination stripping, gated on Sep 4's normalization finding

**Fri · Sep 25**
- [ ] **Build** — test-time augmentation at inference; ensemble via prediction voting across HTR-VT / TrOCR / VLM (beam-hypothesis voting, not character-level)

**Sat · Sep 26**
- [ ] **Learn** — Hinton's distillation paper in full, then re-read HTR-JAND's loss section
- [ ] **Build** — distill the ensemble into a single HTR-VT student (temperature-scaled KD + CTC loss); run the VLM→HTR-VT pseudo-label distillation pass

**Sun · Sep 27**
- [ ] **Build** — error-analysis loop on the worst 50 predictions; submit both the ensemble and the distilled student, compare on CV

**Checkpoint —** an ensemble that beats every one of its members on CV, and a distilled single model that recovers most of that gain without the multi-model overhead.

### Phase 5 — Lock it down (Sep 28 – Oct 5, 8 days)

> Reproducibility and calm are worth more than a last-minute architecture change. Top 10 gets a 48-hour code-review window — be ready before you need to be.

**Mon · Sep 28**
- [ ] **Build** — freeze every seed (data split, model init, augmentation); start a clean re-run to confirm your score reproduces

**Tue · Sep 29**
- [ ] **Build** — finish the reproducibility re-run; fix anything nondeterministic (dataloader workers, cudnn benchmark flags)

**Wed · Sep 30**
- [ ] **Build** — start writing up the solution: approach, what worked, what didn't — not optional if you place well

**Thu · Oct 1**
- [ ] **Build** — finish documentation; clean the code into a code-review-ready state

**Fri · Oct 2**
- [ ] **Build** — pick your 2 private-leaderboard submissions deliberately (distilled model + best ensemble); run the required-ID completeness check against `SampleSubmission.csv`

**Sat · Oct 3**
- [ ] **Build** — buffer day: catch up on anything that slipped; re-verify both final submissions still reproduce

**Sun · Oct 4**
- [ ] **Build** — final sanity pass; get submissions in with real margin — don't burn your last daily cap of 5 on the deadline day itself

**Mon · Oct 5 — close**
- [ ] Nothing new starts today. Submissions lock and the private leaderboard reveals the same day.

**Checkpoint —** a reproducible, documented, deliberately-chosen final pair.

---

## Model options on your hardware

All fit on the laptop 4080's 12GB for line/word-level crops — that card is 12GB GDDR6 at 175W, not the desktop 4080's 16GB, so the VLM row is sized accordingly.

| Family | Why it's here | Watch out for | Role |
|---|---|---|---|
| **HTR-VT** (Li et al., 2024) | CNN stem + ViT encoder + CTC head, built specifically to compete without large external pretraining — validated on the historical READ2016 set | You're implementing from a paper/repo, not a one-line `.from_pretrained()` — budget real debugging time | Anchor / differentiator |
| **TrOCR** (large-handwritten) | Vision transformer + text decoder, pretrained on IAM — strong out-of-the-box handwriting prior, easy HF fine-tuning, architecturally distinct from HTR-VT | Seq2seq decoder can repeat/hallucinate words with too little fine-tuning data; needs matched preprocessing | Ensemble diversity |
| **CRNN + CTC** (PyLaia-style, optional) | The classic Transkribus/READ-pipeline workhorse — quick to train, a useful sanity check | Largely superseded here by HTR-VT once that's working; keep only for a third, cheap CV opinion | Optional sanity check |
| **VLM + QLoRA** (Qwen2.5-VL-3B / InternVL2.5-4B) | General visual-text pretraining transfers surprisingly well to messy real handwriting, per recent low-resource-script results | 12GB fits a 3–4B model with 4-bit + LoRA + gradient checkpointing; a 7B model leaves little headroom for batch size — stretch goal, not default | Second differentiator |

---

## Distillation, concretely

Two teacher→student pairs worth running, both grounded in papers from the last year rather than a vague "try distillation" — and the best answer to the reproducibility rules, since one distilled model is far easier to verify than a live multi-model vote.

**Ensemble → single HTR-VT student.** Run the trained ensemble (HTR-VT + TrOCR + VLM) as teachers over the training set, then fine-tune a fresh HTR-VT student against their temperature-softened outputs plus the real labels. HTR-JAND's ablation found this cut CER by 62% relative to an undistilled baseline — treat that as evidence the mechanism works, not as a number that transfers exactly to this dataset.

**VLM → HTR-VT, specifically.** The recipe behind LightOnOCR-1B, scaled down: the teacher (the fine-tuned VLM) generates pseudo-transcriptions for the test set, hallucinations get filtered by checking agreement against the CTC/TrOCR outputs, and the student trains on real labels plus the filtered pseudo-labels — folding the VLM's stronger real-world generalization into a model fast enough to actually run at submission time.

```
L_total = α · L_ctc(student, labels) + γ · L_kd(student, teacher)
L_kd    = KL( softmax(teacher_logits / τ) || softmax(student_logits / τ) ) · τ²
```

Start temperature τ around 2–4 and keep γ small early — soft targets are mostly noise before the student has learned the basics — then ramp it up as training stabilizes. That staged weighting, not a fixed loss mix, is what HTR-JAND credits for most of the gain.

---

## Metric — replicate it exactly, don't approximate

Both WER and CER here are length-weighted, not simple averages — an off-the-shelf `jiwer.wer()` call will silently track the wrong thing.

```
weight_i   = sqrt(reference_length_i)          # words for WER, chars for CER
edits_i    = levenshtein(prediction_i, reference_i)
score      = sum(edits_i * weight_i) / sum(weight_i * reference_length_i)
final_score = 0.5 * weighted_WER + 0.5 * weighted_CER
```

Longer transcriptions are weighted more heavily (via the square root of their length), so getting a handful of long multi-word crops right matters more than nailing every short single-word one. Build this scorer in Phase 1 and use it — not accuracy, not raw edit distance — as the only number you trust for model selection.

---

## Where the score actually comes from

**High leverage:** pretrained transfer plus a genuinely different architecture (HTR-VT) · domain self-supervised pretraining on all 6,000 images · elastic/affine augmentation matched to ink and paper degradation · style-conditioned diffusion synthesis for rare letterforms · lexicon-constrained decoding · ensembling 2–3 genuinely different architectures.

**Low leverage:** hyperparameter micro-tuning a single model past the first good setting · training a diffusion generator from scratch on one laptop GPU instead of using a released checkpoint · chasing the public leaderboard (only ~30% of the test set) instead of your CV.

---

## Pitfalls specific to this challenge

- **Small LB** — the public leaderboard is a small, noisy slice; a submission that looks worse there can be genuinely better. Select your final two submissions from 5-fold CV, not leaderboard rank.
- **Empty preds** — missing or empty predictions are scored as flat-out wrong; always run a completeness check against `SampleSubmission.csv` before submitting.
- **Reproducibility** — top 10 gets a code request with a 48-hour window; a result that doesn't reproduce gets your rank adjusted down to what the code actually produces. Seed everything now, not in Phase 5.
- **Open-source only** — no AutoML tools, no closed APIs; every pretrained checkpoint and package needs to be genuinely open and available to any other entrant.

---

## Teaming

Teams run up to 4 and can merge later, but a team formed after submissions have started can't be disbanded and members can't leave once a submission is in — so if you bring someone on, do it deliberately (someone strong on the language-model/lexicon side pairs well with a CV background) rather than defaulting into it. Solo is a completely reasonable choice here too, especially for the learning goal.

---

*Plan begins 03 Sep 2026 · solo/team entry into the R.O.A.D. Barbados Historic Handwriting Challenge.*
