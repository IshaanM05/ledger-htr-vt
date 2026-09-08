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
| **Metric** | `1 - 0.5*(corpus CER + corpus WER)`, higher is better — corpus-level (sum of edits / sum of reference lengths), not per-sample averaged. Verified against a real submission to 8 decimal places; see the Metric section for the full derivation. |

---

## The core bet

This is **not** a full-page layout problem — the images are already cropped to individual words or short lines, so segmentation is a non-issue. That reframes it as a small-data transfer-learning problem: ~6,000 crops of 18th–19th century Barbadian legal handwriting is nowhere near enough to train a recognizer from scratch.

Every point of leaderboard movement will come from:

1. **How well you transfer** a model already fluent in handwriting.
2. **How well you augment** the faded ink and irregular penmanship this specific archive has.
3. **How well you combine** a few complementary models rather than betting everything on one architecture.

---

## Scope: what's actually required vs. what's a bet

Solo, 33 days, one 12GB laptop GPU. The original version of this plan tried to schedule ten distinct techniques — that's over-committed for one person. Everything below is sorted into three tiers with explicit gates between them: don't start a Stretch item until the tier below it has already shown a real CV gain.

**Must (the score doesn't exist without these):**
- Exact metric implementation, hand-verified (Phase 1)
- Length-stratified local CV, honestly interpreted (see Validation strategy below — no document/writer metadata exists in this dataset, so this is the best split available, not a perfect one)
- A strong pretrained baseline: TrOCR-base-handwritten, fine-tuned properly
- Augmentation, validated incrementally (one technique at a time against CV, not stacked blind)
- Beam decoding
- A 2-model ensemble

**Should (build if the Must tier is solid and gate below is green):**
- HTR-VT as the second, architecturally-different model — CTC vs. TrOCR's seq2seq gives the ensemble something real to combine, and it's a good build even if it doesn't win outright
- Domain self-supervised (MIM) pretraining, *if the competition rules permit using unlabeled test images this way* — verify this explicitly before running it; if unclear, pretrain on training images only

**Stretch (gate each individually — only proceed if the tier it depends on already helped):**
- One-DM style-conditioned synthesis (gated on: does augmentation alone already show diminishing CV returns?)
- VLM QLoRA fine-tune (gated on: does the 2-model ensemble leave clear error patterns a VLM's general visual-text prior might fix? Treat as a last-resort diversity source, not a central phase — general VLMs are prone to plausible-looking hallucinated words on historical handwriting)
- LLM post-OCR correction (gated on: does a much cheaper, safer rule-based confusion-correction layer — u/v, i/j, long-s style confusions — already fail to close the gap?)
- Ensemble → single-model distillation (gated on: does the ensemble clearly beat every individual member on CV? If not, distillation just adds complexity for no proven benefit.)

If a Stretch item's gate isn't green by the time you'd start it in the schedule below, skip it and reinvest that time into decoding, error analysis, or reproducibility instead — those are Must-tier and always pay off.

---

## Rules & chat findings (verified 2026-09-04)

A full pass through the official rules page and every discussion thread on the competition's Chat tab, done after the first real submission surfaced questions the original version of this plan hadn't resolved. Findings below are organizer-confirmed unless marked otherwise.

**Data quality — actionable now:**
- **21 known-corrupted training rows** (image/label mismatches, empty images) are community-identified and organizer-acknowledged as excludable (thread 33891, IDs shared 18 Jul 2026). Already applied: `src/ledger_htr/data/known_issues.py` holds the list, `cv_split.py` excludes them before making folds. The original `Train.csv` stays untouched, as the organizer explicitly required.
- **Unresolved multi-line labeling ambiguity** (thread 34089, "Must-Read"): some multi-line crops are labeled with only the center/main line's text, others with the full multi-line transcription — no consistent rule, and no organizer resolution was posted as of this check (they asked for example IDs on 5 Aug, last reply 21 Aug still unresolved). Example flagged IDs: `ptuXstzPGsZ5p9Wl`, `Xf53GrwovECETF4H`, `259Ksw56mJJlnipt`, `2KW2WCEcogSZEaxB`. This is real label noise sitting in the training set with no clean fix available yet — worth an EDA pass correlating crop height against transcription word-count to empirically flag likely-affected rows (tall crop + suspiciously short label = candidate for center-line-only labeling), but don't expect to fully resolve it. Revisit if it's still unresolved by the time error analysis (Phase 2/3) is underway.

**Rules confirmations that unblock or reshape parts of this plan:**
- **Test-image self-supervised pretraining is explicitly allowed** (thread 34459, meganomaly/Zindi, 17 Aug 2026): "Transductive pseudo-labeling / self-training on the test images is allowed, provided the entire process is fully automated." MIM/SSL on raw test pixels (no labels touched) clearly qualifies — it's less aggressive than the pseudo-labeling this ruling explicitly covers. The Should-tier SSL pretraining step no longer needs the "verify first" hedge it had.
- **StackMix-style augmentation is explicitly allowed** (thread 34514, meganomaly/Zindi, 24 Aug 2026): segmenting character-level crops from the *provided training images only* and recombining them into new synthetic lines (label = concatenation of the source crops' labels) counts as ordinary data augmentation, not "using an external dataset" — provided it's fully automated, train-images-only, and any segmentation tooling used also complies with the pretrained-model rules. **Tried and deferred, 2026-09-04**: prototyped whitespace-gap word segmentation (binarize, find column-wise ink gaps) against 30 real training lines — 0/30 got an exact word-count match, even after smoothing to merge noise gaps. This isn't a tuning problem: this archive's cursive hand frequently doesn't lift the pen between words at all, so there's often no real gap to find. A crude proportional-width fallback (split by character-count ratio, ignoring actual ink) would produce arbitrary, often letter-splitting boundaries — quality low enough to risk hurting more than helping. **Revised plan: defer StackMix until after HTR-VT exists**, and drive segmentation from its own CTC forced-alignment instead of blind pixel heuristics — this is the standard reason StackMix-style techniques are normally built on top of an existing recognizer rather than as a pure CV preprocessing step. Elastic distortion doesn't have this problem (works directly on the whole line, no segmentation needed) and is the Phase 3 augmentation actually running first.
- **Excluding clearly-corrupted training rows is allowed**, and so is standard hyperparameter search tooling like Optuna — explicitly distinguished from the AutoML ban (thread 33891, meganomaly/Zindi, 16 Jul 2026).
- **Renting cloud GPU compute (RunPod/AWS/Nebius/etc.) is allowed** (thread 33870, meganomaly/Zindi, 16 Jul 2026) as long as the data/solution stay private, no proprietary model API or managed model service is used, and the solution stays fully reproducible with documented hardware/runtime. Worth keeping in back pocket if the 12GB laptop becomes a hard bottleneck for the VLM Stretch-tier work — not needed for anything Must/Should-tier.

**Real constraints this plan hadn't fully accounted for:**
- **Only the competition-provided data may be used for training** — the rules page states this explicitly ("You may use only the datasets provided for this challenge"), confirmed again in an unanswered-but-redundant chat question about MNIST/EMNIST/READ/ICFHR (thread 34612). This matters specifically for HTR-VT: its paper validates on READ2016 and similar external historical-manuscript sets, but we can only use *released pretrained weights* from that lineage if any exist openly — not download and additionally train on READ2016 ourselves. Same applies to any IAM-based augmentation data beyond what a pretrained checkpoint already encodes.
- **"Openly available to everyone" for pretrained models means license permits commercial use, not just free download.** Confirmed twice (threads 34468 and 34053, meganomaly/Zindi): `stanford-oval/churro-3B` — a 3B VLM fine-tuned specifically for historical-document transcription, which would otherwise have been an excellent Stretch-tier VLM candidate — is explicitly **disallowed** because its Qwen Research License is non-commercial, even though the weights are freely downloadable to anyone. `PP-OCRv6 Medium` (Apache-2.0, PaddleOCR's model) was separately confirmed usable (same thread) and is a reasonable addition to the "optional sanity check" row if PyLaia proves inconvenient to set up.
- **Correction, 2026-09-04: `Qwen2.5-VL-3B-Instruct` — this plan's own named VLM candidate — is under that exact same disallowed license, not Apache-2.0.** An earlier pass through this plan claimed it was Apache-2.0, sourced from an unverified claim made in a Zindi chat thread by another participant, not independently checked. Fetching the model's actual `LICENSE` file on Hugging Face (`huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE`) shows it's the "Qwen RESEARCH LICENSE AGREEMENT" — "FOR NON-COMMERCIAL PURPOSES ONLY... If you are commercially using the Materials, you shall request a license from us." Same license family as the already-disallowed CHURRO-3B. **Lesson: verify a license by reading the actual LICENSE file, never by trusting a tag, a claim in chat, or "the base model is probably more open" reasoning** — a fine-tune and its base can carry the same restrictive license even when a HF model card's visible tags don't surface it clearly (Qwen2.5-VL-3B-Instruct's page doesn't even show a "License:" metadata tag the way properly-tagged repos like TrOCR's do — its absence was itself a signal worth checking further).

**Also: Qwen2.5-VL is superseded anyway — check current-generation models, not just current-generation licenses.** Checking Qwen's HF org directly (prompted by a fair "isn't Qwen2.5 old by now" question) turned up `Qwen3-VL-4B-Instruct` and `Qwen3-VL-2B-Instruct` — right-sized for this plan's 12GB budget, and genuinely Apache-2.0 (clean tag, no overriding LICENSE file, unlike 2.5-VL's hidden restrictive one). The Model options table's VLM row now points to `Qwen3-VL-4B-Instruct` as the preferred pick, with `InternVL2.5-4B` (MIT) kept as a second, architecturally-different option.

**Checked `Qwen3.8` too (the actually-newest generation) — wrong fit, but for size, not license.** `Qwen/Qwen3.8-27B` is also Apache-2.0, but the smallest official Qwen3.8 variant is 27B/28B params — no small 2-4B option exists in that generation the way there is for Qwen3-VL. At 4-bit quantization, 28B params alone needs ~14GB, already exceeding the entire 12GB card before any LoRA/activation/KV-cache overhead — not viable on this hardware regardless of license. Most other Qwen3.8 search results were third-party "uncensored/abliterated" re-uploads, not worth pursuing for a competition submission needing clean provenance anyway. Stick with `Qwen3-VL-4B-Instruct` on the current 12GB laptop.

**Contingent upgrade path, if the compute constraint changes** (user flagged 2026-09-04 that it might): `Qwen3.8-27B` is the one to reach for instead — current-generation, Apache-2.0, and meaningfully more capable than the 4B line, just needs real VRAM headroom (rough estimate: 20GB+ for QLoRA at 4-bit with any reasonable batch size, more comfortable at 24GB+). If the resolution is cloud GPU rental rather than new local hardware, that's already confirmed compliant (rules & chat findings above: RunPod/AWS/Nebius allowed if data/solution stay private and it's fully reproducible). Revisit this row once/if more VRAM is actually available rather than assuming the 4B pick is final.
- **HTR-VT's own GitHub repo (`Intellindust-AI-Lab/HTR-VT`, the canonical upstream of the `YutingLi0606/HTR-VT` this project cloned into `external/`) has no LICENSE file at all** — no license badge, nothing in the file tree, checkpoints distributed via an unlicensed Google Drive link. No explicit license means default copyright applies (all rights reserved), not an open license. This doesn't block the plan's actual approach — we're reimplementing the architecture ourselves from the published, peer-reviewed paper (reading their code purely to understand implementation details like patch embedding and span-masking, not redistributing their files), which sidesteps the license question entirely. It *would* matter if we ever wanted to warm-start from their released checkpoints directly rather than training from scratch on this competition's data — that would need explicit Zindi clearance first, same as any other unclearly-licensed pretrained artifact.
- **Submission budget is 200 total, not just 5/day.** The Pitfalls section below already covers the 5/day cap; 200 total across the whole 33-day run is generous but not infinite — don't treat submissions as free just because the daily cap resets.
- The rules page itself is internally inconsistent about the public/private split: the Rules bullet list says 30%/70%, the Submissions section says 20%/80%. Not something we can resolve — just don't be surprised if the actual split doesn't match the 30% figure used elsewhere in this plan.

**Strategy intelligence from the leaderboard's actual top performers:**
- A competitor scoring just above 0.91 (leaderboard top tier as of this check) described their approach in one line (thread 34350): **"Qwen + LoRA + GRPO and beam search."** GRPO (Group Relative Policy Optimization — the RL fine-tuning technique behind DeepSeek-R1 and similar) is not anywhere in this plan's current Stretch tier, and its appearance in a top score is a real signal that VLM + RL fine-tuning (not just supervised QLoRA) is part of what's separating the top of the leaderboard from the rest. Worth reading up on GRPO specifically before the VLM Stretch-tier week, as a possible upgrade to the plain QLoRA supervised fine-tune already planned — not a replacement for the Must/Should tiers, but a reason to take the VLM Stretch item more seriously if it's reached.
- The same competitor added: "I think some of the data might be bad and I shouldn't use all the data" — independent confirmation (from someone scoring near the top) that the training-data cleaning work above (corrupted rows, multi-line ambiguity) is a real lever, not a minor footnote.

---

## The differentiator stack

What most entrants won't try, pulled from papers published in the last 12–18 months and chosen because they actually fit a 6,000-sample archive and one 12GB laptop GPU, not because they're trendy. Framed against the tiers above, not as a guaranteed sequence.

**HTR-VT as the second model (Should-tier).** An encoder-only ViT with a CNN stem, CTC head, SAM optimizer, and span-mask regularization (Li et al., 2024). Its premise is matching CNN-RNN-CTC and TrOCR-scale models *without* needing large external pretraining — a promising fit for this dataset's size on paper, but compare it against the TrOCR baseline on CV rather than assuming it wins; it's also a real implementation-from-repo effort, not a one-line `.from_pretrained()`, so budget real debugging time.

**Domain self-supervised pretraining (Should-tier, confirmed allowed).** Masked-image-modeling on the provided crops before supervised fine-tuning, including test-image pixels. Confirmed by a Zindi organizer in chat ("Answering queries around pseudo-labelling", meganomaly/Zindi, 17 Aug 2026): "Transductive pseudo-labeling / self-training on the test images is allowed, provided the entire process is fully automated" — MIM/SSL on raw test pixels (no labels touched at all) clearly falls within this, being even less aggressive than the pseudo-labeling that ruling explicitly covers. This warms the encoder to this exact paper, ink, and penmanship before it ever sees a transcription.

**Style-conditioned diffusion synthesis (Stretch).** One-DM generates realistic handwriting from a single style exemplar via a diffusion model conditioned on high-frequency stroke detail. Condition it on your own training crops to synthesize more examples of the rare letterforms and ligatures early EDA turns up, instead of generic font-rendered synthetic text. Only worth it once plain augmentation's CV gains have visibly plateaued.

**LLM post-OCR correction — done carefully, and only as a last resort (Stretch).** 2025 research on exactly this setup (English, historical documents) found real gains from open LLMs, but only with alignment-based post-processing to strip hallucinated additions, and only once you know whether your ground truth normalizes archaic spelling (EDA already answered this — see Validation strategy below: the character set is fully modern ASCII, no archaic glyphs). Historical transcription is exact-match sensitive: an LLM correcting "wrong-looking" archaic spelling that's actually correct, or a lexicon snap that mangles a proper noun, actively hurts score. Try a small rule-based confusion-correction layer first (see Technical defaults). Gate every correction change on measured CV improvement; never force every output through it.

**Distillation: ensemble → one lean model (Stretch, gated on ensemble beating its members).** HTR-JAND (2024) reports a 62% relative CER cut from temperature-scaled soft-label distillation in its ablation — that's evidence the mechanism can work, not a number that transfers exactly to this dataset. Distill your ensemble into a single compact HTR-VT student only if the ensemble has already clearly beaten every individual model on CV. Full recipe below.

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

> With this little real data, pretraining and augmentation are where most of the score is actually won. A published result on a comparably small historical-manuscript set (~4k lines) took CER from 1.93 to 1.60 through augmentation and ensembling alone — treat that as the floor, not the ceiling. Augmentation is Must-tier and unconditional; SSL pretraining is Should-tier and conditional (verify test-image use is allowed first); One-DM (Sat) is Stretch-tier — only run it if augmentation's CV gains have already started to plateau by Thursday/Friday.

**Mon · Sep 14**
- [ ] **Learn** — MAE paper — the masked-image-modeling recipe you're about to run
- [ ] **Build** — verify whether the competition rules permit unlabeled test-image use; if yes, implement masked-image-modeling self-supervised warm-up on all 6,000 provided crops (train + test pixels, no labels); if no or unclear, restrict to training crops only

**Tue · Sep 15**
- [ ] **Build** — finish the SSL pretraining run; load those weights as HTR-VT's encoder init; confirm it trains stably from there

**Wed · Sep 16**
- [ ] **Learn** — Lilian Weng's diffusion-models explainer, ahead of Saturday's One-DM work
- [ ] **Build** — add elastic distortion as the first augmentation to try (reported as the strongest single-technique gain in the reference study — verify that holds here before layering on more)

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

> This week is entirely Stretch-tier (see Scope section) — only enter it with a gate already green: the Must+Should tiers are solid on CV and a 2-model ensemble is already beating its members. A VLM fine-tune, careful LLM correction, and distillation are potential differentiators here, not guaranteed ones — each still needs its own gate to clear before it's worth the days it costs.

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

| Family | Why it's here | Watch out for | Role | ~VRAM (typical batch) | ~Time/run (12GB laptop) |
|---|---|---|---|---|---|
| **TrOCR-base-handwritten** | Fast, stable, easy HF fine-tuning — the actual Must-tier baseline | Seq2seq decoder can repeat/hallucinate words with too little fine-tuning data; needs matched preprocessing | Baseline | ~4–6GB at bs 16–32, fp16 | ~20–40 min/fold, few epochs |
| **TrOCR-large-handwritten** | Same lineage, more capacity — only if base's CV clearly has headroom | Meaningfully more VRAM/time than base for a dataset this size; may not pay off | Should-tier upgrade, not default | ~8–10GB at bs 8–16, fp16 | ~1–2 hr/fold |
| **HTR-VT** (Li et al., 2024) | CNN stem + ViT encoder + CTC head, built to compete without large external pretraining — validated on the historical READ2016 set | Implementing from a paper/repo, not `.from_pretrained()` — budget real debugging time | Should-tier second model | ~4–8GB depending on patch/seq config | ~1–2 hr/fold once stable |
| **CRNN + CTC** (PyLaia-style, optional) | Classic Transkribus/READ-pipeline workhorse — quick to train, a useful sanity check | Adds a 4th architecture on top of an already ambitious stack — only build if HTR-VT proves genuinely unstable, not as a parallel Day-1 track | Fallback only, not scheduled by default | ~2–4GB | ~20–30 min/fold |
| **VLM + QLoRA** (`Qwen3-VL-4B-Instruct` — Apache-2.0, verified 2026-09-04, preferred: current-generation, right-sized, genuinely clean license tag with no overriding LICENSE file; `Qwen3-VL-2B-Instruct` as a lighter fallback if VRAM is tight; `InternVL2.5-4B` — MIT, also verified — as a second option for ensemble diversity if two VLMs are ever wanted. **Not** `Qwen2.5-VL-3B-Instruct` (superseded anyway) — its actual LICENSE file is the same non-commercial "Qwen RESEARCH LICENSE AGREEMENT" that got `stanford-oval/churro-3B` disallowed; an earlier pass through this plan wrongly called it Apache-2.0) | General visual-text pretraining transfers surprisingly well to messy real handwriting, per recent low-resource-script results — and a top scorer's disclosed approach ("Qwen + LoRA + GRPO") suggests this is genuinely load-bearing for the leaderboard's top tier, not just a diversity play (though note: if they're literally using Qwen2.5-VL, that itself may be a rules violation on their end, not evidence it's safe for us) | Prone to plausible-but-wrong hallucinated words on historical handwriting; 12GB fits a 3–4B model with 4-bit + LoRA + gradient checkpointing, a 7B model leaves little headroom for batch size; **always check a candidate checkpoint's actual LICENSE file directly, not a claim made about it or its tag alone** — see the Qwen2.5-VL correction here | Stretch, but worth taking seriously if reached | ~9–11GB with 4-bit + LoRA + grad checkpointing | Multiple hours/run — budget accordingly |

All figures above are rough planning estimates for this dataset's crop sizes, not measured — confirm actuals once each model is running and adjust batch size/gradient accumulation accordingly.

---

## Validation strategy

Ideally CV folds would be grouped by document/page/writer so the same scribe's handwriting never appears in both train and validation within a fold — random splitting risks the model "learning the writer" rather than generalizing. **Checked directly: this dataset provides no such metadata.** `Train.csv`/`Test.csv` contain only `ID, Target` / `ID`; IDs are opaque random hashes with no discoverable structure; the download manifest lists only the four top-level files. Document-grouped splitting is therefore not implementable as stated — length-stratified 5-fold (already built, see `src/ledger_htr/data/cv_split.py`) is the best split available, not a perfect one.

What partially bounds the risk: exact-duplicate leakage is negligible (only 22/4,098 rows — 0.5% — share an identical `Target` string with another row, confirmed via EDA). Style leakage from *different* lines by the same scribe landing in different folds is real and currently unmeasurable without metadata. If a trained model's CV error looks suspiciously low relative to leaderboard performance, that's the first thing to suspect — cross-check by eye against the Otsu-comparison samples for visually-similar paper/ink clusters that might hint at shared-source pages.

Practical fold-count tip: use fewer folds (e.g. 3) while iterating quickly on architecture/hyperparameters, and only spend the compute on the full 5-fold run for models being seriously compared or selected as finalists.

## Technical defaults

Concrete defaults for the Must/Should-tier build, so each model shares the same assumptions and comparisons stay fair.

**Preprocessing**
- Resize to a fixed height, pad width dynamically (preserve aspect ratio) — do not force a fixed aspect ratio; EDA measured widths from 267–5,746px on this archive.
- Bucket samples by width so batches don't waste computation on padding.
- Carry an explicit padding mask through to attention layers and CTC decoding.
- Global Otsu binarization is **not sufficient on its own** — EDA's `otsu_comparison.png` shows it turning stained/foxed backgrounds into black blobs on roughly half of a small sample. Use `cv2.adaptiveThreshold` or per-tile Otsu instead if binarization is used at all in the final pipeline (it may not be necessary — test with and without against CV).
- Images are RGB as provided; if a model's pretrained backbone expects 3-channel input (TrOCR, ImageNet-style CNN stems), keep them RGB or convert to grayscale and replicate to 3 channels — don't silently mismatch what the pretrained weights saw.
- Character-level vocabulary: EDA already enumerated it — 81 chars, standard ASCII letters/digits/punctuation, no long-s or other archaic glyphs (`reports/eda_findings.md`). Still include an explicit unknown-character token for robustness against anything unseen at test time.

**Training**
- **CTC timestep check is a correctness requirement, not a nice-to-have**: the encoder's output sequence length must exceed the longest target length (max observed: 120 characters) or CTC loss silently degrades. Assert this explicitly once a model's downsampling factor is fixed, across the actual crop-width distribution — not just the average case.
- Mixed precision, gradient accumulation, gradient clipping, and checkpoint averaging as defaults, not optional extras.
- Compare every change against a fixed baseline on the same folds and seeds — no changing two things at once between CV runs.
- Early-stop on the actual weighted metric (once verified — see Metric section), not on loss alone.

**Augmentation**
- Validate incrementally: add one technique, check CV, keep or drop, then move to the next — don't stack elastic + affine + perspective + cutout all at once and hope. Mild, archive-realistic transforms first (ink fading/contrast jitter, blur/scan noise, small affine/elastic deformation, slight erosion/dilation); hold off on aggressive perspective warps, large rotations, or cutout masking until validation actually shows they help — those can destroy historically meaningful letterforms rather than just adding noise.

**Decoding & post-processing**
- Order of increasing risk/complexity: greedy decode → beam search → KenLM rescoring → lexicon constraints. Each step should be justified by a CV improvement over the previous one, not added by default.
- Never force every prediction onto the nearest lexicon word — proper nouns, place names, and genuinely correct archaic spellings will get mangled. Evaluate error rate on those categories separately before deciding how aggressively to snap.
- Before reaching for general LLM post-correction (Stretch-tier, see Scope section), try a small rule-based confusion-correction layer targeting known historical OCR confusions (u/v, i/j, long-s-adjacent patterns) — much lower hallucination risk, much easier to validate, and may close most of the gap on its own.

**Ensembling**
- Combine at the probability level (normalized beam/hypothesis scores), not naive character-level voting.
- Validate ensemble combination weights on out-of-fold predictions, the same way any other hyperparameter would be tuned.

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

## Metric — verified, not approximated

**This is now fully verified, not assumed.** Two earlier passes through this plan got the formula wrong in different ways — first assuming a 0.7/0.3 CER/WER combination copied from `Starters.zip`'s `eval_metrics.py` defaults, then correctly fixing the combination weight to 0.5/0.5 (per the official rules page) but still assuming a per-sample `sqrt(reference_length)`-weighted average, which turned out to also be wrong. The real formula was reverse-engineered from the Zindi discussion board and confirmed against our own real submission to 8 decimal places — see below.

```
CER_weighted = sum(char_edit_distance_i for all i) / sum(char_reference_length_i for all i)
WER_weighted = sum(word_edit_distance_i for all i) / sum(word_reference_length_i for all i)
final_score  = 1 - 0.5 * (CER_weighted + WER_weighted)
```

This is standard **corpus-level (micro-averaged) WER/CER** — total edits across every sample divided by total reference length across every sample — not a per-sample rate averaged across samples. It's algebraically identical to `mean(edit_distance) / mean(reference_length)` over the same set, and it's exactly what the rules page's plain-English description means by "longer reference transcriptions are weighted more heavily": a sample with a longer reference has more room for edits, so it naturally pulls more weight in the corpus-level sum than a short one would in a simple average of per-sample rates. `final_score` is 1 minus that combined error rate, so **higher is better** (matches the leaderboard's sort order — rank 1 has the highest score).

Missing, empty, or invalid predictions are scored as if the prediction were an empty string (maximal edit distance against the reference), not excluded from the corpus sums — per the rules page: "Missing predictions, empty predictions, or invalid text values will be penalised as incorrect."

**How this was confirmed, concretely:**
1. Zindi discussion "Help Please" (thread 33847, user J0NNY, 4 upvotes, no organizer correction) posted a reverse-engineered snippet: `score = 1 - 0.5 * (wer_weighted/12 + cer_weighted/55)`, where `wer_weighted`/`cer_weighted` are literally the mean edit distance values the leaderboard displays in its "WER Weighted"/"CER Weighted" columns, and 12/55 are the mean reference word/char length of whatever split is being scored (not universal constants — see below).
2. Our own real submission (Ish1105, 2026-09-04) displayed `WER Weighted: 8.483792682`, `CER Weighted: 30.34969263`, `Public Score: 0.370602341`. Plugging into the formula above: `1 - 0.5*(8.483792682/12 + 30.34969263/55) = 0.370602342` — matches to 8 decimal places.
3. This also cleanly resolves an earlier mystery: WER Weighted (8.48) looking *smaller* than CER Weighted (30.35) had seemed backwards (a wrong word usually costs more than one wrong character). It isn't backwards — those are raw mean edit-distance *counts*, not rates, and a ~55-char line naturally accumulates more raw character edits than its ~12-word tokenization accumulates word edits.
4. `evaluations/wer.py` + `cer.py` (the actual source `eval_metrics.py` imports) are still not published anywhere — a direct question about this ("STARTER CODE EVALUATION MODULE", 15 Jul 2026) got zero organizer response. This reverse-engineered formula is our best evidence, verified against real data, not a guess.

The local scorer (`src/ledger_htr/metrics/scorer.py`) implements the general `sum(edits)/sum(lengths)` form directly rather than hardcoding 12/55 — those are just the mean reference lengths of whatever split J0NNY and our submission happened to be scored against (likely the public 20-30% test slice), and hardcoding them would silently give wrong numbers on a local validation fold with different mean lengths. Computing the sums directly makes the scorer self-normalizing to whatever set it's run against.

**Real impact of getting this wrong twice:** the Day 4 TrOCR baseline was originally scored locally at 0.5448 (wrong 0.7/0.3 combination), then 0.5938 (right combination, still-wrong per-sample averaging) — both were error-rate-style numbers (lower looked better) computed against no real reference point. Rerunning the *same* saved checkpoint's validation set through the corrected formula gives **CER=0.4427, WER=0.6982, final=0.4296** (now correctly an accuracy-style number, higher-is-better, matching the leaderboard's Public Score convention). Compared to the actual public leaderboard score of 0.3706, that's a real but modest generalization gap (~0.06, consistent with the mild overfitting already visible in the training curves) — not the alarming near-2x mismatch it looked like before the formula was fixed. That mismatch was mostly a wrong-formula artifact, not a real problem with the model.

---

## Where the score actually comes from

**High leverage:** pretrained transfer plus a genuinely different architecture (HTR-VT) · domain self-supervised pretraining on all 6,000 images · elastic/affine augmentation matched to ink and paper degradation · style-conditioned diffusion synthesis for rare letterforms · lexicon-constrained decoding · ensembling 2–3 genuinely different architectures.

**Low leverage:** hyperparameter micro-tuning a single model past the first good setting · training a diffusion generator from scratch on one laptop GPU instead of using a released checkpoint · chasing the public leaderboard (only ~30% of the test set) instead of your CV.

---

## Pitfalls specific to this challenge

- **Small LB** — the public leaderboard is a small, noisy slice; a submission that looks worse there can be genuinely better. Select your final two submissions from 5-fold CV, not leaderboard rank.
- **Empty preds** — missing or empty predictions are scored as flat-out wrong; always run a completeness check against `SampleSubmission.csv` before submitting.
- **Reproducibility** — top 10 gets a code request with a 48-hour window; a result that doesn't reproduce gets your rank adjusted down to what the code actually produces. Seed everything now, not in Phase 5.
- **Open-source only** — no AutoML tools, no closed APIs; every pretrained checkpoint and package needs to be genuinely open *and* commercially-licensed (research-only licenses like Qwen Research License are explicitly disallowed even when the weights are freely downloadable — see Rules & chat findings above).
- **Submission budget** — 200 total across the challenge, not just the 5/day cap. Don't burn submissions casually; the daily reset doesn't mean they're free.
- **Provided data only** — no external training datasets (MNIST, EMNIST, READ2016, IAM, etc.), even though pretrained *weights* from models trained on such data are fine.

---

## Teaming

Teams run up to 4 and can merge later, but a team formed after submissions have started can't be disbanded and members can't leave once a submission is in — so if you bring someone on, do it deliberately (someone strong on the language-model/lexicon side pairs well with a CV background) rather than defaulting into it. Solo is a completely reasonable choice here too, especially for the learning goal.

---

*Plan begins 03 Sep 2026 · solo/team entry into the R.O.A.D. Barbados Historic Handwriting Challenge.*
