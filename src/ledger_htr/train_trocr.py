import argparse
import os
import time

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

from ledger_htr.augment.augment import build_elastic_transform
from ledger_htr.config import ExperimentConfig, load_config, save_config
from ledger_htr.data.dataset import LedgerSeq2SeqDataset, load_fold_split
from ledger_htr.metrics.scorer import final_score_from_dicts
from ledger_htr.utils.logging import RunLogger
from ledger_htr.utils.seed import set_seed


def build_model(model_name: str, processor: TrOCRProcessor) -> VisionEncoderDecoderModel:
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    # standard TrOCR fine-tuning config (HF walkthrough)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    # generation params live on generation_config, not config, in this transformers version
    model.generation_config.max_length = 140
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model


@torch.no_grad()
def run_validation(model, processor, val_df, val_loader, device) -> dict:
    model.eval()
    model.config.use_cache = True  # safe here: no backward pass, so gradient-checkpointing/cache conflict doesn't apply
    all_preds = []
    for batch in val_loader:
        pixel_values = batch["pixel_values"].to(device)
        generated_ids = model.generate(pixel_values=pixel_values, max_length=model.generation_config.max_length)
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        all_preds.extend(texts)
    model.config.use_cache = False

    preds = dict(zip(val_df["ID"].astype(str), all_preds))
    refs = dict(zip(val_df["ID"].astype(str), val_df["Target"].astype(str)))
    return final_score_from_dicts(preds, refs)


def train(cfg: ExperimentConfig, max_train_samples: int | None = None, max_val_samples: int | None = None) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    processor = TrOCRProcessor.from_pretrained(cfg.model_name)
    model = build_model(cfg.model_name, processor).to(device)

    train_csv = os.path.join(cfg.data_dir, "Train.csv")
    train_df, val_df = load_fold_split(train_csv, cfg.folds_csv, cfg.fold)
    if max_train_samples:
        train_df = train_df.sample(n=min(max_train_samples, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    if max_val_samples:
        val_df = val_df.sample(n=min(max_val_samples, len(val_df)), random_state=cfg.seed).reset_index(drop=True)
    print(f"train: {len(train_df)}, val: {len(val_df)}")

    train_transform = build_elastic_transform() if cfg.use_elastic_augment else None
    if cfg.use_elastic_augment:
        print("elastic distortion augmentation: ON (train split only)")
    train_ds = LedgerSeq2SeqDataset(train_df, cfg.image_dir, processor, transform=train_transform)
    val_ds = LedgerSeq2SeqDataset(val_df, cfg.image_dir, processor)  # always clean, no augmentation
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision and device == "cuda")

    logger = RunLogger(cfg.runs_dir, cfg.run_name)
    logger.log_config(cfg)
    save_config(cfg, os.path.join(logger.run_dir, "resolved_config.yaml"))

    # final_score is accuracy-style (1 - error), higher is better -- see
    # ledger_htr.metrics.scorer's module docstring for the corrected formula.
    best_final = float("-inf")
    best_ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name, "best")

    for epoch in range(cfg.num_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision and device == "cuda"):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg.grad_accum_steps

            if (step + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 20 == 0:
                print(f"epoch {epoch} step {step + 1}/{len(train_loader)} loss={running_loss / (step + 1):.4f}")

        train_loss = running_loss / len(train_loader)
        val_result = run_validation(model, processor, val_df, val_loader, device)
        elapsed = time.time() - epoch_start

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_cer={val_result['cer']:.4f} val_wer={val_result['wer']:.4f} "
            f"val_final={val_result['final']:.4f} ({elapsed:.0f}s)"
        )
        logger.log_metrics(
            step=epoch,
            train_loss=train_loss,
            val_cer=val_result["cer"],
            val_wer=val_result["wer"],
            val_final=val_result["final"],
            elapsed_sec=elapsed,
        )

        if val_result["final"] > best_final:
            best_final = val_result["final"]
            os.makedirs(best_ckpt_dir, exist_ok=True)
            model.save_pretrained(best_ckpt_dir)
            processor.save_pretrained(best_ckpt_dir)
            print(f"  new best (val_final={best_final:.4f}), saved to {best_ckpt_dir}")

    print(f"\nDone. Best val_final={best_final:.4f}, checkpoint at {best_ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples)
