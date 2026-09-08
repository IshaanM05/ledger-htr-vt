import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from ledger_htr.augment.augment import build_elastic_transform_raw
from ledger_htr.config import HTRVTConfig, load_config, save_config
from ledger_htr.data.dataset import load_fold_split
from ledger_htr.data.htrvt_dataset import HTRVTLedgerDataset, build_char_list, htrvt_collate
from ledger_htr.decode.ctc_codec import CTCCodec
from ledger_htr.metrics.scorer import final_score_from_dicts
from ledger_htr.models.htr_vt import create_model
from ledger_htr.optim.sam import SAM
from ledger_htr.utils.ema import EmaModel
from ledger_htr.utils.logging import RunLogger
from ledger_htr.utils.lr_schedule import cosine_lr_with_warmup
from ledger_htr.utils.seed import set_seed


def compute_loss(model, images, texts, codec: CTCCodec, criterion, cfg: HTRVTConfig, device) -> torch.Tensor:
    target_flat, target_lengths = codec.encode(texts)
    batch_size = images.size(0)
    logits = model(images, cfg.mask_ratio, cfg.max_span_length, use_masking=True)
    log_probs = logits.float().permute(1, 0, 2).log_softmax(2)  # (T, B, C) for CTCLoss
    input_lengths = torch.IntTensor([log_probs.size(0)] * batch_size)

    torch.backends.cudnn.enabled = False
    loss = criterion(log_probs, target_flat.to(device), input_lengths.to(device), target_lengths.to(device)).mean()
    torch.backends.cudnn.enabled = True
    return loss


@torch.no_grad()
def run_validation(model, codec: CTCCodec, val_df, val_loader, device) -> dict:
    model.eval()
    all_preds: list[str] = []
    for images, _texts in val_loader:
        images = images.to(device)
        logits = model(images, 0.0, 1, use_masking=False)
        preds_index = logits.argmax(2)  # (B, T)
        all_preds.extend(codec.decode_greedy(preds_index.cpu()))
    model.train()

    ids = val_df["ID"].astype(str).tolist()
    preds = dict(zip(ids, all_preds))
    refs = dict(zip(ids, val_df["Target"].astype(str)))
    return final_score_from_dicts(preds, refs)


def train(cfg: HTRVTConfig, max_train_samples: int | None = None, max_val_samples: int | None = None) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_csv = os.path.join(cfg.data_dir, "Train.csv")
    train_df, val_df = load_fold_split(train_csv, cfg.folds_csv, cfg.fold)
    if max_train_samples:
        train_df = train_df.sample(n=min(max_train_samples, len(train_df)), random_state=cfg.seed).reset_index(drop=True)
    if max_val_samples:
        val_df = val_df.sample(n=min(max_val_samples, len(val_df)), random_state=cfg.seed).reset_index(drop=True)
    print(f"train: {len(train_df)}, val: {len(val_df)}")

    alphabet = build_char_list(train_csv, cfg.folds_csv)
    codec = CTCCodec(alphabet)
    print(f"alphabet size: {len(alphabet)} chars, num_classes (incl. blank): {codec.num_classes}")

    model = create_model(nb_classes=codec.num_classes).to(device)
    total_param = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_param:,}")
    model.train()
    ema = EmaModel(model, cfg.ema_decay)
    ema.ema.to(device)

    train_transform = build_elastic_transform_raw() if cfg.use_elastic_augment else None
    if cfg.use_elastic_augment:
        print("elastic distortion augmentation: ON (train split only)")
    train_ds = HTRVTLedgerDataset(train_df, cfg.image_dir, cfg.target_height, cfg.max_width, transform=train_transform)
    val_ds = HTRVTLedgerDataset(val_df, cfg.image_dir, cfg.target_height, cfg.max_width)  # always clean
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train_batch_size, shuffle=True, num_workers=2, collate_fn=htrvt_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.val_batch_size, shuffle=False, num_workers=2, collate_fn=htrvt_collate
    )

    def cycle(loader):
        while True:
            yield from loader

    train_iter = cycle(train_loader)

    optimizer = SAM(model.parameters(), torch.optim.AdamW, rho=0.05, lr=1e-7, betas=(0.9, 0.99), weight_decay=cfg.weight_decay)
    criterion = torch.nn.CTCLoss(reduction="none", zero_infinity=True)

    logger = RunLogger(cfg.runs_dir, cfg.run_name)
    logger.log_config(cfg)
    save_config(cfg, os.path.join(logger.run_dir, "resolved_config.yaml"))

    best_final = float("-inf")
    best_ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name, "best")
    os.makedirs(best_ckpt_dir, exist_ok=True)

    running_loss = 0.0
    last_avg_loss = None
    window_start = time.time()
    for nb_iter in range(1, cfg.total_iters + 1):
        current_lr = cosine_lr_with_warmup(nb_iter, cfg.warmup_iters, cfg.total_iters, cfg.max_lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        images, texts = next(train_iter)
        images = images.to(device)

        loss = compute_loss(model, images, texts, codec, criterion, cfg, device)
        loss.backward()
        optimizer.ascend_step()
        compute_loss(model, images, texts, codec, criterion, cfg, device).backward()
        optimizer.descend_step()
        ema.update(model, step=nb_iter // 2)
        running_loss += loss.item()

        if nb_iter % cfg.print_every_iters == 0:
            last_avg_loss = running_loss / cfg.print_every_iters
            elapsed = time.time() - window_start
            print(f"iter {nb_iter}/{cfg.total_iters} lr={current_lr:.6f} loss={last_avg_loss:.4f} ({elapsed:.0f}s)")
            running_loss = 0.0
            window_start = time.time()

        if nb_iter % cfg.eval_every_iters == 0 or nb_iter == cfg.total_iters:
            val_result = run_validation(ema.ema, codec, val_df, val_loader, device)
            print(
                f"[iter {nb_iter}] val_cer={val_result['cer']:.4f} val_wer={val_result['wer']:.4f} "
                f"val_final={val_result['final']:.4f}"
            )
            logger.log_metrics(
                step=nb_iter,
                train_loss=last_avg_loss,
                val_cer=val_result["cer"],
                val_wer=val_result["wer"],
                val_final=val_result["final"],
            )
            if val_result["final"] > best_final:
                best_final = val_result["final"]
                torch.save(
                    {"model": model.state_dict(), "ema": ema.ema.state_dict()},
                    os.path.join(best_ckpt_dir, "best.pth"),
                )
                print(f"  new best (val_final={best_final:.4f}), saved to {best_ckpt_dir}")

    print(f"\nDone. Best val_final={best_final:.4f}, checkpoint at {best_ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/htrvt_fold0.yaml")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, config_cls=HTRVTConfig)
    train(cfg, max_train_samples=args.max_train_samples, max_val_samples=args.max_val_samples)
