import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ledger_htr.config import HTRVTPretrainConfig, load_config, save_config
from ledger_htr.data.pretrain_dataset import MaskedImageDataset, all_image_ids, mask_image_strips
from ledger_htr.models.htr_vt import HTRMaskedAutoencoder
from ledger_htr.utils.logging import RunLogger
from ledger_htr.utils.lr_schedule import cosine_lr_with_warmup
from ledger_htr.utils.seed import set_seed


def masked_recon_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE restricted to the masked pixels only (MAE's core principle --
    supervising visible/unmasked pixels would let the model trivially copy
    its own input there instead of learning to infer content)."""
    return ((recon - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)


@torch.no_grad()
def run_validation(model, val_loader, device, cfg: HTRVTPretrainConfig, autocast_enabled: bool) -> float:
    model.eval()
    loss_sum, mask_sum = 0.0, 0.0
    for images in val_loader:
        images = images.to(device)
        masked, mask = mask_image_strips(images, cfg.mask_ratio, cfg.num_mask_strips)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
            recon = model(masked)
        recon = recon.float()
        loss_sum += ((recon - images) ** 2 * mask).sum().item()
        mask_sum += mask.sum().item()
    model.train()
    return loss_sum / max(mask_sum, 1.0)


def train(cfg: HTRVTPretrainConfig, max_samples: int | None = None) -> None:
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_enabled = cfg.use_bf16 and device == "cuda"
    print(f"device: {device}, bf16 autocast: {autocast_enabled}")

    ids = all_image_ids(cfg.data_dir)
    rng = np.random.RandomState(cfg.seed)
    rng.shuffle(ids)
    if max_samples:
        ids = ids[:max_samples]
    n_val = max(int(len(ids) * cfg.val_fraction), 1)
    val_ids, train_ids = ids[:n_val], ids[n_val:]
    print(f"pretrain images: train={len(train_ids)}, val={len(val_ids)} (train+test crops combined, no labels)")

    train_ds = MaskedImageDataset(train_ids, cfg.image_dir, cfg.target_height, cfg.max_width)
    val_ds = MaskedImageDataset(val_ids, cfg.image_dir, cfg.target_height, cfg.max_width)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )

    def cycle(loader):
        while True:
            yield from loader

    train_iter = cycle(train_loader)

    model = HTRMaskedAutoencoder(target_height=cfg.target_height).to(device)
    total_param = sum(p.numel() for p in model.parameters())
    print(f"total params: {total_param:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.max_lr, weight_decay=0.05)

    logger = RunLogger(cfg.runs_dir, cfg.run_name)
    logger.log_config(cfg)
    save_config(cfg, os.path.join(logger.run_dir, "resolved_config.yaml"))

    best_val_loss = float("inf")
    best_ckpt_dir = os.path.join(cfg.checkpoint_dir, cfg.run_name, "best")
    os.makedirs(best_ckpt_dir, exist_ok=True)

    running_loss = 0.0
    last_avg_loss = None
    window_start = time.time()
    for nb_iter in range(1, cfg.total_iters + 1):
        current_lr = cosine_lr_with_warmup(nb_iter, cfg.warmup_iters, cfg.total_iters, cfg.max_lr)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        images = next(train_iter).to(device)
        masked, mask = mask_image_strips(images, cfg.mask_ratio, cfg.num_mask_strips)

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled):
            recon = model(masked)
        loss = masked_recon_loss(recon.float(), images, mask)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if nb_iter % cfg.print_every_iters == 0:
            last_avg_loss = running_loss / cfg.print_every_iters
            elapsed = time.time() - window_start
            print(f"iter {nb_iter}/{cfg.total_iters} lr={current_lr:.6f} recon_loss={last_avg_loss:.5f} ({elapsed:.0f}s)")
            running_loss = 0.0
            window_start = time.time()

        if nb_iter % cfg.eval_every_iters == 0 or nb_iter == cfg.total_iters:
            val_loss = run_validation(model, val_loader, device, cfg, autocast_enabled)
            print(f"[iter {nb_iter}] val_recon_loss={val_loss:.5f}")
            logger.log_metrics(step=nb_iter, train_recon_loss=last_avg_loss, val_recon_loss=val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict()}, os.path.join(best_ckpt_dir, "best.pth"))
                print(f"  new best (val_recon_loss={best_val_loss:.5f}), saved to {best_ckpt_dir}")

    print(f"\nDone. Best val_recon_loss={best_val_loss:.5f}, checkpoint at {best_ckpt_dir}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/htrvt_pretrain.yaml")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, config_cls=HTRVTPretrainConfig)
    train(cfg, max_samples=args.max_samples)
