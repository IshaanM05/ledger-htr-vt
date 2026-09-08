import math


def cosine_lr_with_warmup(step: int, warmup_steps: int, total_steps: int, max_lr: float, min_lr: float = 1e-7) -> float:
    """Linear warmup then cosine decay -- standard schedule, independent implementation."""
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
