import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python/numpy/torch RNGs. Call once at the start of every run."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
