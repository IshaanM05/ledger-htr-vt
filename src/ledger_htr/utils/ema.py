import copy

import torch


class EmaModel:
    """Exponential moving average of a model's weights, evaluated instead of
    the raw training weights -- standard technique, independent implementation
    (no dependency on any specific repo's EMA helper)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int | None = None) -> None:
        decay = self.decay
        if step is not None:
            decay = min(decay, (1 + step) / (10 + step))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            v.copy_(v * decay + (1 - decay) * msd[k].detach())
