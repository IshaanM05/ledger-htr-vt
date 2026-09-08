import torch


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2020) -- independent
    implementation of the published two-step algorithm: perturb weights
    toward the steepest local ascent direction (scaled by rho), compute
    loss/gradients again at that perturbed point, then restore the original
    weights and apply the wrapped optimizer's update using that gradient.
    Not a port of any specific repo's SAM code -- the algorithm itself is
    the published method, not any one implementation's expression of it."""

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **base_kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def ascend_step(self) -> None:
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["_pre_ascent"] = p.data.clone()
                p.add_(p.grad * scale)
        self.zero_grad()

    @torch.no_grad()
    def descend_step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.state[p]["_pre_ascent"])
        self.base_optimizer.step()
        self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        device = self.param_groups[0]["params"][0].device
        grads = [p.grad.norm(2).to(device) for group in self.param_groups for p in group["params"] if p.grad is not None]
        return torch.norm(torch.stack(grads), 2)
