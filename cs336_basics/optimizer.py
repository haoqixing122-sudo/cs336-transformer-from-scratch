import math
from collections.abc import Iterable
import torch
from torch.optim import Optimizer


class Adamw(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                avg = state["exp_avg"]
                sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # Decay the old parameter, then apply the adaptive gradient update.
                p.mul_(1 - lr * wd)
                denom = sq.sqrt().div_(math.sqrt(bias2)).add_(eps)
                p.addcdiv_(avg, denom, value=-lr / bias1)
        return None


def clip_gradient_norm(parameters: Iterable[torch.nn.Parameter], max_norm: float):
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm = sum(p.grad.detach().float().norm(2).item() ** 2
                     for p in params_with_grad) ** 0.5
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params_with_grad:
            p.grad.mul_(scale)
