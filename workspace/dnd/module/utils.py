import math

import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler


class Mean(nn.Module):
    def __init__(self, dim, keepdim=False):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        x = torch.mean(x, dim=self.dim, keepdim=self.keepdim)
        return x


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * (self.last_epoch + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:  # use cosine learning rate
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [base_lr * 0.5 * (1 + math.cos(math.pi * progress)) for base_lr in self.base_lrs]


class RandomGenerator(nn.Module):
    def __init__(self, features=None, intensity=1.0):
        super().__init__()
        self.features = features
        self.intensity = intensity

    def forward(self, x):
        if self.features is not None:
            return self.intensity * torch.randn(
                (x.size(0), *self.features), device=x.device, dtype=x.dtype
            )  # output noise
        else:  # self.features is None
            return self.intensity * torch.randn_like(x)
