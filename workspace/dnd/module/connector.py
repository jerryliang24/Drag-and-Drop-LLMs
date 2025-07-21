import torch
from torch import nn

from .hyperconv import cumprod


class KDEAnomalyDetectorMean(nn.Module):
    def __init__(self, threshold=2.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x):
        # reshape in
        assert x.dim() >= 3
        x_shape = x.shape
        x = x.flatten(start_dim=2)
        # KDE process
        eps = 1e-6
        dim = x.size(-1)
        B, N = x.size(0), x.size(1)
        sigma = x.std(dim=1, unbiased=False).mean(dim=1)
        sigma = sigma + eps
        h = (4 / (dim + 2)) ** (1 / (dim + 4)) * (N ** (-1 / (dim + 4))) * sigma + eps
        h = h.view(B, 1, 1)
        x1 = x.unsqueeze(2)
        x2 = x.unsqueeze(1)
        D = ((x1 - x2) ** 2).mean(-1)
        densities = torch.exp(-D / (2 * h**2 + eps)).sum(dim=2)
        mean = torch.mean(densities, dim=1, keepdim=True)
        std = torch.std(densities, dim=1, keepdim=True)
        threshold = mean - self.threshold * std
        mask = densities > threshold
        selected = [x[i, mask[i], :] for i in range(B)]
        means = torch.stack([s.mean(dim=0) if s.numel() > 0 else torch.zeros(dim).to(x.device) for s in selected])
        # reshape out
        x = torch.unflatten(means, 1, x_shape[2:])
        return x


class AttentionConnector(nn.Module):
    def __init__(self, feature: tuple, num_heads=32):
        super().__init__()
        self.q = nn.Parameter(torch.zeros((1, 1, cumprod(feature))))
        self.attention = nn.MultiheadAttention(
            embed_dim=cumprod(feature),
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x):
        # x.shape = (B, N, 384, 7, 7)
        x_shape = x.shape[-3:]
        kv = x.flatten(start_dim=2)  # (B, N, 18816)
        out = self.attention(self.q.repeat(x.shape[0], 1, 1), kv, kv)[0]
        out = torch.unflatten(out, -1, x_shape).squeeze(1)
        assert out.shape[0] == x.shape[0] and out.shape[1:] == x_shape
        return out
