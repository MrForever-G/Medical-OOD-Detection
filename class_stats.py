# class_stats.py
import torch


class ClassStats:
    def __init__(
        self,
        num_classes: int,
        feat_dim: int,
        device,
        ema: float = 0.99,
        eps: float = 1e-4,
    ):
        self.mu = torch.zeros(num_classes, feat_dim, device=device)
        self.var = torch.ones(num_classes, feat_dim, device=device)
        self.cnt = torch.zeros(num_classes, device=device)
        self.ema = ema
        self.eps = eps

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor):
        for k in labels.unique():
            idx = labels == k
            if idx.any():
                f = feats[idx].mean(dim=0)
                v = feats[idx].var(dim=0, unbiased=False)
                self.mu[k] = self.ema * self.mu[k] + (1 - self.ema) * f
                self.var[k] = self.ema * self.var[k] + (1 - self.ema) * v
                self.cnt[k] += idx.sum()

    @torch.no_grad()
    def sample_latent_inliers(
        self, labels: torch.Tensor, scale: float = 0.5
    ) -> torch.Tensor:
        B = labels.size(0)
        D = self.mu.size(1)
        tk = torch.empty(B, D, device=self.mu.device)
        for i, y in enumerate(labels):
            std = self.var[y].clamp_min(self.eps).sqrt() * scale
            tk[i] = self.mu[y] + std * torch.randn(D, device=self.mu.device)
        return tk
