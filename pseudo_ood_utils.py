# pseudo_ood_utils.py
# 该文档保留了多种图像增强与扰动方法，用于生成伪OOD样本，以提升模型的鲁棒性和泛化能力
# 最后选取了最好的一种用到train.py中
from __future__ import annotations
import torch
import torch.nn.functional as F
import random
import math
from typing import Optional, Tuple, Dict


# 基础扰动/增强（
@torch.no_grad()
def _gaussian_noise(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    if sigma <= 0:
        return x
    n = torch.randn_like(x) * sigma
    return (x + n).clamp_(-5.0, 5.0)


@torch.no_grad()
def _gaussian_blur(
    x: torch.Tensor, k: int = 3, sigma: float | None = None
) -> torch.Tensor:
    if k <= 1:
        return x
    if sigma is None:
        sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    r = k // 2
    coords = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k1 = torch.exp(-(coords**2) / (2 * sigma * sigma))
    k1 = k1 / k1.sum()
    k2 = torch.outer(k1, k1)
    k2 = k2 / k2.sum()
    C = x.size(1)
    w = k2.expand(C, 1, k, k).contiguous()
    xpad = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(xpad, w, stride=1, padding=0, groups=C)


@torch.no_grad()
def _contrast(x: torch.Tensor, alpha: float = 0.9, beta: float = 0.0) -> torch.Tensor:
    m = x.mean(dim=(2, 3), keepdim=True)
    return (alpha * (x - m) + m + beta).clamp_(-5.0, 5.0)


@torch.no_grad()
def _grayscale(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    if random.random() > p:
        return x
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return torch.cat([y, y, y], dim=1)


@torch.no_grad()
def _random_erasing(
    x: torch.Tensor,
    p: float = 0.5,
    s: tuple[float, float] = (0.02, 0.15),
    r: tuple[float, float] = (0.3, 3.3),
) -> torch.Tensor:
    if random.random() > p:
        return x
    B, C, H, W = x.shape
    out = x.clone()
    area = H * W
    for b in range(B):
        S = random.uniform(*s) * area
        ar = random.uniform(*r)
        h = int(round(math.sqrt(S * ar)))
        w = int(round(math.sqrt(S / ar)))
        if h < 1 or w < 1 or h >= H or w >= W:
            continue
        y = random.randrange(0, H - h + 1)
        x0 = random.randrange(0, W - w + 1)
        val = torch.randn((C, 1, 1), device=x.device, dtype=x.dtype) * 0.5
        out[b, :, y : y + h, x0 : x0 + w] = val
    return out


@torch.no_grad()
def _randconv(x: torch.Tensor, k: int | None = None) -> torch.Tensor:
    B, C, H, W = x.shape
    if k is None:
        k = random.choice([3, 5, 7])
    r = k // 2
    w = torch.randn(C, C, k, k, device=x.device, dtype=x.dtype)
    w = w / (w.view(C, -1).norm(dim=1).view(C, 1, 1, 1) + 1e-6)
    xpad = F.pad(x, (r, r, r, r), mode="reflect")
    return F.conv2d(xpad, w, padding=0)


@torch.no_grad()
def _jigsaw(x: torch.Tensor, grid: int = 3, swap_ratio: float = 0.7) -> torch.Tensor:
    B, C, H, W = x.shape
    out = torch.empty_like(x)

    def splits(L: int, g: int):
        q, r = divmod(L, g)
        sizes = [q + 1] * r + [q] * (g - r)
        bounds = [0]
        for s in sizes:
            bounds.append(bounds[-1] + s)
        return sizes, bounds

    _, hb = splits(H, grid)
    _, wb = splits(W, grid)

    def part_perm(n: int, ratio: float):
        if n <= 1 or ratio <= 0:
            return list(range(n))
        k = max(2, int(round(n * ratio)))
        idx = list(range(n))
        random.shuffle(idx)
        sel = idx[:k]
        perm = list(range(n))
        sh = sel[:]
        random.shuffle(sh)
        for i, j in zip(sel, sh):
            perm[i] = j
        return perm

    colp = part_perm(grid, swap_ratio)
    rowp = part_perm(grid, swap_ratio)

    for b in range(B):
        blocks = [[None] * grid for _ in range(grid)]
        for gi in range(grid):
            y1, y2 = hb[gi], hb[gi + 1]
            for gj in range(grid):
                x1, x2 = wb[gj], wb[gj + 1]
                blocks[gi][gj] = x[b : b + 1, :, y1:y2, x1:x2]
        after_col = []
        for gi in range(grid):
            row = [blocks[gi][colp[gj]] for gj in range(grid)]
            after_col.append(row)
        after_row = [after_col[rowp[gi]] for gi in range(grid)]
        rows = [torch.cat(after_row[gi], dim=3) for gi in range(grid)]
        out[b : b + 1] = torch.cat(rows, dim=2)
    return out


@torch.no_grad()
def _apply_op_once(t: torch.Tensor) -> torch.Tensor:
    ops = [
        lambda u: _gaussian_blur(u, k=random.choice([9, 11, 13, 15, 17, 19])),
        lambda u: _randconv(u, k=random.choice([3, 5, 7])),
        lambda u: _contrast(
            u, alpha=0.8 + 0.6 * random.random(), beta=0.2 * (random.random() - 0.5)
        ),
        lambda u: _grayscale(u, p=1.0),
        lambda u: _random_erasing(u, p=1.0),
        lambda u: _jigsaw(u, grid=3, swap_ratio=0.7),
    ]
    return random.choice(ops)(t)


@torch.no_grad()
def _augmix_once(
    x: torch.Tensor, width: int = 4, depth: int = 3, m: float = 0.7
) -> torch.Tensor:
    B = x.size(0)
    mixes = []
    for _ in range(width):
        t = x.clone()
        L = depth if depth > 0 else (1 + int(random.random() * 3))
        for __ in range(L):
            t = _apply_op_once(t)
        mixes.append(t)
    w = torch.distributions.Dirichlet(
        torch.ones(width, device=x.device, dtype=x.dtype)
    ).sample((B,))
    w = w.view(B, width, 1, 1, 1)
    y = sum(w[:, i] * mixes[i] for i in range(width))
    lam = torch.rand(B, 1, 1, 1, device=x.device, dtype=x.dtype) * 0.2
    return (1.0 - lam) * y + lam * x


@torch.no_grad()
def _mixup_interclass(x: torch.Tensor, alpha: float = 0.3) -> torch.Tensor:
    B = x.size(0)
    if B == 1:
        return x
    idx = torch.randperm(B, device=x.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(x.device, x.dtype)
    lam = lam.clamp_(0.3, 0.7).view(B, 1, 1, 1)
    return lam * x + (1.0 - lam) * x[idx]


@torch.no_grad()
def _cutmix_interclass(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    B, C, H, W = x.shape
    if B == 1:
        return x
    idx = torch.randperm(B, device=x.device)
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(x.device, x.dtype)
    lam = lam.clamp_(0.3, 0.7)
    rx = (W * torch.sqrt(1 - lam)).long()
    ry = (H * torch.sqrt(1 - lam)).long()
    out = x.clone()
    for b in range(B):
        cx, cy = random.randrange(W), random.randrange(H)
        x1 = max(cx - rx[b].item() // 2, 0)
        x2 = min(cx + rx[b].item() // 2, W)
        y1 = max(cy - ry[b].item() // 2, 0)
        y2 = min(cy + ry[b].item() // 2, H)
        if x2 > x1 and y2 > y1:
            out[b, :, y1:y2, x1:x2] = x[idx[b], :, y1:y2, x1:x2]
    return out


@torch.no_grad()
def make_pseudo_ood(x: torch.Tensor, mode: str = "kys") -> torch.Tensor:
    if mode == "kys":
        return _augmix_once(x)
    elif mode == "mixup":
        return _mixup_interclass(x, alpha=0.3)
    elif mode == "cutmix":
        return _cutmix_interclass(x, alpha=1.0)
    elif mode == "weak":
        return _gaussian_blur(_gaussian_noise(x, 0.02), k=3)
    elif mode == "auto":
        p = random.random()
        if p < 0.45:
            return _augmix_once(x)
        elif p < 0.75:
            return _mixup_interclass(x, 0.3)
        elif p < 0.95:
            return _cutmix_interclass(x, 1.0)
        else:
            return _gaussian_blur(_gaussian_noise(x, 0.02), k=3)
    else:
        return _augmix_once(x)


@torch.no_grad()
def _decide_recipe(epoch: int, warm_end: int, num_epochs: int) -> Dict:
    """
    返回本 epoch 的配比与强度（A/B/C 三阶段）
    """
    if epoch < warm_end + 3:  # 阶段A：边界成形（warmup后前2~3轮）
        return dict(
            ratios=dict(t1=0.70, t2=0.20, t0=0.10, t3=0.00),
            t1_strength="mid",
            t2_cross_real=False,
        )
    elif epoch < warm_end + 9:  # 阶段B：多样化扩展
        return dict(
            ratios=dict(t1=0.50, t2=0.35, t0=0.15, t3=0.00),
            t1_strength="mid+",
            t2_cross_real=False,
        )
    else:  # 阶段C：巩固与校准
        return dict(
            ratios=dict(t1=0.40, t2=0.30, t0=0.30, t3=0.10),
            t1_strength="mid",
            t2_cross_real=True,
        )


def _t1_augment_once(x_id: torch.Tensor, strength: str) -> torch.Tensor:
    """
    Tier-1：对 ID 批做“单一强扰动”，强度随阶段变化（不叠加多种强扰动）
    """
    if strength == "mid":
        # 适中：从 AugMix/RandConv/Blur/Jigsaw/Erasing/Gray 中选一个
        ops = [
            lambda u: _augmix_once(u, width=3, depth=2),
            lambda u: _randconv(u, k=random.choice([3, 5, 7])),
            lambda u: _gaussian_blur(u, k=random.choice([9, 11, 13])),
            lambda u: _jigsaw(u, grid=3, swap_ratio=0.6),
            lambda u: _random_erasing(u, p=1.0, s=(0.02, 0.10)),
            lambda u: _grayscale(u, p=1.0),
        ]
        return random.choice(ops)(x_id)
    else:
        # mid+：略加强度（AugMix depth ↑ / Erasing 面积略增）
        ops = [
            lambda u: _augmix_once(u, width=4, depth=3),
            lambda u: _randconv(u, k=random.choice([5, 7])),
            lambda u: _gaussian_blur(u, k=random.choice([11, 13, 15])),
            lambda u: _jigsaw(u, grid=3, swap_ratio=0.7),
            lambda u: _random_erasing(u, p=1.0, s=(0.03, 0.15)),
            lambda u: _grayscale(u, p=1.0),
        ]
        return random.choice(ops)(x_id)


@torch.no_grad()
def _mix_id_id(x_id: torch.Tensor) -> torch.Tensor:
    # 边界混合：ID↔ID，只用于 OOD 分支
    if random.random() < 0.5:
        return _mixup_interclass(x_id, alpha=0.4)
    else:
        return _cutmix_interclass(x_id, alpha=1.0)


@torch.no_grad()
def _mix_id_real(x_id: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
    # 边界混合：ID↔真 OOD（把 51 张“稀释放大”）
    B = min(x_id.size(0), x_real.size(0))
    if B <= 0:
        return None
    xi = x_id[:B]
    xr = x_real[:B]
    # 使用 mixup，权重 0.5±0.1；也可按需切换 cutmix
    lam = (
        0.5 + 0.2 * (torch.rand(B, 1, 1, 1, device=xi.device, dtype=xi.dtype) - 0.5)
    ).clamp_(0.3, 0.7)
    return lam * xi + (1.0 - lam) * xr


@torch.no_grad()
def compose_tiered_ood(
    x_id_batch: torch.Tensor,
    real_ood_iter,  # 可能为 None
    epoch: int,
    cfg,  # 需用到 num_epochs / warmup_head_epochs / p_real_ood / batch_size
    device: torch.device,
    # Tier-3（特征域）可选：由 train.py 传入 callable，实现解耦
    sample_latent_fn: Optional[callable] = None,  # (targets, scale)-> Tensor(feat)
    head_forward_fn: Optional[callable] = None,  # (feats)-> logits
    targets: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Dict,
]:
    """
    返回：X_t1, X_t2, logits_t3, X_t0, meta
      - X_t1: 像素域近 OOD（由 ID 批增强）
      - X_t2: 边界混合 OOD（ID↔ID 以及（可选）ID↔真 OOD）
      - logits_t3: 特征域近 OOD（直接 logits；无需过 backbone/BN）
      - X_t0: 真 OOD（原图/轻中度变换由 real_ood_iter 内部决定）
      - meta: 记录本轮配比/采样张数，供日志使用
    """
    B = x_id_batch.size(0)
    recp = _decide_recipe(
        epoch=epoch, warm_end=cfg.warmup_head_epochs, num_epochs=cfg.num_epochs
    )
    r = recp["ratios"]
    strength = recp["t1_strength"]
    cross_real = recp["t2_cross_real"]

    # 目标张数（相对 ID 批，取整）
    n_ood_total = max(1, int(round(B * 0.8)))  # OOD:ID ≈ 0.8:1（可按需调整）
    n_t1 = int(round(n_ood_total * r["t1"]))
    n_t2 = int(round(n_ood_total * r["t2"]))
    n_t0 = int(round(n_ood_total * r["t0"]))
    n_t3 = int(round(n_ood_total * r["t3"]))

    X_t0 = None
    if (real_ood_iter is not None) and (n_t0 > 0):
        xr, _ = next(real_ood_iter)  # 已在 real_ood_iter 内做了标准化管线
        xr = xr.to(device, non_blocking=True)
        if xr.size(0) >= n_t0:
            X_t0 = xr[:n_t0]
        else:
            # 不够则循环取/重复补齐
            reps = (n_t0 + xr.size(0) - 1) // xr.size(0)
            X_t0 = xr.repeat(reps, 1, 1, 1)[:n_t0]
    X_t1 = None
    if n_t1 > 0:
        # 取 ID 子集并扰动（不叠加多种强扰动）
        idx = torch.randperm(B, device=x_id_batch.device)[:n_t1]
        X_t1 = _t1_augment_once(x_id_batch[idx], strength=strength)
    X_t2 = None
    if n_t2 > 0:
        half = n_t2 // 2
        # a) ID↔ID
        idx = torch.randperm(B, device=x_id_batch.device)[: max(1, half)]
        x_id_a = x_id_batch[idx]
        X_a = _mix_id_id(x_id_a)
        # b) ID↔真 OOD（可选，需有 X_t0）
        X_b = None
        if cross_real and (X_t0 is not None):
            # 用另一组 ID 子集，与 X_t0 取相同张数
            m = n_t2 - X_a.size(0)
            if m > 0:
                idx2 = torch.randperm(B, device=x_id_batch.device)[: min(B, m)]
                X_b = _mix_id_real(x_id_batch[idx2], X_t0[: min(X_t0.size(0), m)])
        # 汇总
        parts = [p for p in [X_a, X_b] if (p is not None and p.numel() > 0)]
        if len(parts) > 0:
            X_t2 = torch.cat(parts, dim=0)
            if X_t2.size(0) > n_t2:
                X_t2 = X_t2[:n_t2]
        else:
            X_t2 = None

    logits_t3 = None
    if (
        (n_t3 > 0)
        and (sample_latent_fn is not None)
        and (head_forward_fn is not None)
        and (targets is not None)
    ):
        # 取与 targets 等量或 n_t3 限额的 latent（使用你已有的 ClassStats.sample_latent_inliers）
        m = min(n_t3, targets.size(0))
        idx = torch.randperm(targets.size(0), device=targets.device)[:m]
        feats = sample_latent_fn(
            targets[idx], scale=0.5
        )  # 轻量插值/扰动在 sample_latent_fn 内部或上层实现
        logits_t3 = head_forward_fn(
            feats
        )  # 只过线性头获得 logits，外部会用 eval() 包裹避免 BN 污染

    meta = dict(
        plan=dict(
            epoch=epoch, ratios=r, t1_strength=strength, t2_cross_real=cross_real
        ),
        sizes=dict(
            t1=0 if X_t1 is None else X_t1.size(0),
            t2=0 if X_t2 is None else X_t2.size(0),
            t0=0 if X_t0 is None else X_t0.size(0),
            t3=0 if logits_t3 is None else logits_t3.size(0),
        ),
        total=n_ood_total,
    )
    return X_t1, X_t2, logits_t3, X_t0, meta
