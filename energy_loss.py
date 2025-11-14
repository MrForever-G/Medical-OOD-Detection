# energy_loss.py
import torch
import torch.nn.functional as F


def energy_from_logits(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    return -T * torch.logsumexp(logits / T, dim=1)


def loss_lid(E_li: torch.Tensor, mID: float) -> torch.Tensor:
    return torch.relu(E_li - mID).pow(2).mean()


def confidence_penalty(logits: torch.Tensor, lam: float = 0.0) -> torch.Tensor:
    if lam <= 0:
        return logits.new_tensor(0.0)
    p = logits.softmax(dim=1).clamp_min(1e-8)
    H = -(p * p.log()).sum(dim=1).mean()
    return lam * (-H)  # 惩罚过于自信


def loss_ood_huber(E_oe: torch.Tensor, mOOD: float, delta: float = 2.0) -> torch.Tensor:
    d = torch.relu(mOOD - E_oe)  # 惩罚项
    small = 0.5 * torch.minimum(d, torch.tensor(delta, device=d.device)).pow(2)
    large = delta * (d - delta / 2)
    return torch.where(d <= delta, small, large).mean()


def loss_rel_margin(
    E_ood: torch.Tensor, E_id_ref: torch.Tensor, m_gap: float = 2.0
) -> torch.Tensor:
    """
    相对间隔：强制 E_ood ≥ E_id_ref + m_gap
    E_id_ref: 建议传入当前 batch 的 ID 能量均值/中位数，且应为 detached 标量
    """
    if E_id_ref.dim() > 0:
        E_id_ref = E_id_ref.mean()
    return torch.relu(m_gap + E_id_ref - E_ood).mean()
