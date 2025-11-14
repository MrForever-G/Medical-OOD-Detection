# train.py
import os, json, datetime, logging
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from ramp import build_cosine_ramp
from utils import get_measures
from class_stats import ClassStats
from pseudo_ood_utils import compose_tiered_ood
from energy_loss import (
    energy_from_logits,
    loss_lid,
    confidence_penalty,
    loss_ood_huber,
    loss_rel_margin,
)
from dataclasses import asdict
from real_ood import build_real_ood_iter


@dataclass
# 初预设值 参数修改现已转到 run.py
class TrainConfig:
    # 必填（run.py 每次都会显式传入）
    data_dir: str
    num_classes: int
    num_epochs: int
    batch_size: int
    learning_rate: float
    device: str
    log_root: str

    # 训练/评估与损失的可选超参（提供合理默认，run.py 可覆盖）
    T: float = 1.2
    alpha: float = 0.04
    beta: float = 0.03
    mID: float = -8.0
    mOOD: float = -4.0
    lambda_conf: float = 0.0
    pseudo_ood_mode: str = "kys"
    warmup_head_epochs: int = 3

    # 训练流程与稳定性
    workers: int = 0
    label_smoothing: float = 0.05
    lr_warmup_epochs: int = 2
    lr_min_factor: float = 0.1
    max_grad_norm: float = 5.0
    detach_lid: bool = True

    # 真实 OOD
    use_real_ood: bool = True
    real_ood_dir: str = "./datasets/extract_DDI"
    p_real_ood: float = 0.25
    beta_real: float = 0.05

    # 头部融合与评估温度
    gamma_ood_head: float = 0.3
    T_eval: float = 1.2
    fpr_select: str = "fpr95"  # 'fpr95' | 'fpr92' | 'fpr90'：用于保存“最低FPR”的指标


def log_hparams(cfg, log_dir):
    d = asdict(cfg)
    keep = [
        "data_dir",
        "num_classes",
        "num_epochs",
        "batch_size",
        "learning_rate",
        "T",
        "alpha",
        "beta",
        "mID",
        "mOOD",
        "lambda_conf",
        "pseudo_ood_mode",
        "warmup_head_epochs",
        "lr_warmup_epochs",
        "lr_min_factor",
        "max_grad_norm",
        "detach_lid",
        "workers",
        "label_smoothing",
        "device",
        "use_real_ood",
        "real_ood_dir",
        "p_real_ood",
        "beta_real",
        "gamma_ood_head",
        "T_eval",
        "log_root",
    ]
    txt = " | ".join([f"{k}={d[k]}" for k in keep if k in d])
    logging.info("HParams: " + txt)
    with open(os.path.join(log_dir, "hparams.json"), "w", encoding="utf-8") as f:
        json.dump({k: d[k] for k in keep if k in d}, f, indent=2, ensure_ascii=False)


def setup_logger(log_root: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(log_root, ts)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{ts}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger().info(f"Log directory: {log_dir}")
    return log_dir


def build_dataloaders(cfg: TrainConfig):
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_ds = datasets.ImageFolder(
        os.path.join(cfg.data_dir, "train"), transform=train_tf
    )
    id_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "test"), transform=val_tf)
    ood_ds = datasets.ImageFolder(os.path.join(cfg.data_dir, "ood"), transform=val_tf)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers
    )
    id_loader = DataLoader(
        id_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers
    )
    ood_loader = DataLoader(
        ood_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers
    )
    return train_loader, id_loader, ood_loader


def build_model(cfg: TrainConfig, device):
    model = models.resnet50(weights="IMAGENET1K_V1")
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        feat_dim = model.head.in_features
        model.head = nn.Identity()
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        feat_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    else:
        raise RuntimeError("未找到分类头，请手动适配")

    classifier = nn.Linear(feat_dim, cfg.num_classes)
    ood_head = nn.Linear(feat_dim, 1)

    model.to(device)
    classifier.to(device)
    ood_head.to(device)
    return model, classifier, ood_head, feat_dim


def forward_logits(model, classifier, x: torch.Tensor):
    feats = model(x)
    if feats.dim() > 2:
        feats = torch.flatten(feats, 1)
    logits = classifier(feats)
    return logits, feats


def evaluate_topk(model, classifier, loader, device, ks=(1, 5)):
    model.eval()
    classifier.eval()
    top1 = top5 = tot = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x, y = x.to(device), y.to(device)
            logits, _ = forward_logits(model, classifier, x)
            _, pred = logits.topk(max(ks), 1, True, True)
            pred = pred.t()
            corr = pred.eq(y.view(1, -1).expand_as(pred))
            top1 += corr[:1].reshape(-1).float().sum().item()
            top5 += corr[:5].reshape(-1).float().sum().item()
            tot += y.size(0)
    return 100 * top1 / tot, 100 * top5 / tot


def evaluate_ood(
    model, classifier, id_loader, ood_loader, device, T_eval: float, ood_head=None
):
    model.eval()
    classifier.eval()
    logging.info(f"[Eval ] Using T_eval={T_eval}")
    if ood_head is not None:
        ood_head.eval()

    def run(loader):
        scores = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits, feats = forward_logits(model, classifier, x)
                # 先算“负能量”（训练同款），再取反得到“ID 分数”
                E_neg = energy_from_logits(logits, T_eval)  # = -T * logsumexp(...)
                E_id = -E_neg  # = +T * logsumexp(...)
                maxlog = logits.max(dim=1).values
                s = 0.7 * E_id + 0.3 * maxlog
                if ood_head is not None:
                    s_head = ood_head(feats).squeeze(1)
                    s = 0.5 * s + 0.5 * s_head
                scores.append(s.cpu().numpy())
        return np.concatenate(scores, 0)

    id_s, ood_s = run(id_loader), run(ood_loader)
    auroc, aupr, _ = get_measures(id_s, ood_s, recall_level=0.95)
    _, _, fpr95 = get_measures(id_s, ood_s, recall_level=0.95)
    _, _, fpr92 = get_measures(id_s, ood_s, recall_level=0.92)
    _, _, fpr90 = get_measures(id_s, ood_s, recall_level=0.90)
    return auroc, aupr, fpr95, fpr92, fpr90


def train_and_eval(cfg: TrainConfig):
    log_dir = setup_logger(cfg.log_root)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    log_hparams(cfg, log_dir)
    train_loader, id_loader, ood_loader = build_dataloaders(cfg)
    model, classifier, ood_head, feat_dim = build_model(cfg, device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # ===== Real OOD iterator (optional) =====
    real_ood_iter = None
    real_loader = None  # <- 先定义，避免后面未定义

    if cfg.use_real_ood and os.path.isdir(cfg.real_ood_dir):
        normalize_tf = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        real_bs = max(1, int(cfg.batch_size * cfg.p_real_ood))
        # 关键：把第二个返回值也接住（不是下划线丢弃）
        real_ood_iter, real_loader = build_real_ood_iter(
            real_ood_dir=cfg.real_ood_dir,
            batch_size=real_bs,
            num_workers=cfg.workers,
            normalize_transform=normalize_tf,
        )

    # 诊断日志（Enabled/Disabled）
    if (real_ood_iter is None) or (real_loader is None):
        logging.info(
            f"[Real OOD] Disabled: dir='{cfg.real_ood_dir}' 不存在/为空/非ImageFolder结构"
        )
    else:
        ds = real_loader.dataset
        num_imgs = len(ds.imgs) if hasattr(ds, "imgs") else len(ds)
        num_classes = len(getattr(ds, "classes", []))
        sample_path = (
            ds.imgs[0][0] if hasattr(ds, "imgs") and len(ds.imgs) > 0 else "N/A"
        )
        logging.info(
            f"[Real OOD] Enabled  | dir='{cfg.real_ood_dir}' | classes={num_classes} | "
            f"images={num_imgs} | batch_size={real_loader.batch_size} | sample0='{sample_path}'"
        )

    for p in model.parameters():
        p.requires_grad = False
    for p in classifier.parameters():
        p.requires_grad = True
    opt_h = torch.optim.Adam(classifier.parameters(), lr=5e-4, weight_decay=1e-5)

    best_warm_top1 = -1.0
    best_cls_state = None
    m_real_ema = None

    for e in range(cfg.warmup_head_epochs):
        model.eval()  # 冻结 BN 统计，避免飘
        classifier.train()

        for x, y in tqdm(
            train_loader, desc=f"Warmup {e+1}/{cfg.warmup_head_epochs}", leave=False
        ):
            x, y = x.to(device), y.to(device)
            opt_h.zero_grad()

            with torch.no_grad():  # backbone 前向不建图，省显存且更稳
                feats = model(x)
                if feats.dim() > 2:
                    feats = torch.flatten(feats, 1)

            logits = classifier(feats)  # 只训练分类头
            loss = criterion(logits, y)
            loss.backward()
            opt_h.step()

        with torch.no_grad():
            top1, _ = evaluate_topk(model, classifier, id_loader, device)
        logging.info(f"[Warmup {e+1}/{cfg.warmup_head_epochs}] Top-1={top1:.2f}%")

        # 记录并回载 warm-up 最优分类头
        if top1 > best_warm_top1:
            best_warm_top1 = top1
            best_cls_state = {
                k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()
            }

    if best_cls_state is not None:
        classifier.load_state_dict(best_cls_state, strict=True)
        logging.info(f"[Warmup] Restored best classifier (Top-1={best_warm_top1:.2f}%)")

    for p in model.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        list(model.parameters())
        + list(classifier.parameters())
        + list(ood_head.parameters()),
        lr=cfg.learning_rate,
        weight_decay=1e-5,
    )

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    sched_warm = LinearLR(optimizer, start_factor=0.2, total_iters=cfg.lr_warmup_epochs)
    sched_main = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.num_epochs - cfg.lr_warmup_epochs),
        eta_min=cfg.learning_rate * cfg.lr_min_factor,
    )
    scheduler = SequentialLR(
        optimizer, [sched_warm, sched_main], milestones=[cfg.lr_warmup_epochs]
    )

    class_stats = ClassStats(cfg.num_classes, feat_dim, device)
    best_top1, best_auroc = 0.0, -1.0
    save_ood = os.path.join(log_dir, "best_model_ood.pth")
    save_top1 = os.path.join(log_dir, "best_model_top1.pth")
    history = []
    best_fpr = float("inf")
    save_fpr = os.path.join(log_dir, f"best_model_{cfg.fpr_select}.pth")

    ramp_coef = build_cosine_ramp(
        cfg.num_epochs, start=0.15, end=0.55, c_min=0.10, c_max=0.30
    )

    logging.info("========== Hyper-Parameter & Ramp Preview ==========")
    logging.info("重要训练参数：")
    for k, v in {
        "T": cfg.T,
        "alpha (LID权重)": cfg.alpha,
        "beta (OOD权重)": cfg.beta,
        "mID": cfg.mID,
        "mOOD": cfg.mOOD,
        "warmup_head_epochs": cfg.warmup_head_epochs,
        "num_epochs": cfg.num_epochs,
        "pseudo_ood_mode": cfg.pseudo_ood_mode,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
    }.items():
        logging.info(f"{k:<20} : {v}")

    logging.info("\nRamp系数预览（前20轮）:")
    for ep in range(min(20, cfg.num_epochs)):
        logging.info(f"  Epoch {ep:>2d} → coef={ramp_coef(ep):.2f}")
    logging.info("====================================================")

    for epoch in range(cfg.num_epochs):
        coef = ramp_coef(epoch)
        logging.info(f"========== Epoch {epoch+1}/{cfg.num_epochs} ==========")

        meter_E_li_sum = 0.0
        meter_E_oe_sum = 0.0

        if epoch < cfg.warmup_head_epochs:
            use_ood = True
            use_lid = False
            eff_T = 1.2
        else:
            use_ood = True
            use_lid = True
            eff_T = 1.2

        # LID 的线性日程：从 warmup 结束开始，6 个 epoch 内从 0 → α
        alpha_eff = cfg.alpha * max(
            0.0, min(1.0, (epoch - cfg.warmup_head_epochs) / 6.0)
        )

        model.train()
        classifier.train()
        meters = {
            "ce": 0.0,
            "lid": 0.0,
            "lood": 0.0,
            "conf": 0.0,
            "lood_pseudo": 0.0,
            "lood_real": 0.0,
            "n": 0,
        }
        raw_meters = {
            "ce": 0.0,
            "lid": 0.0,
            "lood": 0.0,
            "conf": 0.0,
            "lood_pseudo": 0.0,
            "lood_real": 0.0,
            "n": 0,
        }

        for inputs, targets in tqdm(
            train_loader, desc=f"Training {epoch+1}/{cfg.num_epochs}", leave=False
        ):
            inputs, targets = inputs.to(device), targets.to(device)
            bs = inputs.size(0)
            optimizer.zero_grad()

            logits_id, feats = forward_logits(model, classifier, inputs)
            E_id_cur = energy_from_logits(logits_id, eff_T).detach()
            E_id_ref = E_id_cur.mean()

            class_stats.update(feats.detach(), targets)

            tk = class_stats.sample_latent_inliers(targets, scale=0.5)
            if cfg.detach_lid:
                tk = tk.detach()
            logits_li = classifier(tk)

            # E_li = energy_from_logits(logits_li, cfg.T)
            if use_lid:
                E_li = energy_from_logits(logits_li, eff_T)
                L_lid = loss_lid(E_li, cfg.mID)
            else:
                L_lid = torch.tensor(0.0, device=logits_id.device)

            if use_ood:

                def _sample_latent_fn(tgts, scale=0.5):
                    tk = class_stats.sample_latent_inliers(tgts, scale=scale)
                    return tk.detach() if cfg.detach_lid else tk

                def _head_forward_fn(feats):
                    if feats.dim() > 2:
                        feats = torch.flatten(feats, 1)
                    return classifier(feats)

                X_t1, X_t2, logits_t3, X_t0, meta_ood = compose_tiered_ood(
                    x_id_batch=inputs,
                    real_ood_iter=(
                        real_ood_iter if (epoch >= cfg.warmup_head_epochs) else None
                    ),
                    epoch=epoch,
                    cfg=cfg,
                    device=device,
                    sample_latent_fn=_sample_latent_fn,
                    head_forward_fn=_head_forward_fn,
                    targets=targets,
                )

                if (
                    (X_t0 is not None)
                    and (X_t0.numel() > 0)
                    and (epoch < cfg.warmup_head_epochs + 10)
                ):
                    max_t0 = max(
                        1, int(0.10 * inputs.size(0))
                    )  # 10% 上限；如需更保守可改 0.05
                    if X_t0.size(0) > max_t0:
                        X_t0 = X_t0[:max_t0]
                        if (
                            isinstance(meta_ood, dict)
                            and "sizes" in meta_ood
                            and "t0" in meta_ood["sizes"]
                        ):
                            meta_ood["sizes"]["t0"] = int(X_t0.size(0))

                L_parts = []

                if X_t1 is not None and X_t1.numel() > 0:
                    _wt = model.training
                    model.eval()
                    logits_t1, feats_t1 = forward_logits(model, classifier, X_t1)
                    model.train(_wt)
                    E_t1 = energy_from_logits(logits_t1, eff_T)
                    L_abs_t1 = loss_ood_huber(
                        torch.clamp(E_t1, -30.0, 10.0), cfg.mOOD, delta=2.0
                    )
                    L_rel_t1 = loss_rel_margin(E_t1, E_id_ref, m_gap=2.0)
                    L_energy_t1 = 0.5 * L_abs_t1 + 0.5 * L_rel_t1
                    s_t1 = ood_head(feats_t1).squeeze(1)
                    L_head_t1 = loss_ood_huber(
                        torch.clamp(s_t1, -30.0, 10.0), cfg.mOOD, delta=2.0
                    )
                    L_parts.append(
                        (1.0 - cfg.gamma_ood_head) * L_energy_t1
                        + cfg.gamma_ood_head * L_head_t1
                    )

                if X_t2 is not None and X_t2.numel() > 0:
                    _wt = model.training
                    model.eval()
                    logits_t2, feats_t2 = forward_logits(model, classifier, X_t2)
                    model.train(_wt)

                    E_t2 = energy_from_logits(logits_t2, eff_T)
                    L_abs_t2 = loss_ood_huber(
                        torch.clamp(E_t2, -30.0, 10.0), cfg.mOOD, delta=2.0
                    )
                    L_rel_t2 = loss_rel_margin(E_t2, E_id_ref, m_gap=2.0)
                    L_energy_t2 = 0.5 * L_abs_t2 + 0.5 * L_rel_t2

                    s_t2 = ood_head(feats_t2).squeeze(1)
                    L_head_t2 = loss_ood_huber(
                        torch.clamp(s_t2, -30.0, 10.0), cfg.mOOD, delta=2.0
                    )

                    L_parts.append(
                        (1.0 - cfg.gamma_ood_head) * L_energy_t2
                        + cfg.gamma_ood_head * L_head_t2
                    )

                if logits_t3 is not None and logits_t3.numel() > 0:
                    E_t3 = energy_from_logits(logits_t3, eff_T)
                    L_abs_t3 = loss_ood_huber(
                        torch.clamp(E_t3, -30.0, 10.0), cfg.mOOD, delta=2.0
                    )
                    L_rel_t3 = loss_rel_margin(E_t3, E_id_ref, m_gap=2.0)
                    L_parts.append(0.5 * L_abs_t3 + 0.5 * L_rel_t3)

                # 伪 OOD（t1+t2+t3）的合并损失
                if len(L_parts) > 0:
                    L_ood = sum(L_parts) / len(
                        L_parts
                    )  # 简单平均，保持权重由 beta*coef 控
                else:
                    L_ood = torch.tensor(0.0, device=inputs.device)

                # 真 OOD（t0）
                if (X_t0 is not None) and (X_t0.numel() > 0):
                    _wt = model.training
                    model.eval()
                    logits_t0, feats_t0 = forward_logits(model, classifier, X_t0)
                    model.train(_wt)
                    E_real = energy_from_logits(logits_t0, eff_T)
                    E_real_safe = torch.clamp(E_real, min=-30.0, max=10.0)
                    with torch.no_grad():
                        m_inst = (
                            torch.quantile(E_real.detach(), 0.65) + 0.15
                        )  # 分位与EMA按我们前面建议
                        m_real_ema = (
                            m_inst
                            if (m_real_ema is None)
                            else (0.95 * m_real_ema + 0.05 * m_inst)
                        )

                    L_energy_real_abs = loss_ood_huber(
                        E_real_safe, float(m_real_ema), delta=2.5
                    )
                    L_energy_real_rel = loss_rel_margin(
                        E_real_safe, E_id_ref, m_gap=2.0
                    )
                    L_energy_real = 0.5 * L_energy_real_abs + 0.5 * L_energy_real_rel

                    s_t0 = ood_head(feats_t0).squeeze(1)
                    L_head_real = loss_ood_huber(
                        torch.clamp(s_t0, -30.0, 10.0), float(m_real_ema), delta=2.5
                    )

                    L_ood_real = (
                        1.0 - cfg.gamma_ood_head
                    ) * L_energy_real + cfg.gamma_ood_head * L_head_real
                else:
                    L_ood_real = torch.tensor(0.0, device=inputs.device)
            else:
                L_ood = torch.tensor(0.0, device=inputs.device)
                L_ood_real = torch.tensor(0.0, device=inputs.device)

            _E_li_log = energy_from_logits(logits_li, eff_T).detach().mean()
            meter_E_li_sum += float(_E_li_log) * bs

            # OOD：把各来源的批均值按样本数加权
            if use_ood:
                sumE = 0.0
                sumN = 0
                if "X_t1" in locals() and X_t1 is not None and X_t1.numel() > 0:
                    sumE += float(E_t1.detach().mean()) * X_t1.size(0)
                    sumN += X_t1.size(0)
                if "X_t2" in locals() and X_t2 is not None and X_t2.numel() > 0:
                    sumE += float(E_t2.detach().mean()) * X_t2.size(0)
                    sumN += X_t2.size(0)
                if (
                    "logits_t3" in locals()
                    and logits_t3 is not None
                    and logits_t3.numel() > 0
                ):
                    sumE += float(E_t3.detach().mean()) * logits_t3.size(0)
                    sumN += logits_t3.size(0)
                if sumN > 0:
                    E_ood_batch_mean = sumE / max(1, sumN)
                    meter_E_oe_sum += E_ood_batch_mean * bs

            L_ce = criterion(logits_id, targets)
            L_conf = confidence_penalty(logits_id, lam=cfg.lambda_conf)
            w_ce = L_ce
            w_lid = alpha_eff * L_lid

            L_ood_raw = L_ood.detach()
            L_ood_real_raw = L_ood_real.detach()
            L_ood = torch.clamp(L_ood, max=5.0)  # 可在 3~8 之间微调
            L_ood_real = torch.clamp(L_ood_real, max=5.0)

            # 伪 OOD 只做“形状正则”：把有效β限制在不超过 0.08（可按需 0.05~0.08）
            beta_pseudo_eff = min(cfg.beta, 0.08)
            w_ood_pseudo = (beta_pseudo_eff * max(coef, 0.15)) * L_ood

            span = 8  # 可调：6~10
            prog = max(0.0, min(1.0, (epoch - cfg.warmup_head_epochs) / float(span)))
            coef_real = prog  # 0→1 的平滑系数

            w_ood_real = coef_real * (cfg.beta_real) * L_ood_real

            w_ood = w_ood_pseudo + w_ood_real
            w_conf = L_conf

            warm = cfg.warmup_head_epochs
            if epoch < warm + 1:
                max_ratio = 0.50  # 上OOD首轮：半数上限，防抖
            elif epoch < warm + 4:
                max_ratio = 0.60
            elif epoch < warm + 8:
                max_ratio = 0.70
            else:
                max_ratio = 1.00  # 训练后1/3彻底放开

            ood_cap = max_ratio * (
                w_ce.detach() + w_lid.detach() + w_conf.detach() + 1e-6
            )
            w_ood = torch.minimum(w_ood, ood_cap)

            L_total = w_ce + w_lid + w_conf + w_ood

            # 反传更新
            optimizer.zero_grad()
            L_total.backward()
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters())
                    + list(classifier.parameters())
                    + list(ood_head.parameters()),
                    max_norm=cfg.max_grad_norm,
                )
            optimizer.step()

            # 累计到 meters：加权项
            bs = inputs.size(0)
            meters["ce"] += float(w_ce) * bs
            meters["lid"] += float(w_lid) * bs
            meters["lood"] += float(w_ood) * bs
            meters["conf"] += float(w_conf) * bs
            meters["n"] += bs
            meters["lood_pseudo"] += float(w_ood_pseudo) * bs
            meters["lood_real"] += float(w_ood_real) * bs

            # 同时累计原始项（用于直观对照）
            raw_meters["ce"] += float(L_ce) * bs
            raw_meters["lid"] += float(L_lid) * bs
            raw_meters["lood"] += float(L_ood_raw) * bs
            raw_meters["conf"] += float(L_conf) * bs
            raw_meters["lood_pseudo"] += float(L_ood_raw) * bs
            raw_meters["lood_real"] += float(L_ood_real_raw) * bs
            raw_meters["n"] += bs

        # 按 epoch 求均值、评估、日志、保存、调度
        for k in ["ce", "lid", "lood", "conf", "lood_pseudo", "lood_real"]:
            meters[k] /= max(1, meters["n"])
            raw_meters[k] /= max(1, raw_meters["n"])

        top1, top5 = evaluate_topk(model, classifier, id_loader, device)
        auroc, aupr, fpr95, fpr92, fpr90 = evaluate_ood(
            model,
            classifier,
            id_loader,
            ood_loader,
            device,
            cfg.T_eval,
            ood_head=ood_head,
        )

        E_li_mean = meter_E_li_sum / max(1, meters["n"])
        E_oe_mean = meter_E_oe_sum / max(1, meters["n"])

        logging.info(
            f"[Energy] mean(E_id)={E_li_mean:.2f} (target mID={cfg.mID:.1f}) | "
            f"mean(E_ood)={E_oe_mean:.2f} (target mOOD={cfg.mOOD:.1f})"
        )

        logging.info(
            "[Loss(w)] L = CE + α·LID + (β·coef)·LOOD + Conf\n"
            f"          = {meters['ce']:.4f} + {meters['lid']:.4f} + {meters['lood']:.4f} + {meters['conf']:.4f} "
            f"= {meters['ce'] + meters['lid'] + meters['lood'] + meters['conf']:.4f} (coef={coef:.2f})"
        )
        logging.info(
            "[Loss(r)] raw = CE + LID + LOOD + Conf\n"
            f"             = {raw_meters['ce']:.4f} + {raw_meters['lid']:.4f} + "
            f"{raw_meters['lood']:.4f} + {raw_meters['conf']:.4f}"
        )
        logging.info(
            f"[OOD parts] (w) pseudo={meters['lood_pseudo']:.6f} | real={meters['lood_real']:.6f} "
            f"(raw pseudo={raw_meters['lood_pseudo']:.6f} | raw real={raw_meters['lood_real']:.6f})"
        )
        logging.info(
            f"[Val ] Top-1={top1:.2f}% | Top-5={top5:.2f}%\n"
            f"[OOD ] AUROC={auroc:.4f} | AUPR={aupr:.4f} | "
            f"FPR95={fpr95:.4f} | FPR92={fpr92:.4f} | FPR90={fpr90:.4f}"
        )

        history.append(
            {
                "epoch": epoch + 1,
                "CE_w": meters["ce"],
                "LID_w": meters["lid"],
                "LOOD_w": meters["lood"],
                "Conf_w": meters["conf"],
                "CE_r": raw_meters["ce"],
                "LID_r": raw_meters["lid"],
                "LOOD_r": raw_meters["lood"],
                "Conf_r": raw_meters["conf"],
                "Top1": top1,
                "Top5": top5,
                "AUROC": auroc,
                "AUPR": aupr,
                "FPR95": fpr95,
                "FPR92": fpr92,
                "FPR90": fpr90,
                "coef": coef,
            }
        )

        if use_ood and "meta_ood" in locals():
            sizes = meta_ood["sizes"]
            plan = meta_ood["plan"]
            logging.info(
                f"[Composer] plan={plan} | sizes={sizes} | total={meta_ood['total']}"
            )

        if auroc > best_auroc:
            best_auroc = auroc
            os.makedirs(os.path.dirname(save_ood), exist_ok=True)
            torch.save(
                {
                    "backbone": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "ood_head": ood_head.state_dict(),
                },
                save_ood,
            )
            logging.info(f"Saved OOD-best model (AUROC={auroc:.4f})")
        if top1 > best_top1:
            best_top1 = top1
            os.makedirs(os.path.dirname(save_top1), exist_ok=True)
            torch.save(
                {"backbone": model.state_dict(), "classifier": classifier.state_dict()},
                save_top1,
            )
            logging.info(f"Saved Top1-best model (Top-1={top1:.2f}%)")
        if cfg.fpr_select == "fpr95":
            cur_fpr = fpr95
        elif cfg.fpr_select == "fpr92":
            cur_fpr = fpr92
        else:
            cur_fpr = fpr90

        if cur_fpr < best_fpr:
            best_fpr = cur_fpr
            os.makedirs(os.path.dirname(save_fpr), exist_ok=True)
            torch.save(
                {
                    "backbone": model.state_dict(),
                    "classifier": classifier.state_dict(),
                    "ood_head": ood_head.state_dict(),
                },
                save_fpr,
            )
            logging.info(f"Saved FPR-best model ({cfg.fpr_select}={cur_fpr:.4f})")

        scheduler.step()

    with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logging.info(
        f"Training complete. Best AUROC={best_auroc:.4f} | Best Top-1={best_top1:.2f}%"
    )
    logging.info(f"Best {cfg.fpr_select}={best_fpr:.4f}")
    return {"log_dir": log_dir, "best_auroc": best_auroc, "best_top1": best_top1}
