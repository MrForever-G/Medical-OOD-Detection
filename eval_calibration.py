import os, argparse, json, math, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_measures


def build_model(num_classes: int, ckpt_path: str, device):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    sd = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def make_loader(root: str, split: str, bs: int, nw: int):
    ds_dir = os.path.join(root, split)
    if not os.path.isdir(ds_dir):
        return None, None
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(ds_dir, transform=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return ds, dl


@torch.no_grad()
def collect_logits(model, dl, device):
    all_logits, all_labels = [], []
    for x, y in tqdm(dl, desc="Collect logits"):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.clone())
    logits = torch.cat(all_logits, 0)
    labels = torch.cat(all_labels, 0) if len(all_labels) else None
    return logits, labels


def fit_temperature(
    logits_id: torch.Tensor, labels_id: torch.Tensor, T_init=1.0, max_iter=200
):
    device = logits_id.device
    log_T = torch.tensor(
        math.log(T_init), device=device, dtype=torch.float32, requires_grad=True
    )
    nll = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS(
        [log_T], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(log_T)
        loss = nll(logits_id / T, labels_id)
        loss.backward()
        return loss

    optimizer.step(closure)
    T_star = float(torch.exp(log_T).detach().cpu())
    return T_star


@torch.no_grad()
def reliability_and_ece(
    logits: torch.Tensor, labels: torch.Tensor, T: float, n_bins=15
):
    probs = F.softmax(logits / T, dim=1)
    conf, pred = probs.max(dim=1)
    correct = (pred == labels).float()
    bin_edges = torch.linspace(0, 1, steps=n_bins + 1)
    ece = torch.tensor(0.0, dtype=torch.float32)
    bins_out = []
    N = len(conf)
    for i in range(n_bins):
        lo, hi = bin_edges[i].item(), bin_edges[i + 1].item()
        mask = (
            (conf >= lo) & (conf < hi)
            if i < n_bins - 1
            else (conf >= lo) & (conf <= hi)
        )
        cnt = int(mask.sum().item())
        if cnt == 0:
            bins_out.append(
                {
                    "bin": i + 1,
                    "lo": lo,
                    "hi": hi,
                    "count": 0,
                    "acc": None,
                    "conf": None,
                }
            )
            continue
        acc_i = float(correct[mask].mean().item())
        conf_i = float(conf[mask].mean().item())
        ece += (cnt / N) * abs(acc_i - conf_i)
        bins_out.append(
            {
                "bin": i + 1,
                "lo": lo,
                "hi": hi,
                "count": cnt,
                "acc": acc_i,
                "conf": conf_i,
            }
        )
    return float(ece.item()), bins_out


@torch.no_grad()
def energy_scores(logits: torch.Tensor, T: float):
    return (-T) * torch.logsumexp(logits / T, dim=1)


def evaluate_ood_by_energy(E_id: np.ndarray, E_ood: np.ndarray):
    scores_id = (-E_id).astype(np.float64)
    scores_ood = (-E_ood).astype(np.float64)
    return get_measures(scores_id, scores_ood)


def parse_args():
    p = argparse.ArgumentParser("Temperature Scaling + Reliability/ECE + OOD(Energy)")
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="数据根目录（含 train/ test/ 与（可选）ood/）",
    )
    p.add_argument(
        "--ckpt", type=str, required=True, help="单体 resnet50+fc 权重（.pth）"
    )
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--test_split", type=str, default="test"
    )  # 没有 val/；用 test 做温度拟合
    p.add_argument("--ood_split", type=str, default="ood")
    p.add_argument("--n_bins", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 输出目录：按 ckpt 推导到 TrainProcess/<run>/calibration
    ckpt_path = Path(args.ckpt).resolve()
    if ckpt_path.parent.name == "checkpoints":
        ckpt_dir = ckpt_path.parent.parent
    else:
        ckpt_dir = ckpt_path.parent
    out_root = ckpt_dir / "calibration"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[Info] 输出目录: {out_root}")

    model = build_model(args.num_classes, str(ckpt_path), device)

    ds_test, dl_test = make_loader(
        args.data_dir, args.test_split, args.batch_size, args.num_workers
    )
    if dl_test is None:
        raise FileNotFoundError(f"未找到 {args.data_dir}/{args.test_split} 目录")
    ds_ood, dl_ood = make_loader(
        args.data_dir, args.ood_split, args.batch_size, args.num_workers
    )

    print(f"[Info] 使用 '{args.test_split}/' 作为温度拟合集与可靠性评测集")
    logits_test, labels_test = collect_logits(model, dl_test, device)
    labels_test = labels_test.to(logits_test.device)

    T_star = fit_temperature(logits_test, labels_test, T_init=1.0, max_iter=200)
    print(f"[Calib] Fitted Temperature T* = {T_star:.4f}")

    ece_pre, bins_pre = reliability_and_ece(
        logits_test, labels_test, T=1.0, n_bins=args.n_bins
    )
    ece_post, bins_post = reliability_and_ece(
        logits_test, labels_test, T=T_star, n_bins=args.n_bins
    )
    print(f"[Reliability] ECE (T=1.0)  = {ece_pre:.4f}")
    print(f"[Reliability] ECE (T=T*)   = {ece_post:.4f}")

    with open(out_root / "reliability_T1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "lo", "hi", "count", "acc", "conf"])
        for b in bins_pre:
            w.writerow([b["bin"], b["lo"], b["hi"], b["count"], b["acc"], b["conf"]])
    with open(out_root / "reliability_Tstar.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "lo", "hi", "count", "acc", "conf"])
        for b in bins_post:
            w.writerow([b["bin"], b["lo"], b["hi"], b["count"], b["acc"], b["conf"]])

    E_id_T1 = energy_scores(logits_test, T=1.0).cpu().numpy()
    E_id_Tstar = energy_scores(logits_test, T=T_star).cpu().numpy()
    np.save(out_root / "energy_id_T1.npy", E_id_T1)
    np.save(out_root / "energy_id_Tstar.npy", E_id_Tstar)

    result = {
        "T_star": T_star,
        "ECE_T1": ece_pre,
        "ECE_Tstar": ece_post,
        "OOD_T1": None,
        "OOD_Tstar": None,
    }

    if dl_ood is not None:
        logits_ood, _ = collect_logits(model, dl_ood, device)
        E_ood_T1 = energy_scores(logits_ood, T=1.0).cpu().numpy()
        E_ood_Tstar = energy_scores(logits_ood, T=T_star).cpu().numpy()
        np.save(out_root / "energy_ood_T1.npy", E_ood_T1)
        np.save(out_root / "energy_ood_Tstar.npy", E_ood_Tstar)
        m_T1 = evaluate_ood_by_energy(E_id_T1, E_ood_T1)
        m_Tstar = evaluate_ood_by_energy(E_id_Tstar, E_ood_Tstar)
        print("[OOD][T=1.0 ]", m_T1)
        print("[OOD][T=T*  ]", m_Tstar)
        result["OOD_T1"] = m_T1
        result["OOD_Tstar"] = m_Tstar

    with open(out_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(
        f"[SAVE] reliability_T1.csv / reliability_Tstar.csv / metrics.json @ {out_root}"
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
