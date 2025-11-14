import os, argparse, csv
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_measures
from pseudo_ood_utils import (
    _gaussian_noise,
    _gaussian_blur,
    _contrast,
    _randconv,
    _jigsaw,
)


def build_model(num_classes, ckpt, device):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m


def make_loader(root, split, bs, nw):
    tfm = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
    )
    ds = datasets.ImageFolder(os.path.join(root, split), transform=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return ds, dl


# 能量函数
def compute_energy(logits, T=1.0):
    return (-T) * torch.logsumexp(logits / T, dim=1)


# 扰动操作集合（按强度等级）
OPS = {
    "gaussian_noise": lambda x, lvl: _gaussian_noise(x, sigma=0.02 * lvl),
    "gaussian_blur": lambda x, lvl: _gaussian_blur(x, k=2 * lvl + 1),
    "contrast": lambda x, lvl: _contrast(x, alpha=0.5 + 0.2 * lvl),
    "randconv": lambda x, lvl: _randconv(x, k=lvl),
    "jigsaw": lambda x, lvl: _jigsaw(x, grid=2 + lvl),
}


def parse_args():
    p = argparse.ArgumentParser("鲁棒性评测 (伪OOD算子作为腐蚀源)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--levels", type=str, default="1,2,3,4,5", help="扰动强度等级列表")
    p.add_argument(
        "--ops",
        type=str,
        default="gaussian_noise,gaussian_blur,contrast,randconv,jigsaw",
        help="要测试的扰动算子，以逗号分隔",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="输出路径；留空则写入 ckpt 对应 TrainProcess/<run>/robustness/",
    )
    return p.parse_args()


@torch.no_grad()
def run_eval(model, dl_id, dl_ood, device, op_name, op_func, levels, T, out_root):
    csv_path = out_root / f"robust_{op_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["op", "level", "AUROC", "AUPR", "FPR95"])

        for lvl in levels:
            logits_id, logits_ood = [], []
            for x, _ in tqdm(dl_id, desc=f"{op_name}(lvl={lvl}) [ID]"):
                x_cor = op_func(x.clone(), lvl)
                x_cor = x_cor.to(device)
                logits = model(x_cor)
                logits_id.append(logits.cpu())

            for x, _ in tqdm(dl_ood, desc=f"{op_name}(lvl={lvl}) [OOD]"):
                x_cor = op_func(x.clone(), lvl)
                x_cor = x_cor.to(device)
                logits = model(x_cor)
                logits_ood.append(logits.cpu())

            logits_id = torch.cat(logits_id, 0)
            logits_ood = torch.cat(logits_ood, 0)
            E_id = compute_energy(logits_id, T=T).numpy()
            E_ood = compute_energy(logits_ood, T=T).numpy()

            auroc, aupr, fpr = get_measures(-E_id, -E_ood, 0.95)
            writer.writerow([op_name, lvl, auroc, aupr, fpr])
            print(
                f"[{op_name}] level={lvl} AUROC={auroc:.4f} AUPR={aupr:.4f} FPR95={fpr:.4f}"
            )

    print(f"[SAVE] {csv_path}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.num_classes, args.ckpt, device)

    # 输出目录：默认 TrainProcess/<run>/robustness/
    if args.out_dir and args.out_dir.strip():
        out_root = Path(args.out_dir).resolve()
    else:
        ckpt_path = Path(args.ckpt).resolve()
        if ckpt_path.parent.name == "checkpoints":
            out_root = ckpt_path.parent.parent / "robustness"
        else:
            out_root = ckpt_path.parent / "robustness"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[Info] 输出目录: {out_root}")

    ds_id, dl_id = make_loader(args.data_dir, "test", args.batch_size, args.num_workers)
    ds_ood, dl_ood = make_loader(
        args.data_dir, "ood", args.batch_size, args.num_workers
    )

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    print(f"[Info] 扰动算子: {ops}")
    print(f"[Info] 扰动强度: {levels}")

    for op_name in ops:
        if op_name not in OPS:
            print(f"[Warn] 未定义算子: {op_name}")
            continue
        op_func = OPS[op_name]
        run_eval(
            model, dl_id, dl_ood, device, op_name, op_func, levels, args.T, out_root
        )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
