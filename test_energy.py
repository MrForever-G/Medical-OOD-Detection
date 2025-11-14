from pathlib import Path
import os, argparse, json, numpy as np
import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import get_measures


def build_model(num_classes, ckpt, device):
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    sd = torch.load(ckpt, map_location=device)
    ood_head = None

    if isinstance(sd, dict) and ("state_dict" in sd or "backbone" in sd or "classifier" in sd):
        if "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]
        if "backbone" in sd and isinstance(sd["backbone"], dict):
            try:
                m.load_state_dict(sd["backbone"], strict=False)
            except:
                m.load_state_dict({k.replace("module.", ""): v for k, v in sd["backbone"].items()}, strict=False)
        else:
            try:
                m.load_state_dict(sd, strict=False)
            except:
                m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)
        if "classifier" in sd and isinstance(sd["classifier"], dict):
            try:
                m.fc.load_state_dict(sd["classifier"], strict=False)
            except:
                m.fc.load_state_dict({k.replace("module.", ""): v for k, v in sd["classifier"].items()}, strict=False)
        if "ood_head" in sd and isinstance(sd["ood_head"], dict):
            in_f = sd["ood_head"].get("in_features", m.fc.in_features)
            ood_head = nn.Linear(in_f, 1)
            try:
                ood_head.load_state_dict(sd["ood_head"], strict=False)
            except:
                ood_head.load_state_dict({k.replace("module.", ""): v for k, v in sd["ood_head"].items()}, strict=False)
    else:
        try:
            m.load_state_dict(sd, strict=False)
        except:
            m.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False)

    m.to(device).eval()
    if ood_head is not None:
        ood_head.to(device).eval()
    return m, ood_head


@torch.no_grad()
def forward_logits_feats(model, x: torch.Tensor):
    feats = None
    def hook(_, __, outp):
        nonlocal feats
        feats = outp
    h = model.avgpool.register_forward_hook(hook)
    logits = model(x)
    h.remove()
    if feats is not None and feats.ndim == 4:
        feats = torch.flatten(feats, 1)
    return logits, feats


def make_loader(root: str, bs: int, nw: int):
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    ds = datasets.ImageFolder(root, transform=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return ds, dl


def energy_id(logits: torch.Tensor, T: float) -> np.ndarray:
    z = logits / T
    e = (-T) * torch.logsumexp(z, dim=1)
    return (-e).detach().cpu().numpy()


def maxlog_id(logits: torch.Tensor, T: float) -> np.ndarray:
    z = logits / T
    m = torch.max(z, dim=1).values
    return m.detach().cpu().numpy()


def infer_scores(model, ood_head, loader, T: float, w_energy: float, w_maxlog: float, gamma_head: float, device):
    scores = []
    for x, _ in tqdm(loader, ncols=100, leave=False):
        x = x.to(device, non_blocking=True)
        logits, feats = forward_logits_feats(model, x)
        s0 = w_energy * energy_id(logits, T) + w_maxlog * maxlog_id(logits, T)
        if (ood_head is not None) and (feats is not None) and (gamma_head > 0):
            s_head = ood_head(feats).squeeze(1).detach().cpu().numpy()
            s = (1.0 - gamma_head) * s0 + gamma_head * s_head
        else:
            s = s0
        scores.append(s)
    return np.concatenate(scores, axis=0)


def auto_guess_ood_dir(data_dir: str):
    cand = [
        os.path.join(data_dir, "ood"),
        os.path.join(os.path.dirname(data_dir.rstrip("/\\")), "extract_DDI"),
        os.path.join(data_dir, "real_ood"),
    ]
    for p in cand:
        if os.path.isdir(p):
            return p
    return data_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--T_list", type=str, default="1.2")
    p.add_argument("--w_energy", type=float, default=0.7)
    p.add_argument("--w_maxlog", type=float, default=0.3)
    p.add_argument("--gamma_ood_head", type=float, default=0.0)
    p.add_argument("--ood_dir", type=str, default="")
    p.add_argument("--id_split", type=str, default="test")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T_eval = float(args.T_list.split(",")[0].strip())

    model, ood_head = build_model(args.num_classes, args.ckpt, device)
    if ood_head is None:
        args.gamma_ood_head = 0.0

    if args.id_split in ("train", "val", "test"):
        cand = os.path.join(args.data_dir, args.id_split)
        id_root = cand if os.path.isdir(cand) else args.data_dir
    else:
        id_root = args.data_dir

    ood_root = args.ood_dir if args.ood_dir else auto_guess_ood_dir(args.data_dir)

    _, dl_id  = make_loader(id_root,  args.batch_size, args.num_workers)
    _, dl_ood = make_loader(ood_root, args.batch_size, args.num_workers)

    out_root = Path(args.ckpt).parent / "eval_energy"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[Info] 输出目录: {str(out_root)}")

    s_id  = infer_scores(model, ood_head, dl_id,  T_eval, args.w_energy, args.w_maxlog, args.gamma_ood_head, device)
    s_ood = infer_scores(model, ood_head, dl_ood, T_eval, args.w_energy, args.w_maxlog, args.gamma_ood_head, device)

    auroc, aupr_in, fpr95 = get_measures(s_id, s_ood, recall_level=0.95)
    _, _, fpr92 = get_measures(s_id, s_ood, recall_level=0.92)
    _, _, fpr90 = get_measures(s_id, s_ood, recall_level=0.90)

    print("===== OOD 评测 =====")
    print(f"t = {T_eval:.4f} | AUROC={auroc:.4f}  AUPR-in={aupr_in:.4f}  "
          f"FPR@TPR90={fpr90:.4f}  FPR@TPR92={fpr92:.4f}  FPR@TPR95={fpr95:.4f}")

    metrics_eval = {
        "T_eval": float(T_eval),
        "AUROC": float(auroc),
        "AUPR_in": float(aupr_in),
        "FPR@TPR90": float(fpr90),
        "FPR@TPR92": float(fpr92),
        "FPR@TPR95": float(fpr95),
        "weights": {
            "w_energy": float(args.w_energy),
            "w_maxlog": float(args.w_maxlog),
            "gamma_ood_head": float(args.gamma_ood_head),
        },
        "counts": {"n_id": int(len(s_id)), "n_ood": int(len(s_ood))},
        "paths": {"id_root": str(id_root), "ood_root": str(ood_root)},
    }
    with open(out_root / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(metrics_eval, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_root / 'metrics_eval.json'}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()
    main()
