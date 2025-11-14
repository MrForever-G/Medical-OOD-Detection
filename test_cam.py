from pathlib import Path
import os, argparse, csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


def build_model(num_classes, ckpt, device, target_layer_name="layer4.2"):
    """
    Build a resnet50 (fc=num_classes), load ckpt, set to eval.
    Additionally, prepare dicts to store activations & gradients on target layer.
    """
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    sd = torch.load(ckpt, map_location=device)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()

    # resolve target layer module by name
    tgt = m
    for part in target_layer_name.split("."):
        if part.isdigit():
            tgt = tgt[int(part)]
        else:
            tgt = getattr(tgt, part)

    feat = {}

    def fwd_hook(mod, inp, out):
        feat["act"] = out.detach()  # [B, C, H, W]

    def bwd_hook(mod, gin, gout):
        feat["grad"] = gout[0].detach()  # [B, C, H, W]

    tgt.register_forward_hook(fwd_hook)
    tgt.register_full_backward_hook(bwd_hook)
    return m, feat


def make_loader(root, split, bs, nw):
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )
    ds = datasets.ImageFolder(os.path.join(root, split), transform=tfm)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return ds, dl


def gradcam_from_feat(feat_dict, eps=1e-6):
    A = feat_dict.get("act", None)  # [B,C,H,W]
    G = feat_dict.get("grad", None)  # [B,C,H,W]
    if (A is None) or (G is None):
        return None
    w = torch.mean(G, dim=(2, 3), keepdim=True)  # [B,C,1,1]
    cam = torch.relu(torch.sum(w * A, dim=1, keepdim=True))  # [B,1,H,W]
    B = cam.size(0)
    cam = cam.view(B, -1)
    cam_min = cam.min(dim=1, keepdim=True).values
    cam_max = cam.max(dim=1, keepdim=True).values
    cam = (cam - cam_min) / (cam_max - cam_min + eps)
    cam = cam.view(B, 1, int(A.size(2)), int(A.size(3)))
    return cam


def overlay_cam_on_image(img_tensor, cam_2d):
    base = transforms.ToPILImage()(img_tensor.cpu().clamp(0, 1))
    cm = (cam_2d.cpu().numpy() * 255).astype(np.uint8)
    R = cm
    G = (cm * 0.5).astype(np.uint8)
    B = 255 - cm
    heat = np.stack([R, G, B], axis=-1)  # HxWx3
    heat_img = Image.fromarray(heat).resize(base.size, resample=Image.BILINEAR)
    alpha = 0.45
    blended = Image.blend(base.convert("RGB"), heat_img.convert("RGB"), alpha=alpha)
    return blended


def compute_energy(logits, T=1.0):
    return (-T) * torch.logsumexp(logits / T, dim=1)


def topk_ratio(cam, k=0.2):
    h, w = cam.shape
    v = cam.reshape(-1)
    topk = max(1, int(round(k * v.numel())))
    vals, _ = torch.topk(v, topk, largest=True)
    return (vals.sum() / (v.sum() + 1e-8)).item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root dir, with subfolders test/ or ood/",
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num_classes", type=int, required=True)
    p.add_argument("--split", type=str, default="test", choices=["test", "ood"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument(
        "--target_layer",
        type=str,
        default="layer4.2",
        help="e.g., layer4.2, layer4.2.conv3",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="可选；若留空则自动写入 ckpt 对应的 TrainProcess/<run>/cam_<split>/",
    )
    p.add_argument("--n_vis", type=int, default=100, help="Max images to save (visual)")
    p.add_argument(
        "--T", type=float, default=1.0, help="Energy temperature for reporting"
    )
    p.add_argument(
        "--topk",
        type=float,
        default=0.2,
        help="Top-k percentage for concentration metric",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, feat = build_model(
        args.num_classes, args.ckpt, device, target_layer_name=args.target_layer
    )

    ds, dl = make_loader(args.data_dir, args.split, args.batch_size, args.num_workers)

    # ===== 输出目录：优先用 --out_dir；否则按 ckpt 推导到 TrainProcess/<run>/cam_<split>/ =====
    if args.out_dir and args.out_dir.strip():
        out_root = Path(args.out_dir).resolve() / args.split
    else:
        ckpt_path = Path(args.ckpt).resolve()
        if ckpt_path.parent.name == "checkpoints":
            base_dir = ckpt_path.parent.parent  # .../TrainProcess/<run>
        else:
            base_dir = ckpt_path.parent  # .../TrainProcess/<run>
        out_root = base_dir / f"cam_{args.split}"
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    print(f"[Info] 输出目录: {out_root}")

    stats_csv = out_root / "stats.csv"
    with open(stats_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "path", "pred", "energy", "maxlog", "topk_ratio"])

        saved = 0
        idx_global = 0
        for x, _ in tqdm(dl, desc=f"Grad-CAM on {args.split}"):
            x = x.to(device, non_blocking=True)
            x.requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits = model(x)  # [B, C]
            pred = torch.argmax(logits, dim=1)  # [B]

            onehot = torch.zeros_like(logits).scatter_(1, pred.view(-1, 1), 1.0)
            loss = torch.sum(onehot * logits)
            loss.backward(retain_graph=False)

            cam = gradcam_from_feat(feat)  # [B,1,h,w] in [0,1]
            if cam is None:
                print("Warning: CAM not available; ensure target_layer is correct.")
                break

            cam_up = F.interpolate(
                cam, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False
            )  # [B,1,224,224]
            cam_up = cam_up.squeeze(1)  # [B,224,224]

            with torch.no_grad():
                E = compute_energy(logits, T=args.T)  # [B]
                maxlog = logits.max(dim=1).values  # [B]

            B = x.size(0)
            for i in range(B):
                path = ds.samples[idx_global + i][0] if hasattr(ds, "samples") else ""
                tkr = topk_ratio(cam_up[i], k=args.topk)
                writer.writerow(
                    [
                        idx_global + i,
                        path,
                        int(pred[i].item()),
                        float(E[i].item()),
                        float(maxlog[i].item()),
                        tkr,
                    ]
                )

                if saved < args.n_vis:
                    vis = overlay_cam_on_image(
                        x[i].detach().cpu(), cam_up[i].detach().cpu()
                    )
                    stem = (
                        os.path.splitext(os.path.basename(path))[0]
                        if path
                        else f"img_{idx_global+i:06d}"
                    )
                    vis.save(out_root / "images" / f"{stem}_cam.png")
                    saved += 1

            idx_global += B

    print(f"Saved stats: {stats_csv}")
    print(f"Saved images (up to {args.n_vis}): {(out_root/'images')}")


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.freeze_support()
    main()
