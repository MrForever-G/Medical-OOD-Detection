# run.py
import argparse, torch
from train import TrainConfig, train_and_eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../datasets/42+156/42+156/")
    p.add_argument("--num_classes", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--log_root", default="TrainProcess")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.02)

    p.add_argument("--beta", type=float, default=0.12)

    p.add_argument("--mID", type=float, default=-10.0)
    p.add_argument("--mOOD", type=float, default=-1.5)
    p.add_argument("--lambda_conf", type=float, default=0.00)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--detach_lid", action="store_true", default=True)
    p.add_argument(
        "--pseudo_mode",
        default="kys",
        choices=["kys", "auto", "mixup", "cutmix", "weak"],
    )
    p.add_argument("--warmup_head_epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--label_smoothing", type=float, default=0.01)
    p.add_argument("--lr_warmup_epochs", type=int, default=2)
    p.add_argument("--lr_min_factor", type=float, default=0.1)

    p.add_argument(
        "--use_real_ood", action="store_true", default=True, help="启用真实 OOD 分支"
    )
    p.add_argument("--real_ood_dir", type=str, default="../datasets/extract_DDI")

    p.add_argument(
        "--p_real_ood",
        type=float,
        default=0.75,
        help="真实OOD批量相对比例(相对ID batch)",
    )

    p.add_argument(
        "--beta_real", type=float, default=0.40, help="真实OOD项的权重（不随ramp缩放）"
    )

    p.add_argument(
        "--gamma_ood_head",
        type=float,
        default=0.3,
        help="OOD头损失在(能量/头)合成中的占比",
    )
    p.add_argument("--T_eval", type=float, default=1.2, help="评估阶段使用的能量温度")

    return p.parse_args()


def main():
    a = parse_args()
    cfg = TrainConfig(
        data_dir=a.data_dir,
        num_classes=a.num_classes,
        num_epochs=a.epochs,
        batch_size=a.batch_size,
        learning_rate=a.lr,
        device=a.device,
        log_root=a.log_root,
        T=a.T,
        alpha=a.alpha,
        beta=a.beta,
        mID=a.mID,
        mOOD=a.mOOD,
        lambda_conf=a.lambda_conf,
        pseudo_ood_mode=a.pseudo_mode,
        warmup_head_epochs=a.warmup_head_epochs,
        workers=a.workers,
        label_smoothing=a.label_smoothing,
        lr_warmup_epochs=a.lr_warmup_epochs,
        lr_min_factor=a.lr_min_factor,
        max_grad_norm=a.max_grad_norm,
        detach_lid=a.detach_lid,
        use_real_ood=a.use_real_ood,
        real_ood_dir=a.real_ood_dir,
        p_real_ood=a.p_real_ood,
        beta_real=a.beta_real,
        gamma_ood_head=a.gamma_ood_head,
        T_eval=a.T_eval,
    )
    torch.backends.cudnn.benchmark = True
    train_and_eval(cfg)


if __name__ == "__main__":
    main()
