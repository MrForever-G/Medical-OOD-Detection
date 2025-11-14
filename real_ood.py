# real_ood.py
import os, itertools
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets, transforms


def build_real_ood_iter(
    real_ood_dir: str,
    batch_size: int,
    num_workers: int = 0,
    normalize_transform=None,
):
    """
    返回：real_ood_iter（无限循环的迭代器）与 real_ood_loader（可选）
    - 采用 ImageFolder 读取目录下每个子文件夹（类别标签无关紧要，只当“非ID”）
    - 用带放回的采样（解决某些类仅1张图的问题）
    - 自带轻量增广；可传入你的 Normalize 以对齐ID预处理
    """
    if not (real_ood_dir and os.path.isdir(real_ood_dir)):
        return None, None

    tf = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
    ]
    if normalize_transform is not None:
        tf.append(normalize_transform)
    real_ood_tf = transforms.Compose(tf)

    ds = datasets.ImageFolder(real_ood_dir, transform=real_ood_tf)
    if len(ds) == 0:
        return None, None

    loader = DataLoader(
        ds,
        batch_size=max(1, batch_size),
        sampler=RandomSampler(ds, replacement=True),  # 关键：带放回
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    it = iter(itertools.cycle(loader))  # 无限循环
    return it, loader
