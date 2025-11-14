import torch, sys, os

# 用法:
#   python convert_ckpt_to_single.py IN_PTH OUT_PTH
# 例子:
#   python convert_ckpt_to_single.py ./TrainProcess/2025-11-03-21-00-37/checkpoints/best_model_ood.pth ./TrainProcess/2025-11-03-21-00-37/checkpoints/best.pth

in_pth = sys.argv[1]
out_pth = sys.argv[2]

ckpt = torch.load(in_pth, map_location="cpu")
if not isinstance(ckpt, dict) or not all(k in ckpt for k in ["backbone", "classifier"]):
    raise RuntimeError(
        "输入的ckpt不是{'backbone','classifier',...}结构，确认传入的是 best_model_*.pth"
    )

backbone_sd = ckpt["backbone"]  # ResNet主干
classifier_sd = ckpt["classifier"]  # 线性分类头

single = {}
# 复制backbone参数
for k, v in backbone_sd.items():
    single[k] = v

# 把classifier映射到ResNet的fc
if "weight" in classifier_sd:
    single["fc.weight"] = classifier_sd["weight"]
if "bias" in classifier_sd:
    single["fc.bias"] = classifier_sd["bias"]

torch.save(single, out_pth)
print(f"[OK] 已导出单体模型权重: {out_pth}")
