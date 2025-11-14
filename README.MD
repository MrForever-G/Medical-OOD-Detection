本项目是医学图像 开放集 / 外分布检测（OOD Detection） 方案
主体基于 ResNet50 + 分类头 + OOD 二分类头
OOD 包含两类：
伪 OOD（Pseudo-OOD）：多层级增强 + 特征扰动
真 OOD（Real-OOD）：来自 DDI 皮肤病数据集和可公开的常见用于OOD检测的数据集
核心指标：AUROC、FPR95、Top-1 等

项目结构：
code/
  run.py                  # 主入口
  train.py                # 核心训练流程
  pseudo_ood_utils.py     # 多层伪OOD生成策略
  real_ood.py             # 真OOD采样
  energy_loss.py          # 能量损失与相对间隔
  class_stats.py          # 类统计 & 特征采样器
  test_energy.py          # OOD能量评测
  test_robust.py          # 伪OOD鲁棒性
  test_cam.py             # 可视化
  eval_calibration.py     # 温度/可靠性
  convert_ckpt_to_single.py #辅助pth文件转换

环境准备：
conda env create -f environment.yml

运行整个项目：
python run.py 
所有调整好的参数已经在run.py中给出

评估测试运行方法：
python convert_ckpt_to_single.py ^
  ".\TrainProcess\best\best_model_fpr95.pth" ^
  ".\TrainProcess\best\best.pth"

python test_energy.py \
  --data_dir ./datasets/42+156/42+156 \
  --ckpt ./TrainProcess/<run>/checkpoints/best.pth \
  --num_classes 42 \
  --T_list "1.2" \
  --gamma_ood_head 0.6

python test_cam.py ^
  --data_dir ../datasets/42+156/42+156/ ^
  --ckpt ./TrainProcess/2025-11-07-18-47-19/best.pth ^
  --num_classes 42 ^
  --split ood ^
  --out_dir outputs_cam ^
  --n_vis 100 ^
  --T 1.0 ^
  --topk 0.2

  python eval_calibration.py ^
  --data_dir "..\datasets\42+156\42+156" ^
  --ckpt ".\2025-11-07-15-23-45\best.pth" ^
  --num_classes 42 ^
  --batch_size 32 ^
  --num_workers 4

  python test_robust.py ^
  --data_dir "..\datasets\42+156\42+156" ^
  --ckpt ".\2025-11-07-15-23-45\best.pth" ^
  --num_classes 42 ^
  --batch_size 32 ^
  --num_workers 4