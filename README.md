# LaBraM EEG Attention — 心算数据集微调

把 **LaBraM-Base** 预训练权重迁移到「静息态 vs 专注态」二分类任务上的完整可复现实验代码。

数据集：**eegmat**（EEG During Mental Arithmetic Tasks, PhysioNet），静息闭眼 vs 连续心算。

> 本项目是 LaBraM 官方仓库的一个**适配层**，不修改任何官方代码。
> 依赖 `935963004/LaBraM` 的 `modeling_finetune.py` / `utils.py` / `checkpoints/labram-base.pth`。

---

## 目录结构

```
LaBraM/
├── checkpoints/labram-base.pth        # 官方预训练权重（随仓库 clone 自带）
├── modeling_finetune.py               # 官方模型定义（未改动）
├── utils.py                           # 官方工具（未改动）
├── data/
│   ├── xin suan/                      # 原始 EDF（SubjectNN_1=静息, SubjectNN_2=心算）
│   └── processed/                     # 预处理产物
└── eeg_attention/                     # ← 本项目代码
    ├── config.py                      # 集中配置（通道、input_chans、预处理参数）
    ├── prepare.py                     # EDF -> LaBraM 输入格式
    ├── dataset.py                     # 数据发现 / 被试划分 / Dataset
    ├── finetune.py                    # 微调主脚本（方案 A：类别权重在此落地）
    ├── diagnose.py                    # 诊断脚本（估计上限，不可用于报告）
    ├── check_signal.py                # 数据可分性验证
    ├── patch_official.md              # 方案 B：官方 run_class_finetuning.py 补丁
    └── results/                       # 实验结果 JSON
```

---

## 环境

```
Python 3.11
torch 2.0.1+cu118
timm 0.4.12
mne 1.4.2
numpy 1.26.4        # 必须 1.x，2.x 与 torch 2.0.1 二进制不兼容
scikit-learn
einops, h5py
```

安装：

```bash
conda create -n labram python=3.11 -y
conda activate labram
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy==1.26.4" mne timm==0.4.12 einops h5py scikit-learn
```

**不需要 deepspeed**：Windows 装不上，且官方微调脚本默认 `--enable_deepspeed=False`，不加该参数就不会 import。

---

## 快速开始

```bash
# 0) 先 clone LaBraM 官方仓库，把本项目放到其 eeg_attention/ 目录下
git clone https://github.com/935963004/LaBraM.git
cd LaBraM

# 1) 预处理：EDF -> npy
python eeg_attention/prepare.py

# 2) 微调
python eeg_attention/finetune.py --epochs 30

# 常用变体
python eeg_attention/finetune.py --freeze-encoder          # 线性探测
python eeg_attention/finetune.py --no-class-weights        # 对照：不加权
python eeg_attention/finetune.py --loso                    # 留一被试（15 折）
python eeg_attention/finetune.py --seeds 42 43 44          # 多 seed
```

---

## 三个必须知道的实现细节

这三条都是从 LaBraM 源码逐行核实 + 本机实测得出的，**搞错任何一条结果都会崩**。

### 1. 输入尺度是 `µV / 100`，不是 µV

LaBraM 全流程统一把输入除以 100：

| 文件 | 行号 | 代码 |
|---|---|---|
| `engine_for_finetuning.py` | 64 | `samples = samples.float().to(device) / 100` |
| `engine_for_finetuning.py` | 173 | `EEG = EEG.float().to(device) / 100` |
| `engine_for_pretraining.py` | 87 | `samples = samples.float().to(device) / 100` |
| `engine_for_vqnsp.py` | 57 | `EEG = batch.float().to(device) / 100` |

实测（以 µV/100 为基准，比较编码器特征）：

| 输入 | 特征相对差异 | Pearson r |
|---|---|---|
| **µV / 100（正确）** | 0.00% | 1.000 |
| µV 原值（忘记除 100） | **36.13%** | 0.905 |
| mV | 66.78% | 0.841 |
| **V 原值** | **82.51%** | 0.716 |

所以：**数据存 µV，训练循环里再 `/100`**（本仓库 `finetune.py` 已内置）。

> 常见误解是「GroupNorm 会把缩放归一化掉，单位无所谓」。实测证明是错的：`GroupNorm` 的 `eps=1e-5` 是固定值，输入幅值太小时归一化直接失效。

### 2. `input_chans` 是 `pos_embed` 的索引，必须与通道顺序一一对应

`modeling_finetune.py:358`：

```python
pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
```

`pos_embed` 形状 `(1, 129, 200)` = 1 个 cls + 128 个电极位。`utils.get_input_chans` 的定义是 `standard_1020.index(ch_name) + 1`。

本项目 8 通道的映射（从 `utils.standard_1020` 实时查出）：

```python
input_chans = [0, 20, 18, 22, 38, 46, 64, 62, 66]
#               ↑  cls  Fz  F3  F4  T7  T8  Pz  P3  P4
```

实测敏感度：交换 F3/F4 的通道位置 → 编码器特征差 **35.76%**；`input_chans` 全部填同一索引 → 差 **92.06%**。两者都真实参与计算。

⚠️ 注意 `standard_1020` 表长 **136**，而 `pos_embed` 只有 **129** 位，索引 **必须 ≤ 127**，否则越界。

### 3. 类别不平衡必须显式加权

`run_class_finetuning.py:452` 默认是 `torch.nn.CrossEntropyLoss()`，**不带 `weight`**。

本任务 rest:focus ≈ **2.9 : 1**。不加权时模型学会「全猜 rest」即可拿到约 74% accuracy，但 **balanced accuracy 掉到 0.5 附近**。

`finetune.py` 已实现：

```python
w = torch.tensor(meta["class_weights"]).to(device)   # 例：[0.671, 1.959]
criterion = torch.nn.CrossEntropyLoss(weight=w)
```

权重由 `prepare.py` 按简单反比算出：`weight_i = total / (n_classes × count_i)`。

⚠️ 加权重会放大 loss 尺度，若沿用官方 `--lr 5e-4`，建议**先降到 2e-4**。

---

## 两种类别权重方案

| | 方案 A（本仓库 `finetune.py`） | 方案 B（官方 `run_class_finetuning.py` + 补丁） |
|---|---|---|
| 改动量 | 无（已内置） | 改 2 处 + 加 1 个参数 |
| 优点 | 自包含、易改、便于接自研评测框架 | 可复用官方的 `layer_decay` / `mixup` / 完整指标 |
| 缺点 | 未实现 `layer_decay`（用 head/encoder 分组 lr 近似） | 需把数据组织成官方 dataset 格式 |
| 适用 | 快速验证、对照实验、自研框架 | 正式大规模实验 |

**方案 B 补丁**见 [`patch_official.md`](patch_official.md)。

---

## 实测结果（本机 RTX 3060 Laptop 6GB）

数据：**15 名完整被试**（Subject00–13, 15），每人 91 静息窗 + 31 心算窗 = 122 窗，共 1827 窗。

| 配置 | 测试 BA | macro-F1 | AUC |
|---|---|---|---|
| 全量微调（加权，11/2/2 划分） | 0.5549 | 0.5571 | 0.6210 |
| 线性探测（加权，11/2/2 划分） | 0.5669 | 0.5605 | 0.7385 |
| 随机窗口划分（**有泄漏**，仅估上限） | 0.6705 | 0.6866 | 0.7747 |
| 单特征基线（仅用相对 alpha） | 0.5802 | — | 0.593 |

### 结论

1. **连有泄漏的随机划分 BA 也只有 0.67**，说明瓶颈**不在跨被试泛化**，而在任务本身在这份数据上的可分辨性。
2. **LaBraM 没有超过「相对 alpha 单特征」这个极简基线**（0.57 vs 0.58）。
3. 训练 loss 可降到 0.009（严重过拟合），但验证/测试 BA 长期停在 0.5 附近。

### 根因（已定位）

逐被试检查「相对 alpha」比值（rest/focus）：

| 分组 | 结果 |
|---|---|
| 训练集 11 人 | **全部 > 1**（1.02 ~ 1.70），方向一致 |
| 验证集 Subject10 | **0.493** ← 方向反了 |
| 验证集 Subject12 | **0.611** ← 方向反了 |
| 测试集 Subject03 / 06 | 1.699 / 1.279，方向正常 |

验证集那两名被试的 alpha 方向与训练集**完全相反**，这直接解释了「验证集 AUC < 0.5」的异常。

**主要瓶颈是被试数太少（15 人）导致被试间差异被放大**。原数据集共 36 人，当前本地只有 15 人的完整配对（另 4 人只有单侧文件，16 人完全缺失）。

---

## 已知限制

1. **原始数据已被采集方 45 Hz 低通**，而 LaBraM 预训练用 0.1–75 Hz。实测 45 Hz 以上功率骤降 65 倍，确认数据本身就没有高频，无法恢复。
2. **静息态是闭眼（eyes closed）**，不是睁眼。闭眼会引起 alpha 爆发，存在「闭眼 vs 睁眼」与「静息 vs 专注」的混淆风险，所有结论需据此打折。
3. **被试数不足**：15 人做跨被试评测，验证/测试集各仅 2 人，指标方差极大。
4. `diagnose.py` 使用随机窗口划分，**结果仅用于诊断上限，绝不可用于报告或与文献对标**。

---

## 引用

- LaBraM: Jiang et al. *Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI*. ICLR 2024. [arXiv:2405.18765](https://arxiv.org/abs/2405.18765) · [GitHub](https://github.com/935963004/LaBraM)
- eegmat: Zyma et al. *EEG During Mental Arithmetic Tasks*. PhysioNet, 2019. [PhysioNet](https://physionet.org/content/eegmat/1.0.0/)

## 许可

本适配层代码遵循 LaBraM 官方仓库的 **MIT** 许可。
