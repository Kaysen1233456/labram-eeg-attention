# 方案 B：给 LaBraM 官方 `run_class_finetuning.py` 打补丁

适用场景：你想用官方脚本跑正式实验，以复用它的 `layer_decay`、`mixup`、`--eval` 等完整特性。

总共 **3 处改动**。改完**不改动官方任何逻辑**，只在「没有传 `--class_weights` 时」保持原行为。

---

## 补丁 1：新增命令行参数

**位置**：`run_class_finetuning.py` 约第 132 行，`# Dataset parameters` 注释块内（与 `--nb_classes`、`--data_path` 等放在一起）。

**加入**：

```python
    parser.add_argument('--class_weights', type=float, nargs='+', default=None,
                        help='CrossEntropyLoss 的类别权重，按类别索引顺序给出，'
                             '例如二分类 --class_weights 0.6714 1.9589')
```

**说明**：`nargs='+'` 允许传任意多个数；`default=None` 保证不传时行为与原来完全一致。

---

## 补丁 2：把权重接到损失函数上

**位置**：`run_class_finetuning.py` 第 447–452 行。

**原文**：

```python
    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
```

**改为**：

```python
    if args.nb_classes == 1:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    elif args.class_weights is not None:
        weight = torch.tensor(args.class_weights, dtype=torch.float).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weight)
        print("Using class weights: %s" % str(args.class_weights))
    else:
        criterion = torch.nn.CrossEntropyLoss()
```

**安全性确认**：`device` 在本文件第 **239** 行已经定义（`device = torch.device(args.device)`），而 `criterion` 在第 452 行，所以 `device` 一定可用，不会出现 `NameError`。

---

## 补丁 3（必要）：把自定义数据集接进 `get_dataset`

官方 `get_dataset()` 只认 `TUAB` / `TUEV` 等内置数据集。要跑自己的数据，必须加一个分支。

**位置**：`run_class_finetuning.py` 第 213 行 `def get_dataset(args):`。

**在 `if/elif` 链里加一个 `elif`**（例如放在 `elif args.dataset == 'TUEV':` 之后）：

```python
    elif args.dataset == 'MENTAL_ARITHMETIC':
        import numpy as np
        from torch.utils.data import Dataset as TorchDataset

        class NpyDataset(TorchDataset):
            """读 eeg_attention/prepare.py 产出的 npy。X 为 3D [N, 8, 400]，单位 uV。"""
            def __init__(self, split, root):
                self.X = np.load("%s/X_%s.npy" % (root, split))
                self.y = np.load("%s/y_%s.npy" % (root, split))

            def __len__(self):
                return len(self.y)

            def __getitem__(self, i):
                # 返回 3D [8, 400]；engine 内部会 rearrange 成 [8, 2, 200] 并 /100
                return torch.tensor(self.X[i], dtype=torch.float32), int(self.y[i])

        root = args.data_path          # 指向 .../data/processed/mental_arithmetic_labram_200hz_8ch
        train_dataset = NpyDataset("train", root)
        val_dataset = NpyDataset("val", root)
        test_dataset = NpyDataset("test", root)

        # 注意：必须大写，且不带 'EEG ' 前缀 —— utils.get_input_chans 直接查 standard_1020
        ch_names = ['FZ', 'F3', 'F4', 'T7', 'T8', 'PZ', 'P3', 'P4']
        args.nb_classes = 2
        metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
```

**三个必须注意的点**：

1. **`ch_names` 必须大写、不带前缀**。`utils.get_input_chans` 的实现是 `standard_1020.index(ch_name) + 1`，区分大小写；而 `standard_1020` 表里存的是 `'FZ'` 这种大写形式。
2. **返回 3D `[8, 400]`**，不是 4D。官方 `engine_for_finetuning.py` 第 174 行会自己做 `rearrange(EEG, 'B N (A T) -> B N A T', T=200)`。
3. **不要在 Dataset 里做 `/100`**。官方 engine 已经做了（第 64 / 173 行），重复做会导致输入缩小 100 倍。

---

## 运行

```bash
python run_class_finetuning.py \
    --model labram_base_patch200_200 \
    --finetune ./checkpoints/labram-base.pth \
    --dataset MENTAL_ARITHMETIC \
    --data_path ./data/processed/mental_arithmetic_labram_200hz_8ch \
    --input_size 400 \
    --nb_classes 2 \
    --class_weights 0.6714 1.9589 \
    --layer_decay 0.65 --warmup_epochs 5 --epochs 30 \
    --drop_path 0.1 --disable_rel_pos_bias --abs_pos_emb --disable_qkv_bias \
    --batch_size 16 --lr 2e-4
```

对照实验（不加权）只需去掉 `--class_weights`：

```bash
python run_class_finetuning.py ... --lr 5e-4      # 不加权时可用官方默认 lr
```

---

## 关于学习率

| 情形 | 建议 lr |
|---|---|
| 不加类别权重 | 官方默认 `5e-4`（为 TUAB 调的） |
| **加了类别权重** | **先降到 `2e-4`** |

原因：加权 CE 会放大 loss 的绝对尺度，等价于放大梯度。若沿用 5e-4 容易震荡。

更稳妥的做法是跑两组对照（加权 / 不加权），把 BA 差值作为一张表放进汇报——这本身就是有价值的证据。

---

## 验证补丁生效

跑起来后，日志里应该出现：

```
Using class weights: [0.6714, 1.9589]
criterion = CrossEntropyLoss()
```

如果没看到 `Using class weights`，说明补丁 1 或 2 没生效（多半是 `elif` 的位置放错了，被前面的 `else` 短路）。

另外确认 `--nb_classes 2` 时 `evaluate()` 走的是 `CrossEntropyLoss` 分支（`is_binary = (args.nb_classes == 1)` 为 `False`），这样 `balanced_accuracy` 会正常出现在指标里。
