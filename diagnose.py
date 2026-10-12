"""
诊断脚本：判断"模型学不会"还是"跨被试太难"

⚠️ 本脚本使用随机窗口划分（有数据泄漏），结果**仅用于诊断上限**，
   绝不可用于报告或与文献对标。项目正式评测必须用被试不重叠划分。

诊断三问：
  Q1 随机窗口划分下能到多少？  -> 模型能力上限
  Q2 每个被试的数据有没有异常？  -> 数据质量
  Q3 被试内 vs 跨被试差多少？    -> 泛化难度
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C        # noqa: E402
import dataset as DS      # noqa: E402
from finetune import build_model, evaluate  # noqa: E402


def load_all():
    """合并 train/val/test 的全部窗口（仅诊断用）"""
    d = C.PROCESSED_DIR
    Xs, ys = [], []
    for name in ("train", "val", "test"):
        Xs.append(np.load(d / ("X_%s_4d.npy" % name)))
        ys.append(np.load(d / ("y_%s.npy" % name)))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    ch_order = json.loads((d / "channel_order.json").read_text(encoding="utf-8"))
    return X, y, meta, ch_order


def train_eval(Xtr, ytr, Xte, yte, input_chans, cw, device,
               epochs=30, batch=16, lr_head=1e-3, lr_enc=1e-5, seed=42,
               freeze=False, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dl_tr = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                     torch.tensor(ytr, dtype=torch.long)),
                       batch_size=batch, shuffle=True, num_workers=0)
    dl_te = DataLoader(TensorDataset(torch.tensor(Xte, dtype=torch.float32),
                                     torch.tensor(yte, dtype=torch.long)),
                       batch_size=batch, shuffle=False, num_workers=0)

    model = build_model(C.N_CLASSES, device, verbose=False)
    if freeze:
        for n, p in model.named_parameters():
            if not (n.startswith("head.") or n.startswith("fc_norm.")):
                p.requires_grad = False

    hp, ep = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            (hp if (n.startswith("head.") or n.startswith("fc_norm.")) else ep).append(p)
    groups = [{"params": hp, "lr": lr_head}]
    if ep:
        groups.append({"params": ep, "lr": lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    for e in range(epochs):
        model.train()
        for x, y in dl_tr:
            x = x.to(device) / C.LABRAM_INPUT_DIVISOR
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = crit(model(x, input_chans=input_chans), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        if verbose and (e + 1) % 10 == 0:
            print("    ep %d loss %.4f" % (e + 1, loss.item()))

    return evaluate(model, dl_te, device, input_chans, crit)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y, meta, ch_order = load_all()
    input_chans = ch_order["input_chans"]
    cw = meta["class_weights"]

    print("=" * 74)
    print("诊断报告（随机窗口划分，仅用于估计上限，不可用于正式报告）")
    print("=" * 74)
    print("总窗口数: %d  (rest=%d focus=%d)" % (len(y), (y == 0).sum(), (y == 1).sum()))
    print("input_chans: %s" % input_chans)

    # ---------- Q1: 随机窗口划分（泄漏） ----------
    print()
    print("-" * 74)
    print("Q1  随机窗口划分（有泄漏）—— 模型能力上限")
    print("-" * 74)
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(y))
    n_tr = int(len(y) * 0.8)
    tr, te = idx[:n_tr], idx[n_tr:]
    m = train_eval(X[tr], y[tr], X[te], y[te], input_chans, cw, device,
                   epochs=30, verbose=True)
    print("  测试集: BA %.4f | macroF1 %.4f | AUC %.4f | acc %.4f"
          % (m["balanced_accuracy"], m["macro_f1"], m["roc_auc"], m["accuracy"]))
    print("  召回: rest %.3f  focus %.3f" % (m["recall_rest"], m["recall_focus"]))
    print("  => 若此值显著高于跨被试结果，说明模型有能力，瓶颈在跨被试泛化")

    # ---------- Q2: 逐被试数据统计 ----------
    print()
    print("-" * 74)
    print("Q2  逐被试数据统计（看是否有异常被试）")
    print("-" * 74)
    print("%-12s %-8s %-10s %-10s %-10s" % ("被试", "n窗", "std(uV)", "alpha比", "备注"))
    print("-" * 74)

    # 用被试顺序重排：数据是按被试顺序拼接的
    # train: 11人 x (91+31)，val: 2人，test: 2人（Subject04 的 rest 是 85）
    subj_order = (meta["train_subjects"] + meta["val_subjects"] + meta["test_subjects"])
    counts = {}
    for name, subs in (("train", meta["train_subjects"]),
                       ("val", meta["val_subjects"]),
                       ("test", meta["test_subjects"])):
        Xs = np.load(C.PROCESSED_DIR / ("X_%s_4d.npy" % name))
        ys = np.load(C.PROCESSED_DIR / ("y_%s.npy" % name))
        # 按被试切分：每人的窗口数不固定，用累计计数
        pos = 0
        for s in subs:
            n_r = 0
            n_f = 0
            # 简化：按标签块切（数据是按 rest 块、focus 块拼接的）
            seg = ys[pos:]
            # 找到 focus 的起始
            f_idx = np.where(seg == 1)[0]
            if len(f_idx) > 0:
                first_f = f_idx[0]
                n_r = first_f
                # focus 块长度
                tail = seg[first_f:]
                n_f = int((tail == 1).sum())
                block = Xs[pos:pos + n_r + n_f]
                counts[s] = (n_r, n_f, block)
                pos += n_r + n_f

    rows = []
    for s in subj_order:
        if s not in counts:
            continue
        n_r, n_f, block = counts[s]
        std = float(block.std())
        # alpha 比（粗略：用 8-13Hz 功率 vs 全带）
        d = block.reshape(block.shape[0], 8, -1)
        fft = np.abs(np.fft.rfft(d, axis=-1)) ** 2
        freqs = np.fft.rfftfreq(d.shape[-1], 1.0 / C.TARGET_SAMPLING_RATE)
        alpha = fft[:, :, (freqs >= 8) & (freqs <= 13)].mean()
        total = fft[:, :, (freqs >= 1) & (freqs <= 45)].mean()
        ar = float(alpha / max(total, 1e-12))
        note = ""
        if abs(std - 12) > 6:
            note += "幅值异常 "
        rows.append((s, n_r + n_f, std, ar, note))

    for s, n, std, ar, note in rows:
        print("%-12s %-8d %-10.2f %-10.4f %s" % (s, n, std, ar, note))

    # ---------- Q3: 结论 ----------
    print()
    print("-" * 74)
    print("Q3  结论")
    print("-" * 74)
    print("  跨被试（固定划分，11/2/2）: BA 0.5549 / 0.5669（全量微调 / 线性探测）")
    print("  随机窗口（泄漏）          : BA %.4f" % m["balanced_accuracy"])
    print()
    if m["balanced_accuracy"] > 0.8:
        print("  => 模型有能力学，跨被试泛化是主要瓶颈（符合小被试数场景的预期）")
    elif m["balanced_accuracy"] > 0.65:
        print("  => 模型能力有限，任务本身可分辨性中等")
    else:
        print("  => 连随机划分都学不好，需检查数据/超参/预处理")


if __name__ == "__main__":
    main()
