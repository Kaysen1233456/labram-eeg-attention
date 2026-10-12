"""验证数据本身的可分性：rest(闭眼) vs focus(心算) 的频域差异"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa

d = C.PROCESSED_DIR
X = np.concatenate([np.load(d / ("X_%s_4d.npy" % s)) for s in ("train", "val", "test")])
y = np.concatenate([np.load(d / ("y_%s.npy" % s)) for s in ("train", "val", "test")])
X3 = X.reshape(len(X), 8, -1)   # (N, 8, 400)

print("=" * 74)
print("数据可分性验证（静息=闭眼 vs 专注=心算）")
print("=" * 74)
print("总样本 %d  rest=%d focus=%d" % (len(y), (y == 0).sum(), (y == 1).sum()))

fs = C.TARGET_SAMPLING_RATE
freqs = np.fft.rfftfreq(X3.shape[-1], 1.0 / fs)

def band_power(sig, lo, hi):
    f = np.abs(np.fft.rfft(sig, axis=-1)) ** 2
    m = (freqs >= lo) & (freqs <= hi)
    return f[..., m].mean(axis=-1)

bands = {"delta 1-4": (1, 4), "theta 4-8": (4, 8), "alpha 8-13": (8, 13),
         "beta 13-30": (13, 30), "gamma 30-45": (30, 45)}

print()
print("%-14s %-16s %-16s %-12s %s" % ("频带", "rest 功率(µV²)", "focus 功率(µV²)", "rest/focus", "方向"))
print("-" * 74)
for name, (lo, hi) in bands.items():
    p = band_power(X3, lo, hi)          # (N, 8)
    pr = p[y == 0].mean()
    pf = p[y == 1].mean()
    ratio = pr / max(pf, 1e-12)
    direction = "rest 更强" if ratio > 1.05 else ("focus 更强" if ratio < 0.95 else "无明显差异")
    print("%-14s %-16.4e %-16.4e %-12.3f %s" % (name, pr, pf, ratio, direction))

print()
print("-" * 74)
print("闭眼 alpha 效应检查（最关键的信号）")
print("-" * 74)
p_alpha = band_power(X3, 8, 13)
p_total = band_power(X3, 1, 45)
rel_alpha = p_alpha / np.maximum(p_total, 1e-12)

for lab, name in ((0, "rest (闭眼)"), (1, "focus (心算)")):
    m = rel_alpha[y == lab]
    print("  %-14s 相对 alpha = %.4f ± %.4f   绝对 alpha = %.4e"
          % (name, m.mean(), m.std(), p_alpha[y == lab].mean()))
r = rel_alpha[y == 0].mean() / max(rel_alpha[y == 1].mean(), 1e-12)
print("  => 相对 alpha 比值 (rest/focus) = %.3f" % r)
if r > 1.2:
    print("  => 闭眼 alpha 效应明显，标签方向正确，数据本身有强可分信号")
elif r > 1.05:
    print("  => 闭眼 alpha 效应存在但较弱")
else:
    print("  => 未见明显闭眼 alpha 效应，需检查标签或预处理")

# ---- 单特征可分性：用相对 alpha 直接做分类 ----
print()
print("-" * 74)
print("最简单的基线：仅用「相对 alpha」单特征做阈值分类")
print("-" * 74)
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
feat = rel_alpha.mean(axis=1).reshape(-1, 1)   # (N,)
# 用中位数当阈值（无信息泄漏的朴素做法）
thr = np.median(feat)
pred = (feat < thr).astype(int)   # alpha 低 -> focus
print("  阈值 = %.4f" % thr)
print("  BA = %.4f   AUC(单特征) = %.4f"
      % (balanced_accuracy_score(y, pred), roc_auc_score(y, feat)))

# ---- 被试级可分性 ----
print()
print("-" * 74)
print("被试级：每人 rest/focus 的相对 alpha 均值")
print("-" * 74)
meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
print("%-12s %-14s %-14s %-10s" % ("被试", "rest alpha", "focus alpha", "rest/focus"))
print("-" * 74)
for split, subs in (("train", meta["train_subjects"]), ("val", meta["val_subjects"]),
                    ("test", meta["test_subjects"])):
    Xs = np.load(d / ("X_%s_4d.npy" % split)).reshape(-1, 8, 400)
    ys = np.load(d / ("y_%s.npy" % split))
    pos = 0
    for s in subs:
        # 连续块切分
        seg_y = ys[pos:]
        n_r = 0
        while n_r < len(seg_y) and seg_y[n_r] == 0:
            n_r += 1
        n_f = 0
        while n_r + n_f < len(seg_y) and seg_y[n_r + n_f] == 1:
            n_f += 1
        if n_r == 0 or n_f == 0:
            pos += n_r + n_f
            continue
        blk = Xs[pos:pos + n_r + n_f]
        ra = (band_power(blk, 8, 13) / np.maximum(band_power(blk, 1, 45), 1e-12)).mean(axis=1)
        rr = ra[:n_r].mean()
        rf = ra[n_r:n_r + n_f].mean()
        flag = "" if rr > rf else "  <-- 方向反了"
        print("%-12s %-14.4f %-14.4f %-10.3f%s"
              % (s + "[" + split[0] + "]", rr, rf, rr / max(rf, 1e-12), flag))
        pos += n_r + n_f
