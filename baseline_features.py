"""
手工频带特征基线（用于和 LaBraM 做公平对照）

目的：回答一个关键问题——
  LaBraM 的 patch_embed 用 GroupNorm 做样本级归一化，会消除「整体增益」，
  而认知负荷/专注度的 EEG 标志（alpha 抑制、theta 增强）是否依赖这个绝对功率？

做法：在完全相同的 LOSO 协议下，比较三类手工特征与 LaBraM 的跨被试表现。
  A. 绝对频带功率（log）           -> 依赖绝对幅值
  B. 相对频带功率（每通道归一化）  -> 只依赖相对模式（≈ GroupNorm 后的信息）
  C. 频带比值（theta/alpha 等）    -> 临床常用的功率比指标

判读：
  若 A 明显强于 B  -> 绝对功率确实关键，GroupNorm 有害
  若 A ≈ B         -> GroupNorm 影响不大，瓶颈在别处

用法：
    python eeg_attention/baseline_features.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def band_powers(X3, fs):
    """X3: [N, ch, T] -> {band: [N, ch]}"""
    freqs = np.fft.rfftfreq(X3.shape[-1], 1.0 / fs)
    psd = np.abs(np.fft.rfft(X3, axis=-1)) ** 2
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs <= hi)
        out[name] = psd[..., m].mean(axis=-1)
    return out


def build_features(X3, fs, kind):
    bp = band_powers(X3, fs)
    n = X3.shape[0]
    total = sum(bp.values()) + 1e-12

    if kind == "abs":            # A. 绝对功率（log）
        feats = [np.log10(bp[b] + 1e-12) for b in BANDS]
    elif kind == "rel":          # B. 相对功率
        feats = [bp[b] / total for b in BANDS]
    elif kind == "ratio":        # C. 比值
        feats = [
            np.log10((bp["theta"] + 1e-12) / (bp["alpha"] + 1e-12)),
            np.log10((bp["theta"] + 1e-12) / (bp["beta"] + 1e-12)),
            np.log10((bp["alpha"] + 1e-12) / (bp["beta"] + 1e-12)),
            np.log10((bp["delta"] + 1e-12) / (bp["alpha"] + 1e-12)),
        ]
    elif kind == "all":          # 全部拼接
        feats = ([np.log10(bp[b] + 1e-12) for b in BANDS] +
                 [bp[b] / total for b in BANDS] +
                 [np.log10((bp["theta"] + 1e-12) / (bp["alpha"] + 1e-12)),
                  np.log10((bp["theta"] + 1e-12) / (bp["beta"] + 1e-12))])
    else:
        raise ValueError(kind)

    # 每个特征在通道维上取均值与标准差 -> 降维
    cols = []
    for f in feats:
        cols.append(f.mean(axis=1))
        cols.append(f.std(axis=1))
    return np.stack(cols, axis=1)   # [N, n_feat]


def loso_eval(Xf, y, subj_ids):
    """留一被试：每折用 1 名被试做测试"""
    subjects = sorted(set(subj_ids))
    bas, aucs, f1s = [], [], []
    per_subj = []
    for s in subjects:
        te = np.array([i for i, x in enumerate(subj_ids) if x == s])
        tr = np.array([i for i, x in enumerate(subj_ids) if x != s])
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(Xf[tr], y[tr])
        p = clf.predict(Xf[te])
        pr = clf.predict_proba(Xf[te])[:, 1]
        ba = balanced_accuracy_score(y[te], p)
        bas.append(ba)
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
        aucs.append(roc_auc_score(y[te], pr) if len(np.unique(y[te])) > 1 else np.nan)
        per_subj.append((s, ba))
    return (np.array(bas), np.array(f1s), np.array(aucs), per_subj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(C.PROCESSED_DIR))
    args = ap.parse_args()

    d = Path(args.data_dir)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))

    # 合并全部数据，同时保留被试归属
    Xs, ys, subs = [], [], []
    for split in ("train", "val", "test"):
        X = np.load(d / ("X_%s_4d.npy" % split)).reshape(-1, len(C.CHANNEL_NAMES), C.WINDOW_SAMPLES)
        y = np.load(d / ("y_%s.npy" % split))
        # 按被试顺序切分（每被试 rest 块 + focus 块）
        pos = 0
        for s in meta["%s_subjects" % split]:
            seg = y[pos:]
            n_r = 0
            while n_r < len(seg) and seg[n_r] == 0:
                n_r += 1
            n_f = 0
            while n_r + n_f < len(seg) and seg[n_r + n_f] == 1:
                n_f += 1
            if n_r + n_f == 0:
                continue
            Xs.append(X[pos:pos + n_r + n_f])
            ys.append(y[pos:pos + n_r + n_f])
            subs += [s] * (n_r + n_f)
            pos += n_r + n_f

    X3 = np.concatenate(Xs)
    y = np.concatenate(ys)
    subs = np.array(subs)
    print("=" * 72)
    print("手工频带特征基线 vs LaBraM（同一 LOSO 协议，15 折）")
    print("=" * 72)
    print("样本 %d | 被试 %d 人 | rest=%d focus=%d"
          % (len(y), len(set(subs)), (y == 0).sum(), (y == 1).sum()))
    print()

    results = {}
    labels = {
        "abs": "A. 绝对频带功率 (log)",
        "rel": "B. 相对频带功率",
        "ratio": "C. 频带比值 (theta/alpha 等)",
        "all": "D. 全部拼接",
    }
    print("%-28s %-8s %-16s %-10s %s" % ("特征", "维数", "BA (mean±std)", "macroF1", "AUC"))
    print("-" * 72)
    for kind in ("abs", "rel", "ratio", "all"):
        Xf = build_features(X3, C.TARGET_SAMPLING_RATE, kind)
        ba, f1, auc, per = loso_eval(Xf, y, subs)
        results[kind] = {"ba_mean": float(np.mean(ba)), "ba_std": float(np.std(ba)),
                         "f1": float(np.nanmean(f1)), "auc": float(np.nanmean(auc)),
                         "per_subject": [(s, float(b)) for s, b in per]}
        print("%-28s %-8d %-16s %-10.4f %.4f"
              % (labels[kind], Xf.shape[1],
                 "%.4f±%.4f" % (np.mean(ba), np.std(ba)),
                 np.nanmean(f1), np.nanmean(auc)))

    print("-" * 72)
    print("参考（同协议）: LaBraM 线性探测 LOSO = 0.5407 ± 0.0786")
    print()

    best = max(results.items(), key=lambda kv: kv[1]["ba_mean"])
    print("最佳手工特征: %s  BA = %.4f" % (labels[best[0]], best[1]["ba_mean"]))
    gap = best[1]["ba_mean"] - 0.5407
    print("与 LaBraM 的差距: %+.4f" % gap)
    print()
    if best[1]["ba_mean"] > 0.5407 + 0.03:
        print("=> 手工特征显著更强，说明 LaBraM 的表征未充分利用频带功率信息")
    elif abs(gap) <= 0.03:
        print("=> 两者接近，说明瓶颈不在特征表达方式，而在数据本身（被试数/质量）")
    else:
        print("=> LaBraM 更强，预训练表征确实带来增益")

    # 绝对 vs 相对：直接回答 GroupNorm 问题
    print()
    print("-" * 72)
    print("关键对比：绝对功率 vs 相对功率（回答 GroupNorm 问题）")
    print("-" * 72)
    a, b = results["abs"]["ba_mean"], results["rel"]["ba_mean"]
    print("  A 绝对功率 BA = %.4f" % a)
    print("  B 相对功率 BA = %.4f" % b)
    print("  差值 (A - B)  = %+.4f" % (a - b))
    if a - b > 0.03:
        print("  => 绝对功率信息确实有额外价值，样本级归一化会损失这部分信息")
    else:
        print("  => 相对功率已包含绝大部分可用信息，样本级归一化的损失有限")

    out = C.RESULTS_DIR / "baseline_features_loso.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n结果已保存: %s" % out)


if __name__ == "__main__":
    main()
