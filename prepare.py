"""
心算数据集 -> LaBraM 输入格式 预处理
================================================================

严格对齐 LaBraM 官方 dataset_maker/make_TUAB.py：

    1. 选通道
    2. 带通 0.5-75 Hz
    3. 50 Hz 陷波
    4. 重采样 500 -> 200 Hz（一步到位，比例 5:2）
    5. get_data(units='uV')   -> 微伏
    6. 切 2 秒窗口（400 点 = 2 个 patch）

输出（同时给两种形状）：
    X_{split}.npy     [N, 8, 400]     3D，LaBraM 官方 dataset 格式
    X_{split}_4d.npy  [N, 8, 2, 200]  4D，直接喂 model.forward
    y_{split}.npy     [N,]

用法：
    python -m eeg_attention.prepare                 # 用全部完整被试
    python -m eeg_attention.prepare --subjects Subject00 Subject01
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# 允许直接 python eeg_attention/prepare.py 运行
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C            # noqa: E402
import dataset as DS          # noqa: E402


def resolve_input_chans():
    """从 LaBraM utils.standard_1020 实时查出 pos_embed 索引。"""
    chans, src = None, None
    try:
        sys.path.insert(0, str(C.LABRAM_ROOT))
        import utils as labram_utils  # noqa
        table = labram_utils.standard_1020
        chans = [0]
        for name in C.LABRAM_CH_NAMES:
            chans.append(table.index(name) + 1)
        src = "LaBraM utils.standard_1020（实时查询）"
    except Exception as exc:
        print("[WARN] 无法导入 LaBraM utils（%s），改用内置映射表。" % exc)
        chans = [0] + [C.FALLBACK_CHAN_INDEX[n] + 1 for n in C.LABRAM_CH_NAMES]
        src = "内置映射表（已核对源码）"
    return chans, src


def process_one_edf(edf_path, label):
    """EDF -> [n_windows, 8, 2, 200] (uV)"""
    import mne
    mne.set_log_level("ERROR")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # 1) 选通道（按 C.EDF_CHANNELS 顺序；MNE 的 pick 会按给定顺序重排）
    picks = []
    for target in C.EDF_CHANNELS:
        hit = None
        for ch in raw.ch_names:
            clean = ch.replace(" ", "").replace("EEG", "").upper()
            if target.upper() == clean:
                hit = ch
                break
        if hit is None:
            raise RuntimeError("通道 %s 未在 %s 中找到，实际通道: %s"
                               % (target, edf_path.name, raw.ch_names))
        picks.append(hit)
    raw.pick(picks)

    # 2) 带通（同时起抗混叠作用）  3) 陷波  4) 重采样
    raw.filter(l_freq=C.BANDPASS_LOW, h_freq=C.BANDPASS_HIGH, verbose=False)
    raw.notch_filter(C.NOTCH_FREQ, verbose=False)
    raw.resample(C.TARGET_SAMPLING_RATE, n_jobs=1, verbose=False)

    # 5) 取微伏（与官方 make_TUAB.py 完全一致）
    data = raw.get_data(units="uV")

    # 6) 切窗口
    n_ch, n_times = data.shape
    n_windows = n_times // C.WINDOW_SAMPLES
    if n_windows == 0:
        raise RuntimeError("%s 数据不足一个窗口（%d 采样点）" % (edf_path.name, n_times))
    data = data[:, : n_windows * C.WINDOW_SAMPLES]
    windows = data.reshape(n_ch, n_windows, C.WINDOW_SAMPLES).transpose(1, 0, 2)
    windows = windows.reshape(n_windows, n_ch, C.N_PATCHES, C.PATCH_SIZE)

    return windows.astype(np.float32), np.full(n_windows, label, dtype=np.int64)


def compute_class_weights(y):
    """简单反比：weight_i = total / (n_classes * count_i)"""
    counts = np.bincount(y, minlength=C.N_CLASSES)
    total = int(counts.sum())
    weights = total / (C.N_CLASSES * counts)
    return weights, counts, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default=str(C.RAW_EDF_DIR))
    ap.add_argument("--out-dir", type=str, default=str(C.PROCESSED_DIR))
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED)
    ap.add_argument("--split-mode", choices=["fixed", "loso"], default="fixed")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    print("=" * 68)
    print("心算数据集 -> LaBraM 格式预处理")
    print("=" * 68)
    print("LaBraM 根目录 : %s" % C.LABRAM_ROOT)
    print("EDF 输入      : %s" % raw_dir)
    print("输出          : %s" % out_dir)
    print("滤波          : %.1f-%.1f Hz 带通 + %.0f Hz 陷波"
          % (C.BANDPASS_LOW, C.BANDPASS_HIGH, C.NOTCH_FREQ))
    print("重采样        : %d -> %d Hz（一步到位，%d:%d）"
          % (C.RAW_SAMPLING_RATE, C.TARGET_SAMPLING_RATE,
             C.RAW_SAMPLING_RATE, C.TARGET_SAMPLING_RATE))
    print("窗口          : %.1f s = %d 点 = %d patch（patch_size=%d）"
          % (C.WINDOW_SECONDS, C.WINDOW_SAMPLES, C.N_PATCHES, C.PATCH_SIZE))
    print("张量          : [N, %d, %d, %d]"
          % (len(C.CHANNEL_NAMES), C.N_PATCHES, C.PATCH_SIZE))

    complete, partial, missing = DS.discover_subjects(raw_dir, verbose=True)
    if not complete:
        print("[ERROR] 没有找到任何完整被试（rest+focus 都要有），终止。")
        return 1

    input_chans, src = resolve_input_chans()
    print("通道映射（%s）" % src)
    print("  input_chans = %s" % input_chans)
    print("  长度=%d（应=%d）  最大索引=%d（上限 128）"
          % (len(input_chans), len(C.CHANNEL_NAMES) + 1, max(input_chans)))
    if max(input_chans) > 128:
        print("  [ERROR] 索引越界！")
        return 1

    # ---- 划分 ----
    if args.subjects:
        subjects = [s for s in args.subjects if s in complete]
        missing_req = [s for s in args.subjects if s not in complete]
        if missing_req:
            print("[WARN] 以下被试不完整，已跳过: %s" % missing_req)
        splits = {"train": subjects, "val": [], "test": []}
    else:
        if args.split_mode == "loso":
            print("[INFO] LOSO 模式：本脚本只生成单次固定划分，LOSO 请在 finetune 里用 --loso 完成。")
        splits = DS.split_subjects(complete, seed=args.seed, mode="fixed")

    print()
    print("被试划分（被试不重叠）：")
    for k in ("train", "val", "test"):
        print("  %-5s %2d 人: %s" % (k, len(splits[k]), ", ".join(splits[k])))

    # ---- 处理 ----
    all_data, fail = {}, []
    for split_name in ("train", "val", "test"):
        ws, ls = [], []
        for subj in splits[split_name]:
            for suffix, label, tag in ((C.REST_SUFFIX, 0, "rest"), (C.FOCUS_SUFFIX, 1, "focus")):
                edf = raw_dir / (subj + suffix)
                try:
                    w, l = process_one_edf(edf, label)
                except Exception as e:
                    print("  [FAIL] %s %s: %s" % (subj, tag, e))
                    fail.append("%s%s" % (subj, suffix))
                    continue
                ws.append(w)
                ls.append(l)
                print("  [%-5s] %-10s %-5s -> %4d 窗" % (split_name, subj, tag, len(w)))
        if ws:
            X = np.concatenate(ws, axis=0)
            y = np.concatenate(ls, axis=0)
        else:
            X = np.zeros((0, len(C.CHANNEL_NAMES), C.N_PATCHES, C.PATCH_SIZE), dtype=np.float32)
            y = np.zeros((0,), dtype=np.int64)
        all_data[split_name] = (X, y)

    if len(all_data["train"][1]) == 0:
        print("[ERROR] 训练集为空，终止。")
        return 1

    # ---- 类别权重 ----
    weights, counts, total = compute_class_weights(all_data["train"][1])
    print()
    print("-" * 68)
    print("类别权重（简单反比）")
    print("  训练集 %d 样本：rest=%d  focus=%d   比例 %.3f : 1"
          % (total, counts[0], counts[1], counts[0] / max(counts[1], 1)))
    print("  class_weights = [%.6f, %.6f]   (rest, focus)" % (weights[0], weights[1]))
    print("-" * 68)

    # ---- 保存 ----
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        X4d, y = all_data[split_name]
        X3d = X4d.reshape(len(X4d), len(C.CHANNEL_NAMES), C.WINDOW_SAMPLES)
        np.save(out_dir / ("X_%s.npy" % split_name), X3d)
        np.save(out_dir / ("X_%s_4d.npy" % split_name), X4d)
        np.save(out_dir / ("y_%s.npy" % split_name), y)
        print("  保存 X_%-5s %s  X_%-5s_4d %s  y_%-5s %s"
              % (split_name, tuple(X3d.shape), split_name, tuple(X4d.shape),
                 split_name, tuple(y.shape)))

    meta = {
        "dataset_name": "mental_arithmetic_labram",
        "description": "心算数据集（eegmat），LaBraM 输入格式，静息(闭眼) vs 专注(心算) 二分类",
        "prepared_for": "LaBraM-Base finetuning (labram_base_patch200_200)",
        "source_edf_dir": str(raw_dir),
        "preprocessing": {
            "channel_selection": C.CHANNEL_NAMES,
            "edf_channel_names": C.EDF_CHANNELS,
            "bandpass_hz": [C.BANDPASS_LOW, C.BANDPASS_HIGH],
            "notch_hz": C.NOTCH_FREQ,
            "original_sampling_rate": C.RAW_SAMPLING_RATE,
            "target_sampling_rate": C.TARGET_SAMPLING_RATE,
            "units": "uV",
            "unit_conversion": "raw.get_data(units='uV')",
            "normalization": "none（LaBraM patch_embed 内含 GroupNorm）",
            "labram_input_divisor": C.LABRAM_INPUT_DIVISOR,
            "labram_input_divisor_note": (
                "LaBraM 的 engine_for_finetuning.py:64/173、engine_for_pretraining.py:87、"
                "engine_for_vqnsp.py:57 全流程都对输入做 float()/100。"
                "本数据存 uV 原值，训练时必须再 /100，否则编码器特征失真约 36%。"
            ),
            "pipeline_order": [
                "1. pick channels", "2. bandpass", "3. notch",
                "4. resample", "5. get_data(uV)", "6. windowing",
            ],
            "bandwidth_caveat": "原始 EDF 采集时已做 45 Hz 低通，实际有效带宽 0.5-45 Hz",
        },
        "tensor_shape": {
            "X_{split}.npy": "[N, 8, 400]",
            "X_{split}_4d.npy": "[N, 8, 2, 200]",
            "labram_cli_input_size": C.WINDOW_SAMPLES,
        },
        "labram_input_chans": input_chans,
        "labram_ch_names": C.LABRAM_CH_NAMES,
        "labram_input_chans_source": src,
        "class_names": C.CLASS_NAMES,
        "class_names_caveat": "静息态为闭眼(eyes closed)，闭眼会引起 alpha 增强，需在结论中声明",
        "class_weights_method": "inverse",
        "class_weights": [float(weights[0]), float(weights[1])],
        "train_class_distribution": {"rest": int(counts[0]), "focus": int(counts[1])},
        "subjects_complete": complete,
        "subjects_partial": [list(p) for p in partial],
        "subjects_missing": missing,
        "n_subjects_train": len(splits["train"]),
        "n_subjects_val": len(splits["val"]),
        "n_subjects_test": len(splits["test"]),
        "train_subjects": splits["train"],
        "val_subjects": splits["val"],
        "test_subjects": splits["test"],
        "train_samples": int(len(all_data["train"][1])),
        "val_samples": int(len(all_data["val"][1])),
        "test_samples": int(len(all_data["test"][1])),
        "random_seed": args.seed,
        "failed_files": fail,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "channel_order.json").write_text(
        json.dumps({
            "ch_names_for_labram": C.LABRAM_CH_NAMES,
            "input_chans": input_chans,
            "channel_order_in_tensor": C.CHANNEL_NAMES,
            "usage": "input_chans 直接传给 model(x, input_chans=...)，长度须为 1+通道数",
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 68)
    print("完成。输出: %s" % out_dir)
    print("  train %s / val %s / test %s"
          % (tuple(all_data["train"][0].shape),
             tuple(all_data["val"][0].shape),
             tuple(all_data["test"][0].shape)))
    print()
    print("  [!] 训练时输入必须再 /100（LaBraM 全流程统一约定）")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
