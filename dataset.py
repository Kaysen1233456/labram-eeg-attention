"""
数据集发现、被试划分与 Dataset 类
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import config as C


# ==================== 数据发现 ====================

def discover_subjects(raw_dir=None, verbose=True):
    """
    扫描 EDF 目录，返回可用于二分类的完整被试列表（rest + focus 都有）。

    Returns:
        complete : list[str]  两个文件都齐的被试
        partial  : list[tuple] (subj, has_rest, has_focus)  只缺一个的
        missing  : list[str]  两个都缺的
    """
    raw_dir = Path(raw_dir or C.RAW_EDF_DIR)
    if not raw_dir.exists():
        raise FileNotFoundError("EDF 目录不存在: %s" % raw_dir)

    complete, partial, missing = [], [], []
    for i in range(C.SUBJECT_START, C.SUBJECT_END + 1):
        subj = "Subject%02d" % i
        has_r = (raw_dir / (subj + C.REST_SUFFIX)).exists()
        has_f = (raw_dir / (subj + C.FOCUS_SUFFIX)).exists()
        if has_r and has_f:
            complete.append(subj)
        elif has_r or has_f:
            partial.append((subj, has_r, has_f))
        else:
            missing.append(subj)

    if verbose:
        print("-" * 68)
        print("EDF 数据发现 | %s" % raw_dir)
        print("-" * 68)
        print("  完整被试（rest+focus 齐全）: %d 人 -> %s"
              % (len(complete), ", ".join(complete) if len(complete) <= 20 else "见列表"))
        if partial:
            print("  仅部分文件: %d 人" % len(partial))
            for s, r, f in partial:
                print("     %s  rest=%s focus=%s" % (s, "有" if r else "缺", "有" if f else "缺"))
        print("  完全缺失: %d 人" % len(missing))
        if missing:
            print("     %s" % ", ".join(missing))
        print("  => 可用于二分类的被试数: %d" % len(complete))
        print("-" * 68)

    return complete, partial, missing


# ==================== 被试划分 ====================

def split_subjects(complete, n_train=None, n_val=None, n_test=None,
                   seed=None, mode="fixed"):
    """
    被试不重叠划分。mode="fixed" 为固定划分；mode="loso" 返回每折 (train, test)。
    """
    n_train = n_train if n_train is not None else C.N_TRAIN_SUBJECTS
    n_val = n_val if n_val is not None else C.N_VAL_SUBJECTS
    n_test = n_test if n_test is not None else C.N_TEST_SUBJECTS
    seed = seed if seed is not None else C.RANDOM_SEED

    subjects = list(complete)
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    if mode == "loso":
        folds = []
        for i in range(len(subjects)):
            test = [subjects[i]]
            rest = [s for j, s in enumerate(subjects) if j != i]
            # 从剩余里切 2 人做 val
            val = rest[:2]
            train = rest[2:]
            folds.append({"train": train, "val": val, "test": test})
        return folds

    need = n_train + n_val + n_test
    if need > len(subjects):
        # 自动按比例缩放
        total = len(subjects)
        n_val = max(1, int(round(total * 0.15)))
        n_test = max(1, int(round(total * 0.15)))
        n_train = total - n_val - n_test
        print("[提示] 被试数不足，划分自动调整为 %d/%d/%d" % (n_train, n_val, n_test))

    train = subjects[:n_train]
    val = subjects[n_train:n_train + n_val]
    test = subjects[n_train + n_val:n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


# ==================== Dataset ====================

class MentalArithmeticDataset(Dataset):
    """
    读预处理好的 npy。X 为 4D [N, 8, 2, 200]（µV 原值）。

    注意：不在这里除 100 —— 除以 100 在训练循环里做，与 LaBraM 官方 engine 一致。
    """

    def __init__(self, x_path, y_path):
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        if len(self.X) != len(self.y):
            raise ValueError("X/y 长度不一致: %d vs %d" % (len(self.X), len(self.y)))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.X[i], dtype=torch.float32), int(self.y[i])


def load_splits(processed_dir=None):
    """读回预处理产物，返回 dict: split -> (X, y) 与 metadata/channel_order。"""
    d = Path(processed_dir or C.PROCESSED_DIR)
    out = {}
    for name in ("train", "val", "test"):
        xp = d / ("X_%s_4d.npy" % name)
        yp = d / ("y_%s.npy" % name)
        if xp.exists() and yp.exists():
            out[name] = (np.load(xp), np.load(yp))
        else:
            out[name] = (np.zeros((0, len(C.CHANNEL_NAMES), C.N_PATCHES, C.PATCH_SIZE),
                                  dtype=np.float32),
                         np.zeros((0,), dtype=np.int64))
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8")) \
        if (d / "metadata.json").exists() else {}
    ch_order = json.loads((d / "channel_order.json").read_text(encoding="utf-8")) \
        if (d / "channel_order.json").exists() else {}
    return out, meta, ch_order
