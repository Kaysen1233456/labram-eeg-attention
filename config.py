"""
集中配置：心算数据集 -> LaBraM 微调

所有路径均以本文件为锚点推导，脚本可任意目录调用。
"""

from pathlib import Path

# ==================== 路径 ====================
# eeg_attention/config.py  ->  eeg_attention/  ->  LaBraM/
EEG_ATTN_DIR = Path(__file__).resolve().parent
LABRAM_ROOT = EEG_ATTN_DIR.parent

# LaBraM 官方资源
LABRAM_MODELING = LABRAM_ROOT / "modeling_finetune.py"
LABRAM_UTILS = LABRAM_ROOT / "utils.py"
LABRAM_CKPT = LABRAM_ROOT / "checkpoints" / "labram-base.pth"

# 数据集
RAW_EDF_DIR = LABRAM_ROOT / "data" / "xin suan"
PROCESSED_DIR = LABRAM_ROOT / "data" / "processed" / "mental_arithmetic_labram_200hz_8ch"
RESULTS_DIR = EEG_ATTN_DIR / "results"

# ==================== 通道 ====================
# EDF 中的通道名（旧命名 T3/T4，含 'EEG ' 前缀）
EDF_CHANNELS = ["Fz", "F3", "F4", "T3", "T4", "Pz", "P3", "P4"]

# 张量中的通道顺序（新命名 T7/T8，与项目原 metadata 一致）
CHANNEL_NAMES = ["Fz", "F3", "F4", "T7", "T8", "Pz", "P3", "P4"]

# LaBraM standard_1020 中的大写名（供 get_input_chans 查表）
LABRAM_CH_NAMES = ["FZ", "F3", "F4", "T7", "T8", "PZ", "P3", "P4"]

# 兜底映射（LaBraM utils 不可用时的硬编码，已逐行核对源码）
FALLBACK_CHAN_INDEX = {
    "FZ": 19, "F3": 17, "F4": 21, "T7": 37,
    "T8": 45, "PZ": 63, "P3": 61, "P4": 65,
}

# ==================== 预处理参数 ====================
# 严格对齐 LaBraM 官方 dataset_maker/make_TUAB.py
RAW_SAMPLING_RATE = 500
TARGET_SAMPLING_RATE = 200
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 75.0
NOTCH_FREQ = 50.0

WINDOW_SECONDS = 2.0
WINDOW_SAMPLES = int(TARGET_SAMPLING_RATE * WINDOW_SECONDS)   # 400
PATCH_SIZE = 200                                              # LaBraM 固定
N_PATCHES = WINDOW_SAMPLES // PATCH_SIZE                      # 2

# LaBraM engine 全流程会做 input / 100，务必保持一致
LABRAM_INPUT_DIVISOR = 100.0

# ==================== 被试与划分 ====================
# EDF 命名：SubjectNN_1.edf = 静息态(闭眼)，SubjectNN_2.edf = 专注态(心算)
REST_SUFFIX = "_1.edf"
FOCUS_SUFFIX = "_2.edf"

SUBJECT_START = 0
SUBJECT_END = 35

RANDOM_SEED = 42
# 固定划分：15 名完整被试 -> 11 / 2 / 2
N_TRAIN_SUBJECTS = 11
N_VAL_SUBJECTS = 2
N_TEST_SUBJECTS = 2

# 类别定义
CLASS_NAMES = ["rest (eyes closed baseline)", "focus (mental arithmetic)"]
N_CLASSES = 2
