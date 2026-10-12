"""
方案 B'：直接用 LaBraM 官方 engine，但**不修改官方任何文件**

比 patch `run_class_finetuning.py` 更干净：
  - 复用官方 train_one_epoch / evaluate（含 /100、rearrange、指标计算）
  - 复用官方 optim_factory（可选 layer_decay）
  - 类别权重通过 criterion 参数注入（官方 train_one_epoch 支持外部传入）
  - 官方仓库保持零改动

与方案 A（finetune.py）的区别：
  - 方案 A：完全自研训练循环，轻量、易改、便于接自研评测框架
  - 方案 B'：走官方 engine，指标口径与官方论文完全一致

用法：
    python eeg_attention/finetune_official_engine.py --epochs 30
    python eeg_attention/finetune_official_engine.py --epochs 30 --layer-decay 0.65
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C        # noqa: E402
import dataset as DS      # noqa: E402

sys.path.insert(0, str(C.LABRAM_ROOT))
import utils as labram_utils                      # noqa: E402
import optim_factory                              # noqa: E402
from engine_for_finetuning import train_one_epoch, evaluate  # noqa: E402


class Args:
    """给官方 create_optimizer 用的最小 args 对象。"""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--layer-decay", type=float, default=1.0,
                    help="<1.0 时启用官方分层学习率衰减（如 0.65）")
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("方案 B'：官方 engine + 不改官方源码")
    print("=" * 72)
    print("设备: %s | epochs: %d | lr: %g | layer_decay: %g"
          % (device, args.epochs, args.lr, args.layer_decay))

    # ---- 数据 ----
    splits, meta, ch_order = DS.load_splits(C.PROCESSED_DIR)
    ch_names = C.LABRAM_CH_NAMES          # ['FZ','F3','F4','T7','T8','PZ','P3','P4']
    print("ch_names: %s" % ch_names)
    print("  -> utils.get_input_chans 会算出: %s" % labram_utils.get_input_chans(ch_names))

    def loader(split, shuffle):
        X, y = splits[split]
        if len(y) == 0:
            return None
        # 官方 engine 期望 3D [N, 8, 400]，它会自己做 rearrange 与 /100
        X3 = torch.tensor(X.reshape(len(X), len(C.CHANNEL_NAMES), C.WINDOW_SAMPLES),
                          dtype=torch.float32)
        ds = TensorDataset(X3, torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, drop_last=False)

    dl_train = loader("train", True)
    dl_val = loader("val", False)
    dl_test = loader("test", False)
    print("样本数: train=%d val=%d test=%d"
          % (len(dl_train.dataset), len(dl_val.dataset) if dl_val else 0,
             len(dl_test.dataset) if dl_test else 0))

    # ---- 模型 ----
    import modeling_finetune
    model = modeling_finetune.labram_base_patch200_200(
        num_classes=C.N_CLASSES, drop_path_rate=0.1,
        use_mean_pooling=True, init_scale=0.001, init_values=0.1)
    ck = torch.load(str(C.LABRAM_CKPT), map_location="cpu")
    sd = {k[len("student."):]: v for k, v in ck["model"].items() if k.startswith("student.")}
    model.load_state_dict(sd, strict=False)
    model.to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print("模型参数量: %.2fM" % (n_par / 1e6))

    # ---- 优化器（可选官方 layer_decay）----
    a = Args(lr=args.lr, weight_decay=args.weight_decay, layer_decay=args.layer_decay,
             opt='adamw', opt_eps=1e-8, opt_betas=None, filter_bias_and_bn=True,
             model='labram_base_patch200_200', epochs=args.epochs)
    if args.layer_decay < 1.0:
        get_num_layer = optim_factory.get_num_layer_for_vit
        param_groups = optim_factory.get_parameter_groups(
            model, args.weight_decay, get_num_layer=get_num_layer,
            get_layer_scale=lambda x: args.layer_decay)
        print("已启用官方 layer_decay = %g" % args.layer_decay)
    else:
        param_groups = optim_factory.add_weight_decay(model, args.weight_decay) \
            if hasattr(optim_factory, "add_weight_decay") \
            else [{"params": [p for p in model.parameters() if p.requires_grad],
                   "weight_decay": args.weight_decay}]
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))

    loss_scaler = labram_utils.NativeScalerWithGradNormCount()

    n_steps = len(dl_train)
    # cosine_scheduler 要求 warmup_epochs * n_steps <= epochs * n_steps，否则断言失败
    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    if warmup_epochs != args.warmup_epochs:
        print("[提示] warmup_epochs 从 %d 调整为 %d（不能超过总 epoch 数）"
              % (args.warmup_epochs, warmup_epochs))
    lr_schedule = labram_utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, n_steps,
        warmup_epochs=warmup_epochs, start_warmup_value=0)

    # ---- 损失函数（类别权重在这里注入）----
    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("损失: CrossEntropyLoss() 无权重（对照）")
    else:
        w = torch.tensor(meta["class_weights"], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        print("损失: CrossEntropyLoss(weight=[%.4f, %.4f])" % (w[0].item(), w[1].item()))

    metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]

    # ---- 训练 ----
    best = {"ba": -1.0, "epoch": -1, "state": None}
    history = []
    for ep in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, dl_train, optimizer, device, ep, loss_scaler,
            max_norm=None, model_ema=None, log_writer=None,
            start_steps=0, lr_schedule_values=lr_schedule, wd_schedule_values=None,
            num_training_steps_per_epoch=n_steps, update_freq=1,
            ch_names=ch_names, is_binary=False)   # False -> 走 CrossEntropy 路径

        msg = "ep %2d/%d  loss %.4f" % (ep + 1, args.epochs, train_stats.get("loss", float("nan")))
        if dl_val is not None:
            vs = evaluate(dl_val, model, device, header="Val:", ch_names=ch_names,
                          metrics=metrics, is_binary=False)
            history.append(vs)
            msg += "  | val BA %.4f  acc %.4f  kappa %.4f" % (
                vs.get("balanced_accuracy", float("nan")),
                vs.get("accuracy", float("nan")),
                vs.get("cohen_kappa", float("nan")))
            if vs.get("balanced_accuracy", -1) > best["ba"]:
                best = {"ba": vs["balanced_accuracy"], "epoch": ep + 1,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "metrics": vs}
        print(msg)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print("最佳 epoch %d, val BA %.4f" % (best["epoch"], best["ba"]))

    result = {"mode": "official_engine", "args": vars(args),
              "best_epoch": best["epoch"], "val": best.get("metrics"), "history": history}

    if dl_test is not None:
        ts = evaluate(dl_test, model, device, header="Test:", ch_names=ch_names,
                      metrics=metrics, is_binary=False)
        result["test"] = ts
        print("\n[测试] BA %.4f | acc %.4f | kappa %.4f | f1w %.4f"
              % (ts.get("balanced_accuracy", float("nan")), ts.get("accuracy", float("nan")),
                 ts.get("cohen_kappa", float("nan")), ts.get("f1_weighted", float("nan"))))

    out = Path(args.out) if args.out else (
        C.RESULTS_DIR / ("official_engine_cw%s.json" % ("off" if args.no_class_weights else "on")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("结果已保存: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
