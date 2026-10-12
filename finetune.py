"""
LaBraM-Base 微调：心算数据集（静息 vs 专注）
================================================================

方案 A：自包含训练脚本，类别权重直接落地在这里。

本脚本同时演示 LaBraM 的三个必须项：
  1. 输入尺度  : uV / 100（官方 engine 全流程约定）
  2. 通道顺序  : input_chans（pos_embed 索引）
  3. 类别权重  : nn.CrossEntropyLoss(weight=...)

用法：
    python eeg_attention/finetune.py                          # 固定划分，加权
    python eeg_attention/finetune.py --no-class-weights       # 对照：不加权
    python eeg_attention/finetune.py --loso                   # 留一被试（15 折）
    python eeg_attention/finetune.py --seeds 42 43 44         # 多 seed
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C        # noqa: E402
import dataset as DS      # noqa: E402


# ==================== 模型 ====================

def build_model(num_classes, device, verbose=True, drop_path=0.1):
    sys.path.insert(0, str(C.LABRAM_ROOT))
    import modeling_finetune  # noqa

    model = modeling_finetune.labram_base_patch200_200(
        num_classes=num_classes, drop_path_rate=drop_path,
        use_mean_pooling=True, init_scale=0.001, init_values=0.1)

    ck = torch.load(str(C.LABRAM_CKPT), map_location="cpu")
    sd = {k[len("student."):]: v for k, v in ck["model"].items() if k.startswith("student.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        new_layers = [m for m in missing if not m.startswith("head.")]
        print("  [权重] 预训练编码器已加载；新增层(非head): %s" % (new_layers or "无"))
        print("  [权重] 丢弃预训练多余键: %d 个（mask_token/lm_head/norm）" % len(unexpected))
    return model.to(device)


# ==================== 评测 ====================

@torch.no_grad()
def evaluate(model, loader, device, input_chans, criterion):
    from sklearn.metrics import (balanced_accuracy_score, f1_score,
                                 roc_auc_score, confusion_matrix)

    if loader is None or len(loader.dataset) == 0:
        return {}

    model.eval()
    logits_all, y_all = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True) / C.LABRAM_INPUT_DIVISOR
        y = y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(x, input_chans=input_chans)
            loss = criterion(out, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        logits_all.append(out.float().cpu())
        y_all.append(y.cpu())

    logits = torch.cat(logits_all).numpy()
    y_true = torch.cat(y_all).numpy()
    prob = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    pred = logits.argmax(1)

    m = {"loss": total_loss / max(n, 1), "n": int(n)}
    m["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred))
    m["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
    m["accuracy"] = float((pred == y_true).mean())
    m["roc_auc"] = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan")
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    m["recall_rest"] = float(cm[0, 0] / max(cm[0].sum(), 1))
    m["recall_focus"] = float(cm[1, 1] / max(cm[1].sum(), 1))
    m["confusion_matrix"] = cm.tolist()
    return m


# ==================== 单次训练 ====================


def prepare_all_subjects(raw_dir, subjects, verbose=True):
    """一次性处理所有被试的 EDF，返回 {subj: (X, y)}，供 LOSO 复用。"""
    import prepare as P
    cache = {}
    for s in subjects:
        ws, ls = [], []
        for suffix, label in ((C.REST_SUFFIX, 0), (C.FOCUS_SUFFIX, 1)):
            f = Path(raw_dir) / (s + suffix)
            if not f.exists():
                continue
            w, l = P.process_one_edf(f, label)
            ws.append(w)
            ls.append(l)
        if ws:
            cache[s] = (np.concatenate(ws), np.concatenate(ls))
            if verbose:
                print("    [缓存] %-10s %4d 窗 (rest=%d focus=%d)"
                      % (s, len(cache[s][1]), (cache[s][1] == 0).sum(), (cache[s][1] == 1).sum()))
    return cache


def stack_subjects(cache, subj_list):
    """把若干被试的数据拼成一个张量。"""
    ws, ls = [], []
    for s in subj_list:
        if s in cache:
            ws.append(cache[s][0])
            ls.append(cache[s][1])
    if not ws:
        return (torch.zeros((0, len(C.CHANNEL_NAMES), C.N_PATCHES, C.PATCH_SIZE)),
                torch.zeros((0,), dtype=torch.long))
    return (torch.tensor(np.concatenate(ws), dtype=torch.float32),
            torch.tensor(np.concatenate(ls), dtype=torch.long))


def run_once(args, train_subj, val_subj, test_subj, seed, tag="", cache=None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dir = Path(args.raw_dir)

    def build(subj_list):
        ws, ls = [], []
        for s in subj_list:
            for suffix, label in ((C.REST_SUFFIX, 0), (C.FOCUS_SUFFIX, 1)):
                f = raw_dir / (s + suffix)
                if not f.exists():
                    continue
                import prepare as P
                w, l = P.process_one_edf(f, label)
                ws.append(w)
                ls.append(l)
        if not ws:
            return (torch.zeros((0, len(C.CHANNEL_NAMES), C.N_PATCHES, C.PATCH_SIZE)),
                    torch.zeros((0,), dtype=torch.long))
        return (torch.tensor(np.concatenate(ws), dtype=torch.float32),
                torch.tensor(np.concatenate(ls), dtype=torch.long))

    if cache is not None:
        Xtr, ytr = stack_subjects(cache, train_subj)
        Xva, yva = stack_subjects(cache, val_subj)
        Xte, yte = stack_subjects(cache, test_subj)
        import prepare as P2
        input_chans, _ = P2.resolve_input_chans()
        cw_t, _, _ = P2.compute_class_weights(ytr.numpy())
        cw = [float(cw_t[0]), float(cw_t[1])]
        print("  [数据] train=%d val=%d test=%d (内存缓存)" % (len(ytr), len(yva), len(yte)))
    elif args.use_cache and C.PROCESSED_DIR.exists() and not tag:
        splits, meta, ch_order = DS.load_splits(C.PROCESSED_DIR)
        Xtr = torch.tensor(splits["train"][0], dtype=torch.float32)
        ytr = torch.tensor(splits["train"][1], dtype=torch.long)
        Xva = torch.tensor(splits["val"][0], dtype=torch.float32)
        yva = torch.tensor(splits["val"][1], dtype=torch.long)
        Xte = torch.tensor(splits["test"][0], dtype=torch.float32)
        yte = torch.tensor(splits["test"][1], dtype=torch.long)
        input_chans = ch_order["input_chans"]
        cw = meta["class_weights"]
        print("  [数据] 使用缓存: train=%d val=%d test=%d" % (len(ytr), len(yva), len(yte)))
    else:
        print("  [数据] 现场处理 EDF ...")
        Xtr, ytr = build(train_subj)
        Xva, yva = build(val_subj)
        Xte, yte = build(test_subj)
        input_chans, _ = None, None
        import prepare as P
        input_chans, _ = P.resolve_input_chans()
        cw_t, cnt, tot = P.compute_class_weights(ytr.numpy())
        cw = [float(cw_t[0]), float(cw_t[1])]
        print("  [数据] train=%d val=%d test=%d" % (len(ytr), len(yva), len(yte)))

    from torch.utils.data import TensorDataset
    dl_tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=args.batch_size,
                       shuffle=True, num_workers=0, drop_last=False)
    dl_va = DataLoader(TensorDataset(Xva, yva), batch_size=args.batch_size,
                       shuffle=False, num_workers=0) if len(yva) else None
    dl_te = DataLoader(TensorDataset(Xte, yte), batch_size=args.batch_size,
                       shuffle=False, num_workers=0) if len(yte) else None

    model = build_model(C.N_CLASSES, device, drop_path=args.drop_path)

    # ---- 线性探测模式：冻结预训练编码器，只训 head + fc_norm ----
    if args.freeze_encoder:
        for name, p in model.named_parameters():
            if not (name.startswith("head.") or name.startswith("fc_norm.")):
                p.requires_grad = False
        print("  [模式] 线性探测：编码器已冻结")

    head_p, enc_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_p if (name.startswith("head.") or name.startswith("fc_norm.")) else enc_p).append(p)
    print("  [模型] 可训练: head/fc_norm=%.1fK  encoder=%.2fM"
          % (sum(p.numel() for p in head_p) / 1e3, sum(p.numel() for p in enc_p) / 1e6))

    groups = [{"params": head_p, "lr": args.lr_head}]
    if enc_p:
        groups.append({"params": enc_p, "lr": args.lr_encoder})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
        print("  [损失] CrossEntropyLoss()  无权重（对照）")
    else:
        w = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
        print("  [损失] CrossEntropyLoss(weight=[%.4f, %.4f])" % (cw[0], cw[1]))

    best = {"balanced_accuracy": -1.0, "epoch": -1, "state": None, "metrics": None}
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True) / C.LABRAM_INPUT_DIVISOR   # uV / 100
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(x, input_chans=input_chans)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * y.size(0)
            n += y.size(0)
        sched.step()

        msg = "  ep %2d/%d  loss %.4f" % (ep, args.epochs, tot / max(n, 1))
        if dl_va is not None:
            vm = evaluate(model, dl_va, device, input_chans, criterion)
            history.append(vm)
            msg += "  | val BA %.4f  F1 %.4f  AUC %.4f" % (
                vm.get("balanced_accuracy", float("nan")),
                vm.get("macro_f1", float("nan")),
                vm.get("roc_auc", float("nan")))
            if vm.get("balanced_accuracy", -1) > best["balanced_accuracy"]:
                best = {"balanced_accuracy": vm["balanced_accuracy"], "epoch": ep,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "metrics": vm}
        msg += "  (%.1fs)" % (time.time() - t0)
        print(msg)

    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print("  -> 最佳 epoch %d, val BA %.4f" % (best["epoch"], best["balanced_accuracy"]))

    res = {"tag": tag, "seed": seed, "best_epoch": best["epoch"],
           "val": best.get("metrics"), "history": history,
           "class_weights": cw, "use_class_weights": not args.no_class_weights}
    if dl_te is not None:
        res["test"] = evaluate(model, dl_te, device, input_chans, criterion)
        t = res["test"]
        print("  -> [测试] BA %.4f | macroF1 %.4f | AUC %.4f | acc %.4f | 召回 rest %.3f focus %.3f"
              % (t.get("balanced_accuracy", float("nan")), t.get("macro_f1", float("nan")),
                 t.get("roc_auc", float("nan")), t.get("accuracy", float("nan")),
                 t.get("recall_rest", float("nan")), t.get("recall_focus", float("nan"))))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, default=str(C.RAW_EDF_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-encoder", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--no-class-weights", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="线性探测：冻结预训练编码器，只训 head/fc_norm")
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--loso", action="store_true", help="留一被试交叉验证")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--use-cache", action="store_true", default=True)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    print("=" * 70)
    print("LaBraM-Base 微调 | 心算数据集 | 方案A(自包含脚本)")
    print("=" * 70)
    print("设备: %s | 类别加权: %s | 模式: %s"
          % ("cuda" if torch.cuda.is_available() else "cpu",
             "关闭" if args.no_class_weights else "开启",
             "LOSO" if args.loso else "固定划分"))

    complete, _, _ = DS.discover_subjects(args.raw_dir, verbose=False)
    if not complete:
        print("[ERROR] 没有完整被试")
        return 1

    all_results = []

    print("\n预处理所有被试（只做一次，供各折复用）...")
    cache = prepare_all_subjects(args.raw_dir, complete, verbose=True)
    print("共缓存 %d 名被试\n" % len(cache))

    if args.loso:
        folds = DS.split_subjects(complete, mode="loso")
        print("LOSO 折数: %d" % len(folds))
        for fi, fold in enumerate(folds):
            print("\n" + "-" * 70)
            print("Fold %d/%d | test=%s" % (fi + 1, len(folds), fold["test"]))
            print("-" * 70)
            r = run_once(args, fold["train"], fold["val"], fold["test"],
                         seed=args.seeds[0], tag="fold%d" % (fi + 1), cache=cache)
            all_results.append(r)
    else:
        splits = DS.split_subjects(complete, seed=args.seeds[0], mode="fixed")
        print("划分: train=%d val=%d test=%d"
              % (len(splits["train"]), len(splits["val"]), len(splits["test"])))
        for sd in args.seeds:
            print("\n" + "-" * 70)
            print("seed = %d" % sd)
            print("-" * 70)
            r = run_once(args, splits["train"], splits["val"], splits["test"],
                         seed=sd, tag="seed%d" % sd, cache=cache)
            all_results.append(r)

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    if args.loso:
        bas = [r["test"]["balanced_accuracy"] for r in all_results if r.get("test")]
        aucs = [r["test"]["roc_auc"] for r in all_results if r.get("test")]
        f1s = [r["test"]["macro_f1"] for r in all_results if r.get("test")]
        print("LOSO (%d 折)  测试集:" % len(bas))
        print("  BA     = %.4f ± %.4f   (min %.4f, max %.4f)"
              % (np.mean(bas), np.std(bas), np.min(bas), np.max(bas)))
        print("  macroF1= %.4f ± %.4f" % (np.mean(f1s), np.std(f1s)))
        print("  AUC    = %.4f ± %.4f" % (np.mean(aucs), np.std(aucs)))
        summary = {"mode": "loso", "n_folds": len(bas),
                   "ba_mean": float(np.mean(bas)), "ba_std": float(np.std(bas)),
                   "f1_mean": float(np.mean(f1s)), "auc_mean": float(np.mean(aucs)),
                   "folds": all_results}
    else:
        for r in all_results:
            t = r.get("test") or {}
            print("seed %-4d  test BA %.4f  macroF1 %.4f  AUC %.4f"
                  % (r["seed"], t.get("balanced_accuracy", float("nan")),
                     t.get("macro_f1", float("nan")), t.get("roc_auc", float("nan"))))
        bas = [r["test"]["balanced_accuracy"] for r in all_results if r.get("test")]
        if bas:
            print("  平均 BA = %.4f ± %.4f" % (np.mean(bas), np.std(bas)))
        summary = {"mode": "fixed", "results": all_results,
                   "ba_mean": float(np.mean(bas)) if bas else None,
                   "ba_std": float(np.std(bas)) if bas else None}

    out = Path(args.out) if args.out else (
        C.RESULTS_DIR / ("finetune_%s_%s_cw%s.json"
                         % ("loso" if args.loso else "fixed",
                            "lp" if args.freeze_encoder else "ft",
                            "off" if args.no_class_weights else "on")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n结果已保存: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
