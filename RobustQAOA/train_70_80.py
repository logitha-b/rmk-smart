"""
train_70_80.py
==============
Entry point for the 70-80% accuracy experiment.

Usage:
    python train_70_80.py                       # default config
    python train_70_80.py --epochs 150
    python train_70_80.py --force-retrain
    python train_70_80.py --config configs/resnet_baseline.yaml
"""

import os, sys, argparse, json, time, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from experiments.baseline_70_80 import run_baseline_70_80, DEFAULT_CFG


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",         type=str,   default=None)
    p.add_argument("--epochs",         type=int,   default=None)
    p.add_argument("--warmup-epochs",  type=int,   default=None, dest="warmup_epochs")
    p.add_argument("--lr-phase1",      type=float, default=None, dest="lr_phase1")
    p.add_argument("--lr-phase2",      type=float, default=None, dest="lr_phase2")
    p.add_argument("--batch-size",     type=int,   default=None, dest="batch_size")
    p.add_argument("--weight-decay",   type=float, default=None, dest="weight_decay")
    p.add_argument("--mixup-alpha",    type=float, default=None, dest="mixup_alpha")
    p.add_argument("--seed",           type=int,   default=None)
    p.add_argument("--num-workers",    type=int,   default=None, dest="num_workers")
    p.add_argument("--no-pretrained",  action="store_true", dest="no_pretrained")
    p.add_argument("--force-retrain",  action="store_true", dest="force_retrain")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = dict(DEFAULT_CFG)

    if args.config:
        import yaml
        with open(args.config) as f:
            cfg.update(yaml.safe_load(f))

    if args.epochs        is not None: cfg["epochs"]         = args.epochs
    if args.warmup_epochs is not None: cfg["warmup_epochs"]  = args.warmup_epochs
    if args.lr_phase1     is not None: cfg["lr_phase1"]      = args.lr_phase1
    if args.lr_phase2     is not None: cfg["lr_phase2"]      = args.lr_phase2
    if args.batch_size    is not None: cfg["batch_size"]     = args.batch_size
    if args.weight_decay  is not None: cfg["weight_decay"]   = args.weight_decay
    if args.mixup_alpha   is not None: cfg["mixup_alpha"]    = args.mixup_alpha
    if args.seed          is not None: cfg["seed"]           = args.seed
    if args.num_workers   is not None: cfg["num_workers"]    = args.num_workers
    if args.no_pretrained:             cfg["pretrained"]     = False
    if args.force_retrain:             cfg["force_retrain"]  = True

    print("\n" + "="*68)
    print("  RobustQAOA — ResNet18 Prunable Baseline (70-80% Target)")
    print("="*68)
    print(f"  Device      : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"  Epochs      : {cfg['epochs']} (warmup={cfg['warmup_epochs']})")
    print(f"  LR phase1   : {cfg['lr_phase1']}  |  phase2: {cfg['lr_phase2']}")
    print(f"  Batch size  : {cfg['batch_size']}")
    print(f"  MixUp alpha : {cfg['mixup_alpha']}")
    print(f"  Pretrained  : {cfg['pretrained']}")
    print(f"  Force retrain: {cfg['force_retrain']}")
    print("="*68)

    t0 = time.time()
    model, metrics, _ = run_baseline_70_80(cfg)
    total = time.time() - t0

    out_dir = cfg["out_dir"]
    print(f"\n  Total wall-clock time: {total:.0f}s ({total/60:.1f} min)")
    print("\n  Files written:")
    for f in [
        "metrics.json", "metrics.csv", "training_history.csv",
        "training_curve.png", "accuracy_curve.png",
        "experiment_config.json",
        "diagnostic_report.md", "comparison_report.md", "comparison_report.csv",
        "checkpoints/baseline_model.pt",
    ]:
        fp = os.path.join(out_dir, f)
        print(f"    [{'OK' if os.path.exists(fp) else 'MISSING'}] {f}")


if __name__ == "__main__":
    main()
