import os
import json
import numpy as np
import torch
from config import CFG
from train import run_training


def load_class_weights():
    cw = np.load(CFG.class_weights_path)
    return torch.tensor(cw, dtype=torch.float32)


def fold_done(fold):
    """Fold selesai = checkpoint DAN log ada."""
    ckpt = f"{CFG.save_dir}/fold{fold}.pt"
    log = f"{CFG.save_dir}/fold{fold}_log.json"
    return os.path.exists(ckpt) and os.path.exists(log)


def run_folds(folds_to_run):
    class_weights = load_class_weights()
    results = {}
    for fold in folds_to_run:
        if fold_done(fold):
            with open(f"{CFG.save_dir}/fold{fold}_log.json") as f:
                prev = json.load(f)
            results[fold] = prev["best_macro_f1"]
            print(f"⏭️  fold {fold} sudah selesai (f1 {results[fold]:.4f}) — skip")
            continue
        print(f"\n{'='*60}\n🚀 FOLD {fold}\n{'='*60}")
        best_f1, mins = run_training(fold=fold, cfg=CFG, class_weights=class_weights)
        log = {
            "fold": fold,
            "best_macro_f1": float(best_f1),
            "minutes_per_epoch": mins,
            "accum_steps": CFG.accum_steps,
        }
        with open(f"{CFG.save_dir}/fold{fold}_log.json", "w") as f:
            json.dump(log, f, indent=2)
        results[fold] = float(best_f1)
    return results


if __name__ == "__main__":
    results = run_folds([0, 1, 2, 3, 4])
    print("\n📊 Hasil semua fold:", results)
