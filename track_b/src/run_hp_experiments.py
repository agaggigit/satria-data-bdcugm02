"""run_hp_experiments.py — Eksperimen HP di fold 0 saja (Task 7, Fase 1 plan v2).

SATU VARIABEL PER RUN. Dibandingkan terhadap baseline Fase 1 yang sudah ada:
batch=64, grad_checkpointing off, 8 epoch tanpa early stop -> val_f1 = 0.9520
(lihat track_b/outputs/fold0_history.json + track_b/results/fold0_curves.png).

Urutan RUNS mengikuti rasio dampak/biaya dari plan: epoch+early-stop dulu
(hampir gratis), lalu img288 (backbone di-finetune utk test di 288), lalu LLRD
(bonus kecil tapi konsisten). vflip TIDAK diuji di sini -- augmentasi
RandomVerticalFlip(p=0.2) sudah hardcode di track_a/src/dataset.py, Track B
tidak bisa mematikan/menyalakannya tanpa koordinasi breaking-change ke Track A.

Jalankan dari track_b/src/ (sys.path & cwd harus di sini, sama seperti cell
notebook 0d) supaya import flat (`from config import CFG`) jalan:
    python run_hp_experiments.py
"""
import copy
import os
import shutil

import numpy as np
import pandas as pd
import torch

from config import CFG
from train import run_training

BASELINE_F1 = 0.9520  # fold0_history.json, batch=64, no grad-checkpointing, 8 epoch

RUNS = [
    ("epochs20", dict(epochs=20, patience=4)),
    ("img288", dict(epochs=20, patience=4, img_size=288, batch=32, accum_steps=2)),
    ("llrd090", dict(epochs=20, patience=4, layer_decay=0.9)),
]


def main():
    class_weights = torch.tensor(np.load(CFG.class_weights_path), dtype=torch.float32)
    rows = []

    for name, overrides in RUNS:
        cfg = copy.copy(CFG)
        for k, v in overrides.items():
            setattr(cfg, k, v)

        print(f"\n{'=' * 60}\n[{name}] {overrides}\n{'=' * 60}")
        best_f1, mins = run_training(fold=0, cfg=cfg, class_weights=class_weights)

        # fold0.pt & fold0_history.json baru ditimpa run ini -- simpan salinan
        # bernama supaya tidak hilang tertimpa run berikutnya di loop ini.
        shutil.copy(f"{cfg.save_dir}/fold0.pt", f"{cfg.save_dir}/hp_{name}_fold0.pt")
        shutil.copy(f"{cfg.save_dir}/fold0_history.json", f"{cfg.save_dir}/hp_{name}_history.json")

        delta = best_f1 - BASELINE_F1
        rows.append({
            "run_name": name,
            "img_size": cfg.img_size,
            "batch": cfg.batch,
            "epochs": cfg.epochs,
            "patience": cfg.patience,
            "layer_decay": cfg.layer_decay,
            "best_f1": best_f1,
            "delta_vs_baseline": delta,
            "minutes_per_epoch": mins,
        })
        print(f"[{name}] fold0 Macro-F1 = {best_f1:.4f} "
              f"(delta {delta:+.4f} vs baseline {BASELINE_F1}) | {mins:.1f} min/epoch")

    df = pd.DataFrame(rows)
    print("\n" + df[["run_name", "best_f1", "delta_vs_baseline", "minutes_per_epoch"]].to_string(index=False))

    os.makedirs("../results", exist_ok=True)
    out_path = "../results/hp_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nTersimpan: {out_path}")


if __name__ == "__main__":
    main()
