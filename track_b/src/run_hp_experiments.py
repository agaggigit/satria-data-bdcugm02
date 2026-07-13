"""run_hp_experiments.py — Eksperimen HP di fold 0 saja (Task 7, Fase 1 plan v2).

SATU VARIABEL PER RUN. Dibandingkan terhadap baseline Fase 1 yang sudah ada:
batch=64, grad_checkpointing off, 8 epoch tanpa early stop -> val_f1 = 0.9520
(lihat track_b/outputs/fold0_history.json + track_b/results/fold0_curves.png).

Urutan RUNS mengikuti rasio dampak/biaya dari plan: epoch+early-stop dulu
(hampir gratis), lalu img288 (backbone di-finetune utk test di 288), lalu LLRD
(bonus kecil tapi konsisten). vflip TIDAK diuji di sini -- augmentasi
RandomVerticalFlip(p=0.2) sudah hardcode di track_a/src/dataset.py, Track B
tidak bisa mematikan/menyalakannya tanpa koordinasi breaking-change ke Track A.

Tiap run punya run_name sendiri ("hp_epochs20", dst) -- checkpoint & history-nya
otomatis ke file terpisah (lihat src/ckpt.py), tidak saling menimpa. Kalau mau
mengulang run yang sama (mis. img288 kepotong kuota), jalankan lagi dengan
override run_name yang sama akan gagal keras (FileExistsError) kecuali kamu
hapus dulu checkpoint lama atau tambahkan allow_overwrite=True di overrides
RUNS -- ini disengaja, bukan bug.

Jalankan dari track_b/src/ (sys.path & cwd harus di sini, sama seperti cell
notebook 0d) supaya import flat (`from config import CFG`) jalan:
    python run_hp_experiments.py                  # jalanin semua run
    python run_hp_experiments.py llrd090           # jalanin run tertentu saja
    python run_hp_experiments.py epochs20 llrd090  # atau beberapa nama sekaligus

Hasil disimpan ke ../results/hp_results.csv SETELAH TIAP RUN (bukan cuma di
akhir) -- kalau kuota Colab habis di tengah run berikutnya, hasil run yang
sudah selesai tetap aman tercatat.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

from config import CFG, make_cfg
from train import run_training

BASELINE_F1 = 0.9520  # fold0_history.json, batch=64, no grad-checkpointing, 8 epoch
RESULTS_PATH = "../results/hp_results.csv"

RUNS = [
    ("epochs20", dict(run_name="hp_epochs20", epochs=20, patience=4)),
    ("img288", dict(run_name="hp_img288", epochs=20, patience=4,
                    img_size=288, batch=32, accum_steps=2)),
    ("llrd090", dict(run_name="hp_llrd090", epochs=20, patience=4, layer_decay=0.9)),
]


def _save_row(row: dict):
    """Update satu baris di hp_results.csv tanpa menghapus baris run lain."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    if os.path.exists(RESULTS_PATH):
        df = pd.read_csv(RESULTS_PATH)
        df = df[df["run_name"] != row["run_name"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(RESULTS_PATH, index=False)
    return df


def main(run_names=None):
    """run_names: list nama run yang mau dijalankan (default semua di RUNS)."""
    selected = RUNS if not run_names else [r for r in RUNS if r[0] in run_names]
    if run_names:
        missing = set(run_names) - {r[0] for r in RUNS}
        assert not missing, f"Nama run tidak dikenal: {missing}. Pilihan: {[r[0] for r in RUNS]}"

    class_weights = torch.tensor(np.load(CFG.class_weights_path), dtype=torch.float32)

    for name, overrides in selected:
        cfg = make_cfg(**overrides)

        print(f"\n{'=' * 60}\n[{name}] {overrides}\n{'=' * 60}")
        best_f1, mins = run_training(fold=0, cfg=cfg, class_weights=class_weights)

        delta = best_f1 - BASELINE_F1
        row = {
            "run_name": name,
            "img_size": cfg.img_size,
            "batch": cfg.batch,
            "epochs_ran": cfg.epochs,
            "patience": cfg.patience,
            "layer_decay": cfg.layer_decay,
            "best_f1": best_f1,
            "delta_vs_baseline": delta,
            "minutes_per_epoch": mins,
            "status": "selesai",
        }
        df = _save_row(row)
        print(f"[{name}] fold0 Macro-F1 = {best_f1:.4f} "
              f"(delta {delta:+.4f} vs baseline {BASELINE_F1}) | {mins:.1f} min/epoch")
        print(f"Tersimpan ke {RESULTS_PATH}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main(run_names=sys.argv[1:] or None)
