"""senior_spec_compare.py — Eksperimen Spesifikasi Senior:
Model: Concat SigLIP2-Base (256) + SigLIP1-Base (256)
Head: MLP 3-Class
Metode: Frozen Head Probe, LoRA, & Last-layer Fine-Tuning
Komparasi: folds.csv (v1) vs folds_v2.csv (v2 data bersih Track A)
"""
import os
import sys
import numpy as np
import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from consistency import fold_consistency
from embed import align_to_subset, assert_fold_unchanged, load_embeddings
from features import concat_features, l2norm
from heads import make_head
from metrics import per_class_f1
from oof import assemble_oof, validate_oof
from probe_cv import run_probe_cv


def load_combo_features(combo: str, split: str) -> np.ndarray:
    """Muat fitur single atau concat."""
    if combo == "siglip2so400m":
        return l2norm(load_embeddings("siglip2so400m", split)[0])
    if combo == "concat_siglip2b256_siglip1b256":
        e2 = load_embeddings("siglip2b256", split)[0]
        e1 = load_embeddings("siglip1b256", split)[0]
        return concat_features([e2, e1])
    raise KeyError(f"combo '{combo}' tidak dikenal.")


def run_senior_spec_eval(combo: str, X_v1: np.ndarray, folds_v1: pd.DataFrame,
                         X_v2: np.ndarray, folds_v2: pd.DataFrame,
                         head_name: str = "mlp", seed: int = 42) -> dict:
    """Hitung 3 varian evaluasi (baseline_v1, naif_v2, adil_v2) untuk spesifikasi senior."""

    print(f"\n[{combo} | {head_name.upper()}] 1/3 Baseline v1...")
    oof_v1, _ = run_probe_cv(X_v1, folds_v1, head_name=head_name, class_weight="balanced", seed=seed)
    c_v1 = fold_consistency(oof_v1, folds_v1)
    pcf1_v1 = per_class_f1(folds_v1["label"].to_numpy(), oof_v1.argmax(axis=1))
    print(f"  -> Baseline v1 Mean CV: {c_v1['mean']:.4f}")

    print(f"[{combo} | {head_name.upper()}] 2/3 Naif v2...")
    oof_v2_naif, _ = run_probe_cv(X_v2, folds_v2, head_name=head_name, class_weight="balanced", seed=seed)
    c_v2_naif = fold_consistency(oof_v2_naif, folds_v2)
    pcf1_v2_naif = per_class_f1(folds_v2["label"].to_numpy(), oof_v2_naif.argmax(axis=1))
    print(f"  -> Naif v2 Mean CV: {c_v2_naif['mean']:.4f}")

    print(f"[{combo} | {head_name.upper()}] 3/3 Adil v2...")
    y_v1 = folds_v1["label"].to_numpy()
    y_v2 = folds_v2["label"].to_numpy()
    fold_v1 = folds_v1["fold"].to_numpy()
    fold_v2 = folds_v2["fold"].to_numpy()

    fold_probs = {}
    for f in sorted(folds_v1["fold"].unique()):
        tr_m = fold_v2 != f
        va_m = fold_v1 == f
        head = make_head(head_name, seed=seed, class_weight="balanced")
        head.fit(X_v2[tr_m], y_v2[tr_m])
        probs = head.predict_proba(X_v1[va_m])
        val_idx = folds_v1.index[va_m].to_numpy()
        fold_probs[int(f)] = (val_idx, probs)

    oof_v2_adil = assemble_oof(fold_probs, n_rows=len(folds_v1))
    validate_oof(oof_v2_adil, folds_v1)
    c_v2_adil = fold_consistency(oof_v2_adil, folds_v1)
    pcf1_v2_adil = per_class_f1(y_v1, oof_v2_adil.argmax(axis=1))
    print(f"  -> Adil v2 Mean CV: {c_v2_adil['mean']:.4f}")

    return {
        "combo": combo,
        "head": head_name,
        "mean_baseline_v1": c_v1["mean"], "min_baseline_v1": c_v1["min"],
        "mean_naif_v2": c_v2_naif["mean"], "min_naif_v2": c_v2_naif["min"],
        "mean_adil_v2": c_v2_adil["mean"], "min_adil_v2": c_v2_adil["min"],
        "delta_adil_minus_baseline": c_v2_adil["mean"] - c_v1["mean"],
        "f1_electronic_v1": float(pcf1_v1[1]),
        "f1_electronic_v2_adil": float(pcf1_v2_adil[1])
    }


def main():
    from config import CFG

    folds_v1 = pd.read_csv(CFG.folds_csv)
    folds_v2 = pd.read_csv(CFG.folds_v2_csv)
    assert_fold_unchanged(folds_v1, folds_v2)

    combos_to_test = ["siglip2so400m", "concat_siglip2b256_siglip1b256"]
    heads_to_test = ["mlp", "knn"]

    for combo in combos_to_test:
        print(f"\n==========================================")
        print(f"--- Loading Features for Model Combo: {combo} ---")
        print(f"==========================================")
        X_v1 = load_combo_features(combo, "train")
        X_v2 = align_to_subset(X_v1, folds_v1, folds_v2)
        for h in heads_to_test:
            res = run_senior_spec_eval(combo, X_v1, folds_v1, X_v2, folds_v2, head_name=h, seed=CFG.seed)

            print(f"\n=== HASIL EKSPERIMEN [{combo} | {h.upper()}] ===")
            for k, v in res.items():
                if isinstance(v, float):
                    print(f"  {k:30s} : {v:.4f}")
                else:
                    print(f"  {k:30s} : {v}")


if __name__ == "__main__":
    main()
