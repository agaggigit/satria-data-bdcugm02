"""nested_validation.py — Nested CV untuk estimasi JUJUR gain threshold/weight.

Prinsip:
- Tune threshold/bobot HANYA di 4 fold
- Terapkan hasilnya ke fold ke-5 yang tidak dipakai tuning
- Rata-rata 5 angka = estimasi jujur skor test

Kenapa ini penting:
- Model embedding dengan OOF ~0.99 hampir tidak punya ruang naik
- Threshold tuning di seluruh OOF = mudah "menemukan" gain palsu (overfit ke OOF)
- Nested validation memisahkan gain NYATA dari gain ILUSI
- Kalau selisih tuning-penuh vs nested > 0.003 → tuning terlalu agresif

Referensi: Track_C_Evaluation_Submission_Plan_v3.md, Fase 1
"""
import numpy as np
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def macro_f1(labels, preds):
    return f1_score(labels, preds, average="macro", labels=[0, 1, 2],
                    zero_division=0.0)


def _best_weights_on_subset(probs: np.ndarray, labels: np.ndarray,
                             lo: float, hi: float, n_steps: int) -> list:
    """Grid search threshold (weight multiplier) di subset OOF.

    Strategi: fix W0=1.0 (kelas Recyclable sebagai jangkar),
    grid search W1 (Electronic) dan W2 (Organic).
    """
    search_space = np.linspace(lo, hi, num=n_steps)
    best_thresholds = [1.0, 1.0, 1.0]
    best_f1_val = macro_f1(labels, np.argmax(probs, axis=1))  # baseline argmax

    for w1 in search_space:
        for w2 in search_space:
            weights = np.array([1.0, w1, w2])
            preds = np.argmax(probs * weights, axis=1)
            f1 = macro_f1(labels, preds)
            if f1 > best_f1_val:
                best_f1_val = f1
                best_thresholds = [1.0, float(w1), float(w2)]

    return best_thresholds


# ---------------------------------------------------------------------------
# Core: nested CV untuk satu OOF
# ---------------------------------------------------------------------------

def nested_cv_threshold(oof_probs: np.ndarray, true_labels: np.ndarray,
                         fold_assignments: np.ndarray,
                         lo: float = 0.5, hi: float = 2.0,
                         n_steps: int = 30, verbose: bool = True) -> dict:
    """Nested CV: tune threshold di 4 fold, evaluasi di fold ke-5.

    Args:
        oof_probs      : [N, 3] probabilitas OOF
        true_labels    : [N] label asli
        fold_assignments: [N] nomor fold per sampel (0..4)
        lo, hi, n_steps: grid search range dan steps
        verbose        : cetak progress tiap fold

    Returns:
        dict dengan:
            nested_f1_per_fold : list[float] — F1 tiap held-out fold
            nested_mean        : float — rata-rata = estimasi jujur skor test
            nested_min         : float — fold terburuk
            nested_std         : float
            illusion_gap       : float — selisih tuning-penuh vs nested_mean
            full_tuned_f1      : float — F1 tuning di seluruh OOF (angka "palsu")
            full_thresholds    : list[float] — threshold dari tuning di seluruh OOF
    """
    folds = sorted(np.unique(fold_assignments))
    nested_f1_per_fold = []

    if verbose:
        print("=" * 55)
        print("NESTED VALIDATION — Estimasi Jujur Gain Threshold")
        print("=" * 55)
        baseline = macro_f1(true_labels, np.argmax(oof_probs, axis=1))
        print(f"  Baseline argmax (seluruh OOF) : {baseline:.5f}")

    for held_out in folds:
        tune_mask = fold_assignments != held_out
        eval_mask = fold_assignments == held_out

        tune_probs  = oof_probs[tune_mask]
        tune_labels = true_labels[tune_mask]
        eval_probs  = oof_probs[eval_mask]
        eval_labels = true_labels[eval_mask]

        # Tune HANYA di 4 fold
        best_thr = _best_weights_on_subset(tune_probs, tune_labels, lo, hi, n_steps)

        # Evaluasi MENTAH-MENTAH di fold ke-5 (no re-tuning)
        eval_preds = np.argmax(eval_probs * np.array(best_thr), axis=1)
        eval_f1 = macro_f1(eval_labels, eval_preds)
        nested_f1_per_fold.append(eval_f1)

        # F1 di 4 fold tuning (untuk lihat selisih)
        tune_preds = np.argmax(tune_probs * np.array(best_thr), axis=1)
        tune_f1 = macro_f1(tune_labels, tune_preds)

        if verbose:
            print(f"  Fold {held_out} held-out | "
                  f"tune F1={tune_f1:.5f} | "
                  f"nested F1={eval_f1:.5f} | "
                  f"delta={eval_f1 - tune_f1:+.5f}")

    # Tuning di SELURUH OOF (angka "menyenangkan" tapi sedikit palsu)
    full_thr = _best_weights_on_subset(oof_probs, true_labels, lo, hi, n_steps)
    full_preds = np.argmax(oof_probs * np.array(full_thr), axis=1)
    full_f1 = macro_f1(true_labels, full_preds)

    nested_mean = float(np.mean(nested_f1_per_fold))
    nested_min  = float(np.min(nested_f1_per_fold))
    nested_std  = float(np.std(nested_f1_per_fold))
    illusion_gap = full_f1 - nested_mean

    if verbose:
        print("-" * 55)
        print(f"  Nested CV   : mean={nested_mean:.5f}  "
              f"min={nested_min:.5f}  std={nested_std:.5f}")
        print(f"  Tuning-penuh: {full_f1:.5f}  "
              f"threshold={full_thr}")
        print(f"  Illusion gap: {illusion_gap:+.5f}")
        if illusion_gap > 0.003:
            print("  ⚠️  Gap > 0.003 — tuning terlalu agresif! "
                  "Pertimbangkan argmax saja.")
        elif illusion_gap > 0.001:
            print("  ⚠️  Gap 0.001–0.003 — gain sebagian nyata, sebagian ilusi. "
                  "Monitor di test set.")
        else:
            print("  ✅  Gap < 0.001 — gain kecil tapi jujur. Threshold layak dipakai.")
        print("=" * 55)

    return {
        "nested_f1_per_fold": nested_f1_per_fold,
        "nested_mean": nested_mean,
        "nested_min": nested_min,
        "nested_std": nested_std,
        "illusion_gap": illusion_gap,
        "full_tuned_f1": full_f1,
        "full_thresholds": full_thr,
    }


# ---------------------------------------------------------------------------
# Core: nested CV untuk weight search multi-komposisi
# ---------------------------------------------------------------------------

def nested_cv_weight_search(list_oof_probs: list, true_labels: np.ndarray,
                              fold_assignments: np.ndarray,
                              n_simplex: int = 200, seed: int = 42,
                              verbose: bool = True) -> dict:
    """Nested CV untuk weight ensemble multi-komposisi.

    Sampling simplex (bukan grid) karena dimensi bisa > 2.
    Dibatasi n_simplex kombinasi untuk menghindari overfit ke OOF.

    Args:
        list_oof_probs  : list of [N, 3] — satu per komposisi
        true_labels     : [N]
        fold_assignments: [N]
        n_simplex       : max kombinasi bobot yang di-coba (plan v3: dibatasi)
        seed            : random seed untuk sampling simplex
    """
    rng = np.random.default_rng(seed)
    n_comp = len(list_oof_probs)
    folds = sorted(np.unique(fold_assignments))
    nested_f1_per_fold = []

    if verbose:
        print("=" * 55)
        print(f"NESTED WEIGHT SEARCH — {n_comp} komposisi, {n_simplex} sampel")
        print("=" * 55)

    # Sampel bobot dari simplex (jumlah = 1, semua ≥ 0)
    raw = rng.dirichlet(np.ones(n_comp), size=n_simplex)
    weight_candidates = raw  # [n_simplex, n_comp]

    for held_out in folds:
        tune_mask = fold_assignments != held_out
        eval_mask = fold_assignments == held_out

        tune_labels = true_labels[tune_mask]
        eval_labels = true_labels[eval_mask]

        # Cari bobot terbaik di 4 fold
        best_weights = np.ones(n_comp) / n_comp
        best_f1_val = 0.0
        for w in weight_candidates:
            # Gabung prob dengan bobot
            tune_combo = sum(w[i] * list_oof_probs[i][tune_mask]
                             for i in range(n_comp))
            preds = np.argmax(tune_combo, axis=1)
            f1 = macro_f1(tune_labels, preds)
            if f1 > best_f1_val:
                best_f1_val = f1
                best_weights = w

        # Evaluasi di fold held-out
        eval_combo = sum(best_weights[i] * list_oof_probs[i][eval_mask]
                         for i in range(n_comp))
        eval_preds = np.argmax(eval_combo, axis=1)
        eval_f1 = macro_f1(eval_labels, eval_preds)
        nested_f1_per_fold.append(eval_f1)

        if verbose:
            print(f"  Fold {held_out} held-out | "
                  f"best_weights={np.round(best_weights, 3).tolist()} | "
                  f"tune F1={best_f1_val:.5f} | nested F1={eval_f1:.5f}")

    nested_mean = float(np.mean(nested_f1_per_fold))
    nested_min  = float(np.min(nested_f1_per_fold))
    nested_std  = float(np.std(nested_f1_per_fold))

    # Bobot terbaik di seluruh OOF (untuk inference test)
    best_w_full = np.ones(n_comp) / n_comp
    best_f1_full = 0.0
    for w in weight_candidates:
        combo = sum(w[i] * list_oof_probs[i] for i in range(n_comp))
        preds = np.argmax(combo, axis=1)
        f1 = macro_f1(true_labels, preds)
        if f1 > best_f1_full:
            best_f1_full = f1
            best_w_full = w

    illusion_gap = best_f1_full - nested_mean

    if verbose:
        print("-" * 55)
        print(f"  Nested CV   : mean={nested_mean:.5f}  "
              f"min={nested_min:.5f}  std={nested_std:.5f}")
        print(f"  Tuning-penuh: {best_f1_full:.5f}  "
              f"best_weights={np.round(best_w_full, 3).tolist()}")
        print(f"  Illusion gap: {illusion_gap:+.5f}")
        if illusion_gap > 0.003:
            print("  ⚠️  Gap > 0.003 — kurangi n_simplex atau pakai bobot seragam.")
        print("=" * 55)

    return {
        "nested_f1_per_fold": nested_f1_per_fold,
        "nested_mean": nested_mean,
        "nested_min": nested_min,
        "nested_std": nested_std,
        "illusion_gap": illusion_gap,
        "full_tuned_f1": float(best_f1_full),
        "best_weights": best_w_full.tolist(),
    }


if __name__ == "__main__":
    # --- Smoke test ---
    np.random.seed(42)
    N = 5000
    # Simulasi OOF yang sangat bagus (argmax baseline ~0.99)
    true_labels = np.random.choice([0, 1, 2], size=N, p=[0.38, 0.15, 0.47])
    # Buat prob yang sebagian besar argmax-nya benar
    oof_probs = np.zeros((N, 3))
    for i, t in enumerate(true_labels):
        oof_probs[i, t] = 0.90 + np.random.uniform(0, 0.08)
        others = [j for j in range(3) if j != t]
        rem = 1.0 - oof_probs[i, t]
        split = np.random.dirichlet([1, 1])
        oof_probs[i, others[0]] = split[0] * rem
        oof_probs[i, others[1]] = split[1] * rem

    fold_assignments = np.arange(N) % 5

    print("=== Single OOF threshold nested CV ===")
    result = nested_cv_threshold(oof_probs, true_labels, fold_assignments,
                                 n_steps=15, verbose=True)
    print(f"\nHasil: nested_mean={result['nested_mean']:.5f}, "
          f"illusion_gap={result['illusion_gap']:+.5f}")
    print("\nSmoke test PASSED")
