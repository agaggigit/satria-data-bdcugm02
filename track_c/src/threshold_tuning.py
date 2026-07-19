"""threshold_tuning.py — Threshold (weight multiplier) tuning di OOF.

Strategi:
  - Fix W0=1.0 (kelas Recyclable sebagai jangkar)
  - Grid search W1 (Electronic) dan W2 (Organic) dalam [lo, hi]
  - n_steps=30 → langkah ≈ 0.052, cukup kasar untuk menghindari overfit

Catatan penting (plan v3):
  - Modul ini hanya menghitung threshold di SELURUH OOF (untuk dipakai ke test)
  - Untuk estimasi JUJUR seberapa gain-nya bertahan di test, pakai nested_validation.py
  - Urutan wajib: ensemble/weight antar model dulu, BARU threshold di atas prob gabungan
  - Setiap kali komposisi berubah, threshold di-tuning ULANG
"""
import numpy as np
from sklearn.metrics import f1_score


def macro_f1(preds, labels):
    return f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0.0)


def tune_thresholds_oof(oof_probs, true_labels, n_steps=30,
                         lo: float = 0.5, hi: float = 2.0):
    """Cari multiplier terbaik per kelas untuk memaksimalkan Macro-F1 di OOF.

    Args:
        oof_probs  : [N, 3] probabilitas OOF (softmax output)
        true_labels: [N] label asli (0/1/2)
        n_steps    : jumlah titik grid per dimensi (default 30 ≈ langkah 0.052)
        lo, hi     : rentang pencarian multiplier

    Returns:
        list [W0, W1, W2] — W0 selalu 1.0 (jangkar)

    Catatan:
        Fungsi ini hanya menghitung "angka menyenangkan" (tuning di seluruh OOF).
        Untuk estimasi jujur gain yang bertahan di test, gunakan nested_validation.py.
    """
    print("Memulai threshold tuning di OOF...")

    # Baseline dengan Argmax
    baseline_preds = np.argmax(oof_probs, axis=1)
    baseline_f1 = macro_f1(baseline_preds, true_labels)
    print(f"  Baseline OOF Macro-F1 (argmax) : {baseline_f1:.5f}")

    best_thresholds = [1.0, 1.0, 1.0]
    best_f1 = baseline_f1

    # Grid search: W0=1.0 (fixed), W1 dan W2 di-search
    search_space = np.linspace(lo, hi, num=n_steps)

    for w1 in search_space:
        for w2 in search_space:
            weights = np.array([1.0, w1, w2])
            preds = np.argmax(oof_probs * weights, axis=1)
            f1 = macro_f1(preds, true_labels)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = [1.0, float(w1), float(w2)]

    gain = best_f1 - baseline_f1
    print(f"  Best OOF Macro-F1 (tuning)     : {best_f1:.5f}  (+{gain:.5f})")
    print(f"  Best Multiplier: W0={best_thresholds[0]:.3f} "
          f"W1={best_thresholds[1]:.3f} W2={best_thresholds[2]:.3f}")

    if gain < 1e-5:
        print("  ℹ️  Gain = 0 — argmax sudah optimal untuk komposisi ini.")
    elif gain > 0.003:
        print("  ⚠️  Gain besar di OOF. Cek nested_validation untuk estimasi jujur!")
    else:
        print("  ✅  Gain kecil dan wajar. Threshold dipakai.")

    return best_thresholds


def apply_thresholds(test_probs, thresholds):
    """Terapkan multiplier threshold ke probabilitas test → label argmax.

    Args:
        test_probs : [N, 3] probabilitas test
        thresholds : [W0, W1, W2] dari tune_thresholds_oof()

    Returns:
        [N] array label integer (0/1/2)
    """
    weights = np.array(thresholds)
    weighted_probs = test_probs * weights
    return np.argmax(weighted_probs, axis=1).astype(int)


if __name__ == "__main__":
    # Smoke test
    np.random.seed(42)
    N = 500
    oof = np.random.dirichlet([5, 1, 3], size=N)   # bias ke kelas 0 & 2
    labels = np.random.choice([0, 1, 2], size=N, p=[0.38, 0.15, 0.47])
    thr = tune_thresholds_oof(oof, labels, n_steps=10)
    preds = apply_thresholds(oof, thr)
    assert preds.shape == (N,), f"Shape salah: {preds.shape}"
    assert set(np.unique(preds)) <= {0, 1, 2}
    print("Smoke test PASSED")

