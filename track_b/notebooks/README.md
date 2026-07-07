# Track B — Panduan Notebook

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini berisi notebook orchestrator Track B. Sesuai konvensi tim, **logika inti
ada di `src/` (file `.py`)** — notebook di sini hanya lapisan tipis untuk memanggil
fungsi, menjalankan training di Colab/Kaggle, dan menampilkan hasil.

---

## Daftar Notebook (rencana)

| # | File | Fase | Isi |
|---|------|------|-----|
| 1 | `01_sanity_scaffold.ipynb` | Fase 0 | Setup env, sanity overfit 1 batch (bukti loop benar) |
| 2 | `02_baseline_fold0.ipynb` | Fase 1 | Swap stub → `dataset.py` asli, training fold 0, catat baseline Macro-F1 |
| 3 | `03_full_5fold.ipynb` | Fase 2 | Training 5 fold + kumpulkan OOF + CV Macro-F1 |

> **Prerequisite Fase 1:** GATE 2 dari Track A harus hijau — `folds.csv`,
> `dataset.py` asli, dan `class_weights.npy` sudah tersedia di storage bersama.

---

## Environment Target

- **Google Colab** — GPU T4, **2 CPU cores** → `num_workers=2` (jangan 4)
- Alternatif: Kaggle (kuota 30 jam GPU/minggu — koordinasikan dengan tim)
- Library inti: `torch`, `timm`, `scikit-learn`

```bash
pip install torch timm scikit-learn pandas numpy matplotlib
```

---

## Alur Data

```
Track A (folds.csv, dataset.py, class_weights.npy)
        │
        ▼
02_baseline_fold0.ipynb ──→ fold0.pt + fold0_baseline_log.json
        │
        ▼
03_full_5fold.ipynb ──→ fold{0..4}.pt + oof.npy + CV Macro-F1 ──→ ⭐ Track C
```

---

*Dibuat untuk BDC Satria Data 2026 — Track B*
