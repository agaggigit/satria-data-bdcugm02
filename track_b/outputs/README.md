# outputs/ — Artefak Track B (di-gitignore)

Folder ini berisi output yang dihasilkan dari training Track B:

| File | Dibuat oleh | Diserahkan ke |
|------|-------------|---------------|
| `fold{0..4}.pt` | training 5-fold | Track C (ensemble + inference test) |
| `oof.npy` | OOF collector — probabilitas `[N, 3]`, index cocok `folds.csv` | Track C (threshold tuning) |
| `fold0_baseline_log.json` | Fase 1 — baseline fold 0 | Report + sanity target Fase 2 |
| `cv_summary.json` | Fase 2 — CV Macro-F1 (mean ± std) | Track C + report |

> ⚠️ File besar (`*.pt`, `*.npy`, `*.png`) di-gitignore — share via Google Drive / Kaggle Dataset.
> Log JSON (`fold0_baseline_log.json`, `cv_summary.json`) **masuk git** — kecil dan jadi
> bukti reproducibility untuk verifikasi panitia + bahan report.
