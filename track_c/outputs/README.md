# outputs/ — Artefak Track C

Folder ini tidak di-push ke git jika berisi output besar, namun karena setup kita menyimpan output langsung ke Google Drive (`/content/drive/MyDrive/BDC2026 apace`), folder ini bisa saja kosong secara lokal.

Namun, di Google Drive, file yang akan di-generate oleh Track C adalah:

| File | Dibuat oleh | Keterangan |
|------|-------------|------------|
| `submission_apace.csv` | `generate_submission.py` | Prediksi final (Ensemble + TTA + Thresholded). Lolos validator format dan siap diunggah ke panitia. |

## Instruksi Handoff Akhir

Track C telah menyediakan pipeline `track_c/notebooks/01_track_c_pipeline.ipynb` yang membungkus semua logic OOF Tuning, Ensemble, TTA, dan pembuatan file submission akhir.
