# Track A — Outputs

## Artefak v1 (sudah tersedia — handoff awal ke Track B & C)

| File | Keterangan |
|------|-----------|
| `folds.csv` | Split 5-fold StratifiedGroupKFold, kolom: `filepath, label, fold` |
| `class_weights.npy` | Bobot `[w0, w1, w2]` untuk CrossEntropyLoss |
| `df_clean.csv` | DataFrame lengkap + phash + group_id |
| `eda_stats.json` | Ringkasan statistik EDA |
| `split_metadata.json` | Seed 42, metode, img_size 224 |
| `skip_list.txt` | File gambar corrupt (kosong — 0 corrupt) |
| `batch_visualization.png` | Visualisasi 1 batch train |
| `class_distribution.png` | Distribusi kelas train |
| `fold_proportions.png` | Proporsi kelas per fold |
| `image_size_dist.png` | Distribusi ukuran gambar |
| `sample_images.png` | Contoh gambar per kelas |

---

## Artefak v2 (dihasilkan pasca cleaning — Gate G2)

> Dihasilkan oleh `05_generate_v2.ipynb` setelah review visual selesai.

| File | Keterangan |
|------|-----------|
| `folds_v2.csv` | Train bersih — drop & relabel diterapkan; fold assignment sampel bertahan tidak berubah |
| `class_weights_v2.npy` | Bobot kelas dari distribusi v2 |
| `cleaning_summary.json` | Ringkasan: jumlah drop/relabel per kelas, metodologi (untuk verifikasi panitia) |

---

## Artefak Diagnosis & Review (intermediate)

| File | Keterangan |
|------|-----------|
| `diagnosis/oof_diagnosis.csv` | Per-sampel: margin, entropy, prediksi, is_correct |
| `diagnosis/oof_confusion_matrix.png` | Confusion matrix OOF |
| `diagnosis/oof_margin_entropy_dist.png` | Distribusi margin & entropy per kelas |
| `label_issues.csv` | Kandidat mislabel dari cleanlab |
| `ambiguous_candidates.csv` | Kandidat ambigu (margin/entropy rendah) |
| `cleaning_log.csv` | Log keputusan review (keep/relabel/drop + alasan) |
| `contact_sheets/` | Grid PNG untuk review visual semi-otomatis |

---

> ⚠️ File binary & output besar di-gitignore. Share via Google Drive folder `BDC2026 apace/output_trackA/`.
