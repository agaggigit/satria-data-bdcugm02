# Track A — Katalog Output

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini menyimpan semua artefak yang dihasilkan oleh pipeline Track A. File binary dan output berukuran besar di-gitignore — disimpan dan dibagikan via Google Drive:

> 📁 **Google Drive:** `BDC2026apace/output_trackA/`
> 🔗 https://drive.google.com/drive/folders/1m8AeQGumLfoGvwP9AZUiTChyXjoUANrD?usp=sharing

---

## Artefak v2 — Data Bersih (Aktif Dipakai)

Dihasilkan oleh `notebooks/05_generate_v2.ipynb` setelah review visual selesai.
**Ini yang dipakai Track B & C — jangan gunakan versi v1 untuk training final.**

| File | Ukuran | Diserahkan ke | Keterangan |
|------|--------|---------------|------------|
| `folds_v2.csv` | ~1.7 MB | Track B & C | Split 5-fold bersih; kolom: `filepath, label, fold`; 25.985 baris |
| `class_weights_v2.npy` | 152 B | Track B | Bobot `[0.883, 2.202, 0.707]` untuk `CrossEntropyLoss` |
| `cleaning_summary.json` | 560 B | Dokumentasi | Angka cleaning: drop/relabel per kelas, metodologi (verifikasi panitia) |

### Cara Load

```python
import pandas as pd, numpy as np, torch

folds   = pd.read_csv("outputs/folds_v2.csv")
weights = torch.tensor(np.load("outputs/class_weights_v2.npy"), dtype=torch.float)

# Untuk training (Track B):
from track_a.src.dataset import get_loaders
train_loader, val_loader = get_loaders(fold=0, img_size=224, batch=32,
                                        folds_csv="outputs/folds_v2.csv")
```

---

## Artefak v1 — Sebelum Cleaning (Referensi Historis)

Dihasilkan oleh `notebooks/01_eda.ipynb` dan `02_split_verify.ipynb`.
Dipakai Track B untuk training awal sebelum data bersih tersedia.

| File | Ukuran | Keterangan |
|------|--------|------------|
| `folds.csv` | ~1.7 MB | Split v1 (26.527 baris); 542 di antaranya sudah di-drop di v2 |
| `class_weights.npy` | 152 B | Bobot kelas v1 (sebelum cleaning) |
| `df_clean.csv` | ~2.6 MB | DataFrame lengkap: `filepath, label, group_id, phash` |
| `eda_stats.json` | 639 B | Statistik EDA: jumlah gambar, duplikat, resolusi median |
| `split_metadata.json` | 118 B | Seed `42`, metode split, img_size untuk reproducibility |
| `skip_list.txt` | — | Filepath gambar corrupt; kosong (0 gambar corrupt ditemukan) |

---

## Visualisasi EDA

Dihasilkan oleh `notebooks/01_eda.ipynb` dan `02_split_verify.ipynb`.

| File | Keterangan |
|------|------------|
| `class_distribution.png` | Bar chart + pie chart distribusi kelas train |
| `image_size_dist.png` | Histogram width, height, aspect ratio gambar |
| `sample_images.png` | Grid 3×5 contoh gambar per kelas |
| `batch_visualization.png` | Sanity check: 1 batch train (dengan augmentasi) vs val (tanpa augmentasi) |
| `fold_proportions.png` | Proporsi kelas per fold — verifikasi keseimbangan split |

---

## Artefak Diagnosis OOF

Dihasilkan oleh `notebooks/03_oof_diagnosis.ipynb`. Digunakan sebagai input untuk Fase cleaning.

Lokasi: `outputs/diagnosis/`

| File | Ukuran | Keterangan |
|------|--------|------------|
| `oof_diagnosis.csv` | ~5.6 MB | Per-sampel (26.527 baris): `filepath, label, pred_class, margin, entropy, is_correct` |
| `oof_confusion_matrix.png` | 94 KB | Confusion matrix OOF — menunjukkan pola kesalahan antar kelas |
| `oof_margin_entropy_dist.png` | 62 KB | Distribusi margin & entropy per kelas (dipakai untuk menentukan threshold ambigu) |

### Kolom `oof_diagnosis.csv`

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `filepath` | str | Path gambar |
| `label` | int | Label asli (0/1/2) |
| `pred_class` | int | Prediksi argmax OOF |
| `margin` | float | max_prob − second_max_prob (makin tinggi = model makin yakin) |
| `entropy` | float | Entropy distribusi probabilitas (makin tinggi = model makin tidak yakin) |
| `is_correct` | int | 1 jika pred_class == label, 0 jika salah |

---

## Artefak Cleaning & Review

Dihasilkan oleh `notebooks/04_cleanlab_cleaning.ipynb` dan review manual tim.

| File | Ukuran | Keterangan |
|------|--------|------------|
| `label_issues.csv` | 780 B | Kandidat mislabel dari cleanlab (sumber 1) |
| `ambiguous_candidates.csv` | 163 KB | Sampel dengan margin/entropy rendah (5% paling tidak yakin) |
| `cleaning_log.csv` | 92 KB | Template log review (kolom `keputusan` kosong) |
| `cleaning_log.xlsx` | 51 KB | Versi Excel dari `cleaning_log.csv` untuk review |
| `cleaning_log_terisi.csv` | 103 KB | Log review yang sudah diisi — **sumber kebenaran folds_v2.csv** |
| `contact_sheets/` | folder | Grid PNG 50 gambar/halaman per pasangan kelas; border merah = probable mislabel |

### Kolom `cleaning_log_terisi.csv`

| Kolom | Nilai Valid | Keterangan |
|-------|-------------|------------|
| `filepath` | str | Path gambar kandidat |
| `label_asli` | 0, 1, 2 | Label sebelum review |
| `pred_class` | 0, 1, 2 | Prediksi OOF |
| `keputusan` | `keep` / `relabel` / `drop` | Keputusan reviewer |
| `label_baru` | 0, 1, 2 | Wajib diisi jika `keputusan=relabel` |
| `alasan` | teks | Alasan singkat keputusan |
| `source` | `cleanlab` / `ambiguous_filter` | Sumber flag kandidat |

---

## Ringkasan Hasil Cleaning

| Metrik | Nilai |
|--------|-------|
| Total sebelum cleaning | 26.527 gambar |
| Gambar di-drop | **542** (2.04%) |
| Gambar direlabel | **18** |
| Total setelah cleaning (`folds_v2.csv`) | **25.985** gambar |

| Kelas | Jumlah (v2) | Class Weight v2 |
|-------|-------------|-----------------|
| Recyclable (0) | 9.808 | 0.883 |
| Electronic (1) | 3.933 | 2.202 |
| Organic (2) | 12.244 | 0.707 |

---

## Catatan Penting

- **Semua file di folder ini di-gitignore** kecuali `README.md` — ukurannya terlalu besar untuk di-commit.
- Gunakan Google Drive link di atas untuk mengunduh artefak.
- `folds_v2.csv` adalah versi final dan bersih — selalu pakai ini untuk training dan evaluasi.
- `cleaning_log_terisi.csv` adalah sumber kebenaran untuk audit panitia — jangan hapus.
- Jangan menjalankan ulang pipeline cleaning dari awal tanpa alasan kuat — hasilnya bisa berbeda karena review manual tidak deterministik.

---

*Track A — BDC Satria Data 2026 | Selesai: 20 Juli 2026*
