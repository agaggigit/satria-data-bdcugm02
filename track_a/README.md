# Track A — Data & Split

**Proyek:** BDC Satria Data 2026 — Klasifikasi Citra Sampah (Recyclable / Electronic / Organic)
**Peran:** Persiapan data — EDA, pembersihan label, stratified split, dan penyerahan artefak ke Track B & C
**Status:** ✅ Selesai — `folds_v2.csv` dan `class_weights_v2.npy` sudah diserahkan
**Periode:** 8–20 Juli 2026

---

## Struktur Direktori

```
track_a/
├── README.md              ← dokumen ini: gambaran besar Track A
├── HANDOFF.md             ← panduan singkat untuk Track B & C (quick start + kontrak API)
├── EDA_FINDINGS.md        ← temuan analitik dataset: distribusi, duplikat, resolusi
├── __init__.py            ← marker package Python
│
├── notebooks/             ← pipeline eksekusi (jalankan di Google Colab, berurutan)
│   ├── README.md          ← panduan tiap notebook: tujuan, input/output, cara menjalankan
│   ├── 01_eda.ipynb       ← Fase 0: scan dataset, cek corrupt, cari duplikat, EDA
│   ├── 02_split_verify.ipynb ← Fase 1: buat folds.csv, verifikasi DataLoader
│   ├── 03_oof_diagnosis.ipynb ← Fase 2: diagnosis OOF probe dari Track B
│   ├── 04_cleanlab_cleaning.ipynb ← Fase 3: temukan kandidat mislabel, review visual
│   └── 05_generate_v2.ipynb ← Fase 4: terapkan cleaning_log → hasilkan folds_v2.csv
│
├── src/                   ← modul Python yang bisa diimpor oleh Track B & C
│   ├── README.md          ← dokumentasi API per fungsi dan kelas
│   ├── dataset.py         ← WasteDataset, transform, make_fold_loaders, make_test_loader
│   ├── utils.py           ← scan dir, cek corrupt, deteksi duplikat, class weights
│   ├── oof_diagnosis.py   ← hitung margin/entropy per sampel, confusion matrix
│   ├── cleanlab_runner.py ← wrapper cleanlab find_label_issues + fallback margin-based
│   ├── ambiguous_filter.py ← filter sampel margin/entropy rendah
│   ├── contact_sheet.py   ← generate grid PNG untuk review visual, inisialisasi cleaning_log
│   └── generate_v2.py     ← terapkan keputusan cleaning_log → folds_v2.csv + weights v2
│
└── outputs/               ← semua artefak yang dihasilkan (di-gitignore, disimpan di Drive)
    ├── README.md          ← katalog file output: nama, versi, tujuan, keterangan
    ├── folds.csv          ← split v1 (sebelum cleaning)
    ├── folds_v2.csv       ← split v2 bersih ← PAKAI INI untuk training
    ├── class_weights.npy  ← bobot kelas v1
    ├── class_weights_v2.npy ← bobot kelas v2 ← PAKAI INI untuk loss function
    ├── cleaning_log_terisi.csv ← keputusan review visual yang sudah diisi
    ├── cleaning_summary.json   ← ringkasan angka cleaning untuk laporan
    ├── diagnosis/         ← output Fase 2: margin, entropy, confusion matrix
    └── contact_sheets/    ← grid PNG review visual per pasangan kelas
```

---

## Alur Kerja Besar

Track A mengerjakan empat fase secara berurutan. Tiap fase bergantung pada output fase sebelumnya.

```
Dataset mentah (26.527 train / 1.458 test)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Fase 0 — EDA & Verifikasi (01_eda.ipynb)   │
│  • Scan semua path gambar + label           │
│  • Cek gambar corrupt → skip_list.txt       │
│  • Hitung perceptual hash → deteksi         │
│    683 gambar dalam 336 grup duplikat        │
│  • Statistik resolusi, channel, distribusi  │
│  OUTPUT: df_clean.csv, eda_stats.json       │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Fase 1 — Split (02_split_verify.ipynb)     │
│  • StratifiedGroupKFold (seed=42, k=5)      │
│    → grup duplikat dijaga dalam satu fold   │
│  • Verifikasi no-overlap + proporsi kelas   │
│  • Verifikasi DataLoader 1 batch            │
│  OUTPUT: folds.csv, class_weights.npy       │
│          [→ GATE G1: diserahkan ke Track B] │
└────────────────────┬────────────────────────┘
                     │
                     ▼ (menunggu Track B: oof_probe.npy)
                     │
┌─────────────────────────────────────────────┐
│  Fase 2 — Diagnosis OOF (03_oof_diagnosis)  │
│  • Load oof_probe.npy dari Track B          │
│  • Hitung margin, entropy per sampel        │
│  • Plot confusion matrix → konfirmasi pola  │
│    kesalahan Recyclable ↔ Organic           │
│  OUTPUT: oof_diagnosis.csv, PNG             │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Fase 3 — Cleaning (04_cleanlab_cleaning)   │
│  • Cleanlab find_label_issues (OOF-based)   │
│    → label_issues.csv                       │
│  • Filter margin/entropy rendah             │
│    → ambiguous_candidates.csv               │
│  • Generate contact sheet PNG               │
│  • Review visual manual per gambar          │
│  OUTPUT: cleaning_log_terisi.csv            │
└────────────────────┬────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│  Fase 4 — Artefak v2 (05_generate_v2)       │
│  • Terapkan keputusan cleaning_log:         │
│    542 gambar di-drop, 18 direlabel         │
│  • Hitung ulang class weights               │
│  • Verifikasi integritas (no-overlap,       │
│    no duplikat lintas fold, mapping label)  │
│  OUTPUT: folds_v2.csv, class_weights_v2.npy │
│          [→ GATE G2: diserahkan ke B & C]   │
└─────────────────────────────────────────────┘
```

---

## Artefak Utama yang Diserahkan

> Semua file tersedia di Google Drive: `BDC2026apace/output_trackA/`
> Link: https://drive.google.com/drive/folders/1m8AeQGumLfoGvwP9AZUiTChyXjoUANrD?usp=sharing

### Untuk Track B & C (wajib pakai versi v2)

| File | Diserahkan ke | Keterangan |
|------|---------------|------------|
| `folds_v2.csv` | Track B & C | Split 5-fold bersih; kolom: `filepath, label, fold` |
| `class_weights_v2.npy` | Track B | Bobot `[0.883, 2.202, 0.707]` untuk `CrossEntropyLoss` |
| `src/dataset.py` | Track B & C | `WasteDataset`, `make_fold_loaders`, `make_test_loader` |

### Untuk arsip & laporan

| File | Keterangan |
|------|------------|
| `cleaning_summary.json` | Ringkasan angka cleaning: 542 drop, 18 relabel, 25.985 total |
| `cleaning_log_terisi.csv` | Log keputusan review per gambar (verifikasi panitia) |
| `eda_stats.json` | Statistik EDA dalam JSON |
| `split_metadata.json` | Seed, metode, img_size untuk reproducibility |

---

## Hasil Cleaning (Ringkasan Angka)

| Metrik | Nilai |
|--------|-------|
| Total sebelum cleaning | 26.527 gambar |
| Gambar di-drop | **542** (2.04%) |
| Gambar direlabel | **18** |
| Total setelah cleaning | **25.985** gambar |

| Kelas | Jumlah (v2) | Class Weight v2 |
|-------|-------------|-----------------|
| Recyclable (0) | 9.808 | 0.883 |
| Electronic (1) | 3.933 | 2.202 |
| Organic (2) | 12.244 | 0.707 |

---

## Konvensi yang Harus Diikuti Track B & C

| Aturan | Nilai |
|--------|-------|
| Mapping label | `0=Recyclable`, `1=Electronic`, `2=Organic` |
| Seed random | `42` |
| IMG_SIZE default | `224` |
| Normalisasi | ImageNet mean/std |
| Split yang dipakai | `folds_v2.csv` — jangan buat split sendiri |
| Cleaning | hanya data train — test tidak pernah disentuh |

---

## Dokumen Terkait di Direktori Ini

| File | Isi |
|------|-----|
| [`HANDOFF.md`](HANDOFF.md) | Quick start untuk Track B & C: contoh kode siap pakai, kontrak API DataLoader |
| [`EDA_FINDINGS.md`](EDA_FINDINGS.md) | Temuan EDA lengkap: distribusi, duplikat, resolusi, implikasi per track |
| [`notebooks/README.md`](notebooks/README.md) | Panduan tiap notebook: tujuan, prasyarat, input/output, cara menjalankan |
| [`src/README.md`](src/README.md) | Dokumentasi API per fungsi: argumen, return value, contoh kode |
| [`outputs/README.md`](outputs/README.md) | Katalog semua file output: nama, versi, tujuan |

---

*Track A — BDC Satria Data 2026 | Selesai: 20 Juli 2026*
