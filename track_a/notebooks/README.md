# Track A — Panduan Notebook

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini berisi lima notebook Jupyter yang membentuk **pipeline lengkap Track A** — dari verifikasi dataset mentah hingga artefak `folds_v2.csv` yang siap diserahkan ke Track B & C. Semua notebook dirancang untuk dijalankan di **Google Colab** secara berurutan.

---

## Alur Antar Notebook

```
01_eda.ipynb
    │  Output: df_clean.csv, eda_stats.json, skip_list.txt, PNG
    ▼
02_split_verify.ipynb
    │  Input:  df_clean.csv, eda_stats.json
    │  Output: folds.csv, class_weights.npy    [→ GATE G1 ke Track B]
    ▼
        (menunggu Track B: oof_probe.npy)
    ▼
03_oof_diagnosis.ipynb
    │  Input:  oof_probe.npy (dari Track B), folds.csv
    │  Output: oof_diagnosis.csv, confusion matrix, margin/entropy PNG
    ▼
04_cleanlab_cleaning.ipynb
    │  Input:  oof_probe.npy, oof_diagnosis.csv, folds.csv
    │  Output: label_issues.csv, ambiguous_candidates.csv,
    │          contact_sheets/, cleaning_log.csv (template kosong)
    │
    │  [→ isi cleaning_log.csv secara manual setelah review visual]
    ▼
05_generate_v2.ipynb
       Input:  folds.csv, cleaning_log_terisi.csv
       Output: folds_v2.csv, class_weights_v2.npy, cleaning_summary.json
                                                    [→ GATE G2 ke Track B & C]
```

---

## Daftar Notebook

| # | File | Fase | Prasyarat | Estimasi Waktu |
|---|------|------|-----------|----------------|
| 1 | [`01_eda.ipynb`](#1-01_edaipynb) | EDA & Verifikasi Dataset | Dataset sudah di-upload ke Drive | ~2–3 jam (termasuk pHash) |
| 2 | [`02_split_verify.ipynb`](#2-02_split_verifyipynb) | Split & Verifikasi DataLoader | `df_clean.csv` dari notebook 01 | ~30–60 menit |
| 3 | [`03_oof_diagnosis.ipynb`](#3-03_oof_diagnosisipynb) | Diagnosis OOF Probe | `oof_probe.npy` dari Track B | ~30 menit |
| 4 | [`04_cleanlab_cleaning.ipynb`](#4-04_cleanlab_cleaningipynb) | Cleaning & Review Visual | `oof_diagnosis.csv` dari notebook 03 | ~2–4 jam + review manual |
| 5 | [`05_generate_v2.ipynb`](#5-05_generate_v2ipynb) | Generate Artefak v2 | `cleaning_log_terisi.csv` dari review | ~15 menit |

---

## 1. `01_eda.ipynb`

**Tujuan:** Memastikan dataset lengkap dan sehat, memahami karakteristiknya, dan menghasilkan metadata yang dibutuhkan untuk membuat split yang aman (bebas leakage).

**Yang dikerjakan:**

| Langkah | Deskripsi |
|---------|-----------|
| Verifikasi jumlah file | Assert 26.527 train / 1.458 test |
| Distribusi kelas | Hitung jumlah & proporsi per kelas; tandai kelas minoritas |
| Statistik resolusi | Sampling 2.000 gambar → distribusi width, height, aspect ratio, channel |
| Cek gambar corrupt | Coba buka tiap gambar dengan PIL; catat yang gagal ke `skip_list.txt` |
| Deteksi duplikat (pHash) | Hitung perceptual hash tiap gambar; deteksi pasangan/grup duplikat |
| Inspeksi visual | Tampilkan 5 sampel per kelas dalam grid |

**Output yang dihasilkan:**

```
outputs/
├── df_clean.csv            ← DataFrame: filepath, label, group_id, phash [INPUT notebook 02]
├── eda_stats.json          ← Statistik EDA terstruktur (juga menentukan metode split)
├── skip_list.txt           ← Daftar filepath gambar corrupt (kosong jika 0 corrupt)
├── class_distribution.png ← Bar chart + pie chart distribusi kelas
├── image_size_dist.png     ← Histogram resolusi gambar
└── sample_images.png       ← Grid 3×5 sampel gambar per kelas
```

**Konfigurasi yang perlu disesuaikan:**

```python
DATA_ROOT  = Path('/content/drive/MyDrive/BDC2026')  # ← path dataset di Drive
OUTPUT_DIR = Path('/content/track_a_outputs')
SAMPLE_N   = 2000    # sampel untuk statistik resolusi; None = semua (lebih lama)
HASH_SIZE  = 8       # ukuran pHash; jangan diubah
```

> ⏱️ **Fase pHash (Fase 1D)** bisa memakan 15–30 menit untuk 26k gambar — ini normal. Simpan `df_clean.csv` ke Drive segera setelah selesai agar tidak hilang jika Colab disconnect.

---

## 2. `02_split_verify.ipynb`

**Tujuan:** Membuat 5-fold stratified split yang bebas leakage duplikat, menghitung class weights, dan memverifikasi PyTorch DataLoader sebelum diserahkan ke Track B.

**Yang dikerjakan:**

| Langkah | Deskripsi |
|---------|-----------|
| Load artefak notebook 01 | Baca `df_clean.csv` dan `eda_stats.json` |
| Pilih metode split | Otomatis: `StratifiedKFold` jika tanpa duplikat, `StratifiedGroupKFold` jika ada duplikat |
| Buat 5-fold split | Seed 42, stratified per kelas, grup duplikat dijaga dalam fold yang sama |
| Verifikasi proporsi | Plot proporsi kelas per fold — harus seimbang |
| Verifikasi no-overlap | Assert tidak ada gambar yang muncul di lebih dari satu fold |
| Hitung class weights | Formula: `n / (n_classes × count[c])` → simpan sebagai `.npy` |
| Verifikasi DataLoader | Iterasi 1 batch train + val; assert shape `[B, 3, 224, 224]` dan label `{0,1,2}` |
| Verifikasi test loader | Buat loader test terurut sesuai `submission.csv`; assert 1.458 gambar |

**Output yang dihasilkan:**

```
outputs/
├── folds.csv               ← ⭐ Artefak handoff ke Track B & C (v1)
├── class_weights.npy       ← ⭐ Bobot untuk CrossEntropyLoss (v1)
├── split_metadata.json     ← Seed, metode, img_size untuk reproducibility
├── fold_proportions.png    ← Plot proporsi kelas tiap fold
└── batch_visualization.png ← Sanity check visual batch train vs val
```

**Konfigurasi:**

```python
RANDOM_SEED = 42   # JANGAN diubah setelah split dibuat
N_FOLDS     = 5
IMG_SIZE    = 224
```

> ✅ **GATE G1:** Setelah notebook ini selesai, `folds.csv` dan `class_weights.npy` diserahkan ke Track B untuk training awal (v1 — sebelum cleaning).

---

## 3. `03_oof_diagnosis.ipynb`

**Tujuan:** Menganalisis kualitas prediksi out-of-fold dari model Track B untuk menentukan threshold kandidat mislabel dan ambigu secara data-driven (bukan tebakan).

**Prasyarat:** `oof_probe.npy` dari Track B — probabilitas OOF hasil linear probe di atas frozen embedding SigLIP2, shape `[N, 3]`.

**Yang dikerjakan:**

| Langkah | Deskripsi |
|---------|-----------|
| Validasi shape OOF | Assert shape `[26527, 3]` dan alignment dengan `folds.csv` |
| Hitung per-sampel: margin | `max_prob - second_max_prob` — seberapa yakin model |
| Hitung per-sampel: entropy | Ketidakpastian distribusi probabilitas |
| Confusion matrix | Visualisasi pola kesalahan antar kelas |
| Distribusi per kelas | Histogram margin & entropy dipisah per kelas |
| Analisis error rate per fold | Konsistensi antar fold |

**Output yang dihasilkan:**

```
outputs/diagnosis/
├── oof_diagnosis.csv              ← Per-sampel: margin, entropy, prediksi, is_correct [INPUT notebook 04]
├── oof_confusion_matrix.png       ← Confusion matrix OOF
└── oof_margin_entropy_dist.png    ← Distribusi margin & entropy per kelas
```

> **Temuan utama (sesuai data aktual):** Recyclable ↔ Organic saling tertukar sebagai pasangan kelas yang paling sering salah. Digunakan untuk menentukan prioritas review visual di Fase 3.

---

## 4. `04_cleanlab_cleaning.ipynb`

**Tujuan:** Mengidentifikasi kandidat mislabel dan ambigu dari dua sumber independen, lalu menghasilkan contact sheet visual untuk review manual tim.

**Prasyarat:** `oof_probe.npy`, `oof_diagnosis.csv` dari notebook 03, `folds.csv`.

**Yang dikerjakan:**

| Langkah | Deskripsi |
|---------|-----------|
| **Cleanlab** | `find_label_issues(labels, oof_probe)` — deteksi mislabel berbasis confident learning |
| **Filter ambigu** | Ambil 5% sampel dengan margin/entropy terburuk |
| **Gabung & prioritas** | Double-flagged (muncul di kedua sumber) = prioritas tertinggi |
| **Contact sheet** | Grid PNG 50 gambar/halaman per pasangan kelas; border merah = probable mislabel |
| **Inisialisasi log** | `cleaning_log.csv` template kosong siap diisi manual |

**Output yang dihasilkan:**

```
outputs/
├── label_issues.csv            ← Kandidat mislabel dari cleanlab
├── ambiguous_candidates.csv    ← Sampel dengan margin/entropy rendah
├── cleaning_log.csv            ← Template kosong untuk review manual
└── contact_sheets/             ← Grid PNG per pasangan kelas
```

**Cara mengisi `cleaning_log.csv`:**

1. Download dari Drive, buka di spreadsheet editor (Excel / Google Sheets)
2. Isi kolom `keputusan`: `keep` / `relabel` / `drop`
3. Isi `label_baru` jika `relabel`; isi `alasan` singkat
4. Upload kembali ke Drive sebagai `cleaning_log_terisi.csv`
5. Jalankan **Cell 9** di notebook ini untuk validasi format

> ⚠️ **Batas keras:** total `drop` ≤ 2–3% dari dataset (~500–800 gambar). Lewat itu berarti threshold terlalu agresif. Prioritaskan `relabel` di atas `drop`.

---

## 5. `05_generate_v2.ipynb`

**Tujuan:** Menerapkan keputusan review dari `cleaning_log_terisi.csv` ke `folds.csv` untuk menghasilkan artefak bersih `folds_v2.csv` yang siap diserahkan ke Track B & C.

**Prasyarat:** `cleaning_log_terisi.csv` yang sudah diisi dan divalidasi.

**Yang dikerjakan:**

| Langkah | Deskripsi |
|---------|-----------|
| Validasi cleaning log | Cek format, tidak ada keputusan kosong, nilai label valid |
| Terapkan drop | Hapus baris dengan `keputusan=drop` dari `folds.csv` |
| Terapkan relabel | Ganti nilai `label` untuk baris dengan `keputusan=relabel` |
| Hitung ulang class weights | Dari distribusi kelas baru pasca cleaning |
| Verifikasi integritas | No-overlap, no duplikat lintas fold, assert mapping label lolos |
| Simpan artefak | `folds_v2.csv`, `class_weights_v2.npy`, `cleaning_summary.json` |

**Output yang dihasilkan:**

```
outputs/
├── folds_v2.csv               ← ⭐ Train bersih: 25.985 gambar (542 drop, 18 relabel)
├── class_weights_v2.npy       ← ⭐ Bobot kelas v2: [0.883, 2.202, 0.707]
└── cleaning_summary.json      ← Ringkasan angka cleaning untuk laporan & verifikasi panitia
```

> ✅ **GATE G2:** Setelah notebook ini selesai, `folds_v2.csv` dan `class_weights_v2.npy` diserahkan ke Track B (retrain) dan Track C (threshold tuning).

---

## Cara Menjalankan di Google Colab

### Persiapan Awal (sekali saja)

1. Upload dataset ke Drive dengan struktur:
   ```
   My Drive/
   └── BDC2026/
       ├── train/
       │   ├── 0_Recyclable/
       │   ├── 1_Electronic/
       │   └── 2_Organic/
       ├── test/
       └── submission.csv
   ```
2. Buat folder output: `My Drive/BDC2026apace/output_trackA/`
3. Klon repo di Colab: `!git clone https://github.com/agaggigit/satria-data-bdcugm02.git /content/repo`

### Menjalankan Notebook

Untuk setiap notebook:
1. Buka di Colab (klik kanan di Drive → *Open with Colaboratory*)
2. Runtime → Change runtime type → pilih **T4 GPU** (notebook 03 ke atas; CPU cukup untuk 01–02)
3. Edit variabel path di cell konfigurasi
4. Runtime → **Run all** (atau `Ctrl+F9`)
5. Setelah selesai, pastikan output sudah tersimpan di Drive

### Simpan Output ke Drive

Colab menghapus `/content/` saat session berakhir. Jalankan di akhir tiap notebook:

```python
import shutil, pathlib
DRIVE_OUTPUT = pathlib.Path('/content/drive/MyDrive/BDC2026apace/output_trackA')
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)
shutil.copytree('/content/track_a_outputs', DRIVE_OUTPUT, dirs_exist_ok=True)
print(f'Output tersimpan di {DRIVE_OUTPUT}')
```

---

## Troubleshooting Umum

| Masalah | Kemungkinan Penyebab | Solusi |
|---------|---------------------|--------|
| `AssertionError: JUMLAH TIDAK COCOK` | Dataset belum lengkap ter-upload | Cek jumlah file di Drive |
| `FileNotFoundError` | Path `DATA_ROOT` salah | Print `list(DATA_ROOT.iterdir())` untuk cek |
| Fase pHash hang sangat lama | Normal untuk 26k gambar | Tunggu atau set `HASH_SIZE=4` |
| Session expired saat pHash | Colab inaktif terlalu lama | Simpan `df_clean.csv` ke Drive lebih awal |
| `assert len(overlap) == 0` gagal | Bug pada split | Jangan lanjut — laporkan ke Track A |
| `assert len(test_ds) == 1458` gagal | Folder test tidak lengkap | Upload ulang folder `test/` ke Drive |
| `oof_probe.npy` tidak ketemu | Track B belum selesai | Minta status ke Track B |

---

*Track A — BDC Satria Data 2026 | Selesai: 20 Juli 2026*
