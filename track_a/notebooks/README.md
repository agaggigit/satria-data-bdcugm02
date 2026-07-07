# Track A — Panduan Notebook

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini berisi dua notebook Jupyter yang mencakup seluruh pipeline Track A,
dari verifikasi dataset mentah hingga artefak siap-handoff ke Track B & C.

---

## Daftar Notebook

| # | File | Fase | Estimasi Waktu |
|---|------|------|----------------|
| 1 | [`01_eda.ipynb`](#1-01_edaipynb--eda--verifikasi-dataset) | Fase 0 + 1 | ~1–2 jam (+ ~30 menit untuk cek duplikat) |
| 2 | [`02_split_verify.ipynb`](#2-02_split_verifyipynb--split--verifikasi-dataloader) | Fase 2 | ~30–60 menit |

> **Urutan wajib:** Jalankan `01_eda.ipynb` dulu sampai selesai sebelum membuka `02_split_verify.ipynb`.
> Notebook kedua bergantung pada output (`df_clean.csv`, `eda_stats.json`) dari notebook pertama.

---

## 1. `01_eda.ipynb` — EDA & Verifikasi Dataset

### Tujuan
Memastikan dataset bersih, memahami karakteristiknya, dan menghasilkan metadata
yang dibutuhkan untuk membuat split yang aman (bebas leakage).

### Yang Dilakukan

| Seksi | Deskripsi |
|-------|-----------|
| **Fase 0 — Setup & Konfigurasi** | Install library, mount Google Drive, set path dataset, verifikasi jumlah file (assert 26.527 train / 1.458 test) |
| **Fase 1A — Distribusi Kelas** | Hitung jumlah gambar per kelas, buat bar chart + pie chart, tandai kelas minoritas |
| **Fase 1B — Statistik Ukuran Gambar** | Sampling 2.000 gambar → distribusi width, height, aspect ratio, channel; hasilkan histogram |
| **Fase 1C — Cek Gambar Corrupt** | Coba buka setiap gambar dengan PIL; catat yang gagal ke `skip_list.txt` |
| **Fase 1D — Cek Duplikat (pHash)** | Hitung perceptual hash tiap gambar, deteksi duplikat/near-duplikat, buat `group_id` |
| **Fase 1E — Inspeksi Visual** | Tampilkan 5 sampel per kelas dalam grid; cek manual apakah ada mislabel |
| **Save Output** | Simpan `df_clean.csv`, `eda_stats.json`, `skip_list.txt`, dan plot PNG |

### Output yang Dihasilkan

```
/content/track_a_outputs/
├── df_clean.csv            ← DataFrame bersih + phash + group_id (input notebook 02)
├── eda_stats.json          ← Ringkasan statistik EDA (untuk report)
├── skip_list.txt           ← Daftar filepath gambar corrupt
├── class_distribution.png ← Bar chart + pie chart distribusi kelas
├── image_size_dist.png     ← Histogram ukuran gambar
└── sample_images.png       ← Grid 3×5 sampel gambar per kelas
```

### Variabel Konfigurasi yang Perlu Diubah

```python
DATA_ROOT  = Path('/content/drive/MyDrive/BDC2026')  # ← path dataset di Drive kamu
OUTPUT_DIR = Path('/content/track_a_outputs')         # boleh dibiarkan
SAMPLE_N   = 2000   # jumlah sampel untuk statistik ukuran gambar (ubah ke None untuk semua)
HASH_SIZE  = 8      # ukuran pHash; jangan diubah kecuali ada alasan
```

---

## 2. `02_split_verify.ipynb` — Split & Verifikasi DataLoader

### Tujuan
Membuat stratified 5-fold split yang bebas leakage, menghitung class weights,
dan memverifikasi PyTorch DataLoader sebelum diserahkan ke Track B & C.

### Yang Dilakukan

| Seksi | Deskripsi |
|-------|-----------|
| **Setup** | Install library, mount Drive, load `df_clean.csv` dan `eda_stats.json` dari notebook 01 |
| **Stratified 5-Fold Split** | Otomatis pilih `StratifiedKFold` atau `StratifiedGroupKFold` (jika ada duplikat) berdasarkan `eda_stats.json` |
| **Verifikasi Proporsi** | Plot bar proporsi kelas tiap fold; pastikan distribusi seimbang |
| **Verifikasi No-Overlap** | Assert tidak ada gambar yang muncul di dua fold sekaligus |
| **Simpan `folds.csv`** | Artefak utama handoff: kolom `filepath, label, fold` + metadata seed |
| **Hitung Class Weights** | `n_samples / (n_classes × count_per_class)` → simpan sebagai `class_weights.npy` |
| **Verifikasi DataLoader** | Iterasi 1 batch train + val; assert shape & label valid |
| **Visualisasi Batch** | Denormalize + tampilkan 8 gambar train (dengan augmentasi) vs 8 gambar val (tanpa augmentasi) |
| **Test Loader** | Buat loader test dengan urutan PERSIS sesuai `submission.csv`; assert 1458 gambar |
| **Final Check** | Verifikasi semua file output ada dan tidak kosong |

### Output yang Dihasilkan

```
/content/track_a_outputs/
├── folds.csv               ← ⭐ Artefak utama → Track B & C
├── class_weights.npy       ← ⭐ Untuk CrossEntropyLoss di Track B
├── split_metadata.json     ← Seed, metode split, img_size untuk reproducibility
└── fold_proportions.png    ← Plot proporsi kelas tiap fold
    batch_visualization.png ← Sanity check visual batch train vs val
```

### Variabel Konfigurasi yang Perlu Diubah

```python
DATA_ROOT   = Path('/content/drive/MyDrive/BDC2026')  # ← sama dengan notebook 01
OUTPUT_DIR  = Path('/content/track_a_outputs')         # ← sama dengan notebook 01
RANDOM_SEED = 42    # ← JANGAN diubah setelah split; dicatat di split_metadata.json
N_FOLDS     = 5     # standar 5-fold
IMG_SIZE    = 224   # resolusi input; sesuaikan dengan backbone yang akan dipakai Track B
```

---

## Cara Menjalankan di Google Colab

### Persiapan Awal (sekali saja)

1. **Upload dataset ke Google Drive**
   - Buat folder `BDC2026/` di `My Drive`
   - Upload folder `train/`, `test/`, dan `submission.csv` ke dalamnya
   - Struktur akhir di Drive:
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

2. **Upload notebook ke Colab**
   - Buka [colab.research.google.com](https://colab.research.google.com)
   - Klik **File → Upload notebook** → pilih `01_eda.ipynb`
   - Atau: klik kanan file di Google Drive → **Open with → Google Colaboratory**

### Menjalankan `01_eda.ipynb`

```
1. Buka di Colab
2. Runtime → Change runtime type → pilih T4 GPU (atau CPU sudah cukup untuk EDA)
3. Jalankan cell pertama (!pip install ...) → tunggu selesai
4. Jalankan cell mount Drive → izinkan akses
5. Edit DATA_ROOT sesuai lokasi dataset kamu
6. Runtime → Run all  (atau Ctrl+F9)
7. Perhatian: Fase 1D (pHash) bisa 15–30 menit, ini normal
8. Setelah selesai, cek folder output_dir di panel Files (ikon folder kiri)
```

> 💡 **Tips:** Jika Colab disconnect saat Fase 1D, output sebagian hilang. Simpan
> `df_clean.csv` ke Drive sesegera mungkin setelah cell "Save Output Fase 1" selesai.

### Menjalankan `02_split_verify.ipynb`

```
1. Pastikan output notebook 01 sudah tersimpan (df_clean.csv, eda_stats.json)
2. Buka 02_split_verify.ipynb di Colab (tab baru)
3. Edit DATA_ROOT dan OUTPUT_DIR agar sama persis dengan notebook 01
4. Runtime → Run all
5. Perhatikan cell "Verifikasi No-Overlap" — harus lolos tanpa error
6. Setelah selesai, download / simpan folds.csv dan class_weights.npy ke Drive
```

### Menyimpan Output ke Drive (Penting!)

Colab menghapus `/content/` saat session berakhir. Jalankan cell ini di akhir
setiap notebook untuk menyimpan output ke Drive:

```python
import shutil
DRIVE_OUTPUT = Path('/content/drive/MyDrive/BDC2026_TrackA_Outputs')
DRIVE_OUTPUT.mkdir(parents=True, exist_ok=True)
shutil.copytree(OUTPUT_DIR, DRIVE_OUTPUT, dirs_exist_ok=True)
print(f'Output disalin ke {DRIVE_OUTPUT}')
```

---

## Skema Penggunaan Alternatif

### Lokal (Jupyter / VS Code)

Jika menjalankan di komputer lokal, ubah path dan hapus bagian mount Drive:

```python
# Ganti bagian mount Drive dengan:
DATA_ROOT  = Path('../../datasets/BDC2026')     # relatif dari folder notebooks/
OUTPUT_DIR = Path('../outputs')                  # simpan ke track_a/outputs/
```

Pastikan environment sudah terinstall:

```bash
pip install jupyter pandas numpy matplotlib seaborn pillow imagehash scikit-learn torch torchvision tqdm
jupyter notebook
```

### Kaggle Notebook

1. Buat notebook baru di [kaggle.com/code](https://kaggle.com/code)
2. Di panel kanan → **+ Add Data** → pilih dataset privat yang sudah diupload
3. Dataset akan di-mount di `/kaggle/input/nama-dataset/`
4. Ubah `DATA_ROOT`:

```python
DATA_ROOT  = Path('/kaggle/input/bdc2026-dataset/BDC2026')
OUTPUT_DIR = Path('/kaggle/working/track_a_outputs')
```

> ⚠️ Kaggle limit GPU 30 jam/minggu — **gunakan CPU untuk EDA**, hemat GPU untuk training di Track B.

---

## Alur Data Antar Notebook

```
01_eda.ipynb
    │
    ├─── df_clean.csv      ──→  02_split_verify.ipynb (input wajib)
    ├─── eda_stats.json    ──→  02_split_verify.ipynb (menentukan metode split)
    └─── skip_list.txt     ──→  (opsional) Track B jika ingin filter corrupt

02_split_verify.ipynb
    │
    ├─── folds.csv         ──→  ⭐ Track B (training) & Track C (evaluasi)
    ├─── class_weights.npy ──→  ⭐ Track B (weighted loss)
    └─── split_metadata.json ─→ Dokumentasi (seed, metode, img_size)
```

---

## Troubleshooting Umum

| Masalah | Kemungkinan Penyebab | Solusi |
|---------|---------------------|--------|
| `AssertionError: JUMLAH TIDAK COCOK` | Dataset belum lengkap ter-upload | Cek jumlah file di Drive, upload ulang |
| `FileNotFoundError` saat buka gambar | Path `DATA_ROOT` salah | Cek dengan `print(list(DATA_ROOT.iterdir()))` |
| Fase 1D hang sangat lama | Dataset besar, CPU lambat | Normal; tunggu atau perkecil `HASH_SIZE=4` |
| Session Colab expired | Inaktif terlalu lama | Simpan `df_clean.csv` ke Drive lebih awal |
| `assert len(overlap) == 0` gagal | Bug pada split | Laporkan ke penanggungjawab Track A |
| `assert len(test_ds) == 1458` gagal | File test tidak lengkap di Drive | Upload ulang folder `test/` |

---

## Dependensi

```
imagehash>=4.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
Pillow>=10.0
scikit-learn>=1.3
torch>=2.0
torchvision>=0.15
tqdm>=4.65
```

---

*Dibuat untuk BDC Satria Data 2026 — Track A | Last updated: 8 Juli 2026*
