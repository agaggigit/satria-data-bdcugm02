# Track A — Temuan EDA

> Dokumen ini adalah referensi untuk **Track B** (training) dan **Track C** (evaluasi).
> Semua angka berasal dari output `01_eda.ipynb` yang dijalankan pada 8 Juli 2026.

---

## Dataset

| | Train | Test |
|--|-------|------|
| **Jumlah gambar** | 26.527 | 1.458 |
| **Gambar corrupt** | 0 | — |
| **Label tersedia** | Ya | Tidak |

---

## Distribusi Kelas (Train)

| Label | Kelas | Jumlah | Proporsi |
|-------|-------|--------|----------|
| 0 | Recyclable | 9.999 | 37.7% |
| 1 | Electronic | 3.961 | **14.9%** ← minoritas |
| 2 | Organic | 12.567 | 47.4% |

### Implikasi untuk Track B

- Kelas **Electronic** hanya sepertiga dari Recyclable dan seperempat dari Organic
- **Wajib pakai `class_weights`** pada loss function — sudah dihitung di `outputs/class_weights.npy`
- Cara pakai:
  ```python
  import numpy as np, torch, torch.nn as nn
  weights = torch.tensor(np.load("track_a/outputs/class_weights.npy"), dtype=torch.float).to(device)
  criterion = nn.CrossEntropyLoss(weight=weights)
  ```
- Pertimbangkan juga augmentasi tambahan khusus kelas Electronic (oversampling / augmentasi lebih agresif)

---

## Statistik Ukuran Gambar (sampel 2.000 gambar)

| Dimensi | Min | Median | Max |
|---------|-----|--------|-----|
| Width | 56 px | 255 px | 8.000 px |
| Height | 91 px | 194 px | 6.000 px |

**Distribusi aspect ratio:** mayoritas antara 1.0–2.0 (landscape ringan), tapi ada outlier hingga 6.0

### Implikasi untuk Track B

- Median gambar sudah kecil (~255×194). Resize ke 224 berarti sedikit downscale width dan sedikit upscale height
- **Rekomendasi IMG_SIZE: 224** (default pretrained backbone) — cukup representatif
- Alternatif: 384 jika GPU memungkinkan dan ingin lebih detail, tapi waktu training 3× lebih lama
- **Eval/Test transform:** gunakan `Resize + CenterCrop` (bukan `Resize` langsung ke square) untuk menghindari distorsi gambar dengan aspect ratio ekstrem
- Train transform `RandomResizedCrop` sudah di-set `scale=(0.7, 1.0)` — aman untuk gambar kecil

---

## Channels (Format Gambar)

Dataset mengandung tiga jenis channel:

| Channel | Format | Jumlah (estimasi) |
|---------|--------|--------------------|
| 1 | Grayscale | Ada |
| 3 | RGB | Mayoritas |
| 4 | RGBA (ada alpha) | Ada |

### Implikasi

- **Sudah di-handle otomatis** oleh `WasteDataset` via `.convert("RGB")` saat loading
- Track B dan C tidak perlu melakukan apapun — gambar selalu keluar sebagai tensor `[3, H, W]`

---

## Duplikat

| Metrik | Nilai |
|--------|-------|
| Grup duplikat (exact pHash) | 336 |
| Total gambar dalam grup | 683 |
| Persentase dari train | 2.6% |

**Contoh duplikat yang ditemukan:**
- `1_Electronic/11.image-23.png` dan `13.image-23.png` (dan 2 file lain) → hash yang sama
- Duplikat muncul di dalam kelas yang sama maupun lintas kelas

### Implikasi untuk Track A (Split)

- `need_group_aware_split = true` → split menggunakan **StratifiedGroupKFold**
- Gambar dalam grup duplikat dijamin masuk ke fold yang sama → **tidak ada leakage**
- File `outputs/folds.csv` sudah mengimplementasikan ini

### Implikasi untuk Track B & C

- Tidak ada yang perlu dilakukan — leakage sudah dicegah di level split
- Jangan membuat split sendiri — selalu pakai `folds.csv` dari Track A

---

## Keputusan yang Perlu Disepakati dengan Track B

| Keputusan | Opsi | Status |
|-----------|------|--------|
| `IMG_SIZE` | 224 (default) atau 384 (lebih detail) | ✅ **224** — komputasi terbatas (Kaggle 30h GPU/minggu) |
| Eval transform | `Resize+CenterCrop` vs `Resize` langsung | ✅ **Resize+CenterCrop** — sudah diupdate di `dataset.py` |
| Augmentasi tambahan kelas Electronic | Oversampling / lebih agresif | ⏳ Diserahkan ke Track B |

---

## File Output Track A

Semua file ini sudah tersedia di Google Drive folder `BDC2026_TrackA_Outputs/`:

| File | Keterangan |
|------|-----------|
| `folds.csv` | Split 5-fold, kolom: `filepath, label, fold` |
| `class_weights.npy` | Bobot `[w0, w1, w2]` untuk CrossEntropyLoss |
| `df_clean.csv` | DataFrame lengkap + phash + group_id |
| `eda_stats.json` | Ringkasan statistik dalam format JSON |
| `split_metadata.json` | Seed, metode split, img_size yang dipakai |

---

*Generated: 8 Juli 2026 — Track A, BDC Satria Data 2026*
