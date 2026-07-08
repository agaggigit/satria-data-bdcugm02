# Handoff Guide: Track A → Track B & C

> Dokumen ini adalah panduan cepat untuk tim **Track B (Modeling)** dan **Track C (Inference)**.
> Semua persiapan data, pembersihan, dan strategi splitting sudah diselesaikan oleh Track A. Kalian bisa langsung fokus ke pembuatan model.

---

## 📁 1. Lokasi Artefak

Semua output dari Track A sudah disimpan di Google Drive: **`BDC2026_TrackA_Outputs/`**
File yang kalian butuhkan:
1. `folds.csv`: Berisi daftar lengkap path gambar, label, dan pembagian 5-Fold. **Jangan buat split/k-fold sendiri.**
2. `class_weights.npy`: Bobot kelas untuk menyeimbangkan loss function.

Kalian juga membutuhkan modul `src/` dari repo ini:
- `track_a/src/dataset.py`: Berisi class `WasteDataset` dan fungsi pembuat DataLoader.

---

## 💻 2. Quick Start: Track B (Training)

Track A sudah membungkus semua kompleksitas augmentasi dan loading gambar. Kalian cukup memanggil satu fungsi: `make_fold_loaders`.

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from track_a.src.dataset import make_fold_loaders

# 1. Load data
df_folds = pd.read_csv("path/to/drive/BDC2026_TrackA_Outputs/folds.csv")

# 2. Buat Dataloader (misal untuk Fold 0)
# Ini otomatis akan menset Fold 0 sebagai Validation, dan Fold 1-4 sebagai Training
train_loader, val_loader = make_fold_loaders(
    df=df_folds, 
    val_fold=0, 
    batch_size=32, 
    num_workers=2
)

# 3. Tangani Class Imbalance (Krusial!)
# Kelas 'Electronic' sangat minoritas (14.9%), wajib gunakan class weights ini di Loss Function.
weights = torch.tensor(np.load("path/to/drive/BDC2026_TrackA_Outputs/class_weights.npy"), dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights.to(device))

# 4. Loop Training seperti biasa...
for images, labels in train_loader:
    # images shape: [32, 3, 224, 224]
    pass
```

---

## 💻 3. Quick Start: Track C (Inference)

Untuk menghasilkan prediksi `submission.csv`, urutan file test **sangat kritis**. Jangan membuat DataLoader manual dengan `ImageFolder`. Gunakan `make_test_loader` dari Track A yang dijamin menjaga urutan sesuai aturan kompetisi.

```python
from track_a.src.dataset import make_test_loader

# 1. Buat test loader
# Param 1: Path folder test asli
# Param 2: Path ke format sample submission
test_loader, df_submission = make_test_loader(
    test_dir="path/to/drive/BDC2026/test/", 
    sample_sub_path="path/to/drive/BDC2026/sample_submission.csv",
    batch_size=32,
    num_workers=2
)

# 2. Loop Inference
all_preds = []
for images in test_loader:
    # Inference model...
    # preds = model(images.to(device)).argmax(dim=1)
    # all_preds.extend(preds.cpu().numpy())
    pass

# 3. Simpan hasil (urutan dijamin aman)
df_submission['label'] = all_preds
df_submission.to_csv("submission_final.csv", index=False)
```

---

## ⚠️ 4. Konteks Teknis Tambahan (TL;DR dari EDA)

Jika kalian membuat custom model/transformasi, perhatikan batasan berikut:

1. **IMG_SIZE = 224**: Sudah ditetapkan untuk menghemat kuota GPU. Jangan diubah kecuali kalian siap menangani resource constraint.
2. **Data Leakage Aman**: Ada 683 gambar duplikat di dataset train. `folds.csv` sudah menggunakan `StratifiedGroupKFold` untuk mengelompokkan kembaran ke dalam fold yang sama. **Leakage sudah dicegah 100%.**
3. **Aspect Ratio**: Gambar memiliki rasio ekstrem (hingga 6.0). Transformasi evaluasi/test menggunakan `Resize + CenterCrop` untuk mencegah gambar terdistorsi (gepeng/memanjang).
4. **Channel Handling**: Dataset berisi gambar Grayscale dan RGBA. DataLoader Track A sudah otomatis mereduksi/menambahkannya menjadi standar RGB (3 channel). Model selalu menerima tensor `[B, 3, 224, 224]`.

> Untuk detail analitik lengkap (distribusi dimensi, outlier, dsb), silakan baca `track_a/EDA_FINDINGS.md`.
