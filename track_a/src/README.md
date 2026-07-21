# Track A — Panduan Modul `src/`

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini berisi semua modul Python Track A. Ada dua kategori:

1. **Modul publik** (`dataset.py`) — diimpor langsung oleh Track B & C saat training dan inference.
2. **Modul internal** (`utils.py`, `oof_diagnosis.py`, dll.) — dipakai oleh notebook Track A; pada umumnya tidak perlu diimpor track lain.

---

## Daftar File & Fungsinya

| File | Kategori | Dipakai oleh | Fungsi Inti |
|------|----------|--------------|-------------|
| [`dataset.py`](#datasetpy) | **Publik** | Track A, B, C | Dataset class, transform, factory DataLoader |
| [`utils.py`](#utilspy) | Internal | Track A (notebook 01, 02) | Scan dir, cek corrupt, deteksi duplikat, class weights |
| [`oof_diagnosis.py`](#oof_diagnosispy) | Internal | Track A (notebook 03) | Hitung margin/entropy per sampel, confusion matrix |
| [`cleanlab_runner.py`](#cleanlab_runnerpy) | Internal | Track A (notebook 04) | Wrapper cleanlab + fallback margin-based |
| [`ambiguous_filter.py`](#ambiguous_filterpy) | Internal | Track A (notebook 04) | Filter sampel margin/entropy rendah |
| [`contact_sheet.py`](#contact_sheetpy) | Internal | Track A (notebook 04) | Generate grid PNG, inisialisasi cleaning_log |
| [`generate_v2.py`](#generate_v2py) | Internal | Track A (notebook 05) | Terapkan cleaning_log → folds_v2.csv & weights v2 |

---

## `dataset.py`

File inti yang mengemas seluruh logika data loading ke dalam antarmuka PyTorch standar. **Track B dan C cukup mengimpor file ini** — tidak perlu menyentuh path gambar atau transformasi secara manual.

### Konstanta Global

```python
LABEL_MAP = {
    "0_Recyclable": 0,
    "1_Electronic": 1,
    "2_Organic":    2,
}
CLASS_NAMES   = ["Recyclable", "Electronic", "Organic"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]   # normalisasi wajib untuk pretrained backbone
IMAGENET_STD  = [0.229, 0.224, 0.225]
```

---

### `get_train_transform(img_size=224)` → `T.Compose`

Transform untuk data **training**, berisi augmentasi untuk meningkatkan generalisasi model.

| Augmentasi | Parameter | Catatan |
|-----------|-----------|---------|
| `RandomResizedCrop` | scale=(0.7, 1.0) | Crop acak lalu resize ke `img_size × img_size` |
| `RandomHorizontalFlip` | p=0.5 | Flip horizontal |
| `RandomVerticalFlip` | p=0.2 | Flip vertikal (relevan untuk gambar sampah dari atas) |
| `ColorJitter` | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05 | Variasi pencahayaan & warna |
| `RandomRotation` | 15° | Rotasi acak ±15 derajat |
| `ToTensor` + `Normalize` | ImageNet mean/std | Wajib untuk pretrained backbone |

```python
from track_a.src.dataset import get_train_transform
transform = get_train_transform(img_size=224)
```

---

### `get_eval_transform(img_size=224)` → `T.Compose`

Transform untuk **validasi dan test** — tanpa augmentasi, hanya resize deterministik + normalisasi. Urutan: `Resize(int(img_size * 1.143))` → `CenterCrop(img_size)` → normalisasi. `CenterCrop` dipakai (bukan `Resize` langsung) untuk menghindari distorsi gambar dengan aspect ratio ekstrem.

> ⚠️ Selalu gunakan fungsi ini saat evaluasi dan inference, bukan `get_train_transform`.

```python
from track_a.src.dataset import get_eval_transform
transform = get_eval_transform(img_size=224)
```

---

### `class WasteDataset(Dataset)`

Kelas dataset utama, mewarisi `torch.utils.data.Dataset`, kompatibel penuh dengan `DataLoader`.

**Argumen konstruktor:**

| Parameter | Tipe | Default | Deskripsi |
|-----------|------|---------|-----------|
| `df` | `pd.DataFrame` | — | DataFrame dengan kolom `filepath`; juga `label` jika bukan test |
| `transform` | `Callable` | `None` | Transform torchvision; `None` = kembalikan PIL Image |
| `is_test` | `bool` | `False` | `True` untuk set test (tanpa label) |

**Return value `__getitem__`:**

| Mode | Return |
|------|--------|
| Train / Val (`is_test=False`) | `(image: Tensor[3, H, W], label: int)` |
| Test (`is_test=True`) | `(image: Tensor[3, H, W], filepath: str)` |

**Perilaku otomatis:**
- Buka gambar dengan PIL, otomatis konversi Grayscale/RGBA → RGB
- Validasi kolom `label` ∈ `{0, 1, 2}` saat init
- Raise `RuntimeError` dengan pesan jelas jika gambar gagal dibuka

```python
from track_a.src.dataset import WasteDataset, get_train_transform
import pandas as pd

df = pd.read_csv("outputs/folds_v2.csv")
dataset = WasteDataset(df[df["fold"] == 0], transform=get_train_transform(224))
img, label = dataset[0]
# img: Tensor[3, 224, 224], label: int (0, 1, atau 2)
```

---

### `get_loaders(fold, img_size, batch, folds_csv, ...)` → `(DataLoader, DataLoader)`

Wrapper tipis di atas `make_fold_loaders` sesuai signature kontrak Track B.

```python
from track_a.src.dataset import get_loaders

train_loader, val_loader = get_loaders(
    fold=0,
    img_size=224,
    batch=32,
    folds_csv="/path/ke/folds_v2.csv",
)
```

---

### `make_fold_loaders(folds_csv, val_fold, ...)` → `(DataLoader, DataLoader)`

Factory function utama untuk membuat pasangan train + val `DataLoader` dari satu fold.

**Argumen:**

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `folds_csv` | — | Path ke `folds_v2.csv` (wajib) |
| `val_fold` | — | Nomor fold validation: `0`–`4` (wajib) |
| `img_size` | `224` | Resolusi input; harus konsisten dengan model |
| `batch_size` | `32` | Ukuran batch |
| `num_workers` | `2` | Worker DataLoader; set `0` di Windows lokal jika error |
| `skip_filepaths` | `None` | Set filepath yang harus dilewati (dari `skip_list.txt`) |

**Jaminan keamanan:**
- Validasi kolom `folds_v2.csv` ada (`filepath`, `label`, `fold`)
- Assert no-overlap antara train dan val set
- Train loader: `shuffle=True`, `drop_last=True`
- Val loader: `shuffle=False`

```python
from track_a.src.dataset import make_fold_loaders

train_loader, val_loader = make_fold_loaders(
    folds_csv="/path/ke/folds_v2.csv",
    val_fold=0,
    img_size=224,
    batch_size=32,
    num_workers=2,
)

for imgs, labels in train_loader:
    # imgs: Tensor[32, 3, 224, 224]
    # labels: Tensor[32], nilai ∈ {0, 1, 2}
    ...
```

---

### `make_test_loader(test_dir, submission_csv, ...)` → `DataLoader`

Factory function untuk test `DataLoader` dengan urutan gambar yang dijamin sesuai `submission.csv`.

> ⚠️ **Krusial:** urutan gambar yang salah menghasilkan submission yang salah. Selalu gunakan fungsi ini — jangan buat test loader manual dengan `ImageFolder`.

**Argumen:**

| Parameter | Default | Deskripsi |
|-----------|---------|-----------|
| `test_dir` | — | Direktori gambar test (wajib) |
| `submission_csv` | — | Template `submission.csv` sebagai penentu urutan (wajib) |
| `img_size` | `224` | Resolusi input |
| `batch_size` | `32` | Ukuran batch |
| `num_workers` | `2` | Worker DataLoader |

**Perilaku:**
- Urutan gambar mengikuti kolom pertama `submission.csv` — `shuffle=False` selalu
- Coba fallback ekstensi (`.jpg`, `.jpeg`, `.png`) jika nama file tanpa ekstensi
- Return per item: `(image_tensor, filepath_str)`

```python
from track_a.src.dataset import make_test_loader

test_loader = make_test_loader(
    test_dir="/path/ke/BDC2026/test",
    submission_csv="/path/ke/submission.csv",
    img_size=224,
    batch_size=64,
)

for imgs, filepaths in test_loader:
    preds = model(imgs.to(device)).argmax(dim=1)
    # preds terurut persis sesuai submission.csv
```

---

### Sanity Check (jalankan sebagai script)

```bash
# Dari folder track_a/
python src/dataset.py outputs/folds_v2.csv
```

Akan membuat loader fold 0, ambil 1 batch, assert shape dan label. Tidak ada error = siap dipakai.

---

## `utils.py`

Fungsi helper untuk notebook EDA dan split. Umumnya tidak diimpor langsung oleh Track B/C.

### Fungsi-fungsi

#### `scan_train_dir(train_dir)` → `pd.DataFrame`
Scan folder `train/`, kumpulkan path gambar + label. Kolom output: `filepath`, `label_name`, `label`.

#### `scan_test_dir(test_dir)` → `pd.DataFrame`
Scan folder `test/`, urutkan alfanumerik. Kolom: `filepath`, `filename`.

#### `check_corrupt(df, verbose=True)` → `(pd.DataFrame, List[str])`
Coba buka tiap gambar dengan PIL. Gambar gagal → masuk `skip_list`. O(n), ~5–10 menit untuk 26k gambar.

#### `get_image_stats(df, sample_n=None)` → `pd.DataFrame`
Kumpulkan width, height, channels, aspect_ratio. Set `sample_n` untuk sampling acak.

#### `compute_phash(filepath, hash_size=8)` → `str | None`
Hitung perceptual hash satu gambar. Return `None` jika gagal dibuka.

#### `find_duplicates(df, hash_size=8, threshold=5)` → `Dict[str, List[str]]`
Deteksi duplikat/near-duplikat via pHash. `threshold=0` = exact match saja (cepat, O(n)). `threshold>0` = near-duplicate (lambat, O(n²)).

```python
from track_a.src.utils import find_duplicates
groups = find_duplicates(df, threshold=0)
# groups = {"filepath_A": ["filepath_A", "filepath_B"], ...}
```

#### `assign_group_ids(df, dup_groups)` → `pd.DataFrame`
Tambahkan kolom `group_id` berdasarkan hasil `find_duplicates()`. Dipakai `StratifiedGroupKFold`.

#### `compute_class_weights(labels, n_classes=3)` → `np.ndarray`
Hitung bobot kelas: `weight[c] = n_samples / (n_classes × count[c])`.

```python
weights = compute_class_weights(df["label"].tolist())
# Misal: array([0.883, 2.202, 0.707])

w_tensor = torch.tensor(weights, dtype=torch.float).to(device)
criterion = torch.nn.CrossEntropyLoss(weight=w_tensor)
```

#### `save_eda_stats(stats, path)` / `save_class_weights(weights, path)`
Helper I/O — simpan dict ke JSON dan array ke `.npy`.

---

## `oof_diagnosis.py`

Dipakai oleh notebook `03_oof_diagnosis.ipynb`.

**Fungsi utama: `build_diagnosis_df(oof, folds_df)`**
Gabungkan `oof_probe.npy` dengan `folds.csv` dan hitung per-sampel: `margin`, `entropy`, `pred_class`, `is_correct`. Return DataFrame yang menjadi input notebook 04.

**Plot functions:** `plot_confusion_matrix()`, `plot_margin_entropy_dist()` — menghasilkan PNG ke folder `outputs/diagnosis/`.

---

## `cleanlab_runner.py`

Dipakai oleh notebook `04_cleanlab_cleaning.ipynb`.

**Fungsi utama: `run_full_cleanlab(oof, folds_df, diag_df, output_dir, ...)`**
Jalankan `cleanlab.filter.find_label_issues()` pada OOF. Jika cleanlab tidak tersedia, fallback ke `run_margin_based_issues()` yang mendeteksi kandidat berdasarkan margin rendah + prediksi salah. Output: `label_issues.csv`.

---

## `ambiguous_filter.py`

Dipakai oleh notebook `04_cleanlab_cleaning.ipynb`.

**Fungsi utama: `run_full_ambiguous_filter(diag_df, folds_df, output_dir, ...)`**
Ambil N% sampel dengan margin/entropy terburuk sebagai kandidat ambigu. Tandai `is_double_flagged=True` jika juga ada di hasil cleanlab. Output: `ambiguous_candidates.csv`.

---

## `contact_sheet.py`

Dipakai oleh notebook `04_cleanlab_cleaning.ipynb`.

**Fungsi utama:**
- `generate_contact_sheets(candidates, output_dir, ...)` — buat grid PNG 50 gambar/halaman; border merah = probable mislabel, oranye = ambigu, hijau = normal.
- `init_cleaning_log(all_candidates, output_path)` — inisialisasi `cleaning_log.csv` template kosong siap diisi.
- `validate_cleaning_log(log_path)` — validasi format log: semua keputusan terisi, nilai valid, label baru ada jika relabel.

---

## `generate_v2.py`

Dipakai oleh notebook `05_generate_v2.ipynb`.

**Fungsi utama: `apply_cleaning_log(folds_df, cleaning_log_df, ...)`**
Terapkan keputusan `keep/relabel/drop` dari log review ke `folds.csv`. Hasilkan `folds_v2.csv`, hitung ulang `class_weights_v2.npy`, simpan `cleaning_summary.json`.

---

## Ringkasan: Siapa Pakai Apa

| Fungsi / Class | Track A (nb) | Track B | Track C |
|----------------|:---:|:---:|:---:|
| `get_train_transform` | ✅ | ✅ (via loader) | — |
| `get_eval_transform` | ✅ | ✅ (via loader) | ✅ (via loader) |
| `WasteDataset` | ✅ | ✅ (via loader) | ✅ (via loader) |
| `get_loaders` | ✅ | ✅ **utama** | — |
| `make_fold_loaders` | ✅ | ✅ **utama** | — |
| `make_test_loader` | ✅ | — | ✅ **utama** |
| `compute_class_weights` | ✅ | ✅ (load .npy) | — |
| `scan_train_dir` | ✅ | — | — |
| `check_corrupt` | ✅ | — | — |
| `find_duplicates` | ✅ | — | — |
| `run_full_cleanlab` | ✅ | — | — |
| `generate_contact_sheets` | ✅ | — | — |
| `apply_cleaning_log` | ✅ | — | — |

---

## Cara Import

Jika menjalankan dari luar folder `track_a/` (misalnya dari Colab):

```python
import sys
sys.path.insert(0, '/content/repo')  # atau path ke root repo

from track_a.src.dataset import get_loaders, make_test_loader
from track_a.src.utils import compute_class_weights
```

Jika menjalankan dari dalam folder `track_a/`:

```python
from src.dataset import make_fold_loaders
from src.utils import scan_train_dir
```

---

*Track A — BDC Satria Data 2026 | Selesai: 20 Juli 2026*
