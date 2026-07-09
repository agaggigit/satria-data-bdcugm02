# BDC Satria Data 2026 — Klasifikasi Citra Sampah

**Kompetisi:** Big Data Challenge (BDC) Satria Data 2026  
**Task:** Klasifikasi citra sampah ke 3 kelas: Recyclable / Electronic / Organic  
**Metrik Utama:** Macro-averaged F1-Score

---

## Struktur Tim & Track

| Track | Peran | Timeline |
|-------|-------|----------|
| **Track A** | Data & Split — EDA, stratified fold, dataloader | 8–10 Juli 2026 |
| **Track B** | Model Training — arsitektur, fine-tuning, training loop | 9–12 Juli 2026 |
| **Track C** | Evaluasi & Submission — inference, ensemble, submit | 12–14 Juli 2026 |

> **Critical path:** Track B & C tidak bisa jalan sebelum Track A menyerahkan `folds.csv` + `dataset.py` yang sudah teruji.

---

## Dataset

```
datasets/BDC2026/
├── train/
│   ├── 0_Recyclable/    # kelas 0
│   ├── 1_Electronic/    # kelas 1
│   └── 2_Organic/       # kelas 2
├── test/                # 1.458 gambar, tanpa label
└── submission.csv       # template submission (urutan wajib diikuti)
```

**Total train:** 26.527 gambar | **Total test:** 1.458 gambar  
> ⚠️ Dataset di-gitignore — unduh sendiri dari `bit.ly/datasetbdc2026`

---

## Struktur Repository

```
.
├── track_a/             # Track A: Data & Split
│   ├── notebooks/
│   │   ├── 01_eda.ipynb          # EDA: distribusi, duplikat, corrupt
│   │   └── 02_split_verify.ipynb # Verifikasi fold & dataloader
│   ├── src/
│   │   ├── dataset.py            # PyTorch Dataset + DataLoader (artefak utama)
│   │   └── utils.py              # Helper functions
│   └── outputs/
│       ├── folds.csv             # Artefak handoff ke Track B & C
│       ├── class_weights.npy     # Bobot kelas untuk Track B
│       └── eda_stats.json        # Statistik EDA
├── track_b/             # Track B: Model & Training
│   ├── notebooks/       # Orchestrator tipis (sanity, baseline fold 0, full 5-fold)
│   ├── src/
│   │   ├── config.py             # CFG — sumber kebenaran konstanta
│   │   ├── model.py              # ConvNeXt-Tiny via timm
│   │   ├── train.py              # Training loop (AMP + grad clipping)
│   │   └── ...                   # seed_utils, losses_metrics, scheduler, dst.
│   └── outputs/
│       ├── fold{0..4}.pt         # Checkpoint terbaik per fold → Track C
│       └── oof.npy               # Probabilitas OOF [N, 3] → Track C
├── track_c/             # Track C: Evaluasi & Submission
│   ├── notebooks/
│   │   └── 01_track_c_pipeline.ipynb  # Orchestrator Colab (jalankan dari sini)
│   ├── src/
│   │   ├── config_c.py           # CFG_C — path, backbone, TTA, team_name
│   │   ├── inference.py          # load_model() + predict_test() + assert_label_mapping()
│   │   ├── threshold_tuning.py   # tune_thresholds_oof() + apply_thresholds()
│   │   ├── ensemble_tta.py       # run_5fold_ensemble_inference() + TTA
│   │   ├── generate_submission.py# Pipeline end-to-end → submission_apace.csv
│   │   └── validator.py          # validate_submission() — wajib lolos sebelum upload
│   └── outputs/                  # Output disimpan di Google Drive (lihat config_c.py)
├── datasets/            # Dataset lokal (di-gitignore)
└── README.md
```

---

## Artefak Handoff (Track A → Track B & C)

| File | Tujuan | Isi |
|------|--------|-----|
| `track_a/outputs/folds.csv` | Track B & C | `filepath, label, fold` — **sumber kebenaran split** |
| `track_a/src/dataset.py` | Track B | `Dataset` class + transform + fungsi dataloader |
| `track_a/outputs/class_weights.npy` | Track B | Bobot untuk weighted CrossEntropy loss |
| `track_a/outputs/eda_stats.json` | Report | Distribusi kelas, ukuran gambar, jumlah duplikat |

---

## Artefak Handoff (Track B → Track C)

| File | Tujuan | Isi |
|------|--------|-----|
| `track_b/outputs/fold{0..4}.pt` | Track C | Checkpoint terbaik per fold (by val Macro-F1) — ensemble + inference test |
| `track_b/outputs/oof.npy` | Track C | Probabilitas OOF `[N, 3]`, index cocok `folds.csv` — threshold tuning |
| `track_b/outputs/cv_summary.json` | Track C + Report | CV Macro-F1 (mean ± std) + config inference |

> ⚠️ Semua artefak `outputs/` di-gitignore (ukuran besar) — share via Google Drive / Kaggle Dataset privat.

---

## Cara Mulai (Track A)

1. Mount dataset di Google Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. Buka `track_a/notebooks/01_eda.ipynb` di Colab
3. Update path `DATA_ROOT` sesuai lokasi dataset kamu
4. Jalankan semua sel secara berurutan

---

## Cara Mulai (Track B)

1. Tunggu GATE 2 hijau dari Track A (`folds.csv` + `dataset.py` + `class_weights.npy` di storage bersama)
2. Sebelum itu, kerjakan yang tidak butuh data asli: `config.py`, `scheduler.py`, scaffold training loop (pakai `dataset_stub.py`)
3. Training di Google Colab (T4, `num_workers=2`) atau Kaggle — lihat `track_b/notebooks/README.md`
4. Konvensi terkunci: seed `42`, img `224`, norm ImageNet, mapping `0/1/2` — detail di `track_b/src/README.md`

---

## Cara Mulai (Track C)

> **Prasyarat:** GATE 3 sudah hijau — Track B sudah menyerahkan `fold0.pt`..`fold4.pt` dan `oof.npy` ke Google Drive.

1. Mount Drive dan clone repo di Google Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !git clone https://github.com/agaggigit/satria-data-bdcugm02.git
   import sys; sys.path.insert(0, '/content/satria-data-bdcugm02')
   ```
2. Buka `track_c/notebooks/01_track_c_pipeline.ipynb` — semua langkah sudah tersusun
3. Pastikan path di `track_c/src/config_c.py` sudah sesuai lokasi Drive kamu (`DRIVE_BASE_PATH`)
4. Jalankan pipeline secara berurutan:
   - **Step 1** — Assert label mapping (safety check)
   - **Step 2** — Threshold tuning di OOF
   - **Step 3** — Ensemble + TTA inference pada test data
   - **Step 4** — Apply threshold → generate `submission_apace.csv`
   - **Step 5** — Validator format wajib lolos sebelum upload

---

## Cara Verifikasi Manual Track C

Berikut langkah untuk memastikan semua kode Track C benar **sebelum checkpoint Track B tersedia**:

### 1. Cek struktur file sudah lengkap
```bash
ls track_c/src/
# Harus ada: config_c.py, inference.py, threshold_tuning.py,
#            ensemble_tta.py, generate_submission.py, validator.py, __init__.py
```

### 2. Smoke test validator dengan data dummy (di Colab)
```python
import sys
sys.path.insert(0, '/content/satria-data-bdcugm02')
import pandas as pd, numpy as np

# Buat submission dummy yang VALID
template = pd.read_csv('/path/ke/submission.csv')  # template panitia
dummy = template.copy()
dummy['predicted'] = np.random.choice([0,1,2], size=len(dummy))
dummy.to_csv('/tmp/dummy_submission.csv', index=False)

from track_c.src.validator import validate_submission
result = validate_submission('/tmp/dummy_submission.csv', '/path/ke/submission.csv')
print('VALID' if result else 'INVALID')  # harus VALID
```

### 3. Smoke test threshold tuning dengan data dummy
```python
import numpy as np
from track_c.src.threshold_tuning import tune_thresholds_oof, apply_thresholds

# Dummy OOF probs & labels
np.random.seed(42)
oof_probs  = np.random.dirichlet([1,1,1], size=500)  # [500, 3]
true_labels = np.random.choice([0,1,2], size=500)

thresholds = tune_thresholds_oof(oof_probs, true_labels)
print('Thresholds:', thresholds)  # harus 3 nilai float

preds = apply_thresholds(oof_probs, thresholds)
print('Preds shape:', preds.shape)  # harus (500,)
print('Unique labels:', np.unique(preds))  # harus subset {0,1,2}
```

### 4. Cek assert label mapping
```python
from track_c.src.config_c import CFG_C
from track_c.src.inference import assert_label_mapping
assert_label_mapping(CFG_C)  # harus tidak error
print('Label mapping OK:', CFG_C.label_map)
```

### 5. Cek nama file output sudah benar
```python
from track_c.src.config_c import CFG_C
team_suffix = f'_{CFG_C.team_name}' if CFG_C.team_name else ''
print(f'Output file: submission{team_suffix}.csv')
# Harus print: submission_apace.csv
```

---

## Environment

- Python 3.10+
- PyTorch ≥ 2.0
- torchvision, PIL/Pillow
- pandas, numpy, matplotlib, seaborn
- imagehash (untuk perceptual hash / cek duplikat — Track A)
- scikit-learn (StratifiedKFold — Track A; Macro-F1 — Track B/C)
- timm (backbone pretrained ConvNeXt — Track B)

```bash
pip install torch torchvision timm pandas numpy matplotlib seaborn imagehash scikit-learn
```

---

## Quick Checklist Track A

- [ ] Dataset terunduh & jumlah terverifikasi (26.527 / 1.458)
- [ ] Distribusi kelas + kelas minoritas diketahui
- [ ] Statistik ukuran gambar → resolusi input ditetapkan
- [ ] Daftar gambar corrupt (skip-list)
- [ ] Cek duplikat selesai → group-aware split jika perlu
- [ ] `folds.csv` dibuat (stratified 5-fold, seed dicatat)
- [ ] Proporsi kelas per fold & no-overlap terverifikasi
- [ ] `dataset.py` bisa iterasi 1 batch tanpa error
- [ ] Transform train vs eval + normalisasi ImageNet
- [ ] `class_weights.npy` dihitung
- [ ] Assert mapping label 0/1/2 lolos
- [ ] Test loader terurut 1..1458
- [ ] Artefak di-freeze & didokumentasikan (handoff ke Track B & C)

---

## Quick Checklist Track B

- [ ] Env siap (`timm`/`torch`, GPU terdeteksi)
- [ ] Training loop jalan di data dummy (stub)
- [ ] Sanity overfit 1 batch lolos (loss ≈ 0)
- [ ] `config.py` + `scheduler.py` (warmup + cosine) dibuat
- [ ] Stub diganti `dataset.py` asli + `class_weights` asli (setelah GATE 2)
- [ ] 1 batch data asli terverifikasi (shape, label, visual)
- [ ] Baseline fold 0 selesai + Macro-F1 tercatat
- [ ] Estimasi waktu/epoch → biaya 5-fold dicek vs budget GPU
- [ ] 5 fold dilatih, checkpoint terbaik tersimpan
- [ ] OOF terkumpul + CV Macro-F1 (mean ± std) dihitung
- [ ] Checkpoint + OOF + config diserahkan ke Track C

---

## Quick Checklist Track C

> Status per **9 Juli 2026** — dikerjakan bersama Antigravity AI

### Fase 0 — Setup & Format Guard
- [x] Template submission dipahami (`id`, `predicted`, 1458 baris, nilai {0,1,2})
- [x] Validator format dibuat — `validator.py` (cek baris, kolom, NaN, urutan ID, nilai label)
- [x] Fungsi inference checkpoint → prob test — `inference.py` (`load_model`, `predict_test`)
- [x] Assert mapping label 0/1/2 = Track A — `assert_label_mapping()` di `inference.py`
- [x] Package `__init__.py` dibuat di semua track (track_a, track_a/src, track_b, track_b/src, track_c, track_c/src)

### Fase 1 — Threshold Tuning
- [x] Baseline Macro-F1 OOF (argmax) dihitung otomatis di `threshold_tuning.py`
- [x] Threshold per kelas di-tuning di OOF — grid search W1, W2 ∈ [0.5, 2.0]
- [x] Fungsi `apply_thresholds()` final dikunci

### Fase 2 — Ensemble & TTA
- [x] Ensemble 5 model (rata-rata prob) — `ensemble_tta.py`
- [x] TTA horizontal flip — toggle via `CFG_C.use_tta`
- [x] Hardcoded `1458` diperbaiki → dinamis `len(test_loader.dataset)`
- [x] Pipeline end-to-end — `generate_submission.py`
- [x] Nama file submission diperbaiki → `submission_apace.csv` (via `CFG_C.team_name`)

### Fase 3 — Submission & Report
- [ ] Validator format dijalankan pada file submission nyata *(butuh checkpoint Track B)*
- [ ] Submission 1 (safety net) terunggah *(butuh checkpoint Track B)*
- [ ] Kode eval didokumentasikan untuk verifikasi panitia
- [ ] Bagian Metrik + Metodologi + poin diskusi report tertulis