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

## Environment

- Python 3.10+
- PyTorch ≥ 2.0
- torchvision, PIL/Pillow
- pandas, numpy, matplotlib, seaborn
- imagehash (untuk perceptual hash / cek duplikat)
- scikit-learn (untuk StratifiedKFold)

```bash
pip install torch torchvision pandas numpy matplotlib seaborn imagehash scikit-learn
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