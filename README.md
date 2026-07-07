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