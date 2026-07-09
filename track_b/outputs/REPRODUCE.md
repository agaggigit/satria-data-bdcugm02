# Reproduksi Hasil Track B — BDC Satria Data 2026

## Ringkas
Fine-tune `convnext_tiny.in12k_ft_in1k` (pre-trained, via timm) untuk klasifikasi 3 kelas sampah.
Stratified-Group 5-Fold CV (folds.csv dari Track A, seed 42, group-aware anti-leakage).

## Urutan reproduksi
1. Clone repo: `git clone https://github.com/agaggigit/satria-data-bdcugm02.git`
2. Mount Drive, set `FOLDS_CSV` env var ke path folds.csv di Drive
3. `pip install timm scikit-learn`
4. `cp track_a/src/dataset.py track_b/src/dataset.py`
5. Jalankan notebook `track_b/notebooks/02_fase1_3_training.ipynb`
   - Atau: `run_all_folds.run_folds([0,1,2,3,4])` → 5 checkpoint
   - Lalu: `collect_oof.collect_oof()` → oof.npy + metrik

## Konfigurasi kunci
- Seed 42 (torch/numpy/random + cudnn deterministic) — di-set ulang tiap fold
- Image 224, batch 32, epochs 8
- AdamW lr 0.0003 wd 0.05, warmup 1 epoch + cosine → 1e-06
- Weighted CrossEntropy (class_weights.npy Track A) + label smoothing 0.1
- AMP fp16 (torch.amp) + grad clipping max_norm 1.0
- Grad checkpointing ON (VRAM)

## Bukti proses (bukan manual)
- Log per fold: fold{N}_log.json (metrik per fold)
- Log environment: environment.json
- Kurva loss/F1 per epoch tercetak di notebook (tersimpan di history Colab)
- Kode lengkap di Git repo tim
