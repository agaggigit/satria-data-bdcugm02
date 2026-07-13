# Track B — Panduan Modul `src/`

**BDC Satria Data 2026 — Klasifikasi Citra Sampah**

Folder ini berisi modul Python inti Track B — training loop, model, loss/metric.
Notebook di `notebooks/` hanya memanggil fungsi dari sini.

> ⚠️ **Owner-only write:** file di folder ini milik Track B.
> `dataset.py` dan `folds.csv` milik Track A — **read-only**, jangan buat versi sendiri.

---

## Daftar File (rencana — hasil Fase 0)

| File | Isi | Status |
|------|-----|--------|
| `config.py` | `CFG` — sumber kebenaran semua konstanta (seed, img_size, LR, epochs, backbone) | Fase 1 Task 1 |
| `seed_utils.py` | `set_seed(42)` — seed semua sumber random | Fase 0 |
| `dataset_stub.py` | Stub `get_loaders(fold, img_size, batch)` — dummy sampai artefak Track A masuk | Fase 0 |
| `model.py` | `build_model()` — ConvNeXt-Tiny `convnext_tiny.in12k_ft_in1k` via timm, grad checkpointing ON | Fase 0 |
| `losses_metrics.py` | `build_loss()` (weighted CE + label smoothing 0.1), `macro_f1()`, `print_report()` | Fase 0 |
| `scheduler.py` | `build_scheduler()` — warmup linear + cosine decay, `.step()` per batch | Fase 1 Task 2 |
| `train.py` | `train_one_epoch()`, `validate()`, `run_training()` — AMP + grad clipping + checkpoint | Fase 0 |
| `sanity_overfit.py` | `sanity_overfit()` — bukti loop benar (loss ≈ 0 di 1 batch) | Fase 0 |

---

## Konvensi Terkunci (dari kontrak workflow — JANGAN ubah tanpa umumkan)

```python
# Mapping label — aturan keras, assert di semua track
LABEL_MAP   = {0: "Recyclable", 1: "Electronic", 2: "Organic"}
SEED        = 42
IMG_SIZE    = 224
# Normalisasi: ImageNet mean/std
# Signature loader: get_loaders(fold, img_size, batch) -> (train_loader, val_loader)
```

### Detail teknis yang sudah diputuskan (Fase 0)

- **AMP API baru:** `from torch.amp import GradScaler`; `GradScaler("cuda")`;
  `torch.autocast(device_type="cuda", dtype=torch.float16)`.
  JANGAN `torch.cuda.amp.autocast` (deprecated).
- **Grad clipping:** `scaler.unscale_(optimizer)` → `clip_grad_norm_(max_norm=1.0)` → `scaler.step()`.
- **GradScaler baru per fold** — jangan reuse antar fold.
- **Grad checkpointing OFF secara default** (13 Juli, revisi) — GPU T4 15GB cuma kepakai
  ~2GB saat training ConvNeXt-Tiny @224 dengan checkpointing ON; itu murni membuang
  kecepatan (checkpointing = recompute activation di backward untuk hemat VRAM yang
  sebenarnya tidak dibutuhkan). Dikontrol via `cfg.grad_checkpointing` (default `False`).
  Nyalakan lagi kalau naik ke resolusi/backbone yang mepet OOM (mis. img_size 288+).
- **Backbone:** `convnext_tiny.in12k_ft_in1k` — wajib didokumentasikan di report (aturan panitia).

---

## Titik Integrasi dengan Track A

Saat GATE 2 hijau, hanya 2 baris berubah di `train.py`:

```python
# SEBELUM (Fase 0)
from dataset_stub import get_loaders
criterion = build_loss(torch.tensor([1.0, 1.0, 1.0]).to(device))

# SESUDAH (Fase 1)
from dataset import get_loaders                    # asli dari Track A
weights = torch.tensor(np.load("class_weights.npy"), dtype=torch.float)
criterion = build_loss(weights.to(device))
```

> ⚠️ **Catatan integrasi:** `dataset.py` Track A saat ini mengekspos
> `make_fold_loaders(folds_csv, val_fold, ...)`, bukan `get_loaders(fold, img_size, batch)`
> sesuai kontrak. Perlu dibicarakan dengan Track A sebelum swap (protokol breaking change).

---

*Dibuat untuk BDC Satria Data 2026 — Track B*
