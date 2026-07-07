"""
oof_utils.py — Kumpulkan prediksi Out-of-Fold (OOF) selama training 5-fold.

Kontrak (Workflow_Koordinasi_ABC.md): oof.npy = probabilitas [N, 3], index
cocok folds.csv. Dipakai Track C untuk threshold tuning per kelas.

Scaffold ini tidak butuh data asli — plumbing-nya sama persis saat dipakai
nanti di Fase 2 (training 5-fold data asli Track A), tinggal isi val_indices
dari folds.csv per fold.
"""
import numpy as np
import torch


def init_oof(n_samples, num_classes=3):
    """Array OOF kosong (NaN) — diisi baris per baris selama 5-fold training."""
    return np.full((n_samples, num_classes), np.nan, dtype=np.float32)


@torch.no_grad()
def predict_probs(model, loader, device):
    """Softmax probabilitas untuk seluruh loader (urutan dipertahankan, no shuffle).
    Dipakai untuk OOF (val loader per fold) maupun ensemble test (Track C)."""
    model.eval()
    use_amp = device == "cuda"
    all_probs = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)
        probs = torch.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def fill_fold_oof(oof, val_indices, probs):
    """Isi baris OOF untuk index validasi fold ini (val_indices: index baris di folds.csv)."""
    assert len(val_indices) == len(probs), \
        f"Jumlah index ({len(val_indices)}) != jumlah prob ({len(probs)})"
    oof[val_indices] = probs
    return oof


def validate_oof_complete(oof):
    """Pastikan seluruh data latih tertutup — tak ada baris NaN tersisa (semua 5 fold sudah isi)."""
    missing = np.isnan(oof).any(axis=1)
    n_missing = int(missing.sum())
    assert n_missing == 0, f"OOF belum lengkap: {n_missing} baris masih NaN"
    print(f"OOF lengkap: {oof.shape[0]} sampel x {oof.shape[1]} kelas")


def save_oof(oof, path="oof.npy"):
    validate_oof_complete(oof)
    np.save(path, oof)
    print(f"OOF disimpan: {path}")
