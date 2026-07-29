"""lora_ft.py — Task 5 (kalau ada waktu GPU, TRACK_B_ARAHAN_V3.md B8):
LoRA & last-layer fine-tuning di vision tower SigLIP.

Paradigma BEDA dari Task 0-4 (head di atas embedding beku): di sini sebagian
backbone ikut dilatih. Full fine-tuning sudah terbukti gagal (representasi
pretrained rusak) -- dua pendekatan anti-overfit di sini:

  lora       : adapter kecil (rank 8-16) di q_proj/v_proj VISION TOWER SAJA,
               backbone asli 100% beku. LR 1e-4, epoch 3-5.
  last_layer : buka 1-2 blok terakhir vision_model.encoder.layers, LLRD --
               LR backbone 10-100x lebih kecil dari LR head (prinsip yang
               dulu terbukti satu-satunya yang membantu, +0.0158).

WAJIB smoke test fold 0 (2 epoch) dulu, catat minutes_per_epoch, baru
putuskan lanjut 5-fold penuh atau tidak (B8) -- jangan langsung full run.

Constraint jujur (dicatat, bukan disembunyikan): smoke_test_loop() di bawah
CPU-only pakai data dummy (pola sama sanity_overfit.py Fase 0) -- ITU yang
sudah diuji & terbukti benar di repo ini. run_smoke_test_fold0() butuh GPU +
gambar train asli (Drive) -- kodenya reuse loaders.get_loaders_b apa adanya
(sama seperti train.py), tapi BELUM ADA bukti eksekusi lokal. Jalankan dari
Colab, bukan mesin ini.
"""
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from vision_classifier import (VisionClassifier, apply_lora, build_llrd_optimizer,
                               configure_last_layer_finetuning,
                               hf_processor_to_data_config)  # noqa: E402

# --- Parameter hasil riset paper SigLIP2-SO400M ref (B8) ---
LORA_RANK = 4                        # Paper ref: rank 4
LORA_ALPHA = 8                       # Paper ref: alpha 8
LORA_DROPOUT = 0.05                  # Paper ref: dropout 0.05
LORA_LR = 1e-4                       # Paper ref: LR 1e-4
LORA_EPOCHS = 30                     # Paper ref: 30 epoch
LAST_LAYER_N_BLOCKS = 1              # B8: 1-2 blok terakhir
LAST_LAYER_HEAD_LR = 3e-4
LAST_LAYER_BACKBONE_LR_RATIO = 1 / 50  # B8: 10-100x lebih kecil dari LR head
MAX_GRAD_NORM = 1.0

# LR khusus SMOKE TEST -- SENGAJA jauh lebih besar dari LR produksi di atas.
# Dua hal berbeda yang tidak boleh dicampur:
#   - LR produksi (LORA_LR=1e-4 dst, B8) kecil ON PURPOSE supaya representasi
#     pretrained tidak rusak. Itu tujuan generalisasi.
#   - smoke test menguji MEKANIKA loop (gradient sampai ke parameter yang benar,
#     optimizer benar-benar meng-update, tak ada NaN) -- untuk itu model harus
#     bisa dipaksa menghafal 1 batch, dan dgn LR produksi itu butuh ribuan step.
# Diukur empiris di fake encoder (200 step): lora konvergen mulai ~1e-3,
# last_layer butuh ~3e-2 (LR efektif backbone-nya head_lr/50). Preseden yang
# sama sudah dipakai sanity_overfit.py (Fase 0): LR uji != LR produksi.
SMOKE_LR = {"lora": 5e-3, "last_layer": 3e-2}


LORA_N_LAST_BLOCKS = None   # None = semua 27 layer, atau int (misal 4) untuk N transformer block terakhir


def build_variant(variant: str, encoder: nn.Module, hidden_size: int,
                  num_classes: int = 3, lr: float = None,
                  n_last_blocks: int | None = LORA_N_LAST_BLOCKS) -> tuple:
    """variant: 'lora' atau 'last_layer'. Bangun VisionClassifier + optimizer
    sesuai parameter hasil riset B8, siap dipakai training loop.

    lr: override LR (dipakai smoke_test_loop dgn SMOKE_LR). None = LR produksi B8.
    Untuk 'last_layer', lr ini adalah LR HEAD; LR backbone tetap
    lr * LAST_LAYER_BACKBONE_LR_RATIO (rasio LLRD tidak ikut berubah)."""
    model = VisionClassifier(encoder, hidden_size=hidden_size, num_classes=num_classes)
    if variant == "lora":
        model = apply_lora(model, r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                           n_last_blocks=n_last_blocks)
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=LORA_LR if lr is None else lr)
    elif variant == "last_layer":
        configure_last_layer_finetuning(model, n_blocks=LAST_LAYER_N_BLOCKS)
        optimizer = build_llrd_optimizer(
            model, head_lr=LAST_LAYER_HEAD_LR if lr is None else lr,
            backbone_lr_ratio=LAST_LAYER_BACKBONE_LR_RATIO)
    else:
        raise KeyError(f"variant '{variant}' tidak dikenal. Pilihan: 'lora', 'last_layer'")
    return model, optimizer


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, accum_steps: int = 1) -> float:
    """AMP + grad clip + gradient accumulation -- disesuaikan untuk model HF (SigLIP2-SO400M).
    Mencegah OutOfMemoryError di T4 GPU untuk model 1B+ @ 384x384."""
    model.train()
    total_loss = 0.0
    use_amp = device == "cuda"
    trainable = [p for p in model.parameters() if p.requires_grad]
    accum_steps = max(1, accum_steps)

    optimizer.zero_grad()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels) / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item() * accum_steps * images.size(0)
    return total_loss / len(loader.dataset)


def smoke_test_loop(variant: str, encoder_factory, hidden_size: int, n_steps: int = 200,
                    batch: int = 4, device: str = "cpu", target_loss: float = 0.2,
                    lr: float = None) -> float:
    """Bukti mekanika loop benar SEBELUM GPU/data asli disentuh -- pola sama
    sanity_overfit.py (Fase 0): paksa model menghafal 1 batch kecil sampai
    loss rendah. `encoder_factory`: callable() -> nn.Module baru (fresh tiap
    panggilan). `images` di sini VEKTOR dummy [batch, hidden_size] (struktur
    forward saja yang diuji, bukan gambar) -- lihat tests/test_lora_ft.py.

    lr: None = pakai SMOKE_LR[variant] (BUKAN LR produksi B8 -- lihat catatan
    di SMOKE_LR). Yang diuji di sini mekanika, bukan hyperparameter produksi."""
    torch.manual_seed(42)
    lr = SMOKE_LR[variant] if lr is None else lr
    model, optimizer = build_variant(variant, encoder_factory(), hidden_size, lr=lr)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]

    images = torch.randn(batch, hidden_size).to(device)
    labels = (torch.arange(batch) % 3).to(device)

    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=MAX_GRAD_NORM)
        optimizer.step()
        losses.append(loss.item())

    final_loss = losses[-1]
    assert not any(l != l for l in losses), f"NaN dalam smoke test '{variant}'"
    assert final_loss < losses[0], \
        f"smoke test '{variant}': loss tidak turun ({losses[0]:.4f} -> {final_loss:.4f})"
    assert final_loss < target_loss, (
        f"smoke test '{variant}' gagal overfit 1 batch: final loss {final_loss:.4f} >= "
        f"{target_loss}. Loop belum terbukti benar -- cek LR/wiring trainable params "
        f"sebelum lanjut ke GPU/data asli."
    )
    return final_loss


def run_smoke_test_fold0(variant: str, cfg, checkpoint: str, max_epochs: int = 2,
                         n_last_blocks: int | None = None) -> dict:
    """Entry point Colab SUNGGUHAN (B8): fold 0, gambar train asli, GPU.
    Reuse loaders.get_loaders_b apa adanya (sama seperti train.py) --
    TIDAK ditulis ulang. BELUM ADA bukti eksekusi lokal (butuh GPU + Drive) --
    jalankan dari notebook Colab. Panggil smoke_test_loop() dulu di CPU utk
    verifikasi mekanika loop sebelum ini."""
    from embed import load_encoder
    from loaders import get_loaders_b
    from losses_metrics import macro_f1
    from seed_utils import set_seed

    set_seed(cfg.seed)
    device = "cuda"

    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    revision = getattr(cfg, "backbone_revision", None)
    encoder, processor = load_encoder(checkpoint, device=device, revision=revision)
    hidden_size = getattr(getattr(encoder.config, "vision_config", encoder.config), "hidden_size", 1152)
    data_config = hf_processor_to_data_config(processor)

    # Otomatis samakan cfg.img_size dengan native size checkpoint (misal 384 atau 256)
    # untuk mencegah mismatch position embedding di Vision Transformer
    model_img_size = data_config["input_size"][1]
    if cfg.img_size != model_img_size:
        from dataclasses import replace
        cfg = replace(cfg, img_size=model_img_size)
        print(f"Aligning cfg.img_size to checkpoint native size: {model_img_size}")

    val_batch_size = getattr(cfg, "val_batch", 32)
    train_loader, val_loader, _ = get_loaders_b(fold=0, cfg=cfg, data_config=data_config, val_batch_size=val_batch_size)

    model, optimizer = build_variant(variant, encoder, hidden_size, num_classes=cfg.num_classes, n_last_blocks=n_last_blocks)
    
    # Verifikasi trainable parameters (Guard #1)
    print(f"\n--- Verifikasi trainable parameters [{variant}] ---")
    if hasattr(model.encoder, "print_trainable_parameters"):
        model.encoder.print_trainable_parameters()
    else:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    model = model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")
    accum_steps = getattr(cfg, "accum_steps", 1)

    epoch_times = []
    best_val_f1 = -1.0
    best_epoch = -1

    for epoch in range(max_epochs):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, accum_steps=accum_steps)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels)
        val_f1 = macro_f1(torch.cat(all_preds), torch.cat(all_labels))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1

        print(f"  [{variant}] epoch {epoch+1}/{max_epochs} | train_loss {tr_loss:.4f} | "
              f"val_f1 {val_f1:.4f} | {elapsed/60:.1f} mnt")

    mins_per_epoch = sum(epoch_times) / len(epoch_times) / 60
    est_5fold_hours = mins_per_epoch * 5 * max_epochs / 60
    print(f"\n[{variant}] minutes_per_epoch = {mins_per_epoch:.2f}")
    print(f"[{variant}] BEST Epoch: {best_epoch} | Best val_f1: {best_val_f1:.4f}")
    print(f"[{variant}] Estimasi 5-fold x {max_epochs} epoch = {est_5fold_hours:.2f} jam GPU")

    return {
        "variant": variant,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "last_val_f1": val_f1,
        "minutes_per_epoch": mins_per_epoch,
        "est_5fold_hours": est_5fold_hours,
    }


if __name__ == "__main__":
    print("Jalankan smoke_test_loop() dulu (CPU, cepat) sebelum run_smoke_test_fold0 (GPU).")
    print("Skrip ini tidak auto-run apa pun -- panggil fungsinya eksplisit dari notebook.")
