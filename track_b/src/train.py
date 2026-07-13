import json
import time
import torch
from torch.amp import GradScaler
from dataset import get_loaders              # ASLI Track A (was: dataset_stub)
from model import build_model
from losses_metrics import build_loss, macro_f1, print_report
from seed_utils import set_seed
from scheduler import build_scheduler


def train_one_epoch(model, loader, optimizer, scheduler, criterion, scaler, device, cfg):
    """Gradient accumulation dipertahankan dari Fase 0.
    cfg.accum_steps > 1: akumulasi N batch sebelum optimizer.step().
    scheduler.step() dipanggil per optimizer step, bukan per batch.
    Mengembalikan juga train_f1 (dari prediksi selama training, tanpa forward
    pass ekstra) dan lr_history per optimizer-step (bukti visual warmup+cosine)."""
    model.train()
    total_loss = 0.0
    accum = cfg.accum_steps
    n_batches = len(loader)
    optimizer.zero_grad()
    all_preds, all_labels, lr_history = [], [], []

    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels) / accum
        scaler.scale(loss).backward()
        all_preds.append(outputs.detach().argmax(dim=1))
        all_labels.append(labels)

        is_last = (i + 1) == n_batches
        if (i + 1) % accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # Hanya step scheduler kalau optimizer benar-benar jalan
            # (scaler skip optimizer kalau ada inf/nan gradient)
            if scaler.get_scale() == scale_before:
                scheduler.step()
            lr_history.append(optimizer.param_groups[0]["lr"])
            optimizer.zero_grad()

        total_loss += loss.item() * accum * images.size(0)

    train_f1 = macro_f1(torch.cat(all_preds), torch.cat(all_labels))
    return total_loss / len(loader.dataset), train_f1, lr_history


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            all_preds.append(outputs.argmax(dim=1))
            all_labels.append(labels)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    return total_loss / len(loader.dataset), macro_f1(preds, labels), preds, labels


def run_training(fold, cfg, class_weights, max_epochs=None):
    """Signature baru Fase 1 — semua config dibaca dari cfg.
    accum_steps dari cfg.accum_steps (default 1, naikkan kalau OOM di 256)."""
    set_seed(cfg.seed)
    device = "cuda"
    epochs = max_epochs or cfg.epochs

    train_loader, val_loader = get_loaders(fold=fold, img_size=cfg.img_size, batch=cfg.batch)
    model = build_model(
        num_classes=cfg.num_classes,
        drop_path_rate=cfg.drop_path_rate,
        grad_checkpointing=getattr(cfg, "grad_checkpointing", False),
    ).to(device)
    criterion = build_loss(class_weights.to(device), label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # steps_per_epoch = optimizer steps, bukan raw batches
    opt_steps_per_epoch = len(train_loader) // cfg.accum_steps
    scheduler = build_scheduler(optimizer, steps_per_epoch=opt_steps_per_epoch, cfg=cfg)
    scaler = GradScaler("cuda")

    if cfg.accum_steps > 1:
        print(f"  grad accumulation: {cfg.accum_steps} steps → batch efektif {cfg.batch * cfg.accum_steps}")

    best_f1 = 0.0
    save_path = f"{cfg.save_dir}/fold{fold}.pt"
    epoch_times = []
    history = []
    lr_per_step = []

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_f1, lr_steps = train_one_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, device, cfg)
        val_loss, val_f1, preds, labels = validate(model, val_loader, criterion, device)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        lr_per_step.extend(lr_steps)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  epoch {epoch+1}/{epochs} | lr {lr_now:.2e} | train {tr_loss:.4f} (f1 {tr_f1:.4f}) | val {val_loss:.4f} | val_f1 {val_f1:.4f} | {elapsed/60:.1f} mnt")
        history.append({
            "epoch": epoch + 1,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "train_f1": tr_f1,
            "val_f1": val_f1,
            "lr": lr_now,
        })
        if val_f1 > best_f1:
            best_f1 = val_f1
            # Workaround for Google Drive FUSE permission error
            local_save_path = f"/tmp/fold{fold}_best.pt"
            torch.save(model.state_dict(), local_save_path)
            import shutil
            shutil.copy(local_save_path, save_path)
            print(f"    checkpoint saved: {save_path} (f1 {best_f1:.4f})")

    mins_per_epoch = (sum(epoch_times) / len(epoch_times)) / 60
    print(f"\n  BEST fold {fold} macro-f1: {best_f1:.4f}")
    print_report(preds, labels)

    history_path = f"{cfg.save_dir}/fold{fold}_history.json"
    try:
        with open(history_path, "w") as f:
            json.dump({"fold": fold, "history": history, "lr_per_step": lr_per_step}, f, indent=2)
        print(f"    history saved: {history_path}")
    except OSError as e:
        print(f"    WARNING: gagal simpan history ({e}); lanjut tanpa history")

    return best_f1, mins_per_epoch


if __name__ == "__main__":
    from config import CFG
    import numpy as np
    cw = np.load(CFG.class_weights_path)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    run_training(fold=0, cfg=CFG, class_weights=class_weights, max_epochs=2)
