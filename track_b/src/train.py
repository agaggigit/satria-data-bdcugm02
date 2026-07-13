import json
import shutil
import time
import torch
from torch.amp import GradScaler
from model import build_model
from loaders import get_loaders_b
from losses_metrics import build_loss, macro_f1, print_report
from seed_utils import set_seed
from scheduler import build_scheduler
from optim_utils import build_optimizer
from ckpt import checkpoint_path, history_path, save_checkpoint


def setup_run(cfg, fold: int) -> dict:
    """Bangun model + loader + optimizer + scheduler. Bisa di-test di CPU tanpa training.

    cfg.backbone dikirim ke build_model() secara eksplisit -- ini yang dulu
    (Fase 1) tidak pernah tersambung, jadi ganti backbone tidak pernah berefek.
    """
    model, data_config = build_model(
        cfg.backbone,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
        drop_path_rate=cfg.drop_path_rate,
        grad_checkpointing=getattr(cfg, "grad_checkpointing", False),
    )

    train_loader, val_loader, val_row_idx = get_loaders_b(fold, cfg, data_config)

    optimizer = build_optimizer(
        model, lr=cfg.lr, weight_decay=cfg.weight_decay,
        layer_decay=getattr(cfg, "layer_decay", None),
    )
    opt_steps_per_epoch = max(1, len(train_loader) // cfg.accum_steps)
    scheduler = build_scheduler(optimizer, steps_per_epoch=opt_steps_per_epoch, cfg=cfg)

    return {
        "model": model,
        "data_config": data_config,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_row_idx": val_row_idx,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


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
    """Signature TIDAK BERUBAH -- Track C & skrip lama bergantung padanya."""
    set_seed(cfg.seed)
    device = "cuda"
    epochs = max_epochs or cfg.epochs

    # Guard sekali di awal: run_name yang sama dipakai ulang tanpa allow_overwrite
    # eksplisit = kemungkinan besar menimpa checkpoint run lain diam-diam.
    ckpt_path = checkpoint_path(cfg, fold)
    hist_path = history_path(cfg, fold)
    if ckpt_path.exists() and not cfg.allow_overwrite:
        raise FileExistsError(
            f"{ckpt_path} sudah ada — run_name '{cfg.run_name}' kemungkinan dipakai ulang. "
            f"Ganti cfg.run_name, atau set cfg.allow_overwrite=True kalau memang sengaja."
        )

    ctx = setup_run(cfg, fold)
    model = ctx["model"].to(device)
    data_config = ctx["data_config"]
    train_loader, val_loader = ctx["train_loader"], ctx["val_loader"]
    optimizer, scheduler = ctx["optimizer"], ctx["scheduler"]

    criterion = build_loss(class_weights.to(device), label_smoothing=cfg.label_smoothing)
    scaler = GradScaler("cuda")

    if cfg.accum_steps > 1:
        print(f"  grad accumulation: {cfg.accum_steps} steps → batch efektif {cfg.batch * cfg.accum_steps}")

    best_f1 = 0.0
    epoch_times = []
    history = []
    lr_per_step = []
    patience = getattr(cfg, "patience", None)  # None = tidak ada early stopping (perilaku lama)
    epochs_no_improve = 0

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
            epochs_no_improve = 0
            # Workaround for Google Drive FUSE permission error: simpan lokal dulu, baru copy.
            local_tmp = f"/tmp/{cfg.run_name}_fold{fold}_best.pt"
            payload = {
                "model_state_dict": model.state_dict(),
                "model_name": cfg.backbone,
                "data_config": data_config,
                "run_name": cfg.run_name,
                "fold": fold,
                "img_size": cfg.img_size,
                "seed": cfg.seed,
                "best_f1": best_f1,
            }
            save_checkpoint(payload, local_tmp, allow_overwrite=True)
            shutil.copy(local_tmp, ckpt_path)
            print(f"    checkpoint saved: {ckpt_path} (f1 {best_f1:.4f})")
        else:
            epochs_no_improve += 1
            if patience is not None and epochs_no_improve >= patience:
                print(f"  early stopping: {epochs_no_improve} epoch tanpa perbaikan val_f1 (patience={patience})")
                break

    mins_per_epoch = (sum(epoch_times) / len(epoch_times)) / 60
    print(f"\n  BEST fold {fold} macro-f1: {best_f1:.4f}")
    print_report(preds, labels)

    try:
        with open(hist_path, "w") as f:
            json.dump({"fold": fold, "run_name": cfg.run_name, "history": history,
                       "lr_per_step": lr_per_step}, f, indent=2)
        print(f"    history saved: {hist_path}")
    except OSError as e:
        print(f"    WARNING: gagal simpan history ({e}); lanjut tanpa history")

    return best_f1, mins_per_epoch


if __name__ == "__main__":
    from config import CFG
    import numpy as np
    cw = np.load(CFG.class_weights_path)
    class_weights = torch.tensor(cw, dtype=torch.float32)
    run_training(fold=0, cfg=CFG, class_weights=class_weights, max_epochs=2)
