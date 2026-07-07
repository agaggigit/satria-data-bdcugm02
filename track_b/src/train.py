import torch
from torch.amp import GradScaler
from dataset_stub import get_loaders
from model import build_model
from losses_metrics import build_loss, macro_f1, print_report
from seed_utils import set_seed

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            all_preds.append(outputs.argmax(dim=1))
            all_labels.append(labels)
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    val_loss = total_loss / len(loader.dataset)
    val_f1 = macro_f1(preds, labels)
    return val_loss, val_f1, preds, labels

def run_training(fold=0, epochs=5, img_size=224, batch=32, lr=3e-4, save_dir="."):
    set_seed(42)
    device = "cuda"

    train_loader, val_loader = get_loaders(fold=fold, img_size=img_size, batch=batch)
    model = build_model().to(device)
    criterion = build_loss(torch.tensor([1.0, 1.0, 1.0]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scaler = GradScaler("cuda")  # satu instance baru per fold

    best_f1 = 0.0
    save_path = f"{save_dir}/fold{fold}.pt"

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_f1, preds, labels = validate(model, val_loader, criterion, device)
        print(f"  epoch {epoch+1}/{epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_f1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)

    # Report per-kelas di akhir fold
    print(f"\n  best val macro-f1: {best_f1:.4f}")
    print_report(preds, labels)
    return best_f1

if __name__ == "__main__":
    run_training(epochs=2)
