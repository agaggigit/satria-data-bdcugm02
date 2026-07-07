import torch
from torch.amp import GradScaler
from model import build_model
from losses_metrics import build_loss
from seed_utils import set_seed

def sanity_overfit(n_steps=200, lr=3e-4, target_loss=0.1):
    """Bukti training loop benar: model HARUS bisa menghafal 1 batch kecil (loss ≈ 0).

    Perbaikan dari versi awal (loss stuck di ~1.06 ≈ ln(3) = tebakan acak):
    - lr 3e-3 → 3e-4  (3e-3 terlalu tinggi untuk fine-tune ConvNeXt, sesuai plan 1e-4–3e-4)
    - clip max_norm 0.5 → 1.0  (0.5 terlalu agresif, gradient tercekik)
    - drop_path_rate=0.0  (stochastic depth = noise antar step, ganggu overfit deterministik)
    - assert final_loss < target_loss  (kriteria milestone B1: loss ≈ 0, bukan cuma "bergerak")
    """
    set_seed(42)
    device = "cuda"

    images = torch.randn(8, 3, 224, 224).to(device)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]).to(device)

    model = build_model(drop_path_rate=0.0).to(device)
    model.set_grad_checkpointing(False)

    criterion = build_loss(
        torch.tensor([1.0, 1.0, 1.0]).to(device),
        label_smoothing=0.0
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    scaler = GradScaler("cuda")

    losses = []
    for step in range(n_steps):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

        if step % 20 == 0:
            print(f"  step {step:3d}: loss {loss.item():.4f}")

    first_loss = losses[0]
    final_loss = losses[-1]
    print(f"\n  first loss: {first_loss:.4f}")
    print(f"  final loss: {final_loss:.4f}")

    # Cek 1: tidak ada NaN
    assert not any(l != l for l in losses), "❌ Ada NaN dalam training"
    # Cek 2: loss turun signifikan (bukan sekadar bergerak)
    assert final_loss < first_loss * 0.5, \
        f"❌ Loss tidak turun berarti: {first_loss:.4f} → {final_loss:.4f}"
    # Cek 3: KRITERIA UTAMA milestone B1 — model menghafal batch (loss ≈ 0)
    assert final_loss < target_loss, \
        f"❌ Gagal overfit 1 batch: final loss {final_loss:.4f} ≥ {target_loss}. " \
        f"Loop belum terbukti benar — cek LR / grad clipping / AMP scale."

    print("✅ Loop terbukti benar:")
    print(f"   - Tidak ada NaN")
    print(f"   - Loss {first_loss:.4f} → {final_loss:.4f} (< {target_loss})")
    print(f"   - Model menghafal 1 batch → gradient, AMP, dan optimizer bekerja")
    return final_loss
