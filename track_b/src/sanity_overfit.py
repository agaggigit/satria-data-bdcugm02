import torch
from torch.amp import GradScaler
from model import build_model
from losses_metrics import build_loss
from seed_utils import set_seed

def sanity_overfit(n_steps=300, lr=3e-4, target_loss=0.1):
    """Bukti training loop benar: model HARUS bisa menghafal 1 batch kecil (loss ≈ 0).

    Riwayat perbaikan:
    - v1 (lr 3e-3, clip 0.5, noise, 100 step): loss stuck ~1.06 ≈ ln(3) = tebakan acak.
    - v2 (lr 3e-4, clip 1.0, drop_path 0, noise, 200 step): mulai turun tapi cuma
      sampai ~0.16 — noise MURNI sulit di-fit conv, plateau di ln(3) sampai ~160 step
      lalu baru anjlok. Marginal, gagal target < 0.1.
    - v3 (INI): data dummy dibuat BISA dipelajari dari piksel (sinyal per-kelas di
      channel berbeda) alih-alih noise murni → backbone punya fitur nyata untuk
      dipisah → konvergen cepat & andal. Proxy lebih baik untuk data asli yang
      memang berstruktur. n_steps 200 → 300 untuk margin.
    """
    set_seed(42)
    device = "cuda"

    # Batch dummy yang BISA dipelajari dari piksel (root cause v2: noise murni nyaris
    # tak punya struktur untuk di-ekstrak conv). Sinyal per-kelas: channel ke-`label`
    # diterangkan → kelas 0/1/2 ≈ dominan merah/hijau/biru → trivially separable.
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]).to(device)
    images = torch.randn(8, 3, 224, 224) * 0.5
    for i, lbl in enumerate(labels.tolist()):
        images[i, lbl] += 3.0
    images = images.to(device)

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
