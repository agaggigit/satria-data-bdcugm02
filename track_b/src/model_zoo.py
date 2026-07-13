"""
model_zoo.py — Backbone alternatif untuk ensemble (submission 2-3, Track C).

Bukan dipakai training utama Fase 1 — baseline tetap
convnext_tiny.in12k_ft_in1k (lihat model.py, wajib didokumentasikan di report).
Kandidat di sini di-scaffold sekarang (tidak butuh data asli Track A) sekadar
untuk memastikan arsitekturnya bisa dibuat + forward pass benar, supaya siap
dipakai kalau butuh diversity ensemble di minggu akhir.
"""
import timm
import torch

# Kandidat eksplorasi arsitektur di luar FAMILY_REGISTRY (lihat model.py) --
# smoke test murni pakai timm langsung, TIDAK lewat build_model(), karena
# kandidat di sini belum didaftarkan (drop_path/native_size/short belum
# didefinisikan). Daftarkan ke FAMILY_REGISTRY dulu kalau mau dipakai training.
CANDIDATE_BACKBONES = [
    "convnext_tiny.in12k_ft_in1k",       # baseline (sudah ada di FAMILY_REGISTRY)
    "convnextv2_tiny.fcmae_ft_in22k_in1k",
    "efficientnet_b3",
]


def test_backbone_forward(backbone, num_classes=3, img_size=224, device="cpu"):
    """Smoke test: backbone bisa dibuat + forward pass shape benar.
    Cukup di CPU — ini cek arsitektur, bukan benchmark kecepatan training."""
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes).to(device)
    model.eval()
    dummy = torch.randn(2, 3, img_size, img_size).to(device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, num_classes), f"Shape salah untuk {backbone}: {out.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{backbone}] OK — output {tuple(out.shape)}, {params:.1f}M params")
    return out.shape


if __name__ == "__main__":
    for bb in CANDIDATE_BACKBONES:
        test_backbone_forward(bb)
