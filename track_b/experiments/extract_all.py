"""extract_all.py — Ekstrak semua backbone kandidat (GPU).

Jalankan dengan `python experiments/extract_all.py` dari mana saja -- sys.path
di-set eksplisit di bawah (jangan andalkan cwd: `python path/to/script.py`
menaruh folder SCRIPT itu sendiri di sys.path[0], bukan cwd tempat kamu `cd`).

DINOv3 gated -- terima lisensinya di HF dan siapkan token SEBELUM menjalankan
ini, bukan setelah 40 menit ekstraksi gagal di tengah.

Gambar di-mirror ke disk lokal DULU (lihat local_cache.py). Ini bukan optimasi
kosmetik: membaca 26.527 gambar train langsung dari Drive ~1,3 gambar/detik =
5,7 JAM per ekstraksi, dan skrip ini melakukan 10 ekstraksi train -> 57 jam.
Dari disk lokal: 74 gambar/detik. Copy-nya sekarang bagian dari skrip, bukan
langkah manual di notebook yang bisa terlupa -- lupa = skrip tetap "jalan
normal", cuma 50x lebih lambat tanpa peringatan apa pun.
"""
import os
import sys

import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from config import CFG
from embed import (assert_aligned, extract_embeddings, free_encoder, is_cached,
                   load_encoder, save_embeddings)
from local_cache import localize_paths

BACKBONES = {
    "siglip2b256":  "google/siglip2-base-patch16-256",
    "siglip2so400m": "google/siglip2-so400m-patch14-384",
    "siglip1b256":  "google/siglip-base-patch16-256",
    "dinov3vitl":   "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "dinov3cnxb":   "facebook/dinov3-convnext-base-pretrain-lvd1689m",
}

LOCAL_ROOT = "/content/img_cache"


def preflight(backbones: dict) -> None:
    """Buktikan SEMUA backbone bisa dibentuk & mengeluarkan embedding, SEBELUM
    ekstraksi apa pun dimulai.

    Tanpa ini, DINOv3 (gated di HF, dan ada di urutan TERAKHIR) baru ketahuan
    gagal setelah dua backbone SigLIP selesai -- berjam-jam GPU terbakar untuk
    kemudian mati di 401. Preflight memindahkan kegagalan itu ke menit pertama.

    Sekalian membuktikan jalur pooling-nya benar: SigLIP lewat
    get_image_features(), DINOv3 lewat pooler_output. Kalau salah satu tidak
    cocok, kita tahu sekarang, bukan setelah 26.527 gambar diproses.
    """
    import tempfile

    from PIL import Image

    tmp = tempfile.mkdtemp()
    dummy_path = os.path.join(tmp, "preflight.jpg")
    Image.new("RGB", (384, 384), color=(120, 90, 60)).save(dummy_path)

    print("=== PREFLIGHT: cek semua backbone bisa dipakai ===")
    for name, ckpt in backbones.items():
        encoder = load_encoder(ckpt)
        try:
            emb = extract_embeddings(ckpt, [dummy_path], batch=1, encoder=encoder)
            assert emb.ndim == 2 and emb.shape[0] == 1, f"{name}: shape aneh {emb.shape}"
            print(f"  OK {name:14s} dim={emb.shape[1]:5d}  ({ckpt})")
        finally:
            free_encoder(encoder)
    print("=== PREFLIGHT HIJAU — ekstraksi boleh mulai ===\n")


folds = pd.read_csv(CFG.folds_csv)
template = pd.read_csv(CFG.sample_sub_path)

preflight(BACKBONES)

# --- Mirror gambar ke disk lokal (sekali; resume-safe kalau runtime putus) ---
train_paths = localize_paths(folds["filepath"].tolist(),
                             os.path.join(LOCAL_ROOT, "train"), desc="copy train")
test_paths = localize_paths([f"{CFG.test_dir}/{i}.jpg" for i in template["id"]],
                            os.path.join(LOCAL_ROOT, "test"), desc="copy test")

SPLITS = [("train", train_paths), ("test", test_paths)]

for name, ckpt in BACKBONES.items():
    # Lewati backbone yang SEMUA variannya sudah ter-cache -- jangan buang waktu
    # load model 3 GB cuma untuk langsung skip semua splitnya.
    wanted = [(name + suffix, split, flips)
              for flips, suffix in [((), ""), (("h", "v"), "_tta")]
              for split, _ in SPLITS]
    if all(is_cached(full_name, split) for full_name, split, _ in wanted):
        print(f"skip {name}: semua varian sudah ada di {CFG.embeddings_dir}")
        continue

    encoder = load_encoder(ckpt)   # SEKALI per backbone, bukan per (flip, split)
    try:
        for flips, suffix in [((), ""), (("h", "v"), "_tta")]:
            for split, paths in SPLITS:
                full_name = name + suffix
                if is_cached(full_name, split):
                    print(f"skip {full_name} {split}: sudah ada")
                    continue

                emb = extract_embeddings(ckpt, paths, batch=32, flips=flips,
                                         encoder=encoder)
                if split == "train":
                    assert_aligned(emb, folds)
                save_embeddings(emb, full_name, split,
                                {"checkpoint": ckpt, "flips": list(flips),
                                 "seed": CFG.seed})
                print(f"{full_name} {split}: {emb.shape}")
    finally:
        free_encoder(encoder)   # bebaskan VRAM sebelum backbone berikutnya masuk

print("\nSemua embedding siap di", CFG.embeddings_dir)
