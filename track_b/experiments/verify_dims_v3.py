"""verify_dims_v3.py — Task 0 (TRACK_B_ARAHAN_V3.md): cetak shape SEMUA embedding
train yang sudah ada, dan konfirmasi dim concat siglip2b256+siglip1b256 == 1536.

GATE untuk A3 (concat SigLIP2@256 + SigLIP1): jangan ekstrak apa pun sebelum
angka ini jelas. Logika (available_embeddings/verify_concat_dim) sudah diuji
CPU-only dengan data sintetis di tests/test_embed.py -- skrip ini cuma
menjalankannya terhadap cache Drive yang sebenarnya.

Jalankan dari track_b/src/ (sama seperti audit_wiring.py):
    python ../experiments/verify_dims_v3.py
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from config import CFG                                            # noqa: E402
from embed import EXPECTED_BACKBONES, available_embeddings, verify_concat_dim  # noqa: E402

CONCAT_A, CONCAT_B, CONCAT_EXPECTED = "siglip2b256", "siglip1b256", 1536


def main() -> None:
    print(f"embeddings_dir: {CFG.embeddings_dir}\n")

    for split in ("train", "test"):
        found = available_embeddings(split, names=EXPECTED_BACKBONES)
        print(f"=== {split} ===")
        for name, info in found.items():
            print(f"  {name:16s} shape={info['shape']!s:16s} dim={info['dim']:5d} "
                  f"checkpoint={info['checkpoint']}")
        missing = [n for n in EXPECTED_BACKBONES if n not in found]
        if missing:
            print(f"  belum ada: {missing}")
        print()

    found_train = available_embeddings("train", names=EXPECTED_BACKBONES)
    verify_concat_dim(found_train, CONCAT_A, CONCAT_B, expected=CONCAT_EXPECTED)
    print(f"GATE A3 HIJAU — dim({CONCAT_A}) + dim({CONCAT_B}) = {CONCAT_EXPECTED}, "
          f"terkonfirmasi lewat kode. Concat siap dikerjakan tanpa ekstraksi ulang.")


if __name__ == "__main__":
    main()
