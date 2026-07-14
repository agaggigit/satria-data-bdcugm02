"""embed.py — Kontrak embedding cache: nama file dari `name`/`split`, guard
anti-overwrite, manifest wajib, dan assert alignment terhadap folds.csv.

Baris ke-i emb_*.npy HARUS = baris ke-i folds.csv. Tidak ada pengecualian.
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import CFG

# Drive (CFG.embeddings_dir), bukan storage lokal runtime -- kalau sesi Colab
# putus, cache embedding (berjam-jam kerja GPU) tidak boleh ikut hilang.
EMB_DIR = Path(CFG.embeddings_dir)


def emb_path(name: str, split: str) -> Path:
    return EMB_DIR / f"{name}_{split}.npy"


def manifest_path(name: str, split: str) -> Path:
    return EMB_DIR / f"{name}_{split}.json"


def save_embeddings(emb: np.ndarray, name: str, split: str,
                    meta: dict, allow_overwrite: bool = False) -> None:
    p = emb_path(name, split)
    if p.exists() and not allow_overwrite:
        raise FileExistsError(
            f"{p} sudah ada. Ganti `name`, atau set allow_overwrite=True kalau sengaja."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, emb.astype(np.float32))

    meta = {**meta, "n_rows": int(emb.shape[0]), "dim": int(emb.shape[1])}
    manifest_path(name, split).write_text(json.dumps(meta, indent=2))


def load_embeddings(name: str, split: str) -> tuple:
    emb = np.load(emb_path(name, split))
    meta = json.loads(manifest_path(name, split).read_text())
    return emb, meta


def assert_aligned(emb: np.ndarray, folds_df) -> None:
    assert emb.shape[0] == len(folds_df), \
        f"jumlah baris embedding {emb.shape[0]} != folds.csv {len(folds_df)}"
    assert not np.isnan(emb).any(), "embedding mengandung NaN"
    assert np.isfinite(emb).all(), "embedding mengandung inf"


@torch.inference_mode()
def extract_embeddings(ckpt: str, filepaths: list, device="cuda",
                       batch: int = 64, flips: tuple = ()) -> np.ndarray:
    """flips: subset dari ('h', 'v'). Embedding asli + tiap flip dirata-ratakan (TTA)."""
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(ckpt, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt)   # preprocessing milik checkpoint ini

    def _pool(inputs):
        if hasattr(model, "get_image_features"):        # SigLIP / SigLIP2
            return model.get_image_features(**inputs)
        return model(**inputs).pooler_output            # DINOv3

    out = []
    for i in range(0, len(filepaths), batch):
        imgs = [Image.open(p).convert("RGB") for p in filepaths[i:i + batch]]

        views = [imgs]
        if "h" in flips:
            views.append([im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs])
        if "v" in flips:
            views.append([im.transpose(Image.FLIP_TOP_BOTTOM) for im in imgs])

        feats = []
        for view in views:
            inputs = processor(images=view, return_tensors="pt").to(device, torch.float16)
            feats.append(_pool(inputs).float().cpu().numpy())

        out.append(np.mean(feats, axis=0))   # TTA: rata-rata di level EMBEDDING

    return np.concatenate(out)
