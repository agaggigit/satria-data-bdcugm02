"""embed.py — Kontrak embedding cache: nama file dari `name`/`split`, guard
anti-overwrite, manifest wajib, dan assert alignment terhadap folds.csv.

Baris ke-i emb_*.npy HARUS = baris ke-i folds.csv. Tidak ada pengecualian.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

# HF Hub's "Xet" download backend (CAS) 401s on unauthenticated requests for
# some files. Env var ini MEMBANTU tapi TIDAK CUKUP -- terbukti 14 Juli:
# model.safetensors lolos, tapi tokenizer.json tetap 401 lewat cas-bridge, dan
# HF_TOKEN pun tidak menolong. Yang benar-benar menyelesaikan: UNINSTALL package
# hf_xet (`pip uninstall -y hf_xet hf-xet`, sudah ada di notebook cell setup).
# Baris ini dipertahankan sebagai lapis kedua, bukan sebagai solusi utama.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

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


def is_cached(name: str, split: str) -> bool:
    """Cache dianggap sah HANYA kalau .npy DAN manifest .json dua-duanya ada.
    Cek .npy saja tidak cukup: load_embeddings() butuh manifest, jadi cache
    tanpa manifest akan lolos skip lalu meledak jauh kemudian di probe_grid."""
    return emb_path(name, split).exists() and manifest_path(name, split).exists()


def assert_aligned(emb: np.ndarray, folds_df) -> None:
    assert emb.shape[0] == len(folds_df), \
        f"jumlah baris embedding {emb.shape[0]} != folds.csv {len(folds_df)}"
    assert not np.isnan(emb).any(), "embedding mengandung NaN"
    assert np.isfinite(emb).all(), "embedding mengandung inf"


def load_encoder(ckpt: str, device="cuda") -> tuple:
    """Load model + processor SEKALI, pakai ulang untuk beberapa panggilan
    extract_embeddings(). Tanpa ini, extract_all.py (5 backbone x 2 flip x 2
    split) akan me-load model 20 kali padahal cukup 5 -- mahal, apalagi untuk
    so400m (~3,5 GB)."""
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(ckpt, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt)   # preprocessing milik checkpoint ini
    return model, processor


def free_encoder(encoder) -> None:
    """Lepas model dari VRAM. WAJIB dipanggil antar backbone di extract_all.py:
    T4 cuma 15 GB, dan menumpuk base -> so400m-384 -> dinov3-vitl berurutan tanpa
    membersihkan cache allocator adalah cara paling mudah kena OOM di tengah jalan."""
    model, _ = encoder
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tensor_from(output):
    """Unwrap ModelOutput -> tensor polos. Versi transformers berbeda
    mengembalikan get_image_features()/pooler_output kadang sebagai tensor
    langsung, kadang dibungkus BaseModelOutputWithPooling -- tangani dua-duanya
    alih-alih asumsi salah satu (root cause AttributeError 'no attribute float')."""
    if torch.is_tensor(output):
        return output
    if getattr(output, "pooler_output", None) is not None:
        return output.pooler_output
    if getattr(output, "image_embeds", None) is not None:
        return output.image_embeds
    raise TypeError(f"Tidak tahu cara ekstrak tensor dari {type(output)}")


@torch.inference_mode()
def extract_embeddings(ckpt: str, filepaths: list, device="cuda",
                       batch: int = 64, flips: tuple = (), encoder=None) -> np.ndarray:
    """flips: subset dari ('h', 'v'). Embedding asli + tiap flip dirata-ratakan (TTA).

    encoder: hasil load_encoder() yang mau dipakai ulang. Kalau None, model
    di-load di sini (perilaku lama, tetap jalan untuk pemanggil yang sudah ada).
    """
    own_encoder = encoder is None
    if own_encoder:
        encoder = load_encoder(ckpt, device)
    model, processor = encoder

    def _pool(inputs):
        if hasattr(model, "get_image_features"):        # SigLIP / SigLIP2
            raw = model.get_image_features(**inputs)
        else:
            raw = model(**inputs)                        # DINOv3
        return _tensor_from(raw)

    try:
        out = []
        batches = range(0, len(filepaths), batch)
        for i in tqdm(batches, desc=f"extract [{ckpt}]", unit="batch"):
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
    finally:
        if own_encoder:
            free_encoder(encoder)
