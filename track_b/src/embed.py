"""embed.py — Kontrak embedding cache: nama file dari `name`/`split`, guard
anti-overwrite, manifest wajib, dan assert alignment terhadap folds.csv.

Baris ke-i emb_*.npy HARUS = baris ke-i folds.csv. Tidak ada pengecualian.
"""
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

try:
    import psutil
except ImportError:                      # psutil ada default di Colab; jangan hard-fail lokal
    psutil = None

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
    """Load model + image processor SEKALI, pakai ulang untuk beberapa panggilan
    extract_embeddings(). Tanpa ini, extract_all.py (5 backbone x 2 flip x 2
    split) akan me-load model 20 kali padahal cukup 5 -- mahal, apalagi untuk
    so400m (~3,5 GB).

    AutoImageProcessor, BUKAN AutoProcessor. AutoProcessor menarik image
    processor DAN tokenizer; tokenizer.json (34 MB, lewat Xet CDN) sempat gagal
    403 "SignatureError: invalid key pair id" -- masalah sisi HF, bukan token
    kita. Kita memang tidak pernah memakai text tower (dilarang: image encoder
    saja), jadi tokenizer itu murni beban: satu file besar yang bisa
    menggagalkan seluruh run tanpa memberi manfaat apa pun. Terverifikasi
    pixel_values-nya identik dengan AutoProcessor."""
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(ckpt, torch_dtype=torch.float16).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(ckpt)   # preprocessing milik checkpoint ini
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


def _open_rgb(path):
    """Buka + decode jadi RGB, lalu TUTUP file handle-nya segera.

    `Image.open(p).convert("RGB")` versi lama tidak pernah menutup `fp`. PIL
    ImageFile menyimpan referensi siklik (tile/fp), jadi refcount CPython tidak
    langsung membebaskannya -- ia menunggu GC siklik yang jarang jalan di dalam
    loop C-berat + inference_mode ini. Akibatnya satu gambar terdekode (~1 MB)
    menumpuk per gambar yang diproses -> RAM naik linear sampai OOM di ~45%
    dataset. Context manager memutus siklus itu di titik decode."""
    with Image.open(path) as im:
        return im.convert("RGB")                         # gambar baru, lepas dari fp


@torch.inference_mode()
def extract_embeddings(ckpt: str, filepaths: list, device="cuda",
                       batch: int = 64, flips: tuple = (), encoder=None,
                       log_ram: bool = True, gc_every: int = 20) -> np.ndarray:
    """flips: subset dari ('h', 'v'). Embedding asli + tiap flip dirata-ratakan (TTA).

    encoder: hasil load_encoder() yang mau dipakai ulang. Kalau None, model
    di-load di sini (perilaku lama, tetap jalan untuk pemanggil yang sudah ada).

    RAM-safe: hasil ditulis ke SATU array pre-alokasi `[N, D]` (bukan list yang
    tumbuh lalu di-concat), gambar tiap batch ditutup eksplisit, dan `gc.collect()`
    dipanggil tiap `gc_every` batch. `log_ram=True` mencetak RSS proses tiap
    `gc_every` batch lewat psutil supaya lonjakan memori kelihatan di batch berapa.
    """
    own_encoder = encoder is None
    if own_encoder:
        encoder = load_encoder(ckpt, device)
    model, processor = encoder

    ram = psutil.Process() if (log_ram and psutil is not None) else None

    def _pool(inputs):
        if hasattr(model, "get_image_features"):        # SigLIP / SigLIP2
            raw = model.get_image_features(**inputs)
        else:
            raw = model(**inputs)                        # DINOv3
        return _tensor_from(raw)

    n = len(filepaths)
    result = None            # dialokasi setelah batch pertama mengungkap D
    try:
        batch_starts = list(range(0, n, batch))
        if ram is not None:
            tqdm.write(f"  [ram] mulai extract [{ckpt}] n={n} "
                       f"rss={ram.memory_info().rss / 1e9:.2f} GB")

        for bi, i in enumerate(tqdm(batch_starts, desc=f"extract [{ckpt}]", unit="batch")):
            chunk = filepaths[i:i + batch]
            imgs = [_open_rgb(p) for p in chunk]

            views = [imgs]
            if "h" in flips:
                views.append([im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs])
            if "v" in flips:
                views.append([im.transpose(Image.FLIP_TOP_BOTTOM) for im in imgs])

            # TTA dirata-rata secara berjalan (running mean) -- tidak menyimpan
            # list feats agar tak ada array transien yang menganggur di RAM.
            acc = None
            for view in views:
                inputs = processor(images=view, return_tensors="pt").to(device, torch.float16)
                feat = _pool(inputs).float().cpu().numpy()
                acc = feat if acc is None else acc + feat
                del inputs, feat
            acc /= len(views)

            if result is None:
                result = np.empty((n, acc.shape[1]), dtype=np.float32)
            result[i:i + len(chunk)] = acc               # tulis di posisi -> alignment terjaga

            # Lepaskan gambar batch ini SEKARANG; jangan tunggu GC siklik.
            for view in views:
                for im in view:
                    im.close()
            del imgs, views, acc

            if (bi + 1) % gc_every == 0:
                gc.collect()
                if ram is not None:
                    tqdm.write(f"  [ram] batch {bi + 1}/{len(batch_starts)} "
                               f"rss={ram.memory_info().rss / 1e9:.2f} GB")

        if result is None:                               # filepaths kosong
            return np.empty((0, 0), dtype=np.float32)
        return result
    finally:
        if own_encoder:
            free_encoder(encoder)
