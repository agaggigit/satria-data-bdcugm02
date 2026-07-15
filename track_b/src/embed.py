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

# CATATAN xet (dikoreksi 15 Juli): huggingface_hub 1.x memakai backend "xet"
# sebagai DEFAULT, dan HF_HUB_DISABLE_XET SUDAH TIDAK EFEKTIF di 1.x (env var
# ter-set tapi request tetap lewat xet-bridge). Solusi yang BENAR bukan mematikan
# xet dan bukan meng-uninstall hf_xet -- justru sebaliknya: paket `hf_xet` WAJIB
# TERPASANG. Tanpa itu, hub 1.x jatuh ke fallback rusak -> "SignatureError:
# invalid key pair id". Pasang di notebook: `pip install -q hf_xet`.
# (Baris os.environ lama yang mematikan xet DIHAPUS -- tidak berguna & menyesatkan.)

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


def resolve_dtype(ckpt: str) -> torch.dtype:
    """Pilih dtype forward pass per checkpoint.

    fp16 MERUSAK sebagian backbone: aktivasi DINOv3 melampaui rentang fp16
    (maks ~65504) -> inf -> NaN, dan `.float()` setelahnya TIDAK menyelamatkan
    karena NaN sudah lahir di dalam forward pass. Gejalanya: assert_aligned
    menolak "embedding mengandung NaN", file tidak tersimpan, DINOv3 hilang
    diam-diam dari cache (terjadi 15 Juli).

    Default AMAN = fp32. SigLIP terbukti stabil di fp16 dan boleh tetap fp16
    untuk hemat VRAM, tapi kebenaran > kecepatan: kalau ragu, fp32.
    Kalau GPU mendukung bf16 (rentang eksponen selebar fp32, jadi tak overflow),
    itu kompromi terbaik untuk model besar -- tapi T4 tidak punya bf16 native,
    jadi di Colab T4 ini efektif fp32 untuk DINOv3."""
    lc = ckpt.lower()
    if "dinov3" in lc:
        return torch.float32          # WAJIB fp32: fp16 -> NaN
    return torch.float16              # SigLIP: fp16 aman & hemat


def load_encoder(ckpt: str, device="cuda", dtype: torch.dtype = None) -> tuple:
    """Load model + image processor SEKALI, pakai ulang untuk beberapa panggilan
    extract_embeddings(). Tanpa ini, extract_all.py (5 backbone x 2 flip x 2
    split) akan me-load model 20 kali padahal cukup 5 -- mahal, apalagi untuk
    so400m (~3,5 GB).

    dtype: kalau None, dipilih otomatis per checkpoint via resolve_dtype()
    (fp32 untuk DINOv3 yang overflow di fp16, fp16 untuk SigLIP). Model DAN input
    harus dtype yang sama -- extract_embeddings membaca model.dtype untuk itu.

    AutoImageProcessor, BUKAN AutoProcessor. AutoProcessor menarik image
    processor DAN tokenizer; tokenizer.json (34 MB, lewat Xet CDN) sempat gagal
    403 "SignatureError: invalid key pair id" -- masalah sisi HF, bukan token
    kita. Kita memang tidak pernah memakai text tower (dilarang: image encoder
    saja), jadi tokenizer itu murni beban: satu file besar yang bisa
    menggagalkan seluruh run tanpa memberi manfaat apa pun. Terverifikasi
    pixel_values-nya identik dengan AutoProcessor."""
    from transformers import AutoImageProcessor, AutoModel

    if dtype is None:
        dtype = resolve_dtype(ckpt)
    model = AutoModel.from_pretrained(ckpt, dtype=dtype).to(device).eval()
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
            # Input HARUS ikut dtype MODEL (model.dtype), bukan hardcode fp16:
            # DINOv3 di-load fp32 (fp16 -> NaN), jadi input fp16 ke model fp32 =
            # error/mismatch. Baca dtype aktual model, jangan asumsi.
            model_dtype = next(model.parameters()).dtype
            acc = None
            for view in views:
                inputs = processor(images=view, return_tensors="pt").to(device, model_dtype)
                feat = _pool(inputs).float().cpu().numpy()
                acc = feat if acc is None else acc + feat
                del inputs, feat
            acc /= len(views)

            # Tangkap NaN/inf SEGERA -- di batch mana, bukan setelah 26.527 gambar
            # lalu gagal misterius di assert_aligned akhir. Kalau lolos ke sini,
            # dtype/overflow belum beres.
            if not np.isfinite(acc).all():
                raise ValueError(
                    f"NaN/inf di embedding [{ckpt}] batch {bi} (baris {i}:{i+len(chunk)}). "
                    f"model dtype={model_dtype}. Kemungkinan fp16 overflow -- "
                    f"backbone ini butuh fp32 (lihat resolve_dtype)."
                )

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


# =====================================================================
# Resume berbasis shard
# ---------------------------------------------------------------------
# Backbone besar (DINOv3) sering putus di tengah sesi Colab -> progres berjam-jam
# hilang. Solusinya: tulis potongan {name}_{split}.part{start:06d}.npy tiap
# SHARD_ROWS gambar. Run ulang melewati shard yang sudah lengkap dan lanjut. Saat
# semua shard lengkap, gabung BERURUTAN indeks -> file final -> hapus shard.
#
# Alignment dijaga oleh index range yang DISIMPAN di manifest tiap shard: merge
# mengurutkan berdasarkan `start`, bukan berdasarkan urutan file di disk.
# =====================================================================

SHARD_ROWS = 4000


def shard_path(name: str, split: str, start: int) -> Path:
    return EMB_DIR / f"{name}_{split}.part{start:06d}.npy"


def shard_manifest_path(name: str, split: str, start: int) -> Path:
    return EMB_DIR / f"{name}_{split}.part{start:06d}.json"


def save_shard(emb: np.ndarray, name: str, split: str, start: int,
               meta: dict) -> None:
    """Tulis satu shard secara atomik: .npy.tmp -> replace -> manifest TERAKHIR.

    Manifest ditulis paling akhir supaya kalau proses mati di tengah np.save,
    yang tertinggal cuma .npy(.tmp) tanpa manifest -> _shard_is_complete() bilang
    "belum lengkap" dan shard itu dihitung ulang, bukan dikira valid.
    """
    p = shard_path(name, split, start)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")
    with open(tmp, "wb") as f:                 # file-object -> np.save tak utak-atik ekstensi
        np.save(f, emb.astype(np.float32))
    os.replace(tmp, p)

    end = int(start) + int(emb.shape[0])
    meta = {**meta, "start": int(start), "end": end,
            "n_rows": int(emb.shape[0]), "dim": int(emb.shape[1])}
    shard_manifest_path(name, split, start).write_text(json.dumps(meta, indent=2))


def _shard_is_complete(name: str, split: str, start: int) -> bool:
    """Lengkap HANYA kalau .npy + manifest ada DAN jumlah baris .npy cocok dengan
    manifest (deteksi .npy parsial/korup akibat proses mati saat menulis)."""
    p, mp = shard_path(name, split, start), shard_manifest_path(name, split, start)
    if not (p.exists() and mp.exists()):
        return False
    try:
        meta = json.loads(mp.read_text())
        arr = np.load(p, mmap_mode="r")        # baca header saja, murah
        rows = arr.shape[0]
        del arr
    except (ValueError, OSError, json.JSONDecodeError):
        return False
    return rows == meta.get("n_rows") and int(start) + rows == meta.get("end")


def merge_shards(name: str, split: str, n_expected: int, final_meta: dict,
                 allow_overwrite: bool = False, cleanup: bool = True) -> np.ndarray:
    """Gabung semua shard {name}_{split}.part*.npy jadi file final.

    Kontrak keras:
    - Merge HANYA kalau total baris semua shard == n_expected. Kalau kurang,
      assert gagal SEBELUM save_embeddings -> file final tidak ditulis (biar
      is_cached tidak salah bilang "selesai" padahal bolong).
    - Urutan mengikuti index range `start` yang tersimpan di manifest, bukan
      urutan glob di disk -> baris ke-i final = baris ke-i folds.csv.
    - Guard anti-overwrite file final tetap lewat save_embeddings.
    """
    manifests = list(EMB_DIR.glob(f"{name}_{split}.part*.json"))
    metas = sorted((json.loads(m.read_text()) for m in manifests),
                   key=lambda d: d["start"])

    total = sum(int(d["n_rows"]) for d in metas)
    assert total == n_expected, (
        f"total baris shard {total} != {n_expected}; shard belum lengkap, "
        f"file final TIDAK ditulis"
    )

    # Kontiguitas: shard harus menutup [0, n_expected) tanpa lubang / tumpang tindih.
    cursor = 0
    for d in metas:
        assert d["start"] == cursor, \
            f"shard tidak kontigu di start {d['start']} (harusnya {cursor}) -- ada lubang/overlap"
        cursor = int(d["end"])
    assert cursor == n_expected, f"shard berhenti di {cursor}, bukan {n_expected}"

    parts = [np.load(shard_path(name, split, int(d["start"]))) for d in metas]
    emb = np.concatenate(parts, axis=0)
    assert emb.shape[0] == n_expected, "jumlah baris hasil concat tidak sesuai -- batal simpan"
    assert np.isfinite(emb).all(), "hasil merge mengandung NaN/inf -- batal simpan"

    save_embeddings(emb, name, split, final_meta, allow_overwrite=allow_overwrite)

    if cleanup:                                # hapus shard hanya setelah final tersimpan
        for d in metas:
            shard_path(name, split, int(d["start"])).unlink(missing_ok=True)
            shard_manifest_path(name, split, int(d["start"])).unlink(missing_ok=True)
    return emb


def extract_embeddings_resumable(ckpt: str, filepaths: list, name: str, split: str,
                                 device="cuda", batch: int = 64, flips: tuple = (),
                                 encoder=None, shard_rows: int = SHARD_ROWS,
                                 final_meta: dict = None, allow_overwrite: bool = False,
                                 log_ram: bool = True, gc_every: int = 20) -> np.ndarray:
    """extract_embeddings dengan resume: hitung per shard `shard_rows` gambar, tulis
    tiap shard ke disk, lewati shard yang sudah lengkap saat run ulang, lalu merge.

    encoder di-load SEKALI (kalau None) dan dipakai ulang lintas shard -- jangan
    load model 3 GB per shard.
    """
    if final_meta is None:
        final_meta = {"checkpoint": ckpt, "flips": list(flips)}

    if is_cached(name, split):                 # sudah selesai di run sebelumnya
        tqdm.write(f"  skip {name}_{split}: file final sudah ada")
        return load_embeddings(name, split)[0]

    n = len(filepaths)
    if n == 0:
        emb = np.empty((0, 0), dtype=np.float32)
        save_embeddings(emb, name, split, final_meta, allow_overwrite=allow_overwrite)
        return emb

    own_encoder = encoder is None
    if own_encoder:
        encoder = load_encoder(ckpt, device)
    try:
        for start in range(0, n, shard_rows):
            end = min(start + shard_rows, n)
            if _shard_is_complete(name, split, start):
                tqdm.write(f"  skip shard {name}_{split} [{start}:{end}] sudah lengkap")
                continue

            emb = extract_embeddings(ckpt, filepaths[start:end], device=device,
                                     batch=batch, flips=flips, encoder=encoder,
                                     log_ram=log_ram, gc_every=gc_every)
            save_shard(emb, name, split, start, dict(final_meta))
            tqdm.write(f"  shard {name}_{split} [{start}:{end}] tersimpan {emb.shape}")
            del emb
            gc.collect()

        return merge_shards(name, split, n, final_meta,
                            allow_overwrite=allow_overwrite)
    finally:
        if own_encoder:
            free_encoder(encoder)
