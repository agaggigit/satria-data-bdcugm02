"""local_cache.py — Mirror gambar dari Drive ke disk lokal Colab sebelum dibaca.

KENAPA INI ADA (angka nyata dari run safety_net.py, 14 Juli):

    train (26.527 gambar, dari disk LOKAL) : 6 menit    -> 74 gambar/detik
    test  ( 1.458 gambar, dari DRIVE)      : 18m39s     ->  1,3 gambar/detik

Test 18x lebih sedikit gambarnya tapi 3x lebih LAMA. Dari 48,7 detik per batch,
~47,8 detik murni menunggu Drive; komputasi GPU-nya cuma ~0,9 detik. Drive FUSE
itu latency-bound per file, bukan bandwidth-bound -- makanya copy PARALEL
(ThreadPoolExecutor) jauh mengalahkan `cp -r` sekuensial: satu thread cuma
menunggu round-trip jaringan, puluhan thread menunggu bersamaan.

Urutan path yang dikembalikan WAJIB sama persis dengan input -- baris ke-i
embedding harus tetap = baris ke-i folds.csv.
"""
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from tqdm.auto import tqdm


def _copy_atomic(job) -> None:
    """Copy lewat file .tmp lalu rename. Kalau proses mati di tengah copy, yang
    tertinggal cuma .tmp -- BUKAN file tujuan yang setengah jadi dan nanti
    dikira valid oleh pengecekan `exists()`."""
    src, dst = job
    tmp = dst + ".tmp"
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def localize_paths(paths, local_root: str, workers: int = 32,
                   desc: str = "copy ke lokal") -> list:
    """Mirror `paths` ke `local_root`, kembalikan path lokal DENGAN URUTAN SAMA.

    File yang sudah ada di lokal tidak di-copy ulang (aman untuk resume setelah
    runtime putus). Struktur subfolder relatif terhadap common root ikut dijaga,
    jadi dua gambar bernama sama di kelas berbeda tidak saling menimpa.
    """
    paths = [str(p) for p in paths]
    if not paths:
        return []

    common = os.path.commonpath([os.path.dirname(p) for p in paths])

    local_paths = [os.path.join(local_root, os.path.relpath(p, common)) for p in paths]

    todo = [(s, d) for s, d in zip(paths, local_paths) if not os.path.exists(d)]
    if todo:
        for d in {os.path.dirname(d) for _, d in todo}:
            os.makedirs(d, exist_ok=True)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(_copy_atomic, todo), total=len(todo),
                      desc=desc, unit="file"))

    return local_paths
