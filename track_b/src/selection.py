"""selection.py — Aturan seleksi B9 (TRACK_B_ARAHAN_V3.md), generik untuk
membandingkan 2 varian (mis. v1 vs v2) ATAU memilih 1 pemenang dari tabel
banyak kandidat (head_grid_v3.csv). Sengaja terpisah dari experiments/
tta_compare.py (yang menerapkan aturan identik tapi spesifik untuk TTA) supaya
dipakai ulang tanpa impor lintas skrip experiments/.

Aturan (sama di tiap tempat, jangan dilonggarkan mepet deadline):
- mean naik tapi min antar-fold turun -> TOLAK (mengejar satu fold beruntung)
- Selisih mean < 0.002 -> menang yang min tertinggi & std terkecil
- Skor test HANYA konteks, bukan kriteria -- fungsi di sini TIDAK PERNAH
  menerima skor test sebagai argumen.
"""
import pandas as pd

MEAN_TIE_THRESHOLD = 0.002


def decide(mean_a: float, min_a: float, mean_b: float, min_b: float) -> str:
    """B (kandidat baru) vs A (baseline/rujukan)."""
    delta = mean_b - mean_a
    if delta <= 0:
        return "TOLAK (mean tidak naik)"
    if min_b >= min_a:
        return "TERIMA"
    return "TOLAK (overfit ke OOF -- mean naik tapi min turun)"


def select_grid_winner(df: pd.DataFrame, mean_tie_threshold: float = MEAN_TIE_THRESHOLD) -> pd.Series:
    """Dari tabel kandidat (kolom mean/min/std wajib ada): kandidat dengan
    mean tertinggi jadi acuan tier; di antara semua kandidat yang mean-nya
    beda < threshold dari acuan, menang yang min tertinggi, tie-break std
    terkecil (probe_grid.py: 'itu yang paling mungkin bertahan di test')."""
    assert len(df) > 0, "tabel kandidat kosong -- tidak ada yang bisa dipilih"
    assert {"mean", "min", "std"} <= set(df.columns), \
        f"tabel kandidat kehilangan kolom wajib: {{'mean','min','std'}} - {set(df.columns)}"

    top_mean = df["mean"].max()
    tier = df[df["mean"] >= top_mean - mean_tie_threshold]
    tier = tier.sort_values(["min", "std"], ascending=[False, True])
    return tier.iloc[0]
