"""Test CPU-only untuk logika perbandingan TTA. Data mock -- tidak memuat
embedding asli, tidak menyentuh Drive. Menguji aturan keputusan (identik
probe_grid.py) dan perakitan baris hasil."""
import os
import sys

# experiments/ bukan package & tidak di sys.path lewat conftest -> tambah manual.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from tta_compare import build_row, decide_tta


# --- decide_tta: tiga cabang aturan + batas ---

def test_tta_dipakai_saat_mean_naik_dan_min_tidak_turun():
    # mean naik, min naik -> jelas dipakai
    assert decide_tta(0.9901, 0.9896, 0.9925, 0.9910) == "TTA dipakai"


def test_tta_dipakai_saat_min_sama_persis():
    # batas: min TTA == min non-TTA memenuhi ">=" -> tetap dipakai
    assert decide_tta(0.9901, 0.9896, 0.9910, 0.9896) == "TTA dipakai"


def test_tolak_saat_mean_naik_tapi_min_turun():
    # gain mean tapi fold terburuk memburuk -> overfit ke OOF
    assert decide_tta(0.9901, 0.9896, 0.9925, 0.9880) == "TOLAK (overfit ke OOF)"


def test_tidak_membantu_saat_mean_turun():
    assert decide_tta(0.9901, 0.9896, 0.9890, 0.9895) == "TIDAK MEMBANTU, buang TTA"


def test_tidak_membantu_saat_mean_sama_persis():
    # batas: delta == 0 bukan "> 0" -> TTA tidak menang, apa pun min-nya
    assert decide_tta(0.9901, 0.9896, 0.9901, 0.9999) == "TIDAK MEMBANTU, buang TTA"


# --- build_row: delta & kolom benar dari dua hasil fold_consistency mock ---

def _cons(mean, mn, std=0.0):
    return {"mean": mean, "min": mn, "std": std, "per_fold": []}


def test_build_row_menghitung_delta_dan_meneruskan_keputusan():
    c_notta = _cons(0.9901, 0.9896)
    c_tta = _cons(0.9925, 0.9910)
    row = build_row("siglip2so400m", "knn", c_notta, c_tta)

    assert row["backbone"] == "siglip2so400m"
    assert row["head"] == "knn"
    assert row["mean_notta"] == 0.9901
    assert row["mean_tta"] == 0.9925
    assert row["min_notta"] == 0.9896
    assert row["min_tta"] == 0.9910
    assert abs(row["delta"] - (0.9925 - 0.9901)) < 1e-12
    assert row["keputusan"] == "TTA dipakai"


def test_build_row_punya_semua_kolom_csv():
    row = build_row("siglip2so400m", "mlp", _cons(0.98, 0.97), _cons(0.97, 0.96))
    assert set(row) == {"backbone", "head", "mean_notta", "mean_tta",
                        "min_notta", "min_tta", "delta", "keputusan"}
    assert row["keputusan"] == "TIDAK MEMBANTU, buang TTA"
