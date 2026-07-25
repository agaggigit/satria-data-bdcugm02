"""Test CPU-only untuk gerbang validasi submission safety_net_final.

Mock data kecil -- tidak memuat embedding asli (26.527 baris) atau menyentuh
Drive. validate_submission mengunci 1458 baris, jadi template mock pun 1458 baris
(tetap ringan: cuma integer, bukan embedding). Menguji jalur yang dipakai script:
make_submission -> validate_submission lewat build_validated_submission."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from safety_net_final import build_validated_submission

rng = np.random.default_rng(42)


def _template():
    return pd.DataFrame({"id": list(range(1, 1459)), "predicted": [0] * 1458})


def test_submission_valid_lolos_dan_urut_id_sama_template():
    template = _template()
    pred = rng.integers(0, 3, size=1458)
    sub = build_validated_submission(pred, template)

    assert list(sub.columns) == ["id", "predicted"]
    assert len(sub) == 1458
    assert (sub["id"].to_numpy() == template["id"].to_numpy()).all()
    assert set(np.unique(sub["predicted"])) <= {0, 1, 2}


def test_tolak_jumlah_baris_salah():
    template = _template().iloc[:-1]           # 1457 baris
    with pytest.raises(AssertionError, match="1458"):
        build_validated_submission(np.zeros(1457, dtype=int), template)


def test_tolak_label_di_luar_0_1_2():
    template = _template()
    pred = np.zeros(1458, dtype=int)
    pred[0] = 3                                  # label ilegal
    with pytest.raises(AssertionError, match=r"\{0, 1, 2\}"):
        build_validated_submission(pred, template)


def test_tolak_ada_nan_di_predicted():
    template = _template()
    pred = np.zeros(1458, dtype=float)
    pred[5] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        build_validated_submission(pred, template)
