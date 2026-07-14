import numpy as np
import pandas as pd

from consistency import fold_consistency


def _df(n=60):
    # label periode 3, fold periode 5 -- gcd(3,5)=1, jadi tiap fold dapat
    # campuran merata dari ketiga kelas (bukan 1 kelas per fold seperti pola
    # asli plan, yang bikin macro_f1 sempurna cuma 0.333 bukan 1.0).
    return pd.DataFrame({
        "filepath": [f"i{i}.jpg" for i in range(n)],
        "label": np.tile([0, 1, 2], n // 3),
        "fold": np.tile(np.arange(5), n // 5),
    })


def test_reports_one_score_per_fold():
    df = _df()
    oof = np.full((len(df), 3), 1 / 3)
    out = fold_consistency(oof, df)
    assert len(out["per_fold"]) == 5
    assert {"mean", "std", "min"} <= set(out)


def test_perfect_oof_gives_mean_one_and_zero_std():
    df = _df()
    oof = np.zeros((len(df), 3))
    oof[np.arange(len(df)), df["label"].to_numpy()] = 1.0
    out = fold_consistency(oof, df)
    assert np.isclose(out["mean"], 1.0)
    assert np.isclose(out["std"], 0.0)
