"""submission.py — Generate + validasi submission.csv.

Test data HANYA untuk prediksi akhir -- tidak untuk fit, seleksi, atau tuning
apa pun.
"""
import numpy as np
import pandas as pd


def make_submission(pred: np.ndarray, template_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"id": template_df["id"].to_numpy(), "predicted": pred})


def validate_submission(df: pd.DataFrame, template_df: pd.DataFrame) -> None:
    assert len(df) == 1458, f"jumlah baris {len(df)} != 1458"
    assert list(df.columns) == ["id", "predicted"], f"kolom salah: {list(df.columns)}"
    assert (df["id"].to_numpy() == template_df["id"].to_numpy()).all(), \
        "urutan id tidak sama dengan template"
    assert not df["predicted"].isna().any(), "ada NaN di kolom predicted"
    vals = set(pd.unique(df["predicted"]))
    assert vals <= {0, 1, 2}, f"nilai predicted di luar {{0, 1, 2}}: {vals - {0, 1, 2}}"
