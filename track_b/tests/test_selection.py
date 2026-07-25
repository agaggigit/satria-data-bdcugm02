import pandas as pd
import pytest

from selection import decide, select_grid_winner


# --- decide: sama semantik dgn tta_compare.decide_tta, generik namanya ---

def test_decide_accepts_when_mean_up_and_min_not_down():
    assert decide(0.9901, 0.9896, 0.9925, 0.9910) == "TERIMA"


def test_decide_accepts_at_the_min_equality_boundary():
    assert decide(0.9901, 0.9896, 0.9910, 0.9896) == "TERIMA"


def test_decide_rejects_when_mean_up_but_min_down():
    assert decide(0.9901, 0.9896, 0.9925, 0.9880) == "TOLAK (overfit ke OOF -- mean naik tapi min turun)"


def test_decide_rejects_when_mean_down():
    assert decide(0.9901, 0.9896, 0.9890, 0.9895) == "TOLAK (mean tidak naik)"


def test_decide_rejects_at_exact_mean_tie():
    assert decide(0.9901, 0.9896, 0.9901, 0.9999) == "TOLAK (mean tidak naik)"


# --- select_grid_winner ---

def _df(rows):
    return pd.DataFrame(rows)


def test_select_grid_winner_picks_clear_highest_mean():
    df = _df([
        {"combo": "a", "head": "linear", "mean": 0.90, "min": 0.88, "std": 0.01},
        {"combo": "a", "head": "mlp", "mean": 0.95, "min": 0.93, "std": 0.01},
        {"combo": "a", "head": "lgbm", "mean": 0.80, "min": 0.75, "std": 0.02},
    ])
    winner = select_grid_winner(df)
    assert winner["head"] == "mlp"


def test_select_grid_winner_prefers_higher_min_within_tie_band():
    """Dua kandidat mean beda < 0.002 -- yang menang HARUS min tertinggi,
    bukan mean tertinggi (persis B9 poin 2)."""
    df = _df([
        {"combo": "a", "head": "mlp", "mean": 0.9520, "min": 0.9400, "std": 0.02},
        {"combo": "a", "head": "linear", "mean": 0.9510, "min": 0.9480, "std": 0.005},  # mean sedikit lbh rendah
    ])
    winner = select_grid_winner(df)
    assert winner["head"] == "linear", \
        "selisih mean < 0.002 harusnya menang min tertinggi, bukan mean tertinggi"


def test_select_grid_winner_outside_tie_band_mean_wins():
    df = _df([
        {"combo": "a", "head": "mlp", "mean": 0.9700, "min": 0.9500, "std": 0.02},
        {"combo": "a", "head": "linear", "mean": 0.9510, "min": 0.9505, "std": 0.001},  # min tinggi tapi mean jauh
    ])
    winner = select_grid_winner(df)
    assert winner["head"] == "mlp", "selisih mean > 0.002 -- mean tertinggi tetap menang"


def test_select_grid_winner_tie_breaks_by_std_when_mean_and_min_equal():
    df = _df([
        {"combo": "a", "head": "mlp", "mean": 0.95, "min": 0.94, "std": 0.02},
        {"combo": "a", "head": "linear", "mean": 0.95, "min": 0.94, "std": 0.005},
    ])
    winner = select_grid_winner(df)
    assert winner["head"] == "linear"


def test_select_grid_winner_rejects_empty_table():
    with pytest.raises(AssertionError, match="kosong"):
        select_grid_winner(pd.DataFrame(columns=["mean", "min", "std"]))


def test_select_grid_winner_rejects_missing_required_columns():
    with pytest.raises(AssertionError, match="kolom wajib"):
        select_grid_winner(pd.DataFrame([{"mean": 0.9}]))
