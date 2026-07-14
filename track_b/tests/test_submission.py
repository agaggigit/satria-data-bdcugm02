import numpy as np
import pandas as pd
import pytest

from submission import make_submission, validate_submission


@pytest.fixture
def template():
    return pd.DataFrame({"id": list(range(1, 1459)), "predicted": [0] * 1458})


def test_submission_has_1458_rows_and_right_columns(template):
    sub = make_submission(np.zeros(1458, dtype=int), template)
    assert len(sub) == 1458
    assert list(sub.columns) == ["id", "predicted"]


def test_submission_id_order_matches_template_exactly(template):
    sub = make_submission(np.random.randint(0, 3, 1458), template)
    assert (sub["id"].to_numpy() == template["id"].to_numpy()).all()


def test_validator_rejects_wrong_row_count(template):
    bad = make_submission(np.zeros(1458, dtype=int), template).iloc[:-1]
    with pytest.raises(AssertionError, match="1458"):
        validate_submission(bad, template)


def test_validator_rejects_label_outside_0_1_2(template):
    bad = make_submission(np.zeros(1458, dtype=int), template)
    bad.loc[0, "predicted"] = 3
    with pytest.raises(AssertionError, match=r"\{0, 1, 2\}"):
        validate_submission(bad, template)


def test_validator_rejects_nan(template):
    bad = make_submission(np.zeros(1458, dtype=float), template)
    bad.loc[5, "predicted"] = np.nan
    with pytest.raises(AssertionError, match="NaN"):
        validate_submission(bad, template)


def test_validator_accepts_valid_submission(template):
    good = make_submission(np.random.randint(0, 3, 1458), template)
    validate_submission(good, template)   # tidak raise
