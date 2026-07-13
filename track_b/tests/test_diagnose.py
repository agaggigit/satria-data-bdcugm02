import numpy as np
from diagnose import verdict, class_action_table


def test_verdict_underfit_when_train_f1_still_low():
    assert verdict(train_f1=0.90, val_f1=0.89) == "underfit"


def test_verdict_overfit_when_train_high_but_val_far_behind():
    assert verdict(train_f1=0.999, val_f1=0.92) == "overfit"


def test_verdict_ok_when_train_high_and_gap_small():
    assert verdict(train_f1=0.995, val_f1=0.985) == "ok"


def test_class_action_table_has_one_row_per_class():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 2, 1, 0, 2, 0])
    df = class_action_table(y_true, y_pred)
    assert len(df) == 3
    assert list(df["class_id"]) == [0, 1, 2]
    assert set(df["class_name"]) == {"Recyclable", "Electronic", "Organic"}


def test_class_action_table_reports_most_confused_partner():
    # Semua Organic (2) salah diprediksi sebagai Recyclable (0).
    y_true = np.array([2, 2, 2, 0])
    y_pred = np.array([0, 0, 0, 0])
    df = class_action_table(y_true, y_pred)
    row = df[df["class_id"] == 2].iloc[0]
    assert row["top_confusion_with"] == "Recyclable"
    assert row["gap_to_target"] > 0
