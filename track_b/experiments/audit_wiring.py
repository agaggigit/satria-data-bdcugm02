"""audit_wiring.py — GATE sebelum Fase 3 (multi-family).

Buktikan wiring benar SEBELUM membakar GPU: untuk tiap family di FAMILY_REGISTRY,
cetak class model yang BENAR-BENAR terbentuk, dan mean/std yang BENAR-BENAR
dipakai loader. Kalau Swin mencetak "ConvNeXt", kalian tahu HARI INI -- bukan
setelah beberapa jam training di GPU yang salah.

Jalankan dari track_b/src/ (sys.path & cwd harus di sini, sama seperti skrip lain):
    python ../experiments/audit_wiring.py [path/ke/folds.csv]

folds_csv default ke cfg.folds_csv (folds_v2.csv Track A) kalau tidak diberi.
pretrained=False sengaja -- ini audit wiring, bukan audit bobot pretrained.
"""
import os
import sys

import pandas as pd

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

from config import make_cfg           # noqa: E402
from model import FAMILY_REGISTRY      # noqa: E402
from transforms import normalize_of    # noqa: E402
from train import setup_run            # noqa: E402


def run_audit(folds_csv: str) -> pd.DataFrame:
    rows = []
    for key, spec in FAMILY_REGISTRY.items():
        cfg = make_cfg(
            backbone=spec.model_name,
            run_name=f"audit_{spec.short}",
            img_size=spec.native_size,
            folds_csv=folds_csv,
            batch=2,
            num_workers=0,
            pretrained=False,
        )
        ctx = setup_run(cfg, fold=0)
        mean, std = normalize_of(ctx["train_loader"].dataset.transform)

        rows.append({
            "short": spec.short,
            "model_name": spec.model_name,
            "model_class": type(ctx["model"]).__name__,
            "img_size": cfg.img_size,
            "loader_mean": str(tuple(round(m, 3) for m in mean)),
            "config_mean": str(tuple(round(m, 3) for m in ctx["data_config"]["mean"])),
            "match": tuple(round(m, 3) for m in mean)
                     == tuple(round(m, 3) for m in ctx["data_config"]["mean"]),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from config import CFG

    folds_csv = sys.argv[1] if len(sys.argv) > 1 else CFG.folds_csv
    df = run_audit(folds_csv)

    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/wiring_audit.csv", index=False)
    print(df.to_string(index=False))

    # Gate: semua harus lolos, kalau tidak -> berhenti, jangan mulai Fase 3.
    assert df["match"].all(), "ADA LOADER YANG NORMALISASINYA TIDAK IKUT BACKBONE"
    assert df["model_class"].nunique() == len(df), \
        "ADA DUA FAMILY YANG MEMBENTUK CLASS MODEL SAMA -- backbone tidak tersambung"
    print("\nWIRING AUDIT HIJAU — Fase 3 boleh mulai.")
