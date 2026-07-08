"""Simulasi Track C: load semua artefak Track B dari nol."""
import json
import numpy as np
import pandas as pd
import torch
import timm
from config import CFG


def verify():
    ok = True

    # 1. Lima checkpoint bisa di-load ke arsitektur yang benar
    for fold in range(5):
        path = f"{CFG.save_dir}/fold{fold}.pt"
        try:
            model = timm.create_model(CFG.backbone, pretrained=False,
                                      num_classes=3, drop_path_rate=CFG.drop_path_rate)
            state = torch.load(path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            print(f"✅ fold{fold}.pt loadable ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
            del model
        except Exception as e:
            print(f"❌ fold{fold}.pt GAGAL: {e}")
            ok = False

    # 2. OOF: shape, dtype, no-NaN, selaras folds.csv
    oof = np.load(f"{CFG.save_dir}/oof.npy")
    df = pd.read_csv(CFG.folds_csv)
    checks = [
        (oof.shape == (len(df), 3), f"shape {oof.shape} vs folds {len(df)}"),
        (oof.dtype == np.float32,    f"dtype {oof.dtype}"),
        (not np.isnan(oof).any(),    "NaN check"),
        (np.allclose(oof.sum(axis=1), 1.0, atol=1e-3), "probabilitas sum≈1"),
    ]
    for passed, desc in checks:
        print(("✅" if passed else "❌"), "oof.npy:", desc)
        ok = ok and passed

    # 3. Meta & summary readable + berisi field wajib
    with open(f"{CFG.save_dir}/oof_meta.json") as f:
        meta = json.load(f)
    for field in ["shape", "index_source", "oof_overall_macro_f1_argmax", "img_size", "seed"]:
        present = field in meta
        print(("✅" if present else "❌"), f"oof_meta.json field '{field}'")
        ok = ok and present

    with open(f"{CFG.save_dir}/cv_summary.json") as f:
        summary = json.load(f)
    print(f"✅ cv_summary.json: CV {summary['cv_mean_macro_f1']:.4f} ± {summary['cv_std_macro_f1']:.4f}, "
          f"OOF {summary['oof_overall_macro_f1_argmax']:.4f}")

    print("\n" + ("🟢 HANDOFF LENGKAP — Track C bisa jalan mandiri" if ok
                  else "🔴 ADA MASALAH — perbaiki sebelum umumkan GATE 3"))
    return ok


if __name__ == "__main__":
    verify()
