"""manifest.py — Buat dan simpan manifest per submission.

Dua kegunaan nyata manifest:
1. Verifikasi video panitia: bisa menelusuri tiap submission ke kode & artefak persisnya
2. Diagnosis: kalau submission 2 skornya turun dari 1, manifest memberi tahu APA yang beda

Format: JSON, satu file per submission.
"""
import json
import os
import hashlib
import datetime
import numpy as np


def _file_hash(path: str, chunk_size: int = 65536) -> str:
    """SHA-256 dari file (untuk bukti artefak yang dipakai)."""
    if not os.path.exists(path):
        return "FILE_NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]  # 16 hex chars cukup untuk identifikasi


def build_manifest(
    submission_number: int,
    submission_path: str,
    cfg,
    active_compositions: list,
    thresholds: list,
    nested_result: dict = None,
    ensemble_weights: dict = None,
    notes: str = "",
) -> dict:
    """Bangun dict manifest untuk satu submission.

    Args:
        submission_number : 1, 2, atau 3
        submission_path   : path ke file submission_apace.csv yang dibuat
        cfg               : CFG_C
        active_compositions: list nama komposisi yang dipakai
        thresholds        : [W0, W1, W2] threshold yang diterapkan
        nested_result     : output dari nested_cv_threshold() (boleh None)
        ensemble_weights  : dict {comp_key: float} bobot ensemble (None = seragam)
        notes             : catatan bebas dari operator
    """
    # Hash artefak kunci untuk traceability
    oof_hashes = {}
    for comp_key in active_compositions:
        comp = cfg.compositions.get(comp_key)
        if comp:
            oof_hashes[comp_key] = {
                "oof_path": comp.oof,
                "oof_hash": _file_hash(comp.oof),
                "emb_test_path": comp.emb_test,
                "emb_test_hash": _file_hash(comp.emb_test),
                "head": comp.head,
                "backbone": comp.backbone,
            }

    manifest = {
        "submission_number": submission_number,
        "team_name": cfg.team_name,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "submission_file": os.path.basename(submission_path),
        "submission_hash": _file_hash(submission_path),

        # Data
        "folds_csv": cfg.folds_csv,
        "folds_csv_hash": _file_hash(cfg.folds_csv),
        "data_version": "v2_cleaned",

        # Model & komposisi
        "active_compositions": active_compositions,
        "artefak": oof_hashes,
        "ensemble_weights": ensemble_weights if ensemble_weights else "uniform",

        # Threshold
        "thresholds": {
            "method": "grid_search_w1_w2",
            "values": thresholds,
            "W0_kelas_Recyclable": thresholds[0] if len(thresholds) > 0 else None,
            "W1_kelas_Electronic": thresholds[1] if len(thresholds) > 1 else None,
            "W2_kelas_Organic":    thresholds[2] if len(thresholds) > 2 else None,
        },

        # Estimasi performa
        "cv_estimate": {
            "nested_mean": nested_result["nested_mean"] if nested_result else None,
            "nested_min": nested_result["nested_min"] if nested_result else None,
            "nested_std": nested_result["nested_std"] if nested_result else None,
            "full_tuned_f1": nested_result["full_tuned_f1"] if nested_result else None,
            "illusion_gap": nested_result["illusion_gap"] if nested_result else None,
        },

        "notes": notes,
    }

    return manifest


def save_manifest(manifest: dict, output_dir: str,
                  submission_number: int) -> str:
    """Simpan manifest ke output_dir/manifest_sub{N}.json."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"manifest_sub{submission_number}.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest tersimpan: {path}")
    return path


def print_manifest_summary(manifest: dict):
    """Cetak ringkasan manifest yang mudah dibaca."""
    print("\n" + "=" * 55)
    print(f"MANIFEST — Submission {manifest['submission_number']}")
    print("=" * 55)
    print(f"  Timestamp      : {manifest['timestamp_utc']}")
    print(f"  File           : {manifest['submission_file']}")
    print(f"  Komposisi      : {manifest['active_compositions']}")
    print(f"  Threshold      : {manifest['thresholds']['values']}")

    cv = manifest.get("cv_estimate", {})
    if cv.get("nested_mean") is not None:
        print(f"  Nested CV mean : {cv['nested_mean']:.5f}")
        print(f"  Nested CV min  : {cv['nested_min']:.5f}")
        print(f"  Illusion gap   : {cv['illusion_gap']:+.5f}")

    print(f"  Notes          : {manifest.get('notes', '-')}")
    print("=" * 55)


if __name__ == "__main__":
    # Smoke test
    print("Smoke test manifest.py ...")
    from types import SimpleNamespace
    import tempfile

    dummy_cfg = SimpleNamespace(
        team_name="apace",
        folds_csv="/tmp/folds_v2.csv",
        compositions={
            "siglip2_knn": SimpleNamespace(
                oof="/tmp/oof_siglip2so400m_knn_v2.npy",
                emb_test="/tmp/siglip2so400m_test.npy",
                head="knn",
                backbone="google/siglip2-so400m-patch14-384",
            )
        }
    )
    dummy_nested = {
        "nested_mean": 0.9928,
        "nested_min": 0.9918,
        "nested_std": 0.0008,
        "full_tuned_f1": 0.9930,
        "illusion_gap": 0.0002,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        # Buat file dummy
        open(os.path.join(tmpdir, "submission_apace.csv"), "w").close()
        m = build_manifest(
            submission_number=2,
            submission_path=os.path.join(tmpdir, "submission_apace.csv"),
            cfg=dummy_cfg,
            active_compositions=["siglip2_knn"],
            thresholds=[1.0, 0.9, 1.1],
            nested_result=dummy_nested,
            notes="Test manifest",
        )
        path = save_manifest(m, tmpdir, 2)
        print_manifest_summary(m)
        assert os.path.exists(path)

    print("Smoke test PASSED")
