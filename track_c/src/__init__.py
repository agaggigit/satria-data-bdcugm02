# track_c/src/__init__.py
# Modul Track C (v2, embedding-based pipeline):
#   config_c.py           — CFG_C + COMPOSITIONS (multi-backbone)
#   inference.py          — assert_label_mapping (tetap dipakai)
#   threshold_tuning.py   — tune_thresholds_oof, apply_thresholds
#   nested_validation.py  — nested_cv_threshold, nested_cv_weight_search  [BARU]
#   ensemble_embedding.py — run_ensemble_inference (berbasis emb cache)   [BARU]
#   manifest.py           — build_manifest, save_manifest                 [BARU]
#   validator.py          — validate_submission
#   generate_submission.py— pipeline E2E (Step 0-8)
