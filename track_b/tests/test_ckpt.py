import torch
import pytest

from config import make_cfg
from ckpt import REQUIRED_CKPT_KEYS, checkpoint_path, history_path, save_checkpoint


def _payload():
    return {
        "model_state_dict": {"w": torch.zeros(1)},
        "model_name": "convnext_tiny.in12k_ft_in1k",
        "data_config": {"input_size": (3, 224, 224), "mean": (0.485, 0.456, 0.406),
                        "std": (0.229, 0.224, 0.225), "crop_pct": 0.95},
        "run_name": "convnext_v2",
        "fold": 0,
        "img_size": 224,
        "seed": 42,
    }


def test_filename_carries_run_name(tmp_path):
    cfg = make_cfg(save_dir=str(tmp_path), run_name="convnext_v2")
    assert checkpoint_path(cfg, 0).name == "convnext_v2_fold0.pt"
    assert history_path(cfg, 0).name == "convnext_v2_fold0_history.json"


def test_two_run_names_never_collide(tmp_path):
    v1 = make_cfg(save_dir=str(tmp_path), run_name="convnext_v1")
    v2 = make_cfg(save_dir=str(tmp_path), run_name="convnext_v2")
    assert checkpoint_path(v1, 0) != checkpoint_path(v2, 0)


def test_save_refuses_to_overwrite_existing_checkpoint(tmp_path):
    path = tmp_path / "convnext_v1_fold0.pt"
    save_checkpoint(_payload(), path)
    with pytest.raises(FileExistsError, match="sudah ada"):
        save_checkpoint(_payload(), path)  # ini yang dulu menimpa v1 diam-diam


def test_save_allows_overwrite_when_explicitly_requested(tmp_path):
    path = tmp_path / "convnext_v1_fold0.pt"
    save_checkpoint(_payload(), path)
    save_checkpoint(_payload(), path, allow_overwrite=True)  # tidak raise


def test_save_rejects_payload_without_data_config(tmp_path):
    # Track C butuh data_config untuk preprocess test set dengan benar.
    bad = _payload()
    del bad["data_config"]
    with pytest.raises(ValueError, match="data_config"):
        save_checkpoint(bad, tmp_path / "x.pt")


def test_required_keys_include_everything_track_c_needs():
    assert {"model_name", "data_config", "run_name", "img_size"} <= REQUIRED_CKPT_KEYS
