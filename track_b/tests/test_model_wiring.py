import numpy as np
import pytest
import torch

from model import FAMILY_REGISTRY, build_model, get_data_config, short_name


def test_build_model_requires_model_name():
    # Bug asli: build_model punya default yang jalan, jadi lupa kirim backbone = senyap.
    with pytest.raises(TypeError):
        build_model(num_classes=3)  # type: ignore[call-arg]


def test_build_model_actually_returns_requested_family():
    model, _ = build_model("swin_tiny_patch4_window7_224.ms_in1k",
                           num_classes=3, pretrained=False)
    name = type(model).__name__.lower()
    assert "swin" in name, f"minta Swin, dapat {type(model).__name__} — backbone tidak tersambung!"


def test_convnext_and_swin_are_not_the_same_class():
    m1, _ = build_model("convnext_tiny.in12k_ft_in1k", num_classes=3, pretrained=False)
    m2, _ = build_model("swin_tiny_patch4_window7_224.ms_in1k", num_classes=3, pretrained=False)
    assert type(m1) is not type(m2)


def test_effnetv2_normalization_is_not_imagenet():
    dc = get_data_config("tf_efficientnetv2_s.in21k_ft_in1k")
    assert not np.allclose(dc["mean"], (0.485, 0.456, 0.406)), \
        "EffNetV2 tidak boleh memakai normalisasi ImageNet"
    assert np.allclose(dc["mean"], (0.5, 0.5, 0.5), atol=1e-3)


def test_short_names_are_explicit_and_unique():
    shorts = [spec.short for spec in FAMILY_REGISTRY.values()]
    assert len(shorts) == len(set(shorts)), "nama pendek bentrok"
    assert "tf" not in shorts, "nama pendek hasil string-slicing bocor ke registry"
    assert short_name("tf_efficientnetv2_s.in21k_ft_in1k") == "effnetv2s"


def test_densenet_does_not_receive_unsupported_drop_path():
    # timm.create_model("densenet201...", drop_path_rate=0.1) -> TypeError.
    # build_model harus menghormati registry dan tidak mengirim kwarg itu.
    model, _ = build_model("densenet201.tv_in1k", num_classes=3, pretrained=False,
                           drop_path_rate=0.1)
    assert model is not None


def test_every_registry_entry_can_be_built_and_outputs_three_logits():
    for spec in FAMILY_REGISTRY.values():
        model, dc = build_model(spec.model_name, num_classes=3, pretrained=False)
        size = spec.native_size
        out = model(torch.randn(1, 3, size, size))
        assert out.shape == (1, 3), f"{spec.short} output shape salah"
        assert len(dc["mean"]) == 3 and len(dc["std"]) == 3
