from dataclasses import dataclass

import timm
import torch.nn as nn


@dataclass(frozen=True)
class FamilySpec:
    model_name: str
    short: str
    drop_path_rate: float | None   # None = arsitektur ini TIDAK mendukung stochastic depth
    native_size: int
    supports_grad_ckpt: bool
    note: str


FAMILY_REGISTRY: dict[str, FamilySpec] = {
    "convnext": FamilySpec(
        model_name="convnext_tiny.in12k_ft_in1k",
        short="convnext",
        drop_path_rate=0.1,
        native_size=224,
        supports_grad_ckpt=True,
        note="Baseline Fase 1/2. Normalisasi ImageNet.",
    ),
    "swin": FamilySpec(
        model_name="swin_tiny_patch4_window7_224.ms_in1k",
        short="swin",
        drop_path_rate=0.2,
        native_size=224,           # window size terikat 224 -- JANGAN diubah ke 288
        supports_grad_ckpt=True,
        note="Transformer. Resolusi TERKUNCI 224 (window attention).",
    ),
    "effnetv2s": FamilySpec(
        model_name="tf_efficientnetv2_s.in21k_ft_in1k",
        short="effnetv2s",
        drop_path_rate=0.2,
        native_size=300,
        supports_grad_ckpt=True,
        note="NORMALISASI 0.5/0.5/0.5, BUKAN ImageNet.",
    ),
    "densenet201": FamilySpec(
        model_name="densenet201.tv_in1k",
        short="densenet201",
        drop_path_rate=None,       # DenseNet tidak punya stochastic depth
        native_size=224,
        supports_grad_ckpt=False,
        note="Cadangan. Tidak mendukung drop_path.",
    ),
}

_BY_MODEL_NAME = {spec.model_name: spec for spec in FAMILY_REGISTRY.values()}


def spec_for(model_name: str) -> FamilySpec:
    if model_name not in _BY_MODEL_NAME:
        raise KeyError(
            f"'{model_name}' tidak ada di FAMILY_REGISTRY. "
            f"Daftarkan dulu (nama pendek, drop_path, native_size) sebelum dipakai."
        )
    return _BY_MODEL_NAME[model_name]


def short_name(model_name: str) -> str:
    return spec_for(model_name).short


def build_model(
    model_name: str,                 # WAJIB. Tidak ada default -- itu sumber bug senyap.
    num_classes: int = 3,
    pretrained: bool = True,
    drop_path_rate: float | None = None,
    grad_checkpointing: bool = False,
) -> tuple[nn.Module, dict]:
    """Return (model, data_config). data_config WAJIB dipakai untuk membangun transform."""
    spec = spec_for(model_name)

    kwargs = dict(pretrained=pretrained, num_classes=num_classes)

    # Kirim drop_path HANYA kalau arsitekturnya mendukung (DenseNet tidak).
    if spec.drop_path_rate is not None:
        kwargs["drop_path_rate"] = (
            drop_path_rate if drop_path_rate is not None else spec.drop_path_rate
        )

    model: nn.Module = timm.create_model(model_name, **kwargs)

    if grad_checkpointing and spec.supports_grad_ckpt and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)

    data_config = timm.data.resolve_model_data_config(model)
    return model, data_config


def get_data_config(model_name: str) -> dict:
    """Preprocessing config tanpa mengunduh bobot."""
    model = timm.create_model(model_name, pretrained=False, num_classes=3)
    return timm.data.resolve_model_data_config(model)
