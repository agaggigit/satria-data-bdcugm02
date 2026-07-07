import timm

def build_model(num_classes=3, pretrained=True, drop_path_rate=0.1,
                backbone="convnext_tiny.in12k_ft_in1k"):
    """Baseline: ConvNeXt-Tiny (in12k→in1k pretrained), wajib didokumentasikan
    di report (aturan panitia BDC). Param `backbone` opsional untuk kandidat
    ensemble lain (lihat model_zoo.py) — tidak mengubah default training utama."""
    try:
        model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )
    except TypeError:
        # Sebagian backbone alternatif (mis. beberapa varian EfficientNet)
        # tidak menerima kwarg drop_path_rate di constructor timm.
        model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)  # hemat VRAM
    return model

def get_data_config(model):
    """Ambil normalisasi + input size yang TEPAT untuk model ini.
    Dipakai Track A untuk transform."""
    return timm.data.resolve_model_data_config(model)
