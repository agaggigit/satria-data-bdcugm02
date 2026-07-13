import timm

def build_model(num_classes=3, pretrained=True, drop_path_rate=0.1,
                backbone="convnext_tiny.in12k_ft_in1k", grad_checkpointing=False):
    """Baseline: ConvNeXt-Tiny (in12k→in1k pretrained), wajib didokumentasikan
    di report (aturan panitia BDC). Param `backbone` opsional untuk kandidat
    ensemble lain (lihat model_zoo.py) — tidak mengubah default training utama.

    grad_checkpointing: default False. ConvNeXt-Tiny @224 cuma pakai ~2GB dari
    15GB VRAM T4 — checkpointing (hemat VRAM, korbankan kecepatan lewat recompute
    di backward pass) tidak dibutuhkan di sini. Nyalakan lagi kalau naik ke
    backbone/resolusi yang lebih berat (mis. img_size 288+) dan mepet OOM."""
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

    if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model

def get_data_config(model):
    """Ambil normalisasi + input size yang TEPAT untuk model ini.
    Dipakai Track A untuk transform."""
    return timm.data.resolve_model_data_config(model)
