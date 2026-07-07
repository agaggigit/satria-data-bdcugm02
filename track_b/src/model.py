import timm

def build_model(num_classes=3, pretrained=True, drop_path_rate=0.1):
    """ConvNeXt-Tiny (in12k→in1k pretrained).
    Wajib didokumentasikan di report (aturan panitia BDC)."""
    model = timm.create_model(
        "convnext_tiny.in12k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    model.set_grad_checkpointing(True)  # hemat VRAM
    return model

def get_data_config(model):
    """Ambil normalisasi + input size yang TEPAT untuk model ini.
    Dipakai Track A untuk transform."""
    return timm.data.resolve_model_data_config(model)
