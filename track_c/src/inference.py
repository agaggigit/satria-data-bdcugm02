import torch
import numpy as np
import os
import timm

def load_model(checkpoint_path: str, backbone: str, num_classes: int = 3, drop_path_rate: float = 0.1, device: str = 'cpu'):
    """
    Load PyTorch model architecture dan state dict.
    Kita pakai arsitektur yang sama dengan Track B.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint tidak ditemukan di {checkpoint_path}")

    try:
        model = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate
        )
    except TypeError:
        # Fallback jika model tidak menerima drop_path_rate
        model = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=num_classes
        )

    # Load state dict — coba weights_only=True dulu (lebih aman),
    # fallback ke False jika checkpoint lama tidak kompatibel.
    # map_location=device memastikan checkpoint GPU bisa dimuat di CPU.
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    return model

@torch.no_grad()
def predict_test(model, test_loader, device: str = 'cpu', use_amp: bool = True):
    """
    Fungsi dasar inference: forward test_loader lalu return probabilitas [N, 3].
    AMP (float16) hanya dipakai jika device adalah CUDA.
    """
    all_probs = []

    for images, filepaths in test_loader:
        images = images.to(device, non_blocking=True)

        # AMP hanya aktif jika ada GPU; di CPU tidak perlu
        if use_amp and device == 'cuda':
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)

        probs = torch.softmax(logits.float(), dim=1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)

def assert_label_mapping(cfg):
    """
    Pengecekan keamanan terakhir agar mapping kita sama persis
    dengan konvensi bersama A, B, dan C.
    """
    assert cfg.label_map[0] == "Recyclable", "Error: Kelas 0 bukan Recyclable!"
    assert cfg.label_map[1] == "Electronic", "Error: Kelas 1 bukan Electronic!"
    assert cfg.label_map[2] == "Organic", "Error: Kelas 2 bukan Organic!"
