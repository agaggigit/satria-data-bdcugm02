"""Transform per-backbone dari timm data_config.

Modul ini dipakai DUA track:
  - Track B: membangun loader training (loaders.py)
  - Track C: preprocess test set saat inference per family
Jangan menduplikasi logikanya di tempat lain.

Urutan augmentasi train mengikuti PERSIS track_a/src/dataset.py
(RandomResizedCrop scale 0.7-1.0, HorizontalFlip, ColorJitter, RandomRotation(15)).
Yang boleh beda hanya Normalize (per-backbone, dari data_config) dan
RandomVerticalFlip (opsional lewat cfg.vflip -- Track A meng-hardcode p=0.2,
tapi Track B sekarang mengontrolnya sendiri lewat parameter vflip eksplisit).
"""
import torchvision.transforms as T


def build_transforms(data_config: dict, img_size: int | None = None,
                      train: bool = False, vflip: bool = False) -> T.Compose:
    size = img_size or data_config["input_size"][1]
    mean = tuple(data_config["mean"])
    std = tuple(data_config["std"])

    if train:
        ops = [
            T.RandomResizedCrop(size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
        ]
        if vflip:
            ops.append(T.RandomVerticalFlip(p=0.2))
        ops += [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    else:
        ops = [
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]

    return T.Compose(ops)


def normalize_of(transform: T.Compose) -> tuple:
    """Ambil (mean, std) yang benar-benar dipakai -- untuk test & audit."""
    for op in transform.transforms:
        if isinstance(op, T.Normalize):
            return tuple(op.mean), tuple(op.std)
    raise ValueError("transform tidak punya Normalize")
