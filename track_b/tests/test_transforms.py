import numpy as np
import torchvision.transforms as T

from model import get_data_config
from transforms import build_transforms, normalize_of

IMAGENET_MEAN = (0.485, 0.456, 0.406)


def test_effnetv2_transform_uses_its_own_normalization_not_imagenet():
    # Ini test yang akan gagal kalau seseorang meng-hardcode ImageNet lagi.
    dc = get_data_config("tf_efficientnetv2_s.in21k_ft_in1k")
    tfm = build_transforms(dc, train=False)
    mean, _ = normalize_of(tfm)
    assert not np.allclose(mean, IMAGENET_MEAN)
    assert np.allclose(mean, (0.5, 0.5, 0.5), atol=1e-3)


def test_convnext_transform_still_reproduces_imagenet_norm():
    # Parity guard: v1 (dataset.py) dan v2 (transforms.py) harus identik untuk ConvNeXt,
    # supaya perbandingan CV v1 vs v2 tetap apple-to-apple.
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    tfm = build_transforms(dc, train=False)
    mean, std = normalize_of(tfm)
    assert np.allclose(mean, IMAGENET_MEAN, atol=1e-3)
    assert np.allclose(std, (0.229, 0.224, 0.225), atol=1e-3)


def test_eval_transform_has_no_random_augmentation():
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    tfm = build_transforms(dc, train=False)
    names = [type(op).__name__ for op in tfm.transforms]
    assert not any(n.startswith("Random") for n in names), f"augmentasi bocor ke eval: {names}"


def test_vflip_only_appears_when_requested_and_only_in_train():
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    off = [type(o).__name__ for o in build_transforms(dc, train=True, vflip=False).transforms]
    on = [type(o).__name__ for o in build_transforms(dc, train=True, vflip=True).transforms]
    ev = [type(o).__name__ for o in build_transforms(dc, train=False, vflip=True).transforms]
    assert "RandomVerticalFlip" not in off
    assert "RandomVerticalFlip" in on
    assert "RandomVerticalFlip" not in ev


def test_img_size_override_wins_over_native_size():
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    tfm = build_transforms(dc, img_size=288, train=False)
    resize = next(o for o in tfm.transforms if isinstance(o, T.Resize))
    assert resize.size == 288


def test_output_tensor_shape_matches_requested_size():
    from PIL import Image
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    tfm = build_transforms(dc, img_size=224, train=True)
    img = Image.new("RGB", (500, 375), color=(120, 90, 60))
    out = tfm(img)
    assert out.shape == (3, 224, 224)


def test_train_augmentation_order_matches_track_a_except_normalize_and_vflip():
    # dataset.py Track A: RandomResizedCrop, HorizontalFlip, ColorJitter, RandomRotation,
    # ToTensor, Normalize. Kita cuma boleh beda di Normalize (per-backbone) dan
    # menyisipkan RandomVerticalFlip opsional.
    dc = get_data_config("convnext_tiny.in12k_ft_in1k")
    names = [type(o).__name__ for o in build_transforms(dc, train=True, vflip=False).transforms]
    assert names == [
        "RandomResizedCrop", "RandomHorizontalFlip", "ColorJitter",
        "RandomRotation", "ToTensor", "Normalize",
    ]
