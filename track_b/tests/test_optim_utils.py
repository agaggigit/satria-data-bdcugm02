import torch
from model import build_model
from optim_utils import build_optimizer


def test_layer_decay_produces_multiple_lr_groups():
    model, _ = build_model("convnext_tiny.in12k_ft_in1k", pretrained=False)
    opt = build_optimizer(model, lr=3e-4, weight_decay=0.05, layer_decay=0.9)
    lrs = {round(g["lr"], 10) for g in opt.param_groups}
    assert len(lrs) > 1, "LLRD aktif tapi semua param group punya LR sama"
    assert max(lrs) <= 3e-4 + 1e-12, "tidak ada layer yang boleh melebihi base LR"


def test_no_layer_decay_gives_single_lr():
    model, _ = build_model("convnext_tiny.in12k_ft_in1k", pretrained=False)
    opt = build_optimizer(model, lr=3e-4, weight_decay=0.05, layer_decay=None)
    lrs = {round(g["lr"], 10) for g in opt.param_groups}
    assert lrs == {3e-4}
