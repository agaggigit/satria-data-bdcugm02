"""optim_utils.py — Optimizer dengan layer-wise LR decay (LLRD) opsional.

LLRD tidak ditulis manual: timm.optim.create_optimizer_v2 sudah tahu struktur
layer tiap backbone dan bisa auto-assign LR per kedalaman (smooth, bukan
freeze on/off keras). scheduler.py (warmup+cosine) tidak berubah — dipakai
apa adanya di atas optimizer ini.
"""
from timm.optim import create_optimizer_v2


def build_optimizer(model, lr: float, weight_decay: float, layer_decay: float = None):
    kwargs = dict(opt="adamw", lr=lr, weight_decay=weight_decay)
    if layer_decay is not None:
        kwargs["layer_decay"] = layer_decay
    optimizer = create_optimizer_v2(model, **kwargs)

    # timm >=0.9 menyimpan faktor LLRD di param_group["lr_scale"], TIDAK
    # otomatis mengubah "lr". Scheduler kita (LambdaLR di scheduler.py) cuma
    # baca "lr" awal tiap grup, jadi kalau tidak dibakar di sini, LLRD diam-diam
    # tidak berpengaruh sama sekali walau param_groups kelihatan banyak.
    if layer_decay is not None:
        for group in optimizer.param_groups:
            group["lr"] = group["lr"] * group.get("lr_scale", 1.0)

    return optimizer
