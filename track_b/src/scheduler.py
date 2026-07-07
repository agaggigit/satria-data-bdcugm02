import math
from torch.optim.lr_scheduler import LambdaLR

def build_scheduler(optimizer, steps_per_epoch, cfg):
    """Warmup linear lalu cosine decay. Dipanggil .step() TIAP BATCH."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.epochs * steps_per_epoch
    min_ratio = cfg.min_lr / cfg.lr

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  # linear warmup 0 → 1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))  # 1 → 0
        return min_ratio + (1 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
