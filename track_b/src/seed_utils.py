import torch
import numpy as np
import random

def set_seed(seed=42):
    """Seed SEMUA sumber randomness. Kontrak: seed=42 (Workflow_Koordinasi_ABC.md)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
