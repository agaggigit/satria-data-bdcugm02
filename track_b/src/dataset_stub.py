import torch
from torch.utils.data import Dataset, DataLoader

class DummyWasteDataset(Dataset):
    """Dummy pengganti dataset asli Track A.
    Output HARUS sama: image [3, H, W] float, label int ∈ {0, 1, 2}."""
    def __init__(self, n_samples, img_size):
        self.n_samples = n_samples
        self.img_size = img_size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        image = torch.randn(3, self.img_size, self.img_size)
        label = idx % 3
        return image, label

def get_loaders(fold, img_size=224, batch=32):
    """Signature IDENTIK dengan kontrak Track A di Workflow_Koordinasi_ABC.md.
    Nanti tinggal ganti: from dataset import get_loaders"""
    train_ds = DummyWasteDataset(n_samples=512, img_size=img_size)
    val_ds = DummyWasteDataset(n_samples=128, img_size=img_size)

    loader_kwargs = dict(
        batch_size=batch,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader
