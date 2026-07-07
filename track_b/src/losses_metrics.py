import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report

CLASS_NAMES = ["Recyclable", "Electronic", "Organic"]  # 0, 1, 2

def build_loss(class_weights=None, label_smoothing=0.1):
    """Weighted CrossEntropy. class_weights: tensor [3] dari Track A.
    Dummy dulu [1,1,1], nanti diganti angka asli."""
    if class_weights is None:
        class_weights = torch.tensor([1.0, 1.0, 1.0])
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

def macro_f1(preds, labels):
    """Macro-F1 persis seperti panitia hitung.
    labels=[0,1,2] eksplisit supaya kelas yang tidak diprediksi tetap dihitung 0."""
    preds = preds.detach().cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    return f1_score(labels, preds, average="macro", labels=[0, 1, 2], zero_division=0.0)

def print_report(preds, labels):
    """Detail per-kelas untuk debugging. Panggil di akhir validation."""
    preds = preds.detach().cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0.0))
