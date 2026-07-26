"""vision_classifier.py — Model untuk Task 5 (LoRA & last-layer fine-tuning,
TRACK_B_ARAHAN_V3.md B8). Beda paradigma dari head-di-atas-embedding-beku
(Task 0-4): di sini backbone vision SigLIP/DINOv3 ikut dilatih (parsial),
bukan cuma diekstrak sekali lalu dibekukan.

HANYA vision tower yang dipakai/dilatih -- SiglipModel/Siglip2Model punya
text_model + vision_model; kompetisi melarang informasi di luar konten
gambar (lihat embed.py: "kita memang tidak pernah memakai text tower").
LoRA/freeze HARUS discoped ke vision_model saja -- lihat VISION_ATTN_PATTERN.

VISION_ATTN_PATTERN diverifikasi lewat inspeksi arsitektur nyata
(google/siglip2-base-patch16-256 via AutoModel.from_config, tanpa unduh
bobot): vision_model.encoder.layers.{i}.self_attn.{q_proj,v_proj}. Kalau
target_modules dikirim sbg LIST ke peft, itu substring-match dan JUGA kena
text_model (terbukti empiris: 48 vision + 48 text ter-adapt) -- HARUS regex
string supaya benar-benar ter-scope vision only (terbukti: vision=48 text=0).
"""
import torch
import torch.nn as nn

VISION_ATTN_PATTERN = r"vision_model\.encoder\.layers\.\d+\.self_attn\.(q_proj|k_proj)"


class VisionClassifier(nn.Module):
    """encoder (SiglipModel/Siglip2Model lengkap, ATAU vision tower saja) ->
    pooled representation -> Linear(num_classes). Dipakai untuk LoRA & last-layer FT."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_classes: int = 3):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(hidden_size, num_classes)

    def _vision_model(self) -> nn.Module:
        return getattr(self.encoder, "vision_model", self.encoder)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self._vision_model()(pixel_values=pixel_values)
        pooled = getattr(out, "pooler_output", None)
        if pooled is None:
            pooled = out.last_hidden_state.mean(dim=1)   # fallback: mean-pool token
        return self.head(pooled)


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def configure_last_layer_finetuning(model: VisionClassifier, n_blocks: int = 1) -> None:
    """Freeze SEMUA parameter, lalu buka HANYA head (baru, wajib dilatih) +
    n_blocks terakhir vision_model.encoder.layers (B8: '1-2 blok terakhir
    saja'). text_model (kalau ada) tetap beku total."""
    freeze_all(model)
    for p in model.head.parameters():
        p.requires_grad = True

    layers = model._vision_model().encoder.layers
    assert 1 <= n_blocks <= len(layers), \
        f"n_blocks={n_blocks} di luar rentang 1..{len(layers)}"
    for layer in layers[len(layers) - n_blocks:]:
        for p in layer.parameters():
            p.requires_grad = True


def apply_lora(model: VisionClassifier, r: int = 4, lora_alpha: int = 8,
               lora_dropout: float = 0.05,
               target_pattern: str = VISION_ATTN_PATTERN) -> VisionClassifier:
    """B8 / Paper SigLIP2-SO400M ref: rank 4, alpha 8, dropout 0.05, target q_proj & k_proj
    DI VISION TOWER SAJA. Backbone asli dibekukan total (freeze_all) -- cuma adapter
    LoRA kecil + head baru yang trainable."""
    from peft import LoraConfig, get_peft_model

    freeze_all(model.encoder)
    lora_cfg = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          target_modules=target_pattern)
    model.encoder = get_peft_model(model.encoder, lora_cfg)
    for p in model.head.parameters():
        p.requires_grad = True
    return model


def hf_processor_to_data_config(processor) -> dict:
    """Konversi AutoImageProcessor (HF) -> dict data_config generik yang dipakai
    build_transforms()/get_loaders_b() (transforms.py, ditulis awalnya utk
    data_config timm, tapi cuma butuh 3 key ini). Diverifikasi lewat processor
    asli google/siglip2-base-patch16-256: SizeDict(height=256,width=256),
    image_mean/std=(0.5,0.5,0.5). `size` bisa berupa dict ATAU objek ber-atribut
    tergantung versi transformers -- tangani dua-duanya."""
    size = processor.size
    height = size["height"] if isinstance(size, dict) else size.height
    width = size["width"] if isinstance(size, dict) else size.width
    return {
        "input_size": (3, height, width),
        "mean": tuple(processor.image_mean),
        "std": tuple(processor.image_std),
    }


def build_llrd_optimizer(model: VisionClassifier, head_lr: float,
                         backbone_lr_ratio: float = 1 / 50,
                         weight_decay: float = 0.05) -> torch.optim.Optimizer:
    """LLRD-lite untuk last-layer FT (B8: 'LR backbone 10-100x lebih kecil dari
    LR head -- prinsip yang dulu terbukti satu-satunya yang membantu, +0.0158').
    Dua grup: head_lr penuh utk model.head, head_lr*ratio utk sisa parameter
    trainable (blok backbone yang dibuka). Parameter beku TIDAK masuk optimizer
    sama sekali (bukan cuma lr=0 -- memang tak pernah di-update)."""
    head_params = [p for p in model.head.parameters() if p.requires_grad]
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("head.")]

    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": head_lr * backbone_lr_ratio})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
