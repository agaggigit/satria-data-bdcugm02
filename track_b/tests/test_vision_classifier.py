"""Test CPU-only, sepenuhnya OFFLINE (tidak unduh bobot/config dari HF Hub)
untuk vision_classifier.py (Task 5, TRACK_B_ARAHAN_V3.md B8).

_FakeSiglipModel meniru struktur nama module ASLI SiglipModel (diverifikasi
manual lewat AutoModel.from_config('google/siglip2-base-patch16-256') tanpa
unduh bobot -- lihat catatan di vision_classifier.py): vision_model.encoder.
layers[i].self_attn.{q_proj,v_proj}, DAN text_model dengan struktur sama
supaya test bisa membuktikan LoRA/freeze TIDAK PERNAH menyentuh text tower
(kompetisi: "image encoder saja")."""
import pytest
import torch
import torch.nn as nn

from vision_classifier import (VISION_ATTN_PATTERN, VisionClassifier,
                               apply_lora, build_llrd_optimizer,
                               configure_last_layer_finetuning, freeze_all,
                               hf_processor_to_data_config)

pytest.importorskip("peft")


class _Output:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output
        self.last_hidden_state = pooler_output.unsqueeze(1)


class _FakeAttn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x):
        return self.out_proj(self.q_proj(x) + self.v_proj(x) + self.k_proj(x))


class _FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = _FakeAttn(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class _FakeTower(nn.Module):
    """Struktur: encoder.layers[i] -- sama utk vision_model MAUPUN text_model."""

    def __init__(self, d, n_layers):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])

    def forward(self, pixel_values):
        x = pixel_values
        for layer in self.encoder.layers:
            x = layer(x)
        return _Output(pooler_output=x)


class _FakeSiglipModel(nn.Module):
    """vision_model DAN text_model dua-duanya ada -- persis SiglipModel asli."""

    def __init__(self, d=16, n_layers=4):
        super().__init__()
        self.vision_model = _FakeTower(d, n_layers)
        self.text_model = _FakeTower(d, n_layers)


D, N_LAYERS, NUM_CLASSES = 16, 4, 3


def _model():
    return VisionClassifier(_FakeSiglipModel(D, N_LAYERS), hidden_size=D, num_classes=NUM_CLASSES)


# --- VisionClassifier: forward shape ---

def test_forward_produces_correct_shape():
    model = _model()
    x = torch.randn(5, D)
    out = model(x)
    assert out.shape == (5, NUM_CLASSES)


# --- apply_lora: scoping vision-only (bukti langsung, bukan asumsi) ---

def test_apply_lora_only_adapts_vision_tower_not_text_tower():
    model = apply_lora(_model(), r=4, lora_alpha=8)
    lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert len(lora_names) > 0, "LoRA tidak terpasang sama sekali"
    assert all("vision_model" in n for n in lora_names), \
        f"ada adapter LoRA di luar vision_model: {[n for n in lora_names if 'vision_model' not in n]}"
    assert not any("text_model" in n for n in lora_names), \
        "LoRA menyentuh text_model -- melanggar aturan 'image encoder saja'"


def test_apply_lora_n_last_blocks_only_attaches_adapters_to_target_layers():
    model = apply_lora(_model(), r=4, lora_alpha=8, n_last_blocks=2)
    lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert len(lora_names) > 0
    assert all("layers.2" in n or "layers.3" in n for n in lora_names), \
        f"LoRA terpasang di luar layer 2 dan 3: {lora_names}"
    assert not any("layers.0" in n or "layers.1" in n for n in lora_names), \
        f"LoRA menyentuh layer 0 atau 1 padahal n_last_blocks=2: {lora_names}"


def test_apply_lora_freezes_original_backbone_weights():
    model = apply_lora(_model(), r=4, lora_alpha=8)
    base_weight_names = [n for n, p in model.named_parameters()
                         if n.endswith(".base_layer.weight") or n.endswith(".base_layer.bias")]
    assert len(base_weight_names) > 0
    for n, p in model.named_parameters():
        if n.endswith(".base_layer.weight") or n.endswith(".base_layer.bias"):
            assert not p.requires_grad, f"bobot asli backbone {n} tidak dibekukan"


def test_apply_lora_text_tower_entirely_frozen():
    model = apply_lora(_model(), r=4, lora_alpha=8)
    for n, p in model.named_parameters():
        if "text_model" in n:
            assert not p.requires_grad, f"parameter text_model {n} ikut trainable"


def test_apply_lora_head_is_trainable():
    model = apply_lora(_model(), r=4, lora_alpha=8)
    assert all(p.requires_grad for p in model.head.parameters())


def test_apply_lora_trainable_fraction_is_small():
    model = apply_lora(_model(), r=4, lora_alpha=8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable / total < 0.5, \
        "adapter LoRA + head trainable > 50% total param -- kemungkinan salah wiring"


def test_apply_lora_forward_backward_only_updates_trainable_params():
    model = apply_lora(_model(), r=4, lora_alpha=8)
    x = torch.randn(4, D)
    y = torch.tensor([0, 1, 2, 0])
    out = model(x)
    loss = nn.functional.cross_entropy(out, y)
    loss.backward()

    for n, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{n} trainable tapi grad None -- putus dari graph"
        else:
            assert p.grad is None or torch.all(p.grad == 0), \
                f"{n} dibekukan tapi punya gradient nonzero -- freeze bocor"


# --- configure_last_layer_finetuning ---

def test_last_layer_ft_opens_only_head_and_last_n_blocks():
    model = _model()
    configure_last_layer_finetuning(model, n_blocks=1)

    assert all(p.requires_grad for p in model.head.parameters())

    layers = model.encoder.vision_model.encoder.layers
    for i, layer in enumerate(layers):
        trainable = all(p.requires_grad for p in layer.parameters())
        if i == len(layers) - 1:
            assert trainable, f"blok terakhir (index {i}) seharusnya trainable"
        else:
            assert not any(p.requires_grad for p in layer.parameters()), \
                f"blok {i} seharusnya beku tapi ada parameter trainable"

    for p in model.encoder.text_model.parameters():
        assert not p.requires_grad, "text_model ikut trainable di last-layer FT"


def test_last_layer_ft_two_blocks_opens_last_two():
    model = _model()
    configure_last_layer_finetuning(model, n_blocks=2)
    layers = model.encoder.vision_model.encoder.layers
    for i, layer in enumerate(layers):
        trainable = any(p.requires_grad for p in layer.parameters())
        expected = i >= len(layers) - 2
        assert trainable == expected, f"blok {i}: trainable={trainable}, expected={expected}"


def test_last_layer_ft_rejects_n_blocks_out_of_range():
    model = _model()
    with pytest.raises(AssertionError, match="di luar rentang"):
        configure_last_layer_finetuning(model, n_blocks=N_LAYERS + 1)
    with pytest.raises(AssertionError, match="di luar rentang"):
        configure_last_layer_finetuning(model, n_blocks=0)


# --- build_llrd_optimizer ---

def test_llrd_optimizer_backbone_lr_much_smaller_than_head_lr():
    model = _model()
    configure_last_layer_finetuning(model, n_blocks=1)
    opt = build_llrd_optimizer(model, head_lr=1e-3, backbone_lr_ratio=1 / 50)

    lrs = {tuple(sorted(id(p) for p in g["params"])): g["lr"] for g in opt.param_groups}
    assert len(opt.param_groups) == 2
    head_group = next(g for g in opt.param_groups
                      if any(p is hp for hp in model.head.parameters() for p in g["params"]))
    backbone_group = next(g for g in opt.param_groups if g is not head_group)

    assert head_group["lr"] == pytest.approx(1e-3)
    assert backbone_group["lr"] == pytest.approx(1e-3 / 50)
    assert backbone_group["lr"] < head_group["lr"]


def test_llrd_optimizer_excludes_frozen_params():
    model = _model()
    configure_last_layer_finetuning(model, n_blocks=1)
    opt = build_llrd_optimizer(model, head_lr=1e-3)

    optimized_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert id(p) in optimized_ids, f"{n} trainable tapi tidak masuk optimizer"
        else:
            assert id(p) not in optimized_ids, f"{n} beku tapi masuk optimizer"


# --- hf_processor_to_data_config: konversi ke data_config generik (build_transforms) ---

class _SizeAttr:
    def __init__(self, height, width):
        self.height, self.width = height, width


class _FakeProcessorAttrStyle:
    """Meniru SizeDict asli (size.height/.width sbg atribut, bukan dict)."""
    def __init__(self):
        self.size = _SizeAttr(256, 256)
        self.image_mean = (0.5, 0.5, 0.5)
        self.image_std = (0.5, 0.5, 0.5)


class _FakeProcessorDictStyle:
    """Versi transformers lain bisa mengembalikan size sbg dict biasa."""
    def __init__(self):
        self.size = {"height": 384, "width": 384}
        self.image_mean = [0.5, 0.5, 0.5]
        self.image_std = [0.5, 0.5, 0.5]


def test_hf_processor_to_data_config_handles_attribute_style_size():
    cfg = hf_processor_to_data_config(_FakeProcessorAttrStyle())
    assert cfg["input_size"] == (3, 256, 256)
    assert cfg["mean"] == (0.5, 0.5, 0.5)
    assert cfg["std"] == (0.5, 0.5, 0.5)


def test_hf_processor_to_data_config_handles_dict_style_size():
    cfg = hf_processor_to_data_config(_FakeProcessorDictStyle())
    assert cfg["input_size"] == (3, 384, 384)


def test_hf_processor_to_data_config_output_is_usable_by_build_transforms():
    from transforms import build_transforms
    cfg = hf_processor_to_data_config(_FakeProcessorAttrStyle())
    tfm = build_transforms(cfg, train=False)   # tidak boleh raise
    assert tfm is not None


def test_llrd_optimizer_single_group_when_no_backbone_params_trainable():
    """Kalau cuma head yang trainable (mis. n_blocks belum dibuka / LoRA murni
    tanpa backbone terbuka), optimizer tidak boleh punya grup backbone kosong."""
    model = _model()
    freeze_all(model)
    for p in model.head.parameters():
        p.requires_grad = True
    opt = build_llrd_optimizer(model, head_lr=1e-3)
    assert len(opt.param_groups) == 1
