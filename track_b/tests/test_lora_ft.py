"""Test CPU-only, OFFLINE untuk lora_ft.py (Task 5). Fake encoder minimal --
detail wiring LoRA/last-layer sudah diuji tuntas di test_vision_classifier.py;
di sini fokus ke smoke_test_loop() (mekanika loop: overfit 1 batch, pola sama
sanity_overfit.py Fase 0) dan build_variant() (LR/optimizer sesuai B8)."""
import os
import sys

import pytest
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

pytest.importorskip("peft")

from lora_ft import (LAST_LAYER_BACKBONE_LR_RATIO, LAST_LAYER_HEAD_LR, LORA_LR,
                     SMOKE_LR, build_variant, smoke_test_loop)


class _Output:
    def __init__(self, pooler_output):
        self.pooler_output = pooler_output


class _FakeLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d)
        self.self_attn.k_proj = nn.Linear(d, d)
        self.self_attn.v_proj = nn.Linear(d, d)
        self.self_attn.out_proj = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x):
        a = self.self_attn.q_proj(x) + self.self_attn.v_proj(x) + self.self_attn.k_proj(x)
        return self.mlp(self.self_attn.out_proj(a))


class _FakeTower(nn.Module):
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
    def __init__(self, d=16, n_layers=4):
        super().__init__()
        self.vision_model = _FakeTower(d, n_layers)
        self.text_model = _FakeTower(d, n_layers)


D, N_LAYERS = 16, 4


def _encoder_factory():
    return lambda: _FakeSiglipModel(D, N_LAYERS)


# --- build_variant: LR/optimizer sesuai parameter awal B8 ---

def test_build_variant_lora_uses_lora_lr_from_b8():
    _, optimizer = build_variant("lora", _FakeSiglipModel(D, N_LAYERS), hidden_size=D)
    assert all(g["lr"] == pytest.approx(LORA_LR) for g in optimizer.param_groups)


def test_build_variant_last_layer_uses_llrd_from_b8():
    _, optimizer = build_variant("last_layer", _FakeSiglipModel(D, N_LAYERS), hidden_size=D)
    lrs = sorted(g["lr"] for g in optimizer.param_groups)
    assert lrs[-1] == pytest.approx(LAST_LAYER_HEAD_LR)
    assert lrs[0] == pytest.approx(LAST_LAYER_HEAD_LR * LAST_LAYER_BACKBONE_LR_RATIO)


def test_build_variant_rejects_unknown_variant():
    with pytest.raises(KeyError, match="tidak dikenal"):
        build_variant("full_finetune", _FakeSiglipModel(D, N_LAYERS), hidden_size=D)


def test_build_variant_lr_override_does_not_change_llrd_ratio():
    """Override LR (dipakai smoke test) hanya menggeser SKALA, rasio LLRD
    head:backbone tetap sesuai B8 -- kalau rasio ikut berubah, smoke test
    tidak lagi menguji konfigurasi yang sama dgn produksi."""
    _, opt = build_variant("last_layer", _FakeSiglipModel(D, N_LAYERS),
                           hidden_size=D, lr=1e-2)
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[-1] == pytest.approx(1e-2)
    assert lrs[0] / lrs[-1] == pytest.approx(LAST_LAYER_BACKBONE_LR_RATIO)


# --- smoke_test_loop: mekanika loop (overfit 1 batch), pola sanity_overfit.py ---

@pytest.mark.parametrize("variant", ["lora", "last_layer"])
def test_smoke_test_loop_overfits_and_returns_low_loss(variant):
    final_loss = smoke_test_loop(variant, _encoder_factory(), hidden_size=D,
                                 n_steps=200, device="cpu", target_loss=0.2)
    assert final_loss < 0.2


def test_smoke_test_loop_raises_when_target_unreachable():
    """Bukti assert bekerja: target mustahil (1e-6) dlm langkah sedikit ->
    HARUS gagal keras, bukan diam-diam 'lolos' dgn loss masih tinggi."""
    with pytest.raises(AssertionError, match="gagal overfit"):
        smoke_test_loop("lora", _encoder_factory(), hidden_size=D,
                        n_steps=3, device="cpu", target_loss=1e-6)


@pytest.mark.parametrize("variant", ["lora", "last_layer"])
def test_smoke_lr_is_deliberately_larger_than_production_lr(variant):
    """Kunci temuan diagnosis: LR produksi B8 (kecil, melindungi representasi
    pretrained) TIDAK cukup untuk memaksa hafal 1 batch dalam langkah wajar.
    Smoke test WAJIB pakai LR sendiri; kalau suatu saat SMOKE_LR diturunkan
    jadi = LR produksi, test ini gagal dan mengingatkan kenapa."""
    production_lr = LORA_LR if variant == "lora" else LAST_LAYER_HEAD_LR
    assert SMOKE_LR[variant] > production_lr


def test_production_lr_cannot_overfit_in_few_steps_that_is_why_smoke_lr_exists():
    """Bukti langsung (bukan klaim di komentar): dgn LR produksi B8, 200 step
    TIDAK cukup menurunkan loss ke target -- inilah alasan SMOKE_LR ada."""
    with pytest.raises(AssertionError, match="gagal overfit"):
        smoke_test_loop("lora", _encoder_factory(), hidden_size=D, n_steps=200,
                        device="cpu", target_loss=0.2, lr=LORA_LR)
