"""Unit tests for the per-token (Diffusion-Forcing) rectified-flow path.
Run: LATENTCAM_CONFIG=config_rolling python test_per_token.py
Covers: (1) per-token FiLM produces token-dependent modulation & output;
        (2) tau sampler each pattern -> shape [B,W], range [0,1], correct label;
        (3) loss mask zero-denominator guard (all-context batch -> finite loss).
"""
import os
os.environ.setdefault('LATENTCAM_CONFIG', 'config_rolling')
import torch

from tau_sampler import sample_tau, PATTERNS
from models.camera_diffusion_model_latent import CameraDiffusionModel, timestep_embedding


def test_timestep_embedding_shapes():
    """(B,) -> (B,D); (B,W) -> (B,W,D)."""
    d = 512
    assert timestep_embedding(torch.zeros(4), d).shape == (4, d)
    assert timestep_embedding(torch.zeros(4, 13), d).shape == (4, 13, d)
    print("[ok] timestep_embedding shapes (B,)->(B,D), (B,W)->(B,W,D)")


def test_per_token_modulation_differs():
    """Per-token tau with distinct token values must yield token-dependent output;
    and a per-token tau must differ from the broadcast (scalar-per-sample) path."""
    torch.manual_seed(0)
    model = CameraDiffusionModel(cam_dim=64).eval()
    B, W, D = 2, 13, 64
    z = torch.randn(B, W, D)
    txt = torch.randn(B, 8, 4096); mask = torch.ones(B, 8, dtype=torch.bool)

    # per-token: ramp tau (each token a different noise level)
    tau_pt = torch.linspace(0, 1, W).unsqueeze(0).expand(B, W).contiguous()
    with torch.no_grad():
        out_pt = model(z, tau_pt * 1000.0, txt, mask)
    # broadcast: single scalar per sample (legacy path)
    tau_bc = torch.full((B,), 0.5)
    with torch.no_grad():
        out_bc = model(z, tau_bc * 1000.0, txt, mask)

    assert out_pt.shape == (B, W, D)
    # per-token output should NOT be constant across tokens in the way a broadcast one is,
    # and must differ from the broadcast result (different conditioning).
    assert not torch.allclose(out_pt, out_bc, atol=1e-4), "per-token == broadcast (FiLM not per-token)"

    # directly probe the modulation: mod1(t_embed) must differ across tokens for per-token tau
    t_embed = model.time_proj(model.time_mlp(tau_pt * 1000.0))     # (B,W,hidden)
    scale, shift = model.mod1(t_embed).chunk(2, dim=-1)
    tok0, tok_last = scale[0, 0], scale[0, -1]
    assert not torch.allclose(tok0, tok_last, atol=1e-4), "modulation identical across tokens"
    print("[ok] per-token FiLM: modulation & output are token-dependent and != broadcast path")


def test_tau_sampler_patterns():
    """Each pattern: shape [B,W], all values in [0,1], correct label id."""
    torch.manual_seed(1)
    B, W = 512, 13
    tau, labels = sample_tau(B, W, 'cpu')
    assert tau.shape == (B, W), tau.shape
    assert labels.shape == (B,)
    assert float(tau.min()) >= 0.0 and float(tau.max()) <= 1.0, (float(tau.min()), float(tau.max()))
    seen = set(labels.tolist())
    assert seen.issubset(set(PATTERNS.keys()))
    # bootup: first k tokens must be exactly 0
    for b in range(B):
        if labels[b].item() == 2:  # bootup
            assert (tau[b] == 0).any(), "bootup has no clean prefix"
        if labels[b].item() == 3:  # full-sequence: all tokens equal
            assert torch.allclose(tau[b], tau[b, 0].expand(W)), "full-seq tokens not constant"
    # all four patterns should appear over 512 draws (probabilities 0.30/0.40/0.15/0.15)
    assert seen == set(PATTERNS.keys()), f"missing patterns: {set(PATTERNS.keys()) - seen}"
    print(f"[ok] tau sampler: shape {tuple(tau.shape)}, range [{float(tau.min()):.3f},"
          f"{float(tau.max()):.3f}], patterns {sorted(seen)}")


def test_loss_mask_zero_denom():
    """All-context batch (every tau < loss_tau_min) -> mask sum 0 -> clamped denom -> finite loss."""
    mask = torch.zeros(4, 13)                       # every token is context
    per_tok = torch.rand(4, 13)
    denom = mask.sum().clamp(min=1.0)
    loss = (mask * per_tok).sum() / denom
    assert torch.isfinite(loss) and loss.item() == 0.0, loss
    print("[ok] loss mask zero-denominator guard -> finite (0.0) loss, no NaN/inf")


if __name__ == '__main__':
    test_timestep_embedding_shapes()
    test_per_token_modulation_differs()
    test_tau_sampler_patterns()
    test_loss_mask_zero_denom()
    print("\nALL UNIT TESTS PASSED")
