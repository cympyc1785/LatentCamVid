"""Per-token tau (flow time) sampler for Diffusion-Forcing training on the causal-VAE latent
(W = number of latent tokens, 13 for 49-frame clips). Returns tau in [0,1] per token and a
pattern label per sample (0=iid, 1=ramp, 2=bootup, 3=full-sequence).

Mixture (per sample):
  (a) 30% iid uniform(0,1)
  (b) 40% monotone ramp: slope=U(0.5,2)/W, offset=U(-0.5,1), +N(0,0.02), clip[0,1]
  (c) 15% boot-up: first k=randint(1,6) tokens = 0, rest = ramp (offset=U(0,0.3))
  (d) 15% full-sequence: all tokens = one U(0,1) scalar
"""
import torch

PATTERNS = {0: 'iid', 1: 'ramp', 2: 'bootup', 3: 'full'}


def _ramp(W, device, offset_lo, offset_hi):
    slope = (torch.rand(1, device=device) * 1.5 + 0.5) / W            # U(0.5,2)/W
    offset = torch.rand(1, device=device) * (offset_hi - offset_lo) + offset_lo
    tau = offset + slope * torch.arange(W, device=device).float()
    tau = tau.clamp(0, 1) + torch.randn(W, device=device) * 0.02
    return tau.clamp(0, 1)


def sample_tau(B, W, device):
    """-> (tau [B,W] in [0,1], labels [B] long)."""
    tau = torch.zeros(B, W, device=device)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    r = torch.rand(B, device=device)
    for b in range(B):
        u = r[b].item()
        if u < 0.30:                                   # (a) iid
            tau[b] = torch.rand(W, device=device); labels[b] = 0
        elif u < 0.70:                                 # (b) ramp
            tau[b] = _ramp(W, device, -0.5, 1.0); labels[b] = 1
        elif u < 0.85:                                 # (c) boot-up
            k = int(torch.randint(1, 6, (1,)).item())
            t = _ramp(W, device, 0.0, 0.3)
            t[:k] = 0.0
            tau[b] = t; labels[b] = 2
        else:                                          # (d) full-sequence
            tau[b] = torch.rand(1, device=device).expand(W); labels[b] = 3
    return tau, labels
