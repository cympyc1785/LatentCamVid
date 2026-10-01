"""[R136, 보류] 인과 레이아웃 카메라 VAE — CamVLA `cam_causal_vae_v1.py` / WAN `CausalConv3d` 식.
사용자 2026-10-01: "인과 레이아웃은 일단 차순위로 미뤄두고 … causal은 따로 파일 만들어서 … 나중에 내 지시에 따라 진행".
models/vae_intr_large.py 의 R136 판 (replicate / subpixel 포함) 에 causal 옵션을 더한 것. 학습·eval 배선은 아직 없다.
  causal=True: conv padding 을 왼쪽에만 k-1, 토큰 k ↔ 프레임 4k-3..4k (토큰 0 = 프레임 0), 업샘플 뒤 앞 1개 crop.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TConv1d(nn.Conv1d):
    """[R136] 시간축 padding 을 직접 붙이는 Conv1d. causal=True 면 왼쪽(과거)에만 k-1 (CamVLA cam_causal_vae_v1 / WAN CausalConv3d),
    아니면 양쪽 (k-1)/2. pad_mode "replicate" 는 끝 프레임 복제 (zero 는 rot6d=0 같은 존재하지 않는 카메라를 붙인다).
    파라미터 이름은 nn.Conv1d 와 같다."""

    def __init__(self, cin, cout, k, stride=1, causal=False, pad_mode="zeros"):
        super().__init__(cin, cout, k, stride=stride, padding=0)
        self.lp, self.rp = (k - 1, 0) if causal else ((k - 1) // 2, (k - 1) // 2)
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, (self.lp, self.rp), mode="replicate" if self.pad_mode == "replicate" else "constant")
        return super().forward(x)


def make_conv(causal=False, pad_mode="zeros"):
    """기본 (False, zeros) 은 기존과 똑같은 nn.Conv1d(padding=(k-1)/2) — 기존 ckpt 와 수치 동일."""
    if not causal and pad_mode == "zeros":
        return lambda cin, cout, k, stride=1: nn.Conv1d(cin, cout, k, stride=stride, padding=(k - 1) // 2)
    return lambda cin, cout, k, stride=1: TConv1d(cin, cout, k, stride=stride, causal=causal, pad_mode=pad_mode)


class SubpixelUp(nn.Module):
    """[R136] WAN Resample(upsample3d) 식 학습 업샘플: conv 로 채널 2배 → 시간축 2칸으로 interleave (depth-to-time)."""

    def __init__(self, cin, cout, conv):
        super().__init__()
        self.conv = conv(cin, 2 * cout, 3)
        self.cout = cout

    def forward(self, x):
        b, _, t = x.shape
        y = self.conv(x).view(b, 2, self.cout, t)
        return torch.stack((y[:, 0], y[:, 1]), dim=-1).reshape(b, self.cout, 2 * t)


class CameraEncoder(nn.Module):
    def __init__(self, in_channels=11, hidden_dim=64, latent_dim=64, causal=False, pad_mode="zeros"):
        super().__init__()
        C = make_conv(causal, pad_mode)

        self.conv_init = nn.Sequential(
            C(in_channels, hidden_dim, 3),
            nn.ReLU(),
        )

        self.down1 = nn.Sequential(
            C(hidden_dim, hidden_dim, 3, 2),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            C(hidden_dim, hidden_dim, 3, 2),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
        )

        self.to_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B, T, 11)
        return:
            mu, logvar: (B, D, T')
        """
        x = x.transpose(1, 2)   # (B, 9, T)
        x = self.conv_init(x)
        x = self.down1(x)
        x = self.conv1(x) + x
        x = self.down2(x)
        h = self.conv2(x) + x
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

class CameraDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=64, out_channels=11, up_mode="nearest", causal=False, pad_mode="zeros"):
        super().__init__()
        C = make_conv(causal, pad_mode)
        # [R136] causal=True: 토큰 k ↔ 프레임 4k-3..4k (토큰 0 = 프레임 0). 업샘플 뒤 **앞** 1개를 자른다 (CamVLA causal_upsample2x).
        #   linear 는 L → 2L-1 align_corners=True (프레임 2k = 토큰 k), subpixel 은 SubpixelUp.
        self.causal, self.up_mode = causal, up_mode
        # [R126] up_mode: "nearest"(기본, 기존 ckpt 와 동일) | "linear". nearest x2 두 번이 주기 4 톱니를 만든다
        #   (tmp/agent/reader-vae-latent-rf.md). linear 는 파라미터가 같아 state_dict 호환이다.
        _up = (lambda: nn.Upsample(scale_factor=2, mode="nearest")) if up_mode == "nearest" else \
              (lambda: nn.Upsample(scale_factor=2, mode="linear", align_corners=False))
        if causal and up_mode == "linear":
            _up = lambda: nn.Identity()               # forward 에서 size=2L-1 align_corners=True 로 직접 보간
        if up_mode == "subpixel":
            self.up1 = nn.Sequential(SubpixelUp(latent_dim, hidden_dim, C), nn.ReLU())
        else:
            self.up1 = nn.Sequential(
                _up(),
                C(latent_dim, hidden_dim, 3),
                nn.ReLU()
            )

        self.conv1 = nn.Sequential(
            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        if up_mode == "subpixel":
            self.up2 = nn.Sequential(SubpixelUp(hidden_dim, hidden_dim, C), nn.ReLU())
        else:
            self.up2 = nn.Sequential(
                _up(),
                C(hidden_dim, hidden_dim, 3),
                nn.ReLU()
            )

        self.conv2 = nn.Sequential(
            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),

            C(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        self.proj = C(hidden_dim, out_channels, 3)

    def forward(self, z):
        """
        z: (B, D, M+1)
        """
        x = self._up(self.up1, z)     # (B, H, 2M+1)
        x = self.conv1(x) + x

        x = self._up(self.up2, x)     # (B, H, 4M+1)
        x = self.conv2(x) + x

        x_hat = self.proj(x)      # (B, 9, 4M+1)
        return x_hat.transpose(1, 2)

    def _up(self, block, x):
        """L → 2L-1. 기존(비인과): 2배 뒤 끝 1개 crop. causal: 앞 1개 crop (토큰 0 = 프레임 0), linear 는 직접 보간."""
        if self.causal and self.up_mode == "linear":
            x = F.interpolate(x, size=2 * x.shape[-1] - 1, mode="linear", align_corners=True)
            return block(x)
        x = block(x)                                  # (B, H, 2L)
        return x[..., 1:] if self.causal else x[..., :-1]


class CameraVAE(nn.Module):
    def __init__(self, latent_dim=64, up_mode="nearest", pad_end=0, causal=False, pad_mode="zeros"):
        super().__init__()
        # [R136] causal / pad_mode 기본 (False, zeros) = 기존 구조. up_mode 에 "subpixel" (WAN depth-to-time) 추가.
        self.encoder = CameraEncoder(latent_dim=latent_dim, causal=causal, pad_mode=pad_mode)
        self.decoder = CameraDecoder(latent_dim=latent_dim, up_mode=up_mode, causal=causal, pad_mode=pad_mode)
        # [R126] pad_end>0: 입력 끝을 마지막 프레임 복제로 pad_end 만큼 늘려 encode 하고 decode 뒤 잘라낸다.
        #   마지막 토큰이 경계에 걸려 이득이 절반인 문제 (enc 51.5 vs ~120, dec 0.84 vs ~1.6) 완화용.
        #   49+4=53=4*13+1 → 토큰 14개. 기본 0 = 기존 동작 (토큰 13).
        self.pad_end = int(pad_end)

    def _pad(self, x):
        if self.pad_end <= 0:
            return x
        return torch.cat([x, x[:, -1:].expand(-1, self.pad_end, -1)], dim=1)

    def _crop(self, x):
        return x[:, :-self.pad_end] if self.pad_end > 0 else x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(self._pad(x))        # (B, D, T')
        z = self.reparameterize(mu, logvar)
        x_hat = self._crop(self.decoder(z))            # (B, T, 11)
        return x_hat, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(self._pad(x))
        mu = mu.transpose(1, 2)
        return mu
    
    def encode_sample(self, x):
        mu, logvar = self.encoder(self._pad(x))        # (B, D, T')
        z = self.reparameterize(mu, logvar)
        z = z.transpose(1, 2)
        return z

    def decode(self, z):
        z = z.transpose(1, 2)
        x = self._crop(self.decoder(z))
        return x
