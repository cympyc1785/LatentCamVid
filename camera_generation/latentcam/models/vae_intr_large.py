import torch
import torch.nn as nn
import torch.nn.functional as F

class TConv1d(nn.Conv1d):
    """[R136] 시간축 양쪽에 (k-1)/2 padding 을 직접 붙이는 Conv1d. pad_mode "replicate" 는 끝 프레임 복제
    (zero 는 rot6d=0 같은 존재하지 않는 카메라를 붙인다). 파라미터 이름은 nn.Conv1d 와 같다.
    인과(causal) 판은 models/vae_causal.py (보류)."""

    def __init__(self, cin, cout, k, stride=1, pad_mode="zeros"):
        super().__init__(cin, cout, k, stride=stride, padding=0)
        self.lp, self.rp = (k - 1) // 2, (k - 1) // 2
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, (self.lp, self.rp), mode="replicate" if self.pad_mode == "replicate" else "constant")
        return super().forward(x)


def make_conv(pad_mode="zeros"):
    """기본 zeros 는 기존과 똑같은 nn.Conv1d(padding=(k-1)/2) — 기존 ckpt 와 수치 동일."""
    if pad_mode == "zeros":
        return lambda cin, cout, k, stride=1: nn.Conv1d(cin, cout, k, stride=stride, padding=(k - 1) // 2)
    return lambda cin, cout, k, stride=1: TConv1d(cin, cout, k, stride=stride, pad_mode=pad_mode)


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


class LinearUp2x(nn.Module):
    """[R140] nn.Upsample(scale_factor=2, mode="linear", align_corners=False) 와 같은 값을 내는 직접 구현.
    PyTorch 의 upsample_linear1d CUDA backward 가 (B·C 큰, T 짧은) 이 모양에서 ~100배 느려 (43.6 ms vs nearest 0.41 ms) 학습 step 이 9배 느렸다.
    align_corners=False ×2: out[2i] = 0.75·x[i] + 0.25·x[i-1], out[2i+1] = 0.75·x[i] + 0.25·x[i+1] (경계는 복제). 파라미터 없음 → ckpt 호환."""

    def forward(self, x):
        prv = torch.cat([x[..., :1], x[..., :-1]], dim=-1)
        nxt = torch.cat([x[..., 1:], x[..., -1:]], dim=-1)
        return torch.stack((0.75 * x + 0.25 * prv, 0.75 * x + 0.25 * nxt), dim=-1).flatten(-2)


class CameraEncoder(nn.Module):
    def __init__(self, in_channels=11, hidden_dim=64, latent_dim=64, pad_mode="zeros"):
        super().__init__()
        C = make_conv(pad_mode)

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
    def __init__(self, latent_dim=64, hidden_dim=64, out_channels=11, up_mode="nearest", pad_mode="zeros"):
        super().__init__()
        C = make_conv(pad_mode)
        # [R136] up_mode "subpixel" = WAN 식 depth-to-time (SubpixelUp). 세 모드 모두 2배 뒤 끝 1개 crop (L → 2L-1).
        self.up_mode = up_mode
        # [R126] up_mode: "nearest"(기본, 기존 ckpt 와 동일) | "linear". nearest x2 두 번이 주기 4 톱니를 만든다
        #   (tmp/agent/reader-vae-latent-rf.md). linear 는 파라미터가 같아 state_dict 호환이다.
        _up = (lambda: nn.Upsample(scale_factor=2, mode="nearest")) if up_mode == "nearest" else \
              (lambda: LinearUp2x())                    # [R140] = nn.Upsample(linear, align_corners=False), backward 가 빠른 판
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
        """L → 2L-1: 2배 뒤 끝 1개 crop (기존과 같음)."""
        return block(x)[..., :-1]


class CameraVAE(nn.Module):
    def __init__(self, latent_dim=64, up_mode="nearest", pad_end=0, pad_mode="zeros"):
        super().__init__()
        # [R136] pad_mode 기본 zeros = 기존 구조. up_mode 에 "subpixel" (WAN depth-to-time) 추가.
        self.encoder = CameraEncoder(latent_dim=latent_dim, pad_mode=pad_mode)
        self.decoder = CameraDecoder(latent_dim=latent_dim, up_mode=up_mode, pad_mode=pad_mode)
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
