import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraEncoder(nn.Module):
    def __init__(self, in_channels=11, hidden_dim=64, latent_dim=64):
        super().__init__()

        self.conv_init = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.down2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
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
    def __init__(self, latent_dim=64, hidden_dim=64, out_channels=11, up_mode="nearest"):
        super().__init__()
        # [R126] up_mode: "nearest"(기본, 기존 ckpt 와 동일) | "linear". nearest x2 두 번이 주기 4 톱니를 만든다
        #   (tmp/agent/reader-vae-latent-rf.md). linear 는 파라미터가 같아 state_dict 호환이다.
        _up = (lambda: nn.Upsample(scale_factor=2, mode="nearest")) if up_mode == "nearest" else \
              (lambda: nn.Upsample(scale_factor=2, mode="linear", align_corners=False))

        self.up1 = nn.Sequential(
            _up(),
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.up2 = nn.Sequential(
            _up(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.proj = nn.Conv1d(hidden_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, z):
        """
        z: (B, D, M+1)
        """
        x = self.up1(z)           # (B, H, 2M+2)
        x = x[..., :-1]           # (B, H, 2M+1)
        x = self.conv1(x) + x

        x = self.up2(x)           # (B, H, 4M+2)
        x = x[..., :-1]           # (B, H, 4M+1)
        x = self.conv2(x) + x

        x_hat = self.proj(x)      # (B, 9, 4M+1)
        return x_hat.transpose(1, 2)

class CameraVAE(nn.Module):
    def __init__(self, latent_dim=64, up_mode="nearest", pad_end=0):
        super().__init__()
        self.encoder = CameraEncoder(latent_dim=latent_dim)
        self.decoder = CameraDecoder(latent_dim=latent_dim, up_mode=up_mode)
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
