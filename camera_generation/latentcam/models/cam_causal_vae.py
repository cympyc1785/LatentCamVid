import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """시간축의 과거(왼쪽) 방향에만 padding을 붙이는 1D conv.

    padding = kernel_size - 1 을 왼쪽에만 붙이므로:
      - stride=1 : 길이 유지 (L -> L)
      - stride=2 : 2k+1 -> k+1  (첫 출력 토큰은 첫 입력 프레임만 참조)
    padding은 첫 프레임을 복제(replicate)해서 채운다.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.time_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        x = F.pad(x, (self.time_pad, 0), mode="replicate")
        return self.conv(x)


def causal_upsample2x(x):
    """(B, C, k+1) -> (B, C, 2k+1)

    각 토큰을 2번씩 복제한 뒤 맨 앞 1개를 잘라내서,
    첫 토큰만 1프레임, 나머지 토큰은 2프레임에 대응되도록 한다.
    [z0, z1, z2] -> [z0, z0, z1, z1, z2, z2] -> [z0, z1, z1, z2, z2]
    """
    x = x.repeat_interleave(2, dim=-1)
    return x[..., 1:]


class CameraEncoder(nn.Module):
    def __init__(self, in_channels=11, hidden_dim=64, latent_dim=64):
        super().__init__()

        self.conv_init = nn.Sequential(
            CausalConv1d(in_channels, hidden_dim, 3),
            nn.ReLU(),
        )

        # 4N+1 -> 2N+1
        self.down1 = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3, stride=2),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        # 2N+1 -> N+1
        self.down2 = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3, stride=2),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
        )

        self.to_mu = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        self.to_logvar = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x):
        """
        x: (B, T, C_in),  T = 4N+1
        return:
            mu, logvar: (B, D, N+1)
        """
        x = x.transpose(1, 2)       # (B, C_in, 4N+1)
        x = self.conv_init(x)
        x = self.down1(x)           # (B, H, 2N+1)
        x = self.conv1(x) + x
        x = self.down2(x)           # (B, H, N+1)
        h = self.conv2(x) + x
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar


class CameraDecoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=64, out_channels=11):
        super().__init__()

        self.conv_in = nn.Sequential(
            CausalConv1d(latent_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        self.conv1 = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        self.conv_mid = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
            CausalConv1d(hidden_dim, hidden_dim, 3),
            nn.ReLU(),
        )

        self.proj = CausalConv1d(hidden_dim, out_channels, 3)

    def forward(self, z):
        """
        z: (B, D, N+1)
        return: (B, 4N+1, C_out)
        """
        x = self.conv_in(z)             # (B, H, N+1)

        x = causal_upsample2x(x)        # (B, H, 2N+1)
        x = self.conv1(x) + x

        x = self.conv_mid(x)
        x = causal_upsample2x(x)        # (B, H, 4N+1)
        x = self.conv2(x) + x

        x_hat = self.proj(x)            # (B, C_out, 4N+1)
        return x_hat.transpose(1, 2)


class CameraVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.in_dim = cfg.in_dim
        self.out_dim = cfg.out_dim
        self.hidden_dim = cfg.hidden_size
        self.latent_dim = cfg.latent_dim
        self.scale_factor = cfg.scale_factor

        self.encoder = CameraEncoder(self.in_dim, self.hidden_dim, self.latent_dim)
        self.decoder = CameraDecoder(self.latent_dim, self.hidden_dim, self.out_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)        # (B, D, N+1)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)             # (B, 4N+1, C_out)
        return x_hat, mu, logvar

    def encode(self, x):
        mu, logvar = self.encoder(x)
        mu = mu.transpose(1, 2)             # (B, N+1, D)
        return mu

    def encode_sample(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = z.transpose(1, 2)
        return z

    def decode(self, z):
        z = z.transpose(1, 2)
        x = self.decoder(z)
        return x