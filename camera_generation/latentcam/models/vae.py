import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraEncoder(nn.Module):
    def __init__(self, in_channels=9, hidden_dim=64, latent_dim=32):
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
        x: (B, T, 9)
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
    def __init__(self, latent_dim=32, hidden_dim=64, out_channels=9):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
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
            nn.Upsample(scale_factor=2, mode="nearest"),
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
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = CameraEncoder(latent_dim=latent_dim)
        self.decoder = CameraDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)        # (B, D, T')
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)             # (B, T, 9)
        return x_hat, mu, logvar
    
    def encode(self, x):
        mu, logvar = self.encoder(x)
        mu = mu.transpose(1, 2)
        return mu

    def decode(self, z):
        z = z.transpose(1, 2)
        x = self.decoder(z)
        return x
