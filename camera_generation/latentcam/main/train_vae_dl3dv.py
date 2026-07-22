"""Re-fit the camera VAE (vae_intr_large.CameraVAE) on DL3DV-960 via base.Trainer, using the
dataset's cam_param under the configured scale_mode (e.g. 'geo_lagernvs' full-alignment).
Mirrors train_vae.py's loss but on the DL3DV-960 dataloader. Saves ckpts + prints the latent
std (use as vae_latent_scale for the generation model).

Run: CUDA_VISIBLE_DEVICES=3 LATENTCAM_CONFIG=config_vae_dl3dv PYTHONPATH=..:. \
     python train_vae_dl3dv.py
"""
import os, importlib
import torch, torch.nn.functional as F
from accelerate import Accelerator

from hydra_cfg import load_cfg
cfg, cfg_dict = load_cfg('config_vae_dl3dv')

from base import Trainer
from models.vae_intr_large import CameraVAE


def vae_loss(x, x_hat, mu, logvar, beta):
    recon = torch.abs(x - x_hat).mean()
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon, kl


def main():
    # fp32 (mixed_precision off): the VAE KL term (logvar.exp / reparam std) is unstable in
    # bf16 -> NaN. VAE is tiny so fp32 is cheap.
    acc = Accelerator(log_with="wandb", mixed_precision="no")
    device = acc.device
    ckpt_dir = os.path.join(cfg.root_dir, 'my_checkpoints', cfg.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    if acc.is_main_process:
        acc.init_trackers("camera-diffusion-training", config=cfg_dict,
                          init_kwargs={"wandb": {"name": cfg.exp_name}})
        from hydra_cfg import save_cfg_yaml
        save_cfg_yaml(cfg_dict, os.path.join(ckpt_dir, 'config.yaml'))
        try:
            import wandb as _wb
            save_cfg_yaml(cfg_dict, os.path.join(_wb.run.dir, 'config.yaml'))
        except Exception:
            pass

    trainer = Trainer(cfg); trainer._make_batch_generator(include_val=False)
    model = CameraVAE(latent_dim=cfg.cam_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model, opt, loader = acc.prepare(model, opt, trainer.batch_generator)

    gstep = 0
    for epoch in range(cfg.epochs):
        model.train(); tot = torch.tensor(0.0, device=device); n = torch.tensor(0, device=device)
        lat_sum = torch.tensor(0.0, device=device); lat_sq = torch.tensor(0.0, device=device); lat_n = torch.tensor(0, device=device)
        from accelerate.utils import tqdm
        for data in tqdm(loader, disable=not acc.is_main_process):
            traj = data['cam_param'].to(device)               # (B,49,11)
            x_hat, mu, logvar = model(traj)
            loss, recon, kl = vae_loss(traj, x_hat, mu, logvar, cfg.vae_beta)
            if not torch.isfinite(loss):          # skip any pathological batch
                opt.zero_grad(); continue
            opt.zero_grad(); acc.backward(loss)
            acc.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            B = traj.shape[0]; tot += loss.detach()*B; n += B
            with torch.no_grad():
                z = acc.unwrap_model(model).encode(traj)      # (B,T',D)
                lat_sum += z.sum(); lat_sq += (z*z).sum(); lat_n += z.numel()
            if acc.is_main_process:
                acc.log({"train/loss": loss.item(), "train/recon": recon.item(), "train/kl": kl.item()}, step=gstep)
            gstep += 1
        tot = acc.reduce(tot, 'sum'); n = acc.reduce(n, 'sum')
        std = (acc.reduce(lat_sq,'sum')/acc.reduce(lat_n,'sum') - (acc.reduce(lat_sum,'sum')/acc.reduce(lat_n,'sum'))**2).clamp(min=0).sqrt()
        if acc.is_main_process:
            acc.log({"train/epoch_loss": (tot/n).item(), "train/latent_std": std.item()}, step=gstep)
            print(f"epoch {epoch} loss={(tot/n).item():.5f} latent_std={std.item():.6f}")
            if (epoch+1) % 20 == 0 or epoch == cfg.epochs-1:
                torch.save(acc.unwrap_model(model).state_dict(), os.path.join(ckpt_dir, f"{epoch+1}.pth"))
                torch.save(acc.unwrap_model(model).state_dict(), os.path.join(ckpt_dir, "last.pth"))
                with open(os.path.join(ckpt_dir, "latent_std.txt"), "w") as f:
                    f.write(f"epoch {epoch+1} latent_std {std.item():.6f}\n")
    acc.end_training()


if __name__ == '__main__':
    main()
