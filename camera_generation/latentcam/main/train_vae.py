import torch
import torch.nn.functional as F
import numpy as np
import os
import datetime

from config_vae import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.models.vae_intr_large import CameraVAE
from accelerate import Accelerator
from accelerate.utils import tqdm, broadcast_object_list

torch.autograd.set_detect_anomaly(True)

def camera_vae_loss(x, x_hat, mu, logvar, beta=cfg.vae_beta):
    """
    x, x_hat: (B, 9)
    """
    recon_loss = torch.abs(x - x_hat).mean()

    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train():
    # hyperparams
    accelerator = Accelerator(log_with="wandb", mixed_precision="bf16")
    device = accelerator.device
    if accelerator.is_main_process:
        exp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        exp_name = None
    exp_name = broadcast_object_list([exp_name])[0]
    result_dir = os.path.join('../results_vae', exp_name)
    os.makedirs(result_dir, exist_ok=True)
    ckpt_dir = os.path.join(result_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    accelerator.init_trackers(
        project_name="my-vae-training", 
        config=cfg_dict,
        init_kwargs={
            "wandb": {
                "name": f"{exp_name}"
            }
        }
    )

    trainer = Trainer(cfg=cfg)
    trainer._make_model()
    trainer._make_batch_generator(include_val=False)

    # model
    model = CameraVAE()
    model.train()
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    train_dataloader = trainer.batch_generator

    model, opt, train_dataloader = accelerator.prepare(
        model, opt, trainer.batch_generator
    )
    global_step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_dataloader)
        total_loss = torch.tensor(0.0, device=device)
        total_recon_loss = torch.tensor(0.0, device=device)
        total_kl_loss = torch.tensor(0.0, device=device)
        total_samples = torch.tensor(0, device=device)
        for data in pbar:
            data_name = data['data_name']
            traj = data['cam_param'].to(device)
            B = traj.shape[0]
            traj_pred, mu, logvar = model(traj)
            loss, recon_loss, kl_loss = camera_vae_loss(traj, traj_pred, mu, logvar)
            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()
            
            global_step += 1

            accelerator.log({
                "train/loss_iter": loss.item(),
                "train/recon_loss_iter": recon_loss.item(),
                "train/kl_loss_iter": kl_loss.item()
            }, step=global_step)

            total_loss += loss.detach() * B
            total_recon_loss += recon_loss.detach() * B
            total_kl_loss += kl_loss.detach() * B
            total_samples += B

            pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.4f}")
        total_loss = accelerator.reduce(total_loss, reduction='sum')
        total_recon_loss = accelerator.reduce(total_recon_loss, reduction='sum')
        total_kl_loss = accelerator.reduce(total_kl_loss, reduction='sum')
        total_samples = accelerator.reduce(total_samples, reduction='sum')

        accelerator.log(
            {"train/epoch_loss": total_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch_recon_loss": total_recon_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch_kl_loss": total_kl_loss / total_samples},
            step=global_step
        )
        accelerator.log(
            {"train/epoch": epoch + 1},
            step=global_step
        )

        # sampling each epoch
        if epoch % cfg.save_epoch == 0:
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), os.path.join(ckpt_dir, f"{epoch}.pth"))
            
    print("Training done.")

if __name__ == "__main__":
    train()
