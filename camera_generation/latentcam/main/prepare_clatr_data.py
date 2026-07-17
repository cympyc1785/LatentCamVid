import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import wandb
import datetime
import time
import json
from pathlib import Path
import random

from config_large import cfg, cfg_dict
from core_pkg.common.base import Trainer
from core_pkg.models.pc_encoder import PCEncoder
from core_pkg.models.vae_intr import CameraVAE
from core_pkg.models.t5 import T5EncoderModel
from core_pkg.common.utils.data_utils import inverse_camera_matrix
from core_pkg.common.utils.eval_utils import run_command_in_dir
from diffusers import DDPMScheduler, DDIMScheduler
from dataset_seg import CamDataset
from torch.utils.data import DataLoader

from evaluate.CLaTr.clip_extraction import load_clip_model, encode_text, save_feats_custom

def train():
    device = 'cuda:0'

    # For evaluation
    clip_model = load_clip_model(cfg.clip_version, device=device)
    trainset_loader = CamDataset(cfg=cfg)
    batch_generator = DataLoader(dataset=trainset_loader, \
                            batch_size=1, \
                            shuffle=False, num_workers=cfg.num_thread, \
                            persistent_workers=True)
    print("Total data cnt: ", trainset_loader.__len__())
    result_dir = '/home/ckd248/data/SCVideo/camera_generation/dataset/clatr_large'
    os.makedirs(result_dir, exist_ok=True)
    train_data_dir = os.path.join(result_dir, 'train')
    os.makedirs(train_data_dir, exist_ok=True)
    test_data_dir = os.path.join(result_dir, 'test')
    os.makedirs(test_data_dir, exist_ok=True)
    train_data_name_list = []
    test_data_name_list = []
    error_data_name_list = []
    for epoch in range(1):
        pbar = tqdm(batch_generator)
        for step, data in enumerate(pbar):
            if random.random() < 0.95:
                data_dir = train_data_dir
            else:
                data_dir = test_data_dir
            # t1 = time.time()
            data_name = data['data_name']
            text_prompt = data['text_prompt']
            traj = data['cam_param'].to(device)
            intrinsics = data['intrinsics']

            matrix_traj_ref = inverse_camera_matrix(traj)
            if (torch.abs(matrix_traj_ref) >= 1e2).any():
                error_data_name_list += data_name
                continue
            matrix_traj_ref[:, :, :3, 1:3] *= -1
            ref_intrinsics = intrinsics
            matrix_traj_ref = matrix_traj_ref.cpu().tolist()
            ref_intrinsics = intrinsics.cpu().tolist()

            seq_embeds, tok_embeds = encode_text(text_prompt, clip_model, max_token_length=None, device=device)
            save_feats_custom(seq_embeds, data_name, Path(os.path.join(result_dir, 'seq')), data_type=os.path.basename(data_dir))
            save_feats_custom(tok_embeds, data_name, Path(os.path.join(result_dir, 'token')), data_type=os.path.basename(data_dir))
            for i in range(len(data_name)):
                text_path = os.path.join(data_dir, f"{data_name[i]}_caption.json")
                traj_path = os.path.join(data_dir, f"{data_name[i]}_transforms_cleaning.json")
                text_json = {
                    "Concise Interaction": text_prompt[i]
                }
                ref_json = {
                    "w": ref_intrinsics[i][0][0][2] * 2,
                    "h": ref_intrinsics[i][0][1][2] * 2,
                    "fl_x": ref_intrinsics[i][0][0][0],
                    "fl_y": ref_intrinsics[i][0][1][1],
                    "cx": ref_intrinsics[i][0][0][2],
                    "cy": ref_intrinsics[i][0][1][2],
                    "frames": [{
                        "transform_matrix": matrix_traj_ref[i][frame_idx], 
                        "monst3r_im_id": frame_idx + 1}
                        for frame_idx in range(len(matrix_traj_ref[i]))]
                }
                with open(text_path, 'w') as f:
                    json.dump(text_json, f, indent=4)
                with open(traj_path, 'w') as f:
                    json.dump(ref_json, f, indent=4)
            if data_dir == train_data_dir:
                train_data_name_list += data_name
            else:
                test_data_name_list += data_name
        train_txt_path = os.path.join(result_dir, 'train_valid.txt')
        with open(train_txt_path, "w", encoding="utf-8") as f:
            for name in train_data_name_list:
                f.write(name + "\n")
        test_txt_path = os.path.join(result_dir, 'test_valid.txt')
        with open(test_txt_path, "w", encoding="utf-8") as f:
            for name in test_data_name_list:
                f.write(name + "\n")
        error_txt_path = os.path.join(result_dir, 'error.txt')
        with open(error_txt_path, "w", encoding="utf-8") as f:
            for name in error_data_name_list:
                f.write(name + "\n")
        print("Train set cnt: ", len(train_data_name_list))
        print("Test set cnt :", len(test_data_name_list))
        print("Error set cnt :", len(error_data_name_list))

if __name__ == "__main__":
    train()