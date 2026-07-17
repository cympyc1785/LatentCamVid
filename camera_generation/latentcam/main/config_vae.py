import os
import os.path as osp
import sys
import torch
import random

def build_dataset_dir_list():
    dataset_dir_list = []

    dl3dv_data_path = '/data1/cympyc1785/SceneData/DL3DV/scenes'
    dl3dv_dataset_dir_list = [os.path.join(dl3dv_data_path, scene_type) for scene_type in sorted(os.listdir(dl3dv_data_path))]
    dataset_dir_list += dl3dv_dataset_dir_list

    dynamicverse_data_path = '/data1/cympyc1785/SceneData/DynamicVerse/scenes'
    dv_dataset_dir_list = [os.path.join(dynamicverse_data_path, scene_type) for scene_type in sorted(os.listdir(dynamicverse_data_path)) if 'dynpose' not in scene_type]
    dataset_dir_list += dv_dataset_dir_list

    dynpose_data_path = '/data1/cympyc1785/SceneData/DynamicVerse/scenes/dynpose-100k'
    dp_dataset_dir_list = [os.path.join(dynpose_data_path, scene_type) for scene_type in sorted(os.listdir(dynpose_data_path))]
    dataset_dir_list += dp_dataset_dir_list

    # datadop_dir_list = ['/home/ckd248/data/SCVideo/camera_generation/tools/GenDoP/DataDoP/ours']
    # dataset_dir_list += datadop_dir_list

    # dataset_dir_list = dataset_dir_list[0:1]

    return dataset_dir_list

class Config:

    exp_name = 'cam_vla'

    # Path
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    dataset_dir_list = build_dataset_dir_list()
    val_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results/20260226_082914_cross_attn/ckpts/600.pth'
    embed_path = '/home/ckd248/data/SCVideo/camera_generation/dataset/pc_embeds_datadop'
    random_seed = 42

    # Training
    lr = 1e-3
    batch_size = 64
    accum_steps = 1
    epochs = 2000
    save_epoch = 50
    sample_data = None
    load_points = False
    load_saved_pc_embeds = False
    max_trans_norm = False

    # Diffusion
    model_type = 'baseline' # ['baseline', 'director', 'baseline_mmdit]
    sampling_type = 'ddim'
    diffusion_max_step = 1000
    diffusion_inference_step = 50
    beta_start = 0.00085
    beta_end = 0.012
    prediction_type = 'epsilon'
    beta_schedule = 'scaled_linear'
    clip_sample = False
    set_alpha_to_one = False
    steps_offset = 1

    # VAE
    vae_beta = 1e-3
    # vae_latent_scale = 0.4467666
    vae_latent_scale = 0.96032625
    vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260302/300.pth'
    # # target: datadp norm: gendop
    # vae_latent_scale = 0.513456
    # vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260309_064024/ckpts/500.pth'
    # # # target: worldtraj norm: gendop
    # vae_latent_scale = 0.53829
    # vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260309_053143/ckpts/250.pth'
    # target: worlrdtraj norm: ours
    # vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260307_035047/ckpts/350.pth'
    # target: datadop norm: ours
    # vae_latent_scale = 0.45017
    # vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260310_133840/ckpts/550.pth'
    # dl3dv7K vae
    vae_latent_scale = 0.472981
    vae_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results_vae/20260316_073439_dl3dv_7K/ckpts/1900.pth'
    # dv vae
    vae_latent_scale = 0.45341
    vae_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results_vae/20260316_073511_dv/ckpts/1000.pth'
    # dl3dv+dv vae
    vae_latent_scale = 0.453496
    vae_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results_vae/20260316_073543_dl3dv+dv/ckpts/700.pth'
    # datadop wo scale norm
    vae_latent_scale = 0.42653
    vae_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results_vae/20260423_022113_datadop_wo_scale_norm/ckpts/300.pth'
    # datadop w max trans norm
    vae_latent_scale = 0.502852
    vae_latent_scale = 0.8653976

    # VLA
    vae_latent_scale = 0.944445
    vae_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results_vae/20260424_062842_datadop_text_w_max_trans_norm/ckpts/400.pth'

    vae_ckpt_path = '/data2/ckd248/SCVideo/camera_generation/results_vae/20260428_070011_cam_vla/ckpts/100.pth'

    # 64, 5e-3
    # vae_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/results_vae/20260310_064050/ckpts/150.pth'

    num_gpus = 1
    num_thread = 4
    num_frames = 48

    # Text encoder
    text_encoder = 'T5' # ['T5', 'CLIP']
    if text_encoder == 'T5':
        text_len = 512
        t5_dtype = torch.bfloat16
        t5_checkpoint_dir = '/home/ckd248/data/SCVideo/camera_generation/tools/Wan2.2-TI2V-5B'
        t5_checkpoint_path = 'models_t5_umt5-xxl-enc-bf16.pth'
        t5_tokenizer_path = 'google/umt5-xxl'
    elif text_encoder == 'CLIP':
        clip_repo_name = 'hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP'

    # Point encoder
    point_encoder = 'concerto' # ['concerto', 'custom']

    clip_version = 'ViT-B/32'
    clip_max_length = 77

    # clatr_ckpt_path = 'CLaTr_checkpoints/epoch99_directorial.ckpt'
    clatr_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/main/evaluate/CLaTr/CLaTr_checkpoints/epoch139_large.ckpt'
    # clatr_ckpt_path = '/home/ckd248/data/SCVideo/camera_generation/main/evaluate/CLaTr/CLaTr_checkpoints/epoch89_dl3dv_seg_kl_1e-8.ckpt'

cfg = Config()
cfg_dict = {
    k: v for k, v in vars(Config).items()
    if not k.startswith("__")
}

from core_pkg.common.utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))