import os
import os.path as osp
import sys
import torch
import random
import numpy as np

# DL3DV-only config (cf. config_large.py which also adds DynamicVerse + dynpose).
# SCVideo scene format: each scene folder has prompt.json + point cloud ply +
# cameras.json (+ images for the image-based geo encoders).
DL3DV_DATA_PATH = '/data1/cympyc1785/SceneData/DL3DV/scenes'


def build_dataset_dir_list():
    """DL3DV-only scene dir list. Guarded so importing this config does not crash
    when the scene data is not present yet (port may not have it locally)."""
    dataset_dir_list = []
    if os.path.isdir(DL3DV_DATA_PATH):
        dataset_dir_list += [
            os.path.join(DL3DV_DATA_PATH, scene_type)
            for scene_type in sorted(os.listdir(DL3DV_DATA_PATH))
        ]
    else:
        print(f"[config] WARNING: DL3DV data path not found: {DL3DV_DATA_PATH} "
              f"(dataset_dir_list is empty)")
    return dataset_dir_list


class Config:

    exp_name = 'dl3dv_latent_baseline'

    # Path
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    ckpt_root = osp.join(root_dir, 'checkpoints')   # local ported checkpoints
    dataset_dir_list = build_dataset_dir_list()     # dataset_seg.py reads cfg.dataset_dir_list
    val_ckpt_path = None                            # set to a local .pth for validation/inference
    load_ckpt_path = None                           # set to resume from a local ckpt
    embed_path = None                               # only used when load_saved_pc_embeds=True
    random_seed = 42

    # Training
    lr = 1e-4
    batch_size = 8            # config_large uses 32; bumped from 2 (VRAM headroom on 81GB card)
    accum_steps = 1
    epochs = 2000
    save_epoch = 1            # (unused; validation runs once per epoch)
    val_step = 5000          # (unused now; validation is per-epoch, not step-based)
    val_max_batches = 20     # cap validation to N batches so it stays cheap
    sample_data = None        # None -> use all scenes (config_large subsamples to 33980)
    load_points = False       # geo path uses images (not point clouds); pc branch -> None, geo overrides
    load_saved_pc_embeds = False
    use_concerto = False      # image-based geo encoder path (Concerto/point cloud not used)
    attn_loss_weight = 1e-2   # only used by model_type='baseline_attn_sup'
    max_trans_norm = False

    # Diffusion
    model_type = 'baseline'   # ['baseline', 'director', 'baseline_mmdit', 'baseline_attn_sup', 'latent']
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

    # Chunk-wise autoregressive (AR): denoise latent chunk-by-chunk, each chunk conditioned
    # on past CLEAN chunks via causal self-attn (model.forward_ar) + text/geo. Default off.
    is_ar = False
    ar_chunk_size = 3          # latent steps per chunk (13 latent -> ~5 chunks)
    causal_vae = False         # AR uses the causal VAE (past-only conv)

    # Per-token noise (Diffusion Forcing) + rectified flow, in the CAUSAL-VAE latent space.
    # per_token_noise=False -> unchanged DDPM epsilon full-sequence path (bit-identical).
    per_token_noise = False
    cfg_dropout_p = 0.1        # text CFG dropout prob (per-token/flow path)
    loss_tau_min = 0.02        # tokens with tau < this are context -> excluded from loss

    # Camera translation normalization scale (see dataset_dl3dv.py's module docstring).
    #   'avg_scale'     : STORED point-cloud avg_scale (SCVideo original), i.e.
    #                     mean ||scene point - first camera|| from avg_scale/<seg>.json.
    #   'cam_dist_mean' : mean CAMERA center-norm of the TARGET segment (the scale depends
    #                     on the target -> not available at inference).
    #   'context_longer': cam_dist_mean over the LONGER out-of-segment side, chunked into
    #                     num_frames windows (target-excluded, so leakage-free AND
    #                     reproducible at inference from context only).
    #   'first_farthest_135' / 'geo_lagernvs' : LagerNVS-style 1.35*max variants.
    # Legacy names still accepted: 'saved_avg_scale'->'avg_scale', 'target_cam'->'cam_dist_mean'.
    scale_mode = 'cam_dist_mean'

    # Intrinsics encoding of cam_param[..., 9:11] — a property of the VAE CKPT, NOT of scale_mode.
    #   'raw'  : fx/2cx, fy/2cy            -> (0.448, 0.796) for DL3DV
    #   'rel'  : the same, divided by frame 0 -> exactly 1.0 for DL3DV (dataset_large.py:313)
    #   'auto' : legacy coupling, 'rel' iff scale_mode == 'avg_scale' (reproduces pre-fix runs
    #            bit-for-bit; use it only to re-run an old experiment)
    # Mismatch vs the ckpt silently destroys the intr channels — see scripts/vae/vae_scale_matrix.py.
    # 'raw' is the correct convention for the default 32-dim ckpt below (20260202_065659, fit
    # before SCVideo added the frame-0 division at data/dataset_large.py:313).
    intr_norm = 'raw'

    # VAE — SCVideo's DL3DV-only config.py setting. cam_dim MUST match the ckpt's latent_dim
    # (20260202_065659 -> 32, 20260302 -> 64) or the state_dict load hard-crashes on to_mu.
    # NOTE 0.4467666 is SCVideo's constant; on our 1264 DL3DV SEGMENTS the same triple measures
    # 0.46312 -> input std 1.0366, not 1.0. Kept verbatim per explicit user decision; see
    # main/conf/config.yaml's comment for the measured matrix.
    use_vae = True
    if use_vae:
        num_cam = 13
        cam_dim = 32
    else:
        num_cam = 49
        cam_dim = 11
    vae_beta = 1e-3
    vae_latent_scale = 0.4467666
    vae_ckpt_path = osp.join(ckpt_root, 'vae_20260202_065659_400.pth')

    num_gpus = 1
    num_thread = 8
    num_frames = 49

    # Text encoder
    text_encoder = 'T5'  # ['T5', 'CLIP']
    if text_encoder == 'T5':
        text_len = 512
        t5_dtype = torch.bfloat16
        t5_checkpoint_dir = "/data1/cympyc1785/LatentCamVid/video_generation/models/DiffSynth-Studio/Wan-AI/Wan2.2-TI2V-5B"
        t5_checkpoint_path = 'models_t5_umt5-xxl-enc-bf16.pth'
        t5_tokenizer_path = 'google/umt5-xxl'
    elif text_encoder == 'CLIP':
        clip_repo_name = 'hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP'

    # Point encoder (legacy point-cloud path; kept for compat, unused when geo_encoder is set)
    point_encoder = 'concerto'  # ['concerto', 'mosaic', 'custom']

    # DL3DV data (dataset_dl3dv.CamDataset reads meta.csv, excludes blacklist.csv)
    dl3dv_root = '/data1/cympyc1785/data/DL3DV/scenes'
    max_scenes = None          # all scenes (set small for smoke)
    lazy_dataset = True        # __init__ builds only the sample index (persisted cache); poses parsed lazily per __getitem__
    train_seg_list = None      # explicit train segment-list file (<batch>/<hash>/<seg>); None = random 90/10
    test_seg_list = None       # explicit val/test segment-list file; val = its first N segments (shuffle=False)

    # Geo encoder (image-based scene encoder feeding camera_diffusion_model_latent's geo latent)
    geo_encoder = 'lagernvs'    # ['lagernvs', 'scenetok']
    geo_latent_dim = 768        # must match CameraDiffusionModel(geo_latent_dim=...)
    geo_num_views = 4           # (even path only) multi-view images fed per scene
    geo_image_hw = [256, 448]   # (H, W) images are loaded/resized to for the geo encoder

    # Geo context-view sampling (leakage ablation)
    #   'even'         : evenly-spaced frames WITHIN the target segment [s:e] (baseline;
    #                    leaks the target trajectory to the VGGT-based encoder).
    #   'random_inseg' : anchor s + (geo_num_views-1) random in-segment frames, re-drawn
    #                    each access (augmentation; still in-segment).
    #   'hybrid'       : in-segment near-anchor views (scene grounding, minimal leak) +
    #                    out-of-segment co-visibility-retrieved views (decorrelated).
    #   'frustum_cover': k views (in/out ignored) whose frustums cover the most nearby
    #                    space (greedy max set-coverage).
    geo_view_sampling = 'even'  # ['even', 'random_inseg', 'hybrid', 'frustum_cover']
    geo_cover_k = 6             # (frustum_cover) number of context views
    geo_cover_radius = 2.5      # (frustum_cover) candidate/space ball = R * segment-scale
                               # (swept 1.0-4.0 over 10 scenes: 2.5 = first radius with 0
                               #  degenerate selections, 82% cover, 62deg viewpoint diversity)
    geo_cover_ndepth = 3        # (frustum_cover) proxy points per ray (grid mode: unused)
    geo_num_inseg = 3           # (hybrid) near-anchor in-segment views
    geo_num_covis = 3           # (hybrid) out-of-segment co-visibility views
    # frustum_cover leakage control + posed geo conditioning
    geo_cover_out_of_seg = False  # (frustum_cover) restrict candidates to the LONGER
                                  #   out-of-segment side (no target frames -> no leak)
    geo_posed = False           # feed the geo views' camera cam_token to the lagernvs
                                #   reconstructor (posed, lagernvs 1.35*max normalization)
                                #   instead of zeros (unposed). Renders/latent in-distribution.
    geo_shuffle_order = False   # permute the ORDER of the selected geo context views each
                                #   access (posed: changes the first/VGGT-reference view ->
                                #   order-invariance augmentation). Default off (fixed order).
    geo_first_view_target_s = False  # geo context view0 = target segment's FIRST camera s
                                #   (+ (k-1) out-of-seg retrieved). LagerNVS anchors to view0 ->
                                #   geo latent frame == target (frame-s) frame. Needs the start
                                #   frame IMAGE available at inference.
    geo_anchor_first_frame = False  # honest context selection: anchor at the target's FIRST
                                #   frame only (known at inference) + radius from CONTEXT
                                #   movement, so selection never uses the unseen target
                                #   [s+1:e]. Default off (legacy target-midpoint anchor).

    # Per-SEGMENT coverage blacklist (scene,segment CSV from scripts/dump_coverage.py at a
    # chosen tau). None -> no coverage filtering (only the scene-level blacklist.csv applies).
    coverage_blacklist_path = None
    geo_inseg_span = None       # (hybrid) frames from anchor to spread in-segment over (None -> num_frames//8)
    geo_covis_radius = 2.0      # (hybrid) candidate center within R * segment-scale of anchor
    geo_covis_theta0 = 10.0     # (hybrid) preferred triangulation angle (deg) for covis score
    geo_covis_max_axis_deg = 80.0  # (hybrid) reject candidates whose optical axis differs > this
    geo_covis_topM = 32         # (hybrid) unused (FPS now runs over all gated candidates by direction)
    # lagernvs backend: frozen VGGT reconstructor -> 768-d scene tokens
    lagernvs_repo_path = '/data1/cympyc1785/LatentCamVid/camera_generation/tools/lagernvs'
    lagernvs_ckpt_path = '/data1/cympyc1785/LatentCamVid/camera_generation/tools/lagernvs/checkpoints/lagernvs_general_512/model.pt'
    # scenetok backend (stub for now)
    scenetok_repo_path = None
    scenetok_ckpt_path = None

    clip_version = 'ViT-B/32'
    clip_max_length = 77

    # All our runs' CLaTr metrics are against this ckpt. config_large.py's clatr_epoch139_large.ckpt
    # is an architecture-identical drop-in but trained on a different corpus -> incomparable scores.
    clatr_ckpt_path = osp.join(ckpt_root, 'clatr_epoch109_dl3dv_seg_2.ckpt')


cfg = Config()
cfg_dict = {
    k: v for k, v in vars(Config).items()
    if not k.startswith("__")
}

from utils.dir import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(cfg.random_seed)
