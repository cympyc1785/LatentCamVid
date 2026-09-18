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
    #   'ctx_longer_135max': same context windows as 'context_longer' but with LagerNVS's own
    #                     denominator form, 1.35*max||center - window's first center||, averaged
    #                     over the windows (leakage-free; lands in LagerNVS's context units).
    #   'first_farthest_135' / 'geo_lagernvs' : LagerNVS-style 1.35*max variants.
    #   'const' : 세그먼트별로 재지 않고 코퍼스 상수 `norm_scale_const` 로만 나눈다
    #             (= scale align 을 뺀 ablation arm; 상수는 기하평균을 쓸 것).
    # Legacy names still accepted: 'saved_avg_scale'->'avg_scale', 'target_cam'->'cam_dist_mean'.
    scale_mode = 'cam_dist_mean'

    # Intrinsics encoding of cam_param[..., 9:11] — a property of the VAE CKPT, NOT of scale_mode.
    #   'raw'  : fx/2cx, fy/2cy            -> (0.448, 0.796) for DL3DV
    #   'rel'  : the same, divided by frame 0 -> exactly 1.0 for DL3DV (dataset_large.py:313)
    #   'auto' : legacy coupling, 'rel' iff scale_mode == 'avg_scale' (reproduces pre-fix runs
    #            bit-for-bit; use it only to re-run an old experiment)
    # Mismatch vs the ckpt silently destroys the intr channels — see scripts/vae/vae_scale_matrix.py.
    # 'rel' is the correct convention for the default 64-dim ckpt below (20260302, fit AFTER
    # SCVideo added the frame-0 division at data/dataset_large.py:313).
    intr_norm = 'rel'

    # [new 2026-08-23] decode 시 target 첫 카메라를 given 으로 취급 (rel[0]=I 강제).
    # False = 기존 동작. 자세한 근거는 conf/config.yaml 의 같은 키 주석 참고.
    anchor_pred_frame0 = False

    # VAE — SCVideo's config_large.py setting. cam_dim MUST match the ckpt's latent_dim
    # (20260302 -> 64, 20260202_065659 -> 32) or the state_dict load hard-crashes on to_mu.
    # NOTE 0.96032625 was fit over SCVideo's MIXED corpus (DL3DV + DynamicVerse + dynpose-100k);
    # this triple measures 0.44696 on DL3DV-only -> input std 0.4654, not 1.0. Kept verbatim per
    # explicit user decision; see main/conf/config.yaml's comment for the measured matrix.
    use_vae = True
    if use_vae:
        num_cam = 13
        cam_dim = 64
    else:
        num_cam = 49
        cam_dim = 11
    vae_beta = 1e-3
    vae_latent_scale = 0.96032625
    vae_ckpt_path = osp.join(ckpt_root, 'vae_20260302_300.pth')

    num_gpus = 1
    num_thread = 8
    num_frames = 49

    # Text encoder
    text_encoder = 'T5'  # ['T5', 'CLIP']
    if text_encoder == 'T5':
        # [2026-09-03] 512 -> 128. 토크나이저가 `padding='max_length'` 라 text_len 은 예산이
        # 아니라 **고정 패딩 길이**다. 실측 코퍼스 최장이 d121 nl 82 tok / DataDoP 77 tok 라
        # 512 중 84%가 pad 였고, umT5-xxl 인코더가 매 스텝 그 길이를 통째로 돈다.
        # 출력은 안 바뀐다: text CA 가 `key_padding_mask=~text_mask` 로 pad 를 빼고
        # (camera_diffusion_model_latent.py:308), text 에 위치 임베딩이 없으며
        # (`text_proj` 는 per-token Linear), umT5 도 attention mask 를 받는다.
        # 기존 ckpt 와 호환된다 — 늘릴 일이 생기면 여기만 올리면 되고, 캡션이 이 값을
        # 넘으면 `tokenizers.py:49` 의 truncation 이 **조용히** 자른다는 것만 기억할 것.
        text_len = 128
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

    # [new 2026-08-16] DataDoP / 다중 코퍼스 혼합. 근거·실측은 conf/config.yaml 의 같은 키 주석 참고.
    # (전부 getattr 기본값으로도 읽히지만, Config() 를 직접 만드는 비-hydra 스크립트를 위해 둔다.)
    datadop_root = '/data1/cympyc1785/data/DataDoP/DataDoP_with_scene'
    datadop_caption_key = 'Concise Interaction'
    datadop_divisor = 'none'       # 'none' (D=1, MonST3R 게이지 그대로) | 'meanray'
    datadop_norm_gain = 1.0
    datadop_norm_trans_min = 0.005
    datadop_norm_trans_max = 1.0
    datadop_window_stride = None   # None = shot 당 120프레임 1 세그먼트
    datadop_index_workers = 24
    datadop_letterbox = False
    sd_pair_mode = 'list'          # 'list' (기존 동작) | 'random_scene'
    datasets = None                # dataset_name='mixed' 일 때만 읽는 코퍼스별 override 리스트
    mix_val_interleave = True
    mix_check_keys = True          # 코퍼스 간 key set 동일 확인 (collate_fn 이 batch[0] 키만 순회)
    norm_scale_gain = 1.0          # m = mean||t||/norm_scale 에 거는 코퍼스 상수 (1.0 = 무동작)
    norm_scale_const = None        # scale_mode='const' 전용 분모. 코퍼스 **기하평균**을 쓸 것
                                   # (da3_7k/front_first_anchor train 29414: 3.982685)

    # Geo encoder (image-based scene encoder feeding camera_diffusion_model_latent's geo latent)
    geo_encoder = 'lagernvs'    # ['lagernvs', 'scenetok', 'custom', 'da3']
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
    #   'front_uniform': [new 2026-08-13] context range = the L=e-s frames immediately
    #                    BEFORE the target, [s-L, s), sampled uniformly to geo_num_views.
    #                    No retrieval, deterministic. Pairs with
    #                    avg_scale_ref='front_first_anchor_same_len' (divisor built from
    #                    the SAME range); dataset_cfg warns when the two disagree.
    #                    Honours geo_first_view_target_s: on -> view0 = frame s and the
    #                    remaining (geo_num_views-1) context views go in REVERSE time order.
    #   'context_uniform': [new 2026-08-23] context range = the WHOLE video [0, N), sampled
    #                    uniformly to geo_num_views. No retrieval, deterministic per scene
    #                    (independent of s/e). For corpora where context is a DIFFERENT clip
    #                    (SD / TRUMANS) "before/inside the target" is undefined and the clip
    #                    itself is the context; dataset_scene_decoupled._ctx_view_idxs already
    #                    does this. NOT leakage-free: target frames can be picked.
    #                    Honours geo_first_view_target_s: on -> view0 = frame s, remaining
    #                    (geo_num_views-1) uniform over [0, N) minus s.
    geo_view_sampling = 'even'  # ['even', 'random_inseg', 'hybrid', 'frustum_cover',
                                #  'front_uniform', 'context_uniform']
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
    geo_cover_centered_at_s = False  # [renamed 2026-07-31 from geo_anchor_first_frame] center the
                                #   coverage ball on the target's FIRST frame s (known at
                                #   inference) + radius from CONTEXT movement, so selection never
                                #   uses the unseen target [s+1:e]. Steers only WHERE the greedy
                                #   search looks; adds no view to the encoder (that is
                                #   geo_first_view_target_s). Default off (legacy target-mid anchor).
    geo_swap_mode = None        # [new] TEST-TIME probe: None | 'inscene'. Take the geo context
                                #   from a DIFFERENT SEGMENT OF THE SAME SCENE, so only WHICH
                                #   REGION it covers changes (orthogonal to geo_test_inseg_k's
                                #   leakage axis). Force-disables the geo latent cache.
    geo_swap_shift = 1          # donor = the (position + shift)-th other segment of this scene
    geo_swap_keep_first = True  # after the swap put view0 back to the target's frame s (anchor and
                                #   frame/scale link untouched; only the other V-1 views move)
    geo_return_idxs = False     # [new] analysis side-channel: attach 'geo_idxs' + 'geo_ctx_c2w'
                                #   (OpenCV c2w of the context views) to every item. Never fed to
                                #   the model; on a geo-cache HIT it costs the frustum_cover search.

    # Per-SEGMENT coverage blacklist (scene,segment CSV from scripts/dump_coverage.py at a
    # chosen tau). None -> no coverage filtering (only the scene-level blacklist.csv applies).
    coverage_blacklist_path = None
    # [new] precomputed frozen geo-latent cache root (main/cache_geo_embeddings.py). None = OFF,
    # i.e. run LagerNVS every step as before. See conf/config.yaml for layout + validity caveats.
    geo_latent_cache_dir = None
    # [new 2026-08-29] scene 키 pre-ln DA3 캐시 (scripts/data/cache_geo_raw_da3.py). None = OFF.
    # geo_latent_cache_dir 과 자르는 지점이 다르다 — 저건 proj 뒤(=학습 대상 뒤)라 da3 에서는
    # 못 쓰고, 이건 ln 직전이라 ln/proj 가 그대로 학습된다. context 가 scene 상수인 코퍼스
    # (Vista4D) 전용. 조건은 conf/config.yaml 주석 참조.
    geo_raw_cache_dir = None
    geo_raw_cache_preload = True    # scene 파일 전량을 __init__ 에서 RAM 에 올린다 (fork 공유)
    geo_raw_cache_mmap = False      # [new 2026-09-18, D200] 그 적재를 mmap 으로 (arm 끼리 공유)
    # [new] per-context-view camera embedding channel-concatenated onto the geo tokens.
    # None = OFF (geo condition stays the plain (M, 768) LagerNVS tokens). 'relfirst' = 11-d pose
    # of each context view relative to the target's FIRST camera (trans / norm_scale), appended
    # raw by the dataset and lifted by a trainable MLP inside the model. See conf/config.yaml.
    geo_cam_embed = None
    geo_cam_embed_dim = 128
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
    # custom backend: models/custom_geo_encoder.py (frozen DINOv2 + trainable Plücker/depth
    # tokenizer). lagernvs 와 달리 TRAINABLE 이고 batch dict 을 받는다 (GeoEncoder.forward_batch).
    custom_geo_dino_path = None     # None -> osp.join(ckpt_root, 'dinov2-large')
    custom_geo_freeze_dino = True
    custom_geo_input_hw = None      # None -> geo_image_hw 를 patch(14) 배수로 내림
    custom_geo_ray_dim = 64
    custom_geo_geo_dim = 256
    # da3 backend: models/da3_geo_encoder.py (frozen Depth-Anything-3 cross-view ViT +
    # trainable LayerNorm; 768 로 내리는 Linear 는 GeoEncoder.proj). custom 과 달리 픽셀
    # Plücker/depth 를 안 쓰고 (dataset 부하 감소) 카메라는 DA3 자신의 cam_enc 로 넣는다
    # (geo_posed=True). vendored tools/Depth-Anything-3 는 읽기 전용 — 수정 0줄.
    da3_geo_repo_path = '/data1/cympyc1785/LatentCamVid/camera_generation/tools/Depth-Anything-3'
    da3_geo_model = 'da3nested-giant-large'   # DA3 registry key. nested 는 anyview 갈래만 올린다
                                #   (metric 갈래는 depth 스케일 정렬 전용이라 feature 와 무관).
                                #   corpus <scene>/da3/*.npz 를 만든 바로 그 모델이다.
    da3_geo_ckpt_path = None    # None -> da3_geo_hf_home 의 HF 캐시 snapshot/model.safetensors
    da3_geo_hf_home = '/data1/cympyc1785/cache/huggingface'
    da3_geo_input_hw = [252, 448]   # patch(14) 배수. 280x504(DA3 native) 대비 등방 x8/9 축소이고
                                #   custom backend 와 같은 격자(18x32 = view 당 576 토큰)다.
                                #   해상도 probe: results/20260812_da3_res_probe{,_giant}
    da3_geo_layers = 'last'     # ['last', 'all'] — out_layers 4개 중 마지막만 / 전부 concat
    da3_geo_layer_fuse = 'concat'
    da3_geo_norm = 'ln'         # cat_token 두 반쪽(un-normed local | normed global)의 스케일을
                                #   맞추는 LayerNorm. 'none' 으로 끄면 한쪽이 다른 쪽을 먹는다.
    da3_geo_debug = False
    # [new 2026-08-29] cam_enc 를 샘플별로 부를지. False = 기존(배치 통째). 근거는
    # da3_geo_encoder.build_cam_token 주석 — B>1 로 부르면 cuBLAS kernel 이 바뀌어 cam_token 이
    # rel 2.9e-7 흔들리고, backbone 이 그걸 pre-ln 토큰에서 rel 2.3e-2 로 증폭한다.
    da3_cam_token_per_sample = False
    # [new 2026-08-16] da3 backend 를 transforms(COLMAP) pose 로도 허용할지. 기본 False =
    # 기존 raise 유지. 근거는 conf/config.yaml 의 같은 키 주석 참고.
    da3_geo_allow_transforms_pose = False

    clip_version = 'ViT-B/32'
    clip_max_length = 77

    # config_large.py's CLaTr ckpt. Architecture-identical drop-in to clatr_epoch109_dl3dv_seg_2
    # (same 260-key state_dict, both 191,546,118 B; the input standardization lives in
    # evaluate/CLaTr/configs/dataset/standardization/0120.yaml, not in the ckpt) but trained on a
    # DIFFERENT corpus -> FD/PRDC/CLaTr-score are NOT comparable with our epoch109 history.
    clatr_ckpt_path = osp.join(ckpt_root, 'clatr_epoch139_large.ckpt')


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
