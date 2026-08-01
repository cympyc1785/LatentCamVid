"""CamDataset-shaped shim over DynamicVerse, so DL3DV-only analysis/render scripts can run on it.

Why a shim and not a real CamDataset subclass: DynamicVerse has no meta CSV and no COLMAP dir, and
its `prompts.json` gives ONE segment covering the whole clip -- so there are no chunk 0/1/2 to
compare. We cut the chunks ourselves, k-th chunk = frames [k*T, (k+1)*T), exactly as DL3DV's
segments are laid out, and drop scenes shorter than n_chunks*T.
`avg_scale/<k>.json` DOES exist (unlike what this docstring said before 2026-08-02) and its file
count is exactly floor(N_frames / num_frames), i.e. keyed by the SAME chunk cut we make -- so
`scale_mode: avg_scale` is a real measurement here, not a `_cam_dist_mean_scale` fallback.

Only the attributes the retrieval path touches are filled in, so the (deterministic, greedy)
`_sample_geo_frustum_cover` that defines the `geo_lagernvs` divisor is the REAL one from
`dataset_dl3dv`, not a reimplementation.

cameras.json convention: `rotation`/`position` are **R_w2c / t_w2c** -- same reading as
`utils/camera_utils.get_camera_params_from_json:356-360`. Every scene in every subset has
cx=252 / cy=140, i.e. the poses were estimated at 504x280 no matter what `video_input.mp4`'s
resolution is, so (h, w) = (2cy, 2cx) and extracted frames are resized to exactly that.

Users:
  scripts/data/norm_divisor_compare.py   DATASET=dynamicverse   (cameras only)
  main/dump_avgscale_render.py           DATASET=dynamicverse   (needs frames -> with_frames=True)
"""
import json
import os

import numpy as np
import torch

DV_ROOT = "/data1/cympyc1785/data/dynamicverse"
FRAME_CACHE = "/data1/cympyc1785/data/dynamicverse_frames"   # <sub>/<scene>/%05d.png @ 504x280


def extract_frames(scene_dir, out_dir, hw, video="video_input.mp4"):
    """Decode <scene_dir>/<video> to out_dir/%05d.png resized to hw=(h, w). Cached: if the dir
    already holds >= the video's frame count, nothing is decoded."""
    import imageio.v2 as imageio
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    have = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
    rd = imageio.get_reader(os.path.join(scene_dir, video))
    if have:
        rd.close()
        return [os.path.join(out_dir, f) for f in have]
    files = []
    for i, fr in enumerate(rd):
        p = os.path.join(out_dir, f"{i:05d}.png")
        Image.fromarray(fr).resize((hw[1], hw[0]), Image.LANCZOS).save(p)
        files.append(p)
    rd.close()
    return files


def load_dynamicverse(cfg, root=DV_ROOT, n_chunks=3, scenes=None, with_frames=False,
                      frame_cache=FRAME_CACHE, verbose=True):
    """Build the shim. `scenes` = optional list of "<subset>/<name>" to restrict to (order kept).
    `with_frames=True` additionally decodes each kept scene's video into `frame_cache` and fills
    `ds.frame_files_list` -- only do that for the few scenes you actually render."""
    T_ = cfg.num_frames
    need = n_chunks * T_
    from dataset_dl3dv import CamDataset
    ds = CamDataset.__new__(CamDataset)
    ds.cfg = cfg
    ds.num_frames = T_
    ds.geo_cover_k = getattr(cfg, 'geo_cover_k', 6)
    ds.geo_cover_radius = getattr(cfg, 'geo_cover_radius', 2.0)
    ds.geo_cover_ndepth = getattr(cfg, 'geo_cover_ndepth', 3)
    ds.geo_cover_out_of_seg = getattr(cfg, 'geo_cover_out_of_seg', False)
    ds.geo_cover_before_only = getattr(cfg, 'geo_cover_before_only', False)
    ds.geo_cover_centered_at_s = getattr(cfg, 'geo_cover_centered_at_s', False)
    ds.geo_first_view_target_s = getattr(cfg, 'geo_first_view_target_s', False)
    ds._geo_idx_memo = {}
    # [new] DynamicVerse 도 <scene>/avg_scale/<k>.json 을 갖고 있다 (2026-08-02 확인). 개수가
    # 정확히 floor(N_frames / num_frames) 라 k 는 우리가 자르는 chunk [k*T,(k+1)*T) 와 1:1 이고,
    # data_name 이 "<subset>/<scene>_<k>" 라 DL3DV 와 똑같이 split('_')[-1] 이 seg_key 가 된다.
    # 값/clamp/실패시 None 은 CamDataset._avg_scale (dataset_dl3dv.py:744) 과 동일.
    ds._scene_dirs = []                            # scene_idx -> <root>/<subset>/<scene>

    def _dv_avg_scale(scene_idx, seg_key, _ds=ds):
        p = os.path.join(_ds._scene_dirs[scene_idx], 'avg_scale', f'{seg_key}.json')
        if not os.path.isfile(p):
            return None
        try:
            return torch.tensor([float(json.load(open(p)))]).clamp(min=1e-5)
        except Exception:
            return None

    ds._avg_scale = _dv_avg_scale
    # [new] `ds[i]` (= the REAL CamDataset.__getitem__) 를 그대로 쓰기 위한 나머지 속성.
    # 이전 사용자(norm_divisor_compare / dump_avgscale_render)는 ds.samples 와 _sample_geo_* 만
    # 건드려서 필요 없었지만, cam_param 을 뽑으려면(vae_scale_matrix DATASET=dynamicverse) 필요하다.
    # 값은 전부 CamDataset.__init__ 과 같은 getattr 기본값이라 기존 경로 동작은 그대로다.
    ds.geo_hw = tuple(getattr(cfg, 'geo_image_hw', (256, 448)))
    ds.geo_num_views = getattr(cfg, 'geo_num_views', 4)
    ds.geo_enabled = bool(getattr(cfg, 'geo_encoder', None))
    ds.geo_latent_cache_dir = None                # DL3DV 용 캐시라 DynamicVerse 엔 없음
    ds.geo_cam_embed = getattr(cfg, 'geo_cam_embed', None)
    ds.geo_view_sampling = getattr(cfg, 'geo_view_sampling', 'even')
    ds.geo_posed = getattr(cfg, 'geo_posed', False)
    ds.geo_shuffle_order = getattr(cfg, 'geo_shuffle_order', False)
    ds.extrinsics_list, ds.intrinsics_list, ds.hw_list, ds.samples = [], [], [], []
    ds.frame_files_list, ds.scene_names = [], []

    if scenes is None:
        pairs = []
        for sub in sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)) and d != "eval_index"):
            pairs += [f"{sub}/{n}" for n in sorted(os.listdir(os.path.join(root, sub)))]
    else:
        pairs = list(scenes)

    for key in pairs:
        sd = os.path.join(root, key)
        cj = os.path.join(sd, "cameras.json")
        if not os.path.exists(cj):
            continue
        try:
            cams = json.load(open(cj))
        except Exception:
            continue
        if len(cams) < need:
            continue
        w2c = np.tile(np.eye(4, dtype=np.float32), (len(cams), 1, 1))
        K = np.zeros((len(cams), 3, 3), dtype=np.float32)
        for i, p in enumerate(cams):
            w2c[i, :3, :3] = p["rotation"]; w2c[i, :3, 3] = p["position"]
            K[i] = [[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]]
        hw = (int(round(2 * cams[0]["cy"])), int(round(2 * cams[0]["cx"])))
        si = len(ds.extrinsics_list)
        ds.extrinsics_list.append(torch.from_numpy(w2c))
        ds.intrinsics_list.append(torch.from_numpy(K))
        ds.hw_list.append(hw)
        ds.scene_names.append(key)
        ds._scene_dirs.append(sd)
        ds.frame_files_list.append(
            extract_frames(sd, os.path.join(frame_cache, key), hw) if with_frames else None)
        for k in range(n_chunks):
            ds.samples.append((si, k * T_, (k + 1) * T_, "", f"{key}_{k}"))
    if verbose:
        # [new] avg_scale 커버리지도 같이 찍는다 -- 하나라도 비면 그 chunk 는 조용히
        # _cam_dist_mean_scale 로 fallback 되므로 avg_scale 행이 섞인 측정이 된다.
        hit = sum(ds._avg_scale(si, dn.split('_')[-1]) is not None
                  for si, _s, _e, _c, dn in ds.samples)
        print(f"[dynamicverse] {len(ds.extrinsics_list)} scenes with >= {need} frames "
              f"-> {len(ds.samples)} chunks ({n_chunks} per scene), "
              f"avg_scale {hit}/{len(ds.samples)}")
    return ds
