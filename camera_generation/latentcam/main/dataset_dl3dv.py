"""DL3DV dataset for the latent (geo-conditioned) camera diffusion model.

Scene list from meta.csv (chunk = '<batch>K/<hash>'), scenes in blacklist.csv
excluded. Each scene: transforms.json (nerfstudio c2w + intrinsics) + images_4/ or images_8/ +
prompts.json (per-segment captions).

Samples are per PROMPT SEGMENT (like the original SCVideo dataset): each segment
in prompts.json has frame_idx=[start,end] (a 49-frame window) and a caption
prompt_camera_with_scene_video.concise.

Per sample:
  data_name    : str  ('<batch>_<hash>_<seg>')
  cam_param    : (num_frames, 11)  rot6d(6)+trans(3)+[fx/w, fy/h](2), normalized
  intrinsics   : (num_frames, 3, 3)
  first_extrinsic : (4, 4)  un-normalized w2c of the segment's first frame
  norm_scale   : (1,)  the divisor the active `scale_mode` produced; also emitted under the
                 legacy key 'avg_scale' (SCVideo's name) for compatibility. NOTE 'avg_scale'
                 as a SCALE_MODE means specifically the stored point-cloud value -- see below.
  images       : (V, 3, H, W)  multi-view RGB (segment frames) for the geo encoder
  text_prompt  : str  (prompt_camera_with_scene_video.concise)
  height/width : int

scale_mode (what the camera translations are divided by):
  'avg_scale'          stored point-cloud avg_scale, <scene>/avg_scale/<seg>.json =
                       mean(||scene point - first camera||)  (SCVideo original; ~10-44)
  'cam_dist_mean'      mean(||camera center_i - center_0||) over the target segment  (~1-2)
  'context_longer'     'cam_dist_mean' computed on out-of-segment context windows instead
  'ctx_longer_135max'  1.35 * max(||center - window's first center||) averaged over the CONTEXT
                       RANGE's num_frames windows (LagerNVS's denominator form, leakage-free)
  'first_farthest_135' 1.35 * max(||center_i - center_0||) over the segment (LagerNVS-style)
  'geo_lagernvs'       1.35 * max(||geo-context center - frame s||) (full LagerNVS alignment)
Legacy aliases accepted: 'saved_avg_scale' -> 'avg_scale', 'target_cam' -> 'cam_dist_mean'.

pose_source (pose/caption/avg_scale 를 어느 코퍼스에서 읽을지) — 자세한 근거는 아래
_POSE_SOURCES 주석 참고. 'transforms'(기본)는 기존 동작 그대로.
  'transforms'  <scene>/{transforms.json, prompts.json, avg_scale/}
  'da3'         <scene>/da3/{pose.npz, prompts.json, avg_scale/}   (Depth Anything 3 예측 pose)

intr_norm (how cam_param's last 2 channels encode the intrinsics) — INDEPENDENT of scale_mode,
because it is a property of the VAE CHECKPOINT (whichever convention that ckpt was trained on):
  'raw'   fx/(2cx), fy/(2cy)                     -> (0.448, 0.796) for DL3DV
  'rel'   the same, divided by frame 0           -> exactly 1.0 for DL3DV (constant intrinsics)
  'auto'  legacy coupling: 'rel' iff scale_mode == 'avg_scale', else 'raw' (kept as the default
          so existing configs reproduce bit-for-bit; new configs should set this explicitly).
SCVideo's current dataset_large.py:313 always applies the frame-0 division ('rel'), but ckpts
trained before that line was added want 'raw'. Measured recon L1 on the intr channels (1264
DL3DV samples, scripts/vae/vae_scale_matrix.py): vae_20260302_300 (64-dim) 0.0022-0.0040 with
'rel' vs 0.361-0.365 with 'raw'; vae_20260202_065659_400 (32-dim) 0.0029-0.0134 with 'raw' vs
0.259-0.263 with 'rel'. Mismatching it silently destroys the intrinsics channels.

Assumptions to validate (see plan): transforms.json OpenGL c2w -> OpenCV w2c;
frame_idx indexes the file-sorted transforms frames.
"""
import os
import os.path as osp
import csv
import json
import hashlib
import random

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from tqdm import tqdm

from utils.data_utils import normalize_camera_extrinsics_and_points

# OpenGL(c2w) -> OpenCV(c2w): flip Y and Z camera axes
_GL2CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))

# old scale_mode spellings -> current name. 'avg_scale' now means ONLY the stored point-cloud
# value; the camera-distance one is 'cam_dist_mean'. Old configs/wandb runs keep working.
_SCALE_MODE_ALIASES = {'saved_avg_scale': 'avg_scale', 'target_cam': 'cam_dist_mean'}


def resolve_scale_mode(cfg):
    m = getattr(cfg, 'scale_mode', 'cam_dist_mean')
    return _SCALE_MODE_ALIASES.get(m, m)


# [new 2026-08-06] pose_source -- scene 의 pose / intrinsics / caption / avg_scale 를 어느
# 코퍼스에서 읽을지. 'transforms' 가 기본이고 기존 동작과 bit-identical 하다.
#   'transforms'  <scene>/transforms.json (nerfstudio OpenGL c2w) + <scene>/prompts.json
#                 + <scene>/avg_scale/<seg>.json                     [기존]
#   'da3'         <scene>/da3/pose.npz    (Depth Anything 3 가 예측한 pose)
#                 + <scene>/da3/prompts.json + <scene>/da3/avg_scale/<seg>.json
#
# da3/pose.npz 의 규약 (실측으로 확정, 2026-08-06):
#   extrinsics (N,3,4) = **w2c, OpenCV** — transforms.json 과 달리 GL->CV flip 이 필요 없고
#     역행렬도 필요 없다. w2c 로 읽고 GT(transforms.json 을 OpenCV w2c 로 변환한 것)와
#     Umeyama 정렬하면 12개 표본 scene 중 11개가 ATE/extent <= 0.004, 회전 평균 <= 0.32deg 다
#     (나머지 1개 8a1b61638a 는 0.194 / 3.49deg). c2w 로 잘못 읽으면 회전 오차가 137~178deg 로
#     튄다.  !! <scene>/pose_eval_vs_gt.json 과 da3_camcond_report.json("convention":"c2w",
#     da3 ATE_norm 0.4764 / Rot_mean 172deg)은 바로 이 잘못된 c2w 해석으로 만들어진 수치라
#     신뢰하면 안 된다 (5개 방법 전부 172~176deg 로 나오는 게 그 증거).
#   intrinsics (N,3,3) = 504x280 픽셀 공간. cx*2=504, cy*2=280 로 전 프레임 상수라
#     cam_param 의 fx/(2cx), fy/(2cy) 는 해상도와 무관하게 그대로 성립한다. 다만 fx 는
#     **프레임마다 조금씩 다르다** (scene 내 std/mean 7e-4 ~ 3e-3). transforms.json 은 scene 당
#     상수였으므로 intr_norm='rel' 에서 [1,1] 정확히가 아니라 1.000 +- 0.003 이 된다.
_POSE_SOURCES = ('transforms', 'da3')


def resolve_pose_source(cfg):
    ps = getattr(cfg, 'pose_source', 'transforms') or 'transforms'
    if ps not in _POSE_SOURCES:
        raise ValueError(f"pose_source must be one of {_POSE_SOURCES}, got {ps!r}")
    return ps


# per-scene image dir, in preference order: 'images_4' = DL3DV-960 (960x540),
# 'images_8' = DL3DV-480 (480x270). Intrinsics/hw always come from transforms.json
# (full 3840x2160), so which one is present only affects image detail.
IMAGE_DIR_NAMES = ('images_4', 'images_8', 'images')


def scene_image_dir(scene_dir, names=IMAGE_DIR_NAMES):
    """First existing image dir in `scene_dir` (one isdir per candidate — no per-frame
    stat, which matters on lustre). None if none of them exist."""
    for cand in names:
        p = osp.join(scene_dir, cand)
        if osp.isdir(p):
            return p
    return None


def frustum_cover_select(centers, faxis, w2c, K, w, h, anchor, seg_scale,
                         k=6, radius=2.0, n_depth=3, return_debug=False, allowed=None,
                         look_centroid=None, ball_center=None, prepicked=None):
    """Pick k views whose view frustums together cover as much of the nearby observed
    space as possible (greedy maximum set coverage).

    Nearby space is proxied by a point cloud sampled along every candidate camera's
    optical axis at a few depths (the observed scene volume), restricted to a ball of
    radius*seg_scale around the anchor. Coverage of a view = proxy points inside its
    frustum (project -> in image, depth>0, capped). First-picked view (max single-view
    coverage) becomes index 0 = VGGT reference.

    allowed: optional iterable of frame indices the candidates are restricted to (e.g.
    out-of-segment frames only). None -> all frames (original in/out-agnostic behavior).
    """
    N = centers.shape[0]
    c_a = centers[anchor]
    rad = radius * seg_scale

    pool = range(N) if allowed is None else list(allowed)
    cand = [j for j in pool if np.linalg.norm(centers[j] - c_a) <= rad]
    if not cand:  # radius too small for the allowed pool -> use the whole allowed pool
        cand = list(pool) if allowed is not None and len(list(pool)) else [anchor]
    prepicked = list(prepicked) if prepicked else []          # fixed views whose coverage is
    for j in prepicked:                                       # subtracted first (e.g. first cam);
        if j not in cand:                                     # their coverage is counted, but they
            cand.append(j)                                    # are NOT returned in the picks.

    # "nearby space" = UNIFORM 3D grid in a ball around where the near cameras look
    # (so a single wide-FoV frustum covers only a cone, not everything). look_centroid lets
    # callers pass a TARGET-FREE centroid (e.g. candidate mean) so the unseen target
    # trajectory does not leak into the look direction; None -> whole-scene mean (legacy).
    # ball center: explicit ball_center (e.g. frame s position -> omnidirectional coverage
    # of the space around the known start, no query-FoV mask) OR the candidates' look-at
    # centroid (default; where the near cameras look).
    if ball_center is not None:
        ball_c = np.asarray(ball_center, dtype=centers.dtype)
    else:
        scene_c = centers.mean(0) if look_centroid is None else look_centroid
        look = np.stack([centers[j] + max(float(np.dot(scene_c - centers[j], faxis[j])),
                                          0.3 * seg_scale) * faxis[j] for j in cand])
        ball_c = look.mean(0)
    g = np.linspace(-rad, rad, 16)
    gx, gy, gz = np.meshgrid(g, g, g, indexing='ij')
    P = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], -1) + ball_c
    P = P[np.linalg.norm(P - ball_c, axis=1) <= rad]     # keep ball
    if P.shape[0] == 0:
        P = ball_c[None]
    # depth cap: generous enough that frustums of the near cameras reach the ball
    # (fixes forward/linear trajectories where the observed region sits ahead).
    zmax = float(np.linalg.norm(centers[cand] - ball_c, axis=1).max()) + rad

    def covered(j):
        """coverage elements = (grid point, viewing-direction bin). Rewarding the same
        point seen from a NEW direction keeps 6 views diverse even under wide FoV."""
        Xc = (w2c[j, :3, :3] @ P.T).T + w2c[j, :3, 3]      # (M,3) camera coords
        z = Xc[:, 2]
        uv = (K @ (Xc / np.clip(z, 1e-6, None)[:, None]).T).T
        ok = (z > 1e-6) & (z < zmax) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.where(ok)[0]
        if len(idx) == 0:
            return set()
        dirv = centers[j] - P[idx]                          # point -> camera direction
        dirv = dirv / (np.linalg.norm(dirv, axis=1, keepdims=True) + 1e-9)
        b = np.round(dirv * 1.5).astype(int)                # quantize each axis to {-1,0,1}
        return set((int(i), int(b[t, 0]), int(b[t, 1]), int(b[t, 2])) for t, i in enumerate(idx))

    cov = {j: covered(j) for j in cand}
    picked, union = [], set()
    for j in prepicked:                                     # subtract fixed-view coverage first
        union |= cov.get(j, set())                          # (not added to `picked`/returned)
    n_target = min(k, sum(1 for j in cand if j not in prepicked))
    while len(picked) < n_target:
        best_j = max((j for j in cand if j not in picked and j not in prepicked),
                     key=lambda x: len(cov[x] - union), default=None)
        if best_j is None:
            break
        gain = len(cov[best_j] - union)
        if gain == 0:                                       # saturated -> pad by FPS (spread)
            feat = {x: np.concatenate([centers[x] / max(seg_scale, 1e-5), faxis[x]]) for x in cand}
            remaining = [x for x in cand if x not in picked and x not in prepicked]
            if not picked and remaining:                    # nothing picked yet: seed nearest
                picked.append(min(remaining, key=lambda x: np.linalg.norm(centers[x] - c_a)))
                remaining.remove(picked[-1])
            while len(picked) < n_target and remaining:
                nxt = max(remaining,
                          key=lambda x: min(np.linalg.norm(feat[x] - feat[p]) for p in picked))
                picked.append(nxt)
                remaining.remove(nxt)
            break
        picked.append(best_j)
        union |= cov[best_j]
    if return_debug:
        pt_union = {t[0] for t in union}                    # point-index coverage (for viz)
        pt_cov = {j: {t[0] for t in cov[j]} for j in cand}
        return picked, P, pt_cov, pt_union
    return picked


def _read_meta_scenes(root, meta_name='meta.csv'):
    with open(osp.join(root, meta_name), newline='') as f:
        return [row['chunk'].strip() for row in csv.DictReader(f)]


def _read_blacklist_hashes(root):
    bl_path = osp.join(root, 'blacklist.csv')
    blocked = set()
    if osp.isfile(bl_path):
        with open(bl_path, newline='') as f:
            for row in csv.DictReader(f):
                # accept both bare hash and '<split>/<hash>' -> normalize to hash for matching
                blocked.add(row['scene'].strip().split('/')[-1])
    return blocked


def _blacklist_fingerprint(root):
    """Short content hash of blacklist.csv, for the index-cache key. The cache is built AFTER the
    blacklist filter (see _load_index), so editing blacklist.csv without this in the key would
    silently keep serving an index that still contains the newly blocked scenes."""
    bl_path = osp.join(root, 'blacklist.csv')
    if not osp.isfile(bl_path):
        return 'none'
    return hashlib.sha1(open(bl_path, 'rb').read()).hexdigest()[:8]


def _read_coverage_blacklist(path):
    """Per-SEGMENT blacklist CSV (columns: scene, segment) from scripts/dump_coverage.py.
    Returns a set of (scene_chunk, segment_key). None/missing -> empty set (no filtering)."""
    blocked = set()
    if path and osp.isfile(path):
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                blocked.add((row['scene'].strip(), row['segment'].strip()))
    return blocked


class _LazyScenes:
    """Scene-indexed view backed by lazy parsing: ds.extrinsics_list[i] parses scene i's
    transforms.json on first access (cached in ds._scene_cache) and returns the chosen field.
    Lets existing `self.<list>[scene_idx]` code work unchanged while deferring the heavy parse
    from __init__ to __getitem__."""
    def __init__(self, ds, field):
        self._ds = ds
        self._field = field

    def __getitem__(self, i):
        return self._ds._load_scene(i)[self._field]

    def __len__(self):
        return len(self._ds.scene_dir_list)


class CamDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, type='train', only_segments=None):
        self.cfg = cfg
        self.type = type
        # inference/render helper: restrict the index to these segments (list of data_name
        # strings, e.g. '1K_<hash>_0', or (chunk, seg_key) tuples). Only those scenes'
        # prompts.json are read -- no full-corpus scan, no index cache. See from_segments().
        self.only_segments = only_segments
        self._geo_idx_memo = {}      # (scene_idx, s, e) -> frustum_cover context views
        # pose / caption / avg_scale 를 어느 코퍼스에서 읽을지 (모듈 상단 _POSE_SOURCES 주석 참고).
        # 'transforms' = 기존 동작.
        self.pose_source = resolve_pose_source(cfg)
        self.root = cfg.dl3dv_root
        self.num_frames = cfg.num_frames
        self.geo_num_views = getattr(cfg, 'geo_num_views', 4)
        self.geo_hw = tuple(getattr(cfg, 'geo_image_hw', (256, 448)))
        self.geo_enabled = bool(getattr(cfg, 'geo_encoder', None))  # skip image loading for text-only
        # [new] precomputed frozen geo-latent cache (cache_geo_embeddings.py). None = OFF, i.e.
        # load images + run LagerNVS every step exactly as before. Only safe when the encoder's
        # proj is Identity (lagernvs native 768 == geo_latent_dim); otherwise proj is trainable
        # and its output must not be frozen into a file.
        self.geo_latent_cache_dir = None
        _cdir = getattr(cfg, 'geo_latent_cache_dir', None)
        if _cdir and self.geo_enabled:
            if int(getattr(cfg, 'geo_latent_dim', 768)) != 768:
                print(f"[geo cache] DISABLED: geo_latent_dim="
                      f"{getattr(cfg, 'geo_latent_dim')} != 768 -> GeoEncoder.proj is a trainable "
                      f"Linear, its output must not be cached")
            else:
                sub = ('first_cam_included' if getattr(cfg, 'geo_first_view_target_s', False)
                       else 'first_cam_not_included')
                self.geo_latent_cache_dir = osp.join(_cdir, sub)
                print(f"[geo cache] reading {self.geo_latent_cache_dir}/<iK>/<data_name>.pt "
                      f"(miss -> on-the-fly LagerNVS)")
        # [new] per-context-view camera embedding concatenated onto the geo tokens.
        # None = OFF (unchanged geo condition). 'relfirst' = 11-d pose relative to the target's
        # first camera (per-VIEW, broadcast over that view's patches); 'plucker' = 6-d Plücker ray
        # per PATCH token in the same frame. See conf/config.yaml and _geo_cam_param /
        # _geo_cam_plucker below.
        self.geo_cam_embed = getattr(cfg, 'geo_cam_embed', None)
        if self.geo_cam_embed not in (None, 'relfirst', 'plucker'):
            raise ValueError(f"geo_cam_embed must be null | 'relfirst' | 'plucker', "
                             f"got {self.geo_cam_embed!r}")
        # [new] test-time probe: force K of the V context views to be TARGET-SEGMENT cameras.
        # null/0 = OFF (unchanged). See _mix_inseg_context and conf/config.yaml.
        self.geo_test_inseg_k = getattr(cfg, 'geo_test_inseg_k', None) or 0
        if self.geo_test_inseg_k and self.geo_latent_cache_dir is not None:
            # 캐시 키는 data_name 뿐이라 context view 가 바뀐 걸 구분 못 한다 -> 반드시 끈다.
            print(f"[geo cache] DISABLED: geo_test_inseg_k={self.geo_test_inseg_k} changes the "
                  f"context views, but the cache is keyed by segment only")
            self.geo_latent_cache_dir = None
        # geo context-view sampling (leakage ablation): 'even' (in-segment) | 'hybrid'
        self.geo_view_sampling = getattr(cfg, 'geo_view_sampling', 'even')
        self.geo_num_inseg = getattr(cfg, 'geo_num_inseg', 3)
        self.geo_num_covis = getattr(cfg, 'geo_num_covis', 3)
        self.geo_inseg_span = getattr(cfg, 'geo_inseg_span', None)
        self.covis_radius = getattr(cfg, 'geo_covis_radius', 1.0)
        self.covis_theta0 = getattr(cfg, 'geo_covis_theta0', 10.0)
        self.covis_max_axis_deg = getattr(cfg, 'geo_covis_max_axis_deg', 60.0)
        self.covis_topM = getattr(cfg, 'geo_covis_topM', 32)
        # frustum_cover mode
        self.geo_cover_k = getattr(cfg, 'geo_cover_k', 6)
        self.geo_cover_radius = getattr(cfg, 'geo_cover_radius', 2.0)
        self.geo_cover_ndepth = getattr(cfg, 'geo_cover_ndepth', 3)
        self.geo_cover_out_of_seg = getattr(cfg, 'geo_cover_out_of_seg', False)
        # restrict out-of-segment context to frames BEFORE the target segment (index < s)
        # only, instead of the longer side. Requires the target segment to have frames
        # before it -> the target segment can be the 2nd segment onward.
        self.geo_cover_before_only = getattr(cfg, 'geo_cover_before_only', False)
        self.geo_posed = getattr(cfg, 'geo_posed', False)
        self.geo_shuffle_order = getattr(cfg, 'geo_shuffle_order', False)
        # honest selection: anchor at the target's FIRST frame only (known at inference);
        # radius scaled by CONTEXT (not the unseen rest of the target segment).
        # [renamed 2026-07-31] geo_anchor_first_frame -> geo_cover_centered_at_s. The old name read
        # as "the first frame goes in as an anchor VIEW", which is what geo_first_view_target_s
        # does; this flag only centers the frustum_cover search ball on frame s and never adds a
        # view. Old key still honored (deprecated) so pre-rename configs / CLI overrides work.
        _legacy = getattr(cfg, 'geo_anchor_first_frame', None)
        _cur = getattr(cfg, 'geo_cover_centered_at_s', None)
        if _cur is None and _legacy is not None:
            print("[cfg] geo_anchor_first_frame is deprecated -> using it as geo_cover_centered_at_s"
                  f"={_legacy}")
        self.geo_cover_centered_at_s = bool(_cur if _cur is not None
                                            else (_legacy if _legacy is not None else False))
        # geo context view0 = target segment's FIRST camera s (rest = out-of-seg retrieved)
        # -> LagerNVS anchors to s, aligning the geo latent frame with the target frame.
        self.geo_first_view_target_s = getattr(cfg, 'geo_first_view_target_s', False)
        # [new] test-time probe: swap the geo context to ANOTHER SEGMENT OF THE SAME SCENE.
        # 'inscene' = donor is the (position+geo_swap_shift)-th other segment of this scene, so the
        # context views stay real images of the same scene in the same world frame (nothing goes
        # out of distribution) and the ONLY thing that changes is WHICH REGION they cover.
        # geo_swap_keep_first puts view0 back to the target's frame s, so the anchor -- and with it
        # the frame/scale link to the generated trajectory -- is untouched. Reading:
        #   predicted trajectory follows the DONOR region -> the model is context-driven
        #   predicted trajectory does not move                -> it is text-driven
        # geo_test_inseg_k measures the same worry along the leakage axis; this is the orthogonal
        # (context-content) axis. Test-time only -- the model never saw this during training.
        self.geo_swap_mode = getattr(cfg, 'geo_swap_mode', None)
        if self.geo_swap_mode not in (None, 'inscene'):
            raise ValueError(f"geo_swap_mode must be null | 'inscene', got {self.geo_swap_mode!r}")
        self.geo_swap_shift = int(getattr(cfg, 'geo_swap_shift', 1) or 1)
        self.geo_swap_keep_first = bool(getattr(cfg, 'geo_swap_keep_first', True))
        self._scene2samples = None      # lazily built: scene_idx -> [sample idx, ...]
        if self.geo_swap_mode and self.geo_latent_cache_dir is not None:
            # geo_test_inseg_k (:289) 와 같은 이유 -- 캐시 키가 data_name 뿐이라 context view 가
            # 바뀐 것을 구분하지 못한다. 끄지 않으면 swap 이 조용히 무효가 된다.
            print(f"[geo cache] DISABLED: geo_swap_mode={self.geo_swap_mode} changes the "
                  f"context views, but the cache is keyed by segment only")
            self.geo_latent_cache_dir = None
        # [new] attach the chosen context view indices (+ their c2w) to every item so an offline
        # script can measure how close the generated trajectory sits to the context cameras.
        # Off by default: on a cache HIT it costs the frustum_cover search the cache exists to skip.
        self.geo_return_idxs = bool(getattr(cfg, 'geo_return_idxs', False))
        self.max_scenes = getattr(cfg, 'max_scenes', None)   # limit #scenes (e.g. smoke test)

        # scene-level (indexed by scene) and sample-level (per prompt segment)
        self.extrinsics_list = []   # [(N,4,4)]   (eager) or _LazyScenes (lazy)
        self.intrinsics_list = []   # [(N,3,3)]
        self.frame_files_list = []  # [[path,...]]
        self.hw_list = []           # [(h,w)]
        self.scene_dir_list = []    # [scene_dir]  (for avg_scale json lookup)
        self.samples = []           # [(scene_idx, start, end, caption, data_name)]
        self._scene_cache = {}      # lazy: scene_idx -> {w2c,intr,frame_files,hw}
        # lazy_dataset (default True): __init__ only builds the sample/scene index (from a persisted
        # cache when available); scene poses/paths are parsed on demand in __getitem__ + cached.
        self.lazy = getattr(cfg, 'lazy_dataset', True) or only_segments is not None
        if self.lazy:
            self._load_index()
            self.extrinsics_list = _LazyScenes(self, 'w2c')
            self.intrinsics_list = _LazyScenes(self, 'intr')
            self.frame_files_list = _LazyScenes(self, 'frame_files')
        else:
            self.load_data()

    # ---- pose_source 별 경로/파싱 ------------------------------------------------------
    # 아래 4개가 'transforms' 와 'da3' 의 유일한 차이점이다. 세그먼트 경계와 키('0','1',...)는
    # 두 prompts.json 이 동일하므로 seg 리스트/샘플 인덱싱 로직은 공유한다.

    def _prompts_path(self, scene_dir):
        return osp.join(scene_dir, 'da3', 'prompts.json') if self.pose_source == 'da3' \
            else osp.join(scene_dir, 'prompts.json')

    def _avg_scale_dir(self, scene_dir):
        return osp.join(scene_dir, 'da3', 'avg_scale') if self.pose_source == 'da3' \
            else osp.join(scene_dir, 'avg_scale')

    def _scene_probe(self, scene_dir):
        """인덱스 빌드용 경량 프로브 -> (n_frames, h, w). 포즈 전체를 파싱하지 않는다."""
        if self.pose_source == 'da3':
            K = np.load(osp.join(scene_dir, 'da3', 'pose.npz'))['intrinsics']
            n = int(K.shape[0])
            return n, int(round(float(K[0, 1, 2]) * 2)), int(round(float(K[0, 0, 2]) * 2))
        tj = json.load(open(osp.join(scene_dir, 'transforms.json')))
        return len(tj['frames']), int(tj['h']), int(tj['w'])

    def _parse_da3(self, scene_dir):
        """<scene>/da3/pose.npz -> (w2c (N,4,4), intr (N,3,3), frame_files, (h,w)).

        extrinsics 는 이미 **OpenCV w2c** 라 _parse_transforms 의 GL->CV flip + inv 가 둘 다
        필요 없다 (모듈 상단 _POSE_SOURCES 주석의 실측 근거 참고). (N,3,4) 를 (N,4,4) 로 채운다.
        frame_files 는 이미지 디렉토리를 정렬해 쓴다 — da3 는 파일명을 따로 저장하지 않고
        (predictions.npz 는 1000 scene 중 3개에만 있다), 정렬 순서가 transforms.json 의
        file_path 정렬 순서와 일치하는 것은 실측 확인했다."""
        z = np.load(osp.join(scene_dir, 'da3', 'pose.npz'))
        E = torch.from_numpy(np.asarray(z['extrinsics'], dtype=np.float32))     # (N,3,4) w2c
        intr = torch.from_numpy(np.asarray(z['intrinsics'], dtype=np.float32))  # (N,3,3)
        n = E.shape[0]
        w2c = torch.eye(4, dtype=torch.float32).repeat(n, 1, 1)
        w2c[:, :3, :4] = E
        h = int(round(float(intr[0, 1, 2]) * 2))
        w = int(round(float(intr[0, 0, 2]) * 2))

        img_dir = scene_image_dir(scene_dir,
                                  getattr(self.cfg, 'image_dir_names', IMAGE_DIR_NAMES))
        if img_dir is None:
            frame_files = [osp.join(scene_dir, 'images', f'frame_{i + 1:05d}.png')
                           for i in range(n)]
        else:
            names = sorted(f for f in os.listdir(img_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))
            if len(names) != n:
                raise ValueError(f"{scene_dir}: da3 pose has {n} frames but {img_dir} has "
                                 f"{len(names)} images")
            frame_files = [osp.join(img_dir, f) for f in names]
        return w2c, intr, frame_files, (h, w)

    def _load_scene(self, i):
        """Lazily parse + cache a scene's poses (used by _LazyScenes and getitem helpers)."""
        c = self._scene_cache.get(i)
        if c is None:
            sd = self.scene_dir_list[i]
            if self.pose_source == 'da3':
                w2c, intr, ff, hw = self._parse_da3(sd)
            else:
                w2c, intr, ff, hw = self._parse_transforms(sd, osp.join(sd, 'transforms.json'))
            c = {'w2c': w2c, 'intr': intr, 'frame_files': ff, 'hw': hw}
            self._scene_cache[i] = c
        return c

    def _load_index(self):
        """Build (samples, scene_dir_list, hw_list) — the lightweight index — reading only
        prompts.json + a per-scene (n,h,w) probe. Persisted to <root>/.latentcam_index/<key>.pt
        so subsequent runs load instantly instead of re-scanning all scenes."""
        meta_name = getattr(self.cfg, 'meta_csv', 'meta.csv')
        cov_path = getattr(self.cfg, 'coverage_blacklist_path', None)
        key = (f"{meta_name}__nf{self.num_frames}__bo{int(self.geo_cover_before_only)}"
               f"__k{self.geo_cover_k}__cb{osp.basename(cov_path) if cov_path else 'none'}"
               f"__ms{self.max_scenes}__bl{_blacklist_fingerprint(self.root)}")
        if self.pose_source != 'transforms':
            # 기본값일 때는 키를 건드리지 않는다 -> 기존 캐시 파일이 그대로 재사용된다.
            key += f"__ps{self.pose_source}"
        cache_dir = osp.join(self.root, '.latentcam_index'); os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, key.replace('/', '_') + '.pt')
        if self.only_segments is not None:
            self._load_index_subset()
            return
        if osp.isfile(cache_path):
            idx = torch.load(cache_path, weights_only=False)
            self.samples = idx['samples']; self.hw_list = idx['hw_list']
            # scene dirs are stored RELATIVE to self.root ('scene_chunks') so the cache
            # survives moving the dataset. Legacy caches hold absolute 'scene_dir_list';
            # accept them only if they still resolve, else fall through and rebuild.
            if 'scene_chunks' in idx:
                self.scene_dir_list = [osp.join(self.root, c) for c in idx['scene_chunks']]
            else:
                self.scene_dir_list = idx['scene_dir_list']
            if self.scene_dir_list and not osp.isdir(self.scene_dir_list[0]):
                print(f"[index cache] STALE (scene dirs do not exist under {self.root}) "
                      f"-> rebuilding: {cache_path}")
                self.samples = []; self.scene_dir_list = []; self.hw_list = []
            else:
                print(f"[index cache] {len(self.samples)} samples / "
                      f"{len(self.scene_dir_list)} scenes <- {cache_path}")
                return
        scenes = _read_meta_scenes(self.root, meta_name)
        blocked = _read_blacklist_hashes(self.root)
        cov_black = _read_coverage_blacklist(cov_path)
        print(f"[index build] scanning {len(scenes)} scenes in {meta_name} "
              f"(prompts + n,h,w probe; cached to {cache_path})")
        for chunk in tqdm(scenes):
            if chunk.split('/')[-1] in blocked:
                continue
            scene_dir = osp.join(self.root, chunk)
            pose_path = (osp.join(scene_dir, 'da3', 'pose.npz') if self.pose_source == 'da3'
                         else osp.join(scene_dir, 'transforms.json'))
            pj_path = self._prompts_path(scene_dir)
            if not (osp.isfile(pose_path) and osp.isfile(pj_path)):
                continue
            try:
                prompts = json.load(open(pj_path))
                n, h, w = self._scene_probe(scene_dir)
            except Exception:
                continue
            scene_idx = len(self.scene_dir_list)
            added = 0
            for seg_key, seg in prompts.items():
                fi = seg.get('frame_idx')
                if not fi or len(fi) != 2:
                    continue
                s, e = int(fi[0]), int(fi[1])
                if e > n or (e - s) < self.num_frames:
                    continue
                if (chunk, seg_key) in cov_black:
                    continue
                if self.geo_cover_before_only and s < self.geo_cover_k:
                    continue
                pcs = seg.get('prompt_camera_with_scene_video')
                caption = pcs.get('concise', "") if isinstance(pcs, dict) else (pcs or "")
                self.samples.append((scene_idx, s, e, caption,
                                     f"{chunk.replace('/', '_')}_{seg_key}"))
                added += 1
            if added > 0:
                self.scene_dir_list.append(scene_dir)
                self.hw_list.append((h, w))
                if self.max_scenes is not None and len(self.scene_dir_list) >= self.max_scenes:
                    break
        torch.save({'samples': self.samples, 'hw_list': self.hw_list,
                    'scene_chunks': [osp.relpath(d, self.root) for d in self.scene_dir_list]},
                   cache_path)
        print(f"[index build] {len(self.samples)} samples / {len(self.scene_dir_list)} scenes "
              f"-> saved {cache_path}")

    def _load_index_subset(self):
        """Index ONLY self.only_segments (inference / rendering). Reads meta.csv once to map the
        flattened scene name back to its chunk, then opens prompts.json for just those scenes.
        Cost is O(#requested segments), not O(#corpus): no full scan, no index cache, and the
        per-segment filters (num_frames / before_only / blacklists) are deliberately NOT applied
        -- the caller asked for these exact segments."""
        meta_name = getattr(self.cfg, 'meta_csv', 'meta.csv')
        flat2chunk = {c.replace('/', '_'): c for c in _read_meta_scenes(self.root, meta_name)}
        want = []
        for seg in self.only_segments:
            if isinstance(seg, (tuple, list)):
                want.append((seg[0], str(seg[1])))
            else:
                flat, seg_key = str(seg).rsplit('_', 1)
                if flat not in flat2chunk:
                    raise KeyError(f"segment '{seg}': scene '{flat}' not in {meta_name}")
                want.append((flat2chunk[flat], seg_key))
        chunk_idx = {}
        for chunk, seg_key in want:
            scene_dir = osp.join(self.root, chunk)
            if chunk not in chunk_idx:
                _n, _h, _w = self._scene_probe(scene_dir)
                chunk_idx[chunk] = len(self.scene_dir_list)
                self.scene_dir_list.append(scene_dir)
                self.hw_list.append((_h, _w))
            seg = json.load(open(self._prompts_path(scene_dir)))[seg_key]
            s, e = int(seg['frame_idx'][0]), int(seg['frame_idx'][1])
            pcs = seg.get('prompt_camera_with_scene_video')
            caption = pcs.get('concise', "") if isinstance(pcs, dict) else (pcs or "")
            self.samples.append((chunk_idx[chunk], s, e, caption,
                                 f"{chunk.replace('/', '_')}_{seg_key}"))
        print(f"[index subset] {len(self.samples)} samples / {len(self.scene_dir_list)} scenes "
              f"(requested segments only)")

    @classmethod
    def from_segments(cls, cfg, segments, type='test'):
        """Lightweight CamDataset over just `segments` -- for inference/rendering where building
        the full corpus index is wasted work. Everything else (__getitem__, geo context sampling,
        normalization) behaves exactly as in training."""
        return cls(cfg, type=type, only_segments=list(segments))

    def load_data(self):
        meta_name = getattr(self.cfg, 'meta_csv', 'meta.csv')
        scenes = _read_meta_scenes(self.root, meta_name)
        blocked = _read_blacklist_hashes(self.root)
        cov_black = _read_coverage_blacklist(getattr(self.cfg, 'coverage_blacklist_path', None))
        print(f"Loading DL3DV: {len(scenes)} scenes in {meta_name}, {len(blocked)} blacklisted"
              + (f", {len(cov_black)} segments coverage-blacklisted" if cov_black else ""))

        for chunk in tqdm(scenes):
            scene_hash = chunk.split('/')[-1]
            if scene_hash in blocked:
                continue
            scene_dir = osp.join(self.root, chunk)
            tj_path = osp.join(scene_dir, 'transforms.json')
            pose_path = (osp.join(scene_dir, 'da3', 'pose.npz') if self.pose_source == 'da3'
                         else tj_path)
            pj_path = self._prompts_path(scene_dir)
            if not (osp.isfile(pose_path) and osp.isfile(pj_path)):
                continue
            try:
                extr, intr, frame_files, (h, w) = (
                    self._parse_da3(scene_dir) if self.pose_source == 'da3'
                    else self._parse_transforms(scene_dir, tj_path))
                prompts = json.load(open(pj_path))
            except Exception:
                continue

            n = extr.shape[0]
            scene_idx = len(self.extrinsics_list)
            added = 0
            for seg_key, seg in prompts.items():
                fi = seg.get('frame_idx')
                if not fi or len(fi) != 2:
                    continue
                s, e = int(fi[0]), int(fi[1])
                if e > n or (e - s) < self.num_frames:
                    continue
                if (chunk, seg_key) in cov_black:   # coverage-filtered out
                    continue
                # before-only geo context: need enough frames before s (excludes the
                # first segment -> target segment can be the 2nd segment onward).
                if self.geo_cover_before_only and s < self.geo_cover_k:
                    continue
                pcs = seg.get('prompt_camera_with_scene_video')
                caption = pcs.get('concise', "") if isinstance(pcs, dict) else (pcs or "")
                self.samples.append((scene_idx, s, e,
                                     caption, f"{chunk.replace('/', '_')}_{seg_key}"))
                added += 1

            if added > 0:
                self.extrinsics_list.append(extr)
                self.intrinsics_list.append(intr)
                self.frame_files_list.append(frame_files)
                self.hw_list.append((h, w))
                self.scene_dir_list.append(scene_dir)
                if self.max_scenes is not None and len(self.extrinsics_list) >= self.max_scenes:
                    break

        print(f"DL3DV: {len(self.samples)} segment samples from "
              f"{len(self.extrinsics_list)} scenes")

    def _parse_transforms(self, scene_dir, tj_path):
        with open(tj_path) as f:
            tj = json.load(f)
        w, h = int(tj['w']), int(tj['h'])
        fx, fy = float(tj['fl_x']), float(tj['fl_y'])
        cx, cy = float(tj['cx']), float(tj['cy'])

        frames = sorted(tj['frames'], key=lambda fr: fr['file_path'])
        c2w = torch.tensor([fr['transform_matrix'] for fr in frames], dtype=torch.float32)
        c2w = c2w @ _GL2CV.to(c2w.dtype)               # OpenGL -> OpenCV c2w
        w2c = torch.linalg.inv(c2w)                    # (N,4,4) w2c
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
        intr = K.unsqueeze(0).repeat(w2c.shape[0], 1, 1)

        img_dir = scene_image_dir(scene_dir,
                                  getattr(self.cfg, 'image_dir_names', IMAGE_DIR_NAMES))
        frame_files = [osp.join(img_dir, osp.basename(fr['file_path'])) if img_dir
                       else osp.join(scene_dir, fr['file_path']) for fr in frames]
        return w2c, intr, frame_files, (h, w)

    @staticmethod
    def _even_indices(n_total, n_pick):
        return np.linspace(0, n_total - 1, n_pick).round().astype(int).tolist()

    @staticmethod
    def _pw_gauss(theta_deg, t0=10.0, s1=5.0, s2=15.0):
        d = theta_deg - t0
        sig = s1 if theta_deg <= t0 else s2
        return float(np.exp(-(d * d) / (2 * sig * sig)))

    def _covis_retrieve(self, centers, faxis, w2c, K, w, h, s, e, k_views):
        """MVSNet-style out-of-segment co-visibility retrieval anchored at frame s.
        frustum/FoV gate + triangulation-angle score, then farthest-point sampling on
        the *viewing direction toward the look-at point X* so picks surround the scene
        point instead of bunching (avoids clustered selections)."""
        c_s, f_s = centers[s], faxis[s]
        seg_scale = np.linalg.norm(centers[s:e] - c_s, axis=1).mean()
        if seg_scale < 1e-6:
            return []
        centroid = centers.mean(0)
        d = max(float(np.dot(centroid - c_s, f_s)), 0.5 * seg_scale)
        X = c_s + d * f_s
        cos_max = np.cos(np.deg2rad(self.covis_max_axis_deg))
        idxs, scores, dirs = [], [], []
        for j in range(centers.shape[0]):
            if s <= j < e:
                continue
            if np.linalg.norm(centers[j] - c_s) > self.covis_radius * seg_scale:
                continue
            if np.dot(f_s, faxis[j]) < cos_max:
                continue
            Xc = w2c[j, :3, :3] @ X + w2c[j, :3, 3]
            z = Xc[2]
            if z <= 1e-6:
                continue
            uv = K @ (Xc / z)
            if not (0 <= uv[0] < w and 0 <= uv[1] < h):     # frustum/FoV gate
                continue
            v1, v2 = c_s - X, centers[j] - X
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            theta = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
            idxs.append(j)
            scores.append(self._pw_gauss(theta, self.covis_theta0))
            dirs.append(v2 / (np.linalg.norm(v2) + 1e-9))   # dir from X toward camera
        if not idxs:
            return []
        order = np.argsort(scores)[::-1]                    # best-score candidate = FPS seed
        dirs = np.asarray(dirs)[order]
        idxs = [idxs[o] for o in order]
        picks = [0]
        while len(picks) < min(k_views, len(idxs)):
            dmin = np.min([np.linalg.norm(dirs - dirs[p], axis=1) for p in picks], axis=0)
            for p in picks:
                dmin[p] = -1.0
            picks.append(int(np.argmax(dmin)))
        return [idxs[p] for p in picks]

    def _sample_geo_random_inseg(self, s, e):
        """Anchor (frame s, = VGGT reference) fixed + (geo_num_views-1) other frames
        randomly sampled from (s, e) — re-drawn every access (augmentation). Still
        in-segment so it does not remove leakage, but avoids memorizing fixed frames.
        (Ordering is irrelevant to VGGT; anchor kept at index 0 like the even baseline.)"""
        k = self.geo_num_views
        pool = list(range(s + 1, e))
        if len(pool) >= k - 1:
            rest = random.sample(pool, k - 1)
        else:                                          # tiny segment fallback
            rest = [random.randrange(s, e) for _ in range(k - 1)]
        return [s] + sorted(rest)

    @staticmethod
    def _np_context_scale(centers, side, num_frames):
        """Mean per-window camera movement over the context `side`, chunked into
        num_frames windows (numpy; mirrors _cam_dist_mean_context). Target-free."""
        T = num_frames
        chunks = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]
        if not chunks and len(side) >= 2:
            chunks = [side]
        if not chunks:
            return None
        sc = [float(np.linalg.norm(centers[ch] - centers[ch[0]], axis=1).mean()) for ch in chunks]
        return max(float(np.mean(sc)), 1e-5)

    def _sample_geo_frustum_cover(self, scene_idx, s, e):
        """Memoized wrapper around _frustum_cover_uncached. The greedy search is deterministic
        in (scene_idx, s, e) and costs ~54 ms/segment, so re-running it every epoch (and twice
        per item under scale_mode 'geo_lagernvs', which needs the same views to build the
        divisor) is pure waste. One dict per worker process, ~40k tiny tuples."""
        k = (scene_idx, s, e)
        v = self._geo_idx_memo.get(k)
        if v is None:
            v = tuple(self._frustum_cover_uncached(scene_idx, s, e))
            self._geo_idx_memo[k] = v
        return list(v)

    def _frustum_cover_uncached(self, scene_idx, s, e):
        """Frustum max-coverage: k views covering the most nearby space. When
        geo_cover_out_of_seg, candidates are restricted to the LONGER out-of-segment side
        (no target frames -> leakage-free). Search anchor/scale:
          - geo_cover_centered_at_s=True (honest): the coverage ball is CENTERED on the target's
            FIRST frame s (known at inference); radius scaled by CONTEXT movement; look_centroid =
            candidate mean. -> selection never uses the unseen target [s+1:e].
          - False (legacy): anchor near the target midpoint, radius from the target seg_scale.
        Neither setting puts frame s (or any target frame) INTO the returned views -- :668 filters
        s out and only geo_first_view_target_s (:682) prepends it. This flag steers WHERE the
        greedy search looks, nothing more."""
        w2c = self.extrinsics_list[scene_idx].numpy().astype(np.float64)
        intr = self.intrinsics_list[scene_idx].numpy().astype(np.float64)
        c2w = np.linalg.inv(w2c)
        centers, faxis = c2w[:, :3, 3], c2w[:, :3, 2]
        K = intr[0]
        h, w = self.hw_list[scene_idx]
        N = centers.shape[0]
        allowed, anchor, look_centroid = None, s, None
        seg_scale = max(float(np.linalg.norm(centers[s:e] - centers[s], axis=1).mean()), 1e-5)
        ball_center = None
        if self.geo_cover_out_of_seg:
            before, after = list(range(0, s)), list(range(e, N))
            # before_only: out-of-target context must come from earlier indices (< s);
            # else pick the longer out-of-segment side (original behavior).
            if self.geo_cover_before_only:
                allowed = before
            else:
                allowed = before if len(before) >= len(after) else after
            if self.geo_cover_centered_at_s:
                anchor = s                                  # known first frame only
                cs = self._np_context_scale(centers, allowed, self.num_frames) if allowed else None
                if cs is not None:
                    seg_scale = cs                          # scale from context, not target
                # cover a ball AROUND frame s (omnidirectional; no query-FoV mask) so the
                # context is not biased to frame s's viewing direction.
                ball_center = centers[s]
            elif allowed:                                    # legacy: anchor near target mid
                mid = (s + e) // 2
                anchor = min(allowed, key=lambda j: np.linalg.norm(centers[j] - centers[mid]))
        # when geo_first_view_target_s: view0 = the target's FIRST camera s (aligns the geo
        # latent frame with the generation model's frame) + (k-1) out-of-seg retrieved.
        k_retr = (self.geo_cover_k - 1) if self.geo_first_view_target_s else self.geo_cover_k
        # subtract the fixed first camera (frame s)'s coverage before the greedy, so the
        # remaining k-1 maximize RESIDUAL coverage (what s does not already cover).
        _prepick = [s] if (self.geo_first_view_target_s
                           and getattr(self.cfg, 'geo_cover_subtract_first', False)) else None
        picks = frustum_cover_select(centers, faxis, w2c, K, w, h, anchor=anchor,
                                     seg_scale=seg_scale, k=k_retr,
                                     radius=self.geo_cover_radius, n_depth=self.geo_cover_ndepth,
                                     allowed=allowed, look_centroid=look_centroid,
                                     ball_center=ball_center, prepicked=_prepick)
        picks = [p for p in picks if p != s]                # s goes in as view0, not a retrieval
        # frustum_cover can return < k when few candidates fall in the radius ball -> pad to
        # exactly k_retr (else collate fails to stack (V,3,H,W)). Pad from the candidate pool.
        pool = [p for p in (list(allowed) if allowed else list(range(N))) if p != s]
        if len(picks) < k_retr and pool:
            for j in [pool[int(round(x))] for x in np.linspace(0, len(pool) - 1, k_retr)]:
                if j not in picks:
                    picks.append(j)
                if len(picks) >= k_retr:
                    break
        while len(picks) < k_retr and picks:                # tiny pool -> repeat
            picks.append(picks[-1])
        picks = picks[:k_retr]
        if self.geo_first_view_target_s:
            picks = [s] + picks                             # view0 = frame s -> total geo_cover_k
        return picks[:self.geo_cover_k]

    def _sample_geo_hybrid(self, scene_idx, s, e):
        """in-segment near-anchor views (grounding, minimal leak) + out-of-segment
        co-visibility views (decorrelated). Falls back to in-segment if covis is short."""
        w2c = self.extrinsics_list[scene_idx].numpy().astype(np.float64)   # (N,4,4)
        intr = self.intrinsics_list[scene_idx].numpy().astype(np.float64)  # (N,3,3)
        c2w = np.linalg.inv(w2c)
        centers = c2w[:, :3, 3]
        faxis = c2w[:, :3, 2]                              # OpenCV forward (+Z col)
        K = intr[0]                                        # DL3DV: shared intrinsics
        h, w = self.hw_list[scene_idx]

        # in-segment: geo_num_inseg evenly within [s, s+span] (near the anchor)
        span = self.geo_inseg_span if self.geo_inseg_span is not None else max(self.num_frames // 8, self.geo_num_inseg)
        span = int(min(span, e - 1 - s))
        inseg = [s + int(round(i)) for i in np.linspace(0, span, self.geo_num_inseg)]

        covis = self._covis_retrieve(centers, faxis, w2c, K, w, h, s, e, self.geo_num_covis)

        idxs = inseg + covis
        need = (self.geo_num_inseg + self.geo_num_covis) - len(idxs)
        if need > 0:   # degenerate scene (too few covis candidates) -> pad in-segment
            idxs += [s + i for i in self._even_indices(e - s, need)]
        return idxs

    def _cam_dist_mean_scale(self, extrinsics):
        """scale_mode='cam_dist_mean': mean(||camera center_i - center_0||) over the given
        cameras. This is a CAMERA-distance scale -- not to be confused with 'avg_scale',
        which is the stored point-cloud distance."""
        e0_inv = torch.linalg.inv(extrinsics[0].float())
        normalized = extrinsics.float() @ e0_inv.unsqueeze(0)
        centers = torch.linalg.inv(normalized)[:, :3, 3]
        return centers.norm(dim=-1).mean().clamp(min=1e-5).unsqueeze(0)

    def _avg_scale(self, scene_idx, seg_key):
        """scale_mode='avg_scale' (SCVideo original): the STORED point-cloud avg_scale,
        scene_dir/avg_scale/<seg_key>.json = mean(||scene point - first camera||).
        Mirrors dataset_large.py. Returns None if the json is missing (caller falls back).
        pose_source='da3' 이면 <scene>/da3/avg_scale/<seg>.json 을 읽는다 (da3 예측 depth 로
        만든 값이라 da3 pose 와 같은 스케일 공간에 있다)."""
        p = osp.join(self._avg_scale_dir(self.scene_dir_list[scene_idx]), f'{seg_key}.json')
        if not osp.isfile(p):
            return None
        try:
            return torch.tensor([float(json.load(open(p)))]).clamp(min=1e-5)
        except Exception:
            return None

    def _first_farthest_scale(self, extrinsics):
        """LagerNVS-style scale = 1.35 * max(||camera center - FIRST camera||) over the segment
        (relative to frame 0). Matches LagerNVS normalize(): scene_scale = 1.35*max ||t||."""
        e0_inv = torch.linalg.inv(extrinsics[0].float())
        normalized = extrinsics.float() @ e0_inv.unsqueeze(0)
        centers = torch.linalg.inv(normalized)[:, :3, 3]
        return (1.35 * centers.norm(dim=-1).max()).clamp(min=1e-5).unsqueeze(0)

    def _geo_lagernvs_scale(self, scene_idx, s, e):
        """FULL-alignment scale = 1.35*max(||geo-context camera center - ANCHOR||), i.e. the
        scene_scale LagerNVS.build_cam_token uses for the geo views.
        Uses the same geo selection so target & geo latent are in one frame+scale. (1).

        anchor (cfg.geo_lagernvs_anchor, see conf/config.yaml for the full rationale):
          's'     (default, original behaviour) -> frame s
          'view0' -> geo_idxs[0], i.e. the very camera build_cam_token re-anchors to
                     (models/geo_encoder.py:109-110). Makes the injected scalar
                     tok = max||c_geo - c_geo[0]|| / D exactly 1/1.35 = 0.7407 even when
                     geo_first_view_target_s is false. With it true the two anchors are the
                     same camera, so both settings return a bit-identical value.
        """
        geo_idxs = self._sample_geo_frustum_cover(scene_idx, s, e)      # [s] + out-of-seg
        c2w = torch.linalg.inv(self.extrinsics_list[scene_idx].float())
        centers = c2w[:, :3, 3]
        anchor = getattr(self.cfg, 'geo_lagernvs_anchor', 's')
        if anchor == 's':
            a = centers[s]
        elif anchor == 'view0':
            # the divisor is computed here, BEFORE the optional geo_shuffle_order permutation
            # at __getitem__, so a shuffled run would anchor on a different camera than the
            # encoder ends up using. Refuse the combination rather than silently mis-anchor.
            if self.geo_shuffle_order:
                raise ValueError("geo_lagernvs_anchor='view0' is incompatible with "
                                 "geo_shuffle_order=True (the encoder's view0 would differ "
                                 "from the one the divisor was measured against)")
            a = centers[geo_idxs[0]]
        else:
            raise ValueError(f"geo_lagernvs_anchor must be 's' | 'view0', got {anchor!r}")
        d = (centers[geo_idxs] - a).norm(dim=-1)                       # dist from the anchor
        return (1.35 * d.max()).clamp(min=1e-5).unsqueeze(0)

    def _context_window_scale(self, scene_idx, s, e, window_fn=None):
        """Divisor measured on the CONTEXT RANGE = the longer out-of-target-segment side,
        chunked into num_frames windows; `window_fn(w2c_window) -> (1,)` is evaluated per
        window and the windows are averaged. Target-view-excluded -> leakage-free and
        reproducible at inference from the context views only. Returns None when the context
        range is too short (caller falls back).

          window_fn=_cam_dist_mean_scale    -> scale_mode 'context_longer'
                                               mean||center - window's first center||
          window_fn=_first_farthest_scale   -> scale_mode 'ctx_longer_135max'
                                               1.35*max||center - window's first center||
                                               (LagerNVS's own denominator form, measured on
                                               num_frames-sized context windows so it lands in
                                               the same units LagerNVS normalizes context with)
        """
        window_fn = window_fn or self._cam_dist_mean_scale
        w2c_all = self.extrinsics_list[scene_idx]              # (N,4,4) w2c
        N = w2c_all.shape[0]
        T = self.num_frames
        side = list(range(0, s)) if s >= (N - e) else list(range(e, N))
        chunks = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]  # full windows
        if not chunks and len(side) >= 2:
            chunks = [side]                                    # short side -> one partial window
        if not chunks:
            return None
        scales = [window_fn(w2c_all[ch]) for ch in chunks]
        return torch.stack(scales, 0).mean(0).clamp(min=1e-5)  # (1,)

    def _cam_dist_mean_context(self, scene_idx, s, e):
        """scale_mode='context_longer' (unchanged): see _context_window_scale."""
        return self._context_window_scale(scene_idx, s, e, self._cam_dist_mean_scale)

    def _first_farthest_context(self, scene_idx, s, e):
        """scale_mode='ctx_longer_135max': see _context_window_scale."""
        return self._context_window_scale(scene_idx, s, e, self._first_farthest_scale)

    def _load_images(self, frame_files, idxs):
        H, W = self.geo_hw
        imgs = []
        for i in idxs:
            im = Image.open(frame_files[i]).convert('RGB').resize((W, H), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        return torch.stack(imgs)

    def _target_frame_idxs(self, s, e):
        """__getitem__ 이 실제로 생성 대상으로 삼는 프레임 인덱스 (segment 가 num_frames 보다
        길면 균등 subsample 한 그것). _mix_inseg_context 가 '진짜 target 프레임'만 context 로
        넣기 위해 쓴다."""
        idxs = list(range(s, e))
        if len(idxs) > self.num_frames:
            idxs = [idxs[i] for i in self._even_indices(len(idxs), self.num_frames)]
        return idxs

    def _mix_inseg_context(self, geo_idxs, s, e):
        """[new] geo_test_inseg_k=K: context V 장 중 앞의 K 장을 TARGET SEGMENT 카메라로 바꾼다
        (V 는 그대로, 나머지 V-K 장은 원래 sampler 가 고른 coverage view 를 순서대로 채움).

        누수 실험 전용 -- 학습 때 못 본 조건을 test 에서 주는 것이다. geo_worldtraj 는
        geo_first_view_target_s: true / geo_cover_out_of_seg: true 라 학습 중 context 에 들어간
        target 카메라가 프레임 s 하나뿐인데, 그 수를 K 로 늘리면 (a) 지표가 좋아지는지 (b) 예측
        궤적이 그 view 들 쪽으로 치우치는지 본다.

        고르는 방식: _target_frame_idxs 를 균등 분할 -- K=1 -> [s] 로 학습 조건과 완전히 같고
        (재현 대조군), K=3 -> [s, 중간, 마지막], K=5 -> [s, 1/4, 2/4, 3/4, 마지막]. 항상
        geo_idxs[0] == s 라 view0 anchor 도 학습 때와 같다."""
        K = int(self.geo_test_inseg_k)
        if K <= 0:
            return geo_idxs
        V = len(geo_idxs)
        tgt = self._target_frame_idxs(s, e)
        inseg = [tgt[i] for i in self._even_indices(len(tgt), min(K, len(tgt)))]
        seen = set(inseg)
        rest = [i for i in geo_idxs if i not in seen]
        return inseg + rest[:max(V - len(inseg), 0)]

    def _geo_cam_param(self, scene_idx, geo_idxs, w2c_s, norm_scale):
        """[new] geo_cam_embed='relfirst': (V, 11) pose of each context view RELATIVE to the
        target segment's first camera s, in exactly the target's cam_param parametrization.

          rel_v = w2c_v @ inv(w2c_s)          # same form as normalize_camera_extrinsics_
                                              # and_points (utils/data_utils.py:24)
          rel_v[:3, 3] /= norm_scale          # the divisor the active scale_mode produced, so the
                                              # context translations live in the SAME units as the
                                              # trajectory the model generates
          out = [rel[:3,0], rel[:3,1], rel[:3,3], fx/2cx, fy/2cy]

        Intrinsics stay RAW (not divided by frame s) even under intr_norm 'rel': the target's own
        intr channels are ~[1,1] by construction there, so raw is the only way each context view's
        FoV reaches the model. Uses full-resolution scene poses, so it is independent of the
        segment subsampling and of whether the geo latent came from the cache."""
        w2c_v = self.extrinsics_list[scene_idx][list(geo_idxs)].float()   # (V,4,4)
        K = self.intrinsics_list[scene_idx][list(geo_idxs)].float()       # (V,3,3)
        rel = w2c_v @ torch.linalg.inv(w2c_s.float()).unsqueeze(0)        # (V,4,4)
        trans = rel[:, :3, 3] / norm_scale
        intr = torch.stack([K[:, 0, 0] / (K[:, 0, 2] * 2),
                            K[:, 1, 1] / (K[:, 1, 2] * 2)], dim=-1)       # (V,2) raw
        return torch.cat([rel[:, :3, 0], rel[:, :3, 1], trans, intr], dim=-1).float()

    def _geo_patch_grid(self):
        """[new] the (Gh, Gw) VGGT patch grid the geo encoder produces for self.geo_hw images.

        Mirrors models/encoder_decoder.py:193-202 (LagerNVS Reconstructor.forward): the longer side
        is resized to 518 and the other side is floored to a multiple of the ViT patch size 14; the
        aggregator's non-patch tokens are stripped (encoder_decoder.py:213-215), so the token count
        per view is exactly Gh*Gw. For the default geo_image_hw (256, 448) -> (21, 37) = 777 tokens
        per view, i.e. M = 6*777 = 4662 -- the shape the geo latent cache holds. attach_geo_cam
        asserts this against the real M, so a mismatch fails loudly instead of silently
        misaligning rays with tokens."""
        H, W = self.geo_hw
        p, S = 14, 518
        if H > W:
            tgt_h, tgt_w = S, (int(S * W / H) // p) * p
        else:
            tgt_w, tgt_h = S, (int(S * H / W) // p) * p
        return tgt_h // p, tgt_w // p

    def _geo_cam_plucker(self, scene_idx, geo_idxs, w2c_s, norm_scale, hw_orig):
        """[new] geo_cam_embed='plucker': (V, Gh*Gw, 6) Plücker ray per geo patch token, in the
        SAME frame and units as the trajectory being generated (target segment's first camera s,
        translations / norm_scale).

          rel_v = w2c_v @ inv(w2c_s)              # frame-s world -> cam v  (as in _geo_cam_param)
          R, t  = rel_v[:3,:3], rel_v[:3,3] / norm_scale
          o     = -R^T t                          # camera center, frame s, target units
          d     = normalize(R^T K^-1 [x, y, 1])   # patch-center ray direction, frame s
          out   = [d (3), o x d (3)]              # standard Plücker (direction, moment)

        Patch centers are taken in NORMALIZED image coords ((j+.5)/Gw, (i+.5)/Gh) and mapped back
        to the ORIGINAL pixel grid before applying the original K. Both resizes on the way to the
        encoder (원본 -> geo_hw in _load_images, geo_hw -> 518-long-side inside the reconstructor)
        are full-frame scalings, so normalized coords survive them exactly -- including the slight
        anisotropic stretch when geo_hw's aspect differs from the scene's.

        Row-major (i over rows, j over cols) to match the ViT token order that
        `einops.rearrange(tokens, "b v p c -> b (v p) c")` flattens (geo_encoder.py:138).

        Unlike 'relfirst' there are no explicit intrinsics channels: FoV is already in d."""
        Gh, Gw = self._geo_patch_grid()
        H0, W0 = float(hw_orig[0]), float(hw_orig[1])
        w2c_v = self.extrinsics_list[scene_idx][list(geo_idxs)].float()   # (V,4,4)
        K = self.intrinsics_list[scene_idx][list(geo_idxs)].float()       # (V,3,3) original px
        rel = w2c_v @ torch.linalg.inv(w2c_s.float()).unsqueeze(0)        # (V,4,4)
        R = rel[:, :3, :3]                                                # (V,3,3)
        t = rel[:, :3, 3] / norm_scale                                    # (V,3)
        Rt = R.transpose(1, 2)                                            # (V,3,3) = c2w rotation
        o = -torch.einsum('vij,vj->vi', Rt, t)                            # (V,3) camera center

        ys = (torch.arange(Gh, dtype=torch.float32) + 0.5) / Gh * H0      # (Gh,)
        xs = (torch.arange(Gw, dtype=torch.float32) + 0.5) / Gw * W0      # (Gw,)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')                    # row-major
        xx, yy = xx.reshape(-1), yy.reshape(-1)                           # (P,)
        fx, fy = K[:, 0, 0], K[:, 1, 1]
        cx, cy = K[:, 0, 2], K[:, 1, 2]
        d_cam = torch.stack([(xx.unsqueeze(0) - cx[:, None]) / fx[:, None],
                             (yy.unsqueeze(0) - cy[:, None]) / fy[:, None],
                             torch.ones(K.shape[0], xx.numel())], dim=-1)   # (V,P,3)
        d = torch.einsum('vij,vpj->vpi', Rt, d_cam)                       # (V,P,3) frame s
        d = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        m = torch.cross(o[:, None, :].expand_as(d), d, dim=-1)            # (V,P,3) moment
        return torch.cat([d, m], dim=-1).float()                          # (V,P,6)

    def _swap_donor(self, idx, scene_idx):
        """geo_swap_mode='inscene': (s, e) of ANOTHER segment of the same scene, or None when the
        scene holds only this one segment (the caller then leaves the context unswapped and flags
        the item so the analysis can drop it)."""
        if self._scene2samples is None:
            m = {}
            for j, sm in enumerate(self.samples):
                m.setdefault(sm[0], []).append(j)
            self._scene2samples = m
        sibs = self._scene2samples.get(scene_idx, [])
        if len(sibs) < 2:
            return None
        j = sibs[(sibs.index(idx) + self.geo_swap_shift) % len(sibs)]
        return None if j == idx else (self.samples[j][1], self.samples[j][2])

    def _attach_geo_ctx(self, out, scene_idx, geo_idxs):
        """geo_return_idxs bookkeeping: the selected context view indices and their c2w (OpenCV),
        so an offline script can measure generated-trajectory vs context-camera distances. Never
        fed to the model -- purely an analysis side-channel."""
        gi = list(geo_idxs)
        out['geo_idxs'] = torch.tensor(gi, dtype=torch.long)
        out['geo_ctx_c2w'] = torch.linalg.inv(self.extrinsics_list[scene_idx][gi].float())

    def _geo_cam_cond(self, scene_idx, geo_idxs, w2c_s, norm_scale, hw_orig):
        """geo_cam_embed dispatch -- 'relfirst' -> (V,11), 'plucker' -> (V,P,6)."""
        if self.geo_cam_embed == 'plucker':
            return self._geo_cam_plucker(scene_idx, geo_idxs, w2c_s, norm_scale, hw_orig)
        return self._geo_cam_param(scene_idx, geo_idxs, w2c_s, norm_scale)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_idx, s, e, caption, data_name = self.samples[idx]
        extrinsics = self.extrinsics_list[scene_idx][s:e]     # (T,4,4) w2c
        intrinsics = self.intrinsics_list[scene_idx][s:e]     # (T,3,3)
        h, w = self.hw_list[scene_idx]
        frame_files = self.frame_files_list[scene_idx]

        # if the segment has more than num_frames, sample evenly to num_frames
        if extrinsics.shape[0] > self.num_frames:
            sel = self._even_indices(extrinsics.shape[0], self.num_frames)
            extrinsics = extrinsics[sel]
            intrinsics = intrinsics[sel]

        _scale_mode = resolve_scale_mode(self.cfg)
        if _scale_mode == 'avg_scale':
            # SCVideo original: normalize by the STORED point-cloud avg_scale
            # (scene_dir/avg_scale/<seg_key>.json). Falls back to cam_dist_mean if missing.
            gs = self._avg_scale(scene_idx, data_name.split('_')[-1])
            norm_scale = gs if gs is not None else self._cam_dist_mean_scale(extrinsics)
        elif _scale_mode == 'geo_lagernvs':
            # FULL alignment: scale = 1.35*max(||geo-context center - frame s||) = the exact
            # scale LagerNVS's build_cam_token uses -> target & geo latent share frame+scale.
            gs = self._geo_lagernvs_scale(scene_idx, s, e)
            norm_scale = gs if gs is not None else self._cam_dist_mean_scale(extrinsics)
        elif _scale_mode == 'context_longer':
            cs = self._cam_dist_mean_context(scene_idx, s, e)
            norm_scale = cs if cs is not None else self._cam_dist_mean_scale(extrinsics)
        elif _scale_mode == 'ctx_longer_135max':
            # LagerNVS's denominator form (1.35*max) measured on the CONTEXT RANGE's
            # num_frames windows -> same units as LagerNVS's own context normalization,
            # and leakage-free (no target view enters the divisor).
            cs = self._first_farthest_context(scene_idx, s, e)
            norm_scale = cs if cs is not None else self._first_farthest_scale(extrinsics)
        elif _scale_mode == 'first_farthest_135':
            # LagerNVS-style: 1.35 * max ||center - first camera|| over the segment
            norm_scale = self._first_farthest_scale(extrinsics)
        else:
            norm_scale = self._cam_dist_mean_scale(extrinsics)
        normalized_extrinsics, _, norm_scale, _ = normalize_camera_extrinsics_and_points(
            extrinsics, avg_scale=norm_scale, max_trans_norm=self.cfg.max_trans_norm)

        # SCVideo original intrinsics (dataset_large.py): width/height from the principal point
        # (cx*2, cy*2) -- for DL3DV that is exactly transforms.json's w,h (verified 0/400 scenes
        # differ), so this also reproduces the legacy fx/w,fy/h branch bit-for-bit.
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        wv = intrinsics[:, 0, 2] * 2
        hv = intrinsics[:, 1, 2] * 2
        normalized_intrinsics = torch.cat([(fx / wv)[:, None], (fy / hv)[:, None]], dim=-1).float()
        _intr_norm = getattr(self.cfg, 'intr_norm', 'auto')
        if _intr_norm == 'auto':      # legacy coupling to scale_mode (see module docstring)
            _intr_norm = 'rel' if _scale_mode == 'avg_scale' else 'raw'
        if _intr_norm == 'rel':
            # frame0-relative -> frame0 intr = [1,1] (dataset_large.py:313)
            normalized_intrinsics = normalized_intrinsics / normalized_intrinsics[0:1, :]
        elif _intr_norm != 'raw':
            raise ValueError(f"intr_norm must be 'auto' | 'rel' | 'raw', got {_intr_norm!r}")

        # [new 2026-08-06] translation 채널(6:9)의 표현. rotation 채널(0:6)은 어느 쪽이든 w2c R 의
        # 앞 두 열 그대로다 -- R 과 R^T 는 정보량도 프레임간 geodesic 거리도 같아서 규약 비교의
        # 변수가 되지 못한다. 실제로 달라지는 건 translation 하나뿐이라 그것만 분기한다.
        #   'w2c' (기본, 기존 동작) t      = normalized_extrinsics[:, :3, 3]
        #   'c2w'                  c      = -R^T t = 카메라 중심
        # ||t|| == ||c|| 이라 norm_scale 분모는 그대로 통하고 크기 분포도 동일하다; 방향만 다르다.
        # w2c 는 Δt = -R_i·Δc - ΔR·c_{i+1} 로 회전이 translation 에 섞여 들어온다
        # (scripts/data/cam_repr_w2c_vs_c2w.py 의 Step 0 진단 참고).
        # !! 디코드 경로도 같이 맞춰야 한다: utils/data_utils.out_to_trajectory(..., trans_repr=...).
        _trans_repr = getattr(self.cfg, 'trans_repr', 'w2c')
        if _trans_repr == 'w2c':
            normalized_trans = normalized_extrinsics[:, :3, 3]
        elif _trans_repr == 'c2w':
            normalized_trans = -torch.einsum(
                'tji,tj->ti', normalized_extrinsics[:, :3, :3], normalized_extrinsics[:, :3, 3])
        else:
            raise ValueError(f"trans_repr must be 'w2c' | 'c2w', got {_trans_repr!r}")

        cam_param = torch.cat([
            normalized_extrinsics[:, :3, 0],
            normalized_extrinsics[:, :3, 1],
            normalized_trans,
            normalized_intrinsics,
        ], dim=-1).float()                                    # (T,11)

        out = {
            'data_name': data_name,
            'cam_param': cam_param,
            'intrinsics': intrinsics.float(),
            'first_extrinsic': extrinsics[0].float(),
            # the divisor the active scale_mode produced. 'avg_scale' is kept as a legacy
            # alias of the same tensor (SCVideo's key name) so all consumers keep working.
            'norm_scale': norm_scale.float(),
            'avg_scale': norm_scale.float(),
            'text_prompt': caption,
            # per-frame (T,) so collate -> (B, T), matching make_intrinsics(fx_fy (B,T,2))
            'height': torch.full((cam_param.shape[0],), float(h)),
            'width': torch.full((cam_param.shape[0],), float(w)),
        }

        # geo encoder input (multi-view images) — only for the geo path; text-only skips it.
        # [new] cache hit short-circuits the whole block: the frozen geo_emb is read straight off
        # disk, so no images are decoded and no context views are selected. A miss falls through
        # to the original on-the-fly path below, so a partial cache is safe.
        if self.geo_enabled and self.geo_latent_cache_dir is not None:
            _p = osp.join(self.geo_latent_cache_dir, data_name.split('_')[0], f'{data_name}.pt')
            if osp.exists(_p):
                _c = torch.load(_p, map_location='cpu', weights_only=False)
                # two on-disk formats: a bare (M, 768) tensor (v1) and {'emb', 'geo_idxs'} (v2,
                # written once geo_idxs was needed to rebuild the per-view camera embedding).
                _idxs = _c['geo_idxs'].tolist() if isinstance(_c, dict) else None
                _emb = (_c['emb'] if isinstance(_c, dict) else _c).float()
                if self.geo_cam_embed is None:
                    out['geo_emb'] = _emb
                    if self.geo_return_idxs:
                        # the cache short-circuit never selected any views, so redo the (memoized,
                        # deterministic) greedy search purely to report what the cached emb used.
                        if _idxs is None and self.geo_view_sampling == 'frustum_cover' \
                                and not self.geo_shuffle_order:
                            _idxs = list(self._sample_geo_frustum_cover(scene_idx, s, e))
                        if _idxs is not None:
                            self._attach_geo_ctx(out, scene_idx, _idxs)
                    return out
                # [new 2026-08-03] v1 캐시는 geo_idxs 를 안 들고 있지만, frustum_cover 는
                # extrinsics 와 (s,e) 만 보는 deterministic greedy 라 캐시를 만든 그 선택을 그대로
                # 재계산할 수 있다. 이걸로 폴백을 막아 first_cam_included/ 트리(39,830개 전부 v1)를
                # camembed arm 에서도 쓴다 -- 없으면 매 스텝 LagerNVS forward 라 ~12x 느려진다.
                # shuffle 이 켜져 있으면 캐시된 emb 의 view 순서와 어긋나므로 재계산하지 않고
                # 기존대로 폴백한다.
                if _idxs is None and self.geo_view_sampling == 'frustum_cover' \
                        and not self.geo_shuffle_order:
                    _idxs = list(self._sample_geo_frustum_cover(scene_idx, s, e))
                if _idxs is not None:            # v1 files carry no view indices -> can't rebuild
                    out['geo_emb'] = _emb
                    out['geo_cam_param'] = self._geo_cam_cond(
                        scene_idx, _idxs, extrinsics[0], norm_scale, (h, w))
                    if self.geo_return_idxs:
                        self._attach_geo_ctx(out, scene_idx, _idxs)
                    return out
                # v1 cache + geo_cam_embed -> fall through to the on-the-fly path below

        if self.geo_enabled:
            # [new] geo_swap_mode='inscene': run the context selection for a DONOR segment of the
            # same scene instead of this one. Everything downstream (images, poses, norm_scale) is
            # still this scene's, so the swap only moves WHERE the context looks.
            g_s, g_e, swapped = s, e, False
            if self.geo_swap_mode == 'inscene':
                _d = self._swap_donor(idx, scene_idx)
                if _d is not None:
                    g_s, g_e = _d
                    swapped = True
            if self.geo_view_sampling == 'frustum_cover':
                geo_idxs = self._sample_geo_frustum_cover(scene_idx, g_s, g_e)
            elif self.geo_view_sampling == 'hybrid':
                geo_idxs = self._sample_geo_hybrid(scene_idx, g_s, g_e)
            elif self.geo_view_sampling == 'random_inseg':
                geo_idxs = self._sample_geo_random_inseg(g_s, g_e)
            else:   # 'even': evenly-spaced in-segment (baseline, leaks trajectory)
                geo_idxs = [g_s + i for i in self._even_indices(g_e - g_s, self.geo_num_views)]
            if swapped and self.geo_swap_keep_first:
                # view0 back to the TARGET's frame s. The slice keeps V exactly: if s was already
                # in the donor's picks, rest has V-1 entries; if not, the extra one is dropped.
                rest = [i for i in geo_idxs if i != s]
                geo_idxs = ([s] + rest)[:len(geo_idxs)]
            if self.geo_swap_mode:
                out['geo_swapped'] = torch.tensor(int(swapped))
            if self.geo_test_inseg_k:    # [new] test-time in-segment context probe
                geo_idxs = self._mix_inseg_context(list(geo_idxs), s, e)
            if self.geo_shuffle_order:   # permute context view order
                geo_idxs = list(geo_idxs)
                if getattr(self.cfg, 'geo_shuffle_keep_first', False) and len(geo_idxs) > 1:
                    rest = geo_idxs[1:]; random.shuffle(rest)   # keep view0 (frame s / ref) fixed
                    geo_idxs = [geo_idxs[0]] + rest
                else:
                    random.shuffle(geo_idxs)                     # shuffle all (VGGT ref changes)
            out['images'] = self._load_images(frame_files, geo_idxs)   # (V,3,H,W)
            # [new] the selected context frame indices. cache_geo_embeddings.py stores them next
            # to the latent so a cache hit can rebuild geo_cam_param without redoing the greedy
            # coverage search; harmless (a (V,) int tensor) on every other path.
            out['geo_idxs'] = torch.tensor(list(geo_idxs), dtype=torch.long)
            if self.geo_return_idxs:
                self._attach_geo_ctx(out, scene_idx, geo_idxs)
            if self.geo_cam_embed is not None:
                out['geo_cam_param'] = self._geo_cam_cond(
                    scene_idx, geo_idxs, extrinsics[0], norm_scale, (h, w))

            # posed geo: raw geo-view camera geometry (geo_encoder builds the lagernvs
            # cam_token from these). c2w OpenCV, intrinsics (fx,fy,cx,cy) px, image hw.
            if self.geo_posed:
                gw2c = self.extrinsics_list[scene_idx][geo_idxs].float()      # (V,4,4) w2c
                gc2w = torch.linalg.inv(gw2c)                                 # (V,4,4) c2w
                gK = self.intrinsics_list[scene_idx][geo_idxs].float()        # (V,3,3)
                geo_fxfycxcy = torch.stack([gK[:, 0, 0], gK[:, 1, 1],
                                            gK[:, 0, 2], gK[:, 1, 2]], dim=-1)  # (V,4) px
                out['geo_c2w'] = gc2w
                out['geo_fxfycxcy'] = geo_fxfycxcy
                out['geo_hw'] = torch.tensor([float(h), float(w)])

        return out
