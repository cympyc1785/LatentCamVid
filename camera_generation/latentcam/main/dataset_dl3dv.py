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
  'avg_scale'          stored point-cloud avg_scale. 파일은 target segment key 로 고르지만
                       **내용의 정의는 pose_source 마다 다르다** (생성기가 서로 다른 스크립트다).
                       transforms: <scene>/avg_scale/<seg>.json
                         = mean(|| scene.ply 점 - target segment 첫 카메라 ||)  (~10-44)
                           (pipeline/workspace/make_avg_scale.py -> normalize_camera_
                            extrinsics_and_points, extrinsics[s:e] 의 [0] 이 기준 카메라)
                       da3:        <scene>/da3/avg_scale/<seg>.json
                         = mean(|| context 점 - context 카메라 중심들의 centroid ||)
                           context range = [0,s) 와 [e,N) 중 프레임이 많은 쪽 **하나**,
                           점 = da3 depth(conf >= 전역 P40, pixel_stride 2) unproject.
                           target 프레임을 전혀 안 써서 **leakage-free** 다.
                           (pipeline/workspace/make_avg_scale_da3.py, 6/6 세그먼트 재현 확인)
                         기준점은 `avg_scale_ref` 로 바꿀 수 있다 (아래).
                       -> 즉 pose_source 를 바꾸면 분모의 **정의 자체**가 바뀐다. 두 arm 의
                          avg_scale 크기 차이(mean 15.52805 vs 4.69791)는 단위 차가 아니다.
  'cam_dist_mean'      mean(||camera center_i - center_0||) over the target segment  (~1-2)
  'context_longer'     'cam_dist_mean' computed on out-of-segment context windows instead
  'ctx_longer_135max'  1.35 * max(||center - window's first center||) averaged over the CONTEXT
                       RANGE's num_frames windows (LagerNVS's denominator form, leakage-free)
  'first_farthest_135' 1.35 * max(||center_i - center_0||) over the segment (LagerNVS-style)
  'geo_lagernvs'       1.35 * max(||geo-context center - frame s||) (full LagerNVS alignment)
  'const'              [new 2026-08-17] 세그먼트마다 재지 않고 **코퍼스 상수** `cfg.norm_scale_const`
                       하나로 나눈다 = "scale align 을 뺀" ablation arm. 위 분모들은 전부
                       세그먼트별로 scene 크기에 맞춰 적응하는데(=scale align), 이건 그 적응만
                       없애고 나머지(앵커 E@inv(E_s), intr_norm, 채널 규약)는 그대로 둔다.
                       상수를 코퍼스 **기하평균**으로 잡으면 mean||t||/분모 의 로그 평균 레벨이
                       보존되므로 vae_latent_scale / diffusion 입력 std 를 안 흔든다 —
                       da3_7k train 29414 세그먼트 실측: geomean 3.982685, med 3.614095.
                       (분산은 원래도 안 줄고 있었다: mean||t|| 원본 log10std 0.3107 vs
                        avg_scale 로 나눈 뒤 0.3144. 분모가 하는 일은 분산 축소가 아니라
                        "scene 크기 대비" 로의 단위 변환이다.)
Legacy aliases accepted: 'saved_avg_scale' -> 'avg_scale', 'target_cam' -> 'cam_dist_mean'.

pose_source (pose/caption/avg_scale 를 어느 코퍼스에서 읽을지) — 자세한 근거는
dataset_cfg.py 의 _POSE_SOURCES 주석 참고. 'transforms'(기본)는 기존 동작 그대로.
  'transforms'  <scene>/{transforms.json, prompts.json, avg_scale/}
  'da3'         <scene>/da3/{pose.npz, prompts.json, avg_scale/}   (Depth Anything 3 예측 pose)

avg_scale_ref (pose_source='da3' + scale_mode='avg_scale' 에서만 의미가 있다) — 저장된 avg_scale
을 **어느 기준점에서 잰 파일**로 읽을지. 점 집합(context range unproject)은 둘이 같고 기준점만
다르다. 다른 pose_source 와 같이 쓰면 에러 (디렉토리가 da3 아래에만 있다).
  'centroid'          (기본, 기존 동작) <scene>/da3/avg_scale/<seg>.json
                      = mean(|| context 점 - context 카메라 중심들의 centroid ||)
  'context_first_cam' <scene>/da3/avg_scale_context_first_cam/<seg>.json
                      = mean(|| context 점 - context range 첫 카메라 ||)
[new 2026-08-13] 아래 둘은 context range 를 target **앞쪽**으로 고정하고 기준점을 **target segment
첫 카메라 s** 로 잡는다. cam_param 은 E @ inv(E_s) 로 frame s 기준 재고정된 뒤 이 분모로 나뉘므로
(utils/data_utils.normalize_camera_extrinsics_and_points) 재고정 기준과 스케일 기준이 일치한다.
s 의 포즈는 추론 시에도 주어지고(first_extrinsic) 점은 target [s,e) 를 안 쓰므로 leakage-free.
  'front_first_anchor'          <scene>/da3/avg_scale_front_first_anchor/<seg>.json
                      context range = [0, s)      ; s == 0 인 세그먼트는 파일 없음
  'front_first_anchor_same_len' <scene>/da3/avg_scale_front_first_anchor_same_len/<seg>.json
                      context range = [s-L, s), L = e-s ; 앞쪽 L 프레임을 못 채우면 파일 없음
파일이 없는 세그먼트는 인덱스에서 제외된다 (dataset_cfg.avg_scale_min_front -> self._min_front,
_load_index/_load_index_subset 의 `s < self._min_front` 필터). da3_7k 는 전 세그먼트 길이 49 /
s in {0,49,98,...} 이라 두 변형이 똑같이 s==0 만 빼서 paired 비교가 된다.
'front_first_anchor_same_len' 는 geo_view_sampling='front_uniform' (같은 [s-L,s) range 에서
geo context view 를 uniform 추출) 과 짝이고, 어긋나면 dataset_cfg 가 경고한다.
실측 (da3_7k train+test 39817 세그먼트, 결측/비유한 0): first_cam mean 6.64851 / med 4.30108,
first_cam/centroid 비 med 1.41159 (p05 1.00945, p95 2.23834, <1 인 것 3.87%). 즉 first_cam 쪽
분모가 대체로 1.4배 커서 정규화된 translation 이 그만큼 작아진다 — vae_latent_scale 을 그대로
두면 diffusion 입력 std 가 arm 마다 달라진다는 점에 주의.

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

# config 해석은 전부 dataset_cfg.py 로 옮겼다 (거기 모듈 docstring 이 "왜"를 설명한다).
# 여기서 re-export 하는 이유는 `from dataset_dl3dv import CamDataset, resolve_scale_mode`
# 같은 기존 import 를 깨지 않기 위해서다 (scripts/data/viz_scene_chunk_scale.py:108).
# _POSE_SOURCES / da3 pose.npz 규약 주석도 dataset_cfg.py 에 있다.
from dataset_cfg import (  # noqa: F401  (re-export)
    _POSE_SOURCES, _SCALE_MODE_ALIASES, AVG_SCALE_DIRS, AVG_SCALE_REFS, DatasetSpec,
    avg_scale_min_front, resolve_dataset_cfg, resolve_pose_source, resolve_scale_mode,
)


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
        self._plucker_grid = {}      # (H0,W0) -> (xx, yy) 픽셀 중심 격자 (상수라 재사용)
        self._scene2samples = None   # lazily built: scene_idx -> [sample idx, ...]
        # config 읽기/검증/폴백은 전부 dataset_cfg.resolve_dataset_cfg 가 한다. spec 의 필드
        # 이름 = 속성 이름이라 self.geo_cover_k 등 기존 접근은 그대로다. 캐스케이드(한 번 켜진
        # geo_latent_cache_dir 이 네 조건에서 차례로 꺼지는 것)와 print 문구도 거기 있다.
        spec = resolve_dataset_cfg(cfg)
        spec.apply_to(self)
        # front-anchored avg_scale 이 요구하는 최소 앞쪽 context 길이. 0 이면 제약 없음(기존 동작).
        # 인덱스 필터(_load_index)와 캐시 키 양쪽에서 쓰이므로 인덱스 빌드 전에 정해 둔다.
        self._min_front = avg_scale_min_front(self.avg_scale_ref, self.num_frames)

        # scene-level (indexed by scene) and sample-level (per prompt segment)
        self.extrinsics_list = []   # [(N,4,4)]   (eager) or _LazyScenes (lazy)
        self.intrinsics_list = []   # [(N,3,3)]
        self.frame_files_list = []  # [[path,...]]
        self.hw_list = []           # [(h,w)]
        self.scene_dir_list = []    # [scene_dir]  (for avg_scale json lookup)
        self.samples = []           # [(scene_idx, start, end, caption, data_name)]
        self._scene_cache = {}      # lazy: scene_idx -> {w2c,intr,frame_files,hw}
        # target_pose_source: scene_idx -> {'w2c': (V,T,4,4), 'intr': (V,T,3,3), 'keys': [str]}
        self._target_pose_cache = {}
        # lazy_dataset (default True): __init__ only builds the sample/scene index (from a persisted
        # cache when available); scene poses/paths are parsed on demand in __getitem__ + cached.
        self.lazy = spec.lazy_dataset or only_segments is not None
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
        # avg_scale_ref (pose_source='da3' 에서만 의미가 있다): 저장된 avg_scale 을 어느 context
        # range 에서 어느 기준점으로 잰 것으로 쓸지. 'centroid' = 기존 동작(<scene>/da3/avg_scale).
        # 매핑은 dataset_cfg.AVG_SCALE_DIRS 한 곳에만 둔다.
        if self.pose_source == 'da3':
            return osp.join(scene_dir, 'da3', AVG_SCALE_DIRS[self.avg_scale_ref])
        return osp.join(scene_dir, 'avg_scale')

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
        필요 없다 (dataset_cfg.py 의 _POSE_SOURCES 주석에 실측 근거). (N,3,4) 를 (N,4,4) 로 채운다.
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
        # [new 2026-08-13] front_* avg_scale_ref 는 세그먼트를 **걸러낸다**(앞쪽 context 부족).
        # 키에 안 넣으면 centroid arm 이 만든 캐시를 그대로 물어와 제외돼야 할 s==0 세그먼트가
        # 섞여 들어온다 -> 분모 파일이 없어 _avg_scale 이 raise. 거르는 변형일 때만 붙여서
        # 기존 캐시는 그대로 재사용되게 한다.
        if self._min_front > 0:
            key += f"__as{self.avg_scale_ref}"
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
                # front-anchored avg_scale: 앞쪽 context 를 못 채우는 세그먼트는 분모 파일 자체가
                # 없다 (scripts/data/make_avg_scale_da3_front_anchor.py 가 안 쓴다) -> 여기서 뺀다.
                if s < self._min_front:
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
                if s < self._min_front:          # front-anchored avg_scale: 분모 파일이 없다
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

    def _sample_geo_front_uniform(self, s, e):
        """[new 2026-08-13] geo_view_sampling='front_uniform': context range 를 target 바로 앞
        **L = e - s 프레임** `[s-L, s)` 으로 고정하고 거기서 geo_num_views 장을 uniform 으로 뽑는다.
        coverage greedy(frustum_cover) 와 달리 retrieval 이 전혀 없다 — 어느 프레임이 뽑힐지는
        (s, e) 만으로 정해져 결정적이고, epoch 마다 같다.

        `avg_scale_ref='front_first_anchor_same_len'` 과 **같은 프레임 집합**을 본다는 게 요점이다
        (분모를 만든 range 와 context view 를 뽑는 range 가 일치). 그래서 이 sampler 는 그 분모와
        같이 쓰는 것을 전제로 한다 — dataset_cfg 가 어긋난 조합에 경고를 낸다.

        geo_first_view_target_s (2026-08-13 추가):
          False (legacy) -> `[s-L, s)` 에서 k 장, **오름차순**. 프레임 s 는 안 들어간다.
          True           -> `[s-L, s]` **s 포함** 구간에서 k 장을 uniform 으로 뽑고 **역순**으로
                            돌린다 -> `[s, s-10, s-20, ...]` (L=49, k=6 기준). s 를 따로 prepend
                            하고 나머지 k-1 을 `[s-L, s)` 에서 뽑으면 마지막 픽이 s-1 이라 view0 과
                            1 프레임짜리 중복 view 가 생긴다 — 포함 구간에서 한 번에 뽑으면 그게
                            없다. frustum_cover 의 같은 플래그(:771 k_retr = geo_cover_k - 1)와
                            동일하게 s 는 k **안에** 들어 토큰 예산 V*P 가 다른 arm 과 같다.
                            이게 필요한 이유: DA3 는 cam_token 을 view0 기준으로 재고정하므로
                            (da3_geo_encoder.build_cam_token) view0 != s 면 geo latent 의 앵커가
                            target rel(= E @ inv(E_s), 앵커 s)과 다른 카메라가 된다. view0 = s 로
                            두면 두 트랙의 R,t 앵커가 같은 카메라가 되고 남는 어긋남은 스케일뿐이다.
                            (역순 자체는 DA3 에 대해 no-op 이다 — cam_token 을 주면 ref-view 재정렬이
                            꺼지고 RoPE/cls 토큰이 view 인덱스를 안 쓴다. 순서를 시간순으로 읽히게
                            둔 것뿐이고 실제 앵커 효과는 view0 = s 에서 나온다.)

        legacy(False) 는 target 프레임 [s,e) 가 절대 안 들어간다 -> leakage-free. True 는 앵커
        프레임 s 하나만 들어간다 (추론 시 s 는 주어지는 프레임이라 leakage 가 아니다).
        앞쪽이 L 보다 짧으면 있는 만큼만 쓰지만, front_first_anchor_same_len 은 그런 세그먼트를
        인덱스에서 이미 뺐다.
        """
        L = e - s
        pool = list(range(max(0, s - L), s))
        k = int(self.geo_num_views)
        if self.geo_first_view_target_s:
            inc = pool + [s]                          # [s-L, s] (s 포함)
            picks = [inc[i] for i in self._even_indices(len(inc), k)]
            return picks[::-1]                        # view0 = s, 그 다음 과거로 역순
        if not pool:                                  # s == 0 (해당 분모에서는 인덱스에서 제외됨)
            return [s] * k
        return [pool[i] for i in self._even_indices(len(pool), k)]

    def _sample_geo_context_uniform(self, scene_idx, s, e):
        """[new 2026-08-23] geo_view_sampling='context_uniform': context range 를 **영상 전체**
        `[0, N)` 로 두고 거기서 geo_num_views 장을 uniform 으로 뽑는다.

        다른 sampler 들과의 차이는 range 하나뿐이다 — 'even' 은 target segment `[s,e)`,
        'front_uniform' 은 바로 앞 `[s-L, s)`, 이건 영상 전체다. retrieval 이 없어서
        (s, e) 와 무관하게 scene 마다 **같은 프레임 집합**이 나온다 (epoch 간 결정적).

        왜 필요한가: SD/TRUMANS 처럼 context 가 target 과 **다른 clip** 인 코퍼스에서는
        "target 앞/안"이라는 range 자체가 정의되지 않는다. 거기서 자연스러운 context 는
        clip 전체이고, `dataset_scene_decoupled._ctx_view_idxs` 가 이미 같은 일을 한다
        (`sd_geo_views` 장을 clip 전체에서 even). 이 sampler 는 그 정의를 DL3DV 쪽에도
        같은 이름으로 열어 둔 것이다.

        ⚠ leakage: 영상 전체를 보므로 target segment 프레임이 **들어올 수 있다**
        (`even` 만큼은 아니지만 leakage-free 도 아니다). 누수 없는 비교군은 'front_uniform'.

        geo_first_view_target_s:
          False -> `[0, N)` 에서 k 장, 오름차순.
          True  -> `[0, N)` 에서 k 장을 뽑은 뒤 **s 에 가장 가까운 픽을 s 로 치환**하고 view0 로
                   올린다. 토큰 예산 V*P 는 다른 arm 과 같다 (frustum_cover :803 k_retr =
                   geo_cover_k - 1 과 같은 "s 는 k 안에" 규칙).

                   s 를 그냥 prepend 하고 나머지 k-1 을 `[0,N) \\ {s}` 에서 뽑으면 안 된다 —
                   s=0 일 때 첫 픽이 프레임 1 이라 view0 과 **1프레임 차이** 중복 view 가
                   생긴다 (DL3DV da3 는 s ∈ {0,49,98,...} 이라 매 scene 첫 세그먼트가 여기
                   걸린다). front_uniform 이 포함 구간에서 한 번에 뽑아 피하는 것과 같은 함정.
        """
        n = len(self.frame_files_list[scene_idx])
        k = int(self.geo_num_views)
        picks = self._even_indices(n, k)
        if not self.geo_first_view_target_s:
            return picks
        j = int(np.argmin(np.abs(np.asarray(picks) - s)))       # s 가 밀어낼 자리
        rest = [p for m, p in enumerate(picks) if m != j and p != s]
        if len(rest) < k - 1:                          # k > N 인 초단축 scene 방어
            rest = (rest + [p for p in range(n) if p != s])[:k - 1]
        return [s] + rest

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
        """scale_mode='avg_scale': the STORED point-cloud avg_scale. seg_key 는 **target
        segment** 의 키지만, 파일 안의 값이 무엇을 기준으로 잰 것인지는 pose_source 마다
        다르다 — 모듈 상단 scale_mode 주석의 표를 볼 것.

          transforms: mean(|| scene.ply 점 - target segment 첫 카메라 ||)   [target 기준]
          da3:        mean(|| context 점 - context 카메라 centroid ||)      [leakage-free]
                      avg_scale_ref='context_first_cam' 이면 같은 점 집합을 **context range
                      첫 카메라** 기준으로 잰 값 (<scene>/da3/avg_scale_context_first_cam).

        Mirrors dataset_large.py. Returns None if the json is missing (caller falls back to
        cam_dist_mean). 단 avg_scale_ref != 'centroid' 일 때는 fallback 이 조용히 분모를
        바꿔 버리므로 그냥 터뜨린다 — 7K 코퍼스는 39817/39817 세그먼트 전부 존재를 확인했다."""
        p = osp.join(self._avg_scale_dir(self.scene_dir_list[scene_idx]), f'{seg_key}.json')
        _strict = self.avg_scale_ref != 'centroid'
        if not osp.isfile(p):
            if _strict:
                raise FileNotFoundError(f"avg_scale_ref={self.avg_scale_ref!r} 인데 파일이 없다: {p}")
            return None
        try:
            return torch.tensor([float(json.load(open(p)))]).clamp(min=1e-5)
        except Exception:
            if _strict:
                raise
            return None

    def _target_poses(self, scene_idx, seg_key):
        """target_pose_source='da3_target_poses': target 궤적만 합성 pseudo-GT 로 갈아끼운다.

        `<scene>/da3/target_poses.npz` 는 변이(variant) 하나당 궤적 하나를 담는다:
            extrinsics (V,T,4,4) OpenCV **w2c**   — pose.npz 와 같은 규약
            intrinsics (V,T,3,3) 픽셀 공간        — 뱅크가 --fixed_focal 이라 T 축으로 상수
            keys       (V,) str                   — prompts.json 의 세그먼트 키
        seg_key 로 V 축을 고른다. 이 파일이 없거나 키가 없으면 **터뜨린다** — 조용히 소스
        궤적으로 학습되면 로그 어디에도 흔적이 안 남고 arm 이 대조군과 같아져 버린다.

        context 는 이 함수를 안 거친다: geo 블록이 `self.extrinsics_list[scene_idx][geo_idxs]`
        (소스 recon) 와 `self.frame_files_list[scene_idx]` (원본 프레임) 를 직접 읽는다.
        """
        cache = self._target_pose_cache.get(scene_idx)
        if cache is None:
            p = osp.join(self.scene_dir_list[scene_idx], 'da3', 'target_poses.npz')
            if not osp.isfile(p):
                raise FileNotFoundError(
                    f"target_pose_source={self.target_pose_source!r} 인데 파일이 없다: {p}")
            z = np.load(p, allow_pickle=True)
            E = np.asarray(z['extrinsics'], dtype=np.float32)          # (V,T,4,4) or (V,T,3,4)
            if E.shape[-2] == 3:
                pad = np.zeros(E.shape[:-2] + (4, 4), dtype=np.float32)
                pad[..., 3, 3] = 1.0
                pad[..., :3, :4] = E
                E = pad
            cache = {'w2c': torch.from_numpy(E),
                     'intr': torch.from_numpy(np.asarray(z['intrinsics'], dtype=np.float32)),
                     'keys': {str(k): i for i, k in enumerate(z['keys'].tolist())}}
            self._target_pose_cache[scene_idx] = cache
        v = cache['keys'].get(str(seg_key))
        if v is None:
            raise KeyError(f"target_poses.npz 에 세그먼트 키 {seg_key!r} 가 없다 "
                           f"(scene={self.scene_dir_list[scene_idx]}, "
                           f"있는 키 {sorted(cache['keys'])[:8]}...)")
        return cache['w2c'][v], cache['intr'][v]

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

    def _geo_pixel_plucker(self, scene_idx, geo_idxs, w2c_s, norm_scale, hw_orig):
        """[new 2026-08-07] geo_encoder='custom': (V, 6, Hc, Wc) Plücker ray per **PIXEL** of the
        encoder's input grid (self.geo_hw), in the SAME frame and units as the trajectory being
        generated — target segment's first camera s, translations / norm_scale.

        수식은 _geo_cam_plucker (patch 격자용) 와 글자 그대로 같다:
          rel_v = w2c_v @ inv(w2c_s);  R,t = rel[:3,:3], rel[:3,3]/norm_scale
          o = -R^T t;  d = normalize(R^T K^-1 [x,y,1]);  out = [d(3), o x d(3)]
        다른 것은 격자뿐이다 -- patch (Gh,Gw) 대신 픽셀 (Hc,Wc). custom 인코더의 GeoTokenizer 가
        stride=patch conv 로 직접 patchify 하므로 여기서 미리 내리면 안 된다.

        픽셀 중심은 NORMALIZED 좌표 ((j+.5)/Wc, (i+.5)/Hc) 로 잡아 원본 픽셀 격자로 되돌린 뒤
        원본 K 를 적용한다. 원본 -> geo_hw 리사이즈는 full-frame scaling 이라 정규화 좌표가
        정확히 보존된다 (da3 의 504x280 도 원본 3840x2160 의 full-frame 리사이즈임을 확인:
        cx,cy 가 정확히 252,140 = 중심).

        Row-major (i over rows, j over cols) — GeoTokenizer 의 conv 출력을 flatten(2) 한 순서와
        DINO 의 patch 토큰 순서가 둘 다 row-major 라 그대로 맞는다.

        전 구간을 **channel-first (V,3,P)** 로 계산한다. _geo_cam_plucker 처럼 (V,P,3) 으로 두면
        마지막 축이 길이 3 이라 norm/cross 의 벡터화가 나빠서 실측 0.26 s/item 이 나왔다
        (P=112896 이라 patch 격자보다 168 배 크다). channel-first + 마지막 permute 제거로 ~4배."""
        Hc, Wc = self.geo_hw
        H0, W0 = float(hw_orig[0]), float(hw_orig[1])
        w2c_v = self.extrinsics_list[scene_idx][list(geo_idxs)].float()   # (V,4,4)
        K = self.intrinsics_list[scene_idx][list(geo_idxs)].float()       # (V,3,3) original px
        rel = w2c_v @ torch.linalg.inv(w2c_s.float()).unsqueeze(0)        # (V,4,4)
        R = rel[:, :3, :3]
        t = rel[:, :3, 3] / norm_scale
        Rt = R.transpose(1, 2)                                            # c2w rotation
        o = -torch.einsum('vij,vj->vi', Rt, t)                            # (V,3) camera center

        # 픽셀 중심 격자는 (Hc,Wc,H0,W0) 에만 의존하는 상수 -> scene 마다 다시 만들지 않는다.
        g = self._plucker_grid.get((H0, W0))
        if g is None:
            ys = (torch.arange(Hc, dtype=torch.float32) + 0.5) / Hc * H0  # (Hc,)
            xs = (torch.arange(Wc, dtype=torch.float32) + 0.5) / Wc * W0  # (Wc,)
            yy, xx = torch.meshgrid(ys, xs, indexing='ij')                # row-major
            g = (xx.reshape(1, -1), yy.reshape(1, -1))                    # (1,P)
            self._plucker_grid[(H0, W0)] = g
        xx, yy = g
        V, P = K.shape[0], Hc * Wc
        fx, fy = K[:, 0, 0:1], K[:, 1, 1:2]
        cx, cy = K[:, 0, 2:3], K[:, 1, 2:3]
        d_cam = torch.stack([(xx - cx) / fx, (yy - cy) / fy,
                             torch.ones(V, 1).expand(V, P)], dim=1)       # (V,3,P)
        d = Rt @ d_cam                                                    # (V,3,P) frame s
        d = d * d.pow(2).sum(1, keepdim=True).clamp(min=1e-16).rsqrt()
        oc = o.unsqueeze(-1)                                              # (V,3,1)
        m = torch.stack([oc[:, 1] * d[:, 2] - oc[:, 2] * d[:, 1],         # o x d, 성분별로 직접
                         oc[:, 2] * d[:, 0] - oc[:, 0] * d[:, 2],         # (torch.cross 는 길이 3
                         oc[:, 0] * d[:, 1] - oc[:, 1] * d[:, 0]], dim=1)  # 축을 요구해 느리다)
        return torch.cat([d, m], dim=1).view(V, 6, Hc, Wc).float()

    def _da3_depth(self, scene_idx, geo_idxs):
        """<scene>/da3/depth.npz 의 지정 프레임 -> (V, H0, W0) float32 (da3 native 280x504).

        depth.npz 는 deflate 압축이라 6 장만 필요해도 scene 전체를 풀어야 한다 (실측 0.62 s).
        cfg.custom_geo_depth_cache_dir 가 있으면 scripts/data/cache_da3_depth.py 가 풀어 둔
        비압축 .npy 를 mmap 으로 열어 해당 프레임만 읽는다 (0.02 s). 캐시가 없으면 npz 로
        폴백하므로 부분 캐시도 안전하다."""
        sd = self.scene_dir_list[scene_idx]
        gi = list(geo_idxs)
        if self.geo_depth_cache_dir:
            p = osp.join(self.geo_depth_cache_dir, osp.relpath(sd, self.root) + '.npy')
            if osp.isfile(p):
                a = np.load(p, mmap_mode='r')
                return torch.from_numpy(np.asarray(a[gi], dtype=np.float32))
        z = np.load(osp.join(sd, 'da3', 'depth.npz'))['depth']
        return torch.from_numpy(np.asarray(z[gi], dtype=np.float32))

    def _geo_depth_maps(self, scene_idx, geo_idxs, norm_scale):
        """[new 2026-08-07] geo_encoder='custom': (logd (V,1,Hc,Wc), valid (V,1,Hc,Wc)).

          logd  = log(depth / norm_scale)   -- Plücker translation 과 **같은 분모**를 써야
                  o + exp(logd)*d 가 target 궤적과 같은 좌표계의 3D 점이 된다.
          valid = 전부 1 (사용자 결정 2026-08-07). da3 depth 는 depth<=0 이 0.0%, conf 는 확률이
                  아니라 상한 없는 값(scene 별 p50 1.86~14.41)이라 코퍼스 공통 임계값을 못 잡는다.
                  채널은 남겨 두므로(GeoTokenizer 의 8ch 계약) 나중에 마스크를 넣을 때 shape 변경
                  없이 여기만 고치면 된다.

        da3 격자(280x504) -> geo_hw 는 nearest 로 내린다. depth 는 물체 경계에서 불연속이라
        bilinear 로 섞으면 존재하지 않는 중간 깊이가 생긴다."""
        dep = self._da3_depth(scene_idx, geo_idxs).unsqueeze(1)           # (V,1,H0,W0)
        if dep.shape[-2:] != self.geo_hw:
            dep = torch.nn.functional.interpolate(dep, size=self.geo_hw, mode='nearest')
        dep = torch.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
        logd = torch.log((dep / norm_scale).clamp(min=1e-6))
        return logd.float(), torch.ones_like(logd)

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

    @staticmethod
    def seg_key(data_name):
        """data_name '<batch>_<hash>_<seg>' -> seg-list 파일의 '<batch>/<hash>/<seg>'.
        base.py 의 명시적 train/test 분할이 쓴다. Scene-Decoupled 로더
        (dataset_scene_decoupled) 는 리스트에 data_name 을 그대로 적으므로 항등으로 덮어쓴다."""
        bh, seg = data_name.rsplit('_', 1)
        batch, h = bh.split('_', 1)
        return f"{batch}/{h}/{seg}"

    def __len__(self):
        return len(self.samples)

    def _target_out(self, extrinsics, intrinsics, norm_scale, h, w, caption, data_name,
                    scale_mode=None):
        """[refactor 2026-08-10] target segment 의 (w2c, K, divisor) -> 모델이 먹는 dict.

        __getitem__ 에서 그대로 떼어낸 것이라 DL3DV 경로의 결과는 bit-identical 하다 (분할 전후
        3개 item 의 전 키 동일 확인). 떼어낸 이유는 Scene-Decoupled 로더
        (dataset_scene_decoupled.SDCamDataset) 가 **분모만 다르고** cam_param 규약
        (intr_norm / trans_repr / normalize_camera_extrinsics_and_points) 은 완전히 같아서다 --
        복사본을 두면 한쪽만 고치는 사고가 난다.

        scale_mode 는 intr_norm='auto' 의 legacy 커플링에만 쓰인다 (None -> cfg 에서 다시 해석).

        [new 2026-08-16] `cfg.norm_scale_gain` (기본 1.0 = 무동작) 은 혼합 학습에서 코퍼스별
        translation 레벨을 맞추려던 상수다. 분모에 거는 것이 **이 파일 안에서는** 유일하게
        안전한 지점이다 (반환된 `out['norm_scale']` 을 나중에 곱하면 cam_param 이 이미 옛 분모로
        나눠진 뒤라 궤적과 geo depth 단위가 갈라진다). gain 을 **나누는** 이유:
        m = mean||t|| / norm_scale 이라 gain>1 이면 m 이 gain 배가 된다.

        !! 그러나 `geo_encoder='da3'` + `geo_posed=True` 에서는 **켜지 말 것** (2026-08-16 확인).
        그 경로의 context 는 `norm_scale` 을 **안 거친다**: `__getitem__` 이 raw `geo_c2w` 를
        월드 단위 그대로 내보내고(:1437-1444), `da3_geo_encoder.build_cam_token` 이 그걸 DA3
        자기 규약(context view 들의 **median camera distance**)으로 다시 정규화한다(:279-282).
        결과: target cam_param 은 gain 배로 커지는데 context cam token 은 **불변**이라, context 가
        함의하는 스케일과 target 크기의 대응이 코퍼스별 상수만큼 어긋난다. 모델은 그 상수를
        context 에서 읽어낼 수 없어 코퍼스 정체성으로 추측해야 하고, 추론 시엔 그 라벨이 없다.
        gain 이 무해한 것은 모든 채널이 같은 `norm_scale` 로 나눠지는 `geo_custom`(lagernvs,
        `_geo_pixel_plucker` + `_geo_depth_maps`) 경로뿐이다 — 거기서는 균일 닮음변환이다.
        => 혼합 arm 은 **gain 없이(전부 1.0)** 간다. 코퍼스 간 레벨 차이는 필터/분모 정의로
        다루고, 이 훅을 켜려면 위 조건을 먼저 확인할 것.
        """
        _scale_mode = scale_mode or resolve_scale_mode(self.cfg)
        _gain = float(getattr(self.cfg, 'norm_scale_gain', 1.0) or 1.0)
        if _gain != 1.0:
            norm_scale = norm_scale / _gain
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
            # 'const' 는 avg_scale 의 ablation 짝이라 같은 쪽에 붙인다 (분모만 다른 arm 끼리
            # intr 규약이 갈리면 비교가 안 된다). 기존 mode 들의 매핑은 그대로다.
            _intr_norm = 'rel' if _scale_mode in ('avg_scale', 'const') else 'raw'
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
        return out

    def __getitem__(self, idx):
        scene_idx, s, e, caption, data_name = self.samples[idx]
        extrinsics = self.extrinsics_list[scene_idx][s:e]     # (T,4,4) w2c
        intrinsics = self.intrinsics_list[scene_idx][s:e]     # (T,3,3)
        h, w = self.hw_list[scene_idx]
        frame_files = self.frame_files_list[scene_idx]

        # [new 2026-08-27] target 궤적만 합성 pseudo-GT 로 교체. context(geo) 블록은 아래에서
        # self.extrinsics_list / self.frame_files_list 를 **직접** 읽으므로 여기 지역 변수만
        # 갈아끼우면 "context = 소스 영상, target = 합성 카메라" 분기가 정확히 성립한다.
        if getattr(self, 'target_pose_source', None):
            extrinsics, intrinsics = self._target_poses(scene_idx, data_name.split('_')[-1])

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
        elif _scale_mode == 'const':
            # [new 2026-08-17] scale-align ablation: 세그먼트별 적응 분모 대신 코퍼스 상수 하나.
            # 파일도 안 읽고 카메라도 안 재므로 avg_scale_ref 는 **분모에는 무관**하다 (인덱스
            # 필터에는 계속 관여한다 -> 대조군과 같은 샘플 집합을 유지하려고 일부러 남긴다).
            _c = getattr(self.cfg, 'norm_scale_const', None)
            if _c is None or float(_c) <= 0:
                raise ValueError("scale_mode='const' 는 cfg.norm_scale_const > 0 이 필요하다 "
                                 f"(got {_c!r}). 코퍼스 기하평균을 쓸 것 — da3_7k/"
                                 "front_first_anchor 는 3.982685")
            norm_scale = torch.tensor([float(_c)])
        elif _scale_mode == 'first_farthest_135':
            # LagerNVS-style: 1.35 * max ||center - first camera|| over the segment
            norm_scale = self._first_farthest_scale(extrinsics)
        else:
            norm_scale = self._cam_dist_mean_scale(extrinsics)
        out = self._target_out(extrinsics, intrinsics, norm_scale, h, w, caption, data_name)
        norm_scale = out['norm_scale']

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
            elif self.geo_view_sampling == 'front_uniform':
                geo_idxs = self._sample_geo_front_uniform(g_s, g_e)
            elif self.geo_view_sampling == 'context_uniform':
                geo_idxs = self._sample_geo_context_uniform(scene_idx, g_s, g_e)
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

            # [new 2026-08-07] geo_encoder='custom' 의 RGBD 입력. lagernvs 경로에서는
            # self.geo_custom 이 False 라 이 블록 자체가 없는 것과 같다.
            if self.geo_custom:
                if self.geo_custom_channels != 'rgb_only':
                    out['geo_plucker_map'] = self._geo_pixel_plucker(
                        scene_idx, geo_idxs, extrinsics[0], norm_scale, (h, w))
                if self.geo_custom_channels == 'full':
                    out['geo_logd'], out['geo_valid'] = self._geo_depth_maps(
                        scene_idx, geo_idxs, norm_scale)

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
