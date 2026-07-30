"""DL3DV-960 dataset for the latent (geo-conditioned) camera diffusion model.

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
  'first_farthest_135' 1.35 * max(||center_i - center_0||) over the segment (LagerNVS-style)
  'geo_lagernvs'       1.35 * max(||geo-context center - frame s||) (full LagerNVS alignment)
Legacy aliases accepted: 'saved_avg_scale' -> 'avg_scale', 'target_cam' -> 'cam_dist_mean'.

Assumptions to validate (see plan): transforms.json OpenGL c2w -> OpenCV w2c;
frame_idx indexes the file-sorted transforms frames.
"""
import os
import os.path as osp
import csv
import json
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
        self.root = cfg.dl3dv_root
        self.num_frames = cfg.num_frames
        self.geo_num_views = getattr(cfg, 'geo_num_views', 4)
        self.geo_hw = tuple(getattr(cfg, 'geo_image_hw', (256, 448)))
        self.geo_enabled = bool(getattr(cfg, 'geo_encoder', None))  # skip image loading for text-only
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
        self.geo_anchor_first_frame = getattr(cfg, 'geo_anchor_first_frame', False)
        # geo context view0 = target segment's FIRST camera s (rest = out-of-seg retrieved)
        # -> LagerNVS anchors to s, aligning the geo latent frame with the target frame.
        self.geo_first_view_target_s = getattr(cfg, 'geo_first_view_target_s', False)
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

    def _load_scene(self, i):
        """Lazily parse + cache a scene's transforms (used by _LazyScenes and getitem helpers)."""
        c = self._scene_cache.get(i)
        if c is None:
            sd = self.scene_dir_list[i]
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
               f"__ms{self.max_scenes}")
        cache_dir = osp.join(self.root, '.latentcam_index'); os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, key.replace('/', '_') + '.pt')
        if self.only_segments is not None:
            self._load_index_subset()
            return
        if osp.isfile(cache_path):
            idx = torch.load(cache_path, weights_only=False)
            self.samples = idx['samples']; self.scene_dir_list = idx['scene_dir_list']
            self.hw_list = idx['hw_list']
            print(f"[index cache] {len(self.samples)} samples / {len(self.scene_dir_list)} scenes "
                  f"<- {cache_path}")
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
            tj_path = osp.join(scene_dir, 'transforms.json')
            pj_path = osp.join(scene_dir, 'prompts.json')
            if not (osp.isfile(tj_path) and osp.isfile(pj_path)):
                continue
            try:
                tj = json.load(open(tj_path)); prompts = json.load(open(pj_path))
                n = len(tj['frames']); h, w = int(tj['h']), int(tj['w'])
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
        torch.save({'samples': self.samples, 'scene_dir_list': self.scene_dir_list,
                    'hw_list': self.hw_list}, cache_path)
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
                tj = json.load(open(osp.join(scene_dir, 'transforms.json')))
                chunk_idx[chunk] = len(self.scene_dir_list)
                self.scene_dir_list.append(scene_dir)
                self.hw_list.append((int(tj['h']), int(tj['w'])))
            seg = json.load(open(osp.join(scene_dir, 'prompts.json')))[seg_key]
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
        print(f"Loading DL3DV-960: {len(scenes)} scenes in {meta_name}, {len(blocked)} blacklisted"
              + (f", {len(cov_black)} segments coverage-blacklisted" if cov_black else ""))

        for chunk in tqdm(scenes):
            scene_hash = chunk.split('/')[-1]
            if scene_hash in blocked:
                continue
            scene_dir = osp.join(self.root, chunk)
            tj_path = osp.join(scene_dir, 'transforms.json')
            pj_path = osp.join(scene_dir, 'prompts.json')
            if not (osp.isfile(tj_path) and osp.isfile(pj_path)):
                continue
            try:
                extr, intr, frame_files, (h, w) = self._parse_transforms(scene_dir, tj_path)
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

        print(f"DL3DV-960: {len(self.samples)} segment samples from "
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
        """Frustum max-coverage: k views covering the most nearby space. When
        geo_cover_out_of_seg, candidates are restricted to the LONGER out-of-segment side
        (no target frames -> leakage-free). Anchor/scale:
          - geo_anchor_first_frame=True (honest): anchor = target's FIRST frame s (known at
            inference); radius scaled by CONTEXT movement; look_centroid = candidate mean.
            -> selection never uses the unseen target [s+1:e].
          - False (legacy): anchor near the target midpoint, radius from the target seg_scale."""
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
            if self.geo_anchor_first_frame:
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
        Mirrors dataset_large.py. Returns None if the json is missing (caller falls back)."""
        p = osp.join(self.scene_dir_list[scene_idx], 'avg_scale', f'{seg_key}.json')
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
        """FULL-alignment scale = 1.35*max(||geo-context camera center - frame s||), i.e. the
        exact scene_scale LagerNVS.build_cam_token uses for the geo views (view0=frame s).
        Uses the same geo selection so target & geo latent are in one frame+scale. (1)."""
        geo_idxs = self._sample_geo_frustum_cover(scene_idx, s, e)      # [s] + out-of-seg
        c2w = torch.linalg.inv(self.extrinsics_list[scene_idx].float())
        centers = c2w[:, :3, 3]
        d = (centers[geo_idxs] - centers[s]).norm(dim=-1)              # dist from frame s
        return (1.35 * d.max()).clamp(min=1e-5).unsqueeze(0)

    def _cam_dist_mean_context(self, scene_idx, s, e):
        """scale_mode='context_longer': cam_dist_mean taken from the LONGER out-of-segment
        side, chunked into num_frames windows — mean over windows of each window's camera
        movement (same metric as _cam_dist_mean_scale, relative to the window's first frame).
        Target-excluded -> leakage-free and reproducible at inference from context only.
        Returns None when the longer side is too short (caller falls back)."""
        w2c_all = self.extrinsics_list[scene_idx]              # (N,4,4) w2c
        N = w2c_all.shape[0]
        T = self.num_frames
        side = list(range(0, s)) if s >= (N - e) else list(range(e, N))
        chunks = [side[i:i + T] for i in range(0, len(side) - T + 1, T)]  # full windows
        if not chunks and len(side) >= 2:
            chunks = [side]                                    # short side -> one partial window
        if not chunks:
            return None
        scales = [self._cam_dist_mean_scale(w2c_all[ch]) for ch in chunks]
        return torch.stack(scales, 0).mean(0).clamp(min=1e-5)  # (1,)

    def _load_images(self, frame_files, idxs):
        H, W = self.geo_hw
        imgs = []
        for i in idxs:
            im = Image.open(frame_files[i]).convert('RGB').resize((W, H), Image.BICUBIC)
            imgs.append(torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255.0)
        return torch.stack(imgs)

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
        elif _scale_mode == 'first_farthest_135':
            # LagerNVS-style: 1.35 * max ||center - first camera|| over the segment
            norm_scale = self._first_farthest_scale(extrinsics)
        else:
            norm_scale = self._cam_dist_mean_scale(extrinsics)
        normalized_extrinsics, _, norm_scale, _ = normalize_camera_extrinsics_and_points(
            extrinsics, avg_scale=norm_scale, max_trans_norm=self.cfg.max_trans_norm)

        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        if _scale_mode == 'avg_scale':
            # SCVideo original intrinsics (dataset_large.py): width/height from principal point
            # (cx*2, cy*2), then normalize relative to the first frame -> frame0 intr = [1,1].
            wv = intrinsics[:, 0, 2] * 2
            hv = intrinsics[:, 1, 2] * 2
            normalized_intrinsics = torch.cat([(fx / wv)[:, None], (fy / hv)[:, None]], dim=-1).float()
            normalized_intrinsics = normalized_intrinsics / normalized_intrinsics[0:1, :]
        else:
            normalized_intrinsics = torch.cat([(fx / w)[:, None], (fy / h)[:, None]], dim=-1).float()

        cam_param = torch.cat([
            normalized_extrinsics[:, :3, 0],
            normalized_extrinsics[:, :3, 1],
            normalized_extrinsics[:, :3, 3],
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

        # geo encoder input (multi-view images) — only for the geo path; text-only skips it
        if self.geo_enabled:
            if self.geo_view_sampling == 'frustum_cover':
                geo_idxs = self._sample_geo_frustum_cover(scene_idx, s, e)
            elif self.geo_view_sampling == 'hybrid':
                geo_idxs = self._sample_geo_hybrid(scene_idx, s, e)
            elif self.geo_view_sampling == 'random_inseg':
                geo_idxs = self._sample_geo_random_inseg(s, e)
            else:   # 'even': evenly-spaced in-segment (baseline, leaks trajectory)
                geo_idxs = [s + i for i in self._even_indices(e - s, self.geo_num_views)]
            if self.geo_shuffle_order:   # permute context view order
                geo_idxs = list(geo_idxs)
                if getattr(self.cfg, 'geo_shuffle_keep_first', False) and len(geo_idxs) > 1:
                    rest = geo_idxs[1:]; random.shuffle(rest)   # keep view0 (frame s / ref) fixed
                    geo_idxs = [geo_idxs[0]] + rest
                else:
                    random.shuffle(geo_idxs)                     # shuffle all (VGGT ref changes)
            out['images'] = self._load_images(frame_files, geo_idxs)   # (V,3,H,W)

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
