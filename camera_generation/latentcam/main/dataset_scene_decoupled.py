"""Scene-Decoupled (SD) 데이터셋 — **cross-clip** context 로 도는 CamDataset 변종.

DL3DV 와 근본적으로 다른 점 하나: DL3DV 는 한 scene 안의 다른 **프레임 구간**을 context 로
쓰지만, SD 는 같은 scene 을 서로 다른 카메라로 찍은 다른 **clip** 을 context 로 쓴다
(scene 하나당 7 clip, frame-0 pose 는 clip 간 동일). 그래서 sample 의 단위가
(target clip, context clip) 쌍이고, `scene_dir_list` 의 원소는 scene 이 아니라 **clip** 이다.
이렇게 두면 `_geo_pixel_plucker` / `_geo_depth_maps` / `_target_out` 같은 부모의 계산은
"scene_idx" 자리에 context clip 인덱스를 넣는 것만으로 손대지 않고 그대로 쓸 수 있다.

좌표계/스케일 (사용자 확정 baseline, 2026-08-10):
  1. clip 마다 da3 가 따로 돌아 **임의 스케일**이라, 각 clip 의 `umeyama_gt.json` sim3
     (X_gt = s·R·X_da3 + t, `convention: gtrot`) 를 pose 와 depth 양쪽에 걸어 **GT meters** 로
     올린다. 두 clip 이 같은 world frame 에 놓이는 건 이 단계 덕분이다.
       w2c 변환:  R' = R_e·Rᵀ,   t' = s·t_e − R'·t     (R' 은 여전히 직교 — 카메라 좌표도 s 배)
       depth  :  d' = s·d
  2. 그 다음 **context clip 의** `avg_scale_align/0.json` (= s·avg_scale, 정확히 성립 확인) 하나로
     target pose / context Plücker translation / context depth 를 전부 나눈다. 분모가 하나라
     부모의 단일 `norm_scale` 경로에 그대로 맞물리고, `o + exp(logd)·d` 가 target 궤적과 같은
     좌표계의 3D 점이 된다.
  분모를 **context** 에서 재는 이유: 추론 때 알고 있는 것이 context clip 뿐이다 (target 은
  생성 대상). target 의 avg_scale 을 쓰면 누수다.

샘플 리스트는 **파일로 고정**한다 (`cfg.train_seg_list` / `test_seg_list`, 한 줄 = data_name =
`<scene>__<target_clip>__<ctx_clip>`). scripts/data/sd_build_seg_lists.py 가 static clip 과
align 품질 하위 컷을 적용해 만든다 — 자세한 건 그 파일 docstring 참고.

기존 DL3DV 경로는 이 파일을 import 조차 하지 않으므로 영향이 없다 (base.py 가
`cfg.dataset_name == 'scene_decoupled'` 일 때만 여기로 분기).
"""
import json
import os
import os.path as osp

import numpy as np
import torch

from dataset_dl3dv import CamDataset

SD_ROOT_DEFAULT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"

# mp4 디코딩이 0 프레임을 낼 때 재시도 횟수/간격. /data1 은 Lustre 라 동시 I/O 가 몰리면
# VideoCapture open 이 일시적으로 실패한다 — 그 한 번에 학습을 죽이지 않기 위한 값.
_DECODE_RETRIES = 3
_DECODE_RETRY_SLEEP = 0.5      # 초. attempt 마다 x(attempt+1) 로 늘어난다


class SDCamDataset(CamDataset):
    """scene_dir_list 의 원소 = clip 디렉토리, samples 의 원소 = (target clip, context clip) 쌍."""

    # ---- base.py 의 seg-list 분할용. DL3DV 는 '<batch>_<hash>_<seg>' 를 슬래시 경로로 되돌리지만
    #      SD 리스트는 data_name 을 그대로 적어 두므로 항등이다.
    @staticmethod
    def seg_key(data_name):
        return data_name

    def __init__(self, cfg, type='train', only_segments=None):
        self.sd_root = getattr(cfg, 'sd_root', None) or SD_ROOT_DEFAULT
        self.sd_split = getattr(cfg, 'sd_split', None) or 'whuman'
        # context view 수. None -> context clip segment 의 **모든** 프레임 (49).
        # dynamic scene 이라 시간축을 통째로 줘야 한다는 사용자 결정(2026-08-10). int 면 균등 subsample.
        self.sd_geo_views = getattr(cfg, 'sd_geo_views', None)
        # [new 2026-08-16] 'list'(기본) = 리스트에 적힌 (target, ctx) 쌍 그대로 — 기존과 bit-identical.
        # 'random_scene' = 고유 target clip 당 1행으로 줄이고, ctx 는 **그 scene 에서 리스트가
        # 허용한 clip 중** 매번 무작위로 뽑는다. 노리는 것 두 가지:
        #   (a) 코퍼스 균형 — whuman 기준 52286 쌍 -> 14017 target 으로 3.7배 축소해서
        #       DL3DV/DataDoP 와 샘플 수가 같은 자릿수가 된다.
        #   (b) epoch 마다 context 가 바뀌는 augmentation.
        self.pair_mode = str(getattr(cfg, 'sd_pair_mode', 'list') or 'list')
        if self.pair_mode not in ('list', 'random_scene'):
            raise ValueError(f"sd_pair_mode must be 'list' | 'random_scene', got {self.pair_mode!r}")
        self.ctx_of = []                      # sample idx -> context clip idx (list 모드의 고정 ctx)
        self.ctx_pool_of = []                 # sample idx -> [ctx 후보 clip idx] (random_scene 전용)
        super().__init__(cfg, type=type, only_segments=only_segments)

    # ------------------------------------------------------------------ index
    def _load_index(self):
        """train/test seg-list 두 파일의 **합집합**을 그대로 인덱싱한다.

        코퍼스를 훑지 않는다 — 어떤 쌍을 쓸지는 이미 리스트로 고정되어 있고, 그게 이 데이터셋의
        요점이다(어떤 clip 쌍이 학습에 들어갔는지 재현 가능해야 arm 비교가 성립). 부모처럼
        <root>/.latentcam_index/ 에 캐시한다 — prompts.json 이 clip 마다 따로라 15k+ 개를 여는데
        Lustre 에서 수 분 걸린다.
        """
        self.root = osp.join(self.sd_root, 'da3', self.sd_split)
        if self.geo_custom:
            # 부모는 cfg.custom_geo_depth_cache_dir(DL3DV 용)를 넣어 뒀다. SD 캐시는
            # <sd_root>/da3_depth_raw/<split>/<scene>/<clip>.npy 이고 relpath(clip_dir, self.root)
            # 가 정확히 '<scene>/<clip>' 이라 부모의 경로 조립 규칙과 그대로 맞는다.
            self.geo_depth_cache_dir = osp.join(self.sd_root, 'da3_depth_raw', self.sd_split)
            print(f"[SD] depth cache {self.geo_depth_cache_dir}")

        names = []
        for k in ('train_seg_list', 'test_seg_list'):
            p = getattr(self.cfg, k, None)
            if not p:
                raise ValueError(f"dataset_name='scene_decoupled' requires cfg.{k} "
                                 f"(scripts/data/sd_build_seg_lists.py 가 만든다)")
            with open(p) as f:
                names += [ln.strip() for ln in f if ln.strip()]
        seen, ordered = set(), []
        for n in names:                       # train/test 가 겹치지 않아야 정상이지만 방어적으로
            if n not in seen:
                seen.add(n); ordered.append(n)

        # 캐시 키에 pair_mode 가 **반드시** 들어가야 한다 — 지금까지는 정렬된 이름 목록만 해시해서
        # 모드를 바꿔도 옛 캐시를 조용히 재사용했다 ("쌍을 줄였는데 샘플 수가 그대로").
        key = (f"sd_{self.sd_split}__nf{self.num_frames}__n{len(ordered)}"
               f"__pm{self.pair_mode}__{_short_hash(ordered)}")
        cache_dir = osp.join(self.sd_root, '.latentcam_index')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, key + '.pt')
        if osp.isfile(cache_path):
            idx = torch.load(cache_path, weights_only=False)
            self.samples = idx['samples']; self.hw_list = idx['hw_list']
            self.ctx_of = idx['ctx_of']
            self.ctx_pool_of = idx.get('ctx_pool_of', [])
            self.scene_dir_list = [osp.join(self.root, c) for c in idx['clip_rel']]
            print(f"[SD index cache] {len(self.samples)} pairs / "
                  f"{len(self.scene_dir_list)} clips <- {cache_path}")
            return

        print(f"[SD index build] {len(ordered)} pairs (prompts.json + intrinsics probe per clip; "
              f"cached to {cache_path})")
        clip2idx, cap_cache = {}, {}

        def clip_idx(scene, clip):
            rel = f"{scene}/{clip}"
            i = clip2idx.get(rel)
            if i is None:
                d = osp.join(self.root, rel)
                K = np.load(osp.join(d, 'pose.npz'))['intrinsics']
                i = len(self.scene_dir_list)
                clip2idx[rel] = i
                self.scene_dir_list.append(d)
                self.hw_list.append((int(round(float(K[0, 1, 2]) * 2)),
                                     int(round(float(K[0, 0, 2]) * 2))))
            return i

        from tqdm import tqdm
        scene_pool, seen_tgt = {}, {}          # scene -> [ctx 후보 clip idx] / (scene,tgt) -> sample idx
        for name in tqdm(ordered):
            scene, tgt, ctx = name.split('__')
            ti, ci = clip_idx(scene, tgt), clip_idx(scene, ctx)
            # 후보 pool 은 **리스트에 ctx 로 등장한 clip 만** 모은다 — sd_build_seg_lists.py 가
            # static clip 과 sim3 품질 하위 컷을 이미 걸러 뒀고, 그 필터를 우회하면 안 된다.
            p = scene_pool.setdefault(scene, [])
            if ci not in p:
                p.append(ci)
            if self.pair_mode == 'random_scene' and (scene, tgt) in seen_tgt:
                continue                       # 같은 target 의 두 번째 쌍부터는 행을 만들지 않는다
            seg = cap_cache.get(ti)
            if seg is None:
                pj = json.load(open(osp.join(self.scene_dir_list[ti], 'prompts.json')))['0']
                pcs = pj.get('prompt_camera_with_scene_video')
                seg = (int(pj['frame_idx'][0]), int(pj['frame_idx'][1]),
                       pcs.get('concise', "") if isinstance(pcs, dict) else (pcs or ""))
                cap_cache[ti] = seg
            s, e, caption = seg
            seen_tgt[(scene, tgt)] = len(self.samples)
            # data_name 은 random_scene 에서도 **리스트에 처음 나온 쌍의 이름 그대로** 둔다.
            # 이걸 '<scene>__<tgt>' 로 줄이면 base.py 의 seg-list 분할(id2idx)이 리스트 id 와
            # 맞지 않아 train/val 이 통째로 비어 버린다. 즉 이름의 세 번째 칸은 random_scene 에서
            # "이 target 이 리스트에 처음 등장했을 때의 ctx"라는 뜻이고, 실제 사용된 ctx 가 아니다.
            self.samples.append((ti, s, e, caption, name))
            self.ctx_of.append(ci)

        if self.pair_mode == 'random_scene':
            # sample 마다 자기 자신(target clip)을 뺀 후보를 미리 굳혀 둔다 -> __getitem__ 은 O(1).
            for i, (ti, _s, _e, _c, name) in enumerate(self.samples):
                scene = name.split('__')[0]
                pool = [c for c in scene_pool[scene] if c != ti]
                self.ctx_pool_of.append(pool if pool else [self.ctx_of[i]])
            n_multi = sum(1 for p in self.ctx_pool_of if len(p) > 1)
            print(f"[SD pair_mode=random_scene] {len(ordered)} 쌍 -> {len(self.samples)} target, "
                  f"ctx 후보 2개 이상인 sample {n_multi} "
                  f"(평균 {np.mean([len(p) for p in self.ctx_pool_of]):.2f}개)")

        torch.save({'samples': self.samples, 'hw_list': self.hw_list, 'ctx_of': self.ctx_of,
                    'ctx_pool_of': self.ctx_pool_of,
                    'clip_rel': [osp.relpath(d, self.root) for d in self.scene_dir_list]},
                   cache_path)
        print(f"[SD index build] {len(self.samples)} pairs / {len(self.scene_dir_list)} clips "
              f"-> saved {cache_path}")

    # ------------------------------------------------------------------ clip 로딩
    def _load_scene(self, i):
        """clip i 의 pose 를 읽어 **sim3 로 GT meters 에 올린 뒤** 캐시한다.

        pose.npz 의 extrinsics 는 OpenCV w2c (X_cam = R_e·X + t_e), da3 임의 스케일.
        umeyama sim3 가 X_gt = s·R·X + t 이므로 카메라 좌표까지 s 배 한 미터 단위 w2c 는
          R' = R_e·Rᵀ,  t' = s·t_e − R'·t
        검증: 카메라 중심 −R'ᵀt' = s·R·(−R_eᵀt_e) + t 로 forward sim3 와 정확히 일치한다.
        """
        c = self._scene_cache.get(i)
        if c is not None:
            return c
        d = self.scene_dir_list[i]
        z = np.load(osp.join(d, 'pose.npz'))
        E = np.asarray(z['extrinsics'], dtype=np.float64)          # (T,3,4) w2c, da3 스케일
        K = np.asarray(z['intrinsics'], dtype=np.float32)          # (T,3,3) 504x294 격자
        u = json.load(open(osp.join(d, 'umeyama_gt.json')))
        if not u.get('moving', False):
            # static clip 은 s/t 자체가 파일에 없다. 리스트 생성 단계에서 걸러지므로 여기 오면 버그.
            raise ValueError(f"{d}: static clip (moving=false) has no sim3 — "
                             f"seg-list 에 들어가면 안 된다")
        s = float(u['s'])
        R = np.asarray(u['R'], dtype=np.float64)
        t = np.asarray(u['t'], dtype=np.float64)
        Rp = E[:, :3, :3] @ R.T                                    # (T,3,3)
        tp = s * E[:, :3, 3] - np.einsum('tij,j->ti', Rp, t)       # (T,3)
        n = E.shape[0]
        w2c = torch.eye(4, dtype=torch.float32).repeat(n, 1, 1)
        w2c[:, :3, :3] = torch.from_numpy(Rp.astype(np.float32))
        w2c[:, :3, 3] = torch.from_numpy(tp.astype(np.float32))
        c = {'w2c': w2c, 'intr': torch.from_numpy(K),
             'frame_files': self._mp4_path(d),
             'hw': (int(round(float(K[0, 1, 2]) * 2)), int(round(float(K[0, 0, 2]) * 2))),
             'sim3_s': s}
        self._scene_cache[i] = c
        return c

    def _mp4_path(self, clip_dir):
        scene, clip = osp.relpath(clip_dir, self.root).split('/')
        return osp.join(self.sd_root, 'video', self.sd_split, scene, clip + '.mp4')

    # ------------------------------------------------------------------ 입력 3종
    def _load_images(self, mp4, idxs):
        """`frame_files` 자리에 mp4 경로가 온다. 필요한 프레임까지만 순차 디코딩.

        mp4 는 672x384 인데 intrinsics 격자는 504x294 로 종횡비가 미세하게 다르다 (da3 가
        비등방으로 리사이즈했다). 둘 다 **full-frame** 리사이즈라 정규화 좌표는 보존되고,
        `_geo_pixel_plucker` 가 (j+.5)/Wc·W0 로 원본 격자에 되돌린 뒤 K 를 걸므로
        여기서 곧장 geo_hw 로 줄여도 Plücker 와 픽셀이 정확히 대응한다.

        [fix 2026-08-10] 0 프레임 디코딩을 즉시 예외로 올리면 **일시적 I/O 실패 한 번에
        학습 전체가 죽는다.** 실제로 sd_whuman_customgeo (wandb fkfqww00) 가 epoch 5 에서
        이렇게 죽었는데, 그 mp4 는 직후 3/3 회 81 프레임 정상 디코딩됐다 (/data1 은 Lustre,
        동시 I/O 가 몰리면 open 이 실패할 수 있다). 그래서 재시도 후에도 실패할 때만 던진다.
        """
        import cv2
        import time
        H, W = self.geo_hw
        want = {int(i) for i in idxs}
        got = {}
        for attempt in range(_DECODE_RETRIES):
            cap = cv2.VideoCapture(mp4)
            got, k = {}, 0
            while len(got) < len(want):
                ok, f = cap.read()
                if not ok:
                    break
                if k in want:
                    got[k] = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                k += 1
            cap.release()
            if got:
                if attempt:
                    print(f"[sd _load_images] {attempt + 1} 번째 시도에서 디코딩 성공: {mp4}")
                break
            if attempt < _DECODE_RETRIES - 1:
                time.sleep(_DECODE_RETRY_SLEEP * (attempt + 1))
        if not got:
            raise RuntimeError(f"no frames decoded from {mp4} ({_DECODE_RETRIES} 회 재시도 후)")
        last = got[max(got)]
        imgs = [torch.from_numpy(np.ascontiguousarray(got.get(int(i), last)))
                .permute(2, 0, 1).float() / 255.0 for i in idxs]
        return torch.stack(imgs)

    def _da3_depth(self, clip_idx, geo_idxs):
        """clip 의 depth 를 **sim3 스케일 s 를 곱해 미터로** 돌려준다 (부모와 다른 유일한 점 +
        SD 레이아웃 `<clip>/depth.npz`). 캐시는 부모와 같은 relpath 규칙."""
        d = self.scene_dir_list[clip_idx]
        gi = list(geo_idxs)
        dep = None
        if self.geo_depth_cache_dir:
            p = osp.join(self.geo_depth_cache_dir, osp.relpath(d, self.root) + '.npy')
            if osp.isfile(p):
                a = np.load(p, mmap_mode='r')
                dep = np.asarray(a[gi], dtype=np.float32)
        if dep is None:
            dep = np.asarray(np.load(osp.join(d, 'depth.npz'))['depth'][gi], dtype=np.float32)
        return torch.from_numpy(dep * self._load_scene(clip_idx)['sim3_s'])

    def _ctx_norm_scale(self, ctx_idx):
        """분모 = **context clip** 의 avg_scale_align (= sim3 s × avg_scale, 즉 정렬 후 미터
        단위의 scene depth). moving clip 에는 항상 있다."""
        p = osp.join(self.scene_dir_list[ctx_idx], 'avg_scale_align', '0.json')
        return torch.tensor([float(json.load(open(p)))]).clamp(min=1e-5)

    def _ctx_view_idxs(self, ctx_idx, s, e):
        idxs = list(range(s, e))
        v = self.sd_geo_views
        if v and len(idxs) > int(v):
            idxs = [idxs[i] for i in self._even_indices(len(idxs), int(v))]
        return idxs

    # ------------------------------------------------------------------ context 선택
    def _pick_ctx(self, idx):
        """random_scene 에서 쓸 context clip. RNG 는 (worker seed, idx) 기반이라 worker/epoch
        마다 달라진다 -> 같은 target 이 epoch 마다 다른 ctx 를 본다 (그게 이 모드의 목적).

        val 도 같이 흔들린다는 점은 알고 쓸 것: val loss 곡선이 조금 더 시끄러워진다. 고정된
        쌍으로 평가하려면 그 arm 은 `sd_pair_mode: list` 로 두거나 only_segments 를 쓴다.
        """
        pool = self.ctx_pool_of[idx]
        if len(pool) == 1:
            return pool[0]
        seed = (int(torch.initial_seed()) + idx * 2654435761) % (2 ** 63 - 1)
        return pool[int(np.random.default_rng(seed).integers(len(pool)))]

    # ------------------------------------------------------------------ item
    def __getitem__(self, idx):
        tgt, s, e, caption, data_name = self.samples[idx]
        ctx = self._pick_ctx(idx) if self.pair_mode == 'random_scene' else self.ctx_of[idx]
        extrinsics = self.extrinsics_list[tgt][s:e]              # (T,4,4) w2c, meters
        intrinsics = self.intrinsics_list[tgt][s:e]
        h, w = self.hw_list[tgt]
        if extrinsics.shape[0] > self.num_frames:
            sel = self._even_indices(extrinsics.shape[0], self.num_frames)
            extrinsics = extrinsics[sel]
            intrinsics = intrinsics[sel]

        norm_scale = self._ctx_norm_scale(ctx)
        out = self._target_out(extrinsics, intrinsics, norm_scale, h, w, caption, data_name)
        norm_scale = out['norm_scale']

        if not self.geo_enabled:                                  # text-only ablation
            return out

        # context 는 **다른 clip** 이지만 1 단계 sim3 로 같은 world frame 에 있으므로,
        # 부모의 헬퍼에 scene_idx 자리로 ctx 를 넘기기만 하면 계산이 그대로 성립한다.
        # (rel = w2c_ctx_v @ inv(w2c_target_s) 가 cross-clip 상대 포즈가 된다.)
        # context clip 의 segment 도 target 과 같은 [s, e) = [0, 49) 다 (prompts.json 규약).
        geo_idxs = self._ctx_view_idxs(ctx, s, e)
        ch, cw = self.hw_list[ctx]
        out['images'] = self._load_images(self.frame_files_list[ctx], geo_idxs)
        out['geo_idxs'] = torch.tensor(geo_idxs, dtype=torch.long)
        if self.geo_cam_embed is not None:
            out['geo_cam_param'] = self._geo_cam_cond(
                ctx, geo_idxs, extrinsics[0], norm_scale, (ch, cw))
        if self.geo_custom:
            if self.geo_custom_channels != 'rgb_only':
                out['geo_plucker_map'] = self._geo_pixel_plucker(
                    ctx, geo_idxs, extrinsics[0], norm_scale, (ch, cw))
            if self.geo_custom_channels == 'full':
                out['geo_logd'], out['geo_valid'] = self._geo_depth_maps(ctx, geo_idxs, norm_scale)
        if self.geo_posed:
            gw2c = self.extrinsics_list[ctx][geo_idxs].float()
            gK = self.intrinsics_list[ctx][geo_idxs].float()
            out['geo_c2w'] = torch.linalg.inv(gw2c)
            out['geo_fxfycxcy'] = torch.stack(
                [gK[:, 0, 0], gK[:, 1, 1], gK[:, 0, 2], gK[:, 1, 2]], dim=-1)
            out['geo_hw'] = torch.tensor([float(ch), float(cw)])
        return out


def _short_hash(items):
    import hashlib
    h = hashlib.sha1()
    for x in items:
        h.update(x.encode())
    return h.hexdigest()[:10]
