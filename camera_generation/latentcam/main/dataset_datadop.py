"""DataDoP — shot 당 궤적 1개, context 는 **첫 view 1장**뿐인 CamDataset 변종.

DL3DV/Scene-Decoupled 와 근본적으로 다른 점: DataDoP 에는 "다른 프레임 구간"도 "다른 clip"도
없다. shot 하나가 곧 하나의 궤적이고, 우리가 가진 시각 정보는 그 shot 의 **frame 0 한 장**
(`<shot>_rgb.png` + `<shot>_depth.npy`) 이다. 그래서 sample 단위 = shot 이고 V=1 이다.

좌표계 (실측 확정, 2026-08-15)
------------------------------
`<shot>_transforms_cleaning.json` 의 `transform_matrix` 는 **OpenGL c2w** (nerfstudio 와 동일)
로, `models/GenDoP/dataset/scripts/Dataset_DataDoP.py` 가 `matrix[:3,1:3] *= -1` 로 만든 것이다.
`traj.txt`(TUM, wxyz quat) 와 rotation 오차 median 0.000°. 그러므로 변환은 DL3DV 의
transforms.json 과 **글자 그대로 같다**: `c2w_cv = M @ _GL2CV` -> `w2c = inv(c2w_cv)`.
  !! `models/GenDoP/core/provider.py` 는 flip 없이 c2w 로 쓴다 — 따라하지 말 것.

프레임/이미지
  json 은 항상 정확히 **120 포즈** (uniform resample + Kalman smooth). `traj.txt` /
  `intrinsics.txt` 는 raw MonST3R 프레임 수(62~132)라 쓰지 않는다.
  이미지는 **`<shot>_rgb.png`** 다. `<scene>.mp4` 는 별도 재다운로드(854x480)라 프레임이 4배
  조밀하고 frame0 이 `_rgb.png` 와 다르다 (best match mp4 frame 1~6) -> **열지 않는다**.
  `_rgb.png` = MonST3R 입력 frame 0 = json frame 0 이다. json 의 `monst3r_im_id` 는 **1-base**
  프레임 번호로 전 코퍼스 22314/22314 가 정확히 1..120 연속이다 (2026-08-16 전수 확인) —
  즉 leading trim 을 이 필드로는 판별할 수 없고, 판별할 필요도 없다. 계획서의 "1.7% 탈락"은
  mp4 쪽 정합 진단(120 shot 표본 중 2개)이었지 `_rgb.png` 쪽 문제가 아니었다. 남은 것은
  **포맷 드리프트 감지용 구조 검사**뿐이다 (id 가 1..120 연속이 아니면 제외).

스케일 (분모 D). 모델이 보는 값은 `m = mean_t||C_t − C_0|| / D`.
  기본은 **`datadop_divisor='none'` = D 를 아예 안 건다 (norm_scale = 1)**. MonST3R 가 이미
  게이지를 걸어 놨기 때문이다 — global aligner 가 pairwise 스케일의 기하평균을 `base_scale=0.5`
  에 못 박아서 world 단위가 shot 마다 임의가 아니다 (results/datadop/monst3r_scale_normalization.md).
  `'meanray'` 는 frame0 의 `mean(depth * ||K^-1 [u+.5,v+.5,1]||)` 로 한 번 더 나눈다 — DL3DV 의
  저장된 avg_scale(`mean||P − C_ctx0||`)과 정의는 같지만 **이미 걸린 게이지에 덧거는** 셈이다.
  전 코퍼스 실측 (n=22314, results/datadop/scale_levels/, 필터 [0.005,1.0] 적용 후):
        D=1        med 0.0806  p95 0.4125  sd(log10 m) 0.488   keep 91.3%
        D=meanray  med 0.1790  p95 0.7519  sd(log10 m) 0.508   keep 87.2%
  D=1 이 기본인 근거는 **분산**이지 레벨 정합이 아니다 (sd(log10 m) 0.488 < 0.508, 그리고 위의
  게이지 논거). 레벨은 어느 쪽도 DL3DV 와 안 맞는다 — 2026-08-16 데이터셋 실측
  (`scripts/data/corpus_scale_probe.py`, cam_param[:,6:9] 직접 측정, n=300):
        DL3DV (da3_7k_da3geo_frontanchor)  med 0.4367  p95 0.8934  sd 0.285
        Scene-Decoupled (sd_whuman_v6)     med 0.0878  p95 0.2480  sd 0.365
        DataDoP (D=1, 필터 [0.005,1.0])    med 0.0911  p95 0.4095  sd 0.489
  즉 DataDoP 는 DL3DV 보다 **4.8배 낮고**, 대신 SD 와는 3.8% 안으로 붙는다. 레벨 정합은
  `norm_scale_gain` (dataset_dl3dv.py `_target_out`) 으로 코퍼스별로 따로 건다.
  (구 주석의 "p95 0.456 과 1.5% 차이" 는 재현되지 않는 옛 앵커였다 — 폐기.)
  `_depth.npy` 는 z-depth 이고 포즈와 같은 world 스케일임을 실측 확인 (상/하단 평균비 median
  1.945 로 disparity 가 아님, depth<=0 비율 0.0000).
  !! D=1 의 대가: `_geo_depth_maps` 의 `log(depth/norm_scale)` 가 `log(depth)` 가 되어 DL3DV
  (정의상 0 근처) 와 오프셋이 다르다 (DataDoP mean depth ~0.3 -> logd ~ -1.2). geo_custom
  경로를 쓸 때만 문제이고, 그때는 경고를 띄운다 — da3 encoder 는 이 채널을 안 쓴다.

V=1 의 구조적 귀결 (버그 아님)
  `geo_idxs=[0]` 이고 cam_param 앵커도 `extrinsics[0]` 이라 `rel = I` 가 된다 ->
  Plücker **moment 채널이 전부 0**, `geo_cam_param` 이 모든 sample 에서 상수.
  정보를 나르는 것은 ray direction(intrinsics)과 `geo_logd` 뿐이다. 이게 "첫 view 만 context"
  라는 정의 그대로다.

기존 두 경로(dl3dv / scene_decoupled)는 이 파일을 import 조차 하지 않는다
(base.py 가 `cfg.dataset_name == 'datadop'` 일 때만 분기).
"""
import json
import os
import os.path as osp

import numpy as np
import torch

from dataset_dl3dv import CamDataset, _GL2CV

DATADOP_ROOT_DEFAULT = "/data1/cympyc1785/data/DataDoP/DataDoP_with_scene"
# 공식 품질 필터 (28971 entry, '<scene>/<shot>' 한 줄씩). 없으면 필터를 걸지 않는다.
DATADOP_VALID_TXT = "/data1/cympyc1785/data/DataDoP/DataDoP_valid.txt"
N_POSES = 120                      # transforms_cleaning.json 은 항상 정확히 이 값


def _shot_stats(base):
    """shot 하나를 훑어 인덱스가 필요로 하는 것만 뽑는다 (worker process 에서 돈다).

    반환 `(ok, payload)`. payload 는 dict 이거나 실패 사유 문자열.
    무거운 것은 `_depth.npy` (h x w float) 하나뿐이고 나머지는 작은 json 이라, 24 proc 기준
    전 코퍼스(22314 shot) 스캔이 약 10 초다 -> 별도 precompute 스크립트 없이 인덱스에 인라인한다.
    """
    try:
        j = json.load(open(base + '_transforms_cleaning.json'))
        fr = j['frames']
        if len(fr) != N_POSES:
            return False, f'n_poses={len(fr)}'
        # 구조 검사: monst3r_im_id 는 1-base 프레임 번호이고 1..120 연속이어야 한다
        # (22314/22314 확인). frame0 이 1 이 아니면 `_rgb.png` 와 json frame0 이 어긋난
        # 새로운 포맷이라는 뜻이므로 조용히 쓰지 말고 뺀다.
        ids = [int(f.get('monst3r_im_id', -1)) for f in fr]
        if ids != list(range(1, N_POSES + 1)):
            return False, 'im_id_not_1..120'
        M = np.asarray([f['transform_matrix'] for f in fr], np.float64)      # (120,4,4) GL c2w
        if not np.isfinite(M).all():
            return False, 'pose_nonfinite'
        C = M[:, :3, 3]                     # GL->CV flip 은 오른쪽 곱이라 4열(카메라 중심) 불변
        mean_disp = float(np.linalg.norm(C - C[0], axis=1).mean())

        w, h = int(j['w']), int(j['h'])
        fx, fy = float(j['fl_x']), float(j['fl_y'])
        cx, cy = float(j['cx']), float(j['cy'])
        dep = np.load(base + '_depth.npy').astype(np.float32)                 # (h,w) z-depth
        if dep.shape != (h, w):
            return False, f'depth_shape={dep.shape}!=({h},{w})'
        if not np.isfinite(dep).all() or float(dep.min()) <= 0:
            return False, 'depth_nonfinite_or_nonpositive'
        # ray length = ||K^-1 [u+.5, v+.5, 1]||  (z-depth -> 광선 거리)
        ux = (np.arange(w, dtype=np.float64) + 0.5 - cx) / fx
        vy = (np.arange(h, dtype=np.float64) + 0.5 - cy) / fy
        ray = np.sqrt(ux[None, :] ** 2 + vy[:, None] ** 2 + 1.0)
        meanray = float((dep * ray).mean())
        if not np.isfinite(meanray) or meanray <= 0:
            return False, 'meanray_bad'

        cap = json.load(open(base + '_caption.json'))
        return True, {'mean_disp': mean_disp, 'meanray': meanray, 'caption': cap,
                      'hw': (h, w), 'K': (fx, fy, cx, cy)}
    except Exception as ex:                                   # 파일 결손/깨짐은 조용히 제외
        return False, f'{type(ex).__name__}: {ex}'


def _scan_one(args):
    base, rel = args
    ok, payload = _shot_stats(base)
    return rel, ok, payload


class DataDoPCamDataset(CamDataset):
    """scene_dir_list 의 원소 = shot **basedir prefix** (`<root>/<scene>/<shot>`, 확장자 없음),
    samples 의 원소 = `(shot_idx, 0, 120, caption, '<scene>__<shot>')`."""

    # SD 와 같은 이유로 항등 — seg-list 에 data_name 을 그대로 적는다.
    @staticmethod
    def seg_key(data_name):
        return data_name

    def __init__(self, cfg, type='train', only_segments=None):
        self.datadop_root = getattr(cfg, 'datadop_root', None) or DATADOP_ROOT_DEFAULT
        self.caption_key = getattr(cfg, 'datadop_caption_key', None) or 'Concise Interaction'
        self.divisor = str(getattr(cfg, 'datadop_divisor', 'none') or 'none')
        self.norm_gain = float(getattr(cfg, 'datadop_norm_gain', 1.0))
        self.trans_min = float(getattr(cfg, 'datadop_norm_trans_min', 0.005))
        self.trans_max = float(getattr(cfg, 'datadop_norm_trans_max', 1.0))
        self.window_stride = getattr(cfg, 'datadop_window_stride', None)
        self.index_workers = int(getattr(cfg, 'datadop_index_workers', 24))
        self.letterbox = bool(getattr(cfg, 'datadop_letterbox', False))
        self.norm_scale_list = []          # shot idx -> float (mean ray, gain 적용 전)
        self._K_list = []                  # shot idx -> (fx,fy,cx,cy)
        self._validate_cfg(cfg)
        super().__init__(cfg, type=type, only_segments=only_segments)

    # ------------------------------------------------------------------ cfg 검증
    def _validate_cfg(self, cfg):
        """DataDoP 에서 **성립하지 않는** 설정을 조용히 통과시키지 않는다.

        V=1 / 프레임 시퀀스 없음 / intrinsics 격자가 DL3DV 와 다름 — 이 셋에서 파생되는 것 전부.
        """
        def g(k, d=None):
            return getattr(cfg, k, d)

        bad = []
        sm = str(g('scale_mode', 'avg_scale') or 'avg_scale')
        if sm in ('context_longer', 'ctx_longer_135max', 'geo_lagernvs', 'first_farthest_135'):
            bad.append(f"scale_mode={sm!r}: context range 개념이 없다 (shot 당 궤적 1개). "
                       f"분모는 frame0 mean ray 하나로 고정된다")
        if str(g('avg_scale_ref', 'centroid') or 'centroid') != 'centroid':
            bad.append("avg_scale_ref: DataDoP 는 저장된 avg_scale 파일을 안 쓴다 "
                       "('centroid' 로 두고 datadop_divisor 로 제어)")
        if str(g('intr_norm', 'auto')) == 'raw':
            bad.append("intr_norm='raw': fx/w 가 DL3DV 의 2~4배라 다른 코퍼스와 축이 안 맞는다 "
                       "('rel' 을 쓸 것 -> cx==w/2 가 정확해서 [1,1] 이 된다)")
        if int(g('geo_num_views', 1) or 1) > 1:
            bad.append(f"geo_num_views={g('geo_num_views')}: context 이미지가 frame0 한 장뿐이다")
        vs = str(g('geo_view_sampling', 'even') or 'even')
        if vs != 'even':
            bad.append(f"geo_view_sampling={vs!r}: 고를 view 가 없다 ('even' + V=1)")
        if g('geo_cover_out_of_seg') or g('geo_cover_before_only'):
            bad.append("geo_cover_*: frustum_cover 경로는 프레임 시퀀스를 요구한다")
        if g('geo_posed') and not g('geo_first_view_target_s'):
            # DataDoP 에서는 geo view0 = json frame0 = target segment 첫 프레임 s 라서
            # "view0 가 앵커 프레임"이 **구조적으로 참**이다. False 로 두면 dataset_cfg 가
            # 앵커 불일치 경고를 띄우는데(:348) 그 경고가 여기서는 거짓이라 헷갈린다.
            bad.append("geo_first_view_target_s=False: DataDoP 는 view0 가 곧 frame s 다 "
                       "(True 로 둘 것 — 경고가 거짓이 된다)")
        for k in ('geo_swap_mode', 'geo_test_inseg_k', 'geo_shuffle_order'):
            if g(k):
                bad.append(f"{k}: V=1 에서 의미가 없다")
        if g('geo_latent_cache_dir'):
            bad.append("geo_latent_cache_dir: 캐시 경로 규약이 DL3DV 전용이다 (data_name split)")
        nf = int(g('num_frames', 49) or 49)
        if nf > N_POSES:
            bad.append(f"num_frames={nf} > {N_POSES} (json 이 항상 {N_POSES} 포즈다)")
        if str(g('geo_encoder', '') or '') == 'lagernvs':
            bad.append("geo_encoder='lagernvs': cross-view 재구성기라 V=1 이면 무의미하다")
        if self.divisor not in ('none', 'meanray'):
            bad.append(f"datadop_divisor={self.divisor!r} ('none' = D 1 | 'meanray')")
        # _validate_cfg 는 super().__init__ 전에 돈다 -> self.geo_* 가 아직 없다. cfg 로만 본다.
        if self.divisor == 'none' and str(g('geo_encoder', '')) == 'custom' \
                and str(g('custom_geo_channels', 'full') or 'full') == 'full':
            # 여기서 raise 하지 않는 이유: D=1 은 translation 레벨을 맞추기 위한 의도적 선택이고,
            # 어긋나는 건 depth 채널의 **오프셋**뿐이라 arm 에 따라 감수할 수 있다.
            print("[DataDoP] WARNING: divisor='none' + geo_custom(full) -> geo_logd = log(depth) "
                  "라 DL3DV(≈0) 와 오프셋이 다르다 (~-1.2). 코퍼스 식별 지름길이 될 수 있다.")
        if bad:
            raise ValueError("dataset_name='datadop' 와 맞지 않는 설정:\n  - "
                             + "\n  - ".join(bad))

    # ------------------------------------------------------------------ index
    def _load_index(self):
        """전 코퍼스 shot 을 훑어 (samples, scene_dir_list, hw_list, norm_scale_list) 를 만든다.

        캐시 키에 필터/캡션/프레임수를 전부 넣는다 — 하나라도 빠지면 옛 캐시를 조용히 재사용해서
        "필터를 바꿨는데 샘플 수가 그대로"인 사고가 난다.
        """
        self.root = self.datadop_root
        if self.only_segments is not None:
            return self._load_index_subset()

        key = (f"datadop__nf{self.num_frames}__cap{_slug(self.caption_key)}"
               f"__d{self.divisor}__g{self.norm_gain:g}"
               f"__t{self.trans_min:g}-{self.trans_max:g}"
               f"__ws{self.window_stride}__ms{self.max_scenes}")
        cache_dir = osp.join(self.root, '.latentcam_index')
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = osp.join(cache_dir, key + '.pt')
        if osp.isfile(cache_path):
            idx = torch.load(cache_path, weights_only=False)
            self.samples = idx['samples']
            self.hw_list = idx['hw_list']
            self.norm_scale_list = idx['norm_scale_list']
            self._K_list = idx['K_list']
            self.scene_dir_list = [osp.join(self.root, r) for r in idx['shot_rel']]
            print(f"[DataDoP index cache] {len(self.samples)} shots <- {cache_path}")
            return

        rels = self._candidate_shots()
        print(f"[DataDoP index build] {len(rels)} candidate shots "
              f"({self.index_workers} proc; cached to {cache_path})")
        from concurrent.futures import ProcessPoolExecutor
        from tqdm import tqdm
        jobs = [(osp.join(self.root, r), r) for r in rels]
        drop = {}
        with ProcessPoolExecutor(max_workers=self.index_workers) as ex:
            for rel, ok, payload in tqdm(ex.map(_scan_one, jobs, chunksize=32), total=len(jobs)):
                if not ok:
                    drop[payload.split(':')[0]] = drop.get(payload.split(':')[0], 0) + 1
                    continue
                D = payload['meanray'] if self.divisor == 'meanray' else 1.0
                m = payload['mean_disp'] / D              # 모델이 보는 translation 크기
                if not (self.trans_min <= m <= self.trans_max):
                    drop['trans_filter'] = drop.get('trans_filter', 0) + 1
                    continue
                cap = payload['caption']
                text = cap.get(self.caption_key) if isinstance(cap, dict) else str(cap)
                if not text:
                    drop['no_caption'] = drop.get('no_caption', 0) + 1
                    continue
                i = len(self.scene_dir_list)
                self.scene_dir_list.append(osp.join(self.root, rel))
                self.hw_list.append(payload['hw'])
                self._K_list.append(payload['K'])
                self.norm_scale_list.append(D)
                for s, e in self._segments():
                    self.samples.append((i, s, e, text, rel.replace('/', '__')))
        print(f"[DataDoP index build] kept {len(self.samples)} samples / "
              f"{len(self.scene_dir_list)} shots | dropped {drop}")
        torch.save({'samples': self.samples, 'hw_list': self.hw_list,
                    'norm_scale_list': self.norm_scale_list, 'K_list': self._K_list,
                    'shot_rel': [osp.relpath(d, self.root) for d in self.scene_dir_list]},
                   cache_path)

    def _segments(self):
        """shot 당 (s,e) 목록. 기본은 통째로 하나 — 캡션이 shot **전체**를 묘사하기 때문이다
        ("정지 -> 좌측 이동 -> 복귀"). stride 를 주면 슬라이딩 윈도우가 되지만 그때는 캡션과
        내용이 체계적으로 어긋난다는 것을 알고 써야 한다."""
        if not self.window_stride:
            return [(0, N_POSES)]
        st = int(self.window_stride)
        return [(s, s + self.num_frames)
                for s in range(0, N_POSES - self.num_frames + 1, st)]

    def _candidate_shots(self):
        """'<scene>/<shot>' 상대 prefix 목록. DataDoP_valid.txt 가 있으면 그 교집합만."""
        valid = None
        if osp.isfile(DATADOP_VALID_TXT):
            with open(DATADOP_VALID_TXT) as f:
                valid = {ln.strip() for ln in f if ln.strip()}
        scenes = sorted(d for d in os.listdir(self.root)
                        if osp.isdir(osp.join(self.root, d)) and not d.startswith('.'))
        if self.max_scenes:
            scenes = scenes[:int(self.max_scenes)]
        rels = []
        for sc in scenes:
            sd = osp.join(self.root, sc)
            for fn in sorted(os.listdir(sd)):
                if not fn.endswith('_transforms_cleaning.json'):
                    continue
                shot = fn[:-len('_transforms_cleaning.json')]
                rel = f'{sc}/{shot}'
                if valid is None or rel in valid:
                    rels.append(rel)
        return rels

    def _load_index_subset(self):
        """only_segments (추론/렌더용) — 필터도 캐시도 없이 지정한 shot 만."""
        for seg in self.only_segments:
            rel = seg if isinstance(seg, str) else '/'.join(seg)
            rel = rel.replace('__', '/') if '__' in rel else rel
            ok, payload = _shot_stats(osp.join(self.root, rel))
            if not ok:
                raise ValueError(f"DataDoP shot {rel!r}: {payload}")
            cap = payload['caption']
            text = cap.get(self.caption_key) if isinstance(cap, dict) else str(cap)
            i = len(self.scene_dir_list)
            self.scene_dir_list.append(osp.join(self.root, rel))
            self.hw_list.append(payload['hw'])
            self._K_list.append(payload['K'])
            self.norm_scale_list.append(
                payload['meanray'] if self.divisor == 'meanray' else 1.0)
            for s, e in self._segments():
                self.samples.append((i, s, e, text or '', rel.replace('/', '__')))

    # ------------------------------------------------------------------ shot 로딩
    def _load_scene(self, i):
        """shot i 의 포즈/K/이미지경로. `frames` 는 이미 시간순이고 `file_path` 가 없으니
        **정렬하지 말 것** (DL3DV 의 `_parse_transforms` 와 다른 유일한 점)."""
        c = self._scene_cache.get(i)
        if c is not None:
            return c
        base = self.scene_dir_list[i]
        j = json.load(open(base + '_transforms_cleaning.json'))
        M = torch.tensor([f['transform_matrix'] for f in j['frames']], dtype=torch.float32)
        c2w = M @ _GL2CV                                  # OpenGL c2w -> OpenCV c2w
        w2c = torch.linalg.inv(c2w)
        fx, fy, cx, cy = self._K_list[i]
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                         dtype=torch.float32).unsqueeze(0).repeat(M.shape[0], 1, 1)
        c = {'w2c': w2c, 'intr': K, 'frame_files': base + '_rgb.png',
             'hw': self.hw_list[i]}
        self._scene_cache[i] = c
        return c

    # ------------------------------------------------------------------ 입력 3종
    def _load_images(self, rgb_path, idxs):
        """`frame_files` 자리에 png 경로 하나가 온다. context 는 frame0 뿐이라 idxs != [0] 은 버그.

        비등방 resize (예: 512x288 -> 252x448) 는 안전하다 — `_geo_pixel_plucker` 가 정규화
        좌표에서 원본 K 를 다시 걸고, DA3 `build_cam_token` 의 fov 도 full-frame resize 에
        불변이다. 다만 h 가 160~416 로 넓어 34% 는 종횡비가 DL3DV 와 크게 다르다
        (datadop_letterbox 는 그 대안을 위한 자리 — 아직 미구현).
        """
        if list(idxs) != [0]:
            raise ValueError(f"DataDoP context 는 frame0 한 장뿐이다 (got geo_idxs={list(idxs)})")
        if self.letterbox:
            raise NotImplementedError("datadop_letterbox 는 아직 미구현이다 (플래그 자리만 있다)")
        from PIL import Image
        H, W = self.geo_hw
        im = Image.open(rgb_path).convert('RGB').resize((W, H), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0                     # (H,W,3)
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)         # (1,3,H,W)

    def _da3_depth(self, shot_idx, geo_idxs):
        """`<shot>_depth.npy` (h,w) -> (1,h,w). 포즈와 같은 world 스케일이라 보정이 없다."""
        if list(geo_idxs) != [0]:
            raise ValueError(f"DataDoP depth 는 frame0 한 장뿐이다 (got {list(geo_idxs)})")
        dep = np.load(self.scene_dir_list[shot_idx] + '_depth.npy').astype(np.float32)
        return torch.from_numpy(dep[None])

    # ------------------------------------------------------------------ item
    def __getitem__(self, idx):
        shot, s, e, caption, data_name = self.samples[idx]
        extrinsics = self.extrinsics_list[shot][s:e]          # (T,4,4) w2c OpenCV
        intrinsics = self.intrinsics_list[shot][s:e]
        h, w = self.hw_list[shot]
        if extrinsics.shape[0] > self.num_frames:
            # _even_indices 는 index 0 을 유지한다 -> target 첫 프레임 = json frame0
            # = context 이미지 = depth 프레임이 정확히 일치한다.
            sel = self._even_indices(extrinsics.shape[0], self.num_frames)
            extrinsics = extrinsics[sel]
            intrinsics = intrinsics[sel]

        # 부모의 scale_mode dispatch 를 타지 않는다 (context range 개념이 없다).
        norm_scale = torch.tensor(
            [self.norm_scale_list[shot] / max(self.norm_gain, 1e-8)]).clamp(min=1e-5)
        out = self._target_out(extrinsics, intrinsics, norm_scale, h, w, caption, data_name)
        norm_scale = out['norm_scale']
        # 'corpus' 키는 여기서 붙이지 않는다 — collate_fn 이 batch[0] 의 키만 순회하므로
        # 코퍼스마다 키 집합이 다르면 혼합 배치에서 조용히 유실된다. MixedCamDataset 이
        # 세 코퍼스 모두에 동일하게 붙인다.

        if not self.geo_enabled:
            return out

        geo_idxs = [0]
        out['images'] = self._load_images(self.frame_files_list[shot], geo_idxs)
        out['geo_idxs'] = torch.tensor(geo_idxs, dtype=torch.long)
        if self.geo_return_idxs:
            self._attach_geo_ctx(out, shot, geo_idxs)
        if self.geo_cam_embed is not None:
            # rel = w2c_0 @ inv(w2c_0) = I -> Plücker moment 가 전부 0 (위 docstring 참고).
            out['geo_cam_param'] = self._geo_cam_cond(
                shot, geo_idxs, extrinsics[0], norm_scale, (h, w))
        if self.geo_custom:
            if self.geo_custom_channels != 'rgb_only':
                out['geo_plucker_map'] = self._geo_pixel_plucker(
                    shot, geo_idxs, extrinsics[0], norm_scale, (h, w))
            if self.geo_custom_channels == 'full':
                out['geo_logd'], out['geo_valid'] = self._geo_depth_maps(shot, geo_idxs, norm_scale)
        if self.geo_posed:
            gw2c = self.extrinsics_list[shot][geo_idxs].float()
            gK = self.intrinsics_list[shot][geo_idxs].float()
            out['geo_c2w'] = torch.linalg.inv(gw2c)
            out['geo_fxfycxcy'] = torch.stack(
                [gK[:, 0, 0], gK[:, 1, 1], gK[:, 0, 2], gK[:, 1, 2]], dim=-1)
            out['geo_hw'] = torch.tensor([float(h), float(w)])
        return out


def _slug(s):
    return ''.join(ch if ch.isalnum() else '-' for ch in str(s))[:24]
