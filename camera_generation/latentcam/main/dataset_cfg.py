"""dataset cfg 해석 — `CamDataset.__init__` 이 하던 config 읽기/검증/폴백을 한곳으로.

왜 떼어냈나: `CamDataset.__init__` 은 190 줄이었고 그중 ~150 줄이 `getattr` + 검증 + print
였다. 실제 초기화(리스트 만들기, 인덱스 로드)는 20 줄 남짓인데 그게 config 해석에 파묻혀
있어서, "이 arm 에서 결국 어떤 값이 서는가"를 알려면 190 줄을 순서대로 읽어야 했다.

떼어낸 세 가지 이득:
  1. `__init__` 이 "spec 받아서 속성에 꽂고 인덱스 만든다"로 납작해진다.
  2. **디스크를 안 탄다.** `resolve_dataset_cfg` 는 순수 함수라 38k 샘플 인덱스를 만들지 않고도
     config 해석만 밀리초 단위로 검사할 수 있다 (`scripts/test/golden_dataset.py` 는 실제
     `ds[i]` 까지 봐야 해서 훨씬 느리다).
  3. 아래 **비활성화 캐스케이드**가 한 화면에 들어온다.

!! 이건 평평한 매핑이 아니라 **순서 있는 상태 기계**다. `geo_latent_cache_dir` 은 한 번 켜졌다가
세 곳에서 차례로 꺼진다:
     geo_latent_dim != 768   -> proj 가 학습되는 Linear 라 그 출력을 얼리면 안 된다
     geo_encoder == 'custom' -> 인코더 자체가 학습된다
     geo_encoder == 'da3'    -> 같은 이유 (LayerNorm/proj 가 학습된다)
     geo_test_inseg_k        -> 캐시 키가 data_name 뿐이라 context view 변경을 구분 못 한다
     geo_swap_mode           -> 같은 이유
순서와 print 문구를 바꾸면 "캐시가 켜진 줄 알았는데 안 켜졌다"류의 무증상 사고가 난다.
동작 동일성은 `scripts/test/golden_dataset.py --mode check` 로 검증한다.

`resolve_scale_mode` / `resolve_pose_source` 도 여기로 옮겼다 (dataset_dl3dv 가 re-export 하므로
`from dataset_dl3dv import resolve_scale_mode` 는 그대로 동작한다).
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

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

# avg_scale_ref -> <scene>/da3/<subdir>. 저장된 avg_scale 을 **어느 context range 에서 어느
# 기준점으로** 잰 파일로 읽을지. 점 집합 규약(conf >= 전역 P40, pixel_stride 2, mean 거리)은
# 넷 다 같고 range/기준점만 다르다. 생성기:
#   centroid / context_first_cam        pipeline/workspace/make_avg_scale_da3{,_firstcam}.py
#   front_first_anchor{,_same_len}      scripts/data/make_avg_scale_da3_front_anchor.py
#   front_centroid{,_same_len}          scripts/data/make_avg_scale_da3_front_anchor.py
#   da3latent                           scripts/data/make_avg_scale_da3_latent.py
#
# 축은 2x2 다 — **range**(어느 프레임에서 점을 모으나 = 인과성) x **origin**(어디서 거리를 재나):
#                       range                       origin
#   centroid            [0,s)/[e,N) 중 긴 쪽         context 카메라 centroid  (비인과, ~50%가 미래)
#   context_first_cam   같음                         context range 첫 카메라  (비인과)
#   front_centroid      [0,s)                        context 카메라 centroid  (인과)
#   front_first_anchor  [0,s)                        target 첫 카메라 s       (인과)
# front_centroid 는 centroid 와 **range 만** 다르다 -> "인과성 비용"을 origin 변경과 분리해 잰다.
AVG_SCALE_DIRS = {
    'centroid':                      'avg_scale',
    'context_first_cam':             'avg_scale_context_first_cam',
    'front_first_anchor':            'avg_scale_front_first_anchor',
    'front_first_anchor_same_len':   'avg_scale_front_first_anchor_same_len',
    # [new 2026-08-16] 2x2 의 빈 칸 (앞쪽 range + centroid origin).
    'front_centroid':                'avg_scale_front_centroid',
    'front_centroid_same_len':       'avg_scale_front_centroid_same_len',
    # [new 2026-08-13] 위 넷은 점 구름에서 잰 **기하학적 거리**지만 이건 **DA3 latent 이 실제로
    # 쓰는 카메라 스케일**이다: M x sigma. M = cam_token 을 만들 때 쓴 per-sample median camera
    # distance, sigma = cam_dec 가 예측한 카메라 center 를 우리가 넣어 준 center 에 맞추는
    # Umeyama scale. cam_enc 로 pose 를 줘도 cam_dec 의 translation 은 항상 예측값이라
    # (cam_dec.py:35 에 echo 경로가 없다) 둘이 어긋난다 -- 200 세그먼트 실측 sigma med 2.90040,
    # ±5% 안이 1.0% 뿐. 이 분모로 나누면 target 궤적이 geo latent 과 같은 스케일 공간에 놓인다.
    # !! front_uniform view 집합에 의존해서 만든 값이라 geo_view_sampling='front_uniform' +
    #    geo_num_views 가 생성 때와 같아야 의미가 있다 (아래 resolve 에서 경고한다).
    'da3latent':                     'avg_scale_da3latent',
    # [new 2026-08-17] 위와 **같은 정의, 다른 view 집합**. da3latent 는 front_uniform 이 고른
    # 6 view 로 만든 값이라 frustum_cover arm 에서는 쓸 수 없다 (M 도 sigma 도 view 에 의존).
    # 이건 frustum_cover(out_of_seg/before_only/centered_at_s) 짝이다 ->
    # da3_7k_da3geo_frontanchor 와 **분모 하나만** 다른 arm 을 만들 수 있다.
    #   200 세그 dry-run: sigma med 2.72238, M*sigma med 3.26031,
    #   ratio(이것/front_first_anchor) med 0.87966 p05 0.41889 p95 1.12623 (log10 std 0.1726 dex)
    'da3latent_cover':               'avg_scale_da3latent_cover',
    # [new 2026-08-30] Vista4D 전용. 위 넷의 2x2 축(range x origin)이 아니라 **점 집합 규약이
    # 다르다** -- conf>=P40 이 아니라 non-sky 전 픽셀, stride 2, z 를 ray 길이로 곱한 거리.
    #   context_first_cam (Vista4D 뱅크에서) = scene_graph.json:scale.S = **frame0 한 장**의
    #     non-sky 평균 ray 길이. context range 는 영상 전체 [0,49) 인데 분모만 첫 프레임이다.
    #   ctx_all_first_cam                    = **전 프레임** depth 를 unproject 한 점군에서
    #     첫 카메라까지의 평균 거리. 기준점(첫 카메라)은 같고 점 집합만 구간 전체로 넓혔다.
    # 생성기: models/Planner/CinemaTraj/scripts/make_avg_scale_vista4d_ctxall.py
    #   (정의는 trumans_to_recon.py:avg_scale_first_cam 을 import 해서 그대로 쓴다)
    # 52편 실측 ratio(ctxall/S) med 1.0110 min 0.8258(soapbox) max 6.1598(camera-lens) --
    # 대부분은 거의 안 변하고, 카메라가 크게 빠지는 씬(camera-lens 는 z_med 가 1.06->20.84)
    # 에서만 분모가 커진다. 그게 이 arm 이 검증하려는 지점이다.
    'ctx_all_first_cam':             'avg_scale_ctx_all_first_cam',
}
AVG_SCALE_REFS = tuple(AVG_SCALE_DIRS)

# [new 2026-08-27] target 궤적만 갈아끼우는 arm. None(기본) 이면 기존 동작 그대로 —
# target 도 context 와 **같은** pose 배열(soure recon)에서 슬라이스한다.
#   'da3_target_poses'  <scene>/da3/target_poses.npz 의 합성 궤적을 target 으로 쓴다.
#     context(이미지·pose·avg_scale 분모)는 소스 그대로라 "이 씬을 이렇게 찍었다면" 을 배운다.
#     Vista4D LBM-Lite pseudo-GT 뱅크가 이 포맷으로 export 된다.
TARGET_POSE_SOURCES = {'da3_target_poses'}

# 앞쪽 context 를 요구하는 변형 -> 세그먼트가 s 앞에 최소 몇 프레임을 가져야 하는가.
# 값이 None 이면 제약 없음. 'nf' 는 num_frames (= target segment 길이) 로 치환된다.
#   front_first_anchor           s >= 1        ([0,s) 가 비면 분모를 못 만든다)
#   front_first_anchor_same_len  s >= num_frames  ([s-L,s) 를 꽉 채워야 한다)
# DL3DV da3 코퍼스는 세그먼트가 전부 길이 49 이고 s 가 0,49,98,... 이라 두 조건이 s==0 제외로
# 같아진다 -> 두 arm 의 학습 샘플 집합이 동일해서 paired 비교가 성립한다.
#   da3latent                    s >= num_frames  (front_uniform 짝이라 같은 제약)
AVG_SCALE_MIN_FRONT = {
    'front_first_anchor':          1,
    'front_first_anchor_same_len': 'nf',
    'da3latent':                   'nf',
    # frustum_cover 짝은 [s-L,s) 를 꽉 채울 필요가 없다 -> 짝 arm(front_first_anchor)과 같은 1.
    # DL3DV da3 는 s in {0,49,98,...} 이라 1 이든 'nf' 든 결과 집합은 같지만(s==0 만 제외),
    # 의도를 "control 과 동일 인덱스"로 못박아 둔다.
    'da3latent_cover':             1,
    # [new 2026-08-16] range 가 같으니 제약도 같다 (origin 만 다르다).
    'front_centroid':              1,
    'front_centroid_same_len':     'nf',
}


def avg_scale_min_front(avg_scale_ref, num_frames):
    """avg_scale_ref 가 요구하는 최소 앞쪽 context 길이 (없으면 0)."""
    v = AVG_SCALE_MIN_FRONT.get(str(avg_scale_ref))
    return 0 if v is None else (int(num_frames) if v == 'nf' else int(v))


def resolve_pose_source(cfg):
    ps = getattr(cfg, 'pose_source', 'transforms') or 'transforms'
    if ps not in _POSE_SOURCES:
        raise ValueError(f"pose_source must be one of {_POSE_SOURCES}, got {ps!r}")
    return ps


@dataclass
class DatasetSpec:
    """`CamDataset` 가 config 에서 읽어 들이는 값 전부. 필드 이름 = 데이터셋 속성 이름이라
    `__init__` 이 그대로 꽂을 수 있고, 기존 `ds.geo_cover_k` 같은 외부 접근도 안 깨진다.

    여기 없는 것: 상태(`_scene_cache`, `_plucker_grid`, `_scene2samples`)와 `only_segments` 에
    의존하는 `lazy` — 둘 다 config 가 아니라서 `__init__` 에 남는다.
    """
    # --- 코퍼스 / 경로
    pose_source: str
    avg_scale_ref: str
    # None(기본) = 기존 동작. target 카메라를 context 카메라와 **같은** pose 배열에서 가져온다.
    # 'da3_target_poses' = target 만 <scene>/da3/target_poses.npz 의 합성 궤적으로 갈아끼운다
    #   (context 이미지·pose·avg_scale 은 그대로 소스). Vista4D pseudo-GT arm 이 쓴다.
    target_pose_source: Optional[str]
    root: str
    num_frames: int
    max_scenes: Optional[int]
    lazy_dataset: bool
    # --- geo 인코더 일반
    geo_enabled: bool
    geo_num_views: int
    geo_hw: Tuple[int, int]
    geo_cam_embed: Optional[str]
    geo_posed: bool
    geo_return_idxs: bool
    geo_latent_cache_dir: Optional[str]
    # [new 2026-08-29] scene 키 pre-ln DA3 캐시. geo_latent_cache_dir 과 **동시에 켤 수 없다**
    # (둘 다 켜면 위 캐시가 먼저 return 해서 이쪽이 죽은 코드가 된다 — dataset_cfg 가 막는다).
    geo_raw_cache_dir: Optional[str]
    geo_raw_cache_preload: bool
    # --- custom geo (RGBD) 경로
    geo_custom: bool
    geo_custom_channels: str
    geo_depth_cache_dir: Optional[str]
    # --- context view 선택
    geo_view_sampling: str
    geo_shuffle_order: bool
    geo_first_view_target_s: bool
    # frustum_cover
    geo_cover_k: int
    geo_cover_radius: float
    geo_cover_ndepth: int
    geo_cover_out_of_seg: bool
    geo_cover_before_only: bool
    geo_cover_centered_at_s: bool
    # hybrid (in-segment + co-visibility)
    geo_num_inseg: int
    geo_num_covis: int
    geo_inseg_span: Optional[int]
    covis_radius: float
    covis_theta0: float
    covis_max_axis_deg: float
    # --- test-time probe (학습 config 는 전부 off)
    geo_test_inseg_k: int
    geo_swap_mode: Optional[str]
    geo_swap_shift: int
    geo_swap_keep_first: bool

    def apply_to(self, ds):
        """spec 의 모든 필드를 데이터셋 인스턴스 속성으로 꽂는다 (`lazy_dataset` 제외 —
        `only_segments` 와 합쳐 `ds.lazy` 가 되므로 호출자가 처리한다)."""
        for f in self.__dataclass_fields__:
            if f != 'lazy_dataset':
                setattr(ds, f, getattr(self, f))


def resolve_dataset_cfg(cfg, verbose=True):
    """cfg -> DatasetSpec. 순수 함수 (디스크 접근 없음). print 순서는 리팩토링 전 `__init__` 과
    동일하게 유지한다 — 로그를 grep 하는 습관이 깨지지 않게."""
    def say(msg):
        if verbose:
            print(msg)

    pose_source = resolve_pose_source(cfg)
    # [new 2026-08-10] da3 의 저장된 avg_scale 을 어느 기준점에서 잰 파일로 읽을지.
    # 'centroid'(기본) = 기존 동작, 'context_first_cam' = <scene>/da3/avg_scale_context_first_cam.
    # [new 2026-08-13] 'front_first_anchor' / 'front_first_anchor_same_len' 은 context range 를
    # target 앞쪽으로 고정하고 기준점을 **target segment 첫 카메라 s** 로 잡은 변형이다
    # (scripts/data/make_avg_scale_da3_front_anchor.py). cam_param 의 재고정 기준(frame s)과
    # 분모의 기준점을 일치시킨다. 앞쪽 context 를 못 채우는 세그먼트는 파일이 없고,
    # dataset_dl3dv._load_index 가 인덱스에서 아예 뺀다.
    avg_scale_ref = str(getattr(cfg, 'avg_scale_ref', 'centroid') or 'centroid')
    if avg_scale_ref not in AVG_SCALE_REFS:
        raise ValueError(f"avg_scale_ref must be one of {sorted(AVG_SCALE_REFS)}, "
                         f"got {avg_scale_ref!r}")
    if avg_scale_ref != 'centroid' and pose_source != 'da3':
        raise ValueError(f"avg_scale_ref={avg_scale_ref!r} 는 pose_source='da3' 에서만 "
                         f"쓸 수 있다 (해당 디렉토리가 da3 아래에만 있다). "
                         f"현재 pose_source={pose_source!r}")

    # [new 2026-08-27] target 궤적만 합성 pseudo-GT 로 갈아끼우는 arm (Vista4D LBM-Lite 뱅크).
    # context(이미지·pose·분모)는 소스 그대로라 "이 씬을 이렇게 찍었다면" 을 배우게 된다.
    # 파일은 da3 아래에만 있으므로 pose_source='da3' 를 강제한다 — 어긋나면 조용히 소스 궤적으로
    # 학습되고 로그에는 아무 표시도 안 남는다.
    target_pose_source = getattr(cfg, 'target_pose_source', None) or None
    if target_pose_source is not None:
        if target_pose_source not in TARGET_POSE_SOURCES:
            raise ValueError(f"target_pose_source must be one of {sorted(TARGET_POSE_SOURCES)} "
                             f"or null, got {target_pose_source!r}")
        if pose_source != 'da3':
            raise ValueError(f"target_pose_source={target_pose_source!r} 는 pose_source='da3' "
                             f"에서만 쓸 수 있다. 현재 pose_source={pose_source!r}")
        say(f"[target pose] <scene>/da3/target_poses.npz 로 target 궤적 교체 "
            f"(context 는 소스 그대로)")

    geo_hw = tuple(getattr(cfg, 'geo_image_hw', (256, 448)))
    geo_enabled = bool(getattr(cfg, 'geo_encoder', None))   # skip image loading for text-only

    # [cascade 1/4] precomputed frozen geo-latent cache (cache_geo_embeddings.py). None = OFF, i.e.
    # load images + run LagerNVS every step exactly as before. Only safe when the encoder's
    # proj is Identity (lagernvs native 768 == geo_latent_dim); otherwise proj is trainable
    # and its output must not be frozen into a file.
    geo_latent_cache_dir = None
    _cdir = getattr(cfg, 'geo_latent_cache_dir', None)
    if _cdir and geo_enabled:
        if int(getattr(cfg, 'geo_latent_dim', 768)) != 768:
            say(f"[geo cache] DISABLED: geo_latent_dim="
                f"{getattr(cfg, 'geo_latent_dim')} != 768 -> GeoEncoder.proj is a trainable "
                f"Linear, its output must not be cached")
        else:
            sub = ('first_cam_included' if getattr(cfg, 'geo_first_view_target_s', False)
                   else 'first_cam_not_included')
            import os.path as osp
            geo_latent_cache_dir = osp.join(_cdir, sub)
            say(f"[geo cache] reading {geo_latent_cache_dir}/<iK>/<data_name>.pt "
                f"(miss -> on-the-fly LagerNVS)")

    # per-context-view camera embedding concatenated onto the geo tokens.
    # None = OFF (unchanged geo condition). 'relfirst' = 11-d pose relative to the target's
    # first camera (per-VIEW, broadcast over that view's patches); 'plucker' = 6-d Plücker ray
    # per PATCH token in the same frame. See conf/config.yaml and dataset_dl3dv 의
    # _geo_cam_param / _geo_cam_plucker.
    geo_cam_embed = getattr(cfg, 'geo_cam_embed', None)
    if geo_cam_embed not in (None, 'relfirst', 'plucker'):
        raise ValueError(f"geo_cam_embed must be null | 'relfirst' | 'plucker', "
                         f"got {geo_cam_embed!r}")

    # [new 2026-08-07] geo_encoder='custom' (models/custom_geo_encoder.py) 전용 RGBD 경로.
    # lagernvs 경로는 아래 어디도 건드리지 않는다 -- 이 플래그가 False 면 __getitem__ 의
    # 동작은 이전과 bit-identical.
    #
    # custom 인코더는 카메라를 view 당 11-d cam_token 이 아니라 **픽셀당 Plücker ray** 로 받는다.
    # 그래서 dataset 이 내보내야 할 것이 세 개 더 있다 (custom_geo_encoder.py 의 입력 계약):
    #   geo_plucker_map (V,6,Hc,Wc)  target segment 첫 카메라 프레임, trans / norm_scale
    #   geo_logd        (V,1,Hc,Wc)  log(depth / norm_scale)  -- Plücker 와 같은 단위여야
    #                                ray x depth 가 3D 점이 된다
    #   geo_valid       (V,1,Hc,Wc)  depth 유효 마스크
    # 해상도 Hc,Wc 는 DINOv2 patch(14) 의 배수여야 하고, GeoTokenizer 가 stride 14 로
    # patchify 한 토큰 수가 DINO patch 토큰 수와 정확히 같아야 한다 (custom_geo_encoder.py 가
    # 불일치를 raise). 그래서 geo_hw 자체를 custom_geo_input_hw 로 바꿔서 _load_images 도
    # 처음부터 그 격자로 주게 한다 -- 인코더 안의 bilinear 재보간을 피하면 Plücker 방향
    # 단위벡터가 정확히 보존된다.
    geo_custom = geo_enabled and str(cfg.geo_encoder) == 'custom'
    # [ablation 2026-08-10] custom_geo_channels: full | no_depth | rgb_only.
    # 안 쓰는 입력은 아예 만들지 않는다 (depth 는 mmap 이라도 view 당 I/O 가 있다).
    geo_custom_channels = str(getattr(cfg, 'custom_geo_channels', 'full') or 'full')
    geo_depth_cache_dir = None
    if geo_custom:
        _p = int(getattr(cfg, 'custom_geo_patch', 14))
        _ihw = getattr(cfg, 'custom_geo_input_hw', None)
        geo_hw = (tuple(int(v) for v in _ihw) if _ihw else
                  ((geo_hw[0] // _p) * _p, (geo_hw[1] // _p) * _p))
        if any(v % _p for v in geo_hw):
            raise ValueError(f"custom_geo_input_hw={geo_hw} must be a multiple of patch {_p}")
        if pose_source != 'da3':
            # depth 는 <scene>/da3/depth.npz 에만 있고, 그 depth 는 da3 pose/intrinsics 와
            # 같은 스케일 공간이다. transforms pose 와 섞으면 ray x depth 가 무의미해진다.
            raise ValueError("geo_encoder='custom' requires pose_source='da3' "
                             f"(depth comes from <scene>/da3/depth.npz), got {pose_source!r}")
        if geo_latent_cache_dir is not None:      # [cascade 2/4]
            # GeoTokenizer/proj 가 **학습되는** 파라미터라 그 출력을 파일로 얼리면 안 된다.
            say("[geo cache] DISABLED: geo_encoder='custom' -> the geo encoder is trainable")
            geo_latent_cache_dir = None
        geo_depth_cache_dir = getattr(cfg, 'custom_geo_depth_cache_dir', None)
        say(f"[geo custom] input_hw={geo_hw} grid="
            f"{geo_hw[0] // _p}x{geo_hw[1] // _p} "
            f"depth_cache={geo_depth_cache_dir or 'OFF (npz fallback, ~0.6s/item)'}")

    # [new 2026-08-12] geo_encoder='da3' (models/da3_geo_encoder.py). custom 과 달리 dataset 이
    # 더 만들어야 할 텐서가 없다 — DA3 는 RGB 만 받고 카메라는 geo_posed 의 geo_c2w /
    # geo_fxfycxcy / geo_hw (이미 있는 키) 로 들어간다. 여기서 하는 일은 두 가지뿐:
    #   1) geo_hw 를 da3_geo_input_hw 로 덮어써서 _load_images 가 처음부터 그 격자로 주게 한다
    #      (인코더 안의 bilinear 재보간을 피한다).
    #   2) 조용히 틀리는 조합을 raise / disable 한다.
    geo_da3 = geo_enabled and str(cfg.geo_encoder) == 'da3'
    if geo_da3:
        _p = 14                                   # DepthAnything3Net.PATCH_SIZE (고정)
        _ihw = getattr(cfg, 'da3_geo_input_hw', None)
        geo_hw = (tuple(int(v) for v in _ihw) if _ihw else
                  ((geo_hw[0] // _p) * _p, (geo_hw[1] // _p) * _p))
        if any(v % _p for v in geo_hw):
            raise ValueError(f"da3_geo_input_hw={geo_hw} must be a multiple of patch {_p}")
        if getattr(cfg, 'geo_lagernvs_skip_ctx_norm', False):
            # lagernvs 의 1.35*max(context) 정규화를 target scale 로 덮어쓰는 플래그다.
            # DA3 는 자기 규약(view0 재고정 + camera center 거리 median)으로 정규화하므로
            # override_scale 을 받을 자리가 없다 (_DA3Backend.build_cam_token 이 raise).
            raise ValueError("geo_lagernvs_skip_ctx_norm=True 는 geo_encoder='da3' 와 같이 "
                             "쓸 수 없다 (lagernvs 전용 정규화)")
        if getattr(cfg, 'geo_shuffle_order', False):
            # DA3 backbone 은 view0 를 reference 로 삼는다 (cam_token 을 주면 ref-view 자동
            # 선택이 꺼지고 입력 순서가 그대로 쓰인다). 순서를 섞으면 기준 프레임이 매번 바뀐다.
            raise ValueError("geo_shuffle_order=True 는 geo_encoder='da3' 와 같이 쓸 수 없다 "
                             "(DA3 는 view0 를 reference 로 쓴다)")
        if getattr(cfg, 'geo_posed', False) and pose_source != 'da3':
            # 원래 이유: "cam_enc 에 넣는 pose 의 스케일 공간이 depth/pose 코퍼스와 달라진다."
            # [2026-08-16] 이 가드는 custom backend 의 근거를 그대로 옮겨 온 것이라 da3 backend
            # 에는 과하다. da3 는 (a) 디스크 depth 를 안 읽고 (geo_logd/geo_valid 는 geo_custom
            # 분기 전용, dataset_dl3dv.py:1417-1423), (b) cam_enc 입력을 view0 재고정 +
            # **per-sample median camera distance** 로 정규화하므로 (build_cam_token) 포즈
            # 코퍼스의 절대 스케일에 불변이다 -> transforms(COLMAP) pose 로도 성립한다.
            # 그래도 기본은 raise 로 둔다: da3 코퍼스 arm 에서 pose_source 를 빠뜨리는 오타를
            # 잡아 주는 값이 크다. 의도적으로 섞는 arm 만 플래그로 opt-in 한다.
            if not getattr(cfg, 'da3_geo_allow_transforms_pose', False):
                raise ValueError("geo_encoder='da3' + geo_posed=True 는 pose_source='da3' 가 "
                                 f"필요하다 (got {pose_source!r}). 의도한 것이면 "
                                 "da3_geo_allow_transforms_pose=true 로 명시할 것")
            say(f"[geo da3] pose_source={pose_source!r} 로 cam_enc 를 태운다 "
                f"(da3_geo_allow_transforms_pose=true). cam_token 은 per-sample median camera "
                f"distance 로 정규화되므로 포즈 코퍼스의 절대 스케일에 불변이다")
        if geo_latent_cache_dir is not None:      # [cascade 2b/5]
            # LayerNorm/proj 가 **학습되는** 파라미터라 그 출력을 파일로 얼리면 안 된다.
            say("[geo cache] DISABLED: geo_encoder='da3' -> the geo encoder is trainable")
            geo_latent_cache_dir = None
        _samp = str(getattr(cfg, 'geo_view_sampling', 'even'))
        _v = (int(getattr(cfg, 'geo_cover_k', 6)) if _samp == 'frustum_cover' else
              (int(getattr(cfg, 'geo_num_inseg', 3)) + int(getattr(cfg, 'geo_num_covis', 3))
               if _samp == 'hybrid' else int(getattr(cfg, 'geo_num_views', 4))))
        if not getattr(cfg, 'geo_posed', False) and _v < 3:
            # cam_token 이 없으면 backbone 이 THRESH_FOR_REF_SELECTION(=3) 이상일 때만
            # reference view 를 고른다. V<3 이면 그 경로가 통째로 빠져 동작이 달라진다.
            raise ValueError(f"geo_encoder='da3' + geo_posed=False 는 geo_num_views>=3 이어야 "
                             f"한다 (got {_v}; DA3 의 reference-view 선택 임계값)")
        _front_pair = ('front_first_anchor_same_len', 'front_centroid_same_len', 'da3latent')
        if _samp == 'front_uniform' and avg_scale_ref not in _front_pair:
            # front_uniform 의 존재 이유가 "분모를 만든 range 와 context view range 를 일치"인데
            # 분모가 다른 range 에서 온 값이면 그 일치가 깨진다. 조용히 어긋나면 안 되는 자리.
            say(f"[geo da3] WARNING: geo_view_sampling='front_uniform' 인데 "
                f"avg_scale_ref={avg_scale_ref!r} 다 -> context view range([s-L,s)) 와 분모를 "
                f"만든 range 가 다르다 (의도한 것이 아니면 {_front_pair} 중 하나로 맞출 것)")
        if _samp == 'context_uniform' and str(avg_scale_ref).startswith('front_'):
            # context_uniform 의 range 는 영상 전체 [0,N) 인데 front_* 분모는 [s-L,s) 에서
            # 만든 값이다. front_uniform guard 와 같은 종류의 어긋남 — 조용히 지나가면
            # context view 가 보는 스케일과 나누는 분모가 다른 구간에서 온다.
            say(f"[geo da3] WARNING: geo_view_sampling='context_uniform' 인데 "
                f"avg_scale_ref={avg_scale_ref!r} 다 -> context view range([0,N)) 와 분모를 "
                f"만든 range([s-L,s)) 가 다르다")
        # [2026-08-13] 앵커 정합. DA3 는 cam_token 을 **view0 기준**으로 재고정한다
        # (da3_geo_encoder.build_cam_token: w2c @ c2w[:, :1], DA3 api.py:439-440 과 동일).
        # target 은 rel = E @ inv(E_s) 로 **프레임 s 기준**이다. 그래서 view0 != s 면 두 트랙의
        # R,t 앵커가 서로 다른 카메라가 되고, latent 이 표현하는 궤적과 모델이 맞춰야 하는 궤적이
        # 강체변환만큼 어긋난 채 학습된다 (스케일 분모로는 못 고치는 축).
        if getattr(cfg, 'geo_posed', False) and not getattr(cfg, 'geo_first_view_target_s', False):
            say("[geo da3] WARNING: geo_posed=True 인데 geo_first_view_target_s=False 다 -> "
                "DA3 의 view0(=cam_token 재고정 기준)가 target 앵커 프레임 s 가 아니라서 geo "
                "latent 과 target rel 의 R,t 앵커가 서로 다른 카메라다")
        if avg_scale_ref == 'da3latent_cover' and _samp != 'frustum_cover':
            # da3latent 와 대칭인 guard. 이 값은 frustum_cover 가 고른 view 집합의 cam_dec
            # 출력으로 만들었다 -> sampler 가 바뀌면 그냥 틀린 분모다.
            raise ValueError(
                f"avg_scale_ref='da3latent_cover' 는 geo_view_sampling='frustum_cover' 에서만 "
                f"유효하다 (got {_samp!r}). front_uniform 짝은 avg_scale_ref='da3latent' 다 — "
                f"scripts/data/make_avg_scale_da3_latent.py --out-dir 참고")
        if avg_scale_ref in ('da3latent', 'da3latent_cover') \
                and not getattr(cfg, 'geo_first_view_target_s', False):
            # 저장된 M*sigma 는 view0 = s 인 view 집합으로 cam_dec 를 돌려 만든 값이다
            # (scripts/data/make_avg_scale_da3_latent.py). view0 가 바뀌면 M 도 sigma 도 바뀐다.
            raise ValueError(
                f"avg_scale_ref={avg_scale_ref!r} 는 geo_first_view_target_s=True 에서만 유효하다 "
                "(분모를 view0 = 프레임 s 인 view 집합으로 만들었다). "
                "scripts/data/make_avg_scale_da3_latent.py 참고")
        if avg_scale_ref == 'da3latent' and _samp != 'front_uniform':
            # da3latent 값은 front_uniform 이 고른 그 view 집합의 cam_dec 출력으로 만들어졌다.
            # view 가 달라지면 latent 이 보는 스케일도 달라져 분모가 그냥 틀린 값이 된다.
            raise ValueError(
                f"avg_scale_ref='da3latent' 는 geo_view_sampling='front_uniform' 에서만 유효하다 "
                f"(got {_samp!r}). 이 분모는 그 view 집합으로 돌린 cam_dec 출력에서 나온 값이라 "
                f"view 가 바뀌면 의미가 없다 — scripts/data/make_avg_scale_da3_latent.py 참고")
        say(f"[geo da3] model={getattr(cfg, 'da3_geo_model', 'da3nested-giant-large')} "
            f"input_hw={geo_hw} grid={geo_hw[0] // _p}x{geo_hw[1] // _p} "
            f"posed={bool(getattr(cfg, 'geo_posed', False))} "
            f"view_sampling={_samp} V={_v} avg_scale_ref={avg_scale_ref} "
            f"view0={'frame s' if getattr(cfg, 'geo_first_view_target_s', False) else 'context'}")

    geo_test_inseg_k = getattr(cfg, 'geo_test_inseg_k', None) or 0
    if geo_test_inseg_k and geo_latent_cache_dir is not None:      # [cascade 3/4]
        # 캐시 키는 data_name 뿐이라 context view 가 바뀐 걸 구분 못 한다 -> 반드시 끈다.
        say(f"[geo cache] DISABLED: geo_test_inseg_k={geo_test_inseg_k} changes the "
            f"context views, but the cache is keyed by segment only")
        geo_latent_cache_dir = None

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
    geo_swap_mode = getattr(cfg, 'geo_swap_mode', None)
    if geo_swap_mode not in (None, 'inscene'):
        raise ValueError(f"geo_swap_mode must be null | 'inscene', got {geo_swap_mode!r}")
    if geo_swap_mode and geo_latent_cache_dir is not None:         # [cascade 4/4]
        # geo_test_inseg_k 와 같은 이유 -- 캐시 키가 data_name 뿐이라 context view 가 바뀐 것을
        # 구분하지 못한다. 끄지 않으면 swap 이 조용히 무효가 된다.
        say(f"[geo cache] DISABLED: geo_swap_mode={geo_swap_mode} changes the "
            f"context views, but the cache is keyed by segment only")
        geo_latent_cache_dir = None

    # [new 2026-08-29] scene 키 pre-ln DA3 캐시. 위 cascade 와 **독립된** 캐시다 (자르는 지점이
    # ln 직전이라 학습되는 ln/proj 를 얼리지 않는다) — 그래서 da3 에서 유일하게 쓸 수 있다.
    # 여기서 다 막는 이유: 캐시 키가 **scene** 이므로 "context view 선택이 scene 만 보고 정해진다"
    # 가 깨지는 순간 조용히 틀린 토큰을 먹인다. 한 조건이라도 어긋나면 켜지 않는다.
    geo_raw_cache_dir = getattr(cfg, 'geo_raw_cache_dir', None) or None
    if geo_raw_cache_dir and geo_enabled:
        _why = []
        if str(getattr(cfg, 'geo_encoder', None)) != 'da3':
            _why.append(f"geo_encoder={getattr(cfg, 'geo_encoder', None)!r} != 'da3' "
                        f"(encode_raw/from_raw 는 da3 backend 에만 있다)")
        if str(getattr(cfg, 'geo_view_sampling', 'even')) != 'context_uniform':
            # 나머지 sampler 는 (s,e) 나 retrieval 을 보므로 변이마다 view 가 달라진다.
            _why.append(f"geo_view_sampling={getattr(cfg, 'geo_view_sampling', 'even')!r} "
                        f"!= 'context_uniform' -> context 가 scene 상수가 아니다")
        if getattr(cfg, 'geo_cam_embed', None) is not None:
            # geo_cam_param 은 target frame0 기준이라 변이마다 다르다. 캐시 경로는 안 만든다.
            _why.append(f"geo_cam_embed={getattr(cfg, 'geo_cam_embed')!r} (변이별 값)")
        if getattr(cfg, 'geo_shuffle_order', False):
            _why.append("geo_shuffle_order=True -> view 순서가 매 스텝 달라진다")
        if geo_swap_mode:
            _why.append(f"geo_swap_mode={geo_swap_mode!r} -> context 가 다른 segment 로 바뀐다")
        if geo_test_inseg_k:
            _why.append(f"geo_test_inseg_k={geo_test_inseg_k} -> context view 가 바뀐다")
        if geo_latent_cache_dir is not None:
            # 데이터셋에서 geo_latent_cache_dir 분기가 먼저 return 한다 -> 이쪽이 죽은 코드가 된다.
            _why.append("geo_latent_cache_dir 과 동시에 켤 수 없다")
        if _why:
            say("[geo raw cache] DISABLED: " + " | ".join(_why))
            geo_raw_cache_dir = None
        else:
            if not getattr(cfg, 'da3_cam_token_per_sample', False):
                # 캐시는 cache_geo_raw_da3.py 가 B=1 로 굽는다. on-the-fly 폴백(캐시 miss)은
                # batch_size 통째로 cam_enc 를 부르는데, cuBLAS kernel 이 바뀌어 cam_token 이
                # rel 2.9e-7 달라지고 backbone 이 그걸 pre-ln 토큰에서 rel 2.3e-2 로 증폭한다
                # (da3_geo_encoder.build_cam_token 주석 참조). 두 경로를 섞으면 같은 배치 안에
                # 서로 다른 게이지의 토큰이 들어간다.
                say("[geo raw cache] WARNING: da3_cam_token_per_sample=false -> 캐시(B=1)와 "
                    "on-the-fly 폴백(B=batch_size)이 rel 2.3e-2 갈린다. true 로 켤 것")
            say(f"[geo raw cache] reading {geo_raw_cache_dir}/<scene_key>.pt "
                f"(pre-ln (V,P,C) fp32; miss -> on-the-fly DA3). ln/proj 는 그대로 학습된다")
    else:
        geo_raw_cache_dir = None

    # honest selection: anchor at the target's FIRST frame only (known at inference);
    # radius scaled by CONTEXT (not the unseen rest of the target segment).
    # [renamed 2026-07-31] geo_anchor_first_frame -> geo_cover_centered_at_s. The old name read
    # as "the first frame goes in as an anchor VIEW", which is what geo_first_view_target_s
    # does; this flag only centers the frustum_cover search ball on frame s and never adds a
    # view. Old key still honored (deprecated) so pre-rename configs / CLI overrides work.
    _legacy = getattr(cfg, 'geo_anchor_first_frame', None)
    _cur = getattr(cfg, 'geo_cover_centered_at_s', None)
    if _cur is None and _legacy is not None:
        say("[cfg] geo_anchor_first_frame is deprecated -> using it as geo_cover_centered_at_s"
            f"={_legacy}")
    geo_cover_centered_at_s = bool(_cur if _cur is not None
                                   else (_legacy if _legacy is not None else False))

    return DatasetSpec(
        pose_source=pose_source,
        avg_scale_ref=avg_scale_ref,
        target_pose_source=target_pose_source,
        root=cfg.dl3dv_root,
        num_frames=cfg.num_frames,
        max_scenes=getattr(cfg, 'max_scenes', None),      # limit #scenes (e.g. smoke test)
        # lazy_dataset (default True): __init__ only builds the sample/scene index (from a
        # persisted cache when available); scene poses/paths are parsed on demand in
        # __getitem__ + cached. only_segments 도 lazy 를 강제하므로 최종 판정은 __init__ 에서.
        lazy_dataset=bool(getattr(cfg, 'lazy_dataset', True)),
        geo_enabled=geo_enabled,
        geo_num_views=getattr(cfg, 'geo_num_views', 4),
        geo_hw=geo_hw,
        geo_cam_embed=geo_cam_embed,
        geo_posed=getattr(cfg, 'geo_posed', False),
        # [new] attach the chosen context view indices (+ their c2w) to every item so an offline
        # script can measure how close the generated trajectory sits to the context cameras.
        # Off by default: on a cache HIT it costs the frustum_cover search the cache exists to skip.
        geo_return_idxs=bool(getattr(cfg, 'geo_return_idxs', False)),
        geo_latent_cache_dir=geo_latent_cache_dir,
        geo_raw_cache_dir=geo_raw_cache_dir,
        geo_raw_cache_preload=bool(getattr(cfg, 'geo_raw_cache_preload', True)),
        geo_custom=geo_custom,
        geo_custom_channels=geo_custom_channels,
        geo_depth_cache_dir=geo_depth_cache_dir,
        # geo context-view sampling (leakage ablation): 'even' (in-segment) | 'frustum_cover'
        # | 'hybrid' | 'random_inseg' | 'front_uniform' ([s-L,s)) | 'context_uniform' ([0,N))
        geo_view_sampling=getattr(cfg, 'geo_view_sampling', 'even'),
        geo_shuffle_order=getattr(cfg, 'geo_shuffle_order', False),
        # geo context view0 = target segment's FIRST camera s (rest = out-of-seg retrieved)
        # -> LagerNVS anchors to s, aligning the geo latent frame with the target frame.
        geo_first_view_target_s=getattr(cfg, 'geo_first_view_target_s', False),
        geo_cover_k=getattr(cfg, 'geo_cover_k', 6),
        geo_cover_radius=getattr(cfg, 'geo_cover_radius', 2.0),
        geo_cover_ndepth=getattr(cfg, 'geo_cover_ndepth', 3),
        geo_cover_out_of_seg=getattr(cfg, 'geo_cover_out_of_seg', False),
        # restrict out-of-segment context to frames BEFORE the target segment (index < s)
        # only, instead of the longer side. Requires the target segment to have frames
        # before it -> the target segment can be the 2nd segment onward.
        geo_cover_before_only=getattr(cfg, 'geo_cover_before_only', False),
        geo_cover_centered_at_s=geo_cover_centered_at_s,
        geo_num_inseg=getattr(cfg, 'geo_num_inseg', 3),
        geo_num_covis=getattr(cfg, 'geo_num_covis', 3),
        geo_inseg_span=getattr(cfg, 'geo_inseg_span', None),
        covis_radius=getattr(cfg, 'geo_covis_radius', 1.0),
        covis_theta0=getattr(cfg, 'geo_covis_theta0', 10.0),
        covis_max_axis_deg=getattr(cfg, 'geo_covis_max_axis_deg', 60.0),
        # geo_covis_topM 은 읽지 않는다: config.py:219 가 이미 "unused" 라고 적어 뒀고
        # self.covis_topM 을 참조하는 코드가 없다 (scripts/context_select/*.py 는 자기 지역
        # 기본값 32 를 쓴다).
        geo_test_inseg_k=geo_test_inseg_k,
        geo_swap_mode=geo_swap_mode,
        geo_swap_shift=int(getattr(cfg, 'geo_swap_shift', 1) or 1),
        geo_swap_keep_first=bool(getattr(cfg, 'geo_swap_keep_first', True)),
    )
