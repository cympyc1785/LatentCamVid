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
    avg_scale_ref = str(getattr(cfg, 'avg_scale_ref', 'centroid') or 'centroid')
    if avg_scale_ref not in ('centroid', 'context_first_cam'):
        raise ValueError(f"avg_scale_ref must be 'centroid' or 'context_first_cam', "
                         f"got {avg_scale_ref!r}")
    if avg_scale_ref != 'centroid' and pose_source != 'da3':
        raise ValueError(f"avg_scale_ref={avg_scale_ref!r} 는 pose_source='da3' 에서만 "
                         f"쓸 수 있다 (해당 디렉토리가 da3 아래에만 있다). "
                         f"현재 pose_source={pose_source!r}")

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
        geo_custom=geo_custom,
        geo_custom_channels=geo_custom_channels,
        geo_depth_cache_dir=geo_depth_cache_dir,
        # geo context-view sampling (leakage ablation): 'even' (in-segment) | 'frustum_cover'
        # | 'hybrid' | 'random_inseg'
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
