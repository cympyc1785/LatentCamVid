"""씬의 길이 단위 `S` 와 시차 게이지. **이 파일이 S 의 유일한 정의처다.**

왜 별도 모듈인가: `S` 는 리포 전체에서 avg_scale / S_da3 / norm_scale 이라는 서로 다른 이름으로
불리는 같은 양이고, 정의가 한 군데라도 어긋나면 OBB extent · 후보 거리 · τ 임계가 전부 조용히
어긋난다. `lbm/cloud.py` 도 여기서 import 해 쓴다.

DA3 depth 는 metric 이 아니다. 그래서 "0.5 미터 앞"이라는 말을 할 수 없고, 대신 씬 자체의
평균 관측 거리를 1 로 놓는 무차원 단위 `u` 를 쓴다 (1 u ≜ S DA3 units).

2026-09-02 부터 기본 정의가 **전 프레임 non-sky 점 → 첫 소스 카메라 거리 평균** 이다 (사용자
지시). 예전 frame0-한장 정의는 `mode="frame0_ray"` 로 남아 있다. 정의를 바꾸면 뱅크 정체성이
바뀌므로 옛 뱅크와 대조할 때는 반드시 그래프의 `scale.mode` 를 확인할 것.
"""
import numpy as np


SCALE_MODES = ("points_first_cam", "frame0_ray")


def scene_scale(depths: np.ndarray, K: np.ndarray, sky_mask: np.ndarray | None = None,
                cam_c2w: np.ndarray | None = None, mode: str = "points_first_cam",
                stride: int = 1):
    """S = **전 프레임** non-sky 점을 world 로 올려 **첫 소스 카메라까지 거리의 평균**.

    (사용자 지시 2026-09-02: "그냥 scene scale 은 sky 제외 mean of valid points distance from
    first source camera 로 정의하고 사용해줘".)

    왜 바꿨나: 예전 정의(`mode="frame0_ray"`)는 **frame 0 한 장**의 평균 ray length 였다. 카메라가
    돌아서 다른 공간을 보면 그 뒤 프레임의 실제 관측 거리가 몇 배가 되는데 S 는 안 따라간다 —
    그러면 `u` 로 표현된 모든 것(OBB extent · 후보 거리 · 게이트 마진 `0.02·S` · τ 임계)이 그
    프레임들에서만 조용히 어긋난다. 72편 실측(`eval/audit_scene_scale.py`)에서 프레임별
    평균 ray length 의 max/min 비가 p50 1.0961 · 12.5% 가 1.5 초과 · 최대 12.4030 이었다.

    dynamic 은 **안 뺀다** — 사용자가 sky 만 제외라고 지정했다. 동적 표면도 그 시각에 카메라가
    실제로 본 거리이고, 빼면 사람이 크게 잡힌 클립에서 분모가 갑자기 배경 거리로 튄다.

    `trumans_to_recon.py:avg_scale_first_cam` 의 `avg_scale` 과 **같은 정의**다 (그쪽은 stride 2,
    여기 기본은 stride 1). 이름만 avg_scale / S_da3 / norm_scale 로 갈릴 뿐 같은 게이지다.

    `mode="frame0_ray"` 는 예전 정의 그대로다 — 옛 뱅크를 재현하거나 대조할 때 쓴다. frame 0
    한 장만 보면 두 식이 같은 값이 되므로(카메라 원점이 자기 자신), 새 정의는 그 일반화다.
    """
    assert mode in SCALE_MODES, f"mode 는 {SCALE_MODES} 중 하나: {mode}"
    K = np.asarray(K, np.float64)
    height, width = depths.shape[-2:]

    if mode == "frame0_ray":
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        pixels = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], axis=-1).astype(np.float64)
        rays = pixels @ np.linalg.inv(K[0]).T                     # h w 3
        ray_len = np.linalg.norm(rays, axis=-1)
        z = depths[0].astype(np.float64)
        valid = np.isfinite(z) & (z > 0)
        if sky_mask is not None:
            valid &= ~sky_mask[0]
        assert valid.any(), "frame0 에 유효 depth 픽셀이 없다"
        return float((z[valid] * ray_len[valid]).mean())

    assert cam_c2w is not None, "mode='points_first_cam' 은 cam_c2w 가 필요하다"
    cam_c2w = np.asarray(cam_c2w, np.float64)
    origin = cam_c2w[0][:3, 3]
    v, u = np.meshgrid(np.arange(0, height, stride), np.arange(0, width, stride), indexing="ij")
    pixels = np.stack([u + 0.5, v + 0.5, np.ones_like(u)], axis=-1).astype(np.float64)
    # 점을 다 쌓지 않고 합/개수만 누적한다 — 49프레임 full-res 면 1천만 점이라 메모리가 아깝고,
    # 평균은 어차피 결합법칙이 성립해서 결과가 비트 단위로 같다.
    total, count = 0.0, 0
    for t in range(len(depths)):
        K_t = K[t] if K.ndim == 3 else K
        rays = pixels @ np.linalg.inv(K_t).T                      # h w 3, z 성분이 1
        z = np.asarray(depths[t], np.float64)[::stride, ::stride]
        valid = np.isfinite(z) & (z > 0)
        if sky_mask is not None:
            valid &= ~sky_mask[t][::stride, ::stride]
        if not valid.any():
            continue
        c2w = cam_c2w[t]
        points = (z[..., None] * rays)[valid] @ c2w[:3, :3].T + c2w[:3, 3]
        total += float(np.linalg.norm(points - origin, axis=1).sum())
        count += int(valid.sum())
    assert count > 0, "유효 depth 픽셀이 하나도 없다 (전부 sky?)"
    return total / count


def assert_scale_mode(graph: dict, allow_legacy: bool = False, expect: str = "points_first_cam"):
    """그래프의 `S` 가 어느 정의로 구워졌는지 확인한다. 뱅크를 굽기 **전에** 부를 것. -> mode

    WHY (F2). 게이트 임계는 전부 `S` 배율이다 — `behind_margin_frac·S`, `behind_clear_frac·S`,
    `obb_clear_floor·S`, `min_ground_clear·S`, 가림 판정의 `0.02·S` (§`lbm/gates.py`). 즉 `S` 의
    정의를 바꾸는 건 게이트 임계를 통째로 옮기는 것과 같은데, 그래프 파일에는 숫자만 남고
    정의는 안 남았었다. 2026-09-02 에 기본 정의가 `frame0_ray` → `points_first_cam` 으로 바뀌었고
    72편 실측 배율(`audit_scene_scale.py` 의 `r_ship`)이 씬마다 다르므로, 옛 그래프로 새 뱅크를
    구우면 **씬마다 다른 배율로 임계가 어긋난 채** 조용히 통과한다. 그래서 여기서 막는다.

    `allow_legacy=True` (CLI `--allow_legacy_scale`) 는 옛 뱅크를 일부러 되만들 때의 탈출구다.
    """
    mode = str(graph.get("scale", {}).get("mode", "frame0_ray"))
    if allow_legacy:
        return mode
    assert mode == expect, (
        f"scene_graph 의 scale.mode 가 '{mode}' 다 (기대: '{expect}'). 게이트 임계가 전부 S 배율이라 "
        f"정의가 다르면 씬마다 다른 배율로 어긋난다 — 그래프를 다시 굽거나 `--allow_legacy_scale` "
        f"로 명시할 것.")
    return mode


def z_median(depths: np.ndarray, sky_mask: np.ndarray | None = None, frame: int = 0):
    """frame 의 non-sky z-depth 중앙값. τ 의 분모이자 parallax_ratio 의 분모."""
    z = depths[frame].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if sky_mask is not None:
        valid &= ~sky_mask[frame]
    return float(np.median(z[valid]))


TAU_DENOM_MODES = ("S", "z_med_frame0")


def tau_denominator(graph: dict, mode: str = "S"):
    """τ = |Δp| / **이 값**. 그래프에서 분모를 꺼내는 유일한 창구. (D150)

    사용자 지시 2026-09-06: "z_med 는 이전에 전체 프레임에서 sky 제외 유효한 depth 의 첫
    카메라로부터의 거리의 평균으로 하기로 했잖아. 적용해줘." — 즉 τ 의 분모를 `S`
    (`scene_scale(mode="points_first_cam")`) 로 바꾼다. `mode="z_med_frame0"` 은 예전 분모
    (frame 0 non-sky z-depth 중앙값, `z_median()`) 그대로다.

    왜 바꾸나: 두 양이 재는 게 다르다. `z_med_frame0` 은 **frame 0 한 장의 z 중앙값**이라
    ① 카메라가 돌아 다른 공간을 보면 안 따라가고 ② z-depth 라 화면 가장자리 ray 길이를
    빼먹고 ③ 중앙값이라 씬 깊이 분포의 꼬리를 못 본다. `S` 는 이미 OBB extent · 후보 거리 ·
    게이트 마진(`0.02·S`)의 게이지이므로, τ 만 다른 게이지를 쓰면 "τ 0.35" 가 씬마다 다른
    비율의 이동을 뜻하게 된다.

    바뀌는 배율은 씬마다 다르다 (전 그래프 실측, `S / z_med_frame0`):

      corpus    n   min    p10    p50    p90     max
      vista    53  0.695  0.951  1.426  3.940  29.854
      dynpose 280  0.610  0.899  1.171  2.404  14.577
      trumans 191  0.771  0.966  1.106  1.280   1.694

    분모가 커지면 같은 이동의 τ 가 **작아진다** — 즉 같은 τ 임계가 더 큰 이동을 허용한다.
    뱅크 경로에서 τ 는 게이트가 아니라 **푸는 변수**라 이 배율이 결과를 그만큼 옮기지는
    않는다 (hole·OBB·approach 가 binding). 직접 움직이는 것은 `knob_range.tau` 클램프와
    `tau_floor_src` 하한, 그리고 board 경로의 `max_tau` 다.

    **뱅크 정체성이 바뀐다.** 그래서 `bank.json` 의 `fixed.tau_denom` 에 모드를 남긴다 —
    옛 뱅크(`hole_bank_k6_d128` / `hole_bank_d129`)는 이 키가 없고 그게 곧 `z_med_frame0` 이다.
    """
    assert mode in TAU_DENOM_MODES, f"mode 는 {TAU_DENOM_MODES} 중 하나: {mode}"
    scale = graph["scale"]
    value = float(scale["S"]) if mode == "S" else float(scale["z_med_frame0"])
    assert value > 0, f"τ 분모가 양수가 아니다 ({mode}={value})"
    return value


def parallax_ratio(cam_c2w: np.ndarray, z_med: float):
    """max_ij‖C_i − C_j‖ / z_med. 0.02 근처면 single-view depth shell 이라 보면 된다."""
    centers = cam_c2w[:, :3, 3]
    spread = float(np.linalg.norm(centers[:, None] - centers[None, :], axis=-1).max())
    return spread / max(z_med, 1e-9)
