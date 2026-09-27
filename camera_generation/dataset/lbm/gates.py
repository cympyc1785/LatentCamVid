"""**공간 제약을 집행하는 곳.** 프롬프트가 아니라 여기서 막는다.

폐기된 CinemaTraj 계획은 "카메라가 벽 속인가 / 소스가 못 본 데를 보나"를 256³ SDF + Adam 손실로
사후에 고치려 했다. 시차 median 0.024 인 single-view depth shell 위에서 그 SDF 는 71% 가
`unknown` 이라 애초에 못 믿을 제약이었다. 여기서는 같은 질문을 **측정**으로 바꾼다:

  G1 behind-surface  후보 위치를 소스 프레임에 되쏘아 "표면 뒤"인지 본다. 우리가 가진 유일한
                     기하 증거가 관측된 표면이고, 그 증거를 직접 쓴다
  G2 coverage        렌더해서 valid 픽셀 비율을 잰다. 하류 video model 이 채워야 할 hole 그 자체
  G3 framing         subject 가 화면 안에, 적당한 크기로, 가려지지 않고 있나
  G4 tau/view-angle  위 셋보다 1000배 싼 사전 필터. 렌더 수를 깎는 용도지 판정용이 아니다
  G5 obb-clearance   카메라 중심이 **scene graph 노드의 OBB 안**인가 (D49). G1/standoff 는 둘 다
                     "관측된 표면"을 재는데, 표면은 물체의 **껍질**이라 물체 내부를 지나가도
                     증거가 얇다 — 반대편 벽이 보이면 G1 이 통과한다. OBB 는 부피라서 그 구멍을
                     막는다. 게다가 "무엇과 부딪혔나"를 노드 이름으로 말해준다 (`fence`/`camel`)
  G6 elevation       subject 대비 고도각이 물체 **바로 위/아래**인가, 카메라가 **지면 아래**인가.
                     `elevation_profile()`. 위 다섯 개가 전부 통과시키는 구멍이라 따로 있다
                     (그 함수 docstring 에 실측). 사다리(`fit_hole_ladder.py`)만 쓰고
                     `evaluate()` 의 `GATE_ORDER` 에는 없다 — 후보 풀은 elevation 을 애초에
                     `{12,28,45}°` 로 열거하므로 이 구멍이 안 생긴다. 생기는 건 **궤적** 쪽이다

**전 후보의 결과를 남긴다** (`trace/gates.csv`). 전부 탈락하면 그게 진단이다 — subject 가 화면
가장자리라 어느 방향에서도 못 본다는 뜻이고, 그때 폴백(소스 시선 ±20°, r=d_ref)으로 9개만
재생성한다.

G3 의 z-buffer 통과율은 **subject 단독 렌더를 분모로** 쓴다. subject 픽셀 수를 subject 점
개수로 나누면 거리에 따라 값이 통째로 움직여서 임계를 못 정한다.
"""
import numpy as np

GATE_ORDER = ("G4_tau", "G4_view_angle", "G1_behind", "G5_obb", "G2_coverage",
              "G3_center", "G3_area", "G3_occlusion")


def tau_and_view_angle(p_world, target_world, src_center, z_med: float):
    """(tau, view_angle_deg). tau 는 `ctx_align.py` 가 쓰는 그 양 — |Δp| / z_med."""
    p_world, target_world = np.asarray(p_world, float), np.asarray(target_world, float)
    src_center = np.asarray(src_center, float)
    tau = float(np.linalg.norm(p_world - src_center) / max(z_med, 1e-9))
    v_plan = target_world - p_world
    v_src = target_world - src_center
    cos = float(np.dot(v_plan, v_src) / max(np.linalg.norm(v_plan) * np.linalg.norm(v_src), 1e-9))
    return tau, float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def behind_surface_frames(p_world, depths, K, cam_c2w, sky_mask, scale: float,
                          frames, margin_frac: float = 0.02,
                          clear_frac: float = 0.0, radius_px: int = 0,
                          dynamic_mask=None, plan_frame=None, channel: str = "all",
                          min_zcam_frac: float = 0.0):
    """후보 위치가 "관측된 표면 뒤"(또는 표면에 너무 붙었는지)인 소스 프레임 인덱스들.

    `z_cam(p) + clear_frac·S > depth_t(u,v) + margin_frac·S` 면 그 프레임 카메라에서 봤을 때 p 가
    표면 **너머**거나 표면에서 `clear_frac−margin_frac` 배 S 만큼도 안 떨어져 있다. 하늘 픽셀과
    무효 depth 는 판정하지 않는다 (하늘 뒤는 벽 뒤가 아니다).

    두 손잡이의 역할이 반대다: `margin_frac` 은 관통을 **봐주는** 여유(depth 잡음용, 기본
    0.02·S), `clear_frac` 은 표면 앞에 **요구하는** 여유. 기본 `clear_frac=0` 이면 예전 판정 그대로다.

    `radius_px > 0` 이면 투영 픽셀 주변 (2r+1)² 패치의 **최소 depth** 로 판정한다. 점 하나만 보면
    얇은 물체 가장자리를 스칠 때 바로 옆 배경 픽셀의 먼 depth 가 잡혀서 안 걸린다.

    **채널** (`channel`, 2026-09-02). 정적 표면은 시간 불변이라 어느 프레임의 관측이든 유효한
    증거지만, **동적 표면은 그 시각에만 거기 있었다.** 전 프레임을 동등하게 쓰면 궤적 48프레임의
    카메라가 frame 0 의 사람 위치에 막힌다 — 그 사람은 이미 걸어가 버렸는데도. 같은 리포의
    `render.standoff` 는 이미 `temporal_persistence=False` 로 그 프레임 점만 장애물로 쓰고
    있어서, 두 게이트가 서로 반대 규약이었다 (`lbm/render.py:105-108`).

    - `"all"` (기본) — 예전 판정 그대로. 동적/정적을 안 가린다.
    - `"static"` — **모든** 샘플 프레임을 쓰되 동적 픽셀만 증거에서 뺀다. `frames` 집합이
      예전과 같으므로 "동적을 뺀 효과"만 순수하게 남는다.
    - `"dynamic"` — **`plan_frame` 한 장**에서 **동적 픽셀만** 본다. 플랜 프레임 f 의 카메라를
      같은 시각의 소스 관측에만 되쏘는 것이라, 동적 충돌을 끄는 게 아니라 시각을 맞춰 **따로**
      재는 것이다 (사용자 지시 2026-09-02: "dynamic 을 완전 빼지는 말고 … 따로 측정").

    **`min_zcam_frac` — degenerate 투영 하한** (D123, 2026-09-04). 이 판정은 p 를 소스 카메라 t
    의 광축 앞 어딘가에 놓고 "그 방향으로 본 표면보다 먼가"를 묻는다. 그런데 p 가 카메라 t 의
    **광학 중심에 거의 겹치면** `uv = (K·cam)[:2]/cam[2]` 가 0 에 가까운 수로 나뉘어 투영 픽셀이
    화면을 임의로 튄다 — 어느 픽셀이 잡히는지가 기하가 아니라 반올림으로 정해진다. 게다가
    `cam[2] ≈ 0` 이면 판정식이 `clear > z_surf + margin` 으로 붕괴해서, **플랜 위치가 아니라 소스
    카메라 t 자기 주변**에 대한 진술이 된다 (p 가 어디 있든 같은 답).

    실측 (parkour `hole_bank_f7_on`, 664변이 x 49프레임 = 142,952 판정쌍): G1 히트 9,888 쌍의
    `z_cam` 이 min 0.0002 / p10 0.0041 / p50 0.0374 u. 하한 0.005·S 면 히트의 12.2%, 0.02·S 면
    31.9% 가 사라진다 — 씬 하나의 edge case 가 아니라 계통 오차다. 순수 lateral truck 은 자기
    시각의 소스 카메라 대비 `z_cam ≈ 0` 이라 정상 플랜도 여기 걸린다.

    하한 아래의 소스 프레임은 **증언에서 뺀다** (기각도 통과도 아니다) — 다른 프레임들이 여전히
    판정한다. 기본 0.0 이면 예전 `cam[2] <= 1e-6` 그대로라 비트 동일.
    """
    if channel not in ("all", "static", "dynamic"):
        raise ValueError(f"channel must be all|static|dynamic, got {channel!r}")
    if channel != "all" and dynamic_mask is None:
        raise ValueError(f"channel={channel!r} requires dynamic_mask")
    if channel == "dynamic":
        if plan_frame is None:
            raise ValueError("channel='dynamic' requires plan_frame")
        frames = [int(plan_frame)]
    p_world = np.asarray(p_world, float)
    height, width = depths.shape[-2:]
    margin, clear = margin_frac * scale, clear_frac * scale
    z_floor = max(float(min_zcam_frac) * scale, 1e-6)   # 기본 0.0 → 1e-6, 예전과 같은 분기
    hits = []
    for t in frames:
        w2c = np.linalg.inv(cam_c2w[t])
        cam = w2c[:3, :3] @ p_world + w2c[:3, 3]
        if cam[2] <= z_floor:
            continue
        uv = (K[t] @ cam)[:2] / cam[2]
        u, v = int(np.floor(uv[0])), int(np.floor(uv[1]))
        if not (0 <= u < width and 0 <= v < height):
            continue
        if radius_px <= 0:
            if sky_mask[t][v, u]:
                continue
            if channel == "static" and dynamic_mask[t][v, u]:
                continue
            if channel == "dynamic" and not dynamic_mask[t][v, u]:
                continue
            z = float(depths[t][v, u])
            if not (np.isfinite(z) and z > 0):
                continue
        else:
            u0, u1 = max(u - radius_px, 0), min(u + radius_px + 1, width)
            v0, v1 = max(v - radius_px, 0), min(v + radius_px + 1, height)
            patch = depths[t][v0:v1, u0:u1]
            usable = np.isfinite(patch) & (patch > 0) & ~sky_mask[t][v0:v1, u0:u1]
            if channel == "static":
                usable &= ~dynamic_mask[t][v0:v1, u0:u1]
            elif channel == "dynamic":
                usable &= dynamic_mask[t][v0:v1, u0:u1]
            if not usable.any():
                continue
            z = float(patch[usable].min())
        if cam[2] + clear > z + margin:
            hits.append(int(t))
    return hits


def behind_profile(poses, depths, K, cam_c2w, sky_mask, scale: float, frames,
                   margin_frac: float = 0.02, clear_frac: float = 0.0, radius_px: int = 0,
                   dynamic_mask=None, time_match: bool = False, detail: bool = False,
                   min_zcam_frac: float = 0.0):
    """궤적 **전 프레임**에 G1 을 건다. (위반 프레임 수, 최대 위반 소스 프레임 수).

    board 경로는 시작 pose 한 장만 본다. 뱅크 경로는 시작 pose 가 소스 카메라 자신이라 정의상
    표면 앞이고, 위험은 전부 **나머지 프레임**에 있다 — 그래서 궤적 전체를 훑는 판이 따로 필요하다.
    렌더가 안 들어서(재투영뿐) 이분법 안에서 후보마다 불러도 비용이 사실상 0 이다.

    `time_match=True` + `dynamic_mask` 면 판정을 **두 채널로 쪼개서 따로** 잰다
    (사용자 지시 2026-09-02: "dynamic 을 완전 빼지는 말고 … 해당 시간의 plan, src 카메라만
    매칭해서 이용해서 따로 측정"):

    - **static 채널** — `frames` 그대로(균등 7~49장), 동적 픽셀만 제외. 프레임 집합이 예전과
      같으므로 "동적을 뺐다"는 효과만 순수하게 남는다.
    - **dynamic 채널** — 플랜 프레임 f ↔ 소스 프레임 `round(f·(N_src−1)/(N_plan−1))` **한 장**,
      그 프레임의 동적 픽셀만. 플랜/소스 프레임 수가 달라도 되도록 비율로 매핑한다.
      매칭 프레임이 `frames` 에 없어도 쓴다 — 균등 7장으로 제한하면 나머지 42프레임의 동적
      물체가 G1 에 아예 안 보여서 "시간축 정합"이 "동적 충돌 끄기"가 되어 버린다.

    게이트가 보는 `bad` 는 **두 채널의 합집합**이라 동적 충돌을 여전히 잡는다. 어느 쪽이
    잡았는지는 `detail=True` 로 따로 받는다. `time_match=False`(기본)면 채널을 안 나누고
    예전 경로 그대로라 비트 단위로 같다.

    `detail=True` → `(bad, worst, {"static": {...}, "dynamic": {...}})`. 두 채널 dict 는
    `mask` (플랜 프레임별 bool 리스트) 도 들고 온다 — `--collision_source both` 가 depth 와
    mesh 의 위반 프레임을 **합집합**으로 세려면 개수가 아니라 어느 프레임인지가 필요하다
    (개수만 더하면 같은 프레임을 두 번 세고, max 를 쓰면 합집합을 과소보고해 게이트가 느슨해진다).

    `min_zcam_frac` 은 `behind_surface_frames` 로 그대로 내려간다 (D123). 기본 0.0 이면 비트 동일.
    """
    num_src = len(cam_c2w)
    num_plan = len(poses)
    base_frames = list(frames)
    split = bool(time_match and dynamic_mask is not None)
    bad, worst = 0, 0
    per_ch = {"static": {"frames": 0, "worst": 0, "mask": [False] * num_plan},
              "dynamic": {"frames": 0, "worst": 0, "mask": [False] * num_plan}}
    for f in range(num_plan):
        if not split:
            hits = behind_surface_frames(poses[f][:3, 3], depths, K=K, cam_c2w=cam_c2w,
                                         sky_mask=sky_mask, scale=scale, frames=base_frames,
                                         margin_frac=margin_frac, clear_frac=clear_frac,
                                         radius_px=radius_px, min_zcam_frac=min_zcam_frac)
            n_hit = len(hits)
        else:
            plan_frame = int(round(f * (num_src - 1) / max(num_plan - 1, 1)))
            channels = {"static": (base_frames, None), "dynamic": (None, plan_frame)}
            n_hit = 0
            for name, (frs, pf) in channels.items():
                hits = behind_surface_frames(
                    poses[f][:3, 3], depths, K=K, cam_c2w=cam_c2w, sky_mask=sky_mask,
                    scale=scale, frames=(frs if frs is not None else [pf]),
                    margin_frac=margin_frac, clear_frac=clear_frac, radius_px=radius_px,
                    dynamic_mask=dynamic_mask, plan_frame=pf, channel=name,
                    min_zcam_frac=min_zcam_frac)
                if hits:
                    per_ch[name]["frames"] += 1
                    per_ch[name]["worst"] = max(per_ch[name]["worst"], len(hits))
                    per_ch[name]["mask"][f] = True
                n_hit += len(hits)
        if n_hit:
            bad += 1
            worst = max(worst, n_hit)
    return (bad, worst, per_ch) if detail else (bad, worst)


def obb_signed_distance(p_g, center, extent, R):
    """점 `p_g` 에서 OBB 표면까지의 **부호 거리** (G 단위 = u). 음수면 박스 **안**.

    박스 로컬로 옮기고 `d = |q| − extent/2` 를 본다. 밖의 성분만 모아 놓은 것이 표면까지의
    유클리드 거리(`‖max(d,0)‖`)이고, 전부 음수면(=내부) 가장 덜 음수인 성분이 가장 가까운
    면까지의 깊이다. 둘을 더하면 안팎이 한 식으로 이어진다 — 이분법이 이 값을 단조로 밀 수
    있어야 하므로 "안이면 0" 같은 포화를 만들면 안 된다.
    """
    q = np.asarray(R, dtype=float).T @ (np.asarray(p_g, dtype=float) - np.asarray(center, float))
    d = np.abs(q) - np.asarray(extent, dtype=float) / 2.0
    return float(np.linalg.norm(np.maximum(d, 0.0)) + min(float(d.max()), 0.0))


def obb_gate_nodes(nodes: list, flat_ratio: float = 0.0, flat_min_extent: float = 0.0):
    """OBB 게이트(G5)·소스 OBB floor 가 볼 노드 — 바닥처럼 **넓고 납작한** 노드를 뺄지. (kept, skipped_ids)

    R62 (2026-09-27): golf 의 VLM 명사 "golf" 가 SAM3 로 코스 잔디 전체를 물어 `dyn_0` OBB 가
    2.34 x 1.76 x 0.11 u 판이 됐다. 카메라가 시작부터 그 판 안이라 dolly_in 36개 중 35개가 이
    노드에 G5 로 막혀 손잡이 바닥(0.005)에 붙었다. 이런 노드는 물체가 아니라 지면이고, 지면은
    ground 게이트(G6)가 따로 본다. 판정: 높이(extent[2], gravity 프레임 z) < flat_ratio x 수평 최대
    extent **이고** 수평 최대 extent >= flat_min_extent (작은 납작 물체 — 매트·종이 — 는 남긴다).
    flat_ratio <= 0 (기본) 이면 전부 그대로 = 기존 동작.
    """
    if flat_ratio <= 0.0:
        return nodes, []
    kept, skipped = [], []
    for n in nodes:
        ex = n["obb"]["extent"]
        horiz = max(float(ex[0]), float(ex[1]))
        if horiz >= flat_min_extent and float(ex[2]) < flat_ratio * horiz:
            skipped.append(n["id"])
        else:
            kept.append(n)
    return kept, skipped


def node_margins(nodes: list, ratio: float = 0.0, floor: float = 0.0,
                 cap: float = float("inf")) -> dict:
    """노드별 OBB 여유 마진 `m_j = clip(ratio · max(extent_j), floor, cap)` (단위 u).

    **왜 노드 크기에 비례해야 하나 (D51).** 절대 마진 `m` 은 모든 박스를 축마다 똑같이 `2m`
    부풀린다. 그래서 `m=0.15` 는 camel 낙타(0.137×0.060×0.122)의 얇은 축을 6배로, avocado
    의자 조각(0.056×0.005×0.022)의 얇은 축을 **61배**로 키워, 두 유령 박스를 거의 같은
    물리적 크기로 수렴시킨다 — 물체가 무엇이든 금지구역 크기가 같아진다는 뜻이다. 실제로
    사용자가 눈으로 고른 두 씬의 적정 마진(camel 0.15 / avocado 0.06)을 각 씬의 지배 노드
    `max(extent)` 로 나누면 1.09 / 1.07 로 2% 안에 겹친다. `d_ref` 로 나누면 0.235/0.313,
    소스 카메라 자신의 여유로 나누면 0.294/0.515 로 어긋난다.

    `cap` 이 필요한 이유: `max(extent)` 는 벽·창문 같은 큰 노드에서 폭주한다 (avocado
    `stat_1` window 는 3.393 u 라 `ratio=1` 이면 방보다 큰 마진이 나온다). 큰 노드는 부분
    관측이 아니라 잘 관측된 면이라 비례 마진의 근거(관측 부족 보정)가 애초에 약하다.

    `cap` 은 **비례 항에만** 걸린다 (`max(floor, min(ratio·size, cap))`). floor 까지 깎으면
    소스 대비 배수 모드(`--obb_clear_src_ratio`)가 cap 에 잘려버린다 — camel 은 floor 가
    0.153 인데 cap 0.12 가 그걸 덮으면 안 된다. `floor ≤ cap` 인 구간에서는 `clip` 과 같다.

    `ratio=0` 이면 전 노드가 `floor` 로 같아져 **절대 마진 시절과 완전히 동일**하다 —
    선택 노드(argmin)도 값도 안 바뀐다.

    **크기 비례는 실측에서 기각됐다 (D51).** `ratio=1.0` 은 camel 을 재현했지만 avocado 는
    절대 0.06 (path 2.25) 보다 나쁜 1.19 였다. avocado 의 binding 노드가 `stat_0` 테이블
    (0.376×0.354×0.024) 이라 `max_ext` 가 판때기의 **너비**를 집어 cap 에 붙어버리고, ratio 가
    아무 일도 안 하기 때문. `min_ext` 로 바꾸면 `stat_0` 은 1% 로 맞는 대신 `stat_4` 가 12배로
    튄다 — 어느 척도를 써도 binding 노드 셋 중 둘만 맞고 그 둘이 척도마다 바뀐다. 그래서
    기본값은 `ratio=0` + `floor = 0.3 × (소스 카메라 자신의 여유)` 다.
    """
    out = {}
    for node in nodes:
        size = float(np.max(np.asarray(node["obb"]["extent"], dtype=float)))
        out[node["id"]] = float(max(floor, min(ratio * size, cap)))
    return out


def obb_clearance(poses, nodes: list, T_gw, frames=None, node_ids=None, margins=None):
    """궤적 프레임별 **모든 노드 OBB** 까지의 최소 부호 거리와 그 노드 id.
    반환 `(dists(f,), ids(f,), slacks(f,))` — `slack = dist − m_j` 가 **판정량**이다.

    왜 G1/standoff 로 안 되나 (D49): 둘 다 **관측된 표면**을 잰다. 표면은 물체의 껍질이라
    카메라가 물체 **내부**를 지나가도 반대쪽 껍질이 뒤에 있으면 G1 은 "표면 앞"으로 읽고,
    standoff 는 그저 "가까운 점"으로만 읽는다. OBB 는 부피 판정이라 그 구멍을 막고, 덤으로
    **무엇과 부딪혔나**를 노드 라벨로 말해준다.

    좌표: `obb` 는 그래프 프레임 G 에 있고 `T_gw[:3,:3] = R_gw/S` 라 **G 좌표가 곧 u 단위**다.
    따라서 world 카메라 중심을 `T_gw` 로 옮기기만 하면 S 로 다시 나눌 필요가 없다.

    동적 노드(`moving`)는 `node_obb_at(node, f)` 로 **그 프레임 위치**를 쓴다 — 정적 노드까지
    프레임별 track 을 쓰면 추정 jitter(camel `fence` 의 `path_len_u` 0.32)가 그대로 판정에
    들어온다. 렌더가 0회다 (노드당 3×3 곱 하나).

    `margins` (`node_margins()` 결과, D51) 를 주면 노드 선택이 **거리가 아니라 slack 최소**로
    바뀐다 — 작고 가까운 노드가 큰 마진을 요구할 때 정작 binding 노드가 딴 데로 잡히는 걸
    막는다. `margins=None` 이거나 전 노드가 같은 값이면 argmin 이 동일해 **기존과 완전히 같다**.
    """
    from scene_graph.obb import node_obb_at                      # 순환 import 회피

    poses = np.asarray(poses, dtype=float)
    T_gw = np.asarray(T_gw, dtype=float)
    frames = range(len(poses)) if frames is None else frames
    keep = [n for n in nodes if node_ids is None or n["id"] in node_ids]
    dists, hits, slacks = [], [], []
    for f in frames:
        p_g = T_gw[:3, :3] @ poses[f][:3, 3] + T_gw[:3, 3]
        best, best_id, best_slack = float("inf"), "", float("inf")
        for node in keep:
            if node.get("moving"):
                center, extent, R = node_obb_at(node, int(f))
            else:
                obb = node["obb"]
                center, extent, R = obb["center"], obb["extent"], obb["R"]
            value = obb_signed_distance(p_g, center, extent, R)
            slack = value - (margins.get(node["id"], 0.0) if margins else 0.0)
            if slack < best_slack:
                best, best_id, best_slack = value, node["id"], slack
        dists.append(best)
        hits.append(best_id)
        slacks.append(best_slack)
    return (np.asarray(dists, dtype=np.float64), hits,
            np.asarray(slacks, dtype=np.float64))


def elevation_profile(poses, node: dict, T_gw, ground_z: float):
    """G6. 프레임별 **subject 대비 고도각**(deg)과 **지면 위 여유**(u). 렌더가 0회다.

    왜 따로 필요한가 — 이 둘은 기존 예산 어느 것도 못 잡는다 (실측 2026-08-22):

      *물체 바로 위/아래*: `rise_reveal` 이 camel 89.96° / avocado 87.13°, `drop_reveal` 이
      −85.98° 까지 간다. 수직 이동은 hole 을 잘 안 늘려서 (바닥·천장에도 점이 있다) 사다리의
      shape doubling 이 천장에 안 걸리고 끝까지 배가된다 — `lateral_frac` 0.35 × 16 이면
      `atan(5.6)` = 80° 다. hole 0.32 · occlusion 0.95 로 **전 예산을 통과**한다.

      *지면 아래*: `drop_reveal`/`pedestal_down` 이 바닥을 뚫는다 (camel 14 변이, avocado 5;
      `dyn_0__drop_reveal__hole0.35` 는 49 중 46 프레임). 그런데 `behind_frac` 이 전부
      **0.0000** 이다 — G1 은 카메라 중심을 소스 프레임에 투영해 깊이와 비교하는데, 바닥 밑
      카메라는 소스 화면 **밖으로** 나가 아예 채점이 안 된다. G5(OBB)도 못 본다: 바닥은
      노드가 아니다 (D54 가 적어둔 맹점 그대로). 가림 지표도 못 덮는다 — 바닥을 아래에서 보면
      점군에 뒷면이 없어 구멍이 되고, 구멍은 "가려짐"이 아니라 "보임"으로 세어진다.

    좌표: `obb_clearance` 와 같다 — `T_gw[:3,:3] = R_gw/S` 라 **G 좌표가 곧 u 단위**이고
    G 의 +z 가 중력 반대(위)다. 따라서 `ground_z` 와 직접 빼면 u 가 나온다.

    반환 `(elev_deg (F,), ground_clear_u (F,))`. `ground_clear` 가 음수면 지면 **아래**다.
    """
    from scene_graph.obb import node_obb_at                      # 순환 import 회피

    poses = np.asarray(poses, dtype=float)
    T_gw = np.asarray(T_gw, dtype=float)
    elev, clear = [], []
    for f in range(len(poses)):
        p_g = T_gw[:3, :3] @ poses[f][:3, 3] + T_gw[:3, 3]
        if node.get("moving"):
            center = np.asarray(node_obb_at(node, int(f))[0], dtype=float)
        else:
            center = np.asarray(node["obb"]["center"], dtype=float)
        delta = p_g - center
        horizontal = float(np.linalg.norm(delta[:2]))
        elev.append(np.degrees(np.arctan2(float(delta[2]), max(horizontal, 1e-9))))
        clear.append(float(p_g[2]) - float(ground_z))
    return np.asarray(elev, dtype=np.float64), np.asarray(clear, dtype=np.float64)


def approach_profile(poses, node: dict, T_gw):
    """G7. 프레임별 **시선축 위 전진 여유**(u) — subject 박스 근접면까지 얼마나 남았나. 렌더 0회.

    왜 G5(OBB clearance)로 안 되나: `obb_signed_distance` 는 **부호 없는 3D 거리**다. 박스
    **옆을** 안전거리로 스쳐 지나가는 궤적은 거리가 계속 마진 위라 통과하는데, 시선 방향으로는
    이미 물체를 **지나쳐** 버렸다. 실측 (2026-08-22, D53 수정 후 뱅크):

      camel   `dyn_0 push_in_arc` 4단 전부 근접면을 0.1972 u 뚫고 **뒷면까지 0.0467 u 통과**.
              그런데 `obb_clear` 는 0.1536 이상 — G5 는 아무것도 못 봤다.
      avocado dolly 84 중 16 이 근접면 침범, 그중 7 이 뒷면 통과 (`stat_4` 0.1570 u).

    "지나쳤다"는 거리가 아니라 **부호**의 문제라, 같은 박스를 재도 축을 하나 정해서 부호 있는
    좌표를 봐야 잡힌다. 그래서 G5 를 고치는 게 아니라 축이 다른 게이트를 하나 더 둔다.

    축 `a` 는 **플랜 시작 카메라 → 노드 중심**(프레임 0)으로 한 번 고정한다. 프레임마다 다시
    잡으면 카메라가 옆으로 돌 때 축도 같이 돌아가 "지나쳤다"가 영원히 안 나온다 — 이 shot 의
    깊이 축은 시작 구도가 정의하는 것이다. 반면 **중심 `c_j(f)` 는 프레임별**로 쓴다 (동적
    노드가 카메라 쪽으로 걸어오면 여유가 실제로 줄어든다).

    `half_a` 는 OBB 를 축 `a` 에 투영한 반폭 (`Σ|R_k · a| · ext_k / 2`) 이라 회전한 박스도 맞다.

    좌표: `obb_clearance`/`elevation_profile` 과 같다 — `T_gw[:3,:3] = R_gw/S` 라 G 좌표가 곧
    u 단위다.

    반환 `(gap (F,), past (F,))`. `gap = −half_a − s` 로 **양수면 근접면 앞**, 음수면 뚫었다.
    `past = s − half_a` 는 양수면 **뒷면까지 통과**했다는 뜻 (진단용, 판정은 `gap` 으로 한다).
    """
    from scene_graph.obb import node_obb_at                      # 순환 import 회피

    poses = np.asarray(poses, dtype=float)
    T_gw = np.asarray(T_gw, dtype=float)
    boxes = [node_obb_at(node, int(f)) if node.get("moving")
             else (node["obb"]["center"], node["obb"]["extent"], node["obb"]["R"])
             for f in range(len(poses))]
    centers = np.stack([np.asarray(b[0], dtype=float) for b in boxes])
    p_g = (T_gw[:3, :3] @ poses[:, :3, 3].T).T + T_gw[:3, 3]     # (F,3) u 단위
    axis = centers[0] - p_g[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        # 시작 카메라가 노드 중심 위다 — 축이 정의가 안 된다. 침범으로 보고 최악을 돌려준다.
        nan = np.full(len(poses), float("nan"))
        return nan, nan
    axis = axis / norm
    # 반폭도 **프레임별**이다 — 동적 노드는 yaw 가 돌아 축 위 투영 폭이 바뀐다.
    half = np.asarray([float(np.sum(np.abs((np.asarray(b[2], dtype=float)
                                            * np.asarray(b[1], dtype=float)[None, :]
                                            / 2.0).T @ axis))) for b in boxes])
    s = np.einsum("fj,j->f", p_g - centers, axis)                # 음수 = 박스 앞
    return -half - s, s - half


def evaluate(candidate: dict, renderer, subject_points, obb_corners_world,
             depths, sky_mask, scale: float, z_med: float, subject_center_world,
             src_center, tile_height: int, tile_width: int, behind_frames,
             max_tau: float = 0.30, max_view_angle_deg: float = 40.0,
             min_coverage: float = 0.55, center_box: float = 0.80,
             subject_area_range=(0.03, 0.50), min_occlusion_pass: float = 0.40,
             render_frame_index: int = 0, K=None):
    """후보 하나를 전 게이트에 통과시킨다. **싼 것부터** 돌고 떨어지면 렌더를 건너뛴다.

    반환 dict 는 그대로 `trace/gates.csv` 의 한 행이자 board 텍스트의 한 줄이 된다.
    """
    row = {**{k: v for k, v in candidate.items() if k not in ("p_g", "look_at_g")},
           "failed": None, "rendered": None}

    tau, view_angle = tau_and_view_angle(candidate["p_world"], subject_center_world,
                                         src_center, z_med)
    row["tau"] = round(tau, 4)
    row["view_angle_deg"] = round(view_angle, 1)
    if tau > max_tau:
        row["failed"] = "G4_tau"
        return row
    if view_angle > max_view_angle_deg:
        row["failed"] = "G4_view_angle"
        return row

    hits = behind_surface_frames(candidate["p_world"], depths, K=renderer.K_src,
                                 cam_c2w=renderer.cam_c2w_src, sky_mask=sky_mask,
                                 scale=scale, frames=behind_frames)
    row["behind_frames"] = len(hits)
    if hits:
        row["failed"] = "G1_behind"
        return row

    # K=None 이면 소스 intrinsics 그대로 (기존 경로). micro-adjust 의 zoom 연산만 K 를 넘긴다 —
    # 그래야 focal 을 바꾼 뒤에도 같은 게이트로 판정된다.
    rendered = renderer.render(candidate["c2w_world"], K=K, frame=render_frame_index,
                               height=tile_height, width=tile_width)
    alone = renderer.render(candidate["c2w_world"], K=K, frame=render_frame_index,
                            height=tile_height, width=tile_width, subset=subject_points)
    row["rendered"] = rendered

    row["coverage"] = round(float(rendered["valid"].mean()), 4)
    subject_pixels = alone["valid"]                       # 가림이 없었을 때의 실루엣
    drawn = subject_pixels & rendered["valid"]
    # 전체 렌더의 depth 가 subject 단독 depth 보다 확실히 앞이면 그 픽셀은 뭔가에 가려진 것.
    occluded = drawn & (rendered["depth"] < alone["depth"] - 0.02 * scale)
    area = float(subject_pixels.mean())
    row["subject_area"] = round(area, 4)
    row["occlusion_pass"] = round(float(1.0 - occluded.sum() / max(subject_pixels.sum(), 1)), 3)

    ys, xs = np.nonzero(subject_pixels)
    if len(xs) == 0:
        row["subject_center"] = None
        row["failed"] = "G3_area"
        return row
    cx, cy = float(xs.mean() / tile_width), float(ys.mean() / tile_height)
    row["subject_center"] = [round(cx, 3), round(cy, 3)]
    row["subject_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    lo, hi = (1 - center_box) / 2, 1 - (1 - center_box) / 2
    if row["coverage"] < min_coverage:
        row["failed"] = "G2_coverage"
    elif not (lo <= cx <= hi and lo <= cy <= hi):
        row["failed"] = "G3_center"
    elif not (subject_area_range[0] <= area <= subject_area_range[1]):
        row["failed"] = "G3_area"
    elif row["occlusion_pass"] < min_occlusion_pass:
        row["failed"] = "G3_occlusion"
    return row


def rank(rows: list):
    """통과 후보를 board 순서로. coverage × subject 크기 적정도.

    크기 적정도는 log 면적이 목표 0.12 에서 얼마나 떨어졌나 — 선형으로 재면 큰 쪽만 이긴다.
    """
    passed = [r for r in rows if r["failed"] is None]
    def score(row):
        area = max(row["subject_area"], 1e-4)
        fit = np.exp(-abs(np.log(area / 0.12)) / 1.2)
        return row["coverage"] * fit * row["occlusion_pass"]
    for row in passed:
        row["board_score"] = round(float(score(row)), 4)
    return sorted(passed, key=lambda r: -r["board_score"])


def write_csv(output_path: str, rows: list):
    """전 후보 1행씩. 통과·탈락 모두 — 탈락 이유 분포가 곧 진단이다."""
    columns = ["cand_id", "board_label", "azimuth_deg", "elevation_deg",
               "d_azimuth_deg", "d_elevation_deg", "distance_ratio",
               "radius_u", "tau", "view_angle_deg", "behind_frames", "coverage",
               "subject_area", "occlusion_pass", "subject_center", "board_score", "failed"]
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")
        for row in rows:
            values = []
            for column in columns:
                value = row.get(column, "")
                if isinstance(value, (list, tuple)):
                    value = "|".join(str(v) for v in value)
                values.append("" if value is None else str(value))
            file.write(",".join(values) + "\n")
    return output_path
