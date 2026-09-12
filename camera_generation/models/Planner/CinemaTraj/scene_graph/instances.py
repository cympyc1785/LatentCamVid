"""SAM3 track → G 좌표 점군 인스턴스. 걸러낸 track 은 전부 이유와 함께 남긴다.

왜 기각 로그를 남기나: 나중에 "후보가 전부 게이트 탈락" 같은 증상이 나오면 원인이 게이트가
아니라 **subject 가 애초에 sliver track 이었다**인 경우가 많다. 조용히 버리면 그 진단이 불가능하다.
`diagnostics.dropped_tracks` 에 기각 사유와 수치를 전부 적는다.

기각 기준 3개는 서로 다른 실패를 잡는다:
  `max_area_frac < 0.002`  — 화면의 0.2% 미만. OBB 를 세울 점이 없다(720p 에서 ~1800 px)
  `visible_frames < 5`     — 스쳐 지나간 오검출. yaw median(5프레임)이 성립 안 함
  `mean_score < 0.3`       — SAM3 자신이 확신 없음

이 셋은 **track 하나하나가 멀쩡한지**만 본다. 개별로는 다 멀쩡한데 **수가 폭발하는** 실패는
`cap_instances` 가 따로 막는다 (car-roundabout 차 69대). 둘은 다른 문제다.
"""
import numpy as np

from .lift import apply_transform, radius_outlier_mask, unproject_mask, valid_pixels


def trim_depth_tail(depth: np.ndarray, mask: np.ndarray, quantile: float):
    """마스크 안 depth 의 양끝 `quantile` 을 잘라낸다. **OBB extent 를 지키는 핵심 단계다.**

    SAM3 마스크는 실루엣에서 몇 px 배경으로 번지고, 그 픽셀의 depth 는 물체가 아니라 뒤쪽
    바닥/벽이다. camel frame1 에서 마스크 안 z 는 2.61~3.52 인데 1~99% 는 2.61~3.10 이다 —
    꼬리 0.42 가 전부 배경이고, `minAreaRect` 는 **최외곽 점 하나**에 끌려가므로 그 꼬리가
    깊이축 extent 를 통째로 늘린다. 실제로 이걸 안 하면 OBB 투영 bbox 가 마스크 대비 4.3배였다.
    `radius_outlier_mask` 로는 못 잡는다 — 배경 픽셀은 서로 인접해 이웃이 충분하다.
    """
    if quantile <= 0 or not mask.any():
        return mask
    z = depth[mask]
    lo, hi = np.quantile(z, [quantile, 1.0 - quantile])
    trimmed = mask.copy()
    trimmed[mask] = (z >= lo) & (z <= hi)
    return trimmed


def trim_depth_mode(depth: np.ndarray, mask: np.ndarray, bins: int = 40,
                    gap_frac: float = 0.005):
    """마스크 안 depth 의 **중앙값이 속한 덩어리만** 남긴다. `trim_depth_tail` 로는 못 잡는 것.

    `trim_depth_tail` 은 양끝 1% 를 자르는 대칭 처리라 "얇은 번짐"만 잡는다. goat 에서 나온 건
    다른 종류다 — 마스크 안 depth 가 **완전히 두 덩어리**였다: 염소 본체가 z 0.62~1.0 에 44k px,
    그리고 **중간이 텅 빈 채로** z 6.5~7.5 에 7000 px (마스크의 14%). SAM3 가 언덕 저편의 다른
    염소 무리까지 같은 track 으로 묶은 것이다. 14% 는 quantile 1% 로 절대 안 잘리고, 그 결과
    OBB 가 시선 방향으로 늘어난 납작한 판(extent `[0.96, 0.05, 0.82]` u)이 되어 투영 크기가
    마스크의 **13배**였다. `radius_outlier_mask` 도 못 잡는다 — 저쪽 덩어리도 7000 점이라
    자기들끼리 이웃이 충분하다.

    log z 히스토그램에서 중앙값 bin 부터 좌우로 **`gap_frac` 미만인 bin 을 만날 때까지** 넓힌다.
    단봉이면 거의 전 구간이 남으므로 camel/avocado 같은 기존 씬은 사실상 그대로다 (camel 은
    이 처리로 4.37~5.2, quantile trim 결과와 같다).
    """
    if bins <= 0 or not mask.any():
        return mask
    z = depth[mask]
    if z.size < 100 or z.min() <= 0:
        return mask
    log_z = np.log(z)
    counts, edges = np.histogram(log_z, bins=bins)
    floor = max(1.0, gap_frac * z.size)
    peak = int(np.searchsorted(edges, np.median(log_z), side="right") - 1)
    peak = int(np.clip(peak, 0, bins - 1))
    lo_bin = peak
    while lo_bin > 0 and counts[lo_bin - 1] >= floor:
        lo_bin -= 1
    hi_bin = peak
    while hi_bin < bins - 1 and counts[hi_bin + 1] >= floor:
        hi_bin += 1
    if lo_bin == 0 and hi_bin == bins - 1:
        return mask                      # 단봉 — 자를 게 없다
    lo, hi = float(np.exp(edges[lo_bin])), float(np.exp(edges[hi_bin + 1]))
    trimmed = mask.copy()
    trimmed[mask] = (z >= lo) & (z <= hi)
    return trimmed


def summarize_tracks(recon: dict):
    """lift **없이** track 요약만 만든다 — `mask.sum()` 뿐이라 unprojection 대비 사실상 공짜.

    왜 따로 두나: `filter_instances` 와 `cap_instances` 의 판정에 쓰이는 양(`max_area_frac`,
    `num_visible_frames`, `mean_score`)은 전부 2D 마스크만 있으면 나온다. 그런데 예전 순서는
    **전부 lift 한 뒤에** 걸렀다. car-roundabout 은 track 이 123개(동적 69 + 정적 54)라 그중
    100개 넘게 버릴 것을 49프레임씩 unproject 하고 나서 버렸다 — graph 빌드 시간이 track 수에
    선형으로 늘어난 주범이다. 이걸 먼저 돌려 사전 선별한 뒤 살아남은 것만 lift 한다.
    """
    depths = recon["depths"]
    frame_area = depths.shape[1] * depths.shape[2]
    summaries = []
    for seg in recon["segs"]:
        for track_id in seg.track_ids():
            track = seg.tracks[track_id]
            area_frac = {f: float(seg.track_mask(track_id, f).sum()) / frame_area
                         for f in track["frames"]}
            summaries.append({
                "kind": seg.kind, "track_id": track_id, "label": track["keyword"],
                "points_by_frame": {},     # 아직 안 올렸다 — min_points 검사는 lift 뒤에
                "max_area_frac": float(max(area_frac.values())) if area_frac else 0.0,
                "mean_score": float(np.mean(track["scores"])) if track["scores"] else 0.0,
                "num_visible_frames": len(track["frames"]),
            })
    return summaries


def build_instances(recon: dict, T_gw: np.ndarray, scale: float,
                    max_points_per_frame: int = 20000, denoise_radius_u: float = 0.03,
                    denoise_min_neighbors: int = 6, depth_trim_quantile: float = 0.01,
                    depth_mode_bins: int = 40, depth_mode_gap_frac: float = 0.005,
                    select: set = None, seed: int = 0):
    """seg track 마다 프레임별 G 점군을 만든다. 기각은 하지 않고 전부 반환.

    `select` 는 `(kind, track_id)` 집합. 주면 그 track 만 lift 한다 (`summarize_tracks` 로 먼저
    걸러낸 결과를 받는 용도). None 이면 전부 — 예전 동작.
    """
    rng = np.random.default_rng(seed)
    depths, K, cam_c2w, sky = recon["depths"], recon["K"], recon["cam_c2w"], recon["sky_mask"]
    num_frames, height, width = depths.shape
    frame_area = height * width

    # valid 마스크는 프레임당 한 번만 계산한다 (edge_bleed 가 track 수만큼 반복되면 낭비).
    valid_by_frame = [valid_pixels(depths[f], sky[f]) for f in range(num_frames)]

    instances = []
    for seg in recon["segs"]:
        for track_id in seg.track_ids():
            if select is not None and (seg.kind, track_id) not in select:
                continue
            track = seg.tracks[track_id]
            points_by_frame, area_frac = {}, {}
            for f in track["frames"]:
                mask = seg.track_mask(track_id, f)
                area_frac[f] = float(mask.sum()) / frame_area
                mask = mask & valid_by_frame[f]
                # 순서가 중요하다: **덩어리 먼저, 꼬리 나중.** 두 덩어리가 남아 있으면 중앙값이
                # 어디에 앉느냐에 따라 quantile 이 엉뚱한 쪽을 자른다.
                mask = trim_depth_mode(depths[f], mask, depth_mode_bins, depth_mode_gap_frac)
                mask = trim_depth_tail(depths[f], mask, depth_trim_quantile)
                if not mask.any():
                    continue
                points_w, _ = unproject_mask(depths[f], K[f], cam_c2w[f], mask)
                points_g = apply_transform(T_gw, points_w)
                if len(points_g) > max_points_per_frame:
                    points_g = points_g[rng.choice(len(points_g), max_points_per_frame, replace=False)]
                keep = radius_outlier_mask(points_g, denoise_radius_u, denoise_min_neighbors)
                if keep.sum() >= 20:
                    points_g = points_g[keep]
                points_by_frame[f] = points_g

            instances.append({
                "kind": seg.kind, "track_id": track_id, "label": track["keyword"],
                "frames": list(track["frames"]), "scores": list(track["scores"]),
                "boxes_xyxy": list(track["boxes"]),
                "points_by_frame": points_by_frame,
                "area_frac_by_frame": area_frac,
                "max_area_frac": float(max(area_frac.values())) if area_frac else 0.0,
                "mean_score": float(np.mean(track["scores"])) if track["scores"] else 0.0,
                "num_visible_frames": len(track["frames"]),
            })
    return instances


def filter_instances(instances: list, min_area_frac: float = 0.002, min_frames: int = 5,
                     min_score: float = 0.3, min_points: int = 200):
    """(kept, dropped). dropped 는 사유 문자열 포함."""
    kept, dropped = [], []
    for inst in instances:
        total_points = sum(len(p) for p in inst["points_by_frame"].values())
        reasons = []
        if inst["max_area_frac"] < min_area_frac:
            reasons.append(f"max_area_frac {inst['max_area_frac']:.5f} < {min_area_frac}")
        if inst["num_visible_frames"] < min_frames:
            reasons.append(f"visible_frames {inst['num_visible_frames']} < {min_frames}")
        if inst["mean_score"] < min_score:
            reasons.append(f"mean_score {inst['mean_score']:.3f} < {min_score}")
        if total_points < min_points:
            reasons.append(f"lifted_points {total_points} < {min_points}")
        (dropped if reasons else kept).append(
            {**inst, "drop_reasons": reasons} if reasons else inst)
    return kept, dropped


def cap_instances(instances: list, per_keyword_top_k: int = 5,
                  max_dyn_nodes: int = 6, max_stat_nodes: int = 10):
    """스케일에서 인스턴스가 폭발할 때 상한을 건다. (kept, dropped).

    왜 필요한가 — 52편 확장 파일럿에서 나온 실측이다. `car-roundabout` 은 동적 track 69개
    (거리의 차 전부), `parkour`/`car-roundabout` 은 정적 56/54개, `goat` 은 `rock` 하나로
    31개를 냈다. 기존 세 기각 기준(`filter_instances`)은 **각 track 이 개별적으로 멀쩡한지**만
    보므로 이걸 못 막는다 — 저 차들은 하나하나 다 진짜 차다.

    폭발이 왜 해로운가 (조용히 망가지는 종류라 적어 둔다):
      ① G5 상자 여유 게이트는 **모든 노드에 대한 최소값**이다. 상자가 수십 개면 자유공간이
         잘게 쪼개져 어느 방향으로도 카메라가 못 나간다 — 게이트가 전부 `obb_limited` 로
         수렴하고 그게 씬 탓인지 노드 수 탓인지 구분이 안 된다.
      ② `rock` 31개는 사실 지형이다. 지형에 OBB 를 세우는 건 의미가 없고, 그 상자가
         카메라를 막는 건 더더욱 의미가 없다.
      ③ graph 빌드 시간이 track 수에 선형이다 (goat 4분 중 대부분).

    순위는 `max_area_frac` (최대 화면 점유율) — `sample_camera_bank.py --min_area_frac` 이
    앵커를 고를 때 쓰는 것과 **같은 양**이라 상·하류가 어긋나지 않는다. 동점은 관측 프레임 수.

    dyn/stat 을 따로 세는 이유: 역할이 다르다. dyn 은 subject 후보고, stat 은 장애물이다.
    dyn 을 6개로 깎아도 subject 는 여전히 1등이지만, stat 을 6개로 깎으면 진짜 벽이 빠질 수 있다.

    **버리는 것은 전부 기록한다** — `diagnostics.dropped_tracks` 에 사유가 남아야 나중에
    "왜 저 물체가 그래프에 없지"가 추적 가능하다.
    """
    if min(per_keyword_top_k, max_dyn_nodes, max_stat_nodes) <= 0:
        return instances, []       # 상한 끔 = 예전 동작 그대로 (집 규칙: 기존 경로 유지)
    rank = lambda inst: (-inst["max_area_frac"], -inst["num_visible_frames"], str(inst["label"]))
    per_kind_cap = {"dyn": max_dyn_nodes, "stat": max_stat_nodes}

    # ① keyword 별 상한 — 같은 이름 31개를 먼저 깎는다. 서로 다른 이름은 아직 안 건드린다.
    survivors, dropped = [], []
    by_keyword = {}
    for inst in instances:
        by_keyword.setdefault((inst["kind"], str(inst["label"])), []).append(inst)
    for (kind, label), group in by_keyword.items():
        group = sorted(group, key=rank)
        survivors.extend(group[:per_keyword_top_k])
        for inst in group[per_keyword_top_k:]:
            dropped.append({**inst, "drop_reasons": [
                f"keyword '{label}' {len(group)}개 중 면적 상위 {per_keyword_top_k} 밖"]})

    # ② kind 별 전역 상한 — 이름이 다 다른데도 많은 경우 (parkour 정적 56개)를 여기서 막는다.
    kept = []
    by_kind = {}
    for inst in survivors:
        by_kind.setdefault(inst["kind"], []).append(inst)
    for kind, group in by_kind.items():
        cap = per_kind_cap.get(kind, max_stat_nodes)
        group = sorted(group, key=rank)
        kept.extend(group[:cap])
        for inst in group[cap:]:
            dropped.append({**inst, "drop_reasons": [
                f"{kind} 노드 {len(group)}개 중 면적 상위 {cap} 밖"]})
    return kept, dropped


def voxel_keys(points: np.ndarray, voxel: float):
    return set(map(tuple, np.floor(points / voxel).astype(np.int64)))


def merge_duplicates(instances: list, voxel_u: float = 0.02, iou_threshold: float = 0.5):
    """같은 keyword 인 track 이 3D 로 겹치면 하나로 합친다. (merged, merge_log).

    SAM3 는 같은 keyword 를 여러 track 으로 쪼개 내놓는다 (avocado-slice 에서 woman/person/human
    이 서로 IoU 0.99). 2D IoU 로 합치는 건 `merge_seg_instances.py` 가 이미 하지만, 거기서
    keyword 가 다르면 안 합쳐지므로 여기서 **3D 로 한 번 더** 본다.
    """
    order = sorted(range(len(instances)), key=lambda i: -sum(
        len(p) for p in instances[i]["points_by_frame"].values()))
    parent = list(range(len(instances)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    keys = [voxel_keys(np.concatenate(list(inst["points_by_frame"].values()), axis=0), voxel_u)
            if inst["points_by_frame"] else set() for inst in instances]
    log = []
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1:]:
            if find(a) == find(b) or not keys[a] or not keys[b]:
                continue
            inter = len(keys[a] & keys[b])
            iou = inter / max(len(keys[a] | keys[b]), 1)
            if iou >= iou_threshold:
                parent[find(b)] = find(a)
                log.append({"kept": f"{instances[a]['kind']}#{instances[a]['track_id']}",
                            "absorbed": f"{instances[b]['kind']}#{instances[b]['track_id']}",
                            "iou_3d": round(float(iou), 4)})

    groups = {}
    for i in range(len(instances)):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for root, members in groups.items():
        if len(members) == 1:
            merged.append(instances[members[0]])
            continue
        base = dict(instances[root])
        base["aliases"] = sorted({instances[i]["label"] for i in members} - {base["label"]})
        points_by_frame = {}
        for i in members:
            for f, points in instances[i]["points_by_frame"].items():
                points_by_frame[f] = np.concatenate([points_by_frame[f], points], axis=0)\
                    if f in points_by_frame else points
        base["points_by_frame"] = points_by_frame
        base["frames"] = sorted({f for i in members for f in instances[i]["frames"]})
        base["num_visible_frames"] = len(base["frames"])
        #    **`frames` 와 나란한 것은 전부 같이 합쳐야 한다.** 예전에는 `frames` 만 합집합으로
        #    바꾸고 `boxes_xyxy`/`scores` 는 root 것을 그대로 뒀다. 그러면 build_node 의
        #    `zip(inst["frames"], inst["boxes_xyxy"])` 가 **짧은 쪽에서 잘리고**, 합집합이 root
        #    보다 앞선 프레임을 포함하면 프레임↔박스가 **한 칸씩 밀려 짝지어진다**.
        #    증상 둘: (a) `best_frame` 이 잘린 꼬리면 `KeyError: 48` 로 씬 전체가 죽는다 —
        #    D182 10,346편 중 73편(0.71%)이 이걸로 죽었고 예외 프레임이 48/47 에 49건 몰렸다.
        #    (b) 안 죽으면 `bbox_xyxy_best`/`track.conf` 가 **조용히 밀린 채** framing 게이트로
        #    간다 — scene_graph 800편 실측 merged 노드 838개 중 34개(4.1%), 비-merged 7663개는 0개.
        #    root 를 먼저 깔고 나머지 멤버로 빈 프레임만 채운다 (겹치면 root 값 유지 = 기존 동작).
        by_frame = {}
        for i in [root] + [m for m in members if m != root]:
            inst = instances[i]
            for f, score, box in zip(inst["frames"], inst["scores"], inst["boxes_xyxy"]):
                by_frame.setdefault(f, (score, box))
        base["scores"] = [by_frame[f][0] for f in base["frames"]]
        base["boxes_xyxy"] = [by_frame[f][1] for f in base["frames"]]
        assert len(base["scores"]) == len(base["frames"]) == len(base["boxes_xyxy"])
        base["max_area_frac"] = max(instances[i]["max_area_frac"] for i in members)
        base["mean_score"] = float(np.mean([instances[i]["mean_score"] for i in members]))
        base["merged_from"] = [f"{instances[i]['kind']}#{instances[i]['track_id']}" for i in members]
        merged.append(base)
    return merged, log


def smooth_centers(centers: np.ndarray, frames: list, num_frames: int,
                   window: int = 11, poly: int = 3):
    """관측 프레임의 center 를 전 프레임으로 선형 보간한 뒤 Savitzky-Golay 로 편다.

    center jitter 는 look-at 을 통해 **곧장 카메라 회전**으로 들어간다. r=0.5u 에서 1% jitter 가
    프레임당 ~1° yaw 라 눈에 보인다. 그래서 여기서 미리 편다.
    """
    from scipy.signal import savgol_filter

    grid = np.arange(num_frames)
    dense = np.stack([np.interp(grid, frames, centers[:, k]) for k in range(3)], axis=-1)
    window = min(window if window % 2 else window + 1, num_frames if num_frames % 2 else num_frames - 1)
    if window >= poly + 2:
        dense = savgol_filter(dense, window_length=window, polyorder=poly, axis=0)
    return dense
