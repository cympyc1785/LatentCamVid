"""TRUMANS mesh GT 로 **G1(표면 뒤) 을 부피로** 판정한다 — depth shell 대신 점유 격자 + EDT.

**왜 필요한가.** 지금 G1(`gates.behind_profile`)이 아는 공간은 소스 카메라가 **본 표면**뿐이다.
카메라 중심을 소스 프레임에 되쏘아 `z_cam > depth + margin` 이면 "표면 뒤"로 친다. 이건 소스가
그 방향을 봤을 때만 성립한다 — 소스 뒤쪽 벽, 소스가 한 번도 안 본 옆방, 가구 내부는 depth 가
없으니 **아무 증거도 못 만든다**. Vista(in-the-wild)는 그 shell 이 가진 전부라 어쩔 수 없지만,
TRUMANS 는 씬이 `.blend` 로 있어서 **부피를 안다**. 알면서 shell 로 판정할 이유가 없다.

**왜 Blender 로 광선을 쏘지 않나.** `scene.ray_cast` 는 프레임마다 `frame_set()` 아마추어 평가가
필요하고, 이 chunk 의 blend 는 **로드만 16.06 s** 다 (502 오브젝트, 실측 `real 16.062s`).
이분법은 변이당 4회 × 영상당 수백 변이라 프로브마다 Blender 를 띄우는 건 불가능하다. 그래서
역할을 쪼갠다 — **Blender 는 삼각형만 뽑고**(`fit/ingest/trumans_export_mesh.py`, chunk 당 1회),
판정은 여기서 numpy/scipy 로 한다. env `vista4d` 에 `open3d`/`pysdf`/`embree` 가 없어서 정확한
mesh proximity 는 못 쓴다 — 점유 격자 + `scipy.ndimage.distance_transform_edt` 가 유일하게
현실적인 경로이고, 어차피 임계가 cm 단위(0.02·S ≈ 3.9 cm)라 5 cm 격자면 해상도가 남는다.

**"안"을 어떻게 정의하나.** 삼각형을 래스터화한 격자는 **껍질**이라 그 자체로는 안팎을 모른다.
`binary_fill_holes` 는 못 쓴다 — 방이 닫힌 상자면 실내가 통째로 "안"으로 메워진다. 대신
**소스 카메라 위치에서 free 공간을 flood-fill** 한다. 소스 카메라가 서 있던 연결 성분이 곧
"카메라가 존재할 수 있는 공간"이고, 거기 안 닿는 free 복셀은 벽 속이거나 봉인된 공동이라
둘 다 G1 위반이다. 이건 게이트의 원래 뜻("소스 카메라가 실제로 서 있던 자리와 같은 공간")을
depth 없이 그대로 옮긴 것이다.

**좌표.** 전부 **blend world (Z-up, metre)** 다. 뱅크 world → blend world 는 앵커 강체변환
하나뿐이고(`bank_to_blender_poses.py:408-418`, 실측 잔차 4.4e-16, 스케일 섞임 없음) 뱅크 단위가
곧 metre 다. 게이트 임계 `margin_frac·S` 도 뱅크 단위라 **그대로 metre 로 넘기면 된다**.

**정적/동적을 나눠 두는 이유** — 게이트가 이미 두 채널이고(`behind_profile(time_match=True)`),
정적 EDT 는 한 번만 구우면 되는데 동적은 프레임마다 다르다. 유클리드 거리라
`min(EDT_static, EDT_dyn)` = 합집합의 EDT 라 두 격자를 따로 들고 있어도 정확도 손해가 없다.

실측 (chunk `tru_1d076f8c_a00_s3f0k6`, 49프레임):
    static  469 오브젝트 / 568,228 삼각형 / AABB (10.83 × 13.08 × 3.17) m
    dynamic  18 오브젝트 / 878,880 삼각형/프레임 (CC_Base_Body 등 캐릭터 리그)

env: `vista4d` (numpy + scipy 만 쓴다)

사용 예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY -m lbm.mesh_collision --mesh_npz out_trumans/<video>/mesh_gt.npz \
        --src_poses out/trumans_recon/<rec>_<tag>/poses_a00.npz \
        --out out_trumans/<video>/mesh_grid.npz --voxel 0.05
"""
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

GRID_FORMAT = "trumans_mesh_grid_v1"


def _tri_sample_levels(tris: np.ndarray, voxel: float, max_level: int = 512) -> np.ndarray:
    """삼각형별 barycentric 분할 수 `m` — 2의 거듭제곱으로 버킷팅.

    `m` 을 최장 변에서 정확히 뽑으면 그룹이 수백 개로 갈려 벡터화가 안 된다. 2의 거듭제곱으로
    올리면 그룹이 10개 남짓이고 표본이 최대 2배 늘 뿐이다 (전부 같은 복셀에 떨어진다).
    """
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    edge = np.maximum.reduce([np.linalg.norm(b - a, axis=1),
                              np.linalg.norm(c - b, axis=1),
                              np.linalg.norm(a - c, axis=1)])
    need = np.ceil(edge / (0.5 * voxel))
    need = np.clip(need, 1.0, float(max_level))
    return np.power(2, np.ceil(np.log2(need))).astype(np.int32)


def _barycentric(level: int) -> np.ndarray:
    """`level` 분할의 barycentric 가중치 (P,3). `level=1` 이면 꼭짓점 3개."""
    i, j = np.meshgrid(np.arange(level + 1), np.arange(level + 1), indexing="ij")
    keep = (i + j) <= level
    u, v = i[keep] / level, j[keep] / level
    return np.stack([u, v, 1.0 - u - v], axis=1)


def voxelize(tris: np.ndarray, origin: np.ndarray, shape: tuple, voxel: float,
             batch_points: int = 4_000_000) -> np.ndarray:
    """삼각형 (T,3,3) → 점유 bool 격자. 표면 표본이라 **껍질**이고 내부는 안 채운다."""
    grid = np.zeros(shape, dtype=bool)
    if len(tris) == 0:
        return grid
    tris = np.asarray(tris, dtype=np.float64)
    levels = _tri_sample_levels(tris, voxel)
    for level in np.unique(levels):
        weights = _barycentric(int(level))                      # (P,3)
        group = tris[levels == level]
        chunk = max(1, batch_points // max(len(weights), 1))
        for start in range(0, len(group), chunk):
            block = group[start:start + chunk]                  # (B,3,3)
            pts = np.einsum("pk,bkd->bpd", weights, block).reshape(-1, 3)
            idx = np.floor((pts - origin) / voxel).astype(np.int64)
            ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
            idx = idx[ok]
            grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def reachable_from(free: np.ndarray, seeds_idx: np.ndarray) -> np.ndarray:
    """`seeds_idx` 가 속한 free 연결 성분들의 합집합.

    씨앗이 점유 복셀에 떨어지면(카메라가 얇은 표면에 스치는 경우) 26-이웃 한 겹을 뒤져 가장
    가까운 free 복셀로 옮긴다. 그래도 없으면 그 씨앗은 버린다 — 남은 씨앗이 하나라도 있으면
    성분은 같으므로(소스 카메라는 전부 한 공간에 있다) 판정이 안 바뀐다.
    """
    from scipy.ndimage import label

    labels, _ = label(free)
    picked, shape = set(), np.asarray(free.shape)
    for seed in seeds_idx:
        if np.any(seed < 0) or np.any(seed >= shape):
            continue
        lab = int(labels[seed[0], seed[1], seed[2]])
        if lab == 0:
            lo = np.maximum(seed - 1, 0)
            hi = np.minimum(seed + 2, shape)
            patch = labels[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            nz = patch[patch > 0]
            if not len(nz):
                continue
            lab = int(np.bincount(nz).argmax())
        picked.add(lab)
    assert picked, "소스 카메라가 어느 free 성분에도 안 닿는다 — 격자/좌표계가 틀렸다"
    return np.isin(labels, list(picked))


def _edt(free: np.ndarray, voxel: float, clip: float) -> np.ndarray:
    """free 복셀 → 가장 가까운 점유 복셀까지의 거리 (m), `clip` 에서 잘라 float16."""
    from scipy.ndimage import distance_transform_edt

    dist = distance_transform_edt(free) * voxel
    return np.minimum(dist, clip).astype(np.float16)


def build_grid(mesh_npz: str, src_c2w: np.ndarray, out: str, voxel: float = 0.05,
               pad: float = 1.0, clip: float = 3.0, verbose: bool = True) -> dict:
    """`mesh_gt.npz` (+ blend world 소스 카메라 c2w) → `mesh_grid.npz`. chunk 당 1회.

    정적 격자는 씬 AABB ∪ 카메라 위치를 `pad` 만큼 부풀린 범위, 동적 격자는 전 프레임 동적
    삼각형의 AABB 합집합 범위다. 동적은 캐릭터 하나라 범위가 훨씬 작아서 49장을 들고 있어도
    정적 한 장보다 가볍다.

    `src_c2w[0]` 을 **앵커**로 같이 저장한다 — 뱅크 world → blend world 가 이 한 장의 곱이라
    (`bank_to_blender_poses.py:408-418`) 소비자가 recon 폴더를 다시 찾을 필요가 없다.
    """
    data = np.load(mesh_npz, allow_pickle=True)
    assert str(data["format"]) == "trumans_mesh_gt_v1", \
        f"mesh_gt 포맷이 아니다: {data['format']}"
    static_tris = np.asarray(data["static_tris"], dtype=np.float64)
    dyn_tris = np.asarray(data["dyn_tris"], dtype=np.float64)
    frame_list = np.asarray(data["frame_list"], dtype=np.int32)
    src_c2w = np.asarray(src_c2w, dtype=np.float64).reshape(-1, 4, 4)
    cam_centers = src_c2w[:, :3, 3]

    corners = [static_tris.reshape(-1, 3)] if len(static_tris) else []
    corners.append(cam_centers)
    if dyn_tris.size:
        corners.append(dyn_tris.reshape(-1, 3))
    allpts = np.concatenate(corners, axis=0)
    s_origin = np.floor((allpts.min(0) - pad) / voxel) * voxel
    s_shape = tuple(int(v) for v in
                    np.ceil((allpts.max(0) + pad - s_origin) / voxel).astype(int) + 1)

    occ_static = voxelize(static_tris, s_origin, s_shape, voxel)
    free_static = ~occ_static
    seeds = np.floor((cam_centers - s_origin) / voxel).astype(np.int64)
    reach = reachable_from(free_static, seeds)
    edt_static = _edt(free_static, voxel, clip)

    if dyn_tris.size:
        dpts = dyn_tris.reshape(-1, 3)
        d_origin = np.floor((dpts.min(0) - pad) / voxel) * voxel
        d_shape = tuple(int(v) for v in
                        np.ceil((dpts.max(0) + pad - d_origin) / voxel).astype(int) + 1)
        edt_dyn = np.stack([_edt(~voxelize(dyn_tris[f], d_origin, d_shape, voxel), voxel, clip)
                            for f in range(len(dyn_tris))], axis=0)
    else:
        d_origin = s_origin.copy()
        d_shape = (1, 1, 1)
        edt_dyn = np.full((len(frame_list), 1, 1, 1), clip, dtype=np.float16)

    makedirs(path.dirname(path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(
        out, format=GRID_FORMAT, source=path.abspath(mesh_npz),
        voxel=float(voxel), pad=float(pad), clip=float(clip),
        frame_list=frame_list,
        static_origin=s_origin, static_shape=np.asarray(s_shape, dtype=np.int32),
        static_edt=edt_static, reachable=np.packbits(reach),
        dyn_origin=d_origin, dyn_shape=np.asarray(d_shape, dtype=np.int32), dyn_edt=edt_dyn,
        cam_centers=cam_centers, anchor_c2w=src_c2w[0],
    )
    info = {"voxel": voxel, "static_shape": s_shape, "dyn_shape": d_shape,
            "static_occupied": int(occ_static.sum()), "reachable": int(reach.sum()),
            "free": int(free_static.sum()), "frames": int(len(frame_list)), "out": out}
    if verbose:
        width = max(len(k) for k in info)
        for key, value in info.items():
            print(f"{key:<{width}}  {value}")
    return info


class MeshClearance:
    """`mesh_grid.npz` 조회기. 질의는 전부 **blend world metre** 로 받는다."""

    def __init__(self, grid_npz: str):
        data = np.load(grid_npz, allow_pickle=True)
        assert str(data["format"]) == GRID_FORMAT, f"격자 포맷이 아니다: {data['format']}"
        self.voxel = float(data["voxel"])
        self.clip = float(data["clip"])
        self.frame_list = np.asarray(data["frame_list"], dtype=np.int32)
        self.s_origin = np.asarray(data["static_origin"], dtype=np.float64)
        self.s_shape = tuple(int(v) for v in data["static_shape"])
        self.static_edt = np.asarray(data["static_edt"])
        count = int(np.prod(self.s_shape))
        self.reachable = np.unpackbits(np.asarray(data["reachable"]))[:count] \
            .astype(bool).reshape(self.s_shape)
        self.d_origin = np.asarray(data["dyn_origin"], dtype=np.float64)
        self.d_shape = tuple(int(v) for v in data["dyn_shape"])
        self.dyn_edt = np.asarray(data["dyn_edt"])
        #    뱅크 world → blend world 앵커. `to_blend(bank_c2w) = anchor_c2w @ bank_c2w`.
        self.anchor_c2w = np.asarray(data["anchor_c2w"], dtype=np.float64)
        self.cam_centers = np.asarray(data["cam_centers"], dtype=np.float64)

    def to_blend(self, bank_c2w):
        """뱅크 world c2w (F,4,4) → blend world c2w. 강체 앵커 곱 하나뿐이다."""
        return self.anchor_c2w @ np.asarray(bank_c2w, dtype=np.float64)

    def assert_anchor(self, src_bank_c2w, tol: float = 1e-6):
        """뱅크 world 소스 카메라를 앵커로 되돌리면 격자를 구울 때 쓴 중심과 같아야 한다.

        좌표 규약이 어긋나면 clearance 가 **조용히** 엉뚱한 곳을 재게 되므로, mesh 게이트를
        켤 때마다 한 번씩 다시 확인한다 (`bank_to_blender_poses.py --verify` 와 같은 검사).
        """
        back = self.to_blend(np.asarray(src_bank_c2w, dtype=np.float64))[:, :3, 3]
        err = float(np.abs(back - self.cam_centers).max())
        assert err < tol, f"뱅크 world -> blend world 앵커가 안 맞는다: |Δt|={err:.3e}"
        return err

    @staticmethod
    def _lookup(field, origin, shape, voxel, pts, outside):
        """격자 밖 점은 `outside` 로 채운다 — 정적은 "모르는 공간", 동적은 "멀다"."""
        idx = np.floor((np.asarray(pts, dtype=np.float64) - origin) / voxel).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
        out = np.full(len(idx), outside, dtype=np.float64)
        sel = idx[ok]
        if len(sel):
            out[ok] = field[sel[:, 0], sel[:, 1], sel[:, 2]].astype(np.float64)
        return out, ok

    def static_clearance(self, pts):
        """(정적 표면까지 거리 m, 소스 카메라와 같은 공간인가).

        **격자 밖은 통과시킨다** (거리 `clip`, reachable True). 격자는 씬 AABB ∪ 카메라를
        `pad` 만큼 부풀린 것이라 그 밖에는 삼각형이 하나도 없다 — "충돌 없음"이 관대한 게
        아니라 **사실**이다. depth 판 G1 도 증거가 없으면 안 잡으므로 규약이 같다. 카메라가
        씬 밖 30 m 로 날아가는 건 G6(지면/고도) 이 잡을 일이지 G1 이 잡을 일이 아니다.
        """
        dist, inside = self._lookup(self.static_edt, self.s_origin, self.s_shape,
                                    self.voxel, pts, self.clip)
        reach, _ = self._lookup(self.reachable.astype(np.float32), self.s_origin,
                                self.s_shape, self.voxel, pts, 1.0)
        return dist, reach > 0.5

    def dynamic_clearance(self, pts, frame: int):
        """`frame` (동적 격자 인덱스) 에서 동적 표면까지 거리 m. 격자 밖은 `clip`."""
        dist, _ = self._lookup(self.dyn_edt[int(frame)], self.d_origin, self.d_shape,
                               self.voxel, pts, self.clip)
        return dist


def human_aim_points(mesh: "MeshClearance", frame: int, fracs=(0.5, 0.7, 0.85)):
    """동적 격자 `frame` 에서 사람 몸통 조준점 3개 (blend world). (3,3) 또는 None(사람 없음).

    `trumans_scene_probe.aim_points`(human = 몸통 높이띠 3점 median xy) 를 격자로 흉내 낸다.
    동적 EDT 가 0 인 복셀 = 사람 표면이 걸친 복셀이다.
    """
    field = np.asarray(mesh.dyn_edt[int(frame)], dtype=np.float32)
    occ = np.argwhere(field <= 0.5 * mesh.voxel)
    if not len(occ):
        return None
    pts = mesh.d_origin + (occ + 0.5) * mesh.voxel
    z0, z1 = float(pts[:, 2].min()), float(pts[:, 2].max())
    out = []
    for frac in fracs:
        z = z0 + frac * (z1 - z0)
        band = pts[np.abs(pts[:, 2] - z) < 1.5 * mesh.voxel]
        xy = np.median((band if len(band) else pts)[:, :2], axis=0)
        out.append([xy[0], xy[1], z])
    return np.asarray(out, dtype=np.float64)


def _static_march(mesh: "MeshClearance", origin, direction, max_dist: float):
    """정적 EDT 위 sphere tracing. -> 첫 표면까지 거리 (못 맞히면 inf). 사람은 안 본다."""
    d = np.asarray(direction, dtype=np.float64)
    d = d / max(np.linalg.norm(d), 1e-12)
    o = np.asarray(origin, dtype=np.float64)
    hit_eps, t = 0.5 * mesh.voxel, 0.0
    while t < max_dist:
        dist, _ = mesh.static_clearance((o + t * d)[None])
        if dist[0] <= hit_eps:
            return t
        t += max(float(dist[0]) - hit_eps, hit_eps)
    return float("inf")


def _static_march_batch(mesh: "MeshClearance", origins, dirs, max_dist, iters: int = 96):
    """`_static_march` 의 벡터판: 광선 N 개를 한꺼번에. -> (N,) 첫 표면 거리 (없으면 inf)."""
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(dirs, dtype=np.float64)
    d = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    lim = np.broadcast_to(np.asarray(max_dist, dtype=np.float64), (len(o),)).copy()
    eps = 0.5 * mesh.voxel
    t = np.zeros(len(o))
    hit = np.full(len(o), np.inf)
    live = np.ones(len(o), dtype=bool)
    for _ in range(iters):
        if not live.any():
            break
        idx = np.flatnonzero(live)
        dist, _ = mesh.static_clearance(o[idx] + t[idx, None] * d[idx])
        got = dist <= eps
        hit[idx[got]] = t[idx[got]]
        live[idx[got]] = False
        step = np.maximum(dist - eps, eps)
        t[idx[~got]] += step[~got]
        live &= t < lim
    return hit


def mesh_ray_profile(poses_blend, mesh: "MeshClearance", probe_distance: float = 1.5,
                     floor_bias: float = 0.0, fast: bool = True):
    """`mesh_ray_profile` 참조 정의 (아래) 의 벡터판. `fast=False` 면 프레임 루프 판."""
    if not fast:
        return _mesh_ray_profile_loop(poses_blend, mesh, probe_distance)
    poses = np.asarray(poses_blend, dtype=np.float64)
    pos = poses[:, :3, 3]
    n = len(pos)
    aims = [human_aim_points(mesh, i) for i in range(n)]
    clr, _ = mesh.static_clearance(pos)
    floor = _static_march_batch(mesh, pos, np.tile([0.0, 0.0, -1.0], (n, 1)), 10.0) + floor_bias
    ok = np.array([a is not None for a in aims])
    sdist = np.full(n, np.inf)
    clear = np.zeros(n, dtype=bool)
    if ok.any():
        A = np.stack([aims[i] for i in np.flatnonzero(ok)])            # (m,3,3)
        P = pos[ok][:, None, :].repeat(A.shape[1], axis=1)             # (m,3,3)
        vec = (A - P).reshape(-1, 3)
        length = np.linalg.norm(vec, axis=1)
        sdist[ok] = length.reshape(A.shape[:2]).min(axis=1)
        blocked = np.isfinite(_static_march_batch(mesh, P.reshape(-1, 3), vec, length))
        clear[ok] = (~blocked).reshape(A.shape[:2]).any(axis=1)
    clearance = np.minimum(clr, probe_distance)
    return {"clear_frac": float(clear.mean()), "min_clearance": float(clearance.min()),
            "min_subject_dist": float(sdist.min()), "min_floor_drop": float(floor.min()),
            "frames": [{"clear": bool(c), "clearance": float(k), "floor_drop": float(f),
                        "subject_dist": float(sd)} for c, k, f, sd in zip(clear, clearance, floor, sdist)]}


def _mesh_ray_profile_loop(poses_blend, mesh: "MeshClearance", probe_distance: float = 1.5):
    """궤적 (F,4,4) blend world -> `trumans_scene_probe --verify_poses` 와 같은 열을 **격자로**.

    이 코드가 답하는 질문: "Blender 를 안 띄우고 mesh 격자만으로 raycast 게이트
    (clearance / floor_drop / subject_dist / 시선) 를 fitting 루프 안에서 잴 수 있나" (R27).

    정의 대응 — Blender 쪽 (`trumans_scene_probe.py:500-534`) 과 격자 쪽:
      clearance     6방향 광선 최단(사람 제외, 1.5 m 포화)  ~ 정적 EDT (전방향이라 ≤ 6방향)
      floor_drop    아래(-Z) 광선 첫 히트                  ~ 정적 EDT sphere tracing (-Z)
      subject_dist  조준점 3개까지 최단 거리               ~ 격자 몸통 3점까지 최단
      clear         조준점 중 하나라도 첫 히트가 subject    ~ 조준점까지 정적 표면에 안 막힘
    프레임 i 는 격자 `dyn_edt[i]` (뱅크 49프레임과 같은 순서) 와 짝이다.
    """
    poses = np.asarray(poses_blend, dtype=np.float64)
    rows = []
    for i in range(len(poses)):
        pos = poses[i, :3, 3]
        aims = human_aim_points(mesh, i)
        clr, _ = mesh.static_clearance(pos[None])
        floor = _static_march(mesh, pos, (0.0, 0.0, -1.0), 10.0)
        if aims is None:
            clear, sdist = False, float("inf")
        else:
            sdist = float(np.min(np.linalg.norm(aims - pos, axis=1)))
            clear = any(_static_march(mesh, pos, a - pos, float(np.linalg.norm(a - pos)))
                        == float("inf") for a in aims)
        rows.append({"clear": bool(clear), "clearance": float(min(clr[0], probe_distance)),
                     "floor_drop": float(floor), "subject_dist": sdist})
    return {"clear_frac": float(np.mean([r["clear"] for r in rows])),
            "min_clearance": float(min(r["clearance"] for r in rows)),
            "min_subject_dist": float(min(r["subject_dist"] for r in rows)),
            "min_floor_drop": float(min(r["floor_drop"] for r in rows)),
            "frames": rows}


def mesh_behind_profile(poses_blend, mesh: MeshClearance, margin: float,
                        detail: bool = False):
    """`gates.behind_profile` 과 **같은 반환 규약**의 mesh 판정. `margin` 은 metre.

    - **static 채널** — 소스 카메라와 같은 free 성분이 아니거나(벽 속·옆방·봉인 공동) 정적
      표면까지 거리가 `margin` 미만이면 위반.
    - **dynamic 채널** — 플랜 프레임 f ↔ 동적 격자 프레임 `round(f·(F−1)/(N−1))` 한 장에서
      캐릭터 표면까지 거리가 `margin` 미만이면 위반. depth 판의 시간 매칭과 같은 식이다.

    `worst` 는 depth 판에서 "위반한 소스 프레임 수"였는데 mesh 에는 소스 프레임이라는 축이
    없다. 열 정의를 흐리지 않도록 **0/1** 로만 채운다 (진단은 `behind_*_frames` 로 본다).
    """
    poses = np.asarray(poses_blend, dtype=np.float64)
    centers = poses[:, :3, 3]
    num_plan = len(centers)
    num_mesh = len(mesh.dyn_edt)

    s_dist, s_reach = mesh.static_clearance(centers)
    static_bad = (~s_reach) | (s_dist < margin)

    dyn_bad = np.zeros(num_plan, dtype=bool)
    for f in range(num_plan):
        mf = int(round(f * (num_mesh - 1) / max(num_plan - 1, 1)))
        dyn_bad[f] = bool(mesh.dynamic_clearance(centers[f:f + 1], mf)[0] < margin)

    bad = int((static_bad | dyn_bad).sum())
    worst = int(bool(bad))
    if not detail:
        return bad, worst
    #    `mask` 는 `gates.behind_profile` 과 같은 규약 — `--collision_source both` 가 두 판의
    #    위반 프레임을 합집합으로 세는 데 쓴다.
    per_ch = {"static": {"frames": int(static_bad.sum()), "worst": int(static_bad.any()),
                         "mask": static_bad.tolist()},
              "dynamic": {"frames": int(dyn_bad.sum()), "worst": int(dyn_bad.any()),
                          "mask": dyn_bad.tolist()}}
    return bad, worst, per_ch


def resolve_mesh_grid(args, out_root: str) -> str:
    """`--collision_source` / `--mesh_grid` → 실제 격자 경로 (depth 경로면 `""`).

    두 굽기 스크립트(`sample_camera_bank.py` / `fit_hole_ladder.py`)가 **같은 규칙**으로
    격자를 찾아야 한다 — τ 뱅크와 사다리가 서로 다른 G1 으로 굽히면 사다리가 τ 뱅크에서
    이미 걸러진 변이를 다시 살려낸다. 그래서 경로 결정을 여기 한 곳에 둔다.

    `--mesh_grid` 가 비면 `<output_root>/<video>/mesh_grid.npz`. 없으면 **죽는다** —
    `mesh`/`both` 라고 해놓고 조용히 depth 로 떨어지면 뱅크 안에 두 규약이 섞이는데 열
    이름이 같아서 사후에 구분이 안 된다 (D105 의 교훈).
    """
    if getattr(args, "collision_source", "depth") == "depth":
        return ""
    grid = args.mesh_grid or path.join(out_root, args.video, "mesh_grid.npz")
    assert path.isfile(grid), (
        f"`--collision_source mesh` 인데 격자가 없다: {grid}\n"
        f"  먼저 `fit/ingest/trumans_export_mesh.py` → `python -m lbm.mesh_collision` 를 돌려라.")
    return grid


def main(args):
    src = np.load(args.src_poses)["cam_c2w"].astype(np.float64)
    build_grid(args.mesh_npz, src, args.out, voxel=args.voxel, pad=args.pad, clip=args.clip)


if __name__ == "__main__":
    parser = ArgumentParser(description="TRUMANS mesh 삼각형 → 점유/EDT 격자")
    parser.add_argument("--mesh_npz", required=True, type=str)      # trumans_export_mesh.py 산출
    parser.add_argument("--src_poses", required=True, type=str)     # poses_a<NN>.npz (blend world)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--voxel", default=0.05, type=float)        # 격자 한 칸 (m)
    parser.add_argument("--pad", default=1.0, type=float)           # AABB 여유 (m)
    parser.add_argument("--clip", default=3.0, type=float)          # 거리 상한 (m, float16 정밀도)
    main(parser.parse_args())
