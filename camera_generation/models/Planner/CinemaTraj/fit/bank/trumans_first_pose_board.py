"""TRUMANS `.blend` 에서 **chunk 별 시작 pose 후보판(board)** 을 만든다 — 게이트만, VLM 없이.

**왜 필요한가.** Lite 는 지금까지 DA3 좌표계를 기준으로 카메라를 놓았다. 그 좌표계는 첫 프레임이
항등인 건 맞지만 *그 프레임이 blender 씬의 어디인지*를 모른다 — 그래서 "카메라가 벽 속인가 /
사람이 보이는가"를 씬 기하로 물을 수가 없었다. LBM 원본은 이 문제를 **후보를 먼저 렌더해서 보고
고르는** 방식으로 풀었다. 이 스크립트는 그 순서만 가져오고 VLM 은 뺀다: chunk 마다 사람 주변
격자를 깔고, 가림·프레이밍·충돌·근접 게이트를 통과한 후보만 **카메라 파라미터 + 렌더 이미지**로
남긴다. 고르는 건 나중 단계(사람이든 VLM이든)의 몫이다.

**LBM 에서 가져온 것 / 안 가져온 것.**
- 가져옴: board-then-filter 순서, occlusion check (카메라 -> 피사체 광선), 후보를 방위각 x 고도 x
  거리 격자로 까는 방식.
- 안 가져옴: Director/Cinematographer/VideoEngineer 4단 핸드오프, VLM 질의, story/layout JSON.
  실행당 렌더가 ~20,000 장에서 chunk 당 ~40 장으로 줄어드는 게 그 차이다.

**게이트 4종** (전부 통과해야 `usable`). 탈락 사유는 후보마다 `reject` 에 남는다 — 전량 탈락이
곧 진단이다 (사람이 벽 구석에 있어서 어느 방향에서도 못 본다는 뜻).
1. `occlusion` — `line_of_sight`: 카메라에서 조준점들로 쏴서 **처음 맞는 게 subject** 인 광선이
   하나라도 있어야 한다. 반대 방향으로 쏘면 시작점이 몸 속이라 자기 mesh 를 즉시 때린다.
   chunk 의 시작·중앙·끝 세 프레임에서 전부 뚫려야 한다 (한 프레임만 보면 앞이 막힌 후보가
   뽑힌다 — action 10 실측 49프레임 중 앞 21장이 막힘).
2. `framing` — subject AABB 8꼭짓점을 near-plane 에서 자른 뒤 투영해 convex hull 을 만들고,
   ① `crop_keep` = 화면 안 hull 면적 / 전체 hull 면적 ② `area_frac` = 화면 안 hull 면적 / (W*H)
   ③ hull 중심이 중앙 safe frame 안. 절대 면적은 shot size 지 품질이 아니므로 **비율**로 잰다
   (`audit_lite_framing.py` 와 같은 정의, cv2 없이 numpy 로 재구현 — Blender 내장 python 에
   cv2 가 없다).
3. `collision` — `clearance_of` 6방향 최단 히트 + 바닥까지 거리 + 씬 AABB 안. 벽 속이면 6방향이
   전부 몇 cm 라 한 값으로 "벽 속"과 "벽에 너무 붙음"을 같이 잡는다.
4. `subject_dist` — 조준점까지 최단 거리 하한. `clearance_of` 가 subject 히트를 **면제**하므로
   (안 그러면 반경 1.5 m 후보가 전부 "벽에 붙었다"로 찍힌다) 이 게이트가 없으면 피사체를 뚫고
   들어가는 후보를 아무것도 못 막는다.

**방위각은 world 기준**이다 (subject 정면 기준이 아니라). `trumans_scene_probe.py` / orchestrator
의 `--anchor_cell` 과 같은 규약이라 이 board 에서 고른 칸을 그대로 넘길 수 있다. 리그의 어느
로컬 축이 정면인지 하드코딩하면 리그가 바뀔 때 조용히 틀린다는 게 probe 에서 이미 확인된 사실.

**반경은 `d_ref` 배수**다. `d_ref` = 사람이 프레임 높이에 30% 여유를 두고 딱 들어가는 거리
(= full shot). 렌즈와 사람 키에 자동으로 맞춰지므로 `--lens` 를 바꿔도 격자가 같은 뜻을 유지한다.
절대 미터로 박고 싶으면 `--radii` 로 덮어쓴다.

**주의: `--cycles` 로 시작하는 CLI 플래그를 절대 만들지 말 것** (Cycles 애드온이 argv 를
prefix-match 하며 훑어서 실행이 통째로 죽는다).

env: Blender 내장 python

예시:
    B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/00add26c-7a26-4a61-b192-b97aa493b3f3
    $B -b $R/00add26c-7a26-4a61-b192-b97aa493b3f3.blend --python fit/bank/trumans_first_pose_board.py -- \
        --chunk_starts 51 100 149 --num_frames 49 --frame_step 1 \
        --out out/board_00add26c
"""
import json
import sys
import time
from argparse import ArgumentParser
from math import atan2, cos, degrees, radians, sin
from os import makedirs, path

import bpy
import numpy as np
from mathutils import Matrix, Vector

RENDERABLE = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME", "GREASEPENCIL"}
# 6방향 clearance 광선. 벽 **속**이면 여섯 방향이 전부 몇 cm 안에서 히트하므로
# min(hit distance) 하나로 "벽 속"과 "벽에 너무 붙음"을 같은 눈금에서 잰다.
CLEARANCE_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
GL2CV = Matrix(((1, 0, 0, 0), (0, -1, 0, 0), (0, 0, -1, 0), (0, 0, 0, 1)))
# `aabb_corners` 의 부호 순서와 짝이 맞는 12개 변. near-plane 절단에 쓴다.
AABB_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def cli_argv():
    """Blender 는 `--` 뒤를 스크립트 몫으로 남긴다. `--` 가 없으면 인자 없음."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


# ---------------------------------------------------------------------------------------
# 씬 조회 (trumans_scene_probe.py 와 같은 정의 — 두 스크립트가 어긋나면 게이트가 조용히 달라진다)
# ---------------------------------------------------------------------------------------
def find_human(scene):
    """(armature, [사람 mesh], 몸통 mesh) — 이름 하드코딩 대신 armature modifier / parent 로 찾는다."""
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    assert armatures, "이 blend 에 ARMATURE 가 없다 — TRUMANS recording 이 맞는지 확인할 것."
    armature = armatures[0]
    meshes = [o for o in scene.objects if o.type == "MESH"
              and (o.parent is armature
                   or any(m.type == "ARMATURE" and m.object is armature for m in o.modifiers))]
    assert meshes, f"armature {armature.name} 가 deform 하는 mesh 를 못 찾았다."
    named = [m for m in meshes if "body" in m.name.lower()]
    body = named[0] if named else max(meshes, key=lambda m: len(m.data.vertices))
    return armature, meshes, body


def find_props(scene, names):
    """`obj_list.txt` 의 이름들 -> 실제 씬 오브젝트. 정확 일치 + 접두 일치 둘 다 받는다.

    `oven` 처럼 한 소품이 `oven_base_01` / `oven_door_01` 로 쪼개져 있어서 접두 일치가 필요하다.
    못 찾은 이름은 조용히 넘기지 않는다 — blend 와 obj_list 가 어긋난 것이므로 즉시 죽는다.
    """
    if not names:
        return []
    by_name = {o.name: o for o in scene.objects if o.type in RENDERABLE}
    found, missing = [], []
    for name in names:
        if name in by_name:
            found.append(by_name[name])
            continue
        prefixed = [o for n, o in sorted(by_name.items()) if n.startswith(name)]
        found.extend(prefixed) if prefixed else missing.append(name)
    assert not missing, (f"--prop_names 중 씬에 없는 이름: {missing}. "
                         f"recording 의 obj_list.txt 와 blend 가 맞는지 확인할 것.")
    unique, seen = [], set()
    for obj in found:
        if obj.name not in seen:
            seen.add(obj.name)
            unique.append(obj)
    return unique


def world_aabb(objects, depsgraph):
    """deform 이 적용된(evaluated) world AABB. `bound_box` 는 로컬이라 matrix_world 를 곱한다."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            point = np.asarray(evaluated.matrix_world @ Vector(corner))
            lo, hi = np.minimum(lo, point), np.maximum(hi, point)
    return lo, hi


def body_points(body_mesh, depsgraph):
    """몸통 위 조준점 3개(70/55/42% 높이). 각 높이띠 정점들의 **중앙값 xy** 를 쓴다.

    AABB 중심을 쓰면 안 되는 이유: 팔을 뻗으면 AABB 가 벌어져 중심이 몸통과 팔 사이 **빈 공간**에
    앉는다. 그 점을 조준하면 광선이 몸을 통과해 뒷벽을 때리고 "가려졌다"로 오판한다.
    루트 본도 안 된다 — 이 리그의 루트는 골반이 아니라 [0,0,-0.946] 이다.
    """
    evaluated = body_mesh.evaluated_get(depsgraph)
    matrix = evaluated.matrix_world
    vertices = np.asarray([matrix @ v.co for v in evaluated.data.vertices], dtype=np.float64)
    z_lo, z_hi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    span = max(1e-6, z_hi - z_lo)
    points = []
    for fraction in (0.70, 0.55, 0.42):
        band = vertices[np.abs(vertices[:, 2] - (z_lo + fraction * span)) < 0.06 * span]
        if len(band) < 8:
            band = vertices
        points.append([float(np.median(band[:, 0])), float(np.median(band[:, 1])),
                       float(z_lo + fraction * span)])
    return points


def prop_points(props, depsgraph):
    """소품 조준점 — 오브젝트마다 evaluated AABB 중심 1점.

    소품은 팔처럼 튀어나온 부위가 없어서 AABB 중심이 거의 항상 mesh **속**이고, 속에 있으면
    카메라 광선의 첫 히트가 그 소품이라 `line_of_sight` 가 바로 통과 판정한다.
    """
    points = []
    for obj in props:
        lo, hi = world_aabb([obj], depsgraph)
        points.append([float(v) for v in (lo + hi) / 2.0])
    return points


def ray(scene, depsgraph, origin, direction, distance):
    """(hit, 거리, 맞은 오브젝트 이름). `scene.ray_cast` 는 `distance` 안에서만 찾는다."""
    length = float(np.linalg.norm(direction))
    if length < 1e-9 or distance <= 0:
        return False, float("inf"), ""
    hit, location, _, _, obj, _ = scene.ray_cast(
        depsgraph, Vector(origin), Vector(np.asarray(direction) / length), distance=distance)
    if not hit:
        return False, float("inf"), ""
    # evaluated 사본이 오므로 `.original` 로 되돌려야 이름이 원본과 맞는다.
    name = getattr(getattr(obj, "original", obj), "name", "") if obj else ""
    return True, float(np.linalg.norm(np.asarray(location) - np.asarray(origin))), name


def line_of_sight(scene, depsgraph, position, targets, subject_names):
    """카메라 -> 조준점들. **처음 맞는 게 subject** 인 광선이 하나라도 있으면 시선이 뚫린 것.

    임계값 대신 맞은 오브젝트 **이름**으로 판정하므로 subject 크기에 안 휘둘린다 — 사람이든
    15 cm 짜리 컵이든 같은 코드가 그대로 돈다.
    """
    nearest = None
    for target in targets:
        direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        distance = float(np.linalg.norm(direction))
        hit, hit_at, name = ray(scene, depsgraph, position, direction, distance + 0.05)
        if hit and name in subject_names:
            return True, None
        if hit and (nearest is None or hit_at < nearest):
            nearest = hit_at
    return False, nearest


def clearance_of(scene, depsgraph, position, probe_distance, subject_names):
    """6방향 최단 히트 거리. 벽 속이면 작다. 히트가 없으면 probe_distance 로 포화시킨다.

    subject mesh 히트는 무시한다 — subject 는 충돌 위험물이 아니라 피사체이고, 카메라-subject
    거리는 `radius` 와 `--min_subject_dist` 가 통제한다. 안 빼면 반경 1.5 m 후보가 전부
    "벽에 붙었다"로 찍힌다.
    """
    best = float(probe_distance)
    for direction in CLEARANCE_DIRS:
        hit, distance, name = ray(scene, depsgraph, position, direction, probe_distance)
        if hit and name not in subject_names:
            best = min(best, distance)
    return best


# ---------------------------------------------------------------------------------------
# 프레이밍 (audit_lite_framing.py 와 같은 정의, cv2 없이 — Blender 내장 python 에 cv2 가 없다)
# ---------------------------------------------------------------------------------------
def aabb_corners(lo, hi):
    """(8,3). `AABB_EDGES` 와 부호 순서가 짝이 맞는다."""
    signs = np.array([[sx, sy, sz] for sz in (0, 1) for sy in (0, 1) for sx in (0, 1)], dtype=float)
    return lo + signs * (hi - lo)


def near_clipped_corners(corners, w2c, z_near):
    """박스를 `z_cam >= z_near` 반공간으로 자른 뒤 남는 꼭짓점들 (world).

    볼록 다면체 ∩ 반공간의 꼭짓점 = (조건을 만족하는 원래 꼭짓점) ∪ (경계를 지나는 edge 의 교점).
    이걸 안 하면 카메라 뒤 꼭짓점이 부호 반전돼 투영되어 hull 이 화면 전체를 덮는다.
    """
    z = corners @ w2c[2, :3] + w2c[2, 3]
    keep = [corners[i] for i in range(8) if z[i] >= z_near]
    for i, j in AABB_EDGES:
        if (z[i] >= z_near) != (z[j] >= z_near):
            alpha = (z_near - z[i]) / (z[j] - z[i])
            keep.append(corners[i] + alpha * (corners[j] - corners[i]))
    return np.asarray(keep, dtype=float) if keep else np.zeros((0, 3))


def convex_hull_2d(points):
    """Andrew monotone chain. 반시계 방향 볼록껍질 (cv2.convexHull 대체)."""
    if len(points) < 3:
        return np.asarray(points, dtype=float)
    order = np.lexsort((points[:, 1], points[:, 0]))
    ordered = points[order]

    def build(seq):
        stack = []
        for point in seq:
            while len(stack) >= 2:
                a, b = stack[-2], stack[-1]
                if (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]) > 1e-12:
                    break
                stack.pop()
            stack.append(point)
        return stack[:-1]

    hull = build(ordered) + build(ordered[::-1])
    return np.asarray(hull, dtype=float) if len(hull) >= 3 else ordered


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def clip_to_rect(poly, width, height):
    """Sutherland-Hodgman. 볼록 다각형 ∩ 이미지 사각형 — 면적을 **해석적으로** 남기려고.

    래스터로 세면 잘림이 안 보인다 (화면 밖 픽셀은 애초에 세어지지 않으므로).
    """
    output = poly
    for axis, limit, keep_greater in ((0, 0.0, True), (0, float(width), False),
                                      (1, 0.0, True), (1, float(height), False)):
        if len(output) == 0:
            return np.zeros((0, 2))
        inside = (output[:, axis] >= limit) if keep_greater else (output[:, axis] <= limit)
        clipped = []
        for i in range(len(output)):
            j = (i + 1) % len(output)
            if inside[i]:
                clipped.append(output[i])
            if inside[i] != inside[j]:
                denominator = output[j, axis] - output[i, axis]
                alpha = (limit - output[i, axis]) / (denominator if abs(denominator) > 1e-12 else 1e-12)
                clipped.append(output[i] + alpha * (output[j] - output[i]))
        output = np.asarray(clipped, dtype=float)
    return output


def framing_of(lo, hi, c2w_cv, K, width, height, z_near):
    """한 프레임의 (crop_keep, area_frac, center_offset, height_frac). 전부 **비율**.

    - `crop_keep`  = 화면 안 hull 면적 / 전체 hull 면적. 1.0 이면 subject 가 통째로 화면 안.
    - `area_frac`  = 화면 안 hull 면적 / (W*H). 너무 작으면 점, 너무 크면 클로즈업 과다.
    - `center_offset` = hull 무게중심의 화면 중앙 대비 [-1,1] 정규화 거리 (max(|dx|,|dy|)).
    - `height_frac` = **화면 밖까지 포함한** 투영 세로 길이 / 화면 높이. shot scale 눈금이다.
      화면 안으로 자르면 안 된다 — 클로즈업은 정의상 화면 밖으로 넘치므로 자르는 순간 full shot
      과 구분이 안 된다. 그래서 `area_frac`(잘린 뒤)과 달리 이 값은 1.0 을 넘을 수 있다.

    AABB 가 실루엣보다 크므로 `crop_keep` 은 잘림을 **과대**보고한다 (팔을 벌리면 특히).
    같은 이유로 `height_frac` 도 shot 을 실제보다 한 칸 타이트하게 읽는다. 순위용으로만 쓸 것.
    """
    w2c = np.linalg.inv(c2w_cv)
    corners = near_clipped_corners(aabb_corners(lo, hi), w2c, z_near)
    if len(corners) < 3:
        return 0.0, 0.0, 1.0, 0.0                  # 박스가 통째로 카메라 뒤
    cam = corners @ w2c[:3, :3].T + w2c[:3, 3]
    uv = np.stack([K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2],
                   K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]], axis=1)
    height_frac = float((uv[:, 1].max() - uv[:, 1].min()) / height)
    hull = convex_hull_2d(uv)
    total = polygon_area(hull)
    if total <= 1e-6:
        return 0.0, 0.0, 1.0, height_frac
    inside = clip_to_rect(hull, width, height)
    area = polygon_area(inside)
    if len(inside) < 3:
        return 0.0, 0.0, 1.0, height_frac          # 화면 안에 hull 이 없다
    center = inside.mean(axis=0)
    offset = max(abs(center[0] / width - 0.5), abs(center[1] / height - 0.5)) * 2.0
    return min(1.0, area / total), area / (width * height), float(offset), height_frac


# 투영 세로 길이(화면 높이 대비) -> shot scale 이름. 경계는 "사람 전신이 딱 들어가면 full" 을
# 기준으로 잡았다: full 이 1.0 언저리고, 허리 위만 보이면 대략 2배, 얼굴만이면 대략 5배.
SHOT_BANDS = [(0.25, "extreme_wide"), (0.50, "wide"), (0.90, "full"), (1.30, "medium_full"),
              (2.00, "medium"), (3.20, "medium_close"), (6.00, "close"), (1e9, "extreme_close")]
# subject 정면과 "subject -> 카메라" 수평 방향 **사이각의 크기** -> 앵글 이름. 0deg = 마주 본다.
# 크기만으로는 왼쪽 옆모습과 오른쪽 옆모습이 같은 `profile` 로 뭉치므로 좌우는 `_l`/`_r` 접미사로
# 따로 붙인다 (`facing_label`). front/back 은 좌우가 정의되지 않아 접미사가 없다.
FACING_BANDS = [(30.0, "front"), (70.0, "front_3q"), (110.0, "profile"),
                (150.0, "back_3q"), (180.1, "back")]
# 좌우 접미사를 붙이지 않는 밴드. 정면/정후면은 부호가 0deg / 180deg 근처에서 프레임마다
# 뒤집혀서 좌우를 붙이면 의미 없는 잡음이 라벨에 실린다.
FACING_UNSIDED = ("front", "back")
# 요약표 출력 순서 = 왼쪽 정면 -> 정면 -> 오른쪽 정면 -> ... 로 한 바퀴. tally 를 이 순서로 찍어야
# "한쪽으로 쏠렸다"가 표에서 보인다.
FACING_ORDER = ["back_3q_l", "profile_l", "front_3q_l", "front", "front_3q_r", "profile_r",
                "back_3q_r", "back", "front_3q", "profile", "back_3q", "unknown"]
# 카메라가 subject **눈높이** 대비 위/아래 몇 도인가 -> 촬영 각도 이름. 0deg = 눈높이.
#
# 격자의 `elevation_deg` 와 **다른 값이다**. 그건 subject AABB **중심**(가슴께) 기준이라 서 있는
# 사람을 elevation 10deg 로 잡아도 카메라는 눈높이보다 한참 아래고, 같은 elevation 이어도 반경이
# 크면 눈높이에 가까워진다. 즉 격자축을 라벨로 옮겨 적은 게 아니라 별개의 양이다.
CAMERA_ANGLE_BANDS = [(-25.0, "low_angle"), (-8.0, "slight_low"), (8.0, "eye_level"),
                      (30.0, "slight_high"), (60.0, "high_angle"), (90.1, "overhead")]
CAMERA_ANGLE_ORDER = [n for _, n in CAMERA_ANGLE_BANDS] + ["unknown"]
# 눈 메시가 없을 때만 쓰는 근사: 발끝에서 신장의 94% (성인 눈높이). 무릎 꿇기/앉기에서 깨진다.
EYE_FALLBACK_FRAC = 0.94


def band_label(value, bands):
    """오름차순 상한 리스트에서 첫 번째로 걸리는 이름. 경계값은 아래 칸에 들어간다."""
    for limit, name in bands:
        if value < limit:
            return name
    return bands[-1][1]


def facing_label(magnitude_deg, side_frac, min_side_frac):
    """사이각 크기 + 좌우 일치도 -> `profile_l` 같은 라벨.

    `side_frac` 은 "카메라가 subject 의 **왼쪽**에 있던 프레임 비율"이다 (0.5 가 반반).
    창 안에서 사람이 카메라 앞을 가로질러 가면 좌우가 실제로 바뀌므로, 한쪽이
    `min_side_frac` 을 못 넘으면 **접미사를 안 붙인다** — 틀린 좌우를 붙이느니 크기만 부른다.
    """
    name = band_label(magnitude_deg, FACING_BANDS)
    if name in FACING_UNSIDED:
        return name
    if side_frac >= min_side_frac:
        return f"{name}_l"
    if (1.0 - side_frac) >= min_side_frac:
        return f"{name}_r"
    return name


def bone_basis(armature, bone_name):
    """지정한 본의 world 3x3.

    **루트 본(`CC_Base_BoneRoot`)을 쓰면 안 된다.** 이 리그(Character Creator `CC_Base_*`,
    SMPL-X 아님)의 루트는 전 프레임 회전 항등 / 위치 `[0,0,-0.946]` 고정이라 몸이 어디를 보든
    안 움직인다. 실측: 루트로 정면을 풀면 보행 일치도 0.09 (= 신호 없음), 골반/척추 체인으로
    풀면 0.68. 그래서 기본은 `CC_Base_Hip` 이고, 없으면 루트로 떨어지되 일치도가 낮게 나와
    `resolve_forward_axis` 가 경고를 남긴다.
    """
    bone = armature.pose.bones.get(bone_name)
    if bone is None:
        roots = [b for b in armature.pose.bones if b.parent is None]
        bone = roots[0] if roots else None
    matrix = armature.matrix_world if bone is None else armature.matrix_world @ bone.matrix
    return np.array([[float(matrix[r][c]) for c in range(3)] for r in range(3)])


def eye_position(eye_meshes, subject_lo, subject_hi, depsgraph):
    """눈 world 위치.

    **키 비례로 추정하면 안 된다.** subject 가 무릎을 꿇거나 앉으면 "발끝에서 신장의 94%" 가
    통째로 틀린다 (실측 chunk f600 은 사람이 선반 앞에 무릎 꿇은 자세다). 리그에 눈 메시
    (`CC_Base_Eye`)가 실제로 있으므로 그 AABB 중심을 직접 읽고, 없을 때만 근사로 떨어진다.
    """
    if eye_meshes:
        lo, hi = world_aabb(eye_meshes, depsgraph)
        return (lo + hi) / 2.0
    point = (subject_lo + subject_hi) / 2.0
    point[2] = subject_lo[2] + EYE_FALLBACK_FRAC * (subject_hi[2] - subject_lo[2])
    return point


def horizontal(vector):
    """수평 성분 단위벡터. 세로 성분은 버린다 — facing 은 방위각 문제다."""
    flat = np.array([float(vector[0]), float(vector[1]), 0.0])
    norm = float(np.linalg.norm(flat))
    return flat / norm if norm > 1e-9 else None


def resolve_forward_axis(scene, armature, subject_meshes, bone_name, sample_step, min_speed,
                         min_agreement):
    """리그의 어느 로컬 축이 **정면**인지 걷는 구간에서 실측한다.

    왜 실측하나: 정면 축을 하드코딩하면 리그가 바뀔 때 조용히 틀린다 (`trumans_scene_probe.py`
    가 3축을 다 실어 보내고 orchestrator 가 교정하게 둔 것과 같은 이유). 여기서는 board 를
    뽑기 전에 한 번만 풀어서 recording 전체에 재사용한다.

    보행 프레임만 쓴다. 앉아서 책을 집는 chunk 만 보면 속도가 0 이라 아무 축이나 이겨버린다.
    반환하는 `agreement` 는 속도 가중 평균 cos. 실측 기준값: `CC_Base_Hip` 로 +0.68,
    정지 루트 본으로 0.09 — `min_agreement` 는 그 사이를 가른다. 사람은 늘 정면으로만 걷지
    않으므로(옆걸음·회전·제자리) 1.0 은 안 나온다.
    """
    frames = list(range(scene.frame_start, scene.frame_end + 1, sample_step))
    centers = []
    bases = []
    for frame in frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        lo, hi = world_aabb(subject_meshes, depsgraph)
        centers.append((lo + hi) / 2.0)
        bases.append(bone_basis(armature, bone_name))

    scores = np.zeros((3, 2))                      # [축][부호 0=+ 1=-]
    weight = 0.0
    moving = 0
    for i in range(len(frames) - 1):
        step = centers[i + 1] - centers[i]
        speed = float(np.linalg.norm(step[:2]))
        if speed < min_speed:
            continue
        direction = horizontal(step)
        if direction is None:
            continue
        moving += 1
        weight += speed
        for axis in range(3):
            local = horizontal(bases[i][:, axis])
            if local is None:                      # 그 축이 거의 수직이면 방위각 정보가 없다
                continue
            cosine = float(np.dot(local, direction))
            scores[axis, 0] += speed * cosine
            scores[axis, 1] -= speed * cosine
    if weight <= 0.0:
        return {"bone": bone_name, "axis": 1, "sign": 1.0, "agreement": 0.0, "moving_samples": 0,
                "resolved": False, "note": "보행 구간 없음 — 축을 못 풀었다"}
    axis, sign_index = np.unravel_index(int(np.argmax(scores)), scores.shape)
    agreement = float(scores[axis, sign_index] / weight)
    # 못 풀었으면 **조용히 넘어가면 안 된다**. facing 은 게이트가 아니라 태그라 틀려도
    # 파이프라인이 안 죽고, 그대로 두면 뒤통수 후보에 front 라벨이 붙은 board 가 나간다.
    return {"bone": bone_name, "axis": int(axis), "sign": 1.0 if sign_index == 0 else -1.0,
            "agreement": agreement, "moving_samples": moving,
            "resolved": bool(agreement >= min_agreement),
            "note": ("" if agreement >= min_agreement else
                     f"일치도 {agreement:.2f} < {min_agreement:.2f} — facing 태그를 믿지 말 것")}


# ---------------------------------------------------------------------------------------
# 카메라
# ---------------------------------------------------------------------------------------
def K_analytic(camd, scene):
    """Blender lens/sensor -> 픽셀 intrinsics. `BKE_camera_params_compute_viewplane()` 재현.

    `trumans_gt_render.py:139` 와 같은 식이다 (324 조합에서 `calc_matrix_camera()` 와 상대오차
    2.3e-7 로 일치 확인됨). 두 스크립트가 어긋나면 board 의 프레이밍과 실제 렌더가 달라진다.
    """
    render = scene.render
    scale = render.resolution_percentage / 100.0
    width, height = int(render.resolution_x * scale), int(render.resolution_y * scale)
    ax, ay = render.pixel_aspect_x, render.pixel_aspect_y
    ycor = ay / ax
    fit = camd.sensor_fit
    fit_hor = (ax * width) >= (ay * height) if fit == "AUTO" else (fit == "HORIZONTAL")
    sensor = camd.sensor_height if fit == "VERTICAL" else camd.sensor_width
    view_fac_px = width if fit_hor else ycor * height
    fx = camd.lens / sensor * view_fac_px
    return (np.array([[fx, 0.0, width * 0.5 - camd.shift_x * view_fac_px],
                      [0.0, fx / ycor, height * 0.5 + camd.shift_y * view_fac_px / ycor],
                      [0.0, 0.0, 1.0]]), width, height)


def new_camera(scene, lens, sensor):
    """`trumans_gt_render.py:251` 과 같은 설정 — sensor_fit AUTO, 3:2 센서, clip 0.05..1000."""
    camd = bpy.data.cameras.new("BoardCamData")
    camd.type = "PERSP"
    camd.lens = lens
    camd.sensor_fit = "AUTO"
    camd.sensor_width = sensor
    camd.sensor_height = sensor * 2.0 / 3.0
    camd.shift_x = camd.shift_y = 0.0
    camd.clip_start, camd.clip_end = 0.05, 1000.0
    cam_obj = bpy.data.objects.new("BoardCam", camd)
    scene.collection.objects.link(cam_obj)
    return cam_obj


def look_at_c2w(position, target, up=(0.0, 0.0, 1.0)):
    """OpenCV c2w (X right / Y down / Z forward). roll 은 `up` 에 대해 0.

    `trumans_to_recon.py:389` 와 같은 정의다. `up` 이 world Z 인 이유는 TRUMANS 가 Z-up 이고
    중력축이 곧 world Z 이기 때문 — DA3 씬처럼 기울어진 world 가 아니라 여기선 안전하다.
    """
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    forward /= max(1e-9, np.linalg.norm(forward))
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(forward, up))) > 0.999:       # 수직 내려다보기 — up 이 퇴화한다
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= max(1e-9, np.linalg.norm(right))
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, down, forward, position
    return c2w


# ---------------------------------------------------------------------------------------
def probe_chunk(scene, args, subject_meshes, subject_names, human_meshes, armature,
                body_mesh, props, frames, cam_obj, K, width, height, scene_lo, scene_hi,
                forward_axis):
    """한 chunk 의 격자 후보 전부를 게이트에 통과시킨다. 렌더는 하지 않는다 (호출자 몫)."""
    # anchor = 격자 원점·조준점·렌더가 모두 놓이는 프레임. `mid` 가 기존 동작이다.
    #
    # `start` 는 이 board 가 실제로 정하는 것에 맞춘 선택지다 — 우리가 뽑는 건 **첫 pose** 고
    # 나머지 48프레임은 궤적이 채운다. 그런데 `mid` 로 두면 카메라가 창 **중간** 위치의 사람을
    # 기준으로 놓이므로, 보행 chunk 에서는 "첫 프레임에 사람이 아직 프레임 밖" 인 후보가
    # 정상으로 통과할 수 있다 (반대로 첫 프레임 기준으론 멀쩡한데 중간에서 어긋나 죽기도 한다).
    anchor = frames[len(frames) // 2] if args.anchor_frame == "mid" else frames[0]
    # 게이트를 창 전체의 **최악값**으로 볼지(`window`, 기존) anchor 한 장만 볼지(`anchor`).
    #
    # `window` 는 사실상 "정지 카메라로 49프레임을 다 찍는다"는 요구다. 창 안에서 사람이
    # 1.9 m 넘게 걷는 chunk 는 그래서 통과가 구조적으로 0 이 된다 (실측 900/1200/1800).
    # 카메라가 궤적으로 따라갈 거라면 그 요구는 첫 pose 에 물릴 게 아니다.
    probe_frames = (sorted({frames[0], anchor, frames[-1]}) if args.gate_frames == "window"
                    else [anchor])

    # 프레임별 subject AABB / 조준점. anchor 프레임 한 장만 보면 사람이 움직이는 구간에서
    # 시작엔 가려져 있다가 중간부터 보이는 후보가 뽑힌다.
    # 눈 메시는 subject 가 아니라 **사람** 쪽에서 찾는다. subject 가 소품이어도 촬영 각도의
    # 기준은 사람 눈높이다 (`human_height` 를 사람으로 두는 것과 같은 이유).
    eye_meshes = [m for m in human_meshes if m.name in set(args.eye_mesh)]
    track = {}
    for frame in probe_frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        lo, hi = world_aabb(subject_meshes, depsgraph)
        human_bounds = (world_aabb(human_meshes, depsgraph) if not eye_meshes else (lo, hi))
        eye = eye_position(eye_meshes, human_bounds[0], human_bounds[1], depsgraph)
        body = body_points(body_mesh, depsgraph)
        prop = prop_points(props, depsgraph) if props else []
        aims = (body if args.subject_kind == "human" else
                prop if args.subject_kind == "object" else body + prop)
        # 정면 벡터는 프레임마다 다시 읽는다 — 145프레임 창에서 사람이 돌아앉으면 anchor 한
        # 장으로 잰 facing 이 시작/끝에서 반대가 된다.
        facing = None if not forward_axis["resolved"] else horizontal(
            bone_basis(armature, forward_axis["bone"])[:, forward_axis["axis"]]
            * forward_axis["sign"])
        track[frame] = {"lo": lo, "hi": hi, "aims": [np.asarray(p) for p in aims],
                        "facing": facing, "eye": eye}

    scene.frame_set(anchor)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    human_lo, human_hi = world_aabb(human_meshes, depsgraph)
    # 바닥은 **항상 사람 발** 기준. subject 가 소품이면 그 AABB 최저점은 책상 상판이라
    # 거기서 아래로 쏘면 floor_z 가 책상 높이로 잡히고 "카메라가 바닥 밑" 게이트가 무의미해진다.
    hit, drop, _ = ray(scene, depsgraph,
                       [(human_lo[0] + human_hi[0]) * 0.5, (human_lo[1] + human_hi[1]) * 0.5,
                        human_lo[2] + 0.30], (0, 0, -1), 5.0)
    floor_z = float(human_lo[2] + 0.30 - drop) if hit else float(human_lo[2])
    # 사람 키 — 카메라 고도와 look-at 높이의 눈금. subject 가 15 cm 짜리 컵이어도 카메라 고도
    # 눈금까지 15 cm 로 줄이면 안 되므로 이 값은 **사람**으로 둔다.
    human_height = float(human_hi[2] - human_lo[2])
    subject_lo, subject_hi = track[anchor]["lo"], track[anchor]["hi"]
    subject_height = float(subject_hi[2] - subject_lo[2])
    obb_center = (subject_lo + subject_hi) / 2.0
    chest = np.asarray(track[anchor]["aims"][0]) if args.anchor_origin == "chest" else obb_center

    # d_ref = 사람이 프레임 높이에 `--fit_margin` 여유를 두고 딱 들어가는 거리 (= full shot).
    # 렌즈/해상도/사람 키에 자동으로 맞춰지므로 `--lens` 를 바꿔도 격자가 같은 뜻을 유지한다.
    #
    # `aim_bias` 를 반드시 넣어야 한다. 조준점이 0.5*h 가 아니라 (0.5+bias)*h 높이라
    # 화면 중앙에서 subject **아래쪽 끝**까지가 (0.5+bias)*h 로 더 멀고, 그쪽이 먼저 잘린다.
    # 빼먹으면(= bias 0 가정) d_ref 가 1.4배 작게 나와서 격자 전체가 "발이 잘리는" 거리에
    # 앉는다 (실측: a00 chunk 51 에서 0.7*d_ref 칸 36개가 **전부** cropped 로 탈락).
    reach = (0.5 + abs(args.aim_bias)) * human_height       # 화면 중앙 -> subject 먼 쪽 끝
    d_ref = float(2.0 * args.fit_margin * K[1, 1] * reach / height)
    # 기본은 **절대 미터** (`--radii`). `--radii_rel` 을 주면 그때만 d_ref 배수 격자로 바뀐다.
    # d_ref 는 절대 격자에서도 계속 계산·기록한다 — 후보마다 `radius_rel` 로 실려서 "이게 몇
    # 배 거리냐"를 여전히 읽을 수 있어야 한다.
    radii = ([float(rel) * d_ref for rel in args.radii_rel] if args.radii_rel
             else [float(r) for r in args.radii])
    # 조준점은 OBB 중심(0.50*h)보다 `--aim_bias` 만큼 위 — 중심을 겨누면 머리가 잘린다.
    aim_target = obb_center + np.array([0.0, 0.0, args.aim_bias * human_height])

    rows = []
    for azimuth in np.arange(0.0, 360.0, float(args.az_step)):
        for elevation in args.elevations:
            for radius in radii:
                phi, theta = radians(float(azimuth)), radians(float(elevation))
                offset = np.array([cos(phi) * cos(theta), sin(phi) * cos(theta),
                                   sin(theta)]) * radius
                position = chest + offset
                c2w_cv = look_at_c2w(position, aim_target)
                row = {
                    "azimuth_deg": float(azimuth), "elevation_deg": float(elevation),
                    "radius": float(radius), "radius_rel": float(radius / max(1e-9, d_ref)),
                    "position": [float(v) for v in position],
                    "look_at": [float(v) for v in aim_target],
                }
                reject = []

                # --- collision: 씬 AABB 안 / 바닥 위 ---------------------------------------
                inside_scene = bool(np.all(position >= scene_lo - args.scene_margin)
                                    and np.all(position <= scene_hi + args.scene_margin))
                row["height_over_floor"] = float(position[2] - floor_z)
                if not inside_scene:
                    reject.append("outside_scene")
                if row["height_over_floor"] < args.min_height:
                    reject.append("below_floor")

                # 씬 밖/바닥 밑이면 광선 6+3발을 아낀다 (108칸 x chunk 수라 누적이 크다).
                if reject:
                    row["reject"] = reject
                    row["usable"] = False
                    rows.append(row)
                    continue

                clear_flags, clearances, drops, dists = [], [], [], []
                crops, areas, offsets, heights, facings, angles = [], [], [], [], [], []
                for frame in probe_frames:
                    scene.frame_set(frame)
                    depsgraph = bpy.context.evaluated_depsgraph_get()
                    entry = track[frame]
                    clear, _ = line_of_sight(scene, depsgraph, position, entry["aims"],
                                             subject_names)
                    clear_flags.append(bool(clear))
                    clearances.append(clearance_of(scene, depsgraph, position,
                                                   float(args.probe_distance), subject_names))
                    drops.append(ray(scene, depsgraph, position, (0, 0, -1), 10.0)[1])
                    dists.append(min(float(np.linalg.norm(a - position)) for a in entry["aims"]))
                    crop, area, center_off, height_frac = framing_of(
                        entry["lo"], entry["hi"], c2w_cv, K, width, height,
                        cam_obj.data.clip_start)
                    crops.append(crop)
                    areas.append(area)
                    offsets.append(center_off)
                    heights.append(height_frac)
                    # facing 은 카메라가 subject 를 **어느 쪽에서** 보는가다. subject 정면과
                    # "subject -> 카메라" 수평 방향의 사이각 — 0deg 면 얼굴을 마주 본다.
                    #
                    # 부호까지 받는다(-180..180). 크기만 재면 왼쪽 옆모습과 오른쪽 옆모습이
                    # 똑같이 `profile` 110deg 로 찍혀서 구분이 안 된다. 씬은 z-up 이고
                    # `horizontal()` 이 z 를 이미 0 으로 눌렀으므로 외적의 z 성분만 보면 된다.
                    # **부호 약속: + 는 카메라가 subject 의 왼쪽**(정면에서 반시계, 위에서 볼 때).
                    to_camera = horizontal(position - (entry["lo"] + entry["hi"]) / 2.0)
                    if entry["facing"] is not None and to_camera is not None:
                        front = entry["facing"]
                        cross_z = float(front[0] * to_camera[1] - front[1] * to_camera[0])
                        facings.append(degrees(atan2(cross_z,
                                                     float(np.dot(front, to_camera)))))
                    # 촬영 각도는 **눈높이 기준**. 카메라가 눈보다 위면 +(내려다봄).
                    # 격자의 `elevation_deg`(= subject 중심 기준)와 같은 값이 아니다.
                    delta = position - entry["eye"]
                    angles.append(degrees(atan2(float(delta[2]),
                                                float(np.linalg.norm(delta[:2])))))

                side_frac = (float(np.mean([f > 0.0 for f in facings])) if facings else 0.5)
                row.update({
                    "clear_frac": float(np.mean(clear_flags)),
                    "clearance": float(min(clearances)),
                    "floor_drop": float(min(drops)),
                    "subject_dist": float(min(dists)),
                    "crop_keep": float(min(crops)),
                    "area_frac": float(np.median(areas)),
                    "center_offset": float(max(offsets)),
                    # shot scale 은 **가장 타이트한** 프레임으로 부른다. 창 안에서 한 번이라도
                    # 얼굴이 화면을 채우면 그 chunk 는 클로즈업이지 full shot 이 아니다.
                    "height_frac": float(max(heights)),
                    "shot": band_label(float(max(heights)), SHOT_BANDS),
                    # facing 크기는 중앙값. 걷는 chunk 는 앞뒤로 90deg 넘게 흔들려서 min/max 로
                    # 부르면 이름이 창 끝단 한 프레임에 끌려간다. 흔들림은 range 로 따로 낸다.
                    #
                    # **부호는 median 으로 뽑으면 안 된다.** 정후면 근처에서 프레임 값이
                    # +178 / -178 을 오가면 중앙값이 0(= 정면)으로 떨어진다. 그래서 크기는
                    # `|각도|` 의 median 으로, 좌우는 **부호 다수결**(`facing_side_frac`)로
                    # 따로 구하고 마지막에 합친다.
                    "facing_deg": float(np.median(np.abs(facings)) *
                                        (1.0 if side_frac >= 0.5 else -1.0)) if facings else None,
                    "facing_range_deg": (float(np.max(np.abs(facings)) - np.min(np.abs(facings)))
                                         if facings else None),
                    # 카메라가 subject 왼쪽에 있던 프레임 비율. 0.5 근처면 창 안에서 사람이
                    # 카메라 앞을 가로질렀다는 뜻이고, 그때는 라벨에서 좌우를 뗀다.
                    "facing_side_frac": float(side_frac) if facings else None,
                    "facing": (facing_label(float(np.median(np.abs(facings))), side_frac,
                                            args.facing_min_side_frac)
                               if facings else "unknown"),
                    # 촬영 각도도 중앙값. 사람이 창 안에서 일어서면 눈이 올라와 각도가 내려간다.
                    "eye_angle_deg": float(np.median(angles)) if angles else None,
                    "eye_angle_range_deg": (float(max(angles) - min(angles)) if angles else None),
                    "camera_angle": (band_label(float(np.median(angles)), CAMERA_ANGLE_BANDS)
                                     if angles else "unknown"),
                })
                if row["clear_frac"] < 1.0:
                    reject.append("occluded")
                if row["clearance"] < args.min_clearance:
                    reject.append("clearance")
                if row["subject_dist"] < args.min_subject_dist:
                    reject.append("too_close")
                if row["crop_keep"] < args.min_crop_keep:
                    reject.append("cropped")
                if not (args.min_area_frac <= row["area_frac"] <= args.max_area_frac):
                    reject.append("area")
                if row["center_offset"] > args.max_center_offset:
                    reject.append("off_center")
                row["reject"] = reject
                row["usable"] = not reject
                rows.append(row)

    return {
        "frames": frames, "anchor_frame": anchor, "probe_frames": probe_frames,
        "floor_z": floor_z, "human_height": human_height, "subject_height": subject_height,
        "d_ref": d_ref, "radii": radii,
        "obb_center": [float(v) for v in obb_center],
        "anchor_origin_point": [float(v) for v in chest],
        "aim_target": [float(v) for v in aim_target],
        "subject_aabb": {"min": [float(v) for v in subject_lo],
                         "max": [float(v) for v in subject_hi]},
        "candidates": rows,
    }


def render_one(scene, cam_obj, K, candidate, out_dir, tag):
    """후보 하나를 anchor 프레임에서 렌더하고 pose/K 를 후보에 심는다."""
    c2w_cv = look_at_c2w(candidate["position"], candidate["look_at"])
    cam_obj.matrix_world = Matrix([list(row) for row in c2w_cv]) @ GL2CV
    bpy.context.view_layer.update()
    name = (f"{tag}_az{int(round(candidate['azimuth_deg'])):03d}"
            f"_el{int(round(candidate['elevation_deg'])):02d}"
            f"_r{candidate['radius']:.2f}")
    scene.render.filepath = path.join(out_dir, name)
    bpy.ops.render.render(write_still=True)
    candidate["image"] = path.join(path.basename(out_dir), name + ".png")
    candidate["cell_id"] = name
    candidate["c2w_opencv"] = [[float(v) for v in r] for r in c2w_cv]
    candidate["c2w_blender_gl"] = [list(r) for r in cam_obj.matrix_world]
    candidate["K"] = [[float(v) for v in r] for r in K]
    candidate["lens_mm"] = float(cam_obj.data.lens)


def pick_reject_samples(candidates, per_reason):
    """탈락 사유마다 **대표 후보 몇 개**를 고른다 (`--render_rejects`).

    통과분만 렌더하면 board 가 "왜 떨어졌는지"를 숫자로만 말한다. 통과 0장인 chunk 에서는
    아예 아무 그림도 안 나와서 원인을 못 본다. 그래서 사유별로 표본을 뽑아 같이 렌더한다.

    사유가 **적게 겹치는 후보**를 먼저 고르는 게 핵심이다. `occluded + cropped + clearance` 가
    한꺼번에 걸린 칸을 보여주면 어느 게 진짜 원인인지 그림에서 못 가린다. 사유 1개짜리가
    있으면 그게 그 게이트의 순수한 반례다.

    `below_floor`/`outside_scene` 은 광선을 쏘기 전에 잘려서 `clear_frac` 같은 필드가 아예
    없다 — 그래도 렌더는 된다 (카메라가 바닥 밑이라는 걸 보여주는 게 목적).
    """
    rejected = [c for c in candidates if not c["usable"]]
    reasons = sorted({r for c in rejected for r in c["reject"]})
    picked, seen = [], set()
    for reason in reasons:
        pool = [c for c in rejected if reason in c["reject"]]
        pool.sort(key=lambda c: (len(c["reject"]), -c.get("crop_keep", 0.0)))
        for candidate in pool[:per_reason]:
            key = id(candidate)
            if key not in seen:
                seen.add(key)
                candidate["reject_primary"] = reason
                picked.append(candidate)
    return picked


def pick_diverse(usable, max_render):
    """`(shot, camera_angle)` 셀 라운드로빈으로 `max_render` 장을 고른다 (`--render_select diverse`).

    `crop` 정렬(= `-crop_keep`)만으로 자르면 **멀고 높은 칸이 구조적으로 1등**이다. `crop_keep` 은
    "몸이 프레임에 얼마나 들어왔나"라서 반경이 크면 다 들어오고 고도가 높으면 내려다봐서 더
    들어온다. 실측(blend `00add26c…`, 14 chunk, 통과 574 -> 렌더 168): `full` 20.7% -> 60.7%,
    `medium` 33.8% -> 2.4%, `medium_close` 11.7% -> **0%(0/67)**, `slight_low` 21.1% -> 3.6%.
    통과가 많은 chunk 일수록 심해서 c00(52장 통과)은 12장이 전부 `full` 이고 방위각만 다른
    사실상 같은 그림이었다.

    그래서 셀을 먼저 나누고 **큰 셀부터 한 장씩** 가져간다. 셀 **안에서는** 기존 기준을 그대로
    쓴다 — 다양성은 셀 사이에서만 사고, 셀 안에서는 여전히 잘 잡힌 칸을 고른다.

    셀 순서를 `-len(cell)` 로 두는 이유: 12장을 다 못 채우면 큰 셀이 더 가져가게 되는데, 그게
    통과분 분포에 가까운 쪽이다. 같은 크기면 `shot`/`angle` 밴드 순으로 고정해 실행 간 재현된다.
    """
    cells = {}
    for candidate in usable:
        cells.setdefault((candidate["shot"], candidate["camera_angle"]), []).append(candidate)
    for pool in cells.values():
        pool.sort(key=lambda c: (-c["crop_keep"], c["center_offset"]))
    shot_order = [n for _, n in SHOT_BANDS]
    order = sorted(cells, key=lambda k: (-len(cells[k]), shot_order.index(k[0]),
                                         CAMERA_ANGLE_ORDER.index(k[1])))
    picked, i = [], 0
    while len(picked) < max_render and any(cells[k] for k in order):
        pool = cells[order[i % len(order)]]
        if pool:
            picked.append(pool.pop(0))
        i += 1
    return picked


def render_candidates(scene, cam_obj, K, chunk, out_dir, tag, max_render, render_rejects=0,
                      render_select="crop"):
    """usable 후보를 anchor 프레임에서 렌더한다. 렌더한 후보에 `image` 경로를 심는다.

    정렬 기준은 `crop_keep` 내림차순 -> `center_offset` 오름차순이다. `--max_render` 로 자를 때
    잘려 나가는 게 **프레이밍이 나쁜 쪽**이 되도록. 단 이 정렬은 shot/angle 을 편향시킨다 —
    `render_select="diverse"` 면 `pick_diverse` 가 셀 라운드로빈으로 고른다 (거기 주석 참고).

    `render_rejects > 0` 이면 탈락분에서도 사유별 표본을 렌더한다 (`pick_reject_samples`).
    """
    usable = [c for c in chunk["candidates"] if c["usable"]]
    usable.sort(key=lambda c: (-c["crop_keep"], c["center_offset"]))
    dropped = max(0, len(usable) - max_render) if max_render > 0 else 0
    if dropped:
        # 잘라낸 몫은 반드시 남긴다 — 조용히 자르면 "전부 렌더했다"로 읽힌다.
        chunk["render_truncated"] = dropped
    if max_render > 0:
        usable = (pick_diverse(usable, max_render) if render_select == "diverse"
                  else usable[:max_render])
    samples = (pick_reject_samples(chunk["candidates"], render_rejects)
               if render_rejects > 0 else [])
    chunk["reject_rendered"] = len(samples)

    scene.frame_set(chunk["anchor_frame"])
    makedirs(out_dir, exist_ok=True)
    for candidate in usable + samples:
        render_one(scene, cam_obj, K, candidate, out_dir, tag)
    return usable + samples


def main(args):
    started = time.time()
    scene = bpy.context.scene
    scene.frame_step = 1                       # TRUMANS blend 는 2 로 저장되어 있다
    armature, human_meshes, body_mesh = find_human(scene)
    props = find_props(scene, args.prop_names)

    # subject 를 사람 전용에서 일반화한다. kind 분기를 여기 한 곳에 모아 둬야 게이트가 조용히
    # 어긋나지 않는다 (`trumans_scene_probe.py` 와 같은 규약).
    if args.subject_kind == "human":
        subject_meshes = list(human_meshes)
    elif args.subject_kind == "event":
        assert props, "--subject_kind event 인데 --prop_names 가 비었다 (그러면 human 과 같다)"
        subject_meshes = list(human_meshes) + props
    else:
        assert props, "--subject_kind object 인데 --prop_names 가 비었다"
        subject_meshes = list(props)
    subject_names = {o.name for o in subject_meshes}
    if args.subject_kind != "object":
        subject_names.add(armature.name)       # armature 는 안 그려지지만 히트 이름으론 나온다
    assert args.anchor_origin != "chest" or args.subject_kind == "human", \
        "--anchor_origin chest 는 human 전용이다 (기존 뱅크 96편 재현용). event/object 는 obb_center."

    # --- 렌더 설정 (게이트의 K 와 실제 렌더가 같은 카메라여야 한다) ---------------------------
    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True    # 프레임 간 재사용 -> 큰 속도 이득
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = args.samples
    cam_obj = new_camera(scene, args.lens, args.sensor)
    scene.camera = cam_obj
    K, width, height = K_analytic(cam_obj.data, scene)

    # `--chunk_stride` 를 주면 recording 전체를 그 간격으로 훑는다 (`--chunk_starts` 대신).
    # 66편 sweep 을 위한 것 — 편마다 프레임 수가 달라서 start 목록을 밖에서 만들려면 blend 를
    # 한 번 더 열어야 한다(편당 ~4 s). span 은 `num_frames` 가 아니라 `(num_frames-1)*step+1`
    # 이다. 마지막 chunk 가 `frame_end` 를 넘지 않는 데까지만 낸다.
    if args.chunk_stride > 0:
        span = (args.num_frames - 1) * args.frame_step + 1
        args.chunk_starts = list(range(scene.frame_start, scene.frame_end - span + 2,
                                       args.chunk_stride))
        assert args.chunk_starts, (f"chunk 0개: frames {scene.frame_start}..{scene.frame_end} "
                                   f"< span {span}")
    assert args.chunk_starts, "--chunk_starts 또는 --chunk_stride 중 하나는 있어야 한다"

    # 정적 씬 AABB — 사람/소품은 프레임마다 움직이므로 뺀다.
    scene.frame_set(int(args.chunk_starts[0]))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    moving = set(human_meshes) | set(props)
    scene_lo, scene_hi = world_aabb(
        [o for o in scene.objects if o.type in RENDERABLE and o not in moving and o.visible_get()],
        depsgraph)

    # 정면 축은 recording 당 한 번만 푼다 (chunk 마다 풀면 앉아 있는 chunk 에서 엉뚱한 축이
    # 이긴다). 사람이 아닌 subject 는 "정면"이 정의되지 않으므로 armature 기준 사람으로 푼다.
    t0 = time.time()
    forward_axis = resolve_forward_axis(scene, armature, human_meshes, args.facing_bone,
                                        args.facing_sample_step, args.facing_min_speed,
                                        args.facing_min_agreement)
    forward_axis["time_s"] = time.time() - t0

    makedirs(args.out, exist_ok=True)
    chunks = []
    for order, start in enumerate(args.chunk_starts):
        frames = [int(start) + i * args.frame_step for i in range(args.num_frames)]
        tag = f"c{order:02d}_f{frames[0]:05d}"
        t0 = time.time()
        chunk = probe_chunk(scene, args, subject_meshes, subject_names, human_meshes, armature,
                            body_mesh, props, frames, cam_obj, K, width, height,
                            scene_lo, scene_hi, forward_axis)
        chunk["chunk"] = order
        chunk["tag"] = tag
        chunk["time_probe_s"] = time.time() - t0
        t0 = time.time()
        rendered = ([] if args.no_render else
                    render_candidates(scene, cam_obj, K, chunk,
                                      path.join(args.out, "renders"), tag, args.max_render,
                                      args.render_rejects, args.render_select))
        chunk["time_render_s"] = time.time() - t0
        chunk["n_rendered"] = len(rendered)
        chunks.append(chunk)

    result = {
        "format": "trumans_first_pose_board_v1",
        "blend": bpy.data.filepath,
        "fps": float(scene.render.fps / max(1, scene.render.fps_base)),
        "num_frames": args.num_frames, "frame_step": args.frame_step,
        "chunk_starts": [int(f) for f in args.chunk_starts],
        # 0 이 아니면 위 목록이 `scene.frame_start..frame_end` 에서 자동 생성된 것이다.
        "chunk_stride": int(args.chunk_stride),
        "frame_range": [int(scene.frame_start), int(scene.frame_end)],
        "armature": armature.name,
        "human_meshes": sorted(o.name for o in human_meshes),
        "subject_kind": args.subject_kind, "subject_meshes": sorted(subject_names),
        "prop_names": list(args.prop_names), "prop_meshes": [o.name for o in props],
        "anchor_origin": args.anchor_origin, "aim_bias": args.aim_bias,
        # 게이트가 창 전체(`window`)인지 anchor 한 장(`anchor`)인지. 이걸 안 적어두면
        # 두 board 의 usable 수를 그냥 비교하게 된다 — 요구 자체가 다르다.
        "anchor_frame": args.anchor_frame, "gate_frames": args.gate_frames,
        "forward_axis": forward_axis,
        "shot_bands": [[limit, name] for limit, name in SHOT_BANDS],
        "facing_bands": [[limit, name] for limit, name in FACING_BANDS],
        # `_l`/`_r` = 카메라가 subject 의 왼쪽/오른쪽. front/back 은 좌우가 없다.
        "facing_order": list(FACING_ORDER),
        "facing_min_side_frac": float(args.facing_min_side_frac),
        # 촬영 각도는 subject 중심(`elevation_deg`)이 아니라 **눈높이** 기준이다.
        "camera_angle_bands": [[limit, name] for limit, name in CAMERA_ANGLE_BANDS],
        "eye_mesh": list(args.eye_mesh),
        "lens_mm": args.lens, "sensor_mm": args.sensor, "res": list(args.res),
        "K": [[float(v) for v in r] for r in K],
        "scene_aabb": {"min": [float(v) for v in scene_lo], "max": [float(v) for v in scene_hi]},
        "gates": {"min_clearance": args.min_clearance, "min_subject_dist": args.min_subject_dist,
                  "min_crop_keep": args.min_crop_keep, "min_area_frac": args.min_area_frac,
                  "max_area_frac": args.max_area_frac,
                  "max_center_offset": args.max_center_offset,
                  "min_height": args.min_height, "scene_margin": args.scene_margin,
                  "probe_distance": args.probe_distance},
        # 0 이 아니면 렌더에 **탈락분 표본이 섞여 있다**. `usable:false` + `reject_primary` 로
        # 구분한다 — 안 적어두면 하류가 board 의 그림을 전부 통과분으로 읽는다.
        "render_rejects": int(args.render_rejects),
        # 렌더된 12장이 통과분의 **표본**인지(`diverse`) `crop_keep` 상위인지(`crop`). 하류가
        # 렌더분으로 태그 분포를 재면 이 값에 따라 답이 3배씩 달라진다 (`pick_diverse` 주석).
        "render_select": args.render_select,
        "grid": {"az_step": args.az_step, "elevations": list(args.elevations),
                 "radii": list(args.radii), "radii_rel": list(args.radii_rel),
                 "fit_margin": args.fit_margin},
        "chunks": chunks,
        "time_total_s": time.time() - started,
    }
    out_json = path.join(args.out, "board.json")
    with open(out_json, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=1)

    print(f"{'blend':22s} {bpy.data.filepath}")
    print(f"{'subject':22s} {args.subject_kind}  members {len(subject_meshes)}  "
          f"props {[o.name for o in props]}")
    print(f"{'camera':22s} lens {args.lens:g} mm  {width}x{height}  fx {K[0][0]:.1f}")
    print(f"{'forward axis':22s} {forward_axis['bone']} "
          f"{'xyz'[forward_axis['axis']]}{'+-'[forward_axis['sign'] < 0]}  "
          f"agreement {forward_axis['agreement']:.2f}  "
          f"moving samples {forward_axis['moving_samples']}  "
          f"{forward_axis['time_s']:.1f} s  {forward_axis['note']}")
    print(f"{'grid':22s} az {int(360 / args.az_step)} x el {len(args.elevations)} x "
          f"r {len(result['chunks'][0]['radii'])} = "
          f"{len(result['chunks'][0]['candidates'])}")
    print(f"{'chunk':>6s} {'frames':>13s} {'anchor':>7s} {'d_ref':>6s} {'usable':>7s} "
          f"{'render':>7s} {'probe_s':>8s} {'render_s':>9s}   top reject")
    for chunk in chunks:
        counts = {}
        for candidate in chunk["candidates"]:
            for reason in candidate["reject"]:
                counts[reason] = counts.get(reason, 0) + 1
        top = "  ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4])
        usable = sum(1 for c in chunk["candidates"] if c["usable"])
        print(f"{chunk['chunk']:6d} {chunk['frames'][0]:5d}..{chunk['frames'][-1]:<6d} "
              f"{chunk['anchor_frame']:7d} {chunk['d_ref']:6.2f} "
              f"{usable:3d}/{len(chunk['candidates']):<3d} {chunk['n_rendered']:7d} "
              f"{chunk['time_probe_s']:8.1f} {chunk['time_render_s']:9.1f}   {top}")
        # 통과분이 어떤 shot / 어떤 앵글로 쏠렸는지. 게이트는 통과했는데 전부 back 이면
        # 숫자만 봐서는 "40장 확보"로 읽히지만 실제로는 뒤통수 40장이다.
        good = [c for c in chunk["candidates"] if c["usable"]]
        for key, order in (("shot", [n for _, n in SHOT_BANDS]),
                           ("facing", FACING_ORDER),
                           ("camera_angle", CAMERA_ANGLE_ORDER)):
            tally = {}
            for candidate in good:
                tally[candidate[key]] = tally.get(candidate[key], 0) + 1
            spread = "  ".join(f"{n} {tally[n]}" for n in order if n in tally)
            print(f"{'':22s}{key:>8s}: {spread}")
    print(f"{'->':22s} {out_json}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", required=True, type=str)                  # 결과 폴더
    # chunk 시작 프레임들. 한 번의 Blender 기동으로 전부 처리한다 (1.6 GB blend 로드가 ~4 s 라
    # chunk 하나씩 물어보면 그 4 s 가 chunk 수만큼 쌓인다).
    parser.add_argument("--chunk_starts", default=[], nargs="+", type=int)
    # 0 이 아니면 `--chunk_starts` 를 무시하고 recording 전체를 이 간격으로 훑는다. 겹치지 않게
    # 하려면 `(num_frames-1)*frame_step+1` (49x3 이면 145) 로 준다. blend 를 두 번 열지 않으려고
    # 프레임 범위를 스크립트 안에서 읽는다 — 편마다 길이가 다르다.
    parser.add_argument("--chunk_stride", default=0, type=int)
    parser.add_argument("--num_frames", default=49, type=int)              # chunk 길이 (Lite 고정값)
    # 모션 프레임 간격. TRUMANS 모션은 30 Hz 라 3 이면 10 fps 샘플, 49장이 4.83초를 덮는다.
    parser.add_argument("--frame_step", default=1, type=int)

    # --- 격자 -----------------------------------------------------------------------------
    parser.add_argument("--az_step", default=30.0, type=float)             # 방위각 간격(도) -> 12방위
    # 고도는 **구 중심(= `--anchor_origin`, 기본 subject AABB 중심) 기준**이다. 카메라 절대
    # 높이가 아니라서 같은 각도라도 subject 가 앉아 있으면 카메라가 낮게 앉는다.
    #
    # 음수 고도(= low angle)를 기본에 넣는 이유: 양수만 두면 `camera_angle` 이 구조적으로
    # `low_angle` 을 못 낸다 (실측: el 10/25/45 만으로 chunk 3개에서 low_angle 0건, 최저가
    # el 10 x r1.10 의 눈높이 -15deg `slight_low`). -10 을 넣으면 나온다.
    #
    # -25 는 **안 넣는다.** 임계 문제가 아니라 물리적으로 땅속이다 — chunk 51/600/1500 에서
    # `height_over_floor` 가 각각 -0.54..0.10 / -0.71..-0.08 / -0.42..0.22 m 로 48/48 전부
    # `below_floor`. `--min_height` 를 0 으로 내려도 대부분 음수라 살아나지 않는다.
    #
    # -10 도 공짜는 아니다. 구 중심이 낮으면(무릎 꿇은 chunk 600 은 중심 ~0.56 m) 48/48 이
    # `below_floor` 로 죽고, 서 있는 chunk 에서만 몇 칸 남는다. 통과 못 해도 비용은 광선 없이
    # 격자 한 줄뿐이라(`reject` 면 광선 6+3 발을 건너뛴다) 기본에 두는 쪽이 싸다.
    parser.add_argument("--elevations", default=[-10.0, 10.0, 25.0, 45.0], nargs="+", type=float)
    # 반경은 기본이 **절대 미터**다. Lite 에는 시작 카메라가 없어서 d_ref 배수로 잡을 기준
    # 자체가 애매하고, 실제로 통과율 천장을 정하는 clearance/occlusion 은 미터 문제라 렌즈를
    # 바꿔도 안 움직인다 (실측: lens 25/18/14 에서 clearance 365 / occluded 334 기각이 바이트
    # 단위로 동일, 렌즈에 반응하는 건 cropped/area 뿐). 상대 격자의 존재 이유였던 "shot scale
    # 의 뜻을 고정한다"는 이제 `shot` 태그가 직접 하므로 반경으로 대리할 필요가 없다.
    # d_ref 배수 격자로 돌리려면 `--radii_rel` 을 준다 (주면 `--radii` 는 무시된다).
    parser.add_argument("--radii", default=[1.1, 1.5, 2.0, 2.6], nargs="+", type=float)
    parser.add_argument("--radii_rel", default=[], nargs="*", type=float)
    parser.add_argument("--fit_margin", default=1.15, type=float)          # d_ref 의 프레임 여유

    # --- subject / 원점 --------------------------------------------------------------------
    parser.add_argument("--subject_kind", default="human",
                        choices=("human", "event", "object"))
    parser.add_argument("--prop_names", default=[], nargs="*", type=str)   # obj_list.txt 이름
    # 격자 원점. obb_center 는 subject-agnostic 이고 `anchor_cond` 가 싣는 값과 원점이 일치한다.
    # chest 는 human 전용이고 기존 뱅크 96편 재현 전용.
    parser.add_argument("--anchor_origin", default="obb_center",
                        choices=("obb_center", "chest"))
    # 격자 원점·조준점·렌더가 놓이는 프레임, 그리고 게이트를 몇 프레임에서 볼지.
    # 기본(`mid` + `window`)이 기존 동작이다. 보행 chunk 를 첫 pose 기준으로 보려면
    # `--anchor_frame start --gate_frames anchor` (`probe_chunk` 상단 주석 참고).
    parser.add_argument("--anchor_frame", default="mid", choices=("mid", "start"))
    parser.add_argument("--gate_frames", default="window", choices=("window", "anchor"))
    # 조준점을 OBB 중심(0.50*h)에서 위로 올리는 양 (사람 키 비율). 0.20 이면 0.70*h — 중심을
    # 그대로 겨누면 머리가 잘린다. `trumans_to_recon.py:1190` 과 같은 값.
    parser.add_argument("--aim_bias", default=0.20, type=float)

    # --- facing 태그 (게이트 아님) ------------------------------------------------------------
    # 리그의 정면 축을 recording 전체의 보행 구간으로 푼다. 하드코딩하지 않는 이유는
    # `resolve_forward_axis` 주석 참고. 샘플 간격은 30 Hz 기준 10 프레임 = 1/3 초.
    parser.add_argument("--facing_sample_step", default=10, type=int)
    # 이 간격 동안의 수평 이동이 이보다 작으면 "서 있다"로 보고 축 투표에서 뺀다 (m).
    parser.add_argument("--facing_min_speed", default=0.02, type=float)
    # 정면을 읽을 본. 루트(`CC_Base_BoneRoot`)는 전 프레임 고정이라 쓰면 안 된다 (`bone_basis`).
    parser.add_argument("--facing_bone", default="CC_Base_Hip", type=str)
    # 보행 일치도가 이보다 낮으면 축을 못 푼 것으로 보고 facing 을 전부 unknown 으로 둔다.
    # 실측: Hip/Pelvis/Waist/Spine02/Head 전부 +0.68, 정지 루트 본 0.09.
    parser.add_argument("--facing_min_agreement", default=0.35, type=float)
    # 좌우 접미사(`_l`/`_r`)를 붙일 최소 프레임 비율. 창 안에서 사람이 카메라 앞을 가로지르면
    # 좌우가 실제로 바뀌므로, 못 넘으면 접미사 없이 크기만 부른다 (`facing_label`).
    parser.add_argument("--facing_min_side_frac", default=0.70, type=float)
    # 촬영 각도(`camera_angle`)의 기준이 되는 눈 메시. 못 찾으면 신장의 94% 로 근사하는데
    # 무릎 꿇기/앉기에서 깨지므로 이름이 다른 리그면 반드시 지정할 것 (`eye_position`).
    parser.add_argument("--eye_mesh", default=["CC_Base_Eye"], nargs="+", type=str)

    # --- 게이트 ----------------------------------------------------------------------------
    parser.add_argument("--probe_distance", default=1.5, type=float)       # clearance 광선 최대 거리
    parser.add_argument("--min_clearance", default=0.20, type=float)       # 벽까지 최소 거리 (D120)
    parser.add_argument("--min_subject_dist", default=0.80, type=float)    # 조준점까지 최소 거리
    parser.add_argument("--min_height", default=0.30, type=float)          # 바닥 위 최소 높이
    parser.add_argument("--scene_margin", default=0.50, type=float)        # 씬 AABB 밖 허용 여유
    parser.add_argument("--min_crop_keep", default=0.85, type=float)       # OBB 가 화면 안에 남는 비율
    parser.add_argument("--min_area_frac", default=0.03, type=float)       # 너무 작으면 점
    parser.add_argument("--max_area_frac", default=0.60, type=float)       # 너무 크면 클로즈업 과다
    parser.add_argument("--max_center_offset", default=0.60, type=float)   # 중앙에서 벗어난 정도

    # --- 렌더 ------------------------------------------------------------------------------
    parser.add_argument("--lens", default=25.0, type=float)                # 기존 뱅크와 같은 25 mm
    parser.add_argument("--sensor", default=36.0, type=float)
    parser.add_argument("--res", default=[960, 540], nargs=2, type=int)
    parser.add_argument("--samples", default=16, type=int)                 # EEVEE TAA 샘플 수
    parser.add_argument("--max_render", default=0, type=int)               # 0 = 통과분 전부
    # 탈락 사유마다 렌더할 대표 후보 수. 0 이면 통과분만 렌더한다(기존 동작). 통과 0장인 chunk 는
    # 이걸 켜야 원인을 그림으로 볼 수 있다. 사유가 적게 겹치는 칸부터 고른다 —
    # `pick_reject_samples` 주석 참고.
    parser.add_argument("--render_rejects", default=0, type=int)
    # `--max_render` 로 자를 때 **무엇을 남길지**. `crop` 이 기존 동작(= `crop_keep` 상위)이고,
    # `diverse` 는 `(shot, camera_angle)` 셀 라운드로빈이다. board 를 사람이 훑거나 VLM 에
    # 보여줄 거면 `diverse` — `crop` 은 wide/high-angle 만 남긴다 (`pick_diverse` 주석에 실측).
    parser.add_argument("--render_select", default="crop", choices=("crop", "diverse"))
    # 게이트만 돌리고 렌더는 건너뛴다 (격자/임계 튜닝할 때).
    parser.add_argument("--no_render", action="store_true")
    main(parser.parse_args(cli_argv()))
