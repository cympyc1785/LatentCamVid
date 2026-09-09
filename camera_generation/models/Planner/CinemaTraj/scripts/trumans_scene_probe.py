"""TRUMANS `.blend` 에서 **사람 궤적**과 **카메라가 설 수 있는 자리**를 headless Blender 로 잰다.

(이름이 비슷한 `trumans_probe.py` 는 배포본 *파일*을 훑어 SMPL-X 로 카메라를 복원하는 별개
스크립트다. 이쪽은 `.blend` *씬 기하*를 광선으로 재는 Blender 워커다.)

**왜 필요한가.** LBM-Lite 는 지금까지 DA3 depth shell 위에서 돌았다. 그 shell 은 소스 카메라가
본 표면 한 겹뿐이라 "카메라가 벽 속에 있나"를 *관측된 표면보다 뒤인가*로 근사할 수밖에 없었다
(`lbm/gates.py` G1). TRUMANS 는 씬 전체가 실제 mesh 로 들어 있으므로 그 근사를 버리고
**진짜 광선을 쏴서** 판정할 수 있다. full-house 씬이라 카메라를 아무데나 두면 벽 안에 박히는데,
그게 depth shell 에서는 안 잡히던 실패 모드였다.

**이 스크립트가 내놓는 것 (JSON 한 장).**
- `human_track` — 프레임별 **subject union** 의 world AABB 중심 / min / max, 그리고 루트 본의
  world 3x3. forward 축을 여기서 **정하지 않는다**: SMPL-X 리그의 어느 로컬 축이 정면인지
  하드코딩하면 리그가 바뀔 때 조용히 틀린다. 대신 3축을 다 실어 보내고, orchestrator 가
  "걷는 방향과 가장 잘 맞는 축"으로 자기 교정한다.
  (이름이 `human_track` 인 건 하위 호환 때문이다. `--subject_kind` 가 `event`/`object` 면
  그 union 이 사람이 아닐 수 있고, 어떤 member 로 쟀는지는 `subject_meshes` 가 말한다.)
- `subject_kind` / `subject_meshes` / `subject_height` / `prop_meshes` — subject 를 사람 전용에서
  {human, event, object} 로 일반화한 흔적. `human` 은 기존 뱅크 96편과 **비트 동일** 경로다.
- `candidates` — anchor 프레임에서 사람을 중심으로 한 (방위각 x 고도 x 거리) 격자. 각 후보마다
  ① `clear` 사람 가슴 높이에서 카메라까지 시선이 뚫리는가 (가림)
  ② `clearance` 카메라 위치에서 6방향(±X ±Y ±Z) 최단 히트 거리 — 벽 속이면 전부 몇 cm 라
     한 값으로 "벽 속"과 "벽에 너무 붙음"을 동시에 잡는다
  ③ `floor_drop` 바로 아래 바닥까지 거리 (공중/지하 방지)
- `scene_aabb`, `floor_z`, `frame_range`, `fps`.

**측정으로 확정된 사실 (재논의 금지).**
- 사람은 ARMATURE `zzy3` 이고 그 자체는 아무것도 렌더하지 않는다. 실제로 그려지는 건 armature
  가 deform 하는 mesh 12개다 → armature modifier / parent 로 찾는다 (`trumans_gt_render.py` 와 동일).
- TRUMANS `.blend` 는 `scene.frame_step = 2` 로 저장되어 있다. 여기선 프레임을 직접 set 하므로
  영향은 없지만, 헷갈리지 않게 명시적으로 1 로 되돌린다.
- `--cycles` 로 시작하는 CLI 플래그를 **절대 만들지 말 것** (Cycles 애드온이 argv 를 prefix-match
  하며 훑어서 실행이 통째로 죽는다).

env: Blender 내장 python

예시:
    B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/00add26c-7a26-4a61-b192-b97aa493b3f3
    $B -b $R/00add26c-7a26-4a61-b192-b97aa493b3f3.blend --python scripts/trumans_scene_probe.py -- \
        --frames 100 148 --anchor_frame 124 --out /tmp/probe_00add26c.json
"""
import json
import sys
from argparse import ArgumentParser
from math import cos, radians, sin

import bpy
import numpy as np
from mathutils import Vector

RENDERABLE = {"MESH", "CURVE", "SURFACE", "META", "FONT", "VOLUME", "GREASEPENCIL"}
# 6방향 clearance 광선. 벽 **속**이면 여섯 방향이 전부 몇 cm 안에서 히트하므로
# min(hit distance) 하나로 "벽 속"과 "벽에 너무 붙음"을 같은 눈금에서 잰다.
CLEARANCE_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def cli_argv():
    """Blender 는 `--` 뒤를 스크립트 몫으로 남긴다. `--` 가 없으면 인자 없음."""
    argv = sys.argv
    return argv[argv.index("--") + 1:] if "--" in argv else []


def find_human(scene):
    """(armature, [사람 mesh]) — 이름 하드코딩 대신 armature modifier / parent 로 찾는다."""
    armatures = [o for o in scene.objects if o.type == "ARMATURE"]
    assert armatures, "이 blend 에 ARMATURE 가 없다 — TRUMANS recording 이 맞는지 확인할 것."
    armature = armatures[0]
    meshes = []
    for obj in scene.objects:
        if obj.type != "MESH":
            continue
        by_parent = obj.parent is armature
        by_modifier = any(m.type == "ARMATURE" and m.object is armature for m in obj.modifiers)
        if by_parent or by_modifier:
            meshes.append(obj)
    assert meshes, f"armature {armature.name} 가 deform 하는 mesh 를 못 찾았다."
    # 조준점을 뽑을 "몸통" mesh. 옷/머리카락 mesh 는 부분만 덮으므로 맨몸 mesh 를 우선한다.
    named = [m for m in meshes if "body" in m.name.lower()]
    body = named[0] if named else max(meshes, key=lambda m: len(m.data.vertices))
    return armature, meshes, body


def find_props(scene, names):
    """`obj_list.txt` 의 이름들 → 실제 씬 오브젝트. 못 찾은 이름은 **조용히 넘기지 않는다**.

    TRUMANS 의 `obj_list.txt` 는 recording 폴더에 있고 이름이 blend 오브젝트와 그대로 맞는다
    (실측 `['cup_01', 'oven_base_01', 'oven_door_01', ...]`). 다만 `oven` 처럼 한 소품이
    `_base` / `_door` 두 오브젝트로 쪼개져 있어서, orchestrator 는 명사 하나에 여러 이름을
    넘긴다 — 그래서 여기서는 **정확 일치 + 접두 일치** 둘 다 받는다.
    """
    if not names:
        return []
    wanted, found, missing = list(names), [], []
    by_name = {o.name: o for o in scene.objects if o.type in RENDERABLE}
    for name in wanted:
        if name in by_name:
            found.append(by_name[name])
            continue
        prefixed = [o for n, o in sorted(by_name.items()) if n.startswith(name)]
        if prefixed:
            found.extend(prefixed)
        else:
            missing.append(name)
    assert not missing, (f"--prop_names 중 씬에 없는 이름: {missing}. "
                         f"recording 의 obj_list.txt 와 blend 가 맞는지 확인할 것.")
    # 같은 오브젝트가 두 이름에 걸릴 수 있다 (접두 일치). 순서를 지키며 중복만 뺀다.
    unique, seen = [], set()
    for obj in found:
        if obj.name not in seen:
            seen.add(obj.name)
            unique.append(obj)
    return unique


def world_aabb(objects, depsgraph):
    """deform 이 적용된(evaluated) world AABB. `bound_box` 는 로컬이라 matrix_world 를 곱한다."""
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            point = np.asarray(evaluated.matrix_world @ Vector(corner))
            lo = np.minimum(lo, point)
            hi = np.maximum(hi, point)
    return lo, hi


def root_pose(armature):
    """루트(부모 없는 첫) 본의 world 4x4 → (3x3 축, 원점).

    **주의: 이 리그의 루트는 골반이 아니다.** 실측 `root_p = [0, 0, -0.946]` — 몸에서 1 m 가까이
    떨어진 리그 원점이다. 그래서 조준점으로 쓰면 안 되고(`body_points` 를 쓴다), 여기서 내놓는
    3x3 은 orchestrator 가 "정면 축"을 자기 교정할 때 참고용으로만 쓴다.
    """
    roots = [b for b in armature.pose.bones if b.parent is None]
    matrix = armature.matrix_world if not roots else armature.matrix_world @ roots[0].matrix
    return ([[float(matrix[r][c]) for c in range(3)] for r in range(3)],
            [float(matrix[r][3]) for r in range(3)])


def body_points(body_mesh, depsgraph):
    """몸통 위의 조준점 3개(가슴/허리/배꼽 높이)를 **실제 정점**에서 뽑는다.

    왜 AABB 중심이 아닌가: 사람이 오븐 쪽으로 팔을 뻗으면 AABB 가 0.82 x 0.69 m 로 벌어져 그
    중심이 몸통과 팔 사이 **빈 공간**에 앉는다. 그 점을 조준하면 광선이 몸을 통과해 뒷벽을 때리고
    "가려졌다"로 오판한다. 왜 루트 본도 아닌가: 이 리그의 루트는 골반이 아니라 [0,0,-0.946] 이다.

    그래서 높이 띠마다 그 띠에 속한 정점들의 **중앙값 xy** 를 쓴다 — 정의상 그 높이 몸통
    단면 안이고, 뻗은 팔은 소수라 중앙값이 안 끌려간다.
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

    사람처럼 높이띠 median 을 쓰지 않는 이유: 소품은 팔처럼 튀어나온 부위가 없어서 AABB 중심이
    거의 항상 mesh **속**이다. 속에 있으면 카메라에서 쏜 광선의 첫 히트가 그 소품이라
    `line_of_sight` 가 바로 통과 판정한다. `oven_base`/`oven_door` 처럼 한 소품이 여러
    오브젝트로 쪼개져 있으면 점도 그만큼 생기고, "하나라도 맞으면 뚫림" 규칙이 그대로 먹는다.
    """
    points = []
    for obj in props:
        lo, hi = world_aabb([obj], depsgraph)
        points.append([float(v) for v in (lo + hi) / 2.0])
    return points


def lbm_obb_points(lo, hi):
    """LBM `occlusion_check` 의 표본 격자를 **그대로** 옮긴 것 (비교용, 기본 경로 아님).

    `cinematographer_quality_worker.py:2550-2555` 원문:
        for dz in (-0.4, -0.1, 0.15, 0.35, 0.5):
          for dx in (-0.3, 0.0, 0.3):
            for dy in (-0.3, 0.0, 0.3):
              center + Vector((dx*half_x, dy*half_y, dz*half_z*2.0))
    `half_* = max(extent*0.5, 0.1)`. dz 에만 2.0 이 곱해져 있어 세로는 상자 높이의 −40%~+50%
    (즉 발끝을 빼고 정수리까지), 가로세로는 상자 폭의 **±15%** 만 쓴다. 이 ±15% 축소가 LBM 이
    "OBB 안 빈 공간" 에 대해 하는 유일한 대비책이다 — 몸통 심 근처로 표본을 모으는 것.
    """
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    centre, extent = (lo + hi) / 2.0, hi - lo
    half = np.maximum(extent * 0.5, 0.1)
    return [[float(v) for v in centre + np.array([dx * half[0], dy * half[1], dz * half[2] * 2.0])]
            for dz in (-0.4, -0.1, 0.15, 0.35, 0.5)
            for dx in (-0.3, 0.0, 0.3)
            for dy in (-0.3, 0.0, 0.3)]


def dense_aim_points(subject_meshes, depsgraph, bands, per_band):
    """LOS 조준점을 LBM 밀도(45점)로 늘린다. 단 **격자가 아니라 실제 정점**에서 뽑는다.

    왜 LBM 처럼 OBB 격자로 안 하는가. LBM `occlusion_check`
    (`Look-Before-Move/Cinematographer/cinematographer_quality_worker.py:2541`) 는 subject OBB
    안에 5x3x3=45 점을 찍는데 **그 점들이 mesh 안이라는 보장이 없다** — 팔 사이, 다리 사이,
    상자 모서리는 전부 공기다. 그리고 광선이 아무것도 안 맞으면(`hit=False`) LBM 은 그걸
    occluded 로 세지 않으므로, OBB 안 빈 공간은 조용히 **통과표**가 된다. LBM 이 그에 대해 하는
    유일한 완화는 측면 표본을 상자 폭의 ±15%(`dx,dy in (-0.3,0,0.3)` × `half_*`)로 **좁혀서**
    몸통 심 근처에 붙이고, 높이를 −0.4~+0.5 h 로 잘라 발을 빼는 것뿐이다.

    표본을 정점에서 뽑으면 그 문제 자체가 사라진다. 정의상 표면 위이므로 가리는 게 없으면 첫
    히트가 subject 이고, `miss` 는 정말로 "아무것도 없다"는 뜻이 되어 열로 찍어 확인할 수 있다.
    (같은 이유로 `body_points` 도 AABB 중심 대신 높이띠 median xy 를 쓴다 — 그 docstring 참고.)

    높이 `bands` 띠 × 띠마다 방위각으로 고르게 `per_band` 개. 방위각으로 고르는 이유는 실루엣
    양 끝(어깨·팔)까지 표본이 가야, 몸통 앞면만 보이는 반쯤 가려진 구도가 1.00 으로 안 찍히기
    때문이다.
    """
    chunks = []
    for mesh in subject_meshes:
        evaluated = mesh.evaluated_get(depsgraph)
        data = getattr(evaluated, "data", None)
        if data is None or not getattr(data, "vertices", None):
            continue
        matrix = evaluated.matrix_world
        chunks.append(np.asarray([matrix @ v.co for v in data.vertices], dtype=np.float64))
    if not chunks:
        return []
    vertices = np.concatenate(chunks, axis=0)
    z_lo, z_hi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
    span = max(1e-6, z_hi - z_lo)
    points = []
    for band_index in range(bands):
        # 위아래 8% 는 버린다 (LBM 이 dz 를 −0.4..+0.5 로 자른 것과 같은 취지 — 발끝/정수리
        # 한 점은 실루엣 극단이라 가림 판정에 정보가 거의 없다).
        lo = z_lo + span * (0.08 + 0.84 * band_index / bands)
        hi = z_lo + span * (0.08 + 0.84 * (band_index + 1) / bands)
        band = vertices[(vertices[:, 2] >= lo) & (vertices[:, 2] < hi)]
        if len(band) < per_band:
            band = vertices
        centre = band[:, :2].mean(axis=0)
        order = np.argsort(np.arctan2(band[:, 1] - centre[1], band[:, 0] - centre[0]))
        pick = order[np.linspace(0, len(order) - 1, per_band).round().astype(int)]
        points.extend([[float(v) for v in band[j]] for j in pick])
    return points


def aim_points(entry):
    """조준 후보점. 하나라도 subject 를 맞히면 시선이 뚫린 것으로 본다.

    한 점만 쓰면 그 점이 하필 팔 사이 빈틈이거나 몸 뒤쪽 실루엣 밖일 때 통째로 기각된다.

    `aim_points` 키는 subject_kind 에 맞춰 probe 가 미리 골라 담은 것이다
    (human = 몸통 3점 / object = 소품 중심들 / event = 둘 다). 없으면 `body_points` 로
    떨어지는데, 이건 **subject_kind 가 생기기 전에 만든 JSON** 을 그대로 읽기 위한 길이다.
    """
    return [np.asarray(p, dtype=np.float64)
            for p in entry.get("aim_points") or entry["body_points"]]


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
    """카메라 -> 조준점들로 쏴서 **처음 맞는 게 subject 인** 광선이 하나라도 있으면 시선이 뚫린 것.

    반대 방향(몸 -> 카메라)으로 쏘면 안 된다: 시작점이 사람 몸 **속**이라 자기 mesh 를 즉시
    때린다 (실측 342개 차단 중 260개가 0.5 m 이내, median 0.066 m — 전부 자기 몸이었다).
    임계값 대신 맞은 오브젝트 이름으로 판정하므로 subject 크기에 안 휘둘린다 — 그래서 사람이든
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


def dense_line_of_sight(scene, depsgraph, position, targets, subject_names):
    """밀집 조준점 전량에 한 발씩. (subject 첫히트 비율, 남에게 막힌 비율, 무히트 비율).

    LBM `occlusion_check` 와 판정 규칙이 같다 — ① 카메라 → 표본점 1발 ② 첫 히트가 subject 면
    가림 아님(자기 몸 뒤쪽 정점도 앞면을 맞으므로 보이는 것으로 친다) ③ 첫 히트가 남이면 가림.
    다른 건 `miss` 를 **따로 센다**는 것뿐이다. LBM 은 miss 를 "안 가려짐"에 합쳐 버려서
    OBB 빈 공간을 통과표로 세는데(`dense_aim_points` docstring), 여기선 그게 열로 보인다.

    boolean `line_of_sight` 를 대체하지 않고 **나란히** 돈다: 기존 게이트(`clear`)는 3점 OR 로
    그대로 두고 이 비율은 열로만 붙는다. 안 그러면 기존 뱅크가 같은 시드에서 다른 결과를 낸다.
    """
    subject = blocked = miss = 0
    for target in targets:
        direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
        hit, _, name = ray(scene, depsgraph, position, direction,
                           float(np.linalg.norm(direction)) + 0.05)
        if not hit:
            miss += 1
        elif name in subject_names:
            subject += 1
        else:
            blocked += 1
    total = max(1, len(targets))
    return subject / total, blocked / total, miss / total


def clearance_of(scene, depsgraph, position, probe_distance, subject_names):
    """6방향 최단 히트 거리. 벽 속이면 작다. 히트가 없으면 probe_distance 로 포화시킨다.

    subject mesh 히트는 무시한다 — subject 는 충돌 위험물이 아니라 피사체이고, 카메라-subject
    거리는 `radius` 와 `min_subject_dist` 가 이미 통제한다. 안 빼면 반경 1.5 m 후보가 전부
    "벽에 붙었다"로 찍힌다.

    ⚠ event/object 에서는 이 면제가 사람보다 **위험하다**. 소품은 보통 책상 위에 있어서
    소품만 빼면 책상이 남아 clearance 가 제 역할을 하지만, 소품이 통째로 벽장 안이면
    벽장 문짝(= subject member)이 면제되어 "벽 속인데 통과"가 될 수 있다. 그래서
    orchestrator 는 event 에서 `--min_clearance` 를 낮추지 않는다.
    """
    best = float(probe_distance)
    for direction in CLEARANCE_DIRS:
        hit, distance, name = ray(scene, depsgraph, position, direction, probe_distance)
        if hit and name not in subject_names:
            best = min(best, distance)
    return best


def main(args):
    scene = bpy.context.scene
    scene.frame_step = 1
    armature, human_meshes, body_mesh = find_human(scene)
    human_set = set(human_meshes)

    # subject 를 사람 전용에서 일반화한다.
    #   human  = 사람 mesh union            (기존 뱅크 96편과 **비트 동일**)
    #   event  = 사람 ∪ 상호작용 소품 union  ("드링크를 마시는 사람" 처럼 행위를 담는 프레이밍)
    #   object = 소품만                      (사람 없이 소품만 겨누는 shot)
    # 아래 코드는 subject 를 `subject_meshes` / `subject_names` / `aim_points` 세 개로만
    # 참조한다 — kind 분기가 여기 한 곳에 모여 있어야 게이트가 조용히 어긋나지 않는다.
    props = find_props(scene, args.prop_names)
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
        # armature 자체는 아무것도 렌더하지 않지만 ray_cast 히트 이름으로는 나올 수 있다.
        subject_names.add(armature.name)

    # `--frames start end [step]`. step 은 **모션 프레임 간격**이다 — TRUMANS 모션은 30 Hz 라
    # step 3 이면 10 fps 로 샘플한 셈이고 49프레임이 4.8초를 덮는다. 기본 1 = 기존 동작.
    start, end = int(args.frames[0]), int(args.frames[1])
    step = int(args.frames[2]) if len(args.frames) > 2 else 1
    assert step >= 1, f"--frames 의 step 은 1 이상이어야 한다: {step}"
    # 샘플 격자를 **목록 하나**로 못 박는다. 아래의 모든 `track` 색인은 이 목록의 위치로 한다
    # (`(frame - start) // step` 산술은 균일 격자에서만 맞는데, LBM 카메라를 재렌더할 때 쓰는
    # 프레임 집합은 step 이 2..3 으로 섞인다 — `lbm_camera_to_poses.subsample`).
    # `--frame_list` 가 없으면 목록이 예전 `range(start, end+1, step)` 와 글자 그대로 같다.
    if args.frame_list:
        sample_frames = [int(f) for f in args.frame_list]
        assert sample_frames == sorted(sample_frames), \
            f"--frame_list 는 오름차순이어야 한다: {sample_frames[:10]}"
        start, end = sample_frames[0], sample_frames[-1]
    else:
        sample_frames = list(range(start, end + 1, step))
    assert sample_frames, f"--frames {args.frames} 가 빈 범위다"
    frame_index = {f: i for i, f in enumerate(sample_frames)}

    # anchor 는 **샘플 격자 위**에 있어야 한다. 안 그러면 아래 `track[frame_index[anchor]]` 이
    # 실제로 그 프레임이 아닌 이웃을 집는다 (step=1 이던 시절엔 항상 격자 위였다).
    if args.frame_list:
        # 비균일 격자 — 가운데 프레임에 가장 가까운 표본으로 스냅한다.
        anchor = int(args.anchor_frame) if args.anchor_frame >= 0 else (start + end) // 2
        anchor = min(sample_frames, key=lambda f: (abs(f - anchor), f))
    else:
        # 균일 격자 — 기존 산술 그대로 (`round` 의 banker's rounding 까지 보존).
        if args.anchor_frame >= 0:
            anchor = int(args.anchor_frame)
        else:
            anchor = start + step * int(round(((start + end) // 2 - start) / step))
        anchor = min(max(anchor, start), start + step * ((end - start) // step))

    # 정적 씬 AABB 는 사람을 뺀 렌더 가능 오브젝트로 잰다 (사람이 움직이면 AABB 도 흔들린다).
    scene.frame_set(anchor)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    #   소품도 사람처럼 프레임마다 움직일 수 있으므로(들었다 놓는다) 정적 AABB 에서 뺀다.
    #   `prop_names` 가 비면 집합이 예전과 같아 human 경로는 그대로다.
    moving_set = human_set | set(props)
    static_objects = [o for o in scene.objects
                      if o.type in RENDERABLE and o not in moving_set and o.visible_get()]
    scene_lo, scene_hi = world_aabb(static_objects, depsgraph)

    track = []
    for frame in sample_frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        # `center`/`min`/`max` 는 **subject union** 이다. human 이면 사람 AABB 그대로라
        # 기존 소비자(`anchor_cond`, 격자 원점, `--aim_bias` 눈금)가 안 바뀐다.
        lo, hi = world_aabb(subject_meshes, depsgraph)
        root_R, root_p = root_pose(armature)
        body = body_points(body_mesh, depsgraph)
        prop = prop_points(props, depsgraph) if props else []
        entry = {
            "frame": frame,
            "center": [float(v) for v in (lo + hi) / 2.0],
            "min": [float(v) for v in lo], "max": [float(v) for v in hi],
            "root_R": root_R, "root_p": root_p,
            "body_points": body,
        }
        if prop:
            entry["prop_points"] = prop
        # 조준점을 kind 에 맞춰 **여기서 한 번** 고른다. `aim_points()` 는 이 키만 읽는다.
        entry["aim_points"] = (body if args.subject_kind == "human" else
                               prop if args.subject_kind == "object" else body + prop)
        # `--los_samples 0`(기본)이면 키 자체가 안 생기고 JSON 이 예전과 글자 그대로 같다.
        if args.los_samples > 0:
            entry["dense_points"] = (
                lbm_obb_points(lo, hi) if args.los_mode == "obb_lbm" else
                dense_aim_points(subject_meshes, depsgraph, int(args.los_bands),
                                 max(1, int(round(args.los_samples / max(1, args.los_bands))))))
        track.append(entry)

    # 바닥 z 는 anchor 프레임 사람 발 아래로 광선을 쏴서 잰다 (씬 AABB 최저점은 지하실/기초를 잡는다).
    scene.frame_set(anchor)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    anchor_track = track[frame_index[anchor]]
    assert anchor_track["frame"] == anchor, \
        f"anchor {anchor} 가 샘플 격자({start}..{end} step {step}) 위에 없다"
    # 바닥은 **항상 사람 발** 기준으로 잰다. subject 가 소품이면 그 AABB 최저점은 책상 상판이라
    # 거기서 아래로 쏘면 floor_z 가 책상 높이로 잡히고, "카메라가 바닥 밑" 게이트가 통째로
    # 무의미해진다 (실내 씬에서 0.7 m 쯤 위를 바닥으로 착각한다).
    human_lo, human_hi = world_aabb(human_meshes, depsgraph)
    foot = human_lo
    human_names = {o.name for o in human_meshes} | {armature.name}
    hit, drop, _ = ray(scene, depsgraph, [human_lo[0] * 0.5 + human_hi[0] * 0.5,
                                          human_lo[1] * 0.5 + human_hi[1] * 0.5,
                                          foot[2] + 0.30], (0, 0, -1), 5.0)
    floor_z = float(foot[2] + 0.30 - drop) if hit else float(foot[2])

    # 사람 키. 후보 고도와 look-at 높이를 **사람** 크기로 정규화하는 눈금이다 — subject 가
    # 15 cm 짜리 컵이어도 카메라 고도 눈금까지 15 cm 로 줄이면 안 되므로 이 값은 사람으로 둔다.
    height = float(human_hi[2] - human_lo[2])
    # subject 자체 크기는 따로 싣는다 (프레이밍 `fill` 계산은 이쪽을 써야 한다).
    subject_height = float(anchor_track["max"][2] - anchor_track["min"][2])
    anchor_aims = aim_points(anchor_track)
    chest_point = np.asarray(anchor_track["body_points"][0], dtype=np.float64)  # 몸통 70% median xy
    obb_center = np.asarray(anchor_track["center"], dtype=np.float64)   # subject OBB 중심 (=0.50·h)
    assert args.anchor_origin != "chest" or args.subject_kind == "human", \
        "--anchor_origin chest 는 human 전용이다 (기존 뱅크 96편 재현용). event/object 는 obb_center."

    # 격자 원점을 무엇으로 둘 것인가.
    # `chest` 는 human 전용 휴리스틱이라 subject 를 event/object 로 일반화하면 정의가 없어지고,
    # 더 중요하게는 `anchor_cond` 가 싣는 값이 **OBB center** 라 원점이 다르면 모델이 조건에
    # 설명이 없는 offset(수평 0.106 m + 수직 0.20·h)을 보게 된다. loss 에는 안 나타나고
    # 프레이밍만 조용히 어긋난다. 그래서 기본은 `obb_center` 고, 흉부 높이는 orchestrator 의
    # `--aim_bias`(사람 키 비율) 로 복원한다 — 하드코딩이 로그에 찍히는 숫자로 바뀐다.
    #
    # "흉부가 더 일정하다"는 근거가 아니다. 200개 랜덤 49프레임 창 실측에서 프레임당 앵커
    # 이동 median 이 흉부(spine3) 0.0090 m vs OBB center 0.0098 m 로 사실상 같고, 둘 사이
    # 수평 gap 은 median 0.106 m 인데 **창 내 std 가 0.027 m** 라 거의 상수 offset 이다.
    #
    # `chest` 는 기존 뱅크 96편 재현용으로 남긴다 — 원점이 바뀌면 `radius` 와 게이트 값이 전부
    # 달라져서 같은 시드로도 다른 카메라가 나온다.
    chest = chest_point if args.anchor_origin == "chest" else obb_center

    # --verify_poses: 격자 대신 **이미 합성한 궤적**을 프레임별로 검증한다. 격자 검증은 anchor
    # 프레임 한 장뿐이라, 카메라가 움직이는 동안 벽을 뚫고 지나가는 건 못 잡는다. 149초짜리
    # 렌더를 태우기 전에 4초로 거른다.
    # cam_c2w 는 (F,4,4) 한 궤적이거나 (K,F,4,4) 궤적 **여러 개**다. 여러 개를 한 번에 받는
    # 이유는 Blender 기동 + 1.6GB blend 로드가 4초라, 후보를 하나씩 물어보면 재시도가 곧 4초씩
    # 쌓이기 때문. 프레임 루프를 바깥에 두면 frame_set 은 K 개든 F 번뿐이다.
    if args.verify_poses:
        poses = np.load(args.verify_poses)["cam_c2w"]
        if poses.ndim == 3:
            poses = poses[None]
        assert poses.ndim == 4 and poses.shape[2:] == (4, 4), \
            f"cam_c2w 는 (F,4,4) 또는 (K,F,4,4) 여야 한다: {poses.shape}"
        assert poses.shape[1] == len(track), f"pose {poses.shape[1]} vs 프레임 {len(track)}"
        paths = [[] for _ in poses]
        for i, entry in enumerate(track):
            scene.frame_set(entry["frame"])
            depsgraph = bpy.context.evaluated_depsgraph_get()
            aims = aim_points(entry)
            dense = entry.get("dense_points") or []
            for k in range(len(poses)):
                position = np.asarray(poses[k][i][:3, 3], dtype=np.float64)
                clear, blocked_at = line_of_sight(scene, depsgraph, position, aims, subject_names)
                paths[k].append({
                    "index": i, "frame": entry["frame"],
                    "position": [float(v) for v in position],
                    "clear": bool(clear), "blocked_at": blocked_at,
                    "clearance": float(clearance_of(scene, depsgraph, position,
                                                    float(args.probe_distance), subject_names)),
                    "floor_drop": float(ray(scene, depsgraph, position, (0, 0, -1), 10.0)[1]),
                    "height_over_floor": float(position[2] - floor_z),
                    # 카메라-사람 거리. `clearance` 는 사람 히트를 **무시**하므로(위 docstring)
                    # 이 열이 없으면 피사체를 뚫고 들어가는 궤적을 아무것도 못 막는다.
                    # 반경 하한이 1.5 m 였을 땐 push_in(dradius -0.55) 도 0.95 m 라 안 드러났지만,
                    # r=1.0 을 격자에 넣은 뒤로는 0.60 m 까지 들어간다.
                    "subject_dist": float(min(np.linalg.norm(a - position) for a in aims)),
                })
                if dense:
                    seen, hidden, missed = dense_line_of_sight(
                        scene, depsgraph, position, dense, subject_names)
                    paths[k][-1].update({"los_frac": float(seen),
                                         "occluded_frac": float(hidden),
                                         "miss_frac": float(missed),
                                         "los_samples": len(dense)})
        summary = [{"path": k,
                    "clear_frac": float(np.mean([r["clear"] for r in rows])),
                    "min_clearance": float(min(r["clearance"] for r in rows)),
                    "min_subject_dist": float(min(r["subject_dist"] for r in rows)),
                    "min_floor_drop": float(min(r["floor_drop"] for r in rows))}
                   for k, rows in enumerate(paths)]
        if args.los_samples > 0:
            # 밀집 LOS 요약. `min_los_frac` 이 게이트용(최악 프레임), `mean_` 은 진단용.
            # `max_occluded_frac` 은 LBM `occlusion_ratio` 와 같은 정의라 직접 비교된다.
            for row, rows in zip(summary, paths):
                row.update({
                    "min_los_frac": float(min(r["los_frac"] for r in rows)),
                    "mean_los_frac": float(np.mean([r["los_frac"] for r in rows])),
                    "max_occluded_frac": float(max(r["occluded_frac"] for r in rows)),
                    "max_miss_frac": float(max(r["miss_frac"] for r in rows)),
                    "los_samples": int(rows[0]["los_samples"]),
                })
        result = {
            "format": "trumans_pose_verify_v1", "blend": bpy.data.filepath,
            "frame_range": [start, end], "frame_step": step,
            "frame_list": sample_frames,   # 실제 표본 격자. 비균일이면 frame_step 만으론 못 되짚는다
            "floor_z": floor_z, "human_height": height,
            "subject_kind": args.subject_kind, "subject_height": subject_height,
            "subject_meshes": sorted(subject_names), "prop_names": list(args.prop_names),
            "human_track": track, "summary": summary, "paths": paths,
            "poses": paths[0],       # 하위 호환: 단일 궤적 소비자용
        }
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=1)
        for row in summary:
            print(f"{'verify path ' + str(row['path']):22s} {len(paths[0])} frames  "
                  f"clear {row['clear_frac']:.2f}  min clearance {row['min_clearance']:.3f}  "
                  f"min subject_dist {row['min_subject_dist']:.3f}"
                  + (f"  los {row['min_los_frac']:.2f}..{row['mean_los_frac']:.2f}"
                     f"  occl<={row['max_occluded_frac']:.2f}  miss<={row['max_miss_frac']:.2f}"
                     f"  (n={row['los_samples']})" if "min_los_frac" in row else ""))
        print(f"{'->':22s} {args.out}")
        return

    candidates = []
    for azimuth in np.arange(0.0, 360.0, float(args.az_step)):
        for elevation in args.elevations:
            for radius in args.radii:
                phi, theta = radians(float(azimuth)), radians(float(elevation))
                offset = np.array([cos(phi) * cos(theta), sin(phi) * cos(theta), sin(theta)]) * float(radius)
                candidates.append({
                    "azimuth_deg": float(azimuth), "elevation_deg": float(elevation),
                    "radius": float(radius), "position": [float(v) for v in (chest + offset)],
                })

    # 후보를 **anchor 한 프레임만** 보고 통과시키면, 사람이 움직이는 구간에서 시작 프레임에는
    # 가려져 있다가 중간부터 보이는 후보가 뽑힌다 (action 10 실측: 49프레임 중 앞 21장이 막힘).
    # 그래서 구간의 시작·중앙·끝 세 프레임에서 다 뚫려야 `clear` 로 친다.
    probe_frames = sorted({start, anchor, end})
    per_frame = {}
    for frame in probe_frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        #    `track` 은 **샘플 격자** 위에 있다. `frame - start` 로 바로 색인하면 step>1 에서
        #    인덱스가 step 배로 튄다 (step 3, end 144 → 144 vs len(track) 49). 비균일 격자
        #    (`--frame_list`)에서는 나눗셈조차 못 쓰므로 목록 위치로 찾는다.
        aims = aim_points(track[frame_index[frame]])
        dense = track[frame_index[frame]].get("dense_points") or []
        rows = []
        for candidate in candidates:
            position = np.asarray(candidate["position"], dtype=np.float64)
            clear, blocked_at = line_of_sight(scene, depsgraph, position, aims, subject_names)
            seen = dense_line_of_sight(scene, depsgraph, position, dense,
                                       subject_names)[0] if dense else None
            rows.append((bool(clear), blocked_at,
                         float(clearance_of(scene, depsgraph, position,
                                            float(args.probe_distance), subject_names)),
                         float(ray(scene, depsgraph, position, (0, 0, -1), 10.0)[1]),
                         seen))
        per_frame[frame] = rows

    for i, candidate in enumerate(candidates):
        rows = [per_frame[f][i] for f in probe_frames]
        blocked = [r[1] for r in rows if r[1] is not None]
        candidate.update({
            "clear": all(r[0] for r in rows),                    # 세 프레임 전부 뚫려야 한다
            "clear_frac": float(np.mean([r[0] for r in rows])),
            "blocked_at": min(blocked) if blocked else None,
            "clearance": min(r[2] for r in rows),                # 최악 프레임 기준
            "floor_drop": min(r[3] for r in rows),
        })
        if rows[0][4] is not None:
            candidate["los_frac"] = float(min(r[4] for r in rows))   # 최악 프레임 기준

    result = {
        "format": "trumans_scene_probe_v1",
        "body_mesh": body_mesh.name,
        "blend": bpy.data.filepath,
        "fps": float(scene.render.fps / max(1, scene.render.fps_base)),
        "frame_range": [start, end], "frame_step": step, "anchor_frame": anchor,
        "frame_list": sample_frames,       # 실제 표본 격자. 비균일이면 frame_step 만으론 못 되짚는다
        "probe_frames": probe_frames,
        "armature": armature.name,
        "human_meshes": sorted(o.name for o in human_meshes),
        # `human_height` 는 **사람** 키다 (카메라 고도·look-at 눈금). subject 가 소품이어도
        # 이 값은 사람으로 유지한다 — 컵을 찍는다고 카메라 고도까지 15 cm 로 줄이면 안 된다.
        # 프레이밍 `fill` 은 `subject_height` 를 써야 한다.
        "human_height": height,
        "subject_kind": args.subject_kind,
        "subject_height": subject_height,
        "subject_meshes": sorted(subject_names),
        "prop_names": list(args.prop_names),
        "prop_meshes": [o.name for o in props],
        "floor_z": floor_z,
        "aim_point": [float(v) for v in chest],
        # 원점 후보를 **둘 다** 싣는다. orchestrator 가 `--aim_bias` 를 계산하고
        # anchor_cond 가 OBB center 를 그대로 쓸 수 있게. 어느 걸 격자 원점으로 썼는지는
        # `anchor_origin` 이 말한다 (기본 obb_center).
        "anchor_origin": args.anchor_origin,
        "chest_point": [float(v) for v in chest_point],
        "obb_center": [float(v) for v in obb_center],
        "scene_aabb": {"min": [float(v) for v in scene_lo], "max": [float(v) for v in scene_hi]},
        "human_track": track,
        "candidates": candidates,
        "probe_distance": float(args.probe_distance),
    }
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=1)

    n_clear = sum(1 for c in candidates if c["clear"])
    n_room = sum(1 for c in candidates if c["clear"] and c["clearance"] >= args.min_clearance)
    print(f"{'blend':22s} {bpy.data.filepath}")
    print(f"{'armature':22s} {armature.name}  meshes {len(human_meshes)}")
    print(f"{'frames':22s} {start}..{end}  anchor {anchor}  fps {result['fps']:g}")
    print(f"{'human_height':22s} {height:.3f} m   floor_z {floor_z:.3f}")
    print(f"{'subject':22s} {args.subject_kind}  h {subject_height:.3f} m  "
          f"members {len(subject_meshes)}  props {[o.name for o in props]}")
    print(f"{'candidates':22s} {len(candidates)}  clear {n_clear}  "
          f"clear&clearance>={args.min_clearance} {n_room}")
    print(f"{'->':22s} {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", required=True, type=str)                 # 결과 JSON
    # start end [step]. end 포함. step 은 모션 프레임 간격 (30 Hz 기준 3 => 10 fps 샘플).
    parser.add_argument("--frames", required=True, nargs="+", type=int)
    # 비균일 표본 격자. 주면 `--frames` 격자 대신 이 목록을 그대로 쓴다 (LBM 카메라 재렌더용).
    parser.add_argument("--frame_list", default=[], nargs="+", type=int)
    parser.add_argument("--anchor_frame", default=-1, type=int)           # -1 = 구간 중앙
    # 격자 원점. obb_center 는 subject-agnostic 이라 event/object 로 일반화되고 anchor_cond 와
    # 원점이 일치한다. chest 는 human 전용이고 **기존 뱅크 96편 재현 전용**이다.
    parser.add_argument("--anchor_origin", default="obb_center",
                        choices=("obb_center", "chest"))
    # subject 정의. human 은 기존 뱅크 96편과 **비트 동일**한 경로다.
    #   event  = 사람 ∪ 소품 (행위를 담는 프레이밍)   object = 소품만
    parser.add_argument("--subject_kind", default="human",
                        choices=("human", "event", "object"))
    # 소품 오브젝트 이름. recording 폴더의 `obj_list.txt` 에 있는 이름을 그대로 넘긴다
    # (`oven` 처럼 명사 하나가 `oven_base_01`/`oven_door_01` 둘로 쪼개져 있으면 둘 다).
    parser.add_argument("--prop_names", default=[], nargs="*", type=str)
    parser.add_argument("--az_step", default=15.0, type=float)            # 방위각 격자 간격(도)
    parser.add_argument("--elevations", default=[0.0, 12.0, 25.0, 40.0], nargs="+", type=float)
    # 반경 격자. 1.0 m 를 넣는 이유는 프레이밍이다 — Lite 렌즈가 25 mm 고정이라
    # `fill = subject_h / frame_h ~= 1.878 * (subject_h/1.521) / r` 이고, 기존 하한 1.5 m 로는
    # fill 이 1.4 를 못 넘어 뱅크 96편에 **medium shot 이 0건**이었다. r=1.0 은 fill 1.55~2.08 로
    # 그 칸을 채운다. 실측(00add26c a00/a04/a07 x 96격자) 통과율은 오히려 r=1.0 이 최고다
    # (usable 75.0 / 45.8 / 53.1 % vs r=1.5 의 46.9 / 32.3 / 24.0 %) — 사람 1 m 앞은 보통 빈 바닥이다.
    # 5.5 m 는 이 실내 3편에선 usable 0 건이지만(4.0 m 도 288 중 1건) 격자 96개 추가 비용이
    # ~0.9 s 뿐이라 남겨 둔다. 넓은 recording 이 나오면 저절로 쓰이고, 아니면 저절로 비어 있다.
    parser.add_argument("--radii", default=[1.0, 1.5, 2.2, 3.0, 4.0, 5.5], nargs="+", type=float)
    parser.add_argument("--verify_poses", default="", type=str)           # npz cam_c2w (N,4,4) 검증 모드
    parser.add_argument("--probe_distance", default=1.5, type=float)      # clearance 광선 최대 거리
    parser.add_argument("--min_clearance", default=0.20, type=float)      # 요약 출력용 임계 (D120)
    # 밀집 LOS. 0(기본)이면 광선도 JSON 키도 예전과 **완전히 같다**. 45 가 LBM `occlusion_check`
    # 와 같은 밀도이고, 표본은 OBB 격자가 아니라 subject **정점**에서 뽑는다
    # (`dense_aim_points` docstring — LBM 격자는 OBB 안 빈 공간을 통과표로 센다).
    parser.add_argument("--los_samples", default=0, type=int)
    parser.add_argument("--los_bands", default=5, type=int)               # 높이 띠 수 (LBM 도 5)
    # `vertex` = 실제 정점(기본).  `obb_lbm` = LBM 45점 OBB 격자 그대로 (비교 전용 —
    # 표본이 공기에 앉으면 `miss_frac` 으로 찍힌다).  `--los_samples` 는 obb_lbm 에서 무시된다.
    parser.add_argument("--los_mode", default="vertex", choices=("vertex", "obb_lbm"))
    main(parser.parse_args(cli_argv()))
