"""카메라를 **벽으로 천천히 밀어 넣으면서** 각 clearance 임계가 *어디서* 걸리는지 말뚝으로 박는다.

**왜 필요한가.** 앞서 만든 `run_raycast_threshold_viz.sh` 는 *같은 궤적*을 `--min_clearance` 만
바꿔 네 번 렌더했다. 그건 "임계를 바꾸면 색이 뒤집힌다"는 것만 보여준다 — 사용자가 보고 싶었던
것은 그 반대다: **임계는 고정해 두고 카메라를 벽 쪽으로 한 걸음씩 밀었을 때 어느 지점에서
0.50 / 0.35 / 0.20 / 0.10 이 차례로 걸리는가.** 임계를 거리 눈금 위의 말뚝으로 보고 싶은 것이다.

그래서 여기서는 궤적을 **합성**한다. 시작점에서 수평 방향 하나를 골라 벽에 닿기 직전까지 등간격
으로 행진하고, 매 걸음에서 `trumans_scene_probe.py` 의 `clearance_of` 를 **그대로** 부른다
(재구현 아님 — 그림과 게이트가 어긋날 수 없다). 임계 T 마다 `clearance < T` 가 처음 성립하는
걸음에 **바닥에서 솟은 색깔 말뚝 + 글자 라벨**을 심는다. 말뚝 사이 간격이 곧 "임계를 0.35 에서
0.20 으로 낮추면 벽에 몇 cm 더 붙을 수 있나"다.

**정직하게 짚어야 하는 것 하나.** `clearance_of` 는 `CLEARANCE_DIRS` 6방향(±x, ±y, **±z**) 의
최솟값이라 **바닥이 벽보다 먼저 걸리는 경우가 잦다** (a17 감사에서 `floor_is_binding_frac`
47.9%). 그러면 벽에 아무리 붙어도 clearance 가 안 떨어진다 — 램프가 평평해진다. 그래서 두 값을
같이 기록·표시한다:

    clearance     6방향 최솟값 = **게이트가 실제로 쓰는 값**
    clearance_h   수평 4방향(±x, ±y) 최솟값 = 벽까지의 여유

둘이 갈라지는 구간이 곧 "바닥이 게이트를 지배하는 구간"이고, 말뚝은 **게이트가 쓰는 값**(전자)
기준으로 박는다 — 그림이 게이트보다 관대하면 볼 이유가 없기 때문이다.

`--cycles` 로 시작하는 CLI 플래그 금지 (Cycles 애드온이 argv 를 prefix-match 로 훑어서 실행이
통째로 죽는다 — 그래서 `--cdevice`). 영상은 안 만든다 (Blender 내장 python 에 imageio 가 없다):
PNG 시퀀스 + `approach.json` 만 떨구고 묶기/플롯은 `vista4d` 쪽에서 한다.

env: Blender 내장 python

예시:
    B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    S=results/20260901_trumans_preset_blender/tru_1d076f8c_a17_s3f0k6
    $B -b <blend> --python scripts/trumans_approach_viz.py -- \
        --audit $S/raycast_probe_rungs.json --start_from_path 0 --start_frame 0 \
        --out /tmp/approach_a17 --steps 40 --thresholds 0.50 0.35 0.20 0.10
"""
import json
import sys
from argparse import ArgumentParser
from math import radians
from os import makedirs, path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPTS_DIR = path.dirname(path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from trumans_frustum_viz import (add_ball, add_wire, ball_mesh,  # noqa: E402
                                 emission_material, frustum_mesh, half_angles, look_at_matrix)
from trumans_raycast_viz import CV_TO_GL, add_tube, cli_argv  # noqa: E402
from trumans_scene_probe import CLEARANCE_DIRS, clearance_of, ray  # noqa: E402

#    말뚝 색. 임계가 클수록(= 보수적일수록) 먼저 걸리므로 위험색 순서로 간다.
POST_COLORS = ((1.00, 0.10, 0.10), (1.00, 0.55, 0.00), (1.00, 0.92, 0.10), (0.20, 1.00, 0.35))
PATH_COLOR = (1.00, 1.00, 1.00)
WALL_RAY_COLOR = (0.20, 0.55, 1.00)     # 행진 방향 광선 = 벽까지 남은 거리
FLOOR_RAY_COLOR = (0.75, 0.25, 1.00)    # 바닥 광선 = 이게 짧으면 바닥이 게이트를 먹는다
CUR_COLOR = (0.10, 1.00, 0.90)          # 현 걸음 표식

HORIZ_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))


def gl_matrix(c2w_opencv):
    return Matrix((np.asarray(c2w_opencv, dtype=float) @ CV_TO_GL).tolist())


def c2w_looking(position, forward):
    """수평 시선 `forward` 를 바라보는 OpenCV c2w. roll 0 (world +z 를 up 으로)."""
    fwd = np.asarray(forward, dtype=float)
    fwd = fwd / np.linalg.norm(fwd)
    down = np.array([0.0, 0.0, -1.0])
    right = np.cross(down, fwd)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2] = right, down, fwd
    c2w[:3, 3] = np.asarray(position, dtype=float)
    return c2w


def add_text(name, body, location, basis, size, material, collection):
    """관측 카메라를 **정확히** 마주 보는 3D 글자 (billboard). 폰트 파일이 필요 없다.

    회전을 오일러(x=90-elev, z=yaw)로 손계산하면 부호를 틀리기 쉽다 — 첫 실측에서 글자가
    180도 뒤집혀 거울상으로 나왔다. 관측 카메라의 회전 행렬을 **그대로** 씌우면 로컬 X/Y 가
    화면의 오른쪽/위와 정의상 일치하므로 각도와 무관하게 항상 정면이다.
    """
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = "CENTER"
    curve.extrude = size * 0.02
    curve.materials.append(material)
    obj = bpy.data.objects.new(name, curve)
    obj.matrix_world = Matrix.Translation(Vector(tuple(float(v) for v in location))) @ basis
    collection.objects.link(obj)
    return obj


def measure(scene, position, probe, subject_names):
    """한 걸음에서의 전방위 측정. **게이트 함수를 그대로 부른다.**"""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    clearance = clearance_of(scene, depsgraph, position, probe, subject_names)
    per_dir, names = [], []
    for direction in CLEARANCE_DIRS:
        hit, distance, name = ray(scene, depsgraph, position, direction, probe)
        if hit and name not in subject_names:
            per_dir.append(float(distance))
            names.append(name)
        else:
            per_dir.append(float(probe))
            names.append(None)
    horiz = [per_dir[i] for i, d in enumerate(CLEARANCE_DIRS) if d in HORIZ_DIRS]
    binding = int(np.argmin(per_dir))
    return {"clearance": float(clearance), "clearance_h": float(min(horiz)),
            "per_dir": per_dir, "hit_names": names,
            "binding_dir": list(CLEARANCE_DIRS[binding]), "binding_name": names[binding],
            "floor_is_binding": bool(CLEARANCE_DIRS[binding] == (0, 0, -1))}


def main(args):
    scene = bpy.context.scene
    with open(args.audit, encoding="utf-8") as file:
        audit = json.load(file)
    frame_list = [int(f) for f in audit["frame_list"]]
    subject_names = set(audit.get("subject_meshes") or [])
    subject_names |= set(audit.get("human_meshes") or [])
    if audit.get("armature"):
        subject_names.add(audit["armature"])
    floor_z = float(audit.get("floor_z") or 0.0)

    #    씬을 한 프레임에 고정한다. 행진하는 동안 사람이 움직이면 걸음마다 다른 씬을 재는 셈이라
    #    램프에 사람의 움직임이 섞인다 — 여기서 보려는 건 **거리 대 clearance** 하나뿐이다.
    frame = frame_list[args.start_frame] if args.frame < 0 else args.frame
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    # ── 시작점 -------------------------------------------------------------------------------
    if args.start:
        start = np.asarray(args.start, dtype=float)
    else:
        entry = audit["paths"][args.start_from_path][args.start_frame]
        start = np.asarray(entry["position"], dtype=float)
    if args.start_height > 0:
        start[2] = floor_z + args.start_height

    # ── 행진 방향: 수평 4방향을 다 재고 고른다 -------------------------------------------------
    probe = float(args.probe_distance)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    #    방향 탐색만 probe 보다 멀리 본다 — probe 1.5 m 로는 "벽이 3 m 밖" 인 방향이 전부
    #    동점(=1.5)으로 뭉개져서 고를 수가 없다. 측정은 아래에서 다시 probe 로 한다.
    scout = {}
    for direction in HORIZ_DIRS:
        hit, distance, name = ray(scene, depsgraph, start, direction, args.scout_distance)
        scout[direction] = (bool(hit), float(distance) if hit else float(args.scout_distance), name)
    if args.direction:
        march = np.asarray(args.direction, dtype=float)
        march[2] = 0.0
        march /= np.linalg.norm(march)
        hit, wall_dist, wall_name = ray(scene, depsgraph, start, tuple(march), args.scout_distance)
        wall_dist = float(wall_dist) if hit else float(args.scout_distance)
    else:
        hits = {d: v for d, v in scout.items() if v[0]}
        assert hits, f"시작점 {start} 에서 수평 4방향 모두 {args.scout_distance} m 안에 벽이 없다"
        #    nearest = 제일 가까운 벽(짧은 램프), farthest = 제일 먼 벽(긴 램프, 기본).
        #    긴 램프가 기본인 이유: 임계 4개가 다 걸리려면 여유 구간이 0.5 m 이상 있어야 한다.
        key = min if args.dir_mode == "nearest_wall" else max
        march_t = key(hits, key=lambda d: hits[d][1])
        march = np.asarray(march_t, dtype=float)
        wall_dist, wall_name = hits[march_t][1], hits[march_t][2]

    travel = max(0.0, wall_dist - float(args.stop_at))
    assert travel > 0.05, f"벽까지 {wall_dist:.3f} m 라 행진 구간이 없다 (stop_at {args.stop_at})"
    step_len = travel / max(1, args.steps - 1)

    # ── 1차: **측정만** (막대를 먼저 심으면 ray_cast 가 그 막대를 때린다 — rayviz 와 같은 함정)
    rows = []
    for i in range(args.steps):
        position = start + march * (step_len * i)
        row = measure(scene, position, probe, subject_names)
        row.update({"step": i, "travel": float(step_len * i),
                    "wall_gap": float(wall_dist - step_len * i),
                    "position": [float(v) for v in position]})
        rows.append(row)

    #    임계별 **최초 위반 걸음**. 램프가 단조롭지 않을 수 있으므로(가구를 스치면 잠깐 떨어졌다
    #    돌아온다) "처음 걸리는 지점"을 그대로 쓴다 — 게이트도 그렇게 판정한다.
    thresholds = sorted((float(t) for t in args.thresholds), reverse=True)
    crossings = []
    for k, thr in enumerate(thresholds):
        hitrow = next((r for r in rows if r["clearance"] < thr), None)
        hitrow_h = next((r for r in rows if r["clearance_h"] < thr), None)
        crossings.append({"threshold": thr, "color": POST_COLORS[k % len(POST_COLORS)],
                          "step": None if hitrow is None else hitrow["step"],
                          "travel": None if hitrow is None else hitrow["travel"],
                          "wall_gap": None if hitrow is None else hitrow["wall_gap"],
                          "step_h": None if hitrow_h is None else hitrow_h["step"],
                          "wall_gap_h": None if hitrow_h is None else hitrow_h["wall_gap"]})

    # ── 관측 카메라 (고정) ---------------------------------------------------------------------
    #    행진 축에 **수직**으로 서야 카메라가 벽에 다가가는 것이 옆모습으로 보인다. 행진 축을
    #    따라 서면 말뚝들이 서로 겹쳐 거리 차이가 안 보인다.
    end = start + march * travel
    mid = 0.5 * (start + end)
    side = np.cross(march, np.array([0.0, 0.0, 1.0]))
    side /= np.linalg.norm(side)
    #    두 옆면 중 **더 트인 쪽**에 선다. 막힌 쪽에 서면 벽이 화면을 통째로 덮는다
    #    (근평면 컷어웨이가 있긴 하지만 여유가 있는 쪽이 언제나 더 낫다).
    #    openness 는 **화면에 실제로 담기는 구간**에서 잰다. `--focus crossings` 면 그건 행진
    #    중점이 아니라 교차점 구간의 중점이고, a17 에서는 두 지점의 트인 쪽이 달랐다.
    hit_travels = [c["travel"] for c in crossings if c["travel"] is not None]
    probe_mid = mid
    if args.focus == "crossings" and hit_travels:
        probe_mid = start + march * (0.5 * (min(hit_travels) + max(hit_travels)))
    open_side = [ray(scene, depsgraph, probe_mid, tuple(s), args.scout_distance)
                 for s in (side, -side)]
    #    `auto` 가 기존 동작. 자동 판정이 옷장처럼 큰 소품 뒤를 고르는 경우가 있어(a17 eye_zoom
    #    실측: 화면 아래 절반이 장롱) 손으로 뒤집는 `flip` 을 열어 둔다.
    if args.obs_side == "flip":
        side = -side
    elif open_side[1][1] > open_side[0][1] or (open_side[0][0] and not open_side[1][0]):
        side = -side
    #    말뚝 + 층층이 쌓은 라벨이 행진 평면 위로 이만큼 솟는다. 이걸 계산에 안 넣으면 라벨이
    #    화면 위로 잘려 나간다 (첫 실측: 말뚝 4개 라벨이 전부 프레임 밖).
    stack_h = float(args.post_rise) + float(args.text_size) * (1.7 * len(crossings) + 3.8)
    #    `--focus crossings` 는 **말뚝이 실제로 박힌 구간만** 프레임에 담는다. 행진 전체를 담으면
    #    (a17 실측) 임계 4개가 마지막 0.42 m 안에 몰려서 5 m 짜리 방 안의 몇 픽셀이 된다 —
    #    "어디서 걸리는지"를 보려는 그림에서 정작 그 구간이 안 보인다.
    if args.focus == "crossings" and hit_travels:
        lo, hi = min(hit_travels), max(hit_travels)
        lo, hi = max(0.0, lo - args.focus_pad), min(travel, hi + args.focus_pad)
        focus_mid = start + march * (0.5 * (lo + hi))
        focus_span = max(hi - lo, 0.5)
    else:
        focus_mid, focus_span = mid, float(travel)
    look_at = focus_mid + np.array([0.0, 0.0, stack_h * 0.5])
    span = max(focus_span, stack_h * 1.3, 1.0)
    obs_r = args.obs_radius or span * args.obs_scale
    eye = (look_at + side * obs_r * np.cos(radians(args.obs_elev))
           + np.array([0.0, 0.0, 1.0]) * obs_r * np.sin(radians(args.obs_elev)))

    # ── 2차: 그린다 ---------------------------------------------------------------------------
    collection = bpy.data.collections.new("approach_viz")
    scene.collection.children.link(collection)
    materials = {"path": emission_material("ap_path", PATH_COLOR, args.emission),
                 "wall": emission_material("ap_wall", WALL_RAY_COLOR, args.emission),
                 "floor": emission_material("ap_floor", FLOOR_RAY_COLOR, args.emission),
                 "cur": emission_material("ap_cur", CUR_COLOR, args.emission)}
    for k, cross in enumerate(crossings):
        materials[f"post{k}"] = emission_material(f"ap_post{k}", cross["color"], args.emission)
    cur_ball = ball_mesh("ap_cur_ball", args.thickness * 3.0, materials["cur"])

    #    행진 경로 전체를 흰 선으로 — 말뚝이 어느 구간에 몰려 있는지가 이 선 위에서 읽힌다.
    add_tube("ap_path", start, end, args.thickness * 0.6, materials["path"], collection)
    #    벽면 표시: 끝점에서 벽까지 남은 토막.
    add_tube("ap_wallgap", end, start + march * wall_dist, args.thickness * 0.5,
             materials["wall"], collection)

    #    글자는 관측 카메라 회전을 그대로 써서 billboard 한다 (§add_text).
    text_rot = look_at_matrix(Vector(eye), Vector(look_at)).to_3x3().to_4x4()

    #    말뚝 라벨은 **임계 순서대로 층을 달리** 세운다. 임계가 가까이 붙어서 걸리면(a17 실측:
    #    0.35/0.20/0.10 이 전부 마지막 걸음) 라벨이 같은 자리에 겹쳐 하나도 못 읽는다.
    for k, cross in enumerate(crossings):
        if cross["step"] is None:
            continue
        base = np.asarray(rows[cross["step"]]["position"], dtype=float)
        rise = args.post_rise + k * args.text_size * 1.7
        foot = np.array([base[0], base[1], floor_z])
        top = np.array([base[0], base[1], base[2] + rise])
        add_tube(f"ap_post_{k}", foot, top, args.thickness * 1.6, materials[f"post{k}"],
                 collection, sides=8)
        add_ball(f"ap_post_ball_{k}",
                 ball_mesh(f"ap_post_ball_m_{k}", args.thickness * 3.4, materials[f"post{k}"]),
                 tuple(base), collection)
        add_text(f"ap_post_txt_{k}",
                 f"{cross['threshold']:.2f}",
                 top + np.array([0.0, 0.0, args.text_size * 0.8]), text_rot,
                 args.text_size, materials[f"post{k}"], collection)

    #    걸음별 지오메트리는 **걸음마다 따로 컬렉션**에 넣고 렌더 직전에 하나만 켠다.
    #    (모든 걸음의 광선을 한 화면에 그리면 40걸음 x 6발 = 240 막대라 아무것도 안 보인다.)
    lens, sensor = float(args.lens), float(args.sensor)
    half_x, half_y = half_angles(lens, sensor, args.res[0], args.res[1], args.frustum_len)
    frustum = frustum_mesh("ap_frustum", half_x, half_y, args.frustum_len, materials["cur"])
    step_cols = []
    for row in rows:
        sub = bpy.data.collections.new(f"ap_step_{row['step']:03d}")
        collection.children.link(sub)
        step_cols.append(sub)
        position = np.asarray(row["position"], dtype=float)
        add_ball(f"ap_cur_{row['step']:03d}", cur_ball, tuple(position), sub)
        add_wire(f"ap_cam_{row['step']:03d}", frustum, gl_matrix(c2w_looking(position, march)),
                 args.thickness, sub)
        for j, direction in enumerate(CLEARANCE_DIRS):
            distance = row["per_dir"][j]
            tip = position + np.asarray(direction, dtype=float) * distance
            #    행진 방향(벽) 과 아래(바닥) 는 판정을 만드는 두 축이라 굵게, 나머지는 가늘게.
            axis = tuple(int(v) for v in direction)
            is_march = bool(np.allclose(np.asarray(axis, dtype=float), march, atol=1e-6))
            key = "wall" if is_march else ("floor" if axis == (0, 0, -1) else "path")
            radius = args.thickness * (0.9 if key in ("wall", "floor") else 0.28)
            add_tube(f"ap_ray_{row['step']:03d}_{j}", position, tip, radius,
                     materials[key], sub)
        #    걸음별 판독값은 **화면상 같은 자리**(행진 중점 위)에 띄운다 — 카메라를 따라다니면
        #    말뚝 라벨과 겹쳐서(첫 실측) 둘 다 못 읽는다. HUD 처럼 한 곳에 고정한다.
        add_text(f"ap_lbl_{row['step']:03d}",
                 f"clr {row['clearance']:.2f}",
                 focus_mid + np.array([0.0, 0.0, args.post_rise
                                 + args.text_size * (1.7 * len(crossings) + 2.6)]),
                 text_rot, args.text_size * 1.15, materials["cur"], sub)

    # ── 렌더 설정 -----------------------------------------------------------------------------
    camd = bpy.data.cameras.new("ApproachCamData")
    camd.type = "PERSP"
    camd.lens = args.obs_lens
    camd.sensor_fit = "AUTO"
    camd.sensor_width, camd.sensor_height = 36.0, 24.0
    cam_obj = bpy.data.objects.new("ApproachCam", camd)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.matrix_world = look_at_matrix(Vector(eye), Vector(look_at))
    #    관측 카메라와 행진 사이의 벽을 근평면으로 잘라 낸다 (rayviz 와 같은 공짜 컷어웨이).
    near = 0.05 if args.clip_cut <= 0 else max(0.05, obs_r - span * args.clip_cut)
    camd.clip_start, camd.clip_end = near, 1000.0
    #    ⚠ 근평면만으로는 **천장을 못 걷어낸다**. 근평면은 시선에 수직인 평면이라 눈에서 먼 쪽
    #    천장은 그대로 남고, 부감 55도 에서 그게 화면의 90% 를 덮었다 (첫 실측: 채도 있는 픽셀
    #    887개가 전부 좌상단 구석, 나머지는 천장 한 장). 높이 기준으로 **오브젝트를 통째로**
    #    렌더에서 빼는 게 각도와 무관하게 확실하다.
    cut_z = float(mid[2]) + float(args.cut_above)
    hidden = 0
    if args.cut_above > 0:
        for obj in scene.objects:
            if obj.type != "MESH":
                continue
            corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            if min(c.z for c in corners) > cut_z:
                obj.hide_render = True
                hidden += 1
    print(f"[approach] cut_above z>{cut_z:.2f} m -> mesh {hidden}개 렌더 제외")

    if args.world == "dark":
        world = scene.world or bpy.data.worlds.new("approach_world")
        scene.world = world
        world.use_nodes = True
        world.node_tree.nodes.clear()
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        bg.inputs["Color"].default_value = (0.16, 0.16, 0.19, 1.0)
        bg.inputs["Strength"].default_value = float(args.world_strength)
        out_node = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs["Background"], out_node.inputs["Surface"])

    scene.render.resolution_x, scene.render.resolution_y = args.res
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = args.bounces
    scene.cycles.device = args.cdevice
    if args.cdevice == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        picked = None
        for api in ("OPTIX", "CUDA"):
            prefs.compute_device_type = api
            prefs.get_devices()
            if any(dv.type == api for dv in prefs.devices):
                for dv in prefs.devices:
                    dv.use = (dv.type == api)
                picked = api
                break
        print(f"[approach] cycles device GPU/{picked or 'NONE(-> CPU fallback)'}")

    frames_dir = path.join(args.out, "march")
    makedirs(frames_dir, exist_ok=True)
    for i, sub in enumerate(step_cols):
        for j, other in enumerate(step_cols):
            other.hide_render = (i != j)
        scene.render.filepath = path.join(frames_dir, f"frame_{i:05d}")
        bpy.ops.render.render(write_still=True)

    summary = {"format": "trumans_approach_viz_v1", "blend": bpy.data.filepath,
               "audit": path.abspath(args.audit), "frame": int(frame), "floor_z": floor_z,
               "start": [float(v) for v in start], "march_dir": [float(v) for v in march],
               "wall_distance": float(wall_dist), "wall_name": wall_name,
               "travel": float(travel), "step_len": float(step_len), "steps": int(args.steps),
               "probe_distance": probe, "stop_at": float(args.stop_at),
               "scout": {str(d): [v[0], v[1], v[2]] for d, v in scout.items()},
               "observer": {"eye": [float(v) for v in eye],
                            "target": [float(v) for v in look_at],
                            "radius": float(obs_r), "elev_deg": float(args.obs_elev),
                            "clip_start": float(near), "cut_z": float(cut_z),
                            "hidden_meshes": int(hidden)},
               "focus": {"mode": args.focus, "center": [float(v) for v in focus_mid],
                         "span": float(focus_span)},
               "crossings": crossings, "rows": rows,
               "floor_binding_frac": float(np.mean([r["floor_is_binding"] for r in rows])),
               "frames_dir": path.abspath(frames_dir)}
    with open(path.join(args.out, "approach.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)

    print(f"\n{'start':<18}{np.round(start, 3).tolist()}   frame {frame}")
    print(f"{'march dir':<18}{np.round(march, 3).tolist()}  -> {wall_name}  "
          f"{wall_dist:.3f} m")
    print(f"{'travel':<18}{travel:.3f} m  /  {args.steps} steps  ({step_len * 100:.1f} cm/step)")
    print(f"{'floor binding':<18}{summary['floor_binding_frac'] * 100:.1f}% of steps")
    print(f"\n{'threshold':>10}{'step':>7}{'moved':>9}{'wall gap':>10}"
          f"{'step(h)':>9}{'wall gap(h)':>13}")
    print("-" * 58)
    for cross in crossings:
        step_h = "--" if cross["step_h"] is None else str(cross["step_h"])
        gap_h = "--" if cross["wall_gap_h"] is None else f"{cross['wall_gap_h']:.3f}"
        if cross["step"] is None:
            step_s, travel_s, gap_s = "--", "never", "--"
        else:
            step_s = str(cross["step"])
            travel_s = f"{cross['travel']:.3f}"
            gap_s = f"{cross['wall_gap']:.3f}"
        print(f"{cross['threshold']:>10.2f}{step_s:>7}{travel_s:>9}"
              f"{gap_s:>10}{step_h:>9}{gap_h:>13}")
    print(f"\nPNG  -> {frames_dir}")
    print(f"JSON -> {path.join(args.out, 'approach.json')}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    # `bank_to_blender_poses.py --raycast` 가 낸 감사 JSON. subject mesh 이름 / floor_z / 프레임 격자.
    parser.add_argument("--audit", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    # 시작점: 명시하거나(--start x y z) 감사의 궤적에서 집는다.
    parser.add_argument("--start", nargs=3, default=None, type=float)
    parser.add_argument("--start_from_path", default=0, type=int)
    parser.add_argument("--start_frame", default=0, type=int)
    # 시작 고도를 floor_z 위 고정값으로 덮어쓴다 (0 이면 원래 pose 고도 유지).
    parser.add_argument("--start_height", default=0.0, type=float)
    # 씬 애니메이션 프레임. -1 이면 frame_list[start_frame].
    parser.add_argument("--frame", default=-1, type=int)
    # 행진 방향. 명시하거나 수평 4방향 중 고른다.
    parser.add_argument("--direction", nargs=3, default=None, type=float)
    parser.add_argument("--dir_mode", default="farthest_wall",
                        choices=["farthest_wall", "nearest_wall"])
    # 방향 탐색 전용 사거리. 측정(clearance)은 --probe_distance 로 한다.
    parser.add_argument("--scout_distance", default=8.0, type=float)
    parser.add_argument("--steps", default=40, type=int)
    # 벽 앞 몇 m 에서 멈출지. 0 이면 벽에 파묻힌다.
    parser.add_argument("--stop_at", default=0.05, type=float)
    parser.add_argument("--probe_distance", default=1.5, type=float)
    parser.add_argument("--thresholds", nargs="+", default=[0.50, 0.35, 0.20, 0.10], type=float)
    parser.add_argument("--post_rise", default=0.55, type=float)
    parser.add_argument("--text_size", default=0.11, type=float)
    parser.add_argument("--lens", default=32.0, type=float)
    parser.add_argument("--sensor", default=36.0, type=float)
    parser.add_argument("--frustum_len", default=0.30, type=float)
    parser.add_argument("--thickness", default=0.012, type=float)
    parser.add_argument("--emission", default=4.0, type=float)
    parser.add_argument("--obs_radius", default=0.0, type=float)   # 0 = travel x obs_scale
    parser.add_argument("--obs_scale", default=2.2, type=float)
    parser.add_argument("--obs_elev", default=22.0, type=float)
    #    관측 카메라가 설 옆면. auto = 더 트인 쪽(기존 동작), flip = 그 반대편.
    parser.add_argument("--obs_side", default="auto", choices=["auto", "flip"])
    parser.add_argument("--obs_lens", default=32.0, type=float)
    parser.add_argument("--clip_cut", default=1.15, type=float)
    # 행진 높이 + 이 값보다 위에 **통째로** 있는 mesh 는 렌더에서 뺀다 (천장 걷어내기). 0 이면 끔.
    parser.add_argument("--cut_above", default=0.7, type=float)
    # 프레임을 행진 전체(march)에 맞출지 **말뚝이 박힌 구간**(crossings)에 맞출지.
    parser.add_argument("--focus", default="march", choices=["march", "crossings"])
    parser.add_argument("--focus_pad", default=0.35, type=float)
    parser.add_argument("--world", default="keep", choices=["keep", "dark"])
    parser.add_argument("--world_strength", default=1.0, type=float)
    parser.add_argument("--res", nargs=2, default=[960, 540], type=int)
    parser.add_argument("--samples", default=24, type=int)
    parser.add_argument("--bounces", default=4, type=int)
    # `--cycles*` 이름 금지 (Cycles 애드온이 argv 를 prefix-match 로 훑는다).
    parser.add_argument("--cdevice", default="GPU", choices=["GPU", "CPU"])
    main(parser.parse_args(cli_argv()))
