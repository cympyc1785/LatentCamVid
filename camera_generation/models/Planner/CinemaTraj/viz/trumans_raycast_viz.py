"""D91 레이캐스트 게이트가 **실제로 쏘는 광선을 씬 안에 그려서** 렌더한다.

**왜 필요한가.** `bank_to_blender_poses.py --raycast` 는 target 궤적 하나당 숫자 4개만 남긴다 —
`clear_frac` / `min_clearance` / `min_subject_dist` / `min_floor_drop`. 이 숫자로는 "왜 떨어졌나"를
못 읽는다. `wall(0.20)` 이 벽에 붙은 건지 소파 등받이인지 사람 손인지, `occluded(0.83)` 이 어느
프레임 어느 조준점이 막힌 건지가 전부 같은 스칼라로 뭉개진다. 게이트는 근사가 아니라 진짜
`scene.ray_cast` 라서 **그림으로 그리면 그대로 보인다** — 광선이 어디서 끊기는지가 곧 판정이다.

그려지는 것은 `trumans_scene_probe.py` 가 쓰는 함수 **그 자체**다 (`ray` / `line_of_sight` /
`clearance_of` 를 import 한다). 다시 구현하지 않으므로 그림과 판정이 어긋날 수 없다.

    시선(LOS)   카메라 -> 조준점 3개. 처음 맞는 게 subject 면 초록(뚫림), 아니면 막힌 지점까지
                주황 + 구슬. **반대 방향으로 쏘면 안 된다** (몸 속에서 시작해 자기 mesh 를 때린다).
    clearance   6방향(±x, ±y, ±z) 최대 `--probe_distance`. subject mesh 히트는 무시한다.
                `--min_clearance` 미만이면 빨강, 그 이상 히트는 파랑, 히트 없으면 흐린 회색으로
                끝까지. **벽 두께와 무관**한 게 이 게이트의 요점이다 (a08 벽은 0.0007 u 라
                OBB signed distance 로는 원리적으로 못 잡는다).
    floor       바로 아래로 10 m. 히트 거리가 `min_floor_drop` = 카메라 고도.
    궤적        전 프레임 위치를 흰 선으로 잇고, 광선을 쏜 프레임에만 절두체를 세운다.

`--hide_scene` 이면 씬 mesh 를 렌더에서 빼고 subject 만 남긴다 — 광선이 벽에 가려 안 보일 때 쓴다
(가려지는 것 자체가 판정이라 기본은 씬을 켜 둔다).

**주의: `--cycles` 로 시작하는 CLI 플래그 금지** (Cycles 애드온이 argv 를 prefix-match 로 훑어서
실행이 통째로 죽는다 — 그래서 `--cdevice`).

영상은 안 만든다 (Blender 내장 python 에 imageio 가 없다). PNG 시퀀스만 떨구고 `concat_videos.py`
/ `stack_videos.py` 가 PNG 디렉토리를 받으므로 그쪽에서 libx264 로 묶는다.

env: Blender 내장 python

예시:
    B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/0ac97866-dccb-47c3-b220-79712041e187
    $B -b $R/0ac97866-dccb-47c3-b220-79712041e187.blend --python viz/trumans_raycast_viz.py -- \
        --poses /tmp/blendcam_a08_pushin/push_in_arc_left.npz \
        --audit /tmp/blendcam_a08_pushin/raycast_audit.json \
        --out /tmp/rayviz_arc --ray_stride 6 --orbit_frames 72
"""
import json
import sys
from argparse import ArgumentParser
from math import cos, pi, radians, sin
from os import makedirs, path

import bpy
import numpy as np
from mathutils import Matrix, Vector

SCRIPTS_DIR = path.dirname(path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
CINEMATRAJ_ROOT = path.dirname(SCRIPTS_DIR)
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from trumans_frustum_viz import (add_ball, add_wire, ball_mesh,  # noqa: E402
                                 emission_material, frustum_mesh, half_angles)
from fit.ingest.trumans_scene_probe import (CLEARANCE_DIRS, clearance_of,  # noqa: E402
                                            line_of_sight, ray)

#    OpenCV c2w (X right / Y down / Z fwd) -> Blender GL (X right / Y up / -Z fwd).
#    `trumans_gt_render.py:537` 이 기록한 규약과 같은 행렬이다.
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])

LOS_CLEAR = (0.10, 1.00, 0.25)          # 시선이 subject 에 닿았다
LOS_BLOCKED = (1.00, 0.45, 0.00)        # 다른 mesh 가 먼저 맞았다
CLR_NEAR = (1.00, 0.08, 0.08)           # clearance < 임계 (벽에 붙음)
CLR_OK = (0.20, 0.55, 1.00)             # clearance 히트지만 임계 이상
CLR_FREE = (0.45, 0.45, 0.50)           # probe_distance 안에 아무것도 없다
FLOOR_COLOR = (0.75, 0.25, 1.00)
PATH_COLOR = (1.00, 1.00, 1.00)
AIM_COLOR = (1.00, 0.95, 0.20)


def cli_argv():
    """`blender -b x.blend --python me.py -- <args>` 의 `--` 뒤만 argparse 에 넘긴다."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def tube_mesh(name: str, a, b, radius: float, material, sides: int = 6):
    """a -> b 를 잇는 각기둥. **월드 좌표로 직접** 만든다 (광선 수십 개라 공유할 이유가 없다).

    `add_wire` 의 Wireframe modifier 를 안 쓰는 이유: 광선은 속이 찬 막대로 보여야 굵기가
    거리감을 준다. 와이어로 만들면 막대의 테두리만 남아 겹칠 때 뭐가 뭔지 안 보인다.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None
    axis = axis / length
    #    axis 와 안 나란한 아무 벡터로 직교 기저를 만든다 (수직 광선에서 z 를 쓰면 터진다).
    seed = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    verts, faces = [], []
    for i in range(sides):
        angle = 2.0 * pi * i / sides
        offset = radius * (cos(angle) * u + sin(angle) * v)
        verts.append(tuple(a + offset))
    for i in range(sides):
        angle = 2.0 * pi * i / sides
        offset = radius * (cos(angle) * u + sin(angle) * v)
        verts.append(tuple(b + offset))
    for i in range(sides):
        j = (i + 1) % sides
        faces.append((i, j, j + sides, i + sides))
    faces.append(tuple(range(sides)))
    faces.append(tuple(range(2 * sides - 1, sides - 1, -1)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(material)
    return mesh


def add_tube(name, a, b, radius, material, collection, sides: int = 6):
    mesh = tube_mesh(name, a, b, radius, material, sides)
    if mesh is None:
        return None
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def gl_matrix(c2w_opencv):
    """OpenCV c2w (4,4) -> Blender `matrix_world`."""
    return Matrix((np.asarray(c2w_opencv, dtype=float) @ CV_TO_GL).tolist())


def pick_frames(count: int, stride: int, must_have=()):
    """광선을 **그릴** 프레임 인덱스. 끝 프레임과 `must_have` 는 stride 와 무관하게 항상 넣는다.

    측정은 전 프레임에서 한다 (§main). 그리는 것만 솎는 이유는 49프레임 x 10발 = 490 막대가
    화면을 통째로 덮기 때문이고, 솎은 격자에 최악 프레임이 안 걸리면 **그림이 게이트보다 관대해
    보인다** — 실측: stride 16 에서 min clearance 가 0.001 이 아니라 0.088 로 나왔다.
    그래서 최악 프레임을 `must_have` 로 강제로 끼워 넣는다.
    """
    picked = sorted({*range(0, count, max(1, stride)), count - 1, *must_have})
    return [i for i in picked if 0 <= i < count]


def main(args):
    scene = bpy.context.scene
    with open(args.audit, encoding="utf-8") as file:
        audit = json.load(file)
    poses = np.asarray(np.load(args.poses)["cam_c2w"], dtype=np.float64)
    assert poses.ndim == 3 and poses.shape[1:] == (4, 4), f"cam_c2w 모양이 이상하다: {poses.shape}"

    frame_list = [int(f) for f in audit["frame_list"]]
    assert len(frame_list) == len(poses), \
        f"프레임 {len(frame_list)}개인데 pose 는 {len(poses)}개다 — audit 와 npz 가 짝이 아니다"
    #    `subject_names` 는 probe 와 같은 집합이어야 한다 — clearance 면제와 LOS 판정이 둘 다
    #    이 이름으로 갈린다. armature 는 렌더는 안 되지만 ray_cast 히트 이름으로는 나온다.
    subject_names = set(audit.get("subject_meshes") or [])
    subject_names |= set(audit.get("human_meshes") or [])
    if audit.get("armature"):
        subject_names.add(audit["armature"])
    aims_by_frame = {int(e["frame"]): e["aim_points"] for e in audit["human_track"]}

    collection = bpy.data.collections.new("raycast_viz")
    scene.collection.children.link(collection)

    if args.hide_scene:
        hidden = 0
        for obj in scene.objects:
            if obj.type == "MESH" and obj.name not in subject_names:
                obj.hide_render = True
                hidden += 1
        print(f"[rayviz] hide_scene: mesh {hidden}개 렌더 제외 (subject {len(subject_names)}개 유지)")

    materials = {key: emission_material(f"ray_{key}", rgb, args.emission)
                 for key, rgb in (("los_clear", LOS_CLEAR), ("los_blocked", LOS_BLOCKED),
                                  ("clr_near", CLR_NEAR), ("clr_ok", CLR_OK),
                                  ("clr_free", CLR_FREE), ("floor", FLOOR_COLOR),
                                  ("path", PATH_COLOR), ("aim", AIM_COLOR))}
    #    위반 광선은 **길이가 곧 위반의 크기**라서 (`end = position + dir * distance`, 아래)
    #    clearance 0.037 m 는 3.7 cm 짜리 토막으로 그려진다 — 반면 통과 광선은 최대
    #    `probe_distance` 1.5 m 다. 실측: 900px 렌더에서 빨강 2 px vs 파랑 353 px 로,
    #    **찾으려는 것이 화면에서 가장 작은 자국**이 된다. `--near_emphasis` 로 이걸 뒤집는다.
    #    `none` 이 기존 동작이며 그때 결과는 비트 동일하다.
    near_ball = args.thickness * 2.2 * (1.0 if args.near_emphasis in ("none", "stripe")
                                        else args.near_ball_scale)
    balls = {key: ball_mesh(f"ball_{key}",
                            near_ball if key == "clr_near" else args.thickness * 2.2,
                            material)
             for key, material in materials.items()}

    lens, sensor = float(args.lens), float(args.sensor)
    half_x, half_y = half_angles(lens, sensor, args.res[0], args.res[1], args.frustum_len)
    frustum = frustum_mesh("frustum_cam", half_x, half_y, args.frustum_len, materials["path"])

    counts = {"los_clear": 0, "los_blocked": 0, "clr_near": 0, "clr_ok": 0, "clr_free": 0}
    rows = []
    geometry = {}          # index -> (segments, points).  그릴 프레임을 고른 뒤에 꺼내 쓴다

    # ── 1차: **측정만** 한다. 막대를 하나라도 먼저 심으면 `scene.ray_cast` 가 그 막대를 때린다
    #    (첫 실측: 카메라에서 출발하는 궤적 선 때문에 전 광선이 거리 0 히트 -> clearance 0.000,
    #    LOS 전량 blocked). depsgraph 는 씬 지오메트리를 가리지 않고 전부 본다.
    #
    #    측정은 **전 프레임**에서 한다. 솎아서 재면 그림의 숫자가 게이트와 달라진다 — 실측:
    #    stride 16(4프레임)에서 min clearance 0.088 / clear_frac 1.00 이 나왔는데 같은 궤적의
    #    49프레임 probe 값은 0.001 / 0.71 이었다. 그림이 게이트보다 관대하면 볼 이유가 없다.
    for index in range(len(poses)):
        segments, points = [], []
        frame = frame_list[index]
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        position = poses[index][:3, 3]
        aims = aims_by_frame[frame]

        # 시선(LOS) 3발. `line_of_sight` 와 같은 순서·같은 판정으로 하나씩 다시 쏜다.
        clear_any, _ = line_of_sight(scene, depsgraph, position, aims, subject_names)
        for k, target in enumerate(aims):
            direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
            distance = float(np.linalg.norm(direction))
            hit, hit_at, name = ray(scene, depsgraph, position, direction, distance + 0.05)
            if hit and name in subject_names:
                key, end = "los_clear", np.asarray(target, dtype=float)
            elif hit:
                key = "los_blocked"
                end = np.asarray(position) + direction / distance * hit_at
            else:
                #    아무것도 안 맞았다 = subject 도 못 맞혔다. `line_of_sight` 는 이걸 "안 뚫림"
                #    으로 친다 (조준점이 mesh 밖으로 삐져나온 프레임에서 나온다).
                key, end = "los_blocked", np.asarray(target, dtype=float)
            counts[key] += 1
            segments.append((f"los_{index:03d}_{k}", position, end, args.thickness, key))
            points.append((f"los_{index:03d}_{k}_p", end, key))
            points.append((f"aim_{index:03d}_{k}", np.asarray(target, dtype=float), "aim"))

        # clearance 6발. subject 히트는 probe 와 똑같이 무시한다.
        clearance = clearance_of(scene, depsgraph, position, float(args.probe_distance),
                                 subject_names)
        for k, direction in enumerate(CLEARANCE_DIRS):
            hit, distance, name = ray(scene, depsgraph, position, direction,
                                      float(args.probe_distance))
            if hit and name not in subject_names:
                key = "clr_near" if distance < args.min_clearance else "clr_ok"
            else:
                key, distance = "clr_free", float(args.probe_distance)
            end = np.asarray(position) + np.asarray(direction, dtype=float) * distance
            counts[key] += 1
            #    히트가 없는 광선은 6방향 x 프레임 수만큼 1.5 m 짜리 막대가 되어 화면을 통째로
            #    덮는다 (첫 실측: 19/34 가 free 라 판정색이 안 보였다). 가늘게 그린다.
            radius = args.thickness * (0.25 if key == "clr_free" else 0.7)
            if key == "clr_near" and args.near_emphasis in ("stripe", "both"):
                #    위반 광선만 **임계 길이까지** 늘려 그린다 — "이 방향으로 임계만큼의 여유가
                #    없다"는 뜻이라 임계가 곧 눈금이다. 실제 표면 위치는 아래 ball 이 표시하므로
                #    거리 정보는 안 잃는다. 막대 길이를 거리로 읽으면 안 된다.
                stripe_end = (np.asarray(position)
                              + np.asarray(direction, dtype=float) * float(args.min_clearance))
                segments.append((f"clr_{index:03d}_{k}", position, stripe_end,
                                 args.thickness * 1.4, key))
            else:
                segments.append((f"clr_{index:03d}_{k}", position, end, radius, key))
            if key != "clr_free":
                points.append((f"clr_{index:03d}_{k}_p", end, key))

        # 바닥. 히트 거리가 그대로 카메라 고도다.
        _, floor_drop, _ = ray(scene, depsgraph, position, (0, 0, -1), 10.0)
        floor_drop = min(floor_drop, 10.0)
        segments.append((f"floor_{index:03d}", position,
                         np.asarray(position) + np.array([0.0, 0.0, -floor_drop]),
                         args.thickness * 0.5, "floor"))

        rows.append({"index": index, "frame": frame, "clear": bool(clear_any),
                     "clearance": float(clearance), "floor_drop": float(floor_drop),
                     "position": [float(v) for v in position]})
        geometry[index] = (segments, points)

    # ── 2차: 측정이 끝난 뒤에 그린다. 최악 프레임(최소 clearance / 최소 floor drop / 시선이
    #    막힌 첫 프레임)은 stride 격자에 안 걸려도 강제로 그린다 — 판정을 만든 프레임이
    #    그림에 없으면 안 된다.
    worst = [min(rows, key=lambda r: r["clearance"])["index"],
             min(rows, key=lambda r: r["floor_drop"])["index"]]
    blocked = [r["index"] for r in rows if not r["clear"]]
    if blocked:
        worst.append(blocked[len(blocked) // 2])
    drawn = pick_frames(len(poses), args.ray_stride, must_have=worst)
    for index in drawn:
        segments, points = geometry[index]
        for name, start, end, radius, key in segments:
            add_tube(name, start, end, radius, materials[key], collection)
        for name, center, key in points:
            add_ball(name, balls[key], tuple(center), collection)
    for index in drawn:
        add_wire(f"cam_{index:03d}", frustum, gl_matrix(poses[index]), args.thickness, collection)
    #    궤적 전체를 흰 선으로. 광선을 쏘는 프레임만 보면 카메라가 그 사이에서 어디를 지나갔는지가
    #    빠진다 — 벽을 스치는 건 대개 중간 프레임이다.
    for i in range(len(poses) - 1):
        add_tube(f"path_{i:03d}", poses[i][:3, 3], poses[i + 1][:3, 3],
                 args.thickness * 0.55, materials["path"], collection)

    # --- 오버뷰 카메라 -----------------------------------------------------------------------
    camd = bpy.data.cameras.new("OrbitCamData")
    camd.type = "PERSP"
    camd.lens = args.orbit_lens
    camd.sensor_fit = "AUTO"
    camd.sensor_width, camd.sensor_height = 36.0, 24.0
    cam_obj = bpy.data.objects.new("OrbitCam", camd)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    #    오버뷰는 궤적 + 조준점을 다 담아야 한다. 중심은 둘의 중점, 반경은 그 중심에서 가장 먼 점.
    subject_center = np.mean([np.mean(aims_by_frame[frame_list[i]], axis=0) for i in drawn], axis=0)
    center = 0.5 * (poses[:, :3, 3].mean(axis=0) + subject_center)
    spread = max(float(np.linalg.norm(poses[:, :3, 3] - center, axis=1).max()),
                 float(np.linalg.norm(subject_center - center)), 0.5)
    orbit_r = args.orbit_radius or spread * args.orbit_scale
    #    씬을 켠 채 밖에서 돌면 가까운 벽이 화면을 통째로 막는다. near clip 을 궤적 앞까지 밀면
    #    카메라와 궤적 사이의 벽만 잘려 나간다 — 공짜 컷어웨이다. `--clip_cut 0` 이면 끈다.
    near = 0.05 if args.clip_cut <= 0 else max(0.05, orbit_r - spread * args.clip_cut)
    camd.clip_start, camd.clip_end = near, 1000.0
    if args.world == "dark":
        world = scene.world or bpy.data.worlds.new("rayviz_world")
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
        print(f"[rayviz] cycles device GPU/{picked or 'NONE(-> CPU fallback)'}")

    #    사람은 광선을 쏜 프레임 중 **가운데** 자세로 세워 둔다. 오버뷰가 한 바퀴 도는 동안
    #    사람이 움직이면 광선(정지)과 몸(움직임)이 어긋나 보인다.
    scene.frame_set(frame_list[drawn[len(drawn) // 2]])
    bpy.context.view_layer.update()

    frames_dir = path.join(args.out, "orbit")
    makedirs(frames_dir, exist_ok=True)
    orbit_z = radians(args.orbit_elev)
    for step in range(args.orbit_frames):
        azimuth = 2.0 * pi * step / args.orbit_frames + radians(args.orbit_az0)
        eye = Vector(center) + Vector((orbit_r * cos(azimuth) * cos(orbit_z),
                                       orbit_r * sin(azimuth) * cos(orbit_z),
                                       orbit_r * sin(orbit_z)))
        from trumans_frustum_viz import look_at_matrix
        cam_obj.matrix_world = look_at_matrix(eye, Vector(center))
        scene.render.filepath = path.join(frames_dir, f"frame_{step:05d}")
        bpy.ops.render.render(write_still=True)

    summary = {"format": "trumans_raycast_viz_v1", "poses": path.abspath(args.poses),
               "audit": path.abspath(args.audit), "blend": bpy.data.filepath,
               "n_frames": int(len(poses)), "measured_frames": int(len(rows)),
               "drawn_frames": drawn, "forced_frames": sorted(set(worst)),
               "gates": {"probe_distance": args.probe_distance,
                         "min_clearance": args.min_clearance},
               "counts": counts, "rows": rows,
               "clear_frac": float(np.mean([r["clear"] for r in rows])),
               "min_clearance_seen": float(min(r["clearance"] for r in rows)),
               "min_floor_drop_seen": float(min(r["floor_drop"] for r in rows)),
               "orbit": {"radius": orbit_r, "elev_deg": args.orbit_elev,
                         "frames": args.orbit_frames, "center": [float(v) for v in center]},
               "frames_dir": path.abspath(frames_dir)}
    with open(path.join(args.out, "rayviz.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)

    print(f"\n{'poses':<18}{path.basename(args.poses)}   {len(poses)}프레임 전량 측정")
    print(f"{'그린 프레임':<18}{drawn}")
    print(f"{'강제 포함':<18}{sorted(set(worst))}  (최소 clearance / 최소 floor drop / 막힌 시선)")
    print(f"{'게이트':<18}probe_distance {args.probe_distance:g}m  "
          f"min_clearance {args.min_clearance:g}m")
    print(f"\n{'광선 종류(전 프레임)':<34}{'개수':>6}")
    print("-" * 42)
    for key, count in counts.items():
        print(f"{key:<34}{count:>6d}")
    print(f"\n{'clear_frac':<18}{summary['clear_frac']:.3f}")
    print(f"{'min clearance':<18}{summary['min_clearance_seen']:.3f} m")
    print(f"{'min floor drop':<18}{summary['min_floor_drop_seen']:.3f} m")
    print(f"\nPNG  -> {frames_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    # `bank_to_blender_poses.py` 가 낸 preset npz (key `cam_c2w`, (F,4,4) blend-world OpenCV).
    parser.add_argument("--poses", required=True, type=str)
    # 같은 폴더의 `raycast_audit.json` — subject mesh 이름 · 조준점 · 프레임 격자를 여기서 읽는다.
    parser.add_argument("--audit", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    # 광선을 그릴 프레임 간격. 전 프레임에 그리면 광선 490발이라 아무것도 안 보인다.
    parser.add_argument("--ray_stride", default=6, type=int)
    # 게이트 임계 — `bank_to_blender_poses.py` 의 기본값과 같아야 그림이 판정과 일치한다.
    parser.add_argument("--probe_distance", default=1.5, type=float)
    parser.add_argument("--min_clearance", default=0.20, type=float)      # D120
    # 절두체 화각. 렌더에 쓴 렌즈와 같게 줄 것 (`selection.json` 의 `lens`).
    parser.add_argument("--lens", default=32.0, type=float)
    parser.add_argument("--sensor", default=36.0, type=float)
    parser.add_argument("--frustum_len", default=0.30, type=float)
    parser.add_argument("--thickness", default=0.012, type=float)
    #    위반(clr_near) 광선을 눈에 띄게 하는 방식. `none` = 기존 동작(비트 동일).
    #    `ball` 끝점 구슬만 키움 / `stripe` 막대를 임계 길이까지 / `both` 둘 다.
    parser.add_argument("--near_emphasis", default="none",
                        choices=["none", "ball", "stripe", "both"])
    parser.add_argument("--near_ball_scale", default=4.0, type=float)
    parser.add_argument("--emission", default=4.0, type=float)
    parser.add_argument("--hide_scene", dest="hide_scene", action="store_true")
    parser.add_argument("--no_hide_scene", dest="hide_scene", action="store_false")
    parser.set_defaults(hide_scene=False)
    parser.add_argument("--orbit_frames", default=72, type=int)
    parser.add_argument("--orbit_radius", default=0.0, type=float)   # 0 = spread x orbit_scale
    parser.add_argument("--orbit_scale", default=2.4, type=float)
    parser.add_argument("--orbit_elev", default=24.0, type=float)
    parser.add_argument("--orbit_az0", default=0.0, type=float)
    parser.add_argument("--orbit_lens", default=32.0, type=float)
    parser.add_argument("--clip_cut", default=1.35, type=float)
    parser.add_argument("--world", default="keep", choices=["keep", "dark"])
    parser.add_argument("--world_strength", default=1.0, type=float)
    parser.add_argument("--res", nargs=2, default=[960, 540], type=int)
    parser.add_argument("--samples", default=24, type=int)
    parser.add_argument("--bounces", default=4, type=int)
    # `--cycles*` 이름 금지 (Cycles 애드온이 argv 를 prefix-match 로 훑는다).
    parser.add_argument("--cdevice", default="GPU", choices=["GPU", "CPU"])
    main(parser.parse_args(cli_argv()))
