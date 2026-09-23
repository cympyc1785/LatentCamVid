"""TRUMANS `.blend` 안에서 **사람 주위 구면 후보 카메라들을 frustum 와이어로 세워** 렌더한다.

**왜 필요한가.** `trumans_first_pose_board.py` 는 후보를 방위각 x 고도 x 반경 격자로 깔고 게이트
4종(가림·프레이밍·충돌·근접)을 먹인 뒤 **통과한 것만 한 장씩 렌더**한다. 그 결과물은 "이 후보에서
보면 이렇게 보인다"는 답은 주지만 **"후보들이 씬 안에서 어디에 어떻게 박혀 있나"** 는 안 보여준다.
어느 방향이 통째로 벽에 먹혔는지, 반경 사다리가 방을 넘어가는지, 고도 45° 칸이 천장을 뚫는지는
후보를 씬 안에 **동시에** 세워봐야 보인다 — 후보 이미지 192장을 아무리 넘겨도 안 보인다.

그래서 이 스크립트는 board 후보를 읽어 **하나도 렌더하지 않고**, 대신 후보마다 카메라 절두체를
와이어 메시로 만들어 씬에 심고, 그 전체를 **바깥에서 도는 오버뷰 카메라**로 찍는다. 판정은
색이다 — 통과는 초록, 탈락은 사유별 색. 벽 뒤 후보는 벽에 가려 안 보이는 게 정상이고 그게 곧
"이 방위는 못 쓴다"는 그림이다. 씬 없이 후보 구면만 보고 싶으면 `--hide_scene`.

**절두체는 실제 렌즈 그대로**다. board 의 `lens_mm`/`sensor_mm`/`res` 로 화각을 계산하므로
와이어의 벌어진 각이 그 후보가 실제로 담는 화각이다. 길이만 `--frustum_len` 으로 줄여 그린다
(안 그러면 절두체끼리 겹쳐 아무것도 안 보인다).

**Wireframe modifier 를 쓰려면 면이 있어야 한다.** verts+edges 만 있는 메시엔 modifier 가 아무
두께도 못 만든다 — Cycles 에서 통째로 안 보인다. 그래서 절두체를 옆면 4 + 밑면 1 의 **면**으로
만들고 Wireframe modifier 로 두께를 준다.

**주의: `--cycles` 로 시작하는 CLI 플래그를 절대 만들지 말 것** (Cycles 애드온이 `--` 를 무시하고
argv 를 prefix-match 로 훑어서 실행이 통째로 죽는다 — 그래서 `--cdevice`).

영상은 이 스크립트가 안 만든다 (Blender 내장 python 엔 imageio 가 없다). PNG 시퀀스만 떨구고,
`concat_videos.py` / `stack_videos.py` 가 PNG 디렉토리를 받으므로 그쪽에서 libx264 로 묶는다.

env: Blender 내장 python

예시:
    B=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    R=/data1/cympyc1785/data/trumans/Data_release/Recordings_blend/0ac97866-dccb-47c3-b220-79712041e187
    $B -b $R/0ac97866-dccb-47c3-b220-79712041e187.blend --python viz/trumans_frustum_viz.py -- \
        --board out/sweep66/0ac97866-dccb-47c3-b220-79712041e187/board.json \
        --chunk 0 --out /tmp/frustum_a08 --orbit_frames 72
"""
import json
import sys
from argparse import ArgumentParser
from math import atan2, cos, degrees, pi, radians, sin
from os import makedirs, path

import bpy
from mathutils import Matrix, Vector

#    탈락 사유 -> 색. 여러 사유가 겹치면 이 순서로 **먼저 걸리는 것**을 쓴다 — 벽 속인 후보가
#    "가림"으로도 찍히는데(자기 벽에 막혀서) 진짜 원인은 벽이므로 clearance 가 앞이다.
REJECT_COLORS = [
    ("clearance", (1.00, 0.10, 0.10), "wall / 벽 속·벽에 붙음"),
    ("below_ground", (0.55, 0.25, 0.05), "ground / 지면 아래"),
    ("outside", (0.55, 0.25, 0.05), "outside / 씬 밖"),
    ("subject", (1.00, 0.10, 0.80), "subject / 피사체에 너무 붙음"),
    ("occluded", (1.00, 0.55, 0.00), "occluded / 시선 막힘"),
    ("cropped", (0.15, 0.45, 1.00), "cropped / 프레이밍 탈락"),
    ("area", (0.15, 0.45, 1.00), "framing / 면적비 탈락"),
    ("center", (0.15, 0.45, 1.00), "framing / 중심 이탈"),
]
PASS_COLOR = (0.10, 1.00, 0.25)
RING_COLOR = (0.35, 0.35, 0.40)
AIM_COLOR = (1.00, 0.95, 0.20)


def cli_argv():
    """`blender -b x.blend --python me.py -- <args>` 의 `--` 뒤만 argparse 에 넘긴다."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


# ---------------------------------------------------------------------------------------
# 재료 / 메시
# ---------------------------------------------------------------------------------------
def emission_material(name: str, rgb, strength: float):
    """Emission 셰이더. 조명과 무관하게 그 색으로 보여야 판정색이 판정색으로 남는다."""
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    emit = tree.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (*rgb, 1.0)
    emit.inputs["Strength"].default_value = strength
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return material


def look_at_matrix(position, target, up=(0.0, 0.0, 1.0)):
    """blender GL 규약(카메라가 -z 를 본다) 4x4. up 과 시선이 나란하면 up 을 갈아끼운다."""
    forward = (Vector(target) - Vector(position))
    assert forward.length > 1e-9, "position 과 look_at 이 같은 점이다"
    forward.normalize()
    up_vec = Vector(up)
    if abs(forward.dot(up_vec)) > 0.999:            # 바로 위/아래를 보는 후보 (elev ±90)
        up_vec = Vector((0.0, 1.0, 0.0))
    right = forward.cross(up_vec)
    right.normalize()
    true_up = right.cross(forward)
    matrix = Matrix.Identity(4)
    for row in range(3):
        matrix[row][0], matrix[row][1] = right[row], true_up[row]
        matrix[row][2], matrix[row][3] = -forward[row], position[row]
    return matrix


def frustum_mesh(name: str, half_x: float, half_y: float, depth: float, material):
    """카메라 로컬 절두체. 꼭짓점 = 원점, 밑면 = -z 쪽 `depth`. 면을 만든다 (Wireframe modifier 용)."""
    verts = [(0.0, 0.0, 0.0),
             (-half_x, -half_y, -depth), (half_x, -half_y, -depth),
             (half_x, half_y, -depth), (-half_x, half_y, -depth)]
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 2, 3, 4)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(material)
    return mesh


def add_wire(name, mesh, matrix, thickness: float, collection):
    """메시를 씬에 심고 Wireframe modifier 로 두께를 준다. 면이 없으면 아무것도 안 나온다.

    재료는 **메시에** 붙여 두고 (`frustum_mesh` / `ball_mesh` 가 색마다 하나씩 만든다) 오브젝트는
    그 메시를 공유한다. 후보 192개마다 메시를 새로 뜨면 .blend 가 그만큼 무거워지고 Cycles
    BVH 빌드가 후보 수에 비례해 늘어난다 — 같은 절두체를 행렬만 바꿔 세우는 것이므로 공유가 맞다.
    """
    obj = bpy.data.objects.new(name, mesh)
    obj.matrix_world = matrix
    modifier = obj.modifiers.new("wire", "WIREFRAME")
    modifier.thickness = thickness
    modifier.use_replace = True                 # 원래 면은 지우고 테두리만 남긴다
    modifier.use_boundary = True
    collection.objects.link(obj)
    return obj


def ball_mesh(name: str, radius: float, material):
    """후보 위치 점(정팔면체). `bpy.ops` 를 안 쓰는 이유는 후보마다 부르면 O(n) ops 오버헤드다."""
    verts = [(radius, 0, 0), (-radius, 0, 0), (0, radius, 0),
             (0, -radius, 0), (0, 0, radius), (0, 0, -radius)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(material)
    return mesh


def add_ball(name, mesh, center, collection):
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    collection.objects.link(obj)
    return obj


def add_ring(name, center, radius, elevation_deg, material, thickness, collection, segments=96):
    """(반경, 고도) 한 칸의 방위각 원. 격자가 구면이라는 걸 보여주는 게 목적이다."""
    z_off = radius * sin(radians(elevation_deg))
    r_xy = radius * cos(radians(elevation_deg))
    verts, faces = [], []
    for i in range(segments):
        angle = 2.0 * pi * i / segments
        verts.append((r_xy * cos(angle), r_xy * sin(angle), z_off))
    #    선 하나짜리 원엔 면이 없어 Wireframe 이 안 먹는다. 아주 얇은 띠(quad 링)로 만든다.
    for i in range(segments):
        verts.append((verts[i][0], verts[i][1], verts[i][2] + thickness))
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((i, j, j + segments, i + segments))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = center
    obj.data.materials.append(material)
    collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------------------
# board 읽기
# ---------------------------------------------------------------------------------------
def pick_chunk(board: dict, wanted):
    """`--chunk` 를 인덱스로도 tag 로도 받는다. board 는 chunk 리스트를 들고 있다."""
    chunks = board["chunks"]
    if wanted is None:
        return chunks[0]
    for chunk in chunks:
        if str(chunk.get("tag")) == str(wanted) or str(chunk.get("chunk")) == str(wanted):
            return chunk
    raise AssertionError(f"chunk {wanted!r} 없음. 있는 것: "
                         f"{[c.get('tag') for c in chunks][:8]} ...")


def reject_style(candidate: dict):
    """후보 -> (색, 범례 키). 통과면 초록."""
    if candidate["usable"]:
        return PASS_COLOR, "pass"
    reasons = " ".join(candidate["reject"])
    for key, rgb, legend in REJECT_COLORS:
        if key in reasons:
            return rgb, legend
    return (0.6, 0.6, 0.6), "other / 기타"


def half_angles(lens_mm: float, sensor_mm: float, width: int, height: int, depth: float):
    """`new_camera` 와 같은 규약(sensor_fit AUTO, 가로가 길면 가로에 센서를 맞춤)의 절두체 반폭."""
    if width >= height:
        half_x = (0.5 * sensor_mm / lens_mm) * depth
        half_y = half_x * height / width
    else:
        half_y = (0.5 * sensor_mm / lens_mm) * depth
        half_x = half_y * width / height
    return half_x, half_y


# ---------------------------------------------------------------------------------------
def main(args):
    scene = bpy.context.scene
    with open(args.board, encoding="utf-8") as file:
        board = json.load(file)
    chunk = pick_chunk(board, args.chunk)
    candidates = chunk["candidates"]
    origin = chunk["anchor_origin_point"]
    aim = chunk["aim_target"]
    #    사람을 후보가 깔린 그 자세로 세운다. 안 하면 씬은 프레임 1 자세인데 후보는 anchor
    #    프레임 기준이라 "사람이 프레임 밖"인 그림이 나온다.
    frame = int(chunk["anchor_frame"])
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    keep = [c for c in candidates
            if (args.only == "all"
                or (args.only == "usable" and c["usable"])
                or (args.only == "rejected" and not c["usable"]))]
    assert keep, f"--only {args.only} 로 남은 후보가 0개다 (전체 {len(candidates)})"

    collection = bpy.data.collections.new("frustum_viz")
    scene.collection.children.link(collection)

    if args.hide_scene:
        #    씬 지오메트리를 렌더에서만 뺀다 (지우지 않는다 — 사람은 남긴다).
        subject = set(board.get("subject_meshes") or []) | set(board.get("human_meshes") or [])
        subject |= set(board.get("prop_meshes") or [])
        hidden = 0
        for obj in scene.objects:
            if obj.type == "MESH" and obj.name not in subject:
                obj.hide_render = True
                hidden += 1
        print(f"[viz] hide_scene: mesh {hidden}개 렌더 제외 (subject {len(subject)}개 유지)")

    shared, counts = {}, {}
    lens, sensor = float(board["lens_mm"]), float(board["sensor_mm"])
    width, height = (int(v) for v in board["res"])
    half_x, half_y = half_angles(lens, sensor, width, height, args.frustum_len)

    for index, candidate in enumerate(keep):
        rgb, legend = reject_style(candidate)
        counts[legend] = counts.get(legend, 0) + 1
        if legend not in shared:
            key = legend.split()[0]
            material = emission_material(f"viz_{key}", rgb, args.emission)
            shared[legend] = (frustum_mesh(f"frustum_{key}", half_x, half_y,
                                           args.frustum_len, material),
                              ball_mesh(f"ball_{key}", args.thickness * 2.0, material))
        cone, ball = shared[legend]
        tag = (f"cand_{index:03d}_az{candidate['azimuth_deg']:.0f}"
               f"_el{candidate['elevation_deg']:.0f}_r{candidate['radius']:.2f}")
        add_wire(tag, cone, look_at_matrix(candidate["position"], candidate["look_at"]),
                 args.thickness, collection)
        add_ball(tag + "_p", ball, candidate["position"], collection)

    if args.rings:
        ring_material = emission_material("viz_ring", RING_COLOR, args.emission * 0.4)
        pairs = sorted({(round(c["radius"], 4), round(c["elevation_deg"], 3)) for c in keep})
        for radius, elevation in pairs:
            add_ring(f"ring_r{radius:.2f}_el{elevation:.0f}", origin, radius, elevation,
                     ring_material, args.thickness * 0.6, collection)
        print(f"[viz] ring {len(pairs)}개 (반경 x 고도)")

    aim_material = emission_material("viz_aim", AIM_COLOR, args.emission)
    add_ball("aim_target", ball_mesh("ball_aim", args.thickness * 4.0, aim_material),
             aim, collection)

    # --- 오버뷰 카메라 ---------------------------------------------------------------------
    camd = bpy.data.cameras.new("OrbitCamData")
    camd.type = "PERSP"
    camd.lens = args.orbit_lens
    camd.sensor_fit = "AUTO"
    camd.sensor_width = 36.0
    camd.sensor_height = 24.0
    cam_obj = bpy.data.objects.new("OrbitCam", camd)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    radius_max = max(float(c["radius"]) for c in keep)
    orbit_r = args.orbit_radius or radius_max * args.orbit_scale
    #    씬을 켜둔 채 밖에서 돌면 **가까운 벽이 화면을 통째로 막는다** (첫 실측: 후보가 한 개도
    #    안 보이고 건물 외벽만 나왔다). near clip 을 구면 앞까지 밀면 카메라와 구면 사이의 벽만
    #    잘려 나가고 반대편 벽·바닥은 남는다 — 즉 공짜 컷어웨이다. `--clip_cut 0` 이면 끈다.
    near = 0.05 if args.clip_cut <= 0 else max(0.05, orbit_r - radius_max * args.clip_cut)
    camd.clip_start, camd.clip_end = near, 1000.0
    if args.world == "dark":
        #    TRUMANS blend 는 도시 HDRI 를 배경으로 쓴다. 후보만 볼 때는 그 배경이 판정색을
        #    통째로 씻어낸다 (실측: 초록 절두체가 배경에 묻혔다).
        world = scene.world or bpy.data.worlds.new("viz_world")
        scene.world = world
        world.use_nodes = True
        world.node_tree.nodes.clear()
        bg = world.node_tree.nodes.new("ShaderNodeBackground")
        #    완전한 검정으로 두면 emission 절두체만 뜨고 **사람이 안 보인다** (씬 조명은
        #    `--hide_scene` 이 같이 꺼버린다). 회색 배경을 약한 ambient 로 써서 사람만 살린다.
        bg.inputs["Color"].default_value = (0.16, 0.16, 0.19, 1.0)
        bg.inputs["Strength"].default_value = float(args.world_strength)
        out_node = world.node_tree.nodes.new("ShaderNodeOutputWorld")
        world.node_tree.links.new(bg.outputs["Background"], out_node.inputs["Surface"])
    orbit_z = radians(args.orbit_elev)
    #    오버뷰는 후보 구면의 중심을 본다. 사람 발밑이 아니라 anchor origin(가슴) 이어야
    #    구면이 화면 중앙에 온다.
    center = Vector(origin)

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
        print(f"[viz] cycles device GPU/{picked or 'NONE(-> CPU fallback)'}")

    frames_dir = path.join(args.out, "orbit")
    makedirs(frames_dir, exist_ok=True)
    for step in range(args.orbit_frames):
        azimuth = 2.0 * pi * step / args.orbit_frames + radians(args.orbit_az0)
        position = center + Vector((orbit_r * cos(azimuth) * cos(orbit_z),
                                    orbit_r * sin(azimuth) * cos(orbit_z),
                                    orbit_r * sin(orbit_z)))
        cam_obj.matrix_world = look_at_matrix(position, center)
        scene.render.filepath = path.join(frames_dir, f"frame_{step:05d}")
        bpy.ops.render.render(write_still=True)

    summary = {"format": "trumans_frustum_viz_v1", "board": path.abspath(args.board),
               "blend": bpy.data.filepath, "chunk": chunk.get("tag"),
               "anchor_frame": frame, "only": args.only,
               "n_candidates": len(candidates), "n_drawn": len(keep), "legend": counts,
               "grid": board.get("grid"), "gates": board.get("gates"),
               "lens_mm": lens, "sensor_mm": sensor, "res": board["res"],
               "frustum_len": args.frustum_len, "anchor_origin_point": origin,
               "aim_target": aim, "orbit": {"radius": orbit_r, "elev_deg": args.orbit_elev,
                                            "frames": args.orbit_frames, "lens": args.orbit_lens},
               "frames_dir": path.abspath(frames_dir)}
    with open(path.join(args.out, "viz.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=1)

    print(f"\n{'chunk':<16}{chunk.get('tag')}   anchor frame {frame}")
    print(f"{'후보':<16}{len(keep)} / {len(candidates)}   (--only {args.only})")
    print(f"{'격자':<16}az {board['grid']['az_step']:g}도 x el {board['grid']['elevations']} "
          f"x r {chunk['radii']}")
    print(f"{'절두체':<16}lens {lens:g}mm  길이 {args.frustum_len:g}m  "
          f"반폭 {half_x:.3f} x {half_y:.3f} m")
    print(f"{'오버뷰':<16}r {orbit_r:.2f}m  elev {args.orbit_elev:g}도  "
          f"{args.orbit_frames}프레임  {args.res[0]}x{args.res[1]}")
    print(f"\n{'색':<34}{'개수':>6}")
    print("-" * 40)
    for legend, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{legend:<34}{count:>6d}")
    print(f"\nPNG  -> {frames_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    # `trumans_first_pose_board.py` 가 낸 board.json.
    parser.add_argument("--board", required=True, type=str)
    # chunk 인덱스(0,1,..) 또는 tag(c00_f00000). 안 주면 첫 chunk.
    parser.add_argument("--chunk", default=None, type=str)
    parser.add_argument("--out", required=True, type=str)
    # 어떤 후보를 그릴지. `rejected` 는 "어디가 왜 막혔나"만 보는 용도.
    parser.add_argument("--only", default="all", choices=["all", "usable", "rejected"])
    # 절두체를 그리는 길이(m). 실제 화각은 유지하고 길이만 줄인다 — 안 줄이면 서로 겹쳐 안 보인다.
    parser.add_argument("--frustum_len", default=0.28, type=float)
    parser.add_argument("--thickness", default=0.010, type=float)   # Wireframe modifier 두께(m)
    parser.add_argument("--emission", default=4.0, type=float)
    # (반경 x 고도) 방위각 원. 격자가 구면이라는 걸 보여준다.
    parser.add_argument("--rings", dest="rings", action="store_true")
    parser.add_argument("--no_rings", dest="rings", action="store_false")
    parser.set_defaults(rings=True)
    # 씬 지오메트리를 렌더에서 빼고 사람 + 후보만. 벽에 가려 안 보이는 후보까지 보고 싶을 때.
    parser.add_argument("--hide_scene", dest="hide_scene", action="store_true")
    parser.add_argument("--no_hide_scene", dest="hide_scene", action="store_false")
    parser.set_defaults(hide_scene=False)
    parser.add_argument("--orbit_frames", default=72, type=int)
    parser.add_argument("--orbit_radius", default=0.0, type=float)   # 0 = 최대 반경 x orbit_scale
    parser.add_argument("--orbit_scale", default=2.6, type=float)
    parser.add_argument("--orbit_elev", default=28.0, type=float)
    parser.add_argument("--orbit_az0", default=0.0, type=float)
    parser.add_argument("--orbit_lens", default=32.0, type=float)
    # 카메라와 후보 구면 사이의 지오메트리를 near clip 으로 잘라낸다 (구면 반경의 배수). 0 = 끔.
    parser.add_argument("--clip_cut", default=1.35, type=float)
    # `dark` 는 배경 HDRI 를 어두운 단색으로 갈아끼운다 — 판정색이 배경에 안 씻기게.
    parser.add_argument("--world", default="keep", choices=["keep", "dark"])
    parser.add_argument("--world_strength", default=1.0, type=float)
    parser.add_argument("--res", nargs=2, default=[960, 540], type=int)
    parser.add_argument("--samples", default=32, type=int)
    parser.add_argument("--bounces", default=4, type=int)
    #    `--cycles` 로 시작하면 Cycles 애드온이 argv 를 prefix-match 로 먹는다. 이름 바꾸지 말 것.
    parser.add_argument("--cdevice", default="GPU", choices=["GPU", "CPU"])
    parsed = parser.parse_args(cli_argv())
    makedirs(parsed.out, exist_ok=True)
    main(parsed)
