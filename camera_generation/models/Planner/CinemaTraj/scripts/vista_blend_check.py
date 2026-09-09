"""`vista_to_lbm_demo.py` 가 구운 `.blend` 가 **소스 영상과 같은 씬인지** 확인한다 (headless).

세 가지를 한 번에 본다. 앞의 둘이 통과해야 LBM 결과를 믿을 수 있다.

1. **소스 카메라 재렌더** — scene graph 의 `cameras.cam_centers_g` + `K` 로 Blender 카메라를
   세우고 프레임 몇 장을 렌더한다. 원본 프레임과 물체가 같은 자리에 있어야 한다. 여기가 틀리면
   `T_gw` / `u_meters` / K 변환 중 하나가 어긋난 것이고, 그 아래 전부가 무의미해진다.
2. **레이캐스트 적중률** — 소스 카메라에서 각 노드 OBB 중심으로 `scene.ray_cast` 를 쏜다.
   이게 LBM 의 `occlusion_check` (`cinematographer_quality_worker.py:2451-2502`)가 하는 일과
   같다. **점군에는 면이 없어서 항상 `hit=False`** 가 나오고 그러면 가림 게이트가 조용히 꺼진다.
   여기서 `hit=True` 가 나와야 어댑터가 그 문제를 실제로 고친 것이다.
3. 오브젝트 이름·면수·애니메이션 유무 표 — LBM 은 `asset_id` 이름으로 오브젝트를 찾는다.

`--` 뒤 인자만 이 스크립트 몫이다 (Blender 규약).

사용 예시:
    BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    $BL out/lbm_demos_vista/camel/camel.blend --background \
        --python scripts/vista_blend_check.py -- \
        --graph out/camel/scene_graph.json --vista out/lbm_demos_vista/camel/_vista.json \
        --frames 0,24,48 --out /tmp/camel_check
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import bpy
import mathutils
import numpy as np


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def blender_camera_matrix(forward, up):
    """OpenCV 시선(forward)·up 을 Blender 카메라 회전으로.

    Blender 카메라는 **로컬 -Z 를 본다**(OpenCV 는 +Z). 로컬 +Y 가 화면 위다(OpenCV 는 +Y 아래).
    그래서 `local_z = -forward`, `local_y = up`, `local_x = local_y × local_z` 다. 이 부호를
    틀리면 렌더가 상하 반전되거나 정반대를 보는데, 회전이 작은 씬에서는 둘 다 "그럴듯"해 보인다.
    """
    local_z = -np.asarray(forward, dtype=np.float64)
    local_z /= np.linalg.norm(local_z)
    local_y = np.asarray(up, dtype=np.float64)
    local_y = local_y - local_z * float(local_y @ local_z)
    local_y /= np.linalg.norm(local_y)
    local_x = np.cross(local_y, local_z)
    return np.stack([local_x, local_y, local_z], axis=1)


def make_camera(scene, name="check_cam"):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    scene.camera = obj
    return obj


def set_camera(obj, position, rot3, K, width, height):
    """`K` 의 fx 로 렌즈를, cx/cy 로 shift 를 잡는다 (센서 폭은 36 mm 기준)."""
    matrix = mathutils.Matrix.Identity(4)
    for row in range(3):
        for col in range(3):
            matrix[row][col] = float(rot3[row][col])
        matrix[row][3] = float(position[row])
    obj.matrix_world = matrix
    data = obj.data
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.0
    data.lens = float(K[0][0]) / float(width) * 36.0
    data.shift_x = (float(width) * 0.5 - float(K[0][2])) / float(width)
    data.shift_y = (float(K[1][2]) - float(height) * 0.5) / float(width)
    data.clip_start = 0.01
    data.clip_end = 10000.0


def main(args):
    with open(args.graph, encoding="utf-8") as file:
        graph = json.load(file)
    with open(args.vista, encoding="utf-8") as file:
        vista = json.load(file)
    u_meters = float(vista["u_meters"])
    width, height = int(graph["width"]), int(graph["height"])
    K_all = graph["cameras"]["K"]
    c2w_all = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)
    transform = np.asarray(graph["frames"]["T_gw"], dtype=np.float64)
    centers = np.asarray(graph["cameras"]["cam_centers_g"], dtype=np.float64) * u_meters

    scene = bpy.context.scene
    scene.render.resolution_x = args.width or width
    scene.render.resolution_y = args.height or height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    camera = make_camera(scene)

    print(f"\n{'object':<16}{'type':<12}{'faces':>10}{'anim':>6}")
    for obj in sorted(scene.objects, key=lambda o: o.name):
        faces = len(obj.data.polygons) if obj.type == "MESH" else 0
        anim = "yes" if (obj.animation_data and obj.animation_data.action) else "-"
        print(f"{obj.name:<16}{obj.type:<12}{faces:>10d}{anim:>6}")

    nodes = list(graph["nodes"])
    makedirs(args.out, exist_ok=True)
    frames = [int(f) for f in args.frames.split(",") if f != ""]
    rows = []
    for frame in frames:
        scene.frame_set(frame)
        #    G 로 옮긴 소스 카메라. `T_gw` 의 회전 부분은 스케일을 품고 있으므로 정규화한다
        #    (안 하면 forward/up 이 길이 0.21 인 채로 들어가 lens 계산과 안 맞는다).
        rot_gw = transform[:3, :3] / np.cbrt(abs(np.linalg.det(transform[:3, :3])))
        forward = rot_gw @ c2w_all[frame][:3, 2]
        up = rot_gw @ (-c2w_all[frame][:3, 1])   # OpenCV +Y 는 아래
        rot3 = blender_camera_matrix(forward, up)
        set_camera(camera, centers[frame], rot3, K_all[frame], width, height)

        target = path.join(args.out, f"render_f{frame:03d}.png")
        scene.render.filepath = target
        bpy.ops.render.render(write_still=True)

        #    LBM 의 가림 판정과 같은 호출. 점군이었다면 여기가 전부 miss 다.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        origin = mathutils.Vector([float(v) for v in centers[frame]])
        for node in nodes:
            obb = node["obb"]
            track = node.get("track") or {}
            if node.get("moving") and track.get("center_smooth"):
                pick = int(np.argmin(np.abs(np.asarray(track["frames"]) - frame)))
                center = np.asarray(track["center_smooth"][pick], dtype=np.float64) * u_meters
            else:
                center = np.asarray(obb["center"], dtype=np.float64) * u_meters
            delta = mathutils.Vector([float(v) for v in center]) - origin
            distance = delta.length
            hit, location, _n, _f, hit_obj, _m = scene.ray_cast(
                depsgraph, origin, delta.normalized(), distance=max(distance * 1.5, 0.0))
            rows.append((frame, node["id"], node["label"], bool(hit),
                         hit_obj.name if hit_obj else "-", float(distance),
                         float((location - origin).length) if hit else float("nan")))

    print(f"\n{'frame':>6}{'node':<10}{'label':<12}{'hit':>5}{'hit_object':<16}"
          f"{'d_obb':>9}{'d_hit':>9}")
    for frame, node_id, label, hit, hit_name, d_obb, d_hit in rows:
        print(f"{frame:>6}{node_id:<10}{label:<12}{('Y' if hit else 'N'):>5}{hit_name:<16}"
              f"{d_obb:>9.2f}{d_hit:>9.2f}")
    n_hit = sum(1 for r in rows if r[3])
    print(f"\nray_cast 적중 {n_hit}/{len(rows)}  "
          f"(0 이면 면이 없다는 뜻 — LBM 가림 게이트가 꺼진다)")
    print(f"-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--graph", required=True, type=str)    # scene_graph.json
    parser.add_argument("--vista", required=True, type=str)    # demo root 의 _vista.json
    parser.add_argument("--frames", default="0,24,48", type=str)
    parser.add_argument("--width", default=0, type=int)        # 0 이면 씬 해상도 그대로
    parser.add_argument("--height", default=0, type=int)
    parser.add_argument("--out", default="/data1/cympyc1785/LatentCamVid/tmp/vista_blend_check", type=str)
    main(parser.parse_args(argv_after_dashdash()))
