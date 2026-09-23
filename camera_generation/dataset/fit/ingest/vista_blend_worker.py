"""Vista4D 점군에서 뽑아낸 삼각형 메시들을 **원본 LBM 이 읽을 수 있는 `.blend`** 로 굽는다 (headless).

왜 점군이 아니라 메시인가: LBM 의 가림 판정 `occlusion_check`
(`Cinematographer/cinematographer_quality_worker.py:2451-2502`)이 **`scene.ray_cast` 를 45개
표본점에 쏘는 dense 레이캐스트**다. 면이 없는 점군(POINTCLOUD/버텍스만 있는 mesh)에는 레이가
절대 안 맞아서 `hit=False` → `occluded=0` → `occlusion_ratio=0.0` → `severely_occluded=False` 가
된다. 즉 게이트가 막는 게 아니라 **조용히 꺼진다** — 로그에는 아무것도 안 남고 벽 뒤 카메라가
전부 통과한다. 그래서 depth 격자를 삼각분할해 **진짜 면**을 만들어 넣는다.

왜 노드별로 오브젝트를 쪼개는가: LBM 은 `asset_id` 로 오브젝트를 찾고
(`director_scene_context_builder.py:254-283` `_find_object_by_asset_id`: 이름 완전일치 → 커스텀
프로퍼티 `asset_id` → `.nnnn` 접미사 → `_` 접두사), 레이가 **focus 오브젝트에 맞으면 가림으로
안 센다**. 배경 메시 한 덩어리에 camel 표면까지 들어 있으면 카메라→camel 레이가 "배경"에 맞아서
가림 100% 로 찍힌다. 그래서 격자 버텍스를 노드 OBB 소속으로 나눠 `dyn_0`/`stat_2`/`background`
같은 별도 오브젝트로 굽는다.

동적 노드는 frame 0 격자에서만 지오메트리를 만들고 scene graph 의 `track` 으로 키프레임을 굽는다.
(다른 프레임의 동적 점을 배경에 넣으면 움직이는 물체가 정지 잔상으로 49프레임 쌓인다.)

입력은 `vista_to_lbm_demo.py` 가 만든 `_geom.npz` + `_geom.json` 두 개다. `--` 뒤 인자만 이
스크립트 몫이다 (Blender 규약).

사용 예시:
    BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    $BL --background --python fit/ingest/vista_blend_worker.py -- \
        --geom /tmp/camel_geom.npz --meta /tmp/camel_geom.json --out /tmp/camel.blend
"""
import json
import sys
from argparse import ArgumentParser

import bpy
import numpy as np


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def clear_scene():
    """기본 큐브/라이트/카메라를 지운다. LBM 은 자기 카메라를 새로 만든다."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def vertex_color_material():
    """버텍스 색을 **Emission 으로** 내보내는 공용 머티리얼 1개.

    왜 diffuse 가 아니라 emission 인가: 점군 색은 이미 원본 촬영의 조명이 구워진 값이다. 여기에
    Blender 조명을 다시 먹이면 그림자가 두 번 들어가 렌더가 흑백 얼룩이 된다 (실측: 기본
    머티리얼 + SUN 으로 camel 씬을 렌더하면 소스와 형상은 같은데 명암이 전부 뭉개졌다).
    LBM 은 이 프리뷰를 **VLM 에게 보여주고 shot 품질을 판정**시키므로, 사람이 못 알아보는
    그림이면 판정 자체가 무의미해진다.
    """
    material = bpy.data.materials.new("vista_vertex_color")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    attribute = tree.nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(attribute.outputs["Color"], emission.inputs["Color"])
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def build_mesh(name: str, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray,
               origin: np.ndarray, material=None):
    """(Nv,3)/(Nf,3)/(Nv,3)u8 → 메시 오브젝트 1개.

    `origin` 을 로컬 원점으로 뺀 좌표를 메시 데이터에 넣고 `obj.location = origin` 으로 되돌린다.
    동적 노드를 그 자리에서 회전시키려면 원점이 OBB 중심에 있어야 한다 (안 그러면 world 원점을
    축으로 돌아서 물체가 씬 밖으로 날아간다).
    """
    mesh = bpy.data.meshes.new(name)
    local = (verts - origin[None, :]).astype(np.float64)
    mesh.from_pydata(local.tolist(), [], faces.tolist())
    mesh.validate(verbose=False)

    if len(colors):
        #    BYTE_COLOR/POINT 는 float RGBA 를 요구한다 (uint8 을 그대로 넣으면 전부 흰색).
        #    **`color` 는 linear 를 받는다.** 점군 색은 sRGB 바이트라 그대로 넣으면 렌더 때
        #    linear→sRGB 가 한 번 더 걸려 전체가 들뜬다 (실측: camel 이 흰 덩어리로 나왔다).
        #    `color_srgb` 슬롯도 있지만 버전마다 있고 없어서 여기서 직접 역감마를 건다.
        attr = mesh.color_attributes.new(name="Col", type="BYTE_COLOR", domain="POINT")
        srgb = colors.astype(np.float32) / 255.0
        linear = np.where(srgb <= 0.04045, srgb / 12.92,
                          ((srgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
        rgba = np.ones((len(verts), 4), dtype=np.float32)
        rgba[:, :3] = linear
        attr.data.foreach_set("color", rgba.reshape(-1))

    if material is not None:
        mesh.materials.append(material)

    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    obj.location = tuple(float(v) for v in origin)
    #    이름 완전일치가 1순위지만 Blender 가 `.001` 을 붙일 수 있으므로 커스텀 프로퍼티도 심는다.
    obj["asset_id"] = name
    bpy.context.scene.collection.objects.link(obj)
    return obj


def bake_track(obj, keyframes):
    """`[[frame, x, y, z, yaw_deg], ...]` 를 location/rotation_euler.z 키프레임으로 굽는다."""
    for frame, x, y, z, yaw_deg in keyframes:
        obj.location = (float(x), float(y), float(z))
        obj.rotation_euler = (0.0, 0.0, float(yaw_deg) * 0.017453292519943295)
        obj.keyframe_insert(data_path="location", frame=int(frame))
        obj.keyframe_insert(data_path="rotation_euler", frame=int(frame))
    #    보간을 LINEAR 로. 기본 BEZIER 는 키 사이에서 오버슈트해 물체가 궤적 밖으로 튄다.
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for point in fcurve.keyframe_points:
                point.interpolation = "LINEAR"


def add_sun(meta):
    """LBM 프리뷰 렌더가 새까맣지 않도록 태양광 1개. 지오메트리 판정에는 영향이 없다."""
    light = bpy.data.lights.new("sun_01", type="SUN")
    light.energy = 3.0
    obj = bpy.data.objects.new("sun_01", light)
    obj.location = (0.0, 0.0, float(meta["scene_bounds"]["z"]) + 5.0)
    obj.rotation_euler = (0.5, 0.2, 0.0)
    bpy.context.scene.collection.objects.link(obj)


def main(args):
    geom = np.load(args.geom)
    with open(args.meta, encoding="utf-8") as file:
        meta = json.load(file)

    clear_scene()
    scene = bpy.context.scene
    scene.frame_start = int(meta["frame_start"])
    scene.frame_end = int(meta["frame_end"])
    scene.render.fps = int(round(float(meta["fps"])))
    scene.render.fps_base = 1.0
    scene.render.resolution_x = int(meta["resolution"][0])
    scene.render.resolution_y = int(meta["resolution"][1])
    scene.render.engine = args.engine
    #    Blender 4.5 기본 view transform 은 `AgX` 라 emission 색을 들어올려 탈색시킨다 (실측:
    #    camel 렌더가 소스 대비 하얗게 뜬다). 점군 색은 이미 sRGB 결과값이므로 톤매핑을 끈다.
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    #    구멍(면이 없는 곳)이 흰색이면 하늘과 구분이 안 된다. 중간 회색으로 둔다.
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.38, 0.42, 1.0)
    scene.world = world

    material = vertex_color_material()
    rows = []
    for entry in meta["objects"]:
        name = entry["name"]
        verts = geom[f"v_{name}"]
        faces = geom[f"f_{name}"]
        colors = geom[f"c_{name}"]
        origin = np.asarray(entry["origin"], dtype=np.float64)
        obj = build_mesh(name, verts, faces, colors, origin, material)
        if entry.get("keyframes"):
            bake_track(obj, entry["keyframes"])
        rows.append((name, entry.get("kind", ""), len(verts), len(faces),
                     bool(entry.get("keyframes"))))

    add_sun(meta)
    bpy.ops.wm.save_as_mainfile(filepath=args.out)

    print(f"\n{'object':<16}{'kind':<12}{'verts':>10}{'faces':>10}{'anim':>6}")
    for name, kind, nv, nf, anim in rows:
        print(f"{name:<16}{kind:<12}{nv:>10d}{nf:>10d}{('yes' if anim else '-'):>6}")
    print(f"{'total':<28}{sum(r[2] for r in rows):>10d}{sum(r[3] for r in rows):>10d}")
    print(f"\nframes {scene.frame_start}..{scene.frame_end}  fps {scene.render.fps}"
          f"  {scene.render.resolution_x}x{scene.render.resolution_y}  {scene.render.engine}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--geom", required=True, type=str)    # vista_to_lbm_demo 가 만든 npz
    parser.add_argument("--meta", required=True, type=str)    # 같은 이름의 json
    parser.add_argument("--out", required=True, type=str)     # 저장할 .blend
    #    BLENDER_EEVEE_NEXT 가 4.5 의 기본 실시간 엔진. CYCLES 는 프리뷰 1장에 수십 초 든다.
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT", type=str)
    main(parser.parse_args(argv_after_dashdash()))
