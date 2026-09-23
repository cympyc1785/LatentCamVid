"""TRUMANS `.blend` 에서 원본 Look-Before-Move 가 요구하는 **레이아웃 수치**를 뽑는다 (headless).

왜 blend 에서 뽑는가: TRUMANS 는 `Object_all/Object_pose/<seq>.npy` 에도 물체 pose 가 있지만
그건 **Blender 의 부모 EMPTY 기준 local 값**이고, LBM 이 원하는 건 asset 의 world 위치·크기다
(`layout_description.assets[].location` / `asset_sheet[].width/depth/height`). 게다가 TRUMANS 는
mesh 를 `<obj>_root_<obj>` EMPTY 에 매달아 애니메이션하므로, mesh 의 `obj.location` 을 그대로
읽으면 전부 0 근처가 나온다. **`matrix_world` 로 world AABB 를 다시 계산해야** 맞는다.

크기도 `obj.dimensions` 를 안 쓴다 — 그건 local bbox × scale 이라 부모 회전이 반영되지 않는다.
8개 bbox 코너를 world 로 보내 축정렬 AABB 를 다시 잡는다.

`--` 뒤 인자만 이 스크립트 몫이다 (Blender 규약). `bpy` 외 import 없음.

사용 예시:
    BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    $BL <scene>.blend --background --python fit/ingest/trumans_blend_layout_worker.py -- \
        --frame 0 --assets cup_01,oven_base_01 --out /tmp/layout.json
"""
import json
import sys
from argparse import ArgumentParser

import bpy
from mathutils import Vector


def argv_after_dashdash():
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def world_aabb(obj):
    """(center(3), size(3)) — world 축정렬. mesh 가 아니면 None."""
    if not hasattr(obj, "bound_box") or obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}:
        return None
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    return (hi + lo) * 0.5, hi - lo


def union_aabb(boxes):
    """자식 mesh 여러 개를 하나의 asset 으로 묶을 때 (armature 캐릭터가 이 경우)."""
    if not boxes:
        return None
    los, his = [], []
    for center, size in boxes:
        half = size * 0.5
        los.append(center - half)
        his.append(center + half)
    lo = Vector((min(v.x for v in los), min(v.y for v in los), min(v.z for v in los)))
    hi = Vector((max(v.x for v in his), max(v.y for v in his), max(v.z for v in his)))
    return (hi + lo) * 0.5, hi - lo


def descendants(obj):
    """obj 와 그 자손 전부. armature 는 스킨 mesh 가 자식으로 달려 있어 이걸로 묶는다."""
    out = [obj]
    stack = list(obj.children)
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(node.children)
    return out


def measure(scene, name: str):
    obj = scene.objects.get(name)
    if obj is None:
        return None
    boxes = [box for child in descendants(obj) if (box := world_aabb(child)) is not None]
    box = union_aabb(boxes)
    if box is None:
        # EMPTY/ARMATURE 인데 mesh 자손이 없다 — 위치만 돌려준다.
        loc = obj.matrix_world.translation
        return {"asset_id": name, "type": obj.type, "n_meshes": 0,
                "center": [round(float(v), 5) for v in loc],
                "size": [0.0, 0.0, 0.0],
                "yaw_deg": round(float(obj.matrix_world.to_euler("XYZ").z) * 57.29577951308232, 4)}
    center, size = box
    return {
        "asset_id": name, "type": obj.type, "n_meshes": len(boxes),
        "center": [round(float(v), 5) for v in center],
        "size": [round(float(v), 5) for v in size],
        # LBM `layout_description.assets[].rotation.z` 는 **도(degree)** 다
        # (`cinematographer_stage.py:483` 에서 math.radians 로 변환한다).
        "yaw_deg": round(float(obj.matrix_world.to_euler("XYZ").z) * 57.29577951308232, 4),
    }


def main(args):
    scene = bpy.context.scene
    scene.frame_set(args.frame)   # 애니메이션이 걸려 있으므로 프레임을 못 박고 재야 한다

    wanted = [n for n in args.assets.split(",") if n]
    if args.include_armatures:
        # 캐릭터는 `obj_list.txt` 에 없다 (그건 상호작용 물체 목록이다). TRUMANS 는 사람이
        # armature 1개로 들어 있고 스킨 mesh 가 그 자식이라, 타입으로 찾는 게 이름 추측보다 안전하다.
        wanted += [o.name for o in scene.objects if o.type == "ARMATURE" and o.name not in wanted]
    assets = [rec for name in wanted if (rec := measure(scene, name)) is not None]
    missing = [name for name in wanted if scene.objects.get(name) is None]

    # 씬 범위: 렌더에 보이는 mesh 전체의 world AABB. LBM `scene_size` 기본값이 ±10 이라
    # 실내 씬(≈5 m)에 그대로 두면 카메라 후보가 벽 밖으로 나간다.
    boxes = [box for obj in scene.objects
             if not obj.hide_render and (box := world_aabb(obj)) is not None]
    bounds = union_aabb(boxes)
    center, size = bounds if bounds else (Vector((0, 0, 0)), Vector((0, 0, 0)))
    half = size * 0.5

    report = {
        "blend": bpy.data.filepath,
        "scene_name": scene.name,
        "frame": args.frame,
        "frame_start": scene.frame_start, "frame_end": scene.frame_end,
        "fps": scene.render.fps / scene.render.fps_base,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "engine": scene.render.engine,
        "n_objects": len(scene.objects),
        "scene_bounds": {
            "center": [round(float(v), 5) for v in center],
            "size": [round(float(v), 5) for v in size],
            "x_negative": round(float(center.x - half.x), 5), "x": round(float(center.x + half.x), 5),
            "y_negative": round(float(center.y - half.y), 5), "y": round(float(center.y + half.y), 5),
            "z_negative": round(float(center.z - half.z), 5), "z": round(float(center.z + half.z), 5),
        },
        "assets": assets,
        "missing": missing,
    }
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=1)

    print(f"\nscene {scene.name!r} frames {scene.frame_start}..{scene.frame_end} @ frame {args.frame}"
          f"  fps {report['fps']:g}  objects {report['n_objects']}")
    b = report["scene_bounds"]
    print(f"bounds x[{b['x_negative']:.2f},{b['x']:.2f}] y[{b['y_negative']:.2f},{b['y']:.2f}]"
          f" z[{b['z_negative']:.2f},{b['z']:.2f}]")
    print(f"\n{'asset_id':32s} {'type':9s} {'n':>3s} {'center':>26s} {'size':>24s} {'yaw':>8s}")
    for rec in assets:
        print(f"{rec['asset_id']:32s} {rec['type']:9s} {rec['n_meshes']:3d}"
              f" {str([f'{v:.2f}' for v in rec['center']]):>26s}"
              f" {str([f'{v:.2f}' for v in rec['size']]):>24s} {rec['yaw_deg']:8.2f}")
    if missing:
        print(f"\n못 찾은 asset {len(missing)}개: {missing}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--frame", default=0, type=int)          # 레이아웃을 잴 프레임
    parser.add_argument("--assets", default="", type=str)        # 쉼표 구분 오브젝트 이름
    # 캐릭터(ARMATURE)를 자동으로 asset 에 추가. 끄면 --assets 에 적은 것만 잰다.
    parser.add_argument("--include_armatures", action="store_true", default=True)
    parser.add_argument("--no_include_armatures", dest="include_armatures", action="store_false")
    parser.add_argument("--out", default="/data1/cympyc1785/LatentCamVid/tmp/trumans_layout.json", type=str)
    main(parser.parse_args(argv_after_dashdash()))
