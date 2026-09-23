"""TRUMANS `.blend` 안에 실제로 뭐가 들어 있는지 headless Blender 로 찍어본다.

왜 필요한가: 원본 Look-Before-Move 는 `.blend` 안의 오브젝트를 **`asset_id` 이름으로** 찾는다
(`director_scene_context_builder.py:254-283` — 정확일치 → 커스텀 프로퍼티 `asset_id` →
`<id>.` / `<id>_` 접두사 순). 그리고 씬을 `Scene_<scene_id>` 규약으로 찾는다(:206-231).
TRUMANS 의 `obj_list.txt` 에 적힌 이름이 정말 오브젝트 이름인지, 씬 이름이 뭔지, armature/action
이 어떻게 붙어 있는지를 **먼저 확인하지 않으면** 어댑터 JSON 을 짜도 전부 빗나간다.
`strings` 로는 문자열이 있다는 것만 알지 그게 오브젝트 이름인지 머티리얼 이름인지 모른다.

Blender 안에서 도는 스크립트라 이 리포의 다른 모듈을 import 하지 않는다(`bpy` 만).

사용 예시:
    BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    $BL <path>.blend --background --python eval/probe_trumans_blend.py -- --out /tmp/probe.json
"""
import json
import sys
from argparse import ArgumentParser

import bpy


def argv_after_dashdash():
    """Blender 가 `--` 뒤 인자만 스크립트 몫으로 넘긴다."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def obj_record(obj):
    dims = tuple(round(float(v), 4) for v in obj.dimensions)
    loc = tuple(round(float(v), 4) for v in obj.location)
    anim = obj.animation_data
    return {
        "name": obj.name,
        "type": obj.type,
        "dimensions": dims,
        "location": loc,
        "parent": obj.parent.name if obj.parent else None,
        # LBM 이 2순위로 보는 커스텀 프로퍼티. TRUMANS 가 안 넣었을 가능성이 크다.
        "custom_keys": sorted(k for k in obj.keys() if not k.startswith("_")),
        "action": anim.action.name if anim and anim.action else None,
        "n_fcurves": len(anim.action.fcurves) if anim and anim.action else 0,
        "hide_render": bool(obj.hide_render),
    }


def main(args):
    report = {
        "blend": bpy.data.filepath,
        "blender_version": bpy.app.version_string,
        "scenes": [],
        "counts": {
            "objects": len(bpy.data.objects), "meshes": len(bpy.data.meshes),
            "armatures": len(bpy.data.armatures), "actions": len(bpy.data.actions),
            "cameras": len(bpy.data.cameras), "collections": len(bpy.data.collections),
            "materials": len(bpy.data.materials), "images": len(bpy.data.images),
        },
        "collections": sorted(c.name for c in bpy.data.collections),
        "actions": [{"name": a.name, "frame_range": [float(v) for v in a.frame_range],
                     "n_fcurves": len(a.fcurves)} for a in bpy.data.actions],
    }
    for scene in bpy.data.scenes:
        objs = [obj_record(o) for o in scene.objects]
        by_type = {}
        for rec in objs:
            by_type[rec["type"]] = by_type.get(rec["type"], 0) + 1
        report["scenes"].append({
            "name": scene.name,
            "frame_start": scene.frame_start, "frame_end": scene.frame_end,
            "fps": scene.render.fps / scene.render.fps_base,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "engine": scene.render.engine,
            "camera": scene.camera.name if scene.camera else None,
            "world": scene.world.name if scene.world else None,
            "n_objects": len(objs), "by_type": by_type,
            # 전량 싣는다 — 어차피 씬당 수십 개고, 이름 규약을 눈으로 봐야 한다.
            "objects": objs,
        })

    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=1)

    print(f"\n{'key':16s} value")
    for key, value in report["counts"].items():
        print(f"{key:16s} {value}")
    for scene in report["scenes"]:
        print(f"\n[scene] {scene['name']}  frames {scene['frame_start']}..{scene['frame_end']}"
              f"  fps {scene['fps']:g}  {scene['resolution']}  {scene['engine']}"
              f"  camera={scene['camera']}  world={scene['world']}")
        print(f"  by_type {scene['by_type']}")
        for rec in scene["objects"][:args.show]:
            print(f"    {rec['type']:9s} {rec['name'][:44]:44s} dim={rec['dimensions']}"
                  f" action={rec['action']} keys={rec['custom_keys']}")
        if scene["n_objects"] > args.show:
            print(f"    ... {scene['n_objects'] - args.show} more")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--out", default="/data1/cympyc1785/LatentCamVid/tmp/trumans_blend_probe.json", type=str)
    parser.add_argument("--show", default=60, type=int)   # 씬당 콘솔에 찍을 오브젝트 수
    main(parser.parse_args(argv_after_dashdash()))
