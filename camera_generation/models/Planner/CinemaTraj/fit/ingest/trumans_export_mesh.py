"""TRUMANS `.blend` 의 **evaluated mesh 삼각형**을 프레임별로 뽑아 `mesh_gt.npz` 로 굽는다.

**왜 필요한가.** 뱅크의 충돌 게이트(G1)는 지금 depth 재투영이다 — 소스 카메라에서 본 **표면
껍데기** 하나뿐이라, 그 시야에서 안 보이거나 화면 밖으로 투영되는 위치는 채점 자체가 안 된다
(`fit_hole_ladder.solve_knob` docstring 의 `drop_reveal` 사례: 바닥 밑 카메라가
`behind_frac` 0.0000). TRUMANS 는 씬 전체가 `.blend` 라 **부피를 안다** — 껍데기가 아니라
실제 mesh 로 판정할 수 있다.

그런데 mesh 판정을 이분법 **안**에 넣으려면 probe 마다 Blender 를 띄울 수가 없다:
- 씬 로드 고정비가 **16.06 s** 다 (실측, 502 objects).
- `scene.ray_cast` 는 `frame_set()` 으로 armature 를 평가해야 하고 그게 프레임당 비용의 대부분이다.
- 이분법은 변이당 4회(`--iterations 4`), 편당 수백 변이다.

그래서 역할을 쪼갠다. **Blender 는 한 번만 띄워서 삼각형만 뽑고**(이 스크립트), 판정은
`lbm/mesh_collision.py` 가 voxel 점유 + EDT 로 O(1) 조회한다. 삼각형은 변이와 무관하게 같으므로
영상(chunk) 당 한 번 구우면 그 뒤 이분법이 몇 번 돌든 Blender 를 다시 안 띄운다.

**static / dynamic 을 나눠서 저장하는 이유** 두 개:
1. 메모리. 정적 기하(벽·가구)는 프레임 불변이라 한 벌이면 되고, 프레임마다 다시 저장하면
   49배가 된다. 사람만 프레임별로 든다.
2. 게이트가 이미 두 채널이다. `--collision_time_match` 는 정적/동적을 **각자의 예산과 따로**
   보고 OR 하는데(`fit_hole_ladder.behind_over`), depth 경로는 동적 픽셀 마스크로 그걸 흉내
   냈다. mesh 경로는 애초에 오브젝트 단위로 갈라져 있어서 그 구분이 정확해진다.

**정적/동적 판정은 이름이 아니라 실측**이다. 첫 프레임과 끝 프레임의 evaluated 정점을 비교해
`max|Δp| > --move_eps` 면 동적이다. 이름 규칙(`CC_Base_*`)으로 가르지 않는 이유 —
`trumans-rig-is-cc-base-not-smplx` 가 말하듯 리그 이름 규약이 SMPL-X 가정과 이미 한 번 어긋났고,
사람이 든 소품(book/phone)은 armature 에 안 붙어 있어도 같이 움직인다.

좌표계: **blend world 그대로** (Z-up, metre). 뱅크 world 와는 `anchor_c2w` 한 번의 **강체**
변환이라 (`bank_to_blender_poses.py:408-418` 이 1e-6 로 검증) 스케일이 안 섞인다 — 그래서
여기서는 아무것도 정규화하지 않고 metre 로 둔다. u 로 바꾸는 나눗셈은 게이트 쪽 한 군데서만 한다.

Blender 안에서 도는 스크립트라 이 리포의 다른 모듈을 import 하지 않는다 (`bpy`/`numpy` 만).

사용 예시:
    BL=/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender
    $BL <path>.blend --background --python fit/ingest/trumans_export_mesh.py -- \
        --frames 1777 1921 3 --out out_trumans/tru_1d076f8c_a00_s3f0k6/mesh_gt.npz
"""
import sys
from argparse import ArgumentParser
from os import makedirs, path

import bpy
import numpy as np

# 삼각형을 가질 수 있는 타입만. ARMATURE/EMPTY/LIGHT/CAMERA 는 렌더 결과가 없고 부피도 없다.
MESH_TYPES = {"MESH", "SURFACE", "META", "CURVE", "FONT"}


def argv_after_dashdash():
    """Blender 가 `--` 뒤 인자만 스크립트 몫으로 넘긴다."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def evaluated_triangles(obj, depsgraph):
    """오브젝트 하나의 **world 좌표 삼각형** (T,3,3) float32. 삼각형이 없으면 None.

    `evaluated_get` 을 쓰는 이유는 armature deform / modifier 를 적용한 뒤의 정점이 필요하기
    때문이다 (`trumans_scene_probe.body_points` 와 같은 이유). `calc_loop_triangles()` 로
    n-gon 을 삼각형으로 쪼갠다 — 사각형 벽 한 장이 그냥 안 세지면 그 벽이 게이트에서 사라진다.
    """
    evaluated = obj.evaluated_get(depsgraph)
    try:
        mesh = evaluated.to_mesh()
    except RuntimeError:                       # curve/text 가 mesh 로 안 떨어지는 경우
        return None
    if mesh is None or len(mesh.vertices) == 0:
        evaluated.to_mesh_clear()
        return None
    mesh.calc_loop_triangles()
    n_tri = len(mesh.loop_triangles)
    if n_tri == 0:
        evaluated.to_mesh_clear()
        return None
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3)
    idx = np.empty(n_tri * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", idx)
    tris_local = verts[idx.reshape(-1, 3)]                        # (T,3,3), 로컬
    matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)  # 4x4
    tris = tris_local @ matrix[:3, :3].T + matrix[:3, 3]
    evaluated.to_mesh_clear()
    return tris.astype(np.float32)


def scene_triangles(objects, depsgraph):
    """오브젝트 목록 → (concat 삼각형, {이름: 삼각형 수}). 순서는 `objects` 순서다."""
    chunks, counts = [], {}
    for obj in objects:
        tris = evaluated_triangles(obj, depsgraph)
        counts[obj.name] = 0 if tris is None else int(len(tris))
        if tris is not None:
            chunks.append(tris)
    if not chunks:
        return np.zeros((0, 3, 3), dtype=np.float32), counts
    return np.concatenate(chunks, axis=0), counts


def vertex_signature(obj, depsgraph):
    """정점 위치의 저차원 지문. 정적/동적 판정에만 쓴다 (정점 전량을 두 번 들고 있지 않으려고).

    AABB 만 보면 **제자리 회전**하는 오브젝트를 정적으로 오판한다. 그래서 AABB 에 좌표 합까지
    붙인다 — 회전하면 합이 거의 반드시 바뀐다.
    """
    tris = evaluated_triangles(obj, depsgraph)
    if tris is None:
        return None
    points = tris.reshape(-1, 3).astype(np.float64)
    return np.concatenate([points.min(0), points.max(0), points.sum(0) / len(points)])


def split_static_dynamic(scene, objects, frames, move_eps: float):
    """첫/끝 프레임 지문을 비교해 (정적, 동적) 오브젝트로 가른다.

    프레임 **두 장**만 보는 게 약점이다 — 갔다가 정확히 돌아오는 왕복은 정적으로 잡힌다
    (`lbm-preset-names-dont-match-motion` 이 딱 그 종류의 사고였다). 그래서 중간 프레임도 같이
    본다. 세 장이면 왕복의 반환점이 걸린다.
    """
    probe = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
    signatures = {}
    for frame in probe:
        scene.frame_set(int(frame))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            signatures.setdefault(obj.name, []).append(vertex_signature(obj, depsgraph))
    static, dynamic = [], []
    for obj in objects:
        rows = [r for r in signatures[obj.name] if r is not None]
        if len(rows) < 2:
            static.append(obj)
            continue
        spread = float(np.abs(np.asarray(rows) - rows[0]).max())
        (dynamic if spread > move_eps else static).append(obj)
    return static, dynamic


def main(args):
    scene = bpy.context.scene
    start, end, step = args.frames
    frames = list(range(int(start), int(end) + 1, int(step)))
    assert frames, f"--frames {args.frames} 가 빈 목록이다"

    objects = [o for o in scene.objects
               if o.type in MESH_TYPES and not o.hide_render]
    assert objects, "렌더되는 mesh 오브젝트가 하나도 없다 — .blend 나 씬 선택이 틀렸다"

    static_objs, dynamic_objs = split_static_dynamic(scene, objects, frames, args.move_eps)

    #    정적 기하는 **첫 프레임에서 한 번만** 뽑는다 (정의상 프레임 불변).
    scene.frame_set(frames[0])
    depsgraph = bpy.context.evaluated_depsgraph_get()
    static_tris, static_counts = scene_triangles(static_objs, depsgraph)

    #    동적은 프레임마다. 오브젝트별 삼각형 수는 프레임에 따라 안 변하므로(같은 mesh 를
    #    deform 만 한다) 프레임마다 같은 개수가 나와야 한다 — 아니면 assert 로 잡는다.
    dyn_frames, dyn_counts = [], None
    for frame in frames:
        scene.frame_set(int(frame))
        depsgraph = bpy.context.evaluated_depsgraph_get()
        tris, counts = scene_triangles(dynamic_objs, depsgraph)
        if dyn_counts is None:
            dyn_counts = counts
        else:
            assert counts == dyn_counts, (
                f"프레임 {frame} 에서 동적 삼각형 수가 바뀌었다 — voxel 격자를 프레임마다 "
                f"다시 재야 한다. {counts} != {dyn_counts}")
        dyn_frames.append(tris)
    dyn_tris = (np.stack(dyn_frames, axis=0) if dyn_frames and len(dyn_frames[0])
                else np.zeros((len(frames), 0, 3, 3), dtype=np.float32))

    makedirs(path.dirname(path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        format="trumans_mesh_gt_v1",
        blend=bpy.data.filepath,
        blender_version=bpy.app.version_string,
        frame_list=np.asarray(frames, dtype=np.int32),
        move_eps=float(args.move_eps),
        static_tris=static_tris,
        dyn_tris=dyn_tris,
        static_names=np.asarray(sorted(o.name for o in static_objs), dtype=object),
        dynamic_names=np.asarray(sorted(o.name for o in dynamic_objs), dtype=object),
    )

    print(f"\n{'blend':<22}{path.basename(bpy.data.filepath)}")
    print(f"{'frames':<22}{len(frames)}   {frames[0]}..{frames[-1]} step {step}")
    print(f"{'static objects':<22}{len(static_objs):>8}   tris {len(static_tris):>9}")
    print(f"{'dynamic objects':<22}{len(dynamic_objs):>8}   tris "
          f"{(dyn_tris.shape[1] if dyn_tris.ndim == 4 else 0):>9} / frame")
    print(f"{'dynamic names':<22}{', '.join(sorted(o.name for o in dynamic_objs)[:8])}")
    lo = static_tris.reshape(-1, 3).min(0) if len(static_tris) else np.zeros(3)
    hi = static_tris.reshape(-1, 3).max(0) if len(static_tris) else np.zeros(3)
    print(f"{'static aabb (m)':<22}[{lo[0]:.2f} {lo[1]:.2f} {lo[2]:.2f}] .. "
          f"[{hi[0]:.2f} {hi[1]:.2f} {hi[2]:.2f}]")
    print(f"\n기록  {args.out}")


if __name__ == "__main__":
    parser = ArgumentParser(description="TRUMANS .blend -> 프레임별 evaluated mesh 삼각형")
    # chunk 의 blend 프레임 번호. `manifest["frames"]` + `manifest["frame_step"]` 를 그대로.
    parser.add_argument("--frames", nargs=3, type=int, required=True,
                        metavar=("START", "END", "STEP"))
    parser.add_argument("--out", required=True, type=str)     # mesh_gt.npz 경로
    # 정적/동적 경계 (metre). 0.5 mm — armature 가 안 붙은 가구는 정확히 0 으로 나오고,
    # deform 되는 옷/살은 프레임당 cm 단위로 움직여서 사이가 넓다.
    parser.add_argument("--move_eps", default=5e-4, type=float)
    main(parser.parse_args(argv_after_dashdash()))
