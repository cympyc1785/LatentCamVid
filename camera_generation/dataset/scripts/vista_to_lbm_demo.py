"""Vista4D 영상 1편(`cloud.npz` + `scene_graph.json`)을 **원본 Look-Before-Move 의 demo root** 로 바꾼다.

왜 이 어댑터가 필요한가: LBM 은 입력이 "`.blend` 씬 + story/layout JSON 4종"인 파이프라인이다.
우리 Vista4D 입력은 동영상 1편에서 뽑은 4D 점군과 scene graph 뿐이다. **LBM 코드를 한 줄도 안
고치고** 돌리려면 우리 쪽에서 그 4종을 합성해 주는 수밖에 없다 — 그게 이 스크립트다.
(TRUMANS 쪽 대응물이 `trumans_to_lbm_demo.py`. 만드는 파일 목록과 스키마가 같다.)

만드는 것:
    <out>/<scene>/<scene>.blend                          ← vista_blend_worker.py 가 굽는다
    <out>/<scene>/layout_measured.json                   ← 우리가 직접 잰 씬 치수
    <out>/<scene>/layout_script/layout_script_v1.json
    <out>/<scene>/animated_models/animated_models_v1.json
    <out>/<scene>/animated_models/selected_animation_v1.json
    <out>/<scene>/_vista.json                            ← 되돌리기용 사이드카 (LBM 은 안 읽는다)
    <out>/<scene>__run.sh                                ← LBM 4단계 실행 스크립트

좌표계: scene graph 의 **G frame** 을 그대로 Blender world 로 쓴다. G 는 이미 중력축이 +Z 인
Z-up 오른손 좌표계라 Blender 규약과 같다 (world = DA3 frame0 카메라 = OpenCV Y-down 을 그대로
쓰면 LBM 이 만드는 카메라 전체에 Dutch angle 이 박힌다). `frames.T_gw` 가 world→G 변환이고
`1 u = scale.S DA3 단위`다.

**단위 배율이 왜 필요한가**: G 는 무차원 `u` 단위이고 camel 씬은 전체가 3 u 안에 들어간다.
LBM 의 임계값들(near clip, "너무 가깝다" 판정, 궤적 크기 기본값)은 전부 **미터 눈금 상수**라
0.13 u 짜리 camel 을 찻잔으로 취급한다. 그래서 `--subject_height_m`(기본 1.8) 로 subject OBB
높이를 맞추는 배율 `u_meters` 를 자동으로 뽑아 곱한다. 되돌릴 수 있게 `_vista.json` 에 적는다.

사용 예시:
    python scripts/vista_to_lbm_demo.py --scene_dir out/camel \
        --out /data1/.../dataset/out/lbm_demos_vista --movement_term orbit_left_arc
    bash out/lbm_demos_vista/camel__run.sh vista_camel 0
"""
import json
import subprocess
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
# R6(2026-09-22) 재분류에서 드라이버(`scripts/`)와 Blender worker(`fit/ingest/`)가 **갈라졌다.**
# `HERE` 옆을 보면 안 나온다 — 형제 폴더를 명시한다 (R8 에서 잡음).
BLEND_WORKER = path.join(path.dirname(HERE), "fit", "ingest", "vista_blend_worker.py")

#    `trumans_to_lbm_demo.py` 의 movement term 표를 그대로 쓴다. **문구를 새로 짓지 않는 게
#    핵심**이다 — Director 의 `infer_movement_intent` / `infer_direction_label` 은 정해진
#    동사구("circles around", "walks around", "cranes up", "pans to the right")로만 preset 을
#    고르고, 안 걸리면 경고 없이 `static` 으로 떨어진다 (그 파일 100-152행 주석에 근거).
#    아래 치환은 **명사만** 바꾼다 (사람 대명사 → subject 라벨, 실내 명사 → 야외 중립어).
PHRASE_SUBS = ((" him", " {subject}"), (" the room", " the scene"),
               (" the table", " {subject}"), (" the floor", " the ground"))


def load_movement_terms(vocab: str):
    """형제 어댑터에서 movement term 표를 **import 로** 가져온다 (복붙하면 갈라진다)."""
    sys.path.insert(0, HERE)
    import trumans_to_lbm_demo as tld
    table = {"base": tld.MOVEMENT_TERMS, "extended": tld.MOVEMENT_TERMS_EXTENDED,
             "crane": tld.MOVEMENT_TERMS_EXTENDED + tld.MOVEMENT_TERMS_CRANE}
    assert vocab in table, f"--movement_vocab 은 {sorted(table)} 중 하나여야 한다: {vocab!r}"
    return table[vocab]


def adapt_phrase(phrase: str, subject: str) -> str:
    for old, new in PHRASE_SUBS:
        phrase = phrase.replace(old, new.format(subject=subject))
    return phrase


def obb_rotation(yaw_rad: float) -> np.ndarray:
    """중력축(+Z) 둘레 yaw 회전. scene graph 의 `obb.R` 과 같은 형태다."""
    cos, sin = float(np.cos(yaw_rad)), float(np.sin(yaw_rad))
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def inside_obb(points: np.ndarray, center: np.ndarray, extent: np.ndarray,
               rot: np.ndarray, inflate: float) -> np.ndarray:
    """(N,3) 점들이 OBB 안인가. `rot` 열이 OBB 축이므로 `rot.T @ (p-c)` 가 로컬 좌표."""
    local = (points - center[None, :]) @ rot
    half = extent[None, :] * (0.5 * inflate)
    return np.all(np.abs(local) <= half, axis=1)


def node_pose_at(node: dict, frame: int, u_meters: float):
    """프레임 `frame` 에서의 (center(3), R(3,3)). 동적이면 track 을, 정적이면 OBB 를 쓴다."""
    obb = node["obb"]
    track = node.get("track") or {}
    if node.get("moving") and track.get("center_smooth"):
        frames = list(track["frames"])
        #    track 은 그 노드가 **보인 프레임만** 담고 있다. 없는 프레임은 가장 가까운 것으로
        #    대신한다 (없다고 원점에 두면 물체가 씬 밖으로 튄다).
        pick = int(np.argmin(np.abs(np.asarray(frames) - frame)))
        center = np.asarray(track["center_smooth"][pick], dtype=np.float64) * u_meters
        yaw = float(track["yaw"][pick]) if track.get("yaw") else float(obb["yaw_rad"])
        return center, obb_rotation(yaw), yaw
    center = np.asarray(obb["center"], dtype=np.float64) * u_meters
    return center, np.asarray(obb["R"], dtype=np.float64), float(obb["yaw_rad"])


def triangulate_frame(points_g, colors, rows, cols, height, width, cam_center,
                      stride: int, edge_depth_frac: float):
    """한 프레임의 depth 격자를 삼각분할한다 → (verts, faces, colors, local_index).

    깊이 불연속에서 면을 자른다: 사각형 4점의 **최대 3D 변 길이**가 카메라까지 평균 거리의
    `edge_depth_frac` 배를 넘으면 버린다. 안 자르면 물체 실루엣과 배경 사이에 고무막이 생겨
    레이캐스트가 빈 공간에서 맞는다 (가림 판정이 통째로 거짓말이 된다).
    """
    grid = np.full((height, width), -1, dtype=np.int64)
    grid[rows, cols] = np.arange(len(rows), dtype=np.int64)
    sub = grid[::stride, ::stride]
    top_left, top_right = sub[:-1, :-1], sub[:-1, 1:]
    bot_left, bot_right = sub[1:, :-1], sub[1:, 1:]
    ok = (top_left >= 0) & (top_right >= 0) & (bot_left >= 0) & (bot_right >= 0)
    if not ok.any():
        return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64),
                np.zeros((0, 3), np.uint8), np.zeros(0, np.int64))

    quad = [arr[ok] for arr in (top_left, top_right, bot_left, bot_right)]
    corner = [points_g[q] for q in quad]
    dist = np.linalg.norm(np.stack(corner, 0) - cam_center[None, None, :], axis=2).mean(0)
    pairs = ((0, 1), (0, 2), (1, 3), (2, 3), (1, 2))
    longest = np.max([np.linalg.norm(corner[i] - corner[j], axis=1) for i, j in pairs], axis=0)
    keep = longest < edge_depth_frac * dist
    quad = [q[keep] for q in quad]

    faces = np.concatenate([np.stack([quad[0], quad[1], quad[2]], axis=1),
                            np.stack([quad[1], quad[3], quad[2]], axis=1)], axis=0)
    used, faces = np.unique(faces, return_inverse=True)
    return (points_g[used].astype(np.float32), faces.reshape(-1, 3),
            colors[used], used)


def build_geometry(args, graph, cloud, u_meters):
    """격자 프레임들을 삼각분할하고 노드 OBB 소속으로 오브젝트를 쪼갠다.

    `background` 는 어느 노드에도 안 속한 표면(지면·먼 배경)이다. 동적 노드 지오메트리는
    **frame 0 격자에서만** 만든다 — 다른 프레임의 동적 점을 배경에 넣으면 움직인 물체가
    정지 잔상으로 쌓여서 카메라가 유령 벽에 막힌다.
    """
    height, width = int(graph["height"]), int(graph["width"])
    transform = np.asarray(graph["frames"]["T_gw"], dtype=np.float64)
    cam_centers = np.asarray(graph["cameras"]["cam_centers_g"], dtype=np.float64) * u_meters

    indices = np.asarray(cloud["indices"])
    points_w = np.asarray(cloud["points_world"], dtype=np.float32)
    colors_all = np.asarray(cloud["colors"])

    nodes = list(graph["nodes"])
    buckets = {node["id"]: {"verts": [], "faces": [], "colors": []} for node in nodes}
    buckets["background"] = {"verts": [], "faces": [], "colors": []}

    grid_frames = [int(f) for f in args.grid_frames.split(",") if f != ""]
    stats = []
    for frame in grid_frames:
        pick = np.flatnonzero(indices[:, 0] == frame)
        assert len(pick), f"프레임 {frame} 에 점이 없다 — --grid_frames 를 확인할 것"
        points_g = ((points_w[pick].astype(np.float64) @ transform[:3, :3].T
                     + transform[:3, 3]) * u_meters)
        verts, faces, colors, _ = triangulate_frame(
            points_g, colors_all[pick], indices[pick, 1], indices[pick, 2],
            height, width, cam_centers[frame], args.stride, args.edge_depth_frac)

        #    버텍스별 소속 노드. 동적이 정적보다 먼저 잡아야 겹칠 때 움직이는 쪽이 이긴다.
        owner = np.full(len(verts), -1, dtype=np.int64)
        ordered = ([i for i, n in enumerate(nodes) if n.get("moving")]
                   + [i for i, n in enumerate(nodes) if not n.get("moving")])
        for i in ordered:
            center, rot, _ = node_pose_at(nodes[i], frame, u_meters)
            extent = np.asarray(nodes[i]["obb"]["extent"], dtype=np.float64) * u_meters
            hit = (owner < 0) & inside_obb(verts.astype(np.float64), center, extent,
                                           rot, args.obb_inflate)
            owner[hit] = i

        for i, node in enumerate(nodes):
            if node.get("moving") and frame != grid_frames[0]:
                #    frame0 이 아닌 동적 점은 **버린다** (배경에도 안 넣는다).
                owner[owner == i] = -2
        emit = {"background": owner == -1}
        for i, node in enumerate(nodes):
            if node.get("moving") and frame != grid_frames[0]:
                continue
            emit[node["id"]] = owner == i

        counts = {}
        for name, mask in emit.items():
            if not mask.any():
                continue
            #    삼각형은 **세 꼭짓점이 같은 오브젝트일 때만** 살린다. 경계 삼각형을 배경에
            #    남기면 물체 실루엣에 배경 껍질이 붙어 focus 레이가 거기 맞는다.
            keep = mask[faces].all(axis=1)
            if not keep.any():
                continue
            sub_faces = faces[keep]
            used, remapped = np.unique(sub_faces, return_inverse=True)
            offset = sum(len(v) for v in buckets[name]["verts"])
            buckets[name]["verts"].append(verts[used])
            buckets[name]["colors"].append(colors[used])
            buckets[name]["faces"].append(remapped.reshape(-1, 3) + offset)
            counts[name] = len(sub_faces)
        stats.append((frame, len(verts), len(faces), counts))

    objects = {}
    for name, parts in buckets.items():
        if not parts["verts"]:
            continue
        objects[name] = {
            "verts": np.concatenate(parts["verts"], axis=0),
            "faces": np.concatenate(parts["faces"], axis=0).astype(np.int64),
            "colors": np.concatenate(parts["colors"], axis=0),
        }
    return objects, stats, cam_centers


def add_ground_plane(objects, graph, u_meters, margin: float):
    """`ground.ground_z` 높이의 평면 1장. 카메라가 지면 아래로 못 가게 막을 면을 준다.

    크기는 점군 XY 범위 + margin 으로 잡는다. LBM 은 render-visible mesh 전체로 씬 범위를
    잡으므로 무한 평면을 넣으면 `scene_size` 가 폭발해 카메라 후보가 씬 밖으로 나간다.
    """
    everything = np.concatenate([obj["verts"] for obj in objects.values()], axis=0)
    lo, hi = everything.min(axis=0), everything.max(axis=0)
    pad = margin * float(np.linalg.norm(hi[:2] - lo[:2]))
    x0, y0, x1, y1 = lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad
    z = float(graph["ground"]["ground_z"]) * u_meters
    objects["ground_plane"] = {
        "verts": np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], np.float32),
        "faces": np.array([[0, 1, 2], [0, 2, 3]], np.int64),
        "colors": np.full((4, 3), 128, np.uint8),
    }


def scene_bounds(objects):
    everything = np.concatenate([obj["verts"] for obj in objects.values()], axis=0)
    lo, hi = everything.min(axis=0).astype(float), everything.max(axis=0).astype(float)
    return {"center": [round((a + b) * 0.5, 5) for a, b in zip(lo, hi)],
            "size": [round(b - a, 5) for a, b in zip(lo, hi)],
            "x_negative": round(lo[0], 5), "x": round(hi[0], 5),
            "y_negative": round(lo[1], 5), "y": round(hi[1], 5),
            "z_negative": round(lo[2], 5), "z": round(hi[2], 5)}


def asset_records(graph, subject_id: str, u_meters: float, blend_path: str):
    """`asset_sheet` / `layout_description.assets` 두 곳에 들어갈 노드별 수치.

    `rotation` 은 **도(degree)** 다 — `cinematographer_stage.py:483,:513` 이 `math.radians` 로
    되돌린다. 라디안을 넣으면 전부 0도 근처로 읽혀 조용히 축정렬 씬이 된다.
    """
    sheet, placed = [], []
    for node in graph["nodes"]:
        center, _rot, yaw = node_pose_at(node, 0, u_meters)
        extent = np.asarray(node["obb"]["extent"], dtype=np.float64) * u_meters
        sheet.append({
            "asset_id": node["id"],
            #    subject 만 `character`. LBM 은 이 값(또는 ARMATURE 타입)으로 인물을 고른다
            #    (`director_stage.py:948`, `:1463-1470`). 점군에는 리그가 없다.
            "asset_type": "character" if node["id"] == subject_id else "prop",
            "description": node["label"],
            "width": round(float(extent[0]), 4), "depth": round(float(extent[1]), 4),
            "height": round(float(extent[2]), 4),
            "front_view_url": "", "top_view_url": "", "left_view_url": "", "thumbnail_url": "",
            "main_file_path": blend_path,
        })
        placed.append({
            "asset_id": node["id"],
            "location": {"x": round(float(center[0]), 5), "y": round(float(center[1]), 5),
                         "z": round(float(center[2]), 5)},
            "rotation": {"x": 0.0, "y": 0.0, "z": round(float(np.degrees(yaw)), 4)},
            "dimensions": {"x": round(float(extent[0]), 5), "y": round(float(extent[1]), 5),
                           "z": round(float(extent[2]), 5)},
        })
    return sheet, placed


def describe_scene(graph, subject_id: str) -> str:
    labels = [node["label"] for node in graph["nodes"] if node["id"] != subject_id]
    subject = next(n["label"] for n in graph["nodes"] if n["id"] == subject_id)
    tail = f" among the {', the '.join(labels)}" if labels else ""
    return f"An outdoor scene captured from a moving camera, showing the {subject}{tail}."


def write_run_script(args, scene: str, demo_root: str, fps: float) -> str:
    """`trumans_to_lbm_demo.emit_frame_shift_runner` 와 같은 env 를 깐다 (frame-shift 만 뺐다).

    Vista demo root 는 창을 안 자르므로 `TRUMANS_FRAME_*` / `BLENDER_USER_SCRIPTS` 훅이 필요
    없다. `STORYBLENDER_MOVEMENT_VOCAB` 은 `== "extended"` **정확 일치**로만 확장 분기를 켜므로
    base 가 아니면 무조건 `extended` 를 내보낸다.
    """
    lbm_root = path.abspath(path.join(HERE, "..", "..", "models", "Planner", "Look-Before-Move"))
    lines = [
        "#!/bin/bash",
        f"# Vista4D demo root `{scene}` 를 원본 LBM 4단계에 태운다 (어댑터가 생성).",
        "#   $1 = run_id   $2 = CUDA device",
        "set -o pipefail",
        f"LBM={lbm_root}",
        f"PY={args.lbm_python}",
        f"export STORYBLENDER_BLENDER_EXE={args.blender}",
        'export CUDA_VISIBLE_DEVICES=${2:-0}',
        f"export ANYLLM_API_BASE={args.api_base}",
        "export ANYLLM_API_KEY=dummy",
        "export ANYLLM_PROVIDER=openai",
        f"export STORYBLENDER_VISION_MODEL={args.vision_model}",
        #    VideoEngineer/Editor 는 `shutil.which("ffmpeg")` 로 찾고 실패하면 하드코딩된
        #    윈도우 경로 `C:\\ffmpeg\\bin\\ffmpeg.exe` 로 떨어져 FileNotFoundError 를 낸다
        #    (`VideoEngineer/video_stage.py:97`). conda env 에도 /usr/bin 에도 ffmpeg 이 없다.
        f"export PATH={args.ffmpeg_path}:$PATH",
        #    씬이 1개뿐이라 LBM 의 다중 씬 분기를 우회해야 한다 (TRUMANS 와 같은 이유).
        "export LBM_SINGLE_SCENE_FALLBACK=1",
        f"export STORYBLENDER_MOVEMENT_VOCAB="
        f"{'base' if args.movement_vocab == 'base' else 'extended'}",
        f"export STORYBLENDER_TRAJECTORY_SCALE={args.trajectory_scale:g}",
        #    LBM 이 **실제로 렌더한** per-frame 카메라를 받아 적는다 (`lbm_camera_dump_startup.py`).
        #    이게 없으면 평가할 카메라가 없어서 VideoEngineer 를 통째로 다시 돌려야 한다.
        #    startup 디렉토리는 TRUMANS 쪽 것을 그대로 쓴다 — 같이 사는 frame-shift 훅은
        #    `TRUMANS_FRAME_OFFSET` 이 0(미설정)이면 즉시 return 하므로 Vista 에 영향이 없다.
        f"export BLENDER_USER_SCRIPTS={args.startup_scripts}",
        f"export LBM_CAMERA_DUMP_DIR={path.join(args.out, '_camdump')}/${{1:-vista_{scene}}}",
        #    훅은 append 모드다. 재실행 때 이어붙지 않게 비우고 시작한다.
        'rm -rf "$LBM_CAMERA_DUMP_DIR"',
        "",
        f'$PY "$LBM/Engine/run_full_pipeline.py" --demo-root {demo_root} \\',
        f'    --run-id "${{1:-vista_{scene}}}" --fps {fps:g} \\',
        #    기본값 `fast` 는 seed 탐색을 건너뛰어 벽만 찍힌 프리뷰 1장을 낸다 (TRUMANS 실측).
        f"    --camera-quality {args.camera_quality}",
        "",
    ]
    script = path.join(args.out, f"{scene}__run.sh")
    with open(script, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    return script


def main(args):
    scene = path.basename(path.normpath(args.scene_dir))
    graph_path = path.join(args.scene_dir, "scene_graph.json")
    cloud_path = path.join(args.scene_dir, "cloud.npz")
    for target in (graph_path, cloud_path):
        assert path.isfile(target), f"없다: {target}"
    with open(graph_path, encoding="utf-8") as file:
        graph = json.load(file)
    cloud = np.load(cloud_path, mmap_mode="r")

    nodes = list(graph["nodes"])
    assert nodes, f"{graph_path} 에 노드가 없다"
    #    subject 후보 우선순위: `moving` → **`dyn_*` 로 추적된 노드** → 전체.
    #    가운데 단계가 필요한 이유: `moving` 은 이동량 임계를 넘은 것만 True 라, 사람이 제자리에서
    #    작업하는 씬(avocado-slice)은 전 노드가 False 가 된다. 그때 "화면 면적 최대"로 떨어지면
    #    **테이블 상판**(두께 0.026 u)이 subject 로 뽑혀 `u_meters` 가 70 이 되고 씬이 404 m 로
    #    부풀었다 (실측). `dyn_*` 는 동적 track 에서 나온 노드라 배우일 가능성이 높다.
    moving = [n for n in nodes if n.get("moving")]
    dynamic = [n for n in nodes if str(n["id"]).startswith("dyn_")] if args.subject_prefer_dyn else []
    pool = moving or dynamic or nodes
    subject_id = args.subject_id or max(pool, key=lambda n: n.get("max_area_frac", 0.0))["id"]
    subject = next(n for n in nodes if n["id"] == subject_id)

    #    u → m 배율. subject OBB 높이를 `--subject_height_m` 에 맞춘다 (§docstring).
    u_meters = args.u_meters
    if u_meters <= 0.0:
        height_u = float(subject["obb"]["extent"][2])
        assert height_u > 0.0, f"{subject_id} 의 OBB 높이가 0 이다"
        u_meters = args.subject_height_m / height_u

    demo_root = path.join(args.out, scene)
    makedirs(path.join(demo_root, "animated_models"), exist_ok=True)
    makedirs(path.join(demo_root, "layout_script"), exist_ok=True)

    objects, stats, cam_centers = build_geometry(args, graph, cloud, u_meters)
    if args.ground_plane:
        add_ground_plane(objects, graph, u_meters, args.ground_margin)
    bounds = scene_bounds(objects)

    blend_path = path.join(demo_root, f"{scene}.blend")
    num_frames = int(graph["num_frames"])
    fps = float(graph["fps"])
    geom_meta = {
        "frame_start": 0, "frame_end": num_frames - 1, "fps": fps,
        "resolution": [int(graph["width"]), int(graph["height"])],
        "scene_bounds": bounds, "objects": [],
    }
    payload = {}
    for name, obj in objects.items():
        node = next((n for n in nodes if n["id"] == name), None)
        origin = np.zeros(3)
        keyframes = []
        if node is not None:
            origin, _rot, _yaw = node_pose_at(node, 0, u_meters)
            if node.get("moving"):
                for frame in range(num_frames):
                    center, _r, yaw = node_pose_at(node, frame, u_meters)
                    keyframes.append([frame, float(center[0]), float(center[1]),
                                      float(center[2]), float(np.degrees(yaw))])
        geom_meta["objects"].append({
            "name": name, "kind": (node or {}).get("kind", "bg"),
            "origin": [float(v) for v in origin], "keyframes": keyframes,
        })
        payload[f"v_{name}"] = obj["verts"]
        payload[f"f_{name}"] = obj["faces"].astype(np.int32)
        payload[f"c_{name}"] = obj["colors"]

    geom_npz = path.join(demo_root, "_geom.npz")
    geom_json = path.join(demo_root, "_geom.json")
    np.savez_compressed(geom_npz, **payload)
    with open(geom_json, "w", encoding="utf-8") as file:
        json.dump(geom_meta, file, ensure_ascii=False, indent=1)

    result = subprocess.run(
        [args.blender, "--background", "--python", BLEND_WORKER, "--",
         "--geom", geom_npz, "--meta", geom_json, "--out", blend_path],
        capture_output=True, text=True)
    assert path.isfile(blend_path), \
        f"blender 가 .blend 를 못 만들었다 (rc={result.returncode})\n{result.stdout[-4000:]}\n{result.stderr[-2000:]}"

    terms = load_movement_terms(args.movement_vocab)
    lookup = dict(terms)
    if args.movement_term:
        assert args.movement_term in lookup, \
            f"--movement_term 이 {args.movement_vocab} 어휘에 없다: {args.movement_term!r}"
        term, phrase = args.movement_term, lookup[args.movement_term]
    else:
        term, phrase = terms[0]
    subject_label = f"the {subject['label']}"
    verb = "moves through the scene" if subject.get("moving") else "stands in the scene"
    base_text = args.shot_description or f"The {subject['label']} {verb}"
    shot_description = f"{base_text.rstrip().rstrip('.')}{adapt_phrase(phrase, subject_label)}"

    sheet, placed = asset_records(graph, subject_id, u_meters, blend_path)
    summary = describe_scene(graph, subject_id)
    files = {
        path.join(demo_root, "animated_models", "animated_models_v1.json"): {
            "story_summary": summary,
            "asset_sheet": sheet,
            "storyboard_outline": [{
                "scene_id": 1, "scene_description": summary,
                "shots": [{"shot_id": 1, "shot_description": shot_description,
                           "movement_term_target": term}],
            }],
        },
        path.join(demo_root, "animated_models", "selected_animation_v1.json"): {
            "note": "Vista4D: 동적 노드 궤적은 .blend 에 키프레임으로 구워져 있다. "
                    "LBM 은 이 파일의 내용을 읽지 않는다 (경로 존재만 확인한다).",
            "video": scene, "fps": fps, "frame_start": 0, "frame_end": num_frames - 1,
            "animations": [{"asset_id": n["id"], "action_name": "baked_in_blend",
                            "frame_start": 0, "frame_end": num_frames - 1}
                           for n in nodes if n.get("moving")],
        },
        path.join(demo_root, "layout_script", "layout_script_v1.json"): {
            "asset_sheet": sheet,
            "scene_details": [{
                "scene_id": 1,
                "scene_setup": {
                    "scene_type": args.scene_type,
                    "asset_ids": [n["id"] for n in nodes],
                    "layout_description": {
                        "description": summary,
                        #    LBM 기본값 ±10 을 그대로 두면 카메라 후보가 점군 밖으로 나간다.
                        "scene_size": {"x_negative": bounds["x_negative"], "x": bounds["x"],
                                       "y_negative": bounds["y_negative"], "y": bounds["y"]},
                        "assets": placed,
                    },
                },
            }],
        },
        path.join(demo_root, "layout_measured.json"): {
            "blend": blend_path, "scene_name": "Scene", "frame": 0,
            "frame_start": 0, "frame_end": num_frames - 1, "fps": fps,
            "resolution": [int(graph["width"]), int(graph["height"])],
            "engine": "BLENDER_EEVEE_NEXT", "n_objects": len(objects),
            "scene_bounds": bounds,
            "assets": [{"asset_id": rec["asset_id"], "type": "MESH", "n_meshes": 1,
                        "center": [rec2["location"][k] for k in "xyz"],
                        "size": [rec2["dimensions"][k] for k in "xyz"],
                        "yaw_deg": rec2["rotation"]["z"]}
                       for rec, rec2 in zip(sheet, placed)],
            "missing": [],
        },
        path.join(demo_root, "_vista.json"): {
            "format": "vista_lbm_demo_v1", "scene": scene,
            "scene_dir": path.abspath(args.scene_dir),
            #    되돌리기용. LBM 이 뱉는 카메라는 이 배율이 곱해진 G frame 좌표라
            #    `u = blender / u_meters`, `world = T_wg @ [u,1]` 로 되돌린다.
            "u_meters": u_meters, "S_da3": float(graph["scale"]["S"]),
            "T_gw": graph["frames"]["T_gw"], "T_wg": graph["frames"]["T_wg"],
            "subject_id": subject_id, "subject_label": subject["label"],
            "movement_term_target": term, "shot_description": shot_description,
            "num_frames": num_frames, "fps": fps,
            "grid_frames": [int(f) for f in args.grid_frames.split(",") if f != ""],
            "stride": args.stride, "edge_depth_frac": args.edge_depth_frac,
            "obb_inflate": args.obb_inflate, "ground_plane": args.ground_plane,
            "scene_bounds": bounds,
        },
    }
    for target, body in files.items():
        with open(target, "w", encoding="utf-8") as file:
            json.dump(body, file, ensure_ascii=False, indent=1)

    script = write_run_script(args, scene, demo_root, fps)

    print(f"{'scene':<16}{scene}")
    print(f"{'subject':<16}{subject_id}  {subject['label']}  "
          f"(moving={bool(subject.get('moving'))})")
    print(f"{'u_meters':<16}{u_meters:.4f}   (1 u = {u_meters:.3f} m; S_da3="
          f"{float(graph['scale']['S']):.4f})")
    print(f"{'frames':<16}0..{num_frames - 1} @ {fps:g} fps")
    print(f"{'movement':<16}{term}")
    print(f"{'shot':<16}{shot_description}")
    print()
    print(f"{'grid_frame':>10}{'verts':>10}{'faces':>10}  per-object faces")
    for frame, nv, nf, counts in stats:
        detail = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"{frame:>10}{nv:>10d}{nf:>10d}  {detail}")
    print()
    print(f"{'object':<16}{'verts':>10}{'faces':>10}")
    for name, obj in sorted(objects.items()):
        print(f"{name:<16}{len(obj['verts']):>10d}{len(obj['faces']):>10d}")
    bnd = bounds
    print(f"\nbounds x[{bnd['x_negative']:.2f},{bnd['x']:.2f}] "
          f"y[{bnd['y_negative']:.2f},{bnd['y']:.2f}] z[{bnd['z_negative']:.2f},{bnd['z']:.2f}]")
    print(f"cam0 {np.round(cam_centers[0], 3)}  cam48 {np.round(cam_centers[-1], 3)}")
    print(f"\n{'demo_root':<16}{demo_root}")
    print(f"{'blend':<16}{blend_path}")
    print(f"{'run':<16}bash {script} vista_{scene} 0")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scene_dir", required=True, type=str)   # out/<scene> (cloud.npz+graph)
    parser.add_argument("--out", required=True, type=str)         # demo root 를 깔 디렉토리
    parser.add_argument("--subject_id", default="", type=str)     # 비면 max_area_frac 최대 동적 노드
    # `moving` 이 전부 False 일 때 `dyn_*` 노드를 먼저 본다. 끄면 예전처럼 전 노드에서 고른다.
    parser.add_argument("--subject_prefer_dyn", action="store_true", default=True)
    parser.add_argument("--no_subject_prefer_dyn", dest="subject_prefer_dyn", action="store_false")
    # u → m 배율. 0 이면 --subject_height_m 로 자동 계산한다.
    parser.add_argument("--u_meters", default=0.0, type=float)
    parser.add_argument("--subject_height_m", default=1.8, type=float)
    # 배경 메시를 만들 프레임들. 카메라가 훑은 영역 전체를 덮으려면 여러 장이 필요하다.
    parser.add_argument("--grid_frames", default="0,12,24,36,48", type=str)
    parser.add_argument("--stride", default=3, type=int)          # 격자 다운샘플 (720x1280 기준)
    parser.add_argument("--edge_depth_frac", default=0.05, type=float)  # 깊이 불연속 면 컷
    parser.add_argument("--obb_inflate", default=1.05, type=float)      # 소속 판정 OBB 여유
    parser.add_argument("--ground_plane", action="store_true", default=True)
    parser.add_argument("--no_ground_plane", dest="ground_plane", action="store_false")
    parser.add_argument("--ground_margin", default=0.15, type=float)
    parser.add_argument("--scene_type", default="outdoor", type=str)
    parser.add_argument("--shot_description", default="", type=str)
    parser.add_argument("--movement_term", default="", type=str)  # 비면 어휘의 첫 항목
    parser.add_argument("--movement_vocab", default="crane", type=str)  # base|extended|crane
    parser.add_argument("--trajectory_scale", default=5.0, type=float)
    parser.add_argument("--camera_quality", default="quality", type=str)
    parser.add_argument("--blender",
                        default="/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender",
                        type=str)
    parser.add_argument("--lbm_python",
                        default="/data1/cympyc1785/miniconda3/envs/lbm/bin/python3.11", type=str)
    parser.add_argument("--api_base", default="http://127.0.0.1:22002/v1", type=str)
    parser.add_argument("--vision_model", default="Qwen/Qwen3-VL-30B-A3B-Instruct", type=str)
    #    형제 어댑터와 같은 경로. /usr/bin 에는 ffmpeg 이 없다 (실측 — VideoEngineer 가 죽었다).
    parser.add_argument("--ffmpeg_path", default="/data1/cympyc1785/tools/bin", type=str)
    #    카메라 덤프 훅이 사는 `BLENDER_USER_SCRIPTS`. TRUMANS 쪽 디렉토리를 재사용한다.
    parser.add_argument("--startup_scripts", type=str, default=path.abspath(
        path.join(HERE, "..", "out", "lbm_demos_w", "_frame_shift")))
    main(parser.parse_args())
