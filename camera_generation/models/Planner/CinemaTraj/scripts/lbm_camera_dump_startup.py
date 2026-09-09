"""LBM 이 **실제로 렌더한** per-frame 카메라를 그대로 받아 적는다 (Blender startup 훅).

**왜 재구현이 아니라 훅인가.** LBM 이 내놓는 카메라는 `camera_packages/*/scene_N_shot_M_camK.json`
의 `trajectory_keyframes` 3~5개다. 그 사이는 Blender 가 채운다 — 그런데 그 채우는 과정이 단순
보간이 아니다: `blender_render_worker.py:475 _subdivide_rotation_keyframes` 가 인접 키의 쿼터니언
각차가 `ROTATION_CONTINUITY_THRESHOLD_RAD` 를 넘으면 slerp 로 중간 키를 **더 심고**, `:537
_normalize_trajectory_plan` 이 preset 에 따라 lens 램프를 얹고, `video_runtime.py:417
_set_action_interpolation` 이 fcurve easing(`ease_in`/`ease_in_out`/`ease_out`)을 붙인다. 게다가
가시성 검증이 실패하면 `blender_render_worker.py:1094` 가 motion_scale 0.75/0.5/0.35/0.2 로
궤적을 **통째로 갈아끼운다** — 즉 카메라 package 의 키프레임이 렌더된 카메라라는 보장이 없다.
이걸 numpy 로 다시 짜면 "LBM 카메라를 평가했다"가 아니라 "LBM 카메라를 흉내낸 것을 평가했다"가
된다. 평가 대상이 베이스라인이므로 그 차이는 그대로 결론의 오차가 된다.

**왜 `render_pre` 가 아니라 `render_complete` 인가 (실측).** 처음엔 프레임마다 발화하는
`render_pre` 에서 `camera.matrix_world` 를 읽었는데, 110프레임 전량이 **같은 값**으로 나왔다.
Blender 4.5.9 에 최소 재현을 짜서 확인한 결과(키 3개로 x 를 0→2 로 옮기고 animation 렌더):

    handler      frame   matrix_world.x   evaluated_get(depsgraph).x   (기대 0/1/2)
    render_pre     1          2.0                  2.0
    render_pre     5          2.0                  2.0
    render_pre     9          2.0                  2.0

`bpy.ops.render.render(animation=True)` 는 depsgraph **사본** 위에서 애니메이션을 평가하므로,
원본 datablock 의 `matrix_world` 는 마지막 `keyframe_insert` 때 값에 그대로 멈춰 있다.
`evaluated_get()` 도 렌더 중에는 같은 값을 준다. fcurve 를 직접 `evaluate()` 하는 길도 재봤지만
Blender 4.4+ 의 slotted action 에서는 object 와 camera-data 가 한 action 을 공유해
`action.fcurves` 가 object 슬롯만 돌려줘 `lens` 가 통째로 빠진다.

그래서 **렌더가 끝난 뒤** `render_complete` 에서 `frame_start..frame_end` 를 `frame_set` 으로
되짚으며 읽는다. 같은 재현 실험에서 이 경로만 x=0/1/2, lens=24/25/26 을 정확히 돌려줬다.
`render_complete` 는 animation 렌더당 **한 번** 발화하고(`render_post` 는 프레임마다), 그 시점에
임시 카메라(`video_render_*`)와 씬 상태가 아직 살아 있다 — `_remove_temp_camera` /
`_restore_scene` 은 `bpy.ops.render.render` 가 반환한 **뒤**에 돈다. `frame_current` 는 우리가
읽기 전후로 저장·복원하므로 LBM 의 하류 로직에 남는 흔적이 없다.

`trumans_frame_shift_startup.py` 와 같은 자리(`BLENDER_USER_SCRIPTS/startup/`)에 산다. LBM 은
`.blend` 를 저장하지 않고 이 훅은 씬을 읽기만 하므로 렌더 결과에 영향이 없다 — 즉 **LBM 코드
0줄 수정**이고, 덤프를 켠 실행과 안 켠 실행의 렌더 픽셀이 같다.

읽는 env:
    LBM_CAMERA_DUMP_DIR   덤프를 쓸 디렉토리. 비어 있거나 없으면 아무 것도 안 한다
                          (= 기존 동작과 완전히 동일).

쓰는 것: `<dir>/cam_<pid>.jsonl` — 한 줄에 프레임 하나. LBM 은 카메라마다
`bpy.ops.render.render` 를 다시 부르므로 한 파일에 여러 카메라가 이어 붙고, `filepath` 로 갈린다.
    {"frame":  씬 프레임 번호 (LBM 이 1..frame_count 로 덮어쓴 그 번호),
     "filepath": scene.render.filepath — 어느 camera package 의 렌더인지 가르는 유일한 키,
     "camera": scene.camera.name,
     "matrix_world": 4x4 행렬 (Blender GL 규약: X right / Y up / Z backward),
     "lens_mm": .., "sensor_width": .., "sensor_fit": .., "res": [W, H], "res_pct": ..}

**규약 변환은 여기서 안 한다.** `matrix_world` 는 Blender GL 그대로 싣고, OpenCV 로 바꾸는 건
읽는 쪽(`lbm_camera_to_poses.py`)이 `c2w_cv = c2w_gl @ diag(1,-1,-1,1)` 로 한다 —
`trumans_gt_render.py:64 GL2CV` 와 같은 식이다. 훅 안에서 바꾸면 그 한 줄이 두 군데로 갈라진다.

사용 예시:
    export BLENDER_USER_SCRIPTS=<out>/_frame_shift          # 이 파일이 그 아래 startup/ 에 있다
    export LBM_CAMERA_DUMP_DIR=<out>/_camdump/<run_id>
    python VideoEngineer/run_video_engineer.py --camera-handoff-path ... --run-id ...
"""
import json
import os

import bpy
from bpy.app.handlers import persistent


def dump_path():
    """덤프 대상 파일. env 가 비면 None — 그때는 훅이 즉시 반환한다."""
    directory = os.environ.get("LBM_CAMERA_DUMP_DIR", "").strip()
    if not directory:
        return None
    os.makedirs(directory, exist_ok=True)
    # 프로세스별 파일. LBM 은 Blender 를 단계마다 새로 띄우므로 pid 로 갈리고, 같은 프로세스가
    # 여러 카메라를 연속으로 렌더해도 `filepath` 로 다시 갈린다.
    return os.path.join(directory, f"cam_{os.getpid()}.jsonl")


def camera_record(scene, camera):
    """현 프레임의 카메라 상태 한 줄. `scene.frame_set` 직후에 부르는 것을 전제로 한다."""
    data = camera.data
    return {
        "frame": int(scene.frame_current),
        # 이 값이 camera package 를 가르는 키다. LBM 은 `<...>/scene_1_shot_1_cam1/frames/frame_`
        # 꼴로 놓으므로(`blender_render_worker.py:1173`) 부모 두 단계면 카메라가 특정된다.
        "filepath": str(scene.render.filepath),
        "camera": camera.name,
        "matrix_world": [[float(v) for v in row] for row in camera.matrix_world],
        "lens_mm": float(getattr(data, "lens", 0.0)),
        "sensor_width": float(getattr(data, "sensor_width", 36.0)),
        "sensor_height": float(getattr(data, "sensor_height", 24.0)),
        "sensor_fit": str(getattr(data, "sensor_fit", "AUTO")),
        "shift_x": float(getattr(data, "shift_x", 0.0)),
        "shift_y": float(getattr(data, "shift_y", 0.0)),
        "res": [int(scene.render.resolution_x), int(scene.render.resolution_y)],
        "res_pct": int(scene.render.resolution_percentage),
    }


@persistent
def on_render_complete(scene, _depsgraph=None):
    """animation 렌더 하나가 끝나면 그 구간을 `frame_set` 으로 되짚어 받아 적는다.

    `frame_start..frame_end` 를 `frame_step` 으로 도는 것은 방금 렌더된 프레임 집합과 정확히
    같다 (`blender_render_worker.py:1141-1148` 이 셋을 다 직접 세팅한다).
    """
    target = dump_path()
    camera = scene.camera
    if target is None or camera is None or camera.data is None:
        return

    restore = int(scene.frame_current)
    step = max(1, int(scene.frame_step))
    records = []
    try:
        frame = int(scene.frame_start)
        while frame <= int(scene.frame_end):
            scene.frame_set(frame)
            records.append(camera_record(scene, camera))
            frame += step
    finally:
        # 읽으려고 움직인 프레임을 되돌린다. LBM 하류가 frame_current 를 보더라도 흔적이 없게.
        scene.frame_set(restore)

    # 한 번에 쓴다. 렌더가 이미 끝난 뒤라 중간에 죽어 잃을 구간이 없다.
    with open(target, "a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def register():
    if on_render_complete not in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.append(on_render_complete)


def unregister():
    if on_render_complete in bpy.app.handlers.render_complete:
        bpy.app.handlers.render_complete.remove(on_render_complete)


register()
