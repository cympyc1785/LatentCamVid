"""LBM 이 띄우는 **모든** Blender 프로세스에서 TRUMANS 애니메이션을 앞으로 당긴다 (startup 훅).

**왜 이게 필요한가.** LBM 의 렌더 워커는 씬 프레임 범위를 무조건 1..frame_count 로 덮어쓴다
(`VideoEngineer/blender_render_worker.py:1141-1142` `scene.frame_start = 1; scene.frame_end =
frame_count`). frame_count 는 shot 길이(≈1.5 s)에서 나오므로, TRUMANS take 처럼 2077 프레임짜리
`.blend` 를 물리면 **어느 shot 이든 항상 take 의 첫 1.5 초만** 렌더된다. 실측: 15 shot 이 전부
같은 구간을 찍었다 (Task #88 의 D1). 즉 `Actions/<seq>.txt` 의 `(start, end)` 창이 통째로 버려진다.

그 창을 살리는 방법은 두 가지뿐이다 — (1) LBM 워커를 고쳐 frame_start 를 창 시작으로 놓거나,
(2) **`.blend` 쪽 애니메이션을 당겨서 창 시작이 프레임 1 에 오게** 하거나. 사용자가 "LBM 코드
0줄 수정"을 못박았으므로 (2) 다. 그런데 창마다 1.66 GB blend 를 복사해 저장하면 15 창 × 7 편 =
170 GB 라 못 쓴다.

그래서 **디스크에 아무것도 안 쓰고** 로드 직후 메모리에서만 당긴다. Blender 는
`BLENDER_USER_SCRIPTS/startup/*.py` 를 기동 시 자동 import 하고, 거기서 등록한 `load_post` 핸들러는
커맨드라인으로 지정한 `.blend` 가 로드된 **직후** 불린다 (실측 확인: 빈 startup 파일에 한 번,
실제 blend 에 한 번, 총 2회 발화하므로 `bpy.data.filepath` 로 거른다). LBM 이 blender 를 몇 번
띄우든(Director probe / Cinematographer preview·quality / VideoEngineer render) 전부 같은 훅을 탄다.

**LBM 은 `.blend` 를 저장하지 않으므로** 이 변형은 프로세스 안에서만 살고 원본은 그대로다.

읽는 env:
    TRUMANS_FRAME_OFFSET   키프레임에 더할 값. 창 시작 `s` 를 프레임 1 로 보내려면 `1 - s`.
                           0 이거나 없으면 아무 것도 안 한다 (= 기존 동작과 동일).
    TRUMANS_FRAME_COUNT    (선택) 창 길이. 주면 scene.frame_end 를 여기에 맞춘다. LBM 이 다시
                           덮어쓰므로 진단용이다.

사용 예시:
    export BLENDER_USER_SCRIPTS=<out>/_frame_shift      # 이 파일이 그 아래 startup/ 에 있다
    export TRUMANS_FRAME_OFFSET=-50                     # 원본 프레임 51 -> 씬 프레임 1
    python Engine/run_full_pipeline.py --demo-root <out>/<uuid>__w01_f0051_0069 ...
"""
import os

import bpy
from bpy.app.handlers import persistent


def used_actions():
    """실제로 assign 된 action 만. TRUMANS blend 은 미사용 take 가 254개라 전량 훑으면 느리다."""
    actions, seen = [], set()

    def take(anim_data):
        action = getattr(anim_data, "action", None)
        if action is not None and action.name not in seen:
            seen.add(action.name)
            actions.append(action)

    for scene in bpy.data.scenes:
        take(scene.animation_data)
        for obj in scene.objects:
            take(obj.animation_data)
            data = getattr(obj, "data", None)
            if data is not None:
                take(getattr(data, "animation_data", None))
                shape_keys = getattr(data, "shape_keys", None)
                if shape_keys is not None:
                    take(shape_keys.animation_data)
    return actions


def shift_action(action, offset: float):
    """fcurve 키를 통째로 평행이동. `keyframe_points` 를 파이썬 루프로 돌면 zzy3 하나가 405
    fcurve × 2077 키라 수 초씩 먹는다 — `foreach_get/set` 로 배열째 옮긴다."""
    for curve in action.fcurves:
        count = len(curve.keyframe_points)
        if count == 0:
            continue
        for attr, stride in (("co", 2), ("handle_left", 2), ("handle_right", 2)):
            buffer = [0.0] * (count * stride)
            curve.keyframe_points.foreach_get(attr, buffer)
            buffer[0::stride] = [value + offset for value in buffer[0::stride]]
            curve.keyframe_points.foreach_set(attr, buffer)
        curve.update()


@persistent
def on_load(_dummy):
    # 빈 startup 파일 로드에도 발화한다 — 실제 파일일 때만 손댄다.
    if not bpy.data.filepath:
        return
    try:
        offset = int(float(os.environ.get("TRUMANS_FRAME_OFFSET", "0") or 0))
    except ValueError:
        offset = 0
    if offset == 0:
        return

    actions = used_actions()
    for action in actions:
        shift_action(action, float(offset))

    count = os.environ.get("TRUMANS_FRAME_COUNT", "")
    for scene in bpy.data.scenes:
        # blend 이 들고 있던 frame_step=2 는 LBM 이 덮어쓰지만, Director 의 scene context 프리뷰는
        # 덮어쓰기 전에 찍힌다. 여기서 미리 1 로 놓고 창 시작(=프레임 1)에 세워둔다.
        scene.frame_step = 1
        scene.frame_start = 1
        if count:
            try:
                scene.frame_end = max(1, int(count))
            except ValueError:
                pass
        scene.frame_set(1)

    print(f"[trumans_frame_shift] offset {offset:+d} applied to {len(actions)} actions "
          f"({bpy.data.filepath})")


def register():
    if on_load not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_load)


def unregister():
    if on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load)


# startup/ 의 모듈은 import 시점에 register() 가 자동 호출되지 않는 경우가 있어(4.x 는 호출한다)
# import 시점에도 한 번 건다. `register()` 가 중복 등록을 막는다.
register()
