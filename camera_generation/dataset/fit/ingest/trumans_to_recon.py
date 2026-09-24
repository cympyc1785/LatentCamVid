"""TRUMANS action 구간 하나 → LBM-Lite 가 그대로 읽는 `recon_and_seg` + `seg_instances`.

**왜 필요한가.** LBM-Lite 의 입력 규약(`scene_graph/io.py:load_scene`)은 Vista4D 배포본 포맷이다:
RGBD + 카메라 + SAM3 track. TRUMANS 는 그 중 **아무것도** 그 포맷으로 주지 않는다. 대신 씬 전체가
`.blend` 로 있으니 우리가 렌더해서 만들면 되고, 그러면 depth 와 사람 마스크가 *추정치가 아니라
정답*이 된다 (DA3 는 정지 카메라에서도 focal 이 −14% 드리프트하고 SAM3 는 실루엣에서 샌다).

**소스 카메라를 합성하는 이유 — 이건 선택이 아니라 데이터의 제약이다.**
- `<seq>_camera_pose.pkl` 은 recording 67편 중 **2편**에만 있다 (00add26c, 0aa05d5a).
- `.blend` 안의 CAMERA 오브젝트 4개는 **전부 정지**다 (`anim=False`, constraint/parent 없음).
정지 카메라를 소스로 쓰면 시차가 0 이라 Lite 의 `τ`(= 이동량/깊이) 축과 view-angle 축이 통째로
무의미해진다. 그래서 소스 카메라를 합성하되 **있는 2편의 통계에 맞춘다**. 단 그 통계는
**사람이 걷느냐 마느냐로 갈린다** — 49프레임 창을 사람 보행 비율로 나눠 재면:

    창 종류            00add26c (창 206/15)        0aa05d5a (창 143/11)
    ------------------------------------------------------------------------
    still (보행 ≤10%)  cam net 0.440  dt 0.0122    cam net 0.548  dt 0.0137
    walk  (보행 ≥80%)  cam net 1.314  dt 0.0309    cam net 0.588  dt 0.0141
                       (사람 net 1.250 → 추종 1.05x)  (사람 net 1.046 → 0.56x)

즉 원래 인용하던 "net 0.58~0.62 m"는 **still 창이 압도적으로 많아서 나온 정지 통계**고,
보행 구간에는 적용하면 안 된다. 실제 카메라 2대는 보행에 대해 서로 다르게 반응한다 — 하나는
거의 1:1 로 따라가고(1.05x) 하나는 사실상 반응하지 않는다(walk 0.588 ≈ still 0.548).
`--track_gain 0.6` 은 그 둘 사이에 앉는다. 채굴한 보행 클립 실측이 net 1.194 m / dt 0.0283 로
00add26c 의 추종 카메라와 거의 같다 — **과한 게 아니라 두 실제 모드 중 하나를 재현한 것**이다.

정리하면 "높이 고정 + 사람 추종 + 느린 호(arc)" 이고, 추종 강도만 실제로 양극단이다.

**시작점은 LBM 방식, 앵커는 사람.** `trumans_scene_probe.py` 가 사람 가슴을 중심으로
(방위각 x 고도 x 거리) 격자를 깔고 **실제 광선**으로 ① 시선이 뚫리는지 ② 벽 속/벽에 붙었는지
③ 공중에 떠 있는지를 잰다. 그 통과분에서만 샘플링하므로, DA3 depth shell 위에서 하던
"관측된 표면보다 뒤인가" 근사(`lbm/gates.py` G1)를 안 쓴다. 사람 외 물체는 앵커하지 않는다
(사용자 지시).

**단계** (Blender 3회 + 변환 1회):
    1. probe(격자)   ~4 s   사람 궤적 + 설 수 있는 자리
    2. 궤적 합성      즉시   시작 pose 샘플 + preset 호 + 사람 추종 look-at
    3. probe(검증)   ~4 s   합성한 49 pose 를 **프레임별로** 다시 광선 검사
    4. gt_render     ~150 s RGB(EEVEE 16spp) + depth/index(Cycles 1spp CPU)
    5. 변환          ~10 s  video.mp4 / depths(EXR f16) / masks / cameras.npz / seg_instances

**규약.**
- `cam_c2w` 는 **OpenCV** (X right / Y down / Z forward), `cam_c2w[0] = I` 로 재앵커한다
  (`load_scene` 하류가 그걸 assert 한다). rigid 변환이라 metre 스케일은 그대로 남는다.
- depth 는 Blender z-planar metre. `utils.media.load_depths` 기본이 **float16** 이라 배경
  sentinel 1e10 은 그대로 쓰면 inf 가 된다 → `--sky_depth`(기본 1000 m)로 clamp 하고 그 픽셀을
  `sky_mask` 로 표시한다.
- 사람 = object pass index 1 (`trumans_gt_render.py` 예약) → `dynamic_mask` + `seg_instances` 의
  track 1. 나머지 index 는 `seg_instances_static/`.

env: `vista4d` (OpenEXR / imageio / numpy). Blender 는 subprocess.

예시:
    python fit/ingest/trumans_to_recon.py --recording 00add26c-7a26-4a61-b192-b97aa493b3f3 \
        --action 3 --seed 0 --out_video tru_00add26c_a03
    python fit/ingest/trumans_to_recon.py --recording 2b4c9b84-... --list_actions
"""
import ast
import json
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from os import listdir, makedirs, path
from zlib import crc32

import numpy as np

HERE = path.dirname(path.abspath(__file__))
CINEMATRAJ_ROOT = path.dirname(path.dirname(HERE))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

TRUMANS_DEFAULT = "/data1/cympyc1785/data/trumans/Data_release"
BLENDER_DEFAULT = "/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender"
VISTA4D_DEFAULT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
EVAL_DATA_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data"

NUM_FRAMES = 49          # Lite 코퍼스 고정 길이
HUMAN_INDEX = 1          # trumans_gt_render.py 가 사람 mesh union 에 예약한 pass index

# 소스 카메라 궤적 어휘. 실제 pkl 2편이 "높이 일정 + 사람 추종 + 느린 호"라 그 변형들이다.
# 값은 49프레임 **전체** 이동량(m) 과 방위각 스윕(도). net |dt| 실측 median 0.58~0.62 에 맞췄다.
SOURCE_PRESETS = {
    "arc_left":    {"sweep_deg": -18.0, "dradius": 0.00, "dheight": 0.00},
    "arc_right":   {"sweep_deg":  18.0, "dradius": 0.00, "dheight": 0.00},
    "push_in":     {"sweep_deg":   0.0, "dradius": -0.55, "dheight": 0.00},
    "pull_out":    {"sweep_deg":   0.0, "dradius":  0.55, "dheight": 0.00},
    "arc_push":    {"sweep_deg": -12.0, "dradius": -0.35, "dheight": 0.00},
    "arc_pull":    {"sweep_deg":  12.0, "dradius":  0.35, "dheight": 0.00},
    "drift":       {"sweep_deg":  -6.0, "dradius": -0.15, "dheight": 0.18},
    "hold":        {"sweep_deg":   0.0, "dradius":  0.00, "dheight": 0.00},
}

# 카메라가 subject 를 뚫고 들어가는 것을 막는 반경 하한. `--preset_scale` 로 dradius 를 키우면
# push_in 계열의 종료 반경이 음수까지 갈 수 있는데, 아래 `synth_source_path` 의 clamp(0.6)는
# 그걸 **조용히** 잘라서 "요청한 것보다 짧은 dolly" 를 만든다. 그래서 후보-preset 짝짓기
# 단계에서 미리 거른다 (`preset_feasible`).
MIN_END_RADIUS = 0.80


# ---------------------------------------------------------------------------------------
# TRUMANS 메타 (trumans_to_lbm_demo.py 와 같은 파일들을 읽는다)
# ---------------------------------------------------------------------------------------
def read_actions(trumans: str, sequence: str):
    """`Actions/<seq>.txt` → [{'id','start','end','text'}]. 탭 3열 `start\\tend\\t설명`."""
    actions = []
    with open(path.join(trumans, "Actions", f"{sequence}.txt"), encoding="utf-8") as file:
        for line in file:
            parts = [p for p in line.rstrip("\n").split("\t") if p != ""]
            if len(parts) == 3:
                start, end, text = parts
            elif len(parts) >= 4:
                start, end, text = parts[1], parts[2], "\t".join(parts[3:])
            else:
                continue
            actions.append({"id": len(actions) + 1, "start": int(start),
                            "end": int(end), "text": text.strip(), "kind": "labeled"})
    return actions


# ---------------------------------------------------------------------------------------
# subject = human / event / object.  "event 단위" 라는 건 subject 가 사람이 아니라
# **사람 ∪ 그 action 이 건드리는 소품** 이라는 뜻이고, 그 소품만 남기면 object 단위가 된다.
#
# 소품 목록은 recording 폴더의 `obj_list.txt` 에 들어 있다 (61/66 recording 에 존재).
# 예: ['cup_01', 'oven_base_01', 'oven_door_01', 'book_right_01', 'pen_01', 'vase_03', ...]
# 한 소품이 `_base`/`_door` 처럼 여러 오브젝트로 쪼개져 있으므로 stem 으로 묶어서 다룬다.
# ---------------------------------------------------------------------------------------
# obj_list stem → action text 에서 쓰이는 명사. 실측으로 61편의 stem 을 **전부** 덮는다
# (`ALIASES 에 없는 stem` 0건). 여기 없는 stem 이 새로 나오면 assert 로 드러나게 해 둔다.
PROP_ALIASES = {
    "movable_chair": ("stool", "chair"), "static_chair": ("stool", "chair"),
    "door": ("door",), "cabinet": ("cabinet", "cupboard"), "drawer": ("drawer",),
    "microwave": ("microwave",), "oven": ("oven",), "laptop": ("laptop",),
    "monitor": ("monitor", "screen"), "keyboard": ("keyboard",), "mouse": ("mouse",),
    "phone": ("phone",), "book": ("book",), "bottle": ("bottle",), "cup": ("cup", "mug"),
    "pen": ("pen",), "vase": ("vase",), "handbag": ("handbag", "bag"),
    "whiteboard": ("whiteboard", "board"),
}
# 목적어가 문장에 없는 표현. TRUMANS action text 의 미스 상위가 전부 여기였다
# ('drink water' 271줄 / 'write' 124 / 'make a call' 24 / 'type' 14).
PROP_VERBS = {
    "drink": ("bottle", "cup"), "write": ("pen",),
    "make a call": ("phone",), "call": ("phone",), "type": ("keyboard",),
}
# 소품 stem 을 caption 의 `target:` 에 쓸 사람 말로. 없으면 stem 을 그대로 쓴다.
PROP_LABELS = {"movable_chair": "stool", "static_chair": "stool", "whiteboard": "whiteboard",
               "handbag": "bag", "monitor": "monitor"}


def read_obj_list(rec_dir: str):
    """recording 의 `obj_list.txt` → blend 오브젝트 이름 리스트. 없으면 빈 리스트.

    파일 내용은 파이썬 리스트 리터럴 **한 줄** 이다 (`['cup_01', 'oven_base_01', ...]`).
    66편 중 61편에만 있다 — 없으면 event/object 를 못 만드므로 human 으로 떨어진다.
    """
    listing = path.join(rec_dir, "obj_list.txt")
    if not path.isfile(listing):
        return []
    with open(listing, encoding="utf-8") as file:
        text = file.read().strip()
    return list(ast.literal_eval(text)) if text else []


def group_props(names):
    """['oven_base_01','oven_door_01','book_right_01'] → {'oven': [...], 'book': [...]}.

    한 소품이 부품별 오브젝트로 쪼개져 있어서 stem 으로 묶어야 union OBB 가 소품 하나가 된다.
    부품만 넘기면(`oven_door_01`) 문이 열릴 때 subject 가 문짝만 따라가 프레이밍이 튄다.
    """
    groups = {}
    for name in names:
        stem = re.sub(r"_(base|door|seat|screen|drawer|left|right)?_?\d*$", "", name)
        stem = re.sub(r"_\d+$", "", stem)
        groups.setdefault(stem, []).append(name)
    return groups


def match_prop(text: str, groups):
    """action text → (stem, [오브젝트 이름]) 또는 None.

    후보를 **그 recording 이 실제로 가진 소품** 으로 먼저 좁힌다. 이게 모호성의 주된 해소책이다
    — 'drink water' 는 어휘상 bottle/cup 둘 다지만 대부분의 recording 은 둘 중 하나만 갖는다.
    그래도 둘 이상 남으면 사전순 첫 stem 을 쓴다 (재현성). 실측 커버리지(9,488 라인):
    단일 매칭 69.2 % / 모호 6.5 % / 무매칭 24.3 %. 무매칭의 대부분(20.9 %)은
    stand up · sit down · squat · lie down 처럼 **소품이 없는 게 맞는** 동작이다.
    """
    lowered = text.lower()
    hits = {stem for stem in groups
            if any(re.search(rf"\b{word}\b", lowered)
                   for word in PROP_ALIASES.get(stem, (stem,)))}
    if not hits:
        for verb, stems in PROP_VERBS.items():
            if re.search(rf"\b{verb}\b", lowered):
                hits |= {s for s in stems if s in groups}
    if not hits:
        return None
    stem = sorted(hits)[0]
    return stem, sorted(groups[stem])


def resolve_subject(kind: str, manual_props, action, rec_dir: str):
    """(`--subject_kind`, `--prop_names`, action) → probe 에 넘길 (kind, 이름들, caption 필드).

    `auto` 만 텍스트 매칭을 돈다. 매칭이 없으면 **human 으로 떨어진다** — 이게 옳은 이유는
    미매칭의 대부분(전체 라인의 20.9 %)이 stand up · sit down · squat · lie down 처럼
    애초에 소품이 없는 동작이기 때문이다. 억지로 소품을 붙이면 카메라가 엉뚱한 걸 겨눈다.

    `--prop_names` 를 직접 주면 매칭을 건너뛴다 (한 recording 을 손으로 몰 때).
    `human` 은 기본값이고 기존 뱅크 96편과 **비트 동일** 경로다 — 매칭조차 돌지 않는다.

    caption 의 `target:` / `event:` 도 여기서 함께 정한다. 한 곳에서 정해야 manifest 의
    subject 와 caption 이 어긋나지 않는다 (probe 에 넘긴 소품과 caption 이 다르면
    "컵을 마시는" 캡션에 카메라는 사람 전신을 잡고 있는 클립이 조용히 생긴다).
    """
    text = action["text"]
    if manual_props:
        stem = None
        names = list(manual_props)
    elif kind == "human":
        stem, names = None, []
    else:
        hit = match_prop(text, group_props(read_obj_list(rec_dir)))
        if hit is None:
            # auto 는 human 으로 떨어지고, event/object 를 **명시**했는데 못 찾으면 에러다
            # (조용히 human 으로 떨어지면 "object 뱅크"에 사람 클립이 섞인다).
            assert kind == "auto", (
                f"--subject_kind {kind} 인데 action '{text}' 에서 소품을 못 찾았다. "
                f"obj_list.txt: {sorted(group_props(read_obj_list(rec_dir)))}")
            stem, names = None, []
        else:
            stem, names = hit
    resolved = "human" if not names else ("event" if kind in ("auto", "human") else kind)
    label = PROP_LABELS.get(stem, stem) if stem else None
    #    caption `target:` 은 카메라가 **무엇을 프레이밍하는지**다. event 는 사람 ∪ 소품이지만
    #    프레이밍의 주인공은 사람이므로 `man` 으로 두고, 소품은 `event:` 문장이 담는다.
    target = "man" if resolved != "object" else (label or "object")
    return resolved, names, {"target": target, "event": text.lower().rstrip("."),
                             "prop": label, "prop_stem": stem}


# TRUMANS SMPL-X 배열(`human_transl`/`human_joints`)은 **y-up** 이다. blend 씬은 z-up 이지만
# 여기서는 프레임 인덱스만 뽑으므로 변환이 필요 없다 — 단 수평면은 반드시 (x, z) 다.
# `[:, :2]` 로 재면 가장 넓은 축(z, std 1.101)을 버리고 수직 bob(y, std 0.149)을 섞어서
# 보행 비율이 19.0% → 7.5% 로 절반 이하로 과소보고된다 (fps 25 실측). 실제로 한 번 그렇게 틀렸다.
SMPL_UP = 1
SMPL_HORIZONTAL = [0, 2]


def sequence_span(trumans: str, sequence: str):
    """`seg_name.npy` 에서 그 시퀀스의 [start, end) 전역 프레임 범위."""
    seg = np.load(path.join(trumans, "seg_name.npy"))
    hit = np.flatnonzero(seg == sequence)
    assert hit.size, f"시퀀스 {sequence} 가 seg_name.npy 에 없다"
    return int(hit[0]), int(hit[-1]) + 1


def mine_walk_actions(trumans: str, sequence: str, speed: float, frac: float,
                      smooth: int, fps: float, limit: int, span: int = NUM_FRAMES):
    """라벨에 **없는** 보행 구간을 `human_transl` 에서 캐 pseudo-action 으로 만든다.

    왜 필요한가: `Actions/*.txt` 9,488 라인의 어휘에 보행이 **0건**이다 (stand up / sit down /
    pick up ... 뿐이고 `turn` 히트 54개는 전부 microwave·oven on-off). 그런데 모션에는
    수평속도 > 0.4 m/s 인 프레임이 **19.0%** (fps 25) 나 있고, 49프레임 창의 80% 이상이 보행인 구간이
    blend 66편 씬에 16,515개 있으며 그 **81% 가 라벨 구간과 전혀 안 겹친다** — action 사이
    이동 구간이기 때문이다. 즉 라벨로 클립을 고르는 한 보행은 후보에 들어올 수가 없고,
    그래서 뱅크가 전부 제자리 조작이라 subject 가 사실상 정지 앵커였다. 움직이는 subject 를
    추종하는 shot 이 있어야 anchor conditioning 과 `--track_gain` 이 일을 한다.

    반환값은 `read_actions()` 와 같은 dict 이고 **목록 뒤에 append 하는 용도**다. 앞에 끼우면
    `--action <i>` 인덱스가 밀려서 시드 키(`recording|action|seed`)가 바뀌고 기존 96편이
    재현이 안 된다.
    """
    lo, hi = sequence_span(trumans, sequence)
    transl = np.load(path.join(trumans, "human_transl.npy"), mmap_mode="r")
    track = np.asarray(transl[lo:hi], dtype=np.float64)[:, SMPL_HORIZONTAL]
    #    `span` 은 클립이 실제로 덮는 **모션 프레임 수** = (NUM_FRAMES-1)*frame_step + 1 이다.
    #    NUM_FRAMES 로 고정하면 step 3 에서 49프레임짜리 보행 창을 캐 놓고 `action_window` 가
    #    그걸 145프레임으로 늘리므로, "창의 80% 이상이 보행"이라는 보장이 통째로 깨진다.
    if len(track) < span + 2:
        return []
    step = np.linalg.norm(np.diff(track, axis=0), axis=1) * fps
    step = np.convolve(step, np.ones(smooth) / smooth, mode="same")
    walking = (step > speed).astype(np.float64)
    # 창을 span 간격으로만 뽑는다 — 겹치는 창은 렌더가 거의 같은 그림이라 낭비다.
    hits = []
    index = 0
    while index + span < len(walking):
        if walking[index:index + span].mean() >= frac:
            net = float(np.linalg.norm(track[index + span] - track[index]))
            hits.append({"start": index, "end": index + span - 1, "net_m": net,
                         "walk_frac": float(walking[index:index + span].mean())})
            index += span
        else:
            index += 5
    hits.sort(key=lambda h: -h["net_m"])            # 많이 걷는 창부터
    return [{"id": None, "start": h["start"], "end": h["end"], "text": "Walk",
             "kind": "walk", "walk_net_m": round(h["net_m"], 3),
             "walk_frac": round(h["walk_frac"], 3)} for h in hits[:limit]]


def sequences_for_scene(trumans: str, uuid: str):
    """그 씬(uuid, `_1`/`_2` 변형 포함)에 속한 시퀀스별 프레임 수. augment 사본은 뺀다."""
    scenes = np.load(path.join(trumans, "scene_list.npy"))
    seg = np.load(path.join(trumans, "seg_name.npy"))
    flag = np.load(path.join(trumans, "scene_flag.npy"))
    rows = [i for i, name in enumerate(scenes) if str(name).startswith(uuid)]
    names, counts = np.unique(seg[np.isin(flag, rows)], return_counts=True)
    return {str(n): int(c) for n, c in zip(names, counts) if "_augment" not in str(n)}


def blend_frame_count(blender: str, blend: str, tmp: str):
    """headless Blender 로 frame_start/end/fps 를 읽는다 (1.6 GB blend 도 실측 3초)."""
    # 이 파일은 **여기서 만들어 쓰는 스크래치**다 (읽어 오는 게 아니다). 예전엔 `HERE` 에
    # 썼는데, 그러면 소스 트리에 `_probe_frames.py` 가 남는다 — R6 이전에 `scripts/` 로
    # 커밋돼 버린 것이 그 흔적이다. `tmp` 옆(=호출자가 정한 작업 폴더)에 쓴다.
    script = path.join(path.dirname(path.abspath(tmp)), "_probe_frames.py")
    with open(script, "w", encoding="utf-8") as file:
        file.write("import bpy, json, sys\n"
                   "s = bpy.context.scene\n"
                   "json.dump({'start': s.frame_start, 'end': s.frame_end,\n"
                   "           'fps': s.render.fps / max(1, s.render.fps_base)},\n"
                   "          open(sys.argv[sys.argv.index('--') + 1], 'w'))\n")
    subprocess.run([blender, "-b", blend, "--python", script, "--", tmp],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open(tmp, encoding="utf-8") as file:
        return json.load(file)


def pick_sequence(trumans: str, uuid: str, n_frames: int, forced: str):
    """blend 프레임 수와 같은 시퀀스를 고른다. 동점이면 사용자에게 넘긴다."""
    table = sequences_for_scene(trumans, uuid)
    if forced:
        assert forced in table, f"{forced} 는 씬 {uuid} 의 시퀀스가 아니다. 후보: {sorted(table)}"
        return forced, table
    hits = [name for name, count in table.items() if count == n_frames]
    assert hits, (f"blend 프레임 수 {n_frames} 와 일치하는 시퀀스가 없다.\n"
                  f"  후보: {sorted(table.items(), key=lambda kv: kv[1])}")
    assert len(hits) == 1, f"프레임 수 {n_frames} 인 시퀀스가 여럿이다: {hits}. --sequence 로 지정할 것."
    return hits[0], table


def action_window(action, frame_lo, frame_hi, step=1):
    """action 구간 → NUM_FRAMES 장을 뽑을 창. 짧으면 중앙에서 늘리고, 길면 중앙을 자른다.

    왜 중앙인가: TRUMANS action 라벨의 앞뒤 몇 프레임은 전/후 동작과의 전이라 그 구간을 쓰면
    "무엇을 하는 shot 인가"가 흐려진다.

    `step` 은 **모션 프레임 간격**이다. TRUMANS 모션은 30 Hz 이므로 step 3 이면 10 fps 로
    샘플한 셈이고, 49장이 덮는 실시간이 1.6초 -> 4.8초가 된다. 창의 폭은 프레임 수가 아니라
    `(NUM_FRAMES-1)*step + 1` 이다.

    반환 `(start, end)` 의 end 는 **포함**이고 `start + (NUM_FRAMES-1)*step` 과 정확히 같다 —
    Blender 쪽 `range(start, end+1, step)` 이 딱 NUM_FRAMES 장을 내도록.
    """
    span = (NUM_FRAMES - 1) * step + 1
    assert frame_hi - frame_lo + 1 >= span, \
        f"시퀀스 길이 {frame_hi - frame_lo + 1} 가 창 {span} (={NUM_FRAMES}장 x step {step}) 보다 짧다"
    center = (int(action["start"]) + int(action["end"])) // 2
    start = center - span // 2
    start = max(frame_lo, min(start, frame_hi - span + 1))
    return start, start + span - 1


def scaled_preset(preset, scale):
    """`SOURCE_PRESETS[preset]` 을 `--preset_scale` 배 한 것.

    왜 필요한가: preset 값은 "49프레임 **전체** 이동량" 이라 실시간 길이에 묶여 있다. step 3 으로
    가면 같은 49장이 1.6초가 아니라 4.8초를 덮으므로, 값을 그대로 두면 각속도/전진속도가 1/3 로
    떨어진다. scale=step 으로 주면 초당 체감 속도가 보존된다.
    """
    spec = SOURCE_PRESETS[preset]
    return {k: v * float(scale) for k, v in spec.items()}


def preset_feasible(candidate, preset, scale, min_end_radius=MIN_END_RADIUS):
    """이 시작 반경에서 이 preset 을 **요청한 크기대로** 돌릴 수 있나.

    `synth_source_path` 의 `max(0.6, ...)` clamp 는 불가능한 dolly 를 조용히 짧은 dolly 로
    바꿔 버린다 — 로그에는 `push_in` 이라고 찍히는데 실제 이동량은 절반인 클립이 생긴다.
    여기서 미리 거르면 그 짝은 아예 후보에 안 들어가고, 어떤 preset 이 왜 빠졌는지가 남는다.
    """
    spec = scaled_preset(preset, scale)
    return float(candidate["radius"]) + spec["dradius"] >= min_end_radius


# ---------------------------------------------------------------------------------------
# 소스 카메라 합성
# ---------------------------------------------------------------------------------------
def look_at_c2w(position, target, up=(0.0, 0.0, 1.0)):
    """OpenCV c2w. 열 순서 [right | down | forward]. roll 은 world +Z(중력축)에 대해 0.

    TRUMANS world 는 Blender 기본이라 +Z 가 위다 — DA3 world 처럼 기울어진 카메라 프레임이
    아니므로 up 을 그대로 쓸 수 있다 (`caption`/`gravity` 논의가 여기선 필요 없다).
    """
    forward = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = np.linalg.norm(forward)
    assert norm > 1e-6, "카메라와 look-at 이 같은 점이다."
    forward /= norm
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:      # 수직으로 내려다보는 특이점
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    matrix = np.eye(4)
    matrix[:3, 0], matrix[:3, 1], matrix[:3, 2] = right, down, forward
    matrix[:3, 3] = position
    assert abs(np.linalg.det(matrix[:3, :3]) - 1.0) < 1e-9
    return matrix


def rotation_log(rotation):
    """SO(3) → (축, 각). `decode/build_poses.py:150` 과 같은 식이다."""
    trace = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(trace))
    if angle < 1e-9:
        return np.array([1.0, 0.0, 0.0]), 0.0
    axis = np.array([rotation[2, 1] - rotation[1, 2],
                     rotation[0, 2] - rotation[2, 0],
                     rotation[1, 0] - rotation[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:      # angle ≈ π. 이 뱅크의 keyframe 간격에서는 안 나온다
        return np.array([1.0, 0.0, 0.0]), 0.0
    return axis / norm, angle


def slerp_rotation(rotation_a, rotation_b, alpha):
    """SO(3) 측지선 보간 (`decode/build_poses.py:176` 의 복제).

    `build_poses` 를 import 하면 `lbm.render` → Vista4D 점군 렌더러 → torch 가 딸려 온다.
    이 스크립트는 Blender orchestration 용이라 torch 를 안 들이려고 순수 numpy 4함수만 옮겼다.
    식이 갈라지면 Vista4D 뱅크와 TRUMANS 뱅크의 조준이 어긋나므로 고칠 땐 양쪽 같이 고칠 것.
    """
    axis, angle = rotation_log(np.asarray(rotation_b) @ np.asarray(rotation_a).T)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    theta = angle * float(alpha)
    exp = np.eye(3) + np.sin(theta) * cross + (1.0 - np.cos(theta)) * (cross @ cross)
    return exp @ np.asarray(rotation_a)


def keyframe_indices(num_frames, count):
    """`linspace` keyframe 프레임 번호. 항상 0 과 F-1 포함, 중복 제거 (`build_poses.py:186`).

    `count == 1` 은 여기서만 추가로 허용한다 (`build_poses` 는 2 이상만 받는다). keyframe 이
    하나면 보간할 구간이 없으므로 **frame 0 의 조준을 49프레임 내내 유지**한다 = 카메라가
    subject 를 전혀 추종하지 않고 preset 이동만 하는 shot. 매 프레임 look-at(기존 동작)의
    반대 극단이고, 그 둘 사이가 k=6 이다.
    """
    assert count >= 1, f"aim_keyframes 는 1 이상이어야 한다 (받은 값 {count})"
    if count == 1:
        return [0]
    return sorted(set(np.rint(np.linspace(0, num_frames - 1,
                                          min(count, num_frames))).astype(int).tolist()))


def smooth_track(centers, window):
    """사람 중심 궤적을 이동평균으로 편다. look-at 을 통해 곧장 카메라 회전으로 들어가서
    1% jitter 가 프레임당 1° yaw 로 보인다."""
    if window <= 1:
        return centers.copy()
    padded = np.pad(centers, ((window // 2, window // 2), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, i], kernel, mode="valid")[:len(centers)] for i in range(3)], axis=1)


def renders_match(render_dir, poses, num_frames, rgb_engine="eevee"):
    """이미 있는 렌더가 **바로 이 pose · 이 엔진으로** 찍힌 것인지. 아니면 다시 찍어야 한다.

    엔진까지 보는 이유: pose 만 대조하면 EEVEE 로 찍어둔 렌더를 `--rgb_engine cycles` 로
    다시 돌렸을 때 **조용히 재사용**한다. 번짐이 그대로 남는데 로그에는 "렌더 재사용" 한 줄만
    찍혀서 눈치챌 방법이 없다.
    """
    want_engine = "CYCLES" if rgb_engine == "cycles" else "BLENDER_EEVEE_NEXT"
    meta_path = path.join(render_dir, "render_meta.json")
    if path.exists(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as file:
                if json.load(file).get("rgb_engine") != want_engine:
                    return False
        except ValueError:
            return False
    elif rgb_engine != "eevee":
        # meta 가 없는 옛 렌더는 EEVEE 로 찍힌 것이다. cycles 를 요청했으면 재사용 금지.
        return False
    cameras_path = path.join(render_dir, "cameras.json")
    if not path.exists(cameras_path):
        return False
    try:
        with open(cameras_path, encoding="utf-8") as file:
            cameras = json.load(file)
        entries = cameras["cameras"]
        if len(entries) != num_frames:
            return False
        rendered = np.asarray([e["c2w_opencv"] for e in entries], dtype=np.float64)
        if not np.allclose(rendered, poses, atol=1e-4):
            return False
    except (KeyError, ValueError):
        return False
    for name in ("rgb", "depth", "index"):
        folder = path.join(render_dir, name)
        if not path.isdir(folder) or len(listdir(folder)) < num_frames:
            return False
    return True


def synth_source_path(probe, candidate, preset, aim_bias, track_gain,
                      preset_scale=1.0, smooth_window=11, aim_keyframes=0, keep_start_z=False):
    """검증된 시작 pose + preset → (49,4,4) OpenCV c2w (TRUMANS world).

    높이는 **일정**하게 유지한다 (실제 pkl 2편의 z-span 이 정확히 0 이었다). `dheight` 가 있는
    preset 만 예외로 램프를 준다.

    `smooth_window` 는 **샘플 프레임** 단위다. step>1 이면 같은 11이 실시간으로는 step 배 넓은
    평활이 되므로 (30 Hz 기준 0.37초 -> step 3 에서 1.1초) 호출부가 줄여서 넘겨야 한다.

    `aim_keyframes` (기본 0 = 기존 동작): 0 이면 **매 프레임** `look_at_c2w` 로 조준한다.
    N>=2 면 Vista4D 뱅크(`decode/build_poses.py` D71, `--aim_keyframes 6`)와 **같은 방식**으로
    `keyframe_indices(49, N)` = [0,10,19,29,38,48] 여섯 지점에서만 조준을 세우고 사이는
    smoothstep slerp 로 잇는다. 위치는 안 건드린다 — preset 모양이 이미 검증(clearance)을
    통과한 것이라 그대로 두어야 verify 결과가 유효하다. 두 코퍼스를 섞어 학습할 때 조준 방식이
    다르면 "카메라가 subject 를 얼마나 빡빡하게 따라보나"가 코퍼스 라벨이 되어 버린다.
    """
    track = probe["human_track"]
    # look-at 기준점은 **probe 가 격자 원점으로 쓴 것과 같아야** 한다. 안 그러면 `--aim_bias` 가
    # 다른 원점 위에 얹혀서 조용히 엉뚱한 높이를 겨눈다 — 실측 0.333 m (0.70·h 를 겨눈다고
    # 적어 놓고 실제로는 0.89·h, 즉 머리 꼭대기를 봤다). probe 가 무엇을 썼는지는 JSON 이 말한다.
    #
    # `obb_center` (기본): subject union AABB 중심 = 0.50·h. subject-agnostic 이라 event/object
    #   로 그대로 일반화되고, `anchor_cond` 가 싣는 값과 원점이 같다. 흉부 높이는 `--aim_bias`
    #   +0.20 으로 복원한다.
    # `chest`: 몸통 mesh 정점의 높이띠별 median xy = 0.70·h. **human 전용**이고 기존 뱅크 96편
    #   재현용이다. AABB 중심도 리그 루트 본도 아닌 이유는 probe 의 `body_points` docstring 참고
    #   (팔을 뻗으면 AABB 중심이 빈 공간, 루트 본은 발밑 1 m 아래).
    #
    # 예전 JSON 에는 `anchor_origin` 키가 없다 → 그때는 chest 였으므로 그렇게 떨어진다.
    origin = probe.get("anchor_origin", "chest")
    if origin == "chest":
        base = np.asarray([t["body_points"][0] for t in track], dtype=np.float64)
    else:
        base = np.asarray([t["center"] for t in track], dtype=np.float64)
    smooth = smooth_track(base, int(smooth_window))

    floor_z = float(probe["floor_z"])
    height = float(probe["human_height"])
    # `aim_bias` 로 원점 기준 위/아래로 민다 (사람 키 비율). obb_center(0.50·h) 에 +0.20 이면
    # 예전 흉부(0.70·h)와 같은 높이가 된다. 0 이면 원점 그대로.
    aim = smooth + np.array([0.0, 0.0, aim_bias * height])
    # 카메라는 사람을 부분적으로만 따라간다. gain 1 이면 사람이 걸을 때 시차가 0 이 된다.
    pivot = aim[0] + track_gain * (aim - aim[0])

    start = np.asarray(candidate["position"], dtype=np.float64)
    offset = start - aim[0]
    radius = float(np.linalg.norm(offset[:2]))
    phi0 = float(np.arctan2(offset[1], offset[0]))
    z0 = float(start[2])

    spec = scaled_preset(preset, preset_scale)
    # ease-in-out: 등속이면 시작/끝에서 카메라가 톡 끊긴다.
    s = np.linspace(0.0, 1.0, NUM_FRAMES)
    ease = 0.5 - 0.5 * np.cos(np.pi * s)

    phi = phi0 + np.radians(spec["sweep_deg"]) * ease
    rad = np.maximum(0.6, radius + spec["dradius"] * ease)
    zed = z0 + spec["dheight"] * ease

    #    바닥 클램프 높이. `keep_start_z` (D272 board 모드) 면 시작 높이보다 위로는 안 올린다 —
    #    board 는 `--min_height 0.30` 으로 0.25·h 보다 낮은 low-angle 후보를 통과시키는데, 그걸
    #    0.25·h 로 끌어올리면 frame0 이 board 가 검증한 pose 가 아니게 된다 (실측 +6.4 cm).
    z_floor = floor_z + 0.25 * height
    if keep_start_z:
        z_floor = min(z_floor, z0)
    positions = np.zeros((NUM_FRAMES, 3))
    for i in range(NUM_FRAMES):
        position = np.array([pivot[i, 0] + rad[i] * np.cos(phi[i]),
                             pivot[i, 1] + rad[i] * np.sin(phi[i]),
                             zed[i] + (pivot[i, 2] - pivot[0, 2])])
        position[2] = max(position[2], z_floor)   # 바닥 밑으로 안 내려간다
        positions[i] = position

    poses = np.zeros((NUM_FRAMES, 4, 4))
    if int(aim_keyframes):
        # keyframe 조준 (Vista4D 뱅크와 동일). keyframe 에서만 look-at 을 세우고 사이는
        # smoothstep slerp — keyframe 에서 각속도가 0 이라 이음매가 안 튄다.
        frames = keyframe_indices(NUM_FRAMES, int(aim_keyframes))
        rotations = [look_at_c2w(positions[f], aim[f])[:3, :3] for f in frames]
        if len(frames) == 1:      # k=1: frame 0 조준을 그대로 유지 (subject 추종 없음)
            poses[:, :3, :3] = rotations[0]
        for index in range(len(frames) - 1):
            a, b = frames[index], frames[index + 1]
            for f in range(a, b + 1):
                x = (f - a) / (b - a)
                x = x * x * (3.0 - 2.0 * x)
                poses[f][:3, :3] = slerp_rotation(rotations[index], rotations[index + 1], x)
        poses[:, :3, 3] = positions
        poses[:, 3, 3] = 1.0
    else:
        for i in range(NUM_FRAMES):
            poses[i] = look_at_c2w(positions[i], aim[i])
    return poses, aim


def path_stats(poses):
    """실제 pkl 2편과 같은 눈금으로 잰다 — 프레임당 |dt| median / 49프레임 net / 총 경로."""
    positions = poses[:, :3, 3]
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {"dt_median": float(np.median(steps)), "dt_max": float(steps.max()),
            "net": float(np.linalg.norm(positions[-1] - positions[0])),
            "path": float(steps.sum())}


# ---------------------------------------------------------------------------------------
# Blender 호출
# ---------------------------------------------------------------------------------------
#    Blender 가 **시그널로** 죽는 경우(rc < 0)만 재시도한다. stride3 뱅크 2 arm 에서 rc=-11
#    (SIGSEGV) 이 9회 났는데, 전부 job 시작 20~28 초 지점 — probe 두 단계(약 19 s)를 지나
#    gt_render 가 EEVEE GPU context 를 잡는 순간이고, 그때 **다른 worker 가 렌더 한복판**
#    (이웃 job 180~350 s) 이었다. 같은 action 을 단독으로 다시 돌리면 171 s 만에 정상 렌더된다
#    (2026-08-24 tru_1d43e076_a13). 즉 데이터가 아니라 GPU context 경합이라, 잠깐 쉬었다
#    다시 띄우면 통과한다. python traceback / rc>0 은 결정적 실패라 재시도하지 않는다.
BLENDER_RETRY_SLEEP = (20.0, 60.0)     # 재시도 전 대기 — 이웃 worker 의 렌더가 끝날 시간을 준다


_PROBE_CLIENT = {}


def run_probe(args, blend, script, script_args, log_tag, retries: int = 0):
    """probe 1회. `--probe_server` 면 recording 상주 서버에서 in-process 로 (D278), 아니면
    예전처럼 Blender 를 새로 띄운다 (`run_blender`, 비트 동일 경로)."""
    if not args.probe_server:
        return run_blender(args.blender, blend, script, script_args, log_tag, retries=retries)
    started = time.time()
    if args.recording not in _PROBE_CLIENT:
        from lbm.blender_raycast import RaycastClient
        _PROBE_CLIENT[args.recording] = RaycastClient(args.recording)
    _PROBE_CLIENT[args.recording].probe(script_args)
    return time.time() - started


def run_blender(blender, blend, script, script_args, log_tag, retries: int = 0):
    started = time.time()
    for attempt in range(retries + 1):
        process = subprocess.run([blender, "-b", blend, "--python", script, "--", *script_args],
                                 capture_output=True, text=True)
        #    Blender 는 `--python` 스크립트가 예외로 죽어도 **rc 0 을 돌려준다**. rc 만 보면 성공으로
        #    통과하고, 산출물이 없어서 한참 뒤 엉뚱한 자리에서 FileNotFoundError 로 터진다
        #    (2026-08-24 `trumans_scene_probe.py` step 색인 버그가 이 경로로 숨었다). 그래서
        #    출력에 Traceback 이 있으면 여기서 실패로 만든다.
        crashed = "Traceback (most recent call last)" in (process.stdout + process.stderr)
        if process.returncode == 0 and not crashed:
            return time.time() - started
        if process.returncode < 0 and not crashed and attempt < retries:
            nap = BLENDER_RETRY_SLEEP[min(attempt, len(BLENDER_RETRY_SLEEP) - 1)]
            print(f"  WARN {log_tag} 시그널 종료 (rc={process.returncode}) — "
                  f"{nap:.0f}s 쉬고 재시도 {attempt + 1}/{retries}")
            time.sleep(nap)
            continue
        print(process.stdout[-4000:])
        print(process.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"{log_tag} 실패 (rc={process.returncode}"
                           + (", python traceback" if crashed else "")
                           + (f", 재시도 {retries}회 소진" if retries else "") + ")")


# ---------------------------------------------------------------------------------------
# avg_scale — 이 클립을 **context 로 쓸 때** 카메라 이동량을 나눌 분모
# ---------------------------------------------------------------------------------------
AVG_SCALE_STRIDE = 2       # DL3DV `make_avg_scale_da3_*.py` 와 같은 pixel stride


def avg_scale_first_cam(depth, cam_c2w, K, sky_mask, dynamic_mask=None,
                        stride: int = AVG_SCALE_STRIDE):
    """**첫 카메라**에서 이 클립 depth 의 모든 점까지의 거리 **평균** (metre).

    왜 이 값인가: context 와 target 이 같은 씬을 본 **다른 카메라**라서, target 궤적을
    씬 크기로 정규화하려면 그 씬 크기를 **context 쪽만 보고** 정해야 한다. context 클립 하나로
    닫히는 양이어야 추론 때도 같은 값을 만들 수 있고, target 을 안 보므로 leakage 가 없다.
    (DL3DV 의 `avg_scale_context_first_cam` 과 같은 정의다 — 거기선 한 영상 안에서 구간을
    context/target 으로 갈랐고, 여기선 클립 자체가 갈려 있어 클립당 **하나**의 값이 된다.)

    왜 centroid 가 아니라 첫 카메라인가: 학습에서 cam_param 은 frame0 기준으로 재고정된 뒤
    이 분모로 나뉜다. 기준점을 frame0 카메라로 맞춰야 "재고정 기준"과 "스케일 기준"이 같아진다.

    depth 는 **z-planar metre** 다 (`trumans_gt_render.py` docstring). 그래서 `inv(K) @ [u,v,1]`
    에 z 를 곱하는 게 맞다 — ray distance 로 착각해 정규화하면 화면 가장자리가 조용히 짧아진다.

    반환 dict 의 `avg_scale` 은 **sky 를 뺀 전 픽셀**이다 (사용자 정의 "모든 점").
    `avg_scale_static` 은 사람까지 뺀 값 — 진단용이다. 사람은 카메라 가까이 크게 잡히므로
    둘이 크게 벌어지면 그 클립의 분모가 씬이 아니라 **사람 거리**를 재고 있다는 신호다.
    """
    num_frames, height, width = depth.shape
    us, vs = np.arange(0, width, stride), np.arange(0, height, stride)
    uu, vv = np.meshgrid(us, vs)
    #    픽셀 중심(+0.5)을 쓴다 — `S` 게이지 정의(`z·‖K⁻¹[u+0.5, v+0.5, 1]‖`)와 같은 규약.
    pix = np.stack([uu.reshape(-1) + 0.5, vv.reshape(-1) + 0.5,
                    np.ones(uu.size)], -1).astype(np.float64)
    origin = np.asarray(cam_c2w[0], dtype=np.float64)[:3, 3]
    rays = (np.linalg.inv(np.asarray(K, dtype=np.float64)) @ pix.T)      # (3, P)
    sums, counts = [0.0, 0.0], [0, 0]
    for i in range(num_frames):
        z = depth[i][vv, uu].reshape(-1).astype(np.float64)
        keep = np.isfinite(z) & (z > 0) & (~sky_mask[i][vv, uu].reshape(-1))
        if not keep.any():
            continue
        c2w = np.asarray(cam_c2w[i], dtype=np.float64)
        points = (c2w[:3, :3] @ (rays[:, keep] * z[keep][None, :])).T + c2w[:3, 3]
        dist = np.linalg.norm(points - origin[None, :], axis=1)
        sums[0] += float(dist.sum()); counts[0] += int(dist.size)
        if dynamic_mask is not None:
            still = ~dynamic_mask[i][vv, uu].reshape(-1)[keep]
            sums[1] += float(dist[still].sum()); counts[1] += int(still.sum())
    assert counts[0] > 0, "avg_scale: 유효한 depth 픽셀이 하나도 없다 (전부 sky?)"
    return {"format": "trumans_avg_scale_v1",
            "avg_scale": sums[0] / counts[0],
            "avg_scale_static": (sums[1] / counts[1]) if counts[1] else None,
            "origin": "first_cam", "points": "all_non_sky",
            "pixel_stride": int(stride), "n_points": counts[0],
            "depth_convention": "z-planar metre"}


# ---------------------------------------------------------------------------------------
# 변환: gt_render 출력 -> recon_and_seg + seg_instances
# ---------------------------------------------------------------------------------------
def convert(render_dir, poses_world, out_recon, out_seg, out_seg_static, fps, sky_depth,
            vista4d_root):
    """`trumans_gt_render.py` 출력 디렉토리를 Lite 입력 포맷으로 옮긴다."""
    if vista4d_root not in sys.path:
        sys.path.insert(0, vista4d_root)
    import imageio.v2 as imageio
    from utils.media import save_cameras, save_depths, save_masks
    from utils.recon_and_seg.seg_sam3_utils import save_seg_instances

    with open(path.join(render_dir, "render_meta.json"), encoding="utf-8") as file:
        meta = json.load(file)
    with open(path.join(render_dir, "cameras.json"), encoding="utf-8") as file:
        cameras = json.load(file)
    with open(path.join(render_dir, "objects.json"), encoding="utf-8") as file:
        objects = json.load(file)

    frames = sorted(int(path.basename(p)[6:11]) for p in
                    __import__("glob").glob(path.join(render_dir, "rgb", "frame_*.png")))
    assert len(frames) == NUM_FRAMES, f"렌더 프레임 {len(frames)} != {NUM_FRAMES}"

    rgb = np.stack([imageio.imread(path.join(render_dir, "rgb", f"frame_{f:05d}.png"))[:, :, :3]
                    for f in frames])
    depth = np.stack([np.load(path.join(render_dir, "depth", f"frame_{f:05d}.npy")) for f in frames])
    index = np.stack([np.load(path.join(render_dir, "index", f"frame_{f:05d}.npy")) for f in frames])
    num_frames, height, width = depth.shape

    # 배경 sentinel 1e10 은 float16 에서 inf 다. clamp 하고 그 픽셀을 sky 로 표시한다.
    sky_mask = depth > float(sky_depth)
    depth = np.minimum(depth, float(sky_depth)).astype(np.float32)
    dynamic_mask = index == HUMAN_INDEX
    static_mask = (~dynamic_mask) & (~sky_mask)

    makedirs(out_recon, exist_ok=True)
    imageio.mimwrite(path.join(out_recon, "video.mp4"), list(rgb), fps=fps,
                     codec="libx264", quality=8, macro_block_size=1)
    save_depths(path.join(out_recon, "depths"), depth, dtype=np.float16)
    save_masks(path.join(out_recon, "dynamic_mask"), dynamic_mask)
    save_masks(path.join(out_recon, "static_mask"), static_mask)
    save_masks(path.join(out_recon, "sky_mask"), sky_mask)

    # 카메라: **frame0 앵커**. rigid 변환이라 metre 스케일은 그대로 남는다.
    world = np.asarray(poses_world, dtype=np.float64)
    anchored = np.linalg.inv(world[0])[None] @ world
    assert np.allclose(anchored[0], np.eye(4), atol=1e-9), "frame0 앵커 실패"
    # `cameras.json` 의 `frames` 는 프레임 **번호** 목록이고, K/c2w 는 `cameras` 항목에 있다.
    K = np.asarray(cameras["cameras"][0]["K"], dtype=np.float64)
    intrinsics = np.tile(np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]), (num_frames, 1))
    save_cameras(path.join(out_recon, "cameras.npz"), anchored, intrinsics)

    # seg_instances: 사람 = track 1. `save_seg_instances` 가 packbits/meta 를 다 해준다.
    def emit_tracks(folder, wanted):
        seg_frames = []
        for f in range(num_frames):
            instances = []
            for track_id, name in wanted:
                mask = index[f] == track_id
                if not mask.any():
                    continue
                ys, xs = np.nonzero(mask)
                instances.append({"id": int(track_id), "keyword": name, "score": 1.0,
                                  "box_xyxy": [float(xs.min()), float(ys.min()),
                                               float(xs.max() + 1), float(ys.max() + 1)],
                                  "mask": mask})
            seg_frames.append(instances)
        makedirs(folder, exist_ok=True)
        save_seg_instances(folder, seg_frames, [n for _, n in wanted], height, width)
        return sum(1 for _, n in wanted)

    # `objects.json` 의 `objects` 는 {pass_index(문자열): {name, type, ...}} 맵이다.
    by_index = {int(k): str(v["name"]) for k, v in objects["objects"].items()}
    present = sorted(int(v) for v in np.unique(index) if v > 0)
    human = [(HUMAN_INDEX, "person")]
    statics = [(i, by_index.get(i, f"object_{i}")) for i in present if i != HUMAN_INDEX]
    n_dyn = emit_tracks(out_seg, human) if HUMAN_INDEX in present else 0
    n_stat = emit_tracks(out_seg_static, statics) if statics else 0

    #    avg_scale 은 **anchored** 카메라로 잰다 (frame0 = 원점). rigid 라 metre 는 그대로다.
    #    recon 폴더 안에 같이 두는 이유: 이 값을 쓰는 쪽은 클립을 context 로 집어들 때이고,
    #    그때 손에 있는 건 work 디렉토리가 아니라 이 폴더다.
    scale = avg_scale_first_cam(depth, anchored, K, sky_mask, dynamic_mask)
    with open(path.join(out_recon, "avg_scale.json"), "w", encoding="utf-8") as file:
        json.dump(scale, file, ensure_ascii=False, indent=1)

    return {"num_frames": num_frames, "height": height, "width": width,
            "avg_scale": scale["avg_scale"], "avg_scale_static": scale["avg_scale_static"],
            "sky_frac": float(sky_mask.mean()), "human_frac": float(dynamic_mask.mean()),
            "depth_p50": float(np.median(depth[~sky_mask])) if (~sky_mask).any() else float("nan"),
            "n_dyn_tracks": n_dyn, "n_static_tracks": n_stat,
            "render_seconds": float(meta.get("elapsed_seconds", float("nan")))}


# ---------------------------------------------------------------------------------------
def main(args):
    rec_dir = path.join(args.trumans, "Recordings_blend", args.recording)
    blend = path.join(rec_dir, f"{args.recording}.blend")
    assert path.isfile(blend), f"blend 이 없다: {blend}"

    work = args.work or path.join(CINEMATRAJ_ROOT, "out", "trumans_recon", args.recording)
    makedirs(work, exist_ok=True)
    timing = {}

    started = time.time()
    probe_meta = blend_frame_count(args.blender, blend, path.join(work, "_frames.json"))
    n_frames = int(probe_meta["end"]) - int(probe_meta["start"]) + 1
    sequence, _ = pick_sequence(args.trumans, args.recording, n_frames, args.sequence)
    actions = read_actions(args.trumans, sequence)
    n_labeled = len(actions)
    if args.walk_actions:
        # **뒤에** append 한다 — 앞에 끼우면 `--action <i>` 가 밀려 시드 키가 바뀌고 기존
        # 96편이 재현이 안 된다. 그래서 인덱스 0..n_labeled-1 은 예전과 비트 단위로 같다.
        #    모션 레이트는 blend 의 render fps 가 아니다 (`--motion_fps` 주석 참고). 0 이면
        #    예전처럼 blend fps 로 떨어진다.
        motion_fps = args.motion_fps if args.motion_fps > 0 else float(probe_meta["fps"])
        actions += mine_walk_actions(args.trumans, sequence, args.walk_speed, args.walk_frac,
                                     args.walk_smooth, motion_fps, args.walk_max,
                                     span=(NUM_FRAMES - 1) * int(args.frame_step) + 1)
    timing["meta"] = time.time() - started

    if args.list_actions:
        # 소품 매칭 결과를 같이 찍는다 — `--subject_kind event` 로 돌릴 action 을 여기서 고른다.
        groups = group_props(read_obj_list(rec_dir))
        print(f"{args.recording}  seq {sequence}  frames {probe_meta['start']}..{probe_meta['end']}"
              f"  fps {probe_meta['fps']:g}   labeled {n_labeled}  walk {len(actions) - n_labeled}")
        print(f"obj_list  {sorted(groups) or '(없음)'}")
        print(f"\n{'idx':>4s} {'kind':>7s} {'start':>6s} {'end':>6s} {'len':>5s} {'prop':>10s}  text")
        print("-" * 90)
        for i, action in enumerate(actions):
            extra = (f"  (net {action['walk_net_m']:.2f} m, frac {action['walk_frac']:.2f})"
                     if action.get("kind") == "walk" else "")
            hit = match_prop(action["text"], groups) if groups else None
            print(f"{i:4d} {action.get('kind', 'labeled'):>7s} {action['start']:6d} "
                  f"{action['end']:6d} {action['end'] - action['start'] + 1:5d} "
                  f"{(hit[0] if hit else '-'):>10s}  {action['text'][:40]}{extra}")
        return

    board_cand, board_frames = None, []
    if args.board_json:
        #    `--board_json/--board_chunk/--board_candidate` (D272, 사용자 지시 2026-09-23 "후보군
        #    시작 카메라로 source video"). 창은 action 이 아니라 **board chunk 의 프레임**이고,
        #    시작 pose 는 자체 probe 격자에서 뽑지 않고 **board 후보 그대로**다. board 격자
        #    (elev -10/10/25/45, r 1.1~2.6) 와 probe 격자 (elev 0/12/25/40, r 1.0~5.5) 가 거의 안
        #    겹쳐서 `--anchor_cell` 로는 못 넘긴다 (tasks.md A1). action 은 창과 가장 많이 겹치는
        #    것을 대표로 잡는다 — subject 결정·manifest 기록용이고, 캡션은 아래 `window_actions`
        #    가 창에 걸친 action 을 전부 싣는다.
        with open(args.board_json, encoding="utf-8") as file:
            board = json.load(file)
        chunk = next((c for c in board["chunks"] if c["tag"] == args.board_chunk), None)
        assert chunk is not None, f"{args.board_json} 에 chunk {args.board_chunk} 가 없다"
        board_cand = dict(chunk["candidates"][int(args.board_candidate)])
        assert board_cand["usable"], (f"{args.board_chunk} 후보 {args.board_candidate} 는 board "
                                      f"게이트 탈락이다: {board_cand['reject']}")
        board_frames = [int(f) for f in chunk["frames"]]
        assert len(board_frames) == NUM_FRAMES, f"board chunk 프레임 {len(board_frames)} != {NUM_FRAMES}"
        step = int(board["frame_step"])
        assert step == int(args.frame_step), (f"board frame_step {step} != --frame_step "
                                              f"{args.frame_step} — 궤적 평활 폭이 어긋난다")
        start, end = board_frames[0], board_frames[-1]
        assert board_frames == list(range(start, end + 1, step)), "board 프레임이 균일 격자가 아니다"
        overlap = [max(0, min(end, int(a["end"])) - max(start, int(a["start"])) + 1) for a in actions]
        args.action = int(np.argmax(overlap)) if max(overlap) > 0 else 0
        action = actions[args.action]
    else:
        assert 0 <= args.action < len(actions), f"--action 은 0..{len(actions) - 1}"
        action = actions[args.action]
        step = int(args.frame_step)
        start, end = action_window(action, int(probe_meta["start"]), int(probe_meta["end"]), step)
    # 작업 파일 접미사. action 모드는 예전 이름(`_aNN`) 그대로, board 모드는 chunk+후보라
    # 같은 chunk 의 후보 여러 개가 병렬로 돌아도 서로의 probe/render 를 덮지 않는다.
    ftag = (f"a{args.action:02d}" if board_cand is None
            else f"{args.board_chunk}_k{int(args.board_candidate):03d}_p{int(args.board_slot)}")

    #    `--poses_override`: 카메라를 여기서 합성하지 않고 **밖에서 받아** 렌더만 한다.
    #    LBM 이 실제로 렌더한 카메라(`lbm_camera_to_poses.py` 산출물)를 Lite 뱅크와 **같은
    #    눈금**(`audit_lite_framing.py`)으로 재려면, 그 카메라로 GT depth/index 를 다시 그려야
    #    한다 — `occl_keep` 은 "평가 대상 카메라를 따라 렌더한 depth" 를 요구하기 때문이다.
    #    창은 npz 의 `trumans_frames` 가 정한다. LBM 은 창 길이와 무관한 `target_frame_count`
    #    만큼 렌더하므로(w01: 19프레임 창 → 110프레임) action 창과 다르고, 49로 솎은 뒤라
    #    간격도 균일하지 않다 (step 2..3). 그래서 아래 Blender 호출은 `--frame_list` 로 간다.
    frame_list, override_poses, override_aim, override_lens = [], None, None, 0.0
    if args.poses_override:
        loaded = np.load(args.poses_override)
        override_poses = np.asarray(loaded["cam_c2w"], dtype=np.float64)
        for key in ("lens_mm", "lens"):                # 어댑터는 `lens`, gt_render 는 `lens_mm`
            if key in loaded:
                override_lens = float(np.asarray(loaded[key]).reshape(-1)[0])
                break
        override_aim = (np.asarray(loaded["aim"], dtype=np.float64) if "aim" in loaded
                        else override_poses[:, :3, 3] + 2.0 * override_poses[:, :3, 2])
        assert "trumans_frames" in loaded, \
            f"{args.poses_override} 에 trumans_frames 가 없다 (lbm_camera_to_poses.py 산출물이 아니다)"
        frame_list = [int(f) for f in loaded["trumans_frames"]]
        assert len(frame_list) == len(override_poses), \
            f"trumans_frames {len(frame_list)} != cam_c2w {len(override_poses)}"
        assert len(frame_list) == NUM_FRAMES, \
            f"--poses_override 는 {NUM_FRAMES}프레임이어야 한다 (받은 것 {len(frame_list)})"
        #    프레임 격자를 **원본 창 안으로 클램프하지 않는다.** LBM 이 창 밖까지 렌더한 것
        #    자체가 결함(D6)이고, 잘라내면 그 결함이 평가에서 사라진다.
        start, end = frame_list[0], frame_list[-1]
        span = int(probe_meta["end"])
        assert end <= span, (f"LBM 이 blend 마지막 프레임({span})을 넘겼다: {start}..{end}. "
                             f"--frames 를 줄여 다시 덤프할 것.")
        #    보고용 실효 step (비균일이라 하나의 정수가 아니다). 재생 fps 눈금에만 쓴다.
        step = max(1, int(round((end - start) / max(1, len(frame_list) - 1))))
    # 평활 폭은 **샘플 프레임** 단위라 step 만큼 줄여야 실시간 폭이 보존된다 (30 Hz 기준 0.37초).
    # 0 = auto. 홀수로 맞춘다 — 짝수면 `smooth_track` 의 valid 컨볼루션이 반 프레임 밀린다.
    smooth_window = int(args.smooth_window) if args.smooth_window > 0 else max(3, round(11 / step))
    smooth_window += (smooth_window + 1) % 2
    # 라벨 창보다 긴 클립을 만들면 caption 이 거짓말이 된다 (49프레임 median 1개 -> 145프레임
    # median 2개). 창에 걸치는 action 을 전부 실어 packer 가 문장을 조립할 수 있게 한다.
    window_actions = [{"index": i, "kind": a.get("kind", "labeled"), "start": int(a["start"]),
                       "end": int(a["end"]), "text": a["text"]}
                      for i, a in enumerate(actions)
                      if int(a["start"]) <= end and int(a["end"]) >= start]
    video = args.out_video or (f"tru_{args.recording[:8]}_a{args.action:02d}" if board_cand is None
                               else f"tru_{args.recording[:8]}_{ftag}")
    print(f"[{video}] action {args.action} '{action['text'][:44]}' -> frames {start}..{end} "
          f"step {step} ({NUM_FRAMES}장, {(NUM_FRAMES - 1) * step / args.motion_fps:.2f}s @ "
          f"{args.motion_fps / step:g}fps)  창 내 action {len(window_actions)}개")

    #    subject 를 정한다. `human`(기본)이면 소품을 찾지도 않으므로 기존 96편과 비트 동일.
    subject_kind, prop_names, caption = resolve_subject(
        args.subject_kind, args.prop_names, action, rec_dir)
    subject_flags = ["--subject_kind", subject_kind]
    if prop_names:
        subject_flags += ["--prop_names", *prop_names]
    print(f"  subject {subject_kind}"
          + (f"  prop {caption['prop']} {prop_names}" if prop_names else ""))

    # 1) 격자 probe
    probe_path = path.join(work, f"probe_{ftag}.json")
    timing["probe_grid"] = run_probe(
        args, blend, path.join(HERE, "trumans_scene_probe.py"),
        ["--frames", str(start), str(end), str(step), "--out", probe_path,
         "--anchor_origin", args.anchor_origin,
         "--min_clearance", str(args.min_clearance), *subject_flags]
        #    override 면 격자를 LBM 프레임 집합에 맞춘다. `human_track[i]` 가 `cam_c2w[i]` 와
        #    짝이 맞아야 `audit_lite_framing.audit_clip` 이 같은 순간의 OBB 를 투영한다.
        + (["--frame_list", *[str(f) for f in frame_list]] if frame_list else []), "probe(grid)",
        retries=args.blender_retries)
    with open(probe_path, encoding="utf-8") as file:
        probe = json.load(file)

    usable = [c for c in probe["candidates"]
              if c["clear"] and c["clearance"] >= args.min_clearance
              and c["floor_drop"] >= args.min_floor_drop]
    #    override 면 이 격자를 **쓰지 않는다** (카메라가 밖에서 온다). 후보가 0개인 것은
    #    그 자체로 진단이지 실패가 아니다 — 세워둔 카메라를 렌더하는 데는 지장이 없다.
    if args.poses_override or board_cand is not None:
        print(f"{'grid (참고용)':22s} usable {len(usable)}/{len(probe['candidates'])}")
    else:
        assert usable, (f"설 수 있는 후보가 없다 (clear&clearance>={args.min_clearance}"
                        f"&floor_drop>={args.min_floor_drop}). probe: {probe_path}")

    #    `--probe_only`: 격자까지만 돌고 나간다. 공유 anchor 를 정하려면 recording 의 **모든**
    #    action 격자를 먼저 봐야 하는데(교집합), 그 pass 에서 궤적 합성/검증/렌더는 낭비다.
    if args.probe_only:
        print(f"{'probe_only':22s} usable {len(usable)} / {len(probe['candidates'])}  -> {probe_path}")
        return

    if args.poses_override:
        #    2)~3) 궤적 합성 + 검증 probe 를 **건너뛴다**. 카메라가 이미 있으니 합성할 것이
        #    없고, 검증은 "우리가 세운 후보가 벽을 뚫나"를 보는 필터라 평가 대상(LBM 카메라)에
        #    적용하면 베이스라인을 우리 기준으로 걸러버린다. 벽 통과 여부는 하류 감사가 잰다.
        poses, aim = override_poses, override_aim
        candidate, preset = {}, (args.preset or "lbm_render")
        clear_frac, min_clearance = float("nan"), float("nan")
        verify_path, tries, k = "", [], -1
        print(f"  poses_override {path.basename(args.poses_override)}  "
              f"{len(poses)}프레임  trumans {start}..{end} (실효 step {step})")
    else:
        #    `--anchor_cell AZ ELEV R`: 시작 pose 를 **격자 한 칸으로 고정**한다. recording 안의
        #    모든 클립에 같은 칸을 주면 프레임0 의 "사람 대비 화각"이 같아져서, text -> 궤적
        #    매핑에서 시작점 교란이 빠진다. 칸이 이 action 에서 못 쓰이면 실패시킨다 (조용히 다른
        #    칸으로 넘어가면 "공유" 가 깨진 채로 코퍼스에 섞인다).
        if args.anchor_cell:
            az, elev, radius = (float(v) for v in args.anchor_cell)
            forced = [c for c in usable
                      if abs(c["azimuth_deg"] - az) < 1e-6 and abs(c["elevation_deg"] - elev) < 1e-6
                      and abs(c["radius"] - radius) < 1e-6]
            assert forced, (f"--anchor_cell {az} {elev} {radius} 이 이 action 의 usable 격자에 없다 "
                            f"(usable {len(usable)}칸). probe: {probe_path}")
            usable = forced

        # 2) 시작 pose 샘플 + 궤적 합성. seed 로 재현 가능하게, 방위각이 몰리지 않도록 구역당 1개.
        # 파이썬 `hash()` 는 문자열에 프로세스마다 다른 salt 를 쓴다 (PYTHONHASHSEED). 그걸로 seed 를
        # 만들면 같은 --seed 로 돌려도 매번 다른 카메라가 나오고, 렌더 재사용도 늘 빗나간다.
        key = f"{args.recording}|{args.action}|{args.seed}".encode()
        if board_cand is not None:
            key = f"{args.recording}|{ftag}|{args.seed}".encode()
        rng = np.random.default_rng(crc32(key))
        #    같은 방위각 구역(45도)에서 여러 개가 뽑히면 "다양한 시작점"이 아니다. 구역을 섞은 뒤
        #    구역마다 하나씩 뽑아 **재시도 순서**를 만든다 — 1지망이 궤적 검증에서 떨어져도
        #    방위각이 다른 2지망으로 넘어간다 (같은 구역 재시도는 사실상 같은 카메라다).
        if board_cand is not None:
            #    board 모드: 시작 pose 는 고정이고 흔드는 축은 preset 하나다. 아래 루프가
            #    `order[i % len(order)]` 로 같은 후보를 돌면서 preset 만 바꾼다.
            order = [[board_cand]]
        elif args.radius_strata:
            #    반경도 같은 이유로 층화해야 한다. 격자 통과율이 반경에 따라 급락해서
            #    (7편 x 384 격자 실측: 1.5 m 50.3% / 2.2 m 23.2% / 3.0 m 8.4% / 4.0 m 2.3%)
            #    방위각만 층화하면 usable 풀이 1.5 m 로 60% 쏠리고 실제 채택은 **76% (72/95)**
            #    까지 간다 — 반경이 사실상 상수인 코퍼스가 된다. 반경을 **바깥 축**으로 두고
            #    clip 마다 순서를 섞어, 먼 반경이 통과율과 무관하게 1지망 자리를 갖게 한다.
            by_bucket = {}
            for candidate in usable:
                by_bucket.setdefault((float(candidate["radius"]),
                                      int(candidate["azimuth_deg"] // 45)), []).append(candidate)
            radii = sorted({radius for radius, _ in by_bucket})
            rng.shuffle(radii)
            quota = max(1, int(args.max_tries) // len(radii))
            picked = []
            for radius in radii:
                sectors = sorted(sector for r, sector in by_bucket if r == radius)
                rng.shuffle(sectors)
                picked.extend((radius, sector) for sector in sectors[:quota])
            #    할당량으로 max_tries 를 못 채우면 남은 조합으로 채운다 (먼 반경이 씨가 마른 씬).
            rest = [k for k in sorted(by_bucket) if k not in set(picked)]
            rng.shuffle(rest)
            order = [by_bucket[k] for k in picked + rest]
        else:
            by_sector = {}
            for candidate in usable:
                by_sector.setdefault(int(candidate["azimuth_deg"] // 45), []).append(candidate)
            sectors = sorted(by_sector)
            rng.shuffle(sectors)
            order = [by_sector[sector] for sector in sectors]
        #    preset 도 같이 흔든다. 격자 검증을 통과한 시작점이 궤적 검증에서 떨어지는 주된 원인은
        #    **sweep 방향**이다 (같은 자리에서 왼쪽으로 돌면 벽, 오른쪽으로 돌면 뚫린다). 시작점만
        #    바꿔 재시도하면 그 축을 못 건드린다.
        presets = [args.preset] if args.preset else sorted(SOURCE_PRESETS)
        #    `--exclude_presets` (D272): 같은 시작 pose 에서 두 번째 클립을 뽑을 때 첫 클립이 쓴
        #    preset 을 뺀다 — 같은 시작점에서 **다른 움직임** 짝을 만드는 용도. 비면 그대로.
        presets = [p for p in presets if p not in set(args.exclude_presets)]
        assert presets, f"--exclude_presets {args.exclude_presets} 가 preset 을 전부 뺐다"
        rng.shuffle(presets)
        tries = []
        dropped = []
        for i in range(int(args.max_tries)):
            bucket = order[i % len(order)]
            candidate = bucket[int(rng.integers(len(bucket)))]
            #    이 반경에서 크기대로 못 돌리는 preset 은 건너뛴다 (`preset_feasible` docstring).
            #    한 바퀴 돌아 전부 불가능하면 그 짝은 버린다 — max_tries 를 못 채우는 게, 로그와
            #    실제 이동량이 어긋나는 것보다 낫다.
            for j in range(len(presets)):
                preset = presets[(i + j) % len(presets)]
                if preset_feasible(candidate, preset, args.preset_scale, args.min_end_radius):
                    tries.append((candidate, preset))
                    break
                dropped.append((round(float(candidate["radius"]), 2), preset))
        assert tries, (f"preset_scale {args.preset_scale} 에서 성립하는 (후보, preset) 짝이 없다 "
                       f"(min_end_radius {args.min_end_radius}). 반경이 더 큰 격자가 필요하다.")
        if dropped:
            print(f"  preset 제외 {len(dropped)}쌍 (반경 부족): "
                  f"{sorted({f'{p}@r{r}' for r, p in dropped})}")

        synthesized = [synth_source_path(probe, c, p, args.aim_bias, args.track_gain,
                                         args.preset_scale, smooth_window, args.aim_keyframes,
                                         keep_start_z=board_cand is not None)
                       for c, p in tries]
        tries_path = path.join(work, f"tries_{ftag}.npz")
        np.savez(tries_path, cam_c2w=np.stack([p for p, _ in synthesized]))

        # 3) 궤적 검증 probe — 격자는 start/anchor/end 세 프레임뿐이라 이동 중 벽 통과를 못 잡는다.
        #    후보 전부를 **한 번의 Blender 기동**으로 검증한다 (기동+로드가 4초라 재시도가 비싸다).
        verify_path = path.join(work, f"verify_{ftag}.json")
        timing["probe_verify"] = run_probe(
            args, blend, path.join(HERE, "trumans_scene_probe.py"),
            #    verify 도 **같은 subject** 로 돌려야 한다. 여기만 human 이면 시선 관통 판정이
            #    사람만 보고, 소품이 벽장 안이든 화면 밖이든 통과해 버린다.
            ["--frames", str(start), str(end), str(step), "--out", verify_path,
             "--verify_poses", tries_path, *subject_flags], "probe(verify)",
            retries=args.blender_retries)
        with open(verify_path, encoding="utf-8") as file:
            verify = json.load(file)

        #    `min_subject_dist` 는 probe 가 새로 내는 열이다. 예전 verify JSON 에는 없으므로
        #    없으면 무한대로 봐서 기존 결과를 그대로 재현한다.
        passed = [row for row in verify["summary"]
                  if row["clear_frac"] >= args.min_clear_frac
                  and row["min_clearance"] >= args.min_clearance
                  and row.get("min_subject_dist", float("inf")) >= args.min_subject_dist]
        if passed and args.hold_fallback:
            #    `hold` 은 전 필드가 0 이라 **절대 충돌하지 않는다**. 방이 좁아 움직이는 preset 이
            #    줄줄이 떨어지면 `passed[0]` 이 자동으로 hold 을 집어, 로그에는 preset 이름이 남는데
            #    실제로는 정지 카메라 클립만 쌓인다 (stride3 ×3 실측: hold 이 ok 의 35%, step1 은 15%).
            #    그래서 hold 은 "움직이는 preset 이 전부 떨어졌을 때"의 대체재로만 쓴다.
            #    후보 추첨(RNG) 은 건드리지 않으므로 이 플래그를 끄면 예전 결과가 그대로 나온다.
            moving = [row for row in passed if tries[int(row["path"])][1] != "hold"]
            chosen = (moving or passed)[0]
        elif passed:
            chosen = passed[0]
        else:
            #    전부 떨어졌으면 제일 나은 걸 보고한다 — 어느 축이 부족했는지가 진단이다.
            chosen = max(verify["summary"], key=lambda r: (r["clear_frac"], r["min_clearance"]))
            message = (f"궤적 검증 실패 (후보 {len(tries)}개 전부): 최선 clear {chosen['clear_frac']:.2f} "
                       f"(>= {args.min_clear_frac}), min clearance {chosen['min_clearance']:.3f} "
                       f"(>= {args.min_clearance}), min subject_dist "
                       f"{chosen.get('min_subject_dist', float('inf')):.3f} (>= {args.min_subject_dist})")
            assert args.force, message + f"\n  {verify_path}\n  --force 로 무시 가능"
            print(f"  WARN {message}")
        k = int(chosen["path"])
        candidate, preset = tries[k]
        poses, aim = synthesized[k]
        clear_frac, min_clearance = chosen["clear_frac"], chosen["min_clearance"]
        print(f"  후보 {len(tries)}개 중 #{k} 채택 "
              f"(az {candidate['azimuth_deg']:.0f} elev {candidate['elevation_deg']:.0f} "
              f"r {candidate['radius']:.1f} {preset})  clear {clear_frac:.2f}  "
              f"clearance {min_clearance:.3f}  "
              f"subject_dist {chosen.get('min_subject_dist', float('nan')):.3f}")
    poses_path = path.join(work, f"poses_{ftag}.npz")
    #    override 면 **LBM 이 쓴 초점거리**를 같이 싣는다. `trumans_gt_render.load_poses_npz` 가
    #    `lens_mm` 을 보면 `--lens` 기본값(25 mm)을 무시하고 그 값을 쓴다 — LBM 은 24 mm 라
    #    이걸 안 실으면 화각이 4% 넓게 렌더돼서 프레이밍 지표가 통째로 어긋난다.
    if args.poses_override and override_lens:
        np.savez(poses_path, cam_c2w=poses, aim=aim, lens_mm=float(override_lens))
    else:
        np.savez(poses_path, cam_c2w=poses, aim=aim)
    stats = path_stats(poses)

    # 4) GT 렌더. 이 단계만 3분대라 나머지 전부를 합친 것보다 20배 비싸다. 같은 pose 로 이미
    #    렌더해 뒀으면 재사용한다 — pose 를 대조하므로 stale 렌더를 집을 위험은 없다.
    render_dir = path.join(work, f"render_{ftag}")
    if renders_match(render_dir, poses, num_frames=NUM_FRAMES, rgb_engine=args.rgb_engine):
        print(f"  렌더 재사용: {render_dir}")
        timing["render"] = 0.0
    elif args.rgb_anim and not frame_list and args.rgb_engine == "cycles":
        #    D276 (사용자 채택 2026-09-24 "blender 렌더 개선안 채택할게"). 한 job 에서 프레임마다
        #    RGB(Cycles GPU) ↔ depth/index(Cycles CPU) 를 번갈아 돌리면 장치가 바뀔 때마다 씬 전체를
        #    다시 올려서 persistent data 가 안 먹는다 (`trumans_gt_render.py:439-460`: 렌더 자체는
        #    4.6%, 95% 가 재동기화). 그래서 job 을 둘로 쪼갠다:
        #      ① RGB 만 `--anim` (animation render 1회 — 재동기화가 job 당 1회)
        #      ② depth,index 만 프레임 루프 (엔진·장치가 안 바뀌어 persistent data 가 먹는다)
        #    실측 (0ac97866 c17 k059, 단독 GPU): 7.8+4.2 s/frame → 2.1+2.9 s/frame.
        #    ①의 rgb/ 를 ② 폴더로 옮겨 `convert` 가 보는 구조는 예전과 같다.
        rgb_dir = render_dir + "_rgb"
        common = ["--poses", poses_path, "--frames", str(start), str(end), str(step),
                  "--res", str(args.res[0]), str(args.res[1]), "--lens", str(args.lens)]
        t_rgb = run_blender(
            args.blender, blend, path.join(CINEMATRAJ_ROOT, "viz", "trumans_gt_render.py"),
            common + ["--passes", "rgb", "--rgb_engine", "cycles",
                      "--rgb_samples", str(args.rgb_samples), "--rgb_cdevice", args.rgb_cdevice,
                      "--anim", "--out", rgb_dir], "gt_render(rgb anim)",
            retries=args.blender_retries)
        t_geo = run_blender(
            args.blender, blend, path.join(CINEMATRAJ_ROOT, "viz", "trumans_gt_render.py"),
            common + ["--passes", "depth,index", "--out", render_dir], "gt_render(depth,index)",
            retries=args.blender_retries)
        import shutil
        dst = path.join(render_dir, "rgb")
        if path.isdir(dst):
            shutil.rmtree(dst)
        shutil.move(path.join(rgb_dir, "rgb"), dst)
        shutil.copy(path.join(rgb_dir, "render_meta.json"), path.join(render_dir, "render_meta_rgb.json"))
        shutil.rmtree(rgb_dir)
        n_rgb = len([f for f in listdir(dst) if f.endswith(".png")])
        assert n_rgb == NUM_FRAMES, f"rgb anim 프레임 {n_rgb} != {NUM_FRAMES}: {dst}"
        timing["render"] = t_rgb + t_geo
        timing["render_rgb"], timing["render_geom"] = t_rgb, t_geo
    else:
        timing["render"] = run_blender(
            args.blender, blend, path.join(CINEMATRAJ_ROOT, "viz", "trumans_gt_render.py"),
            ["--poses", poses_path, "--frames", str(start), str(end), str(step),
             "--res", str(args.res[0]), str(args.res[1]),
             "--passes", "rgb,depth,index", "--lens", str(args.lens),
             "--samples", str(args.samples), "--out", render_dir]
            #    기본 엔진이면 인자를 안 넘긴다 — 기존 렌더 명령과 비트 동일해야 한다.
            #    `--rgb_cdevice` 만 GPU 로 — geometry(1 spp)는 CPU 가 더 빠르다.
            + ([] if args.rgb_engine == "eevee" else
               ["--rgb_engine", args.rgb_engine, "--rgb_samples", str(args.rgb_samples),
                "--rgb_cdevice", args.rgb_cdevice])
            #    override 는 간격이 비균일해서 `--frames` 격자로는 못 만든다.
            + (["--frame_list", *[str(f) for f in frame_list]] if frame_list else []),
            "gt_render", retries=args.blender_retries)

    # 5) 변환
    started = time.time()
    out_recon = path.join(args.eval_data, "eval_data", "recon_and_seg", video)
    out_seg = path.join(args.eval_data, "eval_data", "seg_instances", video)
    out_seg_static = path.join(args.eval_data, "eval_data", "seg_instances_static", video)
    #    mp4 재생 fps. blend 의 render fps 는 **모션 레이트가 아니다** (실측 7편 중 6편이 15,
    #    1편이 25 인데 `human_transl` 은 30 Hz) — 그걸 쓰면 클립이 2배 느리게 재생돼서 사람과
    #    VLM 이 속도를 잘못 읽는다. cameras.npz / depths / masks 같은 수치 산출물은 안 바뀐다.
    out_fps = args.out_fps if args.out_fps > 0 else args.motion_fps / step
    report = convert(render_dir, poses, out_recon, out_seg, out_seg_static,
                     float(out_fps), args.sky_depth, args.vista4d_root)
    timing["convert"] = time.time() - started

    manifest = {
        "format": "trumans_recon_v1", "video": video,
        "recording": args.recording, "sequence": sequence,
        "action_index": args.action, "action": action,
        # 창에 걸치는 action 전부. packer 가 caption 을 조립할 때 `action` 하나만 쓰면
        # step 3 (145프레임 창) 에서 클립의 69% 가 문장에 없는 동작을 보여준다.
        "window_actions": window_actions,
        # D272 board 모드: 어느 board 후보에서 출발했나. 같은 chunk·같은 후보 = 같은 시작 pose.
        "board": (None if board_cand is None else
                  {"json": args.board_json, "chunk": args.board_chunk,
                   "candidate": int(args.board_candidate), "slot": int(args.board_slot),
                   "exclude_presets": list(args.exclude_presets),
                   "azimuth_deg": board_cand["azimuth_deg"],
                   "elevation_deg": board_cand["elevation_deg"],
                   "radius": board_cand["radius"], "shot": board_cand.get("shot"),
                   "position": board_cand["position"], "look_at": board_cand["look_at"]}),
        "frames": [start, end], "frame_step": step,
        # 실제 표본 격자. override 는 간격이 비균일해서 `frame_step` 만으로 못 되짚는다.
        "frame_list": frame_list,
        "fps": float(out_fps), "blend_fps": float(probe_meta["fps"]),
        "motion_fps": float(args.motion_fps),
        "duration_seconds": (NUM_FRAMES - 1) * step / float(args.motion_fps),
        # subject 와 caption 은 같은 `resolve_subject` 한 번에서 나온다 (어긋나면 캡션과
        # 프레이밍이 다른 클립이 조용히 생긴다). caption 의 framing/motion 줄은 아직 없다.
        "subject": {"kind": subject_kind, "prop_names": prop_names,
                    "prop_stem": caption["prop_stem"], "anchor_origin": args.anchor_origin,
                    "human_height": probe.get("human_height"),
                    "subject_height": probe.get("subject_height")},
        "caption": {"target": caption["target"], "event": caption["event"]},
        "source_camera": {
            # `human` 이 아니면 앵커가 사람이 아니다 — 뱅크를 섞어 읽을 때 이 값으로 가른다.
            # override 는 카메라를 우리가 안 만들었다 — LBM 이 렌더한 것을 그대로 재렌더한 것.
            "kind": ("lbm_render_dump" if args.poses_override
                     else f"synthesized_{subject_kind}_anchored"),
            "poses_override": args.poses_override,
            "preset": preset, "seed": args.seed,
            "candidate": candidate, "aim_bias": args.aim_bias, "track_gain": args.track_gain,
            "aim_keyframes": int(args.aim_keyframes),
            "stats": stats,
            "reference": {"note": "실제 pkl 2편(00add26c, 0aa05d5a) 통계에 맞춤",
                          "dt_median": [0.01286, 0.01280], "net49_median": [0.5794, 0.6180]},
        },
        "verify": {"clear_frac": clear_frac, "min_clearance": min_clearance},
        "render": report, "timing_seconds": timing,
        "paths": {"probe": probe_path, "verify": verify_path, "poses": poses_path,
                  "render": render_dir, "recon": out_recon, "seg": out_seg,
                  "seg_static": out_seg_static},
    }
    with open(path.join(work, f"manifest_{ftag}.json"), "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=1)

    print(f"\n{'video':22s} {video}")
    print(f"{'action':22s} [{args.action}] {action['text'][:48]}")
    print(f"{'frames':22s} {start}..{end} step {step}  out fps {out_fps:g} "
          f"(blend {probe_meta['fps']:g}, motion {args.motion_fps:g})  "
          f"{report['width']}x{report['height']}")
    if candidate:
        print(f"{'candidates usable':22s} {len(usable)}/{len(probe['candidates'])}  "
              f"tries {len(tries)} -> #{k}  az {candidate['azimuth_deg']:.0f} "
              f"elev {candidate['elevation_deg']:.0f} r {candidate['radius']:.1f}")
    else:
        print(f"{'poses_override':22s} {args.poses_override}  lens {override_lens or args.lens:g}mm")
    print(f"{'source preset':22s} {preset}")
    # 비교 기준을 창 종류로 가른다. 예전엔 still 통계 하나만 찍어서 보행 클립이 전부 "2배
    # 초과"로 보였다 (실제로는 실측 추종 카메라와 같은 값이다. 위 docstring 표).
    reference = ("walk  pkl 0.0141~0.0309 / 0.59~1.31" if action.get("kind") == "walk"
                 else "still pkl 0.0122~0.0137 / 0.44~0.55")
    print(f"{'path dt_median/net':22s} {stats['dt_median']:.4f} / {stats['net']:.3f} m"
          f"   ({reference})")
    print(f"{'verify clear/clr':22s} {clear_frac:.2f} / {min_clearance:.3f} m")
    print(f"{'sky/human frac':22s} {report['sky_frac']:.3f} / {report['human_frac']:.3f}")
    print(f"{'tracks dyn/static':22s} {report['n_dyn_tracks']} / {report['n_static_tracks']}")
    print(f"{'depth median':22s} {report['depth_p50']:.2f} m")
    # 사람 제외 값과 크게 벌어지면 분모가 씬이 아니라 사람 거리를 재고 있다는 신호다.
    print(f"{'avg_scale (static)':22s} {report['avg_scale']:.3f} m "
          f"({report['avg_scale_static']:.3f} m)")
    for name, seconds in timing.items():
        print(f"{'time.' + name:22s} {seconds:7.1f} s")
    print(f"{'->':22s} {out_recon}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recording", required=True, type=str)      # Recordings_blend 디렉토리명
    parser.add_argument("--action", default=0, type=int)             # Actions/<seq>.txt 의 0-based 줄
    parser.add_argument("--list_actions", action="store_true")       # action 목록만 찍고 종료
    parser.add_argument("--out_video", default="", type=str)         # 비우면 tru_<uuid8>_a<NN>
    parser.add_argument("--sequence", default="", type=str)          # 비우면 프레임 수로 자동 판별

    # 보행 pseudo-action. Actions/*.txt 에 보행 라벨이 0건이라(모션엔 프레임의 19.0%) 이걸 끄면
    # 뱅크가 전부 제자리 조작 동작이 된다. 목록 **뒤에** 붙으므로 기존 --action 인덱스는 불변.
    parser.add_argument("--walk_actions", dest="walk_actions", action="store_true", default=True)
    parser.add_argument("--no_walk_actions", dest="walk_actions", action="store_false")
    parser.add_argument("--walk_speed", default=0.4, type=float)     # 보행 판정 수평속도(m/s)
    parser.add_argument("--walk_frac", default=0.8, type=float)      # 49프레임 중 보행 프레임 비율 하한
    parser.add_argument("--walk_smooth", default=9, type=int)        # 속도 평활 창(프레임)
    parser.add_argument("--walk_max", default=8, type=int)           # 시퀀스당 채굴 상한 (이동량 큰 순)
    # 모션 배열(`human_transl`)의 레이트. **blend 의 render fps 가 아니다** — 그건 편마다 15/25 로
    # 다른 렌더 설정이고, `video_render/<seq>.pkl.mp4` 는 7편 전부 프레임 수가 시퀀스 길이와 1:1
    # 이고 fps 30 이다. blend fps 를 쓰면 fps 15 편에서 같은 걸음이 0.24 m/s 로 찍혀 임계 아래로
    # 떨어져 **7편 중 5편이 채굴 0건**이 된다 (프레임당 이동량 분포는 7편이 사실상 같다).
    # 0 을 주면 예전처럼 blend fps 를 쓴다 (재현용).
    parser.add_argument("--motion_fps", default=30.0, type=float)
    # **모션 프레임 간격**. 모션이 30 Hz 라 step 3 이면 10 fps 로 샘플한 셈이고 49장이
    # (49-1)*3+1 = 145 모션프레임 = 4.83 초를 덮는다. 1 = 기존 동작 (49장 = 1.63 초).
    parser.add_argument("--frame_step", default=1, type=int)
    # 출력 mp4 fps. 0 이면 motion_fps/frame_step (= 실제 재생 속도). 예전 코드는 blend 의
    # render fps(편마다 15/25)를 그대로 썼는데 모션은 30 Hz 라 클립이 2배 느리게 재생됐다.
    parser.add_argument("--out_fps", default=0.0, type=float)
    # 사람 추종 평활 창(프레임). 0 이면 round(11/step) 로 자동 (평활의 **시간** 폭을 step 과
    # 무관하게 유지). step=1 이면 11 이라 기존 동작과 같다.
    parser.add_argument("--smooth_window", default=0, type=int)

    parser.add_argument("--preset", default="", choices=("", *sorted(SOURCE_PRESETS)))
    # SOURCE_PRESETS 값은 "49프레임 클립 전체의 이동량"이라 실시간 길이에 묶여 있다. step 3 이면
    # 클립이 1.63 -> 4.83 초라 같은 카메라 **속도**를 유지하려면 3 배가 필요하다.
    parser.add_argument("--preset_scale", default=1.0, type=float)
    # push_in 계열이 끝 반경을 이 값 아래로 끌면 그 (후보, preset) 짝을 **버린다**.
    # synth_source_path 의 np.maximum(0.6, ...) 클램프는 불가능한 dolly 를 조용히 잘라내고도
    # "성공"으로 찍힌다 — preset 이름과 실제 이동이 어긋나는 그 종류의 버그.
    parser.add_argument("--min_end_radius", default=MIN_END_RADIUS, type=float)
    # 격자 probe 만 돌리고 종료 (렌더 안 함). recording 전체에서 공통 anchor 칸을 고르는
    # --anchor_share 1 패스용. action 당 ~7.5 초.
    parser.add_argument("--probe_only", action="store_true")
    # 카메라를 합성하지 않고 **밖에서 받는다** (`lbm_camera_to_poses.py` 산출 npz: cam_c2w /
    # aim / trumans_frames / lens). 궤적 합성(2단계)과 검증 probe(3단계)를 건너뛰고 GT 렌더 +
    # 변환만 한다. LBM 카메라를 Lite 뱅크와 같은 눈금으로 재기 위한 경로다.
    parser.add_argument("--poses_override", default="", type=str)
    # 시작 pose 를 이 격자 칸으로 **강제** (az elev radius). usable 에 없으면 에러.
    # frame0 카메라를 클립 간 공유하려면 subject-local (방위각, 고도, 반경) 을 고정해야 한다
    # — position 만 공유해도 look_at 이 클립마다 달라 회전이 안 맞는다.
    parser.add_argument("--anchor_cell", default=[], nargs=3, type=float)
    # D272 board 모드 — `trumans_first_pose_board.py` 의 board.json 에서 chunk 창 + 시작 pose.
    # 셋 다 비면 예전 action 모드 그대로 (비트 동일).
    parser.add_argument("--board_json", default="", type=str)
    parser.add_argument("--board_chunk", default="", type=str)        # 예: c00_f00000
    parser.add_argument("--board_candidate", default=-1, type=int)    # chunk["candidates"] 인덱스
    parser.add_argument("--board_slot", default=0, type=int)          # 같은 시작 pose 의 몇 번째 클립
    parser.add_argument("--exclude_presets", default=[], nargs="*", type=str)
    parser.add_argument("--seed", default=0, type=int)               # 시작 pose/preset 샘플 seed
    # 격자 원점. obb_center 는 subject-agnostic 이라 event/object 로 일반화되고 `anchor_cond` 가
    # 싣는 값과 원점이 일치한다. chest 는 human 전용이고 기존 뱅크 96편 재현 전용이다.
    # subject 단위. `human`(기본)은 소품 매칭을 아예 돌지 않아 기존 뱅크 96편과 비트 동일하다.
    #   auto   = action text 에 소품이 잡히면 event, 아니면 human (대량 생성용)
    #   event  = 사람 ∪ 소품   object = 소품만  — 못 찾으면 조용히 떨어지지 않고 에러
    parser.add_argument("--subject_kind", default="human",
                        choices=("human", "auto", "event", "object"))
    # 소품 오브젝트 이름을 직접 지정 (주면 텍스트 매칭을 건너뛴다). obj_list.txt 의 이름 그대로.
    parser.add_argument("--prop_names", default=[], nargs="*", type=str)

    parser.add_argument("--anchor_origin", default="obb_center", choices=("obb_center", "chest"))
    # look-at 높이 보정(사람 키 비율). 안 주면 원점에 맞춰 정한다 — OBB center 는 0.50·h 라
    # 그대로 겨누면 fill≈1.9 에서 머리가 프레임 위로 나간다. +0.20 이 예전 흉부(0.70·h)와 같다.
    # 즉 하드코딩 휴리스틱이 로그에 찍히는 숫자가 된다.
    parser.add_argument("--aim_bias", default=None, type=float)
    parser.add_argument("--track_gain", default=0.6, type=float)     # 사람 추종 정도 (1=완전 추종)
    # 조준 keyframe 수. 0(기본) = 매 프레임 look-at (기존 뱅크 재현). 6 = Vista4D 뱅크와 같은
    # [0,10,19,29,38,48] keyframe + smoothstep slerp — 두 코퍼스를 섞어 학습할 때 조준 방식이
    # 코퍼스 라벨이 되지 않게 맞추는 값이다 (`synth_source_path` docstring).
    parser.add_argument("--aim_keyframes", default=0, type=int)

    # D120. 0.35 → 0.20 (사용자 지시 2026-09-03). `bank_to_blender_poses.py` 와 **같은 값**을
    # 유지해야 소스/target 대조가 성립한다 — 한쪽만 바꾸지 말 것. ±z 는 그대로 둔다.
    parser.add_argument("--min_clearance", default=0.20, type=float)   # 카메라-표면 최소 거리(m)
    parser.add_argument("--min_floor_drop", default=0.30, type=float)  # 카메라 아래 바닥까지 최소(m)
    parser.add_argument("--min_clear_frac", default=0.90, type=float)  # 궤적 중 시선이 뚫린 비율
    # 카메라-사람 최소 거리(m). `min_clearance` 는 사람 히트를 무시하므로 이게 없으면 피사체를
    # 뚫고 들어가는 궤적을 못 막는다. 0.8 은 **예전 반경 집합에선 절대 안 걸리는** 값이다
    # (하한 1.5 m + push_in dradius -0.55 -> 0.95 m). r=1.0 을 넣으면서 생긴 구멍만 닫는다.
    parser.add_argument("--min_subject_dist", default=0.80, type=float)
    parser.add_argument("--max_tries", default=6, type=int)            # 검증에 걸 시작 pose 후보 수
    # `hold`(정지) 은 움직이는 preset 이 전부 떨어졌을 때만 채택. off 면 예전처럼 통과한 첫 후보.
    parser.add_argument("--hold_fallback", dest="hold_fallback", action="store_true", default=False)
    parser.add_argument("--no_hold_fallback", dest="hold_fallback", action="store_false")
    # Blender 가 **시그널로** 죽었을 때만 재시도 (rc<0). 0 이면 예전 동작. 자세한 근거는
    # `run_blender` 위 주석 — worker 경합으로 EEVEE context 가 터지는 게 유일한 관측 사례다.
    parser.add_argument("--blender_retries", default=1, type=int)
    # 재시도 순서를 (반경 x 방위각) 으로 층화. off 면 방위각만 층화하던 예전 순서 그대로.
    parser.add_argument("--radius_strata", dest="radius_strata", action="store_true", default=True)
    parser.add_argument("--no_radius_strata", dest="radius_strata", action="store_false")
    parser.add_argument("--force", action="store_true")                # 검증 실패해도 렌더 강행

    parser.add_argument("--res", default=[960, 540], nargs=2, type=int)
    parser.add_argument("--lens", default=25.0, type=float)          # mm (TRUMANS 씬 카메라와 동일)
    parser.add_argument("--samples", default=16, type=int)           # EEVEE RGB pass 샘플 수
    # RGB 패스 엔진. `trumans_gt_render.py --rgb_engine` 으로 그대로 내려간다. blend 가 3.3.6
    # 저작이라 4.5 의 EEVEE_NEXT 에서는 발광 재질이 타서 옆 물체까지 번진다 — `cycles` 가 그
    # 탈출구다. 기본값 `eevee` 면 인자를 아예 안 넘겨 기존 명령과 비트 동일하다.
    parser.add_argument("--rgb_engine", default="eevee", choices=["eevee", "cycles"], type=str)
    # D276. cycles RGB 를 `--anim` 한 job + depth/index 별 job 으로 (균일 간격만). 끄면 예전 1-job.
    parser.add_argument("--rgb_anim", action="store_true", default=False)
    # D278. probe(grid)/probe(verify) 를 recording 상주 Blender 서버(`lbm/blender_raycast.py`)에서.
    # 끄면 예전처럼 probe 마다 Blender 를 새로 띄운다.
    parser.add_argument("--probe_server", action="store_true", default=False)
    parser.add_argument("--rgb_samples", default=128, type=int)      # rgb_engine=cycles 일 때 spp
    parser.add_argument("--rgb_cdevice", default="GPU", choices=["CPU", "GPU"], type=str)
    parser.add_argument("--sky_depth", default=1000.0, type=float)   # 배경 clamp (float16 상한 회피)

    parser.add_argument("--trumans", default=TRUMANS_DEFAULT, type=str)
    parser.add_argument("--blender", default=BLENDER_DEFAULT, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_DEFAULT, type=str)
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--work", default="", type=str)              # 중간 산출물 (비우면 out/trumans_recon)
    parsed = parser.parse_args()
    if parsed.aim_bias is None:
        # OBB center(0.50·h) 를 그대로 겨누면 머리가 잘린다 → 흉부(0.70·h) 로 올린다.
        parsed.aim_bias = 0.20 if parsed.anchor_origin == "obb_center" else 0.0
    main(parsed)
