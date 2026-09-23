"""SAM3 인스턴스 → VLM → 캡션이 쓸 **지칭 표현**. D120, 사용자 지시 2026-09-03.

왜 SAM3 **뒤**에 붙나 (사용자 제안): 앞단 `extract_nouns_vlm.py` 는 detector 에 먹일 명사를
뽑는 자리라 "lowercase, singular, 1-2 words" 를 강제한다 — GroundingDINO/SAM3 는 caption 의
부분문자열을 라벨로 돌려주므로 형용사구가 길면 phrase 가 엉킨다. 거기서 긴 표현까지 같이
받으면 (a) 명사↔track 대응이 애매해지고 (b) 아직 어떤 인스턴스가 main subject 가 될지 모르는데
전부에 대해 VLM 을 부르게 된다.

여기서 하면 세 가지가 공짜로 해결된다:
  1. **대응이 이미 확정**돼 있다. 노드 id(`dyn_0`)마다 마스크가 있으니 "이 마스크의 물체를
     설명해라"라고 물을 수 있고, 문자열 매칭이 필요 없다.
  2. **호출 수가 앵커 수**다. 뱅크에 실제로 등장한 `anchor_id` 만 물어보면 영상당 1~6회다.
  3. 마스크 crop 을 **여러 프레임** 보여줄 수 있어서 "drinking a coffee" 같은 **행위**가 붙는다.
     한 장으로는 정지 자세만 보이고 행위는 안 보인다.

동명 인스턴스(avocado-slice: `dyn_1`/`dyn_2` 둘 다 avocado, `stat_1`/`stat_2` 둘 다 window)는
**형제 목록을 프롬프트에 같이 넣어** 구별되는 표현을 요구한다. 안 그러면 둘 다 "the avocado"
로 돌아와서 캡션이 어느 쪽을 조준하는지 못 가린다.

D120-b. 그것만으로는 부족했다 — camel 실측에서 두 낙타가 `"the light-colored camel walking
away from the fence"` / `"the camel facing forward"` 로 나왔는데, **둘 다 밝은 색**이라 앞의
것은 구별이 안 된다. 원인은 프롬프트가 아니라 **입력**이다: 패널은 한 번에 한 마리만 강조해
보여주므로 VLM 은 상대 비교를 할 근거가 없다. 그래서 두 가지를 넣는다.
  1. `## MEASURED FACTS` — 마스크에서 잰 화면 좌우 위치 · 면적비 · 깊이 순서를 **결정론적으로**
     프롬프트에 싣는다. 실측상 형제쌍 5개 전부가 프레임 100% 안정이다 (camel 은 dyn_0 이 49/49
     프레임에서 더 크고 더 가깝다, 면적비 3.69배). 유일한 난케이스는 avocado 반쪽 둘로
     `|Δcx| 0.059` 뿐이라 좌/우만 쓸 수 있다.
  2. 형제를 **순서대로** 처리하며 앞서 나온 형제 문구를 프롬프트와 검증기에 같이 넘긴다.
     검증기는 "형제 문구에 없는 내용어가 최소 하나"를 요구한다 — 길이만 보던 예전 규칙은
     `"the light-colored camel ..."` 를 통과시켰다.
소진 시 fallback 도 라벨 하나가 아니라 실측 기반 수식어(`the nearer camel`)를 붙인다.

출력 `out/<video>/instance_desc.json` 은 `build_bank_captions.py --anchor_desc` 가 읽는다.
그쪽에서 파일이 없으면 조용히 예전 라벨(`anchor_label`)로 떨어진다 — 이 스크립트는 **선택**이다.

사용 예시:
    python fit/graph/describe_instances_vlm.py --videos avocado-slice camel \
        --bank_dir hole_bank_k6_d99
    python fit/graph/describe_instances_vlm.py --videos avocado-slice --all_nodes --dry_run
"""
import json
import sys
import time
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
VISTA4D_ROOT = "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D"
EVAL_DATA = "/data1/cympyc1785/data/Vista4D-Eval-Data"

sys.path.insert(0, HERE)
from lbm.vlm import VLMClient          # noqa: E402  — stdlib 만 쓰는 OpenAI 호환 클라이언트
from scene_graph.io import SegInstances  # noqa: E402

# 프롬프트는 영문 (계획서 규칙). 캡션 문장 `The camera <motion> around <target> ...` 의
# `<target>` 자리에 그대로 꽂히므로 **관사로 시작하는 명사구**여야 하고, 문장이 아니어야 한다.
SYSTEM = """You are a vision annotator writing referring expressions for a film-shot caption.

You are shown frames from one video shot. In each frame, exactly one object is highlighted: it \
keeps its original colours while everything else is darkened, and a yellow box marks it. A tight \
crop of the same object is shown underneath.

Write a referring expression for that highlighted object.

Rules:
- a noun phrase, not a sentence. Start with "the". No trailing period.
- at most 14 words.
- it MUST say what the object looks like (colour, clothing, material, markings) and, if it is \
doing something across the frames, what it is doing. A phrase built only out of position and \
size ("the large table on the left") is NOT acceptable on its own.
- position and size are a TIE-BREAKER you append when a sibling with the same base noun exists \
— never a replacement for appearance. Good: "the woman in a striped shirt slicing an avocado". \
Good with a sibling: "the nearer pale camel walking along the fence".
- it must contain the given base noun (or an obvious singular/plural form of it).
- use only what you can see: colour, clothing, material, position relative to other visible \
things, and what the object is doing across the frames.
- if sibling objects with the same base noun are listed, your phrase MUST distinguish this one \
from them. A trait both of them share (e.g. both are light-coloured) does NOT distinguish.
- when a MEASURED FACTS block is given, those numbers were measured from the segmentation and \
are more reliable than your impression. Prefer them for size, side of frame, and depth order.
- never mention the camera, the video, the frames, the highlight, or the yellow box.
Answer with raw JSON only. No markdown fences, no commentary outside the JSON."""

USER = """Video shot: {width}x{height}, {num_frames} frames.
Highlighted object: base noun "{label}"{sibling_note}
Other objects present in this scene: {others}
Frames shown (time order): {frames}
The object {motion_note}
{disambiguation}
Return exactly this JSON object:
{{"description": "the ...", "attributes": ["..."], "action": "... or empty string", \
"confidence": 0.0}}
- "description": the referring expression (the thing we actually use).
- "attributes": the visual cues you relied on, 1-4 short strings.
- "action": what the object is doing, or "" if it is not doing anything.
- "confidence": 0..1, how sure you are that the phrase picks out this object and no other."""


def resolve_seg_folders(eval_data: str, video: str):
    """(동적 폴더, 정적 폴더). `scene_graph/io.load_scene` 과 **같은 탐색 순서**다.

    inline fallback(`recon_and_seg/<video>/seg_instances`)까지 같이 본다 — 새 소스를 1편씩
    넣을 때 symlink 를 안 걸어도 돌아가야 한다 (io.py 의 같은 주석 참조).
    """
    dyn = path.join(eval_data, "eval_data", "seg_instances", video)
    if not path.isfile(path.join(dyn, "meta.json")):
        inline = path.join(eval_data, "eval_data", "recon_and_seg", video, "seg_instances")
        dyn = inline if path.isfile(path.join(inline, "meta.json")) else ""
    stat = path.join(eval_data, "eval_data", "seg_instances_static", video)
    stat = stat if path.isfile(path.join(stat, "meta.json")) else ""
    return dyn, stat


def node_track_ids(node: dict):
    """노드가 실제로 덮는 track id 들. `merged_from` 은 `"dyn#3"` 꼴 문자열이다.

    병합 노드(avocado-slice `dyn_0` = `dyn#0,1,2`)를 대표 track 하나로만 그리면 사람이
    프레임마다 다른 조각으로 보여서 VLM 이 "the arm" 같은 답을 낸다.
    """
    ids = {int(node["track_id"])}
    for entry in node.get("merged_from") or []:
        text = str(entry)
        if "#" in text:
            ids.add(int(text.split("#", 1)[1]))
    return sorted(ids)


def node_mask(seg: SegInstances, track_ids, frame: int):
    """그 프레임의 노드 마스크 (track 합집합)."""
    mask = np.zeros((seg.height, seg.width), dtype=bool)
    for tid in track_ids:
        if tid in seg.tracks:
            mask |= seg.track_mask(tid, frame)
    return mask


def pick_frames(seg: SegInstances, track_ids, count: int):
    """면적 상위에서 **시간으로 퍼뜨려** 고른다. (프레임 인덱스, 면적비) 리스트.

    상위 N 을 그냥 집으면 연속 프레임 N 장이 나와서 행위가 안 보인다 (전부 같은 자세다).
    그래서 시간을 `count` 구간으로 나누고 구간마다 면적 최대 프레임을 하나씩 집는다.
    면적이 0 인 구간은 건너뛴다 — occlusion 구간에 빈 칸을 쓰면 VLM 이 배경을 설명한다.
    """
    areas = np.zeros(seg.num_frames, dtype=np.float64)
    for tid in track_ids:
        track = seg.tracks.get(tid)
        if not track:
            continue
        for frame, box in zip(track["frames"], track["boxes"]):
            x0, y0, x1, y1 = box
            areas[frame] += max(0.0, x1 - x0) * max(0.0, y1 - y0)
    areas /= float(seg.height * seg.width)
    edges = np.linspace(0, seg.num_frames, count + 1).round().astype(int)
    picks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        local = areas[lo:hi]
        if local.max() <= 0:
            continue
        picks.append((int(lo + int(local.argmax())), float(local.max())))
    return picks


def panel(frame_rgb, mask, dim: float, crop_pad: float, tile: int):
    """(강조 전체 프레임, 타이트 crop) 두 장을 세로로 붙인 하나의 패널.

    전체 프레임만 주면 작은 물체(avocado 는 화면의 0.3%)가 몇 픽셀이라 색도 행위도 안 보이고,
    crop 만 주면 "왼쪽 것/오른쪽 것" 같은 **위치 기반 구별**이 불가능해진다. 둘 다 준다.
    """
    from PIL import Image, ImageDraw

    canvas = frame_rgb.astype(np.float32)
    canvas[~mask] *= dim                       # 바깥만 어둡게 — 마스크가 곧 지시봉이다
    canvas = canvas.clip(0, 255).astype(np.uint8)

    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    if len(xs) == 0:                           # 호출자가 거르지만 방어적으로
        x0, y0, x1, y1 = 0, 0, width - 1, height - 1
    else:
        x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    full = Image.fromarray(canvas)
    draw = ImageDraw.Draw(full)
    draw.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=max(2, width // 320))

    pad_x, pad_y = int((x1 - x0 + 1) * crop_pad), int((y1 - y0 + 1) * crop_pad)
    box = (max(x0 - pad_x, 0), max(y0 - pad_y, 0),
           min(x1 + pad_x + 1, width), min(y1 + pad_y + 1, height))
    #    crop 은 **원본 색**에서 딴다. 어둡게 한 캔버스에서 따면 가림 물체가 안 보인다.
    crop = Image.fromarray(frame_rgb).crop(box)

    out = Image.new("RGB", (tile, tile * 2), (0, 0, 0))
    for row, image in enumerate((full, crop)):
        image = image.copy()
        image.thumbnail((tile, tile))
        out.paste(image, ((tile - image.width) // 2,
                          row * tile + (tile - image.height) // 2))
    return out


def build_image(frames, seg, track_ids, picks, out_path, args):
    """패널들을 가로로 이어 한 장으로. 이미지 1장 = VLM 호출 1회."""
    from PIL import Image

    panels = [panel(frames[f], node_mask(seg, track_ids, f), args.dim, args.crop_pad, args.tile)
              for f, _ in picks]
    sheet = Image.new("RGB", (args.tile * len(panels), args.tile * 2), (0, 0, 0))
    for i, tile_image in enumerate(panels):
        sheet.paste(tile_image, (i * args.tile, 0))
    makedirs(path.dirname(out_path), exist_ok=True)
    sheet.save(out_path, quality=95)
    return out_path


def node_stats(node, seg: SegInstances, graph: dict):
    """마스크 bbox 에서 잰 화면 통계 + 깊이. **형제 구별의 근거**다 (D120-b).

    화면 좌표는 마스크에서 직접 재고, 깊이는 `track.center_smooth` 의 노름(없으면 OBB center)을
    쓴다 — 둘 다 graph frame G 원점이 소스 frame0 카메라라 곧 "카메라로부터의 거리"다.
    `present` 는 프레임별 등장 여부라서 형제 비교를 **공존 프레임에서만** 하게 해준다.
    """
    num = seg.num_frames
    cx = np.full(num, np.nan)
    area = np.zeros(num)
    for tid in node_track_ids(node):
        track = seg.tracks.get(tid)
        if not track:
            continue
        for frame, box in zip(track["frames"], track["boxes"]):
            x0, y0, x1, y1 = box
            a = max(0.0, x1 - x0) * max(0.0, y1 - y0) / float(graph["width"] * graph["height"])
            if a > 0:
                area[frame] += a
                cx[frame] = (x0 + x1) / 2.0 / float(graph["width"])
    smooth = (node.get("track") or {}).get("center_smooth")
    if smooth is not None and len(smooth) == num:
        depth = np.linalg.norm(np.asarray(smooth, dtype=np.float64), axis=1)
    else:
        depth = np.full(num, float(np.linalg.norm(node["obb"]["center"])))
    return {"cx": cx, "area": area, "depth": depth, "present": area > 0}


def compare_lines(mine: dict, other: dict, other_label: str):
    """형제 하나에 대한 상대 비교 3줄. 공존 프레임이 없으면 빈 리스트.

    비율·좌우는 **프레임별로 세서 안정도(%)를 같이 준다**. 중앙값만 주면 한 프레임 스침으로
    뒤집히는 관계를 VLM 이 항상 참인 것처럼 쓴다 — 실측 5쌍은 전부 94~100% 로 안정적이었다.
    """
    both = mine["present"] & other["present"]
    if not both.any():
        return [f'- it never appears at the same time as {other_label}']
    left = float(np.mean(mine["cx"][both] < other["cx"][both]))
    bigger = float(np.mean(mine["area"][both] > other["area"][both]))
    nearer = float(np.mean(mine["depth"][both] < other["depth"][both]))
    ratio = float(np.median(mine["area"][both] / np.maximum(other["area"][both], 1e-9)))
    side = "to the LEFT of" if left >= 0.5 else "to the RIGHT of"
    return [
        f'- it is {side} {other_label} in {max(left, 1 - left):.0%} of the frames they share',
        f'- it looks {"BIGGER" if ratio >= 1 else "SMALLER"} — {max(ratio, 1 / max(ratio, 1e-9)):.1f}x '
        f'the on-screen area of {other_label} ({max(bigger, 1 - bigger):.0%} of frames)',
        f'- it is {"NEARER to" if nearer >= 0.5 else "FURTHER from"} the viewpoint than '
        f'{other_label} ({max(nearer, 1 - nearer):.0%} of frames)']


def disambiguation_block(node, sibling_nodes, stats, prior_descs):
    """형제가 있을 때만 붙는 `## MEASURED FACTS` + 이미 쓴 형제 문구."""
    if not sibling_nodes:
        return ""
    mine = stats.get(node["id"])
    lines = [f'\n## MEASURED FACTS about the highlighted object (from the segmentation masks)',
             f'- on-screen area: {100 * np.nanmedian(np.where(mine["present"], mine["area"], np.nan)):.1f}% '
             f'of the frame (median)',
             f'- horizontal position: x={np.nanmedian(mine["cx"]):.2f} (0 = left edge, 1 = right edge)']
    for other in sibling_nodes:
        stat = stats.get(other["id"])
        if stat is None:
            continue
        name = f'the other {other["label"]}'
        lines.append(f'Compared with {name}:')
        lines += ["  " + line for line in compare_lines(mine, stat, name)]
    if prior_descs:
        lines.append('\nPhrases already used for the other object(s) with this same base noun — '
                     'yours must NOT be interchangeable with them:')
        lines += [f'  - "{text}"' for text in prior_descs]
    return "\n".join(lines) + "\n"


STOPWORDS = frozenset("the a an of in on at to with and or is are its his her their that this "
                      "it they from into over under near by for as while".split())

# D120-b. 위치·크기 어휘. `MEASURED FACTS` 를 넣자 VLM 이 **여기서만** 문구를 만들어
# `"the large wooden table on the left"` 처럼 외양·행위가 통째로 빠졌다 (실측: woman 이
# `"...sitting at a table"` → `"the woman in a striped shirt"` 로 줄었다). 구별은 되지만
# 캡션 target 으로는 후퇴다. 그래서 **이 집합 밖 내용어를 최소 1개** 요구한다.
SPATIAL = frozenset("left right leftmost rightmost centre center middle upper lower top bottom "
                    "front back behind foreground background corner side large small larger "
                    "smaller big bigger nearer nearest farther furthest closer closest distant "
                    "far near tall short wide narrow first second".split())


def content_words(text: str, label: str):
    """구별에 실제로 기여하는 단어들. 관사·전치사와 base noun 은 뺀다."""
    stem = label.lower().rstrip("s")
    words = {w.strip(".,;:'\"") for w in text.lower().split()}
    return {w for w in words if w and w not in STOPWORDS and stem not in w}


def fallback_phrase(node, sibling_nodes, stats):
    """VLM 소진 시에도 형제와 갈리는 문구. 실측만으로 만든다 (계획서 §B6: hard-fail 금지)."""
    label = str(node["label"])
    if not sibling_nodes:
        return f"the {label}"
    mine, other = stats.get(node["id"]), stats.get(sibling_nodes[0]["id"])
    if mine is None or other is None:
        return f"the {label}"
    both = mine["present"] & other["present"]
    if not both.any():
        return f"the {label}"
    ratio = float(np.median(mine["area"][both] / np.maximum(other["area"][both], 1e-9)))
    if ratio >= 1.5 or ratio <= 1 / 1.5:          # 크기 차가 뚜렷하면 그게 제일 눈에 띈다
        return f"the {'nearer' if ratio >= 1.5 else 'more distant'} {label}"
    left = float(np.mean(mine["cx"][both] < other["cx"][both]))
    return f"the {label} on the {'left' if left >= 0.5 else 'right'}"


def make_validate(label: str, siblings, prior_descs=()):
    """스키마 + 규칙 검사. `chat_json` 이 위반 목록을 붙여 재질의한다."""
    def validate(payload):
        violations = []
        text = payload.get("description")
        if not isinstance(text, str) or not text.strip():
            violations.append('"description" must be a non-empty string')
            return violations
        clean = " ".join(text.split())
        if not clean.lower().startswith("the "):
            violations.append('"description" must start with "the "')
        if len(clean.split()) > 14:
            violations.append(f'"description" must be at most 14 words (got {len(clean.split())})')
        # D120-b. 위치·크기만으로 된 문구를 거른다 — 구별은 되지만 캡션 target 으로는 후퇴다.
        if not (content_words(clean, label) - SPATIAL):
            violations.append(
                '"description" is built only from position and size words. Add what the '
                f'{label} actually looks like (colour, material, clothing, markings) or what '
                'it is doing, and keep the position only as a tie-breaker.')
        if clean.rstrip().endswith("."):
            violations.append('"description" must not end with a period')
        stem = label.lower().rstrip("s")
        if stem and stem not in clean.lower():
            violations.append(f'"description" must contain the base noun "{label}"')
        for banned in ("camera", "image", "frame", "photo", "video", "yellow box", "highlight"):
            if banned in clean.lower():
                violations.append(f'"description" must not mention "{banned}"')
        if siblings and clean.lower().strip() == f"the {label.lower()}":
            violations.append(f'"description" is just "the {label}" but siblings exist '
                              f'({", ".join(siblings)}) — it must distinguish this one')
        # D120-b. **길이가 아니라 내용**을 본다. `"the light-colored camel ..."` 는 예전 규칙을
        # 통과했지만 형제도 밝은 색이라 구별이 안 됐다. 형제 문구에 없는 내용어를 요구한다.
        for prior in prior_descs:
            fresh = content_words(clean, label) - content_words(prior, label)
            if not fresh:
                violations.append(
                    f'"description" says nothing that "{prior}" does not already say — it is '
                    f'interchangeable with the other {label}. Use the MEASURED FACTS '
                    f'(side of frame / relative size / which is nearer) instead.')
        if not isinstance(payload.get("confidence"), (int, float)):
            violations.append('"confidence" must be a number')
        return violations
    return validate


def describe_node(node, graph, seg, frames, client, args, out_dir, stats=None, prior_descs=()):
    track_ids = node_track_ids(node)
    picks = pick_frames(seg, track_ids, args.num_frames)
    stats = stats or {}
    sibling_nodes = [n for n in graph["nodes"] if n["id"] != node["id"]
                     and str(n["label"]).lower() == str(node["label"]).lower()]
    if not picks:
        return {"node_id": node["id"], "label": node["label"], "source": "no_mask",
                "description": fallback_phrase(node, sibling_nodes, stats)}

    label = str(node["label"])
    siblings = [n["id"] for n in sibling_nodes]
    others = sorted({str(n["label"]) for n in graph["nodes"] if n["id"] != node["id"]})
    moving = bool(node.get("moving")) or float(node.get("path_len_u") or 0.0) > 0.05

    image = build_image(frames, seg, track_ids, picks,
                        path.join(out_dir, "panels", f"{graph['video']}__{node['id']}.jpg"), args)
    sibling_note = ""
    if siblings:
        sibling_note = (f'.\nWARNING: this scene contains {len(siblings) + 1} objects called '
                        f'"{label}". Your phrase must pick out the highlighted one only.')
    prompt = USER.format(
        width=graph["width"], height=graph["height"], num_frames=graph["num_frames"],
        label=label, sibling_note=sibling_note, others=", ".join(others) or "none",
        frames=", ".join(str(f) for f, _ in picks),
        motion_note=("moves during the shot." if moving else "stays roughly in place."),
        disambiguation=disambiguation_block(node, sibling_nodes, stats, prior_descs))

    started = time.time()
    payload, info = client.chat_json(prompt, images=[image], system=SYSTEM,
                                     validate=make_validate(label, siblings, prior_descs),
                                     max_repairs=args.max_repairs,
                                     label=f"{graph['video']}/{node['id']}")
    seconds = time.time() - started
    if payload is None:      # 소진 — 실측 수식어로 떨어진다 (계획서 §B6: hard-fail 금지)
        return {"node_id": node["id"], "label": label, "source": "exhausted",
                "description": fallback_phrase(node, sibling_nodes, stats),
                "image": image, "seconds": round(seconds, 2)}
    return {"node_id": node["id"], "label": label, "source": "vlm",
            "description": " ".join(str(payload["description"]).split()),
            "attributes": payload.get("attributes", []),
            "action": payload.get("action", ""),
            "confidence": float(payload.get("confidence", 0.0)),
            "siblings": siblings, "frames": [f for f, _ in picks],
            "area_frac": [round(a, 5) for _, a in picks],
            "image": image, "seconds": round(seconds, 2),
            "repairs": info["repairs"], "turns": info["turns"]}


def bank_anchor_ids(out_root: str, video: str, bank_dir: str):
    """뱅크에 **실제로 등장한** anchor id. 없으면 None (= 전 노드)."""
    bank = path.join(out_root, video, bank_dir, "bank.json")
    if not bank_dir or not path.isfile(bank):
        return None
    with open(bank, encoding="utf-8") as file:
        return sorted({str(v["anchor_id"]) for v in json.load(file)["variants"]})


def main(args):
    sys.path.insert(0, VISTA4D_ROOT)
    from utils.media import load_video

    client = VLMClient(api_base=args.api_base, model=args.model, temperature=args.temperature)
    if not args.dry_run:
        served = [entry["id"] for entry in client.models().get("data", [])]
        print(f"{'served':<14}{served}", flush=True)

    rows = []
    for video in args.videos:
        with open(path.join(args.output_root, video, "scene_graph.json"), encoding="utf-8") as f:
            graph = json.load(f)
        dyn_folder, stat_folder = resolve_seg_folders(args.eval_data, video)
        segs = {"dyn": SegInstances(dyn_folder, "dyn") if dyn_folder else None,
                "stat": SegInstances(stat_folder, "stat") if stat_folder else None}
        wanted = None if args.all_nodes else bank_anchor_ids(args.output_root, video,
                                                             args.bank_dir)
        nodes = [n for n in graph["nodes"] if wanted is None or n["id"] in wanted]
        frames, _ = load_video(path.join(args.eval_data, "eval_data", "recon_and_seg",
                                         video, "video.mp4"))

        # D120-b. 통계는 **그래프 전 노드**에 대해 미리 잰다 — 형제가 뱅크 anchor 가 아닐 수도
        # 있는데(avocado `dyn_1`/`dyn_2`) 비교 대상으로는 필요하다.
        stats = {n["id"]: node_stats(n, segs[n["kind"]], graph)
                 for n in graph["nodes"] if segs.get(n["kind"]) is not None}
        # 같은 라벨끼리 **연달아** 처리해야 앞 형제의 문구를 뒤 형제에게 넘길 수 있다.
        nodes.sort(key=lambda n: (str(n["label"]).lower(), n["id"]))
        prior_by_label = {}

        records = {}
        for node in nodes:
            seg = segs.get(node["kind"])
            if seg is None:
                print(f"⚠ {video}/{node['id']}: {node['kind']} seg 가 없다 — 건너뜀", flush=True)
                continue
            if args.dry_run:
                picks = pick_frames(seg, node_track_ids(node), args.num_frames)
                build_image(frames, seg, node_track_ids(node), picks,
                            path.join(args.output_root, video, "panels",
                                      f"{video}__{node['id']}.jpg"), args)
                records[node["id"]] = {"node_id": node["id"], "label": node["label"],
                                       "source": "dry_run",
                                       "frames": [f for f, _ in picks]}
                continue
            key = str(node["label"]).lower()
            record = describe_node(node, graph, seg, frames, client, args,
                                   path.join(args.output_root, video), stats=stats,
                                   prior_descs=tuple(prior_by_label.get(key, ())))
            prior_by_label.setdefault(key, []).append(record["description"])
            records[node["id"]] = record
            rows.append((video, node["id"], node["label"], record))
            print(f"[{video:<16}] {node['id']:<7} {node['label']:<12} "
                  f"-> {record.get('description')}", flush=True)

        payload = {"format": "lbm_instance_desc_v1", "video": video,
                   "bank_dir": args.bank_dir if not args.all_nodes else None,
                   "model": args.model, "num_frames": args.num_frames,
                   "descriptions": records}
        target = path.join(args.output_root, video, args.out_name)
        if not args.dry_run:
            with open(target, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=1)
            print(f"wrote {target}", flush=True)

    if rows:
        header = (f"{'video':<18}{'node':<8}{'label':<12}{'conf':>5}{'rep':>4}{'sec':>6}"
                  f"  description")
        print("=" * min(len(header) + 40, 140))
        print(header)
        print("-" * min(len(header) + 40, 140))
        for video, node_id, label, record in rows:
            print(f"{video:<18}{node_id:<8}{label:<12}"
                  f"{record.get('confidence', 0.0):>5.2f}{record.get('repairs', 0):>4}"
                  f"{record.get('seconds', 0.0):>6.1f}  {record.get('description')}")
        print("=" * min(len(header) + 40, 140))
    if not args.dry_run:
        client.save_trace(path.join(args.output_root, "instance_desc_trace"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output_root", default=path.join(HERE, "out"))
    parser.add_argument("--eval_data", default=EVAL_DATA)
    # 기본은 뱅크에 등장한 anchor 만 — 호출 수가 영상당 1~6 회로 떨어진다.
    parser.add_argument("--bank_dir", default="hole_bank_k6_d99", type=str)
    parser.add_argument("--all_nodes", action="store_true", default=False)
    parser.add_argument("--no_all_nodes", dest="all_nodes", action="store_false")
    parser.add_argument("--out_name", default="instance_desc.json", type=str)
    # D169. 3 → 6. 3장으로도 행위는 나오지만(`action` 비어있음 9.0%, 421 노드 실측) 세 종류가
    # 섞인다 — 진짜 이동(walking 49) · 자세/상태(standing, smiling) · 무정보 수동태(being 77,
    # 18%). 또 빈 구간을 건너뛰는 `pick_frames` 특성상 32/421 은 3장도 못 받았다(0장 7 / 1장 12
    # / 2장 13). 6장이면 그 구간 손실이 줄고 방향·궤적이 갈릴 여지가 생긴다. 비용은 이미지 폭이
    # 2배(1344→2688 px) 되는 것뿐 — 호출 수는 그대로 노드당 1회다 (`build_image` 는 한 장으로 합침).
    # 참고로 DynamicVerse stage1 은 25장을 개별 이미지로 보낸다 (batch_process_qwen_pipeline.py:178).
    parser.add_argument("--num_frames", default=6, type=int)
    parser.add_argument("--tile", default=448, type=int)          # 패널 한 칸 픽셀
    parser.add_argument("--dim", default=0.28, type=float)        # 마스크 바깥 밝기 배율
    parser.add_argument("--crop_pad", default=0.35, type=float)   # 타이트 crop 여백 비율
    parser.add_argument("--api_base", default="http://127.0.0.1:22002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_repairs", type=int, default=3)
    # 이미지만 만들고 VLM 을 안 부른다 — 사람이 패널을 먼저 보고 승인하는 경로.
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--no_dry_run", dest="dry_run", action="store_false")
    main(parser.parse_args())
