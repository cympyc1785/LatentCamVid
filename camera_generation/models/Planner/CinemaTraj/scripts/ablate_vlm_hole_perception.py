"""VLM 이 렌더 구멍(magenta)을 **그림으로** 보는지, 아니면 텍스트의 `coverage` 숫자만 읽는지 가른다.

왜 필요한가: `trace/turn_00.json` 에서 VLM 은 "avoids the large magenta regions seen in
'truck_left', 'truck_right', and 'pedestal_up'" 이라고 답했다. 그런데 **같은 프롬프트 텍스트에
preset 마다 `coverage_min`/`coverage_end` 숫자가 이미 들어 있다.** 그러면 "magenta 를 봤다" 와
"0.68/0.60/0.69 를 읽고 magenta 라는 단어를 지어냈다" 가 구분되지 않는다. 이 구분은 설계에
직접 영향을 준다 — 숫자만 읽는 것이라면 board 렌더링은 토큰만 먹는 장식이고, look-before-move
의 전제("VLM 이 렌더를 보고 고른다")가 이 지점에서는 성립하지 않는다.

**두 턴 모두** 잰다 (`--turn`). 1차 실험은 `traj` 턴만 봤는데, board 렌더가 27칸이나 되는 쪽은
후보 `select` 턴이다. 거기서도 그림을 안 본다면 board 를 크게 그리는 것 자체가 낭비다.

조건 (turn 마다 해당되는 것만 만든다):

    A_full        원본 그대로 (그림 + 숫자)                        — trace 재현 대조군
    B_no_num      그림만 (coverage / occlusion_pass 숫자 제거)
    C_no_img      숫자만 (이미지 첨부 안 함)
    D_conflict    숫자만 뒤집는다 (x -> 1-x). 그림은 그대로       — 숫자 채널 개입
    E_paint       숫자 그대로, **A_full 이 고른 타일에 가짜 magenta 를 칠한다** — 그림 채널 개입
    F_paint_nonum E_paint + 숫자 제거. 그림 말고는 근거가 없다     — 결정적 검사
    G_shuffle     후보/preset 행 순서만 섞는다 (그림 그대로)       — 위치 prior 검사
    H_neutral     preset 이름을 M01..M13 으로 바꾼다 (설명은 유지) — 어휘 prior 검사, traj 전용
    I_temp10      A_full 을 temperature 1.0 으로                    — prior 가 온도로 흔들리나

읽는 법:
  · B 가 A 와 같은 선택을 유지하면 → 그림만으로도 된다. C 가 A 와 같으면 → 숫자만으로도 된다.
    **둘 다 같으면 채널이 중복이고, 그때 답을 주는 건 D/E/F 뿐이다.**
  · D 에서 선택이 안 움직이고 F 에서 움직이면 → 그림을 본다.
  · F 에서도 칠해진 타일을 계속 고르면 → 그림을 안 본다. board 는 장식이다.
  · G/H/I 에서도 안 움직이면 그건 지각이 아니라 **prior** 다 (조건과 무관한 고정 답).

**1회 draw 로는 정합/불일치를 판정하지 않는다** (`--draws 3`). 같은 조건에서 답이 흔들리면
그 자체가 결과다.

사용 예시:
    python scripts/ablate_vlm_hole_perception.py --turn both --draws 3
    python scripts/ablate_vlm_hole_perception.py --turn select --videos camel --draws 5
"""
import json
import random
import re
import sys
from argparse import ArgumentParser
from os import makedirs, path

import imageio.v3 as imageio
import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))
sys.path.insert(0, HERE)

from lbm.overlay import HOLE_COLOR  # noqa: E402
from lbm.vlm import VLMClient, parse_json_candidates  # noqa: E402

# ---------------------------------------------------------------- 텍스트 개입 (traj 턴)
# preset 행에서 coverage 두 칸만 지운다. move/rotation/tau_max 는 남긴다 — 구멍과 무관한
# 채널까지 지우면 "숫자를 지웠더니 못 골랐다" 가 coverage 때문인지 알 수 없다.
COVERAGE_COLUMNS = re.compile(r"\s*\|\s*coverage_min [0-9.]+\s*\|\s*coverage_end [0-9.]+")
EXCLUDED_REASON = re.compile(r"\s*coverage drops to [0-9.]+ \(floor [0-9.]+\)")
LEGEND_COVERAGE = re.compile(r";?\s*coverage_end = [^)]*")
TRAJ_ROW = re.compile(r"^([a-z_0-9]+)\s+move .* coverage_min ([0-9.]+)", re.M)

# ---------------------------------------------------------------- 텍스트 개입 (select 턴)
# 후보 행: `A1  d_az -8  d_elev +10  dist 1.2x  coverage 0.92  subject_area 0.12  ...`
SELECT_ROW = re.compile(r"^([A-Z]\d+)\s+d_az .* coverage ([0-9.]+)", re.M)
SELECT_COVERAGE = re.compile(r"\s\scoverage ([0-9.]+)")
SELECT_OCCLUSION = re.compile(r"\s\socclusion_pass [0-9.]+")

WITHHELD_NOTE = ("\n\n## NOTE\nCoverage numbers are withheld this time. Judge missing-data "
                 "(magenta) purely from the rendered board images.")


def redact_numbers(prompt: str, turn: str):
    """coverage 숫자를 지운다. 지운 자리는 비워두지 않고 흔적을 남긴다 — 프롬프트에서 열이
    통째로 사라지면 모델이 '원래 없는 정보' 로 보지만, 우리는 '가려진 정보' 로 알려야
    ①숫자를 못 봤다 ②숫자가 없다고 착각했다 를 구분할 수 있다."""
    if turn == "traj":
        text = COVERAGE_COLUMNS.sub("", prompt)
        text = EXCLUDED_REASON.sub(" (excluded by the coverage gate)", text)
        text = LEGEND_COVERAGE.sub("", text)
    else:
        text = SELECT_COVERAGE.sub("", prompt)
        text = SELECT_OCCLUSION.sub("", text)
    return text + WITHHELD_NOTE


def conflict_numbers(prompt: str, turn: str):
    """coverage 숫자만 뒤집는다 (x -> 1-x). 그림은 그대로다. 그림과 숫자가 **서로 반대**를
    가리키므로 어느 채널을 따르는지 한 번에 드러난다."""
    if turn == "traj":
        def flip(match):
            lo, hi = float(match.group(1)), float(match.group(2))
            return f" | coverage_min {1 - lo:.2f} | coverage_end {1 - hi:.2f}"
        return re.sub(r"\s*\|\s*coverage_min ([0-9.]+)\s*\|\s*coverage_end ([0-9.]+)",
                      flip, prompt)
    return SELECT_COVERAGE.sub(lambda m: f"  coverage {1 - float(m.group(1)):.2f}", prompt)


def shuffle_rows(prompt: str, turn: str, seed: int = 0):
    """후보/preset **행 순서만** 섞는다. 라벨과 숫자는 그대로라 정보량이 같다.
    답이 바뀌면 모델이 내용이 아니라 위치(첫 줄/마지막 줄)를 보고 있었다는 뜻."""
    pattern = TRAJ_ROW if turn == "traj" else SELECT_ROW
    lines = prompt.splitlines()
    hits = [i for i, line in enumerate(lines) if pattern.match(line)]
    if len(hits) < 2:
        return prompt
    block = [lines[i] for i in hits]
    random.Random(seed).shuffle(block)
    for slot, line in zip(hits, block):
        lines[slot] = line
    return "\n".join(lines)


def neutralize_presets(prompt: str, system: str):
    """preset 이름을 `M01`.. 로 바꾼다. **설명 문장은 그대로 둔다** — 의미는 남기고 단어만
    지운다. 'orbit 은 영화적이다' 같은 어휘 prior 와 '도는 게 낫다' 는 의미 판단을 가른다.
    (그림 타일에는 원래 이름이 찍혀 있으므로 이 조건은 이미지 없이 돌린다.)"""
    names = [m.group(1) for m in TRAJ_ROW.finditer(prompt)]
    names += [n for n in re.findall(r"^([a-z_]+)\s+coverage drops", prompt, re.M)
              if n not in names]
    mapping = {name: f"M{index + 1:02d}" for index, name in enumerate(names)}
    text, sys_text = prompt, system
    for name in sorted(mapping, key=len, reverse=True):   # static_hold ⊂ static_hold_dont_look
        token = re.compile(rf"\b{re.escape(name)}\b")
        text = token.sub(mapping[name], text)
        sys_text = token.sub(mapping[name], sys_text)
    return text, sys_text, mapping


# ---------------------------------------------------------------- 그림 개입
def paint_select_tiles(board_png: str, labels, out_png: str, columns: int, count: int,
                       gap: int = 4, frac: float = 0.60):
    """후보 board 의 지정 타일 중앙 `frac` 을 magenta 로 칠한다. 텍스트 숫자는 안 건드린다.
    `contact_sheet` 기하(gap 4px, 균등 타일)를 그대로 역산한다 (`lbm/overlay.py:85`)."""
    sheet = imageio.imread(board_png)
    rows = (count + columns - 1) // columns
    tile_h = (sheet.shape[0] - (rows - 1) * gap) // rows
    tile_w = (sheet.shape[1] - (columns - 1) * gap) // columns
    for label in labels:
        row, col = ord(label[0]) - ord("A"), int(label[1:]) - 1
        y0, x0 = row * (tile_h + gap), col * (tile_w + gap)
        dy, dx = int(tile_h * (1 - frac) / 2), int(tile_w * (1 - frac) / 2)
        sheet[y0 + dy:y0 + tile_h - dy, x0 + dx:x0 + tile_w - dx] = HOLE_COLOR
    makedirs(path.dirname(out_png), exist_ok=True)
    imageio.imwrite(out_png, sheet)
    return out_png


def paint_preset_rows(board_png: str, presets, order, out_png: str, preset_columns: int = 3,
                      frames: int = 3, gap: int = 4, frac: float = 0.60):
    """preset board 에서 지정 preset 의 **중간·마지막 프레임**을 magenta 로 칠한다.
    첫 프레임을 남기는 이유: system 프롬프트가 판정하라고 시킨 실패 양상이 정확히 "첫 프레임은
    멀쩡한데 마지막이 magenta" 이기 때문이다 (`lbm/prompts/system_traj.md`).

    기하: 각 preset 블록 = `contact_sheet(3 tiles, columns=3)` (loop.py:467) → 바깥
    `contact_sheet(rows, columns=preset_columns)` (:470)."""
    sheet = imageio.imread(board_png)
    block_rows = (len(order) + preset_columns - 1) // preset_columns
    block_h = (sheet.shape[0] - (block_rows - 1) * gap) // block_rows
    block_w = (sheet.shape[1] - (preset_columns - 1) * gap) // preset_columns
    tile_w = (block_w - (frames - 1) * gap) // frames
    for name in presets:
        if name not in order:
            continue
        index = order.index(name)
        by, bx = (index // preset_columns) * (block_h + gap), \
                 (index % preset_columns) * (block_w + gap)
        for frame in range(1, frames):                       # 중간·마지막만
            x0 = bx + frame * (tile_w + gap)
            dy, dx = int(block_h * (1 - frac) / 2), int(tile_w * (1 - frac) / 2)
            sheet[by + dy:by + block_h - dy, x0 + dx:x0 + tile_w - dx] = HOLE_COLOR
    makedirs(path.dirname(out_png), exist_ok=True)
    imageio.imwrite(out_png, sheet)
    return out_png


# ---------------------------------------------------------------- 채점용 정답지 / 실행
def coverage_truth(prompt: str, turn: str):
    """원본 프롬프트에서 (preset|label) -> coverage 표를 뽑는다."""
    pattern = TRAJ_ROW if turn == "traj" else SELECT_ROW
    return {m.group(1): float(m.group(2)) for m in pattern.finditer(prompt)}


def pick_of(parsed: dict, turn: str):
    """turn 마다 다른 키에서 '고른 것' 하나를 뽑는다."""
    if turn == "traj":
        return parsed.get("preset")
    picks = parsed.get("picks")
    return picks[0] if isinstance(picks, list) and picks else None


def run(client: VLMClient, prompt: str, system: str, images, draws: int, label: str, turn: str):
    results = []
    for draw in range(draws):
        raw, meta = client.chat(prompt, images=images, system=system, label=f"{label}#{draw}")
        parsed, error = parse_json_candidates(raw)
        parsed = parsed or {}
        # 파싱 실패도 결과다 — 프로덕션 `chat_json` 은 재질의로 덮지만 여기서는 1샷 그대로 잰다.
        results.append({"draw": draw, "pick": pick_of(parsed, turn),
                        "picks": parsed.get("picks"),
                        "confidence": parsed.get("confidence"),
                        "parse_error": error,
                        "finish_reason": meta.get("finish_reason"),
                        "raw_tail": raw[-200:] if error else "",
                        "prompt_tokens": meta.get("prompt_tokens"),
                        "completion_tokens": meta.get("completion_tokens"),
                        "reasoning": parsed.get("reasoning", ""),
                        "observation": parsed.get("observation", "")})
    return results


def load_turn(root: str, video: str, turn: str):
    """(prompt, system, images, extra) — traj 는 trace 에서, select 는 board 에서 재구성한다.
    select 턴은 trace 를 안 남기므로 (loop 이 `chat_json` 을 쓰고 turn_00 만 저장) 같은 입력을
    `contract.txt` + `system_select.md` + board 이미지 2장으로 다시 만든다."""
    out = path.join(root, "out", video)
    if turn == "traj":
        trace = path.join(out, "trace", "turn_00.json")
        if not path.exists(trace):
            return None
        blob = json.load(open(trace, encoding="utf-8"))
        board = json.load(open(path.join(out, "board", "board.json"), encoding="utf-8"))
        return blob["prompt"], blob["system"], blob["images"], {"board": board}
    contract = path.join(out, "board", "contract.txt")
    if not path.exists(contract):
        return None
    prompt = open(contract, encoding="utf-8").read()
    system = open(path.join(root, "lbm", "prompts", "system_select.md"), encoding="utf-8").read()
    images = [path.join(out, "board", "board_candidates.png"),
              path.join(out, "board", "source_frames.png")]
    board = json.load(open(path.join(out, "board", "board.json"), encoding="utf-8"))
    return prompt, system, images, {"board": board}


def build_conditions(turn: str, prompt: str, system: str, images, args, video, paint_targets):
    """(name -> (prompt, system, images, temperature)). `paint_targets` 는 A_full 이 실제로
    고른 것들 — 미리 정한 타일이 아니라 **모델 자신의 답**에 칠해야 개입이 최대가 된다."""
    conditions = {
        "A_full": (prompt, system, images, args.temperature),
        "B_no_num": (redact_numbers(prompt, turn), system, images, args.temperature),
        "C_no_img": (prompt, system, [], args.temperature),
        "D_conflict": (conflict_numbers(prompt, turn), system, images, args.temperature),
        "G_shuffle": (shuffle_rows(prompt, turn), system, images, args.temperature),
        "I_temp10": (prompt, system, images, 1.0),
    }
    if turn == "traj":
        neutral_prompt, neutral_system, mapping = neutralize_presets(prompt, system)
        conditions["H_neutral"] = (neutral_prompt, neutral_system, [], args.temperature)
        conditions["H_neutral"] += (mapping,)
    if paint_targets:
        painted = path.join(args.output, "painted", f"{video}_{turn}.png")
        if turn == "select":
            board_path = path.join(args.root, "out", video, "board", "board.json")
            count = len(json.load(open(board_path, encoding="utf-8"))["candidates"])
            paint_select_tiles(images[0], paint_targets, painted, args.board_columns, count)
        else:
            order = [m.group(1) for m in TRAJ_ROW.finditer(prompt)]
            paint_preset_rows(images[0], paint_targets, order, painted, args.preset_columns)
        painted_images = [painted] + list(images[1:])
        conditions["E_paint"] = (prompt, system, painted_images, args.temperature)
        conditions["F_paint_nonum"] = (redact_numbers(prompt, turn), system, painted_images,
                                       args.temperature)

        # J/K = **위치 대조군**. 같은 넓이를 *꼴찌* 타일에 칠한다. 개입의 양(픽셀 수)은 같고
        # 위치만 다르다. E/F 가 움직였는데 J/K 도 같이 움직이면 그건 "그림이 바뀌면 답이
        # 흔들린다" 는 것이지 "칠한 그 타일을 봤다" 가 아니다. G_shuffle 만으로는 부족한 게
        # 이것 — 9칸 board 에서 G_shuffle 도 답을 움직였기 때문이다.
        rows = [m.group(1) for m in (TRAJ_ROW if turn == "traj" else SELECT_ROW).finditer(prompt)]
        control = [r for r in rows if r not in paint_targets][-len(paint_targets):]
        if control:
            low = path.join(args.output, "painted", f"{video}_{turn}_control.png")
            if turn == "select":
                paint_select_tiles(images[0], control, low, args.board_columns, len(rows))
            else:
                paint_preset_rows(images[0], control, rows, low, args.preset_columns)
            low_images = [low] + list(images[1:])
            conditions["J_paint_low"] = (prompt, system, low_images, args.temperature)
            conditions["K_paint_low_nonum"] = (redact_numbers(prompt, turn), system, low_images,
                                               args.temperature)
            conditions["_control_labels"] = control
    return conditions


ORDER = ["A_full", "B_no_num", "C_no_img", "D_conflict", "E_paint", "F_paint_nonum",
         "G_shuffle", "H_neutral", "I_temp10", "J_paint_low", "K_paint_low_nonum"]


def main(args):
    makedirs(args.output, exist_ok=True)
    turns = ["traj", "select"] if args.turn == "both" else [args.turn]
    report = {"format": "vlm_hole_ablation_v2", "model": args.model, "draws": args.draws,
              "base_temperature": args.temperature, "turns": {}}

    for turn in turns:
        report["turns"][turn] = {}
        for video in args.videos:
            loaded = load_turn(args.root, video, turn)
            if loaded is None:
                print(f"!! {video}/{turn}: 입력 없음 — 5단계 loop 을 먼저 돌릴 것")
                continue
            prompt, system, images, _extra = loaded
            truth = coverage_truth(prompt, turn)

            # A_full 을 먼저 돌려 칠할 대상을 정한다 (모델 자신의 답에 개입해야 최대 효과).
            client = VLMClient(api_base=args.api_base, model=args.model,
                               temperature=args.temperature)
            first = run(client, prompt, system, images, args.draws, f"{video}/{turn}/A_full", turn)
            targets = sorted({r["pick"] for r in first if r["pick"] in truth})

            conditions = build_conditions(turn, prompt, system, images, args, video, targets)
            block = {"coverage_truth": truth, "paint_targets": targets,
                     "control_labels": conditions.pop("_control_labels", []),
                     "conditions": {"A_full": first}}
            for name in ORDER:
                if name not in conditions or name == "A_full":
                    continue
                text, sys_text, imgs, temperature = conditions[name][:4]
                mapping = conditions[name][4] if len(conditions[name]) > 4 else None
                client = VLMClient(api_base=args.api_base, model=args.model,
                                   temperature=temperature)
                got = run(client, text, sys_text, imgs, args.draws,
                          f"{video}/{turn}/{name}", turn)
                if mapping:                                  # M04 -> orbit_left_arc 로 되돌린다
                    back = {v: k for k, v in mapping.items()}
                    for record in got:
                        record["pick_raw"] = record["pick"]
                        record["pick"] = back.get(record["pick"], record["pick"])
                    block.setdefault("neutral_mapping", mapping)
                block["conditions"][name] = got
            report["turns"][turn][video] = block

    out = path.join(args.output, f"hole_ablation_{args.turn}.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"\n{'turn':8s}{'video':16s}{'condition':15s}{'picks (draw 0..n)':56s}"
          f"{'coverage of picks':>20s}")
    for turn, videos in report["turns"].items():
        for video, block in videos.items():
            truth = block["coverage_truth"]
            print(f"{'':8s}{'':16s}{'(painted E/F)':15s}{','.join(block['paint_targets']):56s}")
            print(f"{'':8s}{'':16s}{'(painted J/K)':15s}"
                  f"{','.join(block.get('control_labels', [])):56s}")
            for name, got in block["conditions"].items():
                picks = [str(r["pick"]) for r in got]
                covs = [f"{truth[p]:.2f}" if p in truth else "?" for p in picks]
                print(f"{turn:8s}{video:16s}{name:15s}{','.join(picks):56s}"
                      f"{','.join(covs):>20s}")
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", default=HERE, type=str)
    parser.add_argument("--videos", nargs="*", default=["camel", "avocado-slice"])
    # traj = 궤적 preset 선택 턴, select = 시작 pose 후보 선택 턴 (board 렌더가 큰 쪽)
    parser.add_argument("--turn", default="both", choices=["traj", "select", "both"])
    parser.add_argument("--output", default=path.join(HERE, "out", "ablation"), type=str)
    # 같은 조건 반복 횟수. 1회로는 정합인지 잡음인지 못 가른다.
    parser.add_argument("--draws", default=3, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    # board 기하 — 그림에 가짜 magenta 를 칠할 위치를 역산하는 데만 쓴다
    parser.add_argument("--board_columns", default=9, type=int)
    parser.add_argument("--preset_columns", default=3, type=int)
    parser.add_argument("--api_base", default="http://127.0.0.1:22002/v1", type=str)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-30B-A3B-Instruct", type=str)
    main(parser.parse_args())
