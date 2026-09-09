"""원본 LBM 실행의 **shot 별 final preview** 를 PASS/BLOCK 라벨과 함께 영상 1편으로 잇는다.

왜 영상인가: LBM 은 카메라를 16대 만들었지만 자기 VLM 게이트로 10대를 떨어뜨렸다. 그 판정이
타당했는지는 **떨어진 프레임을 눈으로 봐야** 안다 (대부분 "프레임이 새하얗다" = 카메라가 벽
바깥). contact sheet 한 장으로 16개를 늘어놓으면 타일이 240px 로 줄어 그 판단이 안 된다.
그래서 shot 당 `hold_sec` 씩 물려 재생하는 릴로 만든다.

`camera_shot_report_v1.json` 의 `final_preview_path` + `downstream_eligible` 을 읽고,
`camera_packages/scene_<s>_shot_<n>/*.json` 에서 차단 사유(`hard_block_camera_reason`)를 가져온다.

영상은 `imageio` + `libx264` (cv2 mp4v 는 VS Code 뷰어에서 안 열린다).

사용 예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    $PY scripts/lbm_preview_reel.py \
        --run_dir <LBM>/Cinematographer/output/trumans_00add26c \
        --out /tmp/lbm_previews.mp4
"""
import json
from argparse import ArgumentParser
from glob import glob
from os import path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    """DejaVu 가 없으면 기본 비트맵 폰트로 (크기 지정은 못 하지만 죽지는 않는다)."""
    for candidate in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                      "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def block_reasons(run_dir: str):
    """shot_id -> 차단 사유. 통과한 shot 은 문자열 'None' 이 들어 있어 걸러낸다."""
    reasons = {}
    for pkg in glob(path.join(run_dir, "camera_packages", "*", "*.json")):
        cam = json.load(open(pkg, encoding="utf-8"))
        cam = cam.get("camera", cam)
        review = cam.get("final_preview_llm_review") or {}
        reason = review.get("hard_block_camera_reason")
        if reason and str(reason) != "None":
            reasons[str(cam.get("shot_id"))] = str(reason)
    return reasons


def annotate(img: Image.Image, lines, colour, width: int, height: int):
    """상단 배너 + 하단 사유 텍스트. 원본 프레임은 letterbox 로 보존한다."""
    canvas = Image.new("RGB", (width, height), (16, 16, 16))
    scale = min(width / img.width, (height - 150) / img.height)
    resized = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    canvas.paste(resized, ((width - resized.width) // 2, 70 + (height - 150 - resized.height) // 2))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width, 62], fill=colour)
    draw.text((16, 14), lines[0], font=load_font(34), fill=(0, 0, 0))
    body = load_font(19)
    for i, line in enumerate(lines[1:]):
        draw.text((16, height - 74 + i * 24), line, font=body, fill=(220, 220, 220))
    return np.asarray(canvas)


def wrap(text: str, limit: int, max_lines: int):
    """단어 단위 줄바꿈. 사유 문장이 길어 화면 밖으로 나가는 걸 막는다."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        if len(cur) + len(word) + 1 > limit:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    lines.append(cur)
    return lines[:max_lines]


def main(args):
    report = json.load(open(path.join(args.run_dir, "outputs",
                                      "camera_shot_report_v1.json"), encoding="utf-8"))
    reasons = block_reasons(args.run_dir)
    rows = sorted(report["rows"], key=lambda r: int(r["shot_id"]))

    frames, table = [], []
    for row in rows:
        preview = row.get("final_preview_path") or row.get("preview_frame_path")
        if not preview or not path.exists(preview):
            continue
        ok = bool(row["downstream_eligible"])
        shot = str(row["shot_id"])
        head = (f"shot {shot:>2}  {row['camera_name']}   "
                f"{'PASS -> rendered' if ok else 'BLOCKED by LBM VLM gate'}")
        body = [f"movement {row['movement_tag']}   selection {row['selection_source'] or 'seed'}"]
        body += wrap(reasons.get(shot, ""), 116, 2) if not ok else []
        img = annotate(Image.open(preview).convert("RGB"), [head] + body,
                       (90, 200, 110) if ok else (235, 120, 110), args.width, args.height)
        frames += [img] * int(round(args.hold_sec * args.fps))
        table.append((shot, row["camera_name"], "PASS" if ok else "BLOCK"))

    imageio.mimwrite(args.out, frames, fps=args.fps, codec="libx264",
                     quality=6, macro_block_size=1)

    print(f"\n{'shot':>4} {'camera':>26} {'verdict':>8}")
    for shot, name, verdict in table:
        print(f"{shot:>4} {name:>26} {verdict:>8}")
    n_pass = sum(1 for _, _, v in table if v == "PASS")
    print(f"\n{len(table)} shots  PASS {n_pass}  BLOCK {len(table) - n_pass}")
    print(f"-> {args.out}  ({len(frames) / args.fps:.1f}s)")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)     # Cinematographer/output/<run_id>
    parser.add_argument("--out", default="/tmp/lbm_previews.mp4", type=str)
    parser.add_argument("--hold_sec", default=1.6, type=float)    # shot 당 화면에 머무는 시간
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--height", default=800, type=int)
    main(parser.parse_args())
