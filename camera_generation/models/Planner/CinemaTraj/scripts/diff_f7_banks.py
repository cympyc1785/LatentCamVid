"""F7 시간축 절단 on/off 두 뱅크를 행 단위로 대조한다.

왜 필요한가: `--time_truncate` 는 게이트에 걸린 변이의 궤적을 **바꾼다**. 바뀐 게 정확히
어느 변이인지(그리고 안 바뀐 변이가 정말 비트 단위로 같은지)를 눈으로 확인하지 않으면
"켰더니 뭔가 달라졌다" 이상을 말할 수 없다. 두 가지를 동시에 검사한다:

  ① **회귀 검사** — `hold_from` 이 비었거나 `num_frames` 인 행은 off 뱅크와 **공유 열 전부가
     문자 동일**해야 한다. 하나라도 다르면 F7 이 절단 안 한 변이까지 건드린 것이다.
  ② **변경 목록** — `hold_from < num_frames` 인 행만 뽑아 off 대비 `status`/`binding`/
     `path_len_u`/`tau_max` 가 어떻게 달라졌는지 표로 낸다. 영상으로 보여줄 후보를 여기서 고른다.

사용 예시:
  python scripts/diff_f7_banks.py --video parkour \
      --off_bank hole_bank_f7_off --on_bank hole_bank_f7_on --top 8
"""
import csv
from argparse import ArgumentParser
from os import path

ROOT = path.dirname(path.dirname(path.abspath(__file__)))


def read_bank(folder):
    with open(path.join(folder, "bank.csv"), newline="", encoding="utf-8") as file:
        return {row["variant_id"]: row for row in csv.DictReader(file)}


def main(args):
    off_dir = path.join(args.output_root, args.video, args.off_bank)
    on_dir = path.join(args.output_root, args.video, args.on_bank)
    off, on = read_bank(off_dir), read_bank(on_dir)

    # 공유 열만 본다 — on 뱅크에만 `hold_from` 이 채워져 있다.
    shared = [c for c in off[next(iter(off))].keys() if c in on[next(iter(on))]]
    shared = [c for c in shared if c != "hold_from"]

    only_off = sorted(set(off) - set(on))
    only_on = sorted(set(on) - set(off))
    truncated, regressions, changed = [], [], []
    for vid in sorted(set(off) & set(on)):
        a, b = off[vid], on[vid]
        hold = b.get("hold_from", "")
        is_trunc = hold not in ("", None) and int(hold) < args.num_frames
        diffs = [c for c in shared if a[c] != b[c]]
        if is_trunc:
            truncated.append((vid, int(hold), a, b, diffs))
        elif diffs:
            regressions.append((vid, diffs, a, b))
        if diffs:
            changed.append(vid)

    print(f"video {args.video}   off {args.off_bank} ({len(off)}행)   "
          f"on {args.on_bank} ({len(on)}행)")
    print(f"off 에만 있는 변이 {len(only_off)}   on 에만 있는 변이 {len(only_on)}")
    print(f"절단된 변이 {len(truncated)}   절단 없이 값이 달라진 변이(회귀) {len(regressions)}")

    if only_off[:10]:
        print("\noff 에만:", " ".join(only_off[:10]), "..." if len(only_off) > 10 else "")
    if only_on[:10]:
        print("on 에만: ", " ".join(only_on[:10]), "..." if len(only_on) > 10 else "")

    if regressions:
        print("\n!! 회귀 — 절단 안 한 변이인데 값이 다르다 (있으면 안 된다)")
        for vid, diffs, a, b in regressions[:20]:
            print(f"  {vid}")
            for c in diffs[:6]:
                print(f"    {c:<22} off={a[c]:<14} on={b[c]}")

    if truncated:
        print(f"\n절단된 변이 {len(truncated)}개 — hold_from 이른 순")
        print(f"{'variant_id':<44}{'hold':>5}{'status(off→on)':>34}"
              f"{'path_len_u':>22}{'tau_max':>18}")
        print("-" * 124)
        for vid, hold, a, b, _ in sorted(truncated, key=lambda r: r[1])[:args.top]:
            st = f"{a.get('status','')}→{b.get('status','')}"
            pl = f"{a.get('path_len_u','')}→{b.get('path_len_u','')}"
            tm = f"{a.get('tau_max','')}→{b.get('tau_max','')}"
            print(f"{vid:<44}{hold:>5}{st:>34}{pl:>22}{tm:>18}")

        print("\n영상 후보 (hold_from 이 이른 순 = 가장 많이 잘린 것):")
        print("  " + " ".join(v for v, *_ in sorted(truncated, key=lambda r: r[1])[:args.top]))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", default="parkour")                   # 영상 이름
    parser.add_argument("--output_root", default=path.join(ROOT, "out"))  # 뱅크 상위 폴더
    parser.add_argument("--off_bank", default="hole_bank_f7_off")       # 절단 끈 뱅크
    parser.add_argument("--on_bank", default="hole_bank_f7_on")         # 절단 켠 뱅크
    parser.add_argument("--num_frames", default=49, type=int)           # 플랜 프레임 수
    parser.add_argument("--top", default=10, type=int)                  # 표에 찍을 행 수
    main(parser.parse_args())
