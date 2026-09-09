"""같은 action 에 대해 **LBM arm** 과 **Lite arm** 의 프레이밍 지표를 나란히 놓는다.

왜 필요한가: 두 팔은 `audit_lite_framing.py` 를 **같은 코드·같은 눈금**으로 통과했지만 CSV 가
따로 나온다. 여기서 action 을 키로 붙여 차이를 낸다. 두 팔이 같은 자를 쓴다는 게 이 비교의 전부
이므로 (그러려고 LBM 카메라를 GT 렌더까지 다시 돌렸다) 값은 **가공하지 않고 그대로** 옮긴다.

`framing` = `occl_keep` = (안 가려진 픽셀) / (화면에 투영된 subject OBB 면적).
`crop_keep` 은 곱하지 않고 진단 열로 따로 둔다 (`audit_lite_framing.py` docstring).

출력: `<out>/lbm_vs_lite.csv` + 요약표.

예시:
    python scripts/compare_lbm_vs_lite.py \
        --lbm  out/trumans_lbm_bank/lite_framing.csv \
        --lite out/trumans_lite_bank/lite_framing.csv \
        --out  out/trumans_lbm_bank
"""
import csv
from argparse import ArgumentParser
from os import makedirs, path

COLUMNS = ["framing_mean", "framing_min", "crop_mean", "occl_mean",
           "offscreen_frames", "frames_below", "nearclip_frames", "obb_area_mean"]


def read_by_action(csv_path, recording=""):
    """`action` 을 키로 CSV 를 읽는다. **`recording` 을 반드시 걸러야 한다.**

    Lite 뱅크 CSV 는 recording 7편이 한 파일에 들어 있다. recording 을 안 거르면 같은 action 번호
    끼리 덮어써서 **다른 편의 clip 과 비교**하게 되는데, 두 열 다 그럴듯한 숫자라 조용히 지나간다
    (실측: action 0 이 00add26c 0.9636 대신 다른 편 0.9952 로 찍혔다).
    """
    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file)
                if not recording or row.get("recording", "") == recording]
    by_action = {}
    for row in rows:
        action = int(row["action"])
        assert action not in by_action, \
            (f"{csv_path} 에 recording={recording!r} action={action} 이 2행이다 "
             f"({by_action[action]['video']} vs {row['video']}) — 필터가 부족하다")
        by_action[action] = row
    return by_action


def main():
    parser = ArgumentParser()
    parser.add_argument("--lbm", required=True, type=str)      # LBM arm lite_framing.csv
    parser.add_argument("--lite", required=True, type=str)     # Lite arm lite_framing.csv
    parser.add_argument("--out", required=True, type=str)
    # 비울 경우 LBM CSV 의 `recording` 열에서 자동으로 집는다 (LBM arm 은 항상 1편이다).
    parser.add_argument("--recording", default="", type=str)
    args = parser.parse_args()

    recording = args.recording
    if not recording:
        with open(args.lbm, newline="", encoding="utf-8") as file:
            recs = sorted({row.get("recording", "") for row in csv.DictReader(file)})
        assert len(recs) == 1, f"LBM CSV 에 recording 이 여럿이다 {recs} — --recording 을 줄 것"
        recording = recs[0]
    lbm = read_by_action(args.lbm, recording)
    lite = read_by_action(args.lite, recording)
    #    LBM arm 은 8창뿐이다 (나머지 8창은 Cinematographer 품질 게이트에서 후보 0개, D7).
    #    교집합만 비교한다 — 없는 쪽을 0 으로 채우면 없는 게 나쁜 점수로 읽힌다.
    actions = sorted(set(lbm) & set(lite))
    assert actions, "두 CSV 에 공통 action 이 없다"

    makedirs(args.out, exist_ok=True)
    out_path = path.join(args.out, "lbm_vs_lite.csv")
    fields = (["action", "lbm_video", "lite_video", "lite_preset", "lite_radius"]
              + [f"lbm_{c}" for c in COLUMNS] + [f"lite_{c}" for c in COLUMNS]
              + ["d_framing_mean", "d_framing_min"])
    rows = []
    for action in actions:
        a, b = lbm[action], lite[action]
        row = {"action": action, "lbm_video": a["video"], "lite_video": b["video"],
               "lite_preset": b.get("preset", ""), "lite_radius": b.get("radius", "")}
        row.update({f"lbm_{c}": a.get(c, "") for c in COLUMNS})
        row.update({f"lite_{c}": b.get(c, "") for c in COLUMNS})
        row["d_framing_mean"] = f"{float(a['framing_mean']) - float(b['framing_mean']):+.4f}"
        row["d_framing_min"] = f"{float(a['framing_min']) - float(b['framing_min']):+.4f}"
        rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"{'recording':<12}{recording}")
    print(f"{'actions':<12}{len(rows)}  {[r['action'] for r in rows]}")
    print()
    head = (f"{'act':>4} {'framing(LBM)':>13} {'framing(Lite)':>14} {'Δ':>9} "
            f"{'min(LBM)':>9} {'min(Lite)':>10} {'crop(LBM)':>10} {'crop(Lite)':>11} "
            f"{'below(L/l)':>11} {'nearclip(L/l)':>14}")
    print(head)
    print("-" * len(head))
    for row in rows:
        print(f"{row['action']:>4} {float(row['lbm_framing_mean']):>13.4f} "
              f"{float(row['lite_framing_mean']):>14.4f} {row['d_framing_mean']:>9} "
              f"{float(row['lbm_framing_min']):>9.4f} {float(row['lite_framing_min']):>10.4f} "
              f"{float(row['lbm_crop_mean']):>10.4f} {float(row['lite_crop_mean']):>11.4f} "
              f"{row['lbm_frames_below'] + '/' + row['lite_frames_below']:>11} "
              f"{row['lbm_nearclip_frames'] + '/' + row['lite_nearclip_frames']:>14}")
    print("-" * len(head))
    mean_lbm = sum(float(r["lbm_framing_mean"]) for r in rows) / len(rows)
    mean_lite = sum(float(r["lite_framing_mean"]) for r in rows) / len(rows)
    print(f"{'mean':>4} {mean_lbm:>13.4f} {mean_lite:>14.4f} {mean_lbm - mean_lite:>+9.4f}")
    print()
    print(f"{'out':<12}{out_path}")


if __name__ == "__main__":
    main()
