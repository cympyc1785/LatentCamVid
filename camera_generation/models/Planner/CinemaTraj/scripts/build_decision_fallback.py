"""VLM 없이 `decision.json` 을 만든다 — §B6 의 결정론적 fallback 그대로.

왜 별도 스크립트인가: **파이프라인은 절대 hard-fail 하지 않는다**가 §B6 의 계약이다. VLM 이
안 붙었을 때(지금)와 붙었는데 3회 재질의까지 실패했을 때가 같은 코드를 타야, 나중에 vlm.py 를
넣어도 하류(decode/emit/verify)가 그대로 돈다. `lbm/loop.py` 가 이 함수를 import 해서 쓴다.

규칙:
    후보    **τ headroom 을 남긴 것들 중** board 1위
            (`rank()` 의 `coverage x 크기적정도 x occlusion_pass` 최대)
    preset  orbit_left, sweep = min(45, 0.7 x obs_az_span)
    tracking  drift (동적 subject 기본)
    source  "fallback"  ← `decision.source` 로 남는다. 이게 있으면 결과에 큰 경고를 붙일 것

**τ 는 시작 pose 와 궤적이 나눠 쓰는 하나의 예산이다.** `tau = |p_plan(f) − p_src(f)| / z_med` 에
시작 offset 과 움직임이 같이 들어가므로, board 1위를 그냥 집으면 카메라가 서 버린다 — camel A1 은
`tau_start 0.2595` 라 `target_tau 0.20` 을 이미 넘겨서 `fit_tau` 가 scale 0 을 골랐다(실측).
그래서 여기서 `tau <= start_tau_frac x target_tau` 인 후보로 먼저 거른다 (기본 0.5 = 반반).
남는 게 없으면 τ 최소 후보를 집고 경고한다. 게이트의 `max_tau`(0.30)는 "후보 자체가 갈 수 있는
자리인가"를 보는 값이라 그대로 두고, 예산 배분은 이 층에서 한다.

예시:
    python scripts/build_decision_fallback.py --video camel
"""
import json
import sys
from argparse import ArgumentParser
from os import path

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)


def fallback_decision(graph: dict, board: dict, preset: str = "orbit_left",
                      tracking: str = "lock", target_tau: float = 0.20,
                      orbit_span_frac: float = 0.7, max_sweep_deg: float = 45.0,
                      look_at_bias: float = 0.0, start_tau_frac: float = 0.5,
                      reason: str = "no VLM"):
    """τ headroom 을 남긴 board 1위 + 보수적 preset. `lbm/loop.py` 도 이 함수를 부른다."""
    assert board["candidates"], "board 에 통과 후보가 없다 — gates.csv 의 탈락 사유부터 볼 것"
    headroom = start_tau_frac * target_tau
    affordable = [c for c in board["candidates"] if c["tau"] <= headroom]
    if affordable:
        best, budget_note = affordable[0], f"tau<={headroom:.3f} 인 {len(affordable)}개 중 1위"
    else:
        best = min(board["candidates"], key=lambda c: c["tau"])
        budget_note = (f"!! tau<={headroom:.3f} 인 후보가 없다 — tau 최소({best['tau']:.3f})로 대체. "
                       "궤적에 남는 예산이 거의 없다")
    node = next(n for n in graph["nodes"] if n["id"] == board["subject_id"])
    sweep = min(max_sweep_deg, orbit_span_frac * float(node["obs_az_span_deg"]))
    return {
        "format": "lbm_decision_v1", "video": graph["video"], "subject_id": board["subject_id"],
        "source": "fallback", "model": None,
        "keyframes": [{
            "t": 0, "target": board["subject_id"],
            "composition": {"azimuth_deg": best["azimuth_deg"],
                            "elevation_deg": best["elevation_deg"],
                            "d_azimuth_deg": best.get("d_azimuth_deg"),
                            "d_elevation_deg": best.get("d_elevation_deg"),
                            "distance_ratio": best["distance_ratio"],
                            "thirds_anchor": None,
                            "p_G": best["p_g"], "look_at_G": best["look_at_g"],
                            "focal_scale": 1.0},
            "visibility": {"coverage": best["coverage"], "subject_area": best["subject_area"],
                           "occlusion_pass": best["occlusion_pass"]},
            "relation": None, "motion_preference": None,
            "candidate_id": best["board_label"], "micro_ops": []}],
        "trajectory": {"preset": preset, "params": {"sweep_deg": round(sweep, 2)},
                       "shape": {"sweep_deg": round(sweep, 2)},
                       "speed": "steady", "tracking": tracking,
                       "look_at_bias": look_at_bias, "target_tau": target_tau},
        "gates": {"coverage": best["coverage"], "subject_area": best["subject_area"],
                  "tau": best["tau"], "view_angle_deg": best["view_angle_deg"],
                  "board_score": best["board_score"]},
        "vlm": {"observation": None, "reasoning": f"deterministic fallback ({reason})",
                "confidence": None, "turns": 0, "repairs": 0},
        "budget_note": budget_note,
    }


def main(args):
    out_root = args.output_root or path.join(CINEMATRAJ_ROOT, "out")
    video_folder = path.join(out_root, args.video)
    with open(path.join(video_folder, "scene_graph.json"), encoding="utf-8") as file:
        graph = json.load(file)
    with open(path.join(video_folder, "board", "board.json"), encoding="utf-8") as file:
        board = json.load(file)

    decision = fallback_decision(graph, board, preset=args.preset, tracking=args.tracking,
                                 target_tau=args.target_tau,
                                 orbit_span_frac=args.orbit_span_frac,
                                 look_at_bias=args.look_at_bias,
                                 start_tau_frac=args.start_tau_frac)
    output = args.output or path.join(video_folder, "decision.json")
    with open(output, "w", encoding="utf-8") as file:
        json.dump(decision, file, ensure_ascii=False, indent=1)

    keyframe = decision["keyframes"][0]
    print("!! decision.source = fallback — VLM 이 고른 게 아니다. 결과를 그렇게 읽을 것.")
    print(f"{'subject':<16}{decision['subject_id']}")
    print(f"{'candidate':<16}{keyframe['candidate_id']}  "
          f"d_az {keyframe['composition']['d_azimuth_deg']}  "
          f"d_el {keyframe['composition']['d_elevation_deg']}  "
          f"dist {keyframe['composition']['distance_ratio']}x")
    print(f"{'preset':<16}{decision['trajectory']['preset']}  "
          f"sweep {decision['trajectory']['params']['sweep_deg']} deg  "
          f"tracking {decision['trajectory']['tracking']}")
    print(f"{'gates':<16}cov {decision['gates']['coverage']}  "
          f"subj {decision['gates']['subject_area']}  tau {decision['gates']['tau']}")
    print(f"{'budget':<16}{decision['budget_note']}")
    print(f"\n-> {output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--preset", default="orbit_left", type=str)
    parser.add_argument("--tracking", default="lock", type=str,      # D127: drift 삭제
                        choices=["world", "lock"])
    parser.add_argument("--target_tau", default=0.20, type=float)
    parser.add_argument("--orbit_span_frac", default=0.7, type=float)
    parser.add_argument("--look_at_bias", default=0.0, type=float)
    # tau 예산 중 시작 pose 에 허용할 몫. 나머지는 궤적이 쓴다 (모듈 docstring 참고).
    parser.add_argument("--start_tau_frac", default=0.5, type=float)
    main(parser.parse_args())
