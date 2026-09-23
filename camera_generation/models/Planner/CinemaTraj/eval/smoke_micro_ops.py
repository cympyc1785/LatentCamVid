"""micro-adjust 연산 14종을 VLM 없이 전부 적용해 보고 게이트 결과를 표로 낸다.

왜 필요한가: `lbm/loop.py` 를 camel 에서 처음 end-to-end 로 돌렸더니 VLM 이 round 0 에서
`done:true` 를 내버려 **`ops.py` 가 단 한 번도 실행되지 않았다**. 루프가 "성공"으로 끝나도
micro 경로는 미검증으로 남는다 — 모델이 연산을 안 고르면 영원히 안 밟히는 코드다.

여기서 재는 것은 두 가지다:
  ① `apply_op` 가 clamp / delta 를 제대로 계산하는가 (`CLAMPED` 열)
  ② 적용된 pose 가 게이트를 통과하는가, 떨어지면 어느 게이트인가

게이트 임계는 `loop.build_parser()` 에서 그대로 가져온다. 손으로 다시 적으면 테스트만 통과하고
루프는 떨어지는 상황이 생긴다.

사용 예시:
    CUDA_VISIBLE_DEVICES=1 python eval/smoke_micro_ops.py --video camel
    CUDA_VISIBLE_DEVICES=1 python eval/smoke_micro_ops.py --video camel --start B6 --allow_zoom
"""
import json
import sys
from os import path

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.loop import Session, build_parser, state_line   # noqa: E402
from lbm.ops import OPS, ZOOM_OPS, apply_op, make_state  # noqa: E402


def main(args):
    folder = path.join(args.output_root or path.join(CINEMATRAJ_ROOT, "out"), args.video)
    graph = json.load(open(path.join(folder, "scene_graph.json"), encoding="utf-8"))
    board = json.load(open(path.join(folder, "board", "board.json"), encoding="utf-8"))
    node = next(n for n in graph["nodes"] if n["id"] == board["subject_id"])

    session = Session(args, graph, board, node)
    picked = next(c for c in board["candidates"] if c["board_label"] == args.start) \
        if args.start else board["candidates"][0]
    state = make_state(picked["p_g"], picked["look_at_g"], 1.0)
    base = session.gate(state, picked["board_label"])
    print(f"start {picked['board_label']}   {state_line(base)}")
    print("-" * 112)
    print(f"{'op':<16}{'note':<32}{'gate':<16}{'cov':>8}{'subj':>8}{'tau':>8}{'d_az':>8}{'d_el':>8}")

    rows = []
    for op in OPS:
        if op in ZOOM_OPS and not args.allow_zoom:
            continue
        candidate, note = apply_op(state, op, node, session.d_ref, session.frame)
        if candidate is None:
            print(f"{op:<16}{str(note)[:31]:<32}{'CLAMPED':<16}")
            rows.append({"op": op, "gate": "clamped", "note": note})
            continue
        row = session.gate(candidate, op)
        verdict = row["failed"] or "pass"
        print(f"{op:<16}{str(note)[:31]:<32}{verdict:<16}"
              f"{(row.get('coverage') if row.get('coverage') is not None else float('nan')):>8.3f}"
              f"{(row.get('subject_area') if row.get('subject_area') is not None else float('nan')):>8.3f}"
              f"{(row.get('tau') if row.get('tau') is not None else float('nan')):>8.3f}"
              f"{row['d_azimuth_deg']:>8.1f}{row['d_elevation_deg']:>8.1f}")
        rows.append({"op": op, "gate": verdict, "note": note,
                     "coverage": row.get("coverage"), "subject_area": row.get("subject_area"),
                     "tau": row.get("tau")})

    passed = sum(1 for r in rows if r["gate"] == "pass")
    print("-" * 112)
    print(f"{'total':<16}{len(rows)} ops   pass {passed}   "
          f"gated {sum(1 for r in rows if r['gate'] not in ('pass', 'clamped'))}   "
          f"clamped {sum(1 for r in rows if r['gate'] == 'clamped')}")
    out_path = path.join(folder, "trace", "smoke_micro_ops.json")
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({"start": picked["board_label"], "rows": rows}, file,
                  ensure_ascii=False, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = build_parser()   # 게이트 임계 · 렌더 타일 크기를 루프와 공유한다
    parser.add_argument("--start", default=None, type=str)   # board label, 기본은 1등 후보
    main(parser.parse_args())
