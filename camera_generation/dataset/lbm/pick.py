"""뱅크 행 중 **무엇을 내보낼지** 고르는 규칙 한 벌. `picked` 열을 다는 곳은 여기 하나다.

왜 모듈로 뺐나: 이 규칙은 `fit_hole_ladder.py` 안에 있었는데, 뱅크를 **다시 굽지 않고** 선택만
바꾸고 싶은 일이 생긴다 (D189). `pick` 은 사다리가 다 끝난 뒤의 순수 후처리라 — 행을 지우지도
않고 기하를 다시 만들지도 않는다 — `bank.json` 만 있으면 재실행할 수 있다. 규칙을 두 벌 두면
"다시 고른 뱅크"와 "새로 구운 뱅크"가 조용히 갈라지므로 함수 하나를 양쪽이 부른다.

정렬 키는 D188 ③ 의 것 그대로다 (근거는 `fit_hole_ladder.py` §pick_budget 주석에 있다):

    track 먼저  →  `plan_tier` 오름차순  →  등급(usable 먼저)  →  `hole_fraction` 오름차순

`per_anchor` 만 D189 에서 새로 붙었다. 씬 단위 예산은 anchor 가 둘이어도 한 대만 내보내므로
두 번째 anchor 가 통째로 버려진다 — d188 뱅크 1,252편 실측으로 씬당 0.783 대였다. anchor 마다
예산을 따로 주면 0.935 대가 되고(+19.5%), 같은 (scene, anchor) 는 여전히 한 대뿐이라 같은
물체를 두 번 굽는 일은 안 생긴다 (사용자 확정 2026-09-14 "같은 scene 같은 anchor 안 곂치게").

사용 예시:
    from lbm.pick import pick_rows
    pick_rows(bank["variants"], budget=1, retry_status=("clamped_low", "static"),
              retry_suspect=("hole_over_budget",), per_anchor=True)
"""


def pick_key(row, retry_suspect=()) -> tuple:
    """정렬 키. 작을수록 먼저 뽑힌다. **읽기만 한다** — 열을 다시 쓰지 않는다.

    `variant_usable()` 을 다시 부르지 않는 이유: 그 함수는 `tag_suspects` 를 호출해 `suspect`
    열을 다시 쓰므로, `--no_suspect` 로 비워 둔 열이 되살아난다. 여기서는 이미 확정된 열만 본다.
    """
    tags = set(str(row.get("suspect", "") or "").split("|")) - {""}
    grade = 1 if tags & set(retry_suspect) else 0
    hole = row.get("hole_fraction")
    return (0 if str(row["preset"]).startswith("track_") else 1,
            int(row.get("plan_tier", 0) or 0), grade,
            float(hole) if isinstance(hole, (int, float)) else float("inf"))


def pick_rows(rows: list, budget: int, retry_status=(), retry_suspect=(),
              per_anchor: bool = False) -> list:
    """`rows` 의 `picked` 열을 **제자리에서** 다시 단다. 고른 행 리스트를 돌려준다.

    `budget` 이 0/None 이면 아무것도 안 한다 (열을 건드리지도 않는다) — `--pick_budget` 기본값
    0 이 "예전 뱅크와 동일"을 뜻하던 동작을 그대로 유지하기 위함이다.

    `retry_status` 로 시작하는 `status` 행은 애초에 후보가 아니다 (`clamped_low` 등 = 게이트를
    못 넘긴 행). `retry_suspect` 는 **탈락이 아니라 감점**이다 — 태그가 붙어도 다른 후보가
    없으면 뽑힌다.
    """
    if not budget:
        return [r for r in rows if r.get("picked")]
    budget = int(budget)
    ok = [r for r in rows
          if not any(str(r["status"]).startswith(t) for t in retry_status)]
    for row in rows:
        row["picked"] = ""
    if per_anchor:
        groups = {}
        for row in ok:
            groups.setdefault(row["anchor_id"], []).append(row)
        # anchor 순회 순서는 결정론적이어야 한다 (뱅크를 두 번 고르면 같은 답이 나오게).
        chosen = [r for key in sorted(groups)
                  for r in sorted(groups[key], key=lambda x: pick_key(x, retry_suspect))[:budget]]
    else:
        chosen = sorted(ok, key=lambda x: pick_key(x, retry_suspect))[:budget]
    for row in chosen:
        row["picked"] = "1"
    return chosen
