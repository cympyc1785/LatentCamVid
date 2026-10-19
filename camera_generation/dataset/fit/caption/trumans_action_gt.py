"""이 코드가 답하는 질문: TRUMANS 클립 창 안에서 사람이 무엇을 하는가 — 어떤 물체를 어느 손으로 (GT 로)?

R448 (사용자 2026-10-20 "TRUMANS 클립의 action 캡션 필드 제안대로 만들어줘"). TRUMANS 라벨 (`Actions/<seq>.txt`) 은 동사는
쓸 만하지만 물체·손이 GT 와 어긋나는 행이 있고 (R448 실측: 라벨 물체가 GT 로 움직인 물체 중에 있음 87.1%, 손 일치 81.5%),
프레임의 ~64% 는 라벨이 없다. 그래서 클립 창마다 세 출처를 합친다:

  label    TRUMANS 라벨의 **동사** + GT 로 그 구간에 움직인 물체 (가장 크게) + GT 손 (물체가 움직이는 프레임의 손목 근접).
           GT 물체가 없으면 (키보드 치기 · 앉기 등) 라벨 문장 그대로.
  object   라벨이 없는 구간에서 물체가 움직임 → "handles the <obj> with the <hand> hand" (동사는 높이 변화로: 올라감 lift, 내려감 put down)
  body     골반 (SMPL 관절 0, y 위) — 수평 속도 > 0.4 m/s 가 0.5 s 이상 = walk, 높이 −0.25 m = sit/squat down, +0.25 m = stand up.
           라벨이 같은 동작을 덮으면 넣지 않는다.

물체 움직임 판정은 `fit/graph/trumans_vlm_action_tag.moved_objects` (0.05 m 또는 12°) 그대로. 손: 왼 손목 = 관절 20, 오른 = 21,
둘 다 0.25 m 안이고 거리 비 < 1.5 면 both. 좌표는 TRUMANS 원본 (y 위, Object_pose 와 human_joints 가 같은 world).

입력: board 클립 (`recon_and_seg/tru_<rec8>_c##_f#####_k###_p#`) 의 manifest (`out/trumans_recon/<rec>/manifest_<key>.json`)
      — `frames` [lo, hi] 가 sequence 프레임 번호다 (Actions 와 같은 축). manifest 의 sequence 가 없으면 (`--allow_unlabeled` 로
      만든 클립) `_frames.json` 프레임 수로 sequence 를 다시 찾고, 못 찾으면 events 없이 `sequence: null`.
출력: `out_trumans/<video>/action_gt.json` (`trumans_action_gt_v1`) — events (start · end · source · verb · object · hand · text) 와
      시간순으로 이은 `caption_action` 한 문장.
env: vista4d. 예) python fit/caption/trumans_action_gt.py --workers 24 [--videos tru_00add26c_c00_f00000_k137_p0 ...]
"""
import json
import os
import re
import sys
from argparse import ArgumentParser
from glob import glob
from multiprocessing import Pool
from os import path

import numpy as np

HERE = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.insert(0, HERE)
import fit.graph.trumans_vlm_action_tag as TG  # noqa: E402

T = "/data1/cympyc1785/data/trumans/Data_release"
E = "/data1/cympyc1785/data/TRUMANS-Lite/eval_data/recon_and_seg"
RECON = path.join(HERE, "out", "trumans_recon")
OUT = path.join(HERE, "out_trumans")
FPS = 30.0
SEG = None
BOARD = re.compile(r"tru_([0-9a-f]{8})_(c\d+_f\d+_k\d+_p\d)$")


def read_actions(seq):
    acts = []
    p = path.join(T, "Actions", f"{seq}.txt")
    if not path.isfile(p):
        return acts
    for line in open(p, encoding="utf-8"):
        parts = [x for x in line.rstrip("\n").split("\t") if x]
        if len(parts) >= 3:
            s, e, t = (parts[0], parts[1], parts[2]) if len(parts) == 3 else (parts[1], parts[2], " ".join(parts[3:]))
            acts.append({"start": int(s), "end": int(e), "text": t.strip()})
    return acts


def find_sequence(rec_dir_name, n_frames):
    from fit.ingest.trumans_to_recon import sequences_for_scene
    table = sequences_for_scene(T, rec_dir_name.split("_")[0])
    hits = [k for k, c in table.items() if c == n_frames]
    return hits[0] if len(hits) == 1 else None


def hand_of(J, loc, fr):
    dl = float(np.linalg.norm(J[fr, 20] - loc[fr], axis=1).mean()); dr = float(np.linalg.norm(J[fr, 21] - loc[fr], axis=1).mean())
    if max(dl, dr) < 0.25 and max(dl, dr) / max(min(dl, dr), 1e-6) < 1.5:
        return "both"
    return "left" if dl < dr else "right"


def hand_text(h):
    return " with both hands" if h == "both" else f" with the {h} hand" if h in ("left", "right") else ""


def gt_object(poses, J, s, e):
    """[s, e] 에서 가장 크게 움직인 물체 → (읽기 이름, 손, 높이 변화 m) 또는 None."""
    best = None
    for n in poses:
        m = TG.moved_objects({n: poses[n]}, s, e)
        if m and (best is None or m[0][1] + m[0][2] / 100 > best[1]):
            best = (n, m[0][1] + m[0][2] / 100)
    if best is None:
        return None
    loc = poses[best[0]][0]; e2 = min(e, len(loc) - 1, len(J) - 1); s2 = min(max(s, 0), e2)
    v = np.linalg.norm(np.diff(loc[s2:e2 + 1], axis=0), axis=1)
    fr = np.arange(s2, e2)[v > max(v.max() * 0.2, 1e-4)] if len(v) else np.array([], int)
    hand = hand_of(J, loc, fr) if len(fr) else "none"
    return TG.readable_object(best[0]), hand, float(loc[e2, 1] - loc[s2, 1])


def body_events(J, lo, hi):
    out = []
    p = J[lo:hi + 1, 0]
    if len(p) < 16:
        return out
    sp = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) * FPS
    walk = sp > 0.4; i = 0
    while i < len(walk):
        if walk[i]:
            j = i
            while j < len(walk) and walk[j]:
                j += 1
            if j - i >= FPS * 0.5:
                out.append({"start": lo + i, "end": lo + j, "source": "body", "verb": "walk", "object": "", "hand": "", "text": "walks"})
            i = j
        else:
            i += 1
    dy = p[-1, 1] - p[0, 1]
    if abs(dy) > 0.25:
        k = int(np.argmax(np.abs(p[:, 1] - p[0, 1]) > 0.5 * abs(dy)))
        verb = "stand up" if dy > 0 else ("squat down" if p[:, 1].min() < p[0, 1] - 0.45 and abs(dy) > 0.45 else "sit down")
        out.append({"start": lo + max(k - 15, 0), "end": lo + min(k + 15, len(p) - 1), "source": "body", "verb": verb, "object": "", "hand": "", "text": verb.replace("stand up", "stands up").replace("sit down", "sits down").replace("squat down", "squats down")})
    return out


def third(phrase):
    """명령형 → 3인칭 ("pick up" → "picks up", "close" → "closes")."""
    w = phrase.split(" ", 1); v = w[0].lower()
    v = v[:-1] + "ies" if v.endswith("y") and v[-2:-1] not in "aeiou" else v + "es" if v.endswith(("s", "sh", "ch", "x", "z", "o")) else v + "s"
    return " ".join([v] + w[1:])


def label_text(text, obj, hand):
    """라벨 동사 + GT 물체 · 손. 라벨에 물체가 없으면 (Sit down · Stand up) 라벨 그대로 (3인칭)."""
    s = text.strip().rstrip(".")
    if " the " in s and obj:
        return f"{third(s.split(' the ')[0])} the {obj}{hand_text(hand)}"
    return third(s[0].lower() + s[1:])


def one(video):
    m = BOARD.match(video)
    if not m:
        return video, "not_board"
    mf = glob(path.join(RECON, f"{m.group(1)}-*", f"manifest_{m.group(2)}.json"))
    if not mf:
        return video, "no_manifest"
    man = json.load(open(mf[0])); lo, hi = [int(x) for x in man["frames"]]; rec = path.basename(path.dirname(mf[0]))
    seq = man.get("sequence")
    if not seq:
        fj = path.join(path.dirname(mf[0]), "_frames.json")
        if path.isfile(fj):
            f = json.load(open(fj)); seq = find_sequence(rec, int(f["end"]) - int(f["start"]) + 1)
    res = {"format": "trumans_action_gt_v1", "video": video, "recording": rec, "sequence": seq, "frames": [lo, hi], "events": [], "caption_action": ""}
    if seq:
        hit = np.flatnonzero(SEG == seq)
        J = np.load(path.join(T, "human_joints.npy"), mmap_mode="r")[hit[0]:hit[-1] + 1]
        try:
            raw, _ = TG.read_scene_objects(T, TG.resolve_recording(T, seq)); poses = TG.read_object_poses(T, seq, raw)
        except Exception:  # noqa: BLE001
            poses = {}
        ev, covered = [], np.zeros(hi - lo + 1, bool)
        for a in read_actions(seq):
            s, e = max(a["start"], lo), min(a["end"], hi)
            if e - s + 1 < 8:
                continue
            g = gt_object(poses, J, a["start"], a["end"]) if poses and " the " in a["text"] else None
            obj, hand = (g[0], g[1]) if g else ("", "")
            ev.append({"start": s, "end": e, "source": "label", "verb": a["text"].split(" the ")[0].lower(), "object": obj, "hand": hand,
                       "label_text": a["text"], "text": label_text(a["text"], obj, hand)})
            covered[s - lo:e - lo + 1] = True
        # 라벨 밖 물체 움직임 (30 프레임 창을 15 간격으로)
        if poses:
            for s in range(lo, hi - 29, 15):
                if covered[s - lo:s - lo + 30].mean() > 0.3:
                    continue
                g = gt_object(poses, J, s, s + 29)
                if not g or any(x["source"] == "label" and x["object"] == g[0] and x["end"] >= s - 30 and x["start"] <= s + 59 for x in ev):
                    continue                                       # 라벨 동작 바로 앞뒤의 같은 물체 움직임은 그 라벨의 일부
                prev = next((x for x in reversed(ev) if x["source"] == "object"), None)
                if prev and prev["object"] == g[0] and prev["end"] >= s - 1:
                    prev["end"] = s + 29; prev["dy"] += g[2]; prev["n"] += 1   # 같은 물체를 연달아 다룸 → 한 구간
                else:
                    ev.append({"start": s, "end": s + 29, "source": "object", "object": g[0], "hand": g[1], "dy": g[2], "n": 1})
                covered[s - lo:s - lo + 30] = True
        for x in ev:
            if x["source"] == "object":
                x["verb"] = "picks up" if x["dy"] > 0.05 else "puts down" if x["dy"] < -0.05 else ("holds" if x["n"] > 1 else "handles")
                x["text"] = f"{x['verb']} the {x['object']}{hand_text(x['hand'])}"; x.pop("dy"); x.pop("n")
        for b in body_events(J, lo, hi):
            lab = [x for x in ev if x["source"] == "label" and b["verb"].split()[0] in x["verb"]]
            if not lab and covered[b["start"] - lo:b["end"] - lo + 1].mean() < 0.5:
                ev.append(b)
        ev.sort(key=lambda x: x["start"])
        res["events"] = ev
        texts = [x["text"] for x in ev]
        res["caption_action"] = ("The person " + ", then ".join(texts) + ".") if texts else ""
    os.makedirs(path.join(OUT, video), exist_ok=True)
    json.dump(res, open(path.join(OUT, video, "action_gt.json"), "w"), indent=1)
    return video, "ok" if res["events"] else ("no_sequence" if not seq else "no_events")


def init():
    global SEG
    SEG = np.load(path.join(T, "seg_name.npy"), allow_pickle=True)


def main():
    ap = ArgumentParser()
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    vids = a.videos or sorted(v for v in os.listdir(E) if BOARD.match(v))
    with Pool(a.workers, initializer=init) as p:
        res = p.map(one, vids, chunksize=8)
    from collections import Counter
    print(len(res), dict(Counter(r for _, r in res)))


if __name__ == "__main__":
    main()
