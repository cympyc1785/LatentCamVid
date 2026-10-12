"""이 코드가 답하는 질문: Molmo2 tracking+DA3 lift 궤적 (target_track_molmo2.npz) 과 GT subject 궤적 (target_track.npz, OBB 중심) 은
소스 카메라 화면에서 프레임마다 얼마나, 어떻게 (편향·흔들림·다른 물체) 어긋나나?

두 world 궤적을 `<scene>/da3/pose.npz` (w2c extrinsics · intrinsics, images_4 640x360 규약) 로 프레임별 정사영해 소스 프레임에 겹친다.
  초록 = GT (OBB 중심 track)   빨강 = Molmo2 lift   노랑 x = Molmo2 원 2D 점 (json, 보간 전 — 있는 프레임만)
  꼬리 = 지난 8 프레임. 아래 글자 = 프레임 화면 거리 (px) 와 3D 상대 오차 (|M-G| / |G-cam| 중앙).
R341 (사용자 10-12 "Molmo2 track 과 gt track 을 프레임별로 source camera 로 정사영한걸 overlay 해서 비교"). CPU 만.

  python scripts/viz/track_overlay_molmo2_vs_gt.py --items <video>:<seg>,... --out x.mp4
  python scripts/viz/track_overlay_molmo2_vs_gt.py --sample_bins 0,0.1,0.2,0.4,1,inf --per_bin 2 --out x.mp4   # test 에서 오차 구간별 추출
"""
import argparse, json, os, os.path as osp, random
import numpy as np, cv2, imageio.v2 as iio

C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"


def project(track_world, E, K):
    """(T,3) world -> (T,2) 픽셀 (카메라 뒤면 nan). E = w2c (T,3,4) — scripts/viz/molmo2_vs_obb.py:project 와 같은 규약."""
    Xc = np.einsum("tij,tj->ti", E[:, :, :3], track_world) + E[:, :, 3]
    uv = np.stack([K[:, 0, 0] * Xc[:, 0] / Xc[:, 2] + K[:, 0, 2], K[:, 1, 1] * Xc[:, 1] / Xc[:, 2] + K[:, 1, 2]], -1)
    uv[Xc[:, 2] <= 1e-6] = np.nan
    return uv


def load_item(corpus, v, seg, track_dir):
    d = f"{corpus}/dynpose/{v}/da3"
    g, m = np.load(f"{d}/target_track.npz", allow_pickle=True), np.load(f"{d}/target_track_molmo2.npz", allow_pickle=True)
    gi, mi = list(map(str, g["keys"])).index(seg), list(map(str, m["keys"])).index(seg)
    p = np.load(f"{d}/pose.npz"); E, K = p["extrinsics"].astype(np.float64), p["intrinsics"].astype(np.float64)
    G, M = g["track_world"][gi].astype(np.float64), m["track_world"][mi].astype(np.float64)
    aid = str(m["anchor_id"][mi]); raw = {}
    jp = f"{track_dir}/{v}__{aid}.json"
    if osp.isfile(jp):
        P = json.load(open(jp))["pts"]; cnt = {}
        for q in P:
            cnt[q[1]] = cnt.get(q[1], 0) + 1
        if cnt:
            main = max(cnt, key=cnt.get)
            for f, tid, x, y in P:
                if tid == main and f not in raw:
                    raw[f] = (x, y)
    cam = np.stack([-e[:, :3].T @ e[:, 3] for e in E])
    rel = float(np.median(np.linalg.norm(M - G, axis=-1) / np.linalg.norm(G - cam, axis=-1).clip(1e-6))) if m["valid"][mi] else float("inf")
    return dict(G=G, M=M, gv=bool(g["valid"][gi]), mv=bool(m["valid"][mi]), E=E, K=K, raw=raw, aid=aid, rel=rel,
                label=json.load(open(f"{d}/prompts.json")).get(seg, {}).get("preset", ""))


def render(corpus, v, seg, it, W=480):
    fr = [np.asarray(cv2.imread(f"{corpus}/dynpose/{v}/images_4/{i:05d}.png"))[:, :, ::-1] for i in range(len(it["E"]))]
    h0, w0 = fr[0].shape[:2]; s = W / w0
    ug, um = project(it["G"], it["E"], it["K"]), project(it["M"], it["E"], it["K"])
    out = []
    for f, x in enumerate(fr):
        x = np.ascontiguousarray(cv2.resize(x, (W, int(h0 * s))))
        for uv, col, ok in ((ug, (0, 255, 0), it["gv"]), (um, (255, 40, 40), it["mv"])):
            if not ok:
                continue
            pts = [(int(uv[t, 0] * s), int(uv[t, 1] * s)) for t in range(max(0, f - 8), f + 1) if np.isfinite(uv[t]).all()]
            for a, b in zip(pts, pts[1:]):
                cv2.line(x, a, b, col, 1, cv2.LINE_AA)
            if pts and np.isfinite(uv[f]).all():
                cv2.circle(x, pts[-1], 5, col, 2, cv2.LINE_AA)
        if f in it["raw"]:
            rx, ry = int(it["raw"][f][0] * W), int(it["raw"][f][1] * h0 * s)
            cv2.drawMarker(x, (rx, ry), (255, 230, 0), cv2.MARKER_TILTED_CROSS, 9, 2)
        dpx = np.linalg.norm(ug[f] - um[f]) * s if (it["gv"] and it["mv"] and np.isfinite(ug[f]).all() and np.isfinite(um[f]).all()) else np.nan
        cv2.putText(x, f"{v[:8]}:{seg} {it['aid']} {it['label']}", (4, 14), 0, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(x, f"f{f:02d} d={dpx:.0f}px rel3d={it['rel']:.2f}" + ("" if it["mv"] else " molmo2 INVALID"),
                    (4, x.shape[0] - 6), 0, 0.4, (255, 255, 0), 1, cv2.LINE_AA)
        out.append(x)
    return out


def sample(corpus, bins, per_bin, seed, track_dir):
    segs = [l.strip().split("/") for l in open(f"{corpus}/seg_list_dynpose_s91_test.txt") if l.strip()]
    random.Random(seed).shuffle(segs); picked = {i: [] for i in range(len(bins) - 1)}; seen = set()
    for _, v, seg in segs:
        if v in seen:
            continue
        try:
            it = load_item(corpus, v, seg, track_dir)
        except (ValueError, FileNotFoundError):
            continue
        if not it["gv"]:
            continue
        for i in range(len(bins) - 1):
            lo, hi = bins[i], bins[i + 1]
            if (lo <= it["rel"] < hi or (lo == hi == float("inf") and it["rel"] == lo)) and len(picked[i]) < per_bin:
                picked[i].append((v, seg)); seen.add(v); break
        if all(len(p) >= per_bin for p in picked.values()):
            break
    return [x for i in sorted(picked) for x in picked[i]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=C)
    ap.add_argument("--track_dir", default=f"{C}/molmo2_track")
    ap.add_argument("--items", default="")                  # video:seg,...
    ap.add_argument("--sample_bins", default="")            # 예 0,0.1,0.2,0.4,1,inf (inf,inf 이면 Molmo2 invalid 만)
    ap.add_argument("--per_bin", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = [tuple(x.split(":")) for x in a.items.split(",") if x]
    if a.sample_bins:
        items += sample(a.corpus, [float(b) for b in a.sample_bins.split(",")], a.per_bin, a.seed, a.track_dir)
    clips = []
    for v, seg in items:
        it = load_item(a.corpus, v, seg, a.track_dir); clips.append(render(a.corpus, v, seg, it))
        print(f"{v}:{seg} anchor {it['aid']} rel3d {it['rel']:.3f} valid gt {it['gv']} molmo2 {it['mv']}", flush=True)
    T = min(len(c) for c in clips); H, W = clips[0][0].shape[:2]
    while len(clips) % a.cols:
        clips.append([np.zeros((H, W, 3), np.uint8)] * T)
    rows = [clips[i:i + a.cols] for i in range(0, len(clips), a.cols)]
    os.makedirs(osp.dirname(osp.abspath(a.out)), exist_ok=True)
    iio.mimwrite(a.out, [np.vstack([np.hstack([c[f] for c in r]) for r in rows]) for f in range(T)], fps=8, macro_block_size=1)
    print("saved", a.out)
