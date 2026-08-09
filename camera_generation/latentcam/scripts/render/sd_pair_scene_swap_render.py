"""umeyama sim3 로 맞춘 두 clip 이 **정말 같은 좌표계에 있는가** 를 눈으로 검증한다.

Scene-Decoupled 는 clip 마다 da3 를 따로 돌려 각자 임의 스케일이고, `umeyama_gt.json` 의
sim3 (s,R,t) 로 GT meters 에 맞춘다. cross-clip 학습(context clip 으로 target clip 을 맞추기)
은 이 정렬이 맞다는 데 전적으로 의존한다.

검증 방식 — "scene 만 바꿔서 렌더":
  target clip T 의 frame t 를, **source clip S 의 scene(depth+RGB)** 으로 다시 그린다.
    depth_S[t] --unproject--> S 의 da3 world --sim3_S--> GT meters
                --sim3_T^-1--> T 의 da3 world --project(pose_T[t])--> 화면
  정렬이 완벽하면 결과는 T 의 GT frame t 와 같아야 한다.

**한 프레임에 한 depth만** 쓴다: SD 영상엔 dynamic subject 가 있어서 여러 프레임의 point 를
누적하면 움직이는 물체가 번져 정렬 오차와 구분이 안 된다. 출력 frame t 는 source 의 **같은 t**
프레임 하나만 unproject 한다 (구멍은 생기지만 해석이 깨끗하다).

---------------------------------------------------------------------------
[중요] `umeyama_gt.json` 의 R 은 straight-line clip 에서 못 쓴다
---------------------------------------------------------------------------
GT 카메라(`camera/<split>/<scene>/<scene>_cam.json`, Unreal 좌표계 LH, cm) 와 대조해 보면:
  * da3 의 상대 회전/이동은 GT 와 거의 완벽히 맞는다 (rel-rot 오차 0.2~1.5 deg).
  * `umeyama_gt.json` 의 s,t 도 맞는다 — 카메라 중심이 GT 와 1~2 cm 안에서 일치한다.
  * 그런데 R 은 clip 마다 **전혀 다른 값**이 나온다. 같은 scene 의 clip 두 개를 각자 R 로
    보내면 서로 최대 174 deg 어긋난다.
원인: 이 데이터셋의 카메라 궤적은 대부분 **직선**이다 (중심 좌표의 특이값 [1, 0, 0]).
직선 점집합에 대한 Umeyama 는 그 직선을 축으로 한 회전 1 자유도가 **완전히 미결정**이라
아무 값이나 나오고, 그런데도 `resid_rmse_over_rad` 는 작게 나온다. 즉 **resid 로는 이 고장을
걸러낼 수 없다.**

고치는 법 (`--sim3 gtrot`): R 을 위치가 아니라 **GT 카메라 방향**에서 푼다.
    R_ext[t] @ R_sim^T = A0 @ R_gt_rh[t]      ->      R_sim = R_gt_rh[t]^T @ A0^T @ R_ext[t]
  를 프레임마다 풀고 SVD 로 평균낸다. 여기서
    p_rh = p_ue * diag(1,-1,1),  R_rh = diag(1,-1,1) @ R_ue @ diag(1,-1,1)   (UE LH -> RH)
    A0   = [[0,-1,0],[0,0,-1],[1,0,0]]        (UE 카메라축 -> OpenCV 카메라축)
  A0 는 궤적이 직선이 아닌 clip(특이값 [1,0.12,0.0004]) 에서 실측으로 확인한 상수다.
  R 을 고정한 뒤 s,t 는 위치에서 최소제곱으로 다시 푼다.
  이 방식은 회전만 있으면 풀리므로 **static clip 도 R,t 는 나온다** (s 만 미결정).

격자 주의: depth 는 (294,504), intrinsics 도 그 격자 기준(cx=252, cy=147)인데 mp4 는
672x384 다. RGB 를 504x294 로 resize 해서 쓴다.

usage:
  python scripts/render/sd_pair_scene_swap_render.py                    # scan 후 best/worst 자동
  python scripts/render/sd_pair_scene_swap_render.py --scene <scene> --clips 01_24mm 05_24mm
out -> results/scene_decoupled/pair_scene_swap/{<tag>.mp4, selection.md, per_frame.csv}
"""
import os, sys, json, re, random, itertools, argparse
import numpy as np
import cv2

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
ROOT = "/data1/cympyc1785/data/Scene-Decoupled-Video-dataset"
GRID_W, GRID_H = 504, 294          # da3 격자 = intrinsics 기준 격자
_D = np.diag([1.0, -1.0, 1.0])                              # UE(LH) -> RH
A0 = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])        # UE 카메라축 -> OpenCV 카메라축


# ---------------------------------------------------------------- 로드
def clip_dir(split, scene, clip):
    return os.path.join(ROOT, "da3", split, scene, clip)


def read_gt_cam(split, scene):
    """<scene>_cam.json -> {clip_suffix: (R_rh (T,3,3), p_rh (T,3) meters)}."""
    p = os.path.join(ROOT, "camera", split, scene, f"{scene}_cam.json")
    j = json.load(open(p))
    fr = sorted(j.keys(), key=int)
    out = {}
    for c in j[fr[0]]:
        Rs, ps = [], []
        for f in fr:
            v = [float(x) for x in re.findall(r"-?[\d.eE+-]+", j[f][c])]
            m = np.asarray(v, np.float64).reshape(4, 4)
            Rs.append(_D @ m[:3, :3] @ _D)          # 행 = 카메라축 (UE), LH -> RH
            ps.append(_D @ (m[3, :3] / 100.0))      # cm -> m, LH -> RH
        out[c] = (np.stack(Rs), np.stack(ps))
    return out


def read_sim3_file(split, scene, clip):
    p = os.path.join(clip_dir(split, scene, clip), "umeyama_gt.json")
    if not os.path.isfile(p):
        return None
    u = json.load(open(p))
    if not u.get("moving", False):
        return None
    return (float(u["s"]), np.asarray(u["R"], np.float64), np.asarray(u["t"], np.float64),
            float(u.get("resid_rmse_over_rad", float("nan"))))


def sim3_from_gtrot(ext, gtR, gtp):
    """R 은 GT 카메라 방향에서, s/t 는 위치 최소제곱으로. -> (s,R,t,resid, rot_spread_deg)."""
    n = min(len(ext), len(gtR))
    M = np.zeros((3, 3))
    for t in range(n):
        M += gtR[t].T @ A0.T @ ext[t][:3, :3]       # 프레임별 R_sim 후보를 합산 -> SVD 평균
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    sp = max(_ang(R.T @ (gtR[t].T @ A0.T @ ext[t][:3, :3])) for t in range(n))

    cen = np.stack([-ext[t][:3, :3].T @ ext[t][:3, 3] for t in range(n)])
    P = gtp[:n]
    a, b = cen @ R.T, P
    am, bm = a - a.mean(0), b - b.mean(0)
    s = float((am * bm).sum() / max((am * am).sum(), 1e-12))
    tv = P.mean(0) - s * a.mean(0)
    r = np.linalg.norm(s * a + tv - P, axis=1)
    rad = float(np.linalg.norm(P - P.mean(0), axis=1).mean())
    return s, R, tv, float(np.sqrt((r ** 2).mean()) / max(rad, 1e-9)), float(sp)


def _ang(D):
    return float(np.degrees(np.arccos(np.clip((np.trace(D) - 1) / 2, -1, 1))))


def read_clip(split, scene, clip, gt, variant):
    d = clip_dir(split, scene, clip)
    P = np.load(os.path.join(d, "pose.npz"))
    ext = np.asarray(P["extrinsics"], np.float64)          # (T,3,4) OpenCV w2c
    K = np.asarray(P["intrinsics"], np.float64)            # (T,3,3), 504x294 격자
    dep = np.asarray(np.load(os.path.join(d, "depth.npz"))["depth"], np.float32)
    cap = cv2.VideoCapture(os.path.join(ROOT, "video", split, scene, clip + ".mp4"))
    rgb = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        rgb.append(cv2.resize(f, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA))
    cap.release()

    f3 = read_sim3_file(split, scene, clip)
    suf = clip[len(scene) + 1:]
    gR, gp = gt[suf]
    g3 = sim3_from_gtrot(ext, gR, gp)
    sv = np.linalg.svd(gp - gp.mean(0), compute_uv=False)
    return dict(clip=clip, ext=ext, K=K, dep=dep, rgb=rgb, file3=f3, gt3=g3,
                straightness=float(sv[1] / max(sv[0], 1e-12)),
                sim3=(f3[:3] if variant == "file" else g3[:3]))


# ---------------------------------------------------------------- 렌더
_UV = None


def _uv_grid():
    global _UV
    if _UV is None:
        u, v = np.meshgrid(np.arange(GRID_W, dtype=np.float64),
                           np.arange(GRID_H, dtype=np.float64))
        _UV = (u.ravel(), v.ravel())
    return _UV


def render_swap(S, T, t, sim3_S, sim3_T, radius=1):
    """source clip S 의 frame t scene 으로 target clip T 의 frame t 를 다시 그린다."""
    z = S["dep"][t].astype(np.float64).ravel()
    col = S["rgb"][min(t, len(S["rgb"]) - 1)].reshape(-1, 3)
    u, v = _uv_grid()
    ok = np.isfinite(z) & (z > 0)

    Ks = S["K"][t]
    x = (u[ok] - Ks[0, 2]) / Ks[0, 0] * z[ok]
    y = (v[ok] - Ks[1, 2]) / Ks[1, 1] * z[ok]
    Xc = np.stack([x, y, z[ok]], 1)                              # S 카메라 좌표

    Rs, ts = S["ext"][t][:, :3], S["ext"][t][:, 3]
    Xw = (Xc - ts) @ Rs                                          # S 의 da3 world

    ss, sR, st = sim3_S
    Xg = ss * (Xw @ sR.T) + st                                   # GT meters (RH)
    ts_, tR, tt = sim3_T
    Xt = ((Xg - tt) @ tR) / ts_                                  # T 의 da3 world

    Rt, tt2 = T["ext"][t][:, :3], T["ext"][t][:, 3]
    Xtc = Xt @ Rt.T + tt2                                        # T 카메라 좌표
    zz = Xtc[:, 2]
    front = zz > 1e-6
    Kt = T["K"][t]
    uu = Kt[0, 0] * Xtc[front, 0] / zz[front] + Kt[0, 2]
    vv = Kt[1, 1] * Xtc[front, 1] / zz[front] + Kt[1, 2]
    zf, cf = zz[front], col[ok][front]

    # z-buffer splat: 모든 offset 을 한 배열로 모아 z 내림차순 정렬 후 덮어쓰기(가까운 게 마지막).
    ui, vi = np.rint(uu).astype(np.int64), np.rint(vv).astype(np.int64)
    idx_l, z_l, c_l = [], [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            a, b = ui + dx, vi + dy
            m = (a >= 0) & (a < GRID_W) & (b >= 0) & (b < GRID_H)
            idx_l.append(b[m] * GRID_W + a[m]); z_l.append(zf[m]); c_l.append(cf[m])
    idx = np.concatenate(idx_l); zc = np.concatenate(z_l); cc = np.concatenate(c_l)
    o = np.argsort(-zc)
    buf = np.zeros((GRID_H * GRID_W, 3), np.uint8)
    hit = np.zeros(GRID_H * GRID_W, bool)
    buf[idx[o]] = cc[o]
    hit[idx[o]] = True
    return buf.reshape(GRID_H, GRID_W, 3), hit.reshape(GRID_H, GRID_W)


def banner(img, lines, color=(255, 255, 255)):
    f = img.copy()
    h, w = f.shape[:2]
    ov = f.copy()
    cv2.rectangle(ov, (0, 0), (w, 16 + 20 * len(lines)), (0, 0, 0), -1)
    f = cv2.addWeighted(ov, 0.55, f, 0.45, 0)
    for i, s in enumerate(lines):
        cv2.putText(f, s, (8, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    return f


# ---------------------------------------------------------------- pair 선정
def scan_pairs(split, n_scene, seed):
    base = os.path.join(ROOT, "da3", split)
    scenes = [x for x in sorted(os.listdir(base)) if x != "logs"]
    random.Random(seed).shuffle(scenes)
    out, kept = [], 0
    for sc in scenes:
        if kept >= n_scene:
            break
        sd = os.path.join(base, sc)
        cs = []
        for c in sorted(os.listdir(sd)):
            if c == "logs" or not os.path.isdir(os.path.join(sd, c)):
                continue
            s3 = read_sim3_file(split, sc, c)
            if s3 is not None:
                cs.append((c, s3[3]))
        if len(cs) < 2:
            continue
        kept += 1
        for (a, ra), (b, rb) in itertools.combinations(cs, 2):
            out.append((max(ra, rb), sc, a, b))
    return out, kept


# ---------------------------------------------------------------- 본체
def make_video(split, scene, clips, out_path, fps, radius, csv_rows, tag):
    gt = read_gt_cam(split, scene)
    A = read_clip(split, scene, clips[0], gt, "file")
    B = read_clip(split, scene, clips[1], gt, "file")
    n = min(len(A["rgb"]), len(B["rgb"]), A["dep"].shape[0], B["dep"].shape[0])
    W, H = GRID_W * 3 + 8, GRID_H * 2 + 4
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    hs, vs = np.full((GRID_H, 4, 3), 255, np.uint8), np.full((4, W, 3), 255, np.uint8)
    info = {c["clip"]: c for c in (A, B)}

    for t in range(n):
        rows = []
        for T, S in ((A, B), (B, A)):
            gtf = banner(T["rgb"][t], [f"GT  {T['clip'][-9:]}   frame {t:02d}"], (0, 255, 255))
            panes = [gtf]
            for var, col in (("file", (0, 165, 255)), ("gtrot", (0, 255, 0))):
                s3S = S["file3"][:3] if var == "file" else S["gt3"][:3]
                s3T = T["file3"][:3] if var == "file" else T["gt3"][:3]
                r, hit = render_swap(S, T, t, s3S, s3T, radius)
                mae = float(np.abs(T["rgb"][t][hit].astype(np.float32)
                                   - r[hit].astype(np.float32)).mean()) if hit.any() else float("nan")
                cov = float(hit.mean())
                csv_rows.append(dict(tag=tag, variant=var, scene=scene, target=T["clip"][-9:],
                                     source=S["clip"][-9:], frame=t, mae=mae, coverage=cov))
                panes.append(banner(r, [f"render[{var}]: {T['clip'][-9:]} pose x {S['clip'][-9:]} scene",
                                        f"MAE {mae:6.2f}   coverage {cov*100:5.1f}%"], col))
            rows.append(np.concatenate([panes[0], hs, panes[1], hs, panes[2]], axis=1))
        vw.write(np.concatenate([rows[0], vs, rows[1]], axis=0))
    vw.release()
    return n, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="whuman")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--clips", nargs=2, default=None, help="clip 접미사 2개 (예: 01_24mm 05_24mm)")
    ap.add_argument("--scan", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out-dir",
                    default=os.path.join(HERE, "results", "scene_decoupled", "pair_scene_swap"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.scene and args.clips:
        full = [c if c.startswith(args.scene) else f"{args.scene}_{c}" for c in args.clips]
        jobs, note = [("manual", args.scene, full)], []
    else:
        pairs, kept = scan_pairs(args.split, args.scan, args.seed)
        pairs.sort(key=lambda x: x[0])
        best, worst = pairs[0], pairs[-1]
        r = np.array([p[0] for p in pairs])
        note = [f"scan: scene={kept}  moving-moving pair={len(pairs)}",
                f"  resid_max(pair)  p50 {np.median(r):.4f}  p90 {np.percentile(r,90):.4f}  "
                f"p99 {np.percentile(r,99):.4f}  min {r.min():.4f}  max {r.max():.4f}", "",
                f"best   resid_max={best[0]:.4f}  {best[1]}  {best[2][-9:]} / {best[3][-9:]}",
                f"worst  resid_max={worst[0]:.4f}  {worst[1]}  {worst[2][-9:]} / {worst[3][-9:]}"]
        jobs = [("best", best[1], [best[2], best[3]]),
                ("worst", worst[1], [worst[2], worst[3]])]
        print("\n".join(note))

    csv_rows, diag = [], []
    for tag, sc, cl in jobs:
        out = os.path.join(args.out_dir, f"{tag}__{sc}__{cl[0][-9:]}_x_{cl[1][-9:]}.mp4")
        n, info = make_video(args.split, sc, cl, out, args.fps, args.radius, csv_rows, tag)
        for c in info.values():
            diag.append(f"  [{tag:>6}] {c['clip'][-9:]}  file resid {c['file3'][3]:.4f}   "
                        f"gtrot resid {c['gt3'][3]:.4f}   gtrot R spread {c['gt3'][4]:6.2f} deg   "
                        f"straightness(sv2/sv1) {c['straightness']:.4f}   "
                        f"R(file) vs R(gtrot) {_ang(c['file3'][1] @ c['gt3'][1].T):7.2f} deg")
        print(f"[{tag}] {sc}  {cl[0][-9:]} <-> {cl[1][-9:]}  {n} frames -> {out}")

    keys = ["tag", "variant", "scene", "target", "source", "frame", "mae", "coverage"]
    with open(os.path.join(args.out_dir, "per_frame.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in csv_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    L = ["# umeyama 정렬 검증 — clip pair scene swap 렌더", ""] + note
    L += ["", "## clip 별 sim3 진단", "   straightness = 궤적 중심 좌표의 sv2/sv1. 0 이면 완전 직선",
          "   -> 직선이면 umeyama 의 R 이 축 회전 1 자유도만큼 미결정이라 file R 이 아무 값이나 된다"] + diag
    L += ["", "## 프레임별 MAE (rendered 픽셀 위)"]
    for tag in dict.fromkeys(r["tag"] for r in csv_rows):
        for var in ("file", "gtrot"):
            for (tg, sr) in dict.fromkeys((r["target"], r["source"]) for r in csv_rows
                                          if r["tag"] == tag):
                v = [r for r in csv_rows if r["tag"] == tag and r["variant"] == var
                     and r["target"] == tg and r["source"] == sr]
                if not v:
                    continue
                m = np.array([x["mae"] for x in v], float)
                c = np.array([x["coverage"] for x in v], float)
                L.append(f"  [{tag:>6}][{var:>5}] {tg} pose x {sr} scene   MAE f0 {m[0]:6.2f}  "
                         f"median {np.nanmedian(m):6.2f}  last {m[-1]:6.2f}   "
                         f"coverage median {np.median(c)*100:5.1f}%")
    txt = "\n".join(L) + "\n"
    open(os.path.join(args.out_dir, "selection.md"), "w").write(txt)
    print("\n" + txt + "saved -> " + args.out_dir)


if __name__ == "__main__":
    main()
