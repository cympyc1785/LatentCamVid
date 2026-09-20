"""예측 궤적의 **충돌률**을 두 게이지로 잰다 — G1(depth shell 재투영) 과 kNN(점군 최근접거리).

두 판정은 같은 질문("카메라가 물체 안/뒤로 들어갔나")에 다른 증거를 쓴다.

  **G1** (`lbm/gates.behind_surface_frames`) — 카메라 중심 p 를 소스 프레임 t 에 투영해
        `z_cam(p) + clear·S > depth_t(u,v) + margin·S` 면 그 프레임에서 봤을 때 p 가 표면 **너머**다.
        임계가 `margin_frac` 하나뿐이라 손으로 고를 게 없고, "어느 소스 프레임이 반박하나"까지
        나온다. 대신 **화면 밖은 못 본다** — 뒤로 크게 빠지거나 바닥 밑으로 내려가면 투영이
        화면을 벗어나 아예 채점이 안 된다 (`lbm/gates.elevation_profile` docstring 의 실측).

  **kNN** — 관측 depth 를 전 프레임 unproject 해 점군 X 를 만들고
        `d_k(f) = ‖p(f) − x‖ 의 k 번째 최솟값 / S`. 화면 밖이어도 재지고, 값이 연속량이라
        "얼마나 아슬아슬했나"가 남는다. 대신 **임계 r 을 밖에서 정해야** 하고 그 r 은 점군
        밀도(=`--stride`)에 딸려 있다. 그래서 여기서는 r 을 GT 위에서 **캘리브레이션**한다:
        `r = quantile(GT 의 궤적별 최소 d_k, --knn_target_rate)` → 정의상 GT 가 그 비율로 걸린다.
        k=1 이 아니라 k=10 인 이유는 depth 경계의 떠 있는 점 하나가 판정을 뒤집지 않게 하려는 것
        (`corpus_clearance_probe.py` 는 k=1 이라 edge 필터에 더 민감하다).

궤적 판정은 둘 다 **49프레임 중 하나라도 걸리면 충돌**이다 (프레임 단위 비율은 따로 낸다).

**단위 호(arc) 정규화** — 위 두 비율은 "얼마나 움직였나"를 안 본다. 거의 정지한 궤적은 부딪힐
기회가 없어서 그냥 낮게 나오고, 그게 안전으로 읽힌다. 그래서 `충돌 프레임 수 / 이동거리` 를
같이 낸다:

  `arc_u = Σ_f ‖p(f+1) − p(f)‖ / S`      # 궤적 총 이동거리, 단위 u (1 u ≜ S)
  `knn_per_arc = knn_frames / arc_u`      # u 당 충돌 프레임 수, 단위 1/u

집계는 두 가지를 같이 찍는다 — 둘이 갈리면 소수의 짧은 궤적이 평균을 끌고 있다는 뜻이다:
  · `mean` = 엔트리별 비의 평균 (사용자가 요청한 정의). 짧은 궤적이 비를 폭발시키므로
    `arc_u < --arc_floor_u` 인 엔트리는 **빼고** 그 개수를 `n_static` 으로 남긴다.
  · `pooled` = `Σ knn_frames / Σ arc_u`. 이동거리로 가중돼 짧은 궤적에 안 흔들린다.

**게이지**: `arc_u` 는 충돌 판정과 **정확히 같은 좌표에서** 잰다. eval JSON 의 world pose 는
`gt_cv[0] @ pred_rel` 이라 `|Δp| = |Δt_rel|` (회전은 노름 보존) 이고, kNN 도 그 `p` 를 씬 점군에
그대로 대고 잰다. 즉 `--raw` GenDoP arm 의 DataDoP 단위를 씬 단위로 취급하는 해석이 분자·분모에
동일하게 걸려 있다 — `arc_u` 의 arm 간 비교는 충돌률의 arm 간 비교와 **같은 정도로** 유효하다.

좌표: eval 폴더의 `*_transforms_{ref,pred}.json` 은 **DA3 world c2w (OpenGL)** 이고 스케일도
recon 그대로다 (실측: `transforms_ref` 의 frame0 이 `recon_and_seg/<video>/cameras.npz`
`cam_c2w[0]` 과 소수점까지 같다). 그래서 되살릴 배율이 없다 — 카메라 중심을 그대로 쓰고
정규화 분모 `S` 만 `scene_graph.json` 에서 읽는다 (vista 코퍼스는 `avg_scale/<entry>.json`
과 같은 값이다).

**GenDoP arm 주의**: `gendop_preds_to_eval_dir.py` 가 `rmax` 를 GT 에서 빌려오므로 궤적의
**최대 이동량이 GT 와 같게** 고정돼 있다. 크기가 아니라 모양만 자기 것이다.

사용:
    cd models/Planner/CinemaTraj
    OPENCV_IO_ENABLE_OPENEXR=1 python scripts/eval_collision_rate.py \
        --corpus vista_d121 --out results/20260907_d159_collision
    # 임계를 이미 정했으면
    ... --knn_r 0.0126
    # dynpose d200 5 arm — subject_in_frame 과 같은 200 씬 표본으로
    OPENCV_IO_ENABLE_OPENEXR=1 python scripts/eval_collision_rate.py \
        --corpus dynpose_d200 --scenes ../../../../tmp/d205/sample200_scenes.txt \
        --out results/20260920_d205_collision_d200
"""
import sys
from argparse import ArgumentParser
from json import dump as json_dump, load as json_load
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.gates import behind_profile                                          # noqa: E402
from scene_graph.io import load_scene                                         # noqa: E402

EVAL_DATA = "/data1/cympyc1785/data/Vista4D-Eval-Data"
VISTA4D_ROOT = path.normpath(path.join(CINEMATRAJ_ROOT, "..", "..", "..", "..",
                                       "video_generation", "models", "Vista4D"))
LATENTCAM = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
D156 = path.join(CINEMATRAJ_ROOT, "results/20260906_d156_gendop_d121")
D200 = path.join(LATENTCAM, "eval_my")

# `eval_data` 는 **`eval_data/` 의 부모**다 — `scene_graph/io.py:88` 이 다시 붙인다.
# (`run_gendop_eval.py:60-63` 이 같은 함정을 기록해 뒀다. 거기 상수는 `eval_data` 까지라
#  score 단계에서 `path.dirname` 을 쳐서 넘긴다.)
# arm 라벨은 `tmp/d156/run_clatr_d121.py` 와 같은 규약이다 (D124 = da3+molmo2, D133 = molmo2 단독).
CORPORA = {
    "vista_d121": {
        "prefix": "vista4d",
        "eval_data": EVAL_DATA,
        "graph_root": path.join(CINEMATRAJ_ROOT, "out"),
        "ref": path.join(LATENTCAM,
                         "eval_my/20260904_165917_vista4d_d121_da3_t128__epoch100__seed42"),
        "arms": [
            ("gt", path.join(LATENTCAM,
                             "eval_my/20260904_165917_vista4d_d121_da3_t128__epoch100__seed42"),
             "ref"),                                        # pred 자리에 ref = 뱅크 GT 궤적
            ("d123_da3", path.join(
                LATENTCAM, "eval_my/20260904_165917_vista4d_d121_da3_t128__epoch100__seed42"),
             "pred"),
            ("d133_molmo2", path.join(LATENTCAM, "eval_my/d133_d121_molmo2_nogeo__last"), "pred"),
            ("d124_da3_molmo2", path.join(
                LATENTCAM, "eval_my/20260904_185554_vista4d_d121_molmo2__epoch100__seed42"),
             "pred"),
            # `_raw` = GenDoP 이 낸 크기 그대로. 접미사 없는 폴더는 `rmax` 를 GT 것으로 바꿔
            # 심은 판본이라 **최대 변위가 GT 와 같아진다** — 충돌률은 절대 크기에 반응하므로
            # 그걸 쓰면 GenDoP 이 아니라 GT 를 재게 된다 (d121 p49 기준 median x2.48 확대).
            # 2026-09-07 사용자 지시로 raw 를 기본으로 바꿨다.
            ("gendop_text_p49", path.join(D156, "eval_dir_gendop_text_p49_raw"), "pred"),
            ("gendop_rgbd_p49", path.join(D156, "eval_dir_gendop_rgbd_p49_raw"), "pred"),
            ("gendop_gdstyle_p49",
             path.join(D156, "eval_dir_gendop_rgbd_p49_gdstyle_raw"), "pred"),
            # 위 `_p49` 세 줄은 대조군이다 — `--pose_length 49` 가 `--forbid_eos` 를 켜서 pose
            # 30..48 이 분포 밖(릴리스 ckpt 는 30 pose 학습)이라 충돌 판정도 그 발산을 재게 된다.
            # 아래 두 줄이 판독 경로: 30 pose 를 GenDoP 자신의 보간기로 49프레임화한 것.
            # text_rgbd 한 ckpt, 텍스트 조건만 둘 (우리 D121 캡션 / GenDoP 학습 분포 문장).
            # `_noscale` = scale 토큰(`coords[:,9]`) 을 되나눠 **공식 배포 판본**과 크기를 맞춘
            # 것이다. 배포 `eval.py:239 pose_normalize` 는 scale 이 안 걸린 pose 로 pred JSON 을
            # 쓰고 scale 은 궤적 PNG 에만 쓴다. 되나누면 궤적이 median 1.99배 커지므로 절대
            # 크기에 반응하는 충돌률이 그만큼 달라진다 (2026-09-07 사용자 지시).
            ("gendop_rgbd_p30_slerp",
             path.join(D156, "eval_dir_gendop_rgbd_raw_slerp_noscale"), "pred"),
            ("gendop_gdstyle_p30_slerp",
             path.join(D156, "eval_dir_gendop_rgbd_gdstyle_raw_slerp_noscale"), "pred"),
        ],
    },
    # D200 pooled(d185+d199) 코퍼스의 5 arm. recon 은 `DynPose-100K/eval_data` 에 있고
    # scene_graph 는 `out_dynpose` 다 (`run_gendop_eval.py` 의 `dynpose_d200` 과 같은 루트).
    # 여기서는 `cloud.npz` 를 읽지 않는다 — 점군을 recon depth 에서 직접 세우고
    # `scene_graph.json` 에서는 `S` 만 읽으므로 D178 의 cloud 삭제와 무관하다.
    # 5,144 엔트리 × 1,017 씬 전량은 비싸므로 `--scenes` 로 D205 의 200 씬 표본만 재는 것을 권한다
    # (subject_in_frame 표와 같은 표본이라 열을 나란히 놓을 수 있다).
    "dynpose_d200": {
        "prefix": "dynpose",
        "eval_data": "/data1/cympyc1785/data/DynPose-100K",
        "graph_root": path.join(CINEMATRAJ_ROOT, "out_dynpose"),
        "ref": path.join(D200, "20260918_140904_dynpose_d200_da3__last"),
        "arms": [
            ("gt", path.join(D200, "20260918_140904_dynpose_d200_da3__last"), "ref"),
            ("d200_da3", path.join(D200, "20260918_140904_dynpose_d200_da3__last"), "pred"),
            ("d200_molmo2", path.join(D200, "20260918_140909_dynpose_d200_molmo2_da3__last"),
             "pred"),
            ("d200_molmo2_l21",
             path.join(D200, "20260918_140914_dynpose_d200_molmo2_l21_da3__last"), "pred"),
            ("d200_molmo2_dec",
             path.join(D200, "20260918_140919_dynpose_d200_molmo2_dec_da3__last"), "pred"),
            # ⑤ 만 run 디렉토리 날짜가 다르다 (09-19 재기동본).
            ("d200_molmo2_dec_l21",
             path.join(D200, "20260919_172530_dynpose_d200_molmo2_dec_l21_da3__last"), "pred"),
        ],
    },
}

# OpenGL(c2w, +Y up / −Z fwd) → OpenCV. 중심(translation)은 안 바뀌므로 판정에는 영향이 없지만,
# 나중에 회전을 쓰는 열이 붙을 때를 위해 여기서 규약을 못 박아 둔다.
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_poses(folder: str, name: str, kind: str):
    """`<folder>/test/<name>_transforms_{ref,pred}.json` → (F,4,4) OpenCV world c2w."""
    file_path = path.join(folder, "test", f"{name}_transforms_{kind}.json")
    if not path.isfile(file_path):
        return None
    with open(file_path, encoding="utf-8") as file:
        frames = json_load(file)["frames"]
    poses = np.asarray([f["transform_matrix"] for f in frames], dtype=np.float64)
    return poses @ GL2CV


def scene_points(recon, stride: int, edge_thr: float):
    """전 프레임 depth → (M,3) world 점군. 하늘·무효 depth·깊이 불연속 픽셀은 뺀다.

    불연속을 빼는 이유는 `corpus_clearance_probe.py` 와 같다 — 물체 경계에서 두 표면 사이에
    붕 뜬 점이 생겨 최근접거리를 실제보다 작게 만든다. depth 는 z-depth 로 본다.
    """
    depths, sky = recon["depths"], recon["sky_mask"]
    num_frames, height, width = depths.shape
    vs = np.arange(0, height, stride)
    us = np.arange(0, width, stride)
    grid = np.stack(np.meshgrid(us + 0.5, vs + 0.5, indexing="xy"), axis=-1)  # (V,U,2)
    pix = np.concatenate([grid.reshape(-1, 2), np.ones((grid[..., 0].size, 1))], axis=1)

    chunks = []
    for t in range(num_frames):
        z = depths[t].astype(np.float64)
        ok = np.isfinite(z) & (z > 0) & ~sky[t]
        log_z = np.log(np.clip(z, 1e-6, None))
        grad = np.zeros_like(log_z)
        dv = np.abs(log_z[1:, :] - log_z[:-1, :])
        grad[1:, :] = np.maximum(grad[1:, :], dv)
        grad[:-1, :] = np.maximum(grad[:-1, :], dv)
        du = np.abs(log_z[:, 1:] - log_z[:, :-1])
        grad[:, 1:] = np.maximum(grad[:, 1:], du)
        grad[:, :-1] = np.maximum(grad[:, :-1], du)
        ok &= grad <= edge_thr

        zz = z[np.ix_(vs, us)].reshape(-1)
        mm = ok[np.ix_(vs, us)].reshape(-1)
        rays = pix @ np.linalg.inv(recon["K"][t]).T                # (P,3), z=1 평면
        cam = rays * zz[:, None]
        world = cam @ recon["cam_c2w"][t][:3, :3].T + recon["cam_c2w"][t][:3, 3]
        chunks.append(world[mm])
    return np.concatenate(chunks, axis=0)


def main(args):
    corpus = CORPORA[args.corpus]
    # 예전 코퍼스 dict 에는 이 두 키가 없었다 — 없으면 vista 상수로 떨어져 동작이 같다.
    eval_data = corpus.get("eval_data", EVAL_DATA)
    graph_root = corpus.get("graph_root", path.join(CINEMATRAJ_ROOT, "out"))
    arms = list(corpus["arms"])
    for spec in args.extra_eval_dir:                    # LABEL=DIR, 반복 가능
        label, _, folder = spec.partition("=")
        assert folder, f"--extra_eval_dir 는 LABEL=DIR 꼴이어야 한다: {spec}"
        arms.append((label, folder, "pred"))

    names = [l.strip() for l in open(path.join(corpus["ref"], "test_valid.txt"),
                                     encoding="utf-8") if l.strip()]
    keep = None
    if args.scenes:
        keep = {l.strip() for l in open(args.scenes, encoding="utf-8") if l.strip()}
        names = [n for n in names
                 if n[len(corpus["prefix"]) + 1:].rsplit("_", 1)[0] in keep]
    if args.limit:
        names = names[:args.limit]
    by_scene = {}
    for name in names:
        scene = name[len(corpus["prefix"]) + 1:].rsplit("_", 1)[0]
        by_scene.setdefault(scene, []).append(name)
    # 씬이 많으면 목록을 통째로 찍는 건 로그만 불린다.
    shown = sorted(by_scene) if len(by_scene) <= 12 else f"{sorted(by_scene)[:6]} ..."
    if keep is not None:
        print(f"{'scenes file':<16}{args.scenes}  ({len(by_scene)}/{len(keep)} 씬 매칭)")
    print(f"{'corpus':<16}{args.corpus}\n{'entries':<16}{len(names)}"
          f"\n{'scenes':<16}{len(by_scene)}  {shown}")
    print(f"{'G1':<16}margin {args.behind_margin_frac:g}·S  clear {args.behind_clear_frac:g}·S  "
          f"min_zcam {args.behind_min_zcam:g}·S  src_frames {args.behind_src_frames}  "
          f"radius {args.behind_radius_px}px  time_match {args.time_match}")
    print(f"{'kNN':<16}k={args.knn_k}  stride {args.stride}  edge_thr {args.edge_thr:g}\n")

    from scipy.spatial import cKDTree

    rows = []                                                     # 엔트리 × arm 한 줄씩
    for scene in sorted(by_scene):
        with open(path.join(graph_root, scene, "scene_graph.json"),
                  encoding="utf-8") as file:
            scale = float(json_load(file)["scale"]["S"])
        recon = load_scene(eval_data, scene, VISTA4D_ROOT)
        num_src = len(recon["cam_c2w"])
        src_frames = sorted(set(np.rint(np.linspace(
            0, num_src - 1, min(args.behind_src_frames, num_src))).astype(int).tolist()))

        tree = None
        if args.metric in ("knn", "both"):
            points = scene_points(recon, args.stride, args.edge_thr)
            tree = cKDTree(points)
            print(f"[{scene}] S {scale:.4f}  points {len(points):,}  "
                  f"entries {len(by_scene[scene])}", flush=True)

        for arm, folder, kind in arms:
            done = 0
            for name in by_scene[scene]:
                poses = load_poses(folder, name, kind)
                if poses is None:
                    rows.append({"scene": scene, "entry": name, "arm": arm, "missing": 1})
                    continue
                row = {"scene": scene, "entry": name, "arm": arm, "missing": 0,
                       "n_poses": len(poses)}
                # 이동거리는 점군이 필요 없다 — metric 설정과 무관하게 항상 잰다.
                row["arc_u"] = float(np.linalg.norm(
                    np.diff(poses[:, :3, 3], axis=0), axis=1).sum() / scale)
                if args.metric in ("g1", "both"):
                    bad, worst = behind_profile(
                        poses, recon["depths"], K=recon["K"], cam_c2w=recon["cam_c2w"],
                        sky_mask=recon["sky_mask"], scale=scale, frames=src_frames,
                        margin_frac=args.behind_margin_frac,
                        clear_frac=args.behind_clear_frac,
                        radius_px=args.behind_radius_px,
                        dynamic_mask=recon["dynamic_mask"] if args.time_match else None,
                        time_match=args.time_match,
                        min_zcam_frac=args.behind_min_zcam)
                    row["g1_frames"] = int(bad)
                    row["g1_worst_src"] = int(worst)
                if tree is not None:
                    dist, _ = tree.query(poses[:, :3, 3], k=args.knn_k, workers=-1)
                    row["knn_min"] = float(dist[:, -1].min() / scale)
                    row["knn_med"] = float(np.median(dist[:, -1]) / scale)
                    row["knn_dists"] = (dist[:, -1] / scale).astype(np.float32)
                rows.append(row)
                done += 1
            print(f"  {arm:<20}{done:>5} entries", flush=True)

    # ---- kNN 임계. 안 주면 GT 위에서 목표 충돌률이 나오도록 캘리브레이션한다 ----
    knn_r = args.knn_r
    if args.metric in ("knn", "both") and knn_r is None:
        gt = np.asarray([r["knn_min"] for r in rows
                         if r["arm"] == args.knn_calib_arm and not r["missing"]])
        knn_r = float(np.quantile(gt, args.knn_target_rate))
        print(f"\n{'knn_r':<16}{knn_r:.6f}  "
              f"(= arm '{args.knn_calib_arm}' 의 궤적별 최소 d_{args.knn_k} 의 "
              f"{args.knn_target_rate:.1%} 분위, n={len(gt)})")

    # 엔트리별 충돌 프레임 수는 knn_r 이 정해진 뒤에야 센다 (r 이 GT 캘리브레이션 산물).
    if args.metric in ("knn", "both"):
        for row in rows:
            if not row["missing"]:
                row["knn_frames"] = int((row["knn_dists"] < knn_r).sum())

    def per_arc(sub, key):
        """`<key>_frames / arc_u` 의 mean(정지 궤적 제외) 과 pooled(이동거리 가중)."""
        moving = [r for r in sub if r["arc_u"] >= args.arc_floor_u]
        arcs = np.asarray([r["arc_u"] for r in sub])
        out = {f"{key}_per_arc_mean": round(float(np.mean(
                   [r[f"{key}_frames"] / r["arc_u"] for r in moving])), 4) if moving else None,
               f"{key}_per_arc_pooled": round(float(
                   sum(r[f"{key}_frames"] for r in sub) / arcs.sum()), 4) if arcs.sum() else None,
               "n_static": len(sub) - len(moving),
               "arc_u_med": round(float(np.median(arcs)), 4),
               "arc_u_mean": round(float(arcs.mean()), 4)}
        return out

    def summarize(pool):
        """arm -> 지표 dict. `pool` 이 전체면 코퍼스 요약, scene 하나면 scene 요약."""
        out = {}
        for arm, _folder, _kind in arms:
            sub = [r for r in pool if r["arm"] == arm and not r["missing"]]
            if not sub:
                continue
            entry = {"n": len(sub), "missing": sum(1 for r in pool
                                                   if r["arm"] == arm and r["missing"])}
            if args.metric in ("g1", "both"):
                entry["g1_collision_rate"] = round(
                    float(np.mean([r["g1_frames"] > 0 for r in sub])), 4)
                entry["g1_frame_frac"] = round(
                    float(np.mean([r["g1_frames"] / r["n_poses"] for r in sub])), 4)
                entry["g1_worst_src_max"] = int(max(r["g1_worst_src"] for r in sub))
                entry.update(per_arc(sub, "g1"))
            if args.metric in ("knn", "both"):
                mins = np.asarray([r["knn_min"] for r in sub])
                frame = np.concatenate([r["knn_dists"] for r in sub])
                entry["knn_collision_rate"] = round(float(np.mean(mins < knn_r)), 4)
                entry["knn_frame_frac"] = round(float(np.mean(frame < knn_r)), 4)
                entry["knn_min_p01"] = round(float(np.quantile(mins, 0.01)), 5)
                entry["knn_min_med"] = round(float(np.median(mins)), 5)
                entry.update(per_arc(sub, "knn"))
            out[arm] = entry
        return out

    summary = summarize(rows)
    # scene 별 재집계 — 코퍼스 평균은 엔트리 수가 많은 scene 이 끌고 간다. 씬마다 점군 밀도와
    # 소스 시차가 달라서 같은 knn_r 이 씬별로 다른 엄격도로 작동하므로 갈라서도 봐야 한다.
    by_scene = ({s: summarize([r for r in rows if r["scene"] == s])
                 for s in sorted({r["scene"] for r in rows})} if args.per_scene else {})

    makedirs(args.out, exist_ok=True)
    config = {"corpus": args.corpus, "metric": args.metric, "n_entries": len(names),
              "n_scenes": len(by_scene), "scenes_file": args.scenes,
              "eval_data": eval_data, "graph_root": graph_root,
              "arms": [[a, f, k] for a, f, k in arms],
              "knn": {"k": args.knn_k, "r_normalized": knn_r, "stride": args.stride,
                      "edge_thr": args.edge_thr, "calib_arm": args.knn_calib_arm,
                      "target_rate": args.knn_target_rate,
                      "calibrated": args.knn_r is None},
              "g1": {"margin_frac": args.behind_margin_frac,
                     "clear_frac": args.behind_clear_frac,
                     "min_zcam_frac": args.behind_min_zcam,
                     "radius_px": args.behind_radius_px,
                     "src_frames": args.behind_src_frames, "time_match": args.time_match},
              "arc": {"floor_u": args.arc_floor_u,
                      "definition": "sum |dp| / S over n_poses-1 steps",
                      "per_arc_unit": "collision frames per u of camera path"},
              "rule": "49프레임 중 하나라도 걸리면 충돌"}
    with open(path.join(args.out, "collision_rate.json"), "w", encoding="utf-8") as file:
        json_dump({"config": config, "summary": summary, "by_scene": by_scene}, file,
                  ensure_ascii=False, indent=1)
    columns = ["scene", "entry", "arm", "missing", "n_poses", "arc_u",
               "g1_frames", "g1_worst_src", "knn_frames", "knn_min", "knn_med"]
    with open(path.join(args.out, "collision_per_entry.csv"), "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")
        for row in rows:
            file.write(",".join(str(row.get(c, "")) for c in columns) + "\n")

    print(f"\n{'arm':<20}{'n':>5}{'G1 rate':>10}{'G1 frame':>10}"
          f"{'kNN rate':>10}{'kNN frame':>11}{'d_k p01':>10}{'d_k med':>10}")
    for arm, entry in summary.items():
        print(f"{arm:<20}{entry['n']:>5}"
              f"{entry.get('g1_collision_rate', float('nan')):>10.4f}"
              f"{entry.get('g1_frame_frac', float('nan')):>10.4f}"
              f"{entry.get('knn_collision_rate', float('nan')):>10.4f}"
              f"{entry.get('knn_frame_frac', float('nan')):>11.4f}"
              f"{entry.get('knn_min_p01', float('nan')):>10.5f}"
              f"{entry.get('knn_min_med', float('nan')):>10.5f}")

    # 단위 호 정규화 — 위 표는 "안 움직여서 안 부딪혔다"를 안전으로 읽는다. 여기는 이동거리로 나눈다.
    print(f"\n[per unit arc]  충돌 프레임 수 / arc_u  (단위 1/u, arc_floor {args.arc_floor_u:g}u)")
    print(f"{'arm':<20}{'arc_u med':>11}{'arc_u mean':>12}{'n_static':>10}"
          f"{'kNN/arc mean':>14}{'kNN/arc pool':>14}{'G1/arc mean':>13}{'G1/arc pool':>13}")
    fmt = lambda v: f"{v:>14.4f}" if isinstance(v, float) else f"{'-':>14}"
    for arm, entry in summary.items():
        print(f"{arm:<20}{entry.get('arc_u_med', float('nan')):>11.4f}"
              f"{entry.get('arc_u_mean', float('nan')):>12.4f}"
              f"{entry.get('n_static', -1):>10}"
              + fmt(entry.get("knn_per_arc_mean")) + fmt(entry.get("knn_per_arc_pooled"))
              + fmt(entry.get("g1_per_arc_mean"))[1:] + fmt(entry.get("g1_per_arc_pooled"))[1:])
    # 씬 블록은 JSON 에 전부 남기되 **출력만** 자른다 — 200 씬이면 표가 200개가 된다.
    shown_scenes = list(by_scene.items())[:args.per_scene_max]
    if len(shown_scenes) < len(by_scene):
        print(f"\n[per-scene] {len(by_scene)} 씬 중 앞 {len(shown_scenes)} 개만 출력 "
              f"(전량은 collision_rate.json 의 by_scene)")
    for scene, per_arm in shown_scenes:
        n = max(e["n"] for e in per_arm.values())
        print(f"\n[scene {scene}]  n={n}")
        print(f"{'arm':<26}{'kNN rate':>10}{'kNN frame':>11}{'d_k med':>10}"
              f"{'arc_u med':>11}{'kNN/arc mean':>14}{'kNN/arc pool':>14}")
        for arm, entry in per_arm.items():
            print(f"{arm:<26}"
                  f"{entry.get('knn_collision_rate', float('nan')):>10.4f}"
                  f"{entry.get('knn_frame_frac', float('nan')):>11.4f}"
                  f"{entry.get('knn_min_med', float('nan')):>10.5f}"
                  f"{entry.get('arc_u_med', float('nan')):>11.4f}"
                  + fmt(entry.get("knn_per_arc_mean")) + fmt(entry.get("knn_per_arc_pooled")))
    print(f"\n-> {path.join(args.out, 'collision_rate.json')}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="vista_d121", choices=sorted(CORPORA))
    parser.add_argument("--out", default=path.join(CINEMATRAJ_ROOT,
                                                   "results/20260907_d159_collision"))
    parser.add_argument("--metric", default="both", choices=("g1", "knn", "both"))
    parser.add_argument("--limit", default=0, type=int)          # smoke 용 엔트리 수 제한
    # 씬 표본 파일 (씬 이름 한 줄씩). subject_in_frame 표와 같은 표본을 쓰라고 있는 것이다.
    parser.add_argument("--scenes", default=None)
    # 코퍼스 dict 밖의 eval 폴더를 pred arm 으로 추가 (E.T./GenDoP 베이스라인).
    parser.add_argument("--extra_eval_dir", action="append", default=[],
                        metavar="LABEL=DIR")
    # --- kNN ---
    parser.add_argument("--knn_k", default=10, type=int)         # 경계 떠 있는 점 1개에 안 흔들리게
    parser.add_argument("--knn_r", default=None, type=float)     # 주면 캘리브레이션 안 한다
    parser.add_argument("--knn_calib_arm", default="gt")
    # 이동거리 하한 — 이보다 짧은 궤적은 비(1/u)가 폭발하므로 mean 에서 빼고 개수만 남긴다.
    parser.add_argument("--arc_floor_u", default=0.01, type=float)
    # scene 별 재집계 표 + JSON `by_scene`. 기본 on — 코퍼스 평균만 보면 엔트리 많은 씬에 가려진다.
    parser.add_argument("--per_scene", action="store_true", default=True)
    parser.add_argument("--no_per_scene", dest="per_scene", action="store_false")
    parser.add_argument("--per_scene_max", default=12, type=int)  # 출력만 자른다 (JSON 은 전량)
    parser.add_argument("--knn_target_rate", default=0.01, type=float)
    parser.add_argument("--stride", default=4, type=int)         # 점군 솎음 (밀도가 r 을 정한다)
    parser.add_argument("--edge_thr", default=0.05, type=float)  # |∇log z| 컷
    # --- G1 (기본값은 `fit_hole_ladder.py` 와 같되 clear_frac 만 0 = 순수 관통 판정) ---
    parser.add_argument("--behind_src_frames", default=49, type=int)
    parser.add_argument("--behind_margin_frac", default=0.02, type=float)
    parser.add_argument("--behind_clear_frac", default=0.0, type=float)
    parser.add_argument("--behind_radius_px", default=0, type=int)
    parser.add_argument("--behind_min_zcam", default=0.02, type=float)
    parser.add_argument("--time_match", action="store_true", default=False)
    parser.add_argument("--no_time_match", dest="time_match", action="store_false")
    main(parser.parse_args())
