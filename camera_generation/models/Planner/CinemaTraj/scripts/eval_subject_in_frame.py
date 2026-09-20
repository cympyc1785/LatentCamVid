"""eval 폴더(`test/<name>_transforms_pred.json`) 여러 개를 **같은 점군에 렌더해서** 비교한다.

왜 따로 있나: `verify.py` 는 `out/<video>/` 한 벌(`decision.json` + `poses.npz` + fingerprint)에
묶여 있어서 **모델이 예측한 궤적**에는 못 건다. 여기는 반대로 궤적의 출처를 안 따지고
`_transforms_pred.json` 만 받는다 — 우리 모델 / GenDoP / GT 를 한 표에 올리는 유일한 방법.

지표는 `verify.py:measure()` 의 2·1·3 을 그대로 쓴다 (같은 정의를 두 번 적지 않으려고 import 한다):

  subject_in_frame       subject 실루엣 **중심**이 중앙 `center_box`(0.80) 안에 든 프레임 비율
  hole_fraction          49프레임 평균 `1 - valid.mean()`   (맥락용 — 카메라가 얼마나 벗어났나)
  subject_pixel_coverage subject 면적비 median               (맥락용 — 0 이면 아예 안 보인다)

`--subject_occlusion` 을 켜면 **가림** 열이 붙는다 (`occlusion_series`, 렌더 2-pass):

  subject_visible_frac   가려지지 않은 실루엣 픽셀 비율의 프레임 median
  subject_visible_min    그 최악 프레임

위 세 지표로는 가림이 안 잡힌다 — `subject_in_frame` 은 **위치**(실루엣 중심이 상자 안인가),
`subject_pixel_coverage` 는 **크기**(화면 면적비)라 상판에 가려 반쯤 사라진 subject 도 둘 다
멀쩡히 통과한다. 기본값 off = 기존 표와 비트 단위로 같다.

**게이지 주의**: arm 마다 world 스케일이 다를 수 있다. GenDoP 을 `--no_rescale`(raw)로 변환하면
DataDoP 게이지가 그대로 남아 씬 단위와 무관한 크기로 렌더된다. 그 상태의 `subject_in_frame` 은
"구도가 맞나"와 "크기가 맞나"를 **같이** 재는 값이다. 크기를 GT 에서 빌린 arm 과 비교하려면
`gendop_preds_to_eval_dir.py --rescale` 로 만든 폴더를 따로 넘길 것.

subject 는 entry 마다 다르다 — 코퍼스의 `da3/prompts.json[<entry>]["variant_id"]` 앞머리
(`dyn_1__dolly_in__hole0.2` -> `dyn_1`)가 그 entry 가 조준한 노드다. 씬 단위로 고정하면
anchor 가 2개인 D84 코퍼스에서 절반이 엉뚱한 물체를 잰다.

env: `vista4d` (CloudRenderer).

예시:
    PY=/data1/cympyc1785/miniconda3/envs/vista4d/bin/python
    CUDA_VISIBLE_DEVICES=5 $PY scripts/eval_subject_in_frame.py \
        --ref_eval_dir /data1/.../latentcam/results/20260901_033720_dynpose_d84_k6_dionly \
        --name_prefix dynpose --cloud_root out_dynpose \
        --eval_data /data1/cympyc1785/data/DynPose-LBM \
        --corpus_root /data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose \
        --eval_dir ours=/data1/.../20260901_033720_dynpose_d84_k6_dionly \
        --eval_dir gendop_a=results/20260901_gendop_dynpose_val/eval_dir_case_a \
        --eval_dir gendop_b=results/20260901_gendop_dynpose_val/eval_dir_case_b \
        --out results/20260901_gendop_dynpose_val/subject_in_frame.json
"""
import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from glob import glob
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT, subject_point_mask  # noqa: E402
from lbm.render import add_cloud_source_args, open_renderer                        # noqa: E402
from scripts.build_candidate_board import subject_track_volume                     # noqa: E402
from verify import measure, render_plan                                            # noqa: E402

GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])   # eval JSON(OpenGL c2w) <-> OpenCV c2w, 자기역원


def load_c2w(json_path: str):
    """eval JSON -> OpenCV c2w (T,4,4). `gendop_preds_to_eval_dir.load_transforms` 와 같은 규약."""
    with open(json_path, encoding="utf-8") as file:
        data = json.load(file)
    return np.array([f["transform_matrix"] for f in data["frames"]], dtype=np.float64) @ GL2CV


def load_prompts(corpus_root: str, prefix: str, scene: str):
    """씬 하나의 `prompts.json`. entry 당 한 번씩 열면 875번 파싱하게 되므로 씬 단위로 캐시."""
    with open(path.join(corpus_root, prefix, scene, "da3", "prompts.json"), encoding="utf-8") as f:
        return json.load(f)


def entry_subject(prompts: dict, entry: str):
    """entry 가 조준한 노드 id. `variant_id` 앞머리가 그것이다 (`dyn_1__dolly_in__hole0.2`)."""
    return str(prompts[entry]["variant_id"]).split("__")[0]


def assert_corpus_matches(prompts: dict, ref_eval_dir: str, name: str, entry: str):
    """코퍼스 `prompts.json` 이 **이 eval 폴더를 만든 그 세대**인지 캡션으로 확인한다.

    entry 인덱스는 코퍼스 키가 아니다 — 재굽기하면 변이 수와 순서가 바뀌므로 같은 `27` 이
    다른 물체를 가리킨다. 그런데 `variant_id` 는 형식이 같아서 잘못된 세대를 읽어도 조용히
    `stat_4` 같은 그럴듯한 답이 나오고, subject 지표만 통째로 틀린다 (실제로 d121 eval 을
    재굽기된 `latentcam_da3` 로 재서 그렇게 됐다). eval 폴더의 `<name>_caption.json` 과
    코퍼스 캡션을 대조하면 이 클래스의 버그가 첫 entry 에서 죽는다.
    """
    cap_path = path.join(ref_eval_dir, "test", f"{name}_caption.json")
    if not path.isfile(cap_path):
        return
    with open(cap_path, encoding="utf-8") as file:
        cap = json.load(file)
    want = cap.get("Concise Interaction") if isinstance(cap, dict) else cap
    got = prompts[entry].get("prompt_camera_with_scene_video", {}).get("concise")
    assert want is None or got is None or str(want).strip() == str(got).strip(), (
        f"{name}: --corpus_root 세대가 --ref_eval_dir 과 다르다 (캡션 불일치)\n"
        f"  eval  : {str(want)[:120]}\n  corpus: {str(got)[:120]}")


def in_frame_at(centers: list, center_box: float):
    """`measure()` 와 같은 규칙을 **이미 뽑은 중심 좌표**에 다시 적용한다.

    `center_box` 를 쓸어보려고 49프레임을 다시 렌더할 이유가 없다 — `subject_in_frame` 은
    실루엣 중심 하나에만 걸린 임계값이라, 중심을 저장해 두면 상자 크기는 사후에 바꾼다.
    `None` 은 subject 픽셀이 0인 프레임(=화면 밖이 아니라 **소실**)이고, `measure()` 처럼
    상자 크기와 무관하게 실패로 센다.
    """
    lo, hi = (1 - center_box) / 2, 1 - (1 - center_box) / 2
    hit = [c is not None and lo <= c[0] <= hi and lo <= c[1] <= hi for c in centers]
    return float(np.mean(hit))


def occlusion_series(renderer, poses, rendered: list, subject_points, scale: float, args):
    """프레임별 subject 가시 비율 (1 − 가려진 실루엣 비율). `sample_camera_bank` 와 같은 2-pass.

    왜 두 번 렌더하나: `subject_pixel_coverage`(=실루엣 화면 면적비)는 **가림과 구분이 안 된다** —
    책상 밑으로 내려간 카메라는 상판에 subject 가 가려져도 면적비가 멀쩡히 나온다. 그래서
    subject 점만 그린 실루엣(`alone`)과 전체 렌더의 depth 를 비교해 "그려졌어야 하는데 앞에
    뭔가 있는" 픽셀을 센다 (`lbm/gates.py:338-349`, `scripts/sample_camera_bank.py:690-698` 과
    같은 식). `render.CloudRenderer.measure` 의 `num_subject_points` 판은 픽셀수를 점 개수로
    나눈 **밀도**라 비율이 아니므로 쓰지 않는다.

    분모는 그 프레임 실루엣 픽셀 수라 **화면 밖 성분은 안 들어간다** — 프레임을 벗어난 건
    `subject_in_frame` 이 잡는 몫이고 여기는 순수 가림만 잰다. 실루엣이 통째로 비면 0.

    두 번째 패스는 subject 점만 그리므로 첫 패스보다 훨씬 싸다 (점 수가 실루엣 몫뿐).
    """
    seen = []
    for f, shot in enumerate(rendered):
        alone = renderer.render(poses[f], frame=f, height=args.height, width=args.width,
                                temporal_persistence=args.temporal_persistence,
                                subset=subject_points)
        silhouette = alone["valid"]
        n_sil = int(silhouette.sum())
        if n_sil == 0:
            seen.append(0.0)
            continue
        drawn = silhouette & shot["valid"]
        occluded = drawn & (shot["depth"] < alone["depth"] - 0.02 * scale)
        seen.append(1.0 - float(occluded.sum()) / n_sil)
    return np.asarray(seen, dtype=np.float64)


def score(renderer, poses, args, scene, entry, name, label, subject_id,
          subject_points=None, scale: float = 1.0):
    """한 arm 의 49프레임을 렌더해 지표 한 줄로. `centers` 를 같이 실어 사후 sweep 을 가능하게.

    `subject_points` 를 주면 `subject_visible_frac` 열이 붙는다 (렌더 2-pass). 안 주면 열이
    안 붙고 기존 표와 비트 단위로 같다.
    """
    rendered = render_plan(renderer, poses, height=args.height, width=args.width,
                           temporal_persistence=args.temporal_persistence)
    holes, areas, centers, in_frame = measure(rendered, args.center_box)
    row = {"scene": scene, "entry": entry, "name": name, "arm": label,
           "subject_id": subject_id,
           "subject_in_frame": float(in_frame.mean()),
           "hole_fraction": float(holes.mean()),
           "subject_pixel_coverage": float(np.median(areas)),
           "subject_zero_frames": int((areas == 0).sum()),
           "centers": centers}
    if subject_points is not None:
        seen = occlusion_series(renderer, poses, rendered, subject_points, scale, args)
        # median 으로 접는다 — 한 프레임 스치는 가림에 열이 끌려가면 못 쓴다
        # (`sample_camera_bank.py:725-728` 과 같은 규칙). 최악 프레임은 따로 남긴다.
        row["subject_visible_frac"] = float(np.median(seen))
        row["subject_visible_min"] = float(seen.min())
        row["subject_visible_mean"] = float(seen.mean())
    return row


def main(args):
    arms = dict(a.split("=", 1) for a in args.eval_dir)
    assert arms, "--eval_dir LABEL=DIR 를 하나 이상 줄 것"

    refs = sorted(glob(path.join(args.ref_eval_dir, "test",
                                 f"{args.name_prefix}_*_transforms_ref.json")))
    assert refs, f"{args.ref_eval_dir}/test 에 {args.name_prefix}_* ref 가 없다"
    by_scene = defaultdict(list)
    for ref_path in refs:
        name = path.basename(ref_path)[:-len("_transforms_ref.json")]
        entry = name.rsplit("_", 1)[1]
        scene = name[len(args.name_prefix) + 1:-(len(entry) + 1)]
        by_scene[scene].append((name, entry))

    # 씬 부분집합. 전량이 감당 안 될 때(d200 test 1,017씬 = 실측 429 s/씬 → 단일 GPU 121 h)
    # 표본을 **밖에서 정해** 넣는다. 표본 규칙(랜덤 / 운동 층화 / preset 한정)은 이 스크립트의
    # 관심사가 아니다 — 파일 하나로 받아야 "어떤 표본이었나"가 산출물 옆에 남는다.
    # 기본 None = 전량 = 기존 동작과 비트 단위로 같다.
    if args.scenes:
        with open(args.scenes, encoding="utf-8") as file:
            wanted = {line.strip() for line in file if line.strip()}
        missing = wanted - set(by_scene)
        by_scene = {s: v for s, v in by_scene.items() if s in wanted}
        assert by_scene, f"{args.scenes} 의 씬이 ref_eval_dir 에 하나도 없다"
        print(f"[scenes] {len(by_scene)}/{len(wanted)} 씬 선택"
              + (f"  (ref 에 없음 {len(missing)})" if missing else ""), flush=True)

    # 샤딩은 **씬 단위**다. entry 로 자르면 같은 씬의 점군을 샤드마다 다시 올리게 되는데,
    # 씬 하나 여는 비용(unproject)이 entry 하나 렌더보다 훨씬 크다. 기본값 1/0 은 전량 =
    # 기존 동작과 비트 단위로 같다. `--scenes` 를 **먼저** 거르고 샤딩은 그 뒤다 — 순서가
    # 반대면 샤드마다 표본 크기가 들쭉날쭉해진다.
    scenes = sorted(by_scene)
    if args.num_shards > 1:
        scenes = [s for i, s in enumerate(scenes) if i % args.num_shards == args.shard_id]
        print(f"[shard {args.shard_id}/{args.num_shards}] 씬 {len(scenes)}/{len(by_scene)}",
              flush=True)

    rows, skipped = [], []
    for scene in scenes:
        graph_path = path.join(args.cloud_root, scene, "scene_graph.json")
        # npz 모드는 cloud.npz 가 있어야 하고, memory 모드는 recon 만 있으면 된다.
        needed = [graph_path] + ([path.join(args.cloud_root, scene, "cloud.npz")]
                                 if args.cloud_source == "npz" else [])
        if not all(path.isfile(p) for p in needed):
            skipped += [f"{scene}_{e}: cloud/graph 없음" for _, e in by_scene[scene]]
            continue
        with open(graph_path, encoding="utf-8") as file:
            graph = json.load(file)
        renderer, recon = open_renderer(args, args.cloud_root, graph, video=scene)

        # 가림 판정의 depth 여유(0.02·S)는 씬 게이지를 따라야 한다 — `gates.evaluate` 와 같은 값.
        scene_scale = float(graph["scale"]["S"])

        prompts = load_prompts(args.corpus_root, args.name_prefix, scene)
        picks = sorted(by_scene[scene], key=lambda t: int(t[1]))
        if args.limit:                      # 프로브용 — 씬마다 앞에서 N entry 만
            picks = picks[:args.limit]
        # 처음·중간·끝 세 군데를 본다. 재굽기는 앞머리가 우연히 맞는 경우가 있어(entry 0 은
        # 언제나 첫 anchor 의 첫 preset) 한 개만 보면 통과해 버린다.
        for probe in {0, len(picks) // 2, len(picks) - 1}:
            assert_corpus_matches(prompts, args.ref_eval_dir, *picks[probe])
        for name, entry in picks:
            subject_id = entry_subject(prompts, entry)
            node = next((n for n in graph["nodes"] if n["id"] == subject_id), None)
            if node is None:
                skipped.append(f"{name}: 노드 {subject_id} 없음")
                continue
            subject_points = subject_point_mask(
                renderer.indices, subject_track_volume(recon, node)).cpu().numpy()
            renderer.set_subject(subject_points)
            # 가림 열을 끄면 `occl` 이 None 이라 2-pass 가 아예 안 돈다 (= 기존 동작).
            occl = subject_points if args.subject_occlusion else None

            for label, folder in arms.items():
                pred_path = path.join(folder, "test", f"{name}_transforms_pred.json")
                if not path.isfile(pred_path):
                    skipped.append(f"{name}/{label}: pred 없음")
                    continue
                poses = load_c2w(pred_path)
                assert len(poses) == renderer.num_frames, \
                    f"{pred_path}: {len(poses)} pose vs cloud {renderer.num_frames} 프레임"
                rows.append(score(renderer, poses, args, scene, entry, name,
                                  label, subject_id, subject_points=occl, scale=scene_scale))
            # GT(ref)는 천장. arm 과 같은 렌더 경로로 재서 비교 가능하게 둔다.
            if args.include_gt:
                poses = load_c2w(path.join(args.ref_eval_dir, "test",
                                           f"{name}_transforms_ref.json"))
                rows.append(score(renderer, poses, args, scene, entry, name,
                                  "gt", subject_id, subject_points=occl, scale=scene_scale))
            print(f"  {name:<48} {subject_id:<8} "
                  + "  ".join(f"{r['arm']} {r['subject_in_frame']:.3f}"
                              for r in rows if r["name"] == name), flush=True)
        del renderer

    labels = (["gt"] if args.include_gt else []) + list(arms)
    report = {"format": "subject_in_frame_v1", "center_box": args.center_box,
              "render_mode": "cloud" if args.temporal_persistence else "warp_1to1",
              "ref_eval_dir": args.ref_eval_dir, "arms": arms,
              "entries": len(refs), "subject_occlusion": bool(args.subject_occlusion),
              "num_shards": args.num_shards, "shard_id": args.shard_id,
              "skipped": skipped, "rows": rows}
    if args.out:
        makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=1)

    summarize(rows, labels, args, report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=1)

    if skipped:
        print(f"\nskipped {len(skipped)}")
        for s in skipped[:10]:
            print(f"  {s}")
    if args.out:
        print(f"\n-> {args.out}")


def summarize(rows, labels, args, report):
    """arm 요약표 + center_box sweep 을 찍고 `report["sweep"]` 을 채운다.

    샤드 병합(`--merge`)이 같은 표를 다시 그릴 수 있어야 해서 main 에서 떼어냈다 — 샤드마다
    찍힌 표를 사람이 눈으로 더할 수는 없다 (arm 평균은 샤드 크기 가중이라 단순 평균이 틀린다).
    """
    occl_on = any("subject_visible_frac" in r for r in rows)
    print(f"\n{'arm':<12}{'n':>4}{'subj_in_frame':>15}{'median':>9}"
          f"{'hole':>8}{'subj_cov':>10}{'zero_f':>8}"
          + (f"{'vis_frac':>10}{'vis_min':>9}" if occl_on else ""))
    for label in labels:
        sub = [r for r in rows if r["arm"] == label]
        if not sub:
            continue
        v = np.array([r["subject_in_frame"] for r in sub])
        h = np.array([r["hole_fraction"] for r in sub])
        c = np.array([r["subject_pixel_coverage"] for r in sub])
        z = np.array([r["subject_zero_frames"] for r in sub])
        extra = ""
        if occl_on:
            # entry 단위 median 을 arm 단위로 다시 median — 평균은 완전 가림 몇 개에 끌려간다.
            sv = np.array([r["subject_visible_frac"] for r in sub if "subject_visible_frac" in r])
            sm = np.array([r["subject_visible_min"] for r in sub if "subject_visible_min" in r])
            extra = f"{np.median(sv):>10.4f}{np.median(sm):>9.4f}" if len(sv) else f"{'-':>10}{'-':>9}"
        print(f"{label:<12}{len(sub):>4}{v.mean():>15.4f}{np.median(v):>9.4f}"
              f"{h.mean():>8.4f}{np.median(c):>10.4f}{z.mean():>8.1f}{extra}")
    # center_box sweep — 같은 렌더에서 상자만 줄인다. 1.0 은 "화면 안에 보이기만 하면 성공"이라
    # (1 - 값) 이 곧 **subject 소실 프레임 비율**이고, 거기서부터의 하락이 구도 성분이다.
    boxes = [round(b, 2) for b in np.arange(args.sweep_hi, args.sweep_lo - 1e-9, -args.sweep_step)]
    print(f"\n[center_box sweep] mean subject_in_frame")
    print(f"{'box':<8}" + "".join(f"{l:>10}" for l in labels))
    for box in boxes:
        cells = []
        for label in labels:
            sub = [r for r in rows if r["arm"] == label]
            cells.append(np.mean([in_frame_at(r["centers"], box) for r in sub]) if sub else np.nan)
        print(f"{box:<8.2f}" + "".join(f"{c:>10.4f}" for c in cells))
    report["sweep"] = {"boxes": boxes,
                       "mean": {l: [float(np.mean([in_frame_at(r["centers"], b)
                                                   for r in rows if r["arm"] == l]))
                                    for b in boxes] for l in labels}}


def merge(args):
    """샤드 JSON 들을 한 표로. 렌더는 안 한다 — 저장된 `rows` 만 합친다.

    `(name, arm)` 로 중복을 제거한다: 샤드 경계가 씬이라 원칙적으로 겹치지 않지만, 재실행한
    샤드를 지우지 않고 같은 glob 에 남겨 두면 그 씬만 두 번 세어져 평균이 조용히 기운다.

    `--scenes` 를 같이 주면 **합치면서** 그 씬만 남긴다. 전량 샤드가 이미 있으면 표본 표는
    다시 렌더할 필요가 없다 — 같은 행을 거르기만 하면 되고, 수치는 표본을 따로 돌린 것과
    같다. 열마다 씬 집합이 다르면 비교가 안 되므로, 거른 뒤 **arm 별 씬 수**를 찍는다.
    """
    shards = sorted(glob(args.merge))
    assert shards, f"{args.merge} 에 샤드 JSON 이 없다"
    wanted = None
    if args.scenes:
        with open(args.scenes, encoding="utf-8") as file:
            wanted = {line.strip() for line in file if line.strip()}
    rows, skipped, seen, labels = [], [], set(), ["gt"]
    for shard_path in shards:
        with open(shard_path, encoding="utf-8") as file:
            rep = json.load(file)
        labels += [l for l in rep["arms"] if l not in labels]
        n0 = len(rows)
        for row in rep["rows"]:
            if wanted is not None and row["scene"] not in wanted:
                continue
            key = (row["name"], row["arm"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        skipped += rep.get("skipped", [])
        print(f"  {path.basename(shard_path):<40} rows {len(rep['rows']):>5} "
              f"(+{len(rows) - n0} new)", flush=True)
    labels = [l for l in labels if any(r["arm"] == l for r in rows)]
    names = {r["name"] for r in rows}
    print(f"\nentries {len(names)} / rows {len(rows)} / arms {labels}")
    # 서로 다른 작업의 샤드를 합칠 때(우리 arm 만 돈 샤드 + 베이스라인만 돈 샤드) 한쪽이
    # 덜 끝나 있으면 열마다 분모가 달라진다. 조용히 지나가면 "같은 표"로 읽히므로 찍는다.
    per_arm = {l: len({r["scene"] for r in rows if r["arm"] == l}) for l in labels}
    if len(set(per_arm.values())) > 1:
        print(f"[경고] arm 별 씬 수가 다르다 — 열끼리 같은 분모가 아니다: {per_arm}")
    else:
        print(f"scenes/arm {next(iter(per_arm.values()))}")

    report = {"format": "subject_in_frame_v1", "center_box": args.center_box,
              "merged_from": shards, "scenes_file": args.scenes,
              "arms": {l: "" for l in labels if l != "gt"},
              "entries": len(names), "skipped": skipped, "rows": rows}
    summarize(rows, labels, args, report)
    if args.out:
        makedirs(path.dirname(path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=1)
        print(f"\n-> {args.out}")
    if skipped:
        print(f"\nskipped {len(skipped)}")
        for s in skipped[:10]:
            print(f"  {s}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    # entry 목록과 GT 를 빌려올 기준 eval 폴더 (`test/<prefix>_<scene>_<idx>_transforms_ref.json`)
    parser.add_argument("--ref_eval_dir", required=True)
    parser.add_argument("--eval_dir", action="append", default=[])   # LABEL=DIR, 여러 번
    parser.add_argument("--name_prefix", default="vista4d")          # dynpose 코퍼스면 `dynpose`
    parser.add_argument("--cloud_root", default=path.join(CINEMATRAJ_ROOT, "out"))
    # **`--ref_eval_dir` 을 만든 그 세대의 코퍼스**를 줄 것. entry 인덱스가 코퍼스 키가 아니라
    # 세대마다 뜻이 달라진다 (`assert_corpus_matches` 가 캡션으로 확인한다). d121 eval 은
    # `latentcam_da3_k6_d121`, 재굽기된 `latentcam_da3` 는 변이 수가 500 vs 875 로 다르다.
    parser.add_argument("--corpus_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d121")
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT)
    add_cloud_source_args(parser)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT)
    parser.add_argument("--center_box", type=float, default=0.80)    # verify.py 기본값과 동일
    # 표에 같이 찍을 center_box sweep 범위 (렌더는 한 번, 임계만 다시 건다)
    parser.add_argument("--sweep_hi", type=float, default=1.0)
    parser.add_argument("--sweep_lo", type=float, default=0.1)
    parser.add_argument("--sweep_step", type=float, default=0.1)
    parser.add_argument("--height", type=int, default=None)          # None = 점군 원해상도
    parser.add_argument("--width", type=int, default=None)
    # 뱅크를 fixed_focal 로 만들었으면 여기도 켜야 화각이 같다 (D98 이후 뱅크는 전부 켬).
    parser.add_argument("--fixed_focal", dest="fixed_focal", action="store_true", default=True)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    parser.add_argument("--include_gt", dest="include_gt", action="store_true", default=True)
    parser.add_argument("--no_include_gt", dest="include_gt", action="store_false")
    # 끄면 **순수 1:1 depth warp** 로 잰다 (프레임 f 의 점만 → 점 수 1/49 → 렌더가 그만큼 싸다).
    #
    # 기본값 True 를 "49프레임 뭉개기"로 읽으면 안 된다. `point_cloud.py:63` 이
    #   `visible = (visible & dynamic_mask) | static_mask`
    # 라서 **정적 점만** 전 프레임 True 이고 동적 점은 자기 프레임 하나에서만 True 다. 즉
    # `temporal_persistence=True` 가 곧 "static 누적 + dynamic 1:1" 이고, 이게 가림(occlusion)
    # 판정에 필요한 모드다 — 벽·가구가 다른 프레임에서만 관측됐으면 NTP 에서는 점군에 아예
    # 없어서 subject 가 안 가려진 것처럼 보인다.
    # 두 모드가 갈라지는 지점은 정적 점 하나뿐이므로:
    #   · **동적 subject** — subject 실루엣 자체는 두 모드가 동일(동적 점은 어차피 1프레임).
    #     달라지는 건 z-buffer 를 막는 정적 껍데기뿐.
    #   · **정적 subject** — NTP 는 subject 점도 프레임 f 것만 남겨 실루엣이 1/49 로 얇아진다.
    #     중심 추정이 흔들리고 `subject_zero_frames` 가 늘어 지표가 체계적으로 낮게 나온다.
    #     d121 val 은 정적 subject 가 다수(초반 327 entry 중 205)라 이 편향이 표를 지배한다.
    #   · `hole_fraction` 은 누적분이 빠지므로 **모드 간 비교 금지**.
    parser.add_argument("--temporal_persistence", dest="temporal_persistence",
                        action="store_true", default=True)
    parser.add_argument("--no_temporal_persistence", dest="temporal_persistence",
                        action="store_false")
    # subject **가림** 열(`subject_visible_frac`/`_min`/`_mean`). 프레임마다 subject 만 그린
    # 실루엣을 한 번 더 렌더해 전체 렌더의 depth 와 비교한다 (`occlusion_series`). 두 번째 패스는
    # 점 수가 실루엣 몫뿐이라 싸지만 그래도 렌더가 늘어나므로 기본값은 off — 끄면 열이 안 붙고
    # 기존 표와 비트 단위로 같다. `subject_in_frame`(위치) / `subject_pixel_coverage`(크기) 로는
    # 가림이 안 잡힌다.
    parser.add_argument("--subject_occlusion", dest="subject_occlusion",
                        action="store_true", default=False)
    parser.add_argument("--no_subject_occlusion", dest="subject_occlusion",
                        action="store_false")
    # 씬 **하나당** entry 수 제한 (프로브용). 씬 수는 안 줄인다 — d200 은 씬당 entry 가 6 이하라
    # 이 값으로는 아무것도 안 줄어든다. 씬 수를 줄이려면 `--scenes` / `--num_shards` 를 쓸 것.
    parser.add_argument("--limit", type=int, default=0)              # 0 = 씬당 entry 전량
    # 돌릴 씬만 한 줄에 하나씩 적은 파일. None = 전량(기존 동작). 표본을 파일로 받는 이유는
    # 산출물 옆에 "어떤 표본이었나"가 남아야 해서다 (main() 주석 참조).
    # `--merge` 와 같이 주면 렌더 없이 **합치면서** 거른다 — 전량 샤드에서 표본 표를 뽑을 때.
    parser.add_argument("--scenes", default=None)
    # 씬 단위 샤딩 (`sam3_seg_instances.py` 와 같은 패턴). 기본 1/0 = 전량, 기존 동작 그대로.
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    # 샤드 JSON glob 을 주면 **렌더 없이** 그것들만 합쳐 표를 그린다 (GPU 불필요).
    parser.add_argument("--merge", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    parsed = parser.parse_args()
    if parsed.merge:
        merge(parsed)
    else:
        main(parsed)
