"""`bank/poses.npz` 의 변이들을 **영상**으로 렌더한다 — 타일 애니메이션 1개 + (선택) 변이별 mp4.

왜 PNG 가 아니라 영상인가: `sample_camera_bank.py` 의 `preview.png` 는 변이마다 중간 프레임
**한 장**이다. 그런데 궤적은 한 장으로 판정이 안 된다 — 구멍이 언제 열리는지, subject 가 몇
프레임째에 프레임 밖으로 나가는지, 마지막 10프레임에서만 무너지는지가 전부 안 보인다.
τ 사다리의 요점이 "어디서 무너지나"이므로 프리뷰가 영상이 아니면 사다리를 읽을 수 없다.

## 무엇을 고르나

`--anchors` / `--presets` / `--tau` 로 뱅크를 거른 뒤 남은 것을 그리드에 깐다. 이 세 필터가
그대로 세 가지 읽기가 된다:

    --anchors dyn_0 --presets orbit_left_arc      τ 사다리 5칸 (강도 축)
    --anchors dyn_0 --tau 0.35                    같은 강도에서 preset 16종 (모양 축)
    --presets orbit_left_arc --tau 0.35           같은 카메라를 anchor 별로 (표적 축)

아무것도 안 주면 τ 단으로 층화해 `--max_tiles` 만큼 뽑는다 (뱅크 전체 훑기).

타일 캡션에 실측치(hole/path/τ)를 박는다 — 영상만 보고도 어느 변이인지 알아야 한다.
구멍은 마젠타. 소스 프레임을 왼쪽에 붙이려면 `--with_source` (구멍이 카메라 탓인지 소스
점군 탓인지 가르는 데 필요하다).

`--bank_dir hole_bank` 이면 `fit_hole_ladder.py` 가 만든 **hole 사다리** 뱅크를 같은 방식으로
깐다 (사다리 축만 `target_tau` → `target_hole` 로 바뀌고 나머지 읽기는 동일).

출력:
    <out>/<video>/<bank_dir>/preview.mp4          타일 애니메이션 (전 프레임)
    <out>/<video>/<bank_dir>/clips/<variant>.mp4  `--per_variant` 일 때만

`--no_sheet --per_variant --max_tiles 0` 이면 **타일 시트 없이 뱅크 전량**을 변이별 mp4 로만
떨군다 (`--max_tiles 0` = 상한 없음). 최종 카메라 전량을 depth warp 로 저장할 때 쓴다 — 수백
타일짜리 그리드는 읽을 수도 없고, 전 변이를 동시에 메모리에 들고 있어야 해서 못 돈다.

env: `vista4d` (렌더러가 GPU 를 쓴다)

예시:
    CUDA_VISIBLE_DEVICES=1 python scripts/render_bank_videos.py --video camel
    CUDA_VISIBLE_DEVICES=1 python scripts/render_bank_videos.py --video camel \
        --anchors dyn_0 --presets orbit_left_arc --with_source --per_variant
"""
import json
import sys
from argparse import ArgumentParser
from os import makedirs, path

import numpy as np

CINEMATRAJ_ROOT = path.dirname(path.dirname(path.abspath(__file__)))
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)

from decode.build_poses import deroll_poses                                     # noqa: E402
from lbm.cloud import EVAL_DATA_DEFAULT, VISTA4D_ROOT_DEFAULT                   # noqa: E402
from lbm.cloud import CINEMATRAJ_ROOT as CLOUD_ROOT                             # noqa: E402
from lbm.cloud import subject_point_mask                                        # noqa: E402
from lbm.overlay import contact_sheet, label_tile, paint_holes                  # noqa: E402
from lbm.presets import row_preset                                              # noqa: E402
from lbm.render import add_cloud_source_args, open_renderer                     # noqa: E402
from scene_graph.schema import load_graph                                       # noqa: E402
from scripts.build_candidate_board import subject_track_volume                  # noqa: E402


def write_video(output_path: str, frames, fps: float):
    """imageio + libx264. cv2 의 mp4v 는 VS Code 뷰어에서 안 열린다 (파일은 멀쩡해서 조용히 지나간다).

    [new 2026-09-03 / FIX-D118] 홀수 치수를 **여기서** 짝수로 채운다. `macro_block_size=1` 은
    imageio 의 자동 패딩을 끄는데, libx264 + yuv420p 은 짝수 치수만 받으므로 홀수가 하나라도
    있으면 ffmpeg 가 첫 프레임에서 죽고 `OSError: Broken pipe` 만 올라온다 — 원인이 치수라는
    말이 어디에도 안 나오고, 그때까지의 렌더(수 분)가 통째로 날아간다. 실측: `contact_sheet`
    가 3행 × 타일 225px + gap 4 → 683 을 만들어 두 씬 모두 렌더 직후 터졌다. 격자 높이는
    타일 크기·열 수·타일 개수의 곱이라 호출부에서 짝수를 보장할 수가 없다.
    """
    import imageio.v2 as imageio
    frames = list(frames)
    height, width = frames[0].shape[:2]
    if height % 2 or width % 2:                 # 아래/오른쪽에 1px 배경(격자 gap 과 같은 색)
        pad = ((0, height % 2), (0, width % 2), (0, 0))
        frames = [np.pad(f, pad, mode="edge") for f in frames]
        print(f"{'pad':<14}{width}x{height} -> {width + width % 2}x{height + height % 2} "
              f"(libx264 는 짝수 치수만 받는다)")
    imageio.mimwrite(output_path, frames, fps=fps, codec="libx264",
                     quality=6, macro_block_size=1)
    return output_path


def rung_of(variant: dict):
    """뱅크 두 종류를 같은 코드로 깐다 — τ 사다리는 `target_tau`, hole 사다리는 `target_hole`.
    사다리 축이 뭐냐만 다르고 "한 단씩 층화해 깐다"는 읽기는 똑같다."""
    if "target_tau" in variant:
        return "tau", float(variant["target_tau"])
    # D53. `--hole_mode excess` 뱅크는 `target_hole` 이 anchor 마다 다르다 (정지 hole + Δ).
    # 단을 고르는 축은 눈금 Δ 쪽이라 `hole_delta` 가 있으면 그걸 쓴다 (없는 예전 뱅크는 그대로).
    return "hole", float(variant.get("hole_delta", variant["target_hole"]))


def select(variants: list, anchors, presets, taus, max_tiles: int, seed: int,
           variant_ids=None):
    """필터 → 남은 게 많으면 사다리 단으로 층화해 뽑는다 (그냥 자르면 앞쪽 anchor 만 남는다).

    `variant_ids` 를 주면 **그 목록 순서 그대로** 낸다 — 감사(geometry.csv)에서 고른 특정 변이를
    나란히 보려면 층화 샘플링이 방해가 된다.

    `--presets` 는 **적힌 이름과 정식 이름 둘 다** 받는다. 옛 뱅크는 `straight_ease` 라고 적혀
    있지만 실제로는 `dolly_in` 이라, 한쪽만 받으면 사용자가 어느 이름을 대든 절반은 빈손이 된다.
    """
    if variant_ids:
        table = {v["variant_id"]: (i, v) for i, v in enumerate(variants)}
        missing = [vid for vid in variant_ids if vid not in table]
        assert not missing, f"bank.json 에 없는 variant_id: {missing}"
        return [table[vid] for vid in variant_ids]
    rows = [(i, v) for i, v in enumerate(variants)
            if (not anchors or v["anchor_id"] in anchors)
            and (not presets or v["preset"] in presets or row_preset(v) in presets)
            and (not taus or any(abs(rung_of(v)[1] - t) < 1e-9 for t in taus))]
    assert rows, "필터에 걸리는 변이가 없다. bank.csv 의 anchor_id/preset/사다리 열을 볼 것"
    if max_tiles <= 0:                      # 0 = 상한 없음 (뱅크 전량 덤프)
        return rows
    if len(rows) <= max_tiles:
        return rows
    rng = np.random.default_rng(seed)
    by_rung = {}
    for item in rows:
        by_rung.setdefault(rung_of(item[1])[1], []).append(item)
    per = max(1, max_tiles // len(by_rung))
    keep = []
    for rung in sorted(by_rung):
        pool = by_rung[rung]
        idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        keep.extend(pool[j] for j in sorted(idx))
    return sorted(keep, key=lambda item: item[0])[:max_tiles]


def colorize_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float):
    """렌더 depth (h,w) → turbo 컬러맵 RGB. 구멍은 `paint_holes` 와 같은 색으로 남긴다.

    역깊이(1/z)로 정규화한다 — 선형 z 는 가까운 표면이 전부 같은 색으로 뭉개져서 카메라가
    무엇에 얼마나 가까운지가 안 보인다. 눈금 `lo/hi` 는 **궤적 전체에서 한 번** 잡는다
    (프레임마다 다시 잡으면 카메라가 다가가도 색이 안 변해 비교가 성립하지 않는다).
    """
    import cv2
    z = np.where(valid & np.isfinite(depth) & (depth > 0), depth, np.nan)
    inv = 1.0 / np.clip(z, 1e-6, None)
    t = (inv - 1.0 / hi) / max(1.0 / lo - 1.0 / hi, 1e-9)
    u8 = np.clip(np.nan_to_num(t, nan=0.0) * 255.0, 0, 255).astype(np.uint8)
    rgb = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)[:, :, ::-1]
    return paint_holes(np.ascontiguousarray(rgb), valid & np.isfinite(depth) & (depth > 0))


def render_variant(renderer, poses: np.ndarray, height: int, width: int, stride: int,
                   focal=None, depth_mode: bool = False, depth_range=None,
                   temporal_persistence: bool = True):
    """궤적 전 프레임 렌더 → 구멍 칠한 RGB 리스트. `stride` 로 프레임을 솎을 수 있다.

    `focal` (n,) 이 오면 그 프레임의 소스 K 를 `fx,fy` 만 배율해서 넘긴다 — **intrinsic zoom** 은
    SE(3) 밖이라 `cam_c2w` 로는 못 실린다. `cx,cy` 는 안 건드린다 (zoom 은 주점을 안 옮긴다).
    `None` 이면 `renderer.render` 가 `K_src[frame]` 을 그대로 쓴다 (기존 동작 그대로).

    `depth_mode` 면 splatting RGB 대신 **렌더 depth** 를 컬러맵으로 낸다. 점군 색(=소스 영상
    픽셀)이 빠지므로 텍스처에 가려 안 보이던 기하 — 구멍의 모양, 표면까지의 거리 변화 — 만
    남는다. 기본 off = 기존 동작.

    `temporal_persistence=False` (NTP) 면 **그 프레임에서 유래한 점만** 그린다. SAM3 를 안 돌려
    `dynamic_mask` 가 전부 0 인 점군(`--allow_no_seg`)에서 필요하다 — 그런 점군은 전 점이
    static 이라 TP 렌더가 움직이는 피사체를 49프레임 겹쳐 그린다 (my-clip 실측: dynamic 점
    0/44,994,231, 골퍼가 프레임마다 유령 다발). NTP 는 시간 누적을 포기하는 대신 그 겹침이
    구조적으로 안 생긴다. 기본 True = 기존 릴 비트 동일.
    """
    picks = list(range(0, len(poses), max(stride, 1)))
    shots = []
    for f in picks:
        K = None
        if focal is not None and abs(float(focal[f]) - 1.0) > 1e-9:
            K = np.array(renderer.K_src[f], dtype=np.float64).copy()
            K[0, 0] *= float(focal[f])
            K[1, 1] *= float(focal[f])
        shot = renderer.render(poses[f], K=K, frame=int(f), height=height, width=width,
                               temporal_persistence=temporal_persistence)
        shots.append(shot if depth_mode else paint_holes(shot["rgb"], shot["valid"]))
    if not depth_mode:
        return picks, shots
    if depth_range:
        lo, hi = float(depth_range[0]), float(depth_range[1])
    else:                       # 궤적 전체의 유효 depth 5~95% — 이상치 한 픽셀에 눈금이 끌리지 않게
        pool = np.concatenate([s["depth"][s["valid"] & np.isfinite(s["depth"])
                                          & (s["depth"] > 0)].ravel() for s in shots])
        assert pool.size, "유효 depth 픽셀이 0개다 — 궤적 전체가 구멍이다"
        lo, hi = (float(v) for v in np.percentile(pool, [5, 95]))
        hi = max(hi, lo * 1.01)
        print(f"    depth_range(auto) {lo:.3f} .. {hi:.3f} u")
    return picks, [colorize_depth(s["depth"], s["valid"], lo, hi) for s in shots]


def main(args):
    assert args.sheet or args.per_variant, "--no_sheet 인데 --per_variant 도 아니면 나오는 게 없다"
    out_root = args.output_root or path.join(CLOUD_ROOT, "out")
    bank_folder = path.join(out_root, args.video, args.bank_dir)
    with open(path.join(bank_folder, "bank.json"), encoding="utf-8") as file:
        bank = json.load(file)
    # D87. 리타이밍한 궤적을 렌더하려면 뱅크가 아닌 poses.npz 를 읽어야 한다
    # (`emit_bank.py --keyframe_ease ... --dump_poses` 가 같은 스키마로 떨군다). 순서가
    # `bank.json` 변이 순서와 같아야 하는데 아래 길이 assert 가 그대로 지켜준다.
    poses_npz = np.load(args.poses_npz or path.join(bank_folder, "poses.npz"), allow_pickle=False)
    cam_c2w = poses_npz["cam_c2w"]
    assert len(cam_c2w) == len(bank["variants"]), \
        f"poses.npz {len(cam_c2w)} != bank.json {len(bank['variants'])} — 뱅크를 다시 만들 것"
    # 옛 뱅크엔 없는 키다 (zoom preset 이 생기기 전). 없으면 전부 1.0 = 기존 동작.
    focal_all = poses_npz["focal_scale"] if "focal_scale" in poses_npz.files else None

    graph = load_graph(path.join(out_root, args.video, "scene_graph.json"))
    #    D102. `--deroll` 없이 구워진 옛 뱅크를 **다시 굽지 않고** 지평선만 세워서 렌더한다 —
    #    실측 `tru_0ac97866_a08_s3f0k6` 옛 뱅크: orbit_left 21.8°, truck_left 15.0° (GT 뱅크 0.2°).
    #    광축과 위치는 안 바뀌므로 τ / hole / path 열은 `bank.json` 값 그대로 유효하다
    #    (`decode.build_poses.deroll_poses` docstring). 기본 off = 예전과 비트 동일.
    if args.deroll:
        cam_c2w = np.array(cam_c2w, dtype=np.float64)
        up_world = graph["gravity"]["up_world"]
        skipped = sum(deroll_poses(cam, up_world, np.zeros(len(cam))) for cam in cam_c2w)
        print(f"{'deroll':<14}up_world={np.round(up_world, 4).tolist()} "
              f"({graph['gravity'].get('method')})  roll 미정의 프레임 {skipped}\n")
    renderer, recon = open_renderer(args, out_root, graph)
    nodes = {n["id"]: n for n in graph["nodes"]}

    chosen = select(bank["variants"], args.anchors, args.presets, args.tau,
                    args.max_tiles, args.seed, variant_ids=args.variant_ids)
    print(f"{'video':<14}{args.video}   bank {len(bank['variants'])} -> tiles {len(chosen)}")
    print(f"{'filters':<14}anchors {args.anchors or '전량'}  presets {args.presets or '전량'}  "
          f"tau {args.tau or '전량'}\n")

    header = f"{'variant':<52}{'frames':>8}{'hole':>7}{'path_u':>8}"
    print(header)
    print("-" * len(header))

    # `--no_sheet` 는 타일 애니메이션을 아예 안 만든다 — 뱅크 전량(수백 개)을 변이별 mp4 로
    # 떨구는 용도. 시트를 만들면 ① 타일 수백 개짜리 그리드는 읽을 수도 없고 ② 전 변이의 49프레임을
    # 동시에 메모리에 들고 있어야 해서(480x270 기준 변이당 19 MB) 400 변이면 7.6 GB 다.
    # 시트를 끄면 렌더 즉시 mp4 로 흘려보내므로 상주 메모리가 변이 1개분으로 고정된다.
    stream_only = args.per_variant and not args.sheet
    clip_folder = path.join(bank_folder, "clips")
    if args.per_variant:
        makedirs(clip_folder, exist_ok=True)

    clips, captions = [], []
    for index, row in chosen:
        # subject 마스크는 anchor 마다 다시 깔아야 한다 — 안 하면 직전 anchor 의 실루엣이 남는다.
        renderer.set_subject(subject_point_mask(
            renderer.indices, subject_track_volume(recon, nodes[row["anchor_id"]])).cpu().numpy())
        picks, frames = render_variant(renderer, cam_c2w[index],
                                       args.tile_height, args.tile_width, args.stride,
                                       focal=(None if focal_all is None else focal_all[index]),
                                       depth_mode=args.depth, depth_range=args.depth_range)
        if stream_only:
            write_video(path.join(clip_folder, f"{row['variant_id']}.mp4"), frames, args.fps)
        else:
            clips.append(frames)
            captions.append((row, picks))
        print(f"{row['variant_id']:<52}{len(picks):>8}"
              f"{row['hole_fraction']:>7.3f}{row['path_len_u']:>8.3f}")

    if stream_only:
        print(f"\n변이별 mp4 {len(chosen)}개 -> {clip_folder}   (시트 없음: --no_sheet)")
        return

    length = min(len(c) for c in clips)
    source = None
    if args.with_source:
        import cv2
        video = recon["video"]
        source = [cv2.resize(video[f], (args.tile_width, args.tile_height))
                  for f in range(0, len(video), max(args.stride, 1))][:length]

    # 이 카메라가 어디서 온 것인지 타일 제목에 박는다. 뱅크의 `poses.npz` 는 우리가 fit 한
    # pseudo-GT 라 `GT`, `--poses_npz` 로 외부 파일을 물리면 출처를 모르니 `EXT` 로 두고
    # 모델 추론 결과면 `--pose_kind pred` 처럼 명시해서 부른다. `--pose_kind ""` 면 안 붙는다
    # (예전 영상과 라벨이 같아진다).
    pose_kind = args.pose_kind if args.pose_kind != "auto" else ("EXT" if args.poses_npz else "GT")

    # 프레임마다 그리드를 다시 합성한다. 타일 캡션은 매 프레임 같지만 프레임 번호만 갱신한다.
    sheet = []
    for f in range(length):
        tiles = []
        if source is not None:
            tiles.append(label_tile(source[f], "SOURCE", f"frame {f * args.stride}"))
        for clip, (row, picks) in zip(clips, captions):
            axis, rung = rung_of(row)
            # follow/keyframe 축은 **제목에** 붙인다 (D71/D72). 변이 이름에만 있으면 A/B 영상에서
            # 어느 타일이 어느 축인지 눈으로 못 갈라 비교 자체가 성립하지 않는다. 0 이면 안 붙여서
            # 그 축을 안 쓴 뱅크의 캡션은 예전 그대로다.
            gain = float(row.get("follow_gain", 0) or 0)
            keys = int(row.get("aim_keyframes", 0) or 0)
            # smooth 는 follow 를 쓸 때만 뜻이 있다 (gain 0 이면 offset 이 0 이라 창이 무의미).
            smooth = int(row.get("follow_smooth", 0) or 0) if gain else 0
            # 라벨은 **적힌 이름이 아니라 행이 실제로 만든 카메라의 이름**이다 (`row_preset`).
            # 옛 뱅크는 `straight_ease` 라 적어 놓고 `dolly_in` 을 굽는다 — snowboard 228행 중
            # 108행(47%)이 그렇다. 캡션(`build_bank_captions.py`)은 이미 정식 이름을 쓰고 있어서,
            # 안 고치면 같은 행의 캡션과 타일 라벨이 서로 다른 카메라를 가리킨다.
            preset_name = row_preset(row, args.raw_preset_names)
            label = ((args.label_prefix + " " if args.label_prefix else "")
                     + (f"[{pose_kind}] " if pose_kind else "") + preset_name[:18]
                     + (f" g{gain:g}" if gain else "") + (f" s{smooth}" if smooth else "")
                     + (f" k{keys}" if keys else ""))
            # target 이름을 자막에 같이 박는다. `anchor_id`(dyn_0/stat_4) 만으로는 어느 물체를
            # 겨냥한 카메라인지 영상만 보고 못 가른다. `anchor_label` 열이 없는 옛 뱅크는
            # `row.get` 이 빈 문자열을 돌려주므로 자막이 예전 그대로다.
            alabel = str(row.get("anchor_label", "") or "")
            target = row['anchor_id'] + (f"={alabel[:16]}" if alabel else "")
            tiles.append(label_tile(clip[f], label,
                                    f"{target} {axis}{rung:g} "
                                    f"hole{row['hole_fraction']:.2f} "
                                    f"path{row['path_len_u']:.2f}u  f{picks[f]}"))
        sheet.append(contact_sheet(tiles, args.columns))
    output = write_video(path.join(bank_folder, args.name), sheet, args.fps)

    if args.per_variant:
        for clip, (row, _) in zip(clips, captions):
            write_video(path.join(clip_folder, f"{row['variant_id']}.mp4"), clip, args.fps)
        print(f"\n변이별 mp4 {len(clips)}개 -> {clip_folder}")

    print(f"\n{'tiles':<14}{len(clips)}   frames {length}   fps {args.fps}")
    print(f"-> {output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--eval_data", default=EVAL_DATA_DEFAULT, type=str)
    parser.add_argument("--output_root", default=None, type=str)
    parser.add_argument("--vista4d_root", default=VISTA4D_ROOT_DEFAULT, type=str)
    parser.add_argument("--seg_root", default=None, type=str)
    parser.add_argument("--seg_static_root", default=None, type=str)
    add_cloud_source_args(parser)

    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    # DA3 추정 focal 이 프레임마다 흔들려서(snowboard +6.99%) 렌더에 화각 떨림이 실린다.
    # 켜면 frame0 K 로 전 프레임 고정. 기본 off = 기존 동작.
    parser.add_argument("--fixed_focal", action="store_true", default=False)
    parser.add_argument("--no_fixed_focal", dest="fixed_focal", action="store_false")
    # `bank` = τ 사다리(sample_camera_bank.py), `hole_bank` = hole 사다리(fit_hole_ladder.py).
    parser.add_argument("--bank_dir", default="bank", type=str)
    # D87. 비우면 `<bank_dir>/poses.npz` = 예전 동작. 리타이밍 대조용 경로 교체.
    parser.add_argument("--poses_npz", default="", type=str)
    # D102. 렌더 직전에 중력 기준 roll 을 0 으로 세운다 (뱅크는 안 건드린다). 기본 off.
    parser.add_argument("--deroll", dest="deroll", action="store_true")
    parser.add_argument("--no_deroll", dest="deroll", action="store_false")
    parser.set_defaults(deroll=False)

    # ── 무엇을 깔지 (셋 다 비우면 사다리 단 층화 추출)
    parser.add_argument("--anchors", nargs="*", default=None)
    parser.add_argument("--presets", nargs="*", default=None)
    # 사다리 값. τ 뱅크면 target_tau, hole 뱅크면 target_hole 로 걸린다.
    parser.add_argument("--tau", nargs="*", default=None, type=float)
    # 감사에서 고른 특정 변이를 그 순서대로 (다른 필터·층화를 전부 무시한다)
    parser.add_argument("--variant_ids", nargs="*", default=None)
    # splatting RGB 대신 렌더 depth 를 컬러맵으로. 점군 색이 빠져 기하만 남는다. 기본 off.
    parser.add_argument("--depth", action="store_true", default=False)
    parser.add_argument("--no_depth", dest="depth", action="store_false")
    # depth 눈금(u)을 고정. 안 주면 궤적 전체 유효 depth 의 5~95%. 타일끼리 색을 맞추려면 필수.
    parser.add_argument("--depth_range", nargs=2, default=None, type=float)
    parser.add_argument("--max_tiles", default=10, type=int)   # 타일 하나당 49프레임 렌더다
    parser.add_argument("--seed", default=0, type=int)

    # ── 영상
    parser.add_argument("--name", default="preview.mp4", type=str)
    # 같은 필터를 두 뱅크에 걸어 위아래로 붙일 때 어느 줄이 어느 뱅크인지 (예: BEFORE / AFTER).
    parser.add_argument("--label_prefix", default="", type=str)
    # 카메라 출처 태그. auto = 뱅크 poses.npz 면 GT / --poses_npz 면 EXT. 모델 추론이면 pred.
    parser.add_argument("--pose_kind", default="auto", type=str)
    # 옛 산출물 재현용. 켜면 `bank.json` 에 적힌 문자열을 그대로 라벨에 쓴다 (= 예전 동작, 어긋난 채).
    parser.add_argument("--raw_preset_names", action="store_true", default=False)
    parser.add_argument("--no_raw_preset_names", dest="raw_preset_names", action="store_false")
    parser.add_argument("--columns", default=5, type=int)
    parser.add_argument("--tile_width", default=480, type=int)
    parser.add_argument("--tile_height", default=270, type=int)
    parser.add_argument("--stride", default=1, type=int)       # 1 = 전 프레임
    parser.add_argument("--fps", default=12.0, type=float)
    # 구멍이 카메라 탓인지 소스 점군 탓인지 가르려면 소스를 같이 봐야 한다.
    parser.add_argument("--with_source", action="store_true", default=False)
    parser.add_argument("--no_with_source", dest="with_source", action="store_false")
    parser.add_argument("--per_variant", action="store_true", default=False)
    parser.add_argument("--no_per_variant", dest="per_variant", action="store_false")
    # `--no_per_variant` 와 같이 주면 아무것도 안 나온다 — 뱅크 전량 덤프 전용 스위치다.
    parser.add_argument("--sheet", action="store_true", default=True)
    parser.add_argument("--no_sheet", dest="sheet", action="store_false")
    main(parser.parse_args())
