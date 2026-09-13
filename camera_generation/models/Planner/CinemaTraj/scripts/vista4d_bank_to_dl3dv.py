"""Vista4D pseudo-GT 뱅크(`out/<video>/hole_bank_k6/`) → latentcam 이 읽는 **DL3DV da3 포맷**.

`trumans_lite_to_dl3dv.py` 와 같은 이유로 새 dataset 클래스를 안 만든다 (`dataset_mixed.py`
`_VALID_NAMES` 에 Vista4D 가 없고, 대신 `dl3dv_root` 를 갈아끼우면 그대로 돈다). 다른 것은
**target 궤적이 소스 궤적이 아니라는 것** 하나다.

═══ 왜 target 만 다른 배열인가 ═════════════════════════════════════════════════════════
TRUMANS 는 클립마다 카메라가 하나뿐이라 "클립을 이어 붙여 scene 을 만들고 segment 를 클립으로"
두면 context(다른 클립) ≠ target(이 클립) 이 자연히 성립했다. Vista4D 는 영상 1편에 **소스
궤적 1개 + 합성 궤적 V개(뱅크)** 라 구조가 다르다:

    context = 소스 영상 프레임 49장 (uniform 6장 샘플링)  ← 영상마다 **하나뿐**
    target  = 뱅크 변이 V개 각각의 49프레임 궤적          ← 같은 씬을 다르게 찍은 것

같은 scene 을 V번 재사용하되 target 만 바꾸는 것이므로, 이미지를 V벌 복제하면
49 × ~280 × 52 ≈ 713k 장이 된다. 대신 이미지·소스 pose 는 **한 벌**만 두고 target 궤적을
`da3/target_poses.npz` 에 (V,49,4,4) 로 쌓는다. 데이터셋 쪽 분기는
`target_pose_source: da3_target_poses` 한 줄이다 (`dataset_cfg.TARGET_POSE_SOURCES`).

`prompts.json` 의 모든 segment 가 `frame_idx [0,49]` 인 것은 그래서 정상이다 — segment 는
"프레임 구간"이 아니라 **변이 인덱스**다. context 가 target 과 겹쳐도 누수가 아니다:
context 는 소스 카메라이고 target 은 합성 카메라라 포즈가 아예 다른 배열이다.

═══ 좌표계 / 단위 ══════════════════════════════════════════════════════════════════════
소스(`scene_graph.json:cameras.cam_c2w_world`)와 뱅크(`hole_bank_k6/poses.npz:cam_c2w`)는
**같은 world, 같은 DA3 raw 단위**다 (world = DA3 frame0 카메라, `cam_c2w[0]≈I`). 실측:
basketball-four 소스 |t|max 0.4622 로 `recon_and_seg/cameras.npz` 와 bit 일치, 뱅크 변이 하나의
경로장 0.6827 = `path_len_u` 0.0813 × S 8.397. 즉 뱅크가 보고하는 `_u` 는 S 로 나눈 값이고
**저장된 포즈 자체는 안 나눠져 있다** → 여기서도 안 나눈다 (분모는 avg_scale 로 따로 준다).

═══ intrinsics 는 고정 (사용자 지시) ═══════════════════════════════════════════════════
DA3 는 프레임마다 focal 이 흔들린다 (basketball-four K 가 t 축으로 상수가 아님; 정지 카메라에서도
−14% 드리프트가 관측됐다). 뱅크는 `--fixed_focal` 로 렌더러 K 를 frame0 에 고정해서 만들었으므로
여기서도 **K[0] 을 49프레임 전체에 복사**한다. context 와 target 이 같은 K 를 쓰고, `intr_norm:
rel` 에서 `cam_param[9:11]` 이 정확히 [1,1] 이 된다.

이미지를 줄이면 K 도 같이 줄인다 (`--image_scale`). `hw_list` 는 K 의 cx*2/cy*2 에서 나오므로
둘을 따로 두면 조용히 어긋난다.

═══ avg_scale ══════════════════════════════════════════════════════════════════════════
`scene_graph.json:scale.S` = frame0 non-sky 픽셀의 평균 ray 길이 (DA3 단위). 소스 영상만으로
계산되므로 **누수가 없고** 추론 때도 복원 가능하다. scene 상수라 전 변이가 같은 값을 쓴다.
TRUMANS arm 의 "recording median" 과 같은 자리다.

`dataset_dl3dv.py` 가 실제로 요구하는 것:
  <scene>/da3/pose.npz          extrinsics (N,3,4) OpenCV w2c, intrinsics (N,3,3)
  <scene>/da3/target_poses.npz  extrinsics (V,T,4,4) w2c, intrinsics (V,T,3,3), keys (V,) str
  <scene>/da3/prompts.json      seg → frame_idx [s,e] (e exclusive) + 캡션
  <scene>/da3/<ref dir>/<seg>.json   bare float
  <scene>/images_4/*.png        정렬 순서 = pose 순서, 개수 == N
  <root>/meta_vista4d.csv       `chunk` 열만 읽힌다

사용 예시:
  python scripts/vista4d_bank_to_dl3dv.py --dry_run
  python scripts/vista4d_bank_to_dl3dv.py --workers 8
  python scripts/vista4d_bank_to_dl3dv.py --bank_dir hole_bank_k6 --test_videos camel bmx-bumps
  python scripts/vista4d_bank_to_dl3dv.py --drop_status clamped_low   # D137. 아래 참조
"""
import csv
import json
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from os import listdir, makedirs, path

import numpy as np

HERE = path.dirname(path.abspath(__file__))
CINEMATRAJ_ROOT = path.dirname(HERE)
_LATENTCAM_MAIN = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam/main"
if _LATENTCAM_MAIN not in sys.path:
    sys.path.insert(0, _LATENTCAM_MAIN)      # AVG_SCALE_DIRS 를 두 벌 안 두려고 직접 읽는다
if CINEMATRAJ_ROOT not in sys.path:
    sys.path.insert(0, CINEMATRAJ_ROOT)
from dataset_cfg import AVG_SCALE_DIRS, AVG_SCALE_REFS      # noqa: E402
from lbm.presets import row_preset                          # noqa: E402

RECON_DEFAULT = "/data1/cympyc1785/data/Vista4D-Eval-Data/eval_data/recon_and_seg"


def is_test_hash(video: str, mod: int) -> bool:
    """씬 이름 해시로 홀드아웃 판정. `hash()` 가 아니라 md5 인 이유는 **프로세스 간 불변**이다
    (PYTHONHASHSEED 가 다르면 같은 씬이 run 마다 반대편으로 간다)."""
    from hashlib import md5                                          # noqa: PLC0415
    return int(md5(video.encode("utf-8")).hexdigest(), 16) % mod == 0


def scaled_K(K0, scale):
    """frame0 K → 이미지를 `scale` 배로 줄였을 때의 K. cx/cy 도 같이 줄여야 hw_list 가 맞는다."""
    K = np.asarray(K0, dtype=np.float64).copy()
    K[:2, :] *= float(scale)
    return K


def convert_scene(job):
    """영상 하나 → <out>/<chunk>/{da3/*, images_4/*}. 실패는 예외 대신 dict 로 돌려준다."""
    import imageio.v3 as iio                    # worker 안에서 import

    (video, chunk, out_root, image_dir, image_scale, recon_root, cine_out, bank_dir,
     refs, skip_done, dedup, captions_name, drop_status, drop_suspect, picked_only) = job
    dst = path.join(out_root, *chunk.split("/"))
    da3, img = path.join(dst, "da3"), path.join(dst, image_dir)
    try:
        graph = json.load(open(path.join(cine_out, video, "scene_graph.json"), encoding="utf-8"))
        bank = json.load(open(path.join(cine_out, video, bank_dir, "bank.json"), encoding="utf-8"))
        caps = json.load(open(path.join(cine_out, video, bank_dir, captions_name),
                              encoding="utf-8"))["captions"]
        poses = np.load(path.join(cine_out, video, bank_dir, "poses.npz"))
        bank_c2w = np.asarray(poses["cam_c2w"], dtype=np.float64)          # (V,T,4,4)
        bank_ids = [str(v) for v in poses["variant_id"].tolist()]

        # ---- status 필터 (D137). `--drop_status` 가 빈 리스트면 아래 블록이 통째로 no-op 이라
        #    예전 코퍼스와 비트 동일하게 돈다.
        #    `clamped_low` 는 `fit_hole_ladder.py:513-531` 에서 "knob 을 하한까지 밀었는데
        #    **하한에서도 `over()` 가 여전히 참**" 일 때 찍힌다. 즉 제일 작게 만들어도 게이트를
        #    위반하는 카메라이고, 그 궤적이 그대로 pseudo-GT 로 나간다. d129 dd10 train 612행
        #    실측에서 binding 이 collision 340 / obb 114 / approach 87 / ground 43 / elev 26 /
        #    hole 2 였다 — hole 예산(=τ) 때문에 걸린 건 2건뿐이고 나머지는 물리 게이트다.
        #    `path_len_u` 중앙값이 0.0330 (solved 는 0.6900) 이라 사실상 정지 카메라인데
        #    캡션은 `dolly_in`/`s_curve` 라고 써 있다 — 이름과 실제 이동량이 어긋난 학습 신호.
        #    접두사 매칭이다: `clamped_low` 하나로 `clamped_low+tau_floor` 까지 잡는다
        #    (`+tau_floor` 접미사는 `:910` 에서 붙는 **직교하는** 표시라 `solved+tau_floor`
        #    처럼 정상 행에도 붙는다 — 접미사만으로 거르면 안 된다).
        status_by_id = {v["variant_id"]: str(v.get("status", "")) for v in bank["variants"]}
        keep = list(range(len(bank_ids)))
        n_drop, drop_hist = 0, {}
        if drop_status:
            kept = []
            for i in keep:
                status = status_by_id.get(bank_ids[i], "")
                if any(status.startswith(p) for p in drop_status):
                    drop_hist[status] = drop_hist.get(status, 0) + 1
                    continue
                kept.append(i)
            n_drop, keep = len(keep) - len(kept), kept

        # ---- suspect 필터 (D140). `status` 필터와 **별개 축**이다.
        #    `drop_status` 는 "게이트가 막았다"를 자르고, 이건 "게이트는 통과했는데 설정 탓에
        #    캡션과 기하가 어긋난다"를 자른다 (`fit_hole_ladder.tag_suspects` 참조). 그래서
        #    태그 하나가 붙은 행의 대부분은 `status == "solved"` 이고 위 블록에 안 걸린다.
        #    `suspect` 열은 `|` 로 이은 태그 목록이라 **부분 문자열**이 아니라 토큰으로 맞춘다 —
        #    접두사 매칭을 쓰면 `aim_free_subject_lost` 가 두 번째 토큰일 때 못 잡는다.
        #    빈 리스트면 no-op 이고, `suspect` 열이 아예 없는 예전 뱅크에서도 no-op 이다.
        if drop_suspect:
            susp_by_id = {v["variant_id"]: str(v.get("suspect", "") or "")
                          for v in bank["variants"]}
            kept = []
            for i in keep:
                tags = [t for t in susp_by_id.get(bank_ids[i], "").split("|") if t]
                hit = sorted(set(tags) & set(drop_suspect))
                if hit:
                    key = "suspect:" + "|".join(hit)
                    drop_hist[key] = drop_hist.get(key, 0) + 1
                    continue
                kept.append(i)
            n_drop, keep = n_drop + len(keep) - len(kept), kept

        # ---- picked 필터 (D189). 위 둘과 **또 다른 축**이다 — 저 둘은 행마다 독립으로 "이건
        #    쓰면 안 된다"를 보지만, 이건 씬/anchor 단위 **예산**의 결과다 (`lbm/pick.py`).
        #    뱅크는 재고 목록이라 사다리가 만든 행을 다 들고 있고, 그중 무엇을 코퍼스로 쓸지는
        #    `picked` 열이 정한다. `emit_bank.py --picked_only` 와 같은 열·같은 판정이다.
        #    기본 꺼짐 = 열이 없던 예전 뱅크와 비트 동일. 켰는데 열이 비어 있으면 그 씬은 0행이
        #    되므로 (조용한 전멸) 아래 `n_unpicked` 를 요약에 싣는다.
        n_unpicked = 0
        if picked_only:
            pick_by_id = {v["variant_id"]: bool(v.get("picked")) for v in bank["variants"]}
            kept = [i for i in keep if pick_by_id.get(bank_ids[i])]
            n_unpicked, keep = len(keep) - len(kept), kept
            if n_unpicked:
                drop_hist["unpicked"] = drop_hist.get("unpicked", 0) + n_unpicked
            n_drop += n_unpicked

        # ---- 사다리 붕괴 중복 제거.
        #    hole 사다리 4단은 게이트(obb/ground/approach/elev/shape/collision)가 물리면
        #    같은 knob 에서 멈춘다 — 그러면 4단이 **같은 궤적**이 된다. 47편 실측에서
        #    15624 변이 중 절반이 그랬고, 그중 99.4% 는 pose 배열이 비트 단위로 같았다
        #    (나머지도 위치 차 <= 0.025 u). 그대로 내보내면 학습이 같은 (pose, text) 쌍을
        #    두 번 보고, 게이트가 잘 물리는 좁은 씬이 그만큼 과대표집된다.
        #    판정은 path_len 같은 대리값이 아니라 **pose 배열 자체**로 한다 — 임계값이 없다.
        #    남기는 건 뱅크 순서상 첫 번째 = 사다리 아랫단(작은 target_hole)이다.
        #    **status 필터 뒤에 돈다** — 순서가 뒤바뀌면 pose 가 비트 동일한 붕괴 그룹에서
        #    `clamped_low` 인 아랫단이 대표로 남아 그룹 전체가 통째로 날아간다.
        n_pre_dedup = len(keep)
        if dedup:
            seen, dedup_keep = set(), []
            for i in keep:
                digest = bank_c2w[i].round(9).tobytes()
                if digest in seen:
                    continue
                seen.add(digest)
                dedup_keep.append(i)
            keep = dedup_keep
        n_dup = n_pre_dedup - len(keep)
        bank_c2w = bank_c2w[keep]
        bank_ids = [bank_ids[i] for i in keep]

        src_c2w = np.asarray(graph["cameras"]["cam_c2w_world"], dtype=np.float64)   # (N,4,4)
        n, T = src_c2w.shape[0], bank_c2w.shape[1]
        if T != n:
            return {"video": video, "status": "fail",
                    "why": f"뱅크 프레임 {T} != 소스 프레임 {n}"}
        # --fixed_focal 과 같은 규약: frame0 K 를 전 프레임에 복사한 뒤 이미지 배율만큼 줄인다.
        K = scaled_K(np.asarray(graph["cameras"]["K"], dtype=np.float64)[0], image_scale)
        h = int(round(float(K[1, 2]) * 2))
        w = int(round(float(K[0, 2]) * 2))

        done = (path.isfile(path.join(da3, "target_poses.npz"))
                and path.isdir(img) and len(listdir(img)) == n)
        if skip_done and done:
            return {"video": video, "status": "skip_done", "n": n, "h": h, "w": w,
                    "n_var": len(bank_ids), "n_dup": n_dup, "n_drop": n_drop,
                    "drop_hist": drop_hist}
        makedirs(da3, exist_ok=True)
        makedirs(img, exist_ok=True)

        # ---- 소스 프레임 (context). plugin="FFMPEG": vista4d env 에 pyav 가 없다.
        frames = iio.imread(path.join(recon_root, video, "video.mp4"), plugin="FFMPEG")
        if len(frames) != n:
            return {"video": video, "status": "fail",
                    "why": f"video {len(frames)} != pose {n}"}
        if image_scale != 1.0:
            from PIL import Image
            frames = [np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC))
                      for f in frames]
        for i, fr in enumerate(frames):
            iio.imwrite(path.join(img, f"{i:05d}.png"), fr)

        # ---- 소스 pose (context). w2c (N,3,4), K 는 frame0 고정본.
        np.savez(path.join(da3, "pose.npz"),
                 extrinsics=np.linalg.inv(src_c2w)[:, :3, :4].astype(np.float32),
                 intrinsics=np.repeat(K[None], n, axis=0).astype(np.float32))

        # ---- target 궤적 (합성 pseudo-GT). seg 키는 뱅크 순서의 정수 문자열.
        keys = [str(i) for i in range(len(bank_ids))]
        np.savez(path.join(da3, "target_poses.npz"),
                 extrinsics=np.linalg.inv(bank_c2w).astype(np.float32),      # (V,T,4,4) w2c
                 intrinsics=np.repeat(K[None, None], len(keys), axis=0
                                      ).repeat(T, axis=1).astype(np.float32),
                 keys=np.array(keys),
                 variant_id=np.array(bank_ids))

        by_id = {v["variant_id"]: v for v in bank["variants"]}
        prompts = {}
        for seg, vid in zip(keys, bank_ids):
            row, cap = by_id[vid], caps.get(vid, {})
            prompts[seg] = {
                "frame_idx": [0, n],                       # segment = 변이. 구간은 항상 전체다.
                "prompt_camera_with_scene_video": {"concise": cap.get("prompt", "")},
                # `preset` 은 **행이 실제로 만든 카메라의 정식 이름**이다. 옛 뱅크에 적힌 문자열은
                # `preset_raw` 로 같이 싣는다 (옛 코퍼스와 대조할 때 필요; 여기 플래그를 안 두는
                # 이유는 둘 다 나가므로 예전 값이 소실되지 않아서다). `aim` 도 함께 내보낸다 —
                # 하류(`render_pred_depth_warp.py`)가 이 JSON 만 보고 이름을 풀 수 있어야 한다.
                "variant_id": vid, "preset": row_preset(row), "preset_raw": row["preset"],
                "aim": row.get("aim"), "anchor_label": row["anchor_label"],
                # D121. nl 형식 캡션은 `target_text`/`framing_nl`/`composition` 을 더 들고 온다.
                # **있을 때만** 싣는다 — fields 형식으로 뽑은 코퍼스는 예전과 같아야 한다.
                "caption_fields": {k: cap[k] for k in
                                   ("target", "event", "framing", "motion",
                                    "target_text", "framing_nl", "composition")
                                   if k in cap},
                "tau_max": row["tau_max"], "hole_fraction": row["hole_fraction"],
                "subject_area_med": row["subject_area_med"],
                # D87. subject 가 프레임 안에 든 비율 / 가림 없이 보인 비율. 학습 쪽에서
                # 로깅하려면 코퍼스에 실려 있어야 한다 (학습이 뱅크 CSV 를 열 수는 없다).
                # `--no_subject_visible` 로 구운 뱅크엔 뒤엣것이 없어 None 이 된다 — "가림 0" 과
                # "안 쟀다" 는 하류가 구분해야 하므로 기본값을 안 준다.
                "subject_in_frame": row["subject_in_frame"],
                "subject_visible_frac": row.get("subject_visible_frac"),
                "source": f"vista4d_lbm_lite/{bank_dir}",
            }
        json.dump(prompts, open(path.join(da3, "prompts.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)

        avg_scale = float(graph["scale"]["S"])             # scene 상수, 소스만으로 계산됨
        for ref in refs:
            ref_dir = path.join(da3, AVG_SCALE_DIRS[ref])
            makedirs(ref_dir, exist_ok=True)
            for seg in keys:
                json.dump(avg_scale, open(path.join(ref_dir, f"{seg}.json"), "w"))

        return {"video": video, "status": "ok", "n": n, "h": h, "w": w,
                "n_var": len(keys), "n_dup": n_dup, "n_drop": n_drop,
                "drop_hist": drop_hist, "S": avg_scale}
    except Exception as error:                             # noqa: BLE001
        return {"video": video, "status": "fail", "why": f"{type(error).__name__}: {error}"}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cine_out", default=path.join(CINEMATRAJ_ROOT, "out"))
    parser.add_argument("--bank_dir", default="hole_bank_k6")
    # 뱅크 안의 캡션 파일. 궤적은 그대로 두고 **텍스트만** 바꿔 대조군을 만들 때 쓴다
    # (`build_bank_captions.py --out_name`). 기본값이 예전 하드코딩 이름이라 동작은 그대로다.
    parser.add_argument("--captions_name", default="captions.json", type=str)
    parser.add_argument("--recon_root", default=RECON_DEFAULT)
    parser.add_argument("--out_root",
                        default="/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3")
    parser.add_argument("--videos", nargs="*", default=["all"])
    parser.add_argument("--meta_csv", default="meta_vista4d.csv")
    # chunk 를 두 단계로 두는 이유는 trumans 쪽과 같다: seg-list 분할이 쓰는
    # `CamDataset.seg_key` (dataset_dl3dv.py:1289) 가 data_name 을 `<batch>_<hash>_<seg>` 로
    # 가정한다. Vista4D 영상 이름은 전부 하이픈이라 밑줄 split 이 안전하다.
    parser.add_argument("--chunk_prefix", default="vista4d")
    # scene 단위 holdout. 영상 이름 **접두사**로 매칭한다. 비우면 base.py 랜덤 분할로 떨어진다.
    parser.add_argument("--test_videos", nargs="*",
                        default=["camel", "avocado-slice", "bmx-bumps", "couple-hug"])
    # D189. 이름 **해시**로 홀드아웃을 정한다 (`md5(video) % mod == 0`). 0 이면 꺼짐 = 예전대로
    # `--test_videos` 접두사만. DynPose 처럼 씬이 UUID 이고 코퍼스가 **자라는 중**일 때 필요하다:
    # 접두사 분할은 이미 구운 풀이 정렬 순서로 편향돼 있어 `00`~`0b` 12/256(=4.7% 기대)이 실제로는
    # 865편 중 513편(59.3%)을 잡았다. 해시는 코퍼스가 몇 편이든 같은 씬을 같은 쪽에 두므로
    # d188 이 다 구워진 뒤 다시 내보내도 test 가 train 으로 새지 않는다.
    parser.add_argument("--test_hash_mod", default=0, type=int)
    parser.add_argument("--seg_list_prefix", default="seg_list_vista4d")
    parser.add_argument("--image_dir", default="images_4")     # dl3dv IMAGE_DIR_NAMES 첫 후보
    # 소스는 1280x720. 0.5 면 640x360 -> geo_image_hw (256,448) 로 갈 때 여전히 축소라
    # 업샘플이 안 생기고, PNG 디코드가 4배 싸다. K 도 같은 배율로 줄인다.
    parser.add_argument("--image_scale", default=0.5, type=float)
    parser.add_argument("--avg_scale_refs", nargs="+",
                        default=["context_first_cam", "centroid"],
                        choices=sorted(AVG_SCALE_REFS))
    # hole 사다리 4단이 게이트에 물려 같은 궤적으로 붕괴한 변이를 pose 배열 동일성으로 걸러낸다.
    # 기본 켬 — 47편 실측에서 15624 중 7804(50%)가 중복이었다. `--no_dedup` 으로 예전처럼 전량.
    parser.add_argument("--dedup", action="store_true", default=True)
    parser.add_argument("--no_dedup", dest="dedup", action="store_false")
    # D137. 내보내지 않을 `bank.json` status **접두사** 목록. 기본값이 빈 리스트라 안 주면
    # 예전 코퍼스와 비트 동일하다. `clamped_low` 하나로 `clamped_low+tau_floor` 까지 잡힌다.
    parser.add_argument("--drop_status", nargs="*", default=[],
                        help="예: --drop_status clamped_low  (접두사 매칭, 기본 없음)")
    # D140. 내보내지 않을 `suspect` 태그 목록. `drop_status` 와 **다른 축**이다 — 저건
    # "게이트가 막았다"를 자르고 이건 "게이트는 통과했는데 설정 탓에 캡션과 기하가 어긋난다"를
    # 자른다 (`fit_hole_ladder.tag_suspects`). 태그 목록이라 접두사가 아니라 **토큰** 매칭.
    # 기본값이 빈 리스트라 안 주면 예전 코퍼스와 비트 동일하고, `suspect` 열이 없는 예전
    # 뱅크에서도 no-op 이다.
    parser.add_argument("--drop_suspect", nargs="*", default=[],
                        choices=["aim_free_subject_lost", "static_start_collision",
                                 "motion_preset_no_motion"],
                        help="예: --drop_suspect aim_free_subject_lost  (토큰 매칭, 기본 없음)")
    # D189. `picked` 열이 고른 행만 내보낸다 (`lbm/pick.py` / `emit_bank.py --picked_only` 와
    # 같은 열). 기본 꺼짐 = 열이 없던 예전 뱅크와 비트 동일.
    parser.add_argument("--picked_only", action="store_true", default=False)
    parser.add_argument("--no_picked_only", dest="picked_only", action="store_false")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--skip_done", action="store_true", default=True)
    parser.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.videos == ["all"]:
        videos = sorted(v for v in listdir(args.cine_out)
                        if path.isfile(path.join(args.cine_out, v, args.bank_dir, "bank.json")))
    else:
        videos = list(args.videos)

    jobs, missing = [], []
    for video in videos:
        need = [path.join(args.cine_out, video, "scene_graph.json"),
                path.join(args.cine_out, video, args.bank_dir, "bank.json"),
                path.join(args.cine_out, video, args.bank_dir, "poses.npz"),
                path.join(args.cine_out, video, args.bank_dir, args.captions_name),
                path.join(args.recon_root, video, "video.mp4")]
        gone = [path.basename(p) for p in need if not path.isfile(p)]
        if gone:
            missing.append((video, f"없음: {', '.join(gone)}"))
            continue
        chunk = f"{args.chunk_prefix}/{video}" if args.chunk_prefix else video
        jobs.append((video, chunk, args.out_root, args.image_dir, args.image_scale,
                     args.recon_root, args.cine_out, args.bank_dir, args.avg_scale_refs,
                     args.skip_done, args.dedup, args.captions_name, list(args.drop_status),
                     list(args.drop_suspect), args.picked_only))

    print(f"{'bank_dir':22s} {args.bank_dir}")
    print(f"{'picked_only':22s} {args.picked_only}")
    print(f"{'drop_status':22s} {args.drop_status or '(없음 — 예전과 비트 동일)'}")
    print(f"{'drop_suspect':22s} {args.drop_suspect or '(없음 — 예전과 비트 동일)'}")
    print(f"{'scenes (video)':22s} {len(jobs)}")
    print(f"{'skipped':22s} {len(missing)}")
    for video, why in missing:
        print(f"  - {video}: {why}")
    if args.dry_run:
        for video, *_ in jobs[:5]:
            n = len(json.load(open(path.join(args.cine_out, video, args.bank_dir, "bank.json"),
                                   encoding="utf-8"))["variants"])
            print(f"  {video:<24} {n:5d} variants")
        return

    makedirs(args.out_root, exist_ok=True)
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(convert_scene, job) for job in jobs]
        for future in as_completed(futures):
            results.append(future.result())
            print(f"  done {len(results)}/{len(futures)}  {results[-1]['video']:<24} "
                  f"{results[-1]['status']}", flush=True)

    ok = [r for r in results if r["status"] in ("ok", "skip_done")]
    fail = [r for r in results if r["status"] == "fail"]
    with open(path.join(args.out_root, args.meta_csv), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["chunk", "height", "width", "num_images"])
        for r in sorted(ok, key=lambda x: x["video"]):
            chunk = f"{args.chunk_prefix}/{r['video']}" if args.chunk_prefix else r["video"]
            writer.writerow([chunk, r["h"], r["w"], r["n"]])

    seg_paths = {}
    if (args.test_videos or args.test_hash_mod) and args.chunk_prefix:
        lists = {"train": [], "test": []}
        for r in sorted(ok, key=lambda x: x["video"]):
            chunk = f"{args.chunk_prefix}/{r['video']}"
            held = any(r["video"].startswith(p) for p in args.test_videos)
            if args.test_hash_mod:
                held = held or is_test_hash(r["video"], args.test_hash_mod)
            lists["test" if held else "train"] += [f"{chunk}/{seg}" for seg in range(r["n_var"])]
        for side, ids in lists.items():
            p = path.join(args.out_root, f"{args.seg_list_prefix}_{side}.txt")
            with open(p, "w") as file:
                file.write("\n".join(ids) + "\n")
            seg_paths[side] = (p, len(ids))

    print(f"\n{'video':<24}{'var':>6}{'drop':>6}{'dup':>6}{'frames':>8}{'h x w':>12}{'S':>10}")
    for r in sorted(ok, key=lambda x: x["video"]):
        hw = f"{r['h']}x{r['w']}"
        print(f"{r['video']:<24}{r['n_var']:6d}{r.get('n_drop', 0):6d}{r.get('n_dup', 0):6d}"
              f"{r['n']:8d}{hw:>12}{r.get('S', float('nan')):10.3f}")
    # 중복 제거는 조용히 개수가 줄면 안 되는 값이라 총계를 따로 찍는다.
    kept = sum(r["n_var"] for r in ok)
    dup = sum(r.get("n_dup", 0) for r in ok)
    drop = sum(r.get("n_drop", 0) for r in ok)
    print(f"\n{'변이 (dedup 후)':22s} {kept}"
          + (f"   제거한 사다리 붕괴 중복 {dup} ({dup / max(kept + dup, 1):.0%})"
             if args.dedup else "   (--no_dedup: 중복 제거 안 함)"))
    # status / suspect 필터도 같은 이유로 총계를 찍는다 — 조용히 코퍼스가 줄면 안 된다.
    # `drop_hist` 는 두 필터가 공유하는 히스토그램이고, suspect 쪽 키에는 `suspect:` 접두사가
    # 붙어 있어 (convert_scene) 어느 축에서 잘렸는지 한 표에서 구분된다.
    if args.drop_status or args.drop_suspect:
        hist = {}
        for r in ok:
            for status, cnt in (r.get("drop_hist") or {}).items():
                hist[status] = hist.get(status, 0) + cnt
        base = kept + dup + drop
        print(f"{'status/suspect 제외':22s} {drop} / {base} ({drop / max(base, 1):.2%})"
              f"   status {args.drop_status or '-'}   suspect {args.drop_suspect or '-'}")
        for status, cnt in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {status:<30}{cnt:7d}")
    print(f"\n{'ok':22s} {len([r for r in results if r['status'] == 'ok'])}")
    print(f"{'skip_done':22s} {len([r for r in results if r['status'] == 'skip_done'])}")
    print(f"{'fail':22s} {len(fail)}")
    for r in fail:
        print(f"  - {r['video']}: {r['why']}")
    print(f"{'meta_csv':22s} {path.join(args.out_root, args.meta_csv)} ({len(ok)} rows)")
    print(f"{'avg_scale dirs':22s} {[AVG_SCALE_DIRS[r] for r in args.avg_scale_refs]} "
          f"<- scene_graph.scale.S")
    for side, (p, cnt) in sorted(seg_paths.items()):
        print(f"{'seg_list ' + side:22s} {p} ({cnt} segments)")
    if not seg_paths:
        print(f"{'seg_list':22s} (없음 -> base.py 가 train_frac 랜덤 분할)")


if __name__ == "__main__":
    main()
