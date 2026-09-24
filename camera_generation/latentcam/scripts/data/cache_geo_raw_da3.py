"""DA3 geo 인코더의 **pre-ln** 토큰을 scene 당 파일 하나로 굽는 캐시 빌더.

왜 필요한가
-----------
Vista4D arm 은 "scene = 소스 영상 1편(49프레임), segment = 뱅크 변이"다. context view 는
`geo_view_sampling='context_uniform'` 이라 **(s,e) 를 안 보고 scene 만 보고** 정해지고
(`_even_indices(49,6)` = [0,10,19,29,38,48]), 그 view 들의 카메라도 소스 카메라라 변이와
무관하다. 즉 DA3 backbone 이 매 스텝 계산하는 값이 변이 9,873개에 걸쳐 **scene 52종**뿐이다.
같은 forward 를 189배 반복하고 있었다는 뜻이다.

왜 하필 pre-ln 인가
-------------------
기존 `main/cache_geo_embeddings.py` 는 `GeoEncoder.proj` **뒤**를 얼린다. da3 backend 는
`ln`(LayerNorm 3072) 과 `proj`(3072->768) 가 **학습 대상**이라(`models/geo_encoder.py:201
trainable = True`) 그 지점에서 얼리면 두 모듈이 초기값에 박제된다 — `dataset_cfg.py` 가 da3 에서
`geo_latent_cache_dir` 을 강제로 끄는 이유가 그것이다. 그래서 자르는 선을 `self.ln` **직전**,
즉 DA3 backbone 출력으로 옮겼다 (`models/da3_geo_encoder.py:encode_raw`). 그 위쪽은
`requires_grad_(False)` + 영구 eval 이라 같은 입력에 항상 같은 값이 나온다.

dtype 이 fp32 인 이유 (bf16 으로 저장하면 안 된다)
--------------------------------------------------
처음엔 "학습이 bf16 autocast 아래서 도니 backbone 출력도 bf16" 이라 보고 bf16 으로 저장했는데
**틀렸다**. autocast 는 matmul/linear 만 bf16 으로 내리고 LayerNorm 은 fp32 로 돌린다. DA3 블록은
`x = x + attn(ln(x))` 라 residual stream 이 fp32 로 유지되고, `encode_raw` 가 돌려주는 값도
fp32 다 (실측 `r.dtype == torch.float32`, `|a|max = 667.07`). 그래서 bf16 으로 굽는 순간
왕복 손실이 생긴다 — 실측 `max|d| 1.9993`, `||d||/||a|| 1.78e-3`. 이건 노이즈가 아니라 scene 마다
고정된 편향이라, 학습은 그 편향에 적응하지만 **캐시 없이 도는 추론과 어긋난다**. fp32 로 저장하면
캐시 경로와 on-the-fly 경로가 비트 단위로 같다 (max|d| 0.0000).

batch size 의존성은 없다 (2026-08-29 실측). 같은 샘플을 B=1 / B=2 / B=6 (동료를 다른 scene 으로,
순서까지 바꿔가며) 태워도 max|d| 0.0000 이다 — bf16 autocast 와 fp32 양쪽 모두. 백본의 global
attention 은 `b (s n) c` 라 배치 축을 안 섞고, reference-view 선택 경로는 `x.shape[1]`(=view 수)
만 보며 cam_token 이 있으면 아예 건너뛴다. 즉 B=1 로 구워 batch_size=8 학습에 먹여도 된다.

크기: scene 당 V*P*C*4 B = 6 * 576 * 3072 * 4 = 42.5 MB. Vista4D 52 scene -> ~2.2 GB.

안전장치
--------
* scene 안의 **모든** 샘플에 대해 `_sample_geo_context_uniform` 을 다시 돌려 geo_idxs 가
  정말 상수인지 확인한다. 하나라도 다르면 그 scene 은 굽지 않는다 (조용히 틀린 토큰을 먹이는
  것보다 on-the-fly 로 떨어지는 게 낫다).
* `meta` 에 model/input_hw/layers/posed/num_views/geo_idxs 를 적어 둔다. 설정이 바뀌면
  캐시를 지우고 다시 구울 것 — 학습 쪽은 파일 존재만 보고 읽는다.
* 이미 있으면 건너뛴다 (재실행 안전).

env:
  EXP        experiment 이름 (기본 vista4d_pgt_k6)
  OUT        출력 루트 (기본 <dl3dv_root>/geo_raw_cache_da3)
  SPLITS     train,test (기본) — 두 split 의 scene 을 합집합으로 굽는다
  DEVICE     기본 cuda:0
  LIMIT      앞에서 N scene 만 (기본 0 = 전부)
  SHARDS     [new 2026-09-17] 샤드 총수 (기본 1). scene 을 `i % SHARDS == SHARD` 로 가른다.
  SHARD      샤드 번호 (기본 0)

샤딩 (SHARDS/SHARD): 씬당 파일 하나라 샤드끼리 같은 파일을 안 건드린다 — 10k 규모 코퍼스에서
  한 GPU 로 굽는 데 몇 시간이 걸려서 넣었다. `is_done` 이 skip-done 을 보므로 샤드 경계를
  바꿔 다시 돌려도 안전하다. 기본값 1/0 이면 기존 동작과 **동일**하다.

사용 예:
  CUDA_VISIBLE_DEVICES=2 EXP=vista4d_pgt_k6 \
    /data1/cympyc1785/miniconda3/envs/latentcam/bin/python scripts/data/cache_geo_raw_da3.py
  CUDA_VISIBLE_DEVICES=0 EXP=dynpose_d200_da3 SHARDS=4 SHARD=0 \
    /data1/cympyc1785/miniconda3/envs/latentcam/bin/python scripts/data/cache_geo_raw_da3.py
"""
import os
import os.path as osp
import sys
import time

import torch

HERE = "/data1/cympyc1785/LatentCamVid/camera_generation/latentcam"
sys.path.insert(0, HERE); sys.path.insert(0, osp.join(HERE, "main"))
os.chdir(osp.join(HERE, "main"))
from hydra_cfg import load_cfg                                    # noqa: E402

EXP = os.environ.get("EXP", "vista4d_pgt_k6")
SPLITS = [s.strip() for s in os.environ.get("SPLITS", "train,test").split(',') if s.strip()]
DEVICE = os.environ.get("DEVICE", "cuda:0")
LIMIT = int(os.environ.get("LIMIT", "0"))
SHARDS = max(1, int(os.environ.get("SHARDS", "1")))
SHARD = int(os.environ.get("SHARD", "0"))
# [new 2026-09-24, D274] 저장 dtype. 기본 float32 = 옛 동작. bfloat16 은 `convert_geo_raw_cache.py`
# 가 fp32 캐시를 사후에 내리던 것과 **같은 캐스트**를 굽는 자리에서 한다 — 12 view 는 fp32 로
# 808 GB 라 사후 변환하려면 그만큼을 잠깐 들고 있어야 한다. 학습 쪽은 `geo_raw_cache_dtype:
# bfloat16` 으로 읽는다 (D200 bf16 캐시와 같은 규약).
SAVE_DTYPE = getattr(torch, os.environ.get("SAVE_DTYPE", "float32"))
assert 0 <= SHARD < SHARDS, f"SHARD {SHARD} 가 0~{SHARDS - 1} 밖이다"

cfg, _ = load_cfg("config", overrides=[f"experiment={EXP}"])
# 굽는 동안에는 캐시를 **꺼야** 한다 — 켜져 있으면 dataset 이 캐시를 읽고 early-return 해서
# images/geo_c2w 가 배치에 안 실린다 (닭이 먼저냐 달걀이 먼저냐).
cfg.geo_raw_cache_dir = None
OUT = os.environ.get("OUT") or osp.join(cfg.dl3dv_root, "geo_raw_cache_da3")

from dataset_dl3dv import CamDataset                              # noqa: E402
from base import collate_fn                                       # noqa: E402
from models.geo_encoder import build_geo_encoder                  # noqa: E402
from torch.utils.data import Subset, DataLoader                   # noqa: E402


def scene_plan(ds):
    """scene_key -> (대표 sample idx, geo_idxs). geo_idxs 가 scene 상수가 아니면 뺀다.

    검사를 dataset __getitem__ 이 아니라 sampler 를 직접 불러서 하는 이유: __getitem__ 은
    이미지 6장을 디코드하므로 9,873 샘플을 다 돌면 몇 분이 날아간다. sampler 는 extrinsics 도
    안 보고 프레임 수만 본다."""
    plan, bad = {}, {}
    for i, s in enumerate(ds.samples):
        scene_idx, st, en, _, data_name = s
        key = ds.geo_raw_key(data_name)
        idxs = tuple(ds._sample_geo_context_uniform(scene_idx, st, en))
        if key not in plan:
            plan[key] = (i, idxs)
        elif plan[key][1] != idxs:
            bad.setdefault(key, set()).add(idxs)
    for key in bad:
        plan.pop(key, None)
    return plan, bad


def is_done(path, meta):
    """이미 구워졌나. **존재만으로는 부족하다** — dtype 과 meta 까지 맞아야 한다.

    존재만 봤다가 dtype 을 bf16 -> fp32 로 바꾼 뒤 skip-done 이 낡은 파일을 그대로 살려 두는
    사고를 냈다. 설정을 바꾸면 그냥 다시 돌리면 덮어쓰도록 여기서 판정한다.

    단 `meta['exp']` 는 **비교에서 뺀다** (D163). 이건 설정이 아니라 "누가 처음 구웠나" 라벨이고,
    캐시 키가 scene 이라 여러 experiment 가 같은 디렉토리를 공유하는 게 정상 사용법이다
    (`dynpose_d137_da3.yaml` 이 d129 루트의 캐시를 가리키는 것이 그 예). 라벨까지 비교하면
    새 arm 이 붙을 때마다 내용이 **비트 동일한** 파일 수백 개를 덮어쓰고, 그 디렉토리를 읽고
    있는 다른 학습이 반쯤 쓰인 파일을 torch.load 하다 죽는다. 진짜 설정(da3 모델/입력 해상도/
    layers/posed/num_views/image_hw/first_view) 은 그대로 전부 비교한다."""
    if not osp.exists(path):
        return False
    try:
        c = torch.load(path, map_location='cpu', weights_only=False)
    except Exception:
        return False
    old = dict(c.get('meta') or {}); old.pop('exp', None)
    new = dict(meta); new.pop('exp', None)
    return c['raw'].dtype == SAVE_DTYPE and old == new


def main():
    os.makedirs(OUT, exist_ok=True)
    if str(getattr(cfg, 'geo_view_sampling', None)) != 'context_uniform':
        raise SystemExit(f"geo_view_sampling={getattr(cfg, 'geo_view_sampling', None)!r} — 이 "
                         f"캐시는 context view 가 scene 상수일 때만 성립한다 "
                         f"(context_uniform 만 그렇다)")
    if str(getattr(cfg, 'geo_encoder', None)) != 'da3':
        raise SystemExit(f"geo_encoder={getattr(cfg, 'geo_encoder', None)!r} != 'da3'")

    ge = build_geo_encoder(cfg).to(DEVICE).eval()
    meta = {
        'exp': EXP,
        'da3_geo_model': str(getattr(cfg, 'da3_geo_model', None)),
        'da3_geo_input_hw': list(getattr(cfg, 'da3_geo_input_hw', []) or []),
        'da3_geo_layers': str(getattr(cfg, 'da3_geo_layers', 'last')),
        'geo_posed': bool(getattr(cfg, 'geo_posed', False)),
        'geo_num_views': int(getattr(cfg, 'geo_num_views', 6)),
        'geo_image_hw': list(getattr(cfg, 'geo_image_hw', []) or []),
        'geo_first_view_target_s': bool(getattr(cfg, 'geo_first_view_target_s', False)),
    }

    tot_new, tot_bytes, tot_bad = 0, 0, 0
    for split in SPLITS:
        ds = CamDataset(cfg, split)
        plan, bad = scene_plan(ds)
        tot_bad += len(bad)
        keys = sorted(plan)
        if LIMIT:
            keys = keys[:LIMIT]
        # 샤드 분할은 **정렬된 전체 목록** 위에서 한다 — skip-done 을 먼저 적용하면 샤드마다
        # 다른 목록을 보게 되어 재실행 시 경계가 흔들린다.
        nall = len(keys)
        if SHARDS > 1:
            keys = [k for i, k in enumerate(keys) if i % SHARDS == SHARD]
        todo = [k for k in keys if not is_done(osp.join(OUT, f'{k}.pt'), meta)]
        print(f"[{split}] scenes {len(keys)}/{nall} (shard {SHARD}/{SHARDS}, "
              f"skip-done {len(keys) - len(todo)}, "
              f"non-constant {len(bad)}) -> {OUT}", flush=True)
        for k in sorted(bad):
            print(f"  !! {k}: geo_idxs 가 변이마다 다르다 ({len(bad[k])}종) -> 굽지 않음")
        if not todo:
            continue

        loader = DataLoader(Subset(ds, [plan[k][0] for k in todo]), batch_size=1, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for n, data in enumerate(loader):
                key = ds.geo_raw_key(data['data_name'][0])
                cam_token = None
                if getattr(cfg, 'geo_posed', False) and 'geo_c2w' in data:
                    cam_token = ge.build_cam_token(
                        data['geo_c2w'].to(DEVICE), data['geo_fxfycxcy'].to(DEVICE),
                        data['geo_hw'].to(DEVICE))
                raw = ge.encode_raw(data['images'].to(DEVICE), cam_token)   # (1,V,P,C) fp32
                raw = raw[0].float().cpu().clone()      # bf16 로 내리면 안 된다 — docstring 참조
                if SAVE_DTYPE != torch.float32:       # D274: 명시했을 때만 (convert 와 같은 캐스트)
                    raw = raw.to(SAVE_DTYPE)
                gidx = data['geo_idxs'][0].to(torch.int16).cpu().clone()
                torch.save({'raw': raw, 'geo_idxs': gidx, 'meta': meta},
                           osp.join(OUT, f'{key}.pt'))
                tot_new += 1; tot_bytes += raw.numel() * raw.element_size()
                if (n + 1) % 10 == 0 or n + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  [{split}] {n + 1}/{len(todo)} | {tot_bytes / 1e9:.2f} GB | "
                          f"{el / (n + 1):.2f} s/scene", flush=True)

    print(f"\n{'scenes written':<22}{tot_new}")
    print(f"{'size':<22}{tot_bytes / 1e9:.2f} GB")
    print(f"{'skipped (non-const)':<22}{tot_bad}")
    print(f"{'out':<22}{OUT}")
    print(f"\n학습에서 켜는 법:  geo_raw_cache_dir={OUT}")


if __name__ == "__main__":
    main()
