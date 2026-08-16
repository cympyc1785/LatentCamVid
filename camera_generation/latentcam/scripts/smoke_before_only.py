#!/usr/bin/env python
"""smoke: `geo_cover_before_only: true` 로 바꾼 frontanchor arm 이 정말 앞쪽만 보는지 확인.

체크 항목
  1) 인덱스 크기가 before_only=false 때와 같은가 (29414/3263 유지 -> paired 비교 유지)
  2) 반환된 geo context view 가 전부 `[0, s]` 안에 있는가 (view0 = s, 나머지는 < s)
  3) target segment 프레임 (s, e) 가 context 에 절대 안 들어가는가
사용: python scripts/smoke_before_only.py <experiment_name> [n_samples]
"""
import os
import os.path as osp
import sys

_REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, _REPO)                      # utils/ 패키지
sys.path.insert(0, osp.join(_REPO, 'main'))
os.chdir(osp.join(_REPO, 'main'))

import numpy as np
import torch
from hydra import compose, initialize_config_dir

EXP = sys.argv[1] if len(sys.argv) > 1 else 'da3_7k_da3geo_frontanchor'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300

with initialize_config_dir(config_dir=osp.abspath('conf'), version_base=None):
    cfg = compose(config_name='config', overrides=[f'experiment={EXP}'])

# geo latent/encoder 를 안 태우고 index + view 선택만 본다
from omegaconf import OmegaConf  # noqa: E402
OmegaConf.set_struct(cfg, False)
cfg.geo_return_idxs = True

from base import build_dataset  # noqa: E402

ds = build_dataset(cfg)
print(f'[{EXP}] before_only={ds.geo_cover_before_only} avg_scale_ref={cfg.avg_scale_ref} '
      f'view_sampling={cfg.geo_view_sampling} cover_k={ds.geo_cover_k}')
print(f'[index] samples={len(ds)} scenes={len(ds.scene_dir_list)}')

rng = np.random.default_rng(0)
pick = rng.choice(len(ds), size=min(N, len(ds)), replace=False)
bad_ahead = bad_inseg = bad_view0 = 0
nviews = []
for i in pick:
    si, s, e, _cap, _name = ds.samples[int(i)]
    gi = list(ds._sample_geo_frustum_cover(si, s, e))
    nviews.append(len(gi))
    if gi[0] != s:
        bad_view0 += 1
    for j in gi[1:]:
        if j >= s:
            bad_ahead += 1
        if s <= j < e:
            bad_inseg += 1

print(f'[views] n={len(pick)}  views/sample: min={min(nviews)} max={max(nviews)}')
print(f'[check] view0 != s              : {bad_view0}   (기대 0)')
print(f'[check] view j >= s (뒤쪽/target): {bad_ahead}   (기대 0)')
print(f'[check] view in target [s,e)    : {bad_inseg}   (기대 0)')
ok = bad_view0 == 0 and bad_ahead == 0 and bad_inseg == 0
print('SMOKE', 'PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
