"""`dataset_cfg.resolve_dataset_cfg` 의 캐스케이드/검증 경로 테스트.

golden_dataset.py 가 못 잡는 구멍을 메운다. golden 은 실제 `ds[i]` 텐서를 비교하므로
**켜진 캐시**는 (geo_emb 가 나오는지로) 검증되지만, **꺼지는 네 조건**은 검증되지 않는다 —
어느 arm 도 "캐시 dir 이 설정돼 있는데 custom/swap/inseg_k 라서 꺼진다"는 조합이 아니기
때문이다. 그 조합이 조용히 깨지면 캐시가 켜진 채로 남아 context view 변경이 무시되고,
증상은 "swap probe 를 돌렸는데 결과가 안 변한다" 뿐이다 (에러 없음).

resolve_dataset_cfg 는 순수 함수라 디스크를 안 타므로 이 파일은 1초 안에 돈다.

    python scripts/test/test_dataset_cfg.py
"""
import os.path as osp
import sys
from types import SimpleNamespace as NS

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'main'))

from dataset_cfg import resolve_dataset_cfg  # noqa: E402

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: {got!r}" + ('' if ok else f' (want {want!r})'))
    if not ok:
        FAILS.append(name)


def check_raises(name, cfg):
    try:
        resolve_dataset_cfg(cfg, verbose=False)
    except ValueError as e:
        print(f"  OK   {name}: raised — {str(e)[:60]}")
        return
    print(f"  FAIL {name}: 예외가 안 났다")
    FAILS.append(name)


def base(**kw):
    d = dict(dl3dv_root='/x', num_frames=32, geo_encoder='lagernvs',
             geo_latent_cache_dir='/cache', geo_latent_dim=768, pose_source='da3')
    d.update(kw)
    return NS(**d)


def cache_on(cfg):
    return resolve_dataset_cfg(cfg, verbose=False).geo_latent_cache_dir is not None


print('[geo_latent_cache_dir 캐스케이드]')
check('enabled (조건 없음)', cache_on(base()), True)
check('1/4 geo_latent_dim != 768', cache_on(base(geo_latent_dim=1024)), False)
check('2/4 geo_encoder=custom',
      cache_on(base(geo_encoder='custom', custom_geo_input_hw=(252, 448))), False)
check('3/4 geo_test_inseg_k', cache_on(base(geo_test_inseg_k=2)), False)
check('4/4 geo_swap_mode', cache_on(base(geo_swap_mode='inscene')), False)
check('geo_encoder=None (geo 자체 off)', cache_on(base(geo_encoder=None)), False)
check('cfg 에 캐시 dir 자체가 없음', cache_on(base(geo_latent_cache_dir=None)), False)

print('[캐시 하위 디렉토리 = geo_first_view_target_s]')
for v, sub in ((False, 'first_cam_not_included'), (True, 'first_cam_included')):
    check(f'geo_first_view_target_s={v}',
          resolve_dataset_cfg(base(geo_first_view_target_s=v),
                              verbose=False).geo_latent_cache_dir, f'/cache/{sub}')

print('[검증 (ValueError 나야 함)]')
check_raises('avg_scale_ref 오타', base(avg_scale_ref='zzz'))
check_raises('avg_scale_ref=context_first_cam + pose_source!=da3',
             base(pose_source='transforms', avg_scale_ref='context_first_cam'))
check_raises('geo_cam_embed 오타', base(geo_cam_embed='zzz'))
check_raises('geo_swap_mode 오타', base(geo_swap_mode='zzz'))
check_raises('pose_source 오타', base(pose_source='zzz'))
check_raises('custom + pose_source!=da3', base(geo_encoder='custom', pose_source='transforms'))
check_raises('custom_geo_input_hw 가 patch 배수 아님',
             base(geo_encoder='custom', custom_geo_input_hw=(250, 448)))

print('[geo_anchor_first_frame (deprecated) -> geo_cover_centered_at_s]')
check('legacy 만 있으면 그 값을 쓴다',
      resolve_dataset_cfg(base(geo_anchor_first_frame=True),
                          verbose=False).geo_cover_centered_at_s, True)
check('현재 키가 있으면 그쪽이 이긴다',
      resolve_dataset_cfg(base(geo_anchor_first_frame=True, geo_cover_centered_at_s=False),
                          verbose=False).geo_cover_centered_at_s, False)
check('둘 다 없으면 False',
      resolve_dataset_cfg(base(), verbose=False).geo_cover_centered_at_s, False)

print('[custom geo_hw = patch 배수로 내림]')
check('custom_geo_input_hw 명시',
      resolve_dataset_cfg(base(geo_encoder='custom', custom_geo_input_hw=(252, 448)),
                          verbose=False).geo_hw, (252, 448))
check('명시 없으면 geo_image_hw 를 14 배수로 내림',
      resolve_dataset_cfg(base(geo_encoder='custom', geo_image_hw=(256, 448)),
                          verbose=False).geo_hw, (252, 448))
check('non-custom 은 geo_image_hw 그대로',
      resolve_dataset_cfg(base(geo_image_hw=(256, 448)), verbose=False).geo_hw, (256, 448))

print()
if FAILS:
    print(f"{len(FAILS)} FAIL: {FAILS}")
    sys.exit(1)
print('모두 통과')
