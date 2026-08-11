"""끝난 run 의 **저장된 config 그대로** dataset 을 세워, 고른 context view 가 target
segment `[s, e)` 안으로 들어갔는지 전수 확인한다.

왜 필요한가: leakage-free 여부는 `geo_cover_out_of_seg` / `geo_first_view_target_s` /
`geo_cover_before_only` 세 플래그의 조합으로 결정되고, 코드에는 pool 이 비었을 때
anchor(=frame s)로 떨어지는 fallback 경로가 있다 (`dataset_dl3dv.py:173-174`,
`:917` 의 pad pool). 플래그만 읽고 "안 가져왔다"고 단정하면 그 경로를 놓친다.
그래서 실제로 dataset 이 내보내는 `geo_idxs` 를 받아 `[s, e)` 와 교집합을 센다.

config 은 `scripts/eval_testset.py::build_cfg` 를 그대로 재사용한다 — 현재
`conf/config.yaml` 기본값 위에 run 의 `config.yaml` 을 덮는 식이라, run 이후에 추가된
키도 해석되면서 run 당시 값이 이긴다. 여기서 강제로 켜는 건 `geo_return_idxs` 하나뿐이고
(그래야 `geo_idxs` 가 배치에 붙는다) 이건 **선택 결과를 바꾸지 않는다**.

usage
-----
  python scripts/context_select/verify_geo_leakage.py \
      --run results/20260808_140209_da3_7k_customgeo_nos --n 300
"""
import argparse
import os
import os.path as osp
import sys

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(REPO, 'scripts'))
sys.path.insert(0, osp.join(REPO, 'main'))
sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='results/<run_dir> (config.yaml 필요)')
    ap.add_argument('--n', type=int, default=300, help='검사할 샘플 수 (train split 앞에서부터)')
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--stride', type=int, default=1, help='샘플 인덱스 stride')
    args = ap.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')          # dataset 만 필요, GPU 안 쓴다
    from eval_testset import build_cfg
    # build_cfg 는 (SimpleNamespace, dict) 튜플을 돌려준다
    cfg, _ = build_cfg(osp.join(REPO, args.run) if not osp.isabs(args.run) else args.run,
                       {'geo_return_idxs': True})

    print(f"run   : {args.run}")
    for k in ('geo_view_sampling', 'geo_cover_k', 'geo_cover_out_of_seg',
              'geo_cover_before_only', 'geo_cover_centered_at_s',
              'geo_first_view_target_s', 'geo_cover_subtract_first', 'num_frames'):
        print(f"  {k:26s} {getattr(cfg, k, '<absent>')}")

    from dataset_dl3dv import CamDataset
    ds = CamDataset(cfg, type=args.split)
    n_total = len(ds.samples)
    idxs = list(range(0, n_total, args.stride))[:args.n]
    print(f"\nsamples {n_total} 중 {len(idxs)} 개 검사 (split {args.split}, stride {args.stride})")

    n_leak, n_view0_is_s, worst = 0, 0, []
    for i in idxs:
        scene_idx, s, e, _cap, name = ds.samples[i]
        # frustum_cover 는 selection 함수를 직접 부른다 — __getitem__ 과 같은 경로인데
        # 이미지 디코딩을 건너뛴다. 다른 sampling mode 는 배치를 통째로 받아 geo_idxs 를 읽는다.
        if str(getattr(cfg, 'geo_view_sampling', '')) == 'frustum_cover':
            gi = list(ds._sample_geo_frustum_cover(scene_idx, s, e))
        else:
            gi = ds[i]['geo_idxs'].tolist()
        inside = [j for j in gi if s <= j < e]
        if inside:
            n_leak += 1
            if len(worst) < 10:
                worst.append((name, s, e, gi, inside))
        if gi and gi[0] == s:
            n_view0_is_s += 1

    print(f"\ntarget segment [s,e) 안의 context view 를 가진 샘플 : {n_leak} / {len(idxs)}")
    print(f"view0 == frame s 인 샘플                          : {n_view0_is_s} / {len(idxs)}")
    for name, s, e, gi, inside in worst:
        print(f"  {name}  [s,e)=[{s},{e})  geo_idxs={gi}  겹침={inside}")


if __name__ == '__main__':
    main()
