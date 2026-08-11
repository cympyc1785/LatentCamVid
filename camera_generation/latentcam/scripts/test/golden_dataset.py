"""dataset 리팩토링용 golden 회귀 하네스 — `ds[i]` 출력을 **bit 단위**로 고정한다.

왜 필요한가: `dataset_dl3dv.CamDataset` 는 cfg 속성 41개가 `__init__`(190줄)과
`__getitem__`(150줄, 13 스테이지)에 얽혀 있다. 여기를 손대면 학습 신호가 **조용히**
바뀌고, 그 뒤의 모든 실험 비교가 무효가 된다 (wandb loss 는 멀쩡해 보인다). 그래서
"기능 유지"를 눈이 아니라 해시로 증명한다:

    리팩토링 전:  --mode dump   -> golden/dataset_golden.json
    리팩토링 후:  --mode check  -> 전 arm 전 key 가 sha1 까지 동일해야 PASS

`_target_out` 을 떼어낼 때 쓴 것과 같은 방식(그 docstring 의 "3개 item 의 전 키 동일
확인")을 전 arm 으로 자동화한 것이다.

arm 선정 기준: **실제 run 이 저장한 config** 를 그대로 쓴다 (`scripts/eval_testset.py::build_cfg`
= 현재 conf/config.yaml 기본값 <- run 의 config.yaml <- override). 손으로 쓴 cfg 는 run 과
어긋나기 때문. 어느 run 도 켠 적 없는 분기(hybrid / random_inseg / first_farthest_135 /
swap / shuffle / posed …)는 실제 run config 위에 override 를 얹은 **synthetic arm** 으로 덮는다
— 이 분기들도 P1 에서 config 해석이 바뀌는 대상이라 안 덮으면 리팩토링이 무증상으로 깨뜨린다.

결정성: `random`/`numpy`/`torch` 시드를 **샘플마다** 고정한다. 그래야 shuffle·random_inseg
arm 도 재현된다. 시드가 고정이라 "무작위 분기가 무작위라서 다르다"와 "리팩토링이 깨뜨렸다"를
구분할 수 있다.

usage
-----
  python scripts/test/golden_dataset.py --mode dump              # 리팩토링 전에 한 번
  python scripts/test/golden_dataset.py --mode check             # 리팩토링 후마다
  python scripts/test/golden_dataset.py --mode dump --arms custom_full sd_whuman --n 3
  python scripts/test/golden_dataset.py --mode list              # arm 목록만
"""
import argparse
import hashlib
import json
import os
import os.path as osp
import random
import sys
import traceback

REPO = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, osp.join(REPO, 'scripts'))
sys.path.insert(0, osp.join(REPO, 'main'))
sys.path.insert(0, REPO)

DEFAULT_OUT = osp.join(osp.dirname(osp.abspath(__file__)), 'golden', 'dataset_golden.json')
SEED = 1234

# (arm 이름, results/<run>, override) — override 가 빈 dict 이면 run config 그대로.
# 앞의 10 개는 실제 run 이고, 뒤의 synthetic 은 어느 run 도 안 켠 분기를 덮는다.
ARMS = [
    # --- 실제 run (저장된 config 그대로) -------------------------------------
    ('custom_full',      '20260808_140209_da3_7k_customgeo_nos', {}),
    ('custom_nodepth',   '20260810_150503_da3_7k_customgeo_nos_nodepth', {}),
    ('custom_rgbonly',   '20260810_150504_da3_7k_customgeo_nos_rgbonly', {}),
    ('lagernvs_plucker', '20260803_211159_dl3dv_geo_worldtraj_camembed_plucker', {}),
    ('lagernvs_relfirst_lagernvsnorm',
     '20260803_013512_dl3dv_geo_worldtraj_camembed_lagernvsnorm', {}),
    ('lagernvs_ctx135_intrraw',
     '20260801_053143_dl3dv_geo_worldtraj_ctxlonger135', {}),
    ('lagernvs_camdistmean', '20260730_235854_dl3dv_geo_worldtraj', {}),
    ('textonly_transforms',  '20260809_212844_da3_7k_textonly', {}),
    ('textonly_asfirstcam',  '20260810_192704_da3_7k_da3pose_asfirstcam', {}),
    ('sd_whuman_custom',     '20260810_010755_sd_whuman_customgeo', {}),

    # --- synthetic: 어느 run 도 안 켠 분기 ------------------------------------
    # 전부 custom_full run 위에 얹는다 (geo 경로가 가장 넓게 켜져 있어 파급을 잘 드러낸다).
    ('syn_sampling_hybrid',  '20260808_140209_da3_7k_customgeo_nos',
     {'geo_view_sampling': 'hybrid'}),
    ('syn_sampling_random_inseg', '20260808_140209_da3_7k_customgeo_nos',
     {'geo_view_sampling': 'random_inseg'}),
    ('syn_scale_first_farthest_135', '20260808_140209_da3_7k_customgeo_nos',
     {'scale_mode': 'first_farthest_135'}),
    ('syn_scale_context_longer', '20260808_140209_da3_7k_customgeo_nos',
     {'scale_mode': 'context_longer'}),
    ('syn_intr_auto',        '20260808_140209_da3_7k_customgeo_nos', {'intr_norm': 'auto'}),
    ('syn_trans_c2w',        '20260808_140209_da3_7k_customgeo_nos', {'trans_repr': 'c2w'}),
    ('syn_shuffle',          '20260808_140209_da3_7k_customgeo_nos',
     {'geo_shuffle_order': True}),
    ('syn_shuffle_keepfirst', '20260808_140209_da3_7k_customgeo_nos',
     {'geo_shuffle_order': True, 'geo_shuffle_keep_first': True}),
    ('syn_swap_inscene',     '20260808_140209_da3_7k_customgeo_nos',
     {'geo_swap_mode': 'inscene'}),
    ('syn_test_inseg_k',     '20260808_140209_da3_7k_customgeo_nos', {'geo_test_inseg_k': 2}),
    ('syn_posed',            '20260808_140209_da3_7k_customgeo_nos', {'geo_posed': True}),
    ('syn_return_idxs',      '20260808_140209_da3_7k_customgeo_nos', {'geo_return_idxs': True}),
]


# --------------------------------------------------------------------------- #
def fingerprint(v):
    """`ds[i]` 의 값 하나 -> 비교 가능한 dict. tensor 는 **원시 바이트 sha1** 이라
    1 ulp 차이도 잡힌다. stats 는 불일치했을 때 원인을 좁히려고 같이 남긴다."""
    import torch
    if isinstance(v, torch.Tensor):
        t = v.detach().contiguous()
        d = {'kind': 'tensor', 'dtype': str(t.dtype), 'shape': list(t.shape),
             'sha1': hashlib.sha1(t.numpy().tobytes()).hexdigest()}
        if t.numel() and t.is_floating_point():
            f = t.float()
            d['stats'] = [round(float(f.min()), 6), round(float(f.max()), 6),
                          round(float(f.mean()), 6)]
        elif t.numel():
            d['stats'] = [int(t.min()), int(t.max())]
        return d
    if isinstance(v, str):
        return {'kind': 'str', 'len': len(v),
                'sha1': hashlib.sha1(v.encode()).hexdigest(), 'head': v[:60]}
    # 그 외(리스트/스칼라)는 repr 로. 지금 경로엔 없지만 키가 늘어나도 조용히 빠지지 않게.
    r = repr(v)
    return {'kind': 'repr', 'sha1': hashlib.sha1(r.encode()).hexdigest(), 'head': r[:60]}


def seed_all(i):
    import numpy as np
    import torch
    random.seed(SEED + i)
    np.random.seed(SEED + i)
    torch.manual_seed(SEED + i)


def build_ds(run_dir, overrides):
    from eval_testset import build_cfg
    cfg, _ = build_cfg(run_dir, dict(overrides))
    name = getattr(cfg, 'dataset_name', None) or 'dl3dv'
    if name == 'dl3dv':
        from dataset_dl3dv import CamDataset
        return CamDataset(cfg=cfg, type='train'), cfg
    if name == 'scene_decoupled':
        from dataset_scene_decoupled import SDCamDataset
        return SDCamDataset(cfg=cfg, type='train'), cfg
    raise ValueError(f"unknown dataset_name {name!r}")


def run_arm(name, run, overrides, n):
    """-> {'samples': {idx: {key: fp}}, ...} 또는 {'error': ...}"""
    run_dir = run if osp.isabs(run) else osp.join(REPO, 'results', run)
    ds, cfg = build_ds(run_dir, overrides)
    total = len(ds.samples)
    # 앞에서 n 개가 아니라 **균등 분포**로 뽑는다: 앞쪽은 한 scene 에 몰려 있어
    # scene 별로 갈리는 분기(avg_scale 파일 유무, da3 depth 유무)를 못 건드린다.
    idxs = sorted({(k * total) // n for k in range(n)}) if total >= n else list(range(total))
    got = {}
    for i in idxs:
        seed_all(i)
        out = ds[i]
        got[str(i)] = {k: fingerprint(v) for k, v in sorted(out.items())}
    return {'run': run, 'overrides': overrides, 'dataset_name':
            getattr(cfg, 'dataset_name', None) or 'dl3dv',
            'n_total': total, 'idxs': idxs, 'samples': got}


def diff_arm(ref, got):
    """-> 사람이 읽을 불일치 목록. 빈 리스트면 동일."""
    bad = []
    if ref.get('idxs') != got.get('idxs'):
        bad.append(f"  샘플 인덱스가 다르다: ref {ref.get('idxs')} vs got {got.get('idxs')}")
        return bad
    if ref.get('n_total') != got.get('n_total'):
        bad.append(f"  샘플 총 개수가 다르다: ref {ref['n_total']} vs got {got['n_total']} "
                   f"(인덱스 구성이 바뀌었으므로 아래 비교는 의미가 약하다)")
    for i, rs in ref['samples'].items():
        gs = got['samples'].get(i)
        if gs is None:
            bad.append(f"  [{i}] 샘플 자체가 없다")
            continue
        for k in sorted(set(rs) | set(gs)):
            r, g = rs.get(k), gs.get(k)
            if r is None:
                bad.append(f"  [{i}] key 추가됨: {k} -> {g}")
            elif g is None:
                bad.append(f"  [{i}] key 사라짐: {k} (ref {r})")
            elif r != g:
                if r.get('shape') != g.get('shape') or r.get('dtype') != g.get('dtype'):
                    bad.append(f"  [{i}] {k}: shape/dtype 변화 "
                               f"{r.get('dtype')}{r.get('shape')} -> {g.get('dtype')}{g.get('shape')}")
                else:
                    bad.append(f"  [{i}] {k}: 값 변화 sha1 {r['sha1'][:8]} -> {g['sha1'][:8]}  "
                               f"stats {r.get('stats')} -> {g.get('stats')}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='check', choices=['dump', 'check', 'list'])
    ap.add_argument('--out', default=DEFAULT_OUT, help='golden json 경로 (dump/check 공통)')
    ap.add_argument('--n', type=int, default=4, help='arm 당 샘플 수')
    ap.add_argument('--arms', nargs='*', default=None, help='arm 이름 필터 (기본 전부)')
    args = ap.parse_args()

    arms = [a for a in ARMS if not args.arms or a[0] in args.arms]
    if args.mode == 'list':
        for name, run, ov in ARMS:
            print(f"  {name:32s} {run}{'  ' + json.dumps(ov) if ov else ''}")
        return
    if args.arms:
        unknown = set(args.arms) - {a[0] for a in ARMS}
        if unknown:
            raise SystemExit(f"모르는 arm: {sorted(unknown)} (--mode list 로 확인)")

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')      # dataset 만 필요, GPU 안 쓴다

    results, failed = {}, []
    for name, run, ov in arms:
        print(f"\n{'=' * 78}\n[{name}] {run} {ov if ov else ''}\n{'=' * 78}", flush=True)
        try:
            results[name] = run_arm(name, run, ov, args.n)
            print(f"[{name}] OK  {len(results[name]['samples'])} 샘플, "
                  f"key {sorted(next(iter(results[name]['samples'].values())))}")
        except Exception as e:
            traceback.print_exc()
            results[name] = {'error': f"{type(e).__name__}: {e}"}
            failed.append(name)

    if args.mode == 'dump':
        os.makedirs(osp.dirname(args.out), exist_ok=True)
        payload = {'seed': SEED, 'n': args.n, 'arms': results}
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=1, sort_keys=True)
        print(f"\n=> {args.out} ({len(results)} arms, 실패 {len(failed)}: {failed})")
        if failed:
            print("실패한 arm 은 golden 에 error 로 기록됐다 — 리팩토링 후에도 같은 error 여야 한다.")
        return

    # check
    if not osp.isfile(args.out):
        raise SystemExit(f"golden 이 없다: {args.out}  (먼저 --mode dump)")
    ref_all = json.load(open(args.out))['arms']
    n_ok, n_bad = 0, 0
    print(f"\n\n{'=' * 78}\nCHECK vs {args.out}\n{'=' * 78}")
    for name, _, _ in arms:
        ref, got = ref_all.get(name), results.get(name)
        if ref is None:
            print(f"  SKIP {name} — golden 에 없다 (arm 이 추가됐으면 dump 를 다시)")
            continue
        if 'error' in ref or 'error' in got:
            same = ref.get('error') == got.get('error')
            print(f"  {'OK  ' if same else 'FAIL'} {name} (error) "
                  f"ref={ref.get('error')} got={got.get('error')}")
            n_ok, n_bad = n_ok + same, n_bad + (not same)
            continue
        bad = diff_arm(ref, got)
        if bad:
            n_bad += 1
            print(f"  FAIL {name} — 불일치 {len(bad)} 건")
            for line in bad[:12]:
                print(line)
            if len(bad) > 12:
                print(f"  ... 외 {len(bad) - 12} 건")
        else:
            n_ok += 1
            print(f"  OK   {name}")
    print(f"\n{n_ok} OK / {n_bad} FAIL")
    raise SystemExit(1 if n_bad else 0)


if __name__ == '__main__':
    main()
