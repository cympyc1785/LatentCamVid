"""`geo_raw_cache_dir` 의 pre-ln DA3 캐시를 다른 dtype 으로 다시 굽는다 (D200 속도 knob A).

WHY: D200 5-arm 학습의 스텝 시간을 계측했더니 62~66%가 **DataLoader** 에서 나왔고, 워커
스택을 보면 그 안은 디스크가 아니라 메모리 복사였다 (`np.load` 0.1% / `collate_fn` 의
`torch.stack` 56.4% + `_share_fd_cpu_` 40.6%). 씬당 `raw (6,576,3072) fp32` = 42.4 MB 라
batch 8 이면 스텝마다 340 MB 를 두 번 복사한다 — stack 으로 한 번, 워커→메인 공유메모리로
한 번. 이 텐서를 bf16 으로 내리면 그 두 복사가 그대로 절반이 된다.

**DA3 재추론이 아니다.** 캐시 파일 안에 들어 있는 건 이미 뽑아 둔 feature 이고 여기서 하는
일은 `raw` 텐서의 dtype 캐스팅뿐이다 (`geo_idxs`/`meta` 는 그대로 복사). 8워커 실측
37.85 files/s.

정밀도: 학습 경로는 `train_latent_cam_dm.py:geo_emb_from_raw_cache` 가 `.float()` 로 되돌린
뒤 `geo_encoder.from_raw` 에 넣는다 — 그 아래 LayerNorm 은 fp32 로 돌고, 이어지는 geo
encoder 는 어차피 bf16 autocast 다. 다만 **비트동일은 깨진다** (bf16 은 가수 8비트).
fp32 캐시로 학습하던 run 을 이 캐시로 이어 붙이면 loss 궤적이 미세하게 갈린다.

원본은 지우지 않는다. `--src` 와 `--dst` 를 다른 디렉토리로 두고, 검증이 끝난 뒤 사람이
지운다 (repo 규칙: 중간 산출물 삭제는 사전 승인).

사용 예시:
  python main/convert_geo_raw_cache.py \
    --src /data1/.../latentcam_dynpose_d200/geo_raw_cache_da3 \
    --dst /data1/.../latentcam_dynpose_d200/geo_raw_cache_da3_bf16 --workers 8
  python main/convert_geo_raw_cache.py --src ... --dst ... --stage verify
"""

from argparse import ArgumentParser
from os import path as osp, makedirs, listdir, replace, remove
from multiprocessing import Pool
import time

import torch

DTYPES = {'bfloat16': torch.bfloat16, 'float16': torch.float16, 'float32': torch.float32}

_G = {}


def _init(src, dst, dtype, key):
    _G.update(src=src, dst=dst, dtype=DTYPES[dtype], key=key)


def _convert(name):
    """파일 하나. 부분 파일이 남지 않게 `.tmp` 로 쓰고 rename 한다 (같은 FS 라 원자적)."""
    src, dst, dtype, key = _G['src'], _G['dst'], _G['dtype'], _G['key']
    out = osp.join(dst, name)
    try:
        o = torch.load(osp.join(src, name), map_location='cpu', weights_only=False)
        if key not in o:
            return (name, 'nokey', 0)
        o[key] = o[key].to(dtype)
        tmp = out + '.tmp'
        torch.save(o, tmp)
        replace(tmp, out)
        return (name, 'ok', osp.getsize(out))
    except Exception as e:                                  # 한 파일 실패로 전체를 죽이지 않는다
        for p in (out + '.tmp',):
            if osp.isfile(p):
                try:
                    remove(p)
                except OSError:
                    pass
        return (name, f'err:{type(e).__name__}:{e}', 0)


def stage_convert(a):
    names = sorted(n for n in listdir(a.src) if n.endswith('.pt'))
    makedirs(a.dst, exist_ok=True)
    done = set(n for n in listdir(a.dst) if n.endswith('.pt')) if a.skip_done else set()
    todo = [n for n in names if n not in done]
    print(f"[convert] src={a.src}\n[convert] dst={a.dst}\n"
          f"[convert] {len(names)} files, {len(done)} already done -> {len(todo)} to do, "
          f"dtype={a.dtype}, workers={a.workers}", flush=True)
    t0 = time.time()
    n_ok = n_err = 0
    nbytes = 0
    with Pool(a.workers, initializer=_init, initargs=(a.src, a.dst, a.dtype, a.key)) as p:
        for i, (name, st, sz) in enumerate(p.imap_unordered(_convert, todo, chunksize=4)):
            if st == 'ok':
                n_ok += 1
                nbytes += sz
            else:
                n_err += 1
                print(f"[convert] FAIL {name}: {st}", flush=True)
            if (i + 1) % 500 == 0 or (i + 1) == len(todo):
                el = time.time() - t0
                print(f"[convert] {i + 1}/{len(todo)}  {(i + 1) / el:.2f} files/s  "
                      f"elapsed {el / 60:.1f} min", flush=True)
    print(f"[convert] done ok={n_ok} err={n_err} wrote={nbytes / 2**30:.1f} GiB "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)
    return n_err == 0


def stage_verify(a):
    """개수·키·dtype·shape 를 전량 확인하고, 표본 몇 개는 값까지 원본과 대조한다."""
    src = sorted(n for n in listdir(a.src) if n.endswith('.pt'))
    dst = sorted(n for n in listdir(a.dst) if n.endswith('.pt'))
    print(f"[verify] src {len(src)} / dst {len(dst)} files  "
          f"missing={len(set(src) - set(dst))} extra={len(set(dst) - set(src))}")
    bad = 0
    for i, n in enumerate(dst):
        try:
            o = torch.load(osp.join(a.dst, n), map_location='cpu', weights_only=False)
            t = o[a.key]
            assert t.dtype == DTYPES[a.dtype] and t.ndim == 3, (n, t.dtype, t.shape)
        except Exception as e:
            bad += 1
            print(f"[verify] BAD {n}: {e}")
        if (i + 1) % 2000 == 0:
            print(f"[verify] {i + 1}/{len(dst)}", flush=True)
    rel = []
    for n in dst[::max(1, len(dst) // a.sample)][:a.sample]:
        s = torch.load(osp.join(a.src, n), map_location='cpu', weights_only=False)[a.key].float()
        d = torch.load(osp.join(a.dst, n), map_location='cpu', weights_only=False)[a.key].float()
        rel.append(((d - s).abs().max() / s.abs().max()).item())
    print(f"[verify] sampled {len(rel)} files, max |Δ|/max|x| = {max(rel):.3e} "
          f"(bf16 상대오차 상한 ~3.9e-3)")
    print(f"[verify] bad={bad}")
    return bad == 0 and not (set(src) - set(dst))


def main():
    ap = ArgumentParser()
    ap.add_argument('--src', required=True)                     # 원본 캐시 디렉토리 (그대로 둔다)
    ap.add_argument('--dst', required=True)                     # 새 dtype 캐시를 쓸 디렉토리
    ap.add_argument('--dtype', default='bfloat16', choices=sorted(DTYPES))
    ap.add_argument('--key', default='raw')                     # 캐스팅할 텐서 키
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--stage', default='all', choices=['convert', 'verify', 'all'])
    ap.add_argument('--sample', type=int, default=8)            # verify 에서 값까지 대조할 파일 수
    ap.add_argument('--skip_done', dest='skip_done', action='store_true')
    ap.add_argument('--no_skip_done', dest='skip_done', action='store_false')
    ap.set_defaults(skip_done=True)
    a = ap.parse_args()

    ok = True
    if a.stage in ('convert', 'all'):
        ok = stage_convert(a) and ok
    if a.stage in ('verify', 'all'):
        ok = stage_verify(a) and ok
    print(f"{'=' * 60}\n결과   : {'OK' if ok else 'FAIL'}\nsrc    : {a.src}\ndst    : {a.dst}\n"
          f"dtype  : {a.dtype}\n{'=' * 60}")


if __name__ == '__main__':
    main()
