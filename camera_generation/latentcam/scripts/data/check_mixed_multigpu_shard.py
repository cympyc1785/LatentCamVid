"""multi-GPU 에서 `PerCorpusBatchSampler` 의 "배치 하나 = 코퍼스 하나" 가 유지되는지 본다.

왜 필요한가: `num_processes > 1` 이면 accelerate 가 우리 batch_sampler 를
`BatchSamplerShard` 로 감싼다 (`accelerate/data_loader.py`). 그 래퍼가 배치를 프로세스 수로
쪼개거나 tail 을 padding 하면서 **다른 코퍼스의 인덱스를 한 배치에 섞을 수 있다** — 그러면
`collate_fn` 의 `torch.stack(images (V,3,H,W))` 가 코퍼스별 V 차이(DL3DV 6 / SD 6 /
DataDoP 1) 때문에 터진다. 학습이 시작하자마자 죽으므로 조용한 실패는 아니지만, 2-GPU 로
올리기 전에 여기서 잡는 게 싸다.

GPU 도 데이터도 안 쓴다 — 인덱스만 보고 코퍼스 라벨을 조회한다. 실제 DataLoader 를 도는 게
아니라 `BatchSamplerShard` 에 sampler 를 직접 물려 프로세스별로 순회한다.

usage:
    python scripts/data/check_mixed_multigpu_shard.py [--num_processes 2] [--batch_size 8]
"""
import argparse
import sys

sys.path.insert(0, "main")

from accelerate.data_loader import BatchSamplerShard  # noqa: E402

from mixed_sampler import PerCorpusBatchSampler  # noqa: E402


def run(corpus_of, n_c, bs, nproc, shuffle, drop_last, tag, interleave=None):
    kw = dict(batch_size=bs, weights=None, shuffle=shuffle, drop_last=drop_last,
              seed=42, num_corpora=n_c)
    if interleave is not None:
        kw["interleave"] = interleave
    base = PerCorpusBatchSampler(corpus_of, **kw)

    bad, seen, sizes = [], 0, set()
    for rank in range(nproc):
        sh = BatchSamplerShard(base, num_processes=nproc, process_index=rank,
                               split_batches=False, even_batches=True)
        for bi, batch in enumerate(sh):
            cs = {corpus_of[i] for i in batch}
            seen += 1
            sizes.add(len(batch))
            if len(cs) != 1:
                bad.append((rank, bi, sorted(cs), len(batch)))
    status = "OK" if not bad else f"FAIL {len(bad)}"
    print(f"  [{tag}] nproc={nproc} base_batches={len(base)} sharded_batches={seen} "
          f"batch_sizes={sorted(sizes)} 혼합배치={len(bad)}  -> {status}")
    for r in bad[:5]:
        print(f"      rank{r[0]} batch{r[1]} corpora={r[2]} size={r[3]}")
    return len(bad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_processes", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs=3, default=[29414, 12689, 18344],
                    help="코퍼스별 train sample 수 (기본 = mix_dl3dv_sd_datadop_v1 실측)")
    a = ap.parse_args()

    corpus_of = [c for c, n in enumerate(a.sizes) for _ in range(n)]
    n_c = len(a.sizes)
    print(f"corpora {a.sizes} (총 {len(corpus_of)}), batch_size(=num_gpus*bs 는 아래서 곱함)"
          f" {a.batch_size}")

    total_bad = 0
    for nproc in a.num_processes:
        # base.py 는 bs = num_gpus * cfg.batch_size 로 준다 -> nproc 배로 키우고
        # BatchSamplerShard 가 프로세스마다 다시 쪼갠다.
        bs = a.batch_size * nproc
        total_bad += run(corpus_of, n_c, bs, nproc, True, True, "train")
        total_bad += run(corpus_of, n_c, bs, nproc, False, True, "val", interleave=True)
    print(f"\n총 혼합 배치 {total_bad} -> {'PASS' if total_bad == 0 else 'FAIL'}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
