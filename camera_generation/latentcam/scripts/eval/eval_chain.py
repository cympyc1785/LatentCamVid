"""이 코드가 답하는 질문: 학습이 끝난 run 하나를 test 에서 평가하고 CLaTr · 궤적 · 충돌 지표까지 한 번에 채점하면?

R489e · R492 · R534 에서 같은 체인 (eval_testset → clatr_score_dir → traj_time_metrics → pred_collision_rate shell/esdf) 을
드라이버마다 복사했다 (CLAUDE.md "두 번째로 쓰게 되면 승격"). 단계마다 산출물이 있으면 건너뛴다 (--force 로 다시).
표로 합치는 것은 `dataset/eval/arm_table.py` 몫이다.

- 생성기 설정은 학습 config 그대로 (eval_testset 이 run 의 config.yaml 을 읽는다). 바꿀 것만 --set 으로 (예 prompts_file, test_seg_list).
- 충돌 채점은 OMP/MKL 4 스레드로 묶는다 — R527 에서 worker 16 × 스레드 329 개로 esdf 가 30 분 멈췄다 (FIX.log 10-24).
env: latentcam (충돌은 vista4d 로 따로 부른다). 예:
    python scripts/eval/eval_chain.py --run 20261023_194408_dsv5fallv2_umt5_..._depthvid --arm noM --out ../../tmp/r535/noM --gpu 2
"""
import os
import subprocess
import sys
import time
from argparse import ArgumentParser

LC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # camera_generation/latentcam
DS = os.path.join(os.path.dirname(LC), "dataset")
PL = "/data1/cympyc1785/miniconda3/envs/latentcam/bin/python"
PV = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"
VA = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_v5f_all"
CK_V5FALL = f"{LC}/checkpoints/clatr_dynpose_v5f_all_v2_last.ckpt"


def log(m):
    print(time.strftime("[%m-%d %H:%M] ") + m, flush=True)


def sh(tag, cmd, cwd, out, env):
    rc = subprocess.call(cmd, cwd=cwd, env=env, stdout=open(f"{out}/{tag}.log", "w"), stderr=subprocess.STDOUT)
    log(f"{tag} rc={rc}")
    if rc:
        raise SystemExit(f"{tag} 실패")


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)                          # results/ 아래 run 폴더 이름
    ap.add_argument("--arm", required=True)                          # 지표 파일의 arm 이름
    ap.add_argument("--out", required=True)                          # 채점 산출물 폴더 (eval_my 쪽 덤프는 --eval_dir)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--ckpt", default="last.pth")
    ap.add_argument("--seed", default="42")
    ap.add_argument("--suffix", default="v5alltest")                 # eval_my/<run>__last__seed42__<suffix>
    ap.add_argument("--eval_dir", default=None)                      # 직접 지정하면 suffix 무시
    ap.add_argument("--set", action="append", default=[])            # eval_testset --set 그대로
    ap.add_argument("--clatr", action="append", default=None)        # name=ckpt (여러 개). 기본 v5fall_v2
    ap.add_argument("--corpus_root", default=VA)
    ap.add_argument("--graph_root", default="out_dynpose_v5")
    ap.add_argument("--stages", default="eval,clatr,traj,coll_shell,coll_esdf")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    ev = a.eval_dir or f"{LC}/eval_my/{a.run}__{os.path.splitext(a.ckpt)[0]}__seed{a.seed}__{a.suffix}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": a.gpu, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    clatr = dict(c.split("=", 1) for c in a.clatr) if a.clatr else {"v5fall_v2": CK_V5FALL}
    st = a.stages.split(",")
    log(f"{a.arm}: {ev}")

    if "eval" in st and (a.force or not os.path.isfile(f"{ev}/metrics.json")):
        sets = sum([["--set", s] for s in a.set], [])
        sh("eval", [PL, "scripts/eval_testset.py", "--run", f"results/{a.run}", "--ckpt", a.ckpt, "--sample-seed", a.seed, "--gpu", a.gpu,
                    "--out", ev, *sets], LC, out, {**os.environ, "OMP_NUM_THREADS": "4"})
    if "clatr" in st:
        for n, ck in clatr.items():
            if a.force or not os.path.isfile(f"{out}/clatr_{n}.json"):
                sh(f"clatr_{n}", [PL, "scripts/eval/clatr_score_dir.py", f"{a.arm}={ev}", "--ckpt", ck, "--standardization", "dynpose49_d200",
                                  "--force", "--out_json", f"{out}/clatr_{n}.json"], LC, out, {**env, "PYTHONPATH": ".:main"})
    if "traj" in st and (a.force or not os.path.isfile(f"{out}/traj/summary.json")):
        sh("traj", [PL, "scripts/eval/traj_time_metrics.py", "--arm", f"{a.arm}={ev}", "--corpus_root", a.corpus_root,
                    "--graph_root", f"{DS}/{a.graph_root}", "--out", f"{out}/traj"], LC, out, env)
    for m in ("shell", "esdf"):
        if f"coll_{m}" in st and (a.force or not os.path.isfile(f"{out}/coll_{m}/summary.json")):
            sh(f"coll_{m}", [PV, "eval/pred_collision_rate.py", "--arm", f"{a.arm}={ev}", "--out", f"{out}/coll_{m}", "--method", m,
                             "--workers", "16", "--graph_root", a.graph_root], DS, out, env)
    log(f"=== eval_chain {a.arm} DONE ===")


if __name__ == "__main__":
    main()
