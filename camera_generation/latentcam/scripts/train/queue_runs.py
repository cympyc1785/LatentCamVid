"""학습 run 대기열 — GPU 가 비는 순서대로 experiment 를 screen 에 띄운다.

WHY: "지금 돌리는 게 끝나면 이거 돌려줘"를 사람이 지키고 앉아 있을 수 없다. 기존
`scripts/relaunch_after_stop.sh` 는 **run 하나 : GPU 하나**가 고정이라, 어느 arm 이 먼저
끝날지 모르는 상황(에폭 수가 같아도 arm 마다 it/s 가 2~3배 차이난다)에서는 먼저 빈 GPU 를
놀리게 된다. 여기서는 **GPU 풀**과 **job 큐**를 분리해서, 비는 대로 앞에서부터 꽂는다.

빈 GPU 판정은 **두 조건의 AND** 다.
  ① `nvidia-smi` memory.used 가 임계 밑. 프로세스가 사라진 것만 보면 안 되는 이유는,
     Ctrl+C 로 죽여도 DataLoader worker 가 몇 초~몇십 초 GPU 를 물고 있어서 직후에 꽂으면
     OOM 으로 죽기 때문이다 (relaunch_after_stop.sh 가 GRACE 를 두는 것과 같은 이유).
  ② 그 GPU 를 `CUDA_VISIBLE_DEVICES` 로 **선점한 학습 프로세스가 없을 것**. 메모리만 보면
     안 되는 이유는 반대 방향의 사고다: molmo2 arm 은 기동 직후 4.5 GB text 캐시를
     디스크에서 읽느라 수 분간 GPU 를 0 MiB 로 둔다. 그 창에서 "비었다"고 읽고 꽂으면
     같은 GPU 에 두 개가 겹친다 (2026-09-15 dry_run 에서 실제로 GPU2/3 이 이렇게 찍혔다).
판정 근거는 `/proc/<pid>/environ` — 명령줄에 GPU 가 안 적히고 환경변수로만 들어가기 때문.

기동한 뒤에는 그 run 의 프로세스가 실제로 뜰 때까지 기다린 다음에야 다음 job 을 본다.
안 그러면 방금 꽂은 GPU 가 (아직 모델을 안 올려서) 계속 "비어 있음"으로 읽혀 같은 GPU 에
두 개가 겹친다.

    # 대기열 정의 (job 순서 = 우선순위)
    cat > /data1/cympyc1785/LatentCamVid/tmp/d196/queue.json <<'EOF'
    {"gpus": [0, 6, 7],
     "log_dir": "/data1/cympyc1785/LatentCamVid/tmp/d196/trainlog",
     "jobs": [{"experiment": "dynpose_d194_molmo2_l21_readout",  "screen": "trainq1"},
              {"experiment": "dynpose_d196_molmo2_l21_readout",  "screen": "trainq2"}]}
    EOF

    # 드라이런 — 지금 꽂으면 어디로 가는지만 본다 (아무것도 안 띄운다)
    python scripts/train/queue_runs.py --jobs .../queue.json --dry_run

    # 실제 대기 (screen 안에서)
    screen -dmS trainqueue bash -c \
      "<latentcam-python> scripts/train/queue_runs.py --jobs .../queue.json \
         >> .../queue.log 2>&1"
"""

from argparse import ArgumentParser
from json import dump, load
from os import makedirs, path
from subprocess import run
from sys import stdout
from time import sleep, strftime

REPO = '/data1/cympyc1785/LatentCamVid/camera_generation/latentcam'
PY = '/data1/cympyc1785/miniconda3/envs/latentcam/bin/python'


def say(msg):
    """줄 단위로 바로 흘려보낸다 — screen 안에서 tail 로 볼 수 있어야 한다."""
    print(f'[{strftime("%m-%d %H:%M:%S")}] {msg}')
    stdout.flush()


def gpu_used_mib(gpu):
    """해당 GPU 의 사용 중 메모리(MiB). 읽기 실패하면 -1 (= 안전하게 '차 있음' 취급)."""
    out = run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
               '-i', str(gpu)], capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (IndexError, ValueError):
        return -1


def claimed_gpus():
    """지금 도는 학습 프로세스들이 `CUDA_VISIBLE_DEVICES` 로 선점한 GPU 집합.

    메모리를 아직 안 잡았어도(캐시 로딩 중) 선점으로 친다. 읽기 실패한 pid 는 그냥 건너뛴다
    — 남의 프로세스거나 방금 죽은 것이고, 그런 GPU 는 어차피 ① 메모리 조건이 막는다."""
    out = run(['pgrep', '-f', r'train_latent_cam_dm\.py'], capture_output=True, text=True)
    claimed = set()
    for pid in out.stdout.split():
        try:
            with open(f'/proc/{pid}/environ', 'rb') as file:
                env = file.read().decode('utf-8', 'replace')
        except OSError:
            continue
        for kv in env.split('\0'):
            if kv.startswith('CUDA_VISIBLE_DEVICES='):
                claimed.update(g.strip() for g in kv.split('=', 1)[1].split(',') if g.strip())
    return claimed


def train_running(experiment):
    """`experiment=<이름>` 으로 도는 학습 프로세스가 있나. 끝을 $ 로 anchor 하는 게 핵심 —
    `dynpose_d194_molmo2_l21` 은 `..._l21_readout` 의 prefix 라 anchor 없으면 오검출한다."""
    pat = rf'train_latent_cam_dm\.py experiment={experiment}$'
    return run(['pgrep', '-f', pat], capture_output=True).returncode == 0


def launch(job, gpu, log_dir, repo, py):
    """screen 하나를 띄우고 그 안에서 학습을 건다. PYTHONPATH 가 빠지면 ModuleNotFoundError
    utils 로 즉사하므로 여기서 한 군데에만 적어 둔다."""
    exp, scr = job['experiment'], job['screen']
    log = job.get('log') or path.join(log_dir, f'train_{exp}.log')
    if path.isfile(log):                       # 이전 run 로그를 덮지 않는다
        run(['mv', log, f'{log[:-4]}_{strftime("%Y%m%d_%H%M%S")}.log'])
    cmd = (f'cd {repo}/main && CUDA_VISIBLE_DEVICES={gpu} PYTHONPATH={repo}:. '
           f'{py} -u train_latent_cam_dm.py experiment={exp} > {log} 2>&1')
    run(['screen', '-dmS', scr, 'bash', '-c', cmd])
    say(f'기동  {exp}  GPU{gpu}  screen={scr}  log={log}')
    return log


def wait_until_up(experiment, timeout, poll):
    """기동한 run 이 실제로 프로세스로 뜰 때까지. 안 뜨면 False (로그를 사람이 봐야 한다)."""
    for _ in range(max(1, timeout // poll)):
        if train_running(experiment):
            return True
        sleep(poll)
    return train_running(experiment)


def main():
    parser = ArgumentParser()
    parser.add_argument("--jobs", required=True, type=str)          # 큐 정의 JSON
    parser.add_argument("--free_mib", default=2000, type=int)       # 이 밑이면 "빈 GPU"
    parser.add_argument("--poll", default=120, type=int)            # GPU 확인 주기(초)
    parser.add_argument("--settle", default=180, type=int)          # 기동 후 프로세스 대기(초)
    parser.add_argument("--repo", default=REPO, type=str)
    parser.add_argument("--py", default=PY, type=str)
    parser.add_argument("--state", default=None, type=str)          # 기본 <jobs>.state.json
    parser.add_argument("--dry_run", action="store_true")           # 배치만 보고 안 띄운다
    args = parser.parse_args()

    with open(args.jobs, encoding="utf-8") as file:
        spec = load(file)
    gpus, jobs = list(spec['gpus']), list(spec['jobs'])
    log_dir = spec.get('log_dir') or path.dirname(path.abspath(args.jobs))
    makedirs(log_dir, exist_ok=True)
    state_path = args.state or f'{args.jobs.rsplit(".", 1)[0]}.state.json'

    say(f'큐 {len(jobs)} job / GPU 풀 {gpus} / 빈 기준 <{args.free_mib} MiB / 주기 {args.poll}s')
    for i, job in enumerate(jobs):
        say(f'  {i + 1}. {job["experiment"]}  -> screen {job["screen"]}')

    done, pending = [], list(jobs)
    while pending:
        claimed = claimed_gpus()
        for gpu in gpus:
            if not pending:
                break
            used = gpu_used_mib(gpu)
            if used < 0 or used >= args.free_mib or str(gpu) in claimed:
                continue
            job = pending[0]
            if args.dry_run:
                say(f'[dry] GPU{gpu} 비어 있음({used} MiB) -> {job["experiment"]}')
                pending.pop(0)
                continue
            pending.pop(0)
            log = launch(job, gpu, log_dir, args.repo, args.py)
            claimed.add(str(gpu))
            ok = wait_until_up(job['experiment'], args.settle, 15)
            done.append({**job, 'gpu': gpu, 'log': log, 'started': strftime('%Y-%m-%d %H:%M:%S'),
                         'process_seen': ok})
            with open(state_path, 'w', encoding="utf-8") as file:
                dump({'done': done, 'pending': pending}, file, ensure_ascii=False, indent=2)
            if not ok:
                say(f'!! 기동 실패로 보임: {job["experiment"]} 프로세스 없음 — {log} 확인')
        if pending and not args.dry_run:
            sleep(args.poll)
        elif pending:                                   # dry_run 은 한 바퀴만 돈다
            say(f'[dry] 남은 job {len(pending)} 개 — 지금 빈 GPU 가 그만큼 없다')
            break

    say('=== 대기열 종료 ===')
    for d in done:
        say(f'  {d["experiment"]:<44} GPU{d["gpu"]}  {d["started"]}  '
            f'{"ok" if d["process_seen"] else "PROCESS NOT SEEN"}')


if __name__ == '__main__':
    main()
