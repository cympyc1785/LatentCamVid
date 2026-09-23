"""TRUMANS 뱅크의 어느 변이가 mesh raycast(벽·바닥·가림·피사체 거리)를 통과하나 — 씬 순회 드라이버.

D266 에서 `tmp/scripts/d266/run_raycast.py` 로 처음 썼고 D271(R22, 737 clip 전량) 에서 두 번째로
쓰게 되어 여기로 승격했다. 옛 판본은 R7 이전 경로(`models/Planner/CinemaTraj`,
`scripts/bank_to_blender_poses.py`)를 박고 있어 그대로는 안 돈다.

씬마다 `fit/convert/bank_to_blender_poses.py --raycast` 를 부른다. 그 스크립트는 통과 변이가
0개면 AssertionError 로 죽는데 (설계), 여기서는 그걸 실패가 아니라 **판정 결과**(`none`)로
기록하고 다음 편으로 간다. 산출물 `<out_root>/<video>/<out_name>/selection.json` 의
`raycast.audit` 가 사다리 **전 칸**의 통과 여부를 들고 있고, 코퍼스 export 는
`vista4d_bank_to_dl3dv.py --raycast_dir` 로 그걸 읽는다.

GPU 불필요 — Blender `scene.ray_cast` 는 CPU 다 (`--gpu` 는 생성되는 render.sh 에만 들어간다).

    python exec/run_raycast.py --videos <list> --bank_dir hole_bank_d266T \
        --out_name raycast_d266T --num_shards 4 --shard_id 0
"""
import json
import re
import subprocess
import sys
import time
from argparse import ArgumentParser
from os import path

HERE = path.dirname(path.dirname(path.abspath(__file__)))          # camera_generation/dataset
PY = "/data1/cympyc1785/miniconda3/envs/vista4d/bin/python"

# "raycast 통과/후보  3 / 3 preset  (6 / 7 변이)"
RE_OK = re.compile(r"통과/후보\s+(\d+)\s*/\s*(\d+)\s*preset\s*\((\d+)\s*/\s*(\d+)\s*변이")
NONE_TOKEN = "레이캐스트를 통과한 변이가 0개다"


def load_videos(spec, output_root, bank_dir):
    """`--videos` 파일/콤마 목록. 비면 `<output_root>/*/<bank_dir>/bank.csv` 가 있는 전부."""
    if spec:
        if path.exists(spec):
            return [ln.strip() for ln in open(spec) if ln.strip() and not ln.startswith("#")]
        return [v for v in spec.split(",") if v]
    import os
    root = path.join(HERE, output_root)
    return sorted(v for v in os.listdir(root) if path.exists(path.join(root, v, bank_dir, "bank.csv")))


def main(a):
    vids = load_videos(a.videos, a.output_root, a.bank_dir)[a.shard_id::a.num_shards]
    rows, t0 = [], time.time()
    summary = path.join(a.log_dir, f"{a.out_name}_s{a.shard_id}.json")
    for i, v in enumerate(vids, 1):
        out = path.join(HERE, a.output_root, v, a.out_name)
        if not path.exists(path.join(HERE, a.output_root, v, a.bank_dir, "poses.npz")):
            row = dict(video=v, verdict="nobank")
        elif a.skip_done and (path.exists(path.join(out, "selection.json"))
                              or path.exists(path.join(out, "none.json"))):
            continue
        else:
            cmd = [PY, "fit/convert/bank_to_blender_poses.py", "--video", v,
                   "--output_root", a.output_root, "--bank_dir", a.bank_dir,
                   "--out", out, "--raycast"] + a.extra
            t1 = time.time()
            p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=a.timeout)
            txt = p.stdout + p.stderr
            with open(path.join(a.log_dir, f"{v}.raycast.log"), "w") as f:
                f.write(txt)
            m = RE_OK.search(txt)
            if m and p.returncode == 0:
                row = dict(video=v, verdict="pass", preset_ok=int(m.group(1)),
                           preset_n=int(m.group(2)), var_ok=int(m.group(3)), var_n=int(m.group(4)))
            elif NONE_TOKEN in txt:
                row = dict(video=v, verdict="none")
                # 전량 기각도 done 으로 남긴다 — 안 남기면 재실행마다 같은 씬을 다시 쏜다.
                import os
                os.makedirs(out, exist_ok=True)
                with open(path.join(out, "none.json"), "w") as f:
                    json.dump({"video": v, "bank_dir": a.bank_dir, "verdict": "none"}, f)
            else:
                row = dict(video=v, verdict="error", rc=p.returncode,
                           tail=txt.strip().splitlines()[-1][:200] if txt.strip() else "")
            row["sec"] = round(time.time() - t1, 1)
        rows.append(row)
        print(f"[rc/s{a.shard_id}] {i}/{len(vids)} {v} {row['verdict']} "
              f"{row.get('var_ok', 0)}/{row.get('var_n', 0)} {row.get('sec', 0)}s", flush=True)
        with open(summary, "w") as f:
            json.dump(rows, f, indent=1)
    print(f"[rc/s{a.shard_id}] ALL DONE {len(rows)}편 {round(time.time() - t0)}s", flush=True)


if __name__ == "__main__":
    q = ArgumentParser()
    q.add_argument("--videos", default="", help="목록 파일 또는 콤마 목록. 비면 bank.csv 있는 전부")
    q.add_argument("--output_root", default="out_trumans")
    q.add_argument("--bank_dir", default="hole_bank_d266T")
    q.add_argument("--out_name", default="raycast_d266T", help="<output_root>/<video>/<out_name>/")
    q.add_argument("--log_dir", required=True)
    q.add_argument("--shard_id", default=0, type=int)
    q.add_argument("--num_shards", default=1, type=int)
    q.add_argument("--timeout", default=1800, type=int)
    q.add_argument("--skip_done", dest="skip_done", action="store_true", default=True)
    q.add_argument("--no_skip_done", dest="skip_done", action="store_false")
    q.add_argument("extra", nargs="*", help="`--` 뒤는 bank_to_blender_poses.py 로 그대로")
    sys.exit(main(q.parse_args()))
