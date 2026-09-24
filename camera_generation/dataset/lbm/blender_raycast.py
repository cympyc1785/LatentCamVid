"""`fit/ingest/blender_raycast_server.py` 의 클라이언트 + recording 단위 서버 레지스트리 (D277, R37).

이 코드가 답하는 질문: "fitting 프로세스가 GT mesh raycast 를 물을 때, 같은 recording 의
Blender 서버가 이미 떠 있으면 그걸 쓰고 없으면 띄우는 가장 단순한 방법".

레지스트리는 `<reg_dir>/<recording>.json` (포트·pid). 파일이 있고 ping 이 되면 붙고, 아니면
Blender 를 새로 띄워 ready 파일을 기다린다. 서로 다른 클립(=서로 다른 fit 프로세스)이 같은
recording 이면 **같은 서버**를 쓴다 — `.blend` 로드와 정적 BVH 빌드가 recording 당 1회다.
동시에 두 프로세스가 띄우려 하면 `fcntl` 잠금으로 하나만 띄운다.

    from lbm.blender_raycast import RaycastClient
    rc = RaycastClient("<rec uuid>")
    rc.load_clip("tru_xxx", frames)              # 49 개 blend 프레임
    rows = rc.profile("tru_xxx", [poses_blend])  # (F,4,4) blend world OpenCV c2w 목록
"""
import fcntl
import json
import os
import socket
import subprocess
import time
from glob import glob
from os import path

import numpy as np

HERE = path.dirname(path.dirname(path.abspath(__file__)))                 # camera_generation/dataset
SERVER = path.join(HERE, "fit", "ingest", "blender_raycast_server.py")
BLENDER = "/data1/cympyc1785/tools/blender/blender-4.5.9-linux-x64/blender"
BLEND_ROOT = "/data1/cympyc1785/data/trumans/Data_release/Recordings_blend"
REG_DIR = "/data1/cympyc1785/LatentCamVid/tmp/blender_srv"


def blend_of(recording: str) -> str:
    """recording uuid(또는 앞 8자) -> .blend 경로."""
    # 폴더 이름 == blend 이름인 것만 (`bank_to_blender_poses.chunk_paths` 와 같은 규약).
    # `<uuid> - 副本/` 같은 사본 폴더가 섞여 있다 (0a761819 실측).
    hits = [h for h in glob(path.join(BLEND_ROOT, f"{recording}*", "*.blend"))
            if path.basename(path.dirname(h)) == path.splitext(path.basename(h))[0]]
    assert len(hits) == 1, f"{recording}: blend {len(hits)}개 {hits[:3]}"
    return hits[0]


class RaycastClient:
    def __init__(self, recording: str, reg_dir: str = REG_DIR, timeout: float = 600.0,
                 probe_distance: float = 1.5, body_margin: float = 0.15):
        self.recording = recording
        os.makedirs(reg_dir, exist_ok=True)
        self.reg = path.join(reg_dir, f"{recording[:8]}.json")
        self.lock = self.reg + ".lock"
        self.timeout, self.pd, self.margin = timeout, probe_distance, body_margin
        self.sock = None
        self._connect_or_spawn()

    # ── 연결 ────────────────────────────────────────────────────────────────────────────
    def _try(self):
        if not path.exists(self.reg):
            return False
        try:
            info = json.load(open(self.reg))
            s = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=5)
            s.settimeout(self.timeout)
            self.sock, self.fh = s, s.makefile("rwb")
            return bool(self._call({"op": "ping"}).get("ok"))
        except (OSError, ValueError, KeyError):
            self.sock = None
            return False

    def _connect_or_spawn(self):
        if self._try():
            return
        with open(self.lock, "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)           # 두 프로세스가 동시에 띄우지 않게
            if self._try():
                return
            ready = self.reg + ".tmp"
            if path.exists(ready):
                os.remove(ready)
            env = dict(os.environ)
            log = open(self.reg.replace(".json", ".log"), "w")
            subprocess.Popen([BLENDER, "-b", blend_of(self.recording), "--python", SERVER, "--",
                              "--port", "0", "--ready", ready,
                              "--probe_distance", str(self.pd), "--body_margin", str(self.margin)],
                             stdout=log, stderr=log, env=env, start_new_session=True)
            t0 = time.time()
            while not path.exists(ready):
                assert time.time() - t0 < 300, f"Blender 서버가 300 s 안에 안 떴다 (log {log.name})"
                time.sleep(0.5)
            time.sleep(0.2)
            os.replace(ready, self.reg)
            assert self._try(), "서버는 떴는데 연결이 안 된다"

    def _call(self, req):
        self.fh.write((json.dumps(req) + "\n").encode())
        self.fh.flush()
        out = json.loads(self.fh.readline())
        if not out.get("ok"):
            raise RuntimeError(f"raycast 서버 오류: {out.get('error')}")
        return out

    # ── API ─────────────────────────────────────────────────────────────────────────────
    def load_clip(self, clip: str, frames):
        return self._call({"op": "load_clip", "clip": clip, "frames": [int(f) for f in frames]})

    def profile(self, clip: str, poses_list, per_frame: bool = False):
        poses = [np.asarray(p, dtype=np.float64).round(7).tolist() for p in poses_list]
        return self._call({"op": "profile", "clip": clip, "poses": poses,
                           "per_frame": per_frame})["rows"]

    def probe(self, argv):
        """`trumans_scene_probe.py` 를 서버 안에서 같은 인자로 돌린다 (D278). -> 서버 측 소요 초."""
        return self._call({"op": "probe", "argv": [str(x) for x in argv]})["sec"]

    def shutdown(self):
        try:
            self._call({"op": "shutdown"})
        finally:
            if path.exists(self.reg):
                os.remove(self.reg)
