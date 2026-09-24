"""TRUMANS `.blend` 를 **한 번만** 올려 두고 raycast 질의에 답하는 Blender 상주 서버 (D277, R37).

이 코드가 답하는 질문: "같은 recording 의 여러 클립이 fitting 중에 GT mesh raycast 게이트
(clearance / floor_drop / subject_dist / 시선) 를 Blender 를 매번 안 띄우고 물을 수 있나".

**왜.** 지금 raycast 는 Blender 를 질의마다 띄운다 — 로드 10~16 s (1.6 GB blend, 500 오브젝트).
fitting 은 변이마다 이분법 probe 를 수십 번 부르므로 그 구조로는 fit 안에 못 넣는다. 그래서
mesh 를 5 cm 격자로 구워(`lbm/mesh_collision.py`) 대신 썼는데, 격자는 얇은 물체를 부풀려
Blender 판정과 88.0% 만 맞았다 (R27). 여기서는 **Blender 를 recording 당 1회** 띄워 상주시키고
클립들이 소켓으로 묻는다.

**무엇을 캐시하나.**
  - 정적 BVH: 사람(armature 가 deform 하는 mesh)을 뺀 렌더 가능·보이는 오브젝트 전부를
    world 좌표 삼각형으로 합쳐 `BVHTree.FromPolygons` 한 번. `scene.ray_cast` 와 같은 대상이다.
  - 클립별 조준점: `load_clip` 이 그 클립 프레임마다 `frame_set` 1회로 `body_points`(몸통 70/55/42%
    높이띠 median xy — `trumans_scene_probe.body_points` 를 그대로 import) 를 캐시한다. 이후
    `profile` 은 `frame_set` 을 안 한다 — 비용의 대부분이 armature 평가라서다.

**정의** (`trumans_scene_probe.py --verify_poses` 와 같은 열·같은 임계):
  clearance     6방향 광선 최단 히트 (정적만 = 사람 면제와 같다), `probe_distance` 포화
  floor_drop    -Z 광선 첫 정적 히트
  subject_dist  조준점 3개까지 최단 거리
  clear         조준점 중 하나라도 **사람 몸 앞에서** 정적 표면에 안 막힌다.
                Blender 원판은 "첫 히트가 subject" 인데, 조준점은 몸 속이라 몸 표면이 조준점보다
                `--body_margin`(0.15 m) 앞에 있다고 보고 정적 히트가 그보다 멀면 뚫린 것으로 친다.
                사람 BVH(프레임당 ~88만 삼각형)를 49벌 들고 있지 않으려는 근사다 — 일치율은
                `eval/compare_mesh_gates.py --backend server` 가 잰다.

**프로토콜.** localhost TCP, 요청/응답 모두 JSON 한 줄.
  {"op":"ping"}                                        -> {"ok":true,"blend":...,"static_tris":N}
  {"op":"load_clip","clip":"<id>","frames":[f...]}     -> {"ok":true,"n":49}
  {"op":"profile","clip":"<id>","poses":[[[4x4]..F]..K], "per_frame":false}
                                                       -> {"ok":true,"rows":[{clear_frac,...}]*K}
  {"op":"probe","argv":[...]}                          -> `trumans_scene_probe.main(argv)` in-process
  {"op":"shutdown"}
poses 는 **blend world OpenCV c2w** (`MeshClearance.to_blend` / `bank_to_blender_poses` 와 같은 좌표).

    blender -b <rec>.blend --python fit/ingest/blender_raycast_server.py -- --port 0 --ready <file>
"""
import json
import socket
import sys
import time
from argparse import ArgumentParser
from importlib.util import module_from_spec, spec_from_file_location
from os import path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

HERE = path.dirname(path.abspath(__file__))
_spec = spec_from_file_location("_probe", path.join(HERE, "trumans_scene_probe.py"))
PROBE = module_from_spec(_spec)
_spec.loader.exec_module(PROBE)          # `if __name__` 가드라 main 은 안 돈다 — 함수만 빌린다

RENDERABLE = {"MESH", "CURVE", "SURFACE", "META", "FONT"}
CLEARANCE_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def build_static_bvh(scene, skip):
    """사람을 뺀 정적 삼각형 전부 -> BVH 한 그루 (world 좌표)."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    verts, tris, base = [], [], 0
    for obj in scene.objects:
        if obj.type not in RENDERABLE or obj in skip or not obj.visible_get():
            continue
        ev = obj.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        if mesh is None:
            continue
        mesh.calc_loop_triangles()
        mw = ev.matrix_world
        vs = [tuple(mw @ v.co) for v in mesh.vertices]
        verts.extend(vs)
        tris.extend((base + t.vertices[0], base + t.vertices[1], base + t.vertices[2])
                    for t in mesh.loop_triangles)
        base += len(vs)
        ev.to_mesh_clear()
    return BVHTree.FromPolygons(verts, tris, all_triangles=True), len(tris)


class Server:
    def __init__(self, args):
        self.args = args
        self.scene = bpy.context.scene
        self.scene.frame_step = 1
        self.armature, self.human, self.body = PROBE.find_human(self.scene)
        t0 = time.time()
        self.bvh, self.n_tris = build_static_bvh(self.scene, set(self.human) | {self.armature})
        self.t_bvh = time.time() - t0
        self.clips = {}                  # clip id -> (F,3,3) 조준점

    def hit(self, origin, direction, dist):
        d = Vector(direction)
        if d.length < 1e-12 or dist <= 0:
            return float("inf")
        got = self.bvh.ray_cast(Vector(origin), d.normalized(), dist)
        return float(got[3]) if got[0] is not None else float("inf")

    def load_clip(self, req):
        frames = [int(f) for f in req["frames"]]
        aims = []
        for f in frames:
            self.scene.frame_set(f)
            dg = bpy.context.evaluated_depsgraph_get()
            aims.append(PROBE.body_points(self.body, dg))
        self.clips[req["clip"]] = np.asarray(aims, dtype=np.float64)
        return {"ok": True, "n": len(frames)}

    def profile(self, req):
        aims = self.clips[req["clip"]]
        pd, margin = float(self.args.probe_distance), float(self.args.body_margin)
        rows = []
        for path_poses in req["poses"]:
            P = np.asarray(path_poses, dtype=np.float64)
            assert len(P) == len(aims), f"poses {len(P)} != clip frames {len(aims)}"
            per = []
            for i in range(len(P)):
                pos = P[i, :3, 3]
                clr = min([self.hit(pos, d, pd) for d in CLEARANCE_DIRS] + [pd])
                floor = self.hit(pos, (0.0, 0.0, -1.0), 10.0)
                dists = [float(np.linalg.norm(a - pos)) for a in aims[i]]
                clear = any(self.hit(pos, a - pos, max(dd - margin, 1e-3)) == float("inf")
                            for a, dd in zip(aims[i], dists))
                per.append({"clear": clear, "clearance": clr, "floor_drop": floor,
                            "subject_dist": min(dists)})
            row = {"clear_frac": float(np.mean([r["clear"] for r in per])),
                   "min_clearance": float(min(r["clearance"] for r in per)),
                   "min_subject_dist": float(min(r["subject_dist"] for r in per)),
                   "min_floor_drop": float(min(r["floor_drop"] for r in per))}
            if req.get("per_frame"):
                row["frames"] = per
            rows.append(row)
        return {"ok": True, "rows": rows}

    def serve(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(self.args.port)))
        sock.listen(64)
        port = sock.getsockname()[1]
        with open(self.args.ready, "w") as f:
            json.dump({"port": port, "blend": bpy.data.filepath, "static_tris": self.n_tris,
                       "bvh_s": round(self.t_bvh, 2), "pid": __import__("os").getpid()}, f)
        print(f"[srv] ready port {port}  static tris {self.n_tris}  bvh {self.t_bvh:.1f}s", flush=True)
        # 연결마다 스레드 하나, **요청 처리는 잠금 하나로 직렬화** — bpy/BVH 는 스레드 안전이 아니다.
        # fit 프로세스는 연결을 fit 내내 쥐고 있으므로, 연결 단위로 직렬화하면 같은 recording 의
        # 다른 클립이 앞 클립의 fit 이 끝날 때까지 accept 에서 막힌다 (첫 판본이 그랬다).
        # 요청 단위로 잠그면 클립들의 probe 가 번갈아 들어간다.
        # 유휴 종료 — recording 마다 하나씩 떠서 안 내리면 수십 개가 남는다 (첫 대조 뒤 14개
        # 잔존). 열린 연결이 0 이고 마지막 요청 뒤 `--idle_exit` 초가 지나면 스스로 내린다.
        import os
        import threading
        lock, state = threading.Lock(), {"last": time.time(), "open": 0, "stop": False}

        def handle(conn):
            with state_lock:
                state["open"] += 1
            try:
                with conn, conn.makefile("rwb") as fh:
                    for line in fh:
                        try:
                            req = json.loads(line)
                            op = req.get("op")
                            with lock:
                                if op == "ping":
                                    out = {"ok": True, "blend": bpy.data.filepath,
                                           "static_tris": self.n_tris, "clips": len(self.clips)}
                                elif op == "load_clip":
                                    out = self.load_clip(req)
                                elif op == "profile":
                                    out = self.profile(req)
                                elif op == "probe":
                                    # D278. `trumans_scene_probe.py` 를 **같은 인자로 in-process**
                                    # 실행 — 결과 JSON 은 probe 가 `--out` 에 쓴다. 매번 Blender 를
                                    # 새로 띄워 1.6 GB blend 를 읽던 것(clip 당 2회)을 없앤다.
                                    t0 = time.time()
                                    PROBE.main(PROBE.build_parser().parse_args(req["argv"]))
                                    out = {"ok": True, "sec": round(time.time() - t0, 2)}
                                elif op == "shutdown":
                                    state["stop"] = True
                                    out = {"ok": True}
                                else:
                                    out = {"ok": False, "error": f"unknown op {op}"}
                        except Exception as exc:  # noqa: BLE001 — 한 요청이 죽어도 서버는 산다
                            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                        fh.write((json.dumps(out) + "\n").encode())
                        fh.flush()
                        state["last"] = time.time()
                        if state["stop"]:
                            return
            finally:
                with state_lock:
                    state["open"] -= 1
                    state["last"] = time.time()

        state_lock = threading.Lock()
        sock.settimeout(5.0)
        while not state["stop"]:
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                if state["open"] == 0 and time.time() - state["last"] > float(self.args.idle_exit):
                    print(f"[srv] idle {self.args.idle_exit}s -> exit", flush=True)
                    break
                continue
            conn.settimeout(None)
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
        final = self.args.ready[:-4] if self.args.ready.endswith(".tmp") else self.args.ready
        try:
            os.remove(final)                 # 레지스트리 항목을 지워 다음 클라이언트가 새로 띄우게
        except OSError:
            pass

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--port", default=0, type=int)            # 0 = OS 가 고른다 (ready 파일에 적힌다)
    p.add_argument("--ready", required=True)                 # 포트를 적을 JSON 경로
    p.add_argument("--probe_distance", default=1.5, type=float)
    p.add_argument("--body_margin", default=0.15, type=float)
    p.add_argument("--idle_exit", default=600, type=float)    # 연결 없이 이만큼 지나면 종료
    Server(p.parse_args(PROBE.cli_argv())).serve()
