"""이 코드가 답하는 질문: subject 3D 궤적을 VLM+SAM3+scene graph 없이 Molmo2 tracking + DA3 depth 로만 만들면 어떻게 되나? (target_track 대체 파일)

R326 (사용자 2026-10-12 "MST 의 T 를 Molmo2 tracking + DA3 로 얻은 3d trajectory (49f) 로 교체해서 학습"). 근거 R323 (v1 test 50편):
기준 target_track (OBB 중심) 대비 3D 오차 중앙 0.209 · 모양 0.101 (카메라-subject 거리로 나눔), 22.8 s/편.

stage
  gen   (GPU, env latentcam) (영상, anchor) 쌍마다 Molmo2-4B "Track the {instance 설명}." (images_4 49f, 메타 fps 2 → 프레임마다 0.5 s)
        → <out>/<video>__<anchor>.json {pts: [[frame, track_id, x, y] 정규화], text 앞부분, sec}. 여러 쌍을 --batch 개씩 묶어 생성.
  lift  (CPU, env vista4d) 점이 가장 많은 track id → 점 주변 depth 중앙값 (반폭 W/80, sky 제외) + recon K·c2w → world.
        빠진 프레임은 선형 보간 (양 끝은 가까운 값), 덮인 프레임 < --min_frames 면 valid False.
        → <corpus>/dynpose/<video>/da3/<--name> (target_track.npz 와 같은 키·형식: track_world (V,T,3) · valid · keys · anchor_id).
대상 문구: out_dynpose/<video>/instance_desc.json descriptions[anchor].description (없으면 scene graph 라벨).
예) CUDA_VISIBLE_DEVICES=0 python fit/bank/molmo2_track_lift.py --stage gen --num_shards 3 --shard_id 0 --batch 8
    python fit/bank/molmo2_track_lift.py --stage lift --workers 24
"""
import argparse, json, os, os.path as osp, re, sys, time
import numpy as np

C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
G = "/data1/cympyc1785/LatentCamVid/camera_generation/dataset/out_dynpose"
E = "/data1/cympyc1785/data/DynPose-100K/eval_data/recon_and_seg"
M = "/data1/cympyc1785/LatentCamVid/camera_generation/tools/molmo2/checkpoints/Molmo2-4B"
COORD = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)")       # 닫는 따옴표 없이도 (2048 토큰 잘림)
FRAME = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
PTS = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
FPS = 2.0


def pairs(corpus):
    """(video, anchor) 전부 — target_track.npz 의 anchor_id 기준 (seg 여러 개가 같은 anchor 를 공유)."""
    out = []
    for v in sorted(os.listdir(f"{corpus}/dynpose")):
        p = f"{corpus}/dynpose/{v}/da3/target_track.npz"
        if not osp.isfile(p):
            continue
        z = np.load(p, allow_pickle=True)
        for a in sorted({str(x) for x in z["anchor_id"]}):
            out.append((v, a))
    return out


def phrase(v, aid):
    q = ""
    try:
        q = json.load(open(f"{G}/{v}/instance_desc.json"))["descriptions"][aid]["description"] or ""
    except Exception:
        pass
    if not q:
        try:
            q = next(n["label"] for n in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"] if n["id"] == aid)
        except Exception:
            q = "main subject"
    q = q[4:] if q.lower().startswith("the ") else q
    return f"Track the {q}."


def parse(text):
    pts = []
    for c in COORD.finditer(text):
        for f in FRAME.finditer(c.group(1)):
            t = float(f.group(1))
            for m in PTS.finditer(f.group(2)):
                pts.append([int(round(t * FPS)), m.group(1), float(m.group(2)) / 1000, float(m.group(3)) / 1000])
    return pts


def stage_gen(a):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    P = pairs(a.corpus)
    P = [p for i, p in enumerate(P) if i % a.num_shards == a.shard_id and not osp.isfile(f"{a.out}/{p[0]}__{p[1]}.json")]
    os.makedirs(a.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(M, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    print(f"[gen] shard {a.shard_id}/{a.num_shards}: {len(P)} 쌍 남음", flush=True)
    t_all = time.time(); done = 0
    for b0 in range(0, len(P), a.batch):
        B = P[b0:b0 + a.batch]
        vids, texts, metas = [], [], []
        for v, aid in B:
            fr = np.stack([np.asarray(Image.open(f"{a.corpus}/dynpose/{v}/images_4/{i:05d}.png").convert("RGB")) for i in range(49)])
            msgs = [{"role": "user", "content": [{"type": "text", "text": phrase(v, aid)}, {"type": "video"}]}]
            texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)); vids.append(fr)
            metas.append({"fps": FPS, "total_num_frames": 49, "duration": 49 / FPS, "frames_indices": list(range(49)), "width": fr.shape[2], "height": fr.shape[1]})
        inp = proc(text=texts, videos=vids, return_tensors="pt", do_sample_frames=False, video_metadata=metas, padding=True).to("cuda")
        t0 = time.time()
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=a.max_tokens, do_sample=False)
        dt = (time.time() - t0) / len(B)
        for (v, aid), row in zip(B, g):
            s = proc.tokenizer.decode(row[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            json.dump(dict(video=v, anchor=aid, prompt=phrase(v, aid), pts=parse(s), text=s[:300], sec=round(dt, 2)), open(f"{a.out}/{v}__{aid}.json", "w"))
        done += len(B)
        if (b0 // a.batch) % 10 == 0:
            el = time.time() - t_all
            print(f"[gen] {done}/{len(P)}  {el / done:.1f} s/쌍  ETA {(len(P) - done) * el / done / 3600:.1f} h", flush=True)
    print(f"=== gen shard {a.shard_id} DONE ===", flush=True)


def lift_video(job):
    a, v = job
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_cameras, load_depths, load_masks
    dst = f"{a.corpus}/dynpose/{v}/da3/{a.name}"
    if osp.isfile(dst) and not a.overwrite:
        return v, "skip"
    z = np.load(f"{a.corpus}/dynpose/{v}/da3/target_track.npz", allow_pickle=True)
    keys, aids = [str(k) for k in z["keys"]], [str(x) for x in z["anchor_id"]]
    try:
        d = load_depths(f"{E}/{v}/depths", dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(f"{E}/{v}/sky_mask", desc=None)).astype(bool)
        Cw, K = load_cameras(f"{E}/{v}/cameras.npz"); Cw, K = np.asarray(Cw, float), np.asarray(K, float)
    except Exception as e:      # noqa: BLE001
        return v, f"error {str(e)[:60]}"
    F_, H, W = d.shape; r = max(2, W // 80)
    per = {}
    for aid in sorted(set(aids)):
        jp = f"{a.out}/{v}__{aid}.json"
        tr = np.zeros((F_, 3), np.float32); ok = False
        if osp.isfile(jp):
            P = json.load(open(jp))["pts"]; ids = {}
            for p in P:
                ids[p[1]] = ids.get(p[1], 0) + 1
            if ids:
                main = max(ids, key=ids.get); got = {}
                for f, tid, x, y in P:
                    if tid != main or not (0 <= f < F_) or f in got:
                        continue
                    u, vv = int(x * W), int(y * H)
                    sl = (f, slice(max(0, vv - r), vv + r + 1), slice(max(0, u - r), u + r + 1))
                    win, m = d[sl], np.isfinite(d[sl]) & (d[sl] > 0) & ~sky[sl]
                    if m.sum() < 3:
                        continue
                    zz = float(np.median(win[m]))
                    fx, fy, cx, cy = (K[f][0], K[f][1], K[f][2], K[f][3]) if K.ndim == 2 else (K[f][0, 0], K[f][1, 1], K[f][0, 2], K[f][1, 2])
                    got[f] = (Cw[f] @ np.array([(u + 0.5 - cx) / fx * zz, (vv + 0.5 - cy) / fy * zz, zz, 1.0]))[:3]
                if len(got) >= a.min_frames:
                    fs = np.array(sorted(got)); X = np.stack([got[f] for f in fs])
                    tr = np.stack([np.interp(np.arange(F_), fs, X[:, c]) for c in range(3)], -1).astype(np.float32); ok = True
        per[aid] = (tr, ok)
    np.savez(dst, track_world=np.stack([per[x][0] for x in aids]), valid=np.array([per[x][1] for x in aids]),
             keys=np.array(keys), anchor_id=np.array(aids))
    return v, f"ok {sum(per[x][1] for x in set(aids))}/{len(set(aids))}"


def stage_lift(a):
    from multiprocessing import Pool
    vids = sorted({v for v, _ in pairs(a.corpus)}); bad = 0
    with Pool(a.workers) as p:
        for i, (v, msg) in enumerate(p.imap_unordered(lift_video, [(a, v) for v in vids], chunksize=4)):
            bad += msg.startswith("error")
            if i % 500 == 0 or msg.startswith("error"):
                print(f"[lift {i}/{len(vids)}] {v} {msg}", flush=True)
    print(f"=== lift DONE {len(vids)} (error {bad}) ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "lift"])
    ap.add_argument("--corpus", default=C)
    ap.add_argument("--out", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/molmo2_track")
    ap.add_argument("--name", default="target_track_molmo2.npz")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--min_frames", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    {"gen": stage_gen, "lift": stage_lift}[a.stage](a)
