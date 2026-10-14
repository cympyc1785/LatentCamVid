"""이 코드가 답하는 질문: text 로 subject 를 찾는 방법들 (SAM3 noun / SAM3 설명 문장 / Molmo2 점 → SAM3 / noun 후보 + Molmo2 점 종합 규칙) 중 무엇이 GT subject 를 잘 찾나?

R376 (사용자 2026-10-14 "SAM3 noun, SAM3 desc, Molmo2 pt+SAM3 비교는 1순위, 2순위 100 개씩 나눠서 … Molmo2 비교하는 것들은 1순위 1000 개").
R345 · R355 · R371 · R373 일회성 실험을 일반화했다. 결과는 `viz/subject_find_overlay.py --root <out>` 로 그대로 볼 수 있다.

쌍: v1 s91 test 영상을 seed 0 으로 섞어 앞에서부터, 영상마다 route (`preset_route_<--route>.json` `anchor_ids`, 순위 = route 가
고른 순서) 의 `--rank` 번째 anchor 가 target_track 에 valid 로 있으면 채택 → `<out>/pairs.json`. (`--rank 0` = 옛 R345 규칙: valid anchor 중 무작위)
stage gen  (env latentcam, GPU) Molmo2-4B pointK8 — 8 프레임 이미지 batch "Point to {desc}" + 점별 좌표 토큰 평균 log-prob → gen/<pair>.json
stage sam3 (env vista4d, GPU) SAM3 noun (라벨) text 전파 → 후보 masklet 전부 → sam3_label (평균 점수×프레임 수 최고) + 규칙 4 종
           earliest (가장 이른 단일 점 프레임) · vote (단일 점들이 든 masklet 다수결, 동률은 이른 프레임) · conf (log-prob 최고 점) · wsum (Σ exp(conf))
           맞는 masklet 이 없으면 pointK8f (v1 `molmo2_sam3/`, R346 Molmo2 점 단독 SAM3) 로 대체. 앞 `--desc_n` 쌍은 SAM3 설명 문장 단독 (sam3_desc) 도.
stage eval (env vista4d, CPU) GT = scene graph anchor 의 SAM3 instance mask · target_track. 쌍마다 방법별 IoU 평균 · IoU≥0.5 프레임 비율 (hit) ·
           mask depth 점 축별 10~90% 중점의 3D 오차 / 카메라-GT 거리 (프레임 중앙값). 요약은 R371 표와 같은 열.
SHARD/NSHARD 환경변수로 쌍 나눔.  예) python eval/subject_find_compare.py --out tmp/r376/rank1 --rank 1 --n 1000 --desc_n 100 --stage pairs
"""
import argparse
import json
import os
import os.path as osp
import random
import shutil
import sys
import time

import numpy as np

R = "/data1/cympyc1785/LatentCamVid"
C = "/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200"
G = f"{R}/camera_generation/dataset/out_dynpose"
E = "/data1/cympyc1785/data/DynPose-100K/eval_data/recon_and_seg"
SEG = f"{R}/DATA/DynPose-100K/eval_data"
V4 = f"{R}/video_generation/models/Vista4D"
RULES = ("earliest", "vote", "conf", "wsum")
SHARD = (int(os.environ.get("SHARD", "0")), int(os.environ.get("NSHARD", "1")))
A = None
IDX = {}
log = lambda m: print(time.strftime("[%m-%d %H:%M:%S] ") + m, flush=True)


def methods():
    return ("sam3_label", "sam3_desc", "pointK8f") + tuple(f"rule_{r}" for r in RULES)


def pairs():
    p = f"{A.out}/pairs.json"
    if osp.isfile(p):
        return json.load(open(p))
    vids = []
    for sp in A.splits.split(","):                          # 앞 split 을 다 쓰고 모자라면 다음 split (Molmo2·SAM3 는 우리 데이터 학습 안 함)
        segs = [l.strip().split("/") for l in open(f"{C}/seg_list_dynpose_s91_{sp}.txt") if l.strip()]
        vs = sorted({v for _, v, _ in segs} - set(vids)); random.Random(0).shuffle(vs); vids += vs
    out = []
    for v in vids:
        g = np.load(f"{C}/dynpose/{v}/da3/target_track.npz", allow_pickle=True)
        ok = sorted({str(a) for a, val in zip(g["anchor_id"], g["valid"]) if val})
        if A.rank == 0:
            if ok:
                out.append([v, random.Random(len(out)).choice(ok)])
        else:
            rp = f"{G}/{v}/preset_route_{A.route}.json"
            ids = json.load(open(rp)).get("anchor_ids", []) if osp.isfile(rp) else []
            if len(ids) >= A.rank and ids[A.rank - 1] in ok:
                out.append([v, ids[A.rank - 1]])
        if len(out) >= A.n:
            break
    os.makedirs(A.out, exist_ok=True); json.dump(out, open(p, "w")); return out


def mine():
    return [p for i, p in enumerate(pairs()) if i % SHARD[1] == SHARD[0]]


def phrase(v, aid):
    sys.path.insert(0, f"{R}/camera_generation/dataset/fit/bank")
    import molmo2_track_lift as T
    return T.phrase(v, aid)[len("Track "):]


def stage_gen():
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    sys.path.insert(0, f"{R}/camera_generation/dataset/fit/bank")
    import molmo2_track_lift as T
    os.makedirs(f"{A.out}/gen", exist_ok=True)
    for d in A.reuse_gen:                                   # 같은 규칙으로 이미 만든 gen json 재사용
        for v, aid in mine():
            src, dst = f"{d}/gen/{v}__{aid}.json", f"{A.out}/gen/{v}__{aid}.json"
            if osp.isfile(src) and not osp.isfile(dst):
                shutil.copy(src, dst)
    P = [p for p in mine() if not osp.isfile(f"{A.out}/gen/{p[0]}__{p[1]}.json")]
    log(f"gen shard {SHARD}: {len(P)} 쌍 (재사용 뒤)")
    if not P:
        log(f"=== gen shard {SHARD[0]} DONE ==="); return
    proc = AutoProcessor.from_pretrained(T.M, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(T.M, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    tok = proc.tokenizer; stop = {tok.eos_token_id, tok.pad_token_id}; K8 = T.K_FRAMES["pointK8"]; t0 = time.time()
    for n, (v, aid) in enumerate(P):
        q = "Point to " + phrase(v, aid)
        msgs = [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image"}]}]
        txt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ims = [Image.open(f"{C}/dynpose/{v}/images_4/{f:05d}.png").convert("RGB") for f in K8]
        inp = proc(text=[txt] * len(K8), images=ims, return_tensors="pt", padding=True)
        inp = {k: (x.to("cuda") if hasattr(x, "to") else x) for k, x in dict(inp).items()}
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=64, do_sample=False, output_scores=True, return_dict_in_generate=True)
        L = inp["input_ids"].shape[1]; pts = []
        for b, fk in enumerate(K8):
            ids = g.sequences[b, L:].tolist()
            ids = ids[:next((i for i, t in enumerate(ids) if t in stop), len(ids))]
            lp = [float(torch.log_softmax(g.scores[i][b].float(), -1)[t]) for i, t in enumerate(ids)]
            text, owner = "", []
            for i, pc in enumerate(tok.decode([t], skip_special_tokens=False) for t in ids):
                text += pc; owner += [i] * len(pc)
            for c in T.COORD.finditer(text):
                base = c.start(1)
                for m in T.PTS.finditer(c.group(1)):
                    tk = {owner[base + j] for g_ in (2, 3) for j in range(m.start(g_), m.end(g_)) if base + j < len(owner)}
                    pts.append([fk, m.group(1), float(m.group(2)) / 1000, float(m.group(3)) / 1000, float(np.mean([lp[k] for k in tk])) if tk else -99.0])
        json.dump(dict(video=v, anchor=aid, prompt=q, pts=pts, pointK8=dict(sec=0.0, pts=pts)), open(f"{A.out}/gen/{v}__{aid}.json", "w"))
        if n % 50 == 0:
            el = time.time() - t0; log(f"gen {n + 1}/{len(P)} {el / (n + 1):.1f} s/쌍 ETA {(len(P) - n - 1) * el / (n + 1) / 60:.0f} 분")
    log(f"=== gen shard {SHARD[0]} DONE ===")


def _propagate(pred, frames, text):
    from PIL import Image
    T_ = len(frames)
    sid = pred.handle_request({"type": "start_session", "resource_path": [Image.fromarray(frames[i]) for i in range(T_)]})["session_id"]
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": text}); per = {}
    for r in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid, "propagation_direction": "forward",
                                         "start_frame_index": 0, "max_frame_num_to_track": T_}):
        o = r["outputs"]
        if o is None:
            continue
        for k, oid in enumerate(o["out_obj_ids"]):
            dd = per.setdefault(int(oid), {"p": [], "m": {}})
            dd["p"].append(float(o["out_probs"][k])); dd["m"][r["frame_index"]] = np.asarray(o["out_binary_masks"][k]).astype(bool)
    pred.handle_request({"type": "close_session", "session_id": sid})
    return per


def stage_sam3():
    sys.path.insert(0, V4)
    from utils.media import load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video
    for m in methods():
        os.makedirs(f"{A.out}/sam3/{m}", exist_ok=True)
    desc_set = {tuple(p) for p in pairs()[:A.desc_n]}
    pred = init_sam3_video(); stat = {}; t0 = time.time()
    P = [p for p in mine() if not osp.isfile(f"{A.out}/sam3/rule_wsum/{p[0]}__{p[1]}.npz")
         or (tuple(p) in desc_set and not osp.isfile(f"{A.out}/sam3/sam3_desc/{p[0]}__{p[1]}.npz"))]
    log(f"sam3 shard {SHARD}: {len(P)} 쌍")
    best_of = lambda per: max(per, key=lambda i: np.mean(per[i]["p"]) * len(per[i]["m"])) if per else None
    for n, (v, aid) in enumerate(P):
        frames, _ = load_video(f"{E}/{v}/video.mp4"); T_, H, W, _ = frames.shape

        def save(m, per, i, extra=None):
            out = f"{A.out}/sam3/{m}/{v}__{aid}.npz"
            if i is None:
                np.savez(out, ok=False, how="no_detection"); return "none"
            M = np.zeros((T_, H, W), bool)
            for f, mk in per[i]["m"].items():
                M[f] = mk
            np.savez_compressed(out, ok=True, masks=np.packbits(M, axis=-1), W=W, how="masklet", conf=float(np.mean(per[i]["p"])), n_obj=len(per))
            return "masklet"
        label = next((x.get("label") for x in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"] if x["id"] == aid), "") or ""
        desc = phrase(v, aid).rstrip(".")
        if (v, aid) in desc_set and not osp.isfile(f"{A.out}/sam3/sam3_desc/{v}__{aid}.npz"):
            perd = _propagate(pred, frames, desc); k = save("sam3_desc", perd, best_of(perd))
            stat.setdefault("sam3_desc", {}); stat["sam3_desc"][k] = stat["sam3_desc"].get(k, 0) + 1
        if osp.isfile(f"{A.out}/sam3/rule_wsum/{v}__{aid}.npz"):
            continue
        per = _propagate(pred, frames, label) if label else {}
        gp = f"{A.out}/gen/{v}__{aid}.json"
        Pt = json.load(open(gp))["pts"] if osp.isfile(gp) else []; nf = {}
        for q in Pt:
            nf[q[0]] = nf.get(q[0], 0) + 1
        S = [q for q in Pt if nf[q[0]] == 1] or Pt

        def hit(q):
            f = int(min(T_ - 1, max(0, q[0]))); u, vv = min(W - 1, int(q[2] * W)), min(H - 1, int(q[3] * H))
            return [i for i in per if f in per[i]["m"] and per[i]["m"][f][vv, u]]
        bp = lambda hs: max(hs, key=lambda i: np.mean(per[i]["p"])) if hs else None
        picks = {"sam3_label": best_of(per)}
        if S and per:
            picks["rule_earliest"] = bp(hit(min(S, key=lambda q: q[0]))); picks["rule_conf"] = bp(hit(max(S, key=lambda q: q[4])))
            vote, wsum, first = {}, {}, {}
            for q in S:
                for i in hit(q):
                    vote[i] = vote.get(i, 0) + 1; wsum[i] = wsum.get(i, 0.0) + float(np.exp(q[4])); first[i] = min(first.get(i, 99), q[0])
            picks["rule_vote"] = max(vote, key=lambda i: (vote[i], -first[i])) if vote else None
            picks["rule_wsum"] = max(wsum, key=lambda i: wsum[i]) if wsum else None
        fb = f"{C}/molmo2_sam3/{v}__{aid}.npz"
        fbz = dict(np.load(fb)) if osp.isfile(fb) else dict(ok=False)
        np.savez_compressed(f"{A.out}/sam3/pointK8f/{v}__{aid}.npz", **fbz)
        for m in ("sam3_label",) + tuple(f"rule_{r}" for r in RULES):
            i = picks.get(m)
            if i is None and m != "sam3_label":
                np.savez_compressed(f"{A.out}/sam3/{m}/{v}__{aid}.npz", **{**fbz, "how": "fallback"}); k = "fallback"
            else:
                k = save(m, per, i)
            stat.setdefault(m, {}); stat[m][k] = stat[m].get(k, 0) + 1
        if n % 50 == 0:
            el = time.time() - t0; log(f"sam3 {n + 1}/{len(P)} {el / (n + 1):.1f} s/쌍 ETA {(len(P) - n - 1) * el / (n + 1) / 60:.0f} 분 · {stat}")
    json.dump(stat, open(f"{A.out}/sam3_stat_s{SHARD[0]}.json", "w")); log(f"경로 {stat}"); log(f"=== sam3 shard {SHARD[0]} DONE ===")


def eval_one(p):
    v, aid = p
    sys.path.insert(0, V4)
    from utils.media import load_depths, load_masks, load_cameras
    d = load_depths(f"{E}/{v}/depths", dtype=np.float16, desc=None).astype(np.float32)
    sky = np.asarray(load_masks(f"{E}/{v}/sky_mask", desc=None)).astype(bool)
    Cw, K = load_cameras(f"{E}/{v}/cameras.npz"); Cw, K = np.asarray(Cw, float), np.asarray(K, float)
    F_, H, W = d.shape
    g = np.load(f"{C}/dynpose/{v}/da3/target_track.npz", allow_pickle=True)
    GT = g["track_world"][[str(a) for a in g["anchor_id"]].index(aid)].astype(float)
    dist = np.linalg.norm(GT - Cw[:len(GT), :3, 3], axis=-1).clip(1e-6)
    node = {n["id"]: n for n in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"]}[aid]
    sub = "seg_instances" if aid.startswith("dyn") else "seg_instances_static"
    meta = json.load(open(f"{SEG}/{sub}/{v}/meta.json")); mz = np.load(f"{SEG}/{sub}/{v}/masks.npz")
    ids = {node.get("track_id")} | set(node.get("merged_from") or []); gm = {}
    for f in range(min(F_, len(meta["frames"]))):
        k = [i for i, b in enumerate(meta["frames"][f]) if b.get("id") in ids]
        if k:
            gm[f] = np.unpackbits(mz[f"{f:05d}"][k], axis=-1, count=meta["width"]).astype(bool).any(0)
    res = dict(video=v, anchor=aid, kind=aid.split("_")[0], rank=A.rank, idx=IDX.get((v, aid), -1))
    for m in methods():
        pth = f"{A.out}/sam3/{m}/{v}__{aid}.npz"
        if not osp.isfile(pth):
            continue
        z = np.load(pth)
        res[f"{m}_how"] = str(z["how"]) if "how" in z.files else ""
        if not bool(z["ok"]):
            res.update({f"{m}_err": float("inf"), f"{m}_iou": 0.0, f"{m}_hit": 0.0}); continue
        M = np.unpackbits(z["masks"], axis=-1, count=int(z["W"])).astype(bool); ious, hits, errs = [], [], []
        for f in range(F_):
            mf = M[f] if f < len(M) else np.zeros_like(M[0])
            if f in gm:
                gf = gm[f] if gm[f].shape == mf.shape else gm[f][(np.arange(mf.shape[0]) * gm[f].shape[0] // mf.shape[0])][:, (np.arange(mf.shape[1]) * gm[f].shape[1] // mf.shape[1])]
                u = (gf | mf).sum(); iou = (gf & mf).sum() / u if u else 0.0; ious.append(iou); hits.append(iou >= 0.5)
            md = mf[(np.arange(H) * mf.shape[0] // H)][:, (np.arange(W) * mf.shape[1] // W)] & np.isfinite(d[f]) & (d[f] > 0) & ~sky[f]
            if md.sum() < 10 or f >= len(GT):
                continue
            vv, uu = np.nonzero(md); z_ = d[f][md]
            fx, fy, cx, cy = (K[f][0], K[f][1], K[f][2], K[f][3]) if K.ndim == 2 else (K[f][0, 0], K[f][1, 1], K[f][0, 2], K[f][1, 2])
            Xw = (Cw[f] @ np.stack([(uu + 0.5 - cx) / fx * z_, (vv + 0.5 - cy) / fy * z_, z_, np.ones_like(z_)], 0))[:3].T
            lo, hi = np.percentile(Xw, 10, 0), np.percentile(Xw, 90, 0)
            errs.append(np.linalg.norm((lo + hi) / 2 - GT[f]) / dist[f])
        res.update({f"{m}_err": float(np.median(errs)) if errs else float("inf"), f"{m}_iou": float(np.mean(ious)) if ious else 0.0,
                    f"{m}_hit": float(np.mean(hits)) if hits else 0.0})
    return res


def stage_cands():
    """[R380] 시각화용: 앞 `--cands_n` 쌍에서 SAM3 noun · 설명 문장 text 의 **후보 masklet 전부** 를 저장 (선정 규칙 없이 출력 그대로 보기 위해).
    → cands/{label,desc}/<pair>.npz: masks (K,T,H,W/8) packbits · W · ids · score (masklet 평균 점수) · nfr (등장 프레임 수) · text."""
    sys.path.insert(0, V4)
    from utils.media import load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video
    for k in ("label", "desc"):
        os.makedirs(f"{A.out}/cands/{k}", exist_ok=True)
    P = [p for p in pairs()[:A.cands_n] if not osp.isfile(f"{A.out}/cands/desc/{p[0]}__{p[1]}.npz")]
    P = [p for i, p in enumerate(P) if i % SHARD[1] == SHARD[0]]
    pred = init_sam3_video(); log(f"cands shard {SHARD}: {len(P)} 쌍")
    for n, (v, aid) in enumerate(P):
        frames, _ = load_video(f"{E}/{v}/video.mp4"); T_, H, W, _ = frames.shape
        label = next((x.get("label") for x in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"] if x["id"] == aid), "") or ""
        for k, text in (("label", label), ("desc", phrase(v, aid).rstrip("."))):
            per = _propagate(pred, frames, text) if text else {}
            ids = sorted(per, key=lambda i: -np.mean(per[i]["p"]) * len(per[i]["m"]))
            M = np.zeros((len(ids), T_, H, W), bool)
            for j, i in enumerate(ids):
                for f, mk in per[i]["m"].items():
                    M[j, f] = mk
            np.savez_compressed(f"{A.out}/cands/{k}/{v}__{aid}.npz", masks=np.packbits(M, axis=-1), W=W, ids=np.array(ids, int),
                                score=np.array([np.mean(per[i]["p"]) for i in ids]), nfr=np.array([len(per[i]["m"]) for i in ids]), text=text)
        log(f"cands {n + 1}/{len(P)} {v[:8]} {aid}")
    log(f"=== cands shard {SHARD[0]} DONE ===")


def stage_eval():
    from multiprocessing import Pool
    global IDX
    IDX = {tuple(p): i for i, p in enumerate(pairs())}
    with Pool(A.workers) as p:
        rows = p.map(eval_one, [tuple(x) for x in pairs()])
    json.dump(rows, open(f"{A.out}/eval.json", "w"), indent=1)
    for sub in (None, "dyn", "stat"):
        S = [r for r in rows if sub is None or r["kind"] == sub]
        if not S:
            continue
        print(f"\n[{sub or 'all'}] n {len(S)}")
        print("방법 | n | 맞은 프레임 비율 평균 | 쌍 성공 (≥0.5) | IoU 평균 | 3D 오차 중앙 | 3D 오차 평균 (유한) | 3D <0.2")
        for m in methods():
            T = [r for r in S if f"{m}_iou" in r]
            if not T:
                continue
            h = np.array([r[f"{m}_hit"] for r in T]); e = np.array([r[f"{m}_err"] for r in T], float); f_ = np.isfinite(e)
            print(f"{m} | {len(T)} | {h.mean():.3f} | {np.mean(h >= 0.5):.3f} | {np.mean([r[f'{m}_iou'] for r in T]):.3f} | "
                  f"{np.median(e[f_]) if f_.any() else float('nan'):.3f} | {e[f_].mean() if f_.any() else float('nan'):.3f} | {np.mean(e < 0.2):.3f}")
    print("=== subject_find_compare eval DONE ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=1)                     # route anchor 순위 (1 = 1순위), 0 = R345 무작위 규칙
    ap.add_argument("--route", default="d192")                         # preset_route_<route>.json (v1 = d192 route)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--splits", default="test")                        # 쌍을 뽑을 split 순서 (예 test,train)
    ap.add_argument("--desc_n", type=int, default=0)                   # 앞 n 쌍만 SAM3 설명 문장 단독도
    ap.add_argument("--reuse_gen", action="append", default=[])        # 같은 규칙 gen json 이 있는 폴더 (복사)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--cands_n", type=int, default=32)                 # cands stage: 앞 n 쌍
    ap.add_argument("--stage", required=True)                          # pairs | gen | sam3 | cands | eval (쉼표로 여러 개)
    A = ap.parse_args()
    for st in A.stage.split(","):
        if st == "pairs":
            P = pairs(); print(len(P), "stat", sum(a.startswith("stat") for _, a in P))
        else:
            {"gen": stage_gen, "sam3": stage_sam3, "cands": stage_cands, "eval": stage_eval}[st]()
