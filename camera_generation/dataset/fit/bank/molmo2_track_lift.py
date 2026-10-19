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
  check (CPU, env vista4d) [R343] 품질 통계: Molmo2 원 2D 점 (main id) 이 그 anchor 의 SAM3 mask (seg_instances[_static], scene graph
        node.track_id + merged_from — export_target_track.node_bbox_2d 와 같은 매칭) 안에 있나 (정확히 · 반경 --near_frac·W 안),
        그리고 lift depth (점 주변 창 중앙값) 가 mask 안 depth 중앙값과 얼마나 다른가 → --check_out CSV (쌍마다 1행) + 요약 출력.
  [R346] Molmo2 점 → SAM3 → mask 3D 중심 (R345 검증: 점 lift 보다 GT OBB 중심에 가깝고 같은 정의)
  point    (GPU, env latentcam) --point_mode point1|pointK|pointK8 에 필요한 "Point to the {desc}." → <--point_out>/<video>__<anchor>.json
             (trackf 는 gen 의 tracking json 을 그대로 쓴다 — 이 stage 불필요)
  sam3     (GPU, env vista4d, PYTHONPATH=Vista4D) 영상마다 SAM3 세션 1개, anchor 마다 reset → 고른 점 (frame, x, y) point prompt (obj 1) →
           양방향 전파 → <--sam3_out>/<video>__<anchor>.npz (packbits mask, 고른 점)
  sam3lift (CPU, env vista4d) mask 안 depth 점 → world → 축별 10~90% 구간 중점 = 프레임 중심, 빈 프레임 보간 (덮인 프레임 < --min_frames 면 invalid)
           → <scene>/da3/<--sam3_name> (target_track.npz 와 같은 키·형식)
  comb     (GPU, env vista4d) [R366] text + point 결합 (R359 검증: 100 쌍 쌍 성공 0.86): 라벨 명사 (scene graph label) text 로 SAM3 전파 → 후보 masklet 전부,
           --point_mode 규칙 (기본 pointK8f, --point_out json) 의 점 (f*, x, y) 이 든 masklet 선택 (여럿이면 평균 점수 최고). 없으면 점 단독 결과 (--sam3_out) 로 대체.
           → <corpus>/molmo2_comb[_vote]/<pair>.npz (how 기록). 고르기 규칙 --comb_rule: vote (기본, R384 — 점이 1 개인 프레임들의 다수결)
           · earliest (R366, 가장 이른 단일 점이 든 masklet → molmo2_comb). 같은 전파로 sam3_text_label/<pair>.npz (최고 점수 obj) 도 비어 있으면 채운다 (R358 중복 제거).
           3D 는 sam3lift --sam3_out <corpus>/molmo2_comb --sam3_name target_track_comb.npz
  sam3text (GPU, env vista4d) [R358] SAM3 text 단독 비교군: 라벨 명사 (scene graph label) · 설명 문장 (instance_desc, Molmo2 질문과 같은 출처)
           0 프레임 text prompt → 전파 (매 프레임 detector), obj 여럿이면 평균 점수 × 등장 프레임 수 최고 하나 → <corpus>/sam3_text_{label,desc}/<video>__<anchor>.npz
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


VIDEOS = None                     # [R386] --videos_file 로 영상 제한 (None = 코퍼스 전부)
PAIRS = None                      # [R445] --pairs_file 로 (video, anchor) 쌍 제한 (None = 영상의 anchor 전부)
DESC_TEMPLATE = "the {q}"         # [R434] --desc_template (부위 지정 질문). 기본 = 예전 문장


def pairs(corpus):
    """(video, anchor) 전부 — target_track.npz 의 anchor_id 기준 (seg 여러 개가 같은 anchor 를 공유)."""
    out = []
    for v in sorted(os.listdir(f"{corpus}/dynpose")):
        if VIDEOS is not None and v not in VIDEOS:
            continue
        p = f"{corpus}/dynpose/{v}/da3/target_track.npz"
        if not osp.isfile(p):
            continue
        z = np.load(p, allow_pickle=True)
        for a in sorted({str(x) for x in z["anchor_id"]}):
            if PAIRS is None or (v, a) in PAIRS:
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
    return f"Track {DESC_TEMPLATE.format(q=q)}."                          # [R434] 기본 "the {q}" = 예전 "Track the {q}." 와 같은 문장


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
    print(f"[gen] shard {a.shard_id}/{a.num_shards}: {len(P)} 쌍 남음 (DataLoader workers {a.prep_workers})", flush=True)
    # [R326b] 프레임 읽기 + processor (CPU) 를 DataLoader worker 에서 미리 — GPU generate 와 겹친다 (사용자 10-12 "전처리를 먼저 병렬로").
    batches = [P[b0:b0 + a.batch] for b0 in range(0, len(P), a.batch)]

    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(batches)

        def __getitem__(self, j):
            B = batches[j]; vids, texts, metas = [], [], []
            for v, aid in B:
                fr = np.stack([np.asarray(Image.open(f"{a.corpus}/dynpose/{v}/images_4/{i:05d}.png").convert("RGB")) for i in range(49)])
                msgs = [{"role": "user", "content": [{"type": "text", "text": phrase(v, aid)}, {"type": "video"}]}]
                texts.append(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)); vids.append(fr)
                metas.append({"fps": FPS, "total_num_frames": 49, "duration": 49 / FPS, "frames_indices": list(range(49)), "width": fr.shape[2], "height": fr.shape[1]})
            return B, dict(proc(text=texts, videos=vids, return_tensors="pt", do_sample_frames=False, video_metadata=metas, padding=True))

    dl = torch.utils.data.DataLoader(_DS(), batch_size=None, shuffle=False, num_workers=a.prep_workers,
                                     prefetch_factor=(2 if a.prep_workers > 0 else None), persistent_workers=False)
    t_all = time.time(); done = 0
    for bi, (B, inp) in enumerate(dl):
        inp = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inp.items()}
        t0 = time.time()
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=a.max_tokens, do_sample=False)
        dt = (time.time() - t0) / len(B)
        for (v, aid), row in zip(B, g):
            s = proc.tokenizer.decode(row[inp["input_ids"].shape[1]:], skip_special_tokens=True)
            json.dump(dict(video=v, anchor=aid, prompt=phrase(v, aid), pts=parse(s), text=s[:300], sec=round(dt, 2)), open(f"{a.out}/{v}__{aid}.json", "w"))
        done += len(B)
        if bi % 10 == 0:
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


K_FRAMES = {"pointK": list(range(0, 49, 8)), "pointK8": [int(round(x)) for x in np.linspace(0, 48, 8)]}


def stage_point(a):
    """[R346] Molmo2 pointing (영상 1회 = point1, 또는 K 프레임 이미지 각각 = pointK·pointK8)."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    src = a.point_mode[:-1] if a.point_mode.endswith("f") else a.point_mode
    assert src in ("point1", "pointK", "pointK8"), a.point_mode
    P = [p for i, p in enumerate(pairs(a.corpus)) if i % a.num_shards == a.shard_id and not osp.isfile(f"{a.point_out}/{p[0]}__{p[1]}.json")]
    os.makedirs(a.point_out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(M, trust_remote_code=True, padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    print(f"[point] {src} shard {a.shard_id}/{a.num_shards}: {len(P)} 쌍", flush=True); t_all = time.time()
    for n, (v, aid) in enumerate(P):
        need = range(49) if src == "point1" else K_FRAMES[src]       # K 모드는 필요한 프레임만 읽는다 (49장 읽기가 병목)
        fr = {i: np.asarray(Image.open(f"{a.corpus}/dynpose/{v}/images_4/{i:05d}.png").convert("RGB")) for i in need}
        if src == "point1":
            fr = np.stack([fr[i] for i in range(49)])
        q = "Point to " + phrase(v, aid)[len("Track "):]; pts = []
        if src == "point1":
            msgs = [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "video"}]}]
            meta = {"fps": FPS, "total_num_frames": 49, "duration": 49 / FPS, "frames_indices": list(range(49)), "width": fr.shape[2], "height": fr.shape[1]}
            inp = proc(text=[proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)], videos=[fr], return_tensors="pt",
                       do_sample_frames=False, video_metadata=[meta])
            jobs = [(None, inp)]
        else:
            msgs = [{"role": "user", "content": [{"type": "text", "text": q}, {"type": "image"}]}]
            txt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            Ks = K_FRAMES[src]       # K 프레임을 한 batch 로 (decode 가 짧아 batch 이득이 큼)
            jobs = [(Ks, proc(text=[txt] * len(Ks), images=[Image.fromarray(fr[f]) for f in Ks], return_tensors="pt", padding=True))]
        for f, inp in jobs:
            inp = {k: (x.to("cuda") if hasattr(x, "to") else x) for k, x in dict(inp).items()}
            with torch.inference_mode():
                g = model.generate(**inp, max_new_tokens=256 if f is None else 64, do_sample=False)
            outs = [proc.tokenizer.decode(row[inp["input_ids"].shape[1]:], skip_special_tokens=True) for row in g]
            if f is None:
                pts += parse(outs[0])
            else:
                for fk, s_ in zip(f, outs):
                    for c in COORD.finditer(s_):
                        pts += [[fk, m.group(1), float(m.group(2)) / 1000, float(m.group(3)) / 1000] for m in PTS.finditer(c.group(1))]
        json.dump(dict(video=v, anchor=aid, prompt=q, mode=src, pts=pts), open(f"{a.point_out}/{v}__{aid}.json", "w"))
        if n % 100 == 0:
            el = time.time() - t_all; print(f"[point] {n + 1}/{len(P)} {el / (n + 1):.1f} s/쌍 ETA {(len(P) - n - 1) * el / (n + 1) / 3600:.1f} h", flush=True)
    print(f"=== point shard {a.shard_id} DONE ===", flush=True)


def choose_point(a, v, aid):
    """[R346] --point_mode 규칙으로 (frame, x, y) 하나. *f = confidence 안 쓰고 가장 이른 점 (R345b: log-prob 최고 고르기가 오히려 나쁨)."""
    if a.point_mode.startswith("track"):
        jp = f"{a.out}/{v}__{aid}.json"
        P = json.load(open(jp))["pts"] if osp.isfile(jp) else []
        cnt = {}
        for q in P:
            cnt[q[1]] = cnt.get(q[1], 0) + 1
        if not cnt:
            return None
        main = max(cnt, key=cnt.get); P = [q for q in P if q[1] == main]
    else:
        jp = f"{a.point_out}/{v}__{aid}.json"
        P = json.load(open(jp))["pts"] if osp.isfile(jp) else []
        if not P:
            return None
        nf = {}
        for q in P:
            nf[q[0]] = nf.get(q[0], 0) + 1
        P = [q for q in P if nf[q[0]] == 1] or P            # 같은 프레임에 점 여러 개 = 모호 → 뺀다
    f, _, x, y = min(P, key=lambda q: q[0])
    return int(min(48, max(0, f))), float(x), float(y)


def stage_sam3(a):
    """[R346] 영상마다 SAM3 세션 1개 (프레임 1번 적재), anchor 마다 reset_session 후 point prompt → 양방향 전파."""
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video
    from PIL import Image
    by_v = {}
    for v, aid in pairs(a.corpus):
        by_v.setdefault(v, []).append(aid)
    vids = [v for i, v in enumerate(sorted(by_v)) if i % a.num_shards == a.shard_id]
    os.makedirs(a.sam3_out, exist_ok=True); pred = init_sam3_video(); t_all = time.time(); n_done = 0
    print(f"[sam3] {a.point_mode} shard {a.shard_id}/{a.num_shards}: 영상 {len(vids)}", flush=True)
    for n, v in enumerate(vids):
        todo = [aid for aid in by_v[v] if not osp.isfile(f"{a.sam3_out}/{v}__{aid}.npz")]
        if not todo:
            continue
        frames, _ = load_video(f"{E}/{v}/video.mp4"); T_, H, W, _ = frames.shape
        sid = pred.handle_request({"type": "start_session", "resource_path": [Image.fromarray(frames[i]) for i in range(T_)]})["session_id"]
        for aid in todo:
            out = f"{a.sam3_out}/{v}__{aid}.npz"; ch = choose_point(a, v, aid)
            if ch is None:
                np.savez(out, ok=False); continue
            pred.handle_request({"type": "reset_session", "session_id": sid})
            st = pred._get_session(sid)["state"]
            for f in range(T_):        # point 단독 추적: 빈 cache 로 채움 (R345 FIX — 원래 text 전파 뒤 보정 경로)
                st["cached_frame_outputs"].setdefault(f, {})
            f0, x, y = ch
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": f0, "points": [[x, y]], "point_labels": [1], "obj_id": 1})
            masks = np.zeros((T_, H, W), bool)
            for r in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid, "propagation_direction": "both",
                                                 "start_frame_index": f0, "max_frame_num_to_track": T_}):
                o = r["outputs"]
                if o is None:
                    continue
                for k, oid in enumerate(o["out_obj_ids"]):
                    if int(oid) == 1:
                        masks[r["frame_index"]] |= np.asarray(o["out_binary_masks"][k]).astype(bool)
            np.savez_compressed(out, ok=True, masks=np.packbits(masks, axis=-1), W=W, frame=f0, x=x, y=y)
            n_done += 1
        pred.handle_request({"type": "close_session", "session_id": sid})
        if n % 50 == 0:
            el = time.time() - t_all; print(f"[sam3] 영상 {n + 1}/{len(vids)} · 쌍 {n_done} · {el / (n + 1):.1f} s/영상 ETA {(len(vids) - n - 1) * el / (n + 1) / 3600:.1f} h", flush=True)
    print(f"=== sam3 shard {a.shard_id} DONE ===", flush=True)


def stage_sam3text(a):
    """[R358] SAM3 text 단독 (라벨 · 설명). 영상마다 세션 1개, (anchor, mode) 마다 reset → text prompt → 정방향 전파."""
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video
    from PIL import Image
    by_v = {}
    for v, aid in pairs(a.corpus):
        by_v.setdefault(v, []).append(aid)
    vids = [v for i, v in enumerate(sorted(by_v)) if i % a.num_shards == a.shard_id]
    outd = {m: f"{a.corpus}/sam3_text_{m}" for m in ("label", "desc")}
    for d in outd.values():
        os.makedirs(d, exist_ok=True)
    pred = init_sam3_video(); t_all = time.time(); n_done = 0
    print(f"[sam3text] shard {a.shard_id}/{a.num_shards}: 영상 {len(vids)}", flush=True)
    for n, v in enumerate(vids):
        todo = [(aid, m) for aid in by_v[v] for m in ("label", "desc") if not osp.isfile(f"{outd[m]}/{v}__{aid}.npz")]
        if not todo:
            continue
        try:
            nodes = {x["id"]: x for x in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"]}
        except Exception:      # noqa: BLE001
            nodes = {}
        frames, _ = load_video(f"{E}/{v}/video.mp4"); T_, H, W, _ = frames.shape
        sid = pred.handle_request({"type": "start_session", "resource_path": [Image.fromarray(frames[i]) for i in range(T_)]})["session_id"]
        for aid, m in todo:
            out = f"{outd[m]}/{v}__{aid}.npz"
            text = (nodes.get(aid, {}).get("label") or "") if m == "label" else phrase(v, aid)[len("Track "):].rstrip(".")
            if not text:
                np.savez(out, ok=False, why="no_text"); continue
            pred.handle_request({"type": "reset_session", "session_id": sid})
            pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": text})
            per = {}
            for r in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid, "propagation_direction": "forward",
                                                 "start_frame_index": 0, "max_frame_num_to_track": T_}):
                o = r["outputs"]
                if o is None:
                    continue
                for k, oid in enumerate(o["out_obj_ids"]):
                    dd = per.setdefault(int(oid), {"p": [], "m": {}})
                    dd["p"].append(float(o["out_probs"][k])); dd["m"][r["frame_index"]] = np.asarray(o["out_binary_masks"][k]).astype(bool)
            if not per:
                np.savez(out, ok=False, why="no_detection", text=text); continue
            best = max(per, key=lambda i: np.mean(per[i]["p"]) * len(per[i]["m"]))
            masks = np.zeros((T_, H, W), bool)
            for f, mk in per[best]["m"].items():
                masks[f] = mk
            np.savez_compressed(out, ok=True, masks=np.packbits(masks, axis=-1), W=W, conf=float(np.mean(per[best]["p"])), n_obj=len(per), text=text)
            n_done += 1
        pred.handle_request({"type": "close_session", "session_id": sid})
        if n % 50 == 0:
            el = time.time() - t_all; print(f"[sam3text] 영상 {n + 1}/{len(vids)} · {n_done} · {el / (n + 1):.1f} s/영상 ETA {(len(vids) - n - 1) * el / (n + 1) / 3600:.1f} h", flush=True)
    print(f"=== sam3text shard {a.shard_id} DONE ===", flush=True)


def single_points(a, v, aid):
    """[R384] --point_out json 에서 점이 1 개뿐인 프레임의 점들 [(frame, x, y)] (같은 프레임 여러 점 = 모호 → 뺀다, 없으면 전부)."""
    jp = f"{a.point_out}/{v}__{aid}.json"
    P = json.load(open(jp))["pts"] if osp.isfile(jp) else []
    nf = {}
    for q in P:
        nf[q[0]] = nf.get(q[0], 0) + 1
    return [(int(min(48, max(0, q[0]))), float(q[2]), float(q[3])) for q in ([q for q in P if nf[q[0]] == 1] or P)]


def stage_comb(a):
    """[R366] text (라벨) 후보 masklet + Molmo2 점 선택. 영상마다 세션 1개, anchor 마다 reset."""
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_video
    from utils.recon_and_seg.seg_sam3_official import init_sam3_video
    from PIL import Image
    import shutil
    by_v = {}
    for v, aid in pairs(a.corpus):
        by_v.setdefault(v, []).append(aid)
    vids = [v for i, v in enumerate(sorted(by_v)) if i % a.num_shards == a.shard_id]
    # [R384] 출력 폴더는 규칙별 (예전 earliest = molmo2_comb 그대로 두고, vote = molmo2_comb_vote)
    od, ld = a.comb_out or f"{a.corpus}/molmo2_comb" + ("" if a.comb_rule == "earliest" else f"_{a.comb_rule}"), f"{a.corpus}/sam3_text_label"
    os.makedirs(od, exist_ok=True); os.makedirs(ld, exist_ok=True)
    pred = init_sam3_video(); t_all = time.time(); how = {}
    print(f"[comb] {a.point_mode} shard {a.shard_id}/{a.num_shards}: 영상 {len(vids)}", flush=True)
    for n, v in enumerate(vids):
        todo = [aid for aid in by_v[v] if not osp.isfile(f"{od}/{v}__{aid}.npz")]
        if not todo:
            continue
        try:
            nodes = {x["id"]: x for x in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"]}
        except Exception:      # noqa: BLE001
            nodes = {}
        frames, _ = load_video(f"{E}/{v}/video.mp4"); T_, H, W, _ = frames.shape
        sid = pred.handle_request({"type": "start_session", "resource_path": [Image.fromarray(frames[i]) for i in range(T_)]})["session_id"]
        for aid in todo:
            out = f"{od}/{v}__{aid}.npz"; label = a.label_override or nodes.get(aid, {}).get("label") or ""; ch = choose_point(a, v, aid); per = {}
            if label:
                pred.handle_request({"type": "reset_session", "session_id": sid})
                pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": label})
                for r in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid, "propagation_direction": "forward",
                                                     "start_frame_index": 0, "max_frame_num_to_track": T_}):
                    o = r["outputs"]
                    if o is None:
                        continue
                    for k, oid in enumerate(o["out_obj_ids"]):
                        dd = per.setdefault(int(oid), {"p": [], "m": {}})
                        dd["p"].append(float(o["out_probs"][k])); dd["m"][r["frame_index"]] = np.asarray(o["out_binary_masks"][k]).astype(bool)

            def masks_of(i):
                mm = np.zeros((T_, H, W), bool)
                for f, mk in per[i]["m"].items():
                    mm[f] = mk
                return mm
            lp = f"{ld}/{v}__{aid}.npz"                      # R358 라벨 판도 같은 규칙으로 채움 (비어 있을 때만)
            if label and not osp.isfile(lp) and not a.label_override:          # [R434] 명사를 바꾼 실행은 코퍼스 sam3_text_label 을 채우지 않는다
                if per:
                    best = max(per, key=lambda i: np.mean(per[i]["p"]) * len(per[i]["m"]))
                    np.savez_compressed(lp, ok=True, masks=np.packbits(masks_of(best), axis=-1), W=W, conf=float(np.mean(per[best]["p"])), n_obj=len(per), text=label)
                else:
                    np.savez(lp, ok=False, why="no_detection", text=label)
            pick, why = None, ("no_label" if not label else "no_point")
            if ch is not None and per and a.comb_rule == "vote":
                # [R384] 기본 = vote (R376 1순위 1000 쌍 성공 0.954 vs earliest 0.914): 점이 1 개인 프레임의 점마다 그 점이 든 masklet 에 한 표,
                #   최다 표 (동률은 그 masklet 을 가리킨 가장 이른 프레임). ch 는 대체 경로 (점 단독 SAM3) 기록용으로만 쓴다.
                vote, first = {}, {}
                for f1, x1, y1 in single_points(a, v, aid):
                    u1, v1 = min(W - 1, int(x1 * W)), min(H - 1, int(y1 * H))
                    for i in per:
                        if f1 in per[i]["m"] and per[i]["m"][f1][v1, u1]:
                            vote[i] = vote.get(i, 0) + 1; first[i] = min(first.get(i, 99), f1)
                if vote:
                    pick, why = max(vote, key=lambda i: (vote[i], -first[i])), "text_masklet_vote"
                else:
                    why = "point_outside"
            elif ch is not None and per:
                f0, x, y = ch; u, vv = min(W - 1, int(x * W)), min(H - 1, int(y * H))
                hit = [i for i in per if f0 in per[i]["m"] and per[i]["m"][f0][vv, u]]
                if hit:
                    pick, why = max(hit, key=lambda i: np.mean(per[i]["p"])), "text_masklet"
                else:
                    why = "point_outside"
            elif ch is not None:
                why = "no_candidate"
            if a.save_cands and per:                     # [R386] noun 후보 masklet 전부 (규칙을 바꿔도 SAM3 를 다시 안 돌리게)
                os.makedirs(f"{od}_cands", exist_ok=True)
                ids = sorted(per, key=lambda i: -np.mean(per[i]["p"]) * len(per[i]["m"]))
                np.savez_compressed(f"{od}_cands/{v}__{aid}.npz", masks=np.packbits(np.stack([masks_of(i) for i in ids]), axis=-1), W=W,
                                    ids=np.array(ids, int), score=np.array([np.mean(per[i]["p"]) for i in ids]),
                                    nfr=np.array([len(per[i]["m"]) for i in ids]), text=label)
            if pick is not None:
                np.savez_compressed(out, ok=True, masks=np.packbits(masks_of(pick), axis=-1), W=W, frame=ch[0], x=ch[1], y=ch[2], how=why, n_obj=len(per))
            else:
                src = f"{a.sam3_out}/{v}__{aid}.npz"
                if osp.isfile(src):
                    z = dict(np.load(src)); z["how"] = "fallback_point:" + why; np.savez_compressed(out, **z)
                else:
                    np.savez(out, ok=False, how="none:" + why)
                why = "fallback:" + why
            how[why] = how.get(why, 0) + 1
        pred.handle_request({"type": "close_session", "session_id": sid})
        if n % 50 == 0:
            el = time.time() - t_all; print(f"[comb] 영상 {n + 1}/{len(vids)} · {how} · {el / (n + 1):.1f} s/영상 ETA {(len(vids) - n - 1) * el / (n + 1) / 3600:.1f} h", flush=True)
    print(f"[comb] how {how}\n=== comb shard {a.shard_id} DONE ===", flush=True)


def sam3lift_video(job):
    a, v = job
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_cameras, load_depths, load_masks
    dst = f"{a.corpus}/dynpose/{v}/da3/{a.sam3_name}"
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
    F_, H, W = d.shape; per = {}
    for aid in sorted(set(aids)):
        tr = np.zeros((F_, 3), np.float32); ok = False; mp = f"{a.sam3_out}/{v}__{aid}.npz"
        if osp.isfile(mp):
            zz = np.load(mp)
            if bool(zz["ok"]):
                M = np.unpackbits(zz["masks"], axis=-1, count=int(zz["W"])).astype(bool); got = {}
                for f in range(min(F_, len(M))):
                    mf = M[f][(np.arange(H) * M.shape[1] // H)][:, (np.arange(W) * M.shape[2] // W)] & np.isfinite(d[f]) & (d[f] > 0) & ~sky[f]
                    if mf.sum() < 10:
                        continue
                    vv, uu = np.nonzero(mf); zf = d[f][mf]
                    fx, fy, cx, cy = (K[f][0], K[f][1], K[f][2], K[f][3]) if K.ndim == 2 else (K[f][0, 0], K[f][1, 1], K[f][0, 2], K[f][1, 2])
                    Xw = (Cw[f] @ np.stack([(uu + 0.5 - cx) / fx * zf, (vv + 0.5 - cy) / fy * zf, zf, np.ones_like(zf)], 0))[:3].T
                    got[f] = (np.percentile(Xw, 10, 0) + np.percentile(Xw, 90, 0)) / 2
                if len(got) >= a.min_frames:
                    fs = np.array(sorted(got)); X = np.stack([got[f] for f in fs])
                    tr = np.stack([np.interp(np.arange(F_), fs, X[:, c]) for c in range(3)], -1).astype(np.float32); ok = True
        per[aid] = (tr, ok)
    np.savez(dst, track_world=np.stack([per[x][0] for x in aids]), valid=np.array([per[x][1] for x in aids]),
             keys=np.array(keys), anchor_id=np.array(aids))
    return v, f"ok {sum(per[x][1] for x in set(aids))}/{len(set(aids))}"


def sam3box_video(job):
    """[R391] --sam3_out 의 mask + depth → 프레임별 중력 정렬 OBB (scene_graph.obb.fit_obb, G 좌표) → subject_box.npz 와 같은 형식의
    반축 벡터 (S,T,3,3) world (열 = l·w·h 축 × extent/2). yaw 는 180° 대칭이라 2·yaw 로 unwrap 후 5 프레임 median, 빈 프레임은 가장 가까운 관측,
    extent 는 프레임별 median (GT box 와 같은 규약: yaw 만 프레임별, 크기는 상수). 점은 축별 2~98% 안쪽만 (mask 경계·depth 튐 줄이기)."""
    a, v = job
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/camera_generation/dataset")
    from utils.media import load_cameras, load_depths, load_masks
    from scene_graph.obb import fit_obb
    from scipy.ndimage import median_filter
    dst = f"{a.corpus}/dynpose/{v}/da3/{a.box_name}"
    if osp.isfile(dst) and not a.overwrite:
        return v, "skip"
    z = np.load(f"{a.corpus}/dynpose/{v}/da3/target_track.npz", allow_pickle=True)
    keys, aids = [str(k) for k in z["keys"]], [str(x) for x in z["anchor_id"]]
    try:
        g = json.load(open(f"{G}/{v}/scene_graph.json")); T_wg = np.asarray(g["frames"]["T_wg"], float); T_gw = np.asarray(g["frames"]["T_gw"], float)
        d = load_depths(f"{E}/{v}/depths", dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(f"{E}/{v}/sky_mask", desc=None)).astype(bool)
        Cw, K = load_cameras(f"{E}/{v}/cameras.npz"); Cw, K = np.asarray(Cw, float), np.asarray(K, float)
    except Exception as e:      # noqa: BLE001
        return v, f"error {str(e)[:60]}"
    F_, H, W = d.shape; per = {}; rng = np.random.default_rng(0)
    for aid in sorted(set(aids)):
        mp = f"{a.sam3_out}/{v}__{aid}.npz"; per[aid] = None
        if not osp.isfile(mp):
            continue
        zz = np.load(mp)
        if not bool(zz["ok"]):
            continue
        M = np.unpackbits(zz["masks"], axis=-1, count=int(zz["W"])).astype(bool); yaw, ext = {}, []
        for f in range(min(F_, len(M))):
            mf = M[f][(np.arange(H) * M.shape[1] // H)][:, (np.arange(W) * M.shape[2] // W)] & np.isfinite(d[f]) & (d[f] > 0) & ~sky[f]
            if mf.sum() < 30:
                continue
            vv, uu = np.nonzero(mf)
            if len(vv) > 4000:
                k = rng.choice(len(vv), 4000, replace=False); vv, uu = vv[k], uu[k]
            zf = d[f][vv, uu]
            fx, fy, cx, cy = (K[f][0], K[f][1], K[f][2], K[f][3]) if K.ndim == 2 else (K[f][0, 0], K[f][1, 1], K[f][0, 2], K[f][1, 2])
            Xw = (Cw[f] @ np.stack([(uu + 0.5 - cx) / fx * zf, (vv + 0.5 - cy) / fy * zf, zf, np.ones_like(zf)], 0))
            Xg = (T_gw @ Xw)[:3].T
            lo, hi = np.percentile(Xg, 2, 0), np.percentile(Xg, 98, 0); Xg = Xg[((Xg >= lo) & (Xg <= hi)).all(1)]
            if len(Xg) < 10:
                continue
            o = fit_obb(Xg); yaw[f] = o["yaw_rad"]; ext.append(o["extent"])
        if len(yaw) < a.min_frames:
            continue
        fs = np.array(sorted(yaw)); y2 = np.unwrap(2 * np.array([yaw[f] for f in fs])) / 2
        y2 = median_filter(y2, size=min(5, len(y2)), mode="nearest")
        yf = np.array([y2[np.abs(fs - f).argmin()] for f in range(F_)])
        e = np.median(np.stack(ext), 0) / 2.0
        c, s_ = np.cos(yf), np.sin(yf); zr, on = np.zeros_like(c), np.ones_like(c)
        R = np.stack([np.stack([c, -s_, zr], -1), np.stack([s_, c, zr], -1), np.stack([zr, zr, on], -1)], 1)
        per[aid] = np.einsum("ij,tjk->tik", T_wg[:3, :3], R * e[None, None, :]).astype(np.float32)
    axes = np.zeros((len(keys), F_, 3, 3), np.float32); valid = np.zeros(len(keys), bool)
    for i, aid in enumerate(aids):
        if per.get(aid) is not None:
            axes[i] = per[aid]; valid[i] = True
    np.savez(dst, axes_world=axes, valid=valid, keys=np.array(keys), anchor_id=np.array(aids))
    return v, f"ok {sum(per[x] is not None for x in set(aids))}/{len(set(aids))}"


def stage_sam3box(a):
    from multiprocessing import Pool
    vids = sorted({v for v, _ in pairs(a.corpus)}); bad = 0
    with Pool(a.workers) as p:
        for i, (v, msg) in enumerate(p.imap_unordered(sam3box_video, [(a, v) for v in vids], chunksize=4)):
            bad += msg.startswith("error")
            if i % 200 == 0 or msg.startswith("error"):
                print(f"[sam3box {i}/{len(vids)}] {v} {msg}", flush=True)
    print(f"=== sam3box DONE {len(vids)} (error {bad}) ===", flush=True)


def stage_sam3lift(a):
    from multiprocessing import Pool
    vids = sorted({v for v, _ in pairs(a.corpus)}); bad = 0
    with Pool(a.workers) as p:
        for i, (v, msg) in enumerate(p.imap_unordered(sam3lift_video, [(a, v) for v in vids], chunksize=4)):
            bad += msg.startswith("error")
            if i % 500 == 0 or msg.startswith("error"):
                print(f"[sam3lift {i}/{len(vids)}] {v} {msg}", flush=True)
    print(f"=== sam3lift DONE {len(vids)} (error {bad}) ===", flush=True)


def check_video(job):
    a, v = job
    sys.path.insert(0, "/data1/cympyc1785/LatentCamVid/video_generation/models/Vista4D")
    from utils.media import load_depths, load_masks
    try:
        nodes = {n["id"]: n for n in json.load(open(f"{G}/{v}/scene_graph.json"))["nodes"]}
        d = load_depths(f"{E}/{v}/depths", dtype=np.float16, desc=None).astype(np.float32)
        sky = np.asarray(load_masks(f"{E}/{v}/sky_mask", desc=None)).astype(bool)
    except Exception as e:      # noqa: BLE001
        return [dict(video=v, anchor="", status=f"error {str(e)[:60]}")]
    F_, H, W = d.shape; r = max(2, W // 80); rows = []; seg = {}
    aids = sorted({x for vv, x in pairs_cache[v]})
    for aid in aids:
        row = dict(video=v, anchor=aid, status="ok", kind=aid.split("_")[0])
        node = nodes.get(aid); jp = f"{a.out}/{v}__{aid}.json"
        if node is None or not osp.isfile(jp):
            rows.append({**row, "status": "no_node" if node is None else "no_json"}); continue
        sub = "seg_instances" if aid.startswith("dyn") else "seg_instances_static"
        if sub not in seg:
            mp = f"{a.seg_root}/{sub}/{v}"
            seg[sub] = (json.load(open(f"{mp}/meta.json")), np.load(f"{mp}/masks.npz")) if osp.isfile(f"{mp}/meta.json") else None
        if seg[sub] is None:
            rows.append({**row, "status": "no_seg"}); continue
        meta, mz = seg[sub]; mW, mH = meta["width"], meta["height"]
        ids = {node.get("track_id")} | set(node.get("merged_from") or [])
        P = json.load(open(jp))["pts"]; cnt = {}
        for q in P:
            cnt[q[1]] = cnt.get(q[1], 0) + 1
        main = max(cnt, key=cnt.get) if cnt else None; got = {}
        for f, tid, x, y in P:
            if tid == main and 0 <= f < F_ and f not in got:
                got[f] = (x, y)
        n_pt = n_mask = n_both = n_in = n_near = 0; dz = []
        for f in range(min(F_, len(meta["frames"]))):
            k = [i for i, b in enumerate(meta["frames"][f]) if b.get("id") in ids]
            m = None
            if k:
                m = np.unpackbits(mz[f"{f:05d}"][k], axis=-1, count=mW).astype(bool).any(0); n_mask += 1
            if f in got:
                n_pt += 1
            if m is None or f not in got:
                continue
            n_both += 1; x, y = got[f]
            u, vv = min(mW - 1, int(x * mW)), min(mH - 1, int(y * mH))
            rr = max(1, int(a.near_frac * mW))
            n_in += bool(m[vv, u]); n_near += bool(m[max(0, vv - rr):vv + rr + 1, max(0, u - rr):u + rr + 1].any())
            # lift depth (molmo2_track_lift.lift_video 와 같은 창) vs mask 안 depth 중앙값 (depth 해상도로 최근접 축소)
            md = m[(np.arange(H) * mH // H)][:, (np.arange(W) * mW // W)] & np.isfinite(d[f]) & (d[f] > 0) & ~sky[f]
            ud, vd = int(x * W), int(y * H)
            sl = (f, slice(max(0, vd - r), vd + r + 1), slice(max(0, ud - r), ud + r + 1))
            ok = np.isfinite(d[sl]) & (d[sl] > 0) & ~sky[sl]
            if ok.sum() >= 3 and md.sum() >= 10:
                zm = float(np.median(d[f][md])); dz.append(abs(float(np.median(d[sl][ok])) - zm) / max(zm, 1e-6))
        rows.append({**row, "n_pt": n_pt, "n_mask": n_mask, "n_both": n_both, "in_frac": n_in / max(1, n_both),
                     "near_frac": n_near / max(1, n_both), "dz_med": float(np.median(dz)) if dz else float("nan"),
                     "dz_p90": float(np.percentile(dz, 90)) if dz else float("nan")})
    return rows


def stage_check(a):
    import csv
    from multiprocessing import Pool
    global pairs_cache
    pairs_cache = {}
    for v, aid in pairs(a.corpus):
        pairs_cache.setdefault(v, []).append((v, aid))
    vids = sorted(pairs_cache); out = []
    with Pool(a.workers) as p:
        for i, rows in enumerate(p.imap_unordered(check_video, [(a, v) for v in vids], chunksize=4)):
            out += rows
            if i % 1000 == 0:
                print(f"[check {i}/{len(vids)}]", flush=True)
    keys = ["video", "anchor", "kind", "status", "n_pt", "n_mask", "n_both", "in_frac", "near_frac", "dz_med", "dz_p90"]
    with open(a.check_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in out]
    ok = [r for r in out if r["status"] == "ok" and r["n_both"] > 0]
    print(f"=== check DONE 쌍 {len(out)} · 평가 가능 {len(ok)} (status {dict(__import__('collections').Counter(r['status'] for r in out))}) -> {a.check_out}")
    for kind in ("dyn", "stat", None):
        s = [r for r in ok if kind is None or r["kind"] == kind]
        if not s:
            continue
        inf = np.array([r["in_frac"] for r in s]); nf = np.array([r["near_frac"] for r in s]); dz = np.array([r["dz_med"] for r in s], float)
        print(f"[{kind or 'all'}] n {len(s)} · mask 안 비율 평균 {inf.mean():.3f} 중앙 {np.median(inf):.3f} · ≥0.8 쌍 {np.mean(inf >= 0.8):.3f} · <0.2 쌍 {np.mean(inf < 0.2):.3f}"
              f" · 반경 안 평균 {nf.mean():.3f} · depth 상대차 중앙 {np.nanmedian(dz):.3f} (>0.3 쌍 {np.nanmean(dz > 0.3):.3f})")


pairs_cache = {}


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
    ap.add_argument("--stage", required=True, choices=["gen", "lift", "check", "point", "sam3", "sam3lift", "sam3box", "sam3text", "comb"])
    ap.add_argument("--corpus", default=C)
    ap.add_argument("--out", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/molmo2_track")
    ap.add_argument("--name", default="target_track_molmo2.npz")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--prep_workers", type=int, default=3)   # [R326b] 0 = 예전 (생성 루프 안에서 전처리)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--min_frames", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seg_root", default="/data1/cympyc1785/LatentCamVid/DATA/DynPose-100K/eval_data")   # [R343] seg_instances[_static]
    ap.add_argument("--near_frac", type=float, default=0.02)    # [R343] mask 반경 허용 (영상 폭 비율)
    ap.add_argument("--comb_rule", default="vote", choices=["vote", "earliest"])   # [R384] comb 의 masklet 고르기 (기본 vote, earliest = R366)
    ap.add_argument("--save_cands", action="store_true")                           # [R386] comb: 후보 masklet 전부 <comb_out>_cands/
    ap.add_argument("--videos_file", default="")                                   # [R386] 이 목록의 영상만 (예: v1 test)
    ap.add_argument("--comb_out", default="")                                       # [R384] 비우면 <corpus>/molmo2_comb[_<rule>]
    ap.add_argument("--point_mode", default="pointK8f")      # [R346] trackf | point1 | pointKf | pointK8f (R345b 결과로 고름)
    ap.add_argument("--point_out", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/molmo2_point")
    ap.add_argument("--sam3_out", default="/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d200/molmo2_sam3")
    ap.add_argument("--sam3_name", default="target_track_m2sam3.npz")
    ap.add_argument("--box_name", default="subject_box_vote.npz")          # [R391] sam3box 출력 이름
    # [R434] 부위 지정 (예 얼굴): Molmo2 질문 대상구 템플릿 · comb 의 SAM3 명사. 기본값이면 예전과 같은 질문·명사.
    ap.add_argument("--desc_template", default="the {q}")                    # 예 "the face of the {q}" → "Point to the face of the …"
    ap.add_argument("--label_override", default="")                          # 예 "face"
    ap.add_argument("--pairs_file", default="")                              # [R445] 줄마다 "<video>__<anchor>" — 이 쌍만
    ap.add_argument("--check_out", default="/data1/cympyc1785/LatentCamVid/tmp/r343/molmo2_track_check.csv")
    a = ap.parse_args()
    if a.videos_file:                                   # [R386]
        VIDEOS = set(open(a.videos_file).read().split())
    DESC_TEMPLATE = a.desc_template
    if a.pairs_file:                                    # [R445]
        PAIRS = {tuple(x.split("__", 1)) for x in open(a.pairs_file).read().split()}
    {"gen": stage_gen, "lift": stage_lift, "check": stage_check, "point": stage_point, "sam3": stage_sam3, "sam3lift": stage_sam3lift, "sam3box": stage_sam3box, "sam3text": stage_sam3text, "comb": stage_comb}[a.stage](a)
