"""viser frustum viewer for validation-saved cameras.

root_dir (e.g. results/<exp>/test) holds per-sequence:
  {data_name}_transforms_pred.json   generated cameras (RED)
  {data_name}_transforms_ref.json    target/GT cameras  (BLUE)
  {data_name}_caption.json           text prompt

`--dataset` 으로 배경(회색)을 무엇으로 그릴지 고른다. ref/pred 처리는 두 경우가 완전히 같다.

  dl3dv (기본, 기존 동작)
    data_name = "{type}_{hash}_{seg}" -> scene chunk "{type}/{hash}" 의 transforms.json 에서
    나머지 카메라를 GREY frustum 으로 그린다.

  sd (Scene-Decoupled)
    data_name = "{scene}__{TARGET}__{CONTEXT}" (dataset_scene_decoupled.py:128 과 같은 규약).
    SD 는 scene.ply 가 없으므로 **clip 의 da3 depth 를 unproject 한 point cloud** 를 배경으로
    그리고, 그 clip 의 카메라를 frustum 으로 같이 얹는다.
    좌표: pose.npz 의 w2c 는 da3 임의 스케일이라 umeyama_gt.json 의 sim3 로 GT meters 에
    올린다 (`R' = R_e·Rᵀ`, `t' = s·t_e − R'·t`, depth 는 `d' = s·d`) — dataset 과 같은 식이다.
    ref/pred json 은 `first_extrinsic`(= 이 미터 world 의 target 첫 w2c) 로 복원되므로
    point cloud 와 같은 world frame 에 놓인다.

    `--pc-clips` 로 어느 clip 을 그릴지 고른다 (`context` 기본 = 기존 동작 / `target` / `both`).
    두 clip 은 **서로 다른 umeyama sim3** 를 타고 같은 GT meters world 로 올라오므로, `both`
    로 겹쳐 보는 것이 곧 그 정렬(aligned world)이 맞는지에 대한 검증이다. target 을 그릴 때는
    그 clip 의 pose 와 ref(파랑) 카메라 위치차도 같이 찍는다 — 두 값이 서로 다른 경로로
    나오므로 0 이 아니면 world 가 어긋난 것이다. `--pc-tint` 는 RGB 대신 단색으로 칠해
    두 point cloud 를 구분하고, `--pc-target-cams` 는 target clip 카메라도 그린다.

Coordinate handling: the saved transform_matrix is OpenGL c2w (nerfstudio; y-up, cam looks
-Z), same convention as DL3DV transforms.json. viser camera frustums use OpenCV (cam looks
+Z, y-down), so we convert c2w_opencv = c2w_opengl @ diag(1,-1,-1,1) and pass its rotation
(quaternion) + translation. GL<->CV flip 은 카메라 축만 뒤집고 world 좌표는 건드리지 않으므로
point cloud 는 변환 없이 그대로 겹친다.

GUI: slider to pick sequence + Load button + Next button.

sequence 를 load 할 때마다 터미널에 지표를 찍는다: ref/pred 로 그 자리에서 계산하는 궤적 오차
(pos_rmse / rot, `scripts/render/compare_textonly.py` 와 같은 정의) + `<root>/../preds_scores.csv`
의 그 sequence 행. CSV 의 PRDC/FCD 열은 set 단위 값이 모든 행에 복사된 것이라 per-sequence 에서
빼고 시작할 때 corpus-level 로 한 번만 찍는다 (`load_seq_metrics` 참고).

Run:
  python scripts/viewer/viser_val_cameras.py --root results/<exp>/test [--port 8080]
  python scripts/viewer/viser_val_cameras.py --dataset sd \
      --root results/compare/sd_whuman_textonly_n160/test --port 8081
  python scripts/viewer/viser_val_cameras.py --dataset sd --pc-clips both --pc-tint \
      --root results/compare/sd_whuman_textonly_n160/test --port 8081
"""

import os as _os, sys as _sys, glob as _glob  # scripts/: sibling-module resolution across subfolders
_SCR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_SCR, *sorted(p for p in _glob.glob(_os.path.join(_SCR, '*')) if _os.path.isdir(p))]:
    if _d not in _sys.path: _sys.path.append(_d)

import os, os.path as osp, json, glob, argparse
import numpy as np
import viser
import viser.transforms as vtf

DL3DV_ROOT = '/data1/cympyc1785/data/DL3DV/scenes'
SD_ROOT = '/data1/cympyc1785/data/Scene-Decoupled-Video-dataset'
_GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])


def load_transforms(path):
    """-> (c2w_opengl (N,4,4), fx, fy, w, h)."""
    d = json.load(open(path))
    fr = d['frames']
    c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64)
    return c2w, float(d['fl_x']), float(d['fl_y']), float(d['w']), float(d['h'])


def scene_chunk(data_name):
    p = data_name.split('_')
    return f"{p[0]}/{p[1]}"                      # type/hash (hash = single hex token)


# ---------------------------------------------------------------- Scene-Decoupled 배경
def sd_clip_meters(clip_dir):
    """SD clip 의 pose 를 sim3 로 GT meters 에 올려 돌려준다.

    dataset_scene_decoupled.py::_load_scene 과 **같은 식**이다 (달라지면 point cloud 가
    ref/pred 와 어긋나므로 여기서 재구현하는 대신 식을 그대로 옮겨 적었다):
      pose.npz extrinsics = OpenCV w2c, da3 임의 스케일
      umeyama X_gt = s·R·X + t  ->  R' = R_e·Rᵀ , t' = s·t_e − R'·t , depth d' = s·d
    -> (w2c (T,4,4) float64, K (T,3,3), s)
    """
    z = np.load(osp.join(clip_dir, 'pose.npz'))
    E = np.asarray(z['extrinsics'], dtype=np.float64)               # (T,3,4)
    K = np.asarray(z['intrinsics'], dtype=np.float64)               # (T,3,3)
    u = json.load(open(osp.join(clip_dir, 'umeyama_gt.json')))
    if not u.get('moving', False):
        raise ValueError(f"{clip_dir}: static clip 이라 sim3(s,t) 가 없다")
    s = float(u['s'])
    R = np.asarray(u['R'], dtype=np.float64)
    t = np.asarray(u['t'], dtype=np.float64)
    Rp = E[:, :3, :3] @ R.T
    tp = s * E[:, :3, 3] - np.einsum('tij,j->ti', Rp, t)
    w2c = np.tile(np.eye(4), (len(E), 1, 1))
    w2c[:, :3, :3], w2c[:, :3, 3] = Rp, tp
    return w2c, K, s


def sd_pointcloud(clip_dir, mp4, n_frames=8, stride=3, conf_pct=40.0, max_points=400_000,
                  seed=0, t_range=None):
    """clip 의 da3 depth 를 GT meters world 로 unproject. -> (P (N,3), C (N,3) uint8)

    conf_pct: 선택한 프레임 전체에서 잰 conf 백분위수 아래를 버린다 (avg_scale 생성기와 같은
              기준을 쓰되 여기서는 '보기 좋게' 가 목적이라 값만 옵션으로 뺐다).
    t_range:  (t0, t1) 를 주면 그 반열린 구간에서만 프레임을 고른다. None 이면 clip 전체.
              target clip 을 그릴 때 segment 범위로 자르는 데 쓴다 — clip 은 81 프레임인데
              target segment(ref)는 앞 49 프레임이라, 안 자르면 뒤 32 프레임이 궤적 옆을
              나란히 지나가 **두 번째 궤적처럼 보인다**.
    """
    import cv2
    w2c, K, s = sd_clip_meters(clip_dir)
    dep = np.load(osp.join(clip_dir, 'depth.npz'))['depth']         # (T,h,w) float16, da3 스케일
    cf = np.load(osp.join(clip_dir, 'conf.npz'))['conf']
    T, h, w = dep.shape
    t0, t1 = (0, T) if t_range is None else (max(0, int(t_range[0])), min(T, int(t_range[1])))
    idxs = np.unique(np.linspace(t0, t1 - 1, min(n_frames, t1 - t0)).round().astype(int))

    # 색: mp4 프레임을 depth 격자로 리사이즈. 둘 다 full-frame 이라 정규화 좌표가 보존된다.
    want, rgb = set(int(i) for i in idxs), {}
    cap = cv2.VideoCapture(mp4)
    k = 0
    while len(rgb) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if k in want:
            rgb[k] = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]
        k += 1
    cap.release()

    d_sel = np.asarray(dep[idxs], dtype=np.float32)[:, ::stride, ::stride] * s   # meters
    c_sel = np.asarray(cf[idxs], dtype=np.float32)[:, ::stride, ::stride]
    thr = float(np.percentile(c_sel, conf_pct))
    vs, us = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing='ij')

    P, C = [], []
    for j, i in enumerate(idxs):
        m = (c_sel[j] >= thr) & np.isfinite(d_sel[j]) & (d_sel[j] > 0)
        if not m.any():
            continue
        Ki = np.linalg.inv(K[i])
        pix = np.stack([us[m] + 0.5, vs[m] + 0.5, np.ones(int(m.sum()))], 0)     # (3,M)
        cam = (Ki @ pix) * d_sel[j][m][None]                                     # (3,M)
        Rw, tw = w2c[i, :3, :3], w2c[i, :3, 3]
        P.append((Rw.T @ (cam - tw[:, None])).T)                                 # world (M,3)
        img = rgb.get(int(i))
        C.append(img[::stride, ::stride][m] if img is not None
                 else np.full((int(m.sum()), 3), 180, np.uint8))
    if not P:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    P = np.concatenate(P).astype(np.float32)
    C = np.concatenate(C).astype(np.uint8)
    if len(P) > max_points:
        sel = np.random.default_rng(seed).choice(len(P), max_points, replace=False)
        P, C = P[sel], C[sel]
    return P, C


# ---------------------------------------------------------------- metric 출력
def traj_metrics(ref_c2w, pred_c2w):
    """ref/pred c2w 만으로 계산되는 sequence 단위 궤적 오차.

    정의는 `scripts/render/compare_textonly.py` 와 **같게** 맞췄다 (pos_rmse = 프레임별 위치
    거리의 RMS, rot = R_pred·R_refᵀ 의 각도). 단위는 world 단위 그대로 — SD 는 미터,
    DL3DV 는 nerfstudio 정규화 단위라 두 코퍼스 사이에서 비교하면 안 된다.
    """
    n = min(len(ref_c2w), len(pred_c2w))
    ref, prd = ref_c2w[:n], pred_c2w[:n]
    d = np.linalg.norm(prd[:, :3, 3] - ref[:, :3, 3], axis=1)
    R = np.einsum('tij,tkj->tik', prd[:, :3, :3], ref[:, :3, :3])
    rot = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    plen = lambda c: float(np.linalg.norm(np.diff(c[:, :3, 3], axis=0), axis=1).sum())
    return {'pos_rmse': float(np.sqrt((d ** 2).mean())), 'pos_mean': float(d.mean()),
            'pos_max': float(d.max()), 'pos_end': float(d[-1]),
            'rot_mean': float(rot.mean()), 'rot_max': float(rot.max()),
            'path_ref': plen(ref), 'path_pred': plen(prd)}


def _tid_from_filename(s):
    b = osp.basename(s.strip().rstrip('.'))
    for suf in ('_transforms_ref', '_transforms_pred'):
        if b.endswith(suf):
            return b[:-len(suf)]
    return b


def load_seq_metrics(root):
    """`<root>/../preds_scores.csv` 에서 평가 지표를 읽는다. -> (per_seq, corpus, path)

    주의: 이 CSV 의 PRDC/FCD 열은 **코퍼스 전체에서 한 번** 계산된 값이 모든 행에 그대로
    복사돼 있다 (metrics.json 과 같은 값). sequence 별 수치로 오독하지 않도록, 전 행이
    같은 값인 열은 corpus-level 로 빼서 시작할 때 한 번만 찍고 per-sequence 에서 제외한다.
    (행마다 다른 열만 per-sequence 로 취급 — 우연히 전 행이 같은 per-seq 열이 있으면 그것도
    corpus 쪽으로 가지만, 그때는 어차피 sequence 를 구분하는 정보가 아니다.)
    """
    import csv
    p = osp.join(osp.dirname(osp.abspath(root)), 'preds_scores.csv')
    if not osp.isfile(p):
        return {}, {}, None
    rows = list(csv.DictReader(open(p)))
    if not rows or 'filename' not in rows[0]:
        return {}, {}, None
    cols = [c for c in rows[0] if c and c != 'filename']
    const = [c for c in cols if len({r[c] for r in rows}) == 1]
    var = [c for c in cols if c not in const]
    corpus = {c: rows[0][c] for c in const}
    per = {_tid_from_filename(r['filename']): {c: r[c] for c in var} for r in rows}
    return per, corpus, p


def load_prdc_per_sample(root, manifold_k=3, num_splits=5):
    """`<root>/../preds.npy` 의 CLaTr latent 로 PRDC 를 **sample 별로** 다시 계산한다.

    같은 분해를 `scripts/eval/prdc_per_sample.py::per_sample_prdc` 도 한다 (precision/density
    만). 여기서는 recall/coverage 와 split 인덱스까지 필요해서 초함수로 다시 적었다 — 정의를
    바꾸면 **두 곳 다** 고쳐야 한다.

    `preds_scores.csv` 에는 PRDC 가 set 단위 스칼라 하나로만 남지만, prdc.py 의
    `compute_prdc` 를 보면 `.mean()` 직전까지는 sample 별 값이 있다:
      precision[j] : pred j 가 GT 초구(어떤 GT 의 k-NN 반경) 안에 들어갔나 (0/1)
      density[j]   : pred j 를 감싸는 GT 초구 개수 / k  — 1 을 넘을 수 있는 '깊이'
      recall[i]    : GT i 가 pred 매니폴드 안에 들어갔나 (0/1)
      coverage[i]  : GT i 의 최근접 pred 가 GT i 의 k-NN 반경 안에 있나 (0/1)
    precision/density 는 **pred** 인덱스, recall/coverage 는 **GT** 인덱스다. pred i 와
    ref i 가 같은 sequence 라 한 줄에 같이 찍지만 묻는 게 서로 다르다 (앞 둘 = 이 예측이
    GT 분포 안이냐, 뒤 둘 = 이 GT 가 예측들에 덮였느냐). FCD 는 분포 간 Frechet 거리라
    sample 별 값이 원래 없다.

    재현 조건 (틀리면 집계가 metrics.json 과 안 맞는다):
      - `compute(num_splits=5)` 가 저장 순서대로 5 등분한 **chunk 안에서** 반경을 잰다.
        따라서 sample 값은 그 sample 이 어느 chunk 에 들어갔는지에 의존한다.
      - distance='euclidean', manifold_k=3 (callback.py 의 ManifoldMetrics 설정).
      - recall 은 prdc.py 의 브로드캐스트를 **그대로** 따라했다. 원본 clovaai 구현은
        `fake_nn[None, :]` (열=fake 인덱스) 인데 이 저장소는 `fake_nn.unsqueeze(1)` 이라
        chunk 크기가 같을 때 조용히 broadcast 되어 `d[i,j] < fake_nn[i]` 를 본다.
        여기서 고치면 뷰어 값이 로그된 수치와 어긋나므로 일부러 맞춰 두었다.

    -> (per {tid: {...}}, agg {키: 재계산 집계}, path)
    """
    p = osp.join(osp.dirname(osp.abspath(root)), 'preds.npy')
    if not osp.isfile(p):
        return {}, {}, None
    import torch
    b = np.load(p, allow_pickle=True).item()
    need = ('m_pred_latents', 'm_ref_latents', 'ref_filenames')
    if any(k not in b for k in need):
        return {}, {}, None
    pred = torch.stack(list(b['m_pred_latents'])).float()
    ref = torch.stack(list(b['m_ref_latents'])).float()
    tids = [_tid_from_filename(f) for f in b['ref_filenames']]

    def nn_radius(X):
        d = torch.cdist(X.unsqueeze(0), X.unsqueeze(0), 2).squeeze(0)
        return torch.topk(d, manifold_k + 1, largest=False, dim=-1).values.max(axis=-1).values

    per, acc, off = {}, {k: [] for k in ('precision', 'recall', 'density', 'coverage')}, 0
    for si, (r, f) in enumerate(zip(ref.chunk(num_splits, 0), pred.chunk(num_splits, 0))):
        rr, fr = nn_radius(r), nn_radius(f)
        d = torch.cdist(r.unsqueeze(0), f.unsqueeze(0), 2).squeeze(0)      # (n_ref, n_pred)
        inside = d < rr.unsqueeze(1)                                       # GT 초구 안?
        prec = inside.any(axis=0).to(float)                                # per pred
        dens = inside.sum(axis=0).to(float) / manifold_k                   # per pred
        rec = (d < fr.unsqueeze(1)).any(axis=1).to(float)                  # per GT (위 주석)
        cov = (d.min(axis=1).values < rr).to(float)                        # per GT
        for k, v in (('precision', prec), ('recall', rec), ('density', dens), ('coverage', cov)):
            acc[k].append(float(v.mean()))
        for j in range(len(f)):
            if off + j < len(tids):
                per[tids[off + j]] = {'split': si, 'precision': float(prec[j]),
                                      'density': float(dens[j]), 'recall': float(rec[j]),
                                      'coverage': float(cov[j]),
                                      'n_split': int(len(f))}
        off += len(f)
    agg = {f'clatr/{k}': float(np.mean(v)) for k, v in acc.items()}
    return per, agg, p


def _short(name):
    """긴 SD data_name 을 목록에서 읽을 만하게 줄인다. dl3dv 는 hash 를 앞 8 자만."""
    if '__' in name:
        scene, tgt, ctx = (name.split('__') + ['', ''])[:3]
        pres = lambda s: '_'.join(s.split('_')[-2:])
        return f"{scene}  {pres(tgt)} <- {pres(ctx)}"
    p = name.split('_')
    return f"{p[0]}_{p[1][:8]}_{'_'.join(p[2:])}" if len(p) > 2 else name


def _fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def add_frustums(server, name, c2w_gl, fx, fy, w, h, color, scale, downsample=1):
    """Add camera frustums (OpenGL c2w -> OpenCV) under a named group."""
    fov = 2 * np.arctan2(h / 2, fy)              # vertical fov
    aspect = w / h
    handles = []
    for i in range(0, len(c2w_gl), downsample):
        c2w_cv = c2w_gl[i] @ _GL2CV              # OpenGL c2w -> OpenCV c2w (cam looks +Z)
        R, t = c2w_cv[:3, :3], c2w_cv[:3, 3]
        wxyz = vtf.SO3.from_matrix(R).wxyz
        handles.append(server.scene.add_camera_frustum(
            f"{name}/cam_{i:04d}", fov=float(fov), aspect=float(aspect), scale=scale,
            color=color, wxyz=wxyz, position=t.astype(np.float32)))
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='results/<exp>/test dir')
    ap.add_argument('--dataset', choices=['dl3dv', 'sd'], default='dl3dv',
                    help="배경 소스. dl3dv = scene transforms.json 의 나머지 카메라(기존 동작), "
                         "sd = clip 의 da3 depth point cloud + 그 clip 의 카메라 "
                         "(어느 clip 인지는 --pc-clips)")
    ap.add_argument('--dl3dv-root', default=DL3DV_ROOT)
    ap.add_argument('--port', type=int, default=8080)
    ap.add_argument('--grey-downsample', type=int, default=2, help='subsample scene (grey) cams')
    ap.add_argument('--cam-stride', type=int, default=2,
                    help='target(blue)/pred(red) frustum 을 몇 프레임마다 그릴지 (1 = 전부)')
    ap.add_argument('--context-stride', type=int, default=None,
                    help='sd context(green) frustum stride. 기본 = --grey-downsample')
    ap.add_argument('--no-prdc', action='store_true',
                    help='preds.npy 에서 sample 별 PRDC 재계산을 건너뛴다')
    ap.add_argument('--only', default='all',
                    choices=['all', 'prec-fail', 'dens-low', 'recall-fail', 'cov-fail', 'fail'],
                    help="sample 별 PRDC 로 sequence 목록을 거른다 (slider/Next 도 이 목록만 돈다). "
                         "prec-fail = precision 0, dens-low = density <= --dens-max, "
                         "fail = precision 0 이거나 coverage 0 이거나 density <= --dens-max")
    ap.add_argument('--dens-max', type=float, default=0.0,
                    help="dens-low/fail 의 density 기준 (이 값 **이하**). 기본 0 = GT 초구에 "
                         "하나도 안 걸린 것만")
    ap.add_argument('--list-out', default=None,
                    help='거른 목록을 csv 로 저장할 경로 (기본: 저장 안 함)')
    ap.add_argument('--up', default=None, help="world up. 기본: dl3dv '+y', sd '+z'")
    # --- sd 전용
    ap.add_argument('--sd-root', default=SD_ROOT)
    ap.add_argument('--sd-split', default='whuman')
    ap.add_argument('--pc-frames', type=int, default=49,
                    help='clip 당 unproject 할 프레임 수. clip 전체 길이 T 에 균등 분포시킨다 '
                         '(T=81, --pc-frames 49 면 49장이 81 프레임 전체에 퍼진다 — '
                         'target segment 구간만 뽑는 게 아니다)')
    ap.add_argument('--pc-stride', type=int, default=3, help='depth 격자 픽셀 stride')
    ap.add_argument('--pc-conf-pct', type=float, default=40.0, help='conf 백분위수 컷')
    ap.add_argument('--pc-max-points', type=int, default=400_000)
    ap.add_argument('--pc-size', type=float, default=0.0,
                    help='point 크기 (0 = target extent 로 자동)')
    ap.add_argument('--pc-clips', choices=['context', 'target', 'both'], default='context',
                    help="point cloud 로 unproject 할 clip. 두 clip 모두 자기 umeyama_gt.json 으로 "
                         "GT meters 에 올라가므로 같은 world 에 놓인다 — 'both' 로 켜면 정렬이 "
                         "맞는지 눈으로 확인된다. 기본 'context' = 기존 동작")
    ap.add_argument('--pc-tint', action='store_true',
                    help='RGB 대신 context=초록 / target=파랑 단색으로 칠해 두 point cloud 를 구분 '
                         '(각 clip 의 카메라 색과 같게 맞춘 것)')
    ap.add_argument('--pc-target-cams', action='store_true',
                    help='target clip 의 pose.npz 카메라도 그린다. 색은 **청록** — ref(파랑) 위에 '
                         '정확히 포개지므로 같은 파랑이면 겹쳤는지 어긋났는지 구분이 안 된다')
    ap.add_argument('--pc-seg-only', action='store_true',
                    help='target clip 을 **target segment 범위**(clip 앞 len(ref) 프레임)로 자른다. '
                         'SD clip 은 81 프레임인데 segment 는 앞 49 프레임이라, 안 자르면 뒤 32 '
                         '프레임이 궤적 옆을 median 0.9m 로 나란히 지나가 두 번째 궤적처럼 보인다. '
                         'context 는 clip 전체를 쓰는 게 맞으므로 영향받지 않는다')
    args = ap.parse_args()

    seqs = sorted(osp.basename(p)[:-len('_transforms_pred.json')]
                  for p in glob.glob(osp.join(args.root, '*_transforms_pred.json')))
    if not seqs:
        raise SystemExit(f"no *_transforms_pred.json in {args.root}")
    print(f"{len(seqs)} sequences in {args.root}")

    cam_stride = max(1, int(args.cam_stride))
    ctx_stride = max(1, int(args.context_stride if args.context_stride is not None
                            else args.grey_downsample))

    seq_metrics, corpus_metrics, mpath = load_seq_metrics(args.root)
    unit = 'm' if args.dataset == 'sd' else 'u'          # sd = GT meters, dl3dv = 정규화 단위
    if mpath:
        print(f"metrics: {mpath}  (per-seq {len(seq_metrics)} rows, "
              f"{len(corpus_metrics)} corpus-level cols)")
        if corpus_metrics:
            print("  corpus-level (전 행 동일 = set 단위로 한 번 계산된 값, "
                  "sequence 별 수치가 아니다):")
            print("    " + "  ".join(f"{k} {_fmt(v)}" for k, v in corpus_metrics.items()))
    else:
        print("metrics: preds_scores.csv 없음 -> 궤적 오차만 찍는다")

    prdc, prdc_agg, ppath = ({}, {}, None) if args.no_prdc else load_prdc_per_sample(args.root)
    if ppath:
        # 재계산이 로그된 집계와 맞는지 여기서 확인한다 (chunk 순서/k/거리 중 하나만 어긋나도
        # sample 별 값이 조용히 딴 값이 된다).
        ref_agg = corpus_metrics or {}
        chk = "  ".join(
            f"{k.split('/')[-1]} {v:.4f}"
            + (f"(csv {_fmt(ref_agg[k])})" if k in ref_agg
               and abs(float(ref_agg[k]) - v) > 5e-4 else "")
            for k, v in prdc_agg.items())
        print(f"per-sample PRDC: {ppath} 에서 재계산 (k=3, euclidean, 5 splits)")
        print(f"  재계산 집계 = {chk}   <- csv/metrics.json 과 같아야 맞게 재현된 것")

    if args.only != 'all':
        # 주의: PRDC 는 **전체 순서**로 5 등분한 chunk 안에서 계산된다. 그래서 거르기는
        # 반드시 계산이 끝난 뒤에 한다 — seqs 를 먼저 줄이면 chunk 구성이 달라져 값이 바뀐다.
        if not prdc:
            raise SystemExit(f"--only {args.only} 에는 sample 별 PRDC 가 필요하다 "
                             f"(preds.npy 없음 또는 --no-prdc)")
        def _hit(m):
            if m is None:
                return False
            if args.only == 'prec-fail':
                return m['precision'] < 0.5
            if args.only == 'dens-low':
                return m['density'] <= args.dens_max
            if args.only == 'recall-fail':
                return m['recall'] < 0.5
            if args.only == 'cov-fail':
                return m['coverage'] < 0.5
            return (m['precision'] < 0.5 or m['coverage'] < 0.5
                    or m['density'] <= args.dens_max)
        kept = [s for s in seqs if _hit(prdc.get(s))]
        miss = [s for s in seqs if s not in prdc]
        if miss:
            print(f"  주의: preds.npy 에 없는 sequence {len(miss)} 개는 목록에서 빠졌다")
        print(f"--only {args.only}"
              + (f" (density <= {args.dens_max})" if args.only in ('dens-low', 'fail') else "")
              + f": {len(kept)}/{len(seqs)} sequence")
        if not kept:
            raise SystemExit("조건에 맞는 sequence 가 없다")
        rows = sorted(((s, prdc[s]) for s in kept), key=lambda x: (x[1]['density'],
                                                                   x[1]['precision']))
        tf = lambda v: 'T' if v >= 0.5 else 'F'
        print(f"  {'#':>4}  {'density':>7} {'prec':>4} {'rec':>4} {'cov':>4} {'split':>5}  name")
        for s, m in rows:
            print(f"  {seqs.index(s):>4}  {m['density']:7.4f} {tf(m['precision']):>4} "
                  f"{tf(m['recall']):>4} {tf(m['coverage']):>4} {m['split']:>5}  {_short(s)}")
        if args.list_out:
            import csv as _csv
            with open(args.list_out, 'w', newline='') as fh:
                w_ = _csv.writer(fh)
                w_.writerow(['orig_index', 'name', 'density', 'precision', 'recall',
                             'coverage', 'split'])
                for s, m in rows:
                    w_.writerow([seqs.index(s), s, m['density'], m['precision'],
                                 m['recall'], m['coverage'], m['split']])
            print(f"  -> {args.list_out}")
        seqs = kept

    server = viser.ViserServer(port=args.port)
    # dl3dv 는 OpenGL/nerfstudio world (y-up). sd 의 world 는 umeyama 로 올린 GT(Unreal) 미터
    # 좌표라 z-up 이다.
    server.scene.set_up_direction(args.up or ('+z' if args.dataset == 'sd' else '+y'))

    gui_info = server.gui.add_text("sequence", initial_value=seqs[0], disabled=True)
    gui_cap = server.gui.add_text("caption", initial_value="", disabled=True)
    gui_slider = server.gui.add_slider("index", min=0, max=len(seqs) - 1, step=1, initial_value=0)
    gui_load = server.gui.add_button("Load")
    gui_next = server.gui.add_button("Next")
    state = {"groups": []}

    def clear():
        for g in state["groups"]:
            g.remove()
        state["groups"] = []

    def load(idx):
        clear()
        name = seqs[idx]
        gui_info.value = f"[{idx}/{len(seqs)-1}] {name}"
        cap_p = osp.join(args.root, f"{name}_caption.json")
        if osp.isfile(cap_p):
            try:
                gui_cap.value = str(list(json.load(open(cap_p)).values())[0])[:200]
            except Exception:
                gui_cap.value = ""
        ref_c2w, rfx, rfy, rw, rh = load_transforms(osp.join(args.root, f"{name}_transforms_ref.json"))
        pred_c2w, pfx, pfy, pw, ph = load_transforms(osp.join(args.root, f"{name}_transforms_pred.json"))
        # frustum scale from target extent
        ext = np.linalg.norm(ref_c2w[:, :3, 3] - ref_c2w[0, :3, 3], axis=1).max()
        fscale = float(max(ext, 1e-3)) * 0.08
        if args.dataset == 'sd':
            # 배경 = clip 의 da3 depth point cloud + 그 clip 의 카메라.
            # context(GREEN) / target(BLUE) 는 각자 자기 umeyama_gt.json 으로 GT meters 에
            # 올라가므로 같은 world frame 에 놓인다 — 둘을 같이 그리면 그 정렬이 곧 검증이 된다.
            scene, tgt, ctx = name.split('__')
            # (tag, clip, pc_tint_색, 카메라_색) — 둘을 나눠 둔 이유:
            #   point cloud 는 그게 어느 clip 의 것인지 알아보는 게 목적이라 카메라와 **같은**
            #     색이 맞다 (context=초록, target=파랑).
            #   카메라는 --pc-target-cams 로 켜면 ref(파랑) 위에 정확히 포개지는데, 같은 파랑이면
            #     겹쳤는지 어긋났는지 구분이 안 된다. 그래서 target clip 카메라만 청록으로 뺀다.
            _CTX = ('context', ctx, (40, 190, 90), (40, 190, 90))
            _TGT = ('target_clip', tgt, (40, 90, 230), (0, 200, 200))
            todo = {'context': [_CTX], 'target': [_TGT], 'both': [_CTX, _TGT]}[args.pc_clips]
            for tag, clip, col, cam_col in todo:
                cdir = osp.join(args.sd_root, 'da3', args.sd_split, scene, clip)
                mp4 = osp.join(args.sd_root, 'video', args.sd_split, scene, clip + '.mp4')
                # target clip 의 segment 는 clip 앞부분 len(ref) 프레임이다 (whuman 160/160
                # sequence 에서 offset 0, 위치오차 <1e-3m 로 확인). context 는 clip 전체를
                # 쓰는 게 맞으므로 자르지 않는다.
                seg = ((0, len(ref_c2w)) if (args.pc_seg_only and tag == 'target_clip')
                       else None)
                try:
                    P, C = sd_pointcloud(cdir, mp4, n_frames=args.pc_frames, stride=args.pc_stride,
                                         conf_pct=args.pc_conf_pct, max_points=args.pc_max_points,
                                         t_range=seg)
                    if args.pc_tint:
                        C = np.tile(np.array(col, np.uint8), (len(P), 1))
                    state["groups"].append(server.scene.add_point_cloud(
                        f"pc_{tag}", points=P, colors=C,
                        point_size=float(args.pc_size or fscale * 0.06)))
                    cw2c, cK, _ = sd_clip_meters(cdir)
                    cc2w = np.linalg.inv(cw2c) @ _GL2CV  # CV c2w -> GL c2w (add_frustums 가 되돌린다)
                    n_clip = len(cc2w)
                    if seg is not None:
                        cc2w = cc2w[seg[0]:seg[1]]
                    ch, cw = int(round(cK[0, 1, 2] * 2)), int(round(cK[0, 0, 2] * 2))
                    n_cam = 0
                    if tag == 'context' or args.pc_target_cams:
                        h_ = add_frustums(server, tag, cc2w, cK[0, 0, 0], cK[0, 1, 1], cw, ch,
                                          cam_col, fscale * 0.7, downsample=ctx_stride)
                        state["groups"] += h_
                        n_cam = len(h_)
                    print(f"  sd pc[{tag}]: {len(P)} points from {clip} + {n_cam}/{len(cc2w)} "
                          f"cams (stride {ctx_stride}"
                          + (f", clip {n_clip} 프레임 중 [{seg[0]},{seg[1]}) 만" if seg else "")
                          + ")", flush=True)
                    if tag == 'target_clip':
                        # target clip 의 pose 는 ref(파랑) 가 그리는 바로 그 카메라다. 두 경로가
                        # 다른 곳(pose.npz+sim3 / json 의 first_extrinsic 복원)을 거치므로,
                        # 여기 차이가 0 이 아니면 point cloud 와 ref 의 world 가 어긋난 것이다.
                        n = min(len(cc2w), len(ref_c2w))
                        d = np.linalg.norm(cc2w[:n, :3, 3] - ref_c2w[:n, :3, 3], axis=1)
                        print(f"  align: target clip pose vs ref(blue) 위치차 "
                              f"median {np.median(d):.4f}m  max {d.max():.4f}m  "
                              f"(n={n}, 0 에 가까워야 같은 world)", flush=True)
                except Exception as e:
                    print(f"  sd pc[{tag}] 실패 ({clip}): {type(e).__name__}: {e}", flush=True)
        # grey = the REST of the scene cameras (exclude the target frames, which match exactly)
        sc = (osp.join(args.dl3dv_root, scene_chunk(name), 'transforms.json')
              if args.dataset == 'dl3dv' else None)
        if sc and osp.isfile(sc):
            dj = json.load(open(sc))
            fr = sorted(dj['frames'], key=lambda f: f['file_path'])
            sc_c2w = np.array([f['transform_matrix'] for f in fr], dtype=np.float64)
            eps = max(fscale * 0.05, 1e-4)
            dmin = np.min(np.linalg.norm(sc_c2w[:, :3, 3][:, None] - ref_c2w[:, :3, 3][None], axis=-1), axis=1)
            rest = sc_c2w[dmin > eps]                 # drop target frames from grey
            gh = add_frustums(server, "grey", rest, dj['fl_x'], dj['fl_y'], dj['w'], dj['h'],
                              (150, 150, 150), fscale * 0.6, downsample=args.grey_downsample)
            state["groups"] += gh
        th = add_frustums(server, "target", ref_c2w, rfx, rfy, rw, rh, (40, 90, 230), fscale,
                          downsample=cam_stride)
        ph_ = add_frustums(server, "pred", pred_c2w, pfx, pfy, pw, ph, (230, 40, 40), fscale,
                           downsample=cam_stride)
        state["groups"] += th + ph_
        print(f"loaded [{idx}/{len(seqs)-1}] {name}", flush=True)
        print(f"  cams : target(blue) {len(th)}/{len(ref_c2w)} frames, "
              f"pred(red) {len(ph_)}/{len(pred_c2w)} (stride {cam_stride})", flush=True)
        m = traj_metrics(ref_c2w, pred_c2w)
        print(f"  traj : pos_rmse {m['pos_rmse']:.4f}{unit}  mean {m['pos_mean']:.4f}  "
              f"max {m['pos_max']:.4f}  end {m['pos_end']:.4f} | "
              f"rot mean {m['rot_mean']:.2f}deg  max {m['rot_max']:.2f}deg", flush=True)
        print(f"         path len  ref {m['path_ref']:.4f}{unit}  "
              f"pred {m['path_pred']:.4f}{unit}", flush=True)
        sm = seq_metrics.get(name)
        if sm:
            print("  eval : " + "  ".join(f"{k} {_fmt(v)}" for k, v in sm.items()), flush=True)
        elif seq_metrics:
            print(f"  eval : preds_scores.csv 에 {name} 행이 없다", flush=True)
        pm = prdc.get(name)
        if pm:
            tf = lambda v: 'T' if v >= 0.5 else 'F'
            print(f"  prdc : [pred] precision {tf(pm['precision'])}  "
                  f"density {pm['density']:.4f} ({int(round(pm['density'] * 3))}/3 GT ball) | "
                  f"[GT] recall {tf(pm['recall'])}  coverage {tf(pm['coverage'])}   "
                  f"(split {pm['split']}, n={pm['n_split']})", flush=True)

    @gui_load.on_click
    def _(_):
        load(int(gui_slider.value))

    @gui_next.on_click
    def _(_):
        gui_slider.value = min(int(gui_slider.value) + 1, len(seqs) - 1)
        load(int(gui_slider.value))

    load(0)
    print(f"viser server on port {args.port} (ctrl-C to stop)")
    import time
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
