"""데이터축(vista vs dynpose) 코퍼스를 **학습이 실제로 읽는 단위**로 나란히 센다.

왜 필요한가
-----------
사용자 질문은 "vista(scene 적고 preset 많음) vs dynpose(scene 많고 preset 적음) 중 어느
쪽이 카메라를 더 잘 만드나"다. 그런데 두 코퍼스는 seg-list 크기, 씬 수, preset 분포,
게이트 통과 품질(hole/τ/subject_in_frame)이 전부 다르다. 학습 지표만 비교하면 어느 축이
움직여서 차이가 났는지 못 가른다. 그래서 **모델을 돌리기 전에** 코퍼스 자체를 같은 자로 잰다.

무엇을 읽는가
-------------
`<root>/<batch>/<hash>/da3/prompts.json` — `pose_source: da3` 인 arm 이 실제로 읽는 파일이다
(`<scene>/prompts.json` 이 **아니다**; dataset_dl3dv.py 가 da3 하위로 내려간다). 세그먼트
엔트리에서 뽑는 것:
  preset / preset_raw   preset 다양성 (데이터축의 "preset 많음/적음" 이 여기서 정량화된다)
  aim                   free / look_at — 조준 방식
  anchor_label          target 라벨 (씬 안의 무엇을 잡았나)
  hole_fraction         depth warp 구멍 비율 = 하류 video model 부담
  tau_max               |t|/z_med 최댓값 = 카메라 이동 강도
  subject_in_frame      중앙 80% 안에 subject 가 든 프레임 비율
  subject_area_med      subject 2D 면적 중앙값 (shot size)
  subject_visible_frac  z-buffer 통과율 (가림)
`variant_id` 는 **코퍼스 키가 아니다** — 재굽기해도 이름은 같고 knob/pose 만 바뀐다. 그래서
집계 키로 안 쓰고 진단 출력에만 쓴다.

seg-list 밖 세그먼트는 안 센다. 데이터셋은 prompts.json 을 전량 열거하지만 base.py 가 그
다음에 seg-list 로 Subset 을 뜨므로, **학습이 만지는 집합은 seg-list** 다 (FIX-D138 과 같은
구분). `--split train|test|both` 로 어느 리스트를 볼지 고른다.

usage
-----
  python scripts/eval/corpus_axis_compare.py \
    --corpus vista_d128=/data1/cympyc1785/data/Vista4D-Eval-Data/latentcam_da3_k6_d128:vista4d \
    --corpus dynpose_d137=/data1/cympyc1785/data/DynPose-LBM/latentcam_dynpose_d137:dynpose_dd10
  # <이름>=<root>:<seg_prefix>  ->  <root>/seg_list_<seg_prefix>_{train,test}.txt

out -> stdout 표 + <out>/corpus_axis.json (코퍼스별 raw 카운트와 분위수 전량)
"""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from json import dump, load
from os import makedirs, path
from statistics import median


# 이 필드들만 분위수를 낸다. 전부 [0,1] 이거나(hole/in_frame/area/visible) 무차원(tau)이라
# 코퍼스 간 비교가 성립한다 — PRDC 처럼 표본수에 끌려가는 양이 아니다.
NUMERIC = ['hole_fraction', 'tau_max', 'subject_in_frame',
           'subject_area_med', 'subject_visible_frac']

# 게이트 꼬리 — "이 세그먼트가 학습 신호로 쓸 만한가"의 하한선. 분위수만 보면 med 가 같아도
# 꼬리 두께가 다른 걸 못 본다 (vista 와 dynpose 가 정확히 그 경우다).
# (이름, 필드, 부등호, 임계). 임계는 goals.md 단기목표 완료조건 (a)(c) 에서 따온 값이다.
TAILS = [
    ('subject_visible_frac < 0.05', 'subject_visible_frac', '<', 0.05),
    ('subject_visible_frac < 0.30', 'subject_visible_frac', '<', 0.30),
    ('subject_area_med < 0.005', 'subject_area_med', '<', 0.005),
    ('subject_in_frame < 0.50', 'subject_in_frame', '<', 0.50),
    ('subject_in_frame < 0.85', 'subject_in_frame', '<', 0.85),
    ('hole_fraction > 0.55', 'hole_fraction', '>', 0.55),
    ('tau_max > 1.5', 'tau_max', '>', 1.5),
]

# 캡션이 "추종한다"고 주장하는지 판정하는 키워드. `track_*` preset 인데 `aim=free` 면 카메라는
# 따라 움직이되 **다시 겨냥하지 않는다** — 그런데 캡션은 여전히 프레이밍을 약속한다.
# 그 모순을 세는 데 쓴다 (goals.md 중기목표 1, task #134).
TRACK_WORDS = ('track', 'follow', 'keeping pace', 'keeping it in', 'alongside',
               'stays with', 'staying with')
# 그중에서도 **프레임 안에 유지한다**고 못박는 표현. aim=free 는 이 약속을 지킬 수단이 없다.
FRAME_WORDS = ('keeping it in', 'keeping the', 'in frame', 'in the frame', 'framed',
               'stays in', 'centered', 'keeping pace')


def caption_text(e):
    """세그먼트 엔트리에서 사람이 읽는 캡션 문자열을 이어붙인다."""
    cf = e.get('caption_fields') or {}
    parts = [cf.get('motion') or '', cf.get('framing_nl') or '', cf.get('framing') or '']
    p = e.get('prompt_camera_with_scene_video') or {}
    if isinstance(p, dict):
        parts.append(p.get('concise') or '')
    elif isinstance(p, str):
        parts.append(p)
    return ' '.join(parts).lower()


def preset_family(p):
    """preset 이름 -> 계열. dd_* 는 DataDoP 유래 one-off 라 preset 다양성으로 안 센다."""
    p = p or ''
    if p.startswith('dd_'):
        return 'dd_*(DataDoP)'
    if p.startswith('track_'):
        return 'track_*'
    return 'LBM preset'


def framing_scope_keep(e, scope):
    """`--framing_scope` 필터. subject_in_frame 을 **책임질 수 있는** 세그먼트만 남긴다.

    사용자 지시(2026-09-06): "aim 이 follow 인 것들이나 target 이 없는 free moving 은 물체의
    subject in frame 율이 낮은 건 당연해 이것들 제외한 preset 들만 재줘."

    두 축을 각각 거른다 — 이 코퍼스에서 둘은 **독립**이다:
      · `track_*` preset = **병진 추종**(follow). 카메라가 subject 를 따라 이동한다.
      · `aim` = **회전 조준**. `look_at` 만이 매 프레임 subject 를 다시 겨냥한다.
        `free` 는 조준 자체가 없고(= target 없는 free moving), `traj` 는 궤적 접선을 본다
        (`aim_keyframes=0` 이면 조준 안 함). 둘 다 프레이밍을 약속할 수단이 없다.
    그래서 `aimed_nontrack` 은 `aim == 'look_at'` 이고 `track_*` 도 `dd_*` 도 아닌 것만 남긴다
    (`dd_*` 는 DataDoP 유래 one-off 라 subject 개념 자체가 없다).
    `aimed` 는 track 여부를 안 보고 `aim == 'look_at'` 만 본다 — `track_look_at` 처럼 추종 +
    조준을 둘 다 하는 preset 은 프레이밍을 약속할 수단이 있으므로 남길 근거가 있다.
    """
    if scope == 'all':
        return True
    if (e.get('aim') or '') != 'look_at':
        return False
    if scope == 'aimed':
        return True
    p = e.get('preset') or ''
    return not (p.startswith('track_') or p.startswith('dd_'))


def read_seg_ids(root, prefix, split):
    """seg-list 를 읽어 {'<batch>/<hash>/<seg>'} 집합으로 돌려준다."""
    names = {'train': ['train'], 'test': ['test'], 'both': ['train', 'test']}[split]
    ids, missing = set(), []
    for n in names:
        p = path.join(root, f'seg_list_{prefix}_{n}.txt')
        if not path.exists(p):
            missing.append(p)
            continue
        with open(p) as f:
            ids |= {ln.strip() for ln in f if ln.strip()}
    return ids, missing


def collect(root, ids):
    """seg id 집합 -> 세그먼트 엔트리 리스트. prompts.json 은 씬당 한 번만 연다."""
    by_scene = defaultdict(set)
    for i in ids:
        b, h, s = i.split('/')
        by_scene[(b, h)].add(s)
    rows, no_json, no_entry = [], [], 0
    for (b, h), segs in sorted(by_scene.items()):
        pj = path.join(root, b, h, 'da3', 'prompts.json')
        if not path.exists(pj):
            no_json.append(f'{b}/{h}')
            continue
        with open(pj) as f:
            d = load(f)
        for s in sorted(segs, key=lambda x: int(x) if x.isdigit() else x):
            e = d.get(s)
            if not isinstance(e, dict):
                no_entry += 1
                continue
            e = dict(e)
            e['_scene'] = f'{b}/{h}'
            e['_seg'] = s
            rows.append(e)
    return rows, no_json, no_entry


def quantiles(xs):
    """p10 / median / p90 / mean. 표본이 비면 전부 None."""
    xs = sorted(v for v in xs if isinstance(v, (int, float)))
    if not xs:
        return {'n': 0, 'p10': None, 'med': None, 'p90': None, 'mean': None}
    def q(f):
        return xs[min(len(xs) - 1, max(0, int(round(f * (len(xs) - 1)))))]
    return {'n': len(xs), 'p10': q(0.10), 'med': median(xs), 'p90': q(0.90),
            'mean': sum(xs) / len(xs)}


def summarize(name, root, rows):
    scenes = {r['_scene'] for r in rows}
    presets = Counter(r.get('preset') or '(none)' for r in rows)
    aims = Counter(r.get('aim') or '(none)' for r in rows)
    labels = Counter(r.get('anchor_label') or '(none)' for r in rows)
    per_scene = Counter(r['_scene'] for r in rows)
    # 씬당 고유 preset 수 — "preset 많음/적음" 을 씬 단위로 본 값
    pres_per_scene = defaultdict(set)
    for r in rows:
        pres_per_scene[r['_scene']].add(r.get('preset'))

    # preset 계열 — "uniq preset 전체" 수는 dd_* one-off 가 부풀린다. 계열별로 쪼개야
    # 실제 어휘 크기가 보인다.
    fam = defaultdict(lambda: {'n': 0, 'uniq': set()})
    for r in rows:
        f = fam[preset_family(r.get('preset'))]
        f['n'] += 1
        f['uniq'].add(r.get('preset'))
    fam_out = {k: {'n_seg': v['n'], 'n_uniq': len(v['uniq'])} for k, v in fam.items()}

    # 게이트 꼬리 (개수 + 비율)
    tails = {}
    for tname, key, op, thr in TAILS:   # ← `name` 으로 쓰면 인자 name 을 덮어쓴다
        c = sum(1 for r in rows
                if isinstance(r.get(key), (int, float))
                and ((r[key] < thr) if op == '<' else (r[key] > thr)))
        tails[tname] = {'n': c, 'frac': c / max(len(rows), 1)}

    # track_* x aim 교차표 — 캡션과 기하가 어긋나는 조합을 직접 센다.
    ta = {}
    for r in rows:
        if not (r.get('preset') or '').startswith('track_'):
            continue
        k = r.get('aim') or '(none)'
        d = ta.setdefault(k, {'n': 0, 'visible_lt_005': 0, 'in_frame_lt_05': 0,
                              'caption_claims_track': 0, 'caption_claims_framing': 0})
        d['n'] += 1
        v, i = r.get('subject_visible_frac'), r.get('subject_in_frame')
        if isinstance(v, (int, float)) and v < 0.05:
            d['visible_lt_005'] += 1
        if isinstance(i, (int, float)) and i < 0.50:
            d['in_frame_lt_05'] += 1
        t = caption_text(r)
        if any(w in t for w in TRACK_WORDS):
            d['caption_claims_track'] += 1
        if any(w in t for w in FRAME_WORDS):
            d['caption_claims_framing'] += 1

    return {
        'name': name, 'root': root,
        'n_seg': len(rows), 'n_scene': len(scenes),
        'seg_per_scene': quantiles(list(per_scene.values())),
        'uniq_preset_total': len(presets),
        'uniq_preset_per_scene': quantiles([len(v) for v in pres_per_scene.values()]),
        'preset_counts': dict(presets.most_common()),
        'preset_family': fam_out,
        'gate_tails': tails,
        'track_by_aim': ta,
        'aim_counts': dict(aims.most_common()),
        'anchor_label_top': dict(labels.most_common(15)),
        'n_anchor_label_uniq': len(labels),
        'numeric': {k: quantiles([r.get(k) for r in rows]) for k in NUMERIC},
    }


def main():
    ap = ArgumentParser()
    ap.add_argument('--corpus', action='append', required=True,
                    help='<이름>=<root>:<seg_prefix> 형식. 반복 지정하면 나란히 비교한다.')
    ap.add_argument('--split', default='train', choices=['train', 'test', 'both'],
                    help='어느 seg-list 를 셀지 (기본 train — 학습 신호를 보는 게 목적)')
    ap.add_argument('--out', default='results/compare/corpus_axis',
                    help='corpus_axis.json 이 떨어질 디렉토리')
    ap.add_argument('--top_preset', type=int, default=14,
                    help='표에 찍을 preset 상위 개수 (json 에는 전량 들어간다)')
    ap.add_argument('--framing_scope', default='all',
                    choices=['all', 'aimed', 'aimed_nontrack'],
                    help='게이트를 어느 부분집합에서 잴지. all=전량(기존 동작, 기본), '
                         'aimed=aim==look_at 만, aimed_nontrack=거기서 track_*/dd_* 도 제외')
    a = ap.parse_args()

    summaries = []
    for spec in a.corpus:
        name, rest = spec.split('=', 1)
        root, prefix = rest.rsplit(':', 1)
        ids, missing = read_seg_ids(root, prefix, a.split)
        if missing:
            print(f'[warn] {name}: seg-list 없음 -> {missing}')
        rows, no_json, no_entry = collect(root, ids)
        if no_json:
            print(f'[warn] {name}: da3/prompts.json 없는 씬 {len(no_json)} '
                  f'(예: {no_json[:3]})')
        if no_entry:
            print(f'[warn] {name}: prompts.json 에 엔트리 없는 seg {no_entry}')
        n_all = len(rows)
        if a.framing_scope != 'all':
            rows = [r for r in rows if framing_scope_keep(r, a.framing_scope)]
            print(f'[scope] {name}: {a.framing_scope} -> {len(rows)}/{n_all} seg '
                  f'({100 * len(rows) / max(n_all, 1):.1f}%) 남김')
        s = summarize(name, root, rows)
        s['framing_scope'] = a.framing_scope
        s['n_seg_before_scope'] = n_all
        summaries.append(s)

    W = max(len(s['name']) for s in summaries) + 2
    def row(lbl, vals):
        print(f'  {lbl:26s} ' + ' '.join(f'{v:>{W}}' for v in vals))

    print(f'\n=== 코퍼스 규모 (split={a.split}, scope={a.framing_scope}) ' + '=' * 26)
    row('', [s['name'] for s in summaries])
    row('seg (학습 대상)', [s['n_seg'] for s in summaries])
    row('scene', [s['n_scene'] for s in summaries])
    row('seg/scene med', [f"{s['seg_per_scene']['med']:.0f}" for s in summaries])
    row('uniq preset (전체)', [s['uniq_preset_total'] for s in summaries])
    row('uniq preset/scene med', [f"{s['uniq_preset_per_scene']['med']:.0f}" for s in summaries])
    row('uniq anchor_label', [s['n_anchor_label_uniq'] for s in summaries])

    print(f'\n=== 게이트 수치 (p10 / med / p90) ' + '=' * 34)
    for k in NUMERIC:
        vals = []
        for s in summaries:
            q = s['numeric'][k]
            vals.append('n/a' if q['n'] == 0
                        else f"{q['p10']:.3f}/{q['med']:.3f}/{q['p90']:.3f}")
        row(k, vals)

    print(f'\n=== preset 분포 (상위 {a.top_preset}, % of seg) ' + '=' * 24)
    keys = []
    for s in summaries:
        keys += list(s['preset_counts'])[:a.top_preset]
    for k in dict.fromkeys(keys):
        vals = []
        for s in summaries:
            c = s['preset_counts'].get(k, 0)
            vals.append(f"{100 * c / max(s['n_seg'], 1):5.1f}% ({c})")
        row(k, vals)

    print(f'\n=== preset 계열 (uniq / seg / %) ' + '=' * 31)
    for k in ['dd_*(DataDoP)', 'track_*', 'LBM preset']:
        vals = []
        for s in summaries:
            f = s['preset_family'].get(k, {'n_seg': 0, 'n_uniq': 0})
            vals.append(f"{f['n_uniq']:3d}/{f['n_seg']:5d}/"
                        f"{100 * f['n_seg'] / max(s['n_seg'], 1):5.1f}%")
        row(k, vals)

    print(f'\n=== aim 분포 ' + '=' * 51)
    for k in dict.fromkeys(sum([list(s['aim_counts']) for s in summaries], [])):
        vals = [f"{100 * s['aim_counts'].get(k, 0) / max(s['n_seg'], 1):5.1f}%"
                for s in summaries]
        row(k, vals)

    print(f'\n=== 게이트 꼬리 (n / % of seg) ' + '=' * 33)
    for name, _, _, _ in TAILS:
        vals = [f"{s['gate_tails'][name]['n']:5d} ({100 * s['gate_tails'][name]['frac']:4.1f}%)"
                for s in summaries]
        row(name, vals)

    print(f'\n=== track_* x aim (캡션이 추종을 주장하는 비율) ' + '=' * 17)
    for k in dict.fromkeys(sum([list(s['track_by_aim']) for s in summaries], [])):
        for fld, lbl in [('n', 'n'), ('visible_lt_005', 'visible<0.05'),
                         ('in_frame_lt_05', 'in_frame<0.5'),
                         ('caption_claims_track', '캡션 추종 주장'),
                         ('caption_claims_framing', '캡션 프레이밍 약속')]:
            vals = []
            for s in summaries:
                d = s['track_by_aim'].get(k)
                if not d:
                    vals.append('-')
                elif fld == 'n':
                    vals.append(str(d['n']))
                else:
                    vals.append(f"{d[fld]:5d} ({100 * d[fld] / max(d['n'], 1):4.1f}%)")
            row(f'  aim={k} {lbl}', vals)

    makedirs(a.out, exist_ok=True)
    suffix = '' if a.framing_scope == 'all' else f'_{a.framing_scope}'
    op = path.join(a.out, f'corpus_axis_{a.split}{suffix}.json')
    with open(op, 'w') as f:
        dump({'split': a.split, 'corpora': summaries}, f, indent=2, ensure_ascii=False)
    print(f'\nout -> {op}')


if __name__ == '__main__':
    main()
