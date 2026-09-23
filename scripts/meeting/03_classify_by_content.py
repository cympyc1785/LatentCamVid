# -*- coding: utf-8 -*-
"""버전 A: 오디오를 쓰지 않고, 전사 '내용'만으로 화자를 분류한다.
 - 구조: 00:00:00~01:04:10 학생 A 발표 / 01:04:10~끝 학생 B 발표. 두 구간 모두
   "발표자 vs 교수님" 2-state 문제로 환원된다.
 - 각 세그먼트에 문체 증거(log-odds)를 주고, 발화 턴이 잘 안 바뀐다는 사전지식을
   전이 페널티로 넣어 Viterbi 로 전역 최적 라벨열을 구한다. 페널티는 앞 세그먼트와의
   침묵 간격이 클수록 싸진다(간격이 크면 화자가 바뀐 것).
"""
import json, re
import numpy as np

S = json.load(open('meetings/raw_segments/260908_135919_segments.json'))['segments']
BOUNDARY = 64 * 60 + 10.0          # 01:04:10 — "저는 이제 비디오 받아서 카메라 생성하는 거"

# ---- 문체 증거 (+ = 교수님, - = 발표자) -------------------------------------
PROF = [
    (3.0, r'\?\s*$'),                                    # 질문으로 끝남
    (2.0, r'(뭐에요|뭐예요|뭔가요|어떻게|왜 |왜요|무슨|어디서|어느)'),
    (2.5, r'(아니에요|아닌가요|맞아요\?|맞죠|그러면\?|그런거에요|건가요)'),
    (2.5, r'(해야 (돼|되)|하세요|해주세요|써주세요|하면 (좋|되)|줄 알아야|해보세요)'),
    (2.0, r'(다음 슬라이드|다음은 뭐|넘어가|일단 넘어)'),
    (2.0, r'^(그 다음은|다음은)[^가-힣]*$'),   # 짧은 되묻기만
    (0.6, r'(잖아요|거 아니에요)'),          # 교수 쪽이 약간 많지만 발표자도 씀
    (1.2, r'(그러면\?|그거는\?|이거는\?)'),
    (2.5, r'[가-힣]씨(가|는|한|도|께서|랑|의)?\s'),        # 호칭("상혁씨") = 지도자가 학생을 부름
    (1.5, r'거든요'),
    (1.3, r'(에요|예요|아요|어요|워요|해요|돼요|네요|군요|봐요)\.?$'),  # 해요체
    (1.5, r'(본인이|중요한 스킬|페이퍼 라이팅|논문 쓸 때|프레젠테이션 스킬|알기 쉽)'),
]
PRES = [
    (3.0, r'(습니다|습니당)'),                            # 보고체
    (2.0, r'(했고요|하고요|였고요|이고요|인데요|거고요|합니다|입니다)'),
    (2.5, r'(말씀드|드렸|보고드|설명드)'),
    (2.5, r'^(네,? ?맞습니다|알겠습니다|네 맞|맞습니다|네\.?$|예\.?$)'),
    (1.2, r'(리포트|실험을|평가를|학습을|측정을|진행을|추가를|비교를|확인을)'),
    (1.0, r'^(그래서|그리고|그 다음에|여기서는|이거는|이건 이제|일단은|다음은)'),
    (1.0, r'(보시면|보시는|나왔습니다|나왔고|했었습니다|하고 있습니다|해봤)'),
]
PRIOR = 0.15   # 발표자 쪽 사전 편향. 세그먼트마다 누적되므로 작게 — 크면 파편이 긴 교수 독백을 발표자로 삼킨다
def evidence(t):
    e = 0.0
    for w, p in PROF:
        if re.search(p, t): e += w
    for w, p in PRES:
        if re.search(p, t): e -= w
    return e - PRIOR

# ---- Viterbi ---------------------------------------------------------------
def viterbi(idx):
    n = len(idx)
    # state 0 = 발표자, 1 = 교수님
    d = np.full((n, 2), -1e18); bk = np.zeros((n, 2), int)
    e0 = evidence(S[idx[0]]['text'])
    d[0] = [-e0, e0]
    for k in range(1, n):
        i, j = idx[k], idx[k-1]
        gap = S[i]['start'] - S[j]['end']
        if S[j]['text'].rstrip().endswith('?'):
            lam = 0.15            # 질문 직후 = 답변자로 넘어감
        else:
            lam = 1.4 if gap < 0.3 else (0.9 if gap < 2.0 else 0.35)  # 간격 크면 전환 싸다
        e = evidence(S[i]['text'])
        em = np.array([-e, e])
        for s in range(2):
            cand = d[k-1] - lam * (np.arange(2) != s)
            bk[k, s] = int(np.argmax(cand)); d[k, s] = cand[bk[k, s]] + em[s]
    out = np.zeros(n, int); out[-1] = int(np.argmax(d[-1]))
    for k in range(n-1, 0, -1): out[k-1] = bk[k, out[k]]
    return out

iA = [i for i, s in enumerate(S) if s['start'] < BOUNDARY]
iB = [i for i, s in enumerate(S) if s['start'] >= BOUNDARY]
lab = np.empty(len(S), object)
for idx, pres in ((iA, '학생 A'), (iB, '학생 B')):
    st = viterbi(idx)
    for k, i in enumerate(idx): lab[i] = '교수님' if st[k] else pres

# 확신도: 증거가 0 이거나 라벨과 반대면 낮음
conf = np.array([evidence(s['text']) for s in S])
low  = np.array([(abs(conf[i]) < 0.8) or
                 ((conf[i] > 0) != (lab[i] == '교수님') and abs(conf[i]) >= 0.8)
                 for i in range(len(S))])
# 한 세그먼트 안에 질문+보고체가 섞인 경우 = whisper 가 두 화자를 합친 것
mixed = np.array([bool(re.search(r'\?', s['text'][:-1]) and re.search(r'습니다', s['text']))
                  for s in S])

np.save('tmp/whisper/labA.npy', lab)
np.save('tmp/whisper/lowA.npy', low)
np.save('tmp/whisper/mixedA.npy', mixed)
dur = np.array([s['end'] - s['start'] for s in S])
print('boundary idx', iB[0], S[iB[0]]['text'][:40])
for nm in ('교수님', '학생 A', '학생 B'):
    m = lab == nm
    print(f'  {nm:6} {dur[m].sum()/60:5.1f}분  {m.sum():4d} 세그먼트')
print(f'  저확신 {low.sum()}  혼합세그먼트 {mixed.sum()}')
