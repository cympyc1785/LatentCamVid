"""세그먼트별 화자 임베딩 — 패딩 없이 개별 forward (배치 0-패딩이 임베딩을 오염시켰던 버그 수정)."""
import json, os, wave
import numpy as np, torch
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]="1"
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
SR, MINL, MAXL = 16000, 1.0, 10.0
S=json.load(open('meetings/raw_segments/260908_135919_segments.json'))['segments']
with wave.open('tmp/whisper/meeting.wav','rb') as w:
    audio=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768.
emb=PretrainedSpeakerEmbedding('pyannote/wespeaker-voxceleb-resnet34-LM',device=torch.device('cuda'))
E=np.full((len(S),emb.dimension),np.nan)
for i,s in enumerate(S):
    a,b=int(s['start']*SR),int(s['end']*SR)
    if (b-a)/SR<MINL: continue
    x=audio[a:b][:int(MAXL*SR)]                       # 자르기만, 패딩 없음
    E[i]=np.asarray(emb(torch.from_numpy(x[None,None,:].copy())))[0]
    if i%200==0: print(f"  {i}/{len(S)}",flush=True)
np.save('tmp/whisper/emb.npy',E)
ok=~np.isnan(E[:,0]); N=E[ok]/np.linalg.norm(E[ok],axis=1,keepdims=True)
C=N@N.T; iu=np.triu_indices(len(N),1)
print(f"\n임베딩 {ok.sum()}/{len(S)}  dim {emb.dimension}")
print(f"쌍별 코사인: p5 {np.percentile(C[iu],5):.3f}  median {np.median(C[iu]):.3f}  p95 {np.percentile(C[iu],95):.3f}")
print("-> 분포가 넓으면(예: p5<0.3, p95>0.8) 화자 분리 가능. 좁으면 임베딩이 화자를 못 가른다.")
