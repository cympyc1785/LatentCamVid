"""2차: Whisper 네이티브 long-form (chunk_length_s 없음) — 문장 단위 세그먼테이션 확보.

1차(chunked, batch)는 291초로 빨랐지만 30s+ 세그먼트 41개가 전체의 85%를 담아
화자 배정이 불가능했다. 여기서는 느리더라도 Whisper 자체 알고리즘을 쓴다.
"""
import json, os, time, wave
import numpy as np
import torch
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from transformers import pipeline

WAV, OUT = "tmp/whisper/meeting.wav", "tmp/whisper/segments.json"
t0 = time.time()
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3",
               dtype=torch.float16, device="cuda:0", token=False)
with wave.open(WAV, 'rb') as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
audio = pcm.astype(np.float32) / 32768.0
print(f"[load] {time.time()-t0:.0f}s  audio {len(audio)/16000:.1f}s", flush=True)

t1 = time.time()
res = asr({"raw": audio, "sampling_rate": 16000}, return_timestamps=True,
          generate_kwargs={"language": "korean", "task": "transcribe",
                           "condition_on_prev_tokens": False,   # 반복 루프 억제
                           "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                           "logprob_threshold": -1.0,
                           "compression_ratio_threshold": 1.35,
                           "no_speech_threshold": 0.6})
print(f"[asr] {time.time()-t1:.0f}s  chunks={len(res.get('chunks', []))}", flush=True)

segs = [{"start": (c.get("timestamp") or (None, None))[0],
         "end": (c.get("timestamp") or (None, None))[1],
         "text": (c.get("text") or "").strip()} for c in res.get("chunks", [])]
json.dump({"model": "openai/whisper-large-v3", "mode": "native long-form",
           "text": res.get("text", ""), "segments": segs},
          open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"[done] {len(segs)} segments -> {OUT}  total {time.time()-t0:.0f}s", flush=True)
