# 미팅 오디오 → 화자별 스크립트 재현 절차

산출물: `meetings/raw_segments/` (화자 미분류 전사), `meetings/scripts/` (화자 분류 2종 + 비교).

| 단계 | 스크립트 | 환경 | 비고 |
|---|---|---|---|
| 0 | (ffmpeg) `ffmpeg -i <m4a> -ac 1 -ar 16000 -c:a pcm_s16le meeting.wav` | `infcam` 에만 ffmpeg 있음 | 16kHz mono |
| 1 | `01_transcribe_whisper.py` | `vista4d` | whisper-large-v3 **네이티브 long-form**. `chunk_length_s` 를 주면 30초 배치 경계로 뭉쳐져 diarization 이 불가능해진다 (실측: 41개 세그먼트가 76.8분을 차지) |
| 2 | `02_speaker_embed_pyannote.py` | `diar` | `wespeaker-voxceleb-resnet34-LM` 256-d. **세그먼트를 배치로 zero-pad 하면 안 된다** — 패딩이 임베딩을 지배해 한 클러스터가 88% 를 먹는다. 개별 forward 필수 |
| 3 | `03_classify_by_content.py` | `vista4d` | 오디오 미사용. 합니다체(발표자) ↔ 해요체(교수님) 대립 + Viterbi |

`diar` 환경: `/data1/cympyc1785/miniconda3/envs/diar` (pyannote.audio 4.0.7, torch/torchaudio 2.8.0+cu128).
`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 로 익명 접근 (embedding 모델은 open, `speaker-diarization-3.1` 은 gated).
