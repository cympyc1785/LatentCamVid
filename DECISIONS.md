
## D61 (2026-10-23) main 모델 = v5f_full · GT plan · Molmo2 · 3D box · depth 영상
- 사용자: "우리 main 모델은 우선 GT plan, molmo2 feature, 3d bounding box, depth token (ESDF 말고 그냥 depth) 로 할거고 이후 planner 개선되면 GT plan 만 planner 의 plan 으로 교체".
- 데이터 = v5f_full (= latentcam_dynpose_v5f_all, noroll train 54,950 / test 6,111), TRUMANS 는 시간이 걸려 이번엔 뺀다.
- 인자 = R489 (v5f_all MTb v2) + depth_vid (R482/R511). 학습 R524 (train71).
- 비교 대상 = CCD · Director3D · E.T. · GenDoP (rgbd) release ckpt, raw 크기, v5f_all test 6,111. 채점 CLaTr v5f_all v2 ckpt.
