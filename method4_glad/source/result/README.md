# method4_glad — result

## 파일
- `eval_results_step600.csv` — 학습 중간 체크포인트(step 600) 평가 결과.
- `eval_results_checkpoint20000.csv` — 최종 체크포인트(step 20000, `MAX_TRAIN_STEP` 완주) 평가 결과.
- 컬럼: `clsname,i_auroc,i_ap,i_f1,p_auroc,p_ap,p_f1,pro` (method5_dinomaly와 동일 포맷으로 통일).
- 두 파일 모두 **데스크탑(RTX 5070/12GB)**의 `bs2_grad16_eps_anomaly2_multiclass`(`train_batch_size=2, gradient_accumulation_steps=16`, 실질 batch 32) 실행에서 나온 같은 학습의 두 시점 — 노트북의 별도 시도(`bs1_grad4...`, batch=4)와는 무관.

## PatchCore 대비 H3/H4 판단
| 체크포인트 | grid I-AUROC (H3) | transistor P-AUROC (H4) |
|---|---|---|
| GLAD step600 | 0.955 | 0.786 |
| **GLAD checkpoint20000(최종)** | **0.996** | **0.710** |
| PatchCore 기준 | 0.977 | 0.929 |

- **H3** (grid, global pattern-regularity): **지지**. checkpoint20000에서 PatchCore(0.977)를 넘어섬(0.996). step600→checkpoint20000 사이 계속 개선되는 단조 추세.
- **H4** (transistor, spatial-arrangement): **반박**. PatchCore(0.929)에 크게 못 미침(0.710). 게다가 step600(0.786)→checkpoint20000(0.710)로 **학습이 진행될수록 오히려 나빠지는 비단조 추이** — 원인 미확인.

## 원 논문과의 비교 (multi-class, MVTec-AD, Table S7 "GLAD-256")
논문의 headline 결과(Table 1)는 카테고리별 별도 모델(single-class)이라 직접 비교가 안 되지만, supplementary Table S7에 같은 protocol(15개 카테고리 공동학습, resolution 256, `max_train_steps=20000`, batch=32)의 수치가 따로 있음. **이 재현(데스크탑, `train_batch_size=2 × gradient_accumulation_steps=16` = 실질 batch 32)은 batch를 포함해 논문/공식 스크립트(`train_multi.sh`)와 lr(5e-6)·denoise_step(500)·resolution(256)·mixed_precision(fp16)·optimizer(8-bit Adam)·steps(20000)가 모두 동일함.**

| | I-AUROC | I-AP | I-F1max | P-AUROC | P-AP | P-F1max | PRO |
|---|---|---|---|---|---|---|---|
| GLAD-256 논문 보고치(Table S7, batch=32) | 97.5 | 99.1 | 96.6 | 97.4 | 60.8 | 60.7 | 93.0 |
| 우리 재현(checkpoint20000, batch=32, Mean) | 85.4 | 91.6 | 90.5 | 91.9 | 37.3 | 41.2 | 80.4 |
| 격차 | -12.1 | -7.5 | -6.1 | -5.5 | **-23.5** | **-19.5** | -12.6 |

- **주요 하이퍼파라미터가 논문과 동일한데도 모든 지표에서 격차가 큼**, 특히 pixel-level AP/F1이 20%p 안팎으로 두드러짐. batch 축소 같은 뻔한 설명이 배제된 상태라 원인 미확인 — 추가 조사 필요.
- (확인됨) 노트북·데스크탑 모두 Windows 호환을 위해 원저자 pin에서 교체한 동일 라이브러리 버전(`bitsandbytes==0.49.2`, `diffusers==0.20.0`, `transformers==4.30.2`, `huggingface-hub==0.16.4`, `accelerate==0.19.0` 등, `setup_notes.md` 1절)을 씀 — 즉 "노트북과 데스크탑이 서로 다른 버전을 써서 격차가 생겼다"는 가능성은 배제됨. 다만 이 버전들 자체가 원저자 pin(`bitsandbytes==0.37.2`, `diffusers==0.20.0.dev0`)과는 다르므로, "논문 저자 환경과 우리 환경의 라이브러리 차이"는 두 재현(노트북/데스크탑) 모두에 공통으로 걸리는 원인 후보로 남아있음 — 검증 안 됨.
- DiAD 재현 때도 논문 대비 큰 격차(97.2 vs 87.0, `method3_diad/source/result/README.md` 참고)가 있었는데, 그때는 batch/epoch 축소로 설명 가능했던 것과 달리 이번엔 설정을 맞췄는데도 격차가 있음.

## setting
- multi-class(15개 카테고리 공동학습, `main_multi.py`) — PatchCore(class-separated)와는 setting이 다름 (위 비교는 참고용).
- GLAD 논문의 headline 결과(Table 1, grid 100/99.8·transistor 100/99.4)는 **single-class** 기준이라 이 재현치와 직접 비교 불가. multi-class 기준(Table S7, "GLAD-256")은 위에서 비교함.
- 이 결과(`eval_results_step600.csv`, `eval_results_checkpoint20000.csv`)는 **데스크탑(RTX 5070/12GB)**에서 `train_batch_size=2, gradient_accumulation_steps=16`(실질 batch 32, 논문과 동일)로 학습한 `bs2_grad16_eps_anomaly2_multiclass` 실행 결과. `max_train_steps=20000`, resolution=256, mixed_precision=fp16, 8-bit Adam, gradient_checkpointing 사용 — 공식 스크립트와 batch를 포함해 거의 동일한 설정. 노트북에서 8GB VRAM 제약으로 시도한 별도의 batch=4 축소 버전(`method4_glad/source/config/setup_notes.md`에 기록)은 이 결과와 무관.

## 결론
- `max_train_steps=20000` 완주로 GLAD 재현 종료. batch를 포함한 주요 설정이 논문과 동일(데스크탑 12GB에서 batch=32로 재현).
- **H3(grid)는 지지**로 판단 — PatchCore를 넘어섰고(0.996>0.977) 추이도 단조 개선이라 신뢰도 높음.
- **H4(transistor)는 반박** — PatchCore(0.929)에 크게 못 미치고(0.710), step600(0.786)→checkpoint20000(0.710)로 **학습이 진행될수록 오히려 나빠지는 비단조 추이**를 보임(같은 batch=32 런의 두 시점이라 이 추이 자체는 확실함). 설정이 논문과 동일하므로 "학습 예산 부족" 같은 손쉬운 설명은 배제됨 — 원인은 불명확하지만, 반박 판단 자체의 신뢰도는 batch 문제가 해소되기 전보다 높아짐.
- 다만 위 "원 논문과의 비교"에서 보듯 이 재현 자체가 논문 대비 전 지표에서 큰 격차를 보이므로, H3/H4 판단 모두 "GLAD라는 방법의 실제 성능"이 아니라 "이 특정 재현 결과물"에 대한 판단이라는 점은 유의해야 함.
- 다음 단계로는 (1) transistor 비단조 악화 및 논문 대비 전반적 격차의 원인 조사, (2) 같은 H3/H4 프레임으로 진행 중인 Dinomaly(method5_dinomaly) 결과와 대조해 재구성 기반 방법과 생성 기반(diffusion) 방법 간 판단이 수렴하는지 확인이 필요.
