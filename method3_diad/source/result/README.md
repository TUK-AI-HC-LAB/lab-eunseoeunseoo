# method3_diad — result

## 파일
- `epoch_log.csv` — 학습 중 epoch별 loss 로그(`GitCommitOnEpochEnd` 콜백으로 자동 기록).
- `eval_results_epoch7.csv`, `eval_results_epoch16.csv`, `eval_results_epoch34.csv` — 각 체크포인트에서의 MVTec-AD 15개 카테고리 평가 결과. 컬럼명이 파일마다 순서가 다르니 주의 — `max`=I-AUROC(image-level), `pixel`=P-AUROC(pixel-level, full) 기준으로 비교했음.

## 재현이 이상했던 진짜 이유: DiAD 원 논문과의 격차
`method3_diad/paper/AAAI24_DiAD_...pdf`의 Table 1(image-level)·Table 3(pixel-level)에 있는 저자 보고치와 비교하면, 재현치가 학습을 더 시켜도 논문 수준을 못 따라감:

| | Mean I-AUROC | grid I-AUROC | transistor I-AUROC | Mean P-AUROC | grid P-AUROC | transistor P-AUROC |
|---|---|---|---|---|---|---|
| **DiAD 논문(Table 1/3)** | 97.2 | 98.5 | 99.8 | 96.8 | 96.6 | 95.1 |
| 재현 epoch7 | 80.3 | 58.8 | 87.2 | 87.8 | 63.8 | 94.5 |
| 재현 epoch16 | 84.3 | 76.4 | 92.1 | 89.9 | 64.9 | 92.3 |
| 재현 epoch34 | 87.0 | 65.4 | 95.8 | 90.1 | 63.8 | 92.2 |

**15개 카테고리 전체를 논문과 대조하면(epoch34 기준, I-AUROC 격차 큰 순)**:

| 카테고리 | 논문 I-AUROC | 재현 | 격차 | 논문 P-AUROC | 재현 | 격차 |
|---|---|---|---|---|---|---|
| grid | 98.5 | 65.4 | **33.1** | 96.6 | 63.8 | **32.8** |
| wood | 99.7 | 71.4 | **28.3** | 93.3 | 74.1 | 19.2 |
| pill | 95.7 | 74.6 | 21.1 | 95.7 | 92.1 | 3.6 |
| screw | 90.7 | 72.5 | 18.2 | 97.9 | 87.5 | 10.4 |
| cable | 94.8 | 80.4 | 14.4 | 96.8 | 92.8 | 4.0 |
| capsule | 89.0 | 81.3 | 7.7 | 97.1 | 95.2 | 1.9 |
| toothbrush | 99.7 | 93.3 | 6.4 | 99.0 | 98.2 | 0.8 |
| hazelnut | 99.5 | 93.9 | 5.6 | 98.3 | 96.6 | 1.7 |
| transistor | 99.8 | 95.8 | 4.0 | 95.1 | 92.2 | 2.9 |
| leather | 99.8 | 96.2 | 3.6 | 98.8 | 91.4 | 7.4 |
| carpet | 99.4 | 95.9 | 3.5 | 98.6 | 96.4 | 2.2 |
| metal_nut | 99.1 | 96.1 | 3.0 | 97.3 | 95.2 | 2.1 |
| tile | 96.8 | 95.1 | 1.7 | 92.4 | 86.3 | 6.1 |
| zipper | 95.1 | 93.8 | 1.3 | 96.2 | 92.6 | 3.6 |
| bottle | 99.7 | 99.9 | -0.2 | 98.4 | 97.9 | 0.5 |

- **"grid만 이상하다"는 부정확함** — **grid와 wood 둘 다** I-AUROC 기준 28~33%p라는 압도적인 격차를 보이고(다음으로 큰 pill은 21.1%p로 한 단계 아래), 그 뒤로 pill/screw/cable이 중간 정도(14~21%p) 격차, 나머지 9개 카테고리는 대부분 논문과 1~8%p 이내로 비교적 근접함.
- P-AUROC 기준으로는 grid(32.8%p)가 wood(19.2%p)보다도 확실히 큰 outlier — pixel-level에서는 grid가 특히 심함.
- grid·wood 둘 다 **텍스처 카테고리**인데, 같은 텍스처인 carpet(3.5%p)·leather(3.6%p)·tile(1.7%p)은 정상 범위라 "텍스처라서 다 어렵다"는 설명도 성립하지 않음 — grid·wood 두 카테고리에 특정적인 문제(데이터 전처리, 반복 패턴 특성 등)일 가능성이 있으나 원인은 미확인.
- **Mean**: epoch34까지 가도 논문보다 10%p 이상 낮음 — grid·wood·pill·screw·cable 등 격차가 큰 카테고리들이 평균을 크게 끌어내리고 있음.
- 이 격차(특히 grid·wood)가 "DiAD 방식 자체의 한계"인지 "우리 쪽 재현 설정/버그" 때문인지 epoch 34 시점까지도 구분이 안 됐고, 이게 PatchCore와의 H3/H4 비교보다 먼저 확인됐어야 할 더 근본적인 신뢰성 문제였음.

## PatchCore 대비 H3/H4 판단(위 문제를 감안하고 참고용으로)
| 체크포인트 | grid I-AUROC | transistor P-AUROC |
|---|---|---|
| epoch 7 | 0.588 | 0.945 |
| epoch 16 | 0.764 | 0.923 |
| epoch 34 | 0.654 | 0.922 |
| PatchCore 기준 | 0.977 | 0.929 |

- **H3** (grid, global pattern-regularity): **미결정, 안 좋아지는 쪽**. PatchCore(0.977)에 세 지점 모두 못 미치고 비단조적.
- **H4** (transistor, spatial-arrangement): **미결정, 반박에 가까움**. epoch7만 PatchCore를 넘었고(0.945>0.929) 이후 계속 하락해 PatchCore 근처로 수렴 — 우연히 초반에 높았을 가능성. 다만 transistor는 위 DiAD 논문 대비 격차가 grid만큼 크지 않아(epoch7 P-AUROC 94.5 vs 논문 95.1), grid와 달리 재현 신뢰성 문제로 보이지는 않음 — H4가 반박에 가깝다는 판단 자체는 유지 가능.

> ※ 정정(2026-09-20): 위 PatchCore 0.929는 `anomaly_pixel_auroc`(이상 이미지만 모아 계산)이고, DiAD의 pixel AUROC는 전체 이미지의 모든 픽셀로 계산하는 full-pixel 정의다(`method3_diad/source/code/DiAD/utils/eval_helper.py`). 같은 full-pixel 기준 PatchCore는 **0.963**이라 epoch7의 0.945도 넘지 못한다. "반박에 가까움" 판단은 유지되고 격차만 커진다. 자세한 비교는 `meetings/2026-W39_brief.md` 5절 실험 3.

## 다음 판단
- **2026-09-07 교수님 지시로 재현 중단**. grid·wood 등 DiAD 논문과의 격차가 30%p 안팎으로 너무 커서 이 재현치로 H3/H4를 계속 판단할 실질적 이유가 없다고 보고 중단, GLAD/SimpleNet/Reverse Distillation/Dinomaly로 후보 전환(경위는 `meetings/2026-W37_brief.md` 참고).

## setting
multi-class(15개 카테고리 동시 학습, DiAD/GLAD와 동일 protocol) — PatchCore(class-separated)와 setting이 다름.
