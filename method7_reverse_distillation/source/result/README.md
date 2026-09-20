# method7_reverse_distillation — result

Reverse Distillation(CVPR 2022, 공식 코드 저장소 이름은 RD4AD)

## 파일
- `eval_results_sep.csv` — class-separated(카테고리별 개별 모델, 공식 repo 그대로), batch_size=16·epoch=200(원 논문 설정), 15개 카테고리 전부 완주한 최종 결과.
- `train_console.log` — 학습 stdout(python 버퍼링으로 실시간 반영은 안 되고 프로세스 종료 시점에 flush됨).

## 결과 요약(H3/H4 관련 카테고리)
| 조건 | grid I-AUROC | transistor P-AUROC |
|---|---|---|
| **Reverse Distillation(sep, 최종)** | **1.000** | **0.927** |
| PatchCore 기준 | 0.977 | 0.929 |
| Dinomaly(sep) 참고 | 1.000 | 0.9493 |

## 가설 판단
- **H3** (grid, global pattern-regularity): **지지**. grid I-AUROC 1.000으로 PatchCore(0.977)를 앞섬 — DiAD/GLAD/Dinomaly에 이어 네 번째 재구성 계열 방법에서도 반복.
- **H4** (transistor, spatial-arrangement): **미결정에 가까움**. transistor P-AUROC 0.927로 PatchCore(0.929)와 거의 동일하나 근소하게 낮음(-0.002). Dinomaly(sep)에서는 명확히 앞섰던 것과 달리 Reverse Distillation에서는 그 정도로 뚜렷하지 않아, "재구성 기반 접근이 H4에 특히 강하다"는 가설에 대한 근거가 방법마다 갈리는 모습.

> ※ 정정(2026-09-20): 위 PatchCore 0.929는 anomaly-only 정의이고 RD4AD의 pixel_auroc는 full-pixel 정의다(`method7_reverse_distillation/source/code/RD4AD/test.py`). 같은 full-pixel 기준 PatchCore는 **0.963**이라 차이는 -0.002가 아니라 **-0.036**이며 "거의 동일"이 아니라 뚜렷하게 낮다. 자세한 비교는 `meetings/2026-W39_brief.md` 5절 실험 3.

## 15개 카테고리 전체 결과
| category | pixel_auroc | image_auroc | pixel_aupro |
|---|---|---|---|
| carpet | 0.990 | 0.990 | 0.970 |
| bottle | 0.987 | 0.999 | 0.966 |
| hazelnut | 0.989 | 1.000 | 0.955 |
| leather | 0.994 | 1.000 | 0.991 |
| cable | 0.973 | 0.974 | 0.910 |
| capsule | 0.987 | 0.980 | 0.958 |
| grid | 0.992 | 1.000 | 0.971 |
| pill | 0.982 | 0.966 | 0.965 |
| transistor | 0.927 | 0.966 | 0.783 |
| metal_nut | 0.973 | 1.000 | 0.925 |
| screw | 0.996 | 0.975 | 0.982 |
| toothbrush | 0.990 | 0.994 | 0.941 |
| zipper | 0.982 | 0.979 | 0.955 |
| tile | 0.956 | 0.995 | 0.907 |
| wood | 0.953 | 0.993 | 0.909 |
| **Mean** | **0.978** | **0.987** | **0.939** |

- 원 논문 보고치(mean image AUROC 98.5%, pixel AUROC 97.8%, PRO 93.9%)와 거의 일치 — 재현 신뢰도 확인됨.
- transistor의 pixel_aupro(0.783)가 다른 카테고리 대비 뚜렷하게 낮음. H4(spatial-arrangement 이상)를 픽셀 단위 영역 겹침으로 보면 Reverse Distillation이 특히 약한 카테고리라는 뜻이라, image-level(P-AUROC 0.966)과 pixel-level(P-AUROC/AUPRO) 판단이 엇갈리는 지점.

## setting
- Reverse Distillation 공식 코드/논문 모두 **class-separated(카테고리별 개별 모델) 방식만 존재**, multi-class 옵션 없음. GLAD/Dinomaly(uni, multi-class)와 달리 별도 protocol 변경 없이 **PatchCore(class-separated)와 동일한 setting**으로 바로 비교 가능.
- 하이퍼파라미터는 원 논문 그대로 유지(batch=16, epoch=200, lr=0.005, Adam(0.5,0.999), encoder=wide_resnet50_2, image=256). 로컬 환경 대응은 경로 인자화(`--data_path`/`--ckpt_dir`)와 `np.bool`/`DataFrame.append` 등 최신 numpy/pandas 비호환 API 수정뿐, 원 하이퍼파라미터는 손대지 않음.
- 학습 중 WinCLIP과 GPU를 동시 점유하는 문제가 있어 한 차례 중단 후 WinCLIP 완료 뒤 처음부터 재시작(카테고리 완료 여부를 CSV로 추적해 재개 가능하도록 만들어뒀으나, 이번엔 중단 시점에 완료된 카테고리가 없어 실질적으로는 처음부터 재학습).
