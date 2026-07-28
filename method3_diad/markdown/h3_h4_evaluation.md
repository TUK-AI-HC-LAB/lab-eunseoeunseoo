# H3/H4 검증 — DiAD epoch 7 체크포인트 평가

## 근거 경로
- commit: `ca7dca8` (test.py 버그 수정 시점 — 이 커밋의 코드로 평가 실행)
- sh: `method3_diad/source/run_eval_epoch7.sh`
- code: `method3_diad/source/DiAD/test.py`
- checkpoint: `C:/ai_local/diad_val_ckpt/step_step=3600.ckpt` (epoch 7 도중, global_step 3600 — `.ckpt`는 gitignore 대상이라 repo에는 없음)
- command log: `C:/Users/kelly/AppData/Local/Temp/diad_test_run2.log` (로컬 임시 경로, repo 밖 — *.log는 gitignore 대상)
- raw result: `method3_diad/source/eval_results_epoch7.csv`

## 실험 1: H4 — transistor pixel-level 이상탐지

### 질문
PatchCore가 transistor에서 놓친 spatial-arrangement 이상(부품 위치 이동/누락)을, 전체 이미지를 재구성하는 DiAD가 더 정확한 pixel-level anomaly map으로 잡아낼 수 있는가.

### 가설
H4: PatchCore의 per-patch independent scoring은 spatial-arrangement 이상을 못 잡지만, 전체 이미지 재구성 기반 방법(DiAD)은 정확한 위치에서 재구성이 실패하므로 transistor pixel-level AUROC(P-AUROC)가 PatchCore보다 높아야 한다.

### 설정
- Baseline: PatchCore, transistor P-AUROC = 0.929 (`method1_patchcore/markdown/baseline_analysis.md` L91)
- Method: DiAD, epoch 7 체크포인트(`step_step=3600.ckpt`), MVTec-AD 전체 15개 카테고리 공동학습(multi-class) 모델
- Metric: pixel-level AUROC (`eval_results_epoch7.csv`의 `pixel` 컬럼)

### 기대 결과
H4가 맞다면 DiAD transistor P-AUROC > 0.929.

### 실제 결과
| 지표 | 기대 | 실제 | 해석 | Raw Path |
|---|---|---|---|---|
| transistor P-AUROC | > 0.929 | **0.944843** | 기준 대비 +1.6%p | `method3_diad/source/eval_results_epoch7.csv` |

### 해석
**지지.** 논문 기준(1,000 epoch, batch 12) 대비 극히 일부만 학습한 모델(로컬 8GB GPU, batch 2, epoch 7)임에도 이미 PatchCore baseline을 넘었다. 전체 15개 카테고리 평균 mean I-AUROC가 0.803으로 아직 PatchCore 평균(0.991)에 크게 못 미치는 미성숙한 모델이라는 점을 고려하면, transistor pixel-level 국소화 성능만큼은 구조적으로 PatchCore보다 유리하다는 H4의 핵심 주장과 부합한다. 학습이 더 진행되면 격차가 더 벌어질 가능성이 높다.

## 실험 2: H3 — grid image-level 이상탐지

### 질문
PatchCore의 고정된 3×3 local aggregation이 놓치는 grid의 global periodicity 이상을, 전체 이미지 재구성 기반 DiAD가 더 잘 잡아낼 수 있는가.

### 가설
H3: DiAD의 grid image-level AUROC(I-AUROC)가 PatchCore보다 높아야 한다.

### 설정
- Baseline: PatchCore, grid I-AUROC = 0.977 (`method1_patchcore/markdown/baseline_analysis.md` L84)
- Method/Metric: 위와 동일, `max` 컬럼(image-level score) 사용

### 기대 결과
H3가 맞다면 DiAD grid I-AUROC > 0.977.

### 실제 결과
| 지표 | 기대 | 실제 | 해석 | Raw Path |
|---|---|---|---|---|
| grid I-AUROC | > 0.977 | **0.588137** | 기준 대비 -38.9%p | `method3_diad/source/eval_results_epoch7.csv` |

### 해석
**미결정 (아직 반박도 지지도 아님).** grid는 15개 카테고리 중 가장 낮은 축에 속하고, 전체 평균(0.803)보다도 낮다. 논문의 1,000 epoch 대비 7 epoch만 학습된 상태라 "H3가 틀렸다"와 "아직 grid의 global periodicity를 재구성으로 학습할 만큼 훈련이 안 됐다"를 구분할 수 없다. H4가 이 시점에 이미 baseline을 넘은 것과 비교하면, grid 쪽이 상대적으로 더 많은 학습을 필요로 하는 케이스일 가능성이 있다 — 다음 평가에서 grid I-AUROC의 추이(증가하는지, 계속 낮은 채인지)를 지켜봐야 판단 가능.

## 다음 판단
- H4는 이 시점 evidence로 지지 결론을 내린다.
- H3는 결론을 유보하고, 학습을 계속 진행하며 grid I-AUROC가 epoch에 따라 개선되는지 주기적으로 재평가한다.
