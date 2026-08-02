# H3/H4 검증 — DiAD 체크포인트 평가 (epoch 7 → epoch 16)

## 근거 경로
### epoch 7
- commit: `ca7dca8` (test.py 버그 수정 시점 — 이 커밋의 코드로 평가 실행)
- sh: `method3_diad/source/run_eval_epoch7.sh`
- checkpoint: `C:/ai_local/diad_val_ckpt/step_step=3600.ckpt` (epoch 7 도중, global_step 3600 — `.ckpt`는 gitignore 대상이라 repo에는 없음)
- command log: `C:/Users/kelly/AppData/Local/Temp/diad_test_run2.log` (로컬 임시 경로, repo 밖 — *.log는 gitignore 대상)
- raw result: `method3_diad/source/eval_results_epoch7.csv`

### epoch 16 (재평가)
- commit: `f397a5d` (W31 브리핑 커밋 시점 — 코드 변경 없이 체크포인트만 진행된 상태)
- sh: `method3_diad/source/run_eval_epoch16.sh`
- checkpoint: `C:/ai_local/diad_val_ckpt/step_step=7400.ckpt` (epoch 16 도중, global_step 7400)
- command log: `C:/Users/kelly/AppData/Local/Temp/diad_test_run3.log` (로컬 임시 경로, repo 밖)
- raw result: `method3_diad/source/eval_results_epoch16.csv`

- code: `method3_diad/source/DiAD/test.py` (두 평가 공통)

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
| 지표 | 기대 | epoch 7 | epoch 16 | 해석 | Raw Path |
|---|---|---|---|---|---|
| transistor P-AUROC | > 0.929 | 0.944843 (+1.6%p) | **0.923219** (-0.6%p) | epoch 7에서는 기준 상회, epoch 16에서는 기준 하회로 역전 | `eval_results_epoch7.csv`, `eval_results_epoch16.csv` |

### 해석
**epoch 7 지지 → epoch 16 재역전, 결론을 지지에서 미결정으로 하향 조정.** epoch 7에서는 PatchCore를 넘었지만 epoch 16에서는 다시 기준 아래로 내려왔다(0.945→0.923). 같은 기간 train loss도 0.108(epoch10 최저)→0.118→0.123→0.114로 단조 감소가 아니라 진동하는 패턴을 보이므로, transistor P-AUROC의 하락도 학습이 아직 불안정한 구간에 있다는 노이즈일 가능성이 있다. 다만 이 시점에서 "H4 지지"라고 단정하기는 이르다 — 최소 한 번 더 재평가해서 방향(다시 상회하는지, 계속 기준 근처에서 진동하는지)을 확인해야 한다.

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
| 지표 | 기대 | epoch 7 | epoch 16 | 해석 | Raw Path |
|---|---|---|---|---|---|
| grid I-AUROC | > 0.977 | 0.588137 (-38.9%p) | **0.763576** (-21.9%p) | 여전히 기준 미달이나 격차가 17%p 줄어듦 | `eval_results_epoch7.csv`, `eval_results_epoch16.csv` |

### 해석
**미결정, 다만 개선 추세 확인.** grid I-AUROC가 epoch 7→16 사이 +0.176 상승해 여전히 PatchCore(0.977)에는 못 미치지만 격차가 뚜렷하게 줄었다. 이는 "H3가 틀렸다"보다는 "학습이 더 필요하다"는 설명과 일치하는 방향이다. 다만 두 시점만으로는 이 추세가 계속될지, 어느 지점에서 정체될지 알 수 없다.

## 다음 판단
- H4는 epoch 7 지지 → epoch 16 미결정으로 하향 조정. 다음 재평가에서 방향을 다시 확인해야 최종 판단 가능.
- H3는 여전히 미결정이나, grid I-AUROC가 개선되는 추세가 확인되어 낙관적인 미결정으로 갱신. 학습을 계속 진행하며 다음 체크포인트에서 재평가.
- 두 가설 모두 한두 시점만으로는 최종 결론을 내리기 부족하다는 것 자체가 이번 재평가의 핵심 교훈 — 다음 브리핑에서는 최소 3개 시점(epoch 7/16/X)의 추이를 함께 보고할 것.
