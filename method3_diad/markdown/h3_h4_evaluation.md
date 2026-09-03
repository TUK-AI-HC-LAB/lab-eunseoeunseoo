# H3/H4 검증 — DiAD 체크포인트 평가 (epoch 7 → 16 → 34)

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

### epoch 24 — 재현 불가 (결과 없음)
- `run_eval_epoch24.sh`는 존재하지만(2026-08-05 생성), 대응하는 `eval_results_epoch24.csv`는 한 번도 생성/커밋되지 않았다.
- 정황상 이 시점에 `check_val_every_n_epoch=25` 설정으로 학습 프로세스 내장 validation이 epoch 24 종료 시점에 자동 발동했고(`setup_notes.md` 20절), 그 결과가 별도 로그 파일 없이 콘솔에만 찍히고 사라진 것으로 추정된다 — 확정적 증거는 없다.
- 체크포인트(`step_step=11350.ckpt`)도 `save_top_k=1` 설정으로 이후 진행되며 덮어써져 지금은 존재하지 않아 재실행으로도 재현 불가.
- 재현 불가능한 결과는 가이드 11절 원칙에 따라 evidence로 사용하지 않는다 — 아래 결과 비교에서 epoch 24는 제외.

### epoch 34 (재평가)
- commit: `ca7dca8` (test.py 코드 변경 없이 체크포인트만 진행된 상태 — epoch16과 동일)
- sh: `method3_diad/source/run_eval_epoch34.sh`
- checkpoint: `C:/ai_local/diad_val_ckpt/step_step=15650.ckpt` (epoch 34 도중, global_step 15650)
- command log: `C:/Users/kelly/AppData/Local/Temp/diad_test_run7.log` (로컬 임시 경로, repo 밖; 2026-08-19 00:59 실행)
- raw result: `method3_diad/source/eval_results_epoch34.csv`

- code: `method3_diad/source/DiAD/test.py` (세 평가 공통)

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
| 지표 | 기대 | epoch 7 | epoch 16 | epoch 34 | 해석 | Raw Path |
|---|---|---|---|---|---|---|
| transistor P-AUROC | > 0.929 | 0.944843 (+1.6%p) | 0.923219 (-0.6%p) | **0.921888** (-0.7%p) | 0.929 기준으로 상회/하회를 오가며 뚜렷한 방향성 없음 | `eval_results_epoch7.csv`, `eval_results_epoch16.csv`, `eval_results_epoch34.csv` |

### 해석
**3개 시점 모두 확인한 결과, 미결정 — 뚜렷한 추세 없이 기준선 근처에서 진동.** epoch 7(0.945)→16(0.923)→34(0.922)로, epoch 16 이후 34까지 18 epoch이 더 진행됐음에도 값이 거의 그대로다(0.923→0.922, 차이 0.001). 이는 "학습이 불안정한 구간이라 곧 다시 역전될 것"이라는 epoch16 시점의 가설과 맞지 않는다 — 오히려 PatchCore 기준(0.929) 바로 아래에서 정체된 것에 가깝다. 세 지점만으로 "H4가 반박됐다"고 확정할 수는 없지만, "epoch 7의 우위는 학습 초기의 우연한 결과였고 이후로는 PatchCore와 비슷한 수준에서 안정화됐다"는 해석이 지금까지의 데이터와 가장 잘 맞는다. epoch 24 데이터가 유실되어 epoch16→34 사이의 중간 추이는 확인할 수 없다는 한계가 있다.

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
| 지표 | 기대 | epoch 7 | epoch 16 | epoch 34 | 해석 | Raw Path |
|---|---|---|---|---|---|---|
| grid I-AUROC | > 0.977 | 0.588137 (-38.9%p) | 0.763576 (-21.9%p) | **0.654135** (-32.3%p) | epoch 16에서 개선됐던 것이 epoch 34에서 다시 하락 | `eval_results_epoch7.csv`, `eval_results_epoch16.csv`, `eval_results_epoch34.csv` |

### 해석
**미결정, 다만 개선 추세는 유지되지 않음(역전).** epoch 7→16 사이 +0.176 상승했던 grid I-AUROC가 epoch 34에서 0.654로 다시 하락했다(16 대비 -0.109). epoch 16 시점에 세웠던 "학습이 더 필요해서 개선되는 중"이라는 낙관적 해석은 이번 재평가로 지지되지 않는다. 다만 epoch 24 데이터가 유실되어 16→34 사이 18 epoch 동안 단조 하락이었는지, 24 부근에서 한 번 더 오르내렸는지는 확인할 수 없다 — 이 공백이 해석의 확실성을 낮춘다. train loss는 같은 기간(epoch16≈0.118 → epoch34=0.091, `epoch_log.csv`) 계속 낮아지는 추세인데도 grid I-AUROC는 오히려 떨어졌다는 점이, "단순히 학습을 더 하면 개선된다"는 가설에 대한 반증에 가깝다.

## 다음 판단
- H4는 세 시점(0.945→0.923→0.922) 모두 종합하면 PatchCore 기준(0.929) 근처에서 정체 — epoch 7의 우위가 우연이었을 가능성이 높아 "미결정, 반박에 가까움"으로 갱신.
- H3는 epoch 7→16 개선 후 epoch 34에서 재하락(0.588→0.764→0.654) — 단조 개선이 아니므로 "학습 부족" 가설의 근거가 약해졌다. train loss는 계속 낮아지는데 grid I-AUROC는 함께 오르지 않는다는 점에서, PatchCore 대비 열위가 학습량이 아니라 구조적 한계일 가능성을 시사한다. 여전히 확정적 반박은 아니며 "미결정, 비관적 쪽으로 이동"으로 정리.
- epoch 24 데이터 유실로 세 시점(7/16/34)만 확보됐고, 원래 계획했던 4번째 재평가는 이 시점에서는 추가 정보 가치가 낮다고 판단해 보류 — 데이터 정리·문서화를 우선한다.
- 이번 재평가의 핵심 교훈: 재현성 규칙(가이드 11절)을 지키지 않으면 evidence 자체가 사라진다 — epoch 24는 스크립트만 있고 raw csv가 없어 결과를 아예 인용할 수 없게 됐다. 앞으로는 평가 실행 즉시 csv를 커밋하는 것을 원칙으로 한다.
