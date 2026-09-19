# method5_dinomaly — result

## 파일
- `eval_results_batch4_iter10000.csv` — multi-class(uni), batch_size=4(8GB 노트북 제약으로 축소), 10000 iter 완주 결과.
- `eval_results_batch16_iter10000.csv` — multi-class(uni), batch_size=16(원 논문 설정), 10000 iter 완주 결과. 원본 학습 로그: `train_log_batch16_iter10000.txt`.
- `eval_results_sep_batch16_iter5000.csv` — **class-separated(sep)**, batch_size=16, 카테고리당 5000 iter, 15개 카테고리 전부 완주한 최종 결과(`dinomaly_mvtec_sep.py`). 원본 학습 로그: `train_log_sep_batch16_iter5000.txt`.
- `batch16_inprogress_defaultfan.csv` — uni batch=16 학습 중 GPU 온도/utilization/VRAM 로그(팬 기본 설정, 항목4 진단용).
- `dinomaly_sep_b16_extremefan.csv` — sep 초기 시도(2026-09-16, 팬 Extreme 모드) 때 캡처된 GPU 로그. 이 파일이 커버하는 학습 자체는 이후 CUDA 크래시로 중단된 이전 시도라, 최종 결과(`eval_results_sep_batch16_iter5000.csv`)와는 별개 — 참고용으로만 남김.

## 결과 요약(H3/H4 관련 카테고리만)
| 조건 | grid I-AUROC | transistor P-AUROC |
|---|---|---|
| uni batch=4 | 0.9975 | 0.9238 |
| uni batch=16(원 논문 설정) | 0.9983 | 0.9335 |
| **sep batch=16(class-separated, 최종)** | **1.0000** | **0.9493** |
| PatchCore 기준 | 0.977 | 0.929 |

## 가설 판단
- **H3** (grid, global pattern-regularity): **지지**. uni(batch=4/16)·sep 세 조건 모두 PatchCore를 명확히 앞섬. sep에서는 만점(1.0000).
- **H4** (transistor, spatial-arrangement): uni batch=4에서는 PatchCore에 근소하게 못 미쳐(0.9238<0.929) 미결정이었으나, uni batch=16에서 역전(0.9335>0.929)되었고, **sep(class-separated)에서도 재확인**(0.9493>0.929) — 세 조건 중 가장 큰 격차. training setting(batch size, class-separated 여부)과 무관하게 일관되게 지지로 수렴.
- sep 15개 카테고리 Mean(I-AUROC 99.75%/I-AP 99.92%/P-AUROC 98.38%)은 uni batch=16 Mean(99.64%/99.81%/98.33%)과 거의 같은 수준 — **class-separated로 바꿔도 성능이 크게 달라지지 않음**. 이는 논문 Table 2의 핵심 주장("multi-class 모델 하나로도 class-separated 전용 모델을 이긴다")과도 부합: 이 재현에서는 오히려 두 setting의 차이 자체가 크지 않았음.

## sep 학습 진행 메모
- 최초 시도(2026-09-16, `dinomaly_sep_b16_extremefan.csv`가 남긴 GPU 로그)는 완주 전에 중단됨.
- 이후 재실행(`mvtec_sep_run2_clean`, 2026-09-18 10:02 시작) 중 carpet·grid 완료 직후 `torch.AcceleratorError: CUDA error: unknown error`로 한 차례 크래시(`train_log_sep_batch16_iter5000.txt` 참고) → 크래시 후 자동 재개(auto_restart) 옵션으로 leather부터 이어서 재개, 2026-09-19 04:06 exit code 0으로 15개 카테고리 전부 정상 완주.
- 크래시 전후로 결과 수치의 이상 징후(carpet/grid 값이 leather 이후 카테고리들과 비교해 특별히 튀지 않음)는 없어, 재개 방식이 최종 결과의 신뢰도에 영향을 주지 않은 것으로 보임.

## setting
- uni: multi-class(15개 카테고리 동시 학습, `dinomaly_mvtec_uni.py`) — PatchCore(class-separated)와 setting이 다름.
- sep: class-separated(카테고리별 개별 학습, `dinomaly_mvtec_sep.py`, batch_size=16, 카테고리당 5000 iter) — PatchCore와 동일한 setting. **완주됨.**
