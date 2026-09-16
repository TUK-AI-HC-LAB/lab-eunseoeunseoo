# method5_dinomaly — result

## 파일
- `eval_results_batch4_iter10000.csv` — multi-class(uni), batch_size=4(8GB 노트북 제약으로 축소), 10000 iter 완주 결과.
- `eval_results_batch16_iter10000.csv` — multi-class(uni), batch_size=16(원 논문 설정), 10000 iter 완주 결과. 원본 학습 로그: `train_log_batch16_iter10000.txt`.
- `batch16_inprogress_defaultfan.csv` — 위 batch=16 학습 중 GPU 온도/utilization/VRAM 로그(팬 기본 설정, 항목4 진단용).
- `dinomaly_sep_b16_extremefan.csv` — class-separated(sep) 학습 중 GPU 로그(팬 Extreme 모드, 진행 중).
- `monitor.log`, `monitor_sep.log`, `train_sep_stdout.log` — 백그라운드 실행 표준출력 캡처(디버깅용).

## 결과 요약(H3/H4 관련 카테고리만)
| 조건 | grid I-AUROC | transistor P-AUROC |
|---|---|---|
| batch=4 | 0.9975 | 0.9238 |
| **batch=16(원 논문 설정)** | **0.9983** | **0.9335** |
| PatchCore 기준 | 0.977 | 0.929 |

## 가설 판단
- **H3** (grid, global pattern-regularity): **지지**. batch=4·batch=16 둘 다 PatchCore를 명확히 앞섬.
- **H4** (transistor, spatial-arrangement): batch=4에서는 PatchCore에 근소하게 못 미쳐(0.9238<0.929) 미결정이었으나, **batch=16(원 논문 설정)에서 역전되어 지지로 전환**(0.9335>0.929). batch size가 결과에 실제로 영향을 준다는 직접 증거.
- 전체 15개 카테고리 Mean도 원 논문 보고치(Table 1, I-AUROC 99.6%/P-AUROC 98.4%)와 거의 일치(batch=16: 99.64%/98.33%) — 재현이 정확하다는 근거.

## GPU 진단(항목4)
- batch=16 학습(00:48~08:10, 팬 기본): 온도 최고 65도, 평균 54도, throttle 온도(87~90도)에 한참 못 미침 → **발열 병목 아님**.
- iter당 시간: batch=16 약 2.65초 vs batch=4 약 0.6초(약 4.4배) — 4배 커진 batch만큼의 정상적 연산량 증가로 보임. 브리프에 있던 "26초/iter, 44배 느림" 관측(다른 시도)은 이번 완주에서는 재현 안 됨.
- Extreme 팬 모드 효과: 비교 진행 중(`dinomaly_sep_b16_extremefan.csv`). 초반 데이터로는 기본 팬과 큰 차이 없음 — 애초에 발열 여유가 있었던 것과 일치.

## setting
multi-class(15개 카테고리 동시 학습, `dinomaly_mvtec_uni.py`) — PatchCore(class-separated)와 setting이 다름. class-separated 재현(`dinomaly_mvtec_sep.py`, 15개 카테고리)은 진행 중.
