# W40_method_runs — result

공통 프레임워크(`dinomaly_share_codebase`)로 MVTec·VisA를 실행한 수치 (LabTask #55).
데스크톱 RTX 5070 / 12GB, seed 0, class-separated(카테고리별 개별 학습).

실행이 두 번 있고 학습량이 다르다.

| | 날짜 | 학습량 | 상태 |
|---|---|---|---|
| **A** | 2026-09-25~ | `configs/<method>.yaml` 값 그대로 | 진행 중 |
| **B** | 2026-09-24 | 축소 (rd 10 ep, rd_orig 35 ep, dinomaly 200 iter) | 종료 |

A는 `--meta-epochs` / `--total-iter` / `--batch-size`를 넘기지 않아 config 값이 그대로 적용된다.

| 방법 | A의 학습량 | 출처 |
|---|---|---|
| patchcore | 학습 없음 (memory bank) | `configs/patchcore.yaml`은 `imagesize: 224`만 지정 |
| padim | 학습 없음 (가우시안 적합) | config에 학습 인자 없음 → DEFAULTS |
| rd_orig | meta_epochs 40 | `main.py:146` DEFAULTS |
| rd | meta_epochs 300 | `configs/rd.yaml` |

측정 단위: I-AUROC / P-AUROC, `wall_s`는 job 전체 벽시계 시간, `peak_vram_mb`는 job 중 `nvidia-smi` 2초 간격 최대값.

---

# 1. A — MVTec-AD

| category | patchcore | padim | rd_orig | rd |
|---|---|---|---|---|
| bottle | 1.0000 / 0.9878 | 1.0000 / 0.9872 | 1.0000 / 0.9889 | 1.0000 / 0.9894 |
| cable | 0.9921 / 0.9847 | 0.9072 / 0.9790 | 0.9689 / 0.9741 | 0.9801 / 0.9803 |
| capsule | 0.9749 / 0.9881 | 0.9246 / 0.9855 | 0.9485 / 0.9852 | 0.9673 / 0.9847 |
| carpet | 0.9852 / 0.9895 | (failed) | 0.9912 / 0.9902 | 0.9868 / 0.9887 |
| grid | 0.9942 / 0.9788 | 0.9657 / 0.9707 | 0.9975 / 0.9916 | 0.9916 / 0.9930 |
| hazelnut | 1.0000 / 0.9890 | 0.9504 / 0.9846 | 1.0000 / 0.9901 | 1.0000 / 0.9911 |
| leather | 1.0000 / 0.9934 | 1.0000 / 0.9933 | 1.0000 / 0.9946 | 1.0000 / 0.9940 |
| metal_nut | 1.0000 / 0.9843 | 0.9936 / 0.9800 | 1.0000 / 0.9741 | 1.0000 / 0.9765 |
| pill | 0.9482 / 0.9759 | (미실행) | 0.9705 / 0.9785 | (미실행) |
| screw | 0.9768 / 0.9902 | (미실행) | 0.9699 / 0.9945 | (미실행) |
| tile | 1.0000 / 0.9618 | (미실행) | 0.9949 / 0.9604 | (미실행) |
| toothbrush | 0.9083 / 0.9901 | (미실행) | 0.9889 / 0.9896 | (미실행) |
| transistor | 0.9958 / 0.9470 | (미실행) | 0.9758 / 0.9192 | (미실행) |
| wood | 0.9895 / 0.9441 | (미실행) | 0.9895 / 0.9578 | (미실행) |
| zipper | 0.9879 / 0.9869 | (미실행) | 0.9215 / 0.9789 | (미실행) |
| **Mean (ok만)** | **0.9835 / 0.9794** (15개) | **0.9631 / 0.9829** (7개) | **0.9811 / 0.9778** (15개) | **0.9907 / 0.9872** (8개) |

# 2. A — VisA

| category | rd_orig |
|---|---|
| candle | 0.9362 / 0.9887 |
| capsules | 0.9045 / 0.9937 |
| cashew | 0.9756 / 0.9632 |
| chewinggum | 0.9838 / 0.9855 |
| fryum | 0.9420 / 0.9630 |
| macaroni1 | 0.9647 / 0.9932 |
| macaroni2 | 0.8983 / 0.9931 |
| pcb1 | 0.9696 / 0.9971 |
| pcb2 | 0.9744 / 0.9871 |
| pcb3 | 0.9721 / 0.9924 |
| pcb4 | 0.9987 / 0.9834 |
| pipe_fryum | 0.9946 / 0.9907 |
| **Mean (ok만)** | **0.9595 / 0.9859** (12개) |

# 3. A — 실행 비용

| 방법 | 데이터셋 | 완료 | 총 시간 | 카테고리당 | peak VRAM |
|---|---|---|---|---|---|
| patchcore | mvtec | 15/15 | 0.13h | 31초 | 3,091 MiB |
| padim | mvtec | 7/15 | 0.29h | 2.5분 | 1,720 MiB |
| rd_orig | mvtec | 15/15 | 0.56h | 2.2분 | 8,290 MiB |
| rd_orig | visa | 12/12 | 1.23h | 6.2분 | 8,393 MiB |
| rd | mvtec | 8/15 | 17.33h | 130분 | 11,810 MiB |

---

# 4. B — MVTec-AD

| category | rd_orig | rd | dinomaly |
|---|---|---|---|
| bottle | 1.0000 / 0.9890 | 0.9960 / 0.9866 | 1.0000 / 0.9916 |
| cable | 0.9713 / 0.9748 | 0.9408 / 0.9731 | 0.9859 / 0.9789 |
| capsule | 0.9533 / 0.9849 | 0.8636 / 0.9788 | 0.9637 / 0.9871 |
| carpet | 0.9920 / 0.9902 | 0.9952 / 0.9918 | 1.0000 / 0.9950 |
| grid | 0.9908 / 0.9909 | 0.8906 / 0.9202 | 1.0000 / 0.9952 |
| hazelnut | 1.0000 / 0.9906 | 1.0000 / 0.9913 | 1.0000 / 0.9960 |
| leather | 1.0000 / 0.9944 | 1.0000 / 0.9947 | 1.0000 / 0.9939 |
| metal_nut | 1.0000 / 0.9752 | 0.9927 / 0.9723 | 1.0000 / 0.9748 |
| pill | 0.9733 / 0.9799 | 0.8584 / 0.9557 | (timeout) |
| screw | 0.9637 / 0.9941 | 0.8227 / 0.9860 | 0.9162 / 0.9922 |
| tile | 0.9924 / 0.9588 | 0.9949 / 0.9583 | 1.0000 / 0.9796 |
| toothbrush | 1.0000 / 0.9894 | 0.9889 / 0.9867 | (timeout) |
| transistor | 0.9758 / 0.9150 | 0.9775 / 0.9053 | 0.9896 / 0.9202 |
| wood | 0.9877 / 0.9567 | 0.9921 / 0.9556 | 0.9965 / 0.9759 |
| zipper | 0.9086 / 0.9789 | 0.9031 / 0.9644 | 0.9963 / 0.9859 |
| **Mean (ok만)** | **0.9806 / 0.9775** (15개) | **0.9478 / 0.9680** (15개) | **0.9883 / 0.9820** (13개) |

# 5. B — 실행 비용

| 방법 | 데이터셋 | 완료 | 총 시간 | 카테고리당 | peak VRAM |
|---|---|---|---|---|---|
| rd_orig | mvtec | 15/15 | 1.17h | 4.7분 | 8,033 MiB |
| rd | mvtec | 15/15 | 1.04h | 4.2분 | 11,799 MiB |
| dinomaly | mvtec | 13/15 | 4.04h | 19분 | 11,825 MiB |

# 6. A / B 대조 (같은 카테고리만)

A와 B에서 모두 `ok`인 카테고리만 골라 평균낸 값이다.

| 방법 | 실행 | 학습량 | 비교 카테고리 | I-AUROC | P-AUROC | 카테고리당 |
|---|---|---|---|---|---|---|
| rd | B | 10 ep | 8개 | 0.9599 | 0.9761 | 4.4분 |
| rd | A | 300 ep | 8개 | 0.9907 | 0.9872 | 129.9분 |
| rd | 차이 (A−B) | | | **+0.0309** | **+0.0111** | |
| rd_orig | B | 35 ep | 15개 | 0.9806 | 0.9775 | 4.7분 |
| rd_orig | A | 40 ep | 15개 | 0.9811 | 0.9778 | 2.2분 |
| rd_orig | 차이 (A−B) | | | **+0.0005** | **+0.0003** | |


---

# 7. 실행하지 못한 범위와 이유

| 범위 | 상태 | 이유 |
|---|---|---|
| A: rd MVTec 7개 (pill·screw·tile·toothbrush·transistor·wood·zipper) | 대기 | 9/25 20:30 GPU 반납으로 중단. pill은 219/300 에포크에서 끊겨 재실행 필요 |
| A: rd VisA 12개 | 대기 | MVTec 다음 차례 |
| A: padim MVTec 8개 | 대기 | `spec_mvtec_fast` 단계가 중간에 종료됨 |
| A: padim `carpet` | 실패 | exit code -1, 트레이스백 없이 mahalanobis 계산 35%에서 종료. peak VRAM 1,335 MiB. raw: `run_logs/padim__mvtec__carpet.log` |
| A: winclip·promptad·coad·uniad·simple | 미실행 | 노트북 담당 (`w55_laptop.py`) |
| A: dinomaly · glass | 미실행 | 아래 계산 참조 |
| B: dinomaly `pill` | 실패 | `Dinomaly_lib/utils.py:24` 정수 오버플로 (`Storage size calculation overflowed with sizes=[4, -1178562093]`). `expand_as`로 만든 stride-0 뷰에 boolean 인덱싱. 다른 14개에서는 미발생. raw: `../run_logs/dinomaly_pill_crash_tail.txt` |
| B: dinomaly `toothbrush` | 미실행 | `pill` 크래시로 차례가 오지 않음 |

config 설정 기준 소요시간 (2점 측정, `timing_report.txt`):

| 방법 | config 설정 | MVTec-15 | VisA-12 | 합계 |
|---|---|---|---|---|
| dinomaly | total_iter 5000 | 93.8h | 217.5h | 311.3h |
| glass | meta_epochs 640 | 320.1h | 742.7h | 1,062.8h |

---

# 8. 실행 환경

| 항목 | 값 |
|---|---|
| 기기 | 데스크톱, RTX 5070 12GB, 16 core, Windows 11 |
| 환경 | Python 3.11.9, torch 2.11.0+cu128, torchvision 0.26.0+cu128 (venv `.venv-gpu`) |
| 데이터 | MVTec-AD `C:/ai_local/glad_dataset/MVTec-AD`, VisA `C:/ai_local/glad_dataset/VisA_mvtec` |
| 평가 단위 | class-separated |
| seed | 0 |
| num_workers | 0 |
| 실행 방식 | 방법×카테고리 job마다 `main.py`를 별도 프로세스로 실행, CSV에 append (재개 가능) |

VisA는 공식 `split_csv/1cls.csv` 기준으로 MVTec 디렉터리 구조로 변환. 12 카테고리 / train 8,659 / test-good 962 / test-bad 1,200(마스크 전부 존재). 변환 스크립트 `../scripts/w55_visa_to_mvtec.py`.

peak VRAM 실측에 따른 기기 분담:

| 노트북 (RTX 5060 8GB) | 데스크톱 (RTX 5070 12GB) |
|---|---|
| padim 1,720 · uniad 1,839 · patchcore 3,091 | rd_orig 8,393 MiB |
| simple 3,336 · winclip 3,768 · promptad 5,119 | rd 11,810 MiB |
| coad 6,440 MiB | dinomaly 11,825 MiB |

---

# 9. 파일

| 파일 | 내용 |
|---|---|
| `summary_spec_mvtec_fast.csv` | A: patchcore 15/15 + padim 7/15 |
| `summary_spec_mvtec_rd_orig.csv` | A: rd_orig MVTec, meta_epochs=40 |
| `summary_spec_visa_rd_orig.csv` | A: rd_orig VisA, meta_epochs=40 |
| `summary_spec_mvtec_rd.csv` | A: rd MVTec, meta_epochs=300 |
| `summary_mvtec_rd_orig.csv` | B: rd_orig 15개, meta_epochs=35 |
| `summary_mvtec_rd.csv` | B: rd 15개, meta_epochs=10 |
| `summary_mvtec_dinomaly.csv` | B: dinomaly 13개, total_iter=200 |
| `summary_smoke.csv` | 11개 방법 × bottle 1 epoch — 비용 측정 1번째 점 |
| `summary_timing5.csv` | 학습형 방법 × bottle 5 epoch — 2번째 점 |
| `summary_smoke_dino1/2.csv` | dinomaly 200/600 iter 별도 측정 |
| `summary_mvtec_light.csv` | 시작 1분 만에 종료, 빈 파일 |
| `timing_per_run.csv` | 전 job 구간 분해(기동/데이터/모델/실행), peak VRAM, 이미지당 추론시간 |
| `timing_report.txt` | 위 CSV를 표로 출력 |
| `method_probe.txt` | 11개 방법이 import→계약검사→backbone→생성 중 어디까지 되는지 |

CSV 열: `status`(ok/timeout/failed), `wall_s`, `peak_vram_mb`, `meta_epochs`/`total_iter`, `auroc_mean`, `pixel_auroc_mean`, `sal_f1_mean`, `reason`, `log_path`, `results_csv`.

`timeout`의 제한시간은 프레임워크가 아니라 배치 러너가 건 값이다 (`w55_run_batch.py:199`). 재개 시 `timeout` 행만 재시도되고 `ok`/`failed`는 최종으로 친다.

---

# 10. 재개 방법

```bash
cd No_Submit/code/dinomaly_share_codebase/dinomaly_share_codebase
./.venv-gpu/Scripts/python.exe -u results_w40/w55_spec.py --from-phase 2   # 데스크톱
./.venv-gpu/Scripts/python.exe -u results_w40/w55_laptop.py                # 노트북
```

끝난 job은 건너뛴다. rd는 `pill`부터 이어서 MVTec 7개 → VisA 12개 순으로 진행한다.

job 제한시간은 9/25에 올렸다 (MVTec rd 3.5h → 4.5h, VisA rd 8h → 10h). 실측 1.569 s/iter 기준 최악 카테고리가 MVTec hazelnut 3.27h, VisA pcb3 7.45h로 옛 제한시간의 7% 안이었다. 파이썬이 시작 시 이 값을 읽으므로 실행 중인 프로세스에는 파일 수정이 반영되지 않는다.

중단 시에는 프로세스 트리 전체를 종료해야 한다. 이 환경에는 `wmic`이 없다.

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  ForEach-Object { taskkill /F /T /PID $_.ProcessId }
nvidia-smi --query-gpu=memory.used --format=csv
```

중단으로 죽은 job은 `failed`로 기록되고 재개 시 재시도되지 않으므로, 해당 행을 CSV에서 삭제해야 다시 실행된다.
