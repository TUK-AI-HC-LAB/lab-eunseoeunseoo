# W40_method_runs — result

교수님이 주신 공통 프레임워크(`dinomaly_share_codebase`)로 MVTec을 실행한 결과 (LabTask #55).
**실행일 2026-09-24, 데스크톱 RTX 5070 / 12GB, seed 0, class-separated(카테고리별 개별 학습).**

> `> **[내 판단]**` 블록은 내가 직접 써야 하는 부분(빈칸). 그 밖의 표·수치는 전부 아래 raw 파일로 되짚을 수 있음.

## 파일

| 파일 | 내용 |
|---|---|
| `summary_mvtec_rd_orig.csv` | rd_orig 15개 카테고리, meta_epochs=35 |
| `summary_mvtec_rd.csv` | rd 15개 카테고리, meta_epochs=10 |
| `summary_mvtec_dinomaly.csv` | dinomaly 13개 카테고리, total_iter=200 (pill·toothbrush 미완) |
| `summary_smoke.csv` | 11개 방법 × bottle 1 epoch — 비용·VRAM 측정 1번째 점 |
| `summary_timing5.csv` | 학습형 방법 × bottle 5 epoch — 2번째 점 |
| `summary_smoke_dino1/2.csv` | dinomaly는 epoch이 아닌 total_iter로 도므로 200/600 iter로 별도 측정 |
| `summary_mvtec_light.csv` | patchcore·padim·winclip·promptad — 시작 1분 만에 가용 시간 종료, 빈 파일 |
| `timing_per_run.csv` | 전 job의 구간 분해(기동/데이터/모델/실행), peak VRAM, 이미지당 추론시간 |
| `timing_report.txt` | 위 CSV를 표로 출력한 것 |
| `method_probe.txt` | 11개 방법 각각이 import→계약검사→backbone→생성 중 어디까지 되는지 |

각 CSV 열: `status`(ok/timeout/failed), `wall_s`, `peak_vram_mb`, `meta_epochs`/`total_iter`, `auroc_mean`, `pixel_auroc_mean`, `sal_f1_mean`, `reason`, `log_path`, `results_csv`.

## 결과 요약 (H3/H4 관련 카테고리)

| 조건 | grid I-AUROC | transistor P-AUROC |
|---|---|---|
| **프레임워크 dinomaly** (200 iter) | **1.0000** | 0.9202 |
| **프레임워크 rd_orig** (35 ep) | 0.9908 | 0.9150 |
| **프레임워크 rd** (10 ep) | 0.8906 | 0.9053 |
| PatchCore 기준 | 0.977 | 0.963 (full-pixel 정의, W39 정정본) |
| RD4AD 공식 구현 재현 참고 | 1.000 | 0.927 (epoch 200) |
| Dinomaly(sep) 공식 구현 재현 참고 | 1.000 | 0.9493 (5000 iter/카테고리) |

- grid I-AUROC: dinomaly가 1.0000으로 PatchCore(0.977)를 넘음. rd는 0.8906으로 유일하게 크게 낮은데, 학습량이 논문 설정의 3.3%(10/300 epoch)라는 점과 분리되지 않음.
- **transistor P-AUROC: 세 방법 모두 0.905~0.920으로 PatchCore(0.963)보다 낮음.** 같은 방법의 공식 구현 재현치(RD4AD 0.927, Dinomaly 0.9493)보다도 낮다.

> **[내 판단] H3 / H4**
>
> _(여기에 내 말로:_
> _- **H3**(grid, global pattern-regularity): 지지/반박/미결정 중 무엇이고 근거가 무엇인지. dinomaly 1.0000은 지지 방향이지만 rd 0.8906을 어떻게 다룰지 — 학습 부족 탓으로 돌리는 게 타당한지._
> _- **H4**(transistor, spatial-arrangement): 세 방법이 전부 PatchCore보다 낮다는 게 H4에 어떤 의미인지. 학습량이 부족해서인지, 아니면 프레임워크의 후처리 경로 차이 때문인지 구분할 수 있는지(#54에서 확인한 대로 dinomaly·rd·rd_orig는 모두 공용 `RescaleSegmentor`를 쓰지 않고 자체 predict를 씀)._
> _- 학습량이 논문 설정의 3~87%로 제각각인 수치들을 H3/H4 판단 근거로 쓸 수 있는지.)_

## 15개 카테고리 전체 결과 (I-AUROC / P-AUROC)

| category | rd_orig (35 ep) | rd (10 ep) | dinomaly (200 iter) |
|---|---|---|---|
| bottle | 1.0000 / 0.9890 | 0.9960 / 0.9866 | 1.0000 / 0.9916 |
| cable | 0.9713 / 0.9748 | 0.9408 / 0.9731 | 0.9859 / 0.9789 |
| capsule | 0.9533 / 0.9849 | 0.8636 / 0.9788 | 0.9637 / 0.9871 |
| carpet | 0.9920 / 0.9902 | 0.9952 / 0.9918 | 1.0000 / 0.9950 |
| **grid** | 0.9908 / 0.9909 | 0.8906 / 0.9202 | **1.0000 / 0.9952** |
| hazelnut | 1.0000 / 0.9906 | 1.0000 / 0.9913 | 1.0000 / 0.9960 |
| leather | 1.0000 / 0.9944 | 1.0000 / 0.9947 | 1.0000 / 0.9939 |
| metal_nut | 1.0000 / 0.9752 | 0.9927 / 0.9723 | 1.0000 / 0.9748 |
| pill | 0.9733 / 0.9799 | 0.8584 / 0.9557 | (미실행) |
| screw | 0.9637 / 0.9941 | 0.8227 / 0.9860 | 0.9162 / 0.9922 |
| tile | 0.9924 / 0.9588 | 0.9949 / 0.9583 | 1.0000 / 0.9796 |
| toothbrush | 1.0000 / 0.9894 | 0.9889 / 0.9867 | (미실행) |
| **transistor** | 0.9758 / **0.9150** | 0.9775 / **0.9053** | 0.9896 / **0.9202** |
| wood | 0.9877 / 0.9567 | 0.9921 / 0.9556 | 0.9965 / 0.9759 |
| zipper | 0.9086 / 0.9789 | 0.9031 / 0.9644 | 0.9963 / 0.9859 |
| **Mean (15개)** | **0.9806 / 0.9775** | **0.9478 / 0.9680** | — (13개뿐) |
| **Mean (공통 13개)** | 0.9797 / 0.9764 | 0.9515 / 0.9676 | **0.9883 / 0.9820** |

dinomaly가 13개뿐이라 15개 mean끼리는 비교할 수 없다. 마지막 행이 **dinomaly가 가진 13개로 맞춘 mean**이며 이것만 직접 비교 가능하다.

### 공식 구현 재현치와 대조

| 방법 | 공식 구현 재현 (W39) | 그때 설정 | 이번 프레임워크 | 이번 설정 | 차이 |
|---|---|---|---|---|---|
| RD4AD / rd_orig | 98.7 / 97.8 | epoch **200** | 98.06 / 97.75 | epoch **35** | -0.64 / -0.05 %p |
| Dinomaly (sep) | 99.75 / 98.38 | **5000 iter**/카테고리 | 98.83 / 98.20 | **200 iter** | -0.92 / -0.18 %p |

> **[내 판단] 프레임워크 재현 신뢰도**
>
> _(여기에 내 말로: 학습량을 1/6·1/25로 줄였는데 격차가 1%p 미만인 것을 어떻게 읽을지. 프레임워크 수치를 공식 구현 재현치 대신 쓸 수 있다고 볼지.)_

## setting

| 항목 | 값 |
|---|---|
| 기기 | 데스크톱, RTX 5070 12GB, 16 core, Windows 11 |
| 환경 | Python 3.11.9, torch 2.11.0+cu128, torchvision 0.26.0+cu128 (venv `.venv-gpu`) |
| 데이터 | MVTec-AD, `C:/ai_local/glad_dataset/MVTec-AD` |
| 평가 단위 | class-separated |
| seed | 0 |
| num_workers | **0** — Windows에서 8로 올리면 spawn 오버헤드로 24초 → 92초로 **느려짐**을 실측 |
| 실행 방식 | 방법×카테고리 job마다 `main.py`를 별도 프로세스로 실행, CSV에 append (재개 가능) |

프레임워크 기본값(=논문 설정)과 이번 설정의 차이:

| 방법 | 프레임워크 기본 | 이번 실행 | 비율 |
|---|---|---|---|
| rd_orig | meta_epochs 40 | **35** | 87.5% |
| rd | meta_epochs 300 | **10** | 3.3% |
| dinomaly | total_iter 5000 | **200** | 4% |

epoch 수를 줄인 근거는 2점 측정으로 뽑은 비용이다. 자세한 값은 `timing_report.txt`.

| 방법 | epoch당 | peak VRAM | 8GB 노트북 |
|---|---|---|---|
| rd_orig | 5.4s | 8,282 MiB | **불가** |
| rd | 17.0s | 11,792 MiB | **불가** |
| dinomaly | — (iter 기반) | 11,816 MiB | **불가** |

이 세 방법이 8GB를 넘겨 노트북에서 실행 불가능하므로, 데스크톱 가용일(1일)에 우선 배치했다.

## 실행하지 못한 범위와 이유

| 범위 | 상태 | 이유 |
|---|---|---|
| dinomaly `pill` | 실패 | 프레임워크 버그. `Dinomaly_lib/utils.py:24` 정수 오버플로 (`Storage size calculation overflowed with sizes=[4, -1178562093]`). `expand_as`로 만든 stride-0 뷰에 boolean 인덱싱하면서 크기가 음수로 뒤집힘. 다른 14개 카테고리에서는 발생 안 함. raw: `../run_logs/dinomaly_pill_crash_tail.txt` |
| dinomaly `toothbrush` | 미실행 | `pill` 크래시가 배치 러너까지 중단시켜 차례가 오지 않음 |
| patchcore·padim·winclip·promptad·simple·uniad·glass·coad × MVTec | 미실행 | 데스크톱 가용 시간 내 우선순위에서 밀림. 전부 8GB 노트북에서 가능하므로 9/25~9/28 진행 |
| VisA 전체 (12 카테고리) | 미실행 | 위와 동일. **데이터 변환은 완료** — `C:/ai_local/glad_dataset/VisA_mvtec`, 12 카테고리 / train 8,659 / test-good 962 / test-bad 1,200(마스크 전부 존재), 변환 스크립트 `../scripts/w55_visa_to_mvtec.py` |

**논문 설정으로는 애초에 불가능한 것** (2점 측정에서 계산):

| 방법 | 논문 설정 | 카테고리당 | 15개 합계 |
|---|---|---|---|
| rd | 300 ep | 85분 | **21시간** |
| uniad | 1000 ep | 65분 | **16시간** |
| glass | 640 ep | 21시간 | **13일** |

## 실행을 위해 고쳐야 했던 프레임워크 문제 6건

원본은 전부 `*.orig`로 보존. 패치·스크립트는 `../scripts/`.

| # | 위치 | 증상 | 원인 | 성격 |
|---|---|---|---|---|
| 1 | `metrics_gpu.py:11` | CPU에서 **평가 단계만** `Torch not compiled with CUDA enabled` | `device="cuda"` 기본 인자 (같은 파일 47/79/165행은 이미 fallback 패턴) | 환경 |
| 2 | `datasets/base.py:133` | Windows에서 `num_workers>0` 이 `Can't pickle local object` | transform 첫 항목이 로컬 `lambda x: x` | **Windows 전용** |
| 3 | `trainer_glass.py:236` | glass가 **학습을 통째로 건너뛰고** 에러 없이 종료 | 맨몸 `except:` 가 openpyxl 미설치와 이름 불일치(`mvtec_glass_bottle` vs 키 `mvtec_bottle`)를 삼키고 `distribution=1`(학습 없는 판정 모드)로 전환 | **설계** |
| 4 | `RD_lib/noise.py:19` | rd 시작 즉시 `low is out of bounds for int32` | `np.random.randint(±1e10)` 이 Windows 기본 int32 초과 | **Windows 전용** |
| 5 | `Dinomaly_lib/utils.py:24` | dinomaly `pill` backward 중 크래시 | stride-0 뷰 boolean 인덱싱 오버플로 | **미해결** |
| 6 | scikit-learn 1.9.1 | `sklearn.cluster` `.pyd` 2개가 Windows 앱 제어 정책에 차단 | 파일 단위 차단 | 환경 (1.7.2로 우회) |

**3번이 가장 위험하다.** 실패가 예외로 드러나지 않고 다른 실행 모드로 조용히 바뀌어, exit 0과 함께 그럴듯한 숫자를 내놓는다.

| glass (bottle, 1 epoch) | I-AUROC | P-AUROC | 소요 |
|---|---|---|---|
| 수정 전 (학습 건너뜀) | 0.2452 | 0.4076 | 23초 |
| **수정 후 (실제 학습)** | **0.9452** | **0.9015** | 147초 |

> **[내 판단] 이 6건을 어떻게 볼지**
>
> _(여기에 내 말로: 3번을 #57(코드 기반 한계와 개선안) 후보로 쓸지. 2·4번이 Windows 전용이라는 게 이 프레임워크가 전제한 환경에 대해 무엇을 말해주는지.)_

## 재개 방법

모든 상태가 데스크톱 디스크에 남아 있어 명령 하나로 이어진다. 끝난 job은 건너뛰고, `timeout` 으로 남은 것만 재시도한다.

```bash
cd No_Submit/code/dinomaly_share_codebase/dinomaly_share_codebase
./.venv-gpu/Scripts/python.exe results_w40/w55_runall.py
```

재개 시 dinomaly는 `pill`·`toothbrush` 둘만 재시도하고, 이어서 MVTec 나머지 8개 방법 → VisA 순으로 진행한다.
