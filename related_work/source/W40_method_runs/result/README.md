# W40_method_runs — result

공통 프레임워크(`dinomaly_share_codebase`)로 MVTec·VisA를 실행한 수치 (LabTask #55).
데스크톱 RTX 5070 / 12GB, seed 0, class-separated(카테고리별 개별 학습).
노트북(RTX 5060 Laptop / 8GB)에서도 일부 방법을 나눠 실행했다 (3-0~3-5절).

아래 결과는 모두 config가 정한 학습량 그대로 실행한 것이다(표에서 "A"로 표기). `--meta-epochs` / `--total-iter` / `--batch-size`를 넘기지 않아 config 값이 그대로 적용된다.

> **예외: simple은 백본 가중치를 ImageNet V1으로 바꿔 실행했다.** 프레임워크 기본값(V2)으로는 판별기가 학습되지 않았기 때문이다. 이유와 근거는 3-5절.

| 방법 | 학습량 | 출처 |
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
| pill | 0.9482 / 0.9759 | (미실행) | 0.9705 / 0.9785 | 0.9724 / 0.9815 |
| screw | 0.9768 / 0.9902 | (미실행) | 0.9699 / 0.9945 | 0.9740 / 0.9939 |
| tile | 1.0000 / 0.9618 | (미실행) | 0.9949 / 0.9604 | 0.9935 / 0.9594 |
| toothbrush | 0.9083 / 0.9901 | (미실행) | 0.9889 / 0.9896 | 0.9833 / 0.9919 |
| transistor | 0.9958 / 0.9470 | (미실행) | 0.9758 / 0.9192 | 0.9842 / 0.9443 |
| wood | 0.9895 / 0.9441 | (미실행) | 0.9895 / 0.9578 | 0.9904 / 0.9568 |
| zipper | 0.9879 / 0.9869 | (미실행) | 0.9215 / 0.9789 | 0.9643 / 0.9747 |
| **Mean (ok만)** | **0.9835 / 0.9794** (15개) | **0.9631 / 0.9829** (7개) | **0.9811 / 0.9778** (15개) | **0.9859 / 0.9800** (15개) |

# 2. A — VisA

| category | rd_orig | rd |
|---|---|---|
| candle | 0.9362 / 0.9887 | 0.9605 / 0.9857 |
| capsules | 0.9045 / 0.9937 | 0.8903 / 0.9929 |
| cashew | 0.9756 / 0.9632 | (진행 중) |
| chewinggum | 0.9838 / 0.9855 | (미실행) |
| fryum | 0.9420 / 0.9630 | (미실행) |
| macaroni1 | 0.9647 / 0.9932 | (미실행) |
| macaroni2 | 0.8983 / 0.9931 | (미실행) |
| pcb1 | 0.9696 / 0.9971 | (미실행) |
| pcb2 | 0.9744 / 0.9871 | (미실행) |
| pcb3 | 0.9721 / 0.9924 | (미실행) |
| pcb4 | 0.9987 / 0.9834 | (미실행) |
| pipe_fryum | 0.9946 / 0.9907 | (미실행) |
| **Mean (ok만)** | **0.9595 / 0.9859** (12개) | **0.9254 / 0.9893** (2개) |

# 3. A — 실행 비용

| 방법 | 데이터셋 | 완료 | 총 시간 | 카테고리당 | peak VRAM |
|---|---|---|---|---|---|
| patchcore | mvtec | 15/15 | 0.13h | 31초 | 3,091 MiB |
| padim | mvtec | 7/15 | 0.29h | 2.5분 | 1,720 MiB |
| rd_orig | mvtec | 15/15 | 0.56h | 2.2분 | 8,290 MiB |
| rd_orig | visa | 12/12 | 1.23h | 6.2분 | 8,393 MiB |
| rd | mvtec | 15/15 | 30.80h | 123.2분 | 11,828 MiB |
| rd | visa | 2/12 (진행 중) | 11.94h | 358.1분 | 11,810 MiB |

# 3-0. A (노트북) — 개요

같은 프레임워크를 노트북(RTX 5060 Laptop, VRAM 8GB)에서 실행한 수치이다. 2026-09-25에 시작했고 2026-09-28 새벽까지의 값이다.

- 카테고리마다 따로 학습했고, 학습량 인자를 넘기지 않아 각 `configs/<방법>.yaml` 값이 그대로 쓰였다 (uniad 1000, promptad 100).
- 표의 값은 `I-AUROC / P-AUROC`이다.
- patchcore는 데스크톱(1절)에도 있다. 같은 방법이지만 기기가 달라 두 표의 값이 조금 다르다.

# 3-1. A (노트북) — MVTec-AD

| category | patchcore | winclip | promptad | coad | uniad |
|---|---|---|---|---|---|
| bottle | 1.0000 / 0.9876 | 0.9992 / 0.9584 | 1.0000 / 0.9902 | 1.0000 / 0.9925 | 1.0000 / 0.9841 |
| cable | 0.9946 / 0.9849 | 0.9303 / 0.9136 | 0.9878 / 0.9783 | 0.9987 / 0.9904 | 0.9674 / 0.9739 |
| capsule | 0.9725 / 0.9882 | 0.8827 / 0.9727 | 0.9489 / 0.9270 | 0.9781 / 0.9908 | 0.8568 / 0.9742 |
| carpet | 0.9860 / 0.9895 | 0.9984 / 0.9845 | 0.9988 / 0.9937 | 0.9928 / 0.9954 | 0.9984 / 0.9845 |
| grid | 0.9900 / 0.9825 | 0.9933 / 0.9532 | 0.9942 / 0.9892 | 0.9524 / 0.9922 | 0.8605 / 0.9033 |
| hazelnut | 1.0000 / 0.9892 | 0.9814 / 0.9861 | 1.0000 / 0.9925 | 1.0000 / 0.9943 | 0.9971 / 0.9843 |
| leather | 1.0000 / 0.9934 | 1.0000 / 0.9922 | 1.0000 / 0.9940 | 1.0000 / 0.9973 | 1.0000 / 0.9900 |
| metal_nut | 1.0000 / 0.9843 | 1.0000 / 0.8320 | 1.0000 / 0.9640 | 0.9985 / 0.9867 | 0.9721 / 0.9498 |
| pill | 0.9525 / 0.9742 | 0.9280 / 0.9382 | 0.9686 / 0.9596 | 0.9692 / 0.9830 | 0.9463 / 0.9525 |
| screw | 0.9807 / 0.9905 | 0.9170 / 0.9772 | 0.9143 / 0.9522 | 0.8754 / 0.9926 | 0.8573 / 0.9797 |
| tile | 1.0000 / 0.9611 | 1.0000 / 0.9124 | 1.0000 / 0.9624 | 1.0000 / 0.9774 | 0.9704 / 0.8943 |
| toothbrush | 0.9028 / 0.9900 | 0.9667 / 0.9822 | 0.9472 / 0.9903 | 0.9889 / 0.9922 | 0.9500 / 0.9839 |
| transistor | 0.9958 / 0.9491 | 0.9333 / 0.9280 | 0.9829 / 0.9568 | 0.9825 / 0.9671 | 0.9954 / 0.9889 |
| wood | 0.9886 / 0.9425 | 0.9956 / 0.9474 | 0.9886 / 0.9642 | 0.9868 / 0.9730 | 0.9868 / 0.9325 |
| zipper | 0.9874 / 0.9868 | 0.9769 / 0.9685 | 0.9916 / 0.9189 | 0.9622 / 0.9688 | 0.9443 / 0.9681 |
| **Mean (ok만)** | **0.9834 / 0.9796** (15개) | **0.9669 / 0.9498** (15개) | **0.9815 / 0.9689** (15개) | **0.9790 / 0.9862** (15개) | **0.9535 / 0.9629** (15개) |

# 3-2. A (노트북) — VisA

| category | patchcore | winclip | promptad | coad |
|---|---|---|---|---|
| candle | 0.9698 / 0.9874 | 0.9826 / 0.9569 | 0.9750 / 0.9575 | 0.9325 / 0.9923 |
| capsules | 0.7543 / 0.9764 | 0.8748 / 0.9561 | 0.8323 / 0.9769 | 0.8908 / 0.9910 |
| cashew | 0.9740 / 0.9838 | 0.9622 / 0.9676 | 0.9202 / 0.8705 | 0.9722 / 0.9941 |
| chewinggum | 0.9934 / 0.9864 | 0.9754 / 0.9862 | 0.9800 / 0.9665 | 0.9882 / 0.9913 |
| fryum | 0.9280 / 0.9452 | 0.9256 / 0.9554 | 0.9640 / 0.9013 | 0.9552 / 0.9619 |
| macaroni1 | 0.9188 / 0.9839 | 0.9114 / 0.9049 | 0.9014 / 0.9264 | 0.8593 / 0.9919 |
| macaroni2 | 0.6586 / 0.9343 | 0.7651 / 0.8182 | 0.7533 / 0.9289 | 0.7059 / 0.9878 |
| pcb1 | 0.9693 / 0.9946 | 0.9583 / 0.9670 | 0.9214 / 0.9278 | 0.9536 / 0.9959 |
| pcb2 | 0.9316 / 0.9763 | 0.7733 / 0.9052 | 0.9016 / 0.8396 | 0.8848 / 0.9857 |
| pcb3 | 0.9477 / 0.9851 | 0.8874 / 0.9664 | 0.8880 / 0.9554 | 0.9013 / 0.9875 |
| pcb4 | 0.9925 / 0.9743 | 0.9594 / 0.9814 | 0.9688 / 0.8998 | 0.9935 / 0.9896 |
| pipe_fryum | 0.9978 / 0.9906 | 0.8874 / 0.9545 | 0.9886 / 0.9849 | 0.9722 / 0.9949 |
| **Mean (ok만)** | **0.9197 / 0.9765** (12개) | **0.9052 / 0.9433** (12개) | **0.9162 / 0.9279** (12개) | **0.9175 / 0.9887** (12개) |

# 3-3. A (노트북) — 실행 비용

카테고리당 시간은 job 전체 벽시계 시간의 평균이고, VRAM은 job 중 `nvidia-smi` 최댓값이다.

| 방법 | 데이터셋 | 완료 | 카테고리당 (평균) | peak VRAM |
|---|---|---|---|---|
| patchcore | mvtec | 15/15 | 1.4분 | 2,872 MiB |
| winclip | mvtec | 15/15 | 8.1분 | 4,400 MiB |
| promptad | mvtec | 15/15 | 24.5분 | 2,878 MiB |
| coad | mvtec | 15/15 | 14.9분 | 6,706 MiB |
| uniad | mvtec | 15/15 | 2.31시간 | 1,360 MiB |
| patchcore | visa | 12/12 | 2.1분 | 5,426 MiB |
| winclip | visa | 12/12 | 12.3분 | 7,873 MiB |
| promptad | visa | 12/12 | 1.06시간 | 3,659 MiB |
| coad | visa | 12/12 | 22.2분 | 7,886 MiB |

---

# 3-4. A (노트북) — 제한 시간에 걸린 job

실행 스크립트에는 job당 제한 시간이 있고 넘으면 강제 종료된다 (프레임워크에는 없음). 시간 초과는 방법이 실패한 것이 아니라 스크립트가 종료한 것이다.

| job | 원래 제한 | 결과 |
|---|---|---|
| padim (MVTec bottle, cable, capsule, carpet) | 1800초 | 4개 모두 시간 초과. 로그상 마할라노비스 거리 계산 단계가 초당 약 1.5회 속도였고 끝나기 전에 종료됨. 노트북에서는 제외하고 재실행하지 않음 |
| promptad (MVTec hazelnut, screw) | 1800초 | 제한을 5400초로 올려 재실행, 각각 2024초, 1633초에 완료 |
| uniad (MVTec carpet) | 9000초 | 학습 918/1000 epoch에서 종료. 제한을 21600초로 올려 재실행, 10533초에 완료 |

VisA의 patchcore, winclip, promptad 제한도 3600초에서 10800초로 올려서 돌렸다.

---

# 3-5. simple — 백본 가중치를 V1으로 바꿔 실행한 이유 (예외)

**simple만 프레임워크 설정과 다르게 실행했다.** 학습량 등 config 값은 그대로이고, 백본(WideResNet50)의 사전학습 가중치만 ImageNet V1으로 바꿨다. 나머지 방법은 모두 프레임워크 그대로다.

### V1 / V2 가중치란

PyTorch의 이미지 모델 라이브러리(torchvision)는 WideResNet50에 ImageNet 사전학습 가중치를 두 가지 제공한다. 모델 구조는 같고 학습된 값만 다르다.

| 이름 | 파일 | 설명 | ImageNet 정확도 | 불러오는 방식 |
|---|---|---|---|---|
| V1 (`IMAGENET1K_V1`) | `wide_resnet50_2-95faca4d.pth` | 원래 제공되던 가중치. 원 논문의 학습 방식을 재현 | 78.5% | `pretrained=True` (예전 방식) |
| V2 (`IMAGENET1K_V2`) | `wide_resnet50_2-9ba9bcbe.pth` | torchvision이 개선된 학습 방식으로 다시 학습해 추가 | 81.6% | `weights="DEFAULT"` |

정확도와 파일 이름은 torchvision이 가중치와 함께 제공하는 정보(`Wide_ResNet50_2_Weights`)에서 확인했다.

### 무엇이 문제였나

- 프레임워크의 `backbones.py`는 백본을 `wide_resnet50_2(weights="DEFAULT")`로 불러온다. torchvision에서 이 모델의 DEFAULT는 **ImageNet V2 가중치**다(교수자 제공 원본 zip부터 이렇게 되어 있음).
- SimpleNet 공식 코드는 `wide_resnet50_2(pretrained=True)`, 즉 **V1 가중치**를 쓴다.
- 같은 입력에서 V2의 중간층 특징값은 V1보다 약 9배 크다(layer3 표준편차 V1 0.09, V2 0.85).
- SimpleNet은 정상 특징에 표준편차 0.015의 노이즈를 더해 "가짜 이상 특징"을 만들고, 판별기가 둘을 구분하도록 학습한다. 0.015는 V1 특징 크기에 맞춘 값이라, V2에서는 노이즈가 상대적으로 너무 작아 정상과 가짜 이상이 사실상 같아진다. 그 결과 **판별기가 모든 입력에 0 근처 점수를 내는 상태로 무너져 학습되지 않았다.**

### 근거

bottle 기준. 판별기 정답률은 정상 특징을 정상으로, 가짜 이상 특징을 이상으로 맞힌 비율이다.

| 조건 (bottle) | 판별기 학습 | I-AUROC / P-AUROC |
|---|---|---|
| 프레임워크 그대로 (V2 가중치, 40 epoch) | 안 됨 (정답률* 0.0) | 0.9119 / 0.7439 |
| 가중치만 V1으로 교체 (2 epoch, 원인 확인용) | 됨 (정답률* 0.79) | 0.9929 / 0.9676 |
| SimpleNet 공식 코드 (V1 가중치, 40 epoch, W39 재현) | 됨 (정답률* 0.99) | 1.0000 / 0.9800 |

\* 정답률: 판별기는 특징마다 점수를 내며, 정상은 높게·가짜 이상은 낮게 내도록 학습한다. 여기서 정답률은 정상 특징 중 점수가 +0.5 이상인 비율과 가짜 이상 특징 중 점수가 −0.5 미만인 비율이다(±0.5는 SimpleNet이 정한 기준선, 학습 로그의 `p_true`·`p_fake`). 두 비율은 거의 같아 하나만 적었다. 학습되면 1에 가까워지고, 0이면 모든 특징이 0 근처 점수를 받아 정상과 가짜 이상을 구분하지 못한다는 뜻이다. 프레임워크 그대로 실행했을 때는 손실도 40 epoch 내내 약 1.0에 머물렀는데, 이는 모든 점수가 0일 때의 손실값(정상 쪽 0.5 + 가짜 이상 쪽 0.5)과 같다.

프레임워크 그대로 실행한 cable·capsule도 같은 양상이었다(cable 0.8383 / 0.8174, capsule 0.9278 / 0.9466).

원인을 좁히는 과정에서 아래는 원인이 아님을 확인했다.

| 의심한 원인 | 확인 방법 | 결과 |
|---|---|---|
| 이미지 캐시(데스크톱에서 추가한 속도 개선) | 캐시를 끄고 bottle 실행 | 똑같이 판별기가 무너짐 |
| 라이브러리 버전 | W39 공식 코드 재현 환경이 같은 torch 2.11.0에서 정상 학습됨 | 원인 아님 |
| adaptor 학습률(프레임워크 config가 공식 코드보다 100배 작음) | 학습률만 공식 값으로 바꿔 bottle·cable·capsule 40 epoch 실행 | 세 카테고리 모두 판별기가 무너짐 |

### 적용 방법

`../scripts/main_v1.py`가 `backbones.py`의 wideresnet50 항목만 `weights="IMAGENET1K_V1"`로 바꿔 끼운 뒤 `main.py`를 그대로 실행한다. 프레임워크 파일은 수정하지 않았다. 원시 결과: `summary_spec_mvtec_simple_v1.csv`, `summary_spec_visa_simple_v1.csv`. 프레임워크 그대로(V2) 실행한 3개 카테고리는 `summary_spec_mvtec_simple.csv`에 남겨 두었다.

### simple (V1) — MVTec-AD

카테고리당 약 25.2분, peak VRAM 1,548 MiB.

| category | simple (V1) |
|---|---|
| bottle | 1.0000 / 0.9752 |
| cable | 0.9413 / 0.9544 |
| capsule | 0.9781 / 0.9843 |
| carpet | 0.9502 / 0.9531 |
| grid | 1.0000 / 0.9885 |
| hazelnut | 0.9911 / 0.9674 |
| leather | 1.0000 / 0.9897 |
| metal_nut | 0.9990 / 0.8965 |
| pill | 0.9280 / 0.8800 |
| screw | 0.8897 / 0.9855 |
| tile | 0.9722 / 0.8882 |
| toothbrush | 0.8722 / 0.9851 |
| transistor | 0.9808 / 0.7533 |
| wood | 0.8623 / 0.7905 |
| zipper | 0.9743 / 0.9892 |
| **Mean (ok만)** | **0.9560 / 0.9321** (15개) |

### simple (V1) — VisA

12개 중 8개 완료, 나머지 진행 중(pcb2~pipe_fryum). 카테고리당 약 68.2분(완료분 평균).

| category | simple (V1) |
|---|---|
| candle | 0.9218 / 0.9763 |
| capsules | 0.6845 / 0.9629 |
| cashew | 0.8640 / 0.9655 |
| chewinggum | 0.9920 / 0.9898 |
| fryum | 0.8806 / 0.9158 |
| macaroni1 | 0.8994 / 0.9903 |
| macaroni2 | 0.7102 / 0.9570 |
| pcb1 | 0.8308 / 0.9883 |
| pcb2 | (진행 중) |
| pcb3 | (미실행) |
| pcb4 | (미실행) |
| pipe_fryum | (미실행) |
| **Mean (ok만, 8개)** | **0.8479 / 0.9682** |

### 같은 백본을 쓰는 다른 방법

이 백본 로더(`backbones.py`)를 쓰는 방법은 patchcore·simple·glass뿐이다. padim·rd·rd_orig는 각자 코드에서 V1을 불러오고, 나머지는 WideResNet50을 쓰지 않는다.

patchcore도 V1으로 MVTec 15개를 다시 돌려 비교했다. 평균은 V2 0.9834 / 0.9796, V1 0.9826 / 0.9808로 거의 같아서, **patchcore 결과(3-1절)는 프레임워크 그대로(V2) 둔다.** 카테고리별로는 transistor P-AUROC가 V2 0.9491, V1 0.9628로 차이가 가장 컸다.

---

---

# 4. 실행하지 못한 범위와 이유

| 범위 | 상태 | 이유 |
|---|---|---|
| A: rd VisA fryum | 진행 중 | candle·capsules·cashew·chewinggum 완료 후 진행 중 (2026-09-28 07:51 기준) |
| A: rd VisA 8개 (macaroni1·macaroni2·pcb1·pcb2·pcb3·pcb4·pipe_fryum) | 대기 | fryum 다음 차례 |
| A: padim MVTec 8개 | 대기 | `spec_mvtec_fast` 단계가 중간에 종료됨 |
| A: padim `carpet` | 실패 | exit code -1, 트레이스백 없이 mahalanobis 계산 35%에서 종료. peak VRAM 1,335 MiB. raw: `run_logs/padim__mvtec__carpet.log` |
| A: simple VisA (노트북, 백본 V1) | 진행 중 (8/12) | pcb2부터 이어서 실행 중 (3-5절) |
| A: uniad VisA (노트북) | 대기 | simple VisA 다음 차례 |
| A: padim (노트북) | 제외 | 1800초 제한에 걸림 (3-4절). 데스크톱 담당 |
| A: dinomaly | 미실행 | 실행 시간만 실측(아래), 성능(I-AUROC/P-AUROC)은 아직 없음 |
| A: glass | 미실행 | 아래 계산 참조 |

config 설정 기준 소요시간:

| 방법 | config 설정 | MVTec-15 | VisA-12 | 합계 | 측정 방식 |
|---|---|---|---|---|---|
| dinomaly | total_iter 5000 | 62.6h | 50.1h | 112.7h | bottle 1개, 128 epoch 실측(2.998 s/it) |
| glass | meta_epochs 640 | 132.7h | 315.2h | 447.9h | bottle 5 epoch 실측 평균(평가 포함) |

**dinomaly.** bottle 1개를 config 그대로(total_iter=5000, 총 385 epoch) 돌리다가 속도가 안정된 128 epoch 지점에서 멈추고(2.998 s/iteration, `results_w55/dino5000_probe/run_logs/dinomaly__mvtec__bottle.log`), 여기에 `5000 iter × 2.998s`로 카테고리당 시간(약 4.17h)을 구해 카테고리 수를 곱했다. VisA(12개)가 MVTec(15개)보다 합계가 작은 것은 오타가 아니다 — dinomaly는 `total_iter`가 고정이라 카테고리 크기와 무관하게 항상 약 5,000 iteration을 돌기 때문에, 합계는 거의 카테고리 **개수**에만 비례한다(15개 대 12개). 128~146 epoch 구간(3.39 s/it, 표본 18)까지 반영한 상한은 MVTec 70.6h·VisA 56.5h·합계 127.1h이다. 참고로 Dinomaly 공식 구현은 같은 설정(class-separated, batch 16, 5000 iter)으로 8GB 노트북(RTX 5060)에서 13개 카테고리를 18시간 4분에 끝냈는데(W39 재현), 이 프레임워크 실측은 그보다 3배 가까이 느리다 — 다만 하드웨어가 반대 방향(공식 구현은 8GB 노트북, 이 프레임워크는 12GB 데스크톱)이라 그 차이를 하드웨어 탓으로 볼 수도 없다. peak VRAM은 11,746 MiB로 8GB 노트북에서는 못 돌린다. **끝까지 돌리지 않아서 config 설정(5000 iter)에서의 I-AUROC/P-AUROC는 아직 없다.**

**glass.** bottle을 5 epoch 실행해 잰 epoch당 시간(40.11 / 37.95 / 41.58 / 36.79 / 36.98초, `lg_results/epoch_test_glass.csv`)의 평균에, config epoch 수(640)와 카테고리 수를 곱해 추정했다. `configs/glass.yaml`이 `eval_epochs: 1`이라 매 epoch 전체 평가가 도는데, 위 측정은 학습 시간만 재므로(평가는 별도) 실측 벽시계 기준 평가 비용(epoch당 약 5.7초)을 더한 값을 표에 썼다. 평가를 뺀 학습 시간만이면 MVTec 115.6h·VisA 274.5h·합계 390.1h이다. VisA는 카테고리별 학습 이미지 수 비율로 환산했다.

두 방법 다 이번 주 실행 대상에서 제외한다: glass 447.9h(약 18.7일), dinomaly 112.7h(약 4.7일)에 남은 rd VisA(약 52h)까지 더하면 데스크톱 한 대로는 10/5경까지 걸린다. "실행 불가"가 아니라 "실측 완료, 기간 확보 시 실행 가능"이다.

---

# 5. 실행 환경

| 항목 | 값 |
|---|---|
| 기기 | 데스크톱, RTX 5070 12GB, 16 core, Windows 11 |
| 기기 (노트북) | RTX 5060 Laptop 8GB, Windows 11. torch 2.11.0+cu128 (venv `.venv-gpu`). 데이터 MVTec-AD `C:/ai_local/diad_dataset`, VisA `C:/ai_local/VisA_mvtec` |
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
| simple 3,336 · winclip 3,768 · promptad 5,119 | rd 11,828 MiB |
| coad 6,440 MiB | dinomaly 11,746 MiB (bottle, total_iter=5000 실측) |

---

# 6. 파일

| 파일 | 내용 |
|---|---|
| `summary_spec_mvtec_fast.csv` | A: patchcore 15/15 + padim 7/15 |
| `summary_spec_mvtec_rd_orig.csv` | A: rd_orig MVTec, meta_epochs=40 |
| `summary_spec_visa_rd_orig.csv` | A: rd_orig VisA, meta_epochs=40 |
| `summary_spec_mvtec_rd.csv` | A: rd MVTec, meta_epochs=300 |
| `summary_spec_visa_rd.csv` | A: rd VisA, meta_epochs=300 (진행 중) |
| `summary_smoke.csv` | 11개 방법 × bottle 1 epoch — 비용 측정 1번째 점 |
| `summary_timing5.csv` | 학습형 방법 × bottle 5 epoch — 2번째 점 |
| `summary_smoke_dino1/2.csv` | dinomaly 200/600 iter 별도 측정 (600 iter는 제한시간 초과로 무효) |
| `dino5000_probe/run_logs/dinomaly__mvtec__bottle.log` | dinomaly config 그대로(total_iter=5000) bottle 128 epoch 실측, 4절 소요시간의 근거 |
| `summary_mvtec_light.csv` | 시작 1분 만에 종료, 빈 파일 |
| `timing_per_run.csv` | 전 job 구간 분해(기동/데이터/모델/실행), peak VRAM, 이미지당 추론시간 |
| `timing_report.txt` | 위 CSV를 표로 출력 |
| `method_probe.txt` | 11개 방법이 import→계약검사→backbone→생성 중 어디까지 되는지 |

CSV 열: `status`(ok/timeout/failed), `wall_s`, `peak_vram_mb`, `meta_epochs`/`total_iter`, `auroc_mean`, `pixel_auroc_mean`, `sal_f1_mean`, `reason`, `log_path`, `results_csv`.

`timeout`의 제한시간은 프레임워크가 아니라 배치 러너가 건 값이다 (`w55_run_batch.py:199`). 재개 시 `timeout` 행만 재시도되고 `ok`/`failed`는 최종으로 친다.
