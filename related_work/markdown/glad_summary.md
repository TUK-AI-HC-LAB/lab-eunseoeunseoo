# GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models — 논문 조사 노트

PDF 파일 경로 : `related_work/paper/ECCV24_GLAD_Towards_Better_Reconstruction_with_Global_and_Local_Adaptive_Diffusion_Models_for_Unsupervised_Anomaly_Detection.pdf`

---

## Paper Metadata

| Item | Content |
|---|---|
| Title | GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection |
| Authors | Hang Yao, Ming Liu, Haolin Wang, Zhicun Yin, Zifei Yan, Xiaopeng Hong, Wangmeng Zuo |
| Conference / Journal | ECCV |
| Year | 2024 |
| Paper link | https://arxiv.org/abs/2406.07487 |
| GitHub / Official code | https://github.com/hyao1/GLAD |
| Reason for investigation | H3(전역 패턴 이상)와 관련된 diffusion 기반 anomaly detection 후속 연구 조사 중 선정. DiAD(method3_diad) 재현에서 grid I-AUROC가 개선 후 재하락하는 비단조 패턴을 보여, global/local 재구성을 분리하는 이 논문의 아이디어가 원인 설명이나 확장 방향에 참고가 될지 확인하려고 조사. |

---

## 0. 이 논문이 속한 분야

Diffusion 기반 이상탐지(DiAD 등)는 정상 이미지로 diffusion 모델을 학습시킨 뒤, 테스트 이미지에 노이즈를 씌웠다가 다시 복원(denoise)해서 원본과 비교한다 — 정상 영역은 잘 복원되지만 이상 영역은 "정상처럼" 복원되면서 원본과 달라지고, 그 차이가 곧 anomaly score가 된다.

이 논문은 기존 diffusion 방법들이 **모든 샘플·모든 영역을 똑같은 방식으로 복원**한다는 점을 문제로 지적한다. 구체적으로 두 가지:
1. **(Global) 이상 난이도가 샘플마다 다르다** — 예를 들어 부품이 아예 빠진 결함(missing)은 긁힌 자국(scratch) 하나보다 복원하기 훨씬 어렵다. 그런데도 모든 샘플에 같은 개수의 denoising step을 고정해서 쓴다.
2. **(Local) 이미지 한 장 안에서도 정상 영역과 이상 영역이 다르게 다뤄져야 한다** — 정상 영역은 디테일을 살려야 하고, 이상 영역은 "정상처럼" 다시 그려져야 하는데, 이 둘을 구분하지 않고 균일하게 복원하면 정상 영역의 디테일까지 뭉개지거나, 이상 영역이 충분히 "정상화"되지 않는다.

## 1. 핵심: DiAD와 무엇이 다른가

| 축 | DiAD | GLAD |
|---|---|---|
| 학습 단위 | 15개 카테고리를 하나의 모델로 공동학습(multi-class), 카테고리 조건을 SG network로 주입 | **주 실험(Table 1~4)은 카테고리별로 별도 모델(single-class)**이 headline 결과. multi-category(=DiAD와 같은 조건) 버전은 supplementary Appendix S6/Table S7에 별도로 존재하며, 거기서 DiAD와 직접 비교됨 |
| Denoising step | 모든 샘플에 고정된 step 수 적용 | 샘플마다 다른 step 수를 **자동으로 선택**(Adaptive Denoising Step, ADS) |
| 학습 데이터 | 정상 이미지만 사용 | 정상 이미지 + **합성 이상(synthetic anomaly)**을 섞어 학습(Anomaly-oriented Training Paradigm, ATP) — 비정상 영역에서 비-가우시안 노이즈를 예측하도록 유도 |
| 정상/이상 영역 처리 | SFF(Spatial-**aware** Feature Fusion) 블록이 SGDB3/4 등 **다른 스케일의 특징을 합쳐** 큰 결함 재구성력을 높임(feature pyramid 융합에 가까움, 정상/이상 픽셀을 구분해 다루지는 않음) | SAFF(Spatial-**Adaptive** Feature Fusion)가 anomaly map 기반 **마스크로 정상/이상 픽셀을 구분**해 정상 영역은 원본을, 이상 영역은 복원본을 사용 |
| 이상 점수 | ResNet50 feature 비교 | 다층(3/6/9/12번째 layer) **DINO** feature의 cosine similarity 차이를 layer별로 합산 |

가장 중요한 차이는 **"학습·복원 방식을 샘플/영역 단위로 적응적으로 바꾼다"**는 것 — DiAD를 포함한 기존 방법들이 전부 고정된 절차를 모든 입력에 동일하게 적용하는 것과 대비된다.

> 이름 함정 주의: DiAD의 SFF(Spatial-**aware** Feature Fusion)와 GLAD의 SAFF(Spatial-**Adaptive** Feature Fusion)는 이름이 거의 같지만 다른 메커니즘이다 — 착각하기 쉬우니 인용/비교할 때 주의.

## 2. 용어 (이 논문에서 쓰는 그대로)

- *ADS (Adaptive Denoising Step)*: 테스트 이미지에 노이즈를 점점 더 씌워가며(step 0→T), 매 step마다 "지금까지 복원한 결과"와 "원본을 그대로 다시 diffusion에 넣었을 때의 결과" 차이를 비교한다. 이 차이가 임계값 δ를 넘는 시점의 step을 그 샘플의 복원 시작점으로 쓴다. 즉 이상이 뚜렷할수록(=차이가 빨리 커질수록) 더 많은 step을, 이상이 미묘하면 더 적은 step을 쓰게 된다.
- *ATP (Anomaly-oriented Training Paradigm)*: 학습 중 정상 이미지에 합성 이상을 주입해, 모델이 "정상 노이즈(가우시안)"뿐 아니라 "이상 영역에서 나타나는 비-가우시안 차이"까지 예측하도록 손실함수를 구성. 표준 diffusion 학습(정상만 학습)의 한계 — 이상을 아예 본 적이 없어 이상 영역에서 복원 품질이 불안정한 문제 — 를 겨냥한다.
- *SAFF (Spatial-Adaptive Feature Fusion)*: ADS가 고른 step에서 나온 anomaly map으로 공간 마스크 m을 만들고, `복원 특징 * m + 원본 특징 * (1-m)`으로 최종 특징을 합성. 정상 영역의 디테일 손실을 막으면서 이상 영역만 복원 결과를 쓴다.
- *DINO feature*: self-supervised로 학습된 비전 트랜스포머의 중간 특징. 이 논문은 여러 층(3/6/9/12)의 DINO 특징을 함께 써서 이상 맵을 만든다.

## 3. 방법 — 단계별로, 예시와 함께

전체 흐름: (학습) 정상 이미지 + 합성 이상으로 diffusion denoiser를 ATP 손실로 학습 → (추론) 테스트 이미지마다 ADS로 적절한 step을 고른 뒤, 그 step에서 SAFF로 정상/이상 영역을 분리 융합해 복원 → DINO feature 비교로 최종 점수 산출.

### 3.1 ADS — 샘플마다 다른 시작점 찾기 (추론 시)
테스트 이미지에 노이즈를 단계적으로 씌우며 diffusion으로 복원한 결과(`x̂ᵃ_{t→0}`)와, 같은 이미지를 그대로 다시 diffusion에 통과시킨 결과(`xᵃ_{t→0}`)의 차이를 각 step에서 계산한다. 차이가 임계값 δ를 넘는 첫 step을 그 샘플의 복원 시작 step으로 선택한다. 오차 식은 `x̂ᵃ - x ~ √(1-ᾱ_t)(εᵃ-ε) + √ᾱ_t·n`로 유도되며, `n`이 이상으로 인한 차이 항이다.

### 3.2 ATP — 합성 이상으로 학습 (학습 시)
정상 이미지에 인위적으로 이상을 합성해 넣은 `xᵃ`를 만들고, 모델이 "정상 노이즈 ε"가 아니라 "이상 영역에서 실제로 필요한 보정치까지 포함한 목표"를 예측하도록 손실을 구성한다:
`L_ATP = E[ ‖(ε - (√ᾱ_t/√(1-ᾱ_t))(xᵃ-x)) - ε_θ(xᵃ_t, t)‖₂ ]`
이렇게 하면 모델이 이상 영역에서도(비-가우시안 노이즈까지) 그럴듯한 예측을 하도록 일반화 능력이 넓어진다.

### 3.3 SAFF — 정상/이상 영역을 분리해서 합치기 (추론 시)
ADS가 고른 step의 anomaly map에서 공간 마스크 `m`을 만들고, `x̂ᶠ_t = m·x̂ᵃ_t + (1-m)·xᵃ_t`로 최종 특징을 합성한다. 정상 영역(m≈0)은 원본을 그대로 쓰고, 이상 영역(m≈1)은 복원본을 쓴다.

### 3.4 이상 점수 산출
최종 복원 이미지와 원본을 각각 DINO의 3/6/9/12번째 층에 통과시켜 층별 cosine similarity 맵을 얻고, 이를 합산해 anomaly map(pixel-level)과 그 최댓값 등으로 image-level score를 만든다.

## 4. 실험 결과

데이터셋: MVTec-AD(15 카테고리), MPDD(6 카테고리), VisA(12 카테고리), 저자들이 직접 구성한 PCB-Bank(7 카테고리). 주 실험(Table 1~4)은 **카테고리별 단일 모델(single-class)** 설정.

### 주 실험 — MVTec-AD 카테고리 예시 (single-class, Table 1, I-AUROC/P-AUROC)
| 카테고리 | GLAD | 참고: DiAD(우리 재현, multi-class) |
|---|---|---|
| grid | 100 / 99.8 | 0.654 / 0.638 (epoch34) |
| transistor | 100 / 99.4 | 0.958 / 0.922 (epoch34) |

> 주의: GLAD 쪽은 카테고리마다 **별도로 학습한 모델**(single-class) 기준이라 훨씬 쉬운 설정이다. 우리 DiAD 재현(15개 카테고리 공동학습, multi-class)과 **직접 비교 불가** — 표는 "같은 데이터셋에서 설정이 다르면 얼마나 차이 나는지" 참고용으로만 본다.

### DiAD와의 비교 — multi-category 설정 (Table S7, PDF로 직접 확인·검증함)
GLAD 논문도 DiAD와 같은 조건(15개 카테고리 공동학습)의 변형을 supplementary에서 별도로 돌렸다. MVTec-AD 기준 정확한 수치:

| 방법 | I-AUROC/I-AP/I-F1max | P-AUROC/P-AP/P-F1max/PRO |
|---|---|---|
| DiAD (논문 원 보고치) | 97.2 / 99.0 / 96.5 | 96.8 / 52.6 / 55.5 / 90.7 |
| GLAD-256 (multi-category) | 97.5 / 99.1 / 96.6 | 97.4 / 60.8 / 60.7 / 93.0 |

4개 데이터셋(MVTec-AD/MPDD/VisA/PCB-Bank) 평균으로는 GLAD가 DiAD 대비 I-AUROC/I-AP/I-F1-max/P-AUROC/P-AP/P-F1-max/PRO 전부에서 6.3↑/5.6↑/4.9↑/1.9↑/14.3↑/10.4↑/11.1↑ 개선.

> **중요**: DiAD 논문이 자체 보고한 MVTec-AD multi-class I-AUROC는 **97.2**다. 우리가 로컬 8GB GPU로 재현한 DiAD의 mean I-AUROC는 epoch34 기준 **0.870**(`eval_results_epoch34.csv`) — 논문 보고치에 한참 못 미친다. 즉 우리 grid/transistor 결과가 PatchCore에 못 미치는 것이 "DiAD라는 방법 자체의 한계"인지 "우리 재현이 논문 수준 학습량에 못 미쳐서"인지, 이 비교만으로는 아직 구분이 안 된다 — 오히려 후자(학습량 부족) 쪽에 무게를 싣는 근거가 될 수 있다.

### Ablation (Table 6, MVTec-AD)
| 구성 | I-AUROC | P-AUROC |
|---|---|---|
| Baseline (LDM만) | 98.3 | 98.0 |
| + ADS | 99.0 | 98.5 |
| + ATP | 98.7 | 98.5 |
| + ADS + ATP (SAFF 없이) | 99.2 | 98.3 |
| + ADS + ATP + SAFF (전체) | 99.3 | 98.6 |

세 요소(ADS/ATP/SAFF)를 각각 뺐을 때 성능이 조금씩 떨어져, 세 장치가 서로 다른 역할로 기여한다고 저자들은 해석한다.

### 저자가 밝힌 한계
- ADS가 적절한 step을 고르기 위해 매 step마다 비교 연산을 하므로, **denoising step 수 자체가 줄어들지 않고 추가 연산 비용이 든다**(inference가 더 느려짐).
- 저자들도 이 비교 과정을 가볍게 만드는 것(lightweight evaluation)을 future work로 남겨둠.

## 5. 내 연구와의 연결

- **계열이 같은가/다른가**: 같은 diffusion 재구성 계열이지만, GLAD의 headline 결과는 **single-class**(카테고리별 모델) 기준이라 우리가 재현한 DiAD의 **multi-class**(공동학습) 설정과 조건이 다르다. 공정한 비교 지점은 GLAD 논문의 supplementary multi-category 표(DiAD 포함)뿐이다.
- **접근이 다른가**: DiAD는 모든 샘플에 고정된 denoising 절차를 쓰는 반면, GLAD는 샘플마다 다른 step(ADS)과 영역별로 다른 융합(SAFF)을 쓴다. 우리 H3 실험에서 관찰한 "grid I-AUROC가 epoch에 따라 개선-재하락을 반복"하는 현상은, grid처럼 이상 난이도가 카테고리마다 다른 경우 고정된 denoising 절차가 최적이 아닐 수 있다는 GLAD의 문제의식과 방향이 맞는다.
- **우리 피드백/가설과의 연결**: H3(grid의 전역 패턴 이상)를 직접 반박/지지하는 실험은 아니지만, "DiAD의 고정된 denoising step이 카테고리별 난이도 차이를 반영 못 한다"는 설명은 우리가 관찰한 grid의 불안정한 추이에 대한 **하나의 원인 가설 후보**를 제공한다.
- **참고할 점 / 주의할 점**: GLAD 수치를 DiAD/PatchCore와 나란히 놓고 비교하면 안 됨(학습 설정 자체가 다름 — single-class vs multi-class). 우리 다음 실험으로 고려해볼 만한 것은 GLAD 전체 재현이 아니라, **ADS(샘플별 adaptive step) 아이디어만 떼어서 우리 DiAD 파이프라인에 붙여보는 것** — 비용이 훨씬 적고, grid 재하락이 "고정 step 때문"인지 직접 검증할 수 있다.

## 인용 표기

본문에서 이 논문을 인용할 때는 weekly brief에서 등장 순서대로 `[N]` bracket number를 붙이고, `## 7. 참고문헌`에는 Google Scholar MLA 형식을 그대로 복사해 넣는다.
