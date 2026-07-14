# DiAD Summary

## 논문 메타데이터

| 항목 | 내용 |
|---|---|
| 제목 | DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection |
| 저자 | He et al. |
| 학회 / 저널 | AAAI 2024 |
| 연도 | 2023 (arXiv) / 2024 (AAAI) |
| 논문 링크 | https://arxiv.org/abs/2312.06607 |
| GitHub / 공식 코드 | https://github.com/lewandofskee/DiAD (공식, 저자 본인 리포) |
| 조사 이유 | Candidate C — H3/H4 검증 (PatchCore의 3×3 local aggregation이 global pattern anomaly를 못 잡을 가능성(H3), per-patch independent scoring이 spatial arrangement anomaly를 못 잡을 가능성(H4)). H2(Candidate A / WinCLIP)가 zero-shot·1-shot 모두에서 반박된 뒤, 남은 후보(A 연장 vs C) 검토 결과 C를 선택 |

---

## 문제 정의

Diffusion 모델은 이미지 재구성 성능이 뛰어나지만, 보다 실용적인 **multi-class** 세팅(카테고리마다 모델을 따로 두지 않고 하나의 모델로 전체를 처리)에서는 재구성 과정에서 카테고리 정체성과 픽셀 단위 구조적 정합성을 유지하기 어렵다 — 재구성된 이미지가 다른 카테고리의 외형으로 흘러가거나 세부 구조를 잃을 수 있는데, 이는 "입력과 재구성 결과를 비교"하는 재구성 기반 방법의 이상 탐지 신호 자체를 무너뜨린다.

**핵심 질문**: Diffusion 기반 재구성 방법이 단일 multi-class 모델 안에서도 의미(카테고리)와 픽셀 단위 구조를 동시에 충분히 보존해서, 입력 대비 재구성 차이를 통해 이상을 탐지·위치추정할 수 있는가?

---

## 핵심 아이디어

Denoising 과정을 입력 이미지 자체에서 추출한 semantic 정보로 guide하여, 무조건부(unconditional) 생성이나 class label만으로 conditioning하는 대신 재구성이 입력의 카테고리·구조에 충실하게 유지되면서도 이상 영역은 정상 형태로 "치유"되도록 만든다.

PatchCore가 저장된 정상 패치와의 거리로, WinCLIP이 언어로 정의한 정상/이상 개념과의 거리로 이상을 판단한다면, DiAD는 diffusion 모델의 재구성 결과가 원본과 얼마나 **다른지**로 판단한다 — 차이가 클수록 그 영역을 모델이 이상으로 간주해 정상으로 되돌려 재구성했다는 뜻이다.

---

## 방법 개요

**프레임워크** : Pixel-space Autoencoder → Latent-space Diffusion (Semantic-Guided network + Stable Diffusion denoising network) → Feature-space Anomaly Scoring

1. **Pixel-space Autoencoder** : 사전학습된 encoder E가 입력 이미지 x₀를 latent 표현 z = E(x₀)로 압축하고, decoder D가 최종 재구성 latent ẑ를 다시 픽셀 공간의 재구성 이미지 x̂₀ = D(ẑ)로 복원한다.
2. **Semantic-Guided(SG) Network** : forward diffusion으로 z₀에 노이즈를 T step 추가해 zₜ를 얻는다. Reverse denoising 시, 사전학습되어 고정된 Stable Diffusion(SD) denoising network(encoder 4개 + middle block + decoder 4개 block)와, SD 파라미터를 복제해 초기화한 SG network가 함께 동작한다. SG network는 입력 이미지 x₀를 conv-silu layer로 z와 같은 차원으로 변환한 뒤 zₜ와 더해 SG encoder(SGEB)에 넣고, SG middle block(SGM)의 출력을 SD middle block 출력에 더하며, SG decoder(SGDB)의 출력을 SFF block을 거쳐 SD decoder 출력에 더한다. 즉 원본 이미지의 의미(카테고리) 정보를 denoising 경로 곳곳에 주입해 DDPM의 카테고리 오분류, LDM의 semantic 손실 문제를 동시에 해결한다.
3. **Spatial-aware Feature Fusion(SFF) Block** : SGEB3의 여러 layer 특징을 SGEB4의 각 layer에 통합해 더함으로써, texture 카테고리에서 필요한 미세한 원본 디테일 보존과 object 카테고리에서 필요한 대면적 이상 재구성 능력을 동시에 확보한다. Batch Normalization 대신 Instance Normalization을, ReLU 대신 SiLU를 사용해 카테고리 간 분포 차이가 큰 multi-class 설정에서 개별 샘플 정보 손실을 줄인다.
4. **Feature-space Anomaly Scoring** : 입력 x₀와 재구성 x̂₀를 동일한 사전학습 feature extractor(ResNet50)에 통과시켜 여러 scale(n∈{2,3,4})의 feature map을 얻고, 각 scale에서 코사인 유사도 기반 anomaly map Mⁿ(Eq. 8)을 계산한 뒤 업샘플링하여 합산해 최종 anomaly score map S(Eq. 9)를 만든다. 이미지 단위 score는 8×8 global average pooling을 반복 적용한 뒤의 최댓값으로 계산한다.

### 실험 설정

**데이터셋**
- MVTec-AD : 텍스처 5종 + 객체 10종, 총 5,354개 고해상도 이미지 (학습 3,629 정상 이미지 / 테스트 1,725 정상+이상 이미지). PatchCore와 달리 카테고리별로 모델을 따로 두지 않고 **단일 모델로 15개 카테고리를 모두 처리하는 multi-class 세팅**으로 평가.
- VisA : 12개 객체 subset, 총 10,821개 이미지(정상 9,621 / 이상 1,200, 이상 유형 78종). Complex structure / Multiple instances / Single instance 세 유형으로 구성.

**장비 / 소프트웨어**
- GPU : NVIDIA Tesla V100 32GB 1대 / Optimizer : Adam (lr = 1e-5)
- Feature extractor(기본값) : ImageNet 사전학습 ResNet50 (ablation에서 VGG16/19, ResNet18/34/50/101, WideResNet50/101, EfficientNet-b0/b2/b4 비교)
- Auto-encoder : KL-regularized 버전을 사용, denoising network 학습 전에 fine-tune

**실험 조건**

| Parameter | Value | Description |
|---|---|---|
| 입력 이미지 크기 | 256×256 | MVTec-AD, VisA 공통 리사이즈 |
| 학습 epoch | 1,000 | batch size 12 |
| Forward diffusion timestep T | 1,000 | 초기 노이즈 스텝 수 |
| Inference sampler | DDIM, 10 step | 빠른 역확산 샘플링 |
| Anomaly map smoothing | Gaussian filter, σ=5 | 위치추정 맵 품질 개선 |
| Feature 추출 layer | n ∈ {2, 3, 4} | anomaly score 계산에 사용하는 scale |
| 이미지 단위 score 산출 | 8×8 global average pooling 반복 후 최댓값 | anomaly localization score → classification score 변환 |

**절차**

학습 단계 : 입력 x₀ → encoder E로 z₀ 압축 → forward diffusion으로 zₜ 생성 → SG network(입력 x₀ 기반)와 SD denoising network(고정)가 함께 노이즈 εθ(zₜ, t, cᵢ) 예측 → 실제 노이즈와의 MSE loss(Eq. 6)로 SG network만 학습.

테스트 단계 : zₜ에서 DDIM 10 step으로 역확산해 ẑ 복원 → decoder D로 재구성 이미지 x̂₀ 생성 → x₀, x̂₀를 동일한 사전학습 ResNet50에 통과시켜 다중 scale feature 추출 → 코사인 유사도로 anomaly map 계산 및 업샘플링·합산 → pixel-level score map과 image-level score 산출.

**베이스라인** : Non-diffusion 계열 PaDiM, DRAEM, RD4AD, UniAD(당시 multi-class SOTA)와 Diffusion 계열 DDPM, LDM을 비교 대상으로 삼음.

---

## 결과

**Table 1 : MVTec-AD Image-level Multi-class Anomaly Classification (AUROC/AP/F1max, %, mean)**

| Method | AUROC-cls (↑) | AP-cls (↑) | F1max-cls (↑) |
|---|---:|---:|---:|
| PaDiM | 84.2 | - | - |
| DRAEM | 88.1 | 94.7 | 92.0 |
| RD4AD | 94.6 | 96.5 | 95.2 |
| UniAD | 96.5 | 98.8 | 96.2 |
| DDPM | 71.9 | 81.6 | 86.6 |
| LDM | 76.6 | 87.8 | 88.1 |
| **DiAD (Ours)** | **97.2** | **99.0** | 96.5 |

**Table 2 : VisA Dataset 정량 비교**

| Metric | DRAEM | UniAD | DDPM | LDM | **DiAD** |
|---|---:|---:|---:|---:|---:|
| AUROC-cls | 79.1 | 85.5 | 54.5 | 56.7 | **86.8** |
| AP-cls | 81.9 | 85.5 | 57.9 | 61.4 | **88.3** |
| F1max-cls | 78.9 | 84.4 | 72.3 | 73.1 | **85.1** |
| AUROC-seg | 91.3 | 95.9 | 79.7 | 86.6 | **96.0** |
| AP-seg | 23.5 | 21.0 | 2.2 | 6.0 | **26.1** |
| F1max-seg | 29.5 | 27.0 | 4.5 | 9.9 | **33.0** |
| PRO | 58.8 | **75.6** | 46.8 | 55.0 | 75.2 |

**Table 3 : MVTec-AD Pixel-level Multi-class Anomaly Segmentation (AUROC/AP/F1max, %, mean)**

| Method | AUROC-seg (↑) | AP-seg (↑) | F1max-seg (↑) |
|---|---:|---:|---:|
| PaDiM | 89.5 | - | - |
| DRAEM | 87.2 | 52.5 | 48.6 |
| RD4AD | 96.1 | 48.6 | 53.8 |
| UniAD | **96.8** | 43.4 | 49.5 |
| DDPM | 75.6 | 13.3 | 19.5 |
| LDM | 85.1 | 27.6 | 31.0 |
| **DiAD (Ours)** | **96.8** | **52.6** | **55.5** |

**Table 4 : MVTec-AD PRO metric**

| Method | DRAEM | UniAD | DDPM | LDM | **DiAD** |
|---|---:|---:|---:|---:|---:|
| PRO | 71.1 | 90.4 | 49.0 | 66.3 | **90.7** |

**Figure / Ablation 분석**

- **Figure 1** : Diffusion 백본별 재구성 실패 양상 비교. DDPM은 forward에서 T step 노이즈를 더하며 원본 카테고리 정보를 완전히 잃어 재구성 시 다른 카테고리 형태로 생성되는 categorical error를 보이고, LDM은 class embedder로 카테고리는 유지하지만 나사·헤이즐넛 같은 객체의 방향·세부 구조(semantic) 정합성을 잃는다. DiAD(Ours)는 두 문제 모두에서 자유롭다.
- **Table 5 (Ablation, 구조 설계)** : SD 단독(=LDM 구조, cls/seg AUROC 79.3/89.5) → SG middle block 추가(95.1/91.1) → SGEB3 skip-connection 추가(95.3/89.1) → SGEB4까지 직접 skip-connection 시 오히려 seg AUROC 하락(93.8/91.2, texture 디테일은 보존되지만 대면적 이상 재구성력이 떨어짐) → BN+ReLU 대신 SFF+IN+SiLU 적용 시 96.7/96.7 → 최종 IN+SiLU 조합으로 97.2/96.8 달성. SFF block과 IN+SiLU 조합이 texture 보존과 대면적 재구성의 트레이드오프를 동시에 해결하는 핵심 요인임을 보여준다.
- **Table 6 (Ablation, feature extractor)** : ResNet50이 classification(AUROC-cls 97.2)에서 최고 성능을, WideResNet101이 segmentation(AUROC-seg 96.9, PRO 91.4)에서 최고 성능을 보임. VGG 계열은 전반적으로 가장 낮은 성능.
- **Figure 6 (Ablation, forward diffusion timestep)** : forward diffusion step 수가 늘수록(완전한 Gaussian noise에 가까워질수록) 이상 재구성 능력(AUROC-seg, AP-seg 등)이 향상되지만, 600 step 미만에서는 노이즈 부족으로 이상 영역을 정상으로 되돌리는 재구성 자체가 불충분해 성능이 급격히 저하된다.

## Findings

- **Finding 1 — Multi-class 단일 모델 SOTA 달성** : DiAD는 카테고리별 개별 모델 없이 하나의 모델로 MVTec-AD 15개 카테고리 전체를 처리하면서도 image-level AUROC 97.2%, pixel-level AUROC 96.8%를 달성해(Table 1, 3) non-diffusion SOTA인 UniAD를 능가한다. 특히 pixel-level AP/F1max에서 UniAD 대비 각각 +9.2/+6.0을 기록해(Table 3) multi-class reconstruction 품질이 위치추정 정밀도에 직결됨을 보여준다.

- **Finding 2 — Semantic Guide가 Diffusion 기반 재구성의 근본 문제를 해결** : 같은 diffusion 계열인 DDPM(AUROC-cls 71.9), LDM(76.6) 대비 DiAD(97.2)는 큰 격차로 앞선다(Table 1). Figure 1의 정성적 분석과 결합하면, 이 격차는 diffusion 모델 자체의 표현력 문제가 아니라 "재구성이 입력의 카테고리·구조 정보를 얼마나 보존하는가"의 문제였음을 시사한다 — SG network가 이 부분을 직접 해결한다.

- **Finding 3 — SFF Block이 Texture vs. Object 트레이드오프를 완화** : Table 5 ablation은 SG decoder를 SD decoder에 단순히 skip-connection만 해서는(SGEB4 직접 연결) segmentation 성능이 오히려 떨어짐을 보여준다. SFF block으로 다중 scale 특징을 융합한 뒤에야 texture 카테고리의 세부 보존과 object 카테고리의 대면적 재구성을 동시에 만족시켰다.

- **Finding 4 — 재구성 기반 anomaly detection에서 diffusion step 수가 성능의 필요조건** : Figure 6에서 forward diffusion step이 600 미만이면 성능이 급격히 저하된다. 이는 DDAD 등 제한된 step만 사용하는 선행 diffusion 기반 방법이 대면적 결함을 재구성하지 못하는 이유를 실험적으로 뒷받침한다.

### 요약 — 이론과 모델이 결과로 어떻게 검증되었는가

DiAD의 핵심 가설 — "denoising 과정을 입력 이미지의 semantic 정보로 guide하면, multi-class 단일 모델에서도 카테고리·구조를 보존하며 이상 영역만 정상으로 재구성할 수 있다" — 는 각 실험에서 개별적으로 검증된다.

- **Table 1, 2, 3, 4** → SG network + SFF 조합이 non-diffusion SOTA(UniAD)와 diffusion 베이스라인(DDPM, LDM) 모두를 능가함을 확인
- **Figure 1 (정성 결과)** → DDPM의 categorical error, LDM의 semantic error가 실제로 관찰되며, DiAD가 이를 회피함을 시각적으로 뒷받침
- **Table 5 (구조 ablation)** → SG network의 각 구성요소(middle block guide, SGEB skip-connection, SFF block, IN+SiLU)가 각각 독립적으로 성능에 기여함을 확인
- **Figure 6 (timestep ablation)** → "충분한 noise step이 있어야 재구성 기반 이상 탐지가 성립한다"는 diffusion 고유의 전제 조건을 실증

각 구성요소가 개별적으로 검증되고, 그 조합이 multi-class 세팅에서의 SOTA 성능으로 이어진다는 점에서 PatchCore가 memory bank + coreset + local aggregation의 시너지를 검증한 방식과 구조적으로 유사하다.

---

## 한계

- **저자가 명시한 한계** : 배경의 불필요한 잡음(background impurities)에 취약해 위치추정·분류 오류가 발생할 수 있다. 저자들은 향후 연구로 (1) 배경 간섭에 강인한 diffusion 모델 탐구, (2) multi-modal 정보 활용, (3) 더 큰 모델을 통한 재구성 성능 향상을 제시한다.
- **Inference 비용 미보고** : PatchCore는 inference 속도(초 단위)를 명시적으로 비교했지만(Table 1 계열 논문), DiAD는 DDIM 10 step 샘플링 + SD/SG 두 네트워크의 forward 연산이 필요함에도 inference 시간·throughput을 정량적으로 보고하지 않는다. Memory bank 기반 nearest-neighbor 탐색(PatchCore)보다 구조적으로 무거울 가능성이 높아, 산업 적용 시 속도 트레이드오프 검증이 필요하다.
- **하이퍼파라미터 민감도** : SG decoder를 SD decoder의 4번째 block에만 연결하는 설계, feature 추출 layer n∈{2,3,4} 선택, timestep T=1,000·DDIM step=10 등은 MVTec-AD/VisA 기준으로 경험적으로 정해졌다. 다른 도메인(예: 의료 영상, 3D 표면)에서도 동일 설정이 최적일지는 검증되지 않았다.
- **Backbone 계열 한정** : ablation은 VGG, ResNet, WideResNet, EfficientNet 등 CNN 계열로 한정되어 있다(Table 6). Feature extractor를 ViT 계열로 바꿨을 때의 거동은 다루지 않는다 — 이는 PatchCore가 지적한 것과 동일한 한계다.

---

### Candidate C 검증(H3/H4)과의 연결

이 조사의 목적은 PatchCore의 3×3 local aggregation이 global pattern anomaly를 놓칠 가능성(H3), per-patch independent scoring이 spatial arrangement anomaly를 놓칠 가능성(H4)에 diffusion 기반 재구성이 대안이 될 수 있는지를 살피는 것이었다. DiAD의 강점(Finding 1~3)은 실제로는 "multi-class 단일 모델에서 카테고리·의미 정보를 보존한 채 재구성할 수 있는가"에 집중되어 있고, H3/H4가 요구하는 "global pattern/spatial arrangement 이상을 patch 단위 비교보다 더 잘 잡는가"를 직접 측정하는 실험(예: 카테고리 내 부품 배치가 뒤바뀐 경우, 반복 패턴이 국소적으로는 정상이지만 전체 배열이 깨진 경우)은 포함하지 않는다. Table 3의 카테고리별 수치를 보면 cable(구조적 배선 오류가 많은 카테고리)에서 DiAD의 AUROC-seg가 96.8로 RD4AD(85.1)·UniAD(97.3) 대비 경쟁력 있지만, 이것만으로 H3/H4를 직접 검증했다고 보기는 어렵다.

> 솔직한 평가 : DiAD가 diffusion의 전역적 재구성 능력을 활용한다는 점에서 H3/H4와 방향은 맞지만, 논문 자체가 "global pattern/spatial arrangement anomaly"를 별도로 정의해 측정하지 않으므로 이 연결은 간접적이다. H3/H4를 직접 검증하려면 MVTec-AD 내에서도 구조적 배치 이상이 많은 카테고리(cable, transistor 등)에 대한 category-wise 재현 실험이 추가로 필요하다.

---
