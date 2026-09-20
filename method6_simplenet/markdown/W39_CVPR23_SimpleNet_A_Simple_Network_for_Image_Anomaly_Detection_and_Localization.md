# SimpleNet: A Simple Network for Image Anomaly Detection and Localization

PDF 파일 경로 : `method6_simplenet/paper/W39_CVPR23_SimpleNet_A_Simple_Network_for_Image_Anomaly_Detection_and_Localization.pdf`

---

## Paper Metadata

| Item | Content |
|---|---|
| Title | SimpleNet: A Simple Network for Image Anomaly Detection and Localization |
| Authors | Zhikang Liu, Yiming Zhou, Yuansheng Xu, Zilei Wang |
| Conference / Journal | CVPR |
| Year | 2023 |
| GitHub / Official code | https://github.com/DonaldRR/SimpleNet |
| Reason for investigation | 피드백에서 언급된 4개 방법(GLAD/SimpleNet/Reverse Distillation/Dinomaly) 중 하나. 재구성 계열(DiAD/GLAD/Dinomaly)과 달리 embedding + synthesizing 계열이라 H3/H4에 다른 각도의 증거를 줄 수 있음. |

---

## 1. 기존 방법 3종류의 한계

이 논문의 이상탐지는 학습 때 불량 이미지 없이 정상 이미지만 사용하는 unsupervised(비지도) 설정이다. 
실제 산업 현장에서는 불량 샘플이 드물고 결함 종류도 다양해서 불량 데이터를 충분히 모으기 어렵기 때문이다. 그래서 "정상이 어떤 모습인지"만 학습하고, 테스트 때 정상에서 벗어난 정도로 이상을 판단한다.
논문은 기존 방법을 reconstruction / synthesizing / embedding 세 계열로 나누고 각각의 한계를 지적한다. 

### 1-1. Reconstruction-based
정상 데이터만 학습한 모델이라면 정상 이미지는 잘 복원하지만, 본 적 없는 이상은 제대로 복원하지 못할 것이라는 아이디어를 기반으로 하는 방법이다. (오토인코더, GAN 기반 복원, 이미지 일부를 가리고 채우는 방식 등)

- 인코더-디코더를 $x \mapsto \hat{x} = g(f(x))$ 로 쓰면
  - 정상 입력: $\hat{x} \approx x$ → 복원 오차가 작다.
  - 이상 입력: 결함 영역을 정상처럼 복원하므로 $\hat{x} \neq x$ (결함 위치에서) → 복원 오차가 크다.
- 그래서 pixel-wise reconstruction error(픽셀 단위 복원 오차)를 anomaly map(anomaly score)으로 사용한다: $s_{h,w} = \lVert x_{h,w} - \hat{x}_{h,w} \rVert^2$. (SSIM loss를 함께 쓰는 경우도 많다.)

>> 본 논문에서는 신경망이 너무 잘 generalize하면 anomaly까지 잘 복원해버릴 수 있다고 지적한다. 논문은 이상도 잘 복원되는 경우를 두 가지로 든다.
(1) 결함이 정상 데이터에도 있는 기본 구성 요소(예: 선, 모서리 같은 국소적인 edge)로만 이루어진 경우: 모델이 낯설다고 느끼지 못하고 정상 조각처럼 그대로 그려낸다.
(2) decoder가 "너무 강한" 경우: 정상에서 벗어난 입력이 들어와도 그대로 그려낸다.
두 경우 모두 결함이 있는데도 복원 오차가 작아져서 미탐(이상을 정상으로 잘못 판정)으로 이어진다.
ex) 학습할 때 스크래치가 없었어도 decoder가 충분히 강하면 스크래치가 있는 이미지를 스크래치까지 그대로 복원해버릴 수 있다. 그러면 reconstruction error(복원 오차)가 작아져버리고 정상이라고 잘못 판단할 수 있다.

### 1-2. Synthesizing-based
anomaly 데이터가 없으면 가짜 anomaly를 만들어 정상 이미지에 인위적인 결함을 넣는 아이디어를 기반으로 하는 방법이다. 정상 이미지와 합성 이상 이미지를 구분하도록 분류기를 학습시켜 정상/이상의 결정 경계를 추정한다.

- **CutPaste**: 이미지 패치를 잘라 같은 이미지의 임의 위치에 붙여서 이상을 만들고, CNN이 정상/증강 분포를 구분하도록 학습.
- **DRAEM**: 정상 이미지에 합성 이상을 주입해 end-to-end로 "정상 분포 바로 바깥"의 패턴을 구분하도록 학습.

>> Synthesize한 anomaly가 실제 anomaly처럼 생겼다는 보장이 없는 점을 지적한다. 
실제 결함은 종류가 다양하고 예측 불가능해서 나올 수 있는 모든 결함(outlier)을 빠짐없이 합성해 내는 것은 불가능하다. 합성 이상 feature가 정상 feature에서 멀리 떨어져 있으면 그것으로 학습된 정상 영역은 느슨하게 형성되어, 정상 분포 바로 근처에 있는 미세한 결함이 정상 영역에 포함될 수 있다.

### 1-3. Embedding-based
ImageNet으로 pre-trained된 CNN에서 정상 이미지의 feature를 추출해서 정상 feature 분포를 모델링하는 아이디어를 기반으로 하는 방법이다.
정상 이미지들을 backbone에 넣고 patch(local) feature를 얻어 정상 분포를 만들어 두고, test feature가 그 정상 분포/cluster에서 멀면 anomaly라고 판단한다. (논문 당시 최고 성능을 차지하던 계열)

- **PaDiM** - 정상 feature를 위치별 다변량 가우시안 분포로 모델링하고 Mahalanobis distance로 점수를 낸다.
- **PatchCore** - 정상 patch feature 중 가장 대표적인 것만 골라(coreset) memory bank에 저장하고, 테스트 feature와 가장 가까운 정상 feature까지의 거리를 anomaly score로 쓴다.
- **CS-Flow / CFLOW-AD / DifferNet** - normalizing flow로 정상 feature 분포를 Gaussian으로 변환. 논문은 flow가 full-size feature map만 다룰 수 있고(다운샘플 불가) coupling layer가 일반 conv보다 메모리를 몇 배 더 쓴다고 지적한다.
- **증류(distillation) 계열 (Reverse Distillation 등)** - 정상 샘플만으로 student가 고정된 teacher의 출력을 따라하게 학습하고, 이상 입력에서는 둘의 출력 차이가 커지는 것을 이용. 입력이 teacher와 student를 모두 통과해야 해서 계산량이 두 배가 된다고 지적한다.

>> 지적 1. 도메인 편향
PatchCore 같은 방법은 보통 ImageNet pretrained backbone을 사용하는데 ImageNet은 대략적으로 일반 자연 이미지들이라 산업 이미지와는 분포가 다르다. ImageNet에서 학습한 feature를 그대로 사용하면 feature 표현 자체가 ImageNet 도메인에 편향되어 있어 우리 데이터에 맞지 않을 수 있다.
>> 지적 2. 계산량과 메모리
Embedding-based 방법들은 정상 분포를 모델링하기 위해 추가적인 통계 계산이 들어간다. (PaDiM은 공분산 역행렬 계산, PatchCore는 memory bank에서 최근접 이웃 탐색)
이런 방식이 계산량과 메모리 사용량이 커서 특히 실시간·엣지 환경의 실제 산업 적용에 부담이 될 수 있다고 지적한다.

### 1-4. SimpleNet : Synthesizing-based + Embedding-based
Embedding 계열처럼 pre-trained backbone의 feature를 쓰되, synthesizing 계열처럼 가짜 이상을 만들어 discriminator를 학습시킨다. 다만 이상을 **이미지가 아닌 feature space에서** 만든다.

| 기존 문제 | SimpleNet의 해결 방법 |
| --- | --- |
| ImageNet feature의 도메인 편향 | **Feature Adaptor** (feature를 우리 데이터 도메인으로 변환) |
| 이미지 공간에서 그럴듯한 합성 이상을 만들기 어려움 (결함이 다양해서) | **Feature space에서 anomaly 생성** |
| 합성 이상이 정상에서 너무 멀어 정상 영역이 느슨해짐 | 정상 feature에 **가우시안 noise**를 더하고 σ를 적절히 조절해 정상 영역을 촘촘하게 만듦 |
| 복잡한 통계 계산 (공분산, memory bank 탐색) | **간단한 MLP discriminator** (추론 시 한 번 통과) |

---

## 2. SimpleNet 전체 흐름

구성 요소는 4개: Feature Extractor(사전학습 backbone), Feature Adaptor, Anomalous Feature Generator, Discriminator. 이 중 **Anomalous Feature Generator는 학습 때만 쓰고 추론 때는 버린다.** (논문 Fig. 3)

**학습 (정상 이미지만 사용)**
1. 정상 이미지 $x^i$ → Feature Extractor $F_\phi$ (ImageNet 사전학습 ResNet 계열, **고정**) → 위치별 local feature $o^i_{h,w}$
2. Feature Adaptor $G_\theta$ → adapted feature $q^i_{h,w}$ (우리 데이터 도메인으로 변환, **학습 대상**)
3. Anomalous Feature Generator: $q^i_{h,w}$ 에 Gaussian noise를 더해 가짜 이상 feature $q^{i-}_{h,w}$ 생성
4. Discriminator $D_\psi$ (**학습 대상**)가 $q$ (정상: 양수 목표)와 $q^-$ (가짜 이상: 음수 목표)의 위치별 정상도 점수를 출력
5. truncated $\ell_1$ loss로 $G_\theta$, $D_\psi$를 함께 학습 (Algorithm 1에서 `F = F.detach()`로 backbone은 갱신되지 않음)

**추론 (테스트 이미지)**
1. $x \to F_\phi \to G_\theta \to q_{h,w}$ (noise 생성 없이 한 갈래로만 통과)
2. $s_{h,w} = -D_\psi(q_{h,w})$ → $h \times w$ anomaly map → 입력 해상도로 키우고 가우시안 필터로 부드럽게 함
3. 이미지 anomaly score = anomaly map의 최댓값

즉 extractor + adaptor + discriminator를 이어 붙인 하나의 conv 네트워크이고, memory bank·최근접 탐색·공분산 같은 별도 통계 모듈이 없다.

---

## 3. SimpleNet 요소

각 요소마다 "한마디로"로 큰 그림을 먼저 잡고, 이어서 하는 일을 말과 식으로 함께 적었다.

### 3.1 Feature Extractor
: 이미지를 "칸마다 특징 벡터가 붙은 격자"로 바꾸는 단계이다.

ImageNet으로 미리 학습된 ResNet 계열(WideResNet50)에 이미지를 통과시킨다. 이 네트워크는 학습하지 않고 고정한다. 앞쪽 층은 선·질감 같은 세밀한 정보, 뒤쪽 층은 더 큰 구조 정보를 담고 있는데, 마지막 층은 ImageNet 분류에 너무 치우쳐 있어서 **중간 층 일부(level 2, 3)** 만 쓴다. 층 $l$에서 나온 특징 지도는

$$\phi^{l,i} \in \mathbb{R}^{H_l \times W_l \times C_l}$$

(가로·세로 $H_l \times W_l$ 칸, 칸마다 $C_l$개 숫자)이다.

**① 주변까지 같이 본다 (식 1)**: 한 칸의 특징만 보면 너무 좁아서, 위치 $(h,w)$ 주변 $p \times p$ 칸을 이웃으로 정한다. 기본값은 $p=3$이다.

$$\mathcal{N}_p^{(h,w)}=\{(h',w') \mid h'\in[h-\lfloor p/2\rfloor,\dots,h+\lfloor p/2\rfloor],\ w'\in[w-\lfloor p/2\rfloor,\dots,w+\lfloor p/2\rfloor]\}$$

**② 이웃을 평균 낸다 (식 2)**: 이웃 칸들의 특징을 $f_{agg}$(여기서는 adaptive average pooling)로 합쳐 그 칸의 값으로 쓴다.

$$z^{l,i}_{h,w} = f_{agg}\big(\{\phi^{l,i}_{h',w'} \mid (h',w') \in \mathcal{N}_p^{(h,w)}\}\big)$$

**③ 두 층을 이어 붙인다 (식 3)**: 층마다 격자 크기가 다르므로, 작은 쪽을 가장 큰 크기 $(H_0, W_0)$에 맞게 키운 뒤 같은 칸의 특징끼리 나란히 붙인다.

$$o^i = f_{cat}\big(\text{resize}(z^{l',i}, (H_0, W_0)) \mid l' \in L\big)$$

**결과 (식 4)**: 위 과정 전체를 $o^i = F_\phi(x^i)$ 로 줄여 쓴다. $o^i$는 $H_0 \times W_0$ 칸에 칸마다 $C$개 숫자가 붙은 것이고, WideResNet50 기준 $C = 512 + 1024 = 1536$이다. 이후 단계는 이 칸 하나하나($o_{h,w} \in \mathbb{R}^C$)를 독립적인 샘플처럼 다룬다.

**논문 근거**: level 2+3 조합이 가장 좋고(99.6/98.1), 패치 크기는 3이 가장 좋다(Table 2, Fig. 6).

### 3.2 Feature Adaptor
: ImageNet용 특징을 우리 공장 이미지에 맞게 살짝 바꿔 주는 층이다.

backbone은 ImageNet 사진으로 학습돼서 산업 이미지와는 결이 다르다(도메인 편향). 그래서 뽑은 특징을 그대로 쓰지 않고, 학습 가능한 작은 층 $G_\theta$를 거쳐 우리 데이터에 맞게 옮긴다 (식 5).

$$q^i_{h,w}=G_\theta(o^i_{h,w})$$

논문은 이 층을 **bias 없는 FC 층 1개**라고 설명한다. 입력과 출력의 크기가 같다($\mathbb{R}^C \to \mathbb{R}^C$).

**논문 근거**: (Table 3) 이 단순한 층이 가장 좋다(99.6/98.1). adaptor를 빼면 99.2/97.9, 비선형을 넣어 복잡하게 만들면 98.3/97.2로 오히려 나빠진다(논문은 과적합 가능성을 언급). Fig. 4에서는 가짜 이상과 함께 학습하면 adapted feature의 차원별 표준편차가 더 일정해져서 특징 공간이 촘촘해진다고 보인다.

### 3.3 Anomalous Feature Generator
: 정상 특징을 살짝 흐트러뜨려서 "가짜 이상 특징"을 만드는 단계이다. (학습 때만)

adapted feature의 각 숫자에 작은 무작위 값(가우시안 noise)을 더한다 (식 6). 이미지에 결함을 그리는 것이 아니라 특징 자체를 흔드는 것이다.

$$q^-_{h,w}=q_{h,w}+\epsilon,\qquad \epsilon\in\mathbb{R}^C,\ \text{각 원소는 서로 독립인 } \mathcal N(\mu,\sigma^2)\quad(\text{구현: } \mu=0,\ \sigma=0.015)$$

이렇게 하는 이유: 결함은 종류가 너무 다양해서 이미지 공간에서 모든 이상을 만들 수는 없지만, 특징 공간에서는 단순한 noise로 "정상 바로 근처"에 가짜 이상 샘플을 만들 수 있다.

**noise 크기($\sigma$)가 핵심**이다.
- $\sigma$가 크면 가짜 이상이 정상에서 너무 멀어져 결정 경계가 느슨해지고, 정상에 가까운 미세한 결함을 놓친다.
- $\sigma$가 너무 작으면 학습이 불안정하고 판별기가 정상 특징으로 일반화하지 못한다.
- 논문은 **$\sigma = 0.015$** 근처가 최적이라고 보고한다(Fig. 5).

### 3.4 Discriminator
: 특징 하나를 보고 "정상 같은지"를 점수로 매기는 작은 판별기이다.

칸마다 특징 벡터를 받아 숫자 하나(정상도 점수)를 낸다.

$$D_\psi(q_{h,w}) \in \mathbb{R}$$

정상 특징에는 높은(양수) 점수, 가짜 이상 특징에는 낮은(음수) 점수를 내도록 학습한다. 구조는 2층 MLP: Linear → BatchNorm → LeakyReLU(0.2) → Linear이고, 마지막에 sigmoid 없이 숫자를 그대로 점수로 쓴다.

### 3.5 손실 함수와 학습
: "정상은 충분히 높게, 가짜 이상은 충분히 낮게만 되면 된다"는 기준으로 학습한다.

한 칸의 loss는 정상 항과 가짜 이상 항의 합이다 (식 7).

$$l^i_{h,w}=\max\big(0,\ th^+ - D_\psi(q^i_{h,w})\big)+\max\big(0,\ -th^- + D_\psi(q^{i-}_{h,w})\big),\qquad th^+=0.5,\ th^-=-0.5$$

- 앞 항: 정상 점수가 $0.5$ 이상이면 0, 아니면 부족한 만큼 벌점.
- 뒤 항: 가짜 이상 점수가 $-0.5$ 이하면 0, 아니면 넘친 만큼 벌점.
- 기준을 넘긴 샘플은 loss가 0이라 더 밀어붙이지 않는다. 논문은 이 기준값 $th^\pm$를 "과적합을 막는 절단 항"이라고 설명한다.

전체 목적함수는 이 값을 모든 이미지·칸에 대해 더하고 칸 수로 나눈 것이다 (식 8). adaptor($\theta$)와 판별기($\psi$)를 함께 학습하고 backbone은 고정이다.

$$\mathcal{L}=\min_{\theta,\psi}\sum_{x^i\in\mathcal X_{train}}\sum_{h,w}\frac{l^i_{h,w}}{H_0 \times W_0}$$

**논문 근거**: (Table 3) cross-entropy를 쓰면 99.4/97.8로, 이 loss(99.6/98.1)보다 I-AUROC 0.2%p, P-AUROC 0.3%p 낮았다.

**학습 설정(논문)**: Adam, 학습률 adaptor 0.0001 / 판별기 0.0002, weight decay 0.00001, 160 epoch, 배치 크기 4, 클래스별로 따로 학습(class-separated).

### 3.6 추론과 점수 계산
: 판별기가 "정상 같지 않다"고 느끼는 정도를 이상 점수로 쓴다.

테스트 이미지를 backbone → adaptor → 판별기에 통과시킨다. noise를 만드는 부분은 쓰지 않는다.

**① 이상 점수 (식 9)**: 정상도 점수의 부호를 뒤집는다. 정상이라고 볼수록 낮고, 이상하다고 볼수록 높다.

$$s^i_{h,w}=-D_\psi(q^i_{h,w})$$

**② 이상 지도 (식 10)**: 칸마다 나온 점수를 모으면 이상 지도(anomaly map)가 된다. 입력 이미지 크기로 키운 뒤 가우시안 필터($\sigma=4$)로 부드럽게 한다.

$$S_{AL}(x^i)=\{s^i_{h,w} \mid (h,w) \in W_0 \times H_0\}$$

**③ 이미지 전체 점수 (식 11)**: 지도에서 **가장 높은 값 하나**로 이상 여부를 정한다. 이상 영역이 작든 크든, 가장 크게 반응한 지점만 보면 되기 때문이다.

$$S_{AD}(x^i)=\max_{(h,w)} s^i_{h,w}$$

---

## 4. 결과

### 4-1. MVTec AD (class-separated, Table 1: I-AUROC% / P-AUROC%)

| 계열 | 방법 | 평균 I-AUROC | 평균 P-AUROC |
|---|---|---|---|
| Reconstruction | AE-SSIM | 87 | 69.4 |
| Reconstruction | RIAD | 91.7 | 94.2 |
| Synthesizing | DRÆM | 98.0 | 97.3 |
| Synthesizing | CutPaste | 96.1 | 96.0 |
| Embedding | CS-Flow | 98.7 | - |
| Embedding | PaDiM | 95.8 | 97.5 |
| Embedding | RevDist | 98.5 | 97.8 |
| Embedding | PatchCore | 99.1 | 98.1 |
| **Ours** | **SimpleNet** | **99.6** | **98.1** |

- I-AUROC: 15개 중 9개 클래스에서 최고. 텍스처 평균 99.8, 물체 평균 99.5, 전체 99.6. 같은 WideResNet50 backbone을 쓴 PatchCore(오차 0.9%) 대비 오차가 0.4%로 **55.5% 감소**했다고 주장.
- P-AUROC: 전체 평균 98.1, 물체 평균 98.4(당시 최고), 15개 중 4개 클래스에서 최고.
- **추론 속도**: 3080ti GPU에서 **77 FPS**, PatchCore 대비 약 8배 빠르다고 보고(Fig. 2).
- **단일 클래스 신규성 탐지 (CIFAR-10, Table 5)**: 평균 I-AUROC 86.5.

### 4-2. ablation (절제 실험)
| 항목 | 결과 |
|---|---|
| 사용 level (Table 2) | level 2+3이 99.6/98.1로 최고. level 3만 써도 99.2/97.5, level 1을 포함하면 오히려 하락(1+2+3: 99.1/98.1) |
| 패치 크기 (Fig. 6) | $p=3$이 최적 (국소 정보와 전체 문맥의 균형) |
| adaptor (Table 3) | FC 1층(99.6/98.1) > adaptor 없음(99.2/97.9) > 복잡한 adaptor(98.3/97.2) |
| Loss (Table 3) | truncated $\ell_1$ 99.6/98.1 > CE 99.4/97.8 |
| σ (Fig. 5) | σ=0.015 근처 최적 |
| backbone (Table 4) | ResNet18 98.3/95.7, ResNet50 99.6/98.0, ResNet101 99.2/97.6, WideResNet50 99.6/98.1 — 대체로 안정적 |

---

## 5. 우리 연구에 쓸 수 있는 부분

- **계열**: GLAD/DiAD/Dinomaly(재구성 기반)와 달리 embedding + synthesizing 혼합형이다. 복원 없이 feature 공간에서 정상/가짜 이상을 구분하는 discriminator만 학습하므로, 복원 계열의 identity mapping 문제가 원리상 생기지 않는다.
- **PatchCore와의 관계**: 같은 class-separated 설정, 같은 WideResNet50 layer2·3 feature를 쓰는 직접 비교 대상이다. 차이는 정상 분포를 memory bank에 저장하느냐(PatchCore), adaptor + discriminator로 학습하느냐(SimpleNet)이다.
- **H3/H4 재현 결과** (`source/result/README.md`): 15개 카테고리 평균 I-AUROC 0.9963 / P-AUROC 0.9787 / PRO 0.9127로 논문 수치(99.6/98.1)를 재현했다.
  - H3 (grid I-AUROC): PatchCore 0.977 vs SimpleNet 0.9992(최고 epoch) / 0.9858(마지막 epoch) → 지지
  - H4 (transistor P-AUROC): PatchCore 0.963(full-pixel 기준 정정값) vs SimpleNet 0.9682(최고 epoch) / 0.8995(마지막 epoch) → epoch 선택에 따라 갈리는 조건부 지지
- **참고할 점**: 재현 시 epoch 선택 방식에 따라 결론이 바뀔 수 있으니, 다른 method와 비교할 때 "테스트셋 기준으로 고른 최고 epoch인지, 마지막 epoch인지"를 함께 기록해야 한다.
