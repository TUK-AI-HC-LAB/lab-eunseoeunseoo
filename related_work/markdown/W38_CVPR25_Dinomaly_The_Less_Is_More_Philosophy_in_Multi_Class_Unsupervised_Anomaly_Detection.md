# Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection — 논문 조사 노트

PDF 파일 경로 : `related_work/paper/W38_CVPR25_Dinomaly_The_Less_Is_More_Philosophy_in_Multi_Class_Unsupervised_Anomaly_Detection.pdf`

---

## Paper Metadata

| Item | Content |
|---|---|
| Title | Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection |
| Authors | Jia Guo, Shuai Lu, Weihang Zhang, Fang Chen, Huiqi Li, Hongen Liao |
| Conference / Journal | CVPR |
| Year | 2025 |
| Paper link | https://arxiv.org/abs/2405.14325 |
| GitHub / Official code | https://github.com/guojiajeremy/dinomaly |
| Reason for investigation | 피드백에서 언급된 4개 방법(GLAD/SimpleNet/Reverse Distillation/Dinomaly) 중 하나. Foundation Transformer(DINOv2류) 기반 재구성 방식이라 diffusion 계열(DiAD/GLAD)과 메커니즘이 완전히 달라, H3/H4에 다른 각도의 증거를 줄 수 있을 것으로 기대. diffusion 없이 단일 forward pass로 재구성한다는 점에서 8GB GPU에서도 상대적으로 가벼울 것으로 예상되어 우선 조사 대상으로 선정.

---

## 0. 이 논문이 속한 분야

(직접 읽고 작성)

## 1. 핵심: DiAD/GLAD와 무엇이 다른가

(직접 읽고 작성) — diffusion 기반 재구성(DiAD/GLAD)과 달리 frozen foundation Transformer 특징 위에서 재구성한다는 것이 핵심 차이로 보임. 어느 축에서 갈라지는지 정리.

## 2. 용어 (이 논문에서 쓰는 그대로)

- (직접 읽고 작성 — 예: Foundation Transformer, Noisy Bottleneck, Linear Attention, Loose Reconstruction 등 4가지 핵심 구성요소로 보임)

## 3. 방법 — 단계별로, 예시와 함께

### 3.1 (단계 1 이름)
### 3.2 (단계 2 이름)

## 4. 실험 결과

- (직접 읽고 작성 — 데이터셋 MVTec-AD/VisA/Real-IAD, 핵심 수치 표)

| 지표 | 값 | 비고 |
|---|---|---|
|  |  |  |

## 5. 내 연구와의 연결

- **계열이 같은가/다른가**: (직접 읽고 작성)
- **접근이 다른가**: diffusion 기반 반복 denoising 없이 단일 forward pass 재구성으로 보임 — 학습/추론 비용 측면에서 DiAD/GLAD와 크게 다를 가능성.
- **우리 피드백/가설과의 연결**: H3(grid 전역 패턴)/H4(transistor 배치)에서 diffusion 계열과 다른 메커니즘이 어떻게 다르게(혹은 비슷하게) 실패·성공하는지가 관전 포인트.
- **참고할 점 / 주의할 점**: (직접 읽고 작성 — 8GB GPU에서의 실제 재현 난이도 확인 필요)

## 인용 표기

본문에서 이 논문을 인용할 때는 weekly brief에서 등장 순서대로 `[N]` bracket number를 붙이고, `## 7. 참고문헌`에는 Google Scholar MLA 형식을 그대로 복사해 넣는다.
