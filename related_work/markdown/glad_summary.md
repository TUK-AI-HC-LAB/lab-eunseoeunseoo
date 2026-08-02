# GLAD 논문 요약

PDF: `related_work/paper/ECCV24_GLAD_Towards_Better_Reconstruction_with_Global_and_Local_Adaptive_Diffusion_Models_for_Unsupervised_Anomaly_Detection.pdf`

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
| Reason for investigation | H3(전역 패턴 이상)와 관련된 diffusion 기반 anomaly detection 후속 연구 조사 중, 제목이 "global/local adaptive"로 H3와 직접 관련되어 보여 선정 |

## 핵심 문제

- 기존 diffusion 기반 재구성 방법(DiAD 포함)은 모든 샘플에 동일한 denoising 절차(고정된 timestep, 표준 가우시안 노이즈 가정)를 적용한다.
- 저자들이 지적하는 두 가지 문제: (1) 이미지마다 이상의 재구성 난이도가 다른데 동일한 세팅을 쓰는 것은 비효율적(global 관점), (2) 같은 이미지 안에서도 이상 영역의 노이즈 분포가 정상 영역과 달라서 표준 가우시안 가정이 이상 영역에서는 깨진다(local 관점).

## 핵심 아이디어 / 방법

- Global: 이미지 content와 diffusion prior의 차이를 평가해 샘플마다 적합한 denoising step을 예측(고정 timestep 대신 적응형 timestep).
- Local: 학습 시 synthetic anomaly를 주입해, 이상 영역에서 예측 노이즈가 표준 가우시안 분포를 벗어나도 되도록 학습시키고, 추론 시 spatial-adaptive feature fusion으로 영역별 재구성 품질을 조정.
- 구현상 Stable Diffusion UNet을 파인튜닝하고, VAE(카테고리에 따라 추가 파인튜닝 필요), DINO 기반 feature extractor를 함께 사용(GitHub README 기준).

## 주요 실험 결과

- 데이터셋: MVTec-AD, MPDD, VisA, PCB-Bank(카테고리별 결함이 많은 산업 데이터셋). synthetic anomaly 생성에는 DTD 텍스처 데이터셋 사용.
- 지표: I-AUROC, I-AP, I-F1-max(image-level), P-AUROC, P-AP, P-F1-max, PRO(pixel-level).
- 정확한 수치는 이 문서 작성 시점에 1차 소스(PDF 렌더링 불가, 웹 요약만 확인)로 확인하지 못함 — **unverified**, 필요하면 다음에 GitHub README의 결과 표를 직접 확인해서 채울 것.

| 지표 | 값 | 비고 |
|---|---|---|
| MVTec-AD mean I-AUROC | unverified | 다음에 GitHub README에서 확인 필요 |
| MVTec-AD mean P-AUROC | unverified | 다음에 GitHub README에서 확인 필요 |

## 한계

- 논문이 스스로 인정한 한계: VisA, PCB-Bank처럼 MVTec-AD와 분포가 다른 데이터셋에서는 사전학습 모델과 성능 격차가 커서 VAE를 추가로 파인튜닝해야 함(GitHub README).
- 내가 판단하기에 추가로 보이는 한계: **제목의 "global/local"이 H3가 다루는 "이미지 내 전역 패턴 규칙성(global pattern-regularity) vs 지역 patch 비교"와 같은 의미가 아니다.** GLAD의 "global"은 샘플(이미지) 단위의 denoising step 적응을 말하고, "local"은 이미지 내 영역별 노이즈 분포 적응을 말한다 — 둘 다 재구성 절차의 유연성에 관한 것이지, grid처럼 "국소적으로는 정상이지만 전체 배열이 깨진" 이상을 얼마나 잘 잡느냐를 직접 겨냥한 메커니즘은 아니다.

## 우리 연구와의 연결

- 관련된 원인 가설: H3(grid image-level 판별).
- 이 논문이 H3를 직접 지지/반박하는 근거를 주는가: **간접적으로만.** GLAD는 H3가 요구하는 "전역 패턴 규칙성 위반"을 별도로 측정하지 않는다(DiAD와 마찬가지 한계, `method3_diad/markdown/diad_summary.md` 참고). 다만 "샘플마다 적합한 denoising step이 다르다"는 주장은, 우리 실험에서 grid만 유독 낮게 나오는 현상(`method3_diad/markdown/h3_h4_evaluation.md`)에 대한 하나의 가설을 제공한다 — grid 같은 반복 패턴 이미지가 고정된 timestep(T=1,000)에서는 다른 카테고리보다 재구성이 어려울 수 있다는 것.
- 우리 실험에 그대로 적용하기 어려운 이유: GLAD는 Stable Diffusion UNet + VAE + DINO를 모두 파인튜닝하는 별도 파이프라인이라, 지금 DiAD 재현 위에 바로 이식하기는 어렵다. 8GB GPU 제약에서는 추가 파인튜닝 비용도 감당하기 어려움.
- 다음 판단에 어떻게 쓸 것인가: 지금 당장 GLAD를 도입하기보다는, **"grid가 고정 timestep에서 유독 어려운 카테고리일 수 있다"는 가설을 다음 H3 재평가 해석에 참고**한다. grid I-AUROC가 계속 낮게 정체된다면, timestep을 고정하는 DiAD의 설계 자체가 H3의 반증 방향(=global aggregation 문제가 아니라 diffusion timestep 문제)일 수 있다는 대안 설명으로 고려.

## 인용 표기

본문에서 이 논문을 인용할 때는 weekly brief에서 등장 순서대로 `[N]` bracket number를 붙이고, `## 7. 참고문헌`에는 Google Scholar MLA 형식을 그대로 복사해 넣는다.

MLA: Yao, Hang, et al. "GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection." Computer Vision – ECCV 2024, 2024.
