# WinCLIP Zero-Shot Reproduction — pill 카테고리 (H2 검증)

## 실험: pill 카테고리 zero-shot WinCLIP vs PatchCore

- commit: `39ae367` (mala-lab/WinCLIP)
- sh: `method2_winclip/source/run_pill_zeroshot.sh`
- csv: `method2_winclip/source/WinCLIP/results/pill_zeroshot.csv`

---

### 질문

Zero-shot WinCLIP이 pill 카테고리에서 PatchCore의 I-AUROC(0.968)를 넘어서는가?

### 가설

**H2** (`baseline_analysis.md`에서 도출): PatchCore의 memory bank distance 기반 score는 WideResNet50이 ImageNet pretraining으로 인해 shape-biased되어 있어 color-channel anomaly(예: pill 오염)에 둔감하다. CLIP은 image-text pair로 학습되어 color를 포함한 semantic attribute를 encode하므로, WinCLIP은 이 color bias 문제를 우회할 수 있을 것이다.

**예측**: H2가 맞다면, WinCLIP의 pill I-AUROC가 PatchCore의 pill I-AUROC(0.968)에 근접하거나 이를 초과해야 한다.

### 설정

| Parameter | Value |
|---|---|
| 구현체 | [mala-lab/WinCLIP](https://github.com/mala-lab/WinCLIP) (unofficial, CVPR'24 InCTRL 논문의 WinCLIP baseline으로 사용된 코드) |
| CLIP backbone | OpenCLIP ViT-B-16-plus-240, LAION-400M (`vit_b_16_plus_240-laion400m_e31-8fb26589.pt`) |
| 설정 | zero-shot (shot=0), 텍스트 프롬프트만 사용, 시각 참조 없음 |
| 카테고리 | pill (167 test images) |
| Seed | 10 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8GB) |

**구현체 선택 근거**: 원 논문 저자는 공식 코드를 공개하지 않았다. 두 비공식 재구현체를 검토한 결과, `caoyunkang/WinClip`은 자체 재현 결과가 논문 대비 평균 21.6%p 낮게 나와(70.17% vs 91.81%) 신뢰하기 어려웠다. `mala-lab/WinCLIP`은 CVPR'24 InCTRL 논문에서 WinCLIP baseline 산출에 실제로 사용된 코드로, aggregate 재현치가 논문과 0.6%p 차이로 근접해 이쪽을 채택했다 (자세한 비교는 `winclip_summary.md`의 Implementation Notes 참고).

### 기대 결과

H2가 맞다면 WinCLIP pill I-AUROC ≥ 0.968 (PatchCore 수준 또는 그 이상).

### 실제 결과

| 지표 | PatchCore (pill) | WinCLIP zero-shot 논문 published | WinCLIP zero-shot 오늘 재현 | Raw Path |
|---|---|---|---|---|
| I-AUROC | 0.968 | 0.791 | **0.812** | `method2_winclip/source/WinCLIP/results/pill_zeroshot.csv` |
| AUPR | — | — | 0.963 | 〃 |
| F1-max | — | — | 0.916 | 〃 |

오늘 재현치(0.812)는 논문 published 값(0.791)과 ±2.1%p 이내로 근접하며, 이는 `mala-lab/WinCLIP` 구현체가 신뢰할 만하다는 것을 다시 한번 뒷받침한다 (aggregate 수준뿐 아니라 pill 단일 카테고리에서도 fidelity 확인).

### 해석

**H2는 반박된다.** WinCLIP zero-shot의 pill I-AUROC(0.812)는 PatchCore(0.968)를 15.6%p 밑돈다. 언어만으로 정의한 normal/anomalous 개념이, 학습 없이는 PatchCore가 축적한 memory bank distance 매칭을 이기지 못했다. "CLIP이 color 의미를 encode하므로 shape bias 문제를 우회할 것"이라는 예측 방향 자체는 원 논문의 published per-category 표(pill zero-shot 0.791)에서 이미 근거가 약했고, 오늘 독립 재현으로 재확인되었다.

---

## 다음 판단

zero-shot 조건에서는 H2가 성립하지 않는다. 다만 이 결과가 "CLIP 기반 접근 전체의 실패"를 의미하지는 않는다 — WinCLIP+(few-shot, 1~4장의 정상 참조 이미지 추가)는 언어 점수에 시각적 유사도 점수를 더하는 방식이라, 순수 언어 기반 zero-shot과는 다른 메커니즘이다. WinCLIP+ few-shot을 pill에서 확인하는 것이 H2의 "language+minimal visual reference" 변형 버전을 검증하는 다음 실험이 될 수 있다 (이번 주는 진행하지 않음, 다음 주 후보).

만약 few-shot도 PatchCore에 못 미친다면, H2 계열 가설 전체를 접고 H3/H4(Candidate C, diffusion 기반)로 우선순위를 옮기는 것이 타당하다.
