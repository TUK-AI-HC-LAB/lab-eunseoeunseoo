# WinCLIP Zero-Shot / 1-Shot Reproduction — pill 카테고리 (H2 검증)

## 실험 1: pill 카테고리 zero-shot WinCLIP vs PatchCore

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

## 실험 2: pill 카테고리 WinCLIP+ 1-shot vs PatchCore

- commit: `39ae367` (mala-lab/WinCLIP) + 로컬 수정 (아래 "구현 변경" 참고)
- sh: `method2_winclip/source/run_pill_1shot.sh`
- csv: `method2_winclip/source/WinCLIP/results/pill_1shot.csv`

### 질문

WinCLIP+ (1-shot, 정상 참조 이미지 1장 추가)가 pill 카테고리에서 PatchCore의 I-AUROC(0.968)를 넘어서는가? — 실험 1(zero-shot)에서 H2가 반박된 뒤, "언어만으로는 부족했지만 최소한의 시각 참조를 더하면 충분한가"를 확인하는 후속 실험.

### 가설

H2의 확장판(H2-few-shot): 언어 기반 color semantic에 시각적 유사도(1장의 정상 참조 이미지와의 patch-level 거리)를 더하면, zero-shot의 한계(0.812)를 넘어 PatchCore(0.968) 수준에 도달할 수 있다.

### 설정

실험 1과 동일 (구현체, backbone, 카테고리, seed, GPU). 차이점은 `shot=1`로 설정하고 WinCLIP+ 경로(언어 점수 + 시각 유사도 점수 평균)를 사용한 것.

**구현 변경**: `mala-lab/WinCLIP`의 few-shot `forward()`가 정상 참조 이미지를 정확히 4장(`image[1]`~`image[4]`)으로 하드코딩되어 있어 `shot=1`로는 `IndexError`가 발생했다. `open_clip/model.py`의 `WinCLIP.forward()`를 임의의 shot 개수를 받도록 일반화했다 (`ref_images = image[1:]`로 변경, 나머지 gallery 구성 로직은 원래부터 개수에 무관하게 동작해서 손대지 않음). 로직 자체(코사인 유사도, harmonic mean aggregation)는 바꾸지 않았다.

### 기대 결과

H2-few-shot이 맞다면 WinCLIP+ 1-shot pill I-AUROC ≥ 0.968.

### 실제 결과

| 지표 | PatchCore (pill) | WinCLIP zero-shot | WinCLIP+ 1-shot | Raw Path |
|---|---|---|---|---|
| I-AUROC | 0.968 | 0.812 | **0.853** | `method2_winclip/source/WinCLIP/results/pill_1shot.csv` |
| AUPR | — | 0.963 | 0.972 | 〃 |
| F1-max | — | 0.916 | 0.916 | 〃 |

### 해석

**H2-few-shot도 반박된다.** 정상 참조 이미지 1장을 추가하니 I-AUROC가 0.812 → 0.853으로 +4.1%p 개선되어, 시각 참조가 실제로 도움이 된다는 것은 확인했다. 하지만 여전히 PatchCore(0.968)보다 11.5%p 낮다. 언어 점수와 최소한의 시각 유사도 점수를 합쳐도, PatchCore가 대규모 coreset memory bank로 확보한 매칭 정밀도에는 못 미친다.

---

## 다음 판단

zero-shot과 1-shot 두 조건 모두에서 H2 계열 가설이 반박되었다. 시각 참조를 늘릴수록(4-shot 등) 격차가 더 줄어들 가능성은 남아있지만, 개선 폭(+4.1%p/1장)을 선형 외삽하면 PatchCore를 따라잡으려면 참조 이미지가 최소 3~4장 이상 필요하다는 뜻이고, 그 지점부터는 "zero/few-shot의 실용적 이점"이라는 애초의 동기(카테고리별 재학습 없이 대응)가 희석된다.

H2 계열(Candidate A)에 대한 이번 주 판단: **일단 보류하고 H3/H4(Candidate C, diffusion 기반, DiAD)로 우선순위를 옮기는 쪽이 타당해 보인다.** 다만 이건 잠정 판단이며, 다음 미팅에서 논의가 필요하다 (아래 브리프 8장 질문 참고).
