# method2_winclip — result

## 파일
- `pill_zeroshot.csv` — WinCLIP zero-shot, pill 카테고리(167 test images).
- `pill_1shot.csv` — WinCLIP+ 1-shot(정상 참조 이미지 1장 추가), pill 카테고리.

## 결과 요약
| 조건 | I-AUROC | PatchCore(pill) 대비 |
|---|---|---|
| zero-shot | 0.812 | -15.6%p |
| 1-shot | 0.853 | -11.5%p |
| PatchCore(pill) | 0.968 | 기준 |

## 가설 판단
- **H2** (CLIP의 언어 기반 color semantic이 PatchCore의 shape bias 문제를 우회할 것) → **반박**. zero-shot·1-shot 모두 PatchCore에 못 미침.
- **H2-few-shot** (시각 참조 1장 추가 시 격차 해소) → **반박**. +4.1%p 개선은 있었으나 격차의 절반도 못 좁힘. 선형 외삽 시 PatchCore를 따라잡으려면 참조 이미지 3~4장 이상 필요 — 그 지점부터는 zero/few-shot의 실용적 이점(카테고리별 재학습 불필요)이 희석됨.
- **다음 판단**: H2 계열(Candidate A) 보류, H3/H4(Candidate C, diffusion/reconstruction 기반) 우선순위로 이동.

상세 근거·실험 설정은 `method2_winclip/markdown/winclip_zeroshot_analysis.md` 참고.

## 원 논문 보고치와 대조 (MVTec-AD 15개 카테고리 평균, %)
논문 PDF(`method2_winclip/paper/`)의 Table 1(AC: AUROC/AUPR/F1-max)과 재현 CSV(`mvtec_all_zeroshot.csv`, `mvtec_all_1shot.csv`)의 15개 카테고리 평균을 비교함.

| 조건 | 지표 | 논문 | 재현 | 차이 |
|---|---|---|---|---|
| zero-shot | I-AUROC | 91.8 | 90.41 | -1.39 |
| zero-shot | AUPR | 96.5 | 95.64 | -0.86 |
| zero-shot | F1-max | 92.9 | 92.12 | -0.78 |
| 1-shot (WinCLIP+) | I-AUROC | 93.1±2.0 | 91.45 | -1.65 |
| 1-shot (WinCLIP+) | AUPR | 96.5±0.9 | 95.92 | -0.58 |
| 1-shot (WinCLIP+) | F1-max | 93.7±1.1 | 92.53 | -1.17 |

- 세 지표 모두 재현이 논문보다 0.6~1.7%p 낮음. 논문 1-shot 값은 5개 random seed 평균±표준편차이고, 재현은 seed=10 한 번의 결과라 1-shot 차이(-1.65)는 논문 표준편차(2.0) 안쪽임.
- 원인은 확인하지 않음.
- 이 대조는 이미지 단위 분류(AC)만 해당함. 이 재현은 pixel 단위(AS)를 구현하지 않았음.

## setting
class-separated, zero-shot/few-shot(카테고리별 재학습 없음) — 다른 방법들과 비교축 자체가 다름(학습 데이터 접근량이 0~1장).
