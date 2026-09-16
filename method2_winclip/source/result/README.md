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

## setting
class-separated, zero-shot/few-shot(카테고리별 재학습 없음) — 다른 방법들과 비교축 자체가 다름(학습 데이터 접근량이 0~1장).
