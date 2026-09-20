# method1_patchcore — result

## 파일
- `baseline_20260628.csv` — MVTec-AD 15개 카테고리, PatchCore(WideResNet50, coreset 10%) class-separated 재현 결과. 컬럼: `instance_auroc`(I-AUROC), `full_pixel_auroc`(P-AUROC, 전체 픽셀), `anomaly_pixel_auroc`(P-AUROC, 이상 영역만).

## 결과 요약
- mean I-AUROC 0.991 / P-AUROC(full) 0.981 — 논문(99.0%/98.1%)과 거의 일치.
- 이후 모든 H3/H4 비교의 baseline 기준점: **grid I-AUROC 0.977, transistor P-AUROC(anomaly) 0.929**.

## 가설 판단
- **H1 (재현 정확도)**: 지지. 논문 대비 ±0.1%p 이내로 재현됨.
- 이 결과에서 파생된 후속 가설(상세 근거는 `method1_patchcore/markdown/baseline_analysis.md`):
  - **H2** (PatchCore가 color-channel 이상에 둔감 — pill I-AUROC 0.968로 상대적으로 낮음) → `method2_winclip`에서 검증, **반박됨**.
  - **H3** (고정 3×3 local aggregation이 grid의 global pattern-regularity 이상을 못 잡음 — grid I-AUROC 0.977로 상대적으로 낮음) → `method4_glad`, `method5_dinomaly`에서 검증 중, **현재까지 지지**.
  - **H4** (per-patch independent scoring이 transistor의 spatial-arrangement 이상을 못 잡음 — transistor P-AUROC 0.929로 상대적으로 낮음) → `method3_diad`, `method4_glad`, `method5_dinomaly`에서 검증 중, **method5_dinomaly의 batch=16 결과에서 지지로 전환**.

> ※ 정정(2026-09-20): 다른 방법과 비교할 때는 같은 정의인 `full_pixel_auroc`(transistor **0.963**)를 써야 합니다. 위 0.929는 `anomaly_pixel_auroc`라 metric이 달라, 위 "method5_dinomaly의 batch=16 결과에서 지지로 전환"은 이 기준에서 성립하지 않습니다(Dinomaly 0.9335). 자세한 비교는 `meetings/2026-W39_brief.md` 5절 실험 3.

## setting
class-separated(카테고리별 개별 memory bank) — `method4_glad`/`method5_dinomaly`(multi-class)와 조건이 다르므로 비교표에서 이 차이를 명시할 것.
