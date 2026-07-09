# TUK AI-HC Lab — eunseoeunseoo Repository

## 소개

MVTec AD 기준 Industrial Anomaly Detection (IAD) 방법론을 재현하고, PatchCore 재현 결과에서 도출한 원인 가설(H2/H3/H4)을 후속 방법으로 검증하는 중.

---

## Methods Covered

| # | Folder | Paper | Venue | Status |
|---|---|---|---|---|
| 1 | `method1_patchcore/` | Roth et al., Towards Total Recall in Industrial Anomaly Detection | CVPR 2022 | ✅ Reproduced (mean I-AUROC 99.1%) |
| 2 | `method2_winclip/` | Jeong et al., WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | CVPR 2023 | 🔬 진행 중 — zero-shot/1-shot pill 재현 완료, H2 계열 반박 |

---

## Progress Summary

### 2026-W28 (current) — H2 검증: WinCLIP zero-shot/1-shot vs PatchCore

- WinCLIP zero-shot을 pill 카테고리에서 재현 (mala-lab 구현체): I-AUROC 0.812, PatchCore(0.968) 대비 -15.6%p → H2(zero-shot 버전) 반박
- WinCLIP+ 1-shot 재현: I-AUROC 0.853 (zero-shot 대비 +4.1%p, PatchCore 대비 여전히 -11.5%p) → H2 few-shot 확장판도 반박
- 공식 코드가 없는 논문의 비공식 구현체 간 재현 편차 확인 (caoyunkang 70.2% vs mala-lab 91.2% aggregate) → 신뢰할 수 있는 쪽으로 전환
- H2 계열 가설 전체 반박에 따라 Candidate C(DiAD, diffusion 기반, H3/H4 검증용)로 전환 결정 — 재현 착수 전 논문 읽는 중
- → [meetings/2026-W28_brief.md](meetings/2026-W28_brief.md)
- → [method2_winclip/markdown/winclip_zeroshot_analysis.md](method2_winclip/markdown/winclip_zeroshot_analysis.md)
- → [related_work/markdown/diad_summary.md](related_work/markdown/diad_summary.md)

### 2026-W27 — PatchCore 재현 + 한계 분석

- PatchCore baseline을 MVTec AD 15개 카테고리 전체에 재현: I-AUROC 99.1%, P-AUROC 98.1% (논문과 일치)
- 결과표에서 두 가지 실패 유형(detection failure / localization failure) 발견, H2/H3/H4 가설 도출
- → [meetings/2026-W27_brief.md](meetings/2026-W27_brief.md)
- → [method1_patchcore/markdown/baseline_analysis.md](method1_patchcore/markdown/baseline_analysis.md)

---

## Quick Links

- [meetings/](meetings/)
- [method1_patchcore/](method1_patchcore/)
- [method2_winclip/](method2_winclip/)
- [related_work/](related_work/)
