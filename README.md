# TUK AI-HC Lab — eunseoeunseoo Repository

## 소개

MVTec AD 기준 Industrial Anomaly Detection (IAD) 방법론을 재현하고, PatchCore 재현 결과에서 도출한 원인 가설(H2/H3/H4)을 후속 방법으로 검증하는 중.

---

## Methods Covered

| # | Folder | Paper | Venue | Status |
|---|---|---|---|---|
| 1 | `method1_patchcore/` | Roth et al., Towards Total Recall in Industrial Anomaly Detection | CVPR 2022 | ✅ Reproduced (mean I-AUROC 99.1%) |
| 2 | `method2_winclip/` | Jeong et al., WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | CVPR 2023 | 🔬 진행 중 — zero-shot/1-shot pill 재현 완료, H2 계열 반박 |
| 3 | `method3_diad/` | He et al., DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection | AAAI 2024 | 🔬 학습 진행 중 — epoch 16 재평가: **H3 미결정(개선 추세)**(grid I-AUROC 0.588→0.764), **H4 미결정(역전)**(transistor P-AUROC 0.945→0.923, 0.929 아래로) |

---

## 현재 연구 방향

PatchCore 재현(`method1_patchcore/`)에서 도출한 원인 가설 중 H2 계열은 WinCLIP 재현(`method2_winclip/`)으로 반박되어, 현재는 diffusion 기반 방법인 DiAD(`method3_diad/`)로 H3/H4를 검증하는 중이다. 진행 상세는 아래 weekly brief와 각 `methodN/markdown/`을 참고.

> 참고: 28주차 이후 반복적인 정전으로 실험 환경(노트북) 사용이 간헐적으로 제약되어, 학습 진행 속도가 순수 연산 시간보다 느리다. 세부는 `method3_diad/markdown/setup_notes.md` 참고.

## Weekly Briefs

| Week | Link | 비고 |
|---|---|---|
| 2026-W31 (current) | [meetings/2026-W31_brief.md](meetings/2026-W31_brief.md) | DiAD 재현 실험: H4 지지, H3 미결정 |
| 2026-W28 | [meetings/2026-W28_brief.md](meetings/2026-W28_brief.md) | H2 계열 가설 반박, Candidate C(DiAD)로 전환 |
| 2026-W27 | [meetings/2026-W27_brief.md](meetings/2026-W27_brief.md) | PatchCore 재현, H2/H3/H4 가설 도출 |

---

## Quick Links

- [meetings/](meetings/)
- [method1_patchcore/](method1_patchcore/)
- [method2_winclip/](method2_winclip/)
- [method3_diad/](method3_diad/)
- [related_work/](related_work/)
