# TUK AI-HC Lab — eunseoeunseoo Repository

## 소개

MVTec AD 기준 Industrial Anomaly Detection (IAD) 방법론을 재현하고, PatchCore 재현 결과에서 도출한 원인 가설(H2/H3/H4)을 후속 방법으로 검증하는 중.

---

## Methods Covered

| # | Folder | Paper | Venue | Status |
|---|---|---|---|---|
| 1 | `method1_patchcore/` | Roth et al., Towards Total Recall in Industrial Anomaly Detection | CVPR 2022 | ✅ Reproduced (mean I-AUROC 99.1%) |
| 2 | `method2_winclip/` | Jeong et al., WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | CVPR 2023 | ✅ Reproduced — zero-shot/1-shot pill 재현 완료, H2 계열 반박 |
| 3 | `method3_diad/` | He et al., DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection | AAAI 2024 | ⏸ 중단(2026-09-07, 교수님 지시) — epoch 58까지 학습, H3/H4 최종 판단은 미결정으로 남김(3개 지점 raw 결과: `method3_diad/source/result/eval_results_epoch7/16/34.csv`) |
| 4 | `method4_glad/` | Yao et al., GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection | ECCV 2024 | 🔬 학습 중(데스크톱, RTX 5070/12GB) — `train_batch_size=2`, `gradient_accumulation_steps=16`(effective batch 32)로 원 논문과 동일한 effective batch로 재현 중 |
| 5 | `method5_dinomaly/` | Guo et al., Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection | CVPR 2025 | ✅ multi-class 완료(batch=16, 원 논문 설정) — grid I-AUROC 0.9983, transistor P-AUROC 0.9335로 PatchCore 상회(H3·H4 모두 지지). 🔬 class-separated(15개 카테고리) 재현 진행 중(노트북, RTX 5060/8GB) |

---

## 현재 연구 방향

### 진행 경과
1. PatchCore 재현(`method1_patchcore/`) → grid, transistor에서의 약점으로부터 원인 가설 H2/H3/H4 도출.
2. WinCLIP 재현(`method2_winclip/`) → H2 계열 반박.
3. DiAD 재현(`method3_diad/`) → H3/H4를 epoch 7/16/34 3개 지점에서 검증, 미결정인 채로 2026-09-07 교수님 지시로 중단.
4. GLAD 재현(`method4_glad/`) → 현재 단계. DiAD 대신 GLAD를 직접 재현·실험.

### 현재 상태 — H3/H4 (DiAD, epoch 7→16→34)
| 가설 | 지표 | 추이 | 판단 |
|---|---|---|---|
| H3 | grid I-AUROC | 0.588 → 0.764 → 0.654 | 미결정 (개선 후 재하락) |
| H4 | transistor P-AUROC | 0.945 → 0.923 → 0.922 | 미결정 (반박에 가까움) |

두 지표 모두 train loss가 계속 낮아지는 구간에서도 함께 개선되지 않아, "학습 부족"만으로는 PatchCore 대비 열위를 설명하기 어렵다는 쪽으로 판단이 이동했다.

### 이번 주 업데이트 — GLAD 비교 (2026-W37)
GLAD[ECCV 2024]의 supplementary(multi-category, DiAD와 동일 조건)에 인용된 DiAD 원 논문 mean I-AUROC(97.2)와 우리 재현치(epoch 34 기준 0.870)를 비교한 결과, 우리 재현이 논문 수준 학습에 아직 못 미침을 확인했다. 이는 W36에서 "구조적 한계 쪽에 무게가 실린다"고 봤던 판단과 긴장 관계에 있다 — grid/transistor 열위가 구조적 한계인지, 아직 남은 학습 부족 때문인지 다시 열린 질문이 됐다. 세부는 `meetings/2026-W37_brief.md`.

### 다음 방향
2026-09-07 교수님 지시로 DiAD 재현을 중단하고, 피드백에서 언급된 4개 방법(GLAD/SimpleNet/Reverse Distillation/Dinomaly) 재현으로 전환했다. 현재 두 기기에서 병렬로 진행 중이다:
- **데스크톱(RTX 5070/12GB)**: GLAD(`method4_glad/`) 학습 — `train_batch_size=2`, `gradient_accumulation_steps=16`(effective batch 32)로 원 논문과 동일한 effective batch 재현.
- **노트북(RTX 5060/8GB)**: Dinomaly(`method5_dinomaly/`) — multi-class(batch=16, 원 논문 설정) 학습 완료, grid·transistor 모두 PatchCore 상회. 현재 class-separated(15개 카테고리) 재현 진행 중.

SimpleNet, Reverse Distillation은 아직 착수 전.

### 참고
- 진행 상세는 아래 weekly brief와 각 `methodN/markdown/`을 참고.
- 2026-09-03: 귀국하여 정전으로 인한 실험 환경 제약이 해소되었고, 이날부터 연구를 재개했다.

## Weekly Briefs

| Week | Link | 비고 |
|---|---|---|
| 2026-W38 (current) | [meetings/2026-W38_brief.md](meetings/2026-W38_brief.md) | GLAD·Dinomaly 병행 재현 착수 — Dinomaly batch=4 첫 결과(grid I-AUROC 0.9975로 H3 지지, transistor P-AUROC 0.9238로 H4 근접·미결정), GLAD는 데스크톱에서 effective batch 32로 재학습 중 |
| 2026-W37 | [meetings/2026-W37_brief.md](meetings/2026-W37_brief.md) | GLAD 비교로 DiAD 재현 mean I-AUROC가 논문 보고치에 크게 못 미침을 확인 — H3/H4 "구조적 한계 vs 학습 부족" 판단 재검토 |
| 2026-W36 | [meetings/2026-W36_brief.md](meetings/2026-W36_brief.md) | DiAD 재현 실험 epoch 7/16/34 종합: H3/H4 모두 미결정(비관적 쪽으로 이동), epoch 24 재현 불가 확인 |
| 2026-W31 | [meetings/2026-W31_brief.md](meetings/2026-W31_brief.md) | DiAD 재현 실험: H4 지지, H3 미결정 |
| 2026-W28 | [meetings/2026-W28_brief.md](meetings/2026-W28_brief.md) | H2 계열 가설 반박, Candidate C(DiAD)로 전환 |
| 2026-W27 | [meetings/2026-W27_brief.md](meetings/2026-W27_brief.md) | PatchCore 재현, H2/H3/H4 가설 도출 |

---

## Quick Links

- [meetings/](meetings/)
- [method1_patchcore/](method1_patchcore/)
- [method2_winclip/](method2_winclip/)
- [method3_diad/](method3_diad/)
- [method4_glad/](method4_glad/)
- [method5_dinomaly/](method5_dinomaly/)
- [related_work/](related_work/)
