# TUK AI-HC Lab — eunseoeunseoo Repository

## 소개

MVTec AD 기준 Industrial Anomaly Detection (IAD) 방법론을 재현하고, PatchCore 재현 결과에서 도출한 원인 가설(H2/H3/H4)을 후속 방법으로 검증하는 중.

---

## Methods Covered

| # | Folder | Paper | Venue | Status |
|---|---|---|---|---|
| 1 | `method1_patchcore/` | Roth et al., Towards Total Recall in Industrial Anomaly Detection | CVPR 2022 | ✅ Reproduced (mean I-AUROC 99.1%) |
| 2 | `method2_winclip/` | Jeong et al., WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | CVPR 2023 | ✅ Reproduced — zero-shot/1-shot pill 재현 완료, H2 계열 반박. grid/transistor 전체 재현은 착수 전(H3만, 픽셀 레벨 미구현) |
| 3 | `method3_diad/` | He et al., DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection | AAAI 2024 | ⏸ 중단(2026-09-07, 교수님 지시) — epoch 58까지 학습, H3/H4 최종 판단은 미결정으로 남김(3개 지점 raw 결과: `method3_diad/source/result/eval_results_epoch7/16/34.csv`) |
| 4 | `method4_glad/` | Yao et al., GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection | ECCV 2024 | ✅ 완료(데스크톱, RTX 5070/12GB, batch=32, 20000 step, 원 논문과 동일 설정) — grid I-AUROC 0.996(H3 지지), transistor P-AUROC 0.710(H4 반박, 비단조 악화). 논문 자체 보고치(Table S7) 대비도 전 지표 격차 있음(원인 미확인) |
| 5 | `method5_dinomaly/` | Guo et al., Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection | CVPR 2025 | ✅ multi-class 완료(batch=16, 원 논문 설정) — grid I-AUROC 0.9983, transistor P-AUROC 0.9335로 PatchCore 상회(H3·H4 모두 지지). 🔬 class-separated(PatchCore와 동일 setting, 15개 카테고리)는 데스크톱에서 진행 중(5/15 완료). 노트북에서 batch=16 VRAM 오버서브스크립션(전용 메모리 포화+공유 메모리 spill로 5~8배 저속) 진단·해결(gradient checkpointing, 코드는 로컬에 있고 커밋 전) |
| 6 | `method6_simplenet/` | Liu et al., SimpleNet: A Simple Network for Image Anomaly Detection and Localization | CVPR 2023 | 🔬 학습 중(노트북, RTX 5060/8GB) — class-separated, 공식 repo 그대로(batch=8, meta_epochs=40, gan_epochs=4) |

---

## 현재 연구 방향

### 진행 경과
1. PatchCore 재현(`method1_patchcore/`) → grid, transistor에서의 약점으로부터 원인 가설 H2/H3/H4 도출.
2. WinCLIP 재현(`method2_winclip/`) → H2 계열 반박.
3. DiAD 재현(`method3_diad/`) → H3/H4를 epoch 7/16/34 3개 지점에서 검증, 미결정인 채로 2026-09-07 교수님 지시로 중단.
4. GLAD 재현(`method4_glad/`) → 데스크톱에서 원 논문과 동일 설정(batch=32)으로 완주. H3 지지, H4 반박(비단조 악화).
5. Dinomaly 재현(`method5_dinomaly/`) → multi-class 완료(H3·H4 모두 지지, 논문 수준 재현 확인). class-separated는 진행 중.
6. SimpleNet 재현(`method6_simplenet/`) → 현재 단계, 착수.

### 현재 상태 — H3/H4 누적 현황
| 방법 | setting | grid I-AUROC (H3) | transistor P-AUROC (H4) | 판단 |
|---|---|---|---|---|
| PatchCore (기준) | class-separated | 0.977 | 0.929 | — |
| DiAD (epoch34, 재현) | multi-class | 0.654 | 0.922 | H3 미결정, H4 미결정(반박에 가까움) |
| GLAD (checkpoint20000, 완주) | multi-class | **0.996** | 0.710 | H3 지지, H4 반박(비단조 악화) |
| Dinomaly (batch=16, 완주) | multi-class | 0.9983 | 0.9335 | H3 지지, H4 지지 |
| Dinomaly (class-separated) | class-separated(PatchCore와 동일) | 진행 중(데스크톱, 5/15) | 진행 중 | — |

DiAD·GLAD(둘 다 diffusion 기반) 모두 H4(transistor)를 PatchCore 대비 반박/미결정으로 내는 경향이 반복되는 반면, Dinomaly(diffusion 없는 재구성 방법)는 H4까지 지지 — "재구성/생성 기반 접근 일반"이 아니라 diffusion 계열 특유의 한계일 가능성이 제기됨(추가 검증 필요). GLAD·DiAD 둘 다 논문 자체 보고치 대비 재현 격차가 있어(GLAD는 batch까지 맞췄는데도 격차 있음, 원인 미확인) 이 판단은 잠정적.

### 다음 방향
2026-09-07 교수님 지시로 DiAD 재현을 중단하고, 피드백에서 언급된 4개 방법(GLAD/SimpleNet/Reverse Distillation/Dinomaly) 재현으로 전환했다. GLAD 완주 후 기기 역할이 재배치되어 현재는:
- **데스크톱(RTX 5070/12GB)**: Dinomaly class-separated(`method5_dinomaly/`, PatchCore와 동일 setting) 학습 중, 5/15 완료.
- **노트북(RTX 5060/8GB)**: SimpleNet(`method6_simplenet/`, class-separated) 학습 중.

노트북에서 Dinomaly class-separated 시도 중 batch=16이 VRAM 오버서브스크립션으로 5~8배 저속화되는 문제를 진단·해결(gradient checkpointing)했으나, 이 수정은 아직 커밋되지 않은 상태로 로컬에만 있음(다음 커밋에서 반영 예정) — 데스크톱은 12GB VRAM이라 이 문제 없이 진행 중.

Reverse Distillation은 아직 착수 전.

### 참고
- 진행 상세는 아래 weekly brief와 각 `methodN/markdown/`을 참고.
- 2026-09-03: 귀국하여 정전으로 인한 실험 환경 제약이 해소되었고, 이날부터 연구를 재개했다.

## Weekly Briefs

| Week | Link | 비고 |
|---|---|---|
| 2026-W39 | 작성 중(미커밋) | GLAD 완주(H3 지지/H4 반박), Dinomaly class-separated VRAM 오버서브스크립션 진단·해결. 아직 초안 — 커밋 전 |
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
- [method6_simplenet/](method6_simplenet/)
- [related_work/](related_work/)
