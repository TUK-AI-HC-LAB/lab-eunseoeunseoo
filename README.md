# TUK AI-HC Lab — eunseoeunseoo Repository

## 소개

MVTec AD 기준 Industrial Anomaly Detection (IAD) 방법론을 재현하고, PatchCore 재현 결과에서 도출한 원인 가설(H2/H3/H4)을 후속 방법으로 검증하는 중.

---

## Methods Covered

| # | Folder | Paper | Venue | Status |
|---|---|---|---|---|
| 1 | `method1_patchcore/` | Roth et al., Towards Total Recall in Industrial Anomaly Detection | CVPR 2022 | ✅ Reproduced (mean I-AUROC 99.1%) |
| 2 | `method2_winclip/` | Jeong et al., WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation | CVPR 2023 | ✅ Reproduced — zero-shot/1-shot pill 재현 완료, H2 계열 반박. MVTec 15개 카테고리 zero-shot/1-shot 전체 실행 결과 CSV는 추가됨(`method2_winclip/source/result/mvtec_all_*.csv`), H2 판단은 pill 기준 그대로. H3만, 픽셀 레벨 미구현 |
| 3 | `method3_diad/` | He et al., DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection | AAAI 2024 | ⏸ 중단(2026-09-07, 교수님 지시) — epoch 58까지 학습, H3/H4 최종 판단은 미결정으로 남김(3개 지점 raw 결과: `method3_diad/source/result/eval_results_epoch7/16/34.csv`) |
| 4 | `method4_glad/` | Yao et al., GLAD: Towards Better Reconstruction with Global and Local Adaptive Diffusion Models for Unsupervised Anomaly Detection | ECCV 2024 | ✅ 완료(데스크톱, RTX 5070/12GB, batch=32, 20000 step, 원 논문과 동일 설정) — grid I-AUROC 0.996(H3 지지), transistor P-AUROC 0.710(H4 반박, 비단조 악화). 논문 자체 보고치(Table S7) 대비도 전 지표 격차 있음(원인 미확인) |
| 5 | `method5_dinomaly/` | Guo et al., Dinomaly: The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection | CVPR 2025 | ✅ 완료 — multi-class(batch=16, 원 논문 설정) grid I-AUROC 0.9983 / transistor P-AUROC 0.9335, class-separated(PatchCore와 동일 setting, 15개 카테고리) grid 1.0000 / transistor 0.9493(mean I-AUROC 99.75%, P-AUROC 98.38%)로 PatchCore 상회(H3·H4 모두 지지). 노트북 batch=16 VRAM 오버서브스크립션(5~8배 저속)은 gradient checkpointing으로 해결 |
| 6 | `method6_simplenet/` | Liu et al., SimpleNet: A Simple Network for Image Anomaly Detection and Localization | CVPR 2023 | ✅ 완료(노트북, RTX 5060/8GB, 약 21시간) — class-separated 15개 카테고리, 공식 repo 설정 그대로(batch=8, meta_epochs=40, gan_epochs=4). mean I-AUROC 0.9963 / P-AUROC 0.9787 / PRO 0.9127. 결과는 best epoch 기준이라 마지막 epoch 값과 다를 수 있음(상세는 result README). 논문 Table 1(99.6/98.1) 대비 mean I-AUROC 동일, P-AUROC -0.23%p로 재현 |
| 7 | `method7_reverse_distillation/` | Deng & Li, Anomaly Detection via Reverse Distillation from One-Class Embedding | CVPR 2022 | ✅ 완료 — class-separated 15개 카테고리, batch=16·epoch=200(원 논문 설정). mean I-AUROC 0.987 / P-AUROC 0.978 / PRO 0.939로 논문 보고치와 거의 일치 |

---

## 현재 연구 방향

### 진행 경과
1. PatchCore 재현(`method1_patchcore/`) → grid, transistor에서의 약점으로부터 원인 가설 H2/H3/H4 도출.
2. WinCLIP 재현(`method2_winclip/`) → H2 계열 반박.
3. DiAD 재현(`method3_diad/`) → H3/H4를 epoch 7/16/34 3개 지점에서 검증, 미결정인 채로 2026-09-07 교수님 지시로 중단.
4. GLAD 재현(`method4_glad/`) → 데스크톱에서 원 논문과 동일 설정(batch=32)으로 완주. H3 지지, H4 반박(비단조 악화).
5. Dinomaly 재현(`method5_dinomaly/`) → multi-class·class-separated 모두 완료(H3·H4 모두 지지).
6. SimpleNet 재현(`method6_simplenet/`) → class-separated 완료(H3 지지, H4는 epoch 선택에 따라 갈림).
7. Reverse Distillation 재현(`method7_reverse_distillation/`) → class-separated 완료(H3 지지, H4 미결정에 가까움).

### 현재 상태 — H3/H4 누적 현황
| 방법 | setting | grid I-AUROC (H3) | transistor P-AUROC (H4) | 판단 |
|---|---|---|---|---|
| PatchCore (기준) | class-separated | 0.977 | 0.929 | — |
| DiAD (epoch34, 재현) | multi-class | 0.654 | 0.922 | H3 미결정, H4 미결정(반박에 가까움) |
| GLAD (checkpoint20000, 완주) | multi-class | **0.996** | 0.710 | H3 지지, H4 반박(비단조 악화) |
| Dinomaly (batch=16, 완주) | multi-class | 0.9983 | 0.9335 | H3 지지, H4 지지 |
| Dinomaly (class-separated, 완주) | class-separated(PatchCore와 동일) | 1.0000 | 0.9493 | H3 지지, H4 지지 |
| SimpleNet (best epoch, 완주) | class-separated(PatchCore와 동일) | 0.9992 | 0.9682 | H3 지지, H4 조건부 지지(마지막 epoch 기준 transistor P-AUROC 0.8995로 PatchCore 미달) |
| Reverse Distillation (완주) | class-separated(PatchCore와 동일) | 1.000 | 0.927 | H3 지지, H4 미결정에 가까움(PatchCore 0.929 대비 -0.002) |

> ※ 정정(2026-09-20): 위 표의 PatchCore 0.929는 `anomaly_pixel_auroc`(이상 이미지만 모아 계산)이고 다른 방법의 값은 전체 이미지 기준 full-pixel AUROC라 metric이 다릅니다. 같은 full-pixel 기준 PatchCore는 **0.963**이며, 이 기준에서 Dinomaly의 "H4 지지"(0.9335·0.9493)와 Reverse Distillation의 "거의 동일"(0.927, 실제 차이 -0.036)은 성립하지 않습니다. DiAD·GLAD의 반박/미결정 방향은 그대로이고 격차만 커지며, SimpleNet은 best epoch(0.9682)만 근소하게 높고 마지막 epoch(0.8995)은 낮습니다. 자세한 비교는 `meetings/2026-W39_brief.md` 5절 실험 3.

DiAD·GLAD(둘 다 diffusion 기반)는 H4(transistor)를 PatchCore 대비 반박/미결정으로 내는 반면, Dinomaly는 H4까지 지지, SimpleNet은 epoch 선택에 따라 갈리고 Reverse Distillation은 미결정에 가까워 H4 판단은 방법마다 갈린다. H3(grid)는 diffusion 계열 중 GLAD와 diffusion이 아닌 Dinomaly·SimpleNet·Reverse Distillation 모두에서 지지로 반복된다. GLAD·DiAD는 논문 자체 보고치 대비 재현 격차가 있어(GLAD는 batch까지 맞췄는데도 격차 있음, 원인 미확인) 이 판단은 잠정적.

### 다음 방향
2026-09-07 교수님 지시로 DiAD 재현을 중단하고, 피드백에서 언급된 4개 방법(GLAD/SimpleNet/Reverse Distillation/Dinomaly) 재현으로 전환했고, 4개 모두 재현을 마쳤다(GLAD는 데스크톱, SimpleNet·Reverse Distillation은 노트북, Dinomaly는 노트북 multi-class와 데스크톱 class-separated).

- 재현 결과를 종합해 H3/H4 판단을 갱신하는 것이 다음 단계다.
- SimpleNet 논문 요약(`method6_simplenet/markdown/`)은 아직 미작성(보고치 대조는 result README에 완료).
- 피드백 4(노트북 batch=16 저속 원인) 대응 raw 로그는 `method5_dinomaly/source/result/`에 있다.

### 참고
- 진행 상세는 아래 weekly brief와 각 `methodN/markdown/`을 참고.
- 2026-09-03: 귀국하여 정전으로 인한 실험 환경 제약이 해소되었고, 이날부터 연구를 재개했다.

## Weekly Briefs

| Week | Link | 비고 |
|---|---|---|
| 2026-W39 (current) | [meetings/2026-W39_brief.md](meetings/2026-W39_brief.md) | 7개 방법 재현 완주 후 같은 metric으로 비교(H3 지지, H4는 full-pixel 기준 재해석 — PatchCore 0.929 정정), GLAD 재현 신뢰도 문제, Dinomaly batch=16 저속 원인(VRAM 오버서브스크립션) 확정. 6·7번 피드백은 미착수 |
| 2026-W38 | [meetings/2026-W38_brief.md](meetings/2026-W38_brief.md) | GLAD·Dinomaly 병행 재현 착수 — Dinomaly batch=4 첫 결과(grid I-AUROC 0.9975로 H3 지지, transistor P-AUROC 0.9238로 H4 근접·미결정), GLAD는 데스크톱에서 effective batch 32로 재학습 중 |
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
- [method7_reverse_distillation/](method7_reverse_distillation/)
- [related_work/](related_work/)
