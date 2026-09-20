# method6_simplenet — result

## 파일
- `eval_results_meta40.csv` — 15개 카테고리 최종 결과(공식 코드가 저장한 `results.csv`를 컬럼명만 다른 method와 맞춰 변환, 4자리 반올림). 컬럼: `clsname,i_auroc,p_auroc,pro`. SimpleNet 공식 코드는 I-AUROC/P-AUROC/PRO만 계산해서 AP·F1 컬럼은 없음.
- `results_raw_official.csv` — 공식 코드가 그대로 저장한 원본. 세 번째 컬럼 헤더는 `anomaly_pixel_auroc`로 돼 있지만 값은 PRO임(학습 로그의 `PRO-AUROC` 값과 일치함을 screw·leather로 확인).
- `train_stdout_v2.log` — 이번 실행의 표준출력(meta-epoch별 I-AUROC/P-AUROC/PRO 전부 포함, `*.log`라 git 제외). `train_stdout.log`는 2026-09-17 screw 학습 중 중단된 이전 시도.
- `simplenet_mvtec_all_v2.csv` — 학습 중 GPU 온도/util/VRAM 로그(10초 간격).

## 결과 (class-separated, 15개 카테고리)
| 카테고리 | I-AUROC | P-AUROC | PRO |
|---|---|---|---|
| screw | 0.9877 | 0.9924 | 0.9664 |
| pill | 0.9875 | 0.9832 | 0.9358 |
| capsule | 0.9789 | 0.9892 | 0.9338 |
| carpet | 0.9960 | 0.9786 | 0.8738 |
| grid | 0.9992 | 0.9816 | 0.9333 |
| tile | 0.9982 | 0.9639 | 0.9034 |
| wood | 1.0000 | 0.9401 | 0.8427 |
| zipper | 0.9989 | 0.9886 | 0.9599 |
| cable | 1.0000 | 0.9744 | 0.9061 |
| toothbrush | 1.0000 | 0.9852 | 0.9337 |
| transistor | 1.0000 | 0.9682 | 0.9388 |
| metal_nut | 1.0000 | 0.9868 | 0.8825 |
| bottle | 1.0000 | 0.9800 | 0.9023 |
| hazelnut | 0.9979 | 0.9764 | 0.8115 |
| leather | 1.0000 | 0.9917 | 0.9661 |
| **mean** | **0.9963** | **0.9787** | **0.9127** |

## 결과 선택 방식 (읽을 때 주의)
공식 코드(`simplenet.py` `train()`)는 카테고리마다 meta-epoch 40번을 돌며 매번 **테스트셋 전체를 평가**하고, I-AUROC가 최고인 epoch(동률이면 P-AUROC가 더 높은 epoch)의 기록을 최종 결과로 저장함. 위 표는 그 best 기록이고, 마지막 epoch(39) 값과 다를 수 있음. H3/H4 카테고리의 두 값:

| 카테고리/지표 | best epoch 기준(위 표) | 마지막 epoch(39) 기준 |
|---|---|---|
| grid I-AUROC | 0.9992 | 0.9858 |
| transistor P-AUROC | 0.9682 | 0.8995 |

## PatchCore 대비 H3/H4 판단 (setting 동일: class-separated)
| 가설 | 지표 | PatchCore | SimpleNet(best) | SimpleNet(마지막 epoch) |
|---|---|---|---|---|
| H3 | grid I-AUROC | 0.977 | 0.9992 | 0.9858 |
| H4 | transistor P-AUROC | 0.929 | 0.9682 | 0.8995 |

- **H3**: **지지**. best·마지막 epoch 모두 PatchCore 상회.
- **H4**: best epoch 기준으로는 **지지**(0.9682>0.929)이나, 마지막 epoch 기준으로는 PatchCore에 못 미침(0.8995<0.929). epoch 선택 방식에 따라 판단이 갈리므로 조건부 지지로 기록.

> ※ 정정(2026-09-20): 위 PatchCore 0.929는 anomaly-only 정의이고 SimpleNet의 `full_pixel_auroc`는 full-pixel 정의라 metric이 다르다. 같은 full-pixel 기준 PatchCore는 **0.963**이며, best epoch 0.9682는 근소하게 높고(+0.005) 마지막 epoch 0.8995는 낮아 "조건부 지지"라는 판단 구조는 유지되지만 우위 폭은 0.929 기준(+0.039)보다 훨씬 작다. 자세한 비교는 `meetings/2026-W39_brief.md` 5절 실험 3.

## 원 논문 보고치와 대조 (논문 Table 1, MVTec-AD, class-separated, I-AUROC%/P-AUROC%)
논문 PDF(`method6_simplenet/paper/`) p.6 Table 1의 SimpleNet 열을 읽어 대조함(텍스처 평균 99.8/97.5, 물체 평균 99.5/98.4가 개별 값의 평균과 일치함을 확인). 재현 값은 위 결과 표(best epoch 기준)를 %로 환산.

| 카테고리 | 논문 I | 재현 I | 차이 | 논문 P | 재현 P | 차이 |
|---|---|---|---|---|---|---|
| carpet | 99.7 | 99.6 | -0.1 | 98.2 | 97.9 | -0.3 |
| grid | 99.7 | 99.9 | +0.2 | 98.8 | 98.2 | -0.6 |
| leather | 100.0 | 100.0 | 0.0 | 99.2 | 99.2 | 0.0 |
| tile | 99.8 | 99.8 | 0.0 | 97.0 | 96.4 | -0.6 |
| wood | 100.0 | 100.0 | 0.0 | 94.5 | 94.0 | -0.5 |
| bottle | 100.0 | 100.0 | 0.0 | 98.0 | 98.0 | 0.0 |
| cable | 99.9 | 100.0 | +0.1 | 97.6 | 97.4 | -0.2 |
| capsule | 97.7 | 97.9 | +0.2 | 98.9 | 98.9 | 0.0 |
| hazelnut | 100.0 | 99.8 | -0.2 | 97.9 | 97.6 | -0.3 |
| metal_nut | 100.0 | 100.0 | 0.0 | 98.8 | 98.7 | -0.1 |
| pill | 99.0 | 98.8 | -0.2 | 98.6 | 98.3 | -0.3 |
| screw | 98.2 | 98.8 | +0.6 | 99.3 | 99.2 | -0.1 |
| toothbrush | 99.7 | 100.0 | +0.3 | 98.5 | 98.5 | 0.0 |
| transistor | 100.0 | 100.0 | 0.0 | 97.6 | 96.8 | -0.8 |
| zipper | 99.9 | 99.9 | 0.0 | 98.9 | 98.9 | 0.0 |
| **mean** | **99.6** | **99.63** | +0.03 | **98.1** | **97.87** | -0.23 |

- mean I-AUROC는 논문과 같은 값(99.6), mean P-AUROC는 0.23%p 낮음. 카테고리별 차이는 I-AUROC 최대 0.6%p(screw), P-AUROC 최대 0.8%p(transistor) 이내로, 논문 수치를 재현함.
- H3/H4 카테고리: grid I-AUROC 논문 99.7 / 재현 99.9, transistor P-AUROC 논문 97.6 / 재현 96.8(best epoch 기준). 마지막 epoch(39) 기준 transistor P-AUROC는 89.95라 논문 값과 차이가 큼.
- 논문이 최종 수치를 어떤 epoch 기준으로 골랐는지는 확인하지 않음.
- 논문 본문(p.5)은 "Training epochs 160, batchsize 4"라고 쓰고, 이번에 쓴 공식 `run.sh`는 meta_epochs=40×gan_epochs=4(=160 epoch), batch=8임(batch만 다름).

## setting
- class-separated(카테고리별 별도 모델·discriminator, 공식 repo `main.py` 그대로). PatchCore(class-separated)와 setting 동일, GLAD/Dinomaly(uni)와는 다름.
- 공식 `run.sh` 설정 유지: wideresnet50(layer2·layer3), batch=8, meta_epochs=40, gan_epochs=4, imagesize=288(resize 329), noise_std=0.015, seed=0. 로컬 환경 대응으로 `--gpu 0`, `num_workers=0`만 변경(`source/config/run_mvtec_all.sh`).
- 실행: 노트북(RTX 5060/8GB), 2026-09-18 12:04 시작 → 2026-09-19 09:24 `results.csv` 저장(약 21시간 20분).
