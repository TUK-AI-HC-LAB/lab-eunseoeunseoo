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

## setting
- class-separated(카테고리별 별도 모델·discriminator, 공식 repo `main.py` 그대로). PatchCore(class-separated)와 setting 동일, GLAD/Dinomaly(uni)와는 다름.
- 원 논문 설정 유지: wideresnet50(layer2·layer3), batch=8, meta_epochs=40, gan_epochs=4, imagesize=288(resize 329), noise_std=0.015, seed=0. 로컬 환경 대응으로 `--gpu 0`, `num_workers=0`만 변경(`source/config/run_mvtec_all.sh`).
- 실행: 노트북(RTX 5060/8GB), 2026-09-18 12:04 시작 → 2026-09-19 09:24 `results.csv` 저장(약 21시간 20분).
- 논문 보고치와의 대조는 아직 안 함(논문 요약 `markdown/` 미작성).
