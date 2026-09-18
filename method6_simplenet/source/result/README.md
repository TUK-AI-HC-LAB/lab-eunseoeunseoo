# method6_simplenet — result

(학습 진행 중 — 완료 후 작성 예정)

- class-separated(카테고리별 별도 모델, 공식 repo `main.py` 그대로) 15개 카테고리 순차 학습 중.
- 원 논문 설정 유지(batch=8, meta_epochs=40, gan_epochs=4, imagesize=288 등), `--gpu 0`·`num_workers=0`만 로컬 환경에 맞게 변경.
- 진행 로그: `source/result/train_stdout_v2.log`, GPU 모니터: `source/result/simplenet_mvtec_all_v2.csv`.
