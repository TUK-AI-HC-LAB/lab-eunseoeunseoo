# GLAD 재현 착수 — 환경 설정 기록

## 하드웨어 인벤토리
- 노트북: RTX 5060 Laptop, VRAM 8GB.
- 데스크탑: RTX 5070, VRAM 12GB.
- W37 피드백(`feedback/lab-eunseoeunseoo/2026-W37.md` 1.1절)에서 GLAD/SimpleNet/Reverse Distillation/Dinomaly 재현 및 이 두 머신의 스펙 조사를 요청받아 기록.

## GLAD 재현 환경 구축 (노트북, RTX 5060/8GB)

공식 repo(`https://github.com/hyao1/GLAD`)를 `method4_glad/source/GLAD`에 클론. multi-category 설정(`main_multi.py`, `train_multi.sh` 기반, MVTec-AD 15개 카테고리 공동학습 — DiAD와 동일 조건)으로 재현.

### 환경/의존성 이슈
- `bitsandbytes==0.37.2`(원 저자 pin)가 Windows에서 CUDA 인식을 못 함(Linux 전용 빌드) → `0.49.2`로 교체. DiAD 때(`method3_diad/markdown/setup_notes.md`)와 동일한 이슈.
- `diffusers==0.20.0.dev0`(dev 프리릴리즈, PyPI 미배포) → 안정 버전 `0.20.0`으로 대체.
- `transformers`/`huggingface-hub`를 최신으로 깔면 `diffusers==0.20.0`과 API 불일치 → `transformers==4.30.2`, `huggingface-hub==0.16.4`로 고정.
- `kornia`, `pandas`, `transformers`가 실제로는 import되지만 `requirements.txt`에 누락돼 있었음 → 별도 설치.
- torch는 원 pin(`torch==2.0.1+cu117`) 대신 `2.11.0+cu128`(RTX 5060/Blackwell 계열 지원용, DiAD 때와 동일한 이유로 최신 cu128 빌드 필요) 사용.

### 코드 버그 및 패치
- `main_multi.py` line 529: `os.path.join('model', instance_data_dir.split('/')[-1] + '_' + output_dir + f"_{seed}")` — `--output_dir`을 절대경로로 주면 콜론(`:`)이 경로 중간에 들어가 Windows에서 `OSError: [WinError 123]` 발생. 원래 설계상 `output_dir`은 짧은 태그 문자열이고, 체크포인트는 항상 cwd 기준 상대경로 `./model/...`에 저장되도록 하드코딩돼 있음. → `output_dir`은 태그로만 쓰고, 학습 실행 시 cwd를 OneDrive 밖 `C:/ai_local/glad_run`으로 옮겨 상대경로가 로컬에 떨어지게 함.
- MVTec-AD 데이터셋 루트(`C:/ai_local/diad_dataset/`)에 있던 `license.txt`/`readme.txt`를 `dataset_multiclass.py`의 `get_data_mutil_class`가 카테고리 폴더로 오인식(`os.listdir` 결과를 필터링 없이 순회) → 두 파일을 데이터셋 루트 밖으로 이동.
- `dataset_multiclass.py`의 `anomaly_source`: DTD 텍스처 이미지를 `float32`로 캐스팅한 뒤 `imgaug==0.4.0`의 augmenter(Posterize/Solarize/Invert/Equalize 등)에 넣어 `ValueError: forbidden dtype 'float32'` 발생 → 다른 분기와 동일하게 uint8로 캐스팅 후 augment, 이후 float32로 되돌리도록 패치.
- **공식 코드에 체크포인트 resume 로직이 전혀 없었음** — `accelerator.save_state()`만 있고 `load_state()` 호출이 없어, 저장된 체크포인트와 무관하게 재실행 시 항상 pretrained SD 가중치부터 다시 시작. Windows 자동 업데이트 강제 재부팅으로 실제 진행분(약 13시간, 약 11.5시간)을 두 차례 잃은 뒤 발견. `main()`의 `accelerator.prepare()` 직후에 `output_dir` 내 `checkpoint-N` 중 최신 것을 찾아 `accelerator.load_state()`로 복원하고 `global_step`을 이어가는 로직을 추가(`main_multi.py`). 이후 실제 크래시 상황에서 정상 동작 확인(checkpoint-200에서 재개).
- `dataloader_num_workers` 기본값 8 → 0 (Windows spawn multiprocessing 이슈, DiAD와 동일 패턴).

### 하드웨어 제약에 따른 축소
- 원 논문/공식 스크립트 기준 `train_batch_size=32` → 8GB GPU에서 배치 1개도 7.9GB를 써서 batch=2조차 OOM 위험 → `train_batch_size=1`, `gradient_accumulation_steps=4`(effective batch 4)로 축소. `gradient_accumulation_steps`를 32로 원래 effective batch에 맞추는 것도 검토했으나, micro-batch를 순차 처리하는 구조상 메모리 이득 없이 시간만 대략 8배로 늘어나(step당 약 66초→약 528초 추정) 기각.
- resolution=256(원 논문 multi-category 설정과 동일), denoise_step=500, mixed_precision=fp16, 8-bit Adam, gradient_checkpointing 사용.
- 체크포인트 주기: crash 재발 이후 2000→200 step으로 단축(최대 손실분을 줄이기 위함, resume 로직 추가 후에는 다음 재시작에서 곧바로 활용됨).

### 안정성 인프라 (DiAD 패턴 재사용)
- Windows Update가 로그인 중에도 강제 재부팅하는 것을 확인(9/9, 9/10 두 차례 크래시 원인) → `HKLM:\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU`에 `NoAutoRebootWithLoggedOnUsers=1` 설정.
- `GladTrainingWatchdog` 예약 작업(`C:\Users\kelly\scripts\glad_training_watchdog.ps1`, 15분 주기): 학습 프로세스가 없고 GPU 온도가 75°C 미만이면 자동 재시작.
- 세션 내 Monitor로 GPU 온도(경고 80°C, 강제종료 82°C)·배터리(미충전 상태 20% 이하) 감시.

## 다음에 할 일
- 노트북 팬 강제 풀가동 방법 조사 (W37 피드백 요청 항목, 아직 미착수).
- GLAD 학습을 계속 진행하며 첫 판단 가능한 체크포인트에서 category-wise I-AUROC/P-AUROC 평가 실행.
- SimpleNet/Reverse Distillation/Dinomaly 4개 방법 중 어느 것부터 어느 머신에서 재현할지 배분 결정 — 데스크탑(12GB)이 VRAM 여유가 더 크므로 `method3_diad/markdown/setup_notes.md` 7-8절에서 겪은 8GB 한계(Adam optimizer state OOM)를 우선 이쪽에서 회피 가능한지 검토.
