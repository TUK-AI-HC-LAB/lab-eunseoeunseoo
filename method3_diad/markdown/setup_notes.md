# DiAD 재현 착수 — 환경 설정 기록

## 근거 경로
- source: `method3_diad/source/DiAD` (공식 repo clone, commit `d281300`)
- data manifest 생성 script: `method3_diad/source/build_mvtec_json.py`
- generated manifest: `method3_diad/source/DiAD/training/MVTec-AD/{train,test}.json`

## 왜 바로 학습을 시작하지 못했는가
DiAD 공식 파이프라인은 (1) conda 환경, (2) MVTec-AD 데이터셋, (3) 사전학습 autoencoder(`kl-f8.zip`) + Stable Diffusion v1.5 체크포인트(수 GB), (4) autoencoder finetune, (5) 1000 epoch 학습, (6) test 순서로 진행된다. 이번 세션에서는 (1)(2)까지만 준비했고, (3) 체크포인트 다운로드와 (4)(5)(6)은 아직 실행하지 않았다.

## 이번에 확인/수정한 것

### 1. GPU/CUDA 호환성
- 로컬 GPU: RTX 5060 Laptop 8GB, driver가 CUDA 13.1까지 지원.
- 공식 `environment.yaml`은 `pytorch=1.12.1` + `cudatoolkit=11.3`을 고정하는데, 이 조합은 RTX 5060(Blackwell, sm_120)용 커널을 포함하지 않아 그대로 쓰면 CUDA kernel 호환 문제가 날 가능성이 높다.
- **protocol change**: 원본 `environment.yaml`을 그대로 쓰지 않고, `diad`라는 별도 conda env(python 3.10)를 만들어 `torch`/`torchvision`을 `cu124` wheel(torch 2.6.0+cu124)로 설치.
- **확인된 결과**: `cu124` wheel도 부족했다. 실행 시 `NVIDIA GeForce RTX 5060 Laptop GPU with CUDA capability sm_120 is not compatible with the current PyTorch installation. The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.` 경고 발생 — `torch.cuda.is_available()`은 `True`를 반환하지만 sm_120용 커널이 빌드에 없어 실제 forward/backward 연산에서 깨질 가능성이 높다. cu124는 sm_90까지만 커버하므로, sm_120(Blackwell) 지원이 포함된 더 최신 빌드(cu128 계열 또는 nightly)로 재설치해야 한다.
- 나머지 pip 의존성(`pytorch_lightning`, `einops`, `omegaconf`, `taming-transformers` 등)은 아직 설치 전.

### 2. 데이터 매니페스트 생성
- `mvtecad_dataloader.py`는 `./training/MVTec-AD/{train,test}.json`을 JSON-Lines로 읽고, 각 줄의 `filename`/`maskname`을 `train.py`의 `data_path`와 이어 붙여 이미지를 로드한다. 원본 repo는 이 json이 이미 있다고 가정하고 만드는 코드는 제공하지 않는다.
- 로컬 `No_Submit/Dataset/<category>/{train,test,ground_truth}`가 표준 MVTec-AD 구조와 동일함을 확인 → `build_mvtec_json.py`로 변환.
- 결과: train 3,629 entries, test 1,725 entries — DiAD 논문/요약에 적힌 데이터셋 크기(학습 3,629 정상, 테스트 1,725)와 정확히 일치. 매니페스트 생성 로직이 맞다는 근거로 사용 가능.

### 3. `train.py` 경로 수정
- `data_path`를 원본 하드코딩(`/root/autodl-tmp/mvtecad/`, 원래 저자의 서버 경로)에서 로컬 절대경로(`C:/Users/.../No_Submit/Dataset/`)로 변경.
- 이 경로는 advisor 로컬 머신 기준이라 다른 환경에서 그대로 재현되지 않는다. 실행 전 본인 데이터셋 위치에 맞춰 다시 고쳐야 한다.

## 다음에 필요한 것 (아직 안 함)
1. torch를 sm_120(Blackwell) 지원 빌드(cu128 계열 또는 nightly)로 재설치하고 `torch.randn(1).cuda() @ torch.randn(1).cuda()` 같은 실제 연산으로 재검증.
2. `kl-f8.zip` (사전학습 autoencoder) + `v1-5-pruned.ckpt` (Stable Diffusion v1.5) 다운로드 — 수 GB, 시간 소요.
3. `finetune_autoencoder.py` 실행 → `models/mvtec_ae.ckpt` 생성.
4. `build_model.py` 실행 → `models/diad.ckpt` 생성.
5. `train.py` 실행 (batch_size=12, 1000 epoch 기본값 — 8GB VRAM에서 이 batch size가 OOM 없이 도는지 확인 필요, 필요하면 batch size/accumulate_grad_batches 조정).
6. pip 의존성 전체 설치 및 `pytorch_lightning`/`taming-transformers` 등이 python 3.10 + torch cu128 조합에서 깨지지 않는지 확인.
