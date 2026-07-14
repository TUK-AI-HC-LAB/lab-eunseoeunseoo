# DiAD 재현 착수 — 환경 설정 기록

## 근거 경로
- source: `method3_diad/source/DiAD` (공식 repo clone, commit `d281300`)
- data manifest 생성 script: `method3_diad/source/build_mvtec_json.py`
- generated manifest: `method3_diad/source/DiAD/training/MVTec-AD/{train,test}.json`
- conda env: `diad` (python 3.10), 로컬 전용, repo에는 미포함

## 진행 순서와 확인된 것

### 1. GPU/CUDA 호환성
- 로컬 GPU: RTX 5060 Laptop 8GB, driver가 CUDA 13.1까지 지원.
- 공식 `environment.yaml`은 `pytorch=1.12.1` + `cudatoolkit=11.3`을 고정하는데, 이 조합은 RTX 5060(Blackwell, sm_120)용 커널을 포함하지 않는다.
- **protocol change**: 원본 `environment.yaml` 대신 `diad` conda env(python 3.10)를 만들고 `torch`/`torchvision`을 PyTorch 공식 `cu124` wheel로 설치 → 실행 시 `sm_120 is not compatible with the current PyTorch installation` 경고 확인 (cu124는 sm_90까지만 커버).
- `cu128` wheel(torch 2.11.0+cu128)로 재설치 후 `torch.randn(4,4,device='cuda') @ torch.randn(4,4,device='cuda')` 실제 GPU 연산 성공 → sm_120 문제 해결 확인.
- **재현성 함정**: 이후 나머지 pip 의존성(`pytorch_lightning`, `torchmetrics` 등)을 `--index-url` 없이 그냥 설치했더니, 의존성 해석 과정에서 torch/torchvision이 기본 PyPI 인덱스 버전으로 조용히 재설치되어 CUDA DLL(`c10_cuda.dll`)이 깨짐 (`OSError: ... Error loading c10_cuda.dll`). `pip install --force-reinstall --no-deps torch torchvision --index-url .../cu128`로 복구.
- **교훈**: GPU 특화 wheel(cu128 등)을 쓰는 환경에서는 이후 모든 pip 설치에 `--index-url`을 계속 지정하거나, torch/torchvision을 마지막에 `--no-deps`로 다시 고정해야 한다. 이 반복 설치가 재현성에 실제로 영향을 준 사례로 기록.
- taming-transformers는 PyPI가 아니라 `pip install git+https://github.com/CompVis/taming-transformers.git`로 설치 (repo에 vendor된 `taming/` 폴더는 `lpips` weight 파일 하나만 있고 실제 모듈 코드는 없음).

### 2. 데이터 매니페스트 생성
- `mvtecad_dataloader.py`는 `./training/MVTec-AD/{train,test}.json`을 JSON-Lines로 읽고, 각 줄의 `filename`/`maskname`을 `train.py`의 `data_path`와 이어 붙여 이미지를 로드한다. 원본 repo는 이 json이 이미 있다고 가정하고 만드는 코드는 제공하지 않는다.
- 로컬 `No_Submit/Dataset/<category>/{train,test,ground_truth}`가 표준 MVTec-AD 구조와 동일함을 확인 → `build_mvtec_json.py`로 변환.
- 결과: train 3,629 entries, test 1,725 entries — DiAD 논문/요약에 적힌 데이터셋 크기(학습 3,629 정상, 테스트 1,725)와 정확히 일치. 매니페스트 생성 로직이 맞다는 근거로 사용 가능.

### 3. `train.py` 경로 수정
- `data_path`를 원본 하드코딩(`/root/autodl-tmp/mvtecad/`, 원래 저자의 서버 경로)에서 로컬 절대경로(`C:/Users/.../No_Submit/Dataset/`)로 변경.
- 이 경로는 advisor 로컬 머신 기준이라 다른 환경에서 그대로 재현되지 않는다. 실행 전 본인 데이터셋 위치에 맞춰 다시 고쳐야 한다.

### 4. 체크포인트 다운로드 (finetune 단계 스킵)
- 공식 README는 (a) `kl-f8.zip`으로 autoencoder를 처음부터 finetune하거나, (b) 저자가 미리 finetune해둔 체크포인트를 바로 쓰는 두 경로를 제공. 시간 절약을 위해 (b) 선택.
- `models/v1-5-pruned.ckpt` (Stable Diffusion v1.5, 7.7GB, huggingface에서 직접 다운로드) 확보.
- `models/mvtecad_fs.ckpt` (MVTec용 사전 finetune된 first-stage autoencoder, 823MB, 저자 Google Drive에서 `gdown`으로 다운로드) 확보. **주의**: `build_model.py`가 요구하는 실제 파일명은 `mvtecad_fs.ckpt`이다 (README 본문의 안내와 달리 `mvtec_ae.ckpt`가 아님 — 처음에 잘못된 이름으로 받아서 리네임함).

### 5. `build_model.py` 실행
- `torch.load`가 torch 2.6+부터 기본 `weights_only=True`로 바뀌어서 pytorch_lightning 객체(`ModelCheckpoint`)가 pickle된 구버전 체크포인트 로딩이 깨짐 (`build_model.py`, `sgn/model.py`의 `torch.load` 호출에 `weights_only=False` 명시해서 해결 — 신뢰 가능한 저자 체크포인트라 안전).
- 성공: `models/diad.ckpt` (6.1GB) 생성 완료.

### 6. `train.py` 실행 — 여러 겹의 버전 skew 버그
OneDrive 동기화 폴더 안에서 6GB 체크포인트를 읽으면 응답이 없어짐(디스크 I/O가 사실상 멈춤) → 체크포인트 3개(`v1-5-pruned.ckpt`, `mvtecad_fs.ckpt`, `diad.ckpt`)를 OneDrive 밖 `C:/ai_local/diad_models/`로 이동, `train.py`의 `resume_path`도 그쪽으로 변경. 코드/데이터셋은 OneDrive 안에 남겨둠(용량이 작아 문제 없음).

이후 순서대로 만난 문제와 조치 (모두 `pytorch_lightning==1.9.5`가 저자가 pin한 `1.5.0`과 hook 시그니처/API가 달라서 발생):
1. `cv2.imread`가 경로에 한글(`바탕 화면`)이 있으면 Windows에서 파일을 못 읽고 `None`을 반환 → `mvtecad_dataloader.py`에 `np.fromfile` + `cv2.imdecode` 기반 `imread_unicode()` 헬퍼 추가, `cv2.imread` 호출 두 곳을 교체.
2. `utils/eval_helper.py`, `utils/vis_helper.py`의 `np.int`/`np.bool`/`np.float`가 최신 numpy에서 제거됨 → `int`/`bool`/`float`로 교체.
3. `ldm/models/diffusion/ddpm.py`의 `on_train_batch_start(self, batch, batch_idx, dataloader_idx)` — 최신 PL은 이 인자를 안 넘겨줌 → `dataloader_idx=0` 기본값 추가.
4. `sgn/logger.py`의 `ImageLogger.on_train_batch_end(...)`도 동일 문제 → 동일하게 기본값 추가.
5. `ldm/util.py`의 `log_txt_as_img`가 존재하지 않는 `font/DejaVuSans.ttf`를 요구 → `try/except OSError`로 PIL 기본 폰트로 폴백.

이 5개를 고친 뒤 실제로 sanity check(DDIM 10-step 샘플링, pixel AUROC 계산까지)와 진짜 학습 스텝(forward/backward, loss 값 출력: 0.271→0.235→0.241)이 GPU에서 정상 동작하는 것을 확인했다 — **저자 pin 버전(`pytorch_lightning==1.5.0`, `torch==1.12.1`)을 그대로 안 쓰고 최신 스택으로 옮긴 대가로, 코드 자체의 버전 호환성 버그 5개를 직접 고쳐야 했다.**

### 7. CUDA OOM — batch_size/precision을 낮춰도 재현됨
- `batch_size=12, precision=32`: 3번째 micro-batch까지 정상 진행(loss 출력됨), 4번째(= `accumulate_grad_batches=4`로 누적된 첫 실제 optimizer step) 직전에 `CUDA out of memory` (`Tried to allocate 20.00 MiB`, `18.24 GiB is allocated by PyTorch`라는 이상한 수치 — 8GB GPU인데 18GB 넘게 잡혔다고 나옴).
- `batch_size=2, precision=16`(mixed precision)로 낮춰서 재시도 → **동일한 지점에서 동일하게 OOM** (`18.13 GiB allocated`).
- **해석**: batch_size/precision을 낮춰도 OOM 지점과 규모가 거의 그대로라는 것은, activation memory(배치 크기에 비례)가 병목이 아니라 **trainable parameter 1.3B개에 대한 Adam optimizer state(m, v buffer, 보통 fp32 유지)** 자체가 지배적인 메모리 사용처라는 뜻이다. Adam만으로도 1.3B params × 8bytes ≈ 10.4GB가 필요해, 배치 크기를 아무리 줄여도 8GB GPU 한 대로는 구조적으로 안 맞을 가능성이 높다.
- 이 가설이 맞다면 다음 후보: (a) `bitsandbytes` 8-bit Adam으로 optimizer state를 압축, (b) `only_mid_control=True`가 이미 켜져 있는데도 trainable params가 1.3B나 되는 이유 확인 후 더 많은 layer를 freeze, (c) gradient checkpointing, (d) 더 큰 VRAM 머신으로 이전. 아직 이 중 어느 것도 시도하지 않음.

## 다음에 할 일
1. 위 OOM 가설(Adam optimizer state 지배) 검증 — 예: `sum(p.numel() for p in model.parameters() if p.requires_grad)`으로 실제 trainable param 수 확인.
2. `bitsandbytes` 8-bit Adam 또는 추가 freeze로 재시도.
3. 학습 로그·중간 checkpoint 경로를 evidence로 기록 (아직 optimizer step을 한 번도 성공 못해서 checkpoint 없음).
