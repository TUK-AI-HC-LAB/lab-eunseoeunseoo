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
- 이 가설이 맞다면 다음 후보: (a) `bitsandbytes` 8-bit Adam으로 optimizer state를 압축, (b) `only_mid_control=True`가 이미 켜져 있는데도 trainable params가 1.3B나 되는 이유 확인 후 더 많은 layer를 freeze, (c) gradient checkpointing, (d) 더 큰 VRAM 머신으로 이전.

### 8. `bitsandbytes` 8-bit Adam 적용 — 실제 학습 성공
- `sgn/sgn.py`의 `configure_optimizers()`에서 `torch.optim.AdamW` 대신 `bitsandbytes.optim.AdamW8bit` 사용 (optimizer state를 8-bit로 저장해 약 4배 압축).
- 적용 후 첫 실제 optimizer step이 OOM 없이 통과, loss가 감소하는 정상 학습 확인 → OOM 원인이 activation이 아니라 Adam optimizer state였다는 7절의 가설이 맞았음.
- `batch_size=2`, `precision=16`, `accumulate_grad_batches=4`로 고정. 8GB GPU에서는 이 조합이 실질적인 하한선.

### 9. 학습 속도 현실 점검 — 체크포인트 주기 단축
- 초기 iteration 속도 약 20s/it 기준으로 계산하면 1 epoch ≈ 10시간, 원본 `check_val_every_n_epoch=25` 주기로는 단일 세션 안에 체크포인트가 단 한 번도 저장되지 않음 (전체 1000 epoch 기준으로는 비현실적인 기간이 소요).
- **protocol change**: epoch 기준 val/checkpoint(`ckpt_callback_val_loss`, monitor=`val_acc`)는 그대로 두되, `every_n_train_steps=50`짜리 step 기준 `ModelCheckpoint`(`ckpt_callback_periodic`)를 추가해 세션이 중간에 끊겨도(열 종료, 재부팅 등) 최근 50 step 이내 진행 상황은 항상 보존되도록 함.
- `ModelCheckpoint(save_top_k=2, monitor=None)`으로 처음 설정했다가 `MisconfigurationException: ... No quantity for top_k to track`로 즉시 실패 → `monitor`가 없을 때 `save_top_k`는 0/1/-1(무제한)만 허용됨을 확인, `save_top_k=1`로 수정(최신 체크포인트 1개만 유지, 8GB짜리가 무한히 쌓이는 것도 방지).

### 10. 노트북 열 종료 — 물리적 원인
- 학습 중 노트북이 온도 상승으로 갑자기 꺼져 세션과 백그라운드 프로세스가 함께 종료됨.
- 원인은 GPU/하드웨어 한계가 아니라 노트북을 침대 위에 올려놓아 바닥 흡기구가 막힌 배치 문제로 파악 (거치대로 바닥과 간격을 띄운 뒤 동일 워크로드에서 재발 없음).
- 이후 GPU 온도를 주기적으로 폴링해 85°C 이상이면 경고하는 감시 루틴을 병행 운용.

### 11. 체크포인트가 OneDrive 동기화 폴더에 저장되던 버그 — 24.7GB 불필요 동기화 + 5시간43분 스톨
- `ckpt_dir`을 명시적으로 지정하지 않았을 때 기본값이 `./val_ckpt/`로 잡혀, `method3_diad/source/DiAD/val_ckpt/` 즉 OneDrive 동기화 대상 폴더 안에 8.24GB짜리 체크포인트가 3개(총 24.7GB) 쌓인 것을 뒤늦게 발견.
- 학습 로그에서 step 707→708 사이 한 스텝이 5시간43분 걸린 비정상 구간을 발견, 시점이 체크포인트 저장 직후와 일치 → OneDrive가 백그라운드에서 해당 파일을 업로드하려고 시도하면서 디스크 I/O를 붙잡아 학습이 사실상 멈췄던 것으로 추정(6절의 체크포인트 로딩 스톨과 동일한 근본 원인).
- **protocol change**: `train.py`에 `ckpt_dir = 'C:/ai_local/diad_val_ckpt/'`를 명시하고 두 `ModelCheckpoint` 콜백 모두 이 경로를 쓰도록 변경, 기존 체크포인트도 이 경로로 이동. 오래된 체크포인트는 정리하고 최신 것만 유지.
- `.gitignore`에 `lightning_logs/`, `val_ckpt/`, `log`, `log_image/` 추가 (체크포인트 자체는 이미 `*.ckpt` 규칙으로 제외되지만, 잘못된 경로에 생성된 관련 폴더/로그가 실수로 스테이징되는 것도 방지).

### 12. 재시작 시 `ckpt_path` resume — state_dict strict 로딩 충돌
- 세션이 끊긴 뒤 재시작할 때 `ckpt_dir` 안의 최신 `step_step=*.ckpt`를 찾아 `trainer.fit(..., ckpt_path=resume_ckpt_path)`로 넘겨 전체 trainer state(옵티마이저, 스케줄러, global_step 포함)를 복원하도록 함.
- PL이 내부적으로 `model.load_state_dict(..., strict=True)`를 호출하는데, 저장된 체크포인트에는 첫 validation 때 지연 생성되는 평가용 ResNet50 feature extractor(`pretrained_model.*`) 키가 포함되어 있어 방금 새로 생성한 모델 인스턴스와 키가 안 맞아 `RuntimeError: Unexpected key(s) ... "pretrained_model.*"` 발생.
- 이 키 불일치는 무해하므로(평가 전용 서브모듈, 학습에는 관여하지 않음), resume이 필요한 경우에 한해 `model.load_state_dict`를 `strict=False`로 강제하는 monkeypatch를 `resume_ckpt_path`가 있을 때만 적용.

### 13. 체크포인트 저장 중 크래시 — `_atomic_save`의 메모리 이중 사용
- step 200 체크포인트 저장 도중 `RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos ... vs ...`가 `torch.save(checkpoint, bytesbuffer)` 안에서 발생하며 학습 프로세스 종료.
- 원인: PL(`lightning_fabric.plugins.io.torch_io._atomic_save`)의 기본 저장 방식이 8GB+ 체크포인트 전체를 먼저 메모리상의 `io.BytesIO()` 버퍼에 직렬화한 뒤 디스크에 쓰는 방식이라, 저장 순간 피크 RAM 사용량이 사실상 두 배가 됨. 이 머신은 RAM 23.7GB 총량에 모델·옵티마이저·dataloader가 이미 상주한 상태라 버퍼링 중 메모리 압박으로 zip 컨테이너가 깨진 것으로 추정.
- 디스크에 이미 쓰여 있던 `step_step=150.ckpt`는 손상되지 않음(크래시가 디스크 쓰기 시작 전 버퍼링 단계에서 발생했기 때문).
- **protocol change**: `train.py` 상단에서 `_torch_io._atomic_save`를 `torch.save(checkpoint, filepath)`로 직접 디스크에 쓰는 함수로 monkeypatch해 중간 버퍼링을 제거.
- 수정 후 재시작, `step_step=150.ckpt`에서 정상 resume 후 이전에 크래시했던 지점(step 200)을 통과하고 `step_step=200.ckpt` 저장까지 정상 완료됨을 확인 → 수정 유효.

### 14. Iteration 속도 저하 — 데이터셋을 로컬로 옮겨도 해결 안 됨
- step 200 이후 학습이 진행되며 순간 iteration 속도가 초반 ~3.7s/it에서 ~16-19s/it로 저하되어 그 수준에서 고정됨. GPU는 사용률 100%인데 전력 소모는 ~30W(정상 대비 낮음)로, GPU가 연산이 아니라 무언가를 기다리며 대기하는 패턴으로 관찰됨.
- **가설(기각됨)**: 6/11절과 동일한 OneDrive I/O 경합이 이번엔 `data_path`(학습 이미지, 5GB, 여전히 OneDrive 동기화 폴더 안에 있었음)에서 발생하는 것으로 추정 → `No_Submit/Dataset/`를 `C:/ai_local/diad_dataset/`로 복사(robocopy, 6644 files, 4.9GB, 1분41초 완료)하고 `train.py`의 `data_path`를 로컬 경로로 변경.
- 수정 후 재시작·resume 확인했으나, 실측 순간 속도는 여전히 ~15-16s/it로 **개선 없음**. tqdm이 보여주는 누적 평균(예: 3.49s/it)은 체크포인트 resume 시 이미 완료된 iteration을 순간적으로 "따라잡기"하는 구간이 분모에 섞여 낮게 보이는 착시이며, 실제 스텝 간 timestamp로 재계산하면 이전과 동일한 병목이 그대로 남아 있음을 확인.
- 결론: 이 병목의 원인은 (적어도 주된 원인은) OneDrive 데이터 I/O가 아니었음. 아직 미확인 상태로 남은 후보: sm_120(Blackwell)용 최적화 커널 부재(attention/conv 등에서 비효율적인 fallback 경로 사용 가능성), `bitsandbytes` 8-bit optimizer의 매 스텝 양자화/역양자화 오버헤드, 낮은 여유 RAM(24GB 중 4-6GB대)으로 인한 페이징. 원인 규명에 추가 시간을 쓰는 대신, 현재 속도로도 50-step 체크포인트가 계속 쌓이므로 H3/H4 evidence 확보 목적상 학습은 그대로 진행하기로 결정.

### 15. Epoch 0 완료, 원인 불명의 2시간22분 단발성 스톨
- global_step 478 시점(step_step=450.ckpt 저장 직후)에 epoch 0(1815 iteration) 완주, `train/loss_epoch=0.128`로 epoch 평균 loss 최초 기록. 이후 epoch 1 진행 중 — 크래시 없이 epoch 경계를 정상적으로 넘김.
- 이 과정 중 새벽 시간대(약 04:41~07:03)에 단일 스텝이 **2시간22분** 걸리는 이상 지연 발생 (11절의 5시간43분 스톨과 비슷한 규모). 체크포인트가 이미 로컬(`C:/ai_local`)에 있어 OneDrive가 원인일 가능성은 낮고, 해당 시간대가 Windows Update/Defender 예약 검사와 겹칠 수 있어 그쪽을 의심 중이나 미확인. 별도 개입 없이 스스로 회복해 정상 속도로 복귀함.
- 논문 기준 학습 설정은 `학습 epoch: 1,000, batch size 12` (`method3_diad/markdown/diad_summary.md` 표 참고). 로컬 8GB GPU는 batch size 2로도 겨우 맞춰 실행 중이라 1,000 epoch 완주는 여전히 비현실적 — 목표를 논문과 동일한 epoch 수를 채우는 것이 아니라, 학습이 진행됨에 따라 pixel/image AUROC가 의미 있는 수준으로 개선되는지를 주기적으로 평가해 H3/H4 검증에 쓸 수 있는 시점을 찾는 것으로 조정.

### 16. 장시간 학습 -- epoch 종료마다 자동 commit/push
- 세션 중간 개입 없이도 진행 상황이 실제 시점 그대로 repo에 남도록 자동화가 필요했음.
- `train.py`에 `GitCommitOnEpochEnd(pl.Callback)` 추가: `on_train_epoch_end`마다 `method3_diad/source/epoch_log.csv`에 (epoch, global_step, epoch 평균 loss, 실제 타임스탬프) 한 줄을 append하고, 그 파일 하나만 `git add`->`commit`->`push`.
- 체크포인트(8GB+)는 여전히 git에 올리지 않음 -- 커밋되는 건 가벼운 로그 csv뿐이라 evidence path는 남기되 repo 용량은 늘지 않음.
- 네트워크 단절로 push가 실패해도 학습이 죽지 않도록 git 호출 전체를 try/except로 감싸고 timeout을 둠 -- 실패하면 다음 epoch에서 다시 시도하고, 그 사이 로컬에는 커밋이 계속 쌓여서 다음 성공 시 한 번에 밀림.
- 이 자동화는 epoch이 실제로 끝나는 시점(불규칙한 간격, 스톨 포함)에만 커밋이 발생하므로, 실제 작업 시점을 그대로 반영하는 진짜 진행 기록이다.

### 17. 절전 크래시 이후 두 가지 추가 실패 모드 -- 하이버네이트, 세션 종료, 체크포인트 손상
- (배경) 노트북이 절전 관련 설정 미비로 2026-07-18 13:25~07-19 22:39 사이 약 33시간 하이버네이트됨 (원인: "Standby Battery Budget Exceeded"). 절전에서 깨어나며 CUDA context가 깨져 `torch.cuda.synchronize()`에서 `CUDA error: unknown error` 발생, 프로세스 종료. **조치**: `powercfg`로 AC 전원 연결 시 절전/하이버네이트/모니터 꺼짐을 모두 비활성화(`standby-timeout-ac 0`, `hibernate-timeout-ac 0`).
- 재발 방지로 `C:\Users\kelly\scripts\diad_training_watchdog.ps1`(python train.py 프로세스가 없으면 재시작)을 Windows 작업 스케줄러에 15분 주기(`DiadTrainingWatchdog`)로 등록.
- **새로운 실패 모드 발견**: 학습을 시작한 Claude 세션(Bash 백그라운드 프로세스)이 재시작되면서, 그 자식 프로세스였던 `train.py`도 함께 종료됨 (시스템 재부팅 기록은 없어 하드웨어/OS 문제가 아니라 세션 종료가 원인으로 확인). 하필 `step_step=1050.ckpt` 저장 도중 종료되어 **758MB짜리 손상된(잘린) 체크포인트 파일**이 남음 -- 정상 체크포인트는 항상 ~8.15GB.
- 12절 resume 로직은 `step_step=*.ckpt`를 mtime 기준으로 가장 최신 것을 고르므로, 손상된 파일을 그대로 두면 다음 재시작이 그 파일을 로드하려다 다시 실패하는 악순환에 빠질 뻔했음 -> 손상 파일 삭제 후 그 이전의 정상 체크포인트(`step_step=1000.ckpt`)로 resume.
- **protocol change**: 이후로는 학습을 Bash 백그라운드(Claude 세션의 자식 프로세스)가 아니라 `Start-Process`로 완전히 분리된 프로세스로 띄움 -- 이러면 Claude 세션이 재시작돼도 학습 프로세스는 살아있고, watchdog 스케줄러도 세션과 무관하게 계속 감시 가능.
- 교훈: 체크포인트 저장 도중 죽는 경우를 대비해, resume 로직이 파일 크기나 무결성을 검증하지 않고 무조건 최신 파일을 신뢰하는 것은 위험 -- 지금은 사람이 수동으로 손상 파일을 지웠지만, 다음엔 resume 로직 자체에 최소한의 크기 검증을 추가하는 것도 고려할 만함(아직 미구현).

### 18. H3/H4 검증용 평가 실행 — `test.py` Windows 버그 2건 수정
- epoch 7 체크포인트(`step_step=3600.ckpt`)로 H3(grid I-AUROC vs PatchCore 0.977)/H4(transistor P-AUROC vs PatchCore 0.929) 검증을 위해 `test.py` 최초 실행.
- `MVTecDataset('test')` 호출에 `root` 인자가 빠져 있어 `TypeError` → `train.py`와 동일하게 로컬 데이터셋 경로(`C:/ai_local/diad_dataset/`)를 명시.
- `DataLoader(..., num_workers=8, ...)` + `if __name__ == '__main__':` 가드 없음 → Windows의 spawn 기반 multiprocessing이 worker마다 스크립트 전체(모델 생성·가중치 로딩 포함)를 처음부터 재실행해 멈춘 것처럼 보임(`RuntimeError: freeze_support()...`) → `num_workers=0`으로 변경해 회피(train.py에서도 이미 같은 이유로 0을 쓰고 있었음).
- 수정 후 정상 실행 확인, 1,725개 테스트 이미지 전체 처리 완료. 결과와 H3/H4 판정은 `method3_diad/markdown/h3_h4_evaluation.md` 참고 — **H4 지지**(transistor P-AUROC 0.945 > PatchCore 0.929), **H3 미결정**(grid I-AUROC 0.588 < PatchCore 0.977이지만 전체 평균도 0.803으로 아직 미성숙한 모델이라 반박으로 보기 이름).

### 19. H3/H4 2차 재평가(epoch 16) — H4 역전, H3 개선 추세
- epoch 16 체크포인트(`step_step=7400.ckpt`)로 동일 평가 재실행(`method3_diad/source/run_eval_epoch16.sh`).
- grid I-AUROC: 0.588→**0.764**로 개선(여전히 PatchCore 0.977 미달이나 격차 축소) — H3는 여전히 미결정이나 낙관적.
- transistor P-AUROC: 0.945→**0.923**으로 하락, PatchCore 0.929 아래로 역전 — H4를 지지에서 미결정으로 하향 조정.
- train loss가 epoch 10~13 사이 0.108→0.118→0.123→0.114로 진동하는 것과 같은 시기라, 두 지표의 변화가 노이즈인지 실제 추세인지는 최소 한 번 더 재평가해야 구분 가능. 세부는 `method3_diad/markdown/h3_h4_evaluation.md`.

### 20. 25-epoch마다 도는 내장 validation 비활성화 — 사용 안 하는데 매번 몇 시간씩 소모
- `check_val_every_n_epoch=25` 설정 때문에 epoch 24 종료 시점에 첫 내장 validation이 트리거됨. `ddpm.py`의 `validation_step`을 보니 `test.py`와 동일하게 전체 DDIM 샘플링 + ResNet50 feature 비교 + anomaly map 계산을 1,725개 테스트 이미지 전부에 대해 수행하는 무거운 로직이었음.
- 이미지당 속도가 `test.py` 단독 실행 시 ~2-4s인데 반해 학습 프로세스 안에서는 ~48s로 10배 이상 느림 — 학습용 optimizer state 등이 이미 GPU 메모리를 점유한 상태에서 같은 무거운 추론을 돌리기 때문(메모리 압박, 14/17/19절과 같은 패턴).
- 이 validation이 로깅하는 `val_acc`는 어디에도 실제로 쓰이지 않음 — H3/H4 판단은 항상 `test.py`를 별도로 돌려서 확인해왔음(`method3_diad/markdown/h3_h4_evaluation.md`).
- **protocol change**: `pl.Trainer(...)`에 `limit_val_batches=0` 추가해 내장 validation을 완전히 비활성화. `ckpt_callback_val_loss`(monitor=`val_acc`)는 이제 트리거될 일이 없어 사실상 무력화되지만, step 기준 체크포인트(`ckpt_callback_periodic`)가 이미 진행 상황 보존을 전담하고 있어 문제 없음.

### 21. epoch 24 재평가 결과 유실 확인, epoch 34로 3차 재평가 완료
- `run_eval_epoch24.sh`(step=11350)로 3차 재평가를 시도했던 흔적은 있으나, 대응하는 `eval_results_epoch24.csv`가 한 번도 커밋되지 않았고 로컬에도 없음을 W36 브리핑 작성 중 발견. 임시 로그(`diad_test_run*.log`)도 남아있지 않았고, 체크포인트(`step_step=11350.ckpt`)도 `save_top_k=1`로 이후 덮어써져 재실행으로도 재현 불가.
- 정황상 20절의 내장 validation(epoch 24 종료 시점 자동 발동, 결과 미저장)이 이 시점과 겹쳐 원인일 가능성이 높다고 추정 — 확정 증거는 없음.
- epoch 34(`step_step=15650.ckpt`) 평가는 임시 로그(`diad_test_run7.log`)가 남아있어 복구 가능했음: grid I-AUROC 0.654135, transistor P-AUROC 0.921888. `method3_diad/source/eval_results_epoch34.csv`로 정리해 커밋.
- **교훈**: 평가 실행 직후 raw csv를 바로 커밋하지 않으면(재현성 규칙, 가이드 11절) 임시 로그가 세션 재시작·시간 경과로 유실될 때 evidence 자체가 사라진다. 앞으로는 `test.py` 실행 후 결과 확인 즉시 csv 저장·커밋을 원칙으로 한다.
- 세부 해석은 `method3_diad/markdown/h3_h4_evaluation.md`, weekly brief는 `meetings/2026-W36_brief.md`.

## 다음에 할 일
1. (완료) H3/H4 3차 재평가 — epoch 34로 완료, epoch 24는 재현 불가로 확인. 결과는 `h3_h4_evaluation.md`, `2026-W36_brief.md`에 반영.
2. 다음 학습 방향 결정 — 계속 학습을 진행할지, DiAD 구조 자체의 한계로 보고 다른 후보를 탐색할지 논의 필요.
3. (선택) 14절의 미확인 속도 저하 원인, 15절의 새벽 스톨 원인 중 하나를 골라 profiling — 우선순위는 낮음.
4. (선택) 17절에서 언급한 resume 로직의 체크포인트 크기/무결성 검증 추가.
