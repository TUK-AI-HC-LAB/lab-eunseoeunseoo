# W40 프레임워크 코드 분석

- 대상: anomaly detection 프레임워크 (`dinomaly_share_codebase/dinomaly_share_codebase/`)

## 0. 실행 대상 선택

`experiment.yaml`의 네 값만 바꾸면 실행 대상이 바뀐다.

```yaml
# Change these four values to select an experiment.
method: patchcore
dataset: mvtec
category: bottle
data_path: C:/ai_local/diad_dataset
```

> `experiment.yaml`을 `main.py`가 읽어서 `method`는 `mainmodel`로, `category`는 `subdatasets`로 이름을 바꾼다. 방법 이름과 데이터셋 이름이 등록된 목록에 없다면 에러가 난다.

---

## 1. 전처리

### 1-1. 이미지·마스크 변환 정의

`datasets/base.py:132-156`

```python
        self.transform_img = [
            Crop(*self.args.crop) if self.args.crop is not None else lambda x: x,
            transforms.Resize((resize, resize)),
            ...
            #transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_img = transforms.Compose(self.transform_img)

        self.transform_mask = [
            transforms.Resize((self.args.masksize, self.args.masksize)),
            #transforms.CenterCrop(self.args.masksize),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(self.transform_mask)
```

(`...` 부분은 주석 처리된 ColorJitter, Flip, Affine 등이라 생략)

크기 기본값: `main.py:122-124`

```python
    "resize": 224,
    "imagesize": 224,
    "masksize": 224,
```

> 이미지는 `Resize(224,224)`에서 `ToTensor`에서 `Normalize` 순서로 변환된다. `ToTensor`는 0~255 정수를 0~1 실수로 바꾸고 `(3,224,224)` 모양으로 만든다. `Normalize`는 ImageNet의 평균과 표준편차를 쓴다. 마스크는 `Normalize`를 안하고 `Resize`와 `ToTensor`만 거친다. `CenterCrop`은 주석처리가 돼있는데, 텐서 크기는 `resize`가 정하고 `imagesize`는 데이터셋에는 저장만 되지만 이후 trainer의 `input_shape`로 전달된다.


**shape 변화** (기본값 `resize=masksize=224` 기준, `main.py:122-124`)

- 이미지: 원본 `(H,W,3)` → `Resize` → `ToTensor` → `Normalize` → `(3,224,224)`
- 마스크: 원본 `(H,W)` → `Resize` → `ToTensor` → `(1,224,224)`

### 1-2. `__getitem__`이 돌려주는 값

`datasets/base.py:179-205`

```python
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        text = self._gettext(image_path)

        ret = {
                "image": image,
                "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
                "image_name": "/".join(image_path.split("/")[-4:]),
                "image_path": image_path,
                "text": text,
            }

        # mask loading
        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            if len(mask.mode) > 1: # 만약 mask 가 1차원이 아니면, 1차원으로 변경
                mask = mask.convert("L")
            mask = self.transform_mask(mask)
            mask[mask != 0] = 1.0 # bilinear interpolation 으로 인해 0이 아닌 값이 나올 수 있음
        else:
            mask = torch.zeros(1, self.args.masksize, self.args.masksize)
        ret['mask'] = mask
```

> 이미지 경로를 열어 RGB로 바꾸고, 위 변환을 적용해서 딕셔너리로 돌려주는데 여기서 키는 `image`, `classname`, `anomaly`, `is_anomaly`, `image_name`, `image_path`, `text`, `mask`, `fg`이고(`base.py:186-218`) 여기서 `is_anomaly`는 폴더 이름이 `good`이면 0, 아니면 1이다. 마스크는 테스트이며 파일이 있을때만 읽고 나머지는 전부 0인 `(1,224,224)`를 채운다. 리사이즈때 경계가 회색으로 번져서 0이 아닌 값은 전부 1로 되돌린다.

**shape 변화**

- `image`: `(3,224,224)`
- `mask`: 테스트 + 마스크 파일 있음 → 변환 후 `(1,224,224)`, 값은 0 또는 1 / 그 외 → `torch.zeros(1, masksize, masksize)` = `(1,224,224)`

### 1-3. 데이터셋이 지켜야 하는 약속

`component_registry.py:21-23`, `239-258`

```python
DATASET_ATTRIBUTES = ("imagesize", "data_to_iterate")
DATASET_METHODS = ("having_mask",)
SAMPLE_KEYS = ("image", "mask", "is_anomaly")
```

```python
def validate_dataset_instance(dataset: Any, check_sample: bool = True) -> None:
    missing = [name for name in DATASET_ATTRIBUTES if not hasattr(dataset, name)]
    missing.extend(
        name for name in DATASET_METHODS if not callable(getattr(dataset, name, None))
    )
    if missing:
        raise ComponentContractError(...)

    if check_sample and len(dataset):
        sample = dataset[0]
        if not isinstance(sample, Mapping):
            raise ComponentContractError("dataset sample must be a mapping")
        missing_keys = [key for key in SAMPLE_KEYS if key not in sample]
        if missing_keys:
            raise ComponentContractError(...)
```

> 데이터셋 클래스는 `imagesize`, `data_to_iterate`, `having_mask()`를 가져야 하고, 샘플 딕셔너리에 `image`, `mask`, `is_anomaly` 키가 있어야 한다는 검사 코드도 있다. 다만 `main_dataset.py:214`에서는 `check_sample=False`로 부르기 때문에 이 키 검사는 실행되지 않고, 속성과 메서드만 검사한다.

### 1-4. 이미지 목록 만들기: `data_to_iterate`

`datasets/mvtec.py:26-91` (발췌)

```python
        for classname in self.classnames_to_use:
            classpath = os.path.join(self.source, classname, "train" if self.split ==
                                     DatasetSplit.TRAIN else "test")
            maskpath = os.path.join(self.source, classname, "ground_truth")
            anomaly_types = os.listdir(classpath)
            ...
            for anomaly in anomaly_types:
                anomaly_path = os.path.join(classpath, anomaly)
                anomaly_files = sorted(os.listdir(anomaly_path))
                imgpaths_per_class[classname][anomaly] = [
                    os.path.join(anomaly_path, x) for x in anomaly_files
                ]
                ...
                anomaly_mask_path = os.path.join(maskpath, anomaly)
                if self.split == DatasetSplit.TEST and anomaly != "good" and os.path.exists(anomaly_mask_path): # mask 없으면 건너뜀
                    anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                    maskpaths_per_class[classname][anomaly] = [
                        os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files
                    ]
                else:
                    maskpaths_per_class[classname]["good"] = None
```

```python
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    try: # mask 가 있을 때
                        data_tuple.append(
                            maskpaths_per_class[classname][anomaly][i])
                    except:
                        data_tuple.append(None)
                    data_to_iterate_class.append(data_tuple)
```

`main_dataset.py:174-183`, `217-238` (DataLoader 만들기)

```python
def _make_loader(dataset, batch_size, num_workers, shuffle, drop_last=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        drop_last=drop_last,
    )
```

```python
            train_loader = _make_loader(
                train_dataset,
                args.batch_size,
                args.num_workers,
                shuffle=True,
                drop_last=drop_last,
            )
            ...
            test_loader = _make_loader(
                test_dataset, 1, args.num_workers, shuffle=False
            )
```

**shape 변화**

- 폴더 구조(경로 조립 코드 기준): `<data_path>/<category>/train/<종류>/*`, `<data_path>/<category>/test/<종류>/*`, `<data_path>/<category>/ground_truth/<종류>/*`
- `data_to_iterate`의 한 원소: `[classname, anomaly, image_path, mask_path]` (마스크가 없으면 `None`)
- 학습 DataLoader: `image` `(batch_size,3,224,224)`, `mask` `(batch_size,1,224,224)`
- 테스트 DataLoader: 배치 크기가 1로 고정. `image` `(1,3,224,224)`, `mask` `(1,1,224,224)`

> 이미지 목록은 `MVTecDataset._get_image_data`가 만든다. 카테고리마다 `train` 또는 `test` 폴더 아래의 결함 종류 폴더를 `os.listdir`로 읽어 이미지 경로를 정렬해서 모은다. 마스크는 테스트이고 `good`이 아닐 때만 `ground_truth/<종류>`에서 같은 방식으로 모은다. 이미지와 마스크를 정렬 순서의 i번째끼리 짝지어 `[classname, anomaly, image_path, mask_path]`로 묶고, 짝이 없으면 `mask_path`는 `None`이다. DataLoader는 학습에는 `args.batch_size`를, 테스트에는 1을 배치 크기로 쓴다.

---

## 2. 모델 교체

### 2-1. 방법 등록 표

`component_registry.py:30-38`, `87-121`

```python
@dataclass(frozen=True)
class MethodSpec:
    """How to import and construct one anomaly-detection method."""

    module: str
    class_name: str
    config_name: str | None = None
    backbone_module: str = "backbones"
    constructor_uses_context: bool = False
```

```python
METHOD_REGISTRY: MutableMapping[str, MethodSpec] = {
    "simple": MethodSpec(
        "trainer.trainer_simplenet", "Trainer_SimpleNet", "simple"
    ),
    "patchcore": MethodSpec(
        "trainer.trainer_patchcore", "Trainer_PatchCore", "patchcore"
    ),
    ...
    "rd": MethodSpec("trainer.trainer_rd", "Trainer_RD", "rd"),
    ...
    "padim": MethodSpec("trainer.trainer_padim", "Trainer_PaDiM"),
    "dinomaly": MethodSpec(
        "trainer.trainer_dinomaly", "Trainer_Dinomaly", "dinomaly"
    ),
    "coad": MethodSpec(
        "COAD.COAD",
        "COADNetwork",
        "coad",
        backbone_module="COAD.encoder",
        constructor_uses_context=True,
    ),
}
```

(등록된 11개: simple, patchcore, promptad, winclip, glass, rd, rd_orig, uniad, padim, dinomaly, coad. 중간은 생략했다.)

> `METHOD_REGISTRY`는 방법 이름에서 파일, 클래스, 설정 이름을 구성하는 표이다. `method: patchcore`라는 글자로 `Trainer_PatchCore`를 찾는다. 등록된 방법은 총 11개이며 `coad`만 백본 모듈과 생성 방식이 다르다.

### 2-2. 필요할 때만 불러오기

`component_registry.py:40-50`

```python
    def load_class(self) -> Type[Any]:
        module = importlib.import_module(self.module)
        trainer_class = getattr(module, self.class_name)
        validate_trainer_class(trainer_class)
        return trainer_class

    def create(self, device: Any, context: Mapping[str, Any]) -> Any:
        trainer_class = self.load_class()
        if self.constructor_uses_context:
            return trainer_class(device, dict(context))
        return trainer_class(device)
```

> `importlib.import_module(문자열)`로 고른 방법의 파일만 불러온다 안 쓰는 방법이 요구하는 라이브러리는 설치하지 않아도 된다. `getattr`은 클래스 이름 문자열로 실제 클래스를 꺼낸다.

### 2-3. 모든 방법이 지켜야 하는 약속

`component_registry.py:14-20`, `201-209`, `223-236`

```python
TRAINER_METHODS = (
    "load",
    "train",
    "predict",
    "get_evaluation_metrics",
    "set_model_dir",
)
```

```python
def validate_trainer_class(trainer_class: Type[Any]) -> None:
    if not isinstance(trainer_class, type):
        raise ComponentContractError(f"{trainer_class!r} is not a class")
    missing = [name for name in TRAINER_METHODS if not callable(getattr(trainer_class, name, None))]
    if missing:
        raise ComponentContractError(
            f"{trainer_class.__module__}.{trainer_class.__name__} is missing "
            f"Trainer methods: {', '.join(missing)}"
        )
```

```python
def run_trainer_lifecycle(
    trainer: Any,
    training_data: Any,
    validation_data: Any,
    test_data: Any,
    dataset_name: str,
    collect_metrics: bool = True,
) -> Mapping[str, Any]:
    """Execute the method-independent lifecycle used by the runner."""

    trainer.train(training_data, validation_data, test_data, dataset_name)
    if not collect_metrics:
        return {}
    return validate_metrics(trainer.get_evaluation_metrics())
```
> 모든 trainer는 5개 메서드(`load`, `train`, `predict`, `get_evaluation_metrics`, `set_model_dir`)를 가져야 하고, `validate_trainer_class`가 이를 검사한다. `run_trainer_lifecycle`은 `train`을 부른 뒤 `get_evaluation_metrics`를 부른다.

### 2-4. 방법별 yaml에서 갈리는 값

방법마다 `resize`, `imagesize`, `masksize`가 다르다. yaml에 없는 값은 `main.py:122-124`의 기본값 224를 쓴다. (`configs/*.yaml`)

| 방법 | 파일 | resize | imagesize | masksize |
|---|---|---|---|---|
| patchcore | `configs/patchcore.yaml:2-3` | 224 | 224 | 224 (기본값) |
| dinomaly | `configs/dinomaly.yaml:1-2` | 448 | 392 | 224 (기본값) |
| rd | `configs/rd.yaml:1-3` | 256 | 256 | 256 |
| simple | `configs/simple.yaml:4-5` | 288 | 288 | 224 (기본값) |

> 전처리 코드는 공통이지만 크기 값은 방법별 yaml이 정한다. 이미지 텐서의 크기는 `resize`가 정하고 마스크 텐서의 크기는 `masksize`가 정한다. 그래서 dinomaly는 이미지가 448×448이고 마스크는 224×224이며, simple도 이미지 288×288에 마스크 224×224이다.

### 2-5. 표를 실제로 쓰는 곳: `net()`

`main.py:232`

```python
    run([dataset(args), net(args)], args)
```

`main_net.py:59-111` (발췌)

```python
def net(args):
    """Return the legacy runner hook backed by the method registry."""

    backbone_names = list(args.backbone_names)
    layers_by_backbone = _group_layers(
        backbone_names, args.layers_to_extract_from
    )
    method_spec = get_method_spec(args.mainmodel)

    def get_simplenet(input_shape, device):
        trainers = []
        for raw_backbone_name, layers in zip(backbone_names, layers_by_backbone):
            backbone_name, backbone_seed = _parse_backbone_name(raw_backbone_name)
            backbone = method_spec.load_backbone(backbone_name)
            backbone.name = backbone_name
            backbone.seed = backbone_seed

            context = {
                "args": args,
                "input_shape": input_shape,
                "backbone_name": backbone_name,
                "layers_to_extract_from": layers,
            }
            trainer = method_spec.create(device, context)
            trainer.load(
                backbone=backbone,
                layers_to_extract_from=layers,
                device=device,
                input_shape=input_shape,
                ...
                args=args,
            )
            trainers.append(trainer)
        return trainers

    return "get_simplenet", get_simplenet
```

**입출력**

- `net(args)`의 반환값: 문자열 `"get_simplenet"`과 함수의 쌍
- 그 함수 `get_simplenet(input_shape, device)`의 반환값: trainer 객체의 리스트. 백본 하나당 trainer 하나가 만들어진다.
- `args.mainmodel`(예: `"patchcore"`) → `get_method_spec` → `MethodSpec` → `create()`로 `Trainer_PatchCore` 객체 → `load()`로 설정 전달

> `main()`이 `run([dataset(args), net(args)], args)`를 부른다. `net(args)`는 `("get_simplenet", get_simplenet)` 쌍을 돌려준다. `net()`이 `args.mainmodel`로 `get_method_spec`을 찾아 `method_spec`을 구하고, `get_simplenet(input_shape, device)`는 백본을 불러온 뒤 `method_spec.create()`로 trainer를 만들어 `trainer.load(...)`에 설정값을 넘긴다. 백본 이름 하나마다 trainer 하나가 만들어지고, 결과는 trainer 리스트이다. 방법이 바뀌어도 이 함수의 코드는 그대로이고 `method_spec`만 달라진다.

### 2-6. 실행 순서: `run()`

`main_run.py:18-30`, `44-106` (발췌)

```python
def train_and_collect_metrics(
    trainer, dataloaders, dataset_name, collect_metrics=True
):
    """Run the public Trainer lifecycle used by every registered method."""

    return run_trainer_lifecycle(
        trainer,
        dataloaders["training"],
        dataloaders["validation"],
        dataloaders["testing"],
        dataset_name,
        collect_metrics=collect_metrics,
    )
```

```python
    methods = {key: item for (key, item) in methods}
    ...
    list_of_dataloaders = methods["get_dataloaders"](0)

    device = utils.set_torch_device(gpu)
    ...
            imagesize = dataloaders["training"].dataset.imagesize
            simplenet_list = methods["get_simplenet"](imagesize, device)
    ...
            for i, SimpleNet in enumerate(simplenet_list):
                ...
                metrics = train_and_collect_metrics(
                    SimpleNet,
                    dataloaders,
                    dataset_name,
                    collect_metrics=True,
                )
```

**입출력**

- `methods`: `[dataset(args), net(args)]`(쌍 두 개)를 딕셔너리로 바꾼 것. 키는 `"get_dataloaders"`, `"get_simplenet"`
- `dataloaders`: `{"training": ..., "validation": ..., "testing": ...}`
- `imagesize`: `(3,224,224)` (`datasets/base.py:165`). 이 값이 `get_simplenet(imagesize, device)`의 `input_shape`로 들어가 `trainer.load(input_shape=...)`로 전달된다. 그 뒤 `set_aggregator`에서 `feature_aggregator.feature_dimensions(input_shape)`에 쓰인다 (`trainer/trainer.py:29-32`). `feature_dimensions` 안쪽 동작은 읽지 않았다.
- `metrics`: `train()` 뒤 `get_evaluation_metrics()`가 돌려주는 딕셔너리

> `run()`은 받은 쌍 두 개를 딕셔너리로 바꾸고 `methods["get_dataloaders"](0)`으로 데이터로더 묶음을 받는다. 묶음마다 `dataloaders["training"].dataset.imagesize`를 `get_simplenet`에 넘겨 trainer 리스트를 받고, trainer마다 `train_and_collect_metrics`를 부른다. 이 함수는 `run_trainer_lifecycle`로 `train()`을 부른 뒤 `get_evaluation_metrics()`의 결과를 돌려받는다. 결과는 `result_collect`에 모아 표(`result_df`)로 만들고 csv로 저장한다.

---

## 3. 후처리 (이상 점수·anomaly map)

### 3-1. 공통 경로: 예측 흐름

`trainer/trainer.py:232-256`

```python
    def _predict(self, images, mask_gt):
        images = images.to(torch.float).to(self.device)
        self.forward_modules.eval()
        batchsize = images.shape[0]
        with torch.no_grad():
            st = time()
            features, patch_shapes = self._embed(images)
            features = self._preprocessing_features_predict(features)
            patch_scores = image_scores = self._get_pred_scores(features)
            self.inftime += time() - st

            if isinstance(patch_scores, torch.Tensor):
                patch_scores = patch_scores.cpu().numpy()
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=batchsize)
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = self._preprocessing_predict(images, image_scores)
            image_scores = self._score(image_scores)

            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=batchsize)
            masks = self._get_segmentations(patch_scores, batchsize, patch_shapes)
            self.inftime_pixel += time() - st
            return list(image_scores), list(masks), list(features), list(patch_scores)
```

> 전체적 흐름은 특징 추출을 한 뒤 `_get_pred_scores`(방법이 채우는 칸)에서 패치 점수를 만들고, `_score`가 이를 이미지 점수로 바꾸며, `_get_segmentations`가 패치 점수를 지도로 변환한다. 틀은 공통이고 점수를 내는 방식만 방법마다 다르다.

**shape 변화** (B = 배치 크기, N = 이미지 한 장의 패치 수)

- 입력 `images`: `(B,3,224,224)`
- `_get_pred_scores` 출력 → `unpatch_scores`(`reshape(batchsize, -1, ...)`) → `(B,N)`
- 결과: `list(image_scores)` 길이 B, `list(masks)` 길이 B
- 배치 1일 때 `(784,)` → `(1,784)` → `(1,28,28)`이라는 주석이 `trainer/trainer_patchcore.py:153-155`에 있다. 이 값은 주석 기준이고 실행으로 확인한 값이 아니다.

### 3-2. 이미지 점수: 패치 점수 중 최댓값

`trainer/trainer.py:302-303`, `421-459` (PatchMaker의 `score`, `unpatch_scores`)

```python
    def _score(self, scores):
        return self.patch_maker.score(scores)
```

```python
    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        if was_numpy:
            return x.numpy()
        return x
```
> `PatchMaker.score`는 패치 점수 중 최대값을 이미지 점수로 쓴다. 한 칸만 이상해도 이미지 전체가 이상이 된다. `top_k>1`이면 상위 k개의 평균이다.

**shape 변화**

- `_predict`에서 `score`에 들어가기 전: `(B,N)` → `reshape(*shape[:2], -1)` → `(B,N,1)`
- `score`: `(B,N,1)` → `torch.max(dim=-1)` → `(B,N)` → `torch.max(dim=1)` → `(B,)`
- `top_k > 1`이면: `(B,N)` → `torch.topk(...).values.mean(1)` → `(B,)`

### 3-3. anomaly map: 패치 격자를 원본 크기로 키우고 가우시안 필터 적용

`trainer/trainer.py:311-314`

```python
    def _get_segmentations(self, patch_scores, batchsize, patch_shapes):
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        return self.anomaly_segmentor.convert_to_segmentation(patch_scores)
```

`common.py:97-121`

```python
class RescaleSegmentor:
    def __init__(self, device, target_size=224, smoothing=4):
        self.device = device
        self.target_size = target_size
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores):
        if patch_scores.shape[-1] != self.target_size[-1]: # NOTE target size 와 동일하면 굳이 처리할 필요없음
            with torch.no_grad():
                if isinstance(patch_scores, np.ndarray):
                    patch_scores = torch.from_numpy(patch_scores)
                _scores = patch_scores.to(self.device)
                _scores = _scores.unsqueeze(1)
                _scores = F.interpolate( _scores, size=self.target_size, mode="bilinear", align_corners=False)
                _scores = _scores.squeeze(1)
                patch_scores = _scores.cpu().numpy()

            return [
                ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
                for patch_score in patch_scores
            ]
        else:
            return [patch_score
                for patch_score in patch_scores
            ]
```

> 패치 점수를 (배치, 세로, 가로 ) 격자로 펴서 `RescaleSegmentor`에 넘긴다. 격자를 `bilinear`로 `224x224`까지 키우고, sigma=`last_score_blur_sigma`(기본 1.0)의 가우시안으로 각 픽셀을 자기와 주변 픽셀의 가중 평균으로 바꾼다.

**shape 변화** (h, w = `patch_shapes[0]`, T = `target_size`)

- `_get_segmentations`: `(B,N)` → `reshape(batchsize, scales[0], scales[1])` → `(B,h,w)`
- `RescaleSegmentor`: `(B,h,w)` → `unsqueeze(1)` → `(B,1,h,w)` → `interpolate` → `(B,1,T,T)` → `squeeze(1)` → `(B,T,T)`
- `gaussian_filter`: 이미지 한 장씩 적용하며 shape은 그대로 `(T,T)`. 결과는 `(T,T)` 배열 B개의 리스트
- `patch_scores.shape[-1] == target_size[-1]`이면 `interpolate`와 `gaussian_filter`를 건너뛰고 shape이 그대로다.
- T: `RescaleSegmentor`를 만들 때 `target_size=(masksize, masksize)`를 넘긴다. 기본값은 `(224,224)` (`trainer/trainer.py:63-64`, `main.py:124`)
- h, w: `patchify`가 `patchsize=3`, `stride=1`, `padding=1`로 패치를 만든다. 이 조건에서 패치 격자 크기는 첫 층 특징맵 크기와 같다 (`trainer/trainer.py:58`, `427-436`, `45`). 기본 백본은 `wideresnet50`, 추출 층은 `layer2`, `layer3`이다 (`main.py:141-142`). 특징맵의 실제 크기(224 입력일 때 28)는 백본 구조에서 나오는 값이라 이 코드만으로는 확인하지 못했다.
- `smoothing`: `RescaleSegmentor`의 기본값은 4지만, 만들 때 `smoothing=self.args.last_score_blur_sigma`를 넘겨서 덮어쓴다 (`trainer/trainer.py:65`). `last_score_blur_sigma`의 기본값은 `1.0`이다 (`main.py:166`).

### 3-4. 방법이 채우는 부분: PatchCore

`trainer/trainer_patchcore.py:162-163`

```python
    def _get_pred_scores(self, features):
        return self.anomaly_scorer.predict([features.cpu().numpy()])[0]
```

PatchCore는 `_predict`를 따로 정의하지 않고 위 `_get_pred_scores`만 채워서 3-1의 공통 경로를 쓴다. (`trainer_patchcore.py:138-160`의 `_predict`는 주석 처리돼 있다.)

> `_get_pred_scores` 한 칸만 채워서 3-1의 공통 경로를 그대로 쓴다. 자기 `_predict`는 주석 처리돼 있다. 그 안에선 정상 학습 패치와 최근접 거리를 점수로 쓴다.

**shape 변화**

- 입력 `features` → `anomaly_scorer.predict(...)[0]` → 패치별 점수. 배치 1일 때 `(784,)` (`trainer_patchcore.py:153`의 주석 기준)

### 3-4-1. PatchCore: 특징 추출 `_embed`

`trainer/trainer.py:125-169`

```python
    def _embed(self, images, evaluation=False, original_feature_list=False):
        B = len(images)
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            _ = self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]
        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        if original_feature_list:
            return features

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]
            _features = _features.reshape(_features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:])
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear", align_corners=False).squeeze(1)
            _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        features = self.forward_modules["preprocessing"](features)
        features = self.forward_modules["preadapt_aggregator"](features)
        return features, patch_shapes
```

`trainer/trainer.py:427-440` (`patchify`), `common.py:46-56` (`Preprocessing.forward`), `common.py:90-94` (`Aggregator`)

```python
    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)
        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features
```

```python
    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)
```

```python
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)
```

**shape 변화** (C = 특징 채널 수, N = 패치 수, D = `target_embed_dimension`)

- 입력 `images`: `(B,3,224,224)`
- 백본의 `layer2`, `layer3` 출력: `(B,C,h_i,w_i)`. 3차원 `(B,L,C)`이면 `reshape` 후 `permute`로 `(B,C,√L,√L)`
- `patchify`: `(B,C,h,w)` → `Unfold` → `reshape` → `permute(0,4,1,2,3)` → `(B,N,C,3,3)`. `patch_shapes`는 층마다의 격자 크기 `[h,w]`
- 둘째 층부터는 `interpolate`로 첫 층의 격자 크기에 맞춘다.
- `x.reshape(-1, *x.shape[-3:])` → `(B·N, C, 3, 3)`. `common.py:50-53`의 디버그 주석에 `(10368, 512, 3, 3)` → `(10368, 1536)`이 남아 있다. 어떤 설정에서 나온 값인지는 모른다.
- `preprocessing`: 층마다 `(B·N, D)`를 만들고 `torch.stack(dim=1)` → `(B·N, 층 수, D)`
- `preadapt_aggregator`: `reshape(len, 1, -1)` → `adaptive_avg_pool1d(target_dim)` → `(B·N, D)`
- 반환: `features` `(B·N, D)`, `patch_shapes`

> `_embed`는 이미지 배치를 백본(`feature_aggregator`)에 넣어 `layer2`, `layer3`의 feature map을 꺼낸다. `patchify`가 특징맵을 3×3 패치로 자르고(`Unfold`), 둘째 층의 패치 격자는 `interpolate`로 첫 층의 격자 크기에 맞춘다. 패치를 `(B·N, C, 3, 3)`으로 펴서 층마다 `preprocessing_modules`(`MeanMapper`, 안쪽은 읽지 않았다)를 거치고, `torch.stack`으로 `(B·N, 층 수, D)`를 만든다. `preadapt_aggregator`는 이를 한 줄로 펴서 `adaptive_avg_pool1d`로 D차원으로 줄여 `(B·N, D)`로 만든다. 반환값은 `(features, patch_shapes)`이다.

### 3-4-2. PatchCore: 정상 패치 모음 만들기 `_fill_memory_bank`

`trainer/trainer_patchcore.py:111-136`

```python
    def _fill_memory_bank(self, input_data, fit=True):
        """Computes and sets the support features for SPADE."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)[0]

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                feat = _image_to_features(image)
                feat = self._preprocessing_memory_bank(image, feat)
                features.append(feat.cpu())

        features = np.concatenate(features, axis=0)
        features = self._use_featuresampler(features)

        if fit:
            self.anomaly_scorer.fit(detection_features=[features])
        return features
```

`trainer/trainer_patchcore.py:19-24`

```python
class Trainer_PatchCore(Trainer):
    def initialize_model(self, **kwargs):
        self.coreset_ratio = self.args.coreset_ratio
        self.anomaly_scorer = NearestNeighbourScorer(
            n_nearest_neighbours=1, nn_method=FaissNN(False, 8))
        self.featuresampler = ApproximateGreedyCoresetSampler( self.coreset_ratio, self.device, num_coreset_samples=self.args.patchcore_coreset_num)
```

**shape 변화**

- 학습 이미지 배치 `(B,3,224,224)` → `_embed(...)[0]` → `(B·N, D)`
- 배치마다 나온 것을 `np.concatenate(axis=0)` → `(전체 학습 패치 수, D)`
- `_use_featuresampler`(코어셋 샘플링) → `(줄어든 패치 수, D)`. 남는 개수는 `coreset_ratio`와 `patchcore_coreset_num`으로 정해진다. 샘플러 내부는 읽지 않았다.
- `anomaly_scorer.fit`: 이 특징들을 FAISS 색인에 넣는다.

> `_fill_memory_bank`는 학습 데이터를 배치씩 `_embed`에 넣어 `(B·N, D)` 특징을 모으고 `np.concatenate`로 전부 이어붙인다. 이를 `featuresampler`(`ApproximateGreedyCoresetSampler`)에 넣어 개수를 줄인 뒤, `fit=True`이면 `anomaly_scorer.fit`으로 FAISS 색인에 넣는다. 이 색인이 테스트 때 검색 대상이 되는, 학습 이미지에서 나온 패치 모음이다. 

### 3-4-3. PatchCore: 점수 계산 `NearestNeighbourScorer.predict`

`trainer/trainer_patchcore.py:181-183`, `203-221`, `297-303`, `321-335` (발췌)

```python
        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
```

```python
    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns
```

```python
    def _create_index(self, dimension):
        logger.info("현재 faiss 는 IndexFlatL2 를 쓰는 중")
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)
```

```python
    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)
        ...
```

**shape 변화** (Q = 검색할 패치 수)

- 입력 `query_features`: `(Q, D)`. 배치 `B`장이면 `Q = B·N`
- `search_index.search(query, 1)`: 각 패치마다 가장 가까운 학습 패치 1개를 찾아 `query_distances`와 `query_nns`를 돌려준다. faiss가 `(Q,k)` 모양으로 돌려주는 것은 라이브러리 규칙이고 이 코드에는 없다. `k = n_nearest_neighbours = 1`
- `np.mean(query_distances, axis=-1)` → `(Q,)`
- 이 `(Q,)`가 `trainer_patchcore.py:162-163`의 `_get_pred_scores` 반환값이고, 3-1의 `patch_scores`가 된다.

> `NearestNeighbourScorer.predict`는 검색할 패치 특징 `(Q, D)`를 `FaissNN.run(1, query)`에 넘긴다. `run`은 FAISS 색인(`IndexFlatL2`)에서 각 패치와 가장 가까운 학습 패치 1개를 찾아 거리와 위치를 돌려준다. 거리에 `np.mean(..., axis=-1)`을 적용해 패치별 점수 `(Q,)`가 나오고, 검색 개수가 1이라 평균값은 그 거리 하나와 같다. 이 점수가 `_get_pred_scores`의 반환값이 되어 3-1의 `patch_scores`가 된다.

### 3-5. 공통 경로를 쓰지 않는 방법: Dinomaly

`trainer/trainer_dinomaly.py:150-191`

```python
    @torch.no_grad()
    def predict(self, datas):
        scores_img = []
        scores_maps = []

        gt_list = []
        gt_mask_list = []
        
        self.Dinomaly.eval()
        gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)
        with tqdm.tqdm(datas, desc="Inferring...", leave=False) as data_iterator:
            for data_item in data_iterator:
                gt_list += [data_item['is_anomaly']]
                name = data_item['image_name']
                image = data_item['image'].to(self.device)
                
                output = self.Dinomaly(image)
                en, de = output[0], output[1]
                score_map, _ = cal_anomaly_maps(en, de, image.shape[-1])
                
                score_map = F.interpolate(score_map, size=256, mode='bilinear', align_corners=False)
                gt = F.interpolate(data_item['mask'], size=256, mode='nearest')
                gt_mask_list += [gt]
                
                score_map = gaussian_kernel(score_map).cpu()
                
                anomaly_map = score_map.flatten(1)
                score_img = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * 0.01)]
                score_img = score_img.mean(dim=1)
                
                scores_maps += score_map
                scores_img += score_img

        gt_mask_list = np.asarray(gt_mask_list, dtype=int)
        scores_maps = np.asarray(scores_maps, dtype=float)

        segmentations = [torch.tensor(ma).squeeze() for ma in scores_maps]
        masks_gt = [torch.tensor(ma) for ma in gt_mask_list]
        scores = [sc.cpu().item() for sc in scores_img]
        labels_gt = [gt.cpu().item() for gt in gt_list]

        return scores, segmentations, None, labels_gt, masks_gt
```

(193줄 이후는 `return` 뒤라 실행되지 않는 코드라 제외함)

> 공통 경로를 안쓰고 `predict()` 안에서 전부 처리한다. 인코더와 디코더 출력 차이로 지도를 만들고, 256으로 키운 뒤 가우시안 커널(`kernel_size=5`, `sigma=4`)로 각 픽셀을 주변 픽셀과의 가중 평균으로 바꾼다. 이미지 점수는 지도 값 상위 1%의 평균이다. PatchCore는 패치 점수의 최댓값을 이미지 점수로 쓰고 지도는 `RescaleSegmentor`로 만드는 반면, Dinomaly는 `predict()` 안에서 지도와 점수를 모두 만든다.

**shape 변화**

- 입력 `image`: `(B,3,H,W)`
- `cal_anomaly_maps(en, de, image.shape[-1])`: 층마다 `1 - cosine_similarity` → `unsqueeze(1)` → `interpolate`로 `(B,1,H,W)` → 층별 지도를 `cat` 후 `mean(dim=1, keepdim=True)` → `(B,1,H,W)` (`trainer/Dinomaly_lib/utils.py:208-221`)
- `interpolate(size=256)` → `(B,1,256,256)`
- 마스크: `(B,1,224,224)` → `interpolate(size=256, mode='nearest')` → `(B,1,256,256)`
- `gaussian_kernel(score_map)` 적용 후: `get_gaussian_kernel`의 안쪽은 읽지 않았고, 뒤의 `flatten(1)`이 `(B,65536)`을 전제로 하므로 shape이 그대로라고 본다.
- `flatten(1)` → `(B,65536)` → 내림차순 정렬 후 상위 `int(65536×0.01) = 655`개만 남김 → `mean(dim=1)` → `(B,)`
- 최종 `scores`: 길이 B의 파이썬 float 리스트. `segmentations`: `squeeze()`한 지도의 리스트


