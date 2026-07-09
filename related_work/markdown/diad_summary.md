# DiAD Summary

## 논문 메타데이터

| 항목 | 내용 |
|---|---|
| 제목 | DiAD: A Diffusion-based Framework for Multi-class Anomaly Detection |
| 저자 | He et al. |
| 학회 / 저널 | AAAI 2024 |
| 연도 | 2023 (arXiv) / 2024 (AAAI) |
| 논문 링크 | https://arxiv.org/abs/2312.06607 |
| GitHub / 공식 코드 | https://github.com/lewandofskee/DiAD (공식, 저자 본인 리포) |
| 조사 이유 | Candidate C — H3/H4 검증 (PatchCore의 3×3 local aggregation이 global pattern anomaly를 못 잡을 가능성(H3), per-patch independent scoring이 spatial arrangement anomaly를 못 잡을 가능성(H4)). H2(Candidate A / WinCLIP)가 zero-shot·1-shot 모두에서 반박된 뒤, 남은 후보(A 연장 vs C) 검토 결과 C를 선택 |

---

## 문제 정의

Diffusion 모델은 이미지 재구성 성능이 뛰어나지만, 보다 실용적인 **multi-class** 세팅(카테고리마다 모델을 따로 두지 않고 하나의 모델로 전체를 처리)에서는 재구성 과정에서 카테고리 정체성과 픽셀 단위 구조적 정합성을 유지하기 어렵다 — 재구성된 이미지가 다른 카테고리의 외형으로 흘러가거나 세부 구조를 잃을 수 있는데, 이는 "입력과 재구성 결과를 비교"하는 재구성 기반 방법의 이상 탐지 신호 자체를 무너뜨린다.

**핵심 질문**: Diffusion 기반 재구성 방법이 단일 multi-class 모델 안에서도 의미(카테고리)와 픽셀 단위 구조를 동시에 충분히 보존해서, 입력 대비 재구성 차이를 통해 이상을 탐지·위치추정할 수 있는가?

---

## 핵심 아이디어

Denoising 과정을 입력 이미지 자체에서 추출한 semantic 정보로 guide하여, 무조건부(unconditional) 생성이나 class label만으로 conditioning하는 대신 재구성이 입력의 카테고리·구조에 충실하게 유지되면서도 이상 영역은 정상 형태로 "치유"되도록 만든다.

PatchCore가 저장된 정상 패치와의 거리로, WinCLIP이 언어로 정의한 정상/이상 개념과의 거리로 이상을 판단한다면, DiAD는 diffusion 모델의 재구성 결과가 원본과 얼마나 **다른지**로 판단한다 — 차이가 클수록 그 영역을 모델이 이상으로 간주해 정상으로 되돌려 재구성했다는 뜻이다.

---

## 방법 개요

---

## 결과


---

## 한계

---
