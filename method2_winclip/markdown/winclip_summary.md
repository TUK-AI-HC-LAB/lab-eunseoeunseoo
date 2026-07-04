# WinCLIP Summary

## Paper Metadata

| Item | Content |
|---|---|
| Title | WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation |
| Authors | Jeong et al. |
| Conference / Journal | CVPR 2023 |
| Year | 2023 |
| Paper link | https://arxiv.org/abs/2303.14814 |
| GitHub / Official code | https://github.com/caoyunkang/WinClip |
| Reason for investigation | Candidate A — addresses H2 (ImageNet/shape bias of PatchCore on color anomalies like pill) |

---

## Problem

Existing industrial anomaly detection requires training a separate model for each product category. With few defect images and an enormous variety of product types, this approach does not scale.

**Core question**: Can anomalies be detected with no training at all (zero-shot), or with only a handful of normal images (few-shot)?

---

## Key Idea

Use CLIP (a vision-language model pretrained on image-text pairs) to define "normal" and "anomalous" states as text, then score each image patch by how close it is to those textual concepts.

Where PatchCore scores anomalies by distance to stored normal patches, WinCLIP scores them by distance to language-defined normal/anomalous concepts.

---

## Method (Overview)

**Two core components:**

1. **Compositional Prompt Ensemble (CPE)**
   - Combines state-level words — normal: "flawless", "perfect"; anomalous: "damaged", "with flaw" — to generate a diverse set of text embeddings
   - More robust than any single manually written prompt

2. **Window-based CLIP (WinCLIP)**
   - CLIP natively aligns text only with the full image → poor for pixel-level localization
   - Sliding windows of multiple sizes crop local regions, each passed through CLIP independently
   - Multi-scale scores (2×2, 3×3, full image) are aggregated into a pixel-level anomaly map

**WinCLIP+**: When 1–4 normal reference images are available, combines language-guided scores with cosine similarity to stored normal visual features.

---

## Results (Brief)

| Setting | MVTec AD I-AUROC | MVTec AD P-AUROC |
|---|---|---|
| WinCLIP (zero-shot) | 91.8% | 85.1% |
| WinCLIP+ (1-shot) | 93.1% | 95.2% |
| PatchCore-10% (full-shot, reference) | 99.0% | 98.1% |

- 91.8% with zero training data — strong for a zero-shot method
- 1 normal image gets P-AUROC to 95.2% — approaches PatchCore

---

## Limitations (Brief)

- **Logical anomalies** (missing components, misplaced parts): cannot be defined by language alone → partially addressed by WinCLIP+
- **Tiny defects**: window resolution limits sub-pixel detection
- **Localization precision**: anomaly region is found but boundary edges are coarse

---

## TODO

- [ ] Read paper directly and fill in Method details
- [ ] Fill in experimental setup details (backbone, image size, baselines)
- [ ] Connect to PatchCore H2: check if WinCLIP outperforms PatchCore on pill category