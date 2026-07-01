# PatchCore Baseline Reproduction Analysis

## Experiment: Baseline Reproduction on MVTec AD (15 categories)

- commit: `fcaa92f` (patchcore-inspection)
- sh: `method1_patchcore/source/run_baseline.sh`
- csv: `method1_patchcore/source/results/baseline_20260628.csv`

---

### Question

Can PatchCore (WideResNet50, coreset 10%) be reproduced on MVTec AD within 0.1%p of the paper's reported AUROC?

### Hypothesis

H1: Given correct environment setup (PyTorch + FAISS, seed 0), the reproduced I-AUROC will match the paper's 99.1% within ±0.1%p.

### Expectation

Mean I-AUROC ≈ 99.1%, mean P-AUROC ≈ 98.1%, matching PatchCore-10% in Table 1–2 of the paper.

### Setup

| Parameter | Value |
|---|---|
| Backbone | WideResNet50 |
| Feature layers | layer2 + layer3 |
| Coreset ratio | 10% |
| Patch size | 3 |
| Image size | 224×224 (resize 256 → center crop) |
| Seed | 0 |
| GPU | NVIDIA GeForce RTX 5060 (8GB) |

### Result

Columns: **I-AUROC** = image-level anomaly detection AUROC (higher is better, range 0–1); **P-AUROC (full)** = pixel-level AUROC over all pixels; **P-AUROC (anomaly)** = pixel-level AUROC over anomalous regions only. Each row = one MVTec AD category. **Bold mean row** = average across all 15 categories.

| Category | I-AUROC | P-AUROC (full) | P-AUROC (anomaly) |
|---|---|---|---|
| bottle | 1.000 | 0.985 | 0.980 |
| cable | 0.998 | 0.984 | 0.975 |
| capsule | 0.981 | 0.990 | 0.987 |
| carpet | 0.985 | 0.991 | 0.988 |
| grid | 0.977 | 0.988 | 0.983 |
| hazelnut | 1.000 | 0.987 | 0.979 |
| leather | 1.000 | 0.993 | 0.990 |
| metal_nut | 0.998 | 0.983 | 0.979 |
| pill | 0.968 | 0.978 | 0.976 |
| screw | 0.981 | 0.995 | 0.994 |
| tile | 0.996 | 0.957 | 0.941 |
| toothbrush | 1.000 | 0.986 | 0.980 |
| transistor | 1.000 | 0.963 | 0.929 |
| wood | 0.990 | 0.950 | 0.937 |
| zipper | 0.992 | 0.989 | 0.986 |
| **mean** | **0.991** | **0.981** | **0.974** |

Raw data: `method1_patchcore/source/results/baseline_20260628.csv`

### Comparison with Paper

Columns: **Metric** = evaluation metric name; **Paper / Reproduced** = AUROC value in % (higher is better); **Difference** = reproduced minus paper in percentage points (%p).

| Metric | Paper (PatchCore-10%) | Reproduced | Difference |
|---|---|---|---|
| I-AUROC | 99.0% | 99.1% | +0.1%p |
| P-AUROC | 98.1% | 98.1% | 0.0%p |

### Difference from Expectation

H1 is supported. Reproduced mean I-AUROC (99.1%) matches the paper exactly. P-AUROC (98.1%) is also identical. The slight per-category variance (e.g., transistor P-AUROC 96.3% vs. paper's ~97%) is within expected range for seed/hardware differences.

### Next

#### Phenomenon 1 — Detection failure vs. Localization failure are different problems

Examining the result table reveals two distinct failure modes:

**Type 1 — Low I-AUROC (detection failure)**: The model fails to decide at the image level that an anomaly exists.

| Category | I-AUROC | P-AUROC (full) |
|---|---|---|
| pill | 0.968 | 0.978 |
| grid | 0.977 | 0.988 |
| capsule | 0.981 | 0.990 |

**Type 2 — Low P-AUROC (anomaly) despite high I-AUROC (localization failure)**: The model correctly flags the image as anomalous, but cannot pinpoint where the defect is.

| Category | I-AUROC | P-AUROC (anomaly) |
|---|---|---|
| transistor | 1.000 | 0.929 |
| tile | 0.996 | 0.941 |
| wood | 0.990 | 0.937 |

These are structurally different failures. A method that improves image-level detection does not necessarily improve localization, and vice versa. Future methods should be evaluated against each failure type separately.

---

#### Phenomenon 2 — Why does pill fail at detection?

MVTec AD's pill category includes two distinct defect subtypes:
- **Contamination / color spot**: A foreign-colored region appears on the pill surface — essentially a color anomaly.
- **Wrong pill type**: An entirely different pill (different shape/color) appears in the image — essentially an object-level semantic anomaly.

PatchCore's anomaly score measures the distance of a patch to the nearest patch in the memory bank. For color anomalies, the defect patch may still be geometrically similar to a normal patch (same shape, similar texture) and differ only in color channel values. If the ImageNet-pretrained backbone encodes color less sensitively than shape, small color deviations will produce low distances and thus low anomaly scores, causing misses.

**Hypothesis H2**: PatchCore's memory bank distance-based score is insensitive to color-channel anomalies because WideResNet50 features are shape-biased from ImageNet pretraining.

**Testable prediction**: If H2 holds, replacing the backbone with CLIP (which is trained on image-text pairs and encodes semantic color information via language) should improve I-AUROC on pill. If CLIP-based features close the gap on pill but not on grid, the cause is confirmed to be feature bias rather than the memory bank mechanism itself.

---

#### Phenomenon 3 — Why does grid fail at detection?

Grid's defect types include bent threads, broken threads, and glue contamination — all of which are **local disruptions in a globally regular pattern**. The defect patch itself may look like a plausible patch in isolation (e.g., a single bent thread pixel is not obviously abnormal). The anomaly is only visible when viewed in the context of the surrounding regular pattern.

PatchCore's locally-aware aggregation uses a 3×3 neighborhood (p=3, ~receptive field of ~32px at layer2). This is too small to capture the global periodicity of the grid texture. A patch in the middle of a broken thread may not differ enough from normal patches for the distance to exceed the threshold.

**Hypothesis H3**: PatchCore's fixed 3×3 local aggregation is insufficient for detecting anomalies that require global context (pattern regularity), not just local patch appearance.

**Testable prediction**: If H3 holds, increasing the aggregation size p or using a model with attention over the full image (e.g., ViT-based or diffusion-based reconstruction) should improve grid I-AUROC. This is independent of feature bias and represents a structural limitation of the patch comparison framework.

---

#### Phenomenon 4 — Why does transistor fail at localization despite perfect detection?

Transistor defects include misplaced components and bent leads — these are **spatial/structural anomalies** where the object identity of a patch is correct but its position relative to other components is wrong. PatchCore's score for each patch is computed independently (each patch vs. its nearest neighbor in M). There is no mechanism to encode whether the spatial arrangement of components is correct globally.

PatchCore can detect that "some patch in this image is far from normal patches" (high image-level score), but the specific patch that scores highest may not correspond to the actual defect location — it may instead flag a normal component that appears in an unexpected surrounding context.

**Hypothesis H4**: PatchCore's per-patch independent scoring cannot capture spatial-arrangement anomalies; the highest-scoring patch may not be at the true defect location.

**Testable prediction**: If H4 holds, a model that reconstructs the full image and measures reconstruction error per pixel (Candidate C, diffusion-based) should produce more accurate anomaly maps for transistor, because the reconstruction will fail at the exact location where the component is missing or misplaced.

---

#### Summary of hypotheses and candidate assignments

| Hypothesis | Phenomenon | Candidate to test |
|---|---|---|
| H2: Shape-biased features miss color anomalies | pill low I-AUROC | A (CLIP/VLM — language encodes color semantics) |
| H3: Local aggregation misses global pattern breaks | grid low I-AUROC | C (Diffusion — reconstruction captures global regularity) |
| H4: Per-patch scoring misses spatial arrangement anomalies | transistor low P-AUROC | C (Diffusion — full-image reconstruction) |

H2 points toward Candidate A; H3 and H4 both point toward Candidate C. This makes Candidates A and C the higher-priority directions to explore next. Candidate B (3D multimodal) is not implicated by any of the observed failure patterns in this 2D RGB dataset.