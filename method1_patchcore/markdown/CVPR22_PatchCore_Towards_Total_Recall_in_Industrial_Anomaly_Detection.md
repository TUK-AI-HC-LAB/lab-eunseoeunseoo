# PatchCore Summary

## Paper Metadata

| Item | Content |
|---|---|
| Title | Towards Total Recall in Industrial Anomaly Detection |
| Authors | Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler |
| Conference / Journal | CVPR 2022 |
| Year | 2022 |
| Paper link | https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.html |
| GitHub / Official code | https://github.com/amazon-science/patchcore-inspection |
| Reason for investigation | First method in the lab's IAD onboarding path; baseline that all lab members reproduce first |

---
# PatchCore Paper Summary

## Problem

Three limitations of existing pre-trained model-based industrial anomaly detection:

1. **ImageNet Bias** : Top-layer features are abstract and class-biased, making them unsuitable for capturing subtle, localized industrial defects.
2. **Limited Utilization of Normal Information** : Relies on only a small number of high-level feature vectors, restricting contextual use of normal data.
3. **Performance-Speed Trade-off** : Using all high-resolution features improves detection but causes inference time and memory to grow exponentially.

**Research Hypothesis** : Can building a memory bank from mid-level patch features of normal images reduce ImageNet bias while simultaneously achieving top accuracy in both detection and segmentation, along with fast inference?

## Key Idea

PatchCore models the distribution of normal data **non-parametrically**. A pre-trained CNN converts normal images into a set of patch feature vectors, stored in a **memory bank M**. At test time, the proximity of each patch feature to the normal patches in M is measured; a patch far from all normal patches is flagged as anomalous. The image-level anomaly score is determined by the score of the most anomalous patch.

## Method

**Framework** : Feature Extraction → Memory Bank Construction → Coreset Reduction → Anomaly Score Computation

1. **Feature Extraction** : Extract feature maps φ_{i,j} from intermediate layers j, j+1 of network φ. Aggregate (p×p) neighborhood information at each patch position (h,w) in a locally-aware manner to produce patch features.
2. **Memory Bank Construction** : Collect all patch features from every normal training image x_i ∈ X_N into memory bank M.
3. **Coreset Reduction** : Apply greedy coreset subsampling to remove redundancy from M and select the subset M_c that best represents the distribution.
4. **Anomaly Score Computation** : Compute the maximum distance s* between test patch m_test and its nearest neighbor m* in M_c, then re-weight it based on surrounding patch context to derive the final anomaly score s.

### Experimental Setup

**Datasets**
- MVTec AD : 15 sub-datasets, 5,354 images total. Standard benchmark for industrial anomaly detection.
- Magnetic Tile Defects : 925 defect-free + 392 defective tile images. Specialized dataset with diverse lighting and scale variations.
- Mini Shanghai Tech Campus : Pedestrian video dataset. Used to evaluate applicability to non-industrial domains.

**Equipment / Software**
- GPU : Nvidia Tesla V4 / Software : Python 3.7, PyTorch, Faiss
- Default backbone : WideResNet50 (ResNet50, ResNet101, ResNeXt101 also used for comparison)

**Experimental Conditions**

| Parameter | Value | Description |
|---|---|---|
| Input image size | 224×224 (default), resize to 256×256 then center crop | Fair comparison with prior works |
| Data augmentation | Not used | Requires prior knowledge of class-preserving augmentations |
| Feature extraction layers | Final output of WideResNet50 blocks 2 and 3 | To utilize mid-level features |
| Local aggregation size | p=3 | Optimal value determined by Figure 4 experiments |
| Coreset sampling ratio | 25%, 10%, 1% | Measuring performance change by compression level |
| Anomaly map smoothing | Gaussian kernel, σ=4 | Improve visual quality of segmentation map |

**Procedure**

Training phase: Input normal images → extract feature maps from WideResNet50 layers 2 & 3 → aggregate p=3 neighborhoods to generate patch features → store in memory bank M → apply coreset sampling to produce M_c.

Test phase: Input image → extract P(x_test) in the same manner → for each m_test ∈ P(x_test), search for the nearest neighbor in M_c → compute anomaly score s via Eq. (7) → remap and upsample per-patch scores to generate anomaly segmentation map.

Baselines: Compared against SPADE, PaDiM, DifferNet, PatchSVDD, and other prior works.

## Results

**Table 1 : Image-level Anomaly Detection Performance (AUROC, %)**

| Method | AUROC (↑) | Error Rate (%) (↓) | Misclassifications (↓) |
|---|---:|---:|---:|
| SPADE | 95.5 | 14.5 | - |
| PatchSVDD | 92.1 | 7.9 | - |
| DifferNet | 94.9 | 5.1 | - |
| PaDiM | 97.9 | 2.1 | - |
| **PatchCore-25%** | **99.1** | **0.9** | **42** |
| **PatchCore-10%** | 99.0 | 1.0 | 47 |
| **PatchCore-1%** | 99.0 | 1.0 | 49 |

**Table 2 : Pixel-level Anomaly Segmentation Performance (AUROC, %)**

| Method | AUROC (↑) | Error Rate (%) (↓) |
|---|---:|---:|
| PatchSVDD | 95.7 | 4.3 |
| SPADE | 96.0 | 4.0 |
| PaDiM | 97.5 | 2.5 |
| **PatchCore-25%** | **98.1** | **1.9** |
| **PatchCore-10%** | **98.1** | **1.9** |
| **PatchCore-1%** | 98.0 | 2.0 |

**Figure Analysis**

- **Figure 3** : 2D distribution coverage comparison of coreset vs. random subsampling. In multi-modal distributions, random sampling misses some clusters while coreset better approximates the full spatial support; coreset also shows more uniform coverage in uniform distributions.
- **Figure 4** : Analysis of locally-aware patch aggregation and layer dependency. (Top) Detection/segmentation AUROC is highest at neighborhood size p=3 — both too-local and too-global information degrade performance. (Bottom) Combining layers 2 and 3 (2+3) outperforms either layer alone — confirming synergy across abstraction levels.
- **Figure 5** : Performance retention by sampling method. Coreset maintains near-identical AUROC even at 1% subsampling ratio, while Random and Learned methods degrade sharply — demonstrating the overwhelming efficiency of coreset subsampling.
- **Figure 6** : Low-shot performance. PatchCore-10 outperforms PaDiM and SPADE across all data regimes, approaching previous SOTA with less than 5% of the full training data.

## Findings

- **Finding 1 — New SOTA Achieved** : PatchCore achieves image-level AUROC 99.1% and pixel-level AUROC 98.1% on MVTec AD, reducing the image-level error rate from 2.1% (PaDiM) to 0.9% — a 57% reduction. Validates the effectiveness of the comprehensive memory bank-based patch comparison approach.

- **Finding 2 — Balancing Performance and Efficiency** : Compressing M_c to just 1% of the full memory bank still maintains AUROC 99.0% (Figure 5), as minimax-based coreset selection effectively eliminates redundancy. Inference time (0.17s for PatchCore-1%) is also faster than PaDiM (0.19s), achieving production-viable speed with no performance loss.

- **Finding 3 — Optimality of Locally-Aware Mid-Level Features** : Figure 4 ablation experimentally validates the design choices. Neighborhood size p=3 is optimal — single-pixel features are noise-prone and overly large receptive fields dilute defect locality. Combining mid-level layers (block 2+3) gives the best performance — simultaneously overcoming ImageNet bias in high-level features and the lack of context in low-level features.

- **Finding 4 — Superior Sample Efficiency** : With only 5% of training data (10 samples per dataset), image-level AUROC already reaches 93.6% (Figure 6). The memory bank preserves each normal sample individually, making it highly effective for industrial cold-start scenarios where few normal samples are available.

### Summary — How Theory and Model Were Validated by Results

PatchCore's hypothesis — "an efficiently reduced, locally-aware, mid-level patch feature memory bank modeled non-parametrically is highly effective for anomaly detection" — is thoroughly validated across all experiments.

- **Table 1, 2** → Demonstrates superiority of the memory bank + nearest-neighbor distance approach
- **Figure 5, Table 5** → Confirms that coreset theory (Eq. 5) translates directly into real performance and speed gains
- **Figure 4** → Validates local aggregation (Eq. 2) and mid-level layer selection as optimal design choices
- **Figure 6** → Confirms the inherent data efficiency of the non-parametric memory bank

Each theoretical component is individually validated, and their combined synergy produces the overall SOTA performance.

## Limitations

- **Author-stated Limitation** : Performance fundamentally depends on the transferability of pre-trained features. If the target industrial domain differs significantly in distribution from ImageNet, the feature extractor may degrade and reduce overall effectiveness.
- **Hyperparameter Sensitivity** : Feature layers (2+3), neighborhood size (p=3), and smoothing (σ=4) were empirically chosen based on MVTec AD. Whether these remain optimal for entirely new product types or defect categories requires further validation.
- **Backbone Architecture Dependency** : Study is focused on ResNet-family CNNs. It is unclear how PatchCore performs with Vision Transformers (ViT) or other architectures that extract patch features differently.
- **Anomaly Score Re-weighting** : The re-weighting in Eq. (7) is closer to an empirical heuristic. There is room to introduce a more principled probabilistic framework that accounts for neighborhood density and distribution.

## Open Questions

The following connects each stated limitation of PatchCore to a candidate research direction (A/B/C).

### Q1. ImageNet Bias → Candidate A (VLM/CLIP-based IAD)

**Paper's stated limitation**: "Performance fundamentally depends on the transferability of pre-trained features. If the target domain differs significantly from ImageNet, the feature extractor may degrade."

PatchCore uses patch features extracted from an ImageNet-pretrained CNN (WideResNet50) as the definition of "normal." Because the backbone was trained to classify ImageNet objects, its representations may fail to capture subtle industrial defects (e.g., micro-scratches on metal, texture-level anomalies) when the visual domain diverges from ImageNet.

**Connection**: Vision-language models such as CLIP are trained on image-text pairs, allowing defect concepts like "scratched surface" or "broken edge" to be encoded directly via text descriptions. This shifts the dependency from ImageNet class distributions to language-defined defect semantics, potentially reducing domain-transfer failure.
**Core hypothesis for Candidate A**: Does text-based supervision reduce reliance on ImageNet feature transferability for industrial anomaly detection?

---

### Q2. Validated Only on ResNet Family → Candidate B (3D Multimodal IAD)

**Paper's stated limitation**: "The study is focused on ResNet-family CNNs. It is unclear how PatchCore performs with Vision Transformers or other architectures."

This limitation asks "what happens if we change the backbone?" Candidate B instead changes the **input modality** (RGB → RGB + point cloud), making the connection indirect. However, both fall under the broader gap that PatchCore has only been validated in the narrow design space of 2D RGB + CNN. RGB images alone carry no surface geometry information, so geometry-based defects (dents, deformations) with minimal color change may be missed entirely.

**Connection (indirect)**: Adding depth/point cloud data as a complementary modality could compensate for geometry-based defects that 2D patch features cannot represent.
**Core hypothesis for Candidate B**: Does combining RGB patch features with 3D geometry information improve detection of shape-based anomalies that produce little color change?

> Honest assessment: This connection is weaker than A and C. It extends from a broader motivation ("PatchCore's input space is narrow") rather than a limitation the authors directly stated. Worth re-examining in the next meeting.

---

### Q3. Anomaly Score Re-weighting is a Heuristic → Candidate C (Diffusion-based IAD)

**Paper's stated limitation**: "The re-weighting in Eq. (7) is closer to an empirical heuristic. There is room to introduce a more principled probabilistic framework that accounts for neighborhood density and distribution."

PatchCore's anomaly score is computed as the distance to the nearest normal patch in the memory bank, re-weighted by a heuristic factor (Eq. 7). This is a distance-based approximation, not a direct model of the data distribution.

**Connection**: Diffusion models learn the distribution of normal data directly. During denoising, the reconstruction error at each patch position reflects how far that region deviates from the learned normal distribution — a principled, likelihood-grounded signal rather than a heuristic distance. This could replace PatchCore's empirical re-weighting with a probabilistically motivated anomaly score.
**Core hypothesis for Candidate C**: Does the reconstruction error of a diffusion model provide a more principled and accurate anomaly signal than PatchCore's distance-based heuristic?

---

### Summary

| Candidate | Connected Limitation | Connection Strength |
|---|---|---|
| A (VLM/CLIP) | ImageNet feature dependency (author-stated) | Strong — directly from the paper |
| C (Diffusion) | Heuristic anomaly score re-weighting (author-stated) | Strong — directly from the paper |
| B (3D Multimodal) | Narrow validated input space (inferred, not author-stated) | Weak — extended by inference; needs validation |
