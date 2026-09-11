# WinCLIP Summary

## Paper Metadata

| Item | Content |
|---|---|
| Title | WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation |
| Authors | Jeong et al. |
| Conference / Journal | CVPR 2023 |
| Year | 2023 |
| Paper link | https://arxiv.org/abs/2303.14814 |
| GitHub / Official code | none — paper authors released no official code. Unofficial re-implementations only; accuracy varies widely (see Implementation Notes below) |
| Reason for investigation | Candidate A — addresses H2 (shape bias of PatchCore missing color anomalies like pill) |

### Implementation Notes

No official code exists for WinCLIP. Two unofficial re-implementations were checked:

| Repo | MVTec zero-shot mean I-AUROC (own reported) | Fidelity to paper (91.8%) |
|---|---|---|
| [caoyunkang/WinClip](https://github.com/caoyunkang/WinClip) | 70.17% | Poor — large per-category gaps (e.g. grid 98.8%→48.9%, metal_nut 97.1%→37.6%) |
| [mala-lab/WinCLIP](https://github.com/mala-lab/WinCLIP) | ~91.2% (via CVPR'24 InCTRL paper, which uses this code for its WinCLIP baseline) | Good — matches paper within ~0.6%p in aggregate |

Using `mala-lab/WinCLIP` for reproduction (`method2_winclip/source/WinCLIP/`) because it is the only candidate with external validation (used as a baseline in a peer-reviewed paper).

**Risk carried forward**: this guide's paper-selection priority (`README.md` 10장) ranks "official GitHub code exists" above candidates without it. WinCLIP fails that criterion — the absence of official code is exactly why the two unofficial re-implementations diverge so much. Cross-checking against the paper's own published per-category table (see below) is used to compensate.

**Published pill zero-shot I-AUROC**: 79.10% — from the "Reported" column, pill row, of the per-category table in [caoyunkang/WinClip README](https://github.com/caoyunkang/WinClip) (i-auroc column; that table transcribes the original paper's per-category results, distinct from that repo's own unreliable re-implementation column, which is not used here). This is already well below PatchCore's pill result (96.8%), meaning the H2 zero-shot prediction may already be contradicted by the paper's own numbers before any local reproduction.

*Correction (2026-07-08): an earlier version of this note claimed this number was "confirmed from two independent sources." The second source was a web search snippet describing "0.7908 in segmentation results" from an unspecified unofficial implementation — ambiguous whether it meant image-level or pixel-level AUROC, so it does not actually corroborate the image-level figure above and the claim was overstated. Only the caoyunkang README table is used as the source now.*

---

## Problem

Existing industrial anomaly detection methods require a separately trained model for each product category. Two practical limitations prevent this from scaling:

1. **Defect images are rare** — defects occur infrequently in production, making it hard to collect sufficient training samples.
2. **Too many product types** — building individual models for hundreds of categories across aerospace, automotive, pharmaceutical, and electronics is not feasible.

**Core question**: Can anomalies be detected with no training at all (zero-shot), or with only a handful of normal images (few-shot)?

**Key insight**: "Normal" and "anomalous" are context-dependent concepts that language can define naturally. A vision-language model like CLIP can use text such as "a cracked bottle" or "a contaminated pill" to define defects without any task-specific training.

---

## Key Idea

Use CLIP (pretrained on image-text pairs) to define normal and anomalous states as text, then score each image patch by how close it is to those textual concepts.

| | PatchCore | WinCLIP |
|---|---|---|
| Anomaly criterion | Distance to stored normal patches | Distance to language-defined normal/anomalous concepts |
| Training required | Dozens–hundreds of normal images per category | Zero-shot possible |
| New category | Retrain from scratch | Change text prompt only |
| Backbone | ImageNet-pretrained CNN | CLIP (image-text pairs) |

---

## Method

### Framework: CPE + Window-based CLIP → Anomaly Score

#### 1. Compositional Prompt Ensemble (CPE)

Naively querying CLIP with a single prompt like "damaged" yields poor performance. Instead, WinCLIP systematically combines:

- **State-level words** — normal: "flawless", "perfect", "unblemished"; anomalous: "damaged", "with flaw", "imperfect"
- **Template-level prompts** — "a photo of a [c] for visual inspection", "a cropped photo of [c]", etc.

The resulting text embeddings (dozens of prompts) are ensembled to produce a stable normal/anomalous representation.

**Ablation result**: State-level word diversification alone accounts for the largest gain: 74.0% → 89.8% (+15.8%p).

#### 2. Window-based CLIP (WinCLIP)

**Problem**: CLIP aligns text only with the full image. Feeding the whole image answers "is there a defect?" but cannot localize where.

**Solution**: Slide windows of multiple sizes across the image; pass each cropped region through CLIP independently; aggregate scores into a pixel-level anomaly map.

```
Full image  → CLIP → image-level score
2×2 windows → CLIP × 4  → regional scores
3×3 windows → CLIP × 9  → regional scores
         ↓
Multi-scale aggregation (harmonic mean) → pixel anomaly map
```

Overlapping window scores are combined with **harmonic averaging** — if any window strongly predicts normal, the score for that region is pulled down.

**Ablation result**: Patch tokens alone: 22.4% P-AUROC. WinCLIP: 85.1%.

#### 3. WinCLIP+ (Few-Normal-Shot Extension)

When 1–4 normal reference images are available, WinCLIP+ adds visual reference matching on top of language-guided scoring:

- Store multi-scale patch features of the normal reference images
- Compute cosine similarity between test patch features and stored normal features (similar in spirit to PatchCore)
- Final anomaly score = language score + visual similarity score

WinCLIP uses language only; WinCLIP+ uses language + visual reference.

### Experimental Setup

| Parameter | Value |
|---|---|
| Backbone | OpenCLIP ViT-B/16+ (LAION-400M pretrained) |
| Input image size | 240×240 |
| Datasets | MVTec AD (15 categories), VisA |
| Metrics (image) | AUROC, AUPR, F₁-max |
| Metrics (pixel) | pixel-AUROC, PRO, pixel F₁-max |

---

## Results

### MVTec AD

| Method | Setting | I-AUROC | P-AUROC |
|---|---|---|---|
| WinCLIP | Zero-shot | 91.8% | 85.1% |
| WinCLIP+ | 1-shot | 93.1% | 95.2% |
| WinCLIP+ | 4-shot | — | 96.4% |
| PatchCore-10% | Full-shot (reference) | 99.0% | 98.1% |

- 91.8% I-AUROC with zero training data — +17.8%p over naive CLIP (74.0%)
- 1 normal image gets P-AUROC to 95.2%, approaching PatchCore trained on hundreds

### CPE Ablation (Table 3)

| Configuration | I-AUROC |
|---|---|
| One-class (normal only) | 34.2% |
| Two-class (normal + anomalous words) | 74.0% |
| + State ensemble (word diversification) | 89.8% |
| + Prompt ensemble (template diversification) | 90.8% |
| + Multi-crop (windowed) | 91.8% |

---

## Findings

- **Finding 1 — Language supervision reduces the need for per-category training**: CPE text ensembling alone lifts zero-shot AUROC from 74.0% to 90.8%, demonstrating that language-defined defect semantics carry substantial discriminative signal without any visual training data.

- **Finding 2 — Multi-scale windowing bridges the gap between image-level and pixel-level**: Patch-token features without windowing achieve only 22.4% P-AUROC; WinCLIP's sliding window with harmonic aggregation raises this to 85.1%, showing that preserving language alignment at the local scale is the key design choice for localization.

- **Finding 3 — A single normal image is highly informative**: WinCLIP+ with just 1 reference image reaches 95.2% P-AUROC, nearly matching full-shot PatchCore. The visual reference primarily compensates for logical anomalies that text cannot define.

---

## Limitations

- **Logical anomalies** (missing components, misplaced parts): cannot be defined by language alone — "the screw is missing" requires knowing where it should be. Partially addressed by WinCLIP+, not fully solved.
- **Tiny defects**: window resolution limits detection of sub-pixel anomalies.
- **Localization precision**: anomaly regions are found but boundary edges are coarse; pixel F₁-max is below 60% in many categories — authors explicitly state "low-shot anomaly segmentation is still not solved."
- **Acceptable deviations**: distinguishing intentional design features (holes in fabric, designed PCB scratches) from true defects remains challenging without richer domain context.

---

## Connection to PatchCore H2

PatchCore H2 hypothesis (`baseline_analysis.md`):
> "WideResNet50 features are shape-biased from ImageNet pretraining, making them insensitive to color-channel anomalies (e.g., pill contamination)."

WinCLIP uses CLIP, which is trained on image-text pairs and encodes semantic attributes including color and material state (e.g., "contaminated pill" directly encodes the color deviation as a concept). This makes WinCLIP a direct test of H2.

**Next experiment to validate H2**: Run WinCLIP on MVTec AD and check whether pill I-AUROC exceeds PatchCore's 96.8%. If yes, it confirms that language-based feature encoding resolves the color-bias failure mode.