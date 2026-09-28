# Workstream 5: Pre-Registered Prediction Matrix (Held-Out Invariance Tests)

**Author:** Phillip Nyamwaya
**Purpose:** Pre-registration document for two held-out, out-of-sample invariance tests, committed to git before any new embeddings are extracted or any result is inspected. This document exists specifically to satisfy the HARKing-prevention requirement raised in Prof. Ivor Simpson's 25 September 2026 feedback: the link between each model's training-time inductive bias and its expected perturbation invariance must be stated *before* the corresponding evidence is collected, not fitted to the result afterward.

**Scope note:** Two narrowly targeted, independent tests, not a full new perturbation battery: horizontal/vertical flip (H5) and solarization (H6). Both are entirely absent from the study's existing 18-perturbation × 3-severity battery, both are derived from the same DINOv2-vs-CLIP training-recipe argument, and both are applied at full scale to the study's existing tile set (no new sampling). Running two independent out-of-sample predictions from one theoretical account, rather than one, gives a reader two chances to see the account fail, which is stronger evidence than either test alone.

---

## 1. What is being tested and why it is genuinely held out

The dissertation's perturbation taxonomy (`perturbation_pipeline.py`, `PERTURBATION_REGISTRY`, perturbations 1–18, documented in Appendix A) covers five categories: colour/staining (1–5), imaging artefacts (6–10), geometric transformation (11–13: rotation, scaling, elastic deformation), resolution/quality (14–16), and lighting (17–18). Verified directly against the registry: **no flip transformation of any kind (horizontal or vertical) appears anywhere in this battery.** Rotation (Perturbation 11) covers 90°/180°/270° plus a ±15° fine rotation at the severe level, but a flip is not a rotation and is not implied by one — a flip is a reflection, which no combination of the registered rotation angles can reproduce.

This makes horizontal flip and vertical flip the cleanest available out-of-sample test: they were never generated, never embedded, and never seen by any statistical procedure in this study, so a result computed from them cannot have been shaped, consciously or not, by prior exposure to how these models behave under them.

## 2. The predictions, stated in advance

**H5 (flip, within-model paired test):** Models pretrained under the DINOv2 self-supervised recipe (UNI, Prov-GigaPath tile encoder) will show significantly higher embedding invariance (cosine similarity to the unperturbed baseline) under horizontal flip than under vertical flip. Models pretrained under the CLIP/CoCa contrastive recipe (CONCH, Quilt-LLaVA vision tower) will show no significant difference between the two.

This is a directional, two-part prediction with a built-in negative control: it predicts a specific *asymmetry* (hflip ≠ vflip) for one model family and predicts its *absence* for the other. A result showing no differential anywhere, or showing the differential in the CLIP-family models instead, would disconfirm it — it is not written to be unfalsifiable.

**H6 (solarization, between-family test):** Pooling tile-level cosine similarity to baseline under solarization, the DINOv2-family models (UNI, Prov-GigaPath) will show significantly higher solarization invariance than the CLIP-family models (CONCH, Quilt-LLaVA).

Solarization has no natural within-model sibling condition to pair against (unlike hflip/vflip), so H6 is necessarily a between-group comparison rather than a matched-pairs one; Section 4 explains why this changes the test choice, not just the perturbation. Because DINOv2 applies solarization at a much lower rate and to only one of its two global crops (`RandomSolarize(threshold=128, p=0.2)`, applied only to `global_transfo2_extra` — see Section 2.1a), the trained pressure toward solarization-invariance is weaker than the pressure toward horizontal-flip-invariance (`p=0.5`, applied to both global and local crops). H6's predicted effect size is therefore expected to be smaller than H5's, and this is stated here, before either result is seen, specifically so a smaller or more marginal H6 effect cannot later be read as a failure of the account.

### 2.1 Basis for the DINOv2-family prediction (UNI, Prov-GigaPath)

UNI (Chen et al., 2024, *Nature Medicine*) and the Prov-GigaPath tile encoder (Xu et al., 2024, *Nature*) are both trained under the DINOv2 self-supervised recipe (Oquab et al., 2024, *TMLR*, "DINOv2: Learning Robust Visual Features without Supervision"). The reference DINOv2 training implementation confirms the exact augmentation configuration directly from source (`facebookresearch/dinov2/blob/main/dinov2/data/augmentations.py`, retrieved 28 September 2026):

- `geometric_augmentation_global` and `geometric_augmentation_local` both apply `RandomResizedCrop` followed by `RandomHorizontalFlip(p=0.5)`. No vertical-flip transform appears anywhere in the file.
- `global_transfo2_extra` applies `GaussianBlur(p=0.1)` followed by `RandomSolarize(threshold=128, p=0.2)`, to the second global crop only — not the local crops, and not the first global crop.

A model is not guaranteed to be invariant to a transform simply because it was never trained on it, but it has no *positive, trained pressure* toward invariance to a transform excluded from its own augmentation distribution, whereas it has explicit trained pressure toward invariance to a transform that was included (the augmentation objective directly rewards the model for producing similar representations of a view and its transformed counterpart). This is the basis for both H5 (horizontal flip, included at p=0.5 on both crop types → predicted invariance advantage) and H6 (solarization, included at p=0.2 on one crop type only → predicted invariance advantage, but a smaller one than H5's).

**(a) Solarization threshold and probability, cited exactly, not assumed:** the 128 threshold and 0.2 probability used by `extract_flip_holdout.py`'s `solarize` condition are copied directly from this source file, not chosen independently. Using a different threshold would not be testing the transform DINOv2 was actually trained on.

### 2.2 Basis for the CLIP-family prediction (CONCH, Quilt-LLaVA)

CONCH's image encoder (Lu et al., 2024, *Nature Medicine*) is trained under a CoCa-style contrastive-and-captioning objective; Quilt-LLaVA's vision tower (Seyfioglu et al., 2024, CVPR) is a CLIP ViT-B/32 encoder (`wisdomik/QuiltNet-B-32`). Radford et al. (2021, the CLIP paper) state explicitly that "a random square crop from resized images is the only data augmentation used during training." Neither flip nor solarization is part of this recipe. With no trained pressure toward invariance to any of the three held-out conditions, there is no basis to predict an advantage for these two models on either H5 or H6; the prediction is therefore a genuine null (no significant difference / no advantage), not merely "no prediction made."

## 3. What would confirm, and what would disconfirm, these predictions

| Outcome | Interpretation |
|---|---|
| H5: UNI and Prov-GigaPath both show significantly higher hflip than vflip invariance; CONCH and Quilt-LLaVA show no significant difference | H5 confirmed; direct evidence linking a documented training-recipe detail to a measured invariance asymmetry |
| H6: pooled DINOv2-family solarization invariance significantly exceeds pooled CLIP-family solarization invariance | H6 confirmed; second, independent piece of evidence for the same account |
| One DINOv2-family model confirms H5, the other does not | Partial confirmation; report both results and discuss what differs between UNI and Prov-GigaPath beyond the shared DINOv2 recipe (parameter count, training data, architecture) as a candidate explanation |
| A CLIP-family model shows a significant hflip/vflip differential (H5) or the CLIP-family pool exceeds the DINOv2-family pool on solarization (H6) | Disconfirms the corresponding null prediction; must be reported and discussed honestly, not explained away post hoc |
| No model shows any H5 differential, or H6 shows no between-family difference | Disconfirms that prediction; still a reportable, informative result about the limits of augmentation-recipe-based prediction |
| H5 confirms but H6 does not, or vice versa | Reportable as partial support for the inductive-bias account; discussed against the pre-registered expectation (Section 2, item 2 above) that H6's effect would likely be smaller and therefore harder to detect at the same sample size, given solarization's lower training probability and single-crop application |

## 4. Method (summary; full implementation in `extract_flip_holdout.py`)

- **Sample:** the full existing ACC + GDC baseline tile set already used for the study's 18-perturbation battery (no new sampling procedure, so no new exposure to the shared-RNG tile-selection issue already documented for this pipeline).
- **Held-out condition generation:** horizontal flip, vertical flip (`PIL.Image.transpose`), and solarization (`PIL.ImageOps.solarize`, threshold=128) applied in memory to the existing baseline PNGs, at the point of embedding extraction. No new tiles are written to `perturbation_pipeline.py`'s output tree, and no dependency on that script's `cv2`/`elasticdeform` stack.
- **Embedding extraction:** the four models' existing, already-validated adapters (`extract_embeddings_similarity.py`, `ADAPTER_REGISTRY`), run once per model per held-out condition, on Artemis. The four models can run as four simultaneous, independent Slurm jobs.
- **Reference:** each model's already-cached, already-verified baseline embedding (`embeddings_{model}.h5`, key `baseline__<sample_id>`) is reused directly; it is not recomputed.
- **H5 test:** per-tile cosine similarity to baseline (paired by `sample_id`), compared between the hflip and vflip conditions per model via a two-sided Wilcoxon signed-rank test (matched pairs, non-parametric). Chosen because hflip and vflip are two conditions applied to the *same tile within the same model*, so pairing removes each tile's own baseline embeddability as a nuisance factor.
- **H6 test:** per-tile solarize-vs-baseline cosine similarity, pooled within each training family (DINOv2: UNI + Prov-GigaPath; CLIP: CONCH + Quilt-LLaVA), compared between the two pooled groups by directly reusing `phase_b_statistical_analysis.py`'s own `two_sample_test()` — Shapiro-Wilk normality check, Welch's t-test if both groups pass it, Mann-Whitney U otherwise. Chosen because solarization has no within-model sibling condition to pair against, so the comparison is necessarily between groups rather than within one; reusing the study's own validated decision procedure keeps the test-selection logic identical to H1–H4 rather than introducing a second, inconsistent rule.
- Linear CKA (pooled, ACC-only, GDC-only) is also computed for every condition as a population-level cross-check, following the exact procedure already established in `run_similarity()`.
- **Governance:** the entire experiment runs on Artemis against tiles already resident on Artemis (both ACC and GDC). No data of any kind leaves University of Sussex infrastructure; the GDC-only Route B constraint documented in `artemis_preserve_gdc.sh` does not apply here.

---

*This document must be committed to git, with its commit timestamp preserved, before `extract_flip_holdout.py` is run against real embeddings. The commit hash and timestamp should be cited directly in both the revised dissertation (v13) and the white paper as the anchor establishing that these predictions preceded the evidence.*
