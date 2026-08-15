# Comparative Robustness and Demographic Fairness Evaluation of Vision-Language Foundation Models in Computational Pathology

MRes Advanced Artificial Intelligence dissertation project, University of Sussex.

**Author:** Phillip Nyamwaya
**Supervisor:** Dr. Peter Wijeratne
**Ethics reference:** SEM F-REC: SET 1915

## Project summary

This project evaluates the invariance of four pathology foundation models (UNI, CONCH,
Quilt-LLaVA, and the Prov-GigaPath tile encoder) under an 18-perturbation robustness
framework, applied to a 201-slide African cohort (Aroha Cancer Centre, Kenya) and a
190-slide Western control cohort (TCGA/HCMI, via the NCI Genomic Data Commons). The
central questions concern embedding stability (cosine similarity and linear CKA) and
whether robustness differs systematically between cohorts, with implications for
fairness and demographic generalisability of pathology foundation models.

This repository contains the Python pipeline used to acquire and match data, generate
and apply perturbations, extract embeddings, run statistical analysis, and produce the
dissertation's figures. It does **not** contain patient data, embeddings, or any
Aroha Cancer Centre material; those are covered separately by the project's NDA, MTA,
and DUA and are not distributed here.

## Repository structure and execution order

Scripts are grouped by pipeline phase and numbered in the order they were run. Within
each phase, the number prefix on the filename is the run order; run lower numbers
first.

```
scripts/
  01_data_acquisition/            Phase 0 -- cohort matching and slide acquisition
  02_perturbation_and_extraction/ Phase A -- perturbation generation and embedding extraction
  03_statistical_analysis/        Phase B -- hypothesis testing (H1-H4)
  04_reporting/                   Phase C -- dissertation figures
```

### Phase 0 -- Data acquisition, matching, and preparation

| Order | Script | Purpose |
|---|---|---|
| 1 | `01_gdc_match_manifest.py` | Phase 1 of the two-phase download pipeline. Queries the GDC API to build a manifest of TCGA/HCMI control cases demographically matched to the ACC cohort, resolving programme-specific Google Cloud Storage buckets (`gdc-tcga-phs000178-open`, `gdc-hcmi-open`) for later download. |
| 2 | `02_slide_download_generator_v5.py` | Downloads matched non-African control diagnostic WSIs from GDC/TCGA, with TCIA fallback (slide/histology modality only) and AWS Open Data as a last resort. v5 adds resumable downloads that are safe to interrupt and restart. |
| 3 | `03_gcs_slide_downloader.py` | Phase 2 of the two-phase download pipeline. Downloads the GDC/TCGA-HCMI slides identified in the manifest directly from the resolved Google Cloud Storage buckets over HTTPS. |
| 4 | `04_gdrive_wsi_downloader.py` | Downloads the Aroha Cancer Centre whole-slide images (SVS, NDPI, MRXS, TIFF) from a shared Google Drive folder, with a CSV manifest updated after every download attempt so progress survives interruptions. |
| 5 | `05_wsi_audit.py` | Audits native scan resolution and magnification metadata across both cohorts prior to tile extraction, using one shared OpenSlide-based inspection routine with cohort-specific path resolvers. |
| 6 | `06_downsample_wsi.py` | Reads each WSI at the pyramid level nearest to 20x / 0.5 microns-per-pixel (the pretraining resolution for UNI, CONCH, and Prov-GigaPath), resizes where no native level is close enough, and writes a standardised tiled TIFF per slide across both cohorts. |

### Phase A -- Perturbation generation and embedding extraction

| Order | Script | Purpose |
|---|---|---|
| 7 | `07_perturbation_pipeline.py` | Implements all 18 clinically motivated perturbations defined in Section 5.4 of the dissertation proposal, at three severities, applied to every tile in both cohorts. |
| 8 | `08_preflight_check.py` | Pre-flight validation for embedding extraction, run once per model conda environment immediately before submitting a Slurm GPU job. Checks the active conda environment, model-specific package availability, Hugging Face gated-access tokens (UNI, CONCH), GPU visibility, perturbation tile corpus completeness, and the target embeddings directory, entirely without touching the GPU or the network. |
| 9 | `09_extract_embeddings_similarity.py` | Phase A: extracts embeddings for every baseline and perturbed tile across all study models, with resilient HDF5-backed caching that supports interrupted/resumed runs. Phase B (of this script): computes cosine similarity per tile and linear CKA per perturbation/severity/model from the cached embeddings, entirely on CPU. |

### Phase B -- Statistical analysis (H1-H4)

Validation against synthetic data is run before, not after, the real analysis, following
this project's established practice of preflight validation against synthetic data
before any production statistical run.

| Order | Script | Purpose |
|---|---|---|
| 10 | `10_make_synthetic_data.py` | Generates small synthetic `cosine_similarity.csv` / `cka_summary.csv` files matching the exact schema produced by `extract_embeddings_similarity.py`'s `run_similarity()`, for smoke-testing the statistical analysis script without needing real embeddings. |
| 11 | `11_validate_phase_b_statistical_analysis.py` | Self-contained validation script reproducing every check performed while diagnosing and fixing a zero-variance edge case in `two_sample_test()` (perturbations 3, 17, and 18 at moderate severity are documented identity transforms, yielding a constant cosine similarity of 1.0 in both cohorts, which is mathematically undefined for Shapiro-Wilk / Mann-Whitney U). Confirms the fix on synthetic data before it is trusted on real data. |
| 12 | `12_check_phase_b_dependencies.py` | Verifies the statistical analysis environment (numpy, pandas, scipy, statsmodels, optional scikit-posthocs) has every required package at a known, unmodified version before a long batch run is submitted, using a pip constraints file so no already-installed package is silently upgraded or downgraded. |
| 13 | `13_phase_b_statistical_analysis.py` | Runs the H1-H4 hypothesis tests (model main effect, demographic main effect, model x demographic interaction, model x perturbation interaction) over the real cosine-similarity and linear-CKA metrics, per Section 5.6 of the dissertation proposal (v4, 23 Jul 2026), for both candidate model-cohort configurations (Option A and Option B; see "Model cohort configurations" below). |

### Phase C -- Reporting

| Order | Script | Purpose |
|---|---|---|
| 14 | `14_make_figures.py` | Generates all Phase B results figures used in the dissertation results chapter, for both Model Cohort Option A (UNI, CONCH, Quilt-LLaVA) and Option B (UNI, CONCH, Prov-GigaPath tile encoder), from the H1-H4 JSON summaries and H2/pairwise/outlier CSV tables produced by script 13. |

## Model cohort configurations (Option A vs Option B)

Two candidate primary three-model cohorts are under active consideration, both
preserving the same 3-pairwise x 18-perturbation Bonferroni-corrected structure
(alpha = 0.05/54):

- **Option A** (proposal-aligned): UNI, CONCH, Quilt-LLaVA.
- **Option B**: UNI, CONCH, Prov-GigaPath (tile encoder only), motivated by the shared
  224x224 native input resolution and tile-level inference paradigm across all three
  models. Under Option B, Quilt-LLaVA is repositioned as a documented
  architectural-outlier supplementary comparator (its vision tower is stock
  `openai/clip-vit-large-patch14-336`, which upsamples every 224x224 study tile to
  336x336 before inference).

`13_phase_b_statistical_analysis.py` and `14_make_figures.py` both run either
configuration from a single `MODEL_SETS` dictionary, so the choice between Option A and
Option B does not require code changes. This decision is pending discussion with the
supervisor and is documented here for transparency; it does not block analysis under
either configuration.

## Requirements

See `requirements.txt`. The extraction stage (script 9) additionally requires
model-specific packages and gated Hugging Face access, isolated into separate conda
environments per model (`uni-env`, `conch-env`, `quilt-llava-env`); see
`docs/environment_notes.md`.

## Data and compute environment

- Embedding extraction was run on the University of Sussex Artemis HPC cluster
  (Slurm, NVIDIA A40 GPUs).
- Statistical analysis (Phase B) is CPU-only and does not require HPC access.
- Whole-slide image data (both the ACC-Kenya cohort and the GDC/TCGA-HCMI control
  cohort) is not included in this repository. ACC data is subject to an NDA, MTA, and
  DUA with the Aroha Cancer Centre; GDC/TCGA-HCMI data is open-access but is not
  redistributed here. File paths referenced in these scripts (for example,
  `/its/home/pn254/project2026/...`) are specific to the author's Artemis account and
  will need to be changed for any other environment.
- No API keys, tokens, or credentials are stored in this repository. All scripts read
  credentials from environment variables (for example, `HUGGING_FACE_HUB_TOKEN`,
  `HF_TOKEN`, `GDRIVE_API_KEY`, `TCIA_API_KEY`).

## License

Not yet assigned. Add a `LICENSE` file appropriate to your institution's policy and
the terms of the ACC data-sharing agreements before making this repository public.

## Citation

If this pipeline is useful to your own work, please cite the dissertation once it is
submitted. Citation details will be added here on completion.
