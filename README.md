# Comparative Robustness and Demographic Fairness Evaluation of Vision-Language Foundation Models in Computational Pathology

MRes Advanced Artificial Intelligence dissertation project, University of Sussex.

**Author:** Phillip Nyamwaya
**Supervisor:** Dr. Peter Wijeratne
**Ethics reference:** SEM F-REC: SET 1915
**Repository:** https://github.com/CenturyChallenger/vlm-pathology-invariance-study

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
| 1 | `01_gdc_match_manifest_v4.py` | Phase 1 of the two-phase download pipeline. Queries the GDC API to build a manifest of TCGA/HCMI control cases demographically matched to the ACC cohort. v4 resolves each file's GCS bucket via the NCI CRDC DRS API (programme-agnostic, correct for TCGA and HCMI alike in principle), falling back to the TCGA open bucket if DRS is unreachable, and adds cross-case file_id deduplication (each control file can be assigned to at most one ACC case; a case whose entire 50-candidate pool is already used triggers an automatic retry against a 200-candidate pool). See "Closed" note below for the one real DRS failure this path hit in production and its accepted explanation. |
| 2 | `02_test_matched_controls_patched.py` | Ad hoc verification script (not a formal pytest suite) run against `matched_controls_manifest_patched.tsv` after that manifest has been corrected: confirms zero duplicate `file_id` values among MATCHED rows, and confirms every HCM-prefixed file's `gcs_url` points at the `gdc-hcmi-open` bucket rather than the TCGA bucket. See "Closed" note below. |
| 3 | `03_slide_download_generator_v5.py` | Downloads matched non-African control diagnostic WSIs from GDC/TCGA, with TCIA fallback (slide/histology modality only) and AWS Open Data as a last resort. v5 adds resumable downloads that are safe to interrupt and restart. |
| 4 | `04_gcs_slide_downloader_v2.py` | Phase 2 of the two-phase download pipeline. Downloads the GDC/TCGA-HCMI slides identified in the (patched) manifest directly from the resolved Google Cloud Storage buckets over HTTPS. v2 hardens the MATCHED-row filter: the `status` column is normalised (stripped, upper-cased) before filtering, and rows are additionally required to have a non-null `gcs_url` starting with `https://`, guarding against blank/NaN URLs on rows that pandas would otherwise silently pass through. |
| 5 | `05_gdrive_wsi_downloader.py` | Downloads the Aroha Cancer Centre whole-slide images (SVS, NDPI, MRXS, TIFF) from a shared Google Drive folder, with a CSV manifest updated after every download attempt so progress survives interruptions. |
| 6 | `06_wsi_audit.py` | Audits native scan resolution and magnification metadata across both cohorts prior to tile extraction, using one shared OpenSlide-based inspection routine with cohort-specific path resolvers. |
| 7 | `07_downsample_wsi.py` | Reads each WSI at the pyramid level nearest to 20x / 0.5 microns-per-pixel (the pretraining resolution for UNI, CONCH, and Prov-GigaPath), resizes where no native level is close enough, and writes a standardised tiled TIFF per slide across both cohorts. |

**Closed:** direct comparison of the real, unpatched `matched_controls_manifest.tsv`
against `matched_controls_manifest_patched.tsv` confirms that DRS resolution
(`resolve_gcs_url()` in script 1) genuinely failed for all 15 HCMI-programme files in a
real run -- the unpatched manifest shows the incorrect `gdc-tcga-phs000178-open` bucket
for all 15, exactly matching script 1's coded fallback path, while the patched manifest
shows all 15 correctly resolved to `gdc-hcmi-open`. The accepted explanation for that
failure is the NCI CRDC/Imaging Data Commons User Guide's documented behaviour that a
GUID not yet registered with the DRS resolution service legitimately returns HTTP 404
(National Cancer Institute, Imaging Data Commons, n.d., "Resolving CRDC Globally Unique
Identifiers (GUIDs)"), rather than network-level restriction, which was never
independently confirmed. The mechanism by which the manifest was corrected into its
patched state is not evidenced in this project and is not reconstructed here. A
programme-aware bucket-table resolver (`TCGA-*` -> `gdc-tcga-phs000178-open`, `HCM-*`
-> `gdc-hcmi-open`, bypassing DRS entirely) remains available as a simpler fix
regardless of cause; script 1 as currently checked in does not implement this bypass.

### Phase A -- Perturbation generation and embedding extraction

| Order | Script | Purpose |
|---|---|---|
| 8 | `08_perturbation_pipeline.py` | Implements all 18 clinically motivated perturbations defined in Section 5.4 of the dissertation proposal, at three severities, applied to every tile in both cohorts. |
| 9 | `09_preflight_check.py` | Pre-flight validation for embedding extraction, run once per model conda environment immediately before submitting a Slurm GPU job. Checks the active conda environment, model-specific package availability, Hugging Face gated-access tokens (UNI, CONCH), GPU visibility, perturbation tile corpus completeness, and the target embeddings directory, entirely without touching the GPU or the network. |
| 10 | `10_extract_embeddings_similarity.py` | Phase A: extracts embeddings for every baseline and perturbed tile across all study models, with resilient HDF5-backed caching that supports interrupted/resumed runs. Phase B (of this script): computes cosine similarity per tile and linear CKA per perturbation/severity/model from the cached embeddings, entirely on CPU. |

### Phase B -- Statistical analysis (H1-H4)

Validation against synthetic data is run before, not after, the real analysis, following
this project's established practice of preflight validation against synthetic data
before any production statistical run.

| Order | Script | Purpose |
|---|---|---|
| 11 | `11_make_synthetic_data.py` | Generates small synthetic `cosine_similarity.csv` / `cka_summary.csv` files matching the exact schema produced by `extract_embeddings_similarity.py`'s `run_similarity()`, for smoke-testing the statistical analysis script without needing real embeddings. |
| 12 | `12_validate_phase_b_statistical_analysis.py` | Self-contained validation script reproducing every check performed while diagnosing and fixing a zero-variance edge case in `two_sample_test()` (perturbations 3, 17, and 18 at moderate severity are documented identity transforms, yielding a constant cosine similarity of 1.0 in both cohorts, which is mathematically undefined for Shapiro-Wilk / Mann-Whitney U). Confirms the fix on synthetic data before it is trusted on real data. |
| 13 | `13_check_phase_b_dependencies.py` | Verifies the statistical analysis environment (numpy, pandas, scipy, statsmodels, optional scikit-posthocs) has every required package at a known, unmodified version before a long batch run is submitted, using a pip constraints file so no already-installed package is silently upgraded or downgraded. |
| 14 | `14_phase_b_statistical_analysis.py` | Runs the H1-H4 hypothesis tests (model main effect, demographic main effect, model x demographic interaction, model x perturbation interaction) over the real cosine-similarity and linear-CKA metrics, per Section 5.6 of the dissertation proposal (v4, 23 Jul 2026), for both the confirmed authoritative configuration (Option B) and the alternative configuration retained for the dissertation's sensitivity analysis (Option A; see "Model cohort configurations" below). |

### Phase C -- Reporting

| Order | Script | Purpose |
|---|---|---|
| 15 | `15_make_figures.py` | Generates all 13 Phase B results figures used in the dissertation results chapter, for both the confirmed authoritative Model Cohort Option B (UNI, CONCH, Prov-GigaPath tile encoder) and Option A (UNI, CONCH, Quilt-LLaVA, retained for the sensitivity analysis), from the H1-H4 JSON summaries and H2/pairwise CSV tables produced by script 14. Each figure is written as both PNG and SVG; see `docs/figures.md` for the full figure list, the accessibility rationale behind the colour choices, and how the SVG output is embedded into the dissertation `.docx` files. |

`scripts/04_reporting/data/` bundles the complete real Phase B statistical
output set (23 files: H1-H4, pairwise, and Option B's outlier-supplementary
results in both cosine-similarity and linear-CKA variants, plus a per-model
H4 breakdown) actually used to produce the figures embedded in both
dissertation drafts. These are aggregate statistical summaries only, not
patient data, so the script runs correctly out of the box. Note that
`15_make_figures.py` reads only 10 of these 23 files; see `docs/figures.md`
for exactly which, and for the honestly-documented gap around
`option_b_h4_per_model.csv` (source data for two dissertation figures --
5.2b, 5.5b -- that were produced ad hoc and are not backed by a script
currently held in this project):

```bash
cd scripts/04_reporting
python3 15_make_figures.py                    # both options, reads ./data, writes ./figures
python3 15_make_figures.py --options b         # Option B only
python3 15_make_figures.py --data-dir /path/to/new/phase_b/output --out-dir /path/to/figures
```

## Model cohort configurations (Option A vs Option B)

**Option B was confirmed by Dr. Wijeratne as the authoritative dissertation
configuration.** Both configurations preserve the same 3-pairwise x 18-perturbation
Bonferroni-corrected structure (alpha = 0.05/54):

- **Option B (confirmed, authoritative):** UNI, CONCH, Prov-GigaPath (tile encoder
  only), motivated by the shared 224x224 native input resolution and tile-level
  inference paradigm across all three models. Quilt-LLaVA is repositioned as a
  documented architectural-outlier supplementary comparator (its vision tower is stock
  `openai/clip-vit-large-patch14-336`, which upsamples every 224x224 study tile to
  336x336 before inference).
- **Option A (retained as a record of the alternative configuration considered, not
  pursued further):** UNI, CONCH, Quilt-LLaVA, the original proposal-aligned cohort.

`14_phase_b_statistical_analysis.py` and `15_make_figures.py` both run either
configuration from a single `MODEL_SETS` dictionary. This is why both configurations'
output are still present throughout this repository (e.g. `option_a_*`/`option_b_*`
files in `scripts/04_reporting/data/`) even though only Option B is authoritative --
Option A's results remain available for the dissertation's own sensitivity-analysis
section, which shows the substantive conclusions are robust to this choice.

## Requirements

See `requirements.txt`. The extraction stage (script 10) additionally requires
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
- `matched_controls_manifest*.tsv` and `acc_manifest*.tsv` (the Phase 1 matching
  manifests consumed by scripts 1-4) are also excluded (`.gitignore`): they link
  pseudonymised ACC case IDs to demographic fields (sex, age, diagnosis class), which
  falls under the same ACC data-sharing agreements as the imagery itself, even though
  the manifest contains no image data.
- No API keys, tokens, or credentials are stored in this repository. All scripts read
  credentials from environment variables (for example, `HUGGING_FACE_HUB_TOKEN`,
  `HF_TOKEN`, `GDRIVE_API_KEY`, `TCIA_API_KEY`).

## License

Not yet assigned. Add a `LICENSE` file appropriate to your institution's policy and
the terms of the ACC data-sharing agreements. The repository is already live at the
URL above; if it should be private rather than public, check its visibility setting
on GitHub directly, since this README cannot control that.

## Citation

If this pipeline is useful to your own work, please cite the dissertation once it is
submitted. Citation details will be added here on completion.
