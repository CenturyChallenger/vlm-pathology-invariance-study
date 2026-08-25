# Figures pipeline (script 14: `make_figures.py`)

`scripts/04_reporting/14_make_figures.py` generates the 13 Phase B results
figures used in the dissertation, for both Model Cohort Option A and Option
B. Each figure is written as both `.png` and `.svg` (26 output files total).

## Usage

The script resolves its `data/` and `figures/` directories relative to its
own location by default, so it will look for input files in a `data/`
subdirectory next to the script unless told otherwise:

```bash
# Generate all figures (both options), reading from ./data next to the script
python3 14_make_figures.py

# Generate only one cohort configuration
python3 14_make_figures.py --options a
python3 14_make_figures.py --options b

# Point at a specific Phase B output directory and output location
python3 14_make_figures.py --data-dir /path/to/phase_b_output --out-dir /path/to/figures
```

The script fails fast with a clear error if `--data-dir` does not exist or
is missing an expected file, rather than silently producing an empty or
incorrect figure.

## Input files

Of the full Phase B output set (now 23 files, following the addition of
`option_b_h4_per_model.csv` -- see "Per-model H4 breakdown" below), only these
10 (5 per option) are read:

- `option_{a,b}_h1_model_main_effect.json` and `_cka.json`
- `option_{a,b}_h2_demographic_main_effect.csv`
- `option_{a,b}_h4_model_x_perturbation_interaction.json`
- `option_{a,b}_pairwise_comparisons_bonferroni.csv`

The remaining files (H2/H4/pairwise CKA variants, H3, Option B's
outlier-supplementary CSVs, and the per-model H4 breakdown below) are cited
directly in the dissertation's tables and prose rather than plotted by this
script.

## Per-model H4 significance testing (`14b_compute_h4_per_model.py`)

`scripts/03_statistical_analysis/14b_compute_h4_per_model.py` fits a real
significance test for each model separately, producing
`option_{a,b}_h4_per_model.csv` (51 rows for Option B: 3 models x 17
non-reference perturbations, each row a per-model coefficient with a real
p-value and Bonferroni-corrected significance flag).

It is built directly from `test_h4()` in `14_phase_b_statistical_analysis.py`
(verified against that function's real source before this script was added
to the repo, not assumed): the pooled H4 test fits one mixed-effects model
across all models together (`cosine_similarity ~ C(perturbation_id)`, with
`model` as a random-intercept group), which cannot say whether one model
specifically is more or less sensitive to a given perturbation than
another -- only the pooled, averaged effect. Because each per-model fit
here only sees one model's own tile-level data, the random-intercept term
is no longer needed and the fit collapses to a plain OLS regression on the
same formula minus the grouping -- simpler and faster than the pooled
model, with none of a mixed model's convergence risk.

```bash
python3 14b_compute_h4_per_model.py \
    --cosine-csv /path/to/cosine_similarity.csv \
    --option b \
    --out-dir /path/to/output
```

Input is the raw `cosine_similarity.csv` produced by
`10_extract_embeddings_similarity.py --mode similarity` (the same file the
pooled H4 test in script 14 already reads) -- not the summary CSVs bundled
in `data/`, since a significance test needs the individual tile-level
measurements, not already-averaged numbers. `--alpha-denominator` defaults
to 54, matching the Bonferroni threshold used everywhere else in the
dissertation; the script prints the exact number of tests it actually ran
(51) in case the stricter, exactly-correct 0.05/51 threshold is preferred
instead.

Validated against synthetic data before being added to this repo, not just
taken on trust: run against a synthetic `cosine_similarity.csv` with a
deliberately injected large negative effect on one perturbation, the script
correctly identified that perturbation as the largest and most significant
coefficient for every model; run against a copy missing a required column,
it failed immediately with a clear error rather than partway through a fit.

**CLOSED (25 Aug 2026):** `15_make_figures.py` now reads
`option_{a,b}_h4_per_model.csv` via `fig_h4_perturbation_ranking_per_model()`.
Matching the dissertation's actual rendering (verified directly against the
real dissertation text before implementing, not assumed), this produces
**three separate per-model figures** rather than one combined panel:
`option_b_fig_5_5b_1_uni`, `_2_conch`, `_3_gigapath_tile`, each with real
significance asterisks, sharing the same perturbation row order (matching
Figure 5.5's pooled ranking) and the same x-axis scale across all three, so
the same perturbation can be compared directly across models. If the
per-model CSV is absent for a given option (currently true for Option A, for
which this analysis has not been run), Figure 5.5b is skipped for that
option with a printed note, rather than failing or fabricating a substitute
from different data.

**CLOSED (25 Aug 2026):** `15_make_figures.py` now generates Figure 5.2b via
`fig_cosine_difference_heatmap()`. No source script for this figure was
found anywhere in the project (project files directory checked directly,
plus multiple targeted project-knowledge searches; only an older snapshot
of `make_figures.py` predating both Figure 5.2b and 5.5b was retrievable) --
this function was built fresh, using `mean_a - mean_b` from the same H2 CSV
`fig_cosine_heatmap()` and `fig_demographic_gap_heatmap()` already read
(not `option_{a,b}_h4_per_model.csv`, despite the two being informally
grouped together in earlier notes on this repo -- that remains a real
distinction, not merely a naming coincidence). Verified against the
dissertation's own cited numbers before being considered correct, not
assumed: UNI's raw gaps on Resolution Degradation (+0.215), Downsample-
Upsample (+0.165), and JPEG Compression (+0.134), and CONCH's largest
negative gap on Salt-and-Pepper Noise (-0.144), all reproduced exactly from
the real bundled `option_b_h2_demographic_main_effect.csv`. Same PuOr
colourblind-safe scheme and sign convention as Figure 5.4 (positive/purple
= ACC more stable), auto-scaled symmetric range (unlike Figure 5.4's fixed
+/-2.0, since raw differences are a much smaller magnitude), signed 3-decimal
cell annotations, no significance markers (this figure restates already-
tested means rather than running its own test).

## Output: 13 (+1 Figure 5.2b, +3 per option if the per-model H4 CSV is present) figures, each as PNG and SVG

| Output file (base name) | Dissertation figure | Content |
|---|---|---|
| `shared_fig_5_1_phase_a_throughput` | Figure 5.1 | Phase A embedding extraction throughput by model (values hard-coded from the Phase A extraction log, not read from `data/`) |
| `option_{a,b}_fig_5_2_cosine_heatmap` | Figure 5.2 | Cosine similarity to baseline at moderate severity, by perturbation x model, split by cohort (ACC / GDC) |
| `option_{a,b}_fig_5_2b_cosine_difference_heatmap` | Figure 5.2b | Cosine similarity difference (ACC minus GDC, raw units) by perturbation x model, same sign convention as Figure 5.4 |
| `option_{a,b}_fig_5_3_severity_trends` | Figure 5.3 | Mean cosine similarity vs. severity (mild/moderate/severe), one panel per perturbation category |
| `option_{a,b}_fig_5_4_demographic_gap_heatmap` | Figure 5.4 | Demographic robustness gap (Cohen's d, ACC minus GDC) by perturbation x model, with Bonferroni-significance markers |
| `option_{a,b}_fig_5_5_h4_perturbation_ranking` | Figure 5.5 | H4 mixed-effects perturbation coefficients, ranked, coloured by perturbation category |
| `option_{a,b}_fig_5_5b_{1,2,3}_{model}` | Figure 5.5b(i)/(ii)/(iii) | Per-model H4 perturbation coefficients with real significance testing (script 14b), one figure per model, shared row order and x-axis scale. Only generated if `option_{a,b}_h4_per_model.csv` is present (currently Option B only). |
| `option_{a,b}_fig_5_6_h1_model_main_effect` | Figure 5.6 | H1 model main effect, cosine similarity and linear CKA side by side |
| `option_{a,b}_fig_5_7_pairwise_comparison` | Figure 5.7 | Pairwise model comparison summary (mean Cohen's d per pair) |

## Vector output (SVG)

`matplotlib.pyplot.savefig()` supports SVG natively; every `fig_*` function
calls a `save_both_formats()` helper that writes both `<name>.png` and
`<name>.svg` from the same figure object. SVG preserves the figure as
scalable paths and text rather than a fixed pixel grid, so it can be zoomed
in on, printed at any size, or resized in a downstream document without
pixellation.

This matters for embedding into the dissertation `.docx`: the `docx` npm
library (used by the separate `build_docx.js` conversion script, not
included in this repository) supports embedding SVG directly via
`ImageRun({ type: "svg", ... })`. This was verified end-to-end for this
project, not just assumed from the library's documentation, by building a
real `.docx` with an embedded SVG, converting it to PDF via the same
LibreOffice headless pipeline this project's QA process uses, and inspecting
the resulting PDF's raw content stream directly. That inspection confirmed
LibreOffice renders the embedded SVG as genuine PDF vector path operators
(`m`, `l`, `c`, `re`, `f*`, `S`), not a rasterised `/Image` XObject, so the
vector quality survives through to the final PDF a reader actually opens.

Word's own OOXML SVG support requires a raster fallback image alongside the
SVG, for older Word versions and other tools that cannot render inline SVG.
This is why every `.svg` file here has a same-named `.png` sibling generated
alongside it: both are needed together when embedding, not just the SVG on
its own.

## Accessibility of the colour encoding (Appendix I)

Colour alone is not a reliable encoding channel: roughly 1 in 12 men have
some form of colour vision deficiency (most commonly red-green, i.e.
deuteranopia/protanopia), and colour information is lost entirely under
greyscale photocopying or printing. Every bar and line chart in this figure
set therefore uses two independent, redundant encodings:

1. **Colour**, drawn from the Okabe-Ito palette (Okabe & Ito, 2008; endorsed
   by Wong, B. (2011), "Points of view: Color blindness", *Nature Methods*
   8(6):441, DOI: [10.1038/nmeth.1618](https://doi.org/10.1038/nmeth.1618)),
   designed to remain distinguishable under protanopia, deuteranopia, and
   tritanopia.
2. **Shape**: a unique bar hatch pattern per series, or a unique line style
   + marker shape per series, which survives greyscale conversion
   regardless of colour perception.

Heatmaps use:

- `viridis` (Figure 5.2, cosine similarity): perceptually uniform and
  monotonically increasing in lightness, unlike `RdYlGn`, whose red/green
  endpoints are the classic failure case for deuteranopia/protanopia and
  collapse to similar greys under photocopying.
- `PuOr` (Figure 5.4, demographic gap): a ColorBrewer diverging scheme
  verified colourblind-safe, replacing `RdBu_r`'s red/blue endpoints, which
  are harder to distinguish under tritanopia and whose red end is easily
  confused with the green endpoints used elsewhere in this figure set under
  deuteranopia/protanopia.

Every heatmap cell is also annotated with its signed numeric value (plus a
significance marker where relevant), computed against a luminance-based
text-colour rule (`text_color_for()`, using the standard relative-luminance
formula 0.2126R + 0.7152G + 0.0722B) rather than a hardcoded threshold, so
annotation text stays legible regardless of which colormap is in use.

**Residual limitation**: diverging heatmaps cannot fully guarantee that sign
is recoverable from grey shade alone at moderate effect magnitudes. This is
mitigated by the explicit signed numeric annotation in every cell (see
above), but is noted here as a documented limitation rather than a solved
problem.
