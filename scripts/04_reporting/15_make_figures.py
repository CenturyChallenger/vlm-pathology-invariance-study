"""
make_figures.py
-----------------------------------------------------------------------
Generates all Phase B results figures used in Sections 5.1-5.10 of the
dissertation, for both Model Cohort Option A (UNI, CONCH, Quilt-LLaVA)
and Option B (UNI, CONCH, Prov-GigaPath tile encoder), from the real
`phase_b_statistical_analysis.py` output files (H1-H4 JSON summaries,
H2/pairwise/outlier CSV tables), plus two option-independent Workstream 5
figures (H5, H6) and one effect-size summary figure.

USAGE
-----
    python make_figures.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]

By default, DATA_DIR is "./data" and OUT_DIR is "./figures", both
resolved relative to this script's own location, so the package runs
out of the box from wherever it is unzipped. Pass absolute paths to
point at a different Phase B output directory (for example, a fresh
run's output on the Artemis cluster) without editing the script.

INPUT
-----
DATA_DIR must contain these files per option (produced by
`phase_b_statistical_analysis.py`), the only ones this script reads:
    option_{a,b}_h1_model_main_effect.json
    option_{a,b}_h1_model_main_effect_cka.json
    option_{a,b}_h2_demographic_main_effect.csv
    option_{a,b}_h4_model_x_perturbation_interaction.json
    option_{a,b}_pairwise_comparisons_bonferroni.csv

Note: the CKA variants of H2/H4/pairwise, the H3 files, and the Option B
outlier-supplementary files are part of the full Phase B output set and
are cited directly in the dissertation text/tables, but are not read by
this figure-generation script. The bundled data/ folder includes the
complete 22-file Phase B output set for completeness and provenance,
even though only the 10 files above (5 per option) are actually used
here.

WORKSTREAM 5 (H5, H6) DATA PROVENANCE -- READ BEFORE EDITING
--------------------------------------------------------------
Figures 5.8-5.10 (fig_h5_flip_invariance, fig_h6_solarization_invariance,
fig_workstream5_effect_size_summary) do NOT read a JSON/CSV file the way
every other fig_* function in this script does, because
`flip_holdout_h5_h6_test_results.json` (produced by
`extract_flip_holdout.py --mode test`) has not yet been added to this
project's bundled data/ folder. The summary statistics these three
functions plot are instead hard-coded as module-level constants
(H5_RESULTS, H6_RESULTS below), copied verbatim from the *already
verified and internally cross-checked* results table in Section 5 of
`docs/Nyamwaya_Workstream5_Prediction_Matrix.md` (the post-registration
addendum, added 29 September 2026, itself derived directly from that
JSON file's contents). No number below was estimated, interpolated, or
simulated.

If/when `flip_holdout_h5_h6_test_results.json` is added to DATA_DIR,
these three functions should be refactored to read it directly (the way
fig_h4_perturbation_ranking() reads its JSON), exactly as recommended
for maintainability; H5_RESULTS/H6_RESULTS below should then be deleted
rather than kept as a stale parallel source of truth.

Effect-size relabelling (2 September 2026 verification pass): H5's
per-model effect size is reported here as "2 x Cohen's g", NOT as a
"rank-biserial correlation" as an earlier draft of the prediction matrix
labelled it. The underlying number is unchanged -- it is exactly
(n_pos - n_neg) / (n_pos + n_neg), i.e. twice Cohen's g for a paired
sign comparison (Cohen, 1988, p.147ff; g = P - 0.5 where P is the
proportion of favourable pairs) -- but this is algebraically distinct
from the Wilcoxon-signed-rank rank-biserial correlation of Kerby (2014),
which is computed from the SUM of the signed ranks, not from the COUNT
of positive vs. negative pairs. See the relabelling note in
`docs/Nyamwaya_Workstream5_Prediction_Matrix.md` Section 5.1 for the full
derivation and citation trail.

OUTPUT
------
16 PNG files (+ matching SVGs) written to OUT_DIR, matching the
#FIGURE: references in dissertation.md / dissertation_optionB.md:
    shared_fig_5_1_phase_a_throughput.png
    option_{a,b}_fig_5_2_cosine_heatmap.png
    option_{a,b}_fig_5_3_severity_trends.png
    option_{a,b}_fig_5_4_demographic_gap_heatmap.png
    option_{a,b}_fig_5_5_h4_perturbation_ranking.png
    option_{a,b}_fig_5_6_h1_model_main_effect.png
    option_{a,b}_fig_5_7_pairwise_comparison.png
    shared_fig_5_8_h5_flip_invariance.png
    shared_fig_5_9_h6_solarization_invariance.png
    shared_fig_5_10_workstream5_effect_size_summary.png

DEPENDENCIES
------------
See requirements.txt (numpy, pandas, matplotlib).
-----------------------------------------------------------------------
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# DATA and OUT are resolved in main() from CLI args (or their defaults,
# both relative to this script's location) and then used as module-level
# globals by every fig_* function below via f-string interpolation.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA = str(SCRIPT_DIR / "data")
OUT = str(SCRIPT_DIR / "figures")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------
# Perturbation taxonomy (Appendix A)
# ---------------------------------------------------------------------
PERT_NAMES = {
    1: "H&E Intensity Variation", 2: "Colour Channel Shift", 3: "Saturation Adjustment",
    4: "Hue Rotation", 5: "Stain Norm. Failure",
    6: "Gaussian Blur", 7: "Motion Blur", 8: "Gaussian Noise", 9: "Salt-and-Pepper Noise",
    10: "JPEG Compression",
    11: "Rotation", 12: "Scaling", 13: "Elastic Deformation",
    14: "Downsample-Upsample", 15: "Resolution Degradation", 16: "Sharpness Reduction",
    17: "Brightness Adjustment", 18: "Contrast Adjustment",
}
PERT_CATEGORY = {
    **{i: "1. Colour & Staining" for i in [1, 2, 3, 4, 5]},
    **{i: "2. Imaging Artefacts" for i in [6, 7, 8, 9, 10]},
    **{i: "3. Geometric" for i in [11, 12, 13]},
    **{i: "4. Resolution & Quality" for i in [14, 15, 16]},
    **{i: "5. Lighting" for i in [17, 18]},
}

# ---------------------------------------------------------------------
# Colour and pattern encoding
# ---------------------------------------------------------------------
# Colour alone is not a reliable encoding channel: roughly 1 in 12 men
# have some form of colour vision deficiency (most commonly red-green,
# i.e. deuteranopia/protanopia), and colour information is lost entirely
# under greyscale photocopying or printing. Both bar and line charts
# below therefore use two independent, redundant encodings:
#   (1) colour, drawn from the Okabe-Ito palette (Okabe & Ito, 2008;
#       endorsed by Nature Methods' "Points of view: Color blindness",
#       Wong, 2011), designed to remain distinguishable under
#       protanopia, deuteranopia, and tritanopia; and
#   (2) shape (bar hatch pattern, or line style + marker shape), which
#       survives greyscale conversion regardless of colour perception.
# A reader relying on colour, greyscale value, or pattern alone should
# still be able to distinguish every series in every figure below.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "bluish_green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "reddish_purple": "#CC79A7",
}

CAT_COLORS = {
    "1. Colour & Staining": OKABE_ITO["blue"],
    "2. Imaging Artefacts": OKABE_ITO["vermillion"],
    "3. Geometric": OKABE_ITO["bluish_green"],
    "4. Resolution & Quality": OKABE_ITO["reddish_purple"],
    "5. Lighting": OKABE_ITO["orange"],
}
CAT_HATCHES = {
    "1. Colour & Staining": "",
    "2. Imaging Artefacts": "///",
    "3. Geometric": "xxx",
    "4. Resolution & Quality": "...",
    "5. Lighting": "\\\\",
}
SEVERITY_ORDER = ["mild", "moderate", "severe"]

MODEL_LABELS = {
    "uni": "UNI", "conch": "CONCH", "quilt_llava": "Quilt-LLaVA",
    "gigapath_tile": "Prov-GigaPath",
}
MODEL_COLORS = {
    "uni": OKABE_ITO["blue"], "conch": OKABE_ITO["orange"],
    "quilt_llava": OKABE_ITO["bluish_green"], "gigapath_tile": OKABE_ITO["reddish_purple"],
}
# Bar hatch pattern per model (redundant with colour for greyscale/print).
MODEL_HATCHES = {
    "uni": "", "conch": "///", "quilt_llava": "xxx", "gigapath_tile": "...",
}
# Line style and marker per model (redundant with colour for line charts,
# where hatching is not applicable).
MODEL_LINESTYLES = {
    "uni": "-", "conch": "--", "quilt_llava": "-.", "gigapath_tile": ":",
}
MODEL_MARKERS = {
    "uni": "o", "conch": "s", "quilt_llava": "^", "gigapath_tile": "D",
}
# Fixed colour/hatch triplet for position-based (not model-specific) bar
# charts with exactly three bars, e.g. pairwise comparison summaries.
TRIPLET_COLORS = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["bluish_green"]]
TRIPLET_HATCHES = ["", "///", "xxx"]

# Training-family colour/hatch pair, used only by the Workstream 5
# figures (5.8-5.10), which group by pretraining recipe rather than by
# individual model. Deliberately distinct from CAT_COLORS/MODEL_COLORS
# above so a reader cannot mistake "family" encoding for "perturbation
# category" or "individual model" encoding used elsewhere in the figure
# set.
FAMILY_COLORS = {"dinov2": OKABE_ITO["sky_blue"], "clip": OKABE_ITO["yellow"]}
FAMILY_HATCHES = {"dinov2": "", "clip": "///"}
FAMILY_LABELS = {"dinov2": "DINOv2-family", "clip": "CLIP-family"}

plt.rcParams["hatch.linewidth"] = 1.3


def pert_label(pid):
    return f"{pid:02d}. {PERT_NAMES[pid]}"


def save_both_formats(fig, path_no_ext):
    """
    Save a figure as both PNG (raster, used as the Word-compatibility
    fallback for older/non-Office viewers, and for quick preview) and SVG
    (vector, scales losslessly to any zoom level or print size). Both use
    the same rcParams (dpi, bbox_inches) already set at module load, so
    this only needs to be called once per figure rather than duplicating
    savefig() calls throughout the file.
    """
    fig.savefig(f"{path_no_ext}.png")
    fig.savefig(f"{path_no_ext}.svg")


def text_color_for(cmap, norm, value):
    """
    Return 'white' or 'black' for annotation text drawn on top of a heatmap
    cell, based on the actual relative luminance of the colour that cell
    will be rendered in (not a hand-tuned threshold). This keeps cell text
    legible regardless of which colormap is chosen, since a hard-coded
    "high values are dark" rule (true for RdYlGn) silently breaks for a
    colormap with a different lightness ramp (e.g. viridis, where high
    values are the brightest, not the darkest, part of the scale).
    """
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b  # matplotlib returns 0-1 floats
    return "white" if luminance < 0.5 else "black"


def p_value_stars(p):
    """
    Conventional significance-star annotation, used only for Figures
    5.8-5.10 where per-bar p-values are reported individually rather
    than via a single Bonferroni-corrected alpha threshold applied
    uniformly (contrast with Figures 5.4/5.5, which annotate against a
    single alpha = 0.05/54 threshold because they aggregate many
    comparisons). H5/H6 are two independent, individually pre-registered
    tests, not a family of 54 comparisons, so a per-test alpha = 0.05 is
    the appropriate threshold here, not a Bonferroni-adjusted one.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


# =======================================================================
# FIGURE 5.1: Phase A extraction throughput (option-independent)
# =======================================================================
def fig_phase_a_throughput():
    model_keys = ["uni", "conch", "quilt_llava", "gigapath_tile"]
    labels = ["UNI", "CONCH", "Quilt-LLaVA", "Prov-GigaPath\n(tile)"]
    throughput = [214.8, 333.7, 71.9, 147.5]
    colors = [MODEL_COLORS[m] for m in model_keys]
    hatches = [MODEL_HATCHES[m] for m in model_keys]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, throughput, color=colors, hatch=hatches,
                   edgecolor="black", linewidth=0.8, width=0.6)
    for b, v in zip(bars, throughput):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Throughput (images/second)")
    ax.set_title("Figure 5.1: Phase A Embedding Extraction Throughput by Model")
    ax.set_ylim(0, 380)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/shared_fig_5_1_phase_a_throughput")
    plt.close(fig)


# =======================================================================
# FIGURE 5.2: Cosine similarity heatmap at moderate severity, ACC vs GDC
# =======================================================================
def fig_cosine_heatmap(option, models_order):
    df = pd.read_csv(f"{DATA}/option_{option}_h2_demographic_main_effect.csv")
    df = df[df["severity"] == "moderate"].copy()

    pids = sorted(PERT_NAMES.keys())
    acc_mat = np.zeros((len(pids), len(models_order)))
    gdc_mat = np.zeros((len(pids), len(models_order)))
    for i, pid in enumerate(pids):
        for j, m in enumerate(models_order):
            row = df[(df["perturbation_id"] == pid) & (df["model"] == m)]
            acc_mat[i, j] = row["mean_a"].values[0]
            gdc_mat[i, j] = row["mean_b"].values[0]

    row_labels = [pert_label(p) for p in pids]
    col_labels = [MODEL_LABELS[m] for m in models_order]

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 8.5), sharey=True)
    # Viridis: perceptually uniform, monotonically increasing in lightness,
    # and explicitly designed to remain distinguishable under colour vision
    # deficiency (unlike RdYlGn, whose red/green endpoints are the classic
    # failure case for deuteranopia/protanopia and collapse to similar
    # greys under photocopying). See Appendix I for the full rationale.
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=0.3, vmax=1.0)
    for ax, mat, title in zip(axes, [acc_mat, gdc_mat], ["ACC (African cohort)", "GDC (Western control)"]):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=30, ha="right")
        ax.set_title(title, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt_color = text_color_for(cmap, norm, v)
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=txt_color)
    axes[0].set_yticks(range(len(row_labels)))
    axes[0].set_yticklabels(row_labels, fontsize=8)
    cbar = fig.colorbar(axes[1].images[0], ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("Mean cosine similarity to baseline")
    fig.suptitle(
        "Figure 5.2: Cosine Similarity to Baseline at Moderate Severity,\nby Perturbation, Model, and Cohort "
        + ("(Option A)" if option == "a" else "(Option B)"),
        fontsize=11, fontweight="bold", y=0.995,
    )
    save_both_formats(fig, f"{OUT}/option_{option}_fig_5_2_cosine_heatmap")
    plt.close(fig)


# =======================================================================
# FIGURE 5.3: Severity trend by perturbation category, per model (pooled)
# =======================================================================
def fig_severity_trends(option, models_order):
    df = pd.read_csv(f"{DATA}/option_{option}_h2_demographic_main_effect.csv")
    # Pooled mean across cohorts, weighted by fixed n_a/n_b (1989 / 1865)
    n_a, n_b = df["n_a"].iloc[0], df["n_b"].iloc[0]
    df["pooled_mean"] = (df["mean_a"] * n_a + df["mean_b"] * n_b) / (n_a + n_b)
    df["category"] = df["perturbation_id"].map(PERT_CATEGORY)

    categories = sorted(set(PERT_CATEGORY.values()))
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4), sharey=True)
    for ax, cat in zip(axes, categories):
        sub = df[df["category"] == cat]
        for m in models_order:
            msub = sub[sub["model"] == m]
            means = msub.groupby("severity")["pooled_mean"].mean().reindex(SEVERITY_ORDER)
            ax.plot(SEVERITY_ORDER, means.values, label=MODEL_LABELS[m],
                    color=MODEL_COLORS[m], linestyle=MODEL_LINESTYLES[m],
                    marker=MODEL_MARKERS[m], linewidth=2.2, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.6)
        ax.set_title(cat, fontsize=9)
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean cosine similarity\n(pooled across cohorts)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models_order), bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle(
        "Figure 5.3: Embedding Stability by Perturbation Severity and Category "
        + ("(Option A: UNI, CONCH, Quilt-LLaVA)" if option == "a" else "(Option B: UNI, CONCH, Prov-GigaPath)"),
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_both_formats(fig, f"{OUT}/option_{option}_fig_5_3_severity_trends")
    plt.close(fig)


# =======================================================================
# FIGURE 5.4 (NEW): Demographic robustness gap heatmap (Cohen's d)
# =======================================================================
def fig_demographic_gap_heatmap(option, models_order):
    df = pd.read_csv(f"{DATA}/option_{option}_h2_demographic_main_effect.csv")
    df = df[df["severity"] == "moderate"].copy()

    pids = sorted(PERT_NAMES.keys())
    mat = np.zeros((len(pids), len(models_order)))
    sig = np.zeros((len(pids), len(models_order)), dtype=bool)
    for i, pid in enumerate(pids):
        for j, m in enumerate(models_order):
            row = df[(df["perturbation_id"] == pid) & (df["model"] == m)]
            mat[i, j] = row["cohens_d"].values[0]
            sig[i, j] = bool(row["significant_bonferroni"].values[0])

    row_labels = [pert_label(p) for p in pids]
    col_labels = [MODEL_LABELS[m] for m in models_order]

    fig, ax = plt.subplots(figsize=(5.2, 8.5))
    # PuOr (purple-white-orange): a ColorBrewer diverging scheme verified
    # colourblind-safe, replacing RdBu_r's red/blue endpoints, which are
    # harder to distinguish under tritanopia and whose red end is easily
    # confused with the green endpoints used elsewhere in this figure set
    # under deuteranopia/protanopia. See Appendix I for the full rationale.
    # Every cell is also annotated with its signed numeric value (and a
    # significance marker), so direction and magnitude remain readable
    # from the text alone even if colour is unavailable, e.g. in a
    # black-and-white photocopy.
    cmap = plt.get_cmap("PuOr")
    norm = TwoSlopeNorm(vmin=-2.0, vcenter=0, vmax=2.0)
    im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            marker = "*" if sig[i, j] else ""
            ax.text(j, i, f"{mat[i, j]:.2f}{marker}", ha="center", va="center", fontsize=7,
                    color=text_color_for(cmap, norm, mat[i, j]))
    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label("Cohen's d (ACC − GDC)\npositive = ACC more stable")
    ax.set_title(
        "Figure 5.4: Demographic Robustness Gap (Cohen's d)\nby Perturbation and Model, Moderate Severity\n"
        "(* = significant at Bonferroni-corrected α = 0.05/54)\n"
        + ("Option A" if option == "a" else "Option B"),
        fontsize=10,
    )
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/option_{option}_fig_5_4_demographic_gap_heatmap")
    plt.close(fig)


# =======================================================================
# FIGURE 5.5 (NEW): H4 perturbation vulnerability ranking (mixed-effects coefficients)
# =======================================================================
def fig_h4_perturbation_ranking(option):
    with open(f"{DATA}/option_{option}_h4_model_x_perturbation_interaction.json") as f:
        h4 = json.load(f)
    fe = h4["fixed_effects"]
    intercept = fe["Intercept"]["coef"]

    rows = []
    # perturbation 1 is the reference level (coefficient 0 by construction)
    rows.append((1, 0.0, True))
    for key, val in fe.items():
        if key == "Intercept":
            continue
        pid = int(key.split("T.")[1].rstrip("]"))
        rows.append((pid, val["coef"], val["p_value"] < 0.05 / 54))

    rows.sort(key=lambda r: r[1])
    labels = [pert_label(r[0]) for r in rows]
    coefs = [r[1] for r in rows]
    sig = [r[2] for r in rows]
    colors = [CAT_COLORS[PERT_CATEGORY[r[0]]] for r in rows]
    hatches = [CAT_HATCHES[PERT_CATEGORY[r[0]]] for r in rows]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    bars = ax.barh(labels, coefs, color=colors, hatch=hatches, edgecolor="black", linewidth=0.7)
    for b, s in zip(bars, sig):
        if s:
            x = b.get_width()
            ax.text(x + (0.006 if x >= 0 else -0.006), b.get_y() + b.get_height() / 2, "*",
                    va="center", ha="left" if x >= 0 else "right", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Coefficient (change in cosine similarity relative to\nPerturbation 01 baseline; model intercept = {intercept:.3f})")
    ax.set_title(
        "Figure 5.5: Perturbation-Level Vulnerability Ranking\n(Mixed-Effects Model, Pooled Across Models)\n"
        "(* = significant at Bonferroni-corrected α = 0.05/54)\n"
        + ("Option A" if option == "a" else "Option B"),
        fontsize=10,
    )
    legend_patches = [
        mpatches.Patch(facecolor=CAT_COLORS[k], hatch=CAT_HATCHES[k], edgecolor="black", label=k)
        for k in CAT_COLORS
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/option_{option}_fig_5_5_h4_perturbation_ranking")
    plt.close(fig)


# =======================================================================
# FIGURE 5.6 (NEW): H1 model main effect (cosine + CKA side by side)
# =======================================================================
def fig_h1_model_main_effect(option, models_order):
    with open(f"{DATA}/option_{option}_h1_model_main_effect.json") as f:
        h1_cos = json.load(f)
    with open(f"{DATA}/option_{option}_h1_model_main_effect_cka.json") as f:
        h1_cka = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))
    for ax, h1, title, ylab in zip(
        axes, [h1_cos, h1_cka],
        [f"Cosine similarity\n(Kruskal-Wallis H={h1_cos['statistic']:.1f}, p={h1_cos['p_value']:.2e})",
         f"Linear CKA\n(Kruskal-Wallis H={h1_cka['statistic']:.2f}, p={h1_cka['p_value']:.4f})"],
        ["Mean cosine similarity\n(all tiles, all perturbations)", "Mean linear CKA\n(all perturbation-type cells)"],
    ):
        means = [h1["group_means"][m] for m in models_order]
        colors = [MODEL_COLORS[m] for m in models_order]
        hatches = [MODEL_HATCHES[m] for m in models_order]
        bars = ax.bar([MODEL_LABELS[m] for m in models_order], means, color=colors, hatch=hatches,
                       edgecolor="black", linewidth=0.8, width=0.55)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylab, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.suptitle(
        "Figure 5.6: H1 – Model Main Effect on Embedding Stability ("
        + ("Option A)" if option == "a" else "Option B)"),
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_both_formats(fig, f"{OUT}/option_{option}_fig_5_6_h1_model_main_effect")
    plt.close(fig)


# =======================================================================
# FIGURE 5.7 (NEW): Pairwise model comparison (Cohen's d, cosine)
# =======================================================================
def fig_pairwise_comparison(option, model_pairs):
    df = pd.read_csv(f"{DATA}/option_{option}_pairwise_comparisons_bonferroni.csv")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    labels, means, sig_pcts, colors, hatches = [], [], [], [], []
    for (ma, mb), color, hatch in zip(model_pairs, TRIPLET_COLORS, TRIPLET_HATCHES):
        sub = df[(df["model_a"] == ma) & (df["model_b"] == mb)]
        labels.append(f"{MODEL_LABELS[ma]} vs\n{MODEL_LABELS[mb]}")
        means.append(sub["cohens_d"].mean())
        sig_pcts.append(100 * sub["significant_bonferroni"].mean())
        colors.append(color)
        hatches.append(hatch)

    bars = ax.bar(labels, means, color=colors, hatch=hatches, edgecolor="black", linewidth=0.8, width=0.55)
    for b, v, s in zip(bars, means, sig_pcts):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.08),
                f"d={v:.2f}\n{s:.0f}% sig.", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Cohen's d across 18 perturbations x 3 severities\n(positive = first-named model more stable)")
    ax.set_title(
        "Figure 5.7: Pairwise Model Comparison Summary (Cosine Similarity)\n"
        + ("Option A" if option == "a" else "Option B"),
        fontsize=10, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/option_{option}_fig_5_7_pairwise_comparison")
    plt.close(fig)


# =======================================================================
# WORKSTREAM 5 DATA (H5, H6) -- see module docstring for provenance
# =======================================================================
# H5: within-model, matched-pairs Wilcoxon signed-rank test, hflip vs
# vflip. Source: docs/Nyamwaya_Workstream5_Prediction_Matrix.md, Section 5.1
# (post-registration addendum, 29 September 2026).
#   median_diff       = median(hflip_cosine_sim - vflip_cosine_sim)
#   effect_size_2g     = 2 x Cohen's g = (n_pos - n_neg) / (n_pos + n_neg)
#                        (relabelled from "rank-biserial r"; see the
#                        module docstring and the prediction matrix's own
#                        relabelling note for the derivation)
H5_RESULTS = {
    "uni":            {"family": "dinov2", "median_diff": 0.0275, "effect_size_2g": 0.720,  "p_value": 2.2e-308, "confirmed": True},
    "gigapath_tile":  {"family": "dinov2", "median_diff": 0.0021, "effect_size_2g": 0.182,  "p_value": 1.12e-37, "confirmed": True},
    "conch":          {"family": "clip",   "median_diff": -0.0002, "effect_size_2g": -0.059, "p_value": 4.40e-5,  "confirmed": False},
    "quilt_llava":    {"family": "clip",   "median_diff": 0.0027, "effect_size_2g": 0.505,  "p_value": 5.46e-259, "confirmed": False},
}
H5_MODEL_ORDER = ["uni", "gigapath_tile", "conch", "quilt_llava"]

# H6: between-family (pooled), Mann-Whitney U test, solarize vs baseline.
# Source: same document, Section 5.2.
H6_RESULTS = {
    "dinov2": {"mean_cosine_sim": 0.821, "n": 7708},
    "clip":   {"mean_cosine_sim": 0.655, "n": 7708},
    "cohens_d": 1.186,
    "cliffs_delta": 0.596,
    "p_value": 2.2e-308,
}


# =======================================================================
# FIGURE 5.8 (NEW): H5 -- flip invariance, within-model paired comparison
# =======================================================================
def fig_h5_flip_invariance():
    """
    One bar per model: the median per-tile cosine-similarity difference
    between horizontal and vertical flip (hflip - vflip), from the
    within-model paired Wilcoxon signed-rank test (H5). Bars are ordered
    by training family (DINOv2 first, CLIP second) rather than
    alphabetically, so the pre-registered family grouping (Section 2.1-
    2.2 of the prediction matrix) is visible directly in the bar order,
    not only in the colour/hatch encoding.

    Each bar is annotated with 2 x Cohen's g (the effect size actually
    computed; see H5_RESULTS docstring above for why this is not called
    "rank-biserial correlation") and a conventional significance-star
    marker. A red outline distinguishes the two models whose result
    disconfirmed its pre-registered prediction (CONCH, Quilt-LLaVA),
    consistent with Section 5.1's own framing: the disconfirmations are
    a reported, discussed finding, not a result to visually de-emphasise.
    """
    labels = [MODEL_LABELS[m] for m in H5_MODEL_ORDER]
    diffs = [H5_RESULTS[m]["median_diff"] for m in H5_MODEL_ORDER]
    families = [H5_RESULTS[m]["family"] for m in H5_MODEL_ORDER]
    colors = [FAMILY_COLORS[f] for f in families]
    hatches = [FAMILY_HATCHES[f] for f in families]
    confirmed = [H5_RESULTS[m]["confirmed"] for m in H5_MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        labels, diffs, color=colors, hatch=hatches,
        edgecolor=["black" if c else OKABE_ITO["vermillion"] for c in confirmed],
        linewidth=[0.8 if c else 2.2 for c in confirmed], width=0.55,
    )
    for b, m in zip(bars, H5_MODEL_ORDER):
        r = H5_RESULTS[m]
        y = b.get_height()
        offset = 0.0015 if y >= 0 else -0.0015
        va = "bottom" if y >= 0 else "top"
        ax.text(
            b.get_x() + b.get_width() / 2, y + offset,
            f"2g={r['effect_size_2g']:.3f} {p_value_stars(r['p_value'])} ({'confirmed' if r['confirmed'] else 'disconfirmed'})",
            ha="center", va=va, fontsize=7.5,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    # Extra headroom above the tallest bar (UNI, +0.0275) so its
    # annotation clears the two-line title rather than overlapping it.
    y_max, y_min = max(diffs), min(diffs)
    ax.set_ylim(y_min - 0.003, y_max * 1.45)
    ax.set_ylabel("Median difference in cosine similarity to baseline\n(hflip − vflip)")
    ax.set_title(
        "Figure 5.8: H5 – Flip Invariance, Within-Model Paired Comparison\n"
        "(Wilcoxon signed-rank test; red outline = pre-registered prediction disconfirmed)",
        fontsize=10,
    )
    legend_patches = [
        mpatches.Patch(facecolor=FAMILY_COLORS[k], hatch=FAMILY_HATCHES[k], edgecolor="black", label=FAMILY_LABELS[k])
        for k in ["dinov2", "clip"]
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/shared_fig_5_8_h5_flip_invariance")
    plt.close(fig)


# =======================================================================
# FIGURE 5.9 (NEW): H6 -- solarization invariance, between-family comparison
# =======================================================================
def fig_h6_solarization_invariance():
    """
    Two bars: pooled mean cosine similarity to baseline under
    solarization, DINOv2-family (UNI + Prov-GigaPath) versus CLIP-family
    (CONCH + Quilt-LLaVA), from the between-family Mann-Whitney U test
    (H6). Annotated with both effect sizes reported in Section 5.2 of the
    prediction matrix (Cliff's delta, the effect size matching the
    non-parametric test actually used, and Cohen's d, reported only for
    cross-dissertation comparability and explicitly labelled secondary
    there because both pooled groups failed the Shapiro-Wilk normality
    check).
    """
    families = ["dinov2", "clip"]
    labels = [FAMILY_LABELS[f] for f in families]
    means = [H6_RESULTS[f]["mean_cosine_sim"] for f in families]
    ns = [H6_RESULTS[f]["n"] for f in families]
    colors = [FAMILY_COLORS[f] for f in families]
    hatches = [FAMILY_HATCHES[f] for f in families]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    bars = ax.bar(labels, means, color=colors, hatch=hatches, edgecolor="black", linewidth=0.8, width=0.5)
    for b, v, n in zip(bars, means, ns):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}\n(n={n:,})", ha="center", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean cosine similarity to baseline under solarization")
    stars = p_value_stars(H6_RESULTS["p_value"])
    ax.set_title(
        "Figure 5.9: H6 – Solarization Invariance, Between-Family Comparison\n"
        f"(Mann-Whitney U, {stars}; Cliff's δ = {H6_RESULTS['cliffs_delta']:.3f}, "
        f"Cohen's d = {H6_RESULTS['cohens_d']:.3f} [secondary])",
        fontsize=9.5,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/shared_fig_5_9_h6_solarization_invariance")
    plt.close(fig)


# =======================================================================
# FIGURE 5.10 (NEW): Workstream 5 effect-size summary (H5 + H6 together)
# =======================================================================
def fig_workstream5_effect_size_summary():
    """
    A single forest-plot-style figure placing all five Workstream 5
    effect sizes on one shared [-1, 1] axis: H5's four per-model 2 x
    Cohen's g values and H6's one pooled Cliff's delta. Both statistics
    are rank/sign-based and bounded in [-1, 1], which is what makes a
    shared axis meaningful (Section 5.2 of the prediction matrix makes
    this comparability argument explicitly), but they are NOT the same
    statistic -- H5's is a within-model paired comparison, H6's is a
    between-family pooled comparison -- so the two families are given
    distinct marker shapes and a legend entry making the distinction
    explicit, rather than a single undifferentiated set of dots that
    would visually imply they are interchangeable.
    """
    rows = []  # (label, value, marker_family, color)
    for m in H5_MODEL_ORDER:
        r = H5_RESULTS[m]
        rows.append((f"{MODEL_LABELS[m]} (H5)", r["effect_size_2g"], "h5", FAMILY_COLORS[r["family"]]))
    rows.append(("DINOv2 vs. CLIP,\npooled (H6)", H6_RESULTS["cliffs_delta"], "h6", OKABE_ITO["reddish_purple"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    y_pos = np.arange(len(rows))[::-1]
    for y, (label, value, fam, color) in zip(y_pos, rows):
        marker = "o" if fam == "h5" else "D"
        ax.plot([0, value], [y, y], color="black", linewidth=1.0, zorder=1)
        ax.scatter([value], [y], s=110, color=color, marker=marker,
                   edgecolor="black", linewidth=1.0, zorder=2)
        ax.text(value + (0.04 if value >= 0 else -0.04), y, f"{value:.3f}",
                ha="left" if value >= 0 else "right", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(-1.0, 1.0)
    # Extra headroom below the lowest row (H6) so the legend has clear
    # space rather than sitting on top of the H6 marker/annotation.
    ax.set_ylim(min(y_pos) - 1.1, max(y_pos) + 0.6)
    ax.set_xlabel("Effect size (bounded [−1, 1]; see legend for which statistic)")
    ax.set_title("Figure 5.10: Workstream 5 Effect-Size Summary (H5 and H6)", fontsize=10)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="grey", markeredgecolor="black",
                   markersize=9, label="2 × Cohen's g (H5, within-model paired)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=OKABE_ITO["reddish_purple"], markeredgecolor="black",
                   markersize=9, label="Cliff's δ (H6, between-family pooled)"),
    ]
    ax.legend(handles=legend_handles, loc="lower center", fontsize=7.5, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    fig.tight_layout()
    save_both_formats(fig, f"{OUT}/shared_fig_5_10_workstream5_effect_size_summary")
    plt.close(fig)


# =======================================================================
# RUN ALL
# =======================================================================
def main():
    global DATA, OUT

    parser = argparse.ArgumentParser(
        description="Generate Phase B results figures for the dissertation "
                     "(Model Cohort Option A and/or Option B), plus the "
                     "option-independent Workstream 5 figures (5.8-5.10)."
    )
    parser.add_argument(
        "--data-dir", default=DATA,
        help=f"Directory containing the option_a_*/option_b_* Phase B output "
             f"files (default: {DATA})",
    )
    parser.add_argument(
        "--out-dir", default=OUT,
        help=f"Directory to write generated PNG figures to (default: {OUT})",
    )
    parser.add_argument(
        "--options", default="a,b", choices=["a", "b", "a,b"],
        help="Which cohort configuration(s) to generate figures for "
             "(default: a,b, i.e. both)",
    )
    args = parser.parse_args()

    DATA = args.data_dir
    OUT = args.out_dir
    os.makedirs(OUT, exist_ok=True)

    if not os.path.isdir(DATA):
        raise SystemExit(
            f"Data directory not found: {DATA}\n"
            f"Pass --data-dir pointing at the folder containing the "
            f"option_a_*/option_b_* Phase B output files."
        )

    selected_options = args.options.split(",")

    fig_phase_a_throughput()
    print("Shared: Figure 5.1 (Phase A throughput) done.")

    configs = {
        "a": {
            "models_order": ["uni", "conch", "quilt_llava"],
            "pairs": [("conch", "quilt_llava"), ("conch", "uni"), ("quilt_llava", "uni")],
        },
        "b": {
            "models_order": ["uni", "conch", "gigapath_tile"],
            "pairs": [("conch", "gigapath_tile"), ("conch", "uni"), ("gigapath_tile", "uni")],
        },
    }

    for opt in selected_options:
        cfg = configs[opt]
        fig_cosine_heatmap(opt, cfg["models_order"])
        fig_severity_trends(opt, cfg["models_order"])
        fig_demographic_gap_heatmap(opt, cfg["models_order"])
        fig_h4_perturbation_ranking(opt)
        fig_h1_model_main_effect(opt, cfg["models_order"])
        fig_pairwise_comparison(opt, cfg["pairs"])
        print(f"Option {opt.upper()}: figures done.")

    # Workstream 5 (H5, H6): option-independent, since all four models are
    # tested regardless of which trio (Option A or B) is the dissertation's
    # primary cohort. Generated once per run, not once per option.
    fig_h5_flip_invariance()
    fig_h6_solarization_invariance()
    fig_workstream5_effect_size_summary()
    print("Shared: Figures 5.8-5.10 (Workstream 5) done.")

    print("\nAll figures written to", OUT)
    for fn in sorted(os.listdir(OUT)):
        print(" -", fn)


if __name__ == "__main__":
    main()
