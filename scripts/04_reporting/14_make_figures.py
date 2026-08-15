"""
make_figures.py
-----------------------------------------------------------------------
Generates all Phase B results figures used in Sections 5.1-5.10 of the
dissertation, for both Model Cohort Option A (UNI, CONCH, Quilt-LLaVA)
and Option B (UNI, CONCH, Prov-GigaPath tile encoder), from the real
`phase_b_statistical_analysis.py` output files (H1-H4 JSON summaries,
H2/pairwise/outlier CSV tables).

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

OUTPUT
------
13 PNG files written to OUT_DIR, matching the #FIGURE: references in
dissertation.md / dissertation_optionB.md:
    shared_fig_5_1_phase_a_throughput.png
    option_{a,b}_fig_5_2_cosine_heatmap.png
    option_{a,b}_fig_5_3_severity_trends.png
    option_{a,b}_fig_5_4_demographic_gap_heatmap.png
    option_{a,b}_fig_5_5_h4_perturbation_ranking.png
    option_{a,b}_fig_5_6_h1_model_main_effect.png
    option_{a,b}_fig_5_7_pairwise_comparison.png

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

plt.rcParams["hatch.linewidth"] = 1.3


def pert_label(pid):
    return f"{pid:02d}. {PERT_NAMES[pid]}"


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
    fig.savefig(f"{OUT}/shared_fig_5_1_phase_a_throughput.png")
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
    fig.savefig(f"{OUT}/option_{option}_fig_5_2_cosine_heatmap.png")
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
    fig.savefig(f"{OUT}/option_{option}_fig_5_3_severity_trends.png")
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
    cbar.set_label("Cohen's d (ACC \u2212 GDC)\npositive = ACC more stable")
    ax.set_title(
        "Figure 5.4: Demographic Robustness Gap (Cohen's d)\nby Perturbation and Model, Moderate Severity\n"
        "(* = significant at Bonferroni-corrected \u03b1 = 0.05/54)\n"
        + ("Option A" if option == "a" else "Option B"),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(f"{OUT}/option_{option}_fig_5_4_demographic_gap_heatmap.png")
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
        "(* = significant at Bonferroni-corrected \u03b1 = 0.05/54)\n"
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
    fig.savefig(f"{OUT}/option_{option}_fig_5_5_h4_perturbation_ranking.png")
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
        "Figure 5.6: H1 \u2013 Model Main Effect on Embedding Stability ("
        + ("Option A)" if option == "a" else "Option B)"),
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/option_{option}_fig_5_6_h1_model_main_effect.png")
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
    fig.savefig(f"{OUT}/option_{option}_fig_5_7_pairwise_comparison.png")
    plt.close(fig)


# =======================================================================
# RUN ALL
# =======================================================================
def main():
    global DATA, OUT

    parser = argparse.ArgumentParser(
        description="Generate Phase B results figures for the dissertation "
                     "(Model Cohort Option A and/or Option B)."
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

    print("\nAll figures written to", OUT)
    for fn in sorted(os.listdir(OUT)):
        print(" -", fn)


if __name__ == "__main__":
    main()
