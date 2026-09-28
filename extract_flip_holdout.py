#!/usr/bin/env python3
"""
extract_flip_holdout.py

Workstream 5 (revised): two independent, out-of-sample, HARKing-safe
invariance tests, both derived a priori from the DINOv2-vs-CLIP training
recipe argument described in Nyamwaya_Workstream5_Prediction_Matrix.md.
Read that document first -- it must be git-committed BEFORE this script
is ever run against real (non-mock) embeddings, since the whole point of
both tests is that the prediction precedes the evidence.

=============================================================================
THE TWO HELD-OUT CONDITIONS
=============================================================================

Neither of the following appears in perturbation_pipeline.py's 18-
perturbation battery (confirmed by inspection of PERTURBATION_REGISTRY):

  flip_h / flip_v : horizontal / vertical flip (PIL.Image.transpose).
      DINOv2's reference augmentation pipeline (facebookresearch/dinov2,
      dinov2/data/augmentations.py) applies RandomHorizontalFlip(p=0.5)
      to both global and local crops, and never a vertical flip.

  solarize : PIL.ImageOps.solarize(img, threshold=128).
      DINOv2's pipeline applies RandomSolarize(threshold=128, p=0.2) to
      one of its two global crops only (global_transfo2_extra). Threshold
      and probability are taken directly from the cited source file, not
      assumed.

CLIP's published recipe (Radford et al., 2021) uses only a random square
crop -- none of flip, solarize -- so CONCH and Quilt-LLaVA (both CLIP/
CoCa-family vision encoders) have no trained pressure toward invariance
to any of these three conditions.

=============================================================================
TWO DIFFERENT TEST DESIGNS, FOR A PRINCIPLED REASON
=============================================================================

flip_h vs flip_v is a WITHIN-MODEL, matched-pairs comparison: the same
tile, the same model, two conditions. A paired test (Wilcoxon signed-rank
on the per-tile cosine-similarity difference) is the appropriate choice
here, because it removes each tile's own baseline "embeddability" as a
source of noise and tests only the hflip/vflip difference directly.

solarize has no natural sibling condition to pair against within a model
-- there is nothing to compare it to except baseline, which produces one
number per tile per model, not a difference. The prediction instead
concerns a BETWEEN-MODEL contrast: is solarization invariance higher in
the DINOv2 family than the CLIP family? That is an unpaired, two-group
comparison, so it reuses phase_b_statistical_analysis.py's own
Shapiro-Wilk-gated decision procedure (Welch's t-test if both groups
pass Shapiro-Wilk, Mann-Whitney U otherwise) directly, rather than
introducing a second, inconsistent ad hoc test-selection rule.

=============================================================================
WHAT THIS SCRIPT DOES
=============================================================================

  1. Generates all three held-out conditions from the EXISTING baseline
     tiles, IN MEMORY (PIL only: Image.transpose / ImageOps.solarize). No
     new files are written to perturbation_pipeline.py's output tree, and
     there is no dependency on that pipeline's cv2/elasticdeform stack.
  2. Extracts embeddings for all three conditions, for all four study
     models, by importing and reusing extract_embeddings_similarity.py's
     ADAPTER_REGISTRY, EmbeddingStore, and batching loop directly -- no
     model-loading code is reimplemented.
  3. Writes to a SEPARATE HDF5 store per model
     (embeddings_flip_holdout_{model}.h5, keys "flip_h__<sample_id>",
     "flip_v__<sample_id>", "solarize__<sample_id>"), kept structurally
     apart from the original embeddings_{model}.h5 stores, so this
     held-out test's provenance is auditable as a distinct artifact.
  4. Reuses each model's ALREADY-CACHED baseline embedding
     (embeddings_{model}.h5, key "baseline__<sample_id>"). Baseline is
     never recomputed.
  5. Computes per-tile cosine similarity and linear CKA (pooled/ACC/GDC)
     for all three conditions, using the exact same functions as
     extract_embeddings_similarity.py.
  6. Runs BOTH pre-registered statistical tests (see above) and reports
     each directly against the prediction matrix's stated hypothesis.

Governance: runs entirely on Artemis against ACC+GDC tiles already
resident there. No data leaves University of Sussex infrastructure; the
GDC-only constraint enforced by artemis_preserve_gdc.sh (Route B) does
not apply to this experiment.

Resilience: extraction reuses EmbeddingStore's per-key existence check,
so a job killed mid-run resumes exactly where it left off on restart.

=============================================================================
USAGE (same multi-conda-env pattern as extract_embeddings_similarity.py;
runs fine as four concurrent Slurm jobs, one per model, per conda env)
=============================================================================

  Extraction (once per model, from that model's own conda environment):

    conda activate uni-env
    python extract_flip_holdout.py --mode extract --model uni \
        --tiles-dir outputs/perturbed_tiles \
        --baseline-embeddings-dir outputs/embeddings \
        --holdout-embeddings-dir outputs/embeddings_flip_holdout

    (repeat for conch-env / quilt-llava-env / gigapath -- these four can
    run as separate, simultaneous Slurm jobs)

  Similarity + CKA (any environment with h5py/numpy; CPU-only, cheap):

    python extract_flip_holdout.py --mode similarity \
        --baseline-embeddings-dir outputs/embeddings \
        --holdout-embeddings-dir outputs/embeddings_flip_holdout \
        --cosine-out outputs/flip_holdout_cosine_similarity.csv \
        --cka-out outputs/flip_holdout_cka_summary.csv

  Statistical tests (needs scipy; phase_b_statistical_analysis.py must be
  importable from the same directory for the solarize between-family
  test to reuse its two_sample_test() decision procedure):

    python extract_flip_holdout.py --mode test \
        --cosine-out outputs/flip_holdout_cosine_similarity.csv \
        --stats-out outputs/flip_holdout_h5_test_results.json

  Dry run against the mock adapter first (matches this project's own
  established validation practice):

    python extract_flip_holdout.py --mode extract --model mock \
        --tiles-dir outputs/perturbed_tiles \
        --baseline-embeddings-dir outputs/embeddings \
        --holdout-embeddings-dir outputs/embeddings_flip_holdout_mock_test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Reuse of the existing, already-validated extraction infrastructure.
# Both modules must be importable from the same directory (or PYTHONPATH)
# as this script on Artemis.
# ---------------------------------------------------------------------------
from extract_embeddings_similarity import (  # noqa: E402
    ADAPTER_REGISTRY,
    DEFAULT_BATCH_SIZES,
    DEFAULT_NUM_WORKERS,
    TILE_SIZE,
    EmbeddingStore,
    ModelEnvironmentError,
    cosine_similarity_matrix_rows,
    derive_cohort,
    linear_cka,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Held-out conditions. Each is a callable transform applied to a PIL image.
# Grounded against the cited DINOv2 source (facebookresearch/dinov2,
# dinov2/data/augmentations.py, retrieved 28 Sep 2026):
#   RandomHorizontalFlip(p=0.5)            -- geometric_augmentation_global/local
#   RandomSolarize(threshold=128, p=0.2)   -- global_transfo2_extra (2nd global crop only)
#   No vertical flip anywhere in the pipeline.
# ---------------------------------------------------------------------------
SOLARIZE_THRESHOLD = 128  # facebookresearch/dinov2: RandomSolarize(threshold=128, p=0.2)

HOLDOUT_CONDITIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "flip_h": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    "flip_v": lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
    "solarize": lambda img: ImageOps.solarize(img, threshold=SOLARIZE_THRESHOLD),
}

# Paired (within-model) conditions tested against each other directly.
PAIRED_TEST_CONDITIONS = ("flip_h", "flip_v")
# Unpaired (between-model-family) condition tested against baseline only.
BETWEEN_FAMILY_TEST_CONDITIONS = ("solarize",)

# Per the prediction matrix: which models are DINOv2-trained (predicted
# higher invariance to both held-out conditions) vs CLIP/CoCa-trained
# (predicted no differential advantage on either).
MODEL_FAMILY = {
    "uni": "dinov2",
    "gigapath_tile": "dinov2",
    "conch": "clip_cocacontrastive",
    "quilt_llava": "clip",
}
DINOV2_MODELS = [m for m, fam in MODEL_FAMILY.items() if fam == "dinov2"]
CLIP_MODELS = [m for m, fam in MODEL_FAMILY.items() if fam != "dinov2"]


# ===========================================================================
# Manifest: held-out condition variants of the existing baseline tiles only
# ===========================================================================

def build_baseline_manifest(tiles_dir: Path) -> List[Tuple[str, Path]]:
    """Returns (sample_id, path) for every baseline PNG. These are the only
    images this script ever reads from disk; every held-out condition is
    generated from them in memory at __getitem__ time."""
    baseline_dir = tiles_dir / "baseline"
    if not baseline_dir.exists():
        raise FileNotFoundError(
            f"No baseline/ directory under {tiles_dir}. This script depends "
            f"on the same baseline tiles extract_embeddings_similarity.py "
            f"already used; run that pipeline first."
        )
    records = [(p.stem, p) for p in sorted(baseline_dir.glob("*.png"))]
    log.info("Baseline manifest: %d tiles found under %s.", len(records), baseline_dir)
    return records


class HoldoutTileDataset(Dataset):
    """For each baseline tile, yields every held-out condition not already
    embedded (per-item resume, matching EmbeddingStore's own resilience
    model), applying HOLDOUT_CONDITIONS[cond_name] before the model's own
    preprocessing transform."""

    def __init__(self, records: List[Tuple[str, Path]], model_transform, existing_keys: set):
        self.items: List[Tuple[str, Path, str]] = []
        for sample_id, path in records:
            for cond_name in HOLDOUT_CONDITIONS:
                key = f"{cond_name}__{sample_id}"
                if key not in existing_keys:
                    self.items.append((key, path, cond_name))
        self.model_transform = model_transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        key, path, cond_name = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = HOLDOUT_CONDITIONS[cond_name](img)
        return key, self.model_transform(img)


def _collate(batch):
    keys = [b[0] for b in batch]
    tensors = torch.stack([b[1] for b in batch])
    return keys, tensors


# ===========================================================================
# Phase 1: extraction
# ===========================================================================

def run_extraction(
    model_name: str,
    tiles_dir: Path,
    holdout_embeddings_dir: Path,
    batch_size: Optional[int] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: str = "cuda",
    precision: str = "bf16",
) -> None:
    adapter_cls = ADAPTER_REGISTRY.get(model_name)
    if adapter_cls is None:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(ADAPTER_REGISTRY)}")

    batch_size = batch_size or DEFAULT_BATCH_SIZES.get(model_name, 128)
    records = build_baseline_manifest(tiles_dir)

    store_path = holdout_embeddings_dir / f"embeddings_flip_holdout_{model_name}.h5"
    store = EmbeddingStore(store_path)
    existing_keys = set(store.keys())

    log.info("Loading %s (batch_size=%d, device=%s, precision=%s)...",
              model_name, batch_size, device, precision)
    adapter = adapter_cls(device=device)
    t0 = time.time()
    try:
        adapter.load()
    except ModelEnvironmentError:
        store.close()
        raise
    log.info("%s loaded in %.1fs (embedding_dim=%s).", model_name, time.time() - t0, adapter.embedding_dim)

    dataset = HoldoutTileDataset(records, adapter.get_transform(), existing_keys)
    total_possible = len(records) * len(HOLDOUT_CONDITIONS)
    log.info("%s: %d/%d (sample, condition) pairs already embedded (resuming); %d remaining.",
              model_name, total_possible - len(dataset), total_possible, len(dataset))

    if len(dataset) == 0:
        log.info("%s: nothing to do, every held-out condition already embedded for every tile.", model_name)
        store.close()
        adapter.unload()
        return

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=_collate,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    n_done = 0
    t_start = time.time()

    with torch.inference_mode():
        for batch_idx, (keys, tensors) in enumerate(loader):
            tensors = tensors.to(device, non_blocking=True)
            if amp_dtype is not None and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    embeddings = adapter.extract_batch(tensors)
            else:
                embeddings = adapter.extract_batch(tensors)

            for key, emb in zip(keys, embeddings):
                store.write(key, emb)
            store.flush()  # crash-safety: persist this batch immediately

            n_done += len(keys)
            if batch_idx % 10 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                log.info("%s: %d/%d embedded (%.1f img/s)...", model_name, n_done, len(dataset), rate)

    store.close()
    adapter.unload()
    log.info("%s: held-out extraction complete. %d embeddings in %.1fs.",
              model_name, n_done, time.time() - t_start)


# ===========================================================================
# Phase 2: similarity + CKA against the ALREADY-CACHED baseline
# ===========================================================================

def run_similarity(
    baseline_embeddings_dir: Path,
    holdout_embeddings_dir: Path,
    cosine_out: Path,
    cka_out: Path,
    models: Optional[List[str]] = None,
) -> None:
    if models is None:
        models = [
            p.stem.replace("embeddings_flip_holdout_", "")
            for p in holdout_embeddings_dir.glob("embeddings_flip_holdout_*.h5")
        ]

    cosine_out.parent.mkdir(parents=True, exist_ok=True)
    cka_out.parent.mkdir(parents=True, exist_ok=True)

    with open(cosine_out, "w", newline="", encoding="utf-8") as cos_fh, \
         open(cka_out, "w", newline="", encoding="utf-8") as cka_fh:

        cos_writer = csv.writer(cos_fh)
        cos_writer.writerow(["model", "sample_id", "cohort", "condition", "cosine_similarity"])

        cka_writer = csv.writer(cka_fh)
        cka_writer.writerow(["model", "condition", "cohort", "n_tiles", "linear_cka"])

        for model_name in models:
            baseline_path = baseline_embeddings_dir / f"embeddings_{model_name}.h5"
            holdout_path = holdout_embeddings_dir / f"embeddings_flip_holdout_{model_name}.h5"
            if not baseline_path.exists():
                log.warning("No baseline store for %s at %s; skipping.", model_name, baseline_path)
                continue
            if not holdout_path.exists():
                log.warning("No held-out store for %s at %s; skipping.", model_name, holdout_path)
                continue

            log.info("Computing held-out similarity metrics for %s...", model_name)
            baseline_store = EmbeddingStore(baseline_path)
            holdout_store = EmbeddingStore(holdout_path)

            baselines: Dict[str, np.ndarray] = {}
            for key in baseline_store.keys():
                if key.startswith("baseline__"):
                    baselines[key[len("baseline__"):]] = baseline_store.read(key)

            for cond_name in HOLDOUT_CONDITIONS:
                samples: Dict[str, np.ndarray] = {}
                prefix = f"{cond_name}__"
                for key in holdout_store.keys():
                    if key.startswith(prefix):
                        samples[key[len(prefix):]] = holdout_store.read(key)

                common = [sid for sid in samples if sid in baselines]
                if not common:
                    log.warning("%s/%s: no overlapping baseline+holdout tiles found.", model_name, cond_name)
                    continue

                base_mat = np.stack([baselines[sid] for sid in common])
                pert_mat = np.stack([samples[sid] for sid in common])
                sims = cosine_similarity_matrix_rows(base_mat, pert_mat)
                for sid, sim in zip(common, sims):
                    cos_writer.writerow([model_name, sid, derive_cohort(sid), cond_name, f"{sim:.6f}"])

                for cohort_label, subset_ids in [
                    ("POOLED", common),
                    ("ACC", [sid for sid in common if derive_cohort(sid) == "ACC"]),
                    ("GDC", [sid for sid in common if derive_cohort(sid) == "GDC"]),
                ]:
                    if len(subset_ids) < 3:
                        continue
                    sub_base = np.stack([baselines[sid] for sid in subset_ids])
                    sub_pert = np.stack([samples[sid] for sid in subset_ids])
                    cka_val = linear_cka(sub_base, sub_pert)
                    cka_writer.writerow([model_name, cond_name, cohort_label, len(subset_ids), f"{cka_val:.6f}"])

            baseline_store.close()
            holdout_store.close()
            log.info("%s: held-out similarity metrics written.", model_name)

    log.info("Cosine similarity report: %s", cosine_out)
    log.info("CKA summary report: %s", cka_out)


# ===========================================================================
# Phase 3: the two pre-registered statistical tests
# ===========================================================================

def _load_cosine_csv(cosine_out: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Returns sims[model][condition][sample_id] = cosine_similarity."""
    sims: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    with open(cosine_out, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sims[row["model"]][row["condition"]][row["sample_id"]] = float(row["cosine_similarity"])
    return sims


def run_paired_flip_test(sims: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, dict]:
    """H5: within-model, matched-pairs test. Two-sided Wilcoxon signed-rank
    on paired (by sample_id) flip_h-vs-baseline and flip_v-vs-baseline
    cosine similarities, per model. Justification: same tile, same model,
    two conditions -- a paired non-parametric test removes each tile's own
    baseline embeddability as a nuisance factor, which an unpaired test
    would not."""
    from scipy.stats import wilcoxon

    results = {}
    for model_name, by_cond in sims.items():
        if "flip_h" not in by_cond or "flip_v" not in by_cond:
            continue
        common_ids = sorted(set(by_cond["flip_h"]) & set(by_cond["flip_v"]))
        if len(common_ids) < 10:
            log.warning("%s: only %d paired tiles for flip test; too few for a reliable result.",
                        model_name, len(common_ids))
            continue

        h_vals = np.array([by_cond["flip_h"][sid] for sid in common_ids])
        v_vals = np.array([by_cond["flip_v"][sid] for sid in common_ids])
        diff = h_vals - v_vals

        stat, p_value = wilcoxon(h_vals, v_vals, alternative="two-sided")
        median_diff = float(np.median(diff))
        n_pos, n_neg = int(np.sum(diff > 0)), int(np.sum(diff < 0))
        rank_biserial = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else float("nan")

        family = MODEL_FAMILY.get(model_name, "unknown")
        predicted_direction = "hflip_greater" if family == "dinov2" else "no_significant_difference"
        observed_direction = (
            "hflip_greater" if (p_value < 0.05 and median_diff > 0) else
            "vflip_greater" if (p_value < 0.05 and median_diff < 0) else
            "no_significant_difference"
        )

        results[model_name] = {
            "hypothesis": "H5_flip_within_model_paired",
            "test": "wilcoxon_signed_rank",
            "n_paired_tiles": len(common_ids),
            "training_family": family,
            "median_hflip_cosine_sim": float(np.median(h_vals)),
            "median_vflip_cosine_sim": float(np.median(v_vals)),
            "median_difference_h_minus_v": median_diff,
            "statistic": float(stat),
            "p_value": float(p_value),
            "rank_biserial_effect_size": rank_biserial,
            "predicted_direction": predicted_direction,
            "observed_direction": observed_direction,
            "prediction_confirmed": predicted_direction == observed_direction,
        }
        log.info("H5 %s (%s): predicted=%s observed=%s (p=%.2e, median diff=%.4f, n=%d)",
                  model_name, family, predicted_direction, observed_direction, p_value, median_diff, len(common_ids))
    return results


def run_between_family_solarize_test(sims: Dict[str, Dict[str, Dict[str, float]]]) -> dict:
    """H6: between-model-family, unpaired test. Pools tile-level
    solarize-vs-baseline cosine similarity within each training family
    (DINOv2: UNI + Prov-GigaPath; CLIP: CONCH + Quilt-LLaVA) and compares
    the two pooled groups.

    Reuses phase_b_statistical_analysis.py's own two_sample_test() --
    Shapiro-Wilk normality check, Welch's t-test if both groups pass it,
    Mann-Whitney U otherwise -- so this test's decision procedure is
    identical to, and audited by, the same logic already validated
    elsewhere in this study, rather than a second ad hoc rule.
    """
    try:
        from phase_b_statistical_analysis import two_sample_test
    except ImportError as exc:
        raise SystemExit(
            "phase_b_statistical_analysis.py must be importable (same directory) "
            "for the H6 solarize between-family test, so it reuses that module's "
            "own validated two_sample_test() decision procedure."
        ) from exc

    dinov2_vals, clip_vals = [], []
    for model_name in DINOV2_MODELS:
        dinov2_vals.extend(sims.get(model_name, {}).get("solarize", {}).values())
    for model_name in CLIP_MODELS:
        clip_vals.extend(sims.get(model_name, {}).get("solarize", {}).values())

    if len(dinov2_vals) < 10 or len(clip_vals) < 10:
        log.warning("H6: insufficient data (dinov2 n=%d, clip n=%d); skipping.", len(dinov2_vals), len(clip_vals))
        return {}

    result = two_sample_test(np.array(dinov2_vals), np.array(clip_vals))
    predicted_direction = "dinov2_greater"
    observed_direction = (
        "dinov2_greater" if (result["p_value"] < 0.05 and result["mean_a"] > result["mean_b"]) else
        "clip_greater" if (result["p_value"] < 0.05 and result["mean_a"] < result["mean_b"]) else
        "no_significant_difference"
    )
    result.update({
        "hypothesis": "H6_solarize_between_family",
        "group_a": "dinov2_family (uni, gigapath_tile)",
        "group_b": "clip_family (conch, quilt_llava)",
        "predicted_direction": predicted_direction,
        "observed_direction": observed_direction,
        "prediction_confirmed": predicted_direction == observed_direction,
    })
    log.info("H6 solarize: predicted=%s observed=%s (test=%s, p=%.2e)",
              predicted_direction, observed_direction, result["test_used"], result["p_value"])
    return result


def run_test(cosine_out: Path, stats_out: Path) -> None:
    sims = _load_cosine_csv(cosine_out)
    output = {
        "H5_flip_within_model_paired": run_paired_flip_test(sims),
        "H6_solarize_between_family": run_between_family_solarize_test(sims),
    }
    stats_out.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_out, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    log.info("Test results written to %s", stats_out)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["extract", "similarity", "test"], required=True)
    parser.add_argument("--model", choices=list(ADAPTER_REGISTRY), default=None,
                         help="Required for --mode extract.")
    parser.add_argument("--tiles-dir", type=Path, default=Path("outputs/perturbed_tiles"))
    parser.add_argument("--baseline-embeddings-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--holdout-embeddings-dir", type=Path, default=Path("outputs/embeddings_flip_holdout"))
    parser.add_argument("--cosine-out", type=Path, default=Path("outputs/flip_holdout_cosine_similarity.csv"))
    parser.add_argument("--cka-out", type=Path, default=Path("outputs/flip_holdout_cka_summary.csv"))
    parser.add_argument("--stats-out", type=Path, default=Path("outputs/flip_holdout_h5_h6_test_results.json"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    if args.mode == "extract":
        if args.model is None:
            parser.error("--model is required for --mode extract")
        run_extraction(
            model_name=args.model, tiles_dir=args.tiles_dir,
            holdout_embeddings_dir=args.holdout_embeddings_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
            device=args.device, precision=args.precision,
        )
    elif args.mode == "similarity":
        run_similarity(
            baseline_embeddings_dir=args.baseline_embeddings_dir,
            holdout_embeddings_dir=args.holdout_embeddings_dir,
            cosine_out=args.cosine_out, cka_out=args.cka_out,
        )
    elif args.mode == "test":
        run_test(cosine_out=args.cosine_out, stats_out=args.stats_out)


if __name__ == "__main__":
    main()
