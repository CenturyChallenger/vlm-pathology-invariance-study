#!/usr/bin/env python
"""
preflight_check.py

Pre-flight validation for extract_embeddings_similarity.py, to be run
BEFORE submitting a Slurm GPU job for embedding extraction. Catches the
most common failure modes early, at zero GPU cost:

  1. Wrong conda environment active for the requested model
  2. Missing cross-cutting dependencies (torch, numpy, h5py, Pillow)
  3. Missing model-specific package (timm, conch, or the Quilt-LLaVA /
     LLaVA codebase)
  4. Missing or unauthenticated Hugging Face gated access (UNI, CONCH)
  5. GPU visibility (warns, does not fail, since CPU runs are technically
     possible, just impractically slow at this corpus size)
  6. Perturbation tile corpus completeness under --tiles-dir, checked
     against the expected count derived from PERTURBATION_REGISTRY
     (18 perturbations x 3 severities = 54 combos per tile) and, if
     found, cross-checked against production_progress.csv
  7. --embeddings-dir exists, is writable, and reports any partial /
     stale .h5 outputs from a previous interrupted run

Usage
-----
Run this ONCE from inside each model's conda environment before
extraction, exactly the way you'd run the real extraction command:

    conda activate uni-env
    python preflight_check.py --model uni \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

    conda activate conch-env
    python preflight_check.py --model conch \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

    conda activate quilt-llava-env
    python preflight_check.py --model quilt_llava \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

Omit --model to run only the environment-agnostic checks (tile corpus,
output directory, GPU visibility) -- useful when checking the
similarity-only environment before Phase B.

Exit codes
----------
  0 : all checks passed (or passed with warnings only)
  1 : one or more checks failed; do not submit the Slurm job yet

This script deliberately makes NO network calls (Hugging Face auth is
checked via the locally cached token only) and NEVER loads a model onto
the GPU. It is meant to run in seconds, not minutes.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

# ===========================================================================
# Expected corpus size, mirrored from perturbation_pipeline.py /
# extract_embeddings_similarity.py project notes:
#   18 perturbations x 3 severities = 54 combos per tile
#   3,854 tiles in the finished production run (per production_progress.csv,
#   confirmed 03 Aug 2026: 3,854/3,854 tiles, 208,116/208,116 combos, 0 errors)
# These are defaults, not hard-coded assumptions -- override with
# --expected-tiles if the corpus size ever changes (e.g. a re-run with a
# different --num-slides-per-cohort).
# ===========================================================================
NUM_PERTURBATIONS = 18
NUM_SEVERITIES = 3
COMBOS_PER_TILE = NUM_PERTURBATIONS * NUM_SEVERITIES  # 54
DEFAULT_EXPECTED_TILES = 3854
DEFAULT_EXPECTED_COMBOS = DEFAULT_EXPECTED_TILES * COMBOS_PER_TILE  # 208,116

# Model name -> the package(s) whose import is used as a proxy for "this
# model's dependencies are installed in the currently active environment".
# Mirrors the import checks inside UNIAdapter.load(), CONCHAdapter.load(),
# and QuiltLLaVAAdapter.load() in extract_embeddings_similarity.py.
MODEL_PACKAGE_CHECKS = {
    "uni": ["timm", "huggingface_hub"],
    "conch": ["conch"],
    # The Quilt-LLaVA codebase is typically installed as a local editable
    # package named "llava" (per the Quilt-LLaVA / LLaVA project layout).
    # If your install uses a different top-level module name, pass
    # --quilt-import-name to override this check.
    "quilt_llava": ["llava"],
}

# Which models require a gated Hugging Face token, per project notes.
HF_GATED_MODELS = {"uni", "conch"}

CROSS_CUTTING_PACKAGES = ["torch", "numpy", "h5py", "PIL"]


# ===========================================================================
# Small result-tracking helper
# ===========================================================================

class CheckResult:
    """Accumulates pass / warn / fail outcomes for a clean final summary."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        print(f"  [PASS] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def fail(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  [FAIL] {msg}")

    @property
    def has_failures(self) -> bool:
        return len(self.failures) > 0


# ===========================================================================
# Individual checks
# ===========================================================================

def check_environment_identity(result: CheckResult, expected_model: Optional[str]) -> None:
    """Reports which conda environment is active and flags an obvious mismatch."""
    print("\n[1/6] Environment identity")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    python_path = sys.executable

    if conda_env:
        result.ok(f"Active conda environment: '{conda_env}' (python: {python_path})")
    else:
        result.warn(
            "No CONDA_DEFAULT_ENV set. If you're using a different "
            "environment manager, this is expected and safe to ignore. "
            "If you meant to activate a conda environment, `conda activate` "
            "was likely skipped."
        )

    if expected_model:
        # Not a hard failure -- naming conventions vary -- but a useful
        # sanity nudge, since the project's environments are named
        # uni-env / conch-env / quilt-llava-env.
        expected_substring = expected_model.replace("_", "-")
        if conda_env and expected_substring not in conda_env.lower():
            result.warn(
                f"--model {expected_model} was requested, but the active "
                f"environment is named '{conda_env}', which doesn't "
                f"obviously match. Double-check you're in the right "
                f"environment before submitting the Slurm job."
            )


def check_cross_cutting_packages(result: CheckResult) -> None:
    """torch, numpy, h5py, and Pillow are required in EVERY model environment."""
    print("\n[2/6] Cross-cutting dependencies (required in every model environment)")
    for pkg in CROSS_CUTTING_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown version")
            result.ok(f"{pkg} importable ({version})")
        except ImportError:
            result.fail(
                f"{pkg} not importable in the active environment. "
                f"Install with: pip install torch torchvision h5py numpy "
                f"pillow --break-system-packages"
            )


def check_gpu_visibility(result: CheckResult) -> None:
    """Warns (does not fail) if CUDA isn't visible -- CPU extraction is possible but slow."""
    print("\n[3/6] GPU visibility")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            device_count = torch.cuda.device_count()
            result.ok(f"CUDA available: {device_count} device(s), device 0 = '{device_name}'")
        else:
            result.warn(
                "CUDA not available in this environment. Extraction will "
                "fall back to CPU, which will be dramatically slower across "
                f"{DEFAULT_EXPECTED_TILES} tiles x {COMBOS_PER_TILE} combos. "
                "If this is a login/head node rather than a GPU compute "
                "node, this is expected -- the real Slurm job will have "
                "GPU access."
            )
    except ImportError:
        result.fail("torch is not importable; cannot check GPU visibility.")


def check_model_specific_package(
    result: CheckResult, model: Optional[str], quilt_import_name: str
) -> None:
    """Checks the one model-specific package this environment is meant to provide."""
    print("\n[4/6] Model-specific package")
    if model is None:
        result.warn(
            "--model not specified; skipping model-specific package check. "
            "Pass --model {uni|conch|quilt_llava} to check the package for "
            "that model's environment."
        )
        return

    packages_to_check = MODEL_PACKAGE_CHECKS.get(model, [])
    if model == "quilt_llava" and quilt_import_name != "llava":
        packages_to_check = [quilt_import_name]

    all_ok = True
    for pkg in packages_to_check:
        try:
            importlib.import_module(pkg)
            result.ok(f"'{pkg}' importable (matches {model} adapter's requirement)")
        except ImportError:
            all_ok = False
            result.fail(
                f"'{pkg}' not importable. This environment does not appear "
                f"to have {model}'s dependencies installed. Activate the "
                f"correct conda environment (per project convention: "
                f"{model.replace('_', '-')}-env) or install the missing "
                f"package there."
            )

    if model == "conch" and all_ok:
        # The conch package is unambiguous (a genuinely dedicated pip
        # package), so a clean import here is a strong signal.
        pass
    if model == "quilt_llava" and not all_ok:
        result.warn(
            "If your Quilt-LLaVA install uses a top-level module name "
            "other than 'llava', re-run with --quilt-import-name <name> "
            "to check the correct package."
        )


def check_hf_authentication(result: CheckResult, model: Optional[str]) -> None:
    """
    Checks for a cached or environment-variable Hugging Face token, for
    UNI and CONCH only (both are gated models on Hugging Face per project
    notes). Deliberately does NOT make a network call to verify the token
    is *valid* -- only that one is present -- since this script is meant
    to run instantly and without external dependencies. An invalid or
    expired token will still surface clearly on the real extraction run.
    """
    print("\n[5/6] Hugging Face authentication (UNI / CONCH only)")
    if model not in HF_GATED_MODELS:
        result.ok(f"Model '{model}' does not require gated Hugging Face access; skipping.")
        return

    env_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if env_token:
        result.ok("HUGGING_FACE_HUB_TOKEN / HF_TOKEN environment variable is set.")
        return

    try:
        from huggingface_hub import HfFolder
        cached_token = HfFolder.get_token()
        if cached_token:
            result.ok("Cached Hugging Face token found (via `huggingface-cli login`).")
            return
    except ImportError:
        pass  # already reported as a missing package in check 4, if relevant

    result.fail(
        f"No Hugging Face token found for gated model '{model}'. Before "
        f"running extraction, either run `huggingface-cli login` or set "
        f"`export HUGGING_FACE_HUB_TOKEN=hf_xxx` in this environment. "
        f"You also need approved gated access to 'MahmoodLab/{model}' on "
        f"huggingface.co -- request access on the model page if you "
        f"haven't already."
    )


def check_tile_corpus(
    result: CheckResult, tiles_dir: Path, expected_tiles: int, expected_combos: int
) -> None:
    """
    Counts .png tiles under --tiles-dir and compares against the expected
    corpus size. Structure mirrors PerturbationPipeline.artifact_path():
        {tiles_dir}/{category}/p{id:02d}_{name}/{severity}/{sample_id}.png
    plus a separate baseline/{sample_id}.png per tile.
    """
    print("\n[6/6] Perturbation tile corpus completeness")
    if not tiles_dir.exists():
        result.fail(f"--tiles-dir '{tiles_dir}' does not exist.")
        return

    baseline_dir = tiles_dir / "baseline"
    baseline_count = len(list(baseline_dir.glob("*.png"))) if baseline_dir.exists() else 0

    # All perturbed combo PNGs live under category/pNN_name/severity/*.png,
    # i.e. three directory levels below tiles_dir, excluding the baseline
    # subtree entirely.
    perturbed_count = 0
    for png_path in tiles_dir.rglob("*.png"):
        if baseline_dir in png_path.parents:
            continue
        perturbed_count += 1

    result.ok(f"Baseline tiles found: {baseline_count} (expected: {expected_tiles})") \
        if baseline_count == expected_tiles else \
        result.warn(
            f"Baseline tile count is {baseline_count}, expected {expected_tiles}. "
            f"If this corpus was generated with a different "
            f"--num-slides-per-cohort, this may be expected -- pass "
            f"--expected-tiles to match your actual run."
        )

    if perturbed_count == expected_combos:
        result.ok(f"Perturbed combo tiles found: {perturbed_count} (matches expected {expected_combos})")
    else:
        diff = expected_combos - perturbed_count
        msg = (
            f"Perturbed combo tile count is {perturbed_count}, expected "
            f"{expected_combos} (18 perturbations x 3 severities x "
            f"{expected_tiles} tiles). Difference: {diff:+d}."
        )
        if perturbed_count < expected_combos:
            result.fail(
                msg + " This suggests the perturbation pipeline did not "
                "finish, or --tiles-dir points at the wrong directory. "
                "Cross-check against production_progress.csv before "
                "proceeding to extraction."
            )
        else:
            # More tiles than expected is not dangerous by itself (could be
            # a stale prior run's leftovers, or a corpus regenerated at a
            # larger --num-slides-per-cohort), but is worth a look.
            result.warn(msg + " More tiles than expected is not necessarily an error, but verify this is the corpus you intend to extract from.")

    # Cross-check against production_progress.csv if it's sitting alongside
    # tiles_dir or its parent, since that file is the ground-truth ledger
    # for this project's restart-resilient pipeline design.
    for candidate in [tiles_dir.parent / "production_progress.csv", tiles_dir / "production_progress.csv"]:
        if candidate.exists():
            result.ok(f"Found production_progress.csv at '{candidate}' for cross-reference (not parsed automatically; inspect manually if counts above look off).")
            break


def check_embeddings_output_dir(result: CheckResult, embeddings_dir: Path, model: Optional[str]) -> None:
    """Verifies --embeddings-dir exists (or can be created) and is writable, and flags stale outputs."""
    print("\n[Bonus] Output directory readiness")
    try:
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        result.ok(f"--embeddings-dir '{embeddings_dir}' exists and is writable.")
    except OSError as exc:
        result.fail(f"Cannot create/write --embeddings-dir '{embeddings_dir}': {exc}")
        return

    if model:
        expected_h5 = embeddings_dir / f"embeddings_{model}.h5"
        if expected_h5.exists():
            size_mb = expected_h5.stat().st_size / (1024 * 1024)
            result.warn(
                f"'{expected_h5.name}' already exists ({size_mb:.1f} MB). "
                f"If this is a stale or partial file from a previous run, "
                f"decide whether to delete it or confirm the script's "
                f"resume logic will handle it correctly before proceeding."
            )

    total, used, free = shutil.disk_usage(embeddings_dir)
    free_gb = free / (1024 ** 3)
    if free_gb < 5:
        result.warn(f"Only {free_gb:.1f} GB free at '{embeddings_dir}'. Embedding HDF5 files are usually small, but confirm this isn't a near-full volume.")
    else:
        result.ok(f"{free_gb:.1f} GB free at '{embeddings_dir}'.")


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["uni", "conch", "quilt_llava"], default=None,
        help="Which model this environment is meant to serve. Omit to run "
             "only the environment-agnostic checks.",
    )
    parser.add_argument("--tiles-dir", type=Path, default=Path("outputs/perturbed_tiles"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument(
        "--expected-tiles", type=int, default=DEFAULT_EXPECTED_TILES,
        help=f"Expected number of source tiles (default: {DEFAULT_EXPECTED_TILES}, "
             f"matching the confirmed production run).",
    )
    parser.add_argument(
        "--quilt-import-name", type=str, default="llava",
        help="Override the top-level module name used to check the "
             "Quilt-LLaVA / LLaVA codebase import, if your install differs.",
    )
    args = parser.parse_args()

    expected_combos = args.expected_tiles * COMBOS_PER_TILE

    print("=" * 78)
    print("PRE-FLIGHT CHECK: extract_embeddings_similarity.py")
    print("=" * 78)

    result = CheckResult()
    check_environment_identity(result, args.model)
    check_cross_cutting_packages(result)
    check_gpu_visibility(result)
    check_model_specific_package(result, args.model, args.quilt_import_name)
    check_hf_authentication(result, args.model)
    check_tile_corpus(result, args.tiles_dir, args.expected_tiles, expected_combos)
    check_embeddings_output_dir(result, args.embeddings_dir, args.model)

    print("\n" + "=" * 78)
    print(
        f"SUMMARY: {len(result.passed)} passed, {len(result.warnings)} "
        f"warning(s), {len(result.failures)} failure(s)"
    )
    print("=" * 78)

    if result.has_failures:
        print("\nOne or more checks FAILED. Resolve these before submitting the Slurm job:")
        for msg in result.failures:
            print(f"  - {msg}")
        return 1

    if result.warnings:
        print("\nAll critical checks passed, but review these warnings:")
        for msg in result.warnings:
            print(f"  - {msg}")

    print("\nReady to proceed with extraction in this environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
