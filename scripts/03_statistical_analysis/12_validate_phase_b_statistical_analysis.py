#!/usr/bin/env python3
"""
validate_phase_b_statistical_analysis.py

A single, self-contained script that reproduces every validation step
performed during the session that discovered and fixed a real bug in
two_sample_test() within phase_b_statistical_analysis.py.

BACKGROUND (for a reviewer who wasn't in the original session)
-----------------------------------------------------------------
Perturbations 3, 17, and 18 at moderate severity are documented identity
transforms (multiplier 1.0) in this project's perturbation framework,
meaning cosine similarity is EXACTLY 1.0 for every tile in both the ACC
and GDC cohorts for those cells -- a perfectly constant (zero-variance)
sample in both groups. Running the real Phase B similarity data through
phase_b_statistical_analysis.py surfaced a `scipy` RuntimeWarning
("invalid value encountered in scalar divide") at exactly these cells,
traced to Shapiro-Wilk and Mann-Whitney U both being mathematically
undefined (0/0) for a constant sample. The unhandled case left p_value as
NaN, which then silently propagated into the Bonferroni-corrected
p-value for that cell -- not incorrect in its ultimate effect
(NaN-comparisons-are-False happened to produce the right
significant_bonferroni=False answer) but arrived at by accident rather
than by an explicit, well-defined code path.

two_sample_test() was patched to detect this zero-variance case
explicitly, before calling either statistical test, and return a
well-defined, clearly labelled result instead.

WHAT THIS SCRIPT CHECKS, IN ORDER
-----------------------------------
  1. Syntax check on phase_b_statistical_analysis.py
  2. Unit test, Case 1: identical constants (the real p17/p18 scenario) --
     confirms p_value=1.0, no warning raised, correct labelling
  3. Unit test, Case 2: distinct constants (the other zero-variance
     sub-case, not present in the real data but handled for completeness)
  4. Unit test, Case 3 (regression check): ordinary, non-degenerate data
     still produces an unaffected, standard result -- confirms the fix
     did not change behaviour for the normal code path
  5. Full pipeline run against synthetic data, for BOTH model-set
     configurations, confirming: (a) the run completes with exit code 0,
     (b) the scipy "invalid value encountered" warning does NOT appear
     anywhere in the output, and (c) the separate, already-documented,
     expected statsmodels ConvergenceWarning (H4's thin-3-group-design
     limitation) is still present and does not cause a crash -- i.e. this
     fix did not paper over or suppress a genuinely different warning it
     was never meant to address
  6. Restart resilience, re-confirmed after the fix: a second identical
     run against the same checkpoint directory resumes all work with
     zero recomputation

USAGE
-----
    python validate_phase_b_statistical_analysis.py
    python validate_phase_b_statistical_analysis.py --script /path/to/phase_b_statistical_analysis.py
    python validate_phase_b_statistical_analysis.py --keep-artifacts   # inspect temp files afterwards

Exit code 0 if every check passes; 1 otherwise. Requires only the same
dependencies as phase_b_statistical_analysis.py itself (numpy, pandas,
scipy, statsmodels) -- no GPU, no model-specific packages.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# =============================================================================
# Small result-tracking helper, matching this project's established
# preflight_check.py / check_phase_b_dependencies.py pattern
# =============================================================================

class CheckResult:
    def __init__(self) -> None:
        self.passed: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

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


# =============================================================================
# Check 1: syntax
# =============================================================================

def check_syntax(result: CheckResult, script_path: Path) -> None:
    print("\n[1/6] Syntax check")
    try:
        ast.parse(script_path.read_text(encoding="utf-8"))
        result.ok(f"{script_path.name} parses without syntax errors.")
    except SyntaxError as exc:
        result.fail(f"{script_path.name} has a syntax error: {exc}")


def _import_target_module(script_path: Path):
    """Imports phase_b_statistical_analysis.py by file path, regardless of cwd."""
    spec = importlib.util.spec_from_file_location("phase_b_statistical_analysis", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


# =============================================================================
# Checks 2-4: two_sample_test() unit tests, run with warnings captured
# =============================================================================

def check_zero_variance_identical(result: CheckResult, mod) -> None:
    print("\n[2/6] Unit test: identical constants (real p17/p18 scenario)")
    a = np.ones(2010)  # ACC-cohort-sized sample of constant cosine similarity 1.0
    b = np.ones(1900)  # GDC-cohort-sized sample of constant cosine similarity 1.0

    # NOTE on reliability of warning capture: warnings.catch_warnings() only
    # intercepts warnings raised via Python's `warnings` module. Depending
    # on the installed numpy/scipy version combination, the underlying
    # divide-by-zero can instead surface as a NumPy C-level floating-point
    # error state, which this mechanism does not always catch (confirmed
    # directly: this sub-check still reported "no warning" even when
    # tested against a deliberately un-fixed copy of two_sample_test()).
    # It is kept here as a secondary, best-effort signal only. The
    # authoritative check is the one immediately below, which compares the
    # actual returned p_value and test_used against what a correct
    # implementation must produce -- this is what genuinely discriminates
    # a fixed implementation from a broken one, confirmed via a negative
    # control (running this validator against the pre-fix code correctly
    # produced two [FAIL] results here).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mod.two_sample_test(a, b)

    invalid_value_warnings = [w for w in caught if "invalid value encountered" in str(w.message)]
    if invalid_value_warnings:
        result.fail(f"scipy 'invalid value encountered' warning still fires for identical constants: {invalid_value_warnings[0].message}")
    else:
        result.ok("No scipy warning raised for identical-constant input (best-effort check; see note above -- the functional check below is authoritative).")

    if out["p_value"] == 1.0 and out["test_used"] == "zero_variance_identical":
        result.ok(f"Correct, well-defined result: p_value=1.0, test_used='{out['test_used']}'.")
    else:
        result.fail(f"Unexpected result for identical constants: p_value={out['p_value']}, test_used={out['test_used']}")

    if not (isinstance(out["p_value"], float) and out["p_value"] != out["p_value"]):  # NaN check without numpy dependency here
        result.ok("p_value is not NaN (confirms the original propagation-into-Bonferroni bug is closed).")
    else:
        result.fail("p_value is still NaN -- the original bug is NOT fixed.")


def check_zero_variance_distinct(result: CheckResult, mod) -> None:
    print("\n[3/6] Unit test: distinct constants (the other zero-variance sub-case)")
    a = np.full(50, 0.5)
    b = np.full(50, 0.9)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = mod.two_sample_test(a, b)

    invalid_value_warnings = [w for w in caught if "invalid value encountered" in str(w.message)]
    if invalid_value_warnings:
        result.fail(f"scipy warning still fires for distinct constants: {invalid_value_warnings[0].message}")
    else:
        result.ok("No scipy warning raised for distinct-constant input.")

    if out["p_value"] == 0.0 and out["test_used"] == "zero_variance_distinct_constants":
        result.ok(f"Correct, well-defined result: p_value=0.0, test_used='{out['test_used']}'.")
    else:
        result.fail(f"Unexpected result for distinct constants: p_value={out['p_value']}, test_used={out['test_used']}")


def check_regression_normal_data(result: CheckResult, mod) -> None:
    print("\n[4/6] Regression check: ordinary (non-degenerate) data unaffected by the fix")
    rng = np.random.default_rng(1)
    a = rng.normal(0.9, 0.03, 200)
    b = rng.normal(0.85, 0.03, 180)
    out = mod.two_sample_test(a, b)

    if out["test_used"] == "welch_t_test" and out["p_value"] < 0.05 and out["effect_size"] == "large":
        result.ok(f"Standard code path unaffected: test_used='welch_t_test', p_value={out['p_value']:.3e}, effect_size='large'.")
    else:
        result.fail(f"Unexpected result on ordinary data (possible regression): {out}")


# =============================================================================
# Check 5: full pipeline run against synthetic data, both model sets
# =============================================================================

def _write_synthetic_data(tmp_dir: Path) -> tuple[Path, Path]:
    """
    Generates the same synthetic cosine_similarity.csv / cka_summary.csv
    used throughout this project's validation work, matching the real
    schema, and DELIBERATELY includes rows reproducing the exact p17/p18
    identity-transform scenario (constant value 1.0 in both cohorts) that
    originally surfaced the bug -- a synthetic-only regression test would
    be worthless here if it didn't specifically recreate this condition.
    """
    rng = np.random.default_rng(7)
    models = ["uni", "conch", "quilt_llava", "gigapath_tile"]
    perturbations = list(range(1, 19))
    severities = ["mild", "moderate", "severe"]
    n_acc, n_gdc = 40, 35
    identity_transform_perturbations = {3, 17, 18}  # matches the real framework

    rows = []
    for model in models:
        model_offset = {"uni": 0.0, "conch": 0.02, "quilt_llava": -0.05, "gigapath_tile": 0.01}[model]
        demographic_gap = {"uni": -0.01, "conch": -0.03, "quilt_llava": -0.08, "gigapath_tile": -0.015}[model]
        for pid in perturbations:
            for severity in severities:
                is_identity = severity == "moderate" and pid in identity_transform_perturbations
                for i in range(n_acc):
                    val = 1.0 if is_identity else float(np.clip(
                        0.9 + model_offset + demographic_gap + rng.normal(0, 0.03), 0, 1))
                    rows.append([model, f"ACC_sample_{i}", "ACC", pid, severity, round(val, 6)])
                for i in range(n_gdc):
                    val = 1.0 if is_identity else float(np.clip(
                        0.9 + model_offset + rng.normal(0, 0.03), 0, 1))
                    rows.append([model, f"GDC_sample_{i}", "GDC", pid, severity, round(val, 6)])

    cosine_path = tmp_dir / "synthetic_cosine_similarity.csv"
    pd.DataFrame(rows, columns=["model", "sample_id", "cohort", "perturbation_id", "severity", "cosine_similarity"]).to_csv(cosine_path, index=False)

    cka_rows = []
    for model in models:
        for pid in perturbations:
            for severity in severities:
                for cohort, n in [("POOLED", n_acc + n_gdc), ("ACC", n_acc), ("GDC", n_gdc)]:
                    cka_rows.append([model, pid, severity, cohort, n, round(float(rng.uniform(0.6, 0.95)), 6)])
    cka_path = tmp_dir / "synthetic_cka_summary.csv"
    pd.DataFrame(cka_rows, columns=["model", "perturbation_id", "severity", "cohort", "n_tiles", "linear_cka"]).to_csv(cka_path, index=False)

    return cosine_path, cka_path


def _run_pipeline(script_path: Path, cosine_csv: Path, cka_csv: Path, output_dir: Path, checkpoint_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(script_path),
        "--cosine-csv", str(cosine_csv), "--cka-csv", str(cka_csv),
        "--model-set", "all", "--output-dir", str(output_dir), "--checkpoint-dir", str(checkpoint_dir),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def check_full_pipeline_run(result: CheckResult, script_path: Path, tmp_dir: Path) -> Path:
    print("\n[5/6] Full pipeline run against synthetic data (both model-set configurations)")
    cosine_csv, cka_csv = _write_synthetic_data(tmp_dir)
    output_dir = tmp_dir / "output"
    checkpoint_dir = tmp_dir / "checkpoints"

    proc = _run_pipeline(script_path, cosine_csv, cka_csv, output_dir, checkpoint_dir)
    combined_output = proc.stdout + proc.stderr

    if proc.returncode == 0:
        result.ok("Pipeline run completed with exit code 0 for both model sets.")
    else:
        result.fail(f"Pipeline run exited with code {proc.returncode}. See output below.\n{combined_output}")

    if "invalid value encountered" in combined_output:
        result.fail("The scipy 'invalid value encountered' warning is STILL present in a full pipeline run -- the fix did not take effect end-to-end.")
    else:
        result.ok("No scipy 'invalid value encountered' warning anywhere in the full pipeline output.")

    # The H4 ConvergenceWarning is a SEPARATE, already-documented, expected
    # limitation (thin 3-group design) -- its continued presence here is
    # correct and should NOT be treated as a failure. Checking for it
    # confirms this fix didn't accidentally suppress warnings generally.
    if "ConvergenceWarning" in combined_output:
        result.ok("H4's known ConvergenceWarning (thin 3-group design) is still present and handled gracefully, as expected -- unrelated to this fix, and correctly not suppressed by it.")
    else:
        result.warn("H4's expected ConvergenceWarning did not appear in this synthetic run. Not necessarily a problem (synthetic data may not trigger it every time), but worth a second look if unexpected.")

    for model_set in ["option_a", "option_b"]:
        for fname in ["h2_demographic_main_effect.csv", "h2_demographic_main_effect_cka.csv", "pairwise_comparisons_bonferroni.csv"]:
            fpath = output_dir / model_set / fname
            if fpath.exists():
                result.ok(f"{model_set}/{fname} was written.")
            else:
                result.fail(f"{model_set}/{fname} is MISSING from output.")

    return checkpoint_dir if proc.returncode == 0 else None  # type: ignore[return-value]


# =============================================================================
# Check 6: restart resilience, re-confirmed after the fix
# =============================================================================

def check_restart_resilience(result: CheckResult, script_path: Path, tmp_dir: Path, checkpoint_dir: Path) -> None:
    print("\n[6/6] Restart resilience: re-running against the same checkpoint directory")
    if checkpoint_dir is None:
        result.fail("Skipped: the full pipeline run in step 5 did not complete, so this check cannot run.")
        return

    cosine_csv = tmp_dir / "synthetic_cosine_similarity.csv"
    cka_csv = tmp_dir / "synthetic_cka_summary.csv"
    output_dir = tmp_dir / "output"

    proc = _run_pipeline(script_path, cosine_csv, cka_csv, output_dir, checkpoint_dir)
    combined_output = proc.stdout + proc.stderr

    if proc.returncode != 0:
        result.fail(f"Re-run failed with exit code {proc.returncode}.")
        return

    newly_computed_lines = [line for line in combined_output.splitlines() if "newly computed" in line]
    all_zero = all("(0 newly computed" in line for line in newly_computed_lines)

    if newly_computed_lines and all_zero:
        result.ok(f"All {len(newly_computed_lines)} checkpointed stages resumed with 0 newly computed work on re-run.")
    elif newly_computed_lines:
        offending = [line for line in newly_computed_lines if "(0 newly computed" not in line]
        result.fail(f"Some stages recomputed work unexpectedly on a clean re-run: {offending}")
    else:
        result.fail("No 'newly computed' log lines found at all -- unable to confirm restart resilience.")


# =============================================================================
# Entry point
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--script", type=Path,
        default=Path(__file__).parent / "phase_b_statistical_analysis.py",
        help="Path to phase_b_statistical_analysis.py to validate (default: alongside this script).",
    )
    parser.add_argument(
        "--keep-artifacts", action="store_true",
        help="Do not delete the temporary synthetic data / output / checkpoint directory afterwards.",
    )
    args = parser.parse_args()

    if not args.script.exists():
        print(f"ERROR: --script path does not exist: {args.script}", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"VALIDATING: {args.script}")
    print("=" * 78)

    result = CheckResult()
    check_syntax(result, args.script)

    mod = _import_target_module(args.script)
    check_zero_variance_identical(result, mod)
    check_zero_variance_distinct(result, mod)
    check_regression_normal_data(result, mod)

    tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="phase_b_validation_")
    tmp_dir = Path(tmp_dir_ctx.name)
    try:
        checkpoint_dir = check_full_pipeline_run(result, args.script, tmp_dir)
        check_restart_resilience(result, args.script, tmp_dir, checkpoint_dir)
    finally:
        if args.keep_artifacts:
            tmp_dir_ctx._finalizer.detach()  # type: ignore[attr-defined]  # prevent auto-cleanup
            print(f"\n(Artifacts kept at: {tmp_dir})")
        else:
            tmp_dir_ctx.cleanup()

    print("\n" + "=" * 78)
    print(f"SUMMARY: {len(result.passed)} passed, {len(result.warnings)} warning(s), {len(result.failures)} failure(s)")
    print("=" * 78)

    if result.has_failures:
        print("\nFAILED. Review the [FAIL] lines above before trusting this script's output.")
        return 1

    print("\nAll checks passed. phase_b_statistical_analysis.py's zero-variance fix is confirmed working correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
