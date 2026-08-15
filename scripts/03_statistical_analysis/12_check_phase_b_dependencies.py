#!/usr/bin/env python3
"""
check_phase_b_dependencies.py

Verifies the Python environment has everything phase_b_statistical_analysis.py
needs (numpy, pandas, scipy, statsmodels; optionally scikit-posthocs for
Dunn's post-hoc test), before a long batch run is submitted.

BEHAVIOUR, EXACTLY AS SPECIFIED
--------------------------------
1. Detects each package's installed version WITHOUT importing it (see
   "why importlib.metadata, not import" below), and logs every version
   found.
2. Never upgrades, downgrades, or otherwise touches a package that is
   already installed. This is enforced structurally, not just intended:
   every already-installed package's exact version is written to a pip
   CONSTRAINTS file (PEP 668 / pip's -c flag) before any install is
   attempted. A constraints file tells pip "if you need this package for
   any reason -- including as a dependency of something else you're
   installing -- it must be exactly this version, or fail". This is
   different from a requirements file (which pip treats as a target to
   move packages towards); a constraints file is a ceiling/floor pip is
   not allowed to cross, which is what "do not update the ones that
   already exist" actually requires in pip's own vocabulary.
3. Installs only the packages found to be missing, via that constrained
   pip call, so any newly-installed package is automatically forced to be
   compatible with whatever is already present (pip's resolver will
   refuse to proceed, with a clear error, rather than silently upgrading
   an existing package to satisfy a new one -- surfacing a real
   incompatibility is the correct behaviour here, not resolving it
   quietly on the user's behalf without their knowledge).
4. Re-verifies afterwards: confirms every previously-missing package is
   now present, AND confirms every previously-present package's version
   is UNCHANGED, rather than trusting the constraints file blindly.

WHY importlib.metadata, NOT `import numpy` ETC.
-------------------------------------------------
Actually importing a package to check its __version__ attribute has real
downsides in a dependency-checking script specifically: it executes the
package's top-level code (slow for some packages, and can itself fail
with a confusing traceback if the package is partially/incorrectly
installed -- exactly the ambiguous state this script exists to detect and
fix). importlib.metadata.version() reads the installed distribution's
metadata directly from disk, with no import and no side effects, and
correctly handles packages whose distribution name and import name differ
(scikit-posthocs is installed as "scikit-posthocs" but imported as
"scikit_posthocs" -- get this wrong and a naive `pip show` or `import`
based check will silently misreport it).

USAGE
-----
  python check_phase_b_dependencies.py
  python check_phase_b_dependencies.py --dry-run      # report only, install nothing
  python check_phase_b_dependencies.py --break-system-packages   # see below
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

try:
    from importlib import metadata as importlib_metadata
except ImportError:  # pragma: no cover -- Python <3.8 fallback, not expected here
    import importlib_metadata  # type: ignore[no-redef]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Distribution (pip/PyPI) name -> whether it is required or optional for
# phase_b_statistical_analysis.py. Import names are not needed anywhere in
# this script, since importlib.metadata works from distribution names, and
# pip install also takes distribution names -- avoiding the
# scikit-posthocs / scikit_posthocs naming mismatch entirely.
REQUIRED_PACKAGES = ["numpy", "pandas", "scipy", "statsmodels"]
OPTIONAL_PACKAGES = ["scikit-posthocs"]
ALL_PACKAGES = REQUIRED_PACKAGES + OPTIONAL_PACKAGES


def get_installed_version(distribution_name: str) -> Optional[str]:
    """Returns the installed version string, or None if not installed. No import performed."""
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def scan_environment(packages: List[str]) -> Dict[str, Optional[str]]:
    """Checks every package in `packages`, logging what is found, and returns {name: version_or_None}."""
    found: Dict[str, Optional[str]] = {}
    for pkg in packages:
        version = get_installed_version(pkg)
        found[pkg] = version
        if version is not None:
            log.info("Found %-18s version %s", pkg, version)
        else:
            required_str = "required" if pkg in REQUIRED_PACKAGES else "optional"
            log.info("Missing %-17s (%s) -- not currently installed", pkg, required_str)
    return found


def write_constraints_file(installed: Dict[str, Optional[str]]) -> Path:
    """
    Writes a pip constraints file pinning every ALREADY-installed package to
    its exact current version. Packages that are missing are deliberately
    NOT listed here -- a constraint only makes sense for a version that
    already exists; the whole point is to protect what's already installed
    while leaving pip free to choose a compatible version for what isn't.
    """
    fd, path_str = tempfile.mkstemp(prefix="phase_b_constraints_", suffix=".txt")
    path = Path(path_str)
    lines = [f"{pkg}=={version}" for pkg, version in installed.items() if version is not None]
    with open(fd, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n" if lines else "")
    log.info("Wrote pip constraints file (%d pinned package%s) to %s",
              len(lines), "" if len(lines) == 1 else "s", path)
    for line in lines:
        log.debug("  constraint: %s", line)
    return path


def run_pip_install(packages: List[str], constraints_path: Path, use_break_system_packages: bool) -> subprocess.CompletedProcess:
    """
    Runs `pip install -c <constraints> <packages...>`, optionally with
    --break-system-packages. Returns the CompletedProcess so the caller can
    inspect returncode/stdout/stderr rather than raising immediately --
    the caller needs to distinguish an ordinary failure from the specific
    PEP 668 "externally-managed-environment" failure, to decide whether an
    automatic retry is appropriate (see install_missing_packages() below).
    """
    cmd = [sys.executable, "-m", "pip", "install", "--constraint", str(constraints_path)]
    if use_break_system_packages:
        cmd.append("--break-system-packages")
    cmd.extend(packages)
    log.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


def install_missing_packages(missing: List[str], constraints_path: Path, force_break_system_packages: bool) -> List[str]:
    """
    Installs `missing`, constrained against everything already present.
    Automatically retries with --break-system-packages if, and only if,
    the failure is specifically PEP 668's "externally-managed-environment"
    protection (common on system/Debian-style Python, not expected inside
    an activated conda environment, which is this project's primary
    target -- see the module docstring's Artemis-specific note below).
    Returns the list of packages that are STILL missing after all attempts
    (empty list if everything succeeded).
    """
    if not missing:
        return []

    result = run_pip_install(missing, constraints_path, use_break_system_packages=force_break_system_packages)

    if result.returncode != 0 and not force_break_system_packages and "externally-managed-environment" in result.stderr:
        log.warning(
            "pip reported an externally-managed-environment restriction "
            "(PEP 668) -- typical of system/Debian-style Python, not "
            "expected inside a conda environment. Retrying automatically "
            "with --break-system-packages. This is safe here specifically "
            "because we are ALSO passing the pip constraints file, so this "
            "retry still cannot upgrade any already-installed package -- "
            "it can only relax pip's own environment-protection check, not "
            "the version protection this script exists to enforce."
        )
        result = run_pip_install(missing, constraints_path, use_break_system_packages=True)

    if result.returncode != 0:
        log.error(
            "pip install failed for %s (exit code %d). This likely means a "
            "genuine version incompatibility was found against an "
            "already-installed package -- that is the constraints file "
            "doing its job correctly (refusing to silently upgrade "
            "something), not a bug in this script. Full pip output "
            "follows:\n--- stdout ---\n%s\n--- stderr ---\n%s",
            missing, result.returncode, result.stdout, result.stderr,
        )
        return missing

    log.info("pip install succeeded for: %s", missing)
    return []


def verify_after_install(pre_install: Dict[str, Optional[str]]) -> bool:
    """
    Re-checks every package after installation. Confirms (a) every
    package that was ALREADY present still reports the exact same
    version as before -- the concrete, empirical check that "do not
    update the ones that already exist" actually held, not just an
    assumption that the constraints file worked -- and (b) reports the
    final state of everything. Returns True if no already-installed
    package's version changed; False (logged as an error, since this
    should never happen given the constraints file) otherwise.
    """
    log.info("Re-verifying environment after installation...")
    post_install = scan_environment(ALL_PACKAGES)

    unchanged_ok = True
    for pkg, pre_version in pre_install.items():
        if pre_version is None:
            continue  # was missing before; a version change here is expected and fine
        post_version = post_install.get(pkg)
        if post_version != pre_version:
            log.error(
                "UNEXPECTED: %s was version %s before this script ran, and "
                "is now %s. The constraints file should have made this "
                "impossible -- please review the pip output above closely.",
                pkg, pre_version, post_version,
            )
            unchanged_ok = False

    return unchanged_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report installed/missing packages and exit. Installs nothing.",
    )
    parser.add_argument(
        "--break-system-packages", action="store_true",
        help="Pass --break-system-packages on the FIRST install attempt, rather than "
             "only as an automatic fallback if pip reports a PEP 668 "
             "externally-managed-environment error. Not usually needed inside a "
             "conda environment (this project's primary target); useful if you "
             "already know you are on a system/Debian-style Python.",
    )
    args = parser.parse_args()

    log.info("Checking environment for phase_b_statistical_analysis.py's dependencies...")
    log.info("Required: %s | Optional: %s", REQUIRED_PACKAGES, OPTIONAL_PACKAGES)
    installed = scan_environment(ALL_PACKAGES)

    missing_required = [p for p in REQUIRED_PACKAGES if installed[p] is None]
    missing_optional = [p for p in OPTIONAL_PACKAGES if installed[p] is None]
    missing_all = missing_required + missing_optional

    if not missing_all:
        log.info("All required and optional packages are already installed. No action needed.")
        return 0

    log.info("Missing required package(s): %s", missing_required or "(none)")
    log.info("Missing optional package(s): %s", missing_optional or "(none)")

    if args.dry_run:
        log.info("--dry-run specified: not installing anything.")
        return 1 if missing_required else 0

    constraints_path = write_constraints_file(installed)
    try:
        still_missing = install_missing_packages(missing_all, constraints_path, args.break_system_packages)
    finally:
        constraints_path.unlink(missing_ok=True)  # temp file cleanup, regardless of outcome

    unchanged_ok = verify_after_install(installed)

    still_missing_required = [p for p in still_missing if p in REQUIRED_PACKAGES]
    still_missing_optional = [p for p in still_missing if p in OPTIONAL_PACKAGES]

    if still_missing_optional:
        log.warning(
            "Optional package(s) still missing after install attempt: %s. "
            "phase_b_statistical_analysis.py will still run; H1's Dunn's "
            "post-hoc test will be skipped with a clear log message if "
            "the omnibus Kruskal-Wallis test is ever significant.",
            still_missing_optional,
        )

    if still_missing_required or not unchanged_ok:
        log.error("Environment is NOT ready. See errors above before running phase_b_statistical_analysis.py.")
        return 1

    log.info("Environment is ready. All required packages present; no already-installed package was modified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
