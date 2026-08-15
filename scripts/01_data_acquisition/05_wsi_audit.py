#!/usr/bin/env python3
"""
wsi_audit.py

Audits native scan resolution/magnification metadata across both WSI cohorts
(ACC-Kenya and GDC/TCGA-HCMI controls) prior to tile extraction, using a
single shared inspection routine with two source-specific path resolvers.

Rationale for a single script (not two):
    The OpenSlide inspection logic (MPP lookup, pyramid level enumeration,
    20x/0.5 MPP level selection) is identical for both cohorts. Only the
    file-path resolution differs:
        - ACC:  <acc_root>/<acc_case_id>.svs
        - GDC:  <gdc_root>/<file_id>/<file_name>
    Keeping this in one script means one audit run, one output CSV, and
    a single 'source' column that lets Phase-2/tile-extraction code (and
    later, the resolution-matching analysis for H2) query both cohorts
    from the same table rather than reconciling two separate outputs.

Confirmed paths (Artemis / Sussex HCI, pn254):
    ACC:  /its/home/pn254/project2026/wsi_data/acc_data/<acc_case_id>.svs
    GDC:  /its/home/pn254/project2026/wsi_data/tcga_data/<file_id>/<file_name>
          (file_name already carries its own .svs extension in the
          manifest - do not append a second .svs)

Usage (Slurm batch job recommended — do NOT run interactively, per the
~1 minute Artemis interactive session limit):

    python wsi_audit.py \
        --manifest matched_controls_manifest_patched.tsv \
        --out audit_report.csv

    # --acc-root / --gdc-root default to the confirmed paths above and
    # only need to be passed if auditing from a different location
    # (e.g. a local copy for testing off-cluster).

Output columns:
    source              'ACC' or 'GDC'
    case_id              acc_case_id (ACC) or file_id (GDC)
    tissue_prefix         BP/BN/CP/CN/RP/TP parsed from acc_case_id
    file_path            resolved absolute path
    file_found            bool
    file_size_mb
    native_mpp_x          from OpenSlide metadata (None if missing)
    objective_power       from OpenSlide metadata (None if missing)
    level_count
    level_downsamples     semicolon-joined list
    best_level_for_20x    pyramid level index closest to 0.5 MPP
    achieved_mpp_at_20x   effective MPP at that level
    mpp_deviation         |achieved - target|, target = 0.5
    needs_software_resize  True if deviation exceeds tolerance
    status                OK / MISSING_FILE / MISSING_MPP / OPEN_ERROR
    notes
"""

import argparse
import csv
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import openslide

TARGET_MPP = 0.5      # 20x objective, matches UNI / CONCH / Prov-GigaPath /
                       # (approximated for) Quilt-LLaVA extraction standard
MPP_TOLERANCE = 0.05   # flag slides needing software resize beyond this

# Confirmed Artemis (Sussex HCI) paths - used as CLI defaults so the script
# can be invoked with just --manifest and --out in the common case.
DEFAULT_ACC_ROOT = Path("/its/home/pn254/project2026/wsi_data/acc_data")
DEFAULT_GDC_ROOT = Path("/its/home/pn254/project2026/wsi_data/tcga_data")


@dataclass
class AuditRow:
    source: str
    case_id: str
    tissue_prefix: str
    file_path: str
    file_found: bool
    file_size_mb: Optional[float] = None
    native_mpp_x: Optional[float] = None
    objective_power: Optional[str] = None
    level_count: Optional[int] = None
    level_downsamples: Optional[str] = None
    best_level_for_20x: Optional[int] = None
    achieved_mpp_at_20x: Optional[float] = None
    mpp_deviation: Optional[float] = None
    needs_software_resize: Optional[bool] = None
    status: str = "UNKNOWN"
    notes: str = ""


def parse_tissue_prefix(case_id: str) -> str:
    """Extract BP/BN/CP/CN/RP/TP prefix from an acc_case_id like 'BP-001'."""
    return case_id.split("-")[0] if "-" in case_id else ""


def resolve_acc_path(acc_root: Path, acc_case_id: str) -> Path:
    """
    ACC slides sit directly (flat, no subfolders - confirmed) in
    acc_root, named after acc_case_id: <acc_root>/<acc_case_id>.svs
    """
    direct = acc_root / f"{acc_case_id}.svs"
    if direct.exists():
        return direct
    # Fallback: case-insensitive / alternate-extension search, kept as a
    # safety net in case a small number of files deviate from the
    # confirmed naming convention (e.g. differing case or extension).
    matches = list(acc_root.glob(f"{acc_case_id}.*"))
    return matches[0] if matches else direct


def resolve_gdc_path(gdc_root: Path, file_id: str, file_name: str) -> Path:
    """
    GDC controls sit in a subfolder named after the GDC file_id UUID.

    NOTE: file_name in the manifest already carries its own .svs extension
    (e.g. 'TCGA-AO-A124-01Z-00-DX1.E3C7B017-....svs'), so it is used as-is.
    Do not append '.svs' here - doing so would produce a non-existent
    '....svs.svs' path.
    """
    return gdc_root / file_id / file_name


def inspect_slide(path: Path) -> dict:
    """
    Runs the shared OpenSlide inspection: native MPP, objective power,
    pyramid levels, and the level nearest to the 20x/0.5 MPP target.
    Identical logic regardless of which cohort the slide belongs to.
    """
    result = {
        "file_size_mb": None,
        "native_mpp_x": None,
        "objective_power": None,
        "level_count": None,
        "level_downsamples": None,
        "best_level_for_20x": None,
        "achieved_mpp_at_20x": None,
        "mpp_deviation": None,
        "needs_software_resize": None,
        "status": "OK",
        "notes": "",
    }

    result["file_size_mb"] = round(path.stat().st_size / (1024 * 1024), 2)

    try:
        slide = openslide.OpenSlide(str(path))
    except Exception as exc:  # noqa: BLE001 - want to log and continue, not crash the batch
        result["status"] = "OPEN_ERROR"
        result["notes"] = f"OpenSlide failed to open file: {exc}"
        return result

    mpp_x_raw = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    objective = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    result["objective_power"] = objective

    try:
        mpp_x = float(mpp_x_raw) if mpp_x_raw is not None else None
    except ValueError:
        mpp_x = None

    result["native_mpp_x"] = mpp_x
    result["level_count"] = slide.level_count
    result["level_downsamples"] = ";".join(
        f"{d:.4f}" for d in slide.level_downsamples
    )

    if mpp_x is None or mpp_x <= 0:
        result["status"] = "MISSING_MPP"
        result["notes"] = (
            "No usable MPP-X metadata; cannot select a 20x-equivalent "
            "pyramid level automatically. Flag for manual inspection "
            "(check objective_power field as a fallback, or inspect "
            "scanner export logs)."
        )
        slide.close()
        return result

    best_level, best_diff = 0, float("inf")
    for level in range(slide.level_count):
        downsample = slide.level_downsamples[level]
        effective_mpp = mpp_x * downsample
        diff = abs(effective_mpp - TARGET_MPP)
        if diff < best_diff:
            best_diff, best_level = diff, level

    achieved_mpp = mpp_x * slide.level_downsamples[best_level]
    result["best_level_for_20x"] = best_level
    result["achieved_mpp_at_20x"] = round(achieved_mpp, 4)
    result["mpp_deviation"] = round(best_diff, 4)
    result["needs_software_resize"] = best_diff > MPP_TOLERANCE

    if result["needs_software_resize"]:
        result["notes"] = (
            f"No native pyramid layer within tolerance of {TARGET_MPP} MPP "
            f"(closest available: {achieved_mpp:.3f} MPP). Extraction from "
            "this slide will require an additional resize step, unlike "
            "slides with a native 20x-equivalent layer."
        )

    slide.close()
    return result


def audit_acc(manifest_rows: list[dict], acc_root: Path) -> list[AuditRow]:
    """
    Audits the ACC cohort. Every row in the manifest has an acc_case_id
    (including the 11 NO_MATCH TP cases), so this covers all 201 cases
    regardless of GDC match status.
    """
    rows = []
    for rec in manifest_rows:
        case_id = rec["acc_case_id"]
        prefix = parse_tissue_prefix(case_id)
        path = resolve_acc_path(acc_root, case_id)

        row = AuditRow(
            source="ACC",
            case_id=case_id,
            tissue_prefix=prefix,
            file_path=str(path),
            file_found=path.exists(),
        )
        if not row.file_found:
            row.status = "MISSING_FILE"
            row.notes = f"No ACC file found for {case_id} at expected path."
            rows.append(row)
            continue

        inspection = inspect_slide(path)
        for key, val in inspection.items():
            setattr(row, key, val)
        rows.append(row)
    return rows


def audit_gdc(manifest_rows: list[dict], gdc_root: Path) -> list[AuditRow]:
    """
    Audits the GDC/TCGA-HCMI control cohort. Skips the 11 NO_MATCH rows
    (empty file_id/file_name) rather than attempting a lookup for a file
    that was never allocated - this is expected, not an error condition.
    """
    rows = []
    for rec in manifest_rows:
        file_id = rec.get("file_id", "").strip()
        file_name = rec.get("file_name", "").strip()
        acc_case_id = rec["acc_case_id"]

        if not file_id or not file_name:
            # Expected for TP-016..TP-026 - documented dataset limitation,
            # not a pipeline fault. Skip silently at the row level; the
            # overall summary printed at the end reports the count.
            continue

        prefix = parse_tissue_prefix(acc_case_id)
        path = resolve_gdc_path(gdc_root, file_id, file_name)

        row = AuditRow(
            source="GDC",
            case_id=file_id,
            tissue_prefix=prefix,
            file_path=str(path),
            file_found=path.exists(),
        )
        if not row.file_found:
            row.status = "MISSING_FILE"
            row.notes = f"No GDC file found for {file_id} at expected path."
            rows.append(row)
            continue

        inspection = inspect_slide(path)
        for key, val in inspection.items():
            setattr(row, key, val)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--acc-root",
        type=Path,
        default=DEFAULT_ACC_ROOT,
        help=f"Default: {DEFAULT_ACC_ROOT}",
    )
    parser.add_argument(
        "--gdc-root",
        type=Path,
        default=DEFAULT_GDC_ROOT,
        help=f"Default: {DEFAULT_GDC_ROOT}",
    )
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    print(f"ACC root: {args.acc_root}")
    print(f"GDC root: {args.gdc_root}")

    with open(args.manifest, newline="", encoding="utf-8") as fh:
        manifest_rows = list(csv.DictReader(fh, delimiter="\t"))

    print(f"Loaded {len(manifest_rows)} manifest rows.")

    acc_rows = audit_acc(manifest_rows, args.acc_root)
    gdc_rows = audit_gdc(manifest_rows, args.gdc_root)

    no_match_count = sum(
        1 for r in manifest_rows if not r.get("file_id", "").strip()
    )

    all_rows = acc_rows + gdc_rows
    fieldnames = list(asdict(all_rows[0]).keys()) if all_rows else []

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))

    # Console summary - useful for the Slurm job's stdout log
    def summarize(rows: list[AuditRow], label: str):
        missing = sum(1 for r in rows if r.status == "MISSING_FILE")
        missing_mpp = sum(1 for r in rows if r.status == "MISSING_MPP")
        needs_resize = sum(1 for r in rows if r.needs_software_resize)
        ok = sum(1 for r in rows if r.status == "OK")
        print(
            f"[{label}] total={len(rows)} ok={ok} "
            f"missing_file={missing} missing_mpp={missing_mpp} "
            f"needs_software_resize={needs_resize}"
        )

    print(f"\nAudit written to {args.out}")
    summarize(acc_rows, "ACC")
    summarize(gdc_rows, "GDC")
    print(
        f"[GDC] skipped {no_match_count} NO_MATCH rows "
        "(TP-016..TP-026 - documented TCGA-HNSC coverage ceiling)"
    )


if __name__ == "__main__":
    main()
