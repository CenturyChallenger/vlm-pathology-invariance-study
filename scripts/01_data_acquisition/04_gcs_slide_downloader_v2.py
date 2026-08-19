#!/usr/bin/env python3
"""
gcs_slide_downloader.py  -- Phase 2 of the two-phase download pipeline.

PURPOSE
-------
Download TCGA SVS whole-slide images from the ISB-CGC open-access Google
Cloud Storage bucket, using the manifest TSV produced by gdc_match_manifest.py.

NO CREDENTIALS REQUIRED.  The GCS bucket gdc-tcga-phs000178-open is a
public bucket with allUsers read access, maintained by the NCI Institute for
Systems Biology Cancer Gateway in the Cloud (ISB-CGC).  Files are fetched
over plain HTTPS -- no gsutil, no Google SDK, no GDC token.

Reference:
  https://isb-cancer-genomics-cloud.readthedocs.io/en/latest/sections/data/TCGA-images.html
  "Over 30,000 TCGA tissue slide images in SVS format are available in GCS,
   in the open-access bucket gs://gdc-tcga-phs000178-open/"

DESIGN
------
This script is designed to run as a Slurm batch job with a 24-hour time
limit.  It is fully resumable: every file is written via an atomic
.part -> rename pattern, and a JSON sidecar manifest records completed
downloads.  Re-running after an interruption (session drop, node failure,
job timeout) skips already-complete files and continues from the first
incomplete case.

USAGE
-----
Slurm batch script:

    #!/bin/bash
    #SBATCH --job-name=tcga_svs_download
    #SBATCH --time=24:00:00
    #SBATCH --mem=4G
    #SBATCH --output=logs/download_%j.out
    #SBATCH --error=logs/download_%j.err

    python gcs_slide_downloader.py matched_controls_manifest.tsv \\
        --out_dir tcga_data/ \\
        --max_workers 4

Interactive usage (for testing a few files):

    python gcs_slide_downloader.py matched_controls_manifest.tsv \\
        --out_dir tcga_data/ \\
        --max_workers 1 \\
        --limit 5

Author: Phillip Nyamwaya
"""

import os
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Chunk size for streaming downloads (256 KB balances memory and throughput)
_CHUNK_SIZE = 262144

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gcs_slide_downloader")

# Thread-safe lock for sidecar writes when using multiple workers
_sidecar_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Resumable download helpers (sidecar manifest + atomic writes)
# Identical contract to slide_download_generator_v5/v6 for compatibility.
# ---------------------------------------------------------------------------

_SIDECAR_FILENAME = ".download_manifest.json"


def _sidecar_path(out_dir: Path) -> Path:
    return out_dir / _SIDECAR_FILENAME


def _load_sidecar(out_dir: Path) -> Dict[str, Dict]:
    sp = _sidecar_path(out_dir)
    if not sp.exists():
        return {}
    try:
        with open(sp, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(f"[RESUME] Could not read sidecar {sp}: {e}")
        return {}


def _write_sidecar(out_dir: Path, entry: Dict) -> None:
    """Thread-safe atomic sidecar write."""
    sp  = _sidecar_path(out_dir)
    tmp = sp.with_suffix(".tmp")
    with _sidecar_lock:
        records = _load_sidecar(out_dir)
        records[entry["filename"]] = entry
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2)
            os.replace(tmp, sp)
        except Exception as e:
            logger.error(f"[RESUME] Failed to write sidecar {sp}: {e}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)


def is_download_complete(out_dir: Path, filename: str) -> Tuple[bool, Optional[Path]]:
    """
    Three-way completion check:
      1. Sidecar entry exists for this filename.
      2. File exists on disk.
      3. Disk byte count matches expected_size_bytes in the sidecar.

    When expected_size_bytes == 0 (server omitted Content-Length), we
    accept any non-empty file.
    """
    records  = _load_sidecar(out_dir)
    if filename not in records:
        return False, None
    meta     = records[filename]
    expected = int(meta.get("expected_size_bytes", 0))
    target   = out_dir / filename
    if not target.exists():
        return False, None
    actual = target.stat().st_size
    if actual == 0:
        return False, None
    if expected > 0 and actual != expected:
        logger.warning(
            f"[RESUME] {filename}: sidecar says {expected:,} bytes, "
            f"disk has {actual:,} bytes -- incomplete."
        )
        return False, None
    return True, target


def _remove_partial(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
            logger.info(f"[RESUME] Removed partial: {path.name}")
    except Exception as e:
        logger.warning(f"[RESUME] Could not remove {path}: {e}")


# ---------------------------------------------------------------------------
# HTTP retry wrapper for GCS HTTPS downloads
# ---------------------------------------------------------------------------

def _retry_get_stream(
    url: str,
    max_retries: int = 5,
    timeout: int = 120,
) -> Optional[requests.Response]:
    """
    Open a streaming GET to a GCS public URL.
    Retries on network errors and 5xx responses with exponential backoff.
    Does NOT retry 403/404 -- those indicate a missing or non-public file.
    """
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (403, 404):
                logger.error(
                    f"[GCS] HTTP {r.status_code} for {url} -- "
                    f"file may not be in the open bucket.  Will not retry."
                )
                return None
            logger.warning(f"[GCS] Attempt {attempt}: HTTP {r.status_code} for {url}")
        except requests.exceptions.Timeout:
            logger.warning(f"[GCS] Attempt {attempt}: timeout after {timeout}s")
        except Exception as e:
            logger.warning(f"[GCS] Attempt {attempt}: {e}")
        time.sleep(delay)
        delay *= 2
    logger.error(f"[GCS] All {max_retries} attempts exhausted for {url}")
    return None


# ---------------------------------------------------------------------------
# Single-file download function
# ---------------------------------------------------------------------------

def download_svs(
    gcs_url: str,
    file_name: str,
    out_dir: Path,
    acc_case_id: str,
    file_id: str,
    force: bool = False,
) -> Tuple[bool, str]:
    """
    Download one SVS file from GCS to out_dir.

    Returns (success: bool, message: str).

    Protocol:
      1. Completion check against sidecar -- skip if already complete.
      2. Remove any .part orphan from a prior interrupted run.
      3. Stream to <file_name>.part, fsync, then rename atomically.
      4. Write sidecar entry.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / file_name
    part_path = out_dir / f"{file_name}.part"

    # Step 1: completion check
    if not force:
        complete, existing = is_download_complete(out_dir, file_name)
        if complete:
            return True, f"ALREADY_COMPLETE: {file_name}"
        _remove_partial(part_path)

    # Step 2: open streaming response
    r = _retry_get_stream(gcs_url)
    if r is None:
        return False, f"FAILED_OPEN: {gcs_url}"

    expected_size = int(r.headers.get("Content-Length", 0))

    # Step 3: stream to .part file
    try:
        with open(part_path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=_CHUNK_SIZE):
                if chunk:
                    fh.write(chunk)
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as e:
        _remove_partial(part_path)
        return False, f"STREAM_ERROR: {e}"

    # Step 4: atomic rename
    os.replace(part_path, out_path)
    actual_size = out_path.stat().st_size

    # Validate size if Content-Length was provided
    if expected_size > 0 and actual_size != expected_size:
        logger.error(
            f"[GCS] {file_name}: size mismatch -- expected {expected_size:,}, "
            f"got {actual_size:,}.  Removing corrupt file."
        )
        out_path.unlink(missing_ok=True)
        return False, f"SIZE_MISMATCH: {file_name}"

    # Step 5: sidecar entry
    _write_sidecar(out_dir, {
        "filename":            file_name,
        "expected_size_bytes": actual_size,
        "source":              "GCS",
        "acc_case_id":         acc_case_id,
        "file_id":             file_id,
        "gcs_url":             gcs_url,
        "download_timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    return True, f"OK: {file_name} ({actual_size / 1_073_741_824:.2f} GB)"


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_downloads(
    manifest_path: Path,
    out_dir: Path,
    max_workers: int = 4,
    limit: Optional[int] = None,
    force: bool = False,
    summary_path: Optional[Path] = None,
):
    """
    Read the manifest TSV produced by gdc_match_manifest.py and download
    all matched SVS files from the GCS open-access bucket.

    Parameters
    ----------
    manifest_path  : TSV produced by gdc_match_manifest.py
    out_dir        : Root output directory.  Per-case files go in
                     out_dir/ACC_<acc_case_id>/GCS/<file_name>.svs
    max_workers    : Parallel download threads.  4 is a good default for
                     most HCI environments.  Reduce to 1 for debugging.
    limit          : Stop after this many downloads (useful for testing).
    force          : Re-download even if completion check passes.
    summary_path   : Path to write a results CSV (default: out_dir/download_summary.csv)
    """
    manifest = pd.read_csv(manifest_path, sep="\t")

    # Normalise the status column defensively -- strip whitespace, uppercase,
    # to guard against any trailing spaces or case differences written by Phase 1.
    manifest["status"] = manifest["status"].astype(str).str.strip().str.upper()

    # Filter to MATCHED rows only; guard against null/nan gcs_url values
    # (NO_MATCH rows have empty gcs_url which pandas reads back as float NaN)
    to_download = manifest[
        (manifest["status"] == "MATCHED") &
        (manifest["gcs_url"].notna()) &
        (manifest["gcs_url"].astype(str).str.startswith("https://"))
    ].copy()
    no_match = manifest[manifest["status"] == "NO_MATCH"]

    logger.info(
        f"Manifest: {len(manifest)} total cases  |  "
        f"{len(to_download)} to download  |  "
        f"{len(no_match)} with no GDC match"
    )
    if len(no_match) > 0:
        logger.info(f"No-match cases: {no_match['acc_case_id'].tolist()}")

    if limit:
        to_download = to_download.head(limit)
        logger.info(f"--limit {limit}: restricting to first {limit} rows.")

    if summary_path is None:
        summary_path = out_dir / "download_summary.csv"

    # Load existing summary to avoid re-queuing already-downloaded cases
    completed_ids: set = set()
    summary_rows: List[Dict] = []
    if summary_path.exists():
        try:
            prior = pd.read_csv(summary_path)
            summary_rows  = prior.to_dict(orient="records")
            completed_ids = set(str(r.get("acc_case_id", "")) for r in summary_rows
                                if r.get("download_status") == "SUCCESS")
            logger.info(f"[RESUME] {len(completed_ids)} cases already successful in {summary_path}")
        except Exception as e:
            logger.warning(f"Could not read {summary_path}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the work queue, skipping already-completed cases
    work_items = []
    for _, row in to_download.iterrows():
        acc_id    = str(row["acc_case_id"])
        file_id   = str(row["file_id"])
        file_name = str(row["file_name"])
        url       = str(row["gcs_url"])
        dest_dir  = out_dir / f"ACC_{acc_id}" / "GCS"

        if not force and acc_id in completed_ids:
            logger.info(f"[RESUME] {acc_id} already successful -- skipping.")
            continue

        work_items.append({
            "acc_case_id": acc_id,
            "file_id":     file_id,
            "file_name":   file_name,
            "gcs_url":     url,
            "dest_dir":    dest_dir,
            "row":         row,
        })

    logger.info(f"Queuing {len(work_items)} downloads with {max_workers} parallel worker(s).")

    success_count = 0
    fail_count    = 0

    def _do_download(item: Dict) -> Dict:
        ok, msg = download_svs(
            gcs_url     = item["gcs_url"],
            file_name   = item["file_name"],
            out_dir     = item["dest_dir"],
            acc_case_id = item["acc_case_id"],
            file_id     = item["file_id"],
            force       = force,
        )
        return {"ok": ok, "msg": msg, "item": item}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_do_download, item): item for item in work_items}

        with tqdm(total=len(work_items), desc="Downloading SVS", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                ok, msg, item = result["ok"], result["msg"], result["item"]
                row = item["row"]

                pbar.update(1)

                status = "SUCCESS" if ok else "FAILED"
                if ok:
                    success_count += 1
                    logger.info(f"[{status}] {item['acc_case_id']}: {msg}")
                else:
                    fail_count += 1
                    logger.warning(f"[{status}] {item['acc_case_id']}: {msg}")

                summary_rows.append({
                    "acc_case_id":       item["acc_case_id"],
                    "acc_sex":           row.get("acc_sex", ""),
                    "acc_age":           row.get("acc_age", ""),
                    "acc_diag_class":    row.get("acc_diag_class", ""),
                    "tier_matched":      row.get("tier_matched", ""),
                    "file_id":           item["file_id"],
                    "file_name":         item["file_name"],
                    "control_case_id":   row.get("control_case_id", ""),
                    "control_race":      row.get("control_race", ""),
                    "control_age":       row.get("control_age", ""),
                    "age_delta":         row.get("age_delta", ""),
                    "control_diag_class":row.get("control_diag_class", ""),
                    "gcs_url":           item["gcs_url"],
                    "download_path":     str(item["dest_dir"] / item["file_name"]),
                    "download_status":   status,
                    "message":           msg,
                })

                # Flush summary after every completion so progress survives crashes
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    logger.info(
        f"\nDone.  "
        f"Success: {success_count}  |  "
        f"Failed: {fail_count}  |  "
        f"Summary: {summary_path}"
    )
    if fail_count > 0:
        failed_ids = [
            r["acc_case_id"] for r in summary_rows
            if r.get("download_status") == "FAILED"
        ]
        logger.warning(f"Failed cases: {failed_ids}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 2: Download matched TCGA SVS slides from the ISB-CGC "
            "open-access GCS bucket.  No GDC token or Google account needed. "
            "Run as a Slurm batch job (--time=24:00:00 recommended). "
            "Fully resumable -- re-running skips already-complete files."
        )
    )
    parser.add_argument("manifest",
                        help="Manifest TSV produced by gdc_match_manifest.py")
    parser.add_argument("--out_dir", default="tcga_data",
                        help="Root output directory (default: tcga_data)")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="Parallel download threads (default: 4). "
                             "Check your HCI network policy before raising above 4.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only the first N files (useful for testing)")
    parser.add_argument("--force", action="store_true", default=False,
                        help="Re-download even if files already pass the completion check")
    parser.add_argument("--summary", default=None,
                        help="Path for the results CSV (default: <out_dir>/download_summary.csv)")

    args = parser.parse_args()

    run_downloads(
        manifest_path = Path(args.manifest),
        out_dir       = Path(args.out_dir),
        max_workers   = args.max_workers,
        limit         = args.limit,
        force         = args.force,
        summary_path  = Path(args.summary) if args.summary else None,
    )
