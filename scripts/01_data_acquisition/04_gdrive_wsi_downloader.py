#!/usr/bin/env python3
"""
gdrive_wsi_downloader.py
========================
Download whole-slide images (SVS, NDPI, MRXS, TIF, etc.) from a publicly
shared Google Drive folder with:

  - Recursive sub-folder traversal via Drive API v3 (no auth required for
    public folders -- optional API key raises listing quota).
  - gdown-backed downloads, which correctly handles Google's virus-scan
    confirmation interstitial for files of ANY size, including those above
    the ~100 MB threshold where plain requests-based approaches fail.
  - Resumable downloads: gdown resume=True continues partial downloads
    across Artemis HCI session interruptions.
  - CSV manifest updated after every download attempt so progress survives
    session timeouts.
  - Fast-resume: only files with status=downloaded AND verified on disk are
    skipped. Failed files are always retried on re-run (this was the v2 bug).

Changelog
---------
v1.0.0  Initial release (requests-based download, failed above ~100 MB)
v2.0.0  Switched to gdown backend; added --retry_failed flag
v3.0.0  Fixed silent-skip bug: failed files are now always retried by
        default. Removed --retry_failed flag (no longer needed).
        Failed files are never silently skipped under any circumstance.

Usage
-----
  # Minimal
  python gdrive_wsi_downloader.py "https://drive.google.com/drive/folders/FOLDER_ID"

  # With explicit output directory (Artemis)
  python gdrive_wsi_downloader.py FOLDER_ID --out_dir /scratch/users/pn254/slides

  # Dry run -- list files that would be downloaded, then exit
  python gdrive_wsi_downloader.py FOLDER_ID --dry_run

  # Force re-download everything from scratch (ignores manifest entirely)
  python gdrive_wsi_downloader.py FOLDER_ID --force_redownload

Dependencies
------------
  pip install requests gdown tqdm --user

Author: Phillip Nyamwaya
Version: 3.0.0
"""

import argparse
import csv
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import requests
import gdown


# ---------------------------------------------------------------------------
# >>>  ARTEMIS OUTPUT PATH PLACEHOLDER  <<<
# Set this to your Artemis scratch or home data partition path.
# Example: /its/home/pn254/project2026/wsi_data/acc_data
# ---------------------------------------------------------------------------
ARTEMIS_DEFAULT_OUTPUT_DIR: str = "/scratch/users/YOUR_ARTEMIS_USERNAME/gdrive_slides"


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Google Drive API v3 -- plain HTTP, no SDK required
GDRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
GDRIVE_FILE_FIELDS = "id,name,mimeType,size,parents"
GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"

# Slide file extensions to download (case-insensitive)
SLIDE_EXTENSIONS: Tuple[str, ...] = (
    ".svs",    # Aperio / Leica
    ".ndpi",   # Hamamatsu
    ".mrxs",   # 3DHISTECH Pannoramic
    ".tif",    # Generic TIFF (OME-TIFF, BigTIFF, etc.)
    ".tiff",
    ".scn",    # Leica SCN
    ".vms",    # Hamamatsu VMS
    ".vmu",    # Hamamatsu VMU
    ".bif",    # Ventana BIF
    ".qptiff", # PerkinElmer Phenocycler
)

# CSV manifest columns
MANIFEST_COLUMNS: List[str] = [
    "file_id",
    "filename",
    "mime_type",
    "size_bytes",
    "folder_path",
    "drive_url",
    "local_path",
    "download_timestamp",
    "status",          # "downloaded" | "failed"
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gdrive_wsi_downloader")


# ===========================================================================
# Section 1 -- URL / folder ID parsing
# ===========================================================================

def extract_folder_id(folder_url_or_id: str) -> str:
    """
    Accept either a raw Drive folder ID or any standard sharing URL variant
    and return the bare folder ID string.

    Supported URL shapes:
      https://drive.google.com/drive/folders/<ID>
      https://drive.google.com/drive/folders/<ID>?usp=sharing
      https://drive.google.com/open?id=<ID>

    Raises ValueError if no valid ID can be parsed.
    """
    s = folder_url_or_id.strip()

    # Already a bare ID: alphanumeric + underscores/dashes, no slashes
    if re.match(r'^[A-Za-z0-9_\-]{10,}$', s):
        return s

    parsed = urlparse(s)

    # Pattern: /drive/folders/<ID>
    m = re.search(r'/folders/([A-Za-z0-9_\-]+)', parsed.path)
    if m:
        return m.group(1)

    # Pattern: ?id=<ID>
    qs = parse_qs(parsed.query)
    if "id" in qs:
        return qs["id"][0]

    raise ValueError(
        f"Cannot parse a Drive folder ID from: {folder_url_or_id!r}\n"
        "Supply a raw folder ID or a standard Drive sharing URL."
    )


# ===========================================================================
# Section 2 -- Drive API listing helpers
# ===========================================================================

def _drive_api_get(
    endpoint: str,
    params: Dict,
    api_key: Optional[str],
    max_retries: int = 4,
) -> Optional[dict]:
    """
    GET against Drive API v3 with exponential-backoff retry.
    Returns parsed JSON on HTTP 200, or None after all retries exhausted.
    """
    url = GDRIVE_API_BASE + endpoint
    if api_key:
        params = {**params, "key": api_key}

    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429 or resp.status_code >= 500:
                logger.warning(
                    f"Drive API HTTP {resp.status_code} (attempt {attempt}); "
                    f"retrying in {delay:.0f}s..."
                )
                time.sleep(delay)
                delay *= 2
                continue
            logger.error(f"Drive API error {resp.status_code}: {resp.text[:300]}")
            return None
        except requests.RequestException as exc:
            logger.warning(f"Network error (attempt {attempt}): {exc}")
            time.sleep(delay)
            delay *= 2

    logger.error(f"Drive API: giving up after {max_retries} attempts for {endpoint}")
    return None


def list_folder_contents(
    folder_id: str,
    api_key: Optional[str],
) -> Iterator[dict]:
    """
    Yield every item (file or sub-folder) in a Drive folder, handling
    Drive's paginated response automatically.
    """
    page_token: Optional[str] = None

    while True:
        params: Dict = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": f"nextPageToken,files({GDRIVE_FILE_FIELDS})",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token

        data = _drive_api_get("/files", params, api_key)
        if data is None:
            logger.error(f"Failed to list folder {folder_id}; stopping traversal.")
            return

        for item in data.get("files", []):
            yield item

        page_token = data.get("nextPageToken")
        if not page_token:
            break


def walk_drive_folder(
    folder_id: str,
    api_key: Optional[str],
    current_path: str = "",
) -> Iterator[Tuple[dict, str]]:
    """
    Recursively walk a Drive folder tree.
    Yields (file_resource, containing_folder_path) for every non-folder item.
    """
    for item in list_folder_contents(folder_id, api_key):
        item_path = f"{current_path}/{item['name']}" if current_path else item["name"]

        if item["mimeType"] == GDRIVE_FOLDER_MIME:
            logger.info(f"[Traversal] Entering sub-folder: {item_path}")
            yield from walk_drive_folder(item["id"], api_key, item_path)
        else:
            yield item, current_path


# ===========================================================================
# Section 3 -- Slide file filtering
# ===========================================================================

def is_slide_file(filename: str) -> bool:
    """Return True if filename has a recognised slide extension (case-insensitive)."""
    return Path(filename).suffix.lower() in SLIDE_EXTENSIONS


# ===========================================================================
# Section 4 -- CSV manifest
# ===========================================================================

def load_completed_ids(manifest_path: Path) -> set:
    """
    Read the manifest and return a set of file_ids that have been
    VERIFIED as complete: status=downloaded AND the local file still
    exists on disk AND the size matches.

    CRITICAL DESIGN DECISION (v3 fix):
    Only verified-complete files are returned here. Failed entries are
    intentionally excluded so they are always retried on the next run.
    There is no mechanism to silently skip a failed file -- that was the
    bug in v2 that caused 70 files to be skipped without any download attempt.

    Parameters
    ----------
    manifest_path : Path
        Path to the manifest CSV.

    Returns
    -------
    set
        Set of file_id strings that are confirmed complete and on disk.
    """
    completed: set = set()

    if not manifest_path.exists():
        logger.info("[Manifest] No existing manifest found -- starting fresh.")
        return completed

    # Read all rows; for each file_id keep only the most recent row
    # (the manifest is append-only so later rows override earlier ones
    # for the same file_id)
    latest: Dict[str, dict] = {}
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = row.get("file_id", "").strip()
            if fid:
                latest[fid] = row   # Overwrite with the most recent entry

    for fid, row in latest.items():
        # Only count as complete if ALL three conditions hold:
        #   1. status column says downloaded
        #   2. The recorded local path actually exists on disk
        #   3. The on-disk file size matches the Drive API size (if known)
        if row.get("status") != "downloaded":
            continue

        local_path = Path(row.get("local_path", ""))
        if not local_path.exists():
            logger.warning(
                f"[Manifest] {row.get('filename')} marked downloaded but "
                f"file missing at {local_path} -- will re-download."
            )
            continue

        expected = row.get("size_bytes", "")
        if expected:
            try:
                if local_path.stat().st_size != int(expected):
                    logger.warning(
                        f"[Manifest] {row.get('filename')} size mismatch on disk -- "
                        "will re-download."
                    )
                    continue
            except (ValueError, OSError):
                pass  # If we cannot check size, accept the entry

        completed.add(fid)

    n_failed = sum(1 for r in latest.values() if r.get("status") == "failed")
    logger.info(
        f"[Manifest] {len(completed)} verified complete, "
        f"{n_failed} previously failed (will be retried this run)."
    )
    return completed


def append_manifest_row(manifest_path: Path, row: dict) -> None:
    """
    Append one row to the manifest CSV immediately after a download attempt.
    Creates the file with a header if it does not yet exist.
    Uses append mode so a crash during write cannot corrupt existing rows.
    """
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ===========================================================================
# Section 5 -- Download engine (gdown backend)
# ===========================================================================

def download_drive_file(
    file_id: str,
    dest_path: Path,
    expected_size: Optional[int],
    force: bool = False,
) -> bool:
    """
    Download a single Drive file using gdown.

    gdown handles Google's virus-scan interstitial for files of any size,
    including the JS-rendered confirmation page served for files above ~100 MB
    that breaks plain requests-based approaches. See:
      https://github.com/wkentaro/gdown

    Parameters
    ----------
    file_id : str
        Google Drive file ID.
    dest_path : Path
        Final destination path.
    expected_size : int or None
        Expected byte size from Drive API metadata. None skips size check.
    force : bool
        If True, remove any existing file and restart from byte 0.

    Returns
    -------
    bool
        True if the file downloaded and size-verified successfully.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if force and dest_path.exists():
        dest_path.unlink()
        logger.info(f"[Download] Force mode: removed {dest_path.name}")

    try:
        result = gdown.download(
            id=file_id,
            output=str(dest_path),
            quiet=False,       # Show gdown progress in SLURM .out log
            resume=True,       # Resume partial downloads across session restarts
            use_cookies=True,  # Required for virus-scan confirmation bypass
        )
    except Exception as exc:
        logger.error(f"[Download] gdown exception for {file_id}: {exc}")
        return False

    # gdown returns None on failure, the output path string on success
    if result is None:
        logger.error(f"[Download] gdown returned None for file_id={file_id}")
        return False

    if not dest_path.exists():
        logger.error(
            f"[Download] gdown reported success but {dest_path.name} "
            "not found on disk."
        )
        return False

    actual_size = dest_path.stat().st_size

    # Size verification against Drive API metadata
    if expected_size is not None and actual_size != expected_size:
        logger.error(
            f"[Download] Size mismatch: {dest_path.name} -- "
            f"expected {expected_size:,} B, got {actual_size:,} B. "
            "Removing and will retry on next run."
        )
        dest_path.unlink(missing_ok=True)
        return False

    logger.info(f"[Download] OK: {dest_path.name} ({actual_size / (1024**2):.1f} MB)")
    return True


# ===========================================================================
# Section 6 -- Main orchestration
# ===========================================================================

def run_downloader(
    folder_url_or_id: str,
    out_dir: Path,
    api_key: Optional[str],
    force_redownload: bool,
    dry_run: bool,
    sleep_sec: float,
) -> None:
    """
    Main entry point: walk the Drive folder, filter for slide files,
    skip verified-complete files, download everything else.
    """
    try:
        folder_id = extract_folder_id(folder_url_or_id)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info(f"[Init] Folder ID         : {folder_id}")
    logger.info(f"[Init] Output directory  : {out_dir}")
    logger.info(f"[Init] Force re-download : {force_redownload}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "gdrive_wsi_manifest.csv"

    # Load the set of file_ids that are already verified on disk.
    # Failed entries are deliberately NOT in this set -- they will be retried.
    if force_redownload:
        completed_ids: set = set()
        logger.info("[Init] Force-redownload: all files will be re-downloaded.")
    else:
        completed_ids = load_completed_ids(manifest_path)

    # Walk the Drive folder tree and collect slide files
    logger.info("[Traversal] Walking Drive folder tree...")
    all_slides: List[Tuple[dict, str]] = [
        (fr, fp)
        for fr, fp in walk_drive_folder(folder_id, api_key)
        if is_slide_file(fr["name"])
    ]

    total = len(all_slides)
    logger.info(f"[Traversal] Found {total} slide file(s).")

    if total == 0:
        logger.warning(
            "No slide files found. Check the folder ID and confirm the "
            "folder is shared as 'Anyone with the link'."
        )
        return

    if dry_run:
        logger.info("[Dry Run] Files that would be downloaded or retried:")
        for idx, (fr, fp) in enumerate(all_slides, 1):
            size_mb  = int(fr.get("size", 0)) / (1024 ** 2)
            loc      = f"{fp}/{fr['name']}" if fp else fr["name"]
            tag      = "[SKIP]" if fr["id"] in completed_ids else "[DOWNLOAD]"
            print(f"  {idx:4d}. {tag} {loc}  ({size_mb:.1f} MB)")
        return

    # Download loop
    downloaded_count = 0
    skipped_count    = 0
    failed_count     = 0

    for idx, (file_resource, folder_path) in enumerate(all_slides, 1):
        file_id    = file_resource["id"]
        filename   = file_resource["name"]
        mime_type  = file_resource.get("mimeType", "")
        size_bytes = int(file_resource.get("size", 0)) or None
        drive_url  = f"https://drive.google.com/file/d/{file_id}/view"
        size_label = f"{size_bytes / (1024**2):.1f} MB" if size_bytes else "size unknown"

        # Mirror Drive sub-folder structure locally to avoid filename collisions
        if folder_path:
            safe_parts = [re.sub(r'[<>:"/\\|?*]', '_', p) for p in folder_path.split("/") if p]
            dest_dir   = out_dir / Path(*safe_parts)
        else:
            dest_dir = out_dir

        dest_path = dest_dir / filename

        logger.info(f"[{idx}/{total}] {filename}  ({size_label})")

        # The ONLY skip condition: file is verified complete on disk
        if file_id in completed_ids:
            logger.info(f"  -> Already complete, skipping.")
            skipped_count += 1
            continue

        # Attempt download
        success = download_drive_file(
            file_id=file_id,
            dest_path=dest_path,
            expected_size=size_bytes,
            force=force_redownload,
        )

        status = "downloaded" if success else "failed"
        if success:
            downloaded_count += 1
        else:
            failed_count += 1
            logger.warning(f"  -> Failed. Will retry automatically on next run.")

        # Write manifest row immediately -- survives session interruption
        append_manifest_row(manifest_path, {
            "file_id":            file_id,
            "filename":           filename,
            "mime_type":          mime_type,
            "size_bytes":         size_bytes if size_bytes else "",
            "folder_path":        folder_path,
            "drive_url":          drive_url,
            "local_path":         str(dest_path) if success else "",
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "status":             status,
        })

        if sleep_sec > 0 and idx < total:
            time.sleep(sleep_sec)

    # Summary
    logger.info("=" * 60)
    logger.info(
        f"Run complete. "
        f"Downloaded: {downloaded_count} | "
        f"Skipped (verified): {skipped_count} | "
        f"Failed: {failed_count} | "
        f"Total: {total}"
    )
    if failed_count > 0:
        logger.info(
            f"{failed_count} file(s) failed -- re-run the same command "
            "to retry them automatically. No extra flags needed."
        )
    logger.info(f"Manifest: {manifest_path}")
    logger.info("=" * 60)


# ===========================================================================
# Section 7 -- CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download whole-slide images from a public Google Drive folder.\n"
            "Uses gdown to handle files of any size (including >100 MB).\n"
            "Resumable: simply re-run the same command after an interruption."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard run
  python gdrive_wsi_downloader.py "https://drive.google.com/drive/folders/1AbC..."

  # Explicit output path (Artemis)
  python gdrive_wsi_downloader.py 1AbC... --out_dir /its/home/pn254/project2026/wsi_data/acc_data

  # Dry run: see what would download
  python gdrive_wsi_downloader.py 1AbC... --dry_run

  # Force re-download everything (ignore manifest)
  python gdrive_wsi_downloader.py 1AbC... --force_redownload

Retrying failures:
  Just re-run the same command. Failed files are always retried automatically.
  No extra flags needed.
        """,
    )

    parser.add_argument(
        "folder",
        metavar="FOLDER_URL_OR_ID",
        help="Public Google Drive folder URL or bare folder ID.",
    )
    parser.add_argument(
        "--out_dir",
        default=ARTEMIS_DEFAULT_OUTPUT_DIR,
        help=(
            "Root output directory for downloaded slides. "
            f"Default: {ARTEMIS_DEFAULT_OUTPUT_DIR}. "
            "Update ARTEMIS_DEFAULT_OUTPUT_DIR at the top of this file "
            "to set your Artemis path permanently."
        ),
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("GDRIVE_API_KEY", None),
        help=(
            "Google Drive API key (optional; increases listing quota). "
            "Also reads from the GDRIVE_API_KEY environment variable."
        ),
    )
    parser.add_argument(
        "--force_redownload",
        action="store_true",
        default=False,
        help="Ignore the manifest and re-download all files from scratch.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="List all files that would be downloaded or retried, then exit.",
    )
    parser.add_argument(
        "--sleep_sec",
        type=float,
        default=0.5,
        help="Seconds to sleep between downloads. Default: 0.5",
    )

    args = parser.parse_args()

    run_downloader(
        folder_url_or_id=args.folder,
        out_dir=Path(args.out_dir),
        api_key=args.api_key,
        force_redownload=args.force_redownload,
        dry_run=args.dry_run,
        sleep_sec=args.sleep_sec,
    )
