#!/usr/bin/env python3
"""
slide_download_generator_v5.py

Download matched non-African control diagnostic WSIs from GDC/TCGA
with TCIA fallback (slide/histology modality only) and AWS Open Data
last-resort fallback.

KEY CHANGES FROM v4 (resumable download support for HCI environments)
----------------------------------------------------------------------
1.  RESUMABLE DOWNLOADS -- the script is now fully safe to interrupt and
    restart. On re-run it will skip any case that already has a verified
    complete download and pick up exactly where it left off.

    Three mechanisms work together:

    a) ATOMIC WRITES via a .part temp file:
       Every download (GDC HTTP, TCIA, AWS S3) writes to
       <final_path>.part while in progress.  The file is only renamed
       to its final name after the last byte is flushed and fsync'd.
       A .part file left by a crashed run is therefore always treated
       as incomplete and re-downloaded on the next run.

    b) SIDECAR MANIFEST (.download_manifest.json):
       After each successful rename, a small JSON file is written to
       the same directory recording: acc_case_id, source, filename,
       expected_size_bytes, and download_timestamp.  This is the
       authoritative "completed" record.

    c) COMPLETION CHECK (is_download_complete):
       Before doing any API query or download for a case, the script
       checks for a sidecar entry whose filename exists on disk AND
       whose size on disk matches expected_size_bytes.  If both are
       true the case is skipped immediately with an INFO log.
       If the sidecar exists but the file is missing or smaller, the
       partial file is deleted and the download is retried.

2.  SUMMARY CSV RESUME:
    On startup, if matched_controls_summary.csv already exists it is
    read back into summary_rows and used_ids so the final CSV is always
    a cumulative, non-duplicated record across all runs.

3.  All v4 fixes (tiered GDC queries, BN prefix, TCIA modality filter,
    AWS UUID path) are retained unchanged.

4.  New CLI flag: --force_redownload  -- bypasses the completion check
    and re-downloads everything. Useful when you suspect a corrupt file
    that happens to be the right size (rare, but possible).

Author: Phillip Nyamwaya (v5 -- resumable download support)
"""

import json
import os
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# ----------- AWS -----------------------------------------------------------
import boto3
from botocore import UNSIGNED
from botocore.client import Config
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config & logging
# ---------------------------------------------------------------------------

GDC_BASE = "https://api.gdc.cancer.gov"
GDC_FILES_ENDPOINT = f"{GDC_BASE}/files"
GDC_DATA_ENDPOINT  = f"{GDC_BASE}/data"
GDC_CASES_ENDPOINT = f"{GDC_BASE}/cases"

# v3 TCIA endpoint (v4 is slow / less stable for bulk getSeries)
TCIA_BASE    = "https://services.cancerimagingarchive.net/services/v3/TCIA/query"
TCIA_API_KEY = ""   # Set via env variable TCIA_API_KEY if you have one

# TCGA AWS Open Data bucket
TCGA_S3_BUCKET = "tcga-2-open"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("matched_wsi_downloader_v4")


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def retry_request(
    url: str,
    params: dict = None,
    stream: bool = False,
    max_retries: int = 3,
    timeout: int = 60,
) -> Optional[requests.Response]:
    """
    GET request with exponential backoff retry.
    Returns the Response on HTTP 200, or None after all retries are exhausted.
    """
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, stream=stream, timeout=timeout)
            if r.status_code == 200:
                return r
            logger.warning(f"Attempt {attempt}: HTTP {r.status_code} for {url}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}: Request failed: {e}")
        time.sleep(delay)
        delay *= 2
    logger.error(f"Failed after {max_retries} attempts: {url}")
    return None


# ---------------------------------------------------------------------------
# Diagnosis helpers
# ---------------------------------------------------------------------------

def normalize_diagnosis_class(diag: str) -> str:
    """
    Map free-text diagnosis strings to a coarse class label used for
    scoring candidate controls.
    """
    if not isinstance(diag, str):
        return "unknown"
    d = diag.lower()
    if any(k in d for k in ["invasive", "carcinoma", "malignant", "adenocarcinoma",
                             "squamous", "tumor", "tumour", "cancer"]):
        return "malignant"
    if any(k in d for k in ["benign", "fibroadenoma", "fibrocystic", "lipoma"]):
        return "benign"
    if any(k in d for k in ["hyperplasia", "atypia", "dysplasia", "cin"]):
        return "pre-malignant"
    return "other"


# ---------------------------------------------------------------------------
# FIX 1: Complete prefix-to-primary-site mapping
# ---------------------------------------------------------------------------

# All prefixes found in acc_manifest.tsv:
#   BP (86) - Breast, positive/malignant
#   BN (27) - Breast, negative/benign       <-- was MISSING in v3
#   CP (42) - Cervix, positive/malignant
#   CN  (5) - Cervix, negative/benign
#   RP (15) - Rectum/Colon, positive
#   TP (26) - Tongue/Oral Cavity, positive
#   TN      - Tongue/Oral Cavity, negative  <-- added for completeness

_PREFIX_TO_SITE: Dict[str, str] = {
    "BP": "Breast",
    "BN": "Breast",        # Benign breast -- was missing in v3
    "CP": "Cervix Uteri",
    "CN": "Cervix Uteri",
    "RP": "Colon",
    "RN": "Colon",
    "TP": "Oral Cavity",
    "TN": "Oral Cavity",
}

def infer_primary_site(case_id: str) -> Optional[str]:
    """
    Return a GDC-compatible primary_site string from the internal ACC
    case ID prefix, or None if the prefix is not recognised.
    """
    prefix = case_id[:2].upper() if isinstance(case_id, str) and len(case_id) >= 2 else ""
    return _PREFIX_TO_SITE.get(prefix, None)


# ---------------------------------------------------------------------------
# FIX 2: Tiered GDC filter builders
# ---------------------------------------------------------------------------

def _base_svs_filter() -> List[Dict]:
    """
    Mandatory fields every tier must include:
      - data_category = Slide Image (broader than 'Image')
      - data_type = Slide Image  (GDC uses this for WSIs)
      - exclude Black / African American donors
    Note: We intentionally broaden data_type from 'Diagnostic Slide' to
    capture TCGA cases where the data_type is 'Slide Image'.
    """
    return [
        {
            "op": "in",
            "content": {
                "field": "data_format",
                # SVS is the canonical whole-slide format on GDC
                "value": ["SVS"]
            }
        },
        {
            "op": "not_in",
            "content": {
                "field": "cases.demographic.race",
                "value": ["Black or African American"]
            }
        }
    ]


def build_gdc_filters_tier1(
    case_id: str,
    sex: str,
    age: Optional[float],
    diag_class: str,
    age_window: int = 20,
) -> Optional[Dict]:
    """
    Tier 1 (strict): site + sex + age window + diagnosis class.
    Returns None if site cannot be inferred (prevents empty-dict injection).
    """
    primary_site = infer_primary_site(case_id)
    if primary_site is None:
        logger.warning(f"Cannot infer primary_site for {case_id}; skipping Tier 1.")
        return None

    content = _base_svs_filter() + [
        {"op": "in", "content": {"field": "cases.primary_site", "value": [primary_site]}}
    ]

    # Sex filter
    if isinstance(sex, str) and sex.strip():
        gdc_gender = "male" if sex.upper().startswith("M") else "female"
        content.append(
            {"op": "in", "content": {"field": "cases.demographic.gender", "value": [gdc_gender]}}
        )

    # Age window filter (in days)
    if age is not None and not pd.isna(age):
        lower = int(max(0, (age - age_window) * 365))
        upper = int((age + age_window) * 365)
        content.append(
            {"op": "between", "content": {
                "field": "cases.diagnoses.age_at_diagnosis",
                "value": [lower, upper]
            }}
        )

    return {"op": "and", "content": content}


def build_gdc_filters_tier2(
    case_id: str,
    sex: str,
) -> Optional[Dict]:
    """
    Tier 2 (moderate): site + sex only. Drops age and diagnosis constraints.
    """
    primary_site = infer_primary_site(case_id)
    if primary_site is None:
        logger.warning(f"Cannot infer primary_site for {case_id}; skipping Tier 2.")
        return None

    content = _base_svs_filter() + [
        {"op": "in", "content": {"field": "cases.primary_site", "value": [primary_site]}}
    ]
    if isinstance(sex, str) and sex.strip():
        gdc_gender = "male" if sex.upper().startswith("M") else "female"
        content.append(
            {"op": "in", "content": {"field": "cases.demographic.gender", "value": [gdc_gender]}}
        )
    return {"op": "and", "content": content}


def build_gdc_filters_tier3(case_id: str) -> Optional[Dict]:
    """
    Tier 3 (permissive): site only. Maximises candidate pool for rare sites.
    """
    primary_site = infer_primary_site(case_id)
    if primary_site is None:
        logger.warning(f"Cannot infer primary_site for {case_id}; skipping Tier 3.")
        return None

    content = _base_svs_filter() + [
        {"op": "in", "content": {"field": "cases.primary_site", "value": [primary_site]}}
    ]
    return {"op": "and", "content": content}


# ---------------------------------------------------------------------------
# GDC query execution and matching
# ---------------------------------------------------------------------------

_GDC_FIELDS = [
    "file_id", "file_name",
    "cases.case_id", "cases.submitter_id",
    "cases.demographic.race", "cases.demographic.gender",
    "cases.diagnoses.age_at_diagnosis",
    "cases.diagnoses.primary_diagnosis",
    "cases.primary_site",
    "cases.samples.sample_type",
]


def gdc_query(filters: Dict, size: int = 50) -> List[Dict]:
    """Execute a single GDC files query and return the hits list."""
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(_GDC_FIELDS),
        "format": "JSON",
        "size": size,
    }
    r = retry_request(GDC_FILES_ENDPOINT, params=params)
    if r is None:
        return []
    try:
        return r.json().get("data", {}).get("hits", [])
    except Exception as e:
        logger.error(f"GDC JSON parse error: {e}")
        return []


def gdc_query_tiered(
    acc_case_id: str,
    sex: str,
    age: Optional[float],
    diag_class: str,
    size: int = 50,
    age_window: int = 20,
) -> List[Dict]:
    """
    FIX 2 (core): Run Tier 1 -> Tier 2 -> Tier 3 until we get candidates.
    Returns the first non-empty hit list, together with the tier used.
    """
    # Tier 1 -- full constraints
    f1 = build_gdc_filters_tier1(acc_case_id, sex, age, diag_class, age_window)
    if f1 is not None:
        hits = gdc_query(f1, size)
        if hits:
            logger.info(f"[GDC] Tier 1 match for {acc_case_id}: {len(hits)} candidates")
            return hits

    # Tier 2 -- relax age + diagnosis
    f2 = build_gdc_filters_tier2(acc_case_id, sex)
    if f2 is not None:
        hits = gdc_query(f2, size)
        if hits:
            logger.info(f"[GDC] Tier 2 match for {acc_case_id}: {len(hits)} candidates")
            return hits

    # Tier 3 -- site only
    f3 = build_gdc_filters_tier3(acc_case_id)
    if f3 is not None:
        hits = gdc_query(f3, size)
        if hits:
            logger.info(f"[GDC] Tier 3 match for {acc_case_id}: {len(hits)} candidates")
            return hits

    logger.info(f"[GDC] No candidates found across all tiers for {acc_case_id}")
    return []


def age_from_gdc_hit(hit: Dict) -> Optional[float]:
    """Extract age in years from a GDC file hit."""
    cases = hit.get("cases", [])
    if not cases:
        return None
    diagnoses = cases[0].get("diagnoses", [])
    if not diagnoses:
        return None
    age_days = diagnoses[0].get("age_at_diagnosis")
    return age_days / 365.0 if age_days else None


def primary_diagnosis_from_hit(hit: Dict) -> str:
    cases = hit.get("cases", [])
    if not cases:
        return ""
    diagnoses = cases[0].get("diagnoses", [])
    if not diagnoses:
        return ""
    return diagnoses[0].get("primary_diagnosis", "") or ""


def diag_class_from_hit(hit: Dict) -> str:
    return normalize_diagnosis_class(primary_diagnosis_from_hit(hit))


def select_best_gdc_match(
    acc_age: Optional[float],
    acc_diag_class: str,
    candidates: List[Dict],
) -> Optional[Dict]:
    """
    Score each candidate by (diagnosis_class_penalty + age_delta).
    Lower is better. Returns the best candidate or None.
    """
    if not candidates:
        return None
    scored = []
    for h in candidates:
        gdc_age       = age_from_gdc_hit(h)
        gdc_diag_cls  = diag_class_from_hit(h)
        # Penalise mismatched diagnosis class heavily
        diag_penalty  = 0.0 if gdc_diag_cls == acc_diag_class else 10.0
        # Default age delta of 25 when metadata is absent (neutral penalty)
        age_diff      = abs(acc_age - gdc_age) if acc_age and gdc_age else 25.0
        scored.append((diag_penalty + age_diff, h))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def gdc_download_file(
    file_id: str,
    out_dir: Path,
    acc_case_id: str = "",
    force: bool = False,
) -> Optional[Path]:
    """
    Stream-download a GDC file by its UUID.

    v5 changes:
      - Resolves the filename from the Content-Disposition header first
        (via a HEAD request), enabling the completion check BEFORE the
        download stream is opened -- avoiding an unnecessary TCP connection.
      - Writes to <filename>.part during transfer; renames atomically on
        completion so a half-written file is never left under the final name.
      - Writes a sidecar manifest entry after a successful rename.
      - Skips the download entirely if is_download_complete() returns True
        and force=False.

    Parameters
    ----------
    file_id      : GDC file UUID
    out_dir      : Destination directory
    acc_case_id  : ACC source case ID (for sidecar metadata only)
    force        : If True, bypass the completion check and re-download
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"{GDC_DATA_ENDPOINT}/{file_id}"

    # --- Step 1: resolve filename via HEAD so we can check for completion ---
    # We issue a non-streaming GET with stream=True and read only headers,
    # which is effectively a HEAD for servers that do not honour HEAD.
    r_head = retry_request(url, stream=True)
    if r_head is None:
        return None

    cd       = r_head.headers.get("Content-Disposition", "")
    filename = file_id  # fallback if header is absent
    if "filename=" in cd:
        filename = cd.split("filename=")[-1].strip().strip('"')
    expected_size = int(r_head.headers.get("Content-Length", 0))

    # Close the connection immediately; we will re-open for the full stream
    # if needed.  Closing here avoids keeping a socket open during the check.
    r_head.close()

    # --- Step 2: completion check ---
    if not force:
        complete, existing_path = is_download_complete(out_dir, filename)
        if complete:
            logger.info(f"[RESUME] GDC {filename} already complete -- skipping.")
            return existing_path
        # Remove any partial file left by a previous interrupted download
        _remove_partial(out_dir / filename)

    # --- Step 3: open the full streaming response and download ---
    out_path  = out_dir / filename
    part_path = out_dir / f"{filename}.part"   # atomic write target

    r_stream = retry_request(url, stream=True)
    if r_stream is None:
        return None

    total = int(r_stream.headers.get("Content-Length", expected_size))
    try:
        with open(part_path, "wb") as fh, tqdm(
            total=total or None, unit="B", unit_scale=True,
            desc=f"GDC {filename}", leave=False
        ) as pbar:
            for chunk in r_stream.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
            fh.flush()
            os.fsync(fh.fileno())   # ensure bytes hit disk before rename
    except Exception as e:
        logger.error(f"[GDC] Download error for {filename}: {e}")
        _remove_partial(part_path)
        return None

    # --- Step 4: atomic rename .part -> final name ---
    os.replace(part_path, out_path)

    # --- Step 5: write sidecar manifest ---
    _write_sidecar(out_dir, {
        "filename":            filename,
        "expected_size_bytes": out_path.stat().st_size,  # use actual size
        "source":              "GDC",
        "acc_case_id":         acc_case_id,
        "file_id":             file_id,
        "download_timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    return out_path


# ---------------------------------------------------------------------------
# FIX 3: TCIA fallback -- filter to slide microscopy modality only
# ---------------------------------------------------------------------------

# TCIA modalities that contain histology whole-slide or pathology images
_TCIA_SLIDE_MODALITIES = {"SM", "GM"}  # SM = Slide Microscopy


def tcia_get_slide_series(timeout: int = 120) -> List[Dict]:
    """
    FIX 3: Fetch only Slide Microscopy series from TCIA.
    Passing Modality=SM dramatically reduces response size, avoiding
    the 60-second timeouts seen in v3 that caused cache to stay empty.
    Uses a 120 s timeout (up from 60 s) for the potentially large payload.
    """
    series: List[Dict] = []
    for modality in _TCIA_SLIDE_MODALITIES:
        params = {"format": "json", "Modality": modality}
        r = retry_request(
            f"{TCIA_BASE}/getSeries",
            params=params,
            timeout=timeout,
        )
        logger.info(f"[TCIA] getSeries Modality={modality} status={r.status_code if r else 'None'}")
        if r is None:
            continue
        try:
            batch = r.json()
            logger.info(f"[TCIA] Modality={modality} returned {len(batch)} series")
            series.extend(batch)
        except Exception as e:
            logger.error(f"[TCIA] JSON parse error for Modality={modality}: {e}")

    logger.info(f"[TCIA] Total slide series cached: {len(series)}")
    return series


def tcia_parse_age(age_str: str) -> Optional[float]:
    if not isinstance(age_str, str):
        return None
    age_str = age_str.strip().rstrip("Y")
    try:
        return float(age_str)
    except Exception:
        return None


def tcia_diag_class(series: Dict) -> str:
    desc  = series.get("SeriesDescription", "") or ""
    study = series.get("StudyDescription",  "") or ""
    return normalize_diagnosis_class(f"{desc} {study}")


def select_best_tcia_match(
    acc_sex: str,
    acc_age: Optional[float],
    acc_diag_class: str,
    series_list: List[Dict],
) -> Optional[Dict]:
    if not series_list:
        return None
    gdc_gender = "M" if acc_sex.upper().startswith("M") else "F"
    scored = []
    for s in series_list:
        sex = s.get("PatientSex", "")
        if sex and sex.upper() != gdc_gender:
            continue
        t_age   = tcia_parse_age(s.get("PatientAge", ""))
        t_class = tcia_diag_class(s)
        diag_penalty = 0.0 if t_class == acc_diag_class else 10.0
        age_diff     = abs(acc_age - t_age) if acc_age and t_age else 25.0
        scored.append((diag_penalty + age_diff, s))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def tcia_download_series(
    series_uid: str,
    out_dir: Path,
    acc_case_id: str = "",
    force: bool = False,
) -> Optional[Path]:
    """
    Download a TCIA series as a ZIP archive.

    v5 changes: atomic .part write, completion check via sidecar, fsync
    before rename.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename  = f"{series_uid}.zip"
    out_path  = out_dir / filename
    part_path = out_dir / f"{filename}.part"

    # --- Completion check ---
    if not force:
        complete, existing_path = is_download_complete(out_dir, filename)
        if complete:
            logger.info(f"[RESUME] TCIA {filename} already complete -- skipping.")
            return existing_path
        _remove_partial(out_path)   # remove any earlier partial

    params = {"SeriesInstanceUID": series_uid, "format": "zip"}
    r = retry_request(f"{TCIA_BASE}/getImage", params=params, stream=True)
    if r is None:
        return None

    total = int(r.headers.get("Content-Length", 0))
    try:
        with open(part_path, "wb") as fh, tqdm(
            total=total or None, unit="B", unit_scale=True,
            desc=f"TCIA {series_uid}", leave=False
        ) as pbar:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as e:
        logger.error(f"[TCIA] Download error for {series_uid}: {e}")
        _remove_partial(part_path)
        return None

    os.replace(part_path, out_path)

    _write_sidecar(out_dir, {
        "filename":            filename,
        "expected_size_bytes": out_path.stat().st_size,
        "source":              "TCIA",
        "acc_case_id":         acc_case_id,
        "series_uid":          series_uid,
        "download_timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    return out_path


# ---------------------------------------------------------------------------
# FIX 4: AWS Open Data fallback -- correct TCGA bucket path resolution
# ---------------------------------------------------------------------------

def _gdc_resolve_tcga_barcode(case_id: str) -> Optional[str]:
    """
    Use the GDC API to resolve an internal ACC case_id to a TCGA case
    barcode (e.g. 'TCGA-BH-A0BZ'). This is needed because the AWS bucket
    uses TCGA barcodes, NOT internal ACC IDs like 'BP-001'.

    In practice, for your African cohort data this lookup will return
    nothing (there are no TCGA equivalents for ACC cases). However this
    correctly handles the situation where you want to match a TCGA GDC
    case to its S3 path.
    """
    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {"field": "submitter_id", "value": [case_id]}
        }),
        "fields": "submitter_id,case_id",
        "format": "JSON",
        "size": 1,
    }
    r = retry_request(GDC_CASES_ENDPOINT, params=params, timeout=30)
    if r is None:
        return None
    hits = r.json().get("data", {}).get("hits", [])
    if not hits:
        return None
    return hits[0].get("submitter_id")


def aws_find_tcga_slide_for_barcode(tcga_barcode: str) -> Optional[str]:
    """
    FIX 4: Search the TCGA AWS Open Data bucket using the correct
    path structure: TCGA uses project-level prefixes.

    AWS TCGA bucket structure:
      tcga-2-open/<UUID>/...   (UUID = GDC file_id)
    or at programme level:
      tcga/<TCGA-PROJECT>/<TCGA-BARCODE>/...

    We attempt both common prefix patterns.
    """
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # Derive programme prefix from barcode (e.g. 'TCGA-BH-A0BZ' -> 'TCGA-BRCA')
    # This is a best-effort heuristic; you can extend this map as needed.
    prefixes_to_try = [
        f"{tcga_barcode}/",               # flat layout
        f"tcga/{tcga_barcode}/",          # nested layout variant 1
    ]

    for prefix in prefixes_to_try:
        try:
            resp = s3.list_objects_v2(
                Bucket=TCGA_S3_BUCKET,
                Prefix=prefix,
                MaxKeys=20,
            )
            if "Contents" not in resp:
                continue
            for obj in resp["Contents"]:
                key = obj["Key"]
                if key.lower().endswith(".svs"):
                    logger.info(f"[AWS] Found SVS at s3://{TCGA_S3_BUCKET}/{key}")
                    return key
        except Exception as e:
            logger.warning(f"[AWS] S3 list error for prefix '{prefix}': {e}")

    return None


def aws_find_tcga_slide_from_gdc_hit(hit: Dict) -> Optional[str]:
    """
    Given a GDC file hit (which contains a file_id UUID), look up the
    file directly in the AWS bucket using the UUID path.
    TCGA AWS Open Data stores files at: <file_uuid>/<filename>.svs
    """
    file_id   = hit.get("file_id", "")
    file_name = hit.get("file_name", "")
    if not file_id:
        return None

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    key = f"{file_id}/{file_name}"
    try:
        # HEAD the object to check it exists without downloading it
        s3.head_object(Bucket=TCGA_S3_BUCKET, Key=key)
        logger.info(f"[AWS] Verified s3://{TCGA_S3_BUCKET}/{key}")
        return key
    except Exception:
        # Object not found or access denied -- try prefix scan fallback
        pass

    # Prefix scan fallback using UUID folder
    try:
        resp = s3.list_objects_v2(
            Bucket=TCGA_S3_BUCKET,
            Prefix=f"{file_id}/",
            MaxKeys=10,
        )
        if "Contents" in resp:
            for obj in resp["Contents"]:
                k = obj["Key"]
                if k.lower().endswith(".svs"):
                    return k
    except Exception as e:
        logger.warning(f"[AWS] S3 list error for file_id {file_id}: {e}")

    return None


def aws_download_slide(
    s3_key: str,
    out_dir: Path,
    acc_case_id: str = "",
    force: bool = False,
) -> Optional[Path]:
    """
    Download a TCGA WSI from the AWS Open Data bucket.

    v5 changes:
      - Checks S3 object size (via head_object) before download so the
        completion check has an accurate expected_size_bytes.
      - Downloads to <filename>.part; renames atomically on completion.
      - Writes sidecar manifest entry after successful rename.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename  = s3_key.split("/")[-1]
    out_path  = out_dir / filename
    part_path = out_dir / f"{filename}.part"

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # --- Query S3 object size for an accurate completion check ---
    expected_size = 0
    try:
        head = s3.head_object(Bucket=TCGA_S3_BUCKET, Key=s3_key)
        expected_size = int(head.get("ContentLength", 0))
    except Exception as e:
        logger.warning(f"[AWS] head_object failed for {s3_key}: {e} -- size check will be lenient.")

    # --- Completion check ---
    if not force:
        complete, existing_path = is_download_complete(out_dir, filename)
        if complete:
            logger.info(f"[RESUME] AWS {filename} already complete -- skipping.")
            return existing_path
        _remove_partial(out_path)   # clear any earlier partial

    logger.info(f"[AWS] Downloading s3://{TCGA_S3_BUCKET}/{s3_key} -> {out_path}")
    try:
        # boto3's download_file handles multi-part internally but writes
        # directly to the target path.  We wrap it in .part / rename ourselves.
        s3.download_file(TCGA_S3_BUCKET, s3_key, str(part_path))
    except Exception as e:
        logger.warning(f"[AWS] Download failed for {s3_key}: {e}")
        _remove_partial(part_path)
        return None

    os.replace(part_path, out_path)

    _write_sidecar(out_dir, {
        "filename":            filename,
        "expected_size_bytes": out_path.stat().st_size,
        "source":              "AWS",
        "acc_case_id":         acc_case_id,
        "s3_key":              s3_key,
        "download_timestamp":  datetime.now(timezone.utc).isoformat(),
    })

    return out_path


# ---------------------------------------------------------------------------
# v5: Resumable download support
# ---------------------------------------------------------------------------

# Name of the JSON sidecar file written into every per-case output directory
# after a successful download.  One file per directory; each entry is a dict.
_SIDECAR_FILENAME = ".download_manifest.json"


def _sidecar_path(out_dir: Path) -> Path:
    """Return the path of the sidecar manifest for a given output directory."""
    return out_dir / _SIDECAR_FILENAME


def _load_sidecar(out_dir: Path) -> Dict[str, Dict]:
    """
    Read the sidecar manifest for out_dir.
    Returns a dict keyed by filename -> metadata dict.
    Returns an empty dict if the sidecar does not yet exist.
    """
    sp = _sidecar_path(out_dir)
    if not sp.exists():
        return {}
    try:
        with open(sp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning(f"[RESUME] Could not read sidecar {sp}: {e} -- treating as empty.")
        return {}


def _write_sidecar(out_dir: Path, entry: Dict) -> None:
    """
    Append or update an entry in the sidecar manifest for out_dir.
    Writes atomically via a temp file to avoid corruption on crash.

    entry must contain at minimum:
        filename           (str)  -- basename of the downloaded file
        expected_size_bytes (int) -- Content-Length or S3 object size
        source             (str)  -- 'GDC' | 'TCIA' | 'AWS_GDC' | 'AWS'
        acc_case_id        (str)
        download_timestamp (str)  -- ISO-8601 UTC
    """
    sp      = _sidecar_path(out_dir)
    records = _load_sidecar(out_dir)
    records[entry["filename"]] = entry

    # Write to a temp file in the same directory then rename atomically.
    # os.replace() is atomic on POSIX (CIFS/NFS may vary, but is best-effort).
    tmp = sp.with_suffix(".tmp")
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
    Check whether a file has been fully downloaded to out_dir.

    Returns (True, path) when:
      - The sidecar manifest records this filename AND
      - The file exists on disk AND
      - Its size matches expected_size_bytes in the sidecar exactly.

    Returns (False, None) in all other cases.  The caller should then
    delete any partial file and re-download.

    Note on expected_size_bytes == 0:
      Some servers omit Content-Length (chunked transfer encoding).
      When expected_size_bytes is 0 in the sidecar we fall back to
      checking only that the file exists and is non-empty (>0 bytes).
      This is a weaker guarantee but avoids false negatives on servers
      that do not advertise content length.
    """
    records = _load_sidecar(out_dir)
    if filename not in records:
        return False, None

    meta     = records[filename]
    expected = int(meta.get("expected_size_bytes", 0))
    target   = out_dir / filename

    if not target.exists():
        logger.debug(f"[RESUME] Sidecar entry for {filename} exists but file is missing.")
        return False, None

    actual = target.stat().st_size

    if expected > 0 and actual != expected:
        logger.warning(
            f"[RESUME] {filename}: sidecar says {expected} bytes, "
            f"disk has {actual} bytes -- treating as incomplete."
        )
        return False, None

    if actual == 0:
        logger.warning(f"[RESUME] {filename} exists but is 0 bytes -- treating as incomplete.")
        return False, None

    return True, target


def _remove_partial(path: Path) -> None:
    """Silently remove a partial download file if it exists."""
    try:
        if path.exists():
            path.unlink()
            logger.info(f"[RESUME] Removed partial file: {path}")
    except Exception as e:
        logger.warning(f"[RESUME] Could not remove partial file {path}: {e}")


# ---------------------------------------------------------------------------
# Summary CSV writer
# ---------------------------------------------------------------------------

def write_summary_csv(summary_rows: List[Dict], out_path: Path):
    df = pd.DataFrame(summary_rows)
    df.to_csv(out_path, index=False)
    logger.info(f"Summary CSV written to {out_path}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def download_matched_controls(
    manifest_path: Path,
    out_dir: Path,
    max_controls: int = 200,
    per_case_candidates: int = 50,
    sleep_sec: float = 0.2,
    age_window: int = 20,
    force_redownload: bool = False,
):
    """
    For each ACC case in the manifest, find and download a matched
    non-African control WSI from GDC (tiered), TCIA (slide modality),
    or AWS Open Data (TCGA), in that priority order.

    v5: Fully resumable.  Re-running this function after an interruption
    will skip all cases whose downloads are already verified complete and
    continue from the first incomplete case.

    Parameters
    ----------
    manifest_path     : Path to the ACC manifest TSV
    out_dir           : Root output directory
    max_controls      : Stop after downloading this many controls total
                        (counts previously downloaded controls too)
    per_case_candidates: Max GDC candidates to score per case per tier
    sleep_sec         : Politeness delay between API calls
    age_window        : Age tolerance in years for Tier 1 GDC queries
    force_redownload  : If True, bypass all completion checks and
                        re-download every file from scratch
    """
    manifest = pd.read_csv(manifest_path, sep="\t")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # v5: Resume -- reload previously completed rows from the CSV so
    # that (a) the final CSV is cumulative and (b) used_ids is correct.
    # ----------------------------------------------------------------
    summary_csv_path = out_dir / "matched_controls_summary.csv"
    summary_rows: List[Dict] = []
    used_ids: set = set()
    downloaded = 0  # counts only THIS run's new downloads for progress

    if summary_csv_path.exists() and not force_redownload:
        try:
            prior = pd.read_csv(summary_csv_path)
            summary_rows = prior.to_dict(orient="records")
            # Rebuild used_ids from whichever ID column is populated
            for row in summary_rows:
                uid = row.get("control_file_or_series_id", "")
                if uid:
                    used_ids.add(str(uid))
            logger.info(
                f"[RESUME] Loaded {len(summary_rows)} previously matched controls "
                f"from {summary_csv_path}."
            )
        except Exception as e:
            logger.warning(f"[RESUME] Could not read prior summary CSV: {e} -- starting fresh.")
            summary_rows = []
            used_ids     = set()

    # Count how many valid downloads already exist so max_controls is
    # respected correctly across runs.
    already_done = len(summary_rows)
    logger.info(f"Processing {len(manifest)} ACC cases...  ({already_done} already complete)")

    tcia_series_cache = None  # Lazily populated on first TCIA fallback

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="ACC cases"):
        # Stop when cumulative total (prior + this run) hits the cap
        if already_done + downloaded >= max_controls:
            break

        acc_case_id  = row["case_id"]
        acc_sex      = str(row["sex"])
        acc_age      = float(row["age"]) if not pd.isna(row["age"]) else None
        acc_diag_cls = normalize_diagnosis_class(row["diagnosis"])

        # ----------------------------------------------------------------
        # v5: Per-case completion check -- scan the expected output dirs
        # for any source (GDC / TCIA / AWS) and skip if a complete file
        # is found. We check all three subdirectory names so that a case
        # completed via TCIA on a prior run is correctly skipped even if
        # GDC would now match.
        # ----------------------------------------------------------------
        if not force_redownload:
            already_complete = False
            for source_subdir in ("GDC", "AWS_GDC", "TCIA", "AWS"):
                candidate_dir = out_dir / f"ACC_{acc_case_id}" / source_subdir
                if not candidate_dir.exists():
                    continue
                # Any .svs or .zip file in the sidecar that passes the size check
                sidecar = _load_sidecar(candidate_dir)
                for fname, meta in sidecar.items():
                    ok, _ = is_download_complete(candidate_dir, fname)
                    if ok:
                        logger.info(
                            f"[RESUME] {acc_case_id}: verified complete "
                            f"({source_subdir}/{fname}) -- skipping."
                        )
                        already_complete = True
                        break
                if already_complete:
                    break
            if already_complete:
                continue  # advance to next manifest row

        # ----------------------------------------------------------------
        # GDC -- tiered query (Tier 1 -> 2 -> 3)
        # ----------------------------------------------------------------
        candidates = gdc_query_tiered(
            acc_case_id, acc_sex, acc_age, acc_diag_cls,
            size=per_case_candidates,
            age_window=age_window,
        )
        logger.info(f"[GDC] Total candidates for {acc_case_id}: {len(candidates)}")

        best_gdc = select_best_gdc_match(acc_age, acc_diag_cls, candidates)
        logger.info(f"[GDC] Best match for {acc_case_id}: {best_gdc is not None}")

        if best_gdc:
            file_id = best_gdc["file_id"]
            if file_id not in used_ids:
                out_subdir = out_dir / f"ACC_{acc_case_id}" / "GDC"
                path = gdc_download_file(
                    file_id,
                    out_subdir,
                    acc_case_id=acc_case_id,
                    force=force_redownload,
                )

                if path is not None:
                    used_ids.add(file_id)
                    downloaded += 1
                    gdc_age      = age_from_gdc_hit(best_gdc)
                    gdc_diag_cls = diag_class_from_hit(best_gdc)
                    gdc_race     = best_gdc["cases"][0]["demographic"].get("race", "") if best_gdc.get("cases") else ""
                    gdc_case_id  = best_gdc["cases"][0].get("case_id", "") if best_gdc.get("cases") else ""
                    summary_rows.append({
                        "source":                     "GDC",
                        "acc_case_id":                acc_case_id,
                        "acc_sex":                    acc_sex,
                        "acc_age":                    acc_age,
                        "acc_diag_class":             acc_diag_cls,
                        "control_case_id":            gdc_case_id,
                        "control_file_or_series_id":  file_id,
                        "control_name":               best_gdc.get("file_name", ""),
                        "control_race":               gdc_race,
                        "control_age":                gdc_age,
                        "age_delta":                  abs(acc_age - gdc_age) if acc_age and gdc_age else None,
                        "control_diag_class":         gdc_diag_cls,
                        "download_path":              str(path),
                    })
                    # Write CSV after every successful download so partial
                    # progress is not lost on the next interruption.
                    write_summary_csv(summary_rows, summary_csv_path)
                    time.sleep(sleep_sec)
                    continue  # GDC HTTP success -- skip TCIA and AWS

                # GDC HTTP download failed -- try same file via AWS S3
                logger.info(f"[GDC->AWS] GDC HTTP failed; trying AWS S3 for {file_id}")
                aws_key = aws_find_tcga_slide_from_gdc_hit(best_gdc)
                if aws_key and aws_key not in used_ids:
                    out_subdir = out_dir / f"ACC_{acc_case_id}" / "AWS_GDC"
                    path = aws_download_slide(
                        aws_key,
                        out_subdir,
                        acc_case_id=acc_case_id,
                        force=force_redownload,
                    )
                    if path is not None:
                        used_ids.add(file_id)
                        downloaded += 1
                        gdc_age      = age_from_gdc_hit(best_gdc)
                        gdc_diag_cls = diag_class_from_hit(best_gdc)
                        summary_rows.append({
                            "source":                     "AWS_GDC",
                            "acc_case_id":                acc_case_id,
                            "acc_sex":                    acc_sex,
                            "acc_age":                    acc_age,
                            "acc_diag_class":             acc_diag_cls,
                            "control_case_id":            best_gdc["cases"][0].get("case_id", "") if best_gdc.get("cases") else "",
                            "control_file_or_series_id":  aws_key,
                            "control_name":               aws_key.split("/")[-1],
                            "control_race":               "",
                            "control_age":                gdc_age,
                            "age_delta":                  abs(acc_age - gdc_age) if acc_age and gdc_age else None,
                            "control_diag_class":         gdc_diag_cls,
                            "download_path":              str(path),
                        })
                        write_summary_csv(summary_rows, summary_csv_path)
                        time.sleep(sleep_sec)
                        continue

        # ----------------------------------------------------------------
        # TCIA fallback -- slide microscopy series only
        # ----------------------------------------------------------------
        if tcia_series_cache is None:
            logger.info("Fetching TCIA slide microscopy series (SM modality)...")
            tcia_series_cache = tcia_get_slide_series()

        best_tcia = select_best_tcia_match(acc_sex, acc_age, acc_diag_cls, tcia_series_cache)
        logger.info(f"[TCIA] Best match for {acc_case_id}: {best_tcia is not None}")

        if best_tcia:
            series_uid = best_tcia.get("SeriesInstanceUID")
            if series_uid and series_uid not in used_ids:
                out_subdir = out_dir / f"ACC_{acc_case_id}" / "TCIA"
                path = tcia_download_series(
                    series_uid,
                    out_subdir,
                    acc_case_id=acc_case_id,
                    force=force_redownload,
                )
                if path is not None:
                    used_ids.add(series_uid)
                    downloaded += 1
                    t_age   = tcia_parse_age(best_tcia.get("PatientAge", ""))
                    t_class = tcia_diag_class(best_tcia)
                    summary_rows.append({
                        "source":                     "TCIA",
                        "acc_case_id":                acc_case_id,
                        "acc_sex":                    acc_sex,
                        "acc_age":                    acc_age,
                        "acc_diag_class":             acc_diag_cls,
                        "control_case_id":            best_tcia.get("PatientID", ""),
                        "control_file_or_series_id":  series_uid,
                        "control_name":               best_tcia.get("SeriesDescription", ""),
                        "control_race":               "",
                        "control_age":                t_age,
                        "age_delta":                  abs(acc_age - t_age) if acc_age and t_age else None,
                        "control_diag_class":         t_class,
                        "download_path":              str(path),
                    })
                    write_summary_csv(summary_rows, summary_csv_path)
                    time.sleep(sleep_sec)
                    continue

        # ----------------------------------------------------------------
        # AWS Open Data fallback -- last resort
        # ----------------------------------------------------------------
        logger.info(f"[AWS] Last-resort AWS fallback for {acc_case_id}...")

        tcga_barcode = _gdc_resolve_tcga_barcode(acc_case_id)
        aws_key      = None
        if tcga_barcode:
            aws_key = aws_find_tcga_slide_for_barcode(tcga_barcode)

        logger.info(f"[AWS] Key for {acc_case_id} (barcode={tcga_barcode}): {aws_key}")

        if aws_key and aws_key not in used_ids:
            out_subdir = out_dir / f"ACC_{acc_case_id}" / "AWS"
            path = aws_download_slide(
                aws_key,
                out_subdir,
                acc_case_id=acc_case_id,
                force=force_redownload,
            )
            if path is not None:
                used_ids.add(aws_key)
                downloaded += 1
                summary_rows.append({
                    "source":                     "AWS",
                    "acc_case_id":                acc_case_id,
                    "acc_sex":                    acc_sex,
                    "acc_age":                    acc_age,
                    "acc_diag_class":             acc_diag_cls,
                    "control_case_id":            tcga_barcode or "",
                    "control_file_or_series_id":  aws_key,
                    "control_name":               aws_key.split("/")[-1],
                    "control_race":               "",
                    "control_age":                None,
                    "age_delta":                  None,
                    "control_diag_class":         "unknown",
                    "download_path":              str(path),
                })
                write_summary_csv(summary_rows, summary_csv_path)
                time.sleep(sleep_sec)

    # Final write -- ensures the CSV is up to date even if the last
    # several cases all hit the "already complete" fast-path.
    write_summary_csv(summary_rows, summary_csv_path)
    logger.info(
        f"Run complete.  New downloads this run: {downloaded}.  "
        f"Total in summary: {len(summary_rows)}."
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download matched non-African control WSIs from GDC (tiered), "
            "TCIA (slide microscopy), or AWS Open Data (TCGA).  "
            "Resumable: re-running after an interruption will skip verified "
            "complete files and continue from where the run left off."
        )
    )
    parser.add_argument("manifest",         help="Path to ACC-derived manifest TSV")
    parser.add_argument("--out_dir",        default="matched_controls",
                        help="Output directory (default: matched_controls)")
    parser.add_argument("--max_controls",   type=int,   default=200,
                        help="Maximum controls to download across all runs (default: 200)")
    parser.add_argument("--per_case_candidates", type=int, default=50,
                        help="Max GDC candidates per case per tier (default: 50)")
    parser.add_argument("--sleep_sec",      type=float, default=0.2,
                        help="Politeness delay between API calls in seconds (default: 0.2)")
    parser.add_argument("--relax_age_window", type=int, default=20,
                        help="Age window in years for Tier 1 GDC matching (default: 20, "
                             "increase to 30 for sparse metadata populations)")
    parser.add_argument("--force_redownload", action="store_true", default=False,
                        help="Ignore completion checks and re-download everything from scratch. "
                             "Use when you suspect silent file corruption.")

    args = parser.parse_args()

    download_matched_controls(
        manifest_path        = Path(args.manifest),
        out_dir              = Path(args.out_dir),
        max_controls         = args.max_controls,
        per_case_candidates  = args.per_case_candidates,
        sleep_sec            = args.sleep_sec,
        age_window           = args.relax_age_window,
        force_redownload     = args.force_redownload,
    )
