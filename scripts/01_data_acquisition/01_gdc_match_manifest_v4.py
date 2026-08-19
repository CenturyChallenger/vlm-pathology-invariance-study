#!/usr/bin/env python3
"""
gdc_match_manifest.py  -- Phase 1 of the two-phase download pipeline.

PURPOSE
-------
Query the GDC metadata API to find the best-matching TCGA control slide for
each ACC case in the manifest, then write a download manifest TSV containing
the GDC file_id UUID and filename for each match.

This script performs NO file downloads.  Every call is a small JSON API
request (< 1 KB response for metadata, < 50 KB for candidate lists).  The
entire 201-case manifest can be processed in roughly 10-15 minutes on an
interactive HCI session without any risk of hitting the 1-minute timeout.

The output TSV is consumed by gcs_slide_downloader.py (Phase 2), which
runs as a long Slurm batch job and downloads the matched SVS files from the
ISB-CGC open-access GCS bucket -- no GDC token or Google account required.

OUTPUT FORMAT (matched_controls_manifest.tsv)
---------------------------------------------
acc_case_id  acc_sex  acc_age  acc_diag_class  tier_matched
file_id      file_name  control_case_id  control_race
control_age  age_delta  control_diag_class
gcs_url      status

The gcs_url column contains the full HTTPS URL to the SVS file in the
ISB-CGC public GCS bucket:
  https://storage.googleapis.com/gdc-tcga-phs000178-open/<file_id>/<file_name>

Cases with no GDC match across all three tiers are written with
status="NO_MATCH" so they are visible and auditable.

USAGE
-----
python gdc_match_manifest.py acc_manifest.tsv \
    --out matched_controls_manifest.tsv \
    --age_window 20 \
    --per_case_candidates 50

Author: Phillip Nyamwaya
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GDC_BASE           = "https://api.gdc.cancer.gov"
GDC_FILES_ENDPOINT = f"{GDC_BASE}/files"

# GCS open-access bucket for TCGA slides (used as fallback if DRS is unreachable)
# The authoritative URL for any file is now resolved via the DRS API (see resolve_gcs_url)
GCS_BUCKET_BASE = "https://storage.googleapis.com/gdc-tcga-phs000178-open"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("gdc_match_manifest")


# ---------------------------------------------------------------------------
# HTTP retry wrapper (metadata calls only -- no streaming, no auth needed)
# ---------------------------------------------------------------------------

def retry_get(
    url: str,
    params: dict = None,
    max_retries: int = 5,
    timeout: int = 30,
) -> Optional[requests.Response]:
    """
    GET with exponential backoff.  Used only for GDC metadata API calls
    (not data downloads), so timeouts are short and no auth header is needed.
    401/403 on the metadata endpoint would indicate a misconfiguration;
    log and abort immediately rather than retrying.
    """
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in (401, 403):
                logger.error(f"[AUTH] HTTP {r.status_code} for {url} -- unexpected on metadata endpoint.")
                return None
            logger.warning(f"Attempt {attempt}: HTTP {r.status_code} for {url}")
        except Exception as e:
            logger.warning(f"Attempt {attempt}: {e}")
        time.sleep(delay)
        delay *= 2
    logger.error(f"All {max_retries} attempts failed for {url}")
    return None


# ---------------------------------------------------------------------------
# Diagnosis normalisation
# ---------------------------------------------------------------------------

def normalize_diagnosis_class(diag: str) -> str:
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
# Prefix -> GDC primary_site map
# Returns a LIST of site values -- GDC "in" operator accepts multiple values,
# which is important for TP/TN cases where tongue tumours span several TCGA
# primary_site classifications depending on the submitting institution.
# ---------------------------------------------------------------------------

_PREFIX_TO_SITES: Dict[str, List[str]] = {
    "BP": ["Breast"],
    "BN": ["Breast"],
    "CP": ["Cervix Uteri"],
    "CN": ["Cervix Uteri"],
    "RP": ["Colon"],
    "RN": ["Colon"],
    # FIX 2: TP/TN (tongue/oral) cases -- TCGA distributes these across three
    # primary_site values.  Querying all three simultaneously maximises the
    # candidate pool from TCGA-HNSC without sacrificing anatomical specificity.
    "TP": ["Head and Neck", "Oral Cavity", "Oropharynx"],
    "TN": ["Head and Neck", "Oral Cavity", "Oropharynx"],
}


def infer_primary_sites(case_id: str) -> Optional[List[str]]:
    """Return list of GDC primary_site values for this case ID prefix, or None."""
    prefix = case_id[:2].upper() if isinstance(case_id, str) and len(case_id) >= 2 else ""
    return _PREFIX_TO_SITES.get(prefix)


# Keep the singular alias for backward compatibility with any callers
def infer_primary_site(case_id: str) -> Optional[str]:
    sites = infer_primary_sites(case_id)
    return sites[0] if sites else None


# ---------------------------------------------------------------------------
# GDC filter builders (three tiers, progressively looser)
# ---------------------------------------------------------------------------

def _base_filter() -> List[Dict]:
    """
    SVS format only.

    RACE EXCLUSION -- WHY IT IS NO LONGER IN THE API QUERY
    -------------------------------------------------------
    All previous versions attempted to exclude Black/African American donors
    at the GDC query level using either:
      (a) {"op": "not_in", "content": {"field": "cases.demographic.race", ...}}
      (b) An OR of not_in + {"op": "is", "content": {"value": "missing"}}

    Both fail for TCGA-HNSC and any programme with incomplete race metadata.

    (a) "not_in" on a nested field in GDC only returns records where the
        field carries an explicit non-matching value.  Records where race is
        null, unrecorded, or where the parent demographic entity is absent
        are silently dropped, not included.  This eliminated all of TCGA-HNSC
        (where a large fraction of cases have no recorded race) and produced
        zero candidates for all 26 TP cases across three tiers.

    (b) The GDC API release notes explicitly document that "is missing" does
        not work correctly for nested fields:
        "Fields are not counted as missing if parent field is also missing.
         This may occur with queries of nested fields in the Data Portal
         Advanced Search or an API query using a filter. This behaviour
         could impact results reported using search parameters of
         'IS MISSING' or 'NOT MISSING'."
        Source: https://docs.gdc.cancer.gov/API/Release_Notes/API_Release_Notes/

    CORRECT APPROACH: query without any race filter, retrieve the race value
    in the fields response, then exclude known Black/African American donors
    in Python after retrieval (see filter_candidates_by_race()).  This is
    reliable regardless of metadata completeness, and records with null race
    are treated as acceptable (unknown, not known-African-ancestry).
    """
    return [
        {"op": "in", "content": {"field": "data_format", "value": ["SVS"]}},
    ]


# Race values in GDC that indicate African ancestry -- excluded post-retrieval
_EXCLUDED_RACES = {"black or african american", "african american"}


def filter_candidates_by_race(hits: List[Dict]) -> List[Dict]:
    """
    Post-retrieval race filter.  Removes candidates whose race field is
    explicitly set to a known African-ancestry value.  Candidates with
    null, unrecorded, or 'not reported' race are KEPT -- they are not
    known African-ancestry donors, just donors with incomplete metadata.

    This is called after _query() returns results, before select_best().
    """
    kept = []
    for h in hits:
        cases = h.get("cases", [])
        race  = ""
        if cases:
            demo = cases[0].get("demographic", {}) or {}
            race = (demo.get("race", "") or "").strip().lower()
        if race in _EXCLUDED_RACES:
            continue
        kept.append(h)
    return kept


def _tier1_filter(case_id, sex, age, age_window) -> Optional[Dict]:
    sites = infer_primary_sites(case_id)
    if not sites:
        return None
    content = _base_filter() + [
        # FIX 2: pass the full list of sites -- GDC "in" matches any of them
        {"op": "in", "content": {"field": "cases.primary_site", "value": sites}},
    ]
    if isinstance(sex, str) and sex.strip():
        content.append({"op": "in", "content": {
            "field": "cases.demographic.gender",
            "value": ["male" if sex.upper().startswith("M") else "female"],
        }})
    if age is not None and not pd.isna(age):
        content.append({"op": "between", "content": {
            "field": "cases.diagnoses.age_at_diagnosis",
            "value": [int(max(0, (age - age_window) * 365)), int((age + age_window) * 365)],
        }})
    return {"op": "and", "content": content}


def _tier2_filter(case_id, sex) -> Optional[Dict]:
    sites = infer_primary_sites(case_id)
    if not sites:
        return None
    content = _base_filter() + [
        {"op": "in", "content": {"field": "cases.primary_site", "value": sites}},
    ]
    if isinstance(sex, str) and sex.strip():
        content.append({"op": "in", "content": {
            "field": "cases.demographic.gender",
            "value": ["male" if sex.upper().startswith("M") else "female"],
        }})
    return {"op": "and", "content": content}


def _tier3_filter(case_id) -> Optional[Dict]:
    sites = infer_primary_sites(case_id)
    if not sites:
        return None
    return {"op": "and", "content": _base_filter() + [
        {"op": "in", "content": {"field": "cases.primary_site", "value": sites}},
    ]}


# ---------------------------------------------------------------------------
# GDC candidate query
# ---------------------------------------------------------------------------

_GDC_FIELDS = [
    "file_id", "file_name",
    "cases.case_id", "cases.submitter_id",
    "cases.demographic.race", "cases.demographic.gender",
    "cases.diagnoses.age_at_diagnosis",
    "cases.diagnoses.primary_diagnosis",
    "cases.primary_site",
]


def _query(filters: Dict, size: int) -> List[Dict]:
    r = retry_get(GDC_FILES_ENDPOINT, params={
        "filters": json.dumps(filters),
        "fields":  ",".join(_GDC_FIELDS),
        "format":  "JSON",
        "size":    size,
    })
    if r is None:
        return []
    try:
        return r.json().get("data", {}).get("hits", [])
    except Exception as e:
        logger.error(f"JSON parse error: {e}")
        return []


def query_tiered(case_id, sex, age, diag_class, size, age_window) -> tuple:
    """
    Run tier 1 -> 2 -> 3 in order.
    Returns (hits, tier_label) where tier_label is "T1", "T2", "T3", or "NONE".

    Race exclusion is applied in Python after retrieval (not in the GDC query)
    because GDC's nested-field "is missing" operator does not work reliably for
    cases.demographic.race -- see _base_filter() docstring for the full diagnosis.

    We request 2x the requested size from GDC to ensure enough candidates remain
    after the post-retrieval race filter, since a fraction will be excluded.
    """
    fetch_size = min(size * 2, 200)   # fetch extra to absorb race-filter attrition

    for tier_label, f in [
        ("T1", _tier1_filter(case_id, sex, age, age_window)),
        ("T2", _tier2_filter(case_id, sex)),
        ("T3", _tier3_filter(case_id)),
    ]:
        if f is None:
            continue
        raw_hits = _query(f, fetch_size)
        if not raw_hits:
            continue
        hits = filter_candidates_by_race(raw_hits)
        if hits:
            logger.info(
                f"[GDC] {case_id}: {len(hits)} candidates at {tier_label} "
                f"(from {len(raw_hits)} raw, {len(raw_hits)-len(hits)} race-excluded)"
            )
            return hits, tier_label
        else:
            logger.info(
                f"[GDC] {case_id}: {tier_label} returned {len(raw_hits)} raw hits "
                f"but all excluded by race filter -- trying next tier"
            )
    logger.info(f"[GDC] {case_id}: no candidates across all tiers")
    return [], "NONE"


# ---------------------------------------------------------------------------
# Candidate scoring and best-match selection
# ---------------------------------------------------------------------------

def _hit_age(hit: Dict) -> Optional[float]:
    cases = hit.get("cases", [])
    if not cases:
        return None
    diags = cases[0].get("diagnoses", [])
    if not diags:
        return None
    days = diags[0].get("age_at_diagnosis")
    return days / 365.0 if days else None


def _hit_diag_class(hit: Dict) -> str:
    cases = hit.get("cases", [])
    if not cases:
        return "unknown"
    diags = cases[0].get("diagnoses", [])
    if not diags:
        return "unknown"
    return normalize_diagnosis_class(diags[0].get("primary_diagnosis", ""))


def select_best(acc_age, acc_diag_class, candidates) -> Optional[Dict]:
    """
    Score by (diag_class_mismatch_penalty + age_delta).
    Lower is better.  Returns None if candidates is empty.
    """
    if not candidates:
        return None
    scored = []
    for h in candidates:
        hit_age  = _hit_age(h)
        hit_cls  = _hit_diag_class(h)
        penalty  = 0.0 if hit_cls == acc_diag_class else 10.0
        age_diff = abs(acc_age - hit_age) if acc_age and hit_age else 25.0
        scored.append((penalty + age_diff, h))
    scored.sort(key=lambda x: x[0])
    return scored[0][1]


# ---------------------------------------------------------------------------
# GCS URL resolution via GDC DRS (Data Repository Service) API
# ---------------------------------------------------------------------------

# The NCI CRDC DRS endpoint resolves any GDC file UUID to its correct
# GCS and AWS bucket URLs -- regardless of programme (TCGA, HCMI, etc.).
# This is the authoritative, programme-agnostic method to get the GCS path.
# Reference: https://learn.canceridc.dev/data/organization-of-data/guids-and-uuids
#
# The previous approach of hardcoding "gdc-tcga-phs000178-open/<uuid>/<filename>"
# caused 100% failure in Phase 2 because:
#   - HCM-EXPT / HCM-CSHL files live in a DIFFERENT GCS bucket (not phs000178)
#   - Some TCGA file UUIDs were remapped when GDC migrated from legacy archive
#
# The DRS API returns the correct bucket URL for ALL programmes in one call.

DRS_BASE = "https://nci-crdc.datacommons.io/ga4gh/drs/v1/objects"


def resolve_gcs_url(file_id: str, file_name: str) -> str:
    """
    Resolve a GDC file UUID to its public GCS HTTPS download URL via the
    NCI CRDC DRS (Data Repository Service) API.

    Returns the first GCS access_method URL found, or falls back to the
    legacy TCGA open bucket URL if DRS is unreachable (for resilience).

    DRS response structure:
      {
        "access_methods": [
          {"type": "gs",  "access_url": {"url": "gs://gdc-<programme>-open/<uuid>/<name>"}},
          {"type": "s3",  "access_url": {"url": "s3://gdc-aws-.../<uuid>/<name>"}}
        ]
      }

    We convert the gs:// URL to an https://storage.googleapis.com/... URL
    which can be downloaded by plain requests.get() without any SDK.
    """
    url = f"{DRS_BASE}/dg.4DFC%2F{file_id}"
    r   = retry_get(url, timeout=15, max_retries=3)

    if r is not None:
        try:
            methods = r.json().get("access_methods", [])
            for m in methods:
                gs_url = m.get("access_url", {}).get("url", "")
                if gs_url.startswith("gs://"):
                    # Convert  gs://bucket/path  ->  https://storage.googleapis.com/bucket/path
                    return "https://storage.googleapis.com/" + gs_url[5:]
        except Exception as e:
            logger.warning(f"[DRS] Could not parse DRS response for {file_id}: {e}")

    # Fallback: use the TCGA open bucket (correct for TCGA files even if DRS fails)
    logger.warning(
        f"[DRS] Could not resolve GCS URL for {file_id} via DRS -- "
        f"falling back to gdc-tcga-phs000178-open (may 404 for non-TCGA files)."
    )
    return f"https://storage.googleapis.com/gdc-tcga-phs000178-open/{file_id}/{file_name}"


# ---------------------------------------------------------------------------
# Main manifest generation loop
# ---------------------------------------------------------------------------

def generate_manifest(
    manifest_path: Path,
    out_path: Path,
    age_window: int = 20,
    per_case_candidates: int = 50,
    sleep_sec: float = 0.3,
):
    """
    For each ACC case in the manifest, query GDC for the best-matching
    control slide and write the result (including GCS URL) to out_path.

    v3 changes:
      - GCS URL now resolved via DRS API (correct bucket for ALL programmes).
      - Deduplication: each file_id is assigned to at most ONE ACC case.
        A set of used_file_ids grows as matches are made; candidates already
        assigned are excluded from scoring for subsequent cases.
      - Cases already present in the output file are skipped (resume).
    """
    manifest = pd.read_csv(manifest_path, sep="\t")
    logger.info(f"Loaded {len(manifest)} ACC cases from {manifest_path}")

    # Resume: skip cases already written to the output file
    already_matched: set = set()
    used_file_ids: set   = set()

    if out_path.exists():
        try:
            prior = pd.read_csv(out_path, sep="\t")
            # Only MATCHED rows consume a file_id slot
            matched_prior = prior[prior["status"].str.strip().str.upper() == "MATCHED"]
            already_matched = set(prior["acc_case_id"].tolist())
            used_file_ids   = set(matched_prior["file_id"].dropna().tolist())
            logger.info(
                f"[RESUME] {len(already_matched)} cases already in {out_path} -- will skip. "
                f"({len(used_file_ids)} file_ids already consumed)"
            )
        except Exception as e:
            logger.warning(f"Could not read existing output: {e} -- starting fresh.")

    # Open output in append mode so each row is flushed immediately
    write_header = not out_path.exists() or len(already_matched) == 0
    out_fh = open(out_path, "a", encoding="utf-8")

    columns = [
        "acc_case_id", "acc_sex", "acc_age", "acc_diag_class",
        "tier_matched", "file_id", "file_name",
        "control_case_id", "control_race",
        "control_age", "age_delta", "control_diag_class",
        "gcs_url", "status",
    ]

    if write_header:
        out_fh.write("\t".join(columns) + "\n")

    matched   = 0
    no_match  = 0
    skipped   = 0

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Matching cases"):
        acc_case_id  = row["case_id"]
        acc_sex      = str(row["sex"])
        acc_age      = float(row["age"]) if not pd.isna(row["age"]) else None
        acc_diag_cls = normalize_diagnosis_class(row["diagnosis"])

        if acc_case_id in already_matched:
            skipped += 1
            continue

        candidates, tier = query_tiered(
            acc_case_id, acc_sex, acc_age, acc_diag_cls,
            per_case_candidates, age_window,
        )

        # Deduplication: exclude file_ids already assigned to a prior ACC case
        fresh_candidates = [c for c in candidates if c.get("file_id") not in used_file_ids]
        if not fresh_candidates and candidates:
            logger.warning(
                f"[DEDUP] {acc_case_id}: all {len(candidates)} candidates already used "
                f"-- expanding to 200 candidates for a fresh match."
            )
            # Retry with a much larger candidate pool to find an unused slide
            candidates_large, tier = query_tiered(
                acc_case_id, acc_sex, acc_age, acc_diag_cls,
                200, age_window,
            )
            fresh_candidates = [c for c in candidates_large if c.get("file_id") not in used_file_ids]

        best = select_best(acc_age, acc_diag_cls, fresh_candidates)

        if best:
            file_id   = best["file_id"]
            file_name = best.get("file_name", file_id)
            cases     = best.get("cases", [{}])
            demo      = cases[0].get("demographic", {}) if cases else {}
            hit_age   = _hit_age(best)
            hit_cls   = _hit_diag_class(best)
            ctrl_id   = cases[0].get("case_id", "") if cases else ""
            ctrl_race = demo.get("race", "")
            age_delta = abs(acc_age - hit_age) if acc_age and hit_age else ""

            # Resolve the correct GCS URL for this file via DRS
            resolved_url = resolve_gcs_url(file_id, file_name)
            used_file_ids.add(file_id)

            row_values = [
                acc_case_id, acc_sex, acc_age, acc_diag_cls,
                tier, file_id, file_name,
                ctrl_id, ctrl_race,
                hit_age, age_delta, hit_cls,
                resolved_url, "MATCHED",
            ]
            matched += 1
        else:
            # Write a NO_MATCH row so the case is auditable
            row_values = [
                acc_case_id, acc_sex, acc_age, acc_diag_cls,
                "NONE", "", "", "", "", "", "", "",
                "", "NO_MATCH",
            ]
            no_match += 1

        out_fh.write("\t".join(str(v) for v in row_values) + "\n")
        out_fh.flush()
        time.sleep(sleep_sec)

    out_fh.close()

    logger.info(
        f"Done.  Matched: {matched}  |  No match: {no_match}  |  "
        f"Skipped (already done): {skipped}  |  Output: {out_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 1: Query GDC metadata to match each ACC case to a TCGA "
            "control slide and write a download manifest TSV.  No file "
            "downloads are performed.  Safe to run in an interactive HCI "
            "session.  Re-running resumes from where the last run stopped."
        )
    )
    parser.add_argument("manifest",
                        help="Path to ACC manifest TSV (acc_manifest.tsv)")
    parser.add_argument("--out", default="matched_controls_manifest.tsv",
                        help="Output manifest path (default: matched_controls_manifest.tsv)")
    parser.add_argument("--age_window", type=int, default=20,
                        help="Age tolerance in years for Tier 1 matching (default: 20)")
    parser.add_argument("--per_case_candidates", type=int, default=50,
                        help="Max GDC candidates to score per case per tier (default: 50)")
    parser.add_argument("--sleep_sec", type=float, default=0.3,
                        help="Politeness delay between GDC API calls in seconds (default: 0.3)")

    args = parser.parse_args()

    generate_manifest(
        manifest_path        = Path(args.manifest),
        out_path             = Path(args.out),
        age_window           = args.age_window,
        per_case_candidates  = args.per_case_candidates,
        sleep_sec            = args.sleep_sec,
    )
