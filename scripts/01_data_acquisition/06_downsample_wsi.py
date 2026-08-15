#!/usr/bin/env python3
"""
downsample_wsi.py

Reads each WSI at its pyramid level nearest to 20x / 0.5 microns-per-pixel
(MPP), applies a software resize where the native pyramid has no level
close enough to that target, and writes a standardised, tiled TIFF to a
dedicated "downsampled" output tree, one file per slide, covering both the
ACC-Kenya and GDC/TCGA-HCMI control cohorts from a single run.

WHY THIS EXISTS (context for anyone else reading this script):
    The wsi_audit.py run across both cohorts showed that virtually every
    slide (175/175 ACC, 177/190 GDC) has no native pyramid layer within
    0.05 MPP of the 0.5 MPP / 20x target that UNI, CONCH, and Prov-GigaPath
    were pretrained at (and that Quilt-LLaVA's 224px tile size is matched
    to, per the project's magnification-matching methodology note). This
    script performs that resize ONCE per slide and writes a stable,
    standardised working copy, so every later pipeline stage (tile
    extraction, the 18-perturbation suite, per-model inference) reads
    from a single common-resolution source rather than repeating the
    resize inline and risking subtly different resampling per stage.

KEY DESIGN DECISIONS (and why):

1. Output format is a tiled, single-resolution TIFF written via tifffile,
   NOT a re-saved .svs. Aperio's SVS format is a proprietary TIFF variant;
   writing a generic tiled TIFF and naming it .tiff (not .svs) avoids
   silently misrepresenting the file's provenance downstream.

2. Tiles are generated on demand and streamed directly into the TIFF
   writer via a Python generator, never holding the full resized image
   in memory. A naive "read full level -> resize -> write" approach would
   need multiple GB of RAM per slide for the largest files in this
   cohort (up to ~1.9 GB on disk, meaningfully larger once decompressed);
   streaming keeps peak memory roughly constant regardless of slide size.
   This generator-based tiled-write pattern is a documented tifffile
   usage pattern (see tifffile changelog / PyPI examples for writing
   tiled images from a generator with matching `shape` and `tile` args).

3. Every slide is run through the same resize pipeline, including ones
   whose deviation from 0.5 MPP is within tolerance (no dedicated
   "already fine, just copy" branch). The scale factor computed for
   such slides is naturally close to 1.0, so the resize is a no-op in
   practice, and every output file is guaranteed to be in the same
   standardised tiled-TIFF format regardless of source pyramid quirks.

4. Output writes are atomic: each slide is written to a `.tmp` path
   first, then renamed into place only on success. This means a job
   killed mid-write (Slurm timeout, node failure, OOM) never leaves a
   half-written file that a later re-run would mistake for "already
   done" - which matters directly for the skip-existing logic below.

5. Metadata (native MPP, best level, achieved MPP) is recomputed live
   from OpenSlide on every run, not read from a prior audit_report.csv.
   Reading OpenSlide properties is metadata-only and cheap; trusting a
   CSV that could be stale (for example, BP-073 was MISSING_FILE in the
   last audit but may exist by the time this runs) would reintroduce
   exactly the kind of staleness bug this pipeline has already hit once
   with the GDC bucket URLs. The audit CSV is not required as an input.

6. Idempotent / resumable by design: before processing a slide, the
   script checks whether a valid output file already exists at the
   target path (valid = openable by tifffile with a sane page shape,
   not just "a file exists"). If valid, the slide is skipped. This means
   re-running after downloading the remaining ACC files (219tah.svs
   identity resolved, BP-046 vs BP-046F reconciled, BP-073 re-fetched)
   will process only the newly-available slides.

7. Optional Slurm-array sharding (--shard-index / --num-shards) lets the
   ~390-slide workload be split across parallel array tasks, since each
   slide's resize is CPU-bound and independent of every other slide.

Usage (single Slurm batch job, all slides):
    python downsample_wsi.py \
        --manifest matched_controls_manifest_patched.tsv \
        --out downsample_report.csv

Usage (Slurm array job, e.g. #SBATCH --array=0-9, 10 shards):
    python downsample_wsi.py \
        --manifest matched_controls_manifest_patched.tsv \
        --out downsample_report_${SLURM_ARRAY_TASK_ID}.csv \
        --shard-index ${SLURM_ARRAY_TASK_ID} --num-shards 10

Dependencies (install with --break-system-packages if using system pip):
    pip install openslide-python tifffile imagecodecs pillow --break-system-packages
    (libopenslide must also be available as a system library - confirm
    with `python -c "import openslide"` before submitting the batch job)
"""

import argparse
import csv
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import openslide
import tifffile
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_MPP = 0.5        # 20x-equivalent, matches UNI / CONCH / Prov-GigaPath
                         # and the approximated Quilt-LLaVA extraction standard
MPP_TOLERANCE = 0.05     # consistent with wsi_audit.py's needs_software_resize flag
TILE_SIZE = 512          # output TIFF tile edge length, in pixels
COMPRESSION = "deflate"  # lossless: avoids introducing JPEG-compression artifacts
                         # into a baseline that a later perturbation (JPEG
                         # compression degradation, if present in the 18-item
                         # suite) needs to apply as a controlled, measurable step

DEFAULT_ACC_ROOT = Path("/its/home/pn254/project2026/wsi_data/acc_data")
DEFAULT_GDC_ROOT = Path("/its/home/pn254/project2026/wsi_data/tcga_data")
DEFAULT_ACC_OUT_ROOT = Path("/mnt/lustre/users/inf/pn254/downsampled/acc")
DEFAULT_GDC_OUT_ROOT = Path("/mnt/lustre/users/inf/pn254/downsampled/tcga")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("downsample_wsi")


# ---------------------------------------------------------------------------
# Report row schema
# ---------------------------------------------------------------------------

@dataclass
class DownsampleRow:
    source: str                     # 'ACC' or 'GDC'
    case_id: str                    # acc_case_id or file_id
    input_path: str
    output_path: str
    native_mpp: Optional[float] = None
    source_level: Optional[int] = None
    achieved_mpp_at_level: Optional[float] = None
    scale_factor: Optional[float] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    processing_seconds: Optional[float] = None
    status: str = "UNKNOWN"          # DONE / SKIPPED_EXISTING / MISSING_INPUT / ERROR
    notes: str = ""


# ---------------------------------------------------------------------------
# Path resolution (mirrors wsi_audit.py so both scripts agree on layout)
# ---------------------------------------------------------------------------

def resolve_acc_input(acc_root: Path, acc_case_id: str) -> Path:
    """ACC slides sit flat in acc_root, named <acc_case_id>.svs (confirmed)."""
    return acc_root / f"{acc_case_id}.svs"


def resolve_gdc_input(gdc_root: Path, file_id: str, file_name: str) -> Path:
    """
    GDC controls sit in a subfolder named after the file_id UUID.
    file_name already carries its own .svs extension - do not append one.
    """
    return gdc_root / file_id / file_name


def resolve_acc_output(acc_out_root: Path, acc_case_id: str) -> Path:
    return acc_out_root / f"{acc_case_id}.tiff"


def resolve_gdc_output(gdc_out_root: Path, file_id: str) -> Path:
    # Flat by file_id: the UUID is already globally unique, so no
    # per-file subfolder is needed here (unlike the source GDC layout).
    return gdc_out_root / f"{file_id}.tiff"


# ---------------------------------------------------------------------------
# Core resize / tiling logic
# ---------------------------------------------------------------------------

def select_target_level(slide: "openslide.OpenSlide", target_mpp: float = TARGET_MPP):
    """
    Selects the pyramid level whose effective MPP is closest to target_mpp,
    using the slide's own MPP-X metadata (not the possibly-mislabelled
    objective-power field - see the audit findings on the ACC cohort,
    where objective_power=20 but native MPP is consistently ~0.262,
    which corresponds to a 40x-class scan, not 20x. MPP is the more
    reliable field to key off in exactly this kind of scanner-metadata
    mismatch).

    Returns (level, native_mpp_x, achieved_mpp_at_level).
    Raises ValueError if no usable MPP metadata is present at all.
    """
    mpp_x_raw = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
    try:
        mpp_x = float(mpp_x_raw) if mpp_x_raw is not None else None
    except (TypeError, ValueError):
        mpp_x = None

    if not mpp_x or mpp_x <= 0:
        raise ValueError(
            "No usable MPP-X metadata on this slide; cannot determine "
            "native resolution or select a target level automatically."
        )

    best_level, best_diff = 0, float("inf")
    for level in range(slide.level_count):
        downsample = slide.level_downsamples[level]
        effective_mpp = mpp_x * downsample
        diff = abs(effective_mpp - target_mpp)
        if diff < best_diff:
            best_diff, best_level = diff, level

    achieved_mpp = mpp_x * slide.level_downsamples[best_level]
    return best_level, mpp_x, achieved_mpp


def rgba_to_rgb_white_bg(img_rgba: Image.Image) -> np.ndarray:
    """
    Composite an RGBA region (OpenSlide always returns RGBA) onto a white
    background and return an RGB array. Whole-slide images typically have
    transparent regions outside the scanned tissue area at non-zero
    pyramid levels; compositing onto white (rather than dropping the
    alpha channel outright) avoids those regions rendering as black.
    """
    bg = Image.new("RGB", img_rgba.size, (255, 255, 255))
    bg.paste(img_rgba, mask=img_rgba.split()[3])
    return np.asarray(bg)


def generate_output_tiles(
    slide: "openslide.OpenSlide",
    level: int,
    scale: float,
    tile_size: int = TILE_SIZE,
) -> Iterator[np.ndarray]:
    """
    Generator yielding (tile_size, tile_size, 3) uint8 tiles in row-major
    order, covering the full rescaled output image. Each tile is produced
    by inverse-mapping its output-space coordinates back to the source
    level's coordinate space, reading only that small source region via
    OpenSlide, and resizing it to the tile's exact output size. Nothing
    beyond a single tile's worth of pixels is ever held in memory.

    IMPORTANT: OpenSlide's read_region() location argument must be given
    in the level-0 reference frame regardless of which level is being
    read, per OpenSlide's documented contract (confirmed against the
    OpenSlide Python API reference and openslide-users mailing list
    discussion of this exact gotcha). Concretely: to read at
    (x, y, level=L, size=(w, h)) where level L has downsample d, the
    location argument passed to read_region must be (x * d, y * d), not
    (x, y). Getting this wrong silently reads the wrong region rather
    than raising an error, which is why it's called out explicitly here
    rather than left implicit in the arithmetic below.
    """
    src_w, src_h = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]

    out_w = max(1, round(src_w * scale))
    out_h = max(1, round(src_h * scale))

    n_tiles_x = -(-out_w // tile_size)  # ceiling division
    n_tiles_y = -(-out_h // tile_size)

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            out_x0, out_y0 = tx * tile_size, ty * tile_size
            out_x1 = min(out_x0 + tile_size, out_w)
            out_y1 = min(out_y0 + tile_size, out_h)
            valid_w, valid_h = out_x1 - out_x0, out_y1 - out_y0

            # Inverse-map this output tile back to source-level pixels.
            src_x0 = int(out_x0 / scale)
            src_y0 = int(out_y0 / scale)
            src_x1 = min(src_w, int(round(out_x1 / scale)))
            src_y1 = min(src_h, int(round(out_y1 / scale)))
            src_bw = max(1, src_x1 - src_x0)
            src_bh = max(1, src_y1 - src_y0)

            # Location argument: multiply by this level's downsample to
            # convert into the level-0 reference frame (see docstring).
            location = (int(src_x0 * downsample), int(src_y0 * downsample))
            region_rgba = slide.read_region(location, level, (src_bw, src_bh))
            region_rgb = rgba_to_rgb_white_bg(region_rgba)

            resized = np.asarray(
                Image.fromarray(region_rgb).resize(
                    (valid_w, valid_h), resample=Image.LANCZOS
                )
            )

            # TIFF tiles must all be the declared tile size; pad the
            # unused margin at right/bottom edge tiles with zeros. This
            # padding falls entirely outside the declared image `shape`
            # written below, so it is cropped away transparently on read.
            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile[0:valid_h, 0:valid_w] = resized
            yield tile


def validate_existing_output(path: Path) -> bool:
    """
    Returns True only if `path` exists AND is a readable, non-empty TIFF,
    so a half-written file from a killed prior job is correctly treated
    as needing reprocessing rather than being skipped as "already done".
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            return page.shape[0] > 0 and page.shape[1] > 0
    except Exception:  # noqa: BLE001 - any read failure means "not valid"
        return False


def process_slide(
    source: str,
    case_id: str,
    input_path: Path,
    output_path: Path,
    target_mpp: float = TARGET_MPP,
    tile_size: int = TILE_SIZE,
) -> DownsampleRow:
    """
    Processes a single slide end-to-end: skip-if-already-done check,
    open, level selection, tiled resize + write (atomic via temp file),
    and returns a fully populated DownsampleRow for the CSV report.
    Exceptions are caught here so one bad slide cannot halt the batch.
    """
    row = DownsampleRow(
        source=source,
        case_id=case_id,
        input_path=str(input_path),
        output_path=str(output_path),
    )

    if not input_path.exists():
        row.status = "MISSING_INPUT"
        row.notes = f"Input file not found at {input_path}."
        return row

    if validate_existing_output(output_path):
        row.status = "SKIPPED_EXISTING"
        row.notes = "Valid output already present; not reprocessed."
        return row

    start = time.time()
    tmp_path = output_path.with_suffix(".tiff.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        slide = openslide.OpenSlide(str(input_path))
    except Exception as exc:  # noqa: BLE001
        row.status = "ERROR"
        row.notes = f"OpenSlide failed to open input: {exc}"
        return row

    try:
        level, native_mpp, achieved_mpp = select_target_level(slide, target_mpp)
        scale = achieved_mpp / target_mpp

        src_w, src_h = slide.level_dimensions[level]
        out_w = max(1, round(src_w * scale))
        out_h = max(1, round(src_h * scale))

        tile_gen = generate_output_tiles(slide, level, scale, tile_size)

        # Written to a .tmp path first; only renamed into place on
        # success, so a killed job never leaves a "valid-looking" but
        # truncated file that a later run would mistake for complete.
        tifffile.imwrite(
            str(tmp_path),
            tile_gen,
            dtype="uint8",
            shape=(out_h, out_w, 3),
            tile=(tile_size, tile_size),
            photometric="rgb",
            compression=COMPRESSION,
            description=(
                f"source={source};case_id={case_id};"
                f"native_mpp_x={native_mpp:.5f};source_level={level};"
                f"target_mpp={target_mpp};achieved_mpp_at_level={achieved_mpp:.5f}"
            ),
        )
        tmp_path.rename(output_path)

        row.native_mpp = round(native_mpp, 5)
        row.source_level = level
        row.achieved_mpp_at_level = round(achieved_mpp, 5)
        row.scale_factor = round(scale, 5)
        row.output_width = out_w
        row.output_height = out_h
        row.status = "DONE"

    except Exception as exc:  # noqa: BLE001
        row.status = "ERROR"
        row.notes = f"Processing failed: {exc}"
        tmp_path.unlink(missing_ok=True)  # clean up any partial temp file
    finally:
        slide.close()
        row.processing_seconds = round(time.time() - start, 2)

    return row


# ---------------------------------------------------------------------------
# Manifest handling and batch orchestration
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> list:
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def build_jobs(manifest_rows: list, acc_root: Path, gdc_root: Path,
               acc_out_root: Path, gdc_out_root: Path) -> list:
    """
    Builds the full (source, case_id, input_path, output_path) job list
    for both cohorts. GDC NO_MATCH rows (empty file_id) are skipped, same
    convention as wsi_audit.py, since there is no control slide to process.
    """
    jobs = []
    no_match_count = 0

    for rec in manifest_rows:
        acc_case_id = rec["acc_case_id"]
        jobs.append((
            "ACC", acc_case_id,
            resolve_acc_input(acc_root, acc_case_id),
            resolve_acc_output(acc_out_root, acc_case_id),
        ))

        file_id = rec.get("file_id", "").strip()
        file_name = rec.get("file_name", "").strip()
        if not file_id or not file_name:
            no_match_count += 1
            continue

        jobs.append((
            "GDC", file_id,
            resolve_gdc_input(gdc_root, file_id, file_name),
            resolve_gdc_output(gdc_out_root, file_id),
        ))

    log.info(f"Built {len(jobs)} jobs ({no_match_count} GDC NO_MATCH rows skipped).")
    return jobs


def shard_jobs(jobs: list, shard_index: int, num_shards: int) -> list:
    """Simple modulo sharding for Slurm array jobs: task i handles jobs[i::N]."""
    if num_shards <= 1:
        return jobs
    sharded = jobs[shard_index::num_shards]
    log.info(
        f"Shard {shard_index}/{num_shards}: {len(sharded)} of {len(jobs)} "
        "total jobs assigned to this task."
    )
    return sharded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--acc-root", type=Path, default=DEFAULT_ACC_ROOT)
    parser.add_argument("--gdc-root", type=Path, default=DEFAULT_GDC_ROOT)
    parser.add_argument("--acc-out-root", type=Path, default=DEFAULT_ACC_OUT_ROOT)
    parser.add_argument("--gdc-out-root", type=Path, default=DEFAULT_GDC_OUT_ROOT)
    parser.add_argument("--out", required=True, type=Path, help="Path to write the CSV report")
    parser.add_argument("--target-mpp", type=float, default=TARGET_MPP)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE)
    parser.add_argument("--shard-index", type=int, default=0,
                         help="0-based index of this Slurm array task")
    parser.add_argument("--num-shards", type=int, default=1,
                         help="Total number of Slurm array tasks")
    args = parser.parse_args()

    log.info(f"ACC input root:  {args.acc_root}")
    log.info(f"GDC input root:  {args.gdc_root}")
    log.info(f"ACC output root: {args.acc_out_root}")
    log.info(f"GDC output root: {args.gdc_out_root}")

    manifest_rows = load_manifest(args.manifest)
    log.info(f"Loaded {len(manifest_rows)} manifest rows.")

    jobs = build_jobs(
        manifest_rows, args.acc_root, args.gdc_root,
        args.acc_out_root, args.gdc_out_root,
    )
    jobs = shard_jobs(jobs, args.shard_index, args.num_shards)

    results = []
    for i, (source, case_id, input_path, output_path) in enumerate(jobs, start=1):
        log.info(f"[{i}/{len(jobs)}] {source} {case_id} ...")
        row = process_slide(
            source, case_id, input_path, output_path,
            target_mpp=args.target_mpp, tile_size=args.tile_size,
        )
        results.append(row)
        log.info(f"    -> {row.status} ({row.processing_seconds}s) {row.notes}")

    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(asdict(row))

    def summarize(rows: list, label: str):
        counts = {}
        for r in rows:
            counts[r.status] = counts.get(r.status, 0) + 1
        log.info(f"[{label}] {counts}")

    log.info(f"\nReport written to {args.out}")
    summarize([r for r in results if r.source == "ACC"], "ACC")
    summarize([r for r in results if r.source == "GDC"], "GDC")


if __name__ == "__main__":
    main()
