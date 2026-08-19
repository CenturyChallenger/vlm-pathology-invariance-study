# Environment notes

This project runs Phase A (embedding extraction) on the University of Sussex
Artemis HPC cluster (Slurm, with a legacy SGE compatibility wrapper), one conda
environment per model, and Phase B (statistical analysis) locally or on any
CPU-only machine. These notes exist so the pipeline can be reproduced outside
the author's original HPC account.

## Conda environments (Phase A)

Each foundation model is isolated in its own conda environment to avoid
dependency conflicts between the model codebases:

| Environment | Model | Notes |
|---|---|---|
| `uni-env` | UNI | Requires a Hugging Face token with approved gated access to `MahmoodLab/uni`. |
| `conch-env` | CONCH | Requires a Hugging Face token with approved gated access to `MahmoodLab/conch`. Gated access is per-repository, not per-organisation; approval for one does not grant the other. |
| `quilt-llava-env` | Quilt-LLaVA | Requires the LLaVA codebase and `deepspeed`. **Pin `numpy==1.26.4` in this environment specifically** -- `deepspeed` raises a `numpy.BUFSIZE` `ImportError` under NumPy 2.x. Vision tower is stock `openai/clip-vit-large-patch14-336` (not a domain-adapted encoder); its image processor upsamples every 224x224 input tile to 336x336 before inference. Correct HF repo ID is `wisdomik/Quilt-Llava-v1.5-7b`. |
| (tile-encoder-only, no separate env required) | Prov-GigaPath | Uses `timm.create_model("hf_hub:prov-gigapath/prov-gigapath")` only. Does **not** require the `gigapath` package, `flash-attn`, or `xformers`. |

`h5py` is not covered by base environment setup and must be installed
individually in each of the above environments.

## GPU / CUDA

Confirm the node driver ceiling with `nvidia-smi`'s "CUDA Version:" field
before installing `torch`. On this project's A40 nodes (driver 560.35.05, max
CUDA 12.6):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Installing a `torch` build compiled against a newer CUDA than the node driver
supports will fail silently at runtime rather than at install time; always
verify against `nvidia-smi` first.

## elasticdeform under NumPy 2.x

`elasticdeform` 0.5.1's PyPI wheel is built against the NumPy 1.x ABI and is
not compatible with NumPy 2.x by default. Build from source instead:

```bash
pip install --no-binary elasticdeform --no-build-isolation --no-deps \
    --force-reinstall --no-cache-dir elasticdeform
```

Pin `numpy` to the 2.4.x line and upgrade `numba` to `>=0.64.0` alongside this
change. Verify the fix with an actual perturbation smoke-test run (for
example, perturbation 13), not just a successful `import elasticdeform`.

## Slurm submission pattern (Artemis cluster)

The Artemis Slurm scheduler runs a legacy SGE compatibility wrapper. `#$`-style
directives work for simple single-dash SGE options (`-N`, `-o`, `-j`, `-cwd`,
`-m`, `-l h_rt=`, `-p`), but the wrapper silently ignores Slurm long-option
names under `#$` (`--gres`, `--nodes`, `--cpus-per-task`). Use native
`#SBATCH` directives for those instead; mixing `#$` and `#SBATCH` in the same
script is permitted.

Recommended diagnostic header for every submission script, run before the
Python entry point, to catch a missing GPU allocation immediately rather than
partway through a long job:

```bash
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "JOB_ID=$JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
nvidia-smi
```

Non-interactive `sbatch` shells do not source `~/.bashrc`. Activate conda
explicitly and fail fast if the wrong environment is active:

```bash
source <conda-base>/etc/profile.d/conda.sh
conda activate uni-env
if [[ "$CONDA_DEFAULT_ENV" != "uni-env" ]]; then
    echo "ERROR: expected uni-env, got $CONDA_DEFAULT_ENV" >&2
    exit 1
fi
```

## Interrupted extraction jobs

An OOM-killed job that has called `h5py.File(path, "a")` can leave a
truncated or corrupt `.h5` output file. `extract_embeddings_similarity.py`'s
skip-already-embedded-keys logic cannot recover from a corrupted header;
delete the affected file before resubmitting.

## GDC data access

The NCI CRDC DRS endpoint (`nci-crdc.datacommons.io`) is blocked from Sussex
HCI compute nodes. The programme-specific GDC open-access Google Cloud
Storage bucket is the resolution method confirmed to work in production:

| Programme prefix | GCS bucket |
|---|---|
| `TCGA-*` | `gdc-tcga-phs000178-open` |
| `HCM-*` | `gdc-hcmi-open` |
| `CPTAC-*` | `gdc-cptac-phs001287-open` |
| `TARGET-*` | `gdc-target-phs000218-open` |
| `CGCI-*` | `gdc-cgci-phs000235-open` |

Files are addressed as `https://storage.googleapis.com/<bucket>/<uuid>/<filename>`.

**Discrepancy, documented rather than silently resolved:** the version of
`01_gdc_match_manifest_v4.py` currently in this repository resolves each
file's bucket via a live call to the DRS endpoint above, with a fallback to
the TCGA bucket only (not programme-aware) if that call fails -- it does not
implement the bucket-table approach shown above. Given DRS is unreachable
from Sussex HCI nodes, this script's primary resolution path has not been
confirmed to succeed end-to-end from that environment. Separately,
`matched_controls_manifest_patched.tsv` (excluded from this repository; see
main README) has been verified, via `02_test_matched_controls_patched.py`,
to already contain correct per-programme bucket URLs for every HCM-prefixed
file. No script or log currently in this project evidences how that file was
produced from the unpatched manifest -- it is treated as an open item, not
assumed to have been done via the bucket table above, since that would be
asserting a mechanism without evidence.

The GDC API's `not_in` race filter on nested demographic fields silently
drops records whose parent entity is null; race exclusion is applied in
Python after retrieval rather than relying on the API filter alone.
