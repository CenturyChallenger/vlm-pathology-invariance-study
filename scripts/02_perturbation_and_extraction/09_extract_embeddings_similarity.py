#!/usr/bin/env python3
"""
extract_embeddings_similarity.py

Phase A: extracts embeddings for every tile (baseline + all 18x3 perturbed
variants) produced by perturbation_pipeline.py, across all three study
models (UNI, CONCH, Quilt-LLaVA), with resilient HDF5-backed caching.

Phase B: computes the invariance metrics (cosine similarity per tile, and
linear CKA per perturbation/severity/model) from the cached embeddings,
entirely on CPU, so it can be re-run cheaply without touching the GPU.

=============================================================================
ARCHITECTURE (see conversation for full reasoning; summarised here)
=============================================================================

1. MODEL ADAPTER PATTERN: UNI, CONCH, and Quilt-LLaVA have different
   loading code, preprocessing, and embedding-extraction paths. One
   FoundationModelAdapter interface lets a single batching/inference loop
   serve all three, rather than three duplicated pipelines.

2. SEQUENTIAL MODEL LOADING: models are processed one at a time (fully
   loaded, run to completion, released) rather than concurrently, to
   bound peak GPU memory. Given Prov-GigaPath's documented resource
   failures on this cluster, this is a deliberate conservative choice,
   not an oversight.

3. TWO DECOUPLED PHASES: embedding extraction (GPU-bound, expensive) is
   fully separated from similarity/CKA computation (CPU-bound, cheap).
   Cosine similarity is a per-tile comparison; CKA is NOT per-tile, it
   compares two whole representation matrices (all baseline embeddings
   vs all perturbed embeddings for one perturbation/severity/model) and
   produces one score per combination. Both are cheap once embeddings
   are cached, so re-running metric computation never requires re-running
   the GPU pass.

4. DEDUPLICATION: each tile's baseline (unperturbed) image is embedded
   exactly once per model, then reused as the reference for all 18x3
   perturbation comparisons for that tile. It is read from
   perturbation_pipeline.py's `baseline/` output subfolder.

5. LUSTRE-AWARE STORAGE: one HDF5 file per model
   (embeddings_{model}.h5), not one file per embedding. At full scale
   this is ~190k+ embeddings; writing that many individual files onto a
   parallel filesystem like Lustre is a known metadata-server
   anti-pattern. HDF5 keeps file count low while still supporting cheap
   per-key existence checks for resilience.

6. RESILIENCE: before building the DataLoader for a model, every image
   already present (and non-corrupt) in that model's HDF5 store is
   filtered out of the job list. Embeddings are flushed to the HDF5 file
   after every batch, not buffered until the end, so a job killed by a
   Slurm timeout loses at most one batch of work, not the whole run.

=============================================================================
MODEL LOADING NOTES (grounded against each model's official documentation
as of this writing; NOT executable in this sandbox - no GPU, no gated HF
access. Verify against your already-working installations, e.g. the
Quilt-LLaVA setup noted as already operational on Artemis.)
=============================================================================

  UNI    : timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
           init_values=1e-5, dynamic_img_size=True); requires a
           HuggingFace access token with UNI's gated-access terms
           accepted (see MahmoodLab/UNI on Hugging Face).
  CONCH  : conch.open_clip_custom.create_model_from_pretrained(
           'conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token=...);
           embeddings via model.encode_image(x, proj_contrast=False,
           normalize=False) - the pre-projection, unnormalised
           representation, which MahmoodLab's own README states is the
           form "suitable for linear probe or working with WSIs", i.e.
           the correct choice for representation-comparison work like
           ours, as opposed to proj_contrast=True/normalize=True, which
           is for text-image retrieval similarity scoring specifically.
  Quilt-LLaVA : vision tower is wisdomik/QuiltNet-B-32 (CLIP ViT-B/32,
           224px, matching TILE_SIZE exactly). Loaded via the LLaVA
           codebase (llava.model.builder.load_pretrained_model), then
           the tower is used directly via model.get_vision_tower(),
           bypassing the language model entirely for this study's
           purposes (per project notes: CLI chat inference is not
           relevant to this use case). Feature layer selection follows
           the project's own training config (--mm_vision_select_layer
           -2, i.e. second-to-last transformer layer); the CLS token
           from that layer's hidden states is used as the pooled
           embedding. attn_implementation="sdpa" is NOT used rather than	=> #attn_implementation removed. Refers to more recent Quilt-llava version => 10.08.2026 18:03 :: Phillip Nyamwaya
           flash-attn, per project notes (flash-attn is not a hard
           dependency for tile-level embedding extraction).
  GigaPath (tile encoder only) : timm.create_model(
           "hf_hub:prov-gigapath/prov-gigapath", pretrained=True);
           requires a HuggingFace access token with Prov-GigaPath's
           gated-access terms accepted, and timm>=1.0.3 (older timm
           versions have documented compatibility issues with this
           model, per the official model card). Deliberately uses ONLY
           the tile encoder, not the slide encoder: the slide encoder
           requires the coordinates of every other tile on the same
           slide and runs cross-tile LongNet attention before producing
           any output, which reproduces the exact whole-slide
           aggregation problem documented as the reason TITAN was
           excluded from this study's cohort (see Appendix D - a
           perturbation to one tile would only be one signal among
           potentially thousands aggregated into a slide-level
           representation, not the direct determinant of the output,
           breaking the tile-level perturbation-exposure equivalence
           the whole comparative design depends on). The tile encoder
           alone, like UNI/CONCH/Quilt-LLaVA, embeds each 224x224 tile
           independently with no cross-tile context, which is why it
           is the architecturally appropriate fourth model here. Unlike
           the slide encoder, the tile encoder needs no `gigapath`
           package, flash-attn, or xformers, it is loaded purely
           through timm, exactly like UNI.

Dependencies:
  pip install torch torchvision h5py numpy pillow --break-system-packages
  # Plus each model's own package: timm + huggingface_hub (UNI),
  # conch (pip install git+https://github.com/Mahmoodlab/CONCH.git),
  # and the quilt-llava/LLaVA codebase already installed per project notes.

Usage (MULTI-CONDA-ENV SETUP - one environment per model, as installed on
Artemis):

  Run this script once per model, from inside that model's own conda
  environment. Each invocation only needs that one model's dependencies;
  the deferred-import design means the other two adapters are never
  touched. The three extraction runs write to separate HDF5 files
  (embeddings_uni.h5, embeddings_conch.h5, embeddings_quilt_llava.h5), so
  they don't conflict with each other and can even run concurrently as
  separate Slurm jobs if your allocation allows multiple simultaneous GPU
  jobs, cutting wall-clock time versus running them sequentially (total
  GPU-hours consumed is unchanged either way, this only affects how long
  you wait):

    conda activate uni-env
    python extract_embeddings_similarity.py --mode extract --model uni \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

    conda activate conch-env
    python extract_embeddings_similarity.py --mode extract --model conch \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

    conda activate quilt-llava-env
    python extract_embeddings_similarity.py --mode extract --model quilt_llava \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

    conda activate gigapath
    python extract_embeddings_similarity.py --mode extract --model gigapath_tile \
        --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

  Then, from ANY environment that has h5py/numpy installed (Phase B needs
  no model-specific packages at all, since it only reads cached
  embeddings):

    python extract_embeddings_similarity.py --mode similarity \
        --embeddings-dir outputs/embeddings \
        --cosine-out outputs/cosine_similarity.csv --cka-out outputs/cka_summary.csv

  NOTE ON --model all: this loops through all three adapters within ONE
  process, so it only makes sense if all three models' packages happen to
  be co-installed in the currently active environment. In a multi-env
  setup, running --model all from any single environment will correctly
  skip whichever models aren't installed there (with a clear warning
  telling you which environment to use instead) rather than crashing, but
  it will NOT process the skipped models - you still need the three
  separate invocations above to cover all three models. Every one of the
  three model-specific environments used for extraction also needs
  h5py, numpy, pillow, and torch installed alongside that model's own
  package - these are the only cross-cutting dependencies this script
  needs regardless of which model is selected.

Usage (single shared environment, all three models installed together):
  # Extract embeddings for one model (resilient/resumable):
  python extract_embeddings_similarity.py --mode extract --model uni \
      --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

  # Extract for all three models in sequence:
  python extract_embeddings_similarity.py --mode extract --model all \
      --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings

  # Compute similarity metrics from cached embeddings (CPU-only, fast):
  python extract_embeddings_similarity.py --mode similarity \
      --embeddings-dir outputs/embeddings \
      --cosine-out outputs/cosine_similarity.csv \
      --cka-out outputs/cka_summary.csv

  # Both phases in one invocation:
  python extract_embeddings_similarity.py --mode both --model all \
      --tiles-dir outputs/perturbed_tiles --embeddings-dir outputs/embeddings \
      --cosine-out outputs/cosine_similarity.csv --cka-out outputs/cka_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

TILE_SIZE = 224

# Batch-size defaults tuned conservatively for a <=40GB A40-class GPU.
# UNI (ViT-L/16, ~300M params) is the heaviest of the three; Quilt-LLaVA's
# vision tower alone (ViT-B/32) and CONCH (ViT-B/16, 90M params) are
# lighter, so they can run larger batches on the same hardware. These are
# starting points, not guarantees - tune down if you see OOMs, particularly
# if other jobs are sharing the node's memory on a shared queue.
DEFAULT_BATCH_SIZES = {
    "uni": 256,
    "conch": 384,
    "quilt_llava": 384,
    # GigaPath's tile encoder (DINOv2 ViT-g/14, ~1.1B params) is roughly
    # 3.7x the parameter count of UNI's ViT-L/16 (~300M params), so this
    # starts noticeably more conservative than UNI's batch size despite
    # both being single-tile, no-cross-tile-context forward passes. Per
    # the earlier debugging on this cluster, a correctly-allocated A40
    # (46GB VRAM) has ample headroom even at fp32 for this model class
    # (weights alone are ~4.4GB in fp32), so this figure is a cautious
    # starting point pending an actual observed-throughput calibration
    # run (the same approach used to calibrate UNI's ~215 img/s figure),
    # not a hard ceiling - raise it once you have real throughput data.
    "gigapath_tile": 96,
    "mock": 512,
}

DEFAULT_NUM_WORKERS = 8


# ===========================================================================
# Model Adapter interface
# ===========================================================================

class FoundationModelAdapter(ABC):
    """
    Common interface so one batching/inference loop can serve any of the
    three study models. Subclasses encapsulate everything model-specific:
    loading, preprocessing, and the forward pass used to get a single
    pooled embedding vector per image.
    """

    name: str = "base"
    embedding_dim: Optional[int] = None  # set after load()

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load weights, move to device, set eval mode."""
        raise NotImplementedError

    @abstractmethod
    def get_transform(self) -> Callable[[Image.Image], torch.Tensor]:
        """Returns this model's required preprocessing transform."""
        raise NotImplementedError

    @abstractmethod
    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        """
        Runs a forward pass on a preprocessed batch (B, C, H, W) already
        on the correct device, returns (B, D) float32 embeddings as a
        numpy array on CPU. Must be called within torch.inference_mode().
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Releases the model from GPU memory between models."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelEnvironmentError(RuntimeError):
    """
    Raised when a model's adapter is loaded from a Python environment that
    doesn't have that model's dependencies installed. This is EXPECTED and
    routine in a multi-conda-env setup (one env per model, per project
    notes) - the fix is to activate the correct environment, not a bug in
    this script. Kept as its own exception type so run_extraction's
    --model all path can distinguish "wrong environment, try elsewhere"
    from a genuine error worth stopping the whole run for.
    """
    pass


class UNIAdapter(FoundationModelAdapter):
    """
    UNI (Chen et al., Nature Medicine 2024). ViT-L/16, DINOv2-based,
    masked-modelling pretraining. Loaded via timm's HF Hub integration.
    Requires gated access on Hugging Face and a valid HF token.
    """
    name = "uni"

    def load(self) -> None:
        try:
            import timm
            from huggingface_hub import login  # noqa: F401 - login() must be
            # called by the caller/environment beforehand (e.g.
            # `huggingface-cli login` or HUGGING_FACE_HUB_TOKEN env var);
            # not invoked here to avoid embedding credential-handling
            # logic in this script.
        except ImportError as exc:
            raise ModelEnvironmentError(
                "UNI requires 'timm' and 'huggingface_hub', not found in "
                "the currently active Python environment. If UNI is "
                "installed in its own conda environment, activate it "
                "first: `conda activate <your-uni-env>`."
            ) from exc

        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni", pretrained=True,
            init_values=1e-5, dynamic_img_size=True,
        )
        self.model.eval().to(self.device)
        self.embedding_dim = self.model.num_features

    def get_transform(self):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        return create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))

    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        out = self.model(batch)
        return out.detach().float().cpu().numpy()


class CONCHAdapter(FoundationModelAdapter):
    """
    CONCH (Lu et al., Nature Medicine 2024). ViT-B/16 vision encoder,
    contrastive image-text pretraining. Loaded via the conch package's
    open_clip-derived loader. Requires gated access on Hugging Face.

    Uses proj_contrast=False, normalize=False: the pre-projection,
    unnormalised representation, which is the form MahmoodLab's own
    documentation recommends for representation-comparison / linear-probe
    style use, as opposed to the retrieval-oriented projected+normalised
    embedding (proj_contrast=True, normalize=True).
    """
    name = "conch"

    def load(self) -> None:
        try:
            from conch.open_clip_custom import create_model_from_pretrained
        except ImportError as exc:
            raise ModelEnvironmentError(
                "CONCH requires the 'conch' package (pip install "
                "git+https://github.com/Mahmoodlab/CONCH.git), not found "
                "in the currently active Python environment. If CONCH is "
                "installed in its own conda environment, activate it "
                "first: `conda activate <your-conch-env>`."
            ) from exc

        self.model, self._preprocess = create_model_from_pretrained(
            "conch_ViT-B-16", "hf_hub:MahmoodLab/conch",
        )
        self.model.eval().to(self.device)
        # CONCH's vision encoder is ViT-B/16 (90M params); embedding dim
        # is discovered empirically on first use if not documented
        # directly on the model object.
        self.embedding_dim = getattr(self.model.visual, "output_dim", None)

    def get_transform(self):
        return self._preprocess

    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        out = self.model.encode_image(batch, proj_contrast=False, normalize=False)
        return out.detach().float().cpu().numpy()


class QuiltLLaVAAdapter(FoundationModelAdapter):
    """
    Quilt-LLaVA (Seyfioglu et al., NeurIPS 2024). Vision tower is
    wisdomik/QuiltNet-B-32 (CLIP ViT-B/32, 224px), loaded via the LLaVA
    codebase and used directly, bypassing the language model entirely
    (per project notes: CLI chat inference is not relevant here).
    Feature layer follows the project's training config
    (mm_vision_select_layer=-2); CLS token from that layer is used as
    the pooled embedding.
    """
    name = "quilt_llava"

    def load(self) -> None:
        # Deferred import: only required when this adapter is actually
        # selected, so environments without the LLaVA codebase installed
        # can still run the UNI/CONCH adapters and the mock adapter.
        try:
            from llava.model.builder import load_pretrained_model
            from llava.mm_utils import get_model_name_from_path
        except ImportError as exc:
            raise ModelEnvironmentError(
                "Quilt-LLaVA requires the LLaVA codebase, not found in "
                "the currently active Python environment. If Quilt-LLaVA "
                "is installed in its own conda environment, activate it "
                "first: `conda activate <your-quilt-llava-env>`."
            ) from exc

#        model_path = "wisdomikezogwo/quilt_llava_v1.5" # Older missing repo changed  => 10.08.2026 17:12 :: Phillip Nyamwaya
        model_path = "wisdomik/Quilt-Llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)
        _, self.model, self.image_processor, _ = load_pretrained_model(
            model_path=model_path, model_base=None, model_name=model_name,   #attn_implementation removed. Refers to more recent Quilt-llava version => 10.08.2026 18:03 :: Phillip Nyamwaya
        )
        self.vision_tower = self.model.get_vision_tower()
        self.vision_tower.eval().to(self.device)
        self.select_layer = -2  # matches project's own training config
        self.embedding_dim = self.vision_tower.hidden_size

    def get_transform(self):
        proc = self.image_processor
        def _transform(img: Image.Image) -> torch.Tensor:
            return proc.preprocess(img, return_tensors="pt")["pixel_values"][0]
        return _transform

    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        hidden_states = self.vision_tower.vision_tower(
            batch, output_hidden_states=True
        ).hidden_states[self.select_layer]
        cls_embedding = hidden_states[:, 0, :]  # CLS token, standard CLIP pooling
        return cls_embedding.detach().float().cpu().numpy()


class GigaPathTileAdapter(FoundationModelAdapter):
    """
    Prov-GigaPath (Xu et al., Nature 2024) TILE ENCODER ONLY.

    Deliberately does not load or use the slide encoder. The slide
    encoder (gigapath.slide_encoder.create_model(...)) requires the
    embeddings and coordinates of every other tile on the same slide,
    and runs cross-tile LongNet attention across all of them before
    producing any output - regardless of the global_pool setting, a
    single tile's contribution to that output is mixed with signal from
    potentially thousands of other tiles, not determined by that tile
    alone. That is architecturally the same whole-slide-aggregation
    problem this study's Appendix D documents as the reason TITAN was
    excluded from the cohort: perturbation exposure would no longer be
    equivalent across models, breaking the comparison this study is
    built on. The tile encoder, by contrast, embeds each 224x224 tile
    independently with no cross-tile context, exactly like UNI, CONCH,
    and Quilt-LLaVA, which is what makes it the architecturally
    appropriate fourth model for this specific comparative design.

    Loaded via timm's HF Hub integration only, the same mechanism as
    UNI - no dependency on the `gigapath` package, flash-attn, or
    xformers, all of which are required by the slide encoder but not
    by this class. Requires gated access on Hugging Face
    (prov-gigapath/prov-gigapath) and a valid HF token, and
    timm>=1.0.3 (older timm versions have documented compatibility
    issues with this model per the official model card at
    https://huggingface.co/prov-gigapath/prov-gigapath).
    """
    name = "gigapath_tile"

    def load(self) -> None:
        try:
            import timm
            from huggingface_hub import login  # noqa: F401 - login() must be
            # called by the caller/environment beforehand (e.g.
            # `huggingface-cli login` or HUGGING_FACE_HUB_TOKEN env var),
            # exactly as for UNI; not invoked here to avoid embedding
            # credential-handling logic in this script.
        except ImportError as exc:
            raise ModelEnvironmentError(
                "Prov-GigaPath (tile encoder) requires 'timm' (>=1.0.3) "
                "and 'huggingface_hub', not found in the currently "
                "active Python environment. If GigaPath is installed in "
                "its own conda environment, activate it first: "
                "`conda activate <your-gigapath-env>`."
            ) from exc

        self.model = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", pretrained=True,
        )
        self.model.eval().to(self.device)
        # timm ViT models expose num_features for the pooled output dim,
        # same attribute UNIAdapter reads; expected to be 1536 per the
        # official model card (matches the embed_dim passed to the slide
        # encoder's create_model call, 1536, confirming the tile encoder
        # output dimensionality independently of the slide encoder).
        self.embedding_dim = self.model.num_features

    def get_transform(self):
        # Deliberately the exact transform given in the official model
        # card / README rather than timm's resolve_data_config, so this
        # matches the officially documented and tested preprocessing
        # exactly (Resize(256, bicubic) -> CenterCrop(224), not e.g. a
        # direct Resize(224)). CenterCrop(224) lands on TILE_SIZE exactly,
        # consistent with the other three adapters' input resolution.
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(TILE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        out = self.model(batch)
        return out.detach().float().cpu().numpy()


class MockAdapter(FoundationModelAdapter):
    """
    A dependency-free adapter used ONLY for validating the surrounding
    pipeline (batching, resilience, HDF5 storage, similarity/CKA
    computation) without needing real model weights or a GPU. NOT a
    substitute for UNI/CONCH/Quilt-LLaVA - it derives a deterministic
    64-dim "embedding" from simple pixel statistics (per-channel mean/std
    across a coarse grid), which is perturbation-sensitive enough to
    produce sensible, non-degenerate similarity scores for testing, but
    carries no scientific meaning whatsoever.
    """
    name = "mock"

    def load(self) -> None:
        self.model = "mock"
        self.embedding_dim = 64

    def get_transform(self):
        import torchvision.transforms as T
        return T.Compose([T.Resize((TILE_SIZE, TILE_SIZE)), T.ToTensor()])

    def extract_batch(self, batch: torch.Tensor) -> np.ndarray:
        # batch: (B, 3, 224, 224) in [0, 1]. Split into an 8x8 coarse grid,
        # take per-cell mean per channel -> 8*8*... no, keep it small:
        # 4x4 grid x 3 channels x (mean, std) = 96 -> trim/pad to 64.
        b = batch.shape[0]
        grid = 4
        cell = TILE_SIZE // grid
        feats = []
        for i in range(grid):
            for j in range(grid):
                cell_pixels = batch[:, :, i*cell:(i+1)*cell, j*cell:(j+1)*cell]
                feats.append(cell_pixels.mean(dim=(2, 3)))  # (B, 3)
        emb = torch.cat(feats, dim=1)  # (B, grid*grid*3) = (B, 48)
        # pad to 64 dims with zeros for a round embedding_dim
        pad = torch.zeros(b, 64 - emb.shape[1])
        return torch.cat([emb, pad], dim=1).cpu().numpy()


ADAPTER_REGISTRY: Dict[str, type] = {
    "uni": UNIAdapter,
    "conch": CONCHAdapter,
    "quilt_llava": QuiltLLaVAAdapter,
    "gigapath_tile": GigaPathTileAdapter,
    "mock": MockAdapter,
}


# ===========================================================================
# Image manifest: reads perturbation_pipeline.py's output directory
# ===========================================================================

@dataclass
class ImageRecord:
    key: str              # stable HDF5 dataset key, unique per image
    path: Path
    sample_id: str
    perturbation_id: Optional[int]   # None for baseline
    severity: Optional[str]          # None for baseline
    is_baseline: bool


def build_image_manifest(tiles_dir: Path) -> List[ImageRecord]:
    """
    Walks perturbation_pipeline.py's output tree:
        tiles_dir/baseline/{sample_id}.png
        tiles_dir/{category}/p{NN}_{name}/{severity}/{sample_id}.png
    and returns one ImageRecord per PNG found. The `key` field is what
    gets used as the HDF5 dataset name, so it must be filesystem/HDF5-safe
    and globally unique.
    """
    records: List[ImageRecord] = []

    baseline_dir = tiles_dir / "baseline"
    if not baseline_dir.exists():
        log.warning(
            "No baseline/ directory found under %s. Baseline tiles are "
            "required as the reference for every similarity comparison; "
            "make sure perturbation_pipeline.py has been run with the "
            "baseline-persistence patch (saves tiles_dir/baseline/*.png).",
            tiles_dir,
        )
    else:
        for p in sorted(baseline_dir.glob("*.png")):
            sample_id = p.stem
            records.append(ImageRecord(
                key=f"baseline__{sample_id}", path=p, sample_id=sample_id,
                perturbation_id=None, severity=None, is_baseline=True,
            ))

    for category_dir in sorted(tiles_dir.iterdir()):
        if not category_dir.is_dir() or category_dir.name == "baseline":
            continue
        for pert_dir in sorted(category_dir.iterdir()):
            # directory name format: p{NN}_{Name}
            try:
                pid = int(pert_dir.name.split("_")[0][1:])
            except (ValueError, IndexError):
                log.warning("Skipping unrecognised directory: %s", pert_dir)
                continue
            for severity_dir in sorted(pert_dir.iterdir()):
                severity = severity_dir.name
                for p in sorted(severity_dir.glob("*.png")):
                    sample_id = p.stem
                    key = f"p{pid:02d}__{severity}__{sample_id}"
                    records.append(ImageRecord(
                        key=key, path=p, sample_id=sample_id,
                        perturbation_id=pid, severity=severity, is_baseline=False,
                    ))

    log.info("Image manifest: %d images found under %s (%d baseline).",
              len(records), tiles_dir, sum(1 for r in records if r.is_baseline))
    return records


# ===========================================================================
# HDF5-backed resilient embedding store
# ===========================================================================

class EmbeddingStore:
    """
    One HDF5 file per model. Each embedding is stored as a dataset keyed
    by ImageRecord.key. Deliberately avoids one-file-per-embedding on
    disk (a known metadata-server anti-pattern on Lustre-backed storage)
    while still supporting cheap per-key existence checks for resilience.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 'a' = read/write, create if not exists, preserve existing data -
        # this is what makes resuming a partial run possible at all.
        self._file = h5py.File(str(self.path), "a")

    def has(self, key: str) -> bool:
        return key in self._file

    def write(self, key: str, embedding: np.ndarray) -> None:
        if key in self._file:
            del self._file[key]  # overwrite defensively rather than error
        self._file.create_dataset(key, data=embedding.astype(np.float32))

    def read(self, key: str) -> np.ndarray:
        return self._file[key][()]

    def keys(self) -> List[str]:
        return list(self._file.keys())

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ===========================================================================
# Dataset / DataLoader for batched inference
# ===========================================================================

class TileDataset(Dataset):
    """Thin dataset wrapping a list of ImageRecords needing embedding."""

    def __init__(self, records: List[ImageRecord], transform: Callable):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        rec = self.records[idx]
        img = Image.open(rec.path).convert("RGB")
        return rec.key, self.transform(img)


def _collate(batch: List[Tuple[str, torch.Tensor]]):
    keys = [b[0] for b in batch]
    tensors = torch.stack([b[1] for b in batch])
    return keys, tensors


# ===========================================================================
# Phase A: extraction
# ===========================================================================

def run_extraction(
    model_name: str,
    tiles_dir: Path,
    embeddings_dir: Path,
    batch_size: Optional[int] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    device: str = "cuda",
    precision: str = "bf16",
) -> None:
    """
    Extracts embeddings for every image in tiles_dir for one model,
    skipping images already present in that model's HDF5 store. Flushes
    to disk after every batch for crash-safety.
    """
    adapter_cls = ADAPTER_REGISTRY.get(model_name)
    if adapter_cls is None:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(ADAPTER_REGISTRY)}")

    batch_size = batch_size or DEFAULT_BATCH_SIZES.get(model_name, 128)
    log.info("Loading %s (batch_size=%d, device=%s, precision=%s)...",
              model_name, batch_size, device, precision)

    adapter = adapter_cls(device=device)
    t0 = time.time()
    adapter.load()
    log.info("%s loaded in %.1fs (embedding_dim=%s).", model_name, time.time() - t0, adapter.embedding_dim)

    manifest = build_image_manifest(tiles_dir)
    if not manifest:
        log.error("No images found under %s; aborting extraction for %s.", tiles_dir, model_name)
        return

    store_path = embeddings_dir / f"embeddings_{model_name}.h5"
    store = EmbeddingStore(store_path)

    pending = [r for r in manifest if not store.has(r.key)]
    log.info("%s: %d/%d images already embedded (resuming); %d remaining.",
              model_name, len(manifest) - len(pending), len(manifest), len(pending))

    if not pending:
        log.info("%s: nothing to do, all images already embedded.", model_name)
        store.close()
        adapter.unload()
        return

    dataset = TileDataset(pending, adapter.get_transform())
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
        collate_fn=_collate,
    )

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    n_done = 0
    t_start = time.time()

    with torch.inference_mode():
        for batch_idx, (keys, tensors) in enumerate(loader):
            tensors = tensors.to(device, non_blocking=True)

            if amp_dtype is not None and device == "cuda":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    embeddings = adapter.extract_batch(tensors)
            else:
                embeddings = adapter.extract_batch(tensors)

            for key, emb in zip(keys, embeddings):
                store.write(key, emb)
            store.flush()  # crash-safety: persist this batch immediately

            n_done += len(keys)
            if batch_idx % 10 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                log.info("%s: %d/%d embedded (%.1f img/s)...",
                          model_name, n_done, len(pending), rate)

    store.close()
    adapter.unload()
    log.info("%s: extraction complete. %d images embedded in %.1fs.",
              model_name, n_done, time.time() - t_start)


# ===========================================================================
# Phase B: similarity metrics (CPU-only, cheap - run as often as you like)
# ===========================================================================

def cosine_similarity_matrix_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) arrays -> (N,)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear Centered Kernel Alignment (Kornblith et al., "Similarity of
    Neural Network Representations Revisited", ICML 2019). X and Y are
    (N, D) representation matrices for the SAME N samples (here: N tiles),
    D can differ between X and Y since CKA operates on N x N Gram
    matrices, not on the raw feature dimensions directly - one reason
    it's well suited to comparing representations even if embedding
    dimensionality differs.
    """
    def centering(K: np.ndarray) -> np.ndarray:
        n = K.shape[0]
        unit = np.ones((n, n))
        H = np.eye(n) - unit / n
        return H @ K @ H

    def linear_hsic(A: np.ndarray, B: np.ndarray) -> float:
        L_A = A @ A.T
        L_B = B @ B.T
        return float(np.sum(centering(L_A) * centering(L_B)))

    hsic_xy = linear_hsic(X, Y)
    hsic_xx = linear_hsic(X, X)
    hsic_yy = linear_hsic(Y, Y)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return hsic_xy / denom if denom > 0 else float("nan")


def derive_cohort(sample_id: str) -> str:
    """ACC-cohort tile IDs are ACC_-prefixed; everything else is GDC by
    elimination, since the corpus contains only these two cohorts."""
    return "ACC" if sample_id.startswith("ACC_") else "GDC"


def run_similarity(
    embeddings_dir: Path,
    cosine_out: Path,
    cka_out: Path,
    models: Optional[List[str]] = None,
) -> None:
    """
    Reads cached embeddings for each model and computes:
      - per-tile cosine similarity (baseline vs each perturbed variant),
        tagged with cohort (ACC/GDC)
      - linear CKA across all tiles per (model, perturbation, severity),
        computed THREE times per cell: POOLED (all tiles, supports H1/H4),
        ACC-only, and GDC-only (supports H2/H3) -- each gated by an
        independent n>=3 floor, since a cohort subset can legitimately be
        too small even when the pooled set is not.
    """
    if models is None:
        models = [p.stem.replace("embeddings_", "") for p in embeddings_dir.glob("embeddings_*.h5")]

    cosine_out.parent.mkdir(parents=True, exist_ok=True)
    cka_out.parent.mkdir(parents=True, exist_ok=True)

    with open(cosine_out, "w", newline="", encoding="utf-8") as cos_fh, \
         open(cka_out, "w", newline="", encoding="utf-8") as cka_fh:

        cos_writer = csv.writer(cos_fh)
        cos_writer.writerow(["model", "sample_id", "cohort", "perturbation_id", "severity", "cosine_similarity"])

        cka_writer = csv.writer(cka_fh)
        cka_writer.writerow(["model", "perturbation_id", "severity", "cohort", "n_tiles", "linear_cka"])

        for model_name in models:
            store_path = embeddings_dir / f"embeddings_{model_name}.h5"
            if not store_path.exists():
                log.warning("No embedding store for %s at %s; skipping.", model_name, store_path)
                continue

            log.info("Computing similarity metrics for %s...", model_name)
            store = EmbeddingStore(store_path)
            all_keys = store.keys()

            baselines: Dict[str, np.ndarray] = {}
            perturbed: Dict[Tuple[int, str], Dict[str, np.ndarray]] = {}

            for key in all_keys:
                if key.startswith("baseline__"):
                    sample_id = key[len("baseline__"):]
                    baselines[sample_id] = store.read(key)
                else:
                    pid_str, severity, sample_id = key.split("__", 2)
                    pid = int(pid_str[1:])
                    perturbed.setdefault((pid, severity), {})[sample_id] = store.read(key)

            # --- Per-tile cosine similarity, now cohort-tagged ---
            for (pid, severity), samples in perturbed.items():
                common = [sid for sid in samples if sid in baselines]
                if not common:
                    continue
                base_mat = np.stack([baselines[sid] for sid in common])
                pert_mat = np.stack([samples[sid] for sid in common])
                sims = cosine_similarity_matrix_rows(base_mat, pert_mat)
                for sid, sim in zip(common, sims):
                    cos_writer.writerow([model_name, sid, derive_cohort(sid), pid, severity, f"{sim:.6f}"])

                # --- Linear CKA, computed three ways: POOLED, ACC-only, GDC-only ---
                for cohort_label, subset_ids in [
                    ("POOLED", common),
                    ("ACC", [sid for sid in common if derive_cohort(sid) == "ACC"]),
                    ("GDC", [sid for sid in common if derive_cohort(sid) == "GDC"]),
                ]:
                    if len(subset_ids) < 3:  # CKA is meaningless/unstable below this,
                        continue              # independent floor per cohort group
                    sub_base = np.stack([baselines[sid] for sid in subset_ids])
                    sub_pert = np.stack([samples[sid] for sid in subset_ids])
                    cka_val = linear_cka(sub_base, sub_pert)
                    cka_writer.writerow([model_name, pid, severity, cohort_label, len(subset_ids), f"{cka_val:.6f}"])

            store.close()
            log.info("%s: similarity metrics written for %d perturbation/severity combinations.", model_name, len(perturbed))

    log.info("Cosine similarity report: %s", cosine_out)
    log.info("CKA summary report: %s", cka_out)


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["extract", "similarity", "both"], default="both")
    parser.add_argument("--model", choices=list(ADAPTER_REGISTRY) + ["all"], default="all")
    parser.add_argument("--tiles-dir", type=Path, default=Path("outputs/perturbed_tiles"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("outputs/embeddings"))
    parser.add_argument("--cosine-out", type=Path, default=Path("outputs/cosine_similarity.csv"))
    parser.add_argument("--cka-out", type=Path, default=Path("outputs/cka_summary.csv"))
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Overrides the per-model default batch size")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    models = list(ADAPTER_REGISTRY.keys() - {"mock"}) if args.model == "all" else [args.model]

    if args.mode in ("extract", "both"):
        for model_name in models:
            try:
                run_extraction(
                    model_name=model_name, tiles_dir=args.tiles_dir,
                    embeddings_dir=args.embeddings_dir, batch_size=args.batch_size,
                    num_workers=args.num_workers, device=args.device, precision=args.precision,
                )
            except ModelEnvironmentError as exc:
                if args.model == "all":
                    # Multi-conda-env setups (one env per model, per project
                    # notes) mean this is routine, not a failure: this
                    # environment simply isn't the right one for this
                    # model. Log clearly and move on to the next requested
                    # model rather than aborting the whole run.
                    log.warning(
                        "Skipping %s in this environment: %s\n"
                        "Run this script again with --model %s from that "
                        "model's own conda environment.",
                        model_name, exc, model_name,
                    )
                    continue
                # A single, explicitly-requested model failing its
                # environment check IS something the user needs to see
                # and fix immediately, not silently skip.
                raise

    if args.mode in ("similarity", "both"):
        run_similarity(
            embeddings_dir=args.embeddings_dir, cosine_out=args.cosine_out,
            cka_out=args.cka_out, models=models,
        )


if __name__ == "__main__":
    main()
