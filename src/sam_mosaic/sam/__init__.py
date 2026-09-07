"""SAM predictor backends (SAM2 / SAM3) and mask utilities."""

from pathlib import Path

from sam_mosaic.sam.masks import Mask, apply_black_mask
from sam_mosaic.sam.predictor import SAMPredictor

__all__ = [
    "SAMPredictor",
    "SAM3Predictor",
    "create_predictor",
    "Mask",
    "apply_black_mask",
    "detect_backend",
]


def detect_backend(checkpoint: str, backend: str = "auto") -> str:
    """Resolve which SAM backend to use for a checkpoint.

    Args:
        checkpoint: Checkpoint path (or "hf" for SAM3 auto-download).
        backend: "auto", "sam2" or "sam3". "auto" picks "sam3" when the
            checkpoint filename contains "sam3" (e.g. sam3.pt,
            sam3.1_multiplex.pt), otherwise "sam2".

    Returns:
        Resolved backend name ("sam2" or "sam3").
    """
    if backend not in ("auto", "sam2", "sam3"):
        raise ValueError(f"sam_backend must be 'auto', 'sam2' or 'sam3', got {backend!r}")
    if backend != "auto":
        return backend
    name = Path(str(checkpoint)).stem.lower()
    return "sam3" if "sam3" in name else "sam2"


def create_predictor(
    checkpoint: str,
    model_type: str = "large",
    backend: str = "auto",
    device=None,
):
    """Create the predictor matching the checkpoint's SAM generation.

    SAM2 checkpoints (sam2.1_hiera_*.pt) load through the original
    ``SAMPredictor``; SAM3 checkpoints (sam3*.pt) go through the ``sam3``
    package backend. The two expose the same interface
    (``load_model``/``set_image``/``predict_points_batched``/...), so the
    pipeline treats them interchangeably.

    Args:
        checkpoint: Checkpoint path, or "hf" for SAM3 HuggingFace download.
        model_type: SAM2 model type (ignored for SAM3).
        backend: "auto" (detect from checkpoint name), "sam2" or "sam3".
        device: Torch device (None = auto).

    Returns:
        SAMPredictor or SAM3Predictor instance (model not yet loaded).
    """
    resolved = detect_backend(checkpoint, backend)
    if resolved == "sam3":
        from sam_mosaic.sam.sam3_backend import SAM3Predictor

        return SAM3Predictor(checkpoint, device=device)
    return SAMPredictor(checkpoint, device=device, model_type=model_type)


def __getattr__(name):
    """PEP 562: lazily expose SAM3Predictor (its module imports nothing
    heavy, but keep the sam3-free import path of this package intact)."""
    if name == "SAM3Predictor":
        from sam_mosaic.sam.sam3_backend import SAM3Predictor

        return SAM3Predictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
