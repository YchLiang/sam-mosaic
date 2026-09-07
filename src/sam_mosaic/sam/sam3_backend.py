"""SAM3 predictor backend for segmentation.

SAM3 (Meta, Nov 2025) has a different architecture from SAM2: a DETR-style
detector plus a SAM2-style tracker sharing one ViT vision encoder (848M
params). Its weights are incompatible with ``sam2.build_sam2``, so this
backend wraps the official ``sam3`` package instead.

sam-mosaic drives SAM3 through its point-prompt path (the "SAM 1/2 task"),
which is the interactive instance predictor:

    model = build_sam3_image_model(enable_inst_interactivity=True)
    model.inst_interactive_predictor   # SAM3InteractiveImagePredictor

Two constraints shape the implementation (both mirror the official usage in
``Sam3Image.predict_inst`` / ``Sam3Processor``):

1. The tracker inside the interactive predictor has ``backbone=None``; image
   features must come from the full model's VL backbone via
   ``Sam3Processor.set_image``, then be injected into the predictor.
2. Batched independent point prompts are issued through
   ``SAM3InteractiveImagePredictor._predict`` with point tensors shaped
   ``(B, 1, 2)`` -- the same trick ``SAM2AutomaticMaskGenerator`` uses on
   SAM2. SAM3 ships no automatic mask generator, so thresholding, the
   stability score and box NMS are reimplemented here with the same formulas.

Environment requirements (NOT met by a typical SAM2 env):
    Python >= 3.12, PyTorch >= 2.7, CUDA >= 12.6
    git clone https://github.com/facebookresearch/sam3
    pip install -e ./sam3
    Checkpoint: gated on https://huggingface.co/facebook/sam3 -- request
    access, run ``hf auth login``, then either pass ``checkpoint="hf"`` to
    auto-download ``sam3.pt``, or download it manually and pass its path.
"""

import gc
import hashlib
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sam_mosaic.sam.masks import Mask

# sam3's hole-filling postprocess falls back with a UserWarning when the
# optional triton kernels are missing (always on Windows). Harmless -- the
# masks are unaffected -- but it fires per batch, so keep it out of logs.
warnings.filterwarnings("ignore", message=".*Skipping the post-processing step.*")


class SAM3Predictor:
    """Wrapper around SAM3 exposing the same interface as ``SAMPredictor``.

    Attributes:
        checkpoint_path: Path to the SAM3 checkpoint (.pt), or a sentinel
            such as "hf" to download from the (gated) HuggingFace repo.
        device: Torch device (cuda/cpu).
        model: The SAM3 image model (detector + interactive tracker).
    """

    backend = "sam3"

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        points_per_batch: int = 64,
        enable_segmentation: bool = True,
    ):
        """Initialize SAM3 predictor.

        Args:
            checkpoint_path: Path to the SAM3 checkpoint file, or "hf" to
                auto-download from HuggingFace (requires access + login).
            device: Device to use ('cuda', 'cpu', or None for auto).
            points_per_batch: Number of independent point prompts decoded
                per forward (SAM2's AMG default is 64).
            enable_segmentation: Keep the detector's segmentation head.
                Only needed for SAM3 text/box prompting; point prompting
                works without it (set False to save ~0.5 GB VRAM).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.points_per_batch = points_per_batch
        self.enable_segmentation = enable_segmentation

        self._model = None
        self._processor = None
        self._inst = None  # SAM3InteractiveImagePredictor
        self._state = None  # Sam3Processor state for the current image
        self._current_image = None
        self._total_vram = 0  # device total, cached at load for the VRAM guard
        # Content-keyed cache of the last backbone forward. Multi-pass
        # segmentation re-feeds byte-identical images (passes that added no
        # masks), and the ViT-L forward at 1008x1008 is the dominant per-pass
        # cost -- same rationale as the SAM2 encoder cache.
        self._cache_key = None
        self._cache_state = None

    def load_model(self, debug: bool = False) -> None:
        """Load the SAM3 image model into memory.

        Args:
            debug: If True, print debug messages during loading.

        Raises:
            ImportError: If the ``sam3`` package is not installed.
            FileNotFoundError: If a local checkpoint path doesn't exist.
        """
        def _debug(msg):
            if debug:
                print(f"[DEBUG] {msg}", flush=True)

        if self._model is not None:
            _debug("Model already loaded, skipping")
            return

        _debug(f"Checkpoint path: {self.checkpoint_path}")
        _debug(f"Device: {self.device}")

        try:
            from sam3.model.sam3_image import Sam3Image  # noqa: F401  (sanity import)
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as e:
            raise ImportError(
                "SAM3 backend requires the `sam3` package (Python >= 3.12, "
                "torch >= 2.7, CUDA >= 12.6). Setup:\n"
                "  conda create -n sam3 python=3.12 && conda activate sam3\n"
                "  pip install torch torchvision --index-url "
                "https://download.pytorch.org/whl/cu128\n"
                "  git clone https://github.com/facebookresearch/sam3\n"
                "  pip install -e ./sam3\n"
                "  # then request checkpoint access at huggingface.co/facebook/sam3 "
                "and run `hf auth login`"
            ) from e

        # Resolve checkpoint source
        sentinel = str(self.checkpoint_path).lower()
        if sentinel in ("hf", "auto", "download"):
            checkpoint_arg = None
            load_from_HF = True
            _debug("Checkpoint will be downloaded from HuggingFace (facebook/sam3)")
        elif self.checkpoint_path.exists():
            checkpoint_arg = str(self.checkpoint_path)
            load_from_HF = False
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}. "
                "Pass an existing .pt path or checkpoint='hf' to auto-download "
                "(gated repo; requires access approval + `hf auth login`)."
            )

        _debug("Calling build_sam3_image_model(enable_inst_interactivity=True)...")
        self._model = build_sam3_image_model(
            checkpoint_path=checkpoint_arg,
            load_from_HF=load_from_HF,
            enable_inst_interactivity=True,
            enable_segmentation=self.enable_segmentation,
            device=self.device,
        )
        _debug("build_sam3_image_model() complete")

        self._processor = Sam3Processor(self._model, device=self.device)
        self._inst = self._model.inst_interactive_predictor
        if self._inst is None:
            raise RuntimeError(
                "SAM3 model was built without the interactive predictor; "
                "this should not happen when enable_inst_interactivity=True."
            )
        if self.device.startswith("cuda"):
            self._total_vram = torch.cuda.get_device_properties(0).total_memory
        _debug("SAM3InteractiveImagePredictor ready")

    def set_image(self, image: np.ndarray) -> None:
        """Encode the image for prediction (cached by content).

        Args:
            image: RGB image array of shape (H, W, 3) with dtype uint8.

        Raises:
            ValueError: If image format is invalid.
        """
        if self._processor is None:
            self.load_model()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}")

        key = hashlib.md5(np.ascontiguousarray(image).tobytes()).hexdigest()
        if self._cache_key == key and self._cache_state is not None:
            self._state = self._cache_state
            self._current_image = image
            return

        self._state = self._processor.set_image(image)
        self._cache_key = key
        self._cache_state = self._state
        self._current_image = image

    def _inject_features(self) -> None:
        """Feed the current image's backbone features into the interactive predictor.

        Mirrors ``Sam3Image.predict_inst``: the tracker has no backbone of its
        own, so features computed by ``Sam3Processor.set_image`` are injected
        here and reused across all point batches of this pass.
        """
        backbone_out = self._state["backbone_out"]["sam2_backbone_out"]
        (_, vision_feats, _, _) = self._inst.model._prepare_backbone_features(
            backbone_out
        )
        # Add no_mem_embed, which is added to the lowest-res feat. map during
        # training on videos (same as the SAM2 image predictor).
        vision_feats[-1] = vision_feats[-1] + self._inst.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], self._inst._bb_feat_sizes[::-1]
            )
        ][::-1]
        self._inst._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._inst._is_image_set = True
        self._inst._is_batch = False
        self._inst._orig_hw = [tuple(self._current_image.shape[:2])]

    def _clear_features(self) -> None:
        """Release injected features (as ``predict_inst`` does after each call)."""
        self._inst._features = None
        self._inst._is_image_set = False

    def predict_points_batched(
        self,
        image: np.ndarray,
        points: np.ndarray,
        iou_threshold: float = 0.88,
        stability_threshold: float = 0.92,
        min_mask_area: int = 100,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
    ) -> List[Mask]:
        """Predict masks for many independent point prompts, batched.

        Semantically equivalent to the SAM2 path (``SAM2AutomaticMaskGenerator``
        with a custom point grid): multimask output per point, IoU + stability
        filtering, box NMS. Scores have the same [0, 1] semantics -- SAM3's
        IoU head is sigmoid-based like SAM2's.

        Args:
            image: RGB image array (H, W, 3) uint8.
            points: Array of shape (N, 2) with (x, y) pixel coordinates.
            iou_threshold: Minimum predicted IoU to accept a mask.
            stability_threshold: Minimum stability score to accept a mask.
            min_mask_area: Minimum mask area in pixels.
            box_nms_thresh: Box NMS IoU threshold for overlapping masks.
            crop_n_layers: Sub-crop layers are a SAM2-AMG feature; the SAM3
                backend does not implement them (must be 0).

        Returns:
            List of Mask objects that passed thresholds (NMS-filtered).
        """
        if crop_n_layers > 0:
            raise ValueError(
                "SAM3 backend does not support crop_n_layers > 0 (SAM2-AMG "
                "sub-crop feature). Set crop_n_layers=0."
            )
        if self._processor is None:
            self.load_model()
        if len(points) == 0:
            return []

        h, w = image.shape[:2]
        self.set_image(image)
        self._inject_features()

        device = self._inst.model.device
        all_masks: List[Mask] = []
        kept_boxes: List[torch.Tensor] = []
        kept_scores: List[torch.Tensor] = []

        try:
            for start in range(0, len(points), self.points_per_batch):
                batch = np.asarray(points[start:start + self.points_per_batch],
                                   dtype=np.float32)
                coords = torch.as_tensor(batch, device=device)[:, None, :]  # (B, 1, 2)
                labels = torch.ones(coords.shape[:2], dtype=torch.int, device=device)
                # Normalize pixel coords the same way the public predict() does
                coords = self._inst._transforms.transform_coords(
                    coords, normalize=True, orig_hw=(h, w)
                )

                # (B, 1, 2) points => B independent objects (batched mode),
                # each returning 3 multimask candidates + IoU predictions.
                masks_logits, ious, low_res = self._inst._predict(
                    coords,
                    labels,
                    multimask_output=True,
                    return_logits=True,
                )
                # masks_logits: (B, 3, H, W) float logits at original res
                # ious: (B, 3) sigmoid IoU predictions; low_res: (B, 3, 256, 256)

                # Stability score on low-res logits, identical to SAM2's AMG:
                # calculate_stability_score(logits, thresh=0.0, offset=1.0)
                intersections = (low_res > 1.0).sum(dim=(-2, -1))
                unions = (low_res > -1.0).sum(dim=(-2, -1))
                stability = intersections.float() / unions.float().clamp(min=1)

                binary = masks_logits > self._inst.mask_threshold
                areas = binary.sum(dim=(-2, -1))
                keep = (
                    (ious >= iou_threshold)
                    & (stability >= stability_threshold)
                    & (areas >= min_mask_area)
                )
                if not keep.any():
                    continue

                keep_logits = masks_logits[keep]  # (K, H, W)
                keep_binary = keep_logits > self._inst.mask_threshold
                # Box NMS over the kept masks (same as SAM2 AMG's box NMS)
                from torchvision.ops import masks_to_boxes

                boxes = masks_to_boxes(keep_binary)  # (K, 4) xyxy
                scores = ious[keep]

                all_masks.extend(
                    Mask(
                        # NOTE: store the thresholded binary mask, NOT the
                        # raw logits (return_logits=True gives float logits).
                        data=keep_binary[i].cpu().numpy().astype(np.uint8),
                        score=float(scores[i]),
                        stability=float(stability[keep][i]),
                        point=None,  # Batched - no single point association
                    )
                    for i in range(keep_binary.shape[0])
                )
                kept_boxes.append(boxes)
                kept_scores.append(scores)

                # VRAM guard: release cached blocks before the allocator's
                # reservation grows into WDDM system-memory fallback territory
                # (once spilled, every kernel runs over PCIe -- ~10x slowdown).
                if self._total_vram and torch.cuda.memory_reserved() > 0.85 * self._total_vram:
                    torch.cuda.empty_cache()
        finally:
            self._clear_features()

        if not all_masks:
            return []

        boxes = torch.cat(kept_boxes, dim=0)
        scores = torch.cat(kept_scores, dim=0)
        if boxes.shape[0] > 1 and box_nms_thresh < 1.0:
            from torchvision.ops import nms

            keep = nms(boxes, scores, iou_threshold=box_nms_thresh)
            keep_set = set(keep.cpu().tolist())
            all_masks = [m for i, m in enumerate(all_masks) if i in keep_set]

        return all_masks

    def reset_image(self) -> None:
        """Clear the current image (and its cached features) from memory."""
        self._current_image = None
        self._state = None
        self._cache_key = None
        self._cache_state = None
        if self._inst is not None:
            self._clear_features()
            self._inst.reset_predictor()
        gc.collect()
        torch.cuda.empty_cache()

    def unload_model(self) -> None:
        """Unload the model from memory."""
        self._model = None
        self._processor = None
        self._inst = None
        self._state = None
        self._current_image = None
        self._cache_key = None
        self._cache_state = None
        torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def has_image(self) -> bool:
        """Check if an image is currently set."""
        return self._current_image is not None
