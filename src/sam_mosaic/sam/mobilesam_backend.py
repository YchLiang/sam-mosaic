"""MobileSAM predictor backend for segmentation.

MobileSAM keeps SAM's prompt encoder + mask decoder and swaps the heavy
ViT image encoder for a ~5M-parameter TinyViT, so the whole model is ~40 MB
(~0.1x of SAM2-hiera-large) and the per-pass image encode drops from
hundreds of ms to a few ms -- on this workload the pass cost becomes
mask-decoder-dominated.

The ``mobile_sam`` package (pip install -e of the MobileSAM repo) exposes
the original-SAM API with the same automatic mask generator parameter
surface the SAM2 backend uses (custom ``point_grids``, ``pred_iou_thresh``,
``stability_score_thresh``, ``min_mask_region_area``, ``box_nms_thresh``,
``crop_n_layers``), and its output dicts carry the same keys
(``segmentation`` / ``predicted_iou`` / ``stability_score``). This backend
is therefore a near-drop-in of the SAM2 path; the IoU-head calibration is
also SAM-like, so the same threshold schedule (0.93 start) applies.

Requires: the ``mobile_sam`` package importable in the active environment
(e.g. ``pip install -e <MobileSAM repo>`` plus ``timm``) and a checkpoint
(official ``mobile_sam.pt``, model type ``vit_t``).
"""

import gc
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from sam_mosaic.sam.masks import Mask


class MobileSAMPredictor:
    """Wrapper around MobileSAM exposing the same interface as the other
    backends (``SAMPredictor`` / ``SAM3Predictor``).

    Attributes:
        checkpoint_path: Path to the MobileSAM checkpoint (mobile_sam.pt).
        device: Torch device (cuda/cpu).
        model: Loaded MobileSAM model.
    """

    backend = "mobilesam"

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        model_type: str = "vit_t",
    ):
        """Initialize MobileSAM predictor.

        Args:
            checkpoint_path: Path to the MobileSAM checkpoint file.
            device: Device to use ('cuda', 'cpu', or None for auto).
            model_type: mobile_sam registry key (``vit_t`` for the official
                mobile_sam.pt; vit_b/vit_l/vit_h also exist in the registry).
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type

        self._model = None
        self._predictor = None
        self._current_image = None

    def load_model(self, debug: bool = False) -> None:
        """Load the MobileSAM model into memory.

        Args:
            debug: If True, print debug messages during loading.

        Raises:
            ImportError: If the ``mobile_sam`` package is not installed.
            FileNotFoundError: If the checkpoint file doesn't exist.
        """
        def _debug(msg):
            if debug:
                print(f"[DEBUG] {msg}", flush=True)

        if self._model is not None:
            _debug("Model already loaded, skipping")
            return

        _debug(f"Checkpoint path: {self.checkpoint_path}")
        _debug(f"Model type: {self.model_type}")
        _debug(f"Device: {self.device}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        try:
            # mobile_sam -> timm -> torchvision; keep PIL ahead of it (Windows
            # DLL load-order pitfall, same as the SAM2/torchvision case).
            from PIL import Image  # noqa: F401
            from mobile_sam import SamPredictor, sam_model_registry
        except ImportError as e:
            raise ImportError(
                "MobileSAM backend requires the `mobile_sam` package. Setup:\n"
                "  git clone https://github.com/ChaoningZhang/MobileSAM.git\n"
                "  pip install -e ./MobileSAM  (+ pip install timm)\n"
                "  # checkpoint: MobileSAM/weights/mobile_sam.pt (~40 MB)"
            ) from e

        _debug("Building MobileSAM model ...")
        self._model = sam_model_registry[self.model_type](
            checkpoint=str(self.checkpoint_path)
        )
        self._model.to(device=self.device)
        self._model.eval()
        _debug("model built")

        self._predictor = SamPredictor(self._model)
        _debug("SamPredictor ready")

    def set_image(self, image: np.ndarray) -> None:
        """Set the image for prediction.

        Args:
            image: RGB image array of shape (H, W, 3) with dtype uint8.

        Raises:
            ValueError: If image format is invalid.
        """
        if self._predictor is None:
            self.load_model()

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 image, got {image.dtype}")

        self._predictor.set_image(image)
        self._current_image = image

    def predict_points_batched(
        self,
        image: np.ndarray,
        points: np.ndarray,
        iou_threshold: float = 0.88,
        stability_threshold: float = 0.92,
        min_mask_area: int = 100,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0
    ) -> List[Mask]:
        """Predict masks for a custom point grid via MobileSAM's AMG.

        Same flow as the SAM2 backend: normalized point grid -> AMG ->
        IoU + stability filtering, box NMS, output dicts converted to
        ``Mask`` objects.

        Args:
            image: RGB image array (H, W, 3) uint8.
            points: Array of shape (N, 2) with (x, y) pixel coordinates.
            iou_threshold: Minimum predicted IoU to accept a mask.
            stability_threshold: Minimum stability score to accept a mask.
            min_mask_area: Minimum mask area in pixels.
            box_nms_thresh: Box NMS threshold for overlapping masks.
            crop_n_layers: Sub-crop layers, supported by the SAM-style AMG.

        Returns:
            List of Mask objects that passed thresholds.
        """
        if self._model is None:
            self.load_model()

        from mobile_sam import SamAutomaticMaskGenerator

        h, w = image.shape[:2]

        # Convert pixel coords to normalized coords (0-1)
        points_norm = points.astype(np.float32).copy()
        points_norm[:, 0] /= w  # x
        points_norm[:, 1] /= h  # y

        if crop_n_layers > 0:
            point_grids = [points_norm] * (crop_n_layers + 1)
        else:
            point_grids = [points_norm]

        generator = SamAutomaticMaskGenerator(
            model=self._model,
            points_per_side=None,   # Disable automatic grid
            point_grids=point_grids,  # Use custom points
            pred_iou_thresh=iou_threshold,
            stability_score_thresh=stability_threshold,
            min_mask_region_area=min_mask_area,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
        )

        # Patch the SAM-family bug: when a crop produces no masks,
        # crop_boxes becomes a 1D empty tensor and box_area fails.
        _orig_process_crop = generator._process_crop

        def _safe_process_crop(image, crop_box, crop_layer_idx, orig_size):
            crop_data = _orig_process_crop(image, crop_box, crop_layer_idx, orig_size)
            if len(crop_data["rles"]) == 0:
                crop_data["crop_boxes"] = torch.zeros((0, 4), dtype=torch.int32)
            return crop_data

        generator._process_crop = _safe_process_crop

        mask_outputs = generator.generate(image)

        # Free generator memory immediately
        del generator
        gc.collect()
        torch.cuda.empty_cache()

        masks = []
        for mask_data in mask_outputs:
            m = mask_data['segmentation']
            masks.append(Mask(
                data=m.astype(np.uint8),
                score=float(mask_data.get('predicted_iou', 0.0)),
                stability=float(mask_data.get('stability_score', 0.0)),
                point=None,  # Batched - no single point association
            ))

        del mask_outputs
        return masks

    def reset_image(self) -> None:
        """Clear the current image from memory."""
        self._current_image = None
        if self._predictor is not None:
            self._predictor.reset_image()
        gc.collect()
        torch.cuda.empty_cache()

    def unload_model(self) -> None:
        """Unload model from memory."""
        self._model = None
        self._predictor = None
        self._current_image = None
        torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def has_image(self) -> bool:
        """Check if an image is currently set."""
        return self._current_image is not None
