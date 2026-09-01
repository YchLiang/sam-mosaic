"""Tile processing utilities."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from sam_mosaic.config import Config
from sam_mosaic.sam import SAMPredictor
from sam_mosaic.io import TileInfo, ensure_uint8, ensure_rgb
from sam_mosaic.core.multipass import run_multipass_segmentation


@dataclass
class TileResult:
    """Result from processing a single tile.

    Attributes:
        labels: Label array for the useful area (tile_size x tile_size).
        coverage: Coverage percentage for this tile.
        n_labels: Number of unique labels.
        n_passes: Number of passes used.
        row: Tile row index.
        col: Tile column index.
        pass_stats: Detailed per-pass statistics.
    """
    labels: np.ndarray
    coverage: float
    n_labels: int
    n_passes: int
    row: int
    col: int
    pass_stats: dict = None


def process_tile(
    predictor: SAMPredictor,
    tile_info: TileInfo,
    config: Config,
    start_label: int = 1,
    roi_mask: Optional[np.ndarray] = None
) -> TileResult:
    """Process a single tile with multi-pass segmentation.

    Args:
        predictor: SAM predictor (model should be loaded).
        tile_info: Tile information including image data and crop coords.
        config: Configuration object.
        start_label: Starting label ID for this tile.

    Returns:
        TileResult with labels and statistics.
    """
    # Prepare image
    image = tile_info.data
    image = ensure_uint8(image)
    image = ensure_rgb(image)

    # Run multi-pass segmentation (predictor handles image internally)
    labels, combined_mask, stats = run_multipass_segmentation(
        predictor=predictor,
        image=image,
        seg_config=config.segmentation,
        threshold_config=config.threshold,
        start_label=start_label,
        min_region_area=config.merge.min_mask_area,
        roi_mask=roi_mask
    )

    # Crop to useful area
    crop_x = tile_info.crop_x
    crop_y = tile_info.crop_y
    useful_w = tile_info.useful_w if tile_info.useful_w > 0 else tile_info.tile_size
    useful_h = tile_info.useful_h if tile_info.useful_h > 0 else tile_info.tile_size

    useful_labels = labels[crop_y:crop_y + useful_h, crop_x:crop_x + useful_w]
    useful_mask = combined_mask[crop_y:crop_y + useful_h, crop_x:crop_x + useful_w]

    # Calculate coverage for useful area
    coverage = useful_mask.sum() / useful_mask.size * 100

    # Count unique labels (excluding 0)
    n_labels = len(np.unique(useful_labels)) - 1

    return TileResult(
        labels=useful_labels,
        coverage=coverage,
        n_labels=max(0, n_labels),
        n_passes=stats["passes"],
        row=tile_info.row,
        col=tile_info.col,
        pass_stats=stats,
    )


def calculate_grid_dimensions(
    image_width: int,
    image_height: int,
    tile_size: int
) -> Tuple[int, int, int]:
    """Calculate tile grid dimensions.

    Uses ceiling division so that edge tiles (smaller than tile_size)
    are included, ensuring full image coverage.

    Args:
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        tile_size: Tile size in pixels.

    Returns:
        Tuple of (n_cols, n_rows, total_tiles).
    """
    import math
    n_cols = math.ceil(image_width / tile_size)
    n_rows = math.ceil(image_height / tile_size)
    total_tiles = n_cols * n_rows

    return n_cols, n_rows, total_tiles
