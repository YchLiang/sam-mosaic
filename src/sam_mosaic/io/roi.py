"""ROI mask utilities for shapefile-based region masking."""

import math
from pathlib import Path
from typing import Optional

import numpy as np


def rasterize_roi_mask(
    shp_path: str,
    height: int,
    width: int,
    crs: object,
    transform: object,
) -> np.ndarray:
    """Rasterize a shapefile to a binary mask matching image dimensions.

    Uses fiona to read the SHP geometries and rasterio.features.rasterize
    to burn them into a binary mask at the image's CRS and resolution.

    Args:
        shp_path: Path to shapefile (.shp).
        height: Image height in pixels.
        width: Image width in pixels.
        crs: Coordinate reference system of the image.
        transform: Affine transform of the image.

    Returns:
        Binary uint8 mask (H, W) where 255 = inside ROI, 0 = outside.

    Raises:
        FileNotFoundError: If shapefile does not exist.
        ValueError: If no geometries found in shapefile.
    """
    import fiona
    from shapely.geometry import shape
    from rasterio.features import rasterize

    shp_path = Path(shp_path)
    if not shp_path.exists():
        raise FileNotFoundError(f"ROI shapefile not found: {shp_path}")

    geometries = []
    with fiona.open(str(shp_path)) as src:
        for feature in src:
            geom = shape(feature["geometry"])
            geometries.append(geom)

    if not geometries:
        raise ValueError(f"No geometries found in {shp_path}")

    # Rasterize: burn value 255 for all ROI pixels
    mask = rasterize(
        geometries,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        default_value=255,
        dtype=np.uint8,
    )

    return mask


def crop_roi_mask(
    full_mask: np.ndarray,
    row: int,
    col: int,
    tile_size: int,
    padding: int,
    img_width: int,
    img_height: int,
) -> np.ndarray:
    """Crop a tile-sized region from the full-image ROI mask.

    The crop matches exactly the padded tile bounds as computed by
    load_tile() in reader.py, so the ROI mask aligns pixel-for-pixel
    with the tile data that SAM sees.

    Args:
        full_mask: Full-image ROI mask (img_height, img_width).
        row: Tile row index (0-based).
        col: Tile column index (0-based).
        tile_size: Useful tile area size.
        padding: Padding per side.
        img_width: Full image width.
        img_height: Full image height.

    Returns:
        Cropped ROI mask matching the padded tile dimensions.
    """
    # Replicate the exact same geometry as load_tile() in reader.py
    useful_w = min(tile_size, img_width - col * tile_size)
    useful_h = min(tile_size, img_height - row * tile_size)

    tile_x = col * tile_size
    tile_y = row * tile_size

    pad_left = min(padding, tile_x)
    pad_right = min(padding, img_width - tile_x - useful_w)
    pad_top = min(padding, tile_y)
    pad_bottom = min(padding, img_height - tile_y - useful_h)

    read_x = tile_x - pad_left
    read_y = tile_y - pad_top
    read_w = useful_w + pad_left + pad_right
    read_h = useful_h + pad_top + pad_bottom

    return full_mask[read_y:read_y + read_h, read_x:read_x + read_w]
