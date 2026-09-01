"""Raster to vector conversion (polygonization)."""

from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import numpy as np
import rasterio.features
from rasterio.transform import Affine
from shapely.geometry import shape, mapping


def vectorize_labels(
    labels: np.ndarray,
    output_path: Union[str, Path],
    crs: Optional[object] = None,
    transform: Optional[Affine] = None,
    simplify_tolerance: float = 0.0
) -> int:
    """Convert label raster to vector polygons and save.

    Automatically chooses format based on file extension:
    - .shp -> Shapefile
    - .gpkg -> GeoPackage
    - .geojson -> GeoJSON

    Args:
        labels: Label array of shape (H, W).
        output_path: Output file path.
        crs: Coordinate reference system.
        transform: Affine transform for georeferencing.
        simplify_tolerance: Polygon simplification tolerance (0 = no simplification).

    Returns:
        Number of polygons created.
    """
    output_path = Path(output_path)
    ext = output_path.suffix.lower()

    # Extract polygons
    features = extract_polygons(labels, transform, simplify_tolerance)

    if len(features) == 0:
        return 0

    # Save based on extension
    if ext == ".shp":
        save_shapefile(features, output_path, crs)
    elif ext == ".gpkg":
        save_geopackage(features, output_path, crs)
    elif ext == ".geojson":
        save_geojson(features, output_path, crs)
    else:
        raise ValueError(f"Unsupported output format: {ext}")

    return len(features)


def extract_polygons(
    labels: np.ndarray,
    transform: Optional[Affine] = None,
    simplify_tolerance: float = 0.0
) -> List[Dict[str, Any]]:
    """Extract polygons from label array.

    Uses rasterio.features.shapes for efficient vectorization, then:
    1. Groups polygons by label_id and dissolves each group into a single
       geometry (handles multi-part labels from tile-boundary merging).
    2. Explodes multi-part results into individual polygons.

    Note on simplify_tolerance: simplifying polygons independently
    causes topological overlaps between adjacent polygons because
    shared edges are simplified inconsistently. The default is 0
    (no simplification), which guarantees a topologically clean
    coverage. Use simplify_tolerance > 0 only if you accept minor
    overlaps and prefer smaller file sizes.

    Args:
        labels: Label array of shape (H, W).
        transform: Affine transform for georeferencing.
        simplify_tolerance: Polygon simplification tolerance (default 0.0).
            Set to 0 to guarantee no overlaps. Any value > 0 will introduce
            minor overlaps between adjacent polygons.

    Returns:
        List of feature dictionaries with geometry and properties.
    """
    from shapely.validation import make_valid
    from shapely.ops import unary_union
    from collections import defaultdict

    if transform is None:
        transform = Affine.identity()

    # Step 1: Collect all raw polygons grouped by label_id
    # rasterio.features.shapes returns one polygon per connected region,
    # so a label that was merged across tile boundaries can produce
    # multiple separate polygons with the same label_id.
    grouped = defaultdict(list)
    for geom, value in rasterio.features.shapes(
        labels.astype(np.int32),
        transform=transform
    ):
        value = int(value)
        if value == 0:  # Skip background
            continue
        polygon = shape(geom)
        if not polygon.is_valid:
            polygon = make_valid(polygon)
        grouped[value].append(polygon)

    # Step 2: Dissolve each label_id into a single geometry,
    # then optionally simplify, then explode into individual polygons.
    features = []
    for label_id, polygons in grouped.items():
        # Dissolve all parts of this label into one geometry
        if len(polygons) == 1:
            dissolved = polygons[0]
        else:
            dissolved = unary_union(polygons)

        # Simplify (WARNING: any tolerance > 0 introduces overlaps
        # between adjacent polygons with different label_ids)
        if simplify_tolerance > 0:
            dissolved = dissolved.simplify(simplify_tolerance, preserve_topology=True)

        # Ensure valid
        if not dissolved.is_valid:
            dissolved = make_valid(dissolved)

        # Explode MultiPolygon / GeometryCollection into individual polygons
        geoms_to_save = _extract_single_polygons(dissolved)

        for poly in geoms_to_save:
            features.append({
                "geometry": mapping(poly),
                "properties": {
                    "label_id": label_id,
                    "area_m2": poly.area,
                    "perimeter_m": poly.length,
                }
            })

    return features


def _extract_single_polygons(geom) -> list:
    """Extract individual Polygon geometries from any geometry type.

    Handles Polygon, MultiPolygon, GeometryCollection, etc.

    Args:
        geom: A shapely geometry.

    Returns:
        List of Polygon geometries.
    """
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

    if isinstance(geom, Polygon):
        return [geom]
    elif isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        result = []
        for g in geom.geoms:
            result.extend(_extract_single_polygons(g))
        return result
    else:
        # LineString, Point, etc. — skip non-polygon types
        return []


def save_shapefile(
    features: List[Dict[str, Any]],
    path: Union[str, Path],
    crs: Optional[object] = None
) -> None:
    """Save features to Shapefile.

    Args:
        features: List of feature dictionaries.
        path: Output file path.
        crs: Coordinate reference system.
    """
    import fiona
    from fiona.crs import from_epsg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(features) == 0:
        return

    # Define schema
    schema = {
        "geometry": "Polygon",
        "properties": {
            "label_id": "int",
            "area_m2": "float",
            "perimeter_m": "float",
        }
    }

    # Get CRS
    if crs is None:
        crs_dict = from_epsg(4326)  # Default to WGS84
    elif hasattr(crs, "to_dict"):
        crs_dict = crs.to_dict()
    else:
        crs_dict = crs

    with fiona.open(
        str(path),
        "w",
        driver="ESRI Shapefile",
        crs=crs_dict,
        schema=schema
    ) as dst:
        for feature in features:
            dst.write({
                "geometry": feature["geometry"],
                "properties": feature["properties"],
            })


def save_geopackage(
    features: List[Dict[str, Any]],
    path: Union[str, Path],
    crs: Optional[object] = None
) -> None:
    """Save features to GeoPackage.

    Args:
        features: List of feature dictionaries.
        path: Output file path.
        crs: Coordinate reference system.
    """
    import fiona
    from fiona.crs import from_epsg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(features) == 0:
        return

    schema = {
        "geometry": "Polygon",
        "properties": {
            "label_id": "int",
            "area_m2": "float",
            "perimeter_m": "float",
        }
    }

    if crs is None:
        crs_dict = from_epsg(4326)
    elif hasattr(crs, "to_dict"):
        crs_dict = crs.to_dict()
    else:
        crs_dict = crs

    with fiona.open(
        str(path),
        "w",
        driver="GPKG",
        crs=crs_dict,
        schema=schema,
        layer="segments"
    ) as dst:
        for feature in features:
            dst.write({
                "geometry": feature["geometry"],
                "properties": feature["properties"],
            })


def save_geojson(
    features: List[Dict[str, Any]],
    path: Union[str, Path],
    crs: Optional[object] = None
) -> None:
    """Save features to GeoJSON.

    Args:
        features: List of feature dictionaries.
        path: Output file path.
        crs: Coordinate reference system (ignored for GeoJSON, always WGS84).
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": f["geometry"],
                "properties": f["properties"],
            }
            for f in features
        ]
    }

    with open(path, "w") as f:
        json.dump(geojson, f)
