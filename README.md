# Remote SAMsing

**From Segment Anything to Segment Everything**

[![arXiv](https://img.shields.io/badge/arXiv-2605.00256-b31b1b.svg)](https://arxiv.org/abs/2605.00256)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21841054.svg)](https://doi.org/10.5281/zenodo.21841054)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Cite](https://img.shields.io/badge/Cite-CITATION.cff-green.svg)](CITATION.cff)

Segment large remote sensing images (satellite, aerial, drone) that exceed GPU memory by processing them in tiles and intelligently merging the results. Built on SAM2, the `sam-mosaic` package achieves 91--98% coverage across diverse scenes without any model fine-tuning or manual annotation.

Tested on 7 scenes spanning 5 cm to 4.78 m GSD, two spectral compositions (natural RGB and MNF false-color), and two landscape types (urban and agricultural), including a scalability test on a 36,000 x 54,000 pixel mosaic (1.94 billion pixels).

> **Paper:** O. L. F. de Carvalho, O. A. de Carvalho Junior, A. O. de Albuquerque, and D. Guerreiro e Silva, "Remote SAMsing: From Segment Anything to Segment Everything," *arXiv preprint arXiv:2605.00256*, 2026. [[arXiv]](https://arxiv.org/abs/2605.00256)

## Features

- **Multi-pass segmentation** with adaptive thresholds for high coverage (91--98%)
- **Black mask focusing** to direct SAM toward residual unsegmented areas
- **Best-match boundary merge** at tile edges using LUT + Union-Find (parameter-free, O(n) complexity)
- **Dense Grid point strategy** (default) for robust performance across object scales
- **Adaptive tile padding** to ensure clean merges at boundaries
- **GeoTIFF support** with CRS and georeferencing preservation
- **Multiple output formats**: Raster labels (TIFF) + Vector polygons (Shapefile/GeoPackage)

---

## Quick Start

### 1. Install

**Option A: Using conda (recommended for exact reproducibility)**

```bash
# Clone the repository
git clone https://github.com/osmarluiz/sam-mosaic.git
cd sam-mosaic

# Create conda environment from file
conda env create -f environment.yml
conda activate ts_annotator
```

**Option B: Manual installation**

```bash
# Clone the repository
git clone https://github.com/osmarluiz/sam-mosaic.git
cd sam-mosaic

# Install the package
pip install -e .

# Install SAM2 from PyPI (recommended)
pip install sam2
```

> **Important - SAM2 Version**: This project uses `sam2` from PyPI ([JinsuaFeito-dev fork](https://github.com/JinsuaFeito-dev/segment-anything-2)), NOT the official Facebook repository. This version (1.1.0+) has been tested for stable GPU memory usage during large-scale processing. Do NOT install from `pip install git+https://github.com/facebookresearch/sam2.git` as it may cause memory leaks.

### 2. Download SAM2 Model

Download the SAM2 checkpoint (~857MB):

```bash
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
cd ..
```

> See `checkpoints/README.md` for other model sizes (tiny, small, base).

### 3. Run Segmentation

```bash
# Run from inside the sam-mosaic directory (uses default config)
sam-mosaic /path/to/your/image.tif /path/to/output/

# Example (run from sam-mosaic folder)
sam-mosaic /data/ortofoto.tif /results/segmentation/

# Or specify checkpoint explicitly (can run from anywhere)
sam-mosaic /path/to/image.tif /path/to/output/ --checkpoint /path/to/checkpoints/sam2.1_hiera_large.pt
```

That's it! The tool will generate:
- `labels.tif` - Raster with segment labels
- `segments.shp` - Vectorized polygons (Shapefile)
- `stats.json` - Processing statistics

---

## Using SAM 3

sam-mosaic supports both SAM2 and **SAM3** (Meta, Nov 2025) checkpoints. The
backend is selected automatically from the checkpoint filename (anything
containing `sam3` -> SAM3), or forced explicitly via `sam_backend=` /
`--sam-backend {auto,sam2,sam3}`.

SAM3 cannot be loaded through `sam2.build_sam2` (different architecture: a
DETR-style detector + a SAM2-style tracker sharing one ViT encoder, 848M
params). The SAM3 backend instead drives SAM3's interactive point-prompt
predictor through the official `sam3` package, reimplementing the automatic
point-grid loop (multimask output, IoU + stability filtering, box NMS) with
the same formulas as the SAM2 path.

### Requirements

> **Note**: SAM3 needs a **separate environment** from SAM2. SAM3 requires
> Python >= 3.12, PyTorch >= 2.7 and CUDA >= 12.6, while the SAM2 stack is
> typically older.

```bash
# 1. New conda environment
conda create -n sam3 python=3.12
conda activate sam3

# 2. PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. The sam3 package (no PyPI release; install from the official repo)
git clone https://github.com/facebookresearch/sam3.git
pip install -e ./sam3

# 4. sam-mosaic + its geo dependencies in the same env
pip install -e .

# 5. Checkpoint - the official weights are gated on HuggingFace:
#    request access at https://huggingface.co/facebook/sam3, then either
#      hf auth login          # ...and pass checkpoint="hf" / use sam3.pt path, or
#      download sam3.pt manually into ./checkpoints/
```

### Usage

```python
from sam_mosaic import segment_with_params

result = segment_with_params(
    input_path="image.tif",
    output_dir="output/",
    checkpoint="./checkpoints/sam3.pt",   # backend auto-detected -> SAM3
    tile_size=500,
    max_passes=50,
    target_coverage=95.0,
    points_per_side=48,
    threshold_step=0.02,
)

# Or force the backend explicitly:
result = segment_with_params(..., checkpoint="hf", sam_backend="sam3")
```

CLI: `sam-mosaic input.tif output/ --checkpoint ./checkpoints/sam3.pt --sam-backend auto`

Validation: `test/code/test_sam3_backend.py` checks the integration end to
end on one tile (part 1 runs anywhere; parts 2-3 need the `sam3` env and
weights). Results go to `test/results/`.

### SAM3 backend notes

- Same multi-pass pipeline, point strategies (kmeans / dense_grid), black
  masking and tile merging as the SAM2 backend.
- **Threshold calibration**: SAM3's IoU head scores lower than SAM2's (few
  masks exceed 0.90 on aerial imagery). With the SAM2 default start of 0.93
  the first pass barely admits anything; `iou_start=0.80` /
  `stability_start=0.80` (with the usual `0.60` end) works well.
- **Nodata handling**: SAM3 refuses to segment black nodata / mosaic-gap
  areas, so reported coverage can be lower than SAM2's even though the real
  content is fully segmented (SAM2 tends to cut nodata into fake segments).
  Measured on the test tile: SAM3 63.6% overall but ~93% of actual content,
  24 passes, 39 segments, 15.7s -- vs SAM2 97.0% incl. nodata, 14 passes,
  25 segments, 31.4s (~2x slower).
- The tracker runs under bf16 autocast internally (official behavior), so
  the SAM3 path gets tensor-core acceleration by default.
- Not supported on the SAM3 backend (yet): `crop_n_layers > 0` (a SAM2-AMG
  sub-crop feature; it raises a clear error).
- Encoder features are cached per image content, so multi-pass iterations
  that black-mask nothing skip the expensive ViT forward automatically.
- **Long-run VRAM stability** (matters for 1000+-tile jobs): the periodic
  full model reload the pipeline uses against fragmentation is disabled for
  SAM3 (its 3.4 GB-parameter model makes each reload the fragmentation
  source; the observed failure mode is VRAM saturation -> WDDM system-memory
  fallback -> ~10x slowdown). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  is recommended, and the backend also empties the CUDA cache when the
  allocator reservation passes 85% of VRAM.
- **Windows setup extras** (already applied on this machine's `sam3` env):
  `setuptools<81` (sam3 still imports `pkg_resources`, removed in
  setuptools 81), plus `einops`, `pycocotools`, `psutil` (imported
  unconditionally by sam3 despite being listed as optional extras). The
  `triton` dependency does not exist on Windows; the local sam3 clone at
  `Desktop/sam3` is patched to import `edt_triton` lazily (used only by
  training-time correction sampling, not by this pipeline).

---

## Using MobileSAM

sam-mosaic also supports **MobileSAM** checkpoints (`mobile_sam.pt`, ~40 MB):
the same prompt encoder + mask decoder as SAM, with the ViT image encoder
swapped for a ~5M-parameter TinyViT. On this workload it behaves like a
faster SAM2 (same threshold schedule `0.93 -> 0.60` applies) at roughly half
the wall time and 40% less VRAM, with similar segment character.

Backend selection is automatic (`mobile` in the checkpoint filename) or
force with `sam_backend="mobilesam"` / `--sam-backend mobilesam`.

```bash
# 1. The mobile_sam package (original-SAM API; no dependency conflicts)
git clone https://github.com/ChaoningZhang/MobileSAM.git
pip install -e ./MobileSAM
pip install timm          # TinyViT dependency

# 2. Checkpoint (~40 MB)
cp MobileSAM/weights/mobile_sam.pt ./checkpoints/
```

```python
result = segment_with_params(
    input_path="image.tif", output_dir="output/",
    checkpoint="./checkpoints/mobile_sam.pt",   # auto-detected -> MobileSAM
    tile_size=500, max_passes=50, target_coverage=95.0,
    points_per_side=48, threshold_step=0.02,
)
```

Measured on the 4 test crops (same params as the SAM2/SAM3 comparison, see
`test/results/compare_sam2_vs_sam3/COMPARISON.md`): avg 67.8 segments /
98.2% coverage / 84 s per 1300px crop vs SAM2-large 64.8 / 98.7% / 138 s and
SAM3 89.0 / 96.7% / 92 s. Peak VRAM ~1.9 GB (vs 3.7 GB SAM2 / 7.2 GB SAM3)
-- the lightest backend, suitable for 8 GB GPUs.

---

## Usage

### Command Line (CLI)

```bash
# Basic usage (uses optimized default parameters)
sam-mosaic input.tif output/

# With custom checkpoint path
sam-mosaic input.tif output/ --checkpoint /path/to/sam2.1_hiera_large.pt

# With custom configuration file
sam-mosaic input.tif output/ --config my_config.yaml

# Customize polygon simplification (default: 1.0)
sam-mosaic input.tif output/ --simplify-tolerance 2.0

# Also generate GeoPackage
sam-mosaic input.tif output/ --geopackage
```

### Python API

```python
from sam_mosaic import segment_image

# Basic usage
result = segment_image(
    input_path="data/ortofoto.tif",
    output_dir="results/",
)

print(f"Segments found: {result.n_segments}")
print(f"Coverage: {result.coverage:.1f}%")
print(f"Processing time: {result.processing_time:.1f}s")
print(f"Labels saved to: {result.labels_path}")
print(f"Shapefile saved to: {result.shapefile_path}")
```

---

## Output Files

After running, the output directory will contain:

| File | Description |
|------|-------------|
| `labels.tif` | GeoTIFF raster where each pixel has a segment ID (1, 2, 3, ...). Background = 0. Preserves CRS and georeferencing from input. |
| `segments.shp` | Shapefile with vectorized polygons. Includes attributes: `label_id`, `area_m2`, `perimeter_m`. |
| `segments.gpkg` | GeoPackage (optional, use `--geopackage` flag) |
| `stats.json` | Detailed statistics: segments count, coverage, processing time, per-tile stats. |

---

## Customization

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--simplify-tolerance` | 1.0 | Polygon simplification in map units. Higher = simpler polygons. Use 0 for no simplification. |
| `--tile-size` | 1000 | Tile size in pixels. Reduce if running out of GPU memory. |
| `--padding` | 50 | Extra context pixels around each tile. Helps with boundary merging. |
| `--min-area` | 100 | Remove segments smaller than this (in pixels). |
| `--target-coverage` | 99.0 | Stop segmentation when this coverage % is reached. |
| `--point-strategy` | dense_grid | Point selection: `dense_grid` (default, robust) or `kmeans` (alternative). |
| `--erosion` | 0 | Erosion iterations for point placement. Increase for denser scenes. |
| `--iou-start` | 0.93 | Initial IoU threshold (restrictive). |
| `--iou-end` | 0.60 | Final IoU threshold (permissive). |
| `--stability-start` | 0.93 | Initial stability score threshold. |
| `--stability-end` | 0.60 | Final stability score threshold. |

### Point Strategies

Remote SAMsing supports two point selection strategies for multi-pass segmentation:

**Dense Grid (default)**: Uses a uniform grid filtered by already-segmented areas. Robust across scene types, from urban imagery with small objects to agricultural fields with large parcels.

**K-means**: Clusters points in residual (unsegmented) areas. An alternative when targeting large, homogeneous regions.

```bash
# Default: Dense Grid (works well for most scenes)
sam-mosaic input.tif output/ --checkpoint sam2.pt

# Dense Grid with higher point density for very small objects
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --points-per-side 96

# K-means for large homogeneous regions
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --point-strategy kmeans --erosion 5
```

### Threshold Parameters

SAM2 uses two thresholds to filter predicted masks:

- **IoU threshold** (`--iou-start`, `--iou-end`): Filters masks by predicted IoU score
- **Stability threshold** (`--stability-start`, `--stability-end`): Filters masks by stability score

Both thresholds decrease from start to end across passes, allowing more permissive masks as coverage increases. Strict thresholds capture salient objects first; relaxation occurs only when progress stagnates.

```bash
# More restrictive (fewer but higher quality segments)
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --iou-start 0.95 --stability-start 0.95

# More permissive (higher coverage, may include lower quality segments)
sam-mosaic input.tif output/ --checkpoint sam2.pt \
    --iou-end 0.50 --stability-end 0.50
```

### Polygon Simplification Examples

```bash
# No simplification (keeps all vertices - larger file)
sam-mosaic input.tif output/ --simplify-tolerance 0

# Light simplification (default)
sam-mosaic input.tif output/ --simplify-tolerance 1.0

# More simplification (smaller file, less detail)
sam-mosaic input.tif output/ --simplify-tolerance 3.0

# Heavy simplification
sam-mosaic input.tif output/ --simplify-tolerance 5.0
```

### Using a Configuration File

Create a `config.yaml` file:

```yaml
tile:
  size: 1000           # Tile size in pixels
  padding: 50          # Context padding

threshold:
  iou_start: 0.93      # Initial IoU threshold (restrictive)
  iou_end: 0.60        # Final IoU threshold (permissive)
  step: 0.01           # Threshold decrease per pass

segmentation:
  point_strategy: dense_grid
  points_per_side: 64  # Grid density (64x64 = 4096 points)
  target_coverage: 99.0
  use_black_mask: true
  use_adaptive_threshold: true

merge:
  merge_strategy: best_match
  min_contact_pixels: 20
  min_mask_area: 100
  merge_enclosed_max_area: 500

output:
  simplify_tolerance: 1.0
  save_shapefile: true
  save_geopackage: false

sam_checkpoint: checkpoints/sam2.1_hiera_large.pt
```

Then run:

```bash
sam-mosaic input.tif output/ --config config.yaml
```

---

## How It Works

Remote SAMsing processes each tile in multiple passes with decreasing quality thresholds:

1. **Pass 1**: Uniform grid (64x64 = 4096 points) with strict IoU/stability thresholds (0.93). Captures high-confidence segments first (~60--70% coverage).

2. **Pass 2+**: Points placed only in residual (unsegmented) areas via the Dense Grid strategy. A black mask is applied to already-segmented pixels, directing SAM toward remaining gaps. Thresholds decrease gradually (0.93 -> 0.92 -> ... -> 0.60), but only when coverage progress stagnates.

3. **Stop condition**: Coverage >= 99% or minimum threshold reached.

After all tiles are processed, segments touching at tile boundaries are reconciled through a **best-match merge**: each label pair at a discontinuity line is scored by contact length, and the best match for each label is accepted. Union-Find resolves transitive chains, and a single LUT lookup relabels the full image in O(n) time. This merge is parameter-free and produces a spatially consistent label map for arbitrarily large images with constant GPU memory.

---

## Requirements

- **Python**: 3.12
- **GPU**: NVIDIA GPU with CUDA (recommended). Works on CPU but much slower.
- **RAM**: 16GB+ recommended for large images
- **VRAM**: 8GB+ recommended (tested with 24GB GPU on images up to 1.94 billion pixels)
- **Disk**: ~1GB for SAM2 checkpoint + space for outputs

### Dependencies

- PyTorch 2.9
- SAM2 1.1.0+ (from PyPI)
- CUDA 12.8
- rasterio
- numpy, scipy, scikit-learn
- shapely, fiona
- tqdm, pyyaml

### Tested Configuration

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i9-14900K |
| RAM | 64 GB |
| GPU | NVIDIA RTX 4090 (24 GB VRAM) |
| Python | 3.12 |
| PyTorch | 2.9 |
| SAM2 | 1.1.0 |
| CUDA | 12.8 |

---

## Ablation Studies

The package supports easy ablation experiments:

```python
from sam_mosaic import segment_with_params

# Single-pass only (no multi-pass)
result = segment_with_params(
    "input.tif", "output/single_pass/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    max_passes=1,
    iou_start=0.86,
    use_adaptive_threshold=False
)

# Without black mask
result = segment_with_params(
    "input.tif", "output/no_blackmask/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    use_black_mask=False
)

# Without padding (to show merge artifacts)
result = segment_with_params(
    "input.tif", "output/no_padding/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    padding=0
)

# K-means point strategy (alternative to default Dense Grid)
result = segment_with_params(
    "input.tif", "output/kmeans/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    point_strategy="kmeans",
    erosion_iterations=5
)

# Custom stability thresholds
result = segment_with_params(
    "input.tif", "output/custom_thresholds/",
    checkpoint="checkpoints/sam2.1_hiera_large.pt",
    stability_start=0.90,
    stability_end=0.50
)
```

---

## Troubleshooting

### CUDA out of memory

Reduce tile size:

```bash
sam-mosaic input.tif output/ --tile-size 512
```

### Too many small segments

Increase minimum area filter:

```bash
sam-mosaic input.tif output/ --min-area 200
```

### Shapefile too large / too many vertices

Increase simplification:

```bash
sam-mosaic input.tif output/ --simplify-tolerance 3.0
```

### Segments not merging at tile boundaries

Increase padding:

```bash
sam-mosaic input.tif output/ --padding 100
```

### OpenMP library conflict (Windows)

If you see an error about `libomp.dll` and `libiomp5md.dll` conflict:

```powershell
# PowerShell - set before running
$env:KMP_DUPLICATE_LIB_OK='TRUE'
sam-mosaic input.tif output/
```

```bash
# Bash/CMD
set KMP_DUPLICATE_LIB_OK=TRUE
sam-mosaic input.tif output/
```

### CLI hangs at "Loading SAM2 model..."

Enable debug mode to identify where it hangs:

```powershell
$env:SAM_MOSAIC_DEBUG='1'
$env:KMP_DUPLICATE_LIB_OK='TRUE'
sam-mosaic input.tif output/ --checkpoint path/to/sam2.pt
```

This will print detailed loading steps. The last `[DEBUG]` message before hanging indicates the problem.

---

## Citation

If you use this software in your research, please cite:

```bibtex
@article{carvalho2026remotesamsing,
  title = {Remote {SAMsing}: From Segment Anything to Segment Everything},
  author = {de Carvalho, Osmar Luiz Ferreira and de Carvalho J{\'u}nior, Osmar Ab{\'i}lio and de Albuquerque, Anesmar Olino and Guerreiro e Silva, Daniel},
  journal = {arXiv preprint arXiv:2605.00256},
  year = {2026},
  url = {https://arxiv.org/abs/2605.00256}
}
```

---

## License

MIT License

---

## Acknowledgments

- [SAM2](https://github.com/facebookresearch/sam2) by Meta AI - Segment Anything Model 2
- [sam2 PyPI package](https://pypi.org/project/sam2/) - SAM2 distribution used in this project
- [rasterio](https://rasterio.readthedocs.io/) for GeoTIFF handling
- [shapely](https://shapely.readthedocs.io/) and [fiona](https://fiona.readthedocs.io/) for vector operations
