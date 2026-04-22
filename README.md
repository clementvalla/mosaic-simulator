# Mosaic Simulator

Converts images into realistic mosaic renderings by simulating three historical Roman mosaic techniques. Individual tesserae (tile pieces) are placed with realistic size variation, drift, rotation, and grout spacing.

## Modes

- **Drift** (*Opus Tessellatum*) -- Row-by-row grid placement with cumulative positional drift, like a mason laying tiles left to right.
- **Contour** (*Opus Vermiculatum*) -- Tesserae follow detected edges like worms, with a tunable number of echo rows on each side of every contour (`--inner-rows` / `--outer-rows`). Background fill can differ inside vs. outside detected shapes (`--inner-fill-style` / `--outer-fill-style`, choose drift / radial / concentric).
- **Flow** (*Opus Musivum*) -- Structure-tensor-driven placement across the entire surface. Tesserae align with (or across) image gradients.

## Quick Start

### CLI

```bash
# Basic drift mosaic, 60 tesserae wide
python3 simulate_mosaic.py --image inputs/photo.jpg --mode drift --tesserae-across 60

# Contour mode with radial fill, limited to 8 colors
python3 simulate_mosaic.py --image inputs/photo.jpg --mode contour --tesserae-across 40 --fill-style radial --num-colors 8

# Flow mode, perpendicular to edges, at 50% render resolution
python3 simulate_mosaic.py --image inputs/photo.jpg --mode flow --tesserae-across 80 --flow-direction across --render-percentage 50
```

### GUI

```bash
python3 mosaic_gui.py
```

Native desktop window with a floating control panel. Preview renders fast flat-colored rectangles; Render uses actual tile textures. Save button writes to `output/`. Scroll to zoom, middle-click to pan.

### Tile Extraction

Extract tile templates from a photograph of a real mosaic:

```bash
python3 extract_tiles.py --image images/crop.jpg
python3 extract_tiles.py --image images/crop.jpg --preview-only  # inspect before full extraction
```

Outputs RGBA tile PNGs to `tiles/raw/` for use as templates.

## Key Parameters

### Sizing

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Tesserae across | `--tesserae-across` | 50 | Number of tesserae across the mosaic width. Height auto-calculated from aspect ratio. |
| Render percentage | `--render-percentage` | 100 | % of max tile resolution (10-100). Output pixel size = tile native resolution x this %. |

### Color

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Palette colors | `--num-colors` | 0 (unlimited) | Limit to N colors (1-128). Uses hue-diversity-first quantization. |
| Color influence | `--color-influence` | 1.0 | Blend toward white (0 = all white, 1 = full image color). |
| Color variation | `--color-variation` | 15.0 | Per-tessera brightness jitter for natural stone look. |

Color settings only affect tessera coloring -- placement, edge detection, and flow fields are always computed from the original image.

### Tessera

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Grout width | `--grout-width` | -1 | Pixels between tesserae (negative = overlap). |
| Size jitter | `--size-jitter` | 0.15 | Random size variation per tessera (0.15 = +/-15%). |
| Rotation jitter | `--rotation-jitter` | 3.0 | Max random rotation in degrees. |

### Mode-Specific

| Parameter | CLI Flag | Applies To | Description |
|-----------|----------|------------|-------------|
| Fill style | `--fill-style` | contour | Background fill: `drift`, `radial`, or `concentric` |
| Flow direction | `--flow-direction` | flow | `along` (parallel to edges) or `across` (perpendicular) |
| Edge threshold | `--edge-threshold` | contour | Edge detection sensitivity (0-1) |
| Drift correction | `--drift-correction` | drift | Reset drift every N pixels (0 = never) |

## Installation

Requires Python 3.9+.

```bash
pip install numpy opencv-python Pillow scipy dearpygui
```

`dearpygui` is only needed for the GUI. The CLI and tile extraction work without it.

## Architecture

Each mode runs in two phases: a **plan** phase that produces a `list[Placement]` (one record per tessera with position, size, rotation, and color), and a **render** phase that consumes the list. Raster output is one consumer; the same list is the foundation for future vector (SVG), tabular (CSV), and robotic-arm / g-code exporters. See [`mosaic/placements.py`](mosaic/placements.py) and the Placement data model section of `CLAUDE.md` for details.

## Project Structure

```
simulate_mosaic.py    # CLI entry point
mosaic_gui.py            # GUI entry point (Dear PyGui)
extract_tiles.py         # Tile extraction from mosaic photos
mosaic/                  # Core package
    tiles.py             # Tile template loading and normalization
    compositing.py       # Color quantization, tessera compositing
    placements.py        # Placement dataclass + shared render loop
    mode_drift.py        # Opus Tessellatum placement
    mode_contour.py      # Opus Vermiculatum placement
    mode_flow.py         # Opus Musivum placement
    flow_utils.py        # Shared flow-field utilities
    report.py            # Text report generation
tiles/raw/               # Tile template PNGs
inputs/                  # Source images
output/                  # Generated mosaics and reports
images/                  # Source photos of real mosaics
```
