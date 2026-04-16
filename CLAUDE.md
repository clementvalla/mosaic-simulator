# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A mosaic simulator that converts images into realistic mosaic renderings. It simulates three historical Roman mosaic techniques by placing individual tesserae (tile pieces) with realistic size variation, drift, rotation, and grout.

## Running

**CLI (full render):**
```bash
python3 simulate_mosaic_v2.py --image inputs/photo.jpg --mode drift --tesserae-across 60
python3 simulate_mosaic_v2.py --image inputs/photo.jpg --mode contour --tesserae-across 40 --fill-style radial
python3 simulate_mosaic_v2.py --image inputs/photo.jpg --mode flow --tesserae-across 80 --flow-direction across
python3 simulate_mosaic_v2.py --image inputs/photo.jpg --mode drift --tesserae-across 100 --render-percentage 50
```

Key sizing args: `--tesserae-across N` (number of tesserae across, default 50) and `--render-percentage P` (10-100, percentage of max tile resolution, default 100). Output pixel size is automatically derived from tile template resolution × render percentage.

**GUI (Dear PyGui):**
```bash
python3 mosaic_gui.py
```

**Tile extraction (from a photo of a real mosaic):**
```bash
python3 extract_tiles.py --image images/crop.jpg
python3 extract_tiles.py --image images/crop.jpg --preview-only  # inspect before full extraction
```

Output goes to `output/` (auto-named with mode and timestamp). Reports are saved as `_report.txt` alongside each output image.

## Dependencies

Python with numpy, opencv-python (cv2), Pillow, scipy, dearpygui.

## Architecture

Entry points: `simulate_mosaic_v2.py` (CLI), `mosaic_gui.py` (Dear PyGui GUI). Both call into the `mosaic/` package. The GUI is frontend-only — it imports the same `build_mosaic_*` functions as the CLI.

### `mosaic/` package

- **tiles.py** — `get_max_tile_size()` scans tile PNGs to find the native resolution ceiling. `load_tile_templates()` loads tile template PNGs from `tiles/raw/`, normalizes to (luminance, alpha) pairs. Luminance is normalized so mean over opaque region ≈ 1.0, then multiplied by target color at compositing time.
- **compositing.py** — Core rendering: `composite_tessera()` blends a luminance×color tile with alpha onto the canvas. `composite_tessera_preview()` draws fast rotated rectangles (no texture) for preview mode. `sample_color_nearest()` maps output position back to input pixel color.
- **mode_drift.py** — *Opus Tessellatum*: Row-by-row placement with height map for vertical packing. Each tessera maps to one input pixel. Includes edge-fill pass to extend short rows.
- **mode_contour.py** — *Opus Vermiculatum*: Detects edges via Canny, chains tesserae along contour paths with echo rows on both sides. Background filled with drift rows or flow-field streamlines (radial/concentric).
- **mode_flow.py** — *Opus Musivum*: Structure-tensor-driven placement across entire surface. Multi-wave seeding (surface seeds → fringe seeds → gap cleanup). Supports "along" (parallel to edges) and "across" (perpendicular) flow directions.
- **flow_utils.py** — Shared by contour and flow modes: distance-transform-based flow field computation, fringe seed generation, streamline tracing, and `fill_background_flow()`.
- **report.py** — Generates text reports with dimensions and real-world size estimates.

### Key rendering patterns

All three modes share the same tessera placement loop: jitter size → check occupancy (25% threshold) → sample color from input → rotate template → composite onto canvas. Contour and flow modes use a boolean `occupied_map` to prevent overlap; drift mode uses a `height_map` array for vertical stacking.

Preview mode (`preview=True`) skips tile texture loading entirely and draws flat colored rectangles — used by the GUI for fast parameter tuning.

### `mosaic_gui.py` — Dear PyGui interface

Native desktop window (not browser-based). Full-bleed letterboxed canvas with a floating lil-gui-style control panel. Renders mosaic in a background thread to keep the UI responsive. Textures are managed via DPG dynamic textures with counter-based tags to avoid alias conflicts on resize. Generation runs in a thread; the same `generate_mosaic` logic as the CLI.

### `extract_tiles.py`

Standalone tool that segments tesserae from a photograph of a real mosaic using adaptive thresholding + connected components. Outputs raw RGBA tiles to `tiles/raw/` (which the simulator then uses as templates) plus augmented variants (flips, rotations, brightness jitter) to `tiles/augmented/`.

## Key directories

- `tiles/raw/` — Tile template PNGs (input to the simulator, output of extract_tiles.py)
- `inputs/` — Source images to convert into mosaics
- `output/` — Generated mosaic images and reports
- `images/` — Source photos of real mosaics (input to extract_tiles.py)
