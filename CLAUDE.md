# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A mosaic simulator that converts images into realistic mosaic renderings. It simulates three historical Roman mosaic techniques by placing individual tesserae (tile pieces) with realistic size variation, drift, rotation, and grout.

## Running

**CLI (full render):**
```bash
python3 simulate_mosaic.py --image inputs/photo.jpg --mode drift --tesserae-across 60
python3 simulate_mosaic.py --image inputs/photo.jpg --mode contour --tesserae-across 40 --fill-style radial
python3 simulate_mosaic.py --image inputs/photo.jpg --mode contour --tesserae-across 40 --inner-rows 4 --outer-rows 1 --inner-fill-style concentric --outer-fill-style drift
python3 simulate_mosaic.py --image inputs/photo.jpg --mode flow --tesserae-across 80 --flow-direction across
python3 simulate_mosaic.py --image inputs/photo.jpg --mode drift --tesserae-across 100 --render-percentage 50
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

Entry points: `simulate_mosaic.py` (CLI), `mosaic_gui.py` (Dear PyGui GUI). Both call into the `mosaic/` package. The GUI is frontend-only — it imports the same `build_mosaic_*` functions as the CLI.

### `mosaic/` package

- **tiles.py** — `get_max_tile_size()` scans tile PNGs to find the native resolution ceiling. `load_tile_templates()` loads tile template PNGs from `tiles/raw/`, normalizes to (luminance, alpha) pairs. Luminance is normalized so mean over opaque region ≈ 1.0, then multiplied by target color at compositing time.
- **compositing.py** — Core rendering: `composite_tessera()` blends a luminance×color tile with alpha onto the canvas. `composite_tessera_preview()` draws fast rotated rectangles (no texture) for preview mode. `sample_color_nearest()` maps output position back to input pixel color.
- **mode_drift.py** — *Opus Tessellatum*: Row-by-row placement with height map for vertical packing. Each tessera maps to one input pixel. Includes edge-fill pass to extend short rows.
- **mode_contour.py** — *Opus Vermiculatum*: Detects edges via Canny, chains tesserae along contour paths with user-controllable echo rows on each side (`inner_rows`/`outer_rows`). Background fill supports per-region styles: `inner_fill_style` (inside enclosed shapes) vs. `outer_fill_style` (outside). Shape mask is built by morph-closing the Canny output and filling external contours; when both styles match, falls back to a single-pass fill.
- **mode_flow.py** — *Opus Musivum*: Structure-tensor-driven placement across entire surface. Multi-wave seeding (surface seeds → fringe seeds → gap cleanup). Supports "along" (parallel to edges) and "across" (perpendicular) flow directions.
- **flow_utils.py** — Shared by contour and flow modes: distance-transform-based flow field computation, fringe seed generation, streamline tracing, and `plan_background_flow()` (returns a placement list).
- **placements.py** — `Placement` dataclass (x, y, w, h, angle, color, template_idx) + `render_placements()` (the single compositing loop) + `rasterize_placements()` (allocate canvas, render, crop — used by the GUI for cached preview→render replays).
- **report.py** — Generates text reports with tessera size and output dimensions in px, mm, cm, and inches.

### Placement data model — the core abstraction

The simulator is organized around a single canonical data structure: a `list[Placement]` describing every tessera's final position, size, rotation, and base color. Raster output is just one consumer of that list; the same list is intended to feed vector (SVG), tabular (CSV), and robotic-arm / g-code exporters.

All three modes follow a two-phase pattern:

1. **Plan** — the mode-specific planner (`build_mosaic_drift` / `_contour` / `_flow`) decides where every tessera goes, updating the boolean `occupied_map` (contour, flow) or `height_map` (drift) as it goes, and appends `Placement` records to a list. No pixels are painted.
2. **Render** — `render_placements()` in `mosaic/placements.py` iterates the list once and composites each record onto a fresh canvas. `rasterize_placements()` wraps allocation + render + crop for the raster path.

`build_mosaic_*` returns `(canvas_uint8, count, placements, plan_info)`; `plan_info` carries the canvas shape and crop strategy needed to replay the render step from the cached placement list.

Preview mode (`preview=True`) runs the same plan phase, then the render phase draws flat colored rectangles via `composite_tessera_preview` instead of blending tile textures — so the GUI can cache preview placements and reuse them on the next Render click without re-running the (expensive) plan phase.

When adding a new output format (SVG, CSV, g-code, etc.), plug in at the render phase: consume the `placements` list returned by `build_mosaic_*` and translate each `Placement` to the target format. Do not add output-specific logic inside the mode planners.

### `mosaic_gui.py` — Dear PyGui interface

Native desktop window (not browser-based). Full-bleed letterboxed canvas with a floating lil-gui-style control panel. Renders mosaic in a background thread to keep the UI responsive. Textures are managed via DPG dynamic textures with counter-based tags to avoid alias conflicts on resize. Generation runs in a thread; the same `generate_mosaic` logic as the CLI.

### `extract_tiles.py`

Standalone tool that segments tesserae from a photograph of a real mosaic using adaptive thresholding + connected components. Outputs raw RGBA tiles to `tiles/raw/` (which the simulator then uses as templates) plus augmented variants (flips, rotations, brightness jitter) to `tiles/augmented/`.

## Key directories

- `tiles/raw/` — Tile template PNGs (input to the simulator, output of extract_tiles.py)
- `inputs/` — Source images to convert into mosaics
- `output/` — Generated mosaic images and reports
- `images/` — Source photos of real mosaics (input to extract_tiles.py)
