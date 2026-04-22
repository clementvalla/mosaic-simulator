"""Mode A: Drift placement (opus tessellatum).

Row-by-row placement with cumulative drift, mimicking hand-laid Roman
tessellatum construction.
"""

import time

import numpy as np

from .compositing import sample_color_nearest, jitter_size
from .placements import Placement, render_placements


def _plan_drift(input_image, color_image, tessera_size, grout_width,
                size_jitter, rotation_jitter, rng):
    """Run the drift placement logic and return (placements, canvas_shape).

    No canvas is painted here — this function only decides where each
    tessera goes.
    """
    in_h, in_w = input_image.shape[:2]
    nominal_cell = tessera_size + grout_width

    # Allocate canvas shape with headroom for drift
    headroom = 1.15
    est_w = int(in_w * nominal_cell * headroom) + tessera_size
    est_h = int(in_h * nominal_cell * headroom) + tessera_size

    # Pre-compute rotation angles for jitter
    if rotation_jitter > 0:
        rot_angles = np.linspace(-rotation_jitter, rotation_jitter, 11)
    else:
        rot_angles = [0.0]

    grout_gap = max(grout_width, -5)
    height_map = np.zeros(est_w, dtype=np.float64)
    max_x_extent = 0
    row_end_x = []  # (cursor_x_end, edge_color) per row
    placements = []
    t_start = time.time()

    for row in range(in_h):
        if row % 20 == 0 and row > 0:
            elapsed = time.time() - t_start
            pct = row / in_h * 100
            print(f"  Row {row}/{in_h} ({pct:.0f}%) - {elapsed:.1f}s elapsed")

        row_height_snap = height_map.copy()
        cursor_x = 0

        for col in range(in_w):
            jw, jh = jitter_size(tessera_size, size_jitter, rng)

            x_start = max(cursor_x, 0)
            x_end = min(cursor_x + jw, est_w)
            if x_end > x_start:
                y_pos = int(row_height_snap[x_start:x_end].max()) + grout_gap
            else:
                y_pos = 0
            y_pos = max(y_pos, 0)

            color = sample_color_nearest(color_image, row, col)
            angle = rng.choice(rot_angles) if rotation_jitter > 0 else 0.0

            placements.append(Placement(
                x=cursor_x, y=y_pos, w=jw, h=jh,
                angle=float(angle), color=color,
            ))

            if x_end > x_start:
                height_map[x_start:x_end] = np.maximum(
                    height_map[x_start:x_end], y_pos + jh)

            cursor_x += jw + grout_gap

        edge_color = sample_color_nearest(color_image, row, in_w - 1)
        row_end_x.append((cursor_x, edge_color))
        max_x_extent = max(max_x_extent, cursor_x)

    # Edge-fill pass: extend short rows to max_x_extent
    fill_count = 0
    for (rx, edge_col) in row_end_x:
        while rx < max_x_extent:
            jw, _ = jitter_size(tessera_size, size_jitter, rng)

            x_start = max(rx, 0)
            x_end = min(rx + jw, est_w)
            if x_end > x_start:
                y_pos = int(height_map[x_start:x_end].max()) + grout_gap
            else:
                y_pos = 0
            y_pos = max(y_pos, 0)

            _, jh = jitter_size(tessera_size, size_jitter, rng)
            angle = rng.choice(rot_angles) if rotation_jitter > 0 else 0.0

            placements.append(Placement(
                x=rx, y=y_pos, w=jw, h=jh,
                angle=float(angle), color=edge_col,
            ))

            if x_end > x_start:
                height_map[x_start:x_end] = np.maximum(
                    height_map[x_start:x_end], y_pos + jh)

            rx += jw + grout_gap
            fill_count += 1

    if fill_count > 0:
        print(f"  Edge fill: {fill_count:,} extra tesserae")

    # Final canvas size = fit-to-extent (will be cropped to est_w/est_h)
    final_w = min(max_x_extent + tessera_size, est_w)
    final_h = min(int(height_map[:final_w].max()) + tessera_size, est_h)

    return placements, (est_h, est_w), (final_h, final_w)


def build_mosaic_drift(input_image, templates, tessera_size, grout_width,
                       grout_color, color_variation, size_jitter,
                       rotation_jitter, drift_correction_interval, rng,
                       preview=False, color_image=None):
    """Build mosaic with row-by-row placement and cumulative drift.

    Returns (canvas_uint8, count, placements).
    """
    if color_image is None:
        color_image = input_image
    in_h, in_w = input_image.shape[:2]

    t_start = time.time()

    placements, (est_h, est_w), (final_h, final_w) = _plan_drift(
        input_image, color_image, tessera_size, grout_width,
        size_jitter, rotation_jitter, rng,
    )

    canvas = np.full((est_h, est_w, 3), grout_color, dtype=np.float32)
    render_placements(canvas, placements, templates, color_variation, rng,
                      preview=preview)

    canvas = canvas[:final_h, :final_w]

    elapsed = time.time() - t_start
    total_placed = len(placements)
    print(f"Rendering complete: {total_placed:,} tesserae in {elapsed:.1f}s")
    print(f"Input pixels: {in_h * in_w:,} | Tesserae placed: {total_placed:,}")
    print(f"Output dimensions: {final_w} x {final_h} px")

    plan_info = {
        "canvas_shape": (est_h, est_w),
        "crop": "fixed",
        "fixed_shape": (final_h, final_w),
    }
    return (np.clip(canvas, 0, 255).astype(np.uint8),
            total_placed, placements, plan_info)
