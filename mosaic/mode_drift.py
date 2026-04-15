"""Mode A: Drift placement (opus tessellatum).

Row-by-row placement with cumulative drift, mimicking hand-laid Roman
tessellatum construction.
"""

import time

import cv2
import numpy as np

from .compositing import composite_tessera, composite_tessera_preview, resize_template, sample_color_nearest


def build_mosaic_drift(input_image, templates, tessera_size, grout_width,
                       grout_color, color_variation, size_jitter,
                       rotation_jitter, drift_correction_interval, rng,
                       preview=False):
    """Build mosaic with row-by-row placement and cumulative drift.

    Each tessera maps to exactly one input pixel (nearest-neighbor, no
    interpolation). Tessera sizes are jittered, causing positional drift.
    Every drift_correction_interval pixels, the cursor nudges back toward
    the nominal grid position.
    """
    in_h, in_w = input_image.shape[:2]
    n_templates = len(templates) if templates else 0

    # Scale: nominal canvas pixels per input pixel
    nominal_cell = tessera_size + grout_width

    # Allocate canvas with headroom for drift
    headroom = 1.15
    est_w = int(in_w * nominal_cell * headroom) + tessera_size
    est_h = int(in_h * nominal_cell * headroom) + tessera_size
    canvas = np.full((est_h, est_w, 3), grout_color, dtype=np.float32)

    # Pre-compute rotation angles for jitter
    if rotation_jitter > 0:
        rot_angles = np.linspace(-rotation_jitter, rotation_jitter, 11)
    else:
        rot_angles = [0.0]

    t_start = time.time()
    max_x_extent = 0
    total_placed = 0
    total_pixels = in_h * in_w

    # Height map: tracks the bottom Y of the lowest tessera at each X column
    height_map = np.zeros(est_w, dtype=np.float64)

    # Effective grout for spacing (allow negative = overlap)
    grout_gap = max(grout_width, -tessera_size // 3)  # clamp to prevent extreme overlap

    # Track row end positions for edge fill
    row_end_x = []  # (cursor_x_end, edge_color) per row

    for row in range(in_h):
        if row % 20 == 0 and row > 0:
            elapsed = time.time() - t_start
            pct = row / in_h * 100
            print(f"  Row {row}/{in_h} ({pct:.0f}%) - {elapsed:.1f}s elapsed")

        # Snapshot height map so all tesserae in this row use the same Y reference
        row_height_snap = height_map.copy()
        cursor_x = 0

        for col in range(in_w):
            # Jitter tessera size
            jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))
            jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))

            # Find Y from height map snapshot: highest point in the X footprint
            x_start = max(cursor_x, 0)
            x_end = min(cursor_x + jw, est_w)
            if x_end > x_start:
                y_pos = int(row_height_snap[x_start:x_end].max()) + grout_gap
            else:
                y_pos = 0
            y_pos = max(y_pos, 0)

            # Color from exact input pixel (no interpolation)
            color = sample_color_nearest(input_image, row, col)

            # Rotation angle
            angle = rng.choice(rot_angles) if rotation_jitter > 0 else 0.0

            if preview:
                composite_tessera_preview(
                    canvas, color, cursor_x + jw // 2, y_pos + jh // 2,
                    jw, jh, angle, color_variation, rng)
            else:
                t_idx = rng.integers(n_templates)
                lum_base, alpha_base = templates[t_idx]
                lum, alpha = resize_template(lum_base, alpha_base, jw, jh)
                if abs(angle) > 0.1:
                    center = (jw / 2, jh / 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    lum = cv2.warpAffine(lum, M, (jw, jh))
                    alpha = cv2.warpAffine(alpha, M, (jw, jh))
                composite_tessera(canvas, lum, alpha, color, cursor_x, y_pos,
                                  color_variation, rng)
            total_placed += 1

            # Update height map
            if x_end > x_start:
                height_map[x_start:x_end] = np.maximum(
                    height_map[x_start:x_end], y_pos + jh
                )

            cursor_x += jw + grout_gap

        # Record for edge fill
        edge_color = sample_color_nearest(input_image, row, in_w - 1)
        row_end_x.append((cursor_x, edge_color))
        max_x_extent = max(max_x_extent, cursor_x)

    # Edge-fill pass: extend short rows to max_x_extent
    fill_placed = 0
    for (rx, edge_col) in row_end_x:
        while rx < max_x_extent:
            jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))

            x_start = max(rx, 0)
            x_end = min(rx + jw, est_w)
            if x_end > x_start:
                y_pos = int(height_map[x_start:x_end].max()) + grout_gap
            else:
                y_pos = 0
            y_pos = max(y_pos, 0)

            jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))

            angle = rng.choice(rot_angles) if rotation_jitter > 0 else 0.0

            if preview:
                composite_tessera_preview(
                    canvas, edge_col, rx + jw // 2, y_pos + jh // 2,
                    jw, jh, angle, color_variation, rng)
            else:
                t_idx = rng.integers(n_templates)
                lum_base, alpha_base = templates[t_idx]
                lum, alpha = resize_template(lum_base, alpha_base, jw, jh)
                if abs(angle) > 0.1:
                    center = (jw / 2, jh / 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    lum = cv2.warpAffine(lum, M, (jw, jh))
                    alpha = cv2.warpAffine(alpha, M, (jw, jh))
                composite_tessera(canvas, lum, alpha, edge_col, rx, y_pos,
                                  color_variation, rng)

            if x_end > x_start:
                height_map[x_start:x_end] = np.maximum(
                    height_map[x_start:x_end], y_pos + jh
                )

            rx += jw + grout_gap
            fill_placed += 1
            total_placed += 1

    if fill_placed > 0:
        print(f"  Edge fill: {fill_placed:,} extra tesserae")

    # Crop canvas to actual extent
    final_w = min(max_x_extent + tessera_size, est_w)
    final_h = min(int(height_map[:final_w].max()) + tessera_size, est_h)
    canvas = canvas[:final_h, :final_w]

    elapsed = time.time() - t_start
    print(f"Rendering complete: {total_placed:,} tesserae in {elapsed:.1f}s")
    print(f"Input pixels: {total_pixels:,} | Tesserae placed: {total_placed:,}")
    print(f"Output dimensions: {final_w} x {final_h} px")

    return np.clip(canvas, 0, 255).astype(np.uint8), total_placed
