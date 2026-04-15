"""Mode B: Contour placement (opus vermiculatum).

Contour-following tessera placement — detects edges, chains tesserae along
each path like a worm, with echo rows paralleling the contour on both sides.
"""

import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .compositing import composite_tessera, composite_tessera_preview, resize_template, sample_color_nearest
from .flow_utils import fill_background_flow


def compute_edge_field(input_image):
    """Compute edge tangent angle and strength for each pixel.

    Returns:
        tangent_deg: (H, W) array of edge tangent angles in degrees
        edge_strength: (H, W) array normalized to [0, 1]
    """
    gray = np.mean(input_image, axis=2).astype(np.float32)
    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    # Edge magnitude
    mag = np.sqrt(gx**2 + gy**2)
    edge_strength = mag / (mag.max() + 1e-8)
    # Tangent direction (perpendicular to gradient: rotate 90°)
    tangent_deg = np.degrees(np.arctan2(-gx, gy))
    return tangent_deg, edge_strength


def walk_contour_path(pts_out, step):
    """Walk along an output-space path at regular intervals.

    pts_out: (N, 2) array of (x, y) positions in output space
    step: spacing between tessera placements

    Yields (x, y, tangent_angle_deg) at each placement point.
    """
    diffs = np.diff(pts_out, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_arc = arc_lengths[-1]

    arc = 0
    while arc < total_arc:
        idx = np.searchsorted(arc_lengths, arc, side='right') - 1
        idx = np.clip(idx, 0, len(pts_out) - 2)
        seg_len = max(seg_lengths[idx], 1e-8)
        t = np.clip((arc - arc_lengths[idx]) / seg_len, 0, 1)
        pos = pts_out[idx] * (1 - t) + pts_out[idx + 1] * t

        # Tangent from nearby points for smoothness
        i_prev = max(0, idx - 1)
        i_next = min(len(pts_out) - 1, idx + 2)
        dx = pts_out[i_next][0] - pts_out[i_prev][0]
        dy = pts_out[i_next][1] - pts_out[i_prev][1]
        angle = np.degrees(np.arctan2(-dx, dy))

        yield int(pos[0]), int(pos[1]), angle
        arc += step


def offset_contour(pts_out, distance):
    """Offset a contour path by a distance along its normals.

    Positive distance = offset to the right of the path direction.
    Returns offset points, filtering out any that fold back on themselves.
    """
    if len(pts_out) < 3:
        return pts_out.copy()

    # Compute tangent vectors (smoothed over a few points)
    tangents = np.zeros_like(pts_out)
    for i in range(len(pts_out)):
        i0 = max(0, i - 2)
        i1 = min(len(pts_out) - 1, i + 2)
        tangents[i] = pts_out[i1] - pts_out[i0]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    tangents = tangents / lengths

    # Normal = perpendicular to tangent (rotate 90° clockwise)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)

    return pts_out + normals * distance


def smooth_path(pts, sigma=3):
    """Gaussian-smooth a path to remove pixel-level jitter."""
    if len(pts) < sigma * 2:
        return pts
    smoothed = np.copy(pts)
    smoothed[:, 0] = gaussian_filter1d(pts[:, 0], sigma=sigma)
    smoothed[:, 1] = gaussian_filter1d(pts[:, 1], sigma=sigma)
    return smoothed


def build_mosaic_contour(input_image, templates, tessera_size, grout_width,
                         grout_color, color_variation, size_jitter,
                         rotation_jitter, edge_threshold, rng,
                         fill_style="drift", preview=False):
    """Build mosaic with contour-following tessera placement (opus vermiculatum).

    Detects contours, then chains tesserae along each path like a worm —
    each tessera's position is determined by advancing from where the
    previous one ended, in the direction of the path tangent.
    Echo rows parallel the contour on both sides.
    No drift background — contour only.
    """
    in_h, in_w = input_image.shape[:2]
    n_templates = len(templates) if templates else 0

    nominal_cell = tessera_size + grout_width
    grout_gap = max(grout_width, -tessera_size // 3)

    # Canvas sized to fit nominal grid
    canvas_w = int(in_w * nominal_cell) + tessera_size * 2
    canvas_h = int(in_h * nominal_cell) + tessera_size * 2
    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.float32)

    t_start = time.time()
    total_placed = 0

    # Occupancy map: tracks which pixels already have a tessera
    occupied_map = np.zeros((canvas_h, canvas_w), dtype=np.bool_)

    # --- Detect contours ---
    print("  Detecting contours...")
    gray = np.mean(input_image, axis=2).astype(np.uint8)
    median_val = np.median(gray)
    canny_lo = int(max(0, 0.5 * median_val))
    canny_hi = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray, canny_lo, canny_hi)

    contours_raw, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_contour_len = 20  # filter out tiny fragments
    contours_filtered = [c for c in contours_raw if len(c) >= min_contour_len]
    # Sort longest first so major contours get priority
    contours_filtered.sort(key=lambda c: len(c), reverse=True)
    print(f"  Found {len(contours_filtered)} contour paths (from {len(contours_raw)} raw)")

    # --- Chain tesserae along each contour ---
    echo_rows = 2  # parallel rows on each side (2 = 5 total: center + 2 per side)

    for contour in contours_filtered:
        pts_input = contour.reshape(-1, 2).astype(np.float64)  # (N, 2): col, row
        # Smooth path in input space to remove pixel-level jaggedness
        pts_input = smooth_path(pts_input, sigma=3)
        # Convert to output space
        pts_out = pts_input * nominal_cell

        for echo in range(-echo_rows, echo_rows + 1):
            if echo == 0:
                path = pts_out
            else:
                path = offset_contour(pts_out, echo * (tessera_size + grout_gap))

            if len(path) < 3:
                continue

            # Compute arc-length parameterization
            diffs = np.diff(path, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_arc = arc_lengths[-1]

            if total_arc < tessera_size:
                continue

            # Chain: start at arc=0, place tessera, advance by its width + grout
            arc = 0
            prev_angle = None

            while arc < total_arc:
                # Interpolate position on path
                idx = np.searchsorted(arc_lengths, arc, side='right') - 1
                idx = np.clip(idx, 0, len(path) - 2)
                seg_len = max(seg_lengths[idx], 1e-8)
                t = np.clip((arc - arc_lengths[idx]) / seg_len, 0, 1)
                pos = path[idx] * (1 - t) + path[idx + 1] * t

                # Tangent from nearby path points (wide window for smooth direction)
                i_prev = max(0, idx - 5)
                i_next = min(len(path) - 1, idx + 5)
                dx = path[i_next][0] - path[i_prev][0]
                dy = path[i_next][1] - path[i_prev][1]
                path_angle = np.degrees(np.arctan2(dy, dx))

                # Smooth angle transition from previous tessera
                if prev_angle is not None:
                    # Unwrap angle difference to avoid jumps
                    diff = path_angle - prev_angle
                    diff = (diff + 180) % 360 - 180
                    # Blend: 70% path direction, 30% continuity from previous
                    path_angle = prev_angle + diff * 0.7

                # Small random jitter on top
                angle = path_angle + rng.uniform(-rotation_jitter * 0.3,
                                                  rotation_jitter * 0.3)
                prev_angle = path_angle

                # Jitter tessera size
                jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))
                jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))

                # Center tessera on path position
                out_x = int(pos[0] - jw / 2)
                out_y = int(pos[1] - jh / 2)

                # Check occupancy — skip if footprint is already taken
                oy0 = max(0, out_y)
                oy1 = min(canvas_h, out_y + jh)
                ox0 = max(0, out_x)
                ox1 = min(canvas_w, out_x + jw)
                if oy1 > oy0 and ox1 > ox0:
                    footprint = occupied_map[oy0:oy1, ox0:ox1]
                    # Skip if more than 25% of footprint is already occupied
                    if footprint.sum() > 0.25 * footprint.size:
                        arc += jw + grout_gap
                        continue

                # Get color from input pixel at this position
                in_col = np.clip(int(round(pos[0] / nominal_cell)), 0, in_w - 1)
                in_row = np.clip(int(round(pos[1] / nominal_cell)), 0, in_h - 1)

                color = sample_color_nearest(input_image, in_row, in_col)

                if preview:
                    composite_tessera_preview(
                        canvas, color, int(pos[0]), int(pos[1]),
                        jw, jh, angle, color_variation, rng)
                else:
                    t_idx = rng.integers(n_templates)
                    lum, alpha = resize_template(*templates[t_idx], jw, jh)
                    if abs(angle) > 0.1:
                        center = (jw / 2, jh / 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        lum = cv2.warpAffine(lum, M, (jw, jh))
                        alpha = cv2.warpAffine(alpha, M, (jw, jh))
                    composite_tessera(canvas, lum, alpha, color, out_x, out_y,
                                      color_variation, rng)
                total_placed += 1

                # Mark footprint as occupied
                if oy1 > oy0 and ox1 > ox0:
                    occupied_map[oy0:oy1, ox0:ox1] = True

                # Advance along path by this tessera's width + grout
                arc += jw + grout_gap

    contour_placed = total_placed
    print(f"  Contour pass: {contour_placed:,} tesserae")

    # --- Fill pass: fill remaining space ---
    if fill_style in ("radial", "concentric"):
        print(f"  Filling background with {fill_style} field...")
        fill_placed = fill_background_flow(
            canvas, occupied_map, input_image, templates,
            tessera_size, grout_gap, nominal_cell,
            color_variation, size_jitter, rotation_jitter, rng,
            fill_style=fill_style, preview=preview,
        )
        total_placed += fill_placed
        print(f"  {fill_style.capitalize()} fill: {fill_placed:,} tesserae")

    else:  # drift
        print("  Filling background with drift rows...")
        # Pre-compute rotation angles for jitter
        if rotation_jitter > 0:
            rot_angles = np.linspace(-rotation_jitter, rotation_jitter, 11)
        else:
            rot_angles = [0.0]

        # Use height map for vertical packing (same as drift mode)
        height_map = np.zeros(canvas_w, dtype=np.float64)

        for row in range(in_h):
            if row % 20 == 0 and row > 0:
                elapsed = time.time() - t_start
                pct = row / in_h * 100
                print(f"  Fill row {row}/{in_h} ({pct:.0f}%) - {elapsed:.1f}s")

            row_height_snap = height_map.copy()
            cursor_x = 0

            for col in range(in_w):
                jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))
                jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))

                x_start = max(cursor_x, 0)
                x_end = min(cursor_x + jw, canvas_w)
                if x_end > x_start:
                    y_pos = int(row_height_snap[x_start:x_end].max()) + grout_gap
                else:
                    y_pos = 0
                y_pos = max(y_pos, 0)

                # Check occupancy — skip if already taken by contour pass
                oy0 = max(0, y_pos)
                oy1 = min(canvas_h, y_pos + jh)
                ox0 = max(0, cursor_x)
                ox1 = min(canvas_w, cursor_x + jw)
                if oy1 > oy0 and ox1 > ox0:
                    footprint = occupied_map[oy0:oy1, ox0:ox1]
                    if footprint.sum() > 0.25 * footprint.size:
                        # Still advance cursor and height map so rows don't get stuck
                        cursor_x += jw + grout_gap
                        if x_end > x_start:
                            height_map[x_start:x_end] = np.maximum(
                                height_map[x_start:x_end], y_pos + jh)
                        continue

                # Map output position back to input space for correct color
                in_row_actual = np.clip(int(round(y_pos / nominal_cell)), 0, in_h - 1)
                in_col_actual = np.clip(int(round(cursor_x / nominal_cell)), 0, in_w - 1)
                color = sample_color_nearest(input_image, in_row_actual, in_col_actual)

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

                # Mark occupied
                if oy1 > oy0 and ox1 > ox0:
                    occupied_map[oy0:oy1, ox0:ox1] = True

                if x_end > x_start:
                    height_map[x_start:x_end] = np.maximum(
                        height_map[x_start:x_end], y_pos + jh)

                cursor_x += jw + grout_gap

        fill_placed = total_placed - contour_placed
        print(f"  Drift fill: {fill_placed:,} tesserae")

    # Crop to content (find bounding box of non-grout pixels)
    grout_arr = np.array(grout_color, dtype=np.float32)
    diff_from_grout = np.abs(canvas - grout_arr).sum(axis=2)
    occupied = diff_from_grout > 1.0
    if occupied.any():
        rows_occ = np.any(occupied, axis=1)
        cols_occ = np.any(occupied, axis=0)
        r0, r1 = np.argmax(rows_occ), len(rows_occ) - np.argmax(rows_occ[::-1])
        c0, c1 = np.argmax(cols_occ), len(cols_occ) - np.argmax(cols_occ[::-1])
        # Add small margin
        margin = tessera_size // 2
        r0 = max(0, r0 - margin)
        c0 = max(0, c0 - margin)
        r1 = min(canvas_h, r1 + margin)
        c1 = min(canvas_w, c1 + margin)
        canvas = canvas[r0:r1, c0:c1]

    final_h, final_w = canvas.shape[:2]

    elapsed = time.time() - t_start
    print(f"Rendering complete: {total_placed:,} tesserae in {elapsed:.1f}s")
    print(f"Output dimensions: {final_w} x {final_h} px")

    return np.clip(canvas, 0, 255).astype(np.uint8), total_placed
