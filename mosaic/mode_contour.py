"""Mode B: Contour placement (opus vermiculatum).

Contour-following tessera placement — detects edges, chains tesserae along
each path like a worm, with echo rows paralleling the contour on both sides.
"""

import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .compositing import (
    sample_color_nearest, jitter_size, check_occupancy, crop_to_content,
    ROTATION_JITTER_SCALE,
)
from .flow_utils import plan_background_flow
from .placements import Placement, render_placements


def compute_edge_field(input_image):
    """Compute edge tangent angle and strength for each pixel.

    Returns:
        tangent_deg: (H, W) array of edge tangent angles in degrees
        edge_strength: (H, W) array normalized to [0, 1]
    """
    gray = np.mean(input_image, axis=2).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    edge_strength = mag / (mag.max() + 1e-8)
    tangent_deg = np.degrees(np.arctan2(-gx, gy))
    return tangent_deg, edge_strength


def walk_contour_path(pts_out, step):
    """Walk along an output-space path at regular intervals.

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

    tangents = np.zeros_like(pts_out)
    for i in range(len(pts_out)):
        i0 = max(0, i - 2)
        i1 = min(len(pts_out) - 1, i + 2)
        tangents[i] = pts_out[i1] - pts_out[i0]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-8)
    tangents = tangents / lengths

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


def _plan_contour_tessera(occupied_map, pos_x, pos_y, angle, color,
                          tessera_size, size_jitter, rng, grout_gap=0):
    """Plan one contour-chain tessera centered on (pos_x, pos_y).

    Returns (placement_or_None, jw). jw is returned even when None so the
    caller can still advance the arc pointer.
    """
    jw, jh = jitter_size(tessera_size, size_jitter, rng)
    out_x = int(pos_x - jw / 2)
    out_y = int(pos_y - jh / 2)

    blocked, oy0, oy1, ox0, ox1 = check_occupancy(
        occupied_map, out_x, out_y, jw, jh,
        grout_gap=grout_gap, tessera_size=tessera_size)
    if blocked:
        return None, jw

    occupied_map[oy0:oy1, ox0:ox1] = True
    return Placement(x=out_x, y=out_y, w=jw, h=jh,
                     angle=float(angle), color=color), jw


def _compute_shape_mask(edges, nominal_cell, canvas_h, canvas_w):
    """Rasterize enclosed-edge regions to a canvas-space bool mask.

    Morphologically closes the Canny output to merge near-parallel edge rims,
    finds external contours, then fills each one. Returns a (canvas_h, canvas_w)
    bool mask where True = inside an enclosed edge boundary ("the shape").
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    ext_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
    if not ext_contours:
        return np.zeros((canvas_h, canvas_w), dtype=np.bool_)

    shape_u8 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    scaled = [(c.astype(np.float32) * nominal_cell).astype(np.int32)
              for c in ext_contours]
    cv2.drawContours(shape_u8, scaled, -1, 255, thickness=cv2.FILLED)
    return shape_u8 > 0


def _plan_drift_fill(occupied_map, in_h, in_w, canvas_w,
                     nominal_cell, tessera_size, grout_gap,
                     size_jitter, rotation_jitter,
                     color_image, rng, t_start, label):
    """Drift-style row-by-row fill. Returns list[Placement]."""
    if rotation_jitter > 0:
        rot_angles = np.linspace(-rotation_jitter, rotation_jitter, 11)
    else:
        rot_angles = [0.0]

    height_map = np.zeros(canvas_w, dtype=np.float64)
    placements = []

    for row in range(in_h):
        if row % 20 == 0 and row > 0:
            elapsed = time.time() - t_start
            pct = row / in_h * 100
            print(f"  {label} row {row}/{in_h} ({pct:.0f}%) - {elapsed:.1f}s")

        row_height_snap = height_map.copy()
        cursor_x = 0

        for col in range(in_w):
            jw, jh = jitter_size(tessera_size, size_jitter, rng)

            x_start = max(cursor_x, 0)
            x_end = min(cursor_x + jw, canvas_w)
            if x_end > x_start:
                y_pos = int(row_height_snap[x_start:x_end].max()) + grout_gap
            else:
                y_pos = 0
            y_pos = max(y_pos, 0)

            blocked, oy0, oy1, ox0, ox1 = check_occupancy(
                occupied_map, cursor_x, y_pos, jw, jh,
                grout_gap=grout_gap, tessera_size=tessera_size)
            if blocked:
                cursor_x += jw + grout_gap
                if x_end > x_start:
                    height_map[x_start:x_end] = np.maximum(
                        height_map[x_start:x_end], y_pos + jh)
                continue

            in_row_actual = np.clip(int(round(y_pos / nominal_cell)), 0, in_h - 1)
            in_col_actual = np.clip(int(round(cursor_x / nominal_cell)), 0, in_w - 1)
            color = sample_color_nearest(color_image, in_row_actual, in_col_actual)

            angle = rng.choice(rot_angles) if rotation_jitter > 0 else 0.0

            occupied_map[oy0:oy1, ox0:ox1] = True
            placements.append(Placement(
                x=cursor_x, y=y_pos, w=jw, h=jh,
                angle=float(angle), color=color,
            ))

            if x_end > x_start:
                height_map[x_start:x_end] = np.maximum(
                    height_map[x_start:x_end], y_pos + jh)

            cursor_x += jw + grout_gap

    return placements


def _run_fill_plan(style, occupied_map, input_image, tessera_size, grout_gap,
                   nominal_cell, size_jitter, rotation_jitter, rng,
                   color_image, in_h, in_w, canvas_w, t_start, label):
    """Dispatch to the requested fill strategy. Returns list[Placement]."""
    if style in ("radial", "concentric"):
        print(f"  {label}: {style} fill...")
        return plan_background_flow(
            occupied_map, input_image, tessera_size, grout_gap, nominal_cell,
            size_jitter, rotation_jitter, rng,
            fill_style=style, color_image=color_image,
        )
    print(f"  {label}: drift fill...")
    return _plan_drift_fill(occupied_map, in_h, in_w, canvas_w,
                            nominal_cell, tessera_size, grout_gap,
                            size_jitter, rotation_jitter,
                            color_image, rng, t_start, label)


def _plan_contour(input_image, color_image, tessera_size, grout_width,
                  size_jitter, rotation_jitter, edge_threshold, rng,
                  fill_style, inner_rows, outer_rows,
                  inner_fill_style, outer_fill_style):
    """Full contour-mode planner. Returns (placements, canvas_shape)."""
    in_h, in_w = input_image.shape[:2]
    nominal_cell = tessera_size + grout_width
    grout_gap = max(grout_width, -5)

    canvas_w = int(in_w * nominal_cell) + tessera_size * 2
    canvas_h = int(in_h * nominal_cell) + tessera_size * 2
    occupied_map = np.zeros((canvas_h, canvas_w), dtype=np.bool_)
    t_start = time.time()

    # --- Detect contours ---
    print("  Detecting contours...")
    gray = np.mean(input_image, axis=2).astype(np.uint8)
    median_val = np.median(gray)
    canny_lo = int(max(0, 0.5 * median_val))
    canny_hi = int(min(255, 1.5 * median_val))
    edges = cv2.Canny(gray, canny_lo, canny_hi)

    contours_raw, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_contour_len = 20
    contours_filtered = [c for c in contours_raw if len(c) >= min_contour_len]
    contours_filtered.sort(key=lambda c: len(c), reverse=True)
    print(f"  Found {len(contours_filtered)} contour paths (from {len(contours_raw)} raw)")

    # --- Chain tesserae along each contour ---
    inner_rows = max(0, int(inner_rows))
    outer_rows = max(0, int(outer_rows))
    placements = []

    for contour in contours_filtered:
        pts_input = contour.reshape(-1, 2).astype(np.float64)
        pts_input = smooth_path(pts_input, sigma=3)
        pts_out = pts_input * nominal_cell

        for echo in range(-inner_rows, outer_rows + 1):
            if echo == 0:
                path = pts_out
            else:
                path = offset_contour(pts_out, echo * (tessera_size + grout_gap))

            if len(path) < 3:
                continue

            diffs = np.diff(path, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            arc_lengths = np.concatenate([[0], np.cumsum(seg_lengths)])
            total_arc = arc_lengths[-1]

            if total_arc < tessera_size:
                continue

            arc = 0
            prev_angle = None

            while arc < total_arc:
                idx = np.searchsorted(arc_lengths, arc, side='right') - 1
                idx = np.clip(idx, 0, len(path) - 2)
                seg_len = max(seg_lengths[idx], 1e-8)
                t = np.clip((arc - arc_lengths[idx]) / seg_len, 0, 1)
                pos = path[idx] * (1 - t) + path[idx + 1] * t

                i_prev = max(0, idx - 5)
                i_next = min(len(path) - 1, idx + 5)
                dx = path[i_next][0] - path[i_prev][0]
                dy = path[i_next][1] - path[i_prev][1]
                path_angle = np.degrees(np.arctan2(dy, dx))

                if prev_angle is not None:
                    d = path_angle - prev_angle
                    d = (d + 180) % 360 - 180
                    path_angle = prev_angle + d * 0.7

                angle = path_angle + rng.uniform(
                    -rotation_jitter * ROTATION_JITTER_SCALE,
                    rotation_jitter * ROTATION_JITTER_SCALE)
                prev_angle = path_angle

                in_col = np.clip(int(round(pos[0] / nominal_cell)), 0, in_w - 1)
                in_row = np.clip(int(round(pos[1] / nominal_cell)), 0, in_h - 1)
                color = sample_color_nearest(color_image, in_row, in_col)

                p, jw = _plan_contour_tessera(
                    occupied_map, pos[0], pos[1], angle, color,
                    tessera_size, size_jitter, rng, grout_gap=grout_gap)
                if p is not None:
                    placements.append(p)

                arc += jw + grout_gap

    contour_placed = len(placements)
    print(f"  Contour pass: {contour_placed:,} tesserae")

    # --- Fill pass (optionally split inside/outside shape) ---
    inner_style = inner_fill_style or fill_style
    outer_style = outer_fill_style or fill_style
    fill_args = dict(
        occupied_map=occupied_map, input_image=input_image,
        tessera_size=tessera_size, grout_gap=grout_gap,
        nominal_cell=nominal_cell, size_jitter=size_jitter,
        rotation_jitter=rotation_jitter, rng=rng, color_image=color_image,
        in_h=in_h, in_w=in_w, canvas_w=canvas_w, t_start=t_start,
    )

    if inner_style == outer_style:
        fill = _run_fill_plan(
            inner_style, label=f"Fill ({inner_style})", **fill_args)
        placements.extend(fill)
        print(f"  Fill: {len(fill):,} tesserae")
    else:
        print("  Computing shape mask for inside/outside split...")
        shape_mask = _compute_shape_mask(edges, nominal_cell, canvas_h, canvas_w)
        shape_pixels = int(shape_mask.sum())
        print(f"  Shape mask: {shape_pixels:,} px inside / "
              f"{shape_mask.size - shape_pixels:,} px outside")

        if shape_pixels == 0:
            print("  No enclosed regions; running outer fill only.")
            fill = _run_fill_plan(
                outer_style, label=f"Fill ({outer_style})", **fill_args)
            placements.extend(fill)
            print(f"  Fill: {len(fill):,} tesserae")
        else:
            saved = occupied_map.copy()
            occupied_map[~shape_mask] = True
            inner_fill = _run_fill_plan(
                inner_style, label=f"Inner fill ({inner_style})", **fill_args)
            occupied_map[~shape_mask] = saved[~shape_mask]
            placements.extend(inner_fill)
            print(f"  Inner fill: {len(inner_fill):,} tesserae")

            saved = occupied_map.copy()
            occupied_map[shape_mask] = True
            outer_fill = _run_fill_plan(
                outer_style, label=f"Outer fill ({outer_style})", **fill_args)
            occupied_map[shape_mask] = saved[shape_mask]
            placements.extend(outer_fill)
            print(f"  Outer fill: {len(outer_fill):,} tesserae")

    return placements, (canvas_h, canvas_w)


def build_mosaic_contour(input_image, templates, tessera_size, grout_width,
                         grout_color, color_variation, size_jitter,
                         rotation_jitter, edge_threshold, rng,
                         fill_style="drift", preview=False, color_image=None,
                         inner_rows=2, outer_rows=2,
                         inner_fill_style=None, outer_fill_style=None):
    """Build mosaic with contour-following tessera placement (opus vermiculatum).

    Returns (canvas_uint8, count, placements).
    """
    if color_image is None:
        color_image = input_image
    t_start = time.time()

    placements, (canvas_h, canvas_w) = _plan_contour(
        input_image, color_image, tessera_size, grout_width,
        size_jitter, rotation_jitter, edge_threshold, rng,
        fill_style, inner_rows, outer_rows,
        inner_fill_style, outer_fill_style,
    )

    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.float32)
    render_placements(canvas, placements, templates, color_variation, rng,
                      preview=preview)

    canvas = crop_to_content(canvas, grout_color, margin=tessera_size // 2)
    final_h, final_w = canvas.shape[:2]

    elapsed = time.time() - t_start
    total_placed = len(placements)
    print(f"Rendering complete: {total_placed:,} tesserae in {elapsed:.1f}s")
    print(f"Output dimensions: {final_w} x {final_h} px")

    plan_info = {
        "canvas_shape": (canvas_h, canvas_w),
        "crop": "content",
        "fixed_shape": None,
    }
    return (np.clip(canvas, 0, 255).astype(np.uint8),
            total_placed, placements, plan_info)
