"""Flow-field computation, seed generation, streamline tracing, and background fill.

Shared by both contour (opus vermiculatum) and flow (opus musivum) modes.
"""

import cv2
import numpy as np

from .compositing import (
    sample_color_nearest, jitter_size, check_occupancy, ROTATION_JITTER_SCALE,
)
from .placements import Placement


def compute_flow_field(occupied_map):
    """Compute a flow field that radiates outward from occupied (contour) regions.

    Uses the distance transform of unoccupied space; gradient points outward,
    isoline tangent (perpendicular) gives tessera orientation — like tree rings.

    Returns:
        flow_angle: (H, W) gradient direction in degrees (outward from contours)
        isoline_angle: (H, W) tessera orientation in degrees (parallel to contour echoes)
    """
    empty_mask = (~occupied_map).astype(np.uint8) * 255
    dist = cv2.distanceTransform(empty_mask, cv2.DIST_L2, 5)

    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=3)

    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=5)

    flow_angle = np.degrees(np.arctan2(gy, gx))
    isoline_angle = flow_angle + 90.0

    magnitude = np.sqrt(gx**2 + gy**2)
    weak = magnitude < 0.01
    flow_angle[weak] = 0.0
    isoline_angle[weak] = 90.0

    return flow_angle, isoline_angle


def generate_flow_seeds(occupied_map, cell_step, rng):
    """Generate seed points along the fringe of occupied regions.

    Finds unoccupied pixels adjacent to occupied ones, then subsamples
    at approximately cell_step spacing.

    Returns: (N, 2) array of (x, y) seed positions.
    """
    kernel = np.ones((cell_step, cell_step), dtype=np.uint8)
    dilated = cv2.dilate(occupied_map.astype(np.uint8), kernel, iterations=1)
    fringe = (dilated > 0) & (~occupied_map)

    ys, xs = np.where(fringe)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    grid_xs = xs // cell_step
    grid_ys = ys // cell_step
    keys = grid_ys * (occupied_map.shape[1] // cell_step + 1) + grid_xs
    _, unique_idx = np.unique(keys, return_index=True)

    seeds = np.stack([xs[unique_idx], ys[unique_idx]], axis=1)
    rng.shuffle(seeds)

    return seeds


def trace_streamline(seed_x, seed_y, step_angle, orient_angle, cell_step,
                     canvas_w, canvas_h, max_steps):
    """Trace a streamline from a seed point, stepping along step_angle field.

    Returns list of (x, y, orientation_angle) tuples for tessera placement.
    """
    positions = []
    x, y = float(seed_x), float(seed_y)

    for _ in range(max_steps):
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or ix >= canvas_w or iy < 0 or iy >= canvas_h:
            break

        orient_deg = orient_angle[iy, ix]
        positions.append((ix, iy, orient_deg))

        angle_rad = np.radians(step_angle[iy, ix])
        x += np.cos(angle_rad) * cell_step
        y += np.sin(angle_rad) * cell_step

    return positions


def _make_fill_placement(occupied_map, px, py, orient_deg,
                         color_image, tessera_size, nominal_cell,
                         size_jitter, rotation_jitter,
                         rng, grout_gap=0):
    """Plan one fill-tessera placement. Returns Placement or None if blocked.

    Mutates occupied_map to mark the placement as taken.
    """
    in_h, in_w = color_image.shape[:2]
    jw, jh = jitter_size(tessera_size, size_jitter, rng)
    tx, ty = px - jw // 2, py - jh // 2

    blocked, oy0, oy1, ox0, ox1 = check_occupancy(
        occupied_map, tx, ty, jw, jh,
        grout_gap=grout_gap, tessera_size=tessera_size)
    if blocked:
        return None

    angle = orient_deg + rng.uniform(-rotation_jitter * ROTATION_JITTER_SCALE,
                                     rotation_jitter * ROTATION_JITTER_SCALE)

    in_row = np.clip(int(round(py / nominal_cell)), 0, in_h - 1)
    in_col = np.clip(int(round(px / nominal_cell)), 0, in_w - 1)
    color = sample_color_nearest(color_image, in_row, in_col)

    occupied_map[oy0:oy1, ox0:ox1] = True
    return Placement(x=tx, y=ty, w=jw, h=jh,
                     angle=float(angle), color=color)


def plan_background_flow(occupied_map, input_image, tessera_size, grout_gap,
                         nominal_cell, size_jitter, rotation_jitter, rng,
                         fill_style="radial", color_image=None):
    """Plan background-fill placements using flow-field streamlines from contours.

    fill_style:
        "radial" — streamlines radiate outward (perpendicular to contours),
                   tesserae oriented along isolines
        "concentric" — streamlines follow isolines (parallel to contours),
                       tesserae oriented along the isoline direction

    Multi-wave approach: trace streamlines from contour fringes, recompute
    flow field after each wave to fill gaps. Returns list[Placement].
    Mutates occupied_map as it goes.
    """
    if color_image is None:
        color_image = input_image
    canvas_h, canvas_w = occupied_map.shape
    cell_step = tessera_size + grout_gap
    max_steps = max(canvas_w, canvas_h) // cell_step + 1

    placements = []
    num_waves = 3

    for wave in range(num_waves):
        print(f"    {fill_style.capitalize()} fill wave {wave + 1}/{num_waves}...")

        flow_angle, isoline_angle = compute_flow_field(occupied_map)

        if fill_style == "radial":
            step_angle = flow_angle
            orient_angle = isoline_angle
        else:  # concentric
            step_angle = isoline_angle
            orient_angle = isoline_angle

        seeds = generate_flow_seeds(occupied_map, cell_step, rng)
        if len(seeds) == 0:
            print(f"    No seeds found, stopping.")
            break

        print(f"    {len(seeds)} seed points")
        wave_placed = 0

        for seed in seeds:
            sx, sy = seed
            positions = trace_streamline(sx, sy, step_angle, orient_angle,
                                         cell_step, canvas_w, canvas_h,
                                         max_steps)

            for px, py, orient_deg in positions:
                p = _make_fill_placement(
                    occupied_map, px, py, orient_deg,
                    color_image, tessera_size, nominal_cell,
                    size_jitter, rotation_jitter, rng, grout_gap=grout_gap,
                )
                if p is not None:
                    placements.append(p)
                    wave_placed += 1

        print(f"    Wave {wave + 1}: {wave_placed:,} tesserae")

    # Gap cleanup: scan remaining unoccupied cells
    print(f"    {fill_style.capitalize()} fill gap cleanup...")
    flow_angle, isoline_angle = compute_flow_field(occupied_map)
    gap_placed = 0

    for gy in range(0, canvas_h, cell_step):
        for gx in range(0, canvas_w, cell_step):
            blocked, _, _, _, _ = check_occupancy(
                occupied_map, gx, gy, tessera_size, tessera_size,
                grout_gap=grout_gap, tessera_size=tessera_size)
            if blocked:
                continue

            cy = min(gy + tessera_size // 2, canvas_h - 1)
            cx = min(gx + tessera_size // 2, canvas_w - 1)
            orient_deg = isoline_angle[cy, cx]

            p = _make_fill_placement(
                occupied_map,
                gx + tessera_size // 2, gy + tessera_size // 2,
                orient_deg,
                color_image, tessera_size, nominal_cell,
                size_jitter, rotation_jitter, rng, grout_gap=grout_gap,
            )
            if p is not None:
                placements.append(p)
                gap_placed += 1

    print(f"    Gap cleanup: {gap_placed:,} tesserae")
    return placements
