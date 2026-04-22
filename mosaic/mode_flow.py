"""Mode C: Flow placement (opus musivum).

Structure-tensor-driven whole-surface placement. The entire surface is driven
by a single orientation field derived from the image — edge-following emerges
naturally from the structure tensor.
"""

import time

import cv2
import numpy as np

from .compositing import (
    sample_color_nearest, jitter_size, check_occupancy, crop_to_content,
    ROTATION_JITTER_SCALE,
)
from .flow_utils import generate_flow_seeds, trace_streamline
from .placements import Placement, render_placements


def compute_structure_tensor_field(input_image, sigma_integrate=4.0):
    """Compute dominant orientation from the image's structure tensor.

    Returns:
        orientation_deg: (H, W) float32, dominant orientation in degrees [-90, 90]
        coherence: (H, W) float32, orientation strength in [0, 1]
    """
    gray = np.mean(input_image, axis=2).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)

    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    J11 = Ix * Ix
    J12 = Ix * Iy
    J22 = Iy * Iy

    J11 = cv2.GaussianBlur(J11, (0, 0), sigmaX=sigma_integrate)
    J12 = cv2.GaussianBlur(J12, (0, 0), sigmaX=sigma_integrate)
    J22 = cv2.GaussianBlur(J22, (0, 0), sigmaX=sigma_integrate)

    orientation_rad = 0.5 * np.arctan2(2.0 * J12, J11 - J22)
    orientation_deg = np.degrees(orientation_rad)

    trace = J11 + J22 + 1e-8
    coherence = np.sqrt((J11 - J22)**2 + 4.0 * J12**2) / trace
    coherence = np.clip(coherence, 0, 1)

    return orientation_deg, coherence


def upscale_angle_field(angle_deg, target_w, target_h):
    """Upscale an angle field using angle-safe interpolation."""
    angle_rad = np.radians(angle_deg) * 2.0
    cos_comp = np.cos(angle_rad).astype(np.float32)
    sin_comp = np.sin(angle_rad).astype(np.float32)

    cos_up = cv2.resize(cos_comp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    sin_up = cv2.resize(sin_comp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return np.degrees(np.arctan2(sin_up, cos_up)) * 0.5


def generate_surface_seeds(canvas_w, canvas_h, cell_step, rng):
    """Generate seed points across the entire canvas surface."""
    spacing = cell_step * 2
    xs = np.arange(cell_step, canvas_w - cell_step, spacing)
    ys = np.arange(cell_step, canvas_h - cell_step, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    jitter = cell_step * 0.5
    grid_x = grid_x + rng.uniform(-jitter, jitter, len(grid_x))
    grid_y = grid_y + rng.uniform(-jitter, jitter, len(grid_y))

    grid_x = np.clip(grid_x, 0, canvas_w - 1).astype(np.int32)
    grid_y = np.clip(grid_y, 0, canvas_h - 1).astype(np.int32)

    seeds = np.stack([grid_x, grid_y], axis=1)
    rng.shuffle(seeds)
    return seeds


def _plan_flow_tessera(occupied_map, px, py, orient_deg,
                       color_image, tessera_size, nominal_cell,
                       size_jitter, rotation_jitter,
                       rng, grout_gap=0):
    """Plan one flow-mode tessera. Returns Placement or None if blocked."""
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


def _plan_flow(input_image, color_image, tessera_size, grout_width,
               size_jitter, rotation_jitter, sigma_integrate, flow_direction, rng):
    """Full flow-mode planner. Returns (placements, canvas_shape)."""
    in_h, in_w = input_image.shape[:2]

    nominal_cell = tessera_size + grout_width
    grout_gap = max(grout_width, -5)
    cell_step = tessera_size + grout_gap

    canvas_w = int(in_w * nominal_cell) + tessera_size * 2
    canvas_h = int(in_h * nominal_cell) + tessera_size * 2
    occupied_map = np.zeros((canvas_h, canvas_w), dtype=np.bool_)

    # --- Structure tensor ---
    print("  Computing structure tensor field...")
    orient_input, _ = compute_structure_tensor_field(
        input_image, sigma_integrate=sigma_integrate)
    print("  Upscaling orientation field to canvas...")
    orient_canvas = upscale_angle_field(orient_input, canvas_w, canvas_h)

    if flow_direction == "along":
        step_angle = orient_canvas
        orient_angle = orient_canvas
    else:
        step_angle = orient_canvas + 90.0
        orient_angle = orient_canvas

    max_steps = max(canvas_w, canvas_h) // cell_step + 1
    placements = []

    plan_args = dict(
        occupied_map=occupied_map, color_image=color_image,
        tessera_size=tessera_size, nominal_cell=nominal_cell,
        size_jitter=size_jitter, rotation_jitter=rotation_jitter,
        rng=rng, grout_gap=grout_gap,
    )

    # --- Wave 1: broad surface seeding ---
    print("  Wave 1: surface seeds...")
    seeds = generate_surface_seeds(canvas_w, canvas_h, cell_step, rng)
    print(f"    {len(seeds)} seed points")
    wave_placed = 0
    for seed in seeds:
        sx, sy = seed
        for direction_offset in [0.0, 180.0]:
            s_angle = step_angle if direction_offset == 0.0 else step_angle + 180.0
            positions = trace_streamline(sx, sy, s_angle, orient_angle,
                                         cell_step, canvas_w, canvas_h,
                                         max_steps)
            for px, py, orient_deg in positions:
                p = _plan_flow_tessera(px=px, py=py, orient_deg=orient_deg,
                                       **plan_args)
                if p is not None:
                    placements.append(p)
                    wave_placed += 1
    print(f"    Wave 1: {wave_placed:,} tesserae")

    # --- Waves 2-3: fringe seeding to fill gaps ---
    for wave in range(2, 4):
        print(f"  Wave {wave}: fringe seeds...")
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
                p = _plan_flow_tessera(px=px, py=py, orient_deg=orient_deg,
                                       **plan_args)
                if p is not None:
                    placements.append(p)
                    wave_placed += 1
        print(f"    Wave {wave}: {wave_placed:,} tesserae")

    # --- Gap cleanup ---
    print("  Gap cleanup...")
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
            orient_deg = orient_angle[cy, cx]

            p = _plan_flow_tessera(
                px=gx + tessera_size // 2, py=gy + tessera_size // 2,
                orient_deg=orient_deg, **plan_args)
            if p is not None:
                placements.append(p)
                gap_placed += 1
    print(f"    Gap cleanup: {gap_placed:,} tesserae")

    return placements, (canvas_h, canvas_w)


def build_mosaic_flow(input_image, templates, tessera_size, grout_width,
                      grout_color, color_variation, size_jitter,
                      rotation_jitter, rng, sigma_integrate=4.0,
                      flow_direction="along", preview=False, color_image=None):
    """Build mosaic using structure-tensor flow field (opus musivum).

    Returns (canvas_uint8, count, placements).
    """
    if color_image is None:
        color_image = input_image

    t_start = time.time()

    placements, (canvas_h, canvas_w) = _plan_flow(
        input_image, color_image, tessera_size, grout_width,
        size_jitter, rotation_jitter, sigma_integrate, flow_direction, rng,
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
