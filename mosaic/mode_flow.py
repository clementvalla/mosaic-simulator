"""Mode C: Flow placement (opus musivum).

Structure-tensor-driven whole-surface placement. The entire surface is driven
by a single orientation field derived from the image — edge-following emerges
naturally from the structure tensor.
"""

import time

import cv2
import numpy as np

from .compositing import composite_tessera, composite_tessera_preview, resize_template, sample_color_nearest
from .flow_utils import generate_flow_seeds, trace_streamline


def compute_structure_tensor_field(input_image, sigma_integrate=4.0):
    """Compute dominant orientation from the image's structure tensor.

    The structure tensor captures the local gradient distribution. Its dominant
    eigenvector gives the orientation along which the image varies least —
    i.e., parallel to edges.

    Returns:
        orientation_deg: (H, W) float32, dominant orientation in degrees [-90, 90]
        coherence: (H, W) float32, orientation strength in [0, 1]
    """
    gray = np.mean(input_image, axis=2).astype(np.float32)

    # Smooth slightly to reduce noise before gradient
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)

    # Image gradients
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Structure tensor components
    J11 = Ix * Ix
    J12 = Ix * Iy
    J22 = Iy * Iy

    # Integration window — smooth each component
    J11 = cv2.GaussianBlur(J11, (0, 0), sigmaX=sigma_integrate)
    J12 = cv2.GaussianBlur(J12, (0, 0), sigmaX=sigma_integrate)
    J22 = cv2.GaussianBlur(J22, (0, 0), sigmaX=sigma_integrate)

    # Dominant orientation (direction of least change = along edges)
    orientation_rad = 0.5 * np.arctan2(2.0 * J12, J11 - J22)
    orientation_deg = np.degrees(orientation_rad)

    # Coherence: how strongly directional the local region is
    trace = J11 + J22 + 1e-8
    coherence = np.sqrt((J11 - J22)**2 + 4.0 * J12**2) / trace
    coherence = np.clip(coherence, 0, 1)

    return orientation_deg, coherence


def upscale_angle_field(angle_deg, target_w, target_h):
    """Upscale an angle field using angle-safe interpolation.

    Decomposes into sin/cos of doubled angle, resizes, then recovers angle.
    Avoids artifacts at wrapping boundaries.
    """
    angle_rad = np.radians(angle_deg) * 2.0  # double angle for half-circle [-90,90]
    cos_comp = np.cos(angle_rad).astype(np.float32)
    sin_comp = np.sin(angle_rad).astype(np.float32)

    cos_up = cv2.resize(cos_comp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    sin_up = cv2.resize(sin_comp, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    return np.degrees(np.arctan2(sin_up, cos_up)) * 0.5


def generate_surface_seeds(canvas_w, canvas_h, cell_step, rng):
    """Generate seed points across the entire canvas surface.

    Regular grid at 2x cell_step spacing with random jitter, shuffled.
    """
    spacing = cell_step * 2
    xs = np.arange(cell_step, canvas_w - cell_step, spacing)
    ys = np.arange(cell_step, canvas_h - cell_step, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    grid_x = grid_x.ravel()
    grid_y = grid_y.ravel()

    # Add jitter
    jitter = cell_step * 0.5
    grid_x = grid_x + rng.uniform(-jitter, jitter, len(grid_x))
    grid_y = grid_y + rng.uniform(-jitter, jitter, len(grid_y))

    # Clip to canvas bounds
    grid_x = np.clip(grid_x, 0, canvas_w - 1).astype(np.int32)
    grid_y = np.clip(grid_y, 0, canvas_h - 1).astype(np.int32)

    seeds = np.stack([grid_x, grid_y], axis=1)
    rng.shuffle(seeds)
    return seeds


def build_mosaic_flow(input_image, templates, tessera_size, grout_width,
                      grout_color, color_variation, size_jitter,
                      rotation_jitter, rng, sigma_integrate=4.0,
                      flow_direction="along", preview=False):
    """Build mosaic using structure-tensor flow field (opus musivum).

    The entire surface is driven by a single orientation field derived from
    the image. No separate contour/fill phases — edge-following emerges
    naturally from the structure tensor.
    """
    in_h, in_w = input_image.shape[:2]
    n_templates = len(templates) if templates else 0

    nominal_cell = tessera_size + grout_width
    grout_gap = max(grout_width, -tessera_size // 3)
    cell_step = tessera_size + grout_gap

    # Canvas sized to fit nominal grid
    canvas_w = int(in_w * nominal_cell) + tessera_size * 2
    canvas_h = int(in_h * nominal_cell) + tessera_size * 2
    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.float32)

    occupied_map = np.zeros((canvas_h, canvas_w), dtype=np.bool_)

    t_start = time.time()
    total_placed = 0

    # --- Compute structure tensor at input resolution ---
    print("  Computing structure tensor field...")
    orient_input, coherence_input = compute_structure_tensor_field(
        input_image, sigma_integrate=sigma_integrate
    )

    # --- Upscale to canvas resolution (angle-safe) ---
    print("  Upscaling orientation field to canvas...")
    orient_canvas = upscale_angle_field(orient_input, canvas_w, canvas_h)
    coherence_canvas = cv2.resize(coherence_input, (canvas_w, canvas_h),
                                  interpolation=cv2.INTER_LINEAR)

    # Derive stepping and orientation angles
    if flow_direction == "along":
        # Streamlines follow edges, tesserae orient along edges
        step_angle = orient_canvas
        orient_angle = orient_canvas
    else:  # across
        # Streamlines cross edges (radiate outward), tesserae orient along edges
        step_angle = orient_canvas + 90.0
        orient_angle = orient_canvas

    max_steps = max(canvas_w, canvas_h) // cell_step + 1

    # --- Wave 1: broad surface seeding ---
    print("  Wave 1: surface seeds...")
    seeds = generate_surface_seeds(canvas_w, canvas_h, cell_step, rng)
    print(f"    {len(seeds)} seed points")
    wave_placed = 0

    for seed in seeds:
        sx, sy = seed
        # Trace bidirectionally: forward and backward
        for direction_offset in [0.0, 180.0]:
            if direction_offset == 0.0:
                s_angle = step_angle
            else:
                s_angle = step_angle + 180.0

            positions = trace_streamline(sx, sy, s_angle, orient_angle,
                                         cell_step, canvas_w, canvas_h,
                                         max_steps)

            for px, py, orient_deg in positions:
                jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))
                jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))

                tx = px - jw // 2
                ty = py - jh // 2

                oy0 = max(0, ty)
                oy1 = min(canvas_h, ty + jh)
                ox0 = max(0, tx)
                ox1 = min(canvas_w, tx + jw)
                if oy1 <= oy0 or ox1 <= ox0:
                    continue

                footprint = occupied_map[oy0:oy1, ox0:ox1]
                if footprint.sum() > 0.25 * footprint.size:
                    continue

                angle = orient_deg + rng.uniform(-rotation_jitter * 0.3,
                                                  rotation_jitter * 0.3)

                in_row = np.clip(int(round(py / nominal_cell)), 0, in_h - 1)
                in_col = np.clip(int(round(px / nominal_cell)), 0, in_w - 1)
                color = sample_color_nearest(input_image, in_row, in_col)

                if preview:
                    composite_tessera_preview(
                        canvas, color, px, py, jw, jh, angle,
                        color_variation, rng)
                else:
                    t_idx = rng.integers(n_templates)
                    lum_base, alpha_base = templates[t_idx]
                    lum, alpha = resize_template(lum_base, alpha_base, jw, jh)
                    if abs(angle) > 0.1:
                        center = (jw / 2, jh / 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        lum = cv2.warpAffine(lum, M, (jw, jh))
                        alpha = cv2.warpAffine(alpha, M, (jw, jh))
                    composite_tessera(canvas, lum, alpha, color, tx, ty,
                                      color_variation, rng)

                occupied_map[oy0:oy1, ox0:ox1] = True
                wave_placed += 1

    total_placed += wave_placed
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
                jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))
                jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))

                tx = px - jw // 2
                ty = py - jh // 2

                oy0 = max(0, ty)
                oy1 = min(canvas_h, ty + jh)
                ox0 = max(0, tx)
                ox1 = min(canvas_w, tx + jw)
                if oy1 <= oy0 or ox1 <= ox0:
                    continue

                footprint = occupied_map[oy0:oy1, ox0:ox1]
                if footprint.sum() > 0.25 * footprint.size:
                    continue

                angle = orient_deg + rng.uniform(-rotation_jitter * 0.3,
                                                  rotation_jitter * 0.3)

                in_row = np.clip(int(round(py / nominal_cell)), 0, in_h - 1)
                in_col = np.clip(int(round(px / nominal_cell)), 0, in_w - 1)
                color = sample_color_nearest(input_image, in_row, in_col)

                if preview:
                    composite_tessera_preview(
                        canvas, color, px, py, jw, jh, angle,
                        color_variation, rng)
                else:
                    t_idx = rng.integers(n_templates)
                    lum_base, alpha_base = templates[t_idx]
                    lum, alpha = resize_template(lum_base, alpha_base, jw, jh)
                    if abs(angle) > 0.1:
                        center = (jw / 2, jh / 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        lum = cv2.warpAffine(lum, M, (jw, jh))
                        alpha = cv2.warpAffine(alpha, M, (jw, jh))
                    composite_tessera(canvas, lum, alpha, color, tx, ty,
                                      color_variation, rng)

                occupied_map[oy0:oy1, ox0:ox1] = True
                wave_placed += 1

        total_placed += wave_placed
        print(f"    Wave {wave}: {wave_placed:,} tesserae")

    # --- Gap cleanup ---
    print("  Gap cleanup...")
    gap_placed = 0
    for gy in range(0, canvas_h, cell_step):
        for gx in range(0, canvas_w, cell_step):
            oy0 = max(0, gy)
            oy1 = min(canvas_h, gy + tessera_size)
            ox0 = max(0, gx)
            ox1 = min(canvas_w, gx + tessera_size)
            if oy1 <= oy0 or ox1 <= ox0:
                continue

            footprint = occupied_map[oy0:oy1, ox0:ox1]
            if footprint.sum() > 0.25 * footprint.size:
                continue

            jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))
            jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))

            cy = min(gy + tessera_size // 2, canvas_h - 1)
            cx = min(gx + tessera_size // 2, canvas_w - 1)
            angle = orient_angle[cy, cx] + rng.uniform(-rotation_jitter * 0.3,
                                                        rotation_jitter * 0.3)

            in_row = np.clip(int(round(gy / nominal_cell)), 0, in_h - 1)
            in_col = np.clip(int(round(gx / nominal_cell)), 0, in_w - 1)
            color = sample_color_nearest(input_image, in_row, in_col)

            if preview:
                composite_tessera_preview(
                    canvas, color, gx + tessera_size // 2, gy + tessera_size // 2,
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
                composite_tessera(canvas, lum, alpha, color, gx, gy,
                                  color_variation, rng)
            occupied_map[oy0:oy1, ox0:ox1] = True
            gap_placed += 1

    total_placed += gap_placed
    print(f"    Gap cleanup: {gap_placed:,} tesserae")

    # --- Crop to content ---
    grout_arr = np.array(grout_color, dtype=np.float32)
    diff_from_grout = np.abs(canvas - grout_arr).sum(axis=2)
    occupied = diff_from_grout > 1.0
    if occupied.any():
        rows_occ = np.any(occupied, axis=1)
        cols_occ = np.any(occupied, axis=0)
        r0, r1 = np.argmax(rows_occ), len(rows_occ) - np.argmax(rows_occ[::-1])
        c0, c1 = np.argmax(cols_occ), len(cols_occ) - np.argmax(cols_occ[::-1])
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
