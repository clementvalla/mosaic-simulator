"""Flow-field computation, seed generation, streamline tracing, and background fill.

Shared by both contour (opus vermiculatum) and flow (opus musivum) modes.
"""

import cv2
import numpy as np

from .compositing import composite_tessera, composite_tessera_preview, resize_template, sample_color_nearest


def compute_flow_field(occupied_map):
    """Compute a flow field that radiates outward from occupied (contour) regions.

    Uses the distance transform of unoccupied space; gradient points outward,
    isoline tangent (perpendicular) gives tessera orientation — like tree rings.

    Returns:
        flow_angle: (H, W) gradient direction in degrees (outward from contours)
        isoline_angle: (H, W) tessera orientation in degrees (parallel to contour echoes)
    """
    # Distance from each empty pixel to nearest occupied pixel
    empty_mask = (~occupied_map).astype(np.uint8) * 255
    dist = cv2.distanceTransform(empty_mask, cv2.DIST_L2, 5)

    # Smooth to reduce noise in the gradient
    dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=3)

    # Gradient of distance field
    gx = cv2.Sobel(dist, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(dist, cv2.CV_32F, 0, 1, ksize=5)

    # Gradient direction (outward from contours) in degrees
    flow_angle = np.degrees(np.arctan2(gy, gx))

    # Isoline tangent = perpendicular to gradient (tessera orientation)
    isoline_angle = flow_angle + 90.0

    # Mark low-magnitude regions (singularities between equidistant contours)
    magnitude = np.sqrt(gx**2 + gy**2)
    # Where gradient is too weak, default to 0°
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
    # Dilate occupied map to find the fringe
    kernel = np.ones((cell_step, cell_step), dtype=np.uint8)
    dilated = cv2.dilate(occupied_map.astype(np.uint8), kernel, iterations=1)
    fringe = (dilated > 0) & (~occupied_map)

    # Get fringe pixel coordinates
    ys, xs = np.where(fringe)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Subsample: grid-bucket approach for ~cell_step spacing
    # Quantize to grid cells and keep one per cell
    grid_xs = xs // cell_step
    grid_ys = ys // cell_step
    # Unique grid cells
    keys = grid_ys * (occupied_map.shape[1] // cell_step + 1) + grid_xs
    _, unique_idx = np.unique(keys, return_index=True)

    seeds = np.stack([xs[unique_idx], ys[unique_idx]], axis=1)

    # Shuffle so streamlines don't all start from one corner
    rng.shuffle(seeds)

    return seeds


def trace_streamline(seed_x, seed_y, step_angle, orient_angle, cell_step,
                     canvas_w, canvas_h, max_steps):
    """Trace a streamline from a seed point, stepping along step_angle field.

    step_angle: (H, W) direction to advance in degrees
    orient_angle: (H, W) tessera orientation in degrees

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

        # Step along the stepping direction
        angle_rad = np.radians(step_angle[iy, ix])
        x += np.cos(angle_rad) * cell_step
        y += np.sin(angle_rad) * cell_step

    return positions


def fill_background_flow(canvas, occupied_map, input_image, templates,
                         tessera_size, grout_gap, nominal_cell,
                         color_variation, size_jitter, rotation_jitter, rng,
                         fill_style="radial", preview=False):
    """Fill background using flow-field streamlines from contours.

    fill_style:
        "radial" — streamlines radiate outward (perpendicular to contours),
                   tesserae oriented along isolines
        "concentric" — streamlines follow isolines (parallel to contours),
                       tesserae oriented along the isoline direction

    Multi-wave approach: trace streamlines from contour fringes,
    recompute flow field after each wave to fill gaps.
    """
    in_h, in_w = input_image.shape[:2]
    canvas_h, canvas_w = canvas.shape[:2]
    n_templates = len(templates) if templates else 0
    cell_step = tessera_size + grout_gap
    max_steps = max(canvas_w, canvas_h) // cell_step + 1

    total_placed = 0
    num_waves = 3

    for wave in range(num_waves):
        print(f"    {fill_style.capitalize()} fill wave {wave + 1}/{num_waves}...")

        # (Re)compute flow field from current occupancy
        flow_angle, isoline_angle = compute_flow_field(occupied_map)

        # Choose stepping and orientation directions based on fill style
        if fill_style == "radial":
            # Step outward (gradient), orient along isolines
            step_angle = flow_angle
            orient_angle = isoline_angle
        else:  # concentric
            # Step along isolines (parallel to contours), orient along isolines
            step_angle = isoline_angle
            orient_angle = isoline_angle

        # Generate seeds along the fringe of occupied area
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
                # Size jitter
                jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))
                jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                         int(tessera_size * 0.7))

                # Center tessera on the streamline point
                tx = px - jw // 2
                ty = py - jh // 2

                # Bounds check
                oy0 = max(0, ty)
                oy1 = min(canvas_h, ty + jh)
                ox0 = max(0, tx)
                ox1 = min(canvas_w, tx + jw)
                if oy1 <= oy0 or ox1 <= ox0:
                    continue

                # Occupancy check — skip if >25% already taken
                footprint = occupied_map[oy0:oy1, ox0:ox1]
                if footprint.sum() > 0.25 * footprint.size:
                    continue

                # Rotate to follow isoline (+ small jitter)
                angle = orient_deg + rng.uniform(-rotation_jitter * 0.3,
                                                  rotation_jitter * 0.3)

                # Map output position to input pixel for color
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

                # Mark occupied
                occupied_map[oy0:oy1, ox0:ox1] = True
                wave_placed += 1

        total_placed += wave_placed
        print(f"    Wave {wave + 1}: {wave_placed:,} tesserae")

    # --- Gap cleanup: scan remaining unoccupied cells ---
    print(f"    {fill_style.capitalize()} fill gap cleanup...")
    flow_angle, isoline_angle = compute_flow_field(occupied_map)
    gap_placed = 0

    # Scan on a grid at cell_step intervals
    for gy in range(0, canvas_h, cell_step):
        for gx in range(0, canvas_w, cell_step):
            # Check if center region is mostly unoccupied
            oy0 = max(0, gy)
            oy1 = min(canvas_h, gy + tessera_size)
            ox0 = max(0, gx)
            ox1 = min(canvas_w, gx + tessera_size)
            if oy1 <= oy0 or ox1 <= ox0:
                continue

            footprint = occupied_map[oy0:oy1, ox0:ox1]
            if footprint.sum() > 0.25 * footprint.size:
                continue

            # Size jitter
            jw = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))
            jh = max(int(tessera_size * rng.uniform(1 - size_jitter, 1 + size_jitter)),
                     int(tessera_size * 0.7))

            # Orient by flow field at this point
            cy = min(gy + tessera_size // 2, canvas_h - 1)
            cx = min(gx + tessera_size // 2, canvas_w - 1)
            angle = isoline_angle[cy, cx] + rng.uniform(-rotation_jitter * 0.3,
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

    return total_placed
