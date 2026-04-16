"""Shared utilities for color sampling, template resizing, and compositing."""

import math

import cv2
import numpy as np


def _allocate_proportional(counts, total_slots, min_each=1):
    """Distribute total_slots across groups proportional to counts.

    Guarantees at least min_each per group. Uses largest-remainder method.
    """
    n = len(counts)
    if n == 0:
        return []
    base = min(min_each, total_slots // max(n, 1))
    allocated = np.full(n, base, dtype=int)
    remaining = total_slots - allocated.sum()
    if remaining > 0:
        fracs = counts / max(counts.sum(), 1) * remaining
        floors = np.floor(fracs).astype(int)
        allocated += floors
        leftover = remaining - floors.sum()
        if leftover > 0:
            remainders = fracs - floors
            for idx in np.argsort(-remainders)[:leftover]:
                allocated[idx] += 1
    return allocated.tolist()


def _safe_kmeans(pixels, k):
    """Run k-means on pixels, handling edge cases. Returns array of centers."""
    n_unique = len(np.unique(pixels, axis=0))
    k = min(k, n_unique, len(pixels))
    if k <= 1:
        return [pixels.mean(axis=0)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        pixels.astype(np.float32), k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    return list(centers)


def quantize_colors(image, n_colors):
    """Reduce image to n_colors with maximum hue diversity.

    Separates chromatic/neutral pixels, selects maximally spread hues via
    greedy farthest-point on the circular hue axis, then subdivides each
    hue group by value/saturation using k-means.

    image: (H, W, 3) float32 RGB (0-255 range)
    n_colors: number of palette colors (1-128)
    Returns: quantized image, same shape and dtype.
    """
    from scipy.spatial.distance import cdist

    h, w = image.shape[:2]
    pixels_rgb = image.reshape(-1, 3).astype(np.float32)

    # Trivial case
    if n_colors == 1:
        mean_color = pixels_rgb.mean(axis=0)
        return np.full_like(image, mean_color)

    # Convert to uint8 then HSV so ranges are H=[0,179], S=[0,255], V=[0,255]
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    img_hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    pixels_hsv = img_hsv.reshape(-1, 3).astype(np.float32)

    # Separate chromatic vs neutral
    SAT_THRESH = 30.0
    chromatic_mask = pixels_hsv[:, 1] >= SAT_THRESH
    neutral_mask = ~chromatic_mask
    n_chrom = chromatic_mask.sum()
    n_neut = neutral_mask.sum()

    # Allocate slots between neutral and chromatic
    if n_neut == 0:
        k_neut, k_chrom = 0, n_colors
    elif n_chrom == 0:
        k_neut, k_chrom = n_colors, 0
    else:
        frac = n_neut / len(pixels_rgb)
        k_neut = max(1, min(n_colors // 3, round(frac * n_colors)))
        k_chrom = n_colors - k_neut

    palette = []

    # --- Neutral centers via plain k-means in RGB ---
    if k_neut > 0 and n_neut > 0:
        centers = _safe_kmeans(pixels_rgb[neutral_mask], k_neut)
        freed = k_neut - len(centers)
        palette.extend(centers)
        k_chrom += freed  # give unused slots to chromatic
    elif k_neut > 0:
        k_chrom += k_neut

    # --- Hue-diversity selection for chromatic pixels ---
    if k_chrom > 0 and n_chrom > 0:
        hues = pixels_hsv[chromatic_mask, 0]  # [0, 179]
        chrom_rgb = pixels_rgb[chromatic_mask]

        N_BINS = 36
        hue_hist, bin_edges = np.histogram(hues, bins=N_BINS, range=(0, 180))
        non_empty = np.where(hue_hist > 0)[0]
        n_groups = min(k_chrom, len(non_empty))

        if n_groups <= 1:
            # Only one hue region — just k-means the whole chromatic set
            palette.extend(_safe_kmeans(chrom_rgb, k_chrom))
        else:
            # Greedy farthest-point on circular hue bins
            selected = [non_empty[np.argmax(hue_hist[non_empty])]]
            remaining_bins = set(non_empty.tolist()) - set(selected)
            while len(selected) < n_groups:
                best = max(remaining_bins, key=lambda b: min(
                    min(abs(b - s), N_BINS - abs(b - s)) for s in selected))
                selected.append(best)
                remaining_bins.discard(best)

            # Assign each chromatic pixel to nearest selected hue bin
            pixel_bins = np.clip(np.digitize(hues, bin_edges) - 1, 0, N_BINS - 1)
            # For each pixel bin, find closest selected bin (circular)
            selected_arr = np.array(selected)
            group_labels = np.empty(len(pixel_bins), dtype=int)
            for i, pb in enumerate(pixel_bins):
                dists = np.minimum(np.abs(pb - selected_arr),
                                   N_BINS - np.abs(pb - selected_arr))
                group_labels[i] = selected_arr[np.argmin(dists)]

            # Count pixels per group and allocate sub-slots
            unique_groups, group_counts = np.unique(group_labels, return_counts=True)
            slots = _allocate_proportional(group_counts, k_chrom, min_each=1)

            # Sub-cluster each hue group in RGB
            for gid, k in zip(unique_groups, slots):
                grp_pixels = chrom_rgb[group_labels == gid]
                palette.extend(_safe_kmeans(grp_pixels, k))

    # --- Final assignment: map every pixel to nearest palette entry ---
    palette = np.array(palette, dtype=np.float32)
    if len(palette) == 0:
        return image.copy()
    dists = cdist(pixels_rgb, palette, metric='euclidean')
    labels = np.argmin(dists, axis=1)
    quantized = palette[labels].reshape(h, w, 3)
    return quantized.astype(np.float32)


def apply_color_influence(image, influence):
    """Blend image color toward white based on influence (0=white, 1=full color).

    image: (H, W, 3) float32 RGB (0-255 range)
    influence: float 0.0 to 1.0
    Returns: blended image, same shape and dtype.
    """
    return image * influence + 255.0 * (1.0 - influence)


def sample_color_nearest(input_image, pixel_row, pixel_col):
    """Get the exact color of a specific input pixel (no interpolation)."""
    h, w = input_image.shape[:2]
    r = np.clip(pixel_row, 0, h - 1)
    c = np.clip(pixel_col, 0, w - 1)
    return input_image[r, c].astype(np.float32)


def resize_template(lum, alpha, target_w, target_h):
    """Resize a (lum, alpha) template to a specific pixel size."""
    r_lum = cv2.resize(lum, (target_w, target_h), interpolation=cv2.INTER_AREA)
    r_alpha = cv2.resize(alpha, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return r_lum, r_alpha


def composite_tessera(canvas, lum, alpha, color, x, y, color_variation, rng):
    """Composite one tessera onto canvas at integer position (x, y).

    color: (3,) float32 RGB
    Returns: True if placed, False if out of bounds.
    """
    th, tw = lum.shape
    canvas_h, canvas_w = canvas.shape[:2]

    # Bounds check
    cy0, cy1 = max(0, y), min(canvas_h, y + th)
    cx0, cx1 = max(0, x), min(canvas_w, x + tw)
    if cy1 <= cy0 or cx1 <= cx0:
        return False

    ty0, tx0 = cy0 - y, cx0 - x
    ty1, tx1 = ty0 + (cy1 - cy0), tx0 + (cx1 - cx0)

    # Brightness jitter (same offset for all channels to avoid color shifts)
    jitter = rng.uniform(-color_variation, color_variation)
    color_j = np.clip(color + jitter, 0, 255)

    # Recolor: luminance * target color
    tile_rgb = lum[ty0:ty1, tx0:tx1, np.newaxis] * color_j[np.newaxis, np.newaxis, :]
    a = alpha[ty0:ty1, tx0:tx1, np.newaxis]

    region = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = tile_rgb * a + region * (1.0 - a)
    return True


def composite_tessera_preview(canvas, color, cx, cy, w, h, angle, color_variation, rng):
    """Draw a rotated colored rectangle — fast preview without tile textures.

    Canvas should be pre-filled with grout color so gaps become grout.
    """
    jitter = rng.uniform(-color_variation, color_variation)
    color_j = np.clip(color + jitter, 0, 255).astype(np.float32)

    # Compute 4 corners of rotated rectangle
    hw, hh = w / 2.0, h / 2.0
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    corners = np.array([
        [cx + (-hw) * cos_a - (-hh) * sin_a, cy + (-hw) * sin_a + (-hh) * cos_a],
        [cx + ( hw) * cos_a - (-hh) * sin_a, cy + ( hw) * sin_a + (-hh) * cos_a],
        [cx + ( hw) * cos_a - ( hh) * sin_a, cy + ( hw) * sin_a + ( hh) * cos_a],
        [cx + (-hw) * cos_a - ( hh) * sin_a, cy + (-hw) * sin_a + ( hh) * cos_a],
    ], dtype=np.int32)

    cv2.fillConvexPoly(canvas, corners, color_j.tolist())
    return True
