"""Shared utilities for color sampling, template resizing, and compositing."""

import math

import cv2
import numpy as np


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
