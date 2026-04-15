"""Tile template loading and normalization."""

import os
import sys

import numpy as np
from PIL import Image


def load_tile_templates(tiles_dir, tessera_size):
    """Load tile PNGs, resize to tessera_size, extract luminance + alpha.

    Returns list of (luminance, alpha) tuples, both float32 arrays
    of shape (tessera_size, tessera_size) with values in [0, 1].
    """
    files = sorted(f for f in os.listdir(tiles_dir) if f.endswith(".png"))
    if not files:
        sys.exit(f"No .png files found in {tiles_dir}")

    templates = []
    for fname in files:
        pil_img = Image.open(os.path.join(tiles_dir, fname)).convert("RGBA")
        rgba = np.array(pil_img, dtype=np.float32)

        alpha = rgba[:, :, 3] / 255.0
        rgb = rgba[:, :, :3]
        lum = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]) / 255.0

        h, w = lum.shape
        scale = min(tessera_size / w, tessera_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        lum_pil = Image.fromarray((lum * 255).astype(np.uint8)).resize(
            (new_w, new_h), Image.LANCZOS
        )
        alpha_pil = Image.fromarray((alpha * 255).astype(np.uint8)).resize(
            (new_w, new_h), Image.LANCZOS
        )

        lum_canvas = np.zeros((tessera_size, tessera_size), dtype=np.float32)
        alpha_canvas = np.zeros((tessera_size, tessera_size), dtype=np.float32)
        ox = (tessera_size - new_w) // 2
        oy = (tessera_size - new_h) // 2
        lum_canvas[oy : oy + new_h, ox : ox + new_w] = (
            np.array(lum_pil, dtype=np.float32) / 255.0
        )
        alpha_canvas[oy : oy + new_h, ox : ox + new_w] = (
            np.array(alpha_pil, dtype=np.float32) / 255.0
        )

        # Normalize luminance so mean over opaque region ≈ 1.0
        opaque = alpha_canvas > 0.1
        if opaque.any():
            mean_lum = lum_canvas[opaque].mean()
            if mean_lum > 0:
                lum_canvas = lum_canvas / mean_lum
                lum_canvas = np.clip(lum_canvas, 0, 2.0)

        templates.append((lum_canvas, alpha_canvas))

    print(f"Loaded {len(templates)} tile templates from {tiles_dir}")
    return templates
