"""Placement records and the shared render loop.

Each mode runs a planning phase that fills a `list[Placement]` without
painting the canvas, then `render_placements()` composites the whole list
in one pass. This lets the GUI cache a preview's placements and re-run
only the compositing step when the user hits Render, and gives future
exporters (SVG / CSV / g-code) a canonical data structure to consume.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .compositing import (
    composite_tessera, composite_tessera_preview,
    resize_template, rotate_tile, crop_to_content,
)


@dataclass
class Placement:
    """One tessera's final position, size, orientation, and base color.

    x, y: canvas top-left (pixels). For modes that compute a center point,
    the planner converts to top-left before constructing the record.
    w, h: jittered size (pixels).
    angle: final rotation in degrees, including any per-tessera jitter.
    color: pre-variation RGB, float32, 0-255 range. The color-variation
    jitter is applied at render time (one extra rng draw per tessera).
    template_idx: populated at render time for full renders. None for
    preview-only records or records that haven't been rendered yet.
    """
    x: int
    y: int
    w: int
    h: int
    angle: float
    color: np.ndarray
    template_idx: Optional[int] = None


def render_placements(canvas, placements, templates, color_variation,
                      rng, preview=False):
    """Composite every placement onto the canvas.

    Mutates `canvas` in place. Sets `template_idx` on each Placement when
    rendering at full fidelity (useful for downstream exporters).
    """
    n_templates = len(templates) if templates else 0
    if preview:
        for p in placements:
            cx = p.x + p.w // 2
            cy = p.y + p.h // 2
            composite_tessera_preview(canvas, p.color, cx, cy,
                                      p.w, p.h, p.angle,
                                      color_variation, rng)
        return

    if n_templates == 0:
        raise ValueError("render_placements(preview=False) requires templates")

    for p in placements:
        t_idx = p.template_idx
        if t_idx is None:
            t_idx = int(rng.integers(n_templates))
            p.template_idx = t_idx
        lum, alpha = resize_template(*templates[t_idx], p.w, p.h)
        lum, alpha = rotate_tile(lum, alpha, p.angle)
        composite_tessera(canvas, lum, alpha, p.color, p.x, p.y,
                          color_variation, rng)


def rasterize_placements(placements, canvas_shape, grout_color, tessera_size,
                         templates, color_variation, rng, preview=False,
                         crop="content", fixed_shape=None):
    """Allocate a canvas, render placements, crop, return uint8 image.

    crop:
      "content" — `crop_to_content(margin=tessera_size // 2)` (contour / flow).
      "fixed"   — `canvas[:fixed_shape[0], :fixed_shape[1]]` (drift's
                  height-map-derived extent).
    """
    canvas_h, canvas_w = canvas_shape
    canvas = np.full((canvas_h, canvas_w, 3), grout_color, dtype=np.float32)
    render_placements(canvas, placements, templates, color_variation, rng,
                      preview=preview)

    if crop == "content":
        canvas = crop_to_content(canvas, grout_color, margin=tessera_size // 2)
    elif crop == "fixed" and fixed_shape is not None:
        canvas = canvas[:fixed_shape[0], :fixed_shape[1]]
    return np.clip(canvas, 0, 255).astype(np.uint8)
