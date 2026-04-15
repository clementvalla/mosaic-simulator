"""Mosaic simulation package — realistic tessera placement in three modes.

Modes:
    drift   — Opus Tessellatum (row-by-row with cumulative drift)
    contour — Opus Vermiculatum (contour-following worm-like chains)
    flow    — Opus Musivum (structure-tensor-driven whole-surface)
"""

from .tiles import load_tile_templates
from .compositing import sample_color_nearest, resize_template, composite_tessera, composite_tessera_preview
from .report import generate_report
from .mode_drift import build_mosaic_drift
from .mode_contour import build_mosaic_contour
from .mode_flow import build_mosaic_flow

__all__ = [
    "load_tile_templates",
    "sample_color_nearest",
    "resize_template",
    "composite_tessera",
    "composite_tessera_preview",
    "generate_report",
    "build_mosaic_drift",
    "build_mosaic_contour",
    "build_mosaic_flow",
]
