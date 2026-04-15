#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulate a mosaic with realistic placement logic (drift, variable sizing).

V2 of the mosaic simulator. Instead of a perfect pixel grid, tesserae have
slightly randomized sizes and accumulate positional drift, mimicking real
opus tessellatum construction.

Three placement modes:
    drift   — Opus Tessellatum (row-by-row with cumulative drift)
    contour — Opus Vermiculatum (contour-following worm-like chains)
    flow    — Opus Musivum (structure-tensor-driven whole-surface)
"""

import argparse
import os
import sys
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from mosaic import (
    load_tile_templates,
    generate_report,
    build_mosaic_drift,
    build_mosaic_contour,
    build_mosaic_flow,
)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate mosaic with realistic placement (V2)")
    parser.add_argument("--image", default="inputs/doodle_xs.jpg", help="Input image path")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: auto-named in output/)")
    parser.add_argument("--tiles-dir", default="tiles/raw",
                        help="Tile template directory")
    parser.add_argument("--mode", choices=["drift", "contour", "flow"], default="drift",
                        help="Placement mode (drift=opus tessellatum, contour=opus vermiculatum, flow=opus musivum)")
    parser.add_argument("--flow-direction", choices=["along", "across"], default="along",
                        help="Flow mode: streamlines follow edges (along) or cross them (across)")
    parser.add_argument("--tessera-size", type=int, default=30,
                        help="Base tessera size in output pixels")
    parser.add_argument("--grout-width", type=int, default=-1,
                        help="Grout width in pixels (0=touch, negative=overlap)")
    parser.add_argument("--grout-color", type=int, nargs=3, default=[40, 40, 40],
                        help="Grout RGB color")
    parser.add_argument("--color-variation", type=float, default=15.0,
                        help="Random color jitter range")
    parser.add_argument("--rotation-jitter", type=float, default=3.0,
                        help="Max random rotation in degrees")
    parser.add_argument("--size-jitter", type=float, default=0.15,
                        help="Tessera size variation as fraction (0.15 = +/-15%%)")
    parser.add_argument("--real-tessera-mm", type=float, default=10.0,
                        help="Real-world tessera size in mm")
    parser.add_argument("--real-grout-mm", type=float, default=1.0,
                        help="Real-world grout width in mm")
    parser.add_argument("--max-input-size", type=int, default=500,
                        help="Downsample input if larger than this")
    parser.add_argument("--drift-correction", type=int, default=0,
                        help="Correct drift every N pixels (0=never)")
    parser.add_argument("--edge-threshold", type=float, default=0.15,
                        help="Edge strength threshold for contour mode (0-1)")
    parser.add_argument("--fill-style", choices=["drift", "radial", "concentric"],
                        default="drift",
                        help="Background fill for contour mode (drift=rows, radial=radiating outward, concentric=parallel echoes)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Build output path: output/<input_basename>_<mode>_<timestamp>.png
    if args.output is None:
        os.makedirs("output", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.image))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("output", f"{base}_{args.mode}_{ts}.png")

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Load input image
    print(f"Loading {args.image}...")
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not load image: {args.image}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    print(f"Input size: {img_rgb.shape[1]} x {img_rgb.shape[0]} px")

    # Downsample if needed
    h, w = img_rgb.shape[:2]
    max_dim = max(h, w)
    if max_dim > args.max_input_size:
        scale = args.max_input_size / max_dim
        new_w, new_h = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Downsampled to: {new_w} x {new_h} px")

    # Load tile templates
    print(f"Loading tile templates from {args.tiles_dir}...")
    templates = load_tile_templates(args.tiles_dir, args.tessera_size)

    # Build mosaic
    print(f"Rendering mosaic (mode: {args.mode})...")
    if args.mode == "drift":
        mosaic, count = build_mosaic_drift(
            img_rgb, templates, args.tessera_size, args.grout_width,
            args.grout_color, args.color_variation, args.size_jitter,
            args.rotation_jitter, args.drift_correction, rng
        )
    elif args.mode == "contour":
        mosaic, count = build_mosaic_contour(
            img_rgb, templates, args.tessera_size, args.grout_width,
            args.grout_color, args.color_variation, args.size_jitter,
            args.rotation_jitter, args.edge_threshold, rng,
            fill_style=args.fill_style
        )
    elif args.mode == "flow":
        mosaic, count = build_mosaic_flow(
            img_rgb, templates, args.tessera_size, args.grout_width,
            args.grout_color, args.color_variation, args.size_jitter,
            args.rotation_jitter, rng,
            flow_direction=args.flow_direction
        )

    # Save output
    print(f"Saving output to {args.output}...")
    out_pil = Image.fromarray(mosaic)
    if args.output.lower().endswith((".jpg", ".jpeg")):
        out_pil.save(args.output, quality=95)
    else:
        out_pil.save(args.output)

    # Report
    report = generate_report(
        args.image, img_rgb.shape, mosaic.shape, count,
        args.tessera_size, args.grout_width,
        args.real_tessera_mm, args.real_grout_mm, args.mode
    )
    report_path = os.path.splitext(args.output)[0] + "_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\n{report}")
    print(f"\nReport saved to {report_path}")
    print("Done!")


if __name__ == "__main__":
    main()
