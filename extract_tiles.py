#!/usr/bin/env python3
"""Extract individual mosaic tesserae from a photograph with alpha masks."""

import argparse
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image


def load_and_preprocess(image_path):
    """Load image and enhance contrast for grout detection."""
    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE to boost low-contrast grout lines
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return img, gray, enhanced


def segment_tiles(enhanced, min_area=100, max_area=2000, block_size=25, c_value=3):
    """Detect grout lines and segment individual tiles via connected components."""
    # Adaptive threshold: tiles are bright, grout is dark
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, blockSize=block_size, C=c_value
    )

    # Use connected components to find tile regions directly
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    # Filter by area — skip label 0 (background)
    tile_masks = []
    for i in range(1, n_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            # Extract contour from this component's mask
            component_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                tile_masks.append(contours[0])

    return tile_masks, thresh


def extract_tile_images(img_bgr, contours, pad=0):
    """Extract each tile as RGBA with alpha mask from contour.

    pad: pixels to dilate each mask outward to recapture tile edges
    that the threshold classified as grout.
    """
    h_img, w_img = img_bgr.shape[:2]
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad*2+1, pad*2+1))
    tile_images = []
    for cnt in contours:
        # Create mask for this contour and dilate to recover clipped edges
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        # Recompute bounding rect after dilation
        x, y, w, h = cv2.boundingRect(mask)
        # Clamp to image bounds
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, w_img), min(y + h, h_img)
        # Crop region
        tile_bgr = img_bgr[y0:y1, x0:x1]
        tile_mask = mask[y0:y1, x0:x1]
        # Convert BGR -> RGB and combine with alpha
        tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
        tile_rgba = np.dstack([tile_rgb, tile_mask])
        tile_images.append(tile_rgba)
    return tile_images


def make_preview(tile_images, output_path, max_tiles=120, cell_size=50, cols=12):
    """Create a contact sheet of sample tiles on a dark background."""
    samples = tile_images[:max_tiles]
    rows = (len(samples) + cols - 1) // cols
    sheet_w = cols * cell_size + (cols + 1) * 4
    sheet_h = rows * cell_size + (rows + 1) * 4
    sheet = np.full((sheet_h, sheet_w, 3), 40, dtype=np.uint8)  # dark gray bg

    for i, tile in enumerate(samples):
        r, c = divmod(i, cols)
        # Resize tile to fit cell, preserving aspect ratio
        h, w = tile.shape[:2]
        scale = min(cell_size / w, cell_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_tile = Image.fromarray(tile)
        pil_tile = pil_tile.resize((new_w, new_h), Image.LANCZOS)

        # Composite onto dark background
        bg = Image.new("RGBA", (cell_size, cell_size), (40, 40, 40, 255))
        offset_x = (cell_size - new_w) // 2
        offset_y = (cell_size - new_h) // 2
        bg.paste(pil_tile, (offset_x, offset_y), pil_tile)

        # Place in sheet
        px = c * (cell_size + 4) + 4
        py = r * (cell_size + 4) + 4
        cell_rgb = np.array(bg.convert("RGB"))
        sheet[py:py+cell_size, px:px+cell_size] = cell_rgb

    cv2.imwrite(output_path, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"Preview saved to {output_path} ({len(samples)} tiles shown)")


def save_raw_tiles(tile_images, output_dir):
    """Save each tile as a PNG with alpha transparency."""
    os.makedirs(output_dir, exist_ok=True)
    for i, tile in enumerate(tile_images):
        pil_img = Image.fromarray(tile)
        pil_img.save(os.path.join(output_dir, f"tile_{i:04d}.png"))
    print(f"Saved {len(tile_images)} raw tiles to {output_dir}")


def augment_tile(tile_rgba):
    """Generate 5 augmented variants of a tile."""
    pil = Image.fromarray(tile_rgba)
    augmented = []

    # 1. Horizontal flip
    augmented.append(np.array(pil.transpose(Image.FLIP_LEFT_RIGHT)))

    # 2. Vertical flip
    augmented.append(np.array(pil.transpose(Image.FLIP_TOP_BOTTOM)))

    # 3. Rotate +15°
    rot_pos = pil.rotate(-15, expand=True, resample=Image.BICUBIC)
    augmented.append(np.array(rot_pos))

    # 4. Rotate -15°
    rot_neg = pil.rotate(15, expand=True, resample=Image.BICUBIC)
    augmented.append(np.array(rot_neg))

    # 5. Brightness/contrast jitter
    arr = tile_rgba.copy().astype(np.float32)
    brightness = random.uniform(-20, 20)
    contrast = random.uniform(0.85, 1.15)
    arr[:, :, :3] = np.clip(arr[:, :, :3] * contrast + brightness, 0, 255)
    augmented.append(arr.astype(np.uint8))

    return augmented


def save_augmented_tiles(tile_images, output_dir):
    """Generate and save augmented tiles."""
    os.makedirs(output_dir, exist_ok=True)
    total = 0
    for i, tile in enumerate(tile_images):
        variants = augment_tile(tile)
        for j, var in enumerate(variants):
            pil_img = Image.fromarray(var)
            pil_img.save(os.path.join(output_dir, f"tile_{i:04d}_aug_{j}.png"))
            total += 1
    print(f"Saved {total} augmented tiles to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract mosaic tesserae")
    parser.add_argument("--image", default="images/crop.jpg", help="Input image path")
    parser.add_argument("--preview-only", action="store_true", help="Only generate preview")
    parser.add_argument("--min-area", type=int, default=100, help="Min tile area in px²")
    parser.add_argument("--max-area", type=int, default=2000, help="Max tile area in px²")
    parser.add_argument("--block-size", type=int, default=25, help="Adaptive threshold block size")
    parser.add_argument("--c-value", type=int, default=3, help="Adaptive threshold C constant")
    parser.add_argument("--output", default="tiles", help="Output directory")
    args = parser.parse_args()

    print(f"Loading {args.image}...")
    img, gray, enhanced = load_and_preprocess(args.image)
    print(f"Image size: {img.shape[1]}x{img.shape[0]}")

    print("Segmenting tiles...")
    contours, mask = segment_tiles(enhanced, args.min_area, args.max_area, args.block_size, args.c_value)
    print(f"Found {len(contours)} tile candidates")

    if len(contours) == 0:
        # Save debug mask
        cv2.imwrite(os.path.join(args.output, "debug_mask.png"), mask)
        sys.exit("No tiles found! Check debug_mask.png and adjust parameters.")

    # Sort tiles top-to-bottom, left-to-right
    contours.sort(key=lambda c: (cv2.boundingRect(c)[1] // 20, cv2.boundingRect(c)[0]))

    # Area stats
    areas = [cv2.contourArea(c) for c in contours]
    print(f"Area stats: min={min(areas):.0f}, max={max(areas):.0f}, "
          f"mean={np.mean(areas):.0f}, median={np.median(areas):.0f}")

    print("Extracting tile images...")
    tile_images = extract_tile_images(img, contours)

    os.makedirs(args.output, exist_ok=True)

    # Always save preview
    preview_path = os.path.join(args.output, "preview.png")
    make_preview(tile_images, preview_path)

    # Save debug: threshold mask and contour overlay
    debug_overlay = img.copy()
    cv2.drawContours(debug_overlay, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(args.output, "debug_contours.png"), debug_overlay)
    cv2.imwrite(os.path.join(args.output, "debug_mask.png"), mask)
    print("Debug images saved (debug_contours.png, debug_mask.png)")

    if args.preview_only:
        print("\n--preview-only mode: stopping here. Inspect tiles/preview.png")
        return

    # Full extraction
    raw_dir = os.path.join(args.output, "raw")
    save_raw_tiles(tile_images, raw_dir)

    aug_dir = os.path.join(args.output, "augmented")
    save_augmented_tiles(tile_images, aug_dir)

    print(f"\nDone! {len(tile_images)} raw + {len(tile_images)*5} augmented = "
          f"{len(tile_images)*6} total tile assets")


if __name__ == "__main__":
    main()
