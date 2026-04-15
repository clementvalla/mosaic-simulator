#!/usr/bin/env python3
"""Mosaic Simulator GUI — Dear PyGui interface with lil-gui-style controls."""

import os
import threading
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import dearpygui.dearpygui as dpg

from mosaic import (
    load_tile_templates,
    generate_report,
    build_mosaic_drift,
    build_mosaic_contour,
    build_mosaic_flow,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(SCRIPT_DIR, "tiles", "raw")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
DEFAULT_IMAGE = os.path.join(SCRIPT_DIR, "inputs", "doodle_xs.jpg")

MODES = ["drift", "contour", "flow"]
MODE_LABELS = ["Opus Tessellatum (grid)", "Opus Vermiculatum (contour)", "Opus Musivum (flow)"]

# ── State ──

input_image = None      # numpy RGB float32
output_image = None     # numpy RGB uint8
showing_output = False
busy = False

_tex_counter = 0        # incremented each time we need a new texture size
_cur_tex_tag = None     # tag of the currently active texture
_tex_w = 0
_tex_h = 0


def load_image_file(path):
    """Load an image file, return RGB uint8 numpy array or None."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _img_to_rgba_flat(img_rgb_uint8):
    """Convert RGB uint8 image to flat RGBA float32 array for DPG."""
    rgba = np.dstack([img_rgb_uint8, np.full(img_rgb_uint8.shape[:2], 255, dtype=np.uint8)])
    return (rgba.astype(np.float32) / 255.0).ravel()


def update_texture(img_rgb_uint8):
    """Push a numpy RGB uint8 image into the DPG dynamic texture."""
    global _tex_counter, _cur_tex_tag, _tex_w, _tex_h
    h, w = img_rgb_uint8.shape[:2]
    flat = _img_to_rgba_flat(img_rgb_uint8)
    if _tex_w != w or _tex_h != h:
        # Need a new texture at this size
        _tex_counter += 1
        new_tag = f"_canvas_tex_{_tex_counter}"
        dpg.add_dynamic_texture(w, h, flat, tag=new_tag, parent="tex_registry")
        # Point the image widget at the new texture
        if dpg.does_item_exist("canvas_image"):
            dpg.configure_item("canvas_image", texture_tag=new_tag)
        # Clean up old texture
        if _cur_tex_tag and dpg.does_item_exist(_cur_tex_tag):
            dpg.delete_item(_cur_tex_tag)
        _cur_tex_tag = new_tag
        _tex_w, _tex_h = w, h
        # Re-letterbox for the new dimensions
        try:
            resize_callback()
        except Exception:
            pass
    else:
        dpg.set_value(_cur_tex_tag, flat)


def show_input():
    global showing_output
    if input_image is not None:
        showing_output = False
        update_texture((input_image.clip(0, 255)).astype(np.uint8))


def show_output():
    global showing_output
    if output_image is not None:
        showing_output = True
        update_texture(output_image)


def get_mode():
    idx = dpg.get_value("mode_combo")
    # DPG combo returns the string value, not index
    if isinstance(idx, str):
        return MODES[MODE_LABELS.index(idx)] if idx in MODE_LABELS else "drift"
    return MODES[idx]


def set_status(text):
    dpg.set_value("status_text", text)


def on_file_selected(sender, app_data):
    """Callback when user picks a file from the file dialog."""
    global input_image
    selections = app_data.get("selections", {})
    if selections:
        path = list(selections.values())[0]
    else:
        path = app_data.get("file_path_name", "")
    if not path:
        return
    img = load_image_file(path)
    if img is None:
        set_status(f"Failed to load: {os.path.basename(path)}")
        return
    input_image = img.astype(np.float32)
    update_texture(img)
    set_status(f"Loaded: {os.path.basename(path)} ({img.shape[1]}x{img.shape[0]})")


def _get_fill_style():
    val = dpg.get_value("fill_style")
    if isinstance(val, str):
        return val
    return ["drift", "radial", "concentric"][val]


def _get_flow_direction():
    val = dpg.get_value("flow_direction")
    if isinstance(val, str):
        return val
    return ["along", "across"][val]


def run_generate(preview=False):
    """Run mosaic generation (called in a thread)."""
    global output_image, busy
    if input_image is None:
        set_status("No image loaded.")
        return
    if busy:
        return
    busy = True
    label = "Preview" if preview else "Render"
    set_status(f"{label}ing...")

    try:
        img_rgb = input_image.copy()
        mode_key = get_mode()
        tessera_size = int(dpg.get_value("tessera_size"))
        grout_width = int(dpg.get_value("grout_width"))
        size_jitter = dpg.get_value("size_jitter")
        rotation_jitter = dpg.get_value("rotation_jitter")
        color_variation = dpg.get_value("color_variation")
        preview_max_size = int(dpg.get_value("preview_max_size"))
        max_input_size = int(dpg.get_value("max_input_size"))
        seed_val = int(dpg.get_value("seed_val"))
        drift_correction = int(dpg.get_value("drift_correction"))
        edge_threshold = dpg.get_value("edge_threshold")
        fill_style = _get_fill_style()
        flow_direction = _get_flow_direction()

        effective_max = preview_max_size if preview else max_input_size
        h, w = img_rgb.shape[:2]
        if max(h, w) > effective_max:
            scale = effective_max / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        templates = None if preview else load_tile_templates(TILES_DIR, tessera_size)
        rng = np.random.default_rng(seed_val if seed_val != 0 else None)
        grout_color = [40, 40, 40]

        if mode_key == "drift":
            mosaic, count = build_mosaic_drift(
                img_rgb, templates, tessera_size, grout_width, grout_color,
                color_variation, size_jitter, rotation_jitter, drift_correction, rng,
                preview=preview,
            )
        elif mode_key == "contour":
            mosaic, count = build_mosaic_contour(
                img_rgb, templates, tessera_size, grout_width, grout_color,
                color_variation, size_jitter, rotation_jitter, edge_threshold, rng,
                fill_style=fill_style, preview=preview,
            )
        elif mode_key == "flow":
            mosaic, count = build_mosaic_flow(
                img_rgb, templates, tessera_size, grout_width, grout_color,
                color_variation, size_jitter, rotation_jitter, rng,
                flow_direction=flow_direction, preview=preview,
            )

        output_image = mosaic.astype(np.uint8)
        update_texture(output_image)

        saved_path = None
        if not preview:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"gui_{mode_key}_{ts}.png"
            saved_path = os.path.join(OUTPUT_DIR, out_name)
            Image.fromarray(output_image).save(saved_path)

        report = generate_report(
            "gui_input", img_rgb.shape, mosaic.shape, count,
            tessera_size, grout_width, 10.0, 1.0, mode_key,
        )
        if saved_path:
            report += f"\nSaved to: {saved_path}"

        dpg.set_value("report_text", report)
        tag = "[preview]" if preview else "[saved]"
        set_status(f"{tag} {count} tesserae, {mosaic.shape[1]}x{mosaic.shape[0]}px")

    except Exception as e:
        set_status(f"Error: {e}")
    finally:
        busy = False


def on_preview():
    threading.Thread(target=run_generate, args=(True,), daemon=True).start()


def on_render():
    threading.Thread(target=run_generate, args=(False,), daemon=True).start()


def on_mode_change():
    """Show/hide mode-specific controls."""
    mode = get_mode()
    dpg.configure_item("grp_drift", show=(mode == "drift"))
    dpg.configure_item("grp_contour", show=(mode == "contour"))
    dpg.configure_item("grp_flow", show=(mode == "flow"))


# ── Build UI ──

dpg.create_context()

# Texture registry — initial 1x1 placeholder
with dpg.texture_registry(tag="tex_registry"):
    _cur_tex_tag = "_canvas_tex_0"
    dpg.add_dynamic_texture(1, 1, [0.11, 0.11, 0.11, 1.0], tag=_cur_tex_tag)
    _tex_w, _tex_h = 1, 1

# File dialog
with dpg.file_dialog(
    directory_selector=False, show=False, tag="file_dialog",
    callback=on_file_selected, width=600, height=400,
):
    dpg.add_file_extension(".jpg")
    dpg.add_file_extension(".jpeg")
    dpg.add_file_extension(".png")
    dpg.add_file_extension(".bmp")
    dpg.add_file_extension(".tif")
    dpg.add_file_extension(".tiff")
    dpg.add_file_extension(".*")

# Theme: dark, compact
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 6, 4)
        dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 4, 2)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 6, 2)
        dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing, 4, 2)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 2)
        dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize, 8)
        dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 2)
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (26, 26, 26, 240))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBg, (30, 30, 30))
        dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, (40, 50, 60))
        dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 55, 70))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (50, 70, 90))
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (50, 80, 110))
        dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 60, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (60, 80, 100))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (88, 136, 170))
        dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, (110, 170, 210))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (42, 42, 42))
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (55, 55, 55))
        dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (88, 170, 136))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (220, 220, 220))

# Green render button theme
with dpg.theme() as render_theme:
    with dpg.theme_component(dpg.mvButton):
        dpg.add_theme_color(dpg.mvThemeCol_Button, (40, 120, 70))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (50, 150, 85))
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (60, 170, 95))

dpg.bind_theme(global_theme)

# ── Primary window (full-bleed canvas) ──
with dpg.window(tag="primary_window"):
    dpg.add_image(_cur_tex_tag, tag="canvas_image")

# ── Floating control panel ──
PANEL_W = 320

with dpg.window(
    label="Mosaic Sim",
    tag="control_panel",
    width=PANEL_W, height=600,
    pos=[10, 10],
    no_close=True,
    no_collapse=False,
    no_resize=False,
    no_scrollbar=False,
):
    # Actions
    dpg.add_button(label="Load Image", callback=lambda: dpg.show_item("file_dialog"), width=-1)
    dpg.add_spacer(height=2)

    with dpg.group(horizontal=True):
        dpg.add_button(label="Preview", callback=on_preview, width=PANEL_W // 2 - 14)
        btn_render = dpg.add_button(label="Render", callback=on_render, width=-1)
        dpg.bind_item_theme(btn_render, render_theme)

    dpg.add_spacer(height=2)
    with dpg.group(horizontal=True):
        dpg.add_button(label="Show Input", callback=lambda: show_input(), width=PANEL_W // 2 - 14)
        dpg.add_button(label="Show Output", callback=lambda: show_output(), width=-1)

    dpg.add_spacer(height=4)
    dpg.add_text("Ready", tag="status_text", color=(136, 204, 255))
    dpg.add_separator()

    # Mode
    dpg.add_combo(MODE_LABELS, default_value=MODE_LABELS[0], tag="mode_combo",
                  label="Mode", callback=lambda: on_mode_change(), width=-90)
    dpg.add_spacer(height=2)

    # Tessera
    if dpg.add_collapsing_header(label="Tessera", default_open=True):
        dpg.add_slider_int(tag="tessera_size", label="Size", default_value=30,
                           min_value=5, max_value=100, width=-90)
        dpg.add_slider_int(tag="grout_width", label="Grout", default_value=-1,
                           min_value=-10, max_value=20, width=-90)
        dpg.add_slider_float(tag="size_jitter", label="Size Jitter", default_value=0.15,
                             min_value=0.0, max_value=0.5, format="%.2f", width=-90)
        dpg.add_slider_float(tag="rotation_jitter", label="Rotation", default_value=3.0,
                             min_value=0.0, max_value=30.0, format="%.1f", width=-90)

    # Color
    if dpg.add_collapsing_header(label="Color", default_open=True):
        dpg.add_slider_float(tag="color_variation", label="Variation", default_value=15.0,
                             min_value=0.0, max_value=50.0, format="%.0f", width=-90)

    # Scale
    if dpg.add_collapsing_header(label="Scale", default_open=True):
        dpg.add_slider_int(tag="preview_max_size", label="Preview px", default_value=100,
                           min_value=40, max_value=300, width=-90)
        dpg.add_slider_int(tag="max_input_size", label="Render px", default_value=500,
                           min_value=100, max_value=2000, width=-90)
        dpg.add_input_int(tag="seed_val", label="Seed (0=rand)", default_value=0,
                          step=1, width=-90)

    # Mode-specific options
    if dpg.add_collapsing_header(label="Mode Options", default_open=True):
        with dpg.group(tag="grp_drift", show=True):
            dpg.add_slider_int(tag="drift_correction", label="Drift Fix", default_value=0,
                               min_value=0, max_value=200, width=-90)
        with dpg.group(tag="grp_contour", show=False):
            dpg.add_slider_float(tag="edge_threshold", label="Edge Thresh", default_value=0.15,
                                 min_value=0.0, max_value=1.0, format="%.2f", width=-90)
            dpg.add_combo(["drift", "radial", "concentric"], default_value="drift",
                          tag="fill_style", label="Fill Style", width=-90)
        with dpg.group(tag="grp_flow", show=False):
            dpg.add_combo(["along", "across"], default_value="along",
                          tag="flow_direction", label="Flow Dir", width=-90)

    # Report
    if dpg.add_collapsing_header(label="Report", default_open=False):
        dpg.add_input_text(tag="report_text", multiline=True, readonly=True,
                           height=120, width=-1)

# ── Viewport setup ──

dpg.create_viewport(title="Mosaic Simulator", width=1200, height=800)
dpg.setup_dearpygui()

# Load default image
if os.path.exists(DEFAULT_IMAGE):
    _img = load_image_file(DEFAULT_IMAGE)
    if _img is not None:
        input_image = _img.astype(np.float32)
        update_texture(_img)
        set_status(f"Loaded: plant1.jpg ({_img.shape[1]}x{_img.shape[0]})")

dpg.set_primary_window("primary_window", True)


def resize_callback():
    """Letterbox the canvas image to fill viewport without distortion."""
    vw = dpg.get_viewport_client_width()
    vh = dpg.get_viewport_client_height()
    if _tex_w <= 1 or _tex_h <= 1:
        dpg.configure_item("canvas_image", width=vw, height=vh)
        return
    # Fit image inside viewport preserving aspect ratio
    scale = min(vw / _tex_w, vh / _tex_h)
    iw = int(_tex_w * scale)
    ih = int(_tex_h * scale)
    # Center with offset
    ox = (vw - iw) // 2
    oy = (vh - ih) // 2
    dpg.configure_item("canvas_image", width=iw, height=ih, pos=[ox, oy])


dpg.set_viewport_resize_callback(lambda s, d: resize_callback())

dpg.show_viewport()
resize_callback()
dpg.start_dearpygui()
dpg.destroy_context()
