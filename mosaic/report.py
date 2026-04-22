"""Text report generation for mosaic simulations."""


def generate_report(input_path, input_shape, output_shape, tessera_count,
                    tessera_size, grout_width, real_tessera_mm, real_grout_mm,
                    mode="drift"):
    """Generate a text report with mosaic dimensions and real-world estimates."""
    in_h, in_w = input_shape[:2]
    out_h, out_w = output_shape[:2]

    real_cell_mm = real_tessera_mm + real_grout_mm
    real_w_mm = in_w * real_cell_mm + real_grout_mm
    real_h_mm = in_h * real_cell_mm + real_grout_mm
    real_tessera_in = real_tessera_mm / 25.4

    lines = [
        "Mosaic Simulation Report",
        "=" * 40,
        f"Mode: {mode}",
        f"Input image: {input_path}",
        f"Mosaic grid: {in_w} x {in_h} tesserae",
        f"Total tesserae: {tessera_count:,}",
        f"",
        f"Tessera size: {tessera_size} px, {real_tessera_mm:.0f}mm, {real_tessera_in:.2f} inches (output)",
        f"Output dimensions px: {out_w} x {out_h} px",
        f"Output dimensions mm: {real_w_mm:.0f} x {real_h_mm:.0f} mm",
        f"Output dimensions cm: {real_w_mm / 10:.1f} x {real_h_mm / 10:.1f} cm",
        f"Output dimensions in: {real_w_mm / 25.4:.1f} x {real_h_mm / 25.4:.1f} in",
    ]
    return "\n".join(lines)
