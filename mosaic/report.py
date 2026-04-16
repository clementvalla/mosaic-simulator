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

    lines = [
        "Mosaic Simulation Report (V2)",
        "=" * 40,
        f"Mode: {mode}",
        f"Input image: {input_path}",
        f"Mosaic grid: {in_w} x {in_h} tesserae",
        f"Output dimensions: {out_w} x {out_h} px",
        f"",
        f"Total tesserae: {tessera_count:,}",
        f"Tessera size: {tessera_size} px (output)",
        f"Grout width: {grout_width} px (output)",
        f"",
        f"Real-world estimates:",
        f"  Tessera size: {real_tessera_mm:.1f} mm",
        f"  Grout width: {real_grout_mm:.1f} mm",
        f"  Mosaic width:  {real_w_mm:.1f} mm ({real_w_mm / 10:.1f} cm, {real_w_mm / 1000:.2f} m)",
        f"  Mosaic height: {real_h_mm:.1f} mm ({real_h_mm / 10:.1f} cm, {real_h_mm / 1000:.2f} m)",
        f"  Mosaic area: {real_w_mm * real_h_mm / 1e6:.2f} m²",
    ]
    return "\n".join(lines)
