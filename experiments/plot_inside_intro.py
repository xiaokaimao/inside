"""PDF-only INSIDE overview, adapted from the supplied two-panel schematic.

Panel (a) uses exact four-player, size-two coalition directions. Panel (b)
is a conceptual illustration: six independent random directions on a sphere
are contrasted with the balanced directions of the six-coalition toy design.
The continuous IID illustration is not a draw from the discrete n=4 slice.
Both use unit 3D vectors and the same orthographic camera. The horizontal
ellipse and soft radial gray sphere shading are visual depth cues.

Run from the repository root: python -m experiments.plot_inside_intro
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.tri import Triangulation
import numpy as np

from experiments.result_paths import by_format
from frame_ofa.design import per_size_frame_coupled_design
from frame_ofa.geometry import centered_directions, efficiency_projector


WIDTH_MM, HEIGHT_MM = 180.0, 58.0
RADIUS_MM = 16.5
CENTERS_MM = ((74.0, 26.0), (117.0, 26.0), (160.0, 26.0))
NAVY = "#0C2458"
BLUE = "#0F4D92"
TEAL = "#42949E"
RED = "#B64342"
DARK = "#272727"
GRAY = "#767676"
LIGHT = "#CFCECE"
# Fixed only for reproducibility; generate six directions once, without
# rejecting or selecting batches for clustering or a desired discrepancy.
DEFAULT_IID_SEED = 0

# Columns are an orthonormal basis of 1-perp, in R^4. These are the
# directions of {1,2}, {1,3}, and {1,4}, respectively.
ZERO_SUM_BASIS = np.array(
    [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
    dtype=float,
) / 2

# Camera rows are screen-right, screen-up, and depth, with positive depth
# facing the reader. Opposite directions stay opposite under this projection.
CAMERA = np.array([[0, -0.8, 0.6], [0.8, 0.36, 0.48], [0.6, -0.48, -0.64]])


def apply_publication_style() -> None:
    """Use figures4papers colors and physical-size publication typography."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.0,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "text.color": DARK,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def projected_directions(coalitions: np.ndarray) -> np.ndarray:
    """Return true normalized directions in the common camera coordinates."""
    return centered_directions(coalitions) @ ZERO_SUM_BASIS @ CAMERA.T


def moment_errors(coalitions: np.ndarray) -> dict[str, float]:
    directions = centered_directions(coalitions)
    target = efficiency_projector(4) / 3
    return {
        "first_moment_norm": float(np.linalg.norm(directions.mean(axis=0))),
        "second_moment_frobenius": float(
            np.linalg.norm(directions.T @ directions / len(directions) - target)
        ),
    }


def build_example(iid_seed: int = DEFAULT_IID_SEED, design_seed: int = 0) -> dict:
    """Build exact coalition geometry and a six-direction IID schematic."""
    subsets = list(combinations(range(4), 2))
    legal = np.zeros((6, 4), dtype=bool)
    for row, subset in zip(legal, subsets):
        row[list(subset)] = True
    iid_vectors = np.random.default_rng(iid_seed).normal(size=(6, 3))
    iid_vectors /= np.linalg.norm(iid_vectors, axis=1, keepdims=True)
    iid_coords = iid_vectors @ CAMERA.T
    design = per_size_frame_coupled_design(
        num_players=4,
        num_samples=6,
        seed=design_seed,
        candidate_pool=64,
        mean_balance=1 / 16,
    )
    inside = design.coalitions
    lookup = {tuple(row): i for i, row in enumerate(legal)}
    inside_indices = np.array([lookup[tuple(row)] for row in inside])
    coords = projected_directions(legal)

    # Check scientific identities in R^4 and R^3 before projecting to the page.
    # Two-dimensional projected lengths/angles need not equal their 3D values.
    np.testing.assert_allclose(ZERO_SUM_BASIS.T @ ZERO_SUM_BASIS, np.eye(3))
    np.testing.assert_allclose(ZERO_SUM_BASIS.sum(axis=0), 0)
    np.testing.assert_allclose(CAMERA @ CAMERA.T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(coords, axis=1), 1, atol=1e-14)
    np.testing.assert_allclose(coords[1], -coords[4], atol=1e-14)
    np.testing.assert_allclose(
        coords @ coords.T,
        centered_directions(legal) @ centered_directions(legal).T,
        atol=1e-14,
    )
    np.testing.assert_allclose(np.linalg.norm(iid_coords, axis=1), 1, atol=1e-14)
    if len(iid_coords) != len(inside_indices) or len(set(inside_indices)) != 6:
        raise ValueError("The toy comparison requires six samples and full design coverage")
    np.testing.assert_allclose(centered_directions(inside).mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(
        centered_directions(inside).T @ centered_directions(inside) / 6,
        efficiency_projector(4) / 3,
        atol=1e-14,
    )
    return {
        "legal": legal,
        "coords": coords,
        "iid_vectors": iid_vectors,
        "iid_coords": iid_coords,
        "inside_indices": inside_indices,
        "iid_seed": iid_seed,
        "design_seed": design_seed,
        "iid_sphere_moments": {
            "first_moment_norm": float(np.linalg.norm(iid_vectors.mean(axis=0))),
            "second_moment_frobenius": float(np.linalg.norm(
                iid_vectors.T @ iid_vectors / 6 - np.eye(3) / 3)),
        },
        "inside_moments": moment_errors(inside),
    }


def sphere_axes(fig, center: tuple[float, float], name: str):
    half_size = 19.0
    x, y = center
    ax = fig.add_axes(
        [(x - half_size) / WIDTH_MM, (y - half_size) / HEIGHT_MM,
         2 * half_size / WIDTH_MM, 2 * half_size / HEIGHT_MM],
        label=name,
    )
    limit = half_size / RADIUS_MM
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
    ax.set_axis_off()
    # Restore the supplied diagram's radial gradient: center (34%, 26%),
    # radius 72% of the diameter, and white / #E8E8E8 / #CFCFCF / #9B9B9B
    # stops. Screen y points upward, hence the highlight at (-0.32, +0.48).
    # Gouraud interpolation preserves the vector-only PDF output.
    rings, sectors = 48, 96
    polar = np.linspace(0, np.pi / 2, rings + 1)[1:, None]
    azimuth = np.linspace(0, 2 * np.pi, sectors, endpoint=False)[None, :]
    normals = np.column_stack((
        (np.sin(polar) * np.cos(azimuth)).ravel(),
        (np.sin(polar) * np.sin(azimuth)).ravel(),
        np.broadcast_to(np.cos(polar), (rings, sectors)).ravel(),
    ))
    normals = np.vstack(([0.0, 0.0, 1.0], normals))
    triangles = [[0, 1 + i, 1 + (i + 1) % sectors] for i in range(sectors)]
    for ring in range(rings - 1):
        for i in range(sectors):
            a = 1 + ring * sectors + i
            b = 1 + ring * sectors + (i + 1) % sectors
            triangles.extend(((a, a + sectors, b + sectors), (a, b + sectors, b)))
    radial_distance = np.linalg.norm(normals[:, :2] - [-0.32, 0.48], axis=1) / 1.44
    brightness = np.interp(radial_distance, [0, 0.46, 0.78, 1],
                           np.array([255, 232, 207, 155]) / 255)
    mesh = Triangulation(normals[:, 0], normals[:, 1], triangles)
    ax.tripcolor(mesh, brightness, shading="gouraud", cmap="gray",
                 vmin=0, vmax=1, rasterized=False, zorder=0)
    ax.add_patch(Circle((0, 0), 1, facecolor="none", edgecolor="#C8C8C8",
                        linewidth=0.4, zorder=1))
    # Inset, slightly lowered horizontal ellipse, matching the supplied
    # diagram's proportions. A decorative depth cue, not a coordinate axis.
    theta = np.linspace(0, 2 * np.pi, 361)
    ax.plot(0.82 * np.cos(theta), -0.094 + 0.24 * np.sin(theta),
            color="#A8A8A8", alpha=0.62, linewidth=0.48,
            linestyle=(0, (1.8, 2.3)), zorder=1)
    return ax


def draw_directions(ax, coords, colors: list[str]) -> None:
    """Draw each supplied unit direction once, without multiplicity labels."""
    for i in np.argsort(coords[:, 2]):
        x, y, depth = coords[i]
        color = colors[i]
        ax.add_patch(FancyArrowPatch(
            (0, 0), (x, y), arrowstyle="-|>", mutation_scale=7.5,
            shrinkA=0, shrinkB=1.5, linewidth=1.1, color=color,
            linestyle="-" if depth >= 0 else (0, (2.4, 1.7)), zorder=4,
        ))
        ax.add_patch(Circle((x, y), 0.039, facecolor=color, edgecolor="white",
                            linewidth=0.5, zorder=5))
    ax.add_patch(Circle((0, 0), 0.022, facecolor=GRAY, edgecolor="none", zorder=6))


def build_figure(example: dict):
    apply_publication_style()
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    page = fig.add_axes([0, 0, 1, 1], label="page_annotations")
    page.set(xlim=(0, WIDTH_MM), ylim=(0, HEIGHT_MM))
    page.set_axis_off()

    page.text(4, 53, "(a) Coalitions → unit directions",
              fontsize=9.0, fontweight="bold", va="center")
    page.text(99, 53, "(b) Within-size sampling",
              fontsize=9.0, fontweight="bold", va="center")
    page.plot([95, 95], [3.5, 54.5], color="#E5E8EC", linewidth=0.7)

    examples = [(0, "1,2", RED), (1, "1,3", TEAL), (4, "2,4", NAVY)]
    for number, ((index, members, color), y) in enumerate(zip(examples, (39, 25.5, 12)), 1):
        page.text(4, y + 5.6, rf"$S_{number}=\{{{members}\}}$", color=color,
                  fontsize=8.6, va="center")
        row = example["legal"][index]
        for player, selected in enumerate(row):
            x = 6.0 + player * 5.2
            page.add_patch(Circle((x, y), 1.75, facecolor=NAVY if selected else "white",
                                  edgecolor=NAVY if selected else "#969CA5", linewidth=0.75))
            page.text(x, y, str(player + 1), ha="center", va="center", fontsize=6.6,
                      color="white" if selected else DARK)
        vector = ",".join(str(int(value)) for value in row)
        page.text(26, y, rf"$z_S=({vector})$", fontsize=8.2, va="center")
        page.add_patch(FancyArrowPatch((50.5, y), (56.7, y), arrowstyle="-|>",
                                      mutation_scale=7, color=color, linewidth=0.85))

    axes = [sphere_axes(fig, center, name) for center, name in
            zip(CENTERS_MM, ("map", "iid", "inside"))]
    coords = example["coords"]
    draw_directions(axes[0], coords[[0, 1, 4]], [color for _, _, color in examples])
    label_offsets = {0: (-0.08, -0.14, "right"), 1: (0.15, 0.16, "left"),
                     4: (-0.20, -0.22, "right")}
    for number, (index, _, color) in enumerate(examples, 1):
        dx, dy, align = label_offsets[index]
        x, y = coords[index, :2]
        axes[0].text(x + dx, y + dy, rf"$u_{number}$", fontsize=8.8,
                     ha=align, va="center", color=color, zorder=8)

    page.text(74, 5.5, "Center + normalize", ha="center", va="center",
              fontsize=7.2, color=GRAY)

    draw_directions(axes[1], example["iid_coords"], ["#60646A"] * 6)
    draw_directions(axes[2], coords[example["inside_indices"]], [BLUE] * 6)
    for x, title, color in (
        (117, "IID", "#60646A"),
        (160, "INSIDE", BLUE),
    ):
        page.text(x, 46, title, ha="center", fontsize=8.5, fontweight="bold", color=color)
    page.text(117, 5.5, "Random directions", fontsize=7.7, ha="center", va="center", color=GRAY)
    page.text(160, 5.5, "Balanced directions", fontsize=7.7,
              ha="center", va="center", color=BLUE)
    return fig, axes


def export(output: Path, iid_seed: int = DEFAULT_IID_SEED, design_seed: int = 0,
           audit_alignment: bool = False, metadata: Path | None = None) -> Path:
    if output.suffix.lower() != ".pdf":
        raise ValueError("This script exports PDF only; use a .pdf output path")
    example = build_example(iid_seed, design_seed)
    fig, axes = build_figure(example)
    try:
        fig.canvas.draw()
        if audit_alignment:
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(
                fig, axes=axes, panel_ids=["map", "iid", "inside"],
                row_groups=[["map", "iid", "inside"]],
                json_out=by_format(output.with_suffix(".alignment.json")),
                overlay_svg=None, strict=True, tolerance_pt=1.5,
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf", dpi=300,
                    metadata={"Title": "INSIDE: fixed-size coalition design",
                              "Subject": "Coalition mapping and conceptual random-versus-balanced directions; not an error benchmark",
                              "Creator": "Matplotlib"})
    finally:
        plt.close(fig)
    if metadata is not None:
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text(json.dumps({
            "archetype": "two-panel method schematic",
            "source_category": "structural adaptation of supplied schematic",
            "claim": "Within-size coalition design balances directions and improves moment coverage",
            "interpretation": "Panel a is an exact n=4 s=2 map; panel b is a conceptual continuous-sphere comparison, not literal IID sampling from the n=4 coalition slice or an error benchmark",
            "dimensions_mm": [WIDTH_MM, HEIGHT_MM],
            "vertical_layout": "Compact 58 mm canvas; sphere-to-bottom-caption center gap 4 mm; sphere-to-method-title baseline gap 3.5 mm",
            "n": 4, "s": 2, "samples_per_batch": 6,
            "iid_seed": iid_seed, "design_seed": design_seed,
            "iid_selection": "One batch of six IID standard-normal 3D vectors, normalized to the sphere; no rejection, seed search, duplicates or manual clustering",
            "design_function": "per_size_frame_coupled_design",
            "candidate_pool": 64, "mean_balance": 1 / 16,
            "legal_coalitions": example["legal"].astype(int).tolist(),
            "iid_vectors_3d": example["iid_vectors"].tolist(),
            "iid_camera_coordinates": example["iid_coords"].tolist(),
            "inside_indices": example["inside_indices"].tolist(),
            "zero_sum_basis": ZERO_SUM_BASIS.tolist(), "camera": CAMERA.tolist(),
            "iid_sphere_moments": example["iid_sphere_moments"], "inside_coalition_moments": example["inside_moments"],
            "depth_encoding": "dashed direction shafts are behind the viewing plane",
            "sphere_style": "original soft radial gray gradient; no specular or contact shadow; narrower horizontal ellipse, slightly below center, uniformly light dashed",
            "sphere_style_reference": "supplied original schematic",
            "radial_gradient": {"center_xy": [-0.32, 0.48], "radius": 1.44,
                                "stops": [0, 0.46, 0.78, 1],
                                "gray_8bit": [255, 232, 207, 155]},
            "ellipse": {"radius_x": 0.82, "radius_y": 0.24, "center_y": -0.094},
            "display_labels": "INSIDE abbreviates INSIDE-Coalition; no unsampled markers or multiplicity labels",
            "mapping_annotation": "Center + normalize below panel a sphere, 7.2 pt; mapping formula and complement equation omitted",
            "label_shorthand": "u_i denotes the normalized direction of coalition S_i",
            "uncertainty": "Not applicable: schematic coordinates and one continuous-sphere IID illustration",
        }, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/pdf/inside_intro_figure.pdf"))
    parser.add_argument("--iid-seed", type=int, default=DEFAULT_IID_SEED)
    parser.add_argument("--design-seed", type=int, default=0)
    parser.add_argument("--audit-alignment", action="store_true")
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()
    print(export(args.output, args.iid_seed, args.design_seed, args.audit_alignment, args.metadata))


if __name__ == "__main__":
    main()
