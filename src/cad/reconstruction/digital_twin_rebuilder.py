
import argparse
import json
import math
import webbrowser
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

SELECTION_PATH = (
    ROOT / "data" / "wall_selection"
    / "selected_wall_footprints.json"
)

CAD_PATH = (
    ROOT / "data" / "geometry"
    / "normalized_segments.json"
)

OUTPUT_DIR = ROOT / "data" / "digital_twin_v2"

COLORS = {
    "interior": "#2563EB",
    "lining": "#8B5CF6",
    "partition": "#16A34A",
    "exterior": "#D97706",
}


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_ring(points):
    result = []

    for point in points:
        xy = [float(point[0]), float(point[1])]

        if not all(math.isfinite(v) for v in xy):
            raise ValueError("Non-finite polygon coordinate")

        if not result or math.dist(xy, result[-1]) > 1e-8:
            result.append(xy)

    if (
        len(result) > 1
        and math.dist(result[0], result[-1]) < 1e-8
    ):
        result.pop()

    return result


def signed_area(points):
    return sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    ) / 2.0


def make_wall_mesh(item, height):
    points = clean_ring(item["polygon"])

    # The current footprint resolver produces rectangular
    # polygons. Refuse unsupported shapes instead of
    # silently triangulating them incorrectly.
    if len(points) != 4:
        raise ValueError(
            f"{item['id']}: expected 4 polygon corners, "
            f"received {len(points)}"
        )

    if abs(signed_area(points)) < 1e-9:
        raise ValueError(
            f"{item['id']}: degenerate footprint"
        )

    vertices = (
        [[x, y, 0.0] for x, y in points]
        + [[x, y, height] for x, y in points]
    )

    triangles = []

    # Bottom and top faces.
    triangles.extend([
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
    ])

    # Four side faces.
    for i in range(4):
        j = (i + 1) % 4

        triangles.extend([
            (i, j, j + 4),
            (i, j + 4, i + 4),
        ])

    return vertices, triangles


def build_interior(fig, footprints, height):
    counts = Counter()

    for item in footprints:
        category = item.get("category", "interior")

        vertices, triangles = make_wall_mesh(
            item, height
        )

        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]

        i = [t[0] for t in triangles]
        j = [t[1] for t in triangles]
        k = [t[2] for t in triangles]

        color = COLORS.get(category, "#64748B")

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=color,
                opacity=0.92,
                flatshading=True,
                name=category,
                legendgroup=category,
                showlegend=counts[category] == 0,
                hovertemplate=(
                    f"ID: {item['id']}"
                    f"<br>Category: {category}"
                    f"<br>Thickness: "
                    f"{item.get('estimated_thickness_m', 0):.3f} m"
                    "<br>Status: provisional"
                    "<extra></extra>"
                ),
            )
        )

        counts[category] += 1

    return counts


def build_exterior_reference(fig, cad, height):
    """
    Exterior CAD linework is shown at floor and roof
    reference levels. This does not infer wall thickness.
    """
    segments = cad.get("segments", [])

    exterior = [
        item for item in segments
        if item.get("category") == "exterior"
    ]

    if not exterior:
        raise ValueError(
            "No exterior CAD segments found. "
            "Cannot build the exterior reference."
        )

    xs = []
    ys = []
    zs = []

    for segment in exterior:
        start = segment["start"]
        end = segment["end"]

        for z in (0.0, height):
            xs.extend([
                float(start[0]),
                float(end[0]),
                None,
            ])
            ys.extend([
                float(start[1]),
                float(end[1]),
                None,
            ])
            zs.extend([z, z, None])

    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line=dict(
                color=COLORS["exterior"],
                width=4,
            ),
            name="Exterior CAD reference",
            hoverinfo="name",
        )
    )

    # Add vertical reference edges only at unique
    # CAD endpoints. This is a wireframe, not a wall.
    unique_points = set()

    for segment in exterior:
        for key in ("start", "end"):
            point = segment[key]
            unique_points.add((
                round(float(point[0]), 5),
                round(float(point[1]), 5),
            ))

    vx, vy, vz = [], [], []

    for x, y in sorted(unique_points):
        vx.extend([x, x, None])
        vy.extend([y, y, None])
        vz.extend([0.0, height, None])

    fig.add_trace(
        go.Scatter3d(
            x=vx,
            y=vy,
            z=vz,
            mode="lines",
            line=dict(
                color="rgba(217,119,6,0.20)",
                width=1,
            ),
            name="Exterior height guides",
            hoverinfo="skip",
            showlegend=False,
            visible="legendonly",
        )
    )

    return len(exterior)


def make_figure(selection, cad, height):
    fig = go.Figure()

    selected = selection["selected_footprints"]

    counts = build_interior(
        fig, selected, height
    )

    exterior_count = build_exterior_reference(
        fig, cad, height
    )

    fig.update_layout(
        title=dict(
            text=(
                "BuildProgress | Reconstructed Digital Twin"
                "<br><sup>"
                "Solid interiors: CAD-derived provisional "
                "footprints | Exterior: CAD wireframe | "
                "Height: assumed"
                "</sup>"
            ),
            x=0.04,
        ),
        paper_bgcolor="#F8FAFC",
        font=dict(
            family="Arial",
            color="#172033",
        ),
        scene=dict(
            xaxis=dict(
                title="CAD X (m)",
                backgroundcolor="#F1F5F9",
            ),
            yaxis=dict(
                title="CAD Y (m)",
                backgroundcolor="#F1F5F9",
            ),
            zaxis=dict(
                title="Assumed height (m)",
                backgroundcolor="#F1F5F9",
                range=[0, height * 1.25],
            ),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=-1.7, z=1.4)
            ),
        ),
        legend=dict(
            x=0.02,
            y=0.88,
        ),
        margin=dict(
            l=0, r=0, t=95, b=0
        ),
    )

    return fig, counts, exterior_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--selection",
        type=Path,
        default=SELECTION_PATH,
    )

    parser.add_argument(
        "--cad",
        type=Path,
        default=CAD_PATH,
    )

    parser.add_argument(
        "--height",
        type=float,
        default=3.0,
        help="Assumed wall height in meters",
    )

    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated viewer",
    )

    args = parser.parse_args()

    if not math.isfinite(args.height) or args.height <= 0:
        parser.error("--height must be positive and finite")

    selection = read_json(args.selection)
    cad = read_json(args.cad)

    selected = selection.get(
        "selected_footprints", []
    )

    if not selected:
        raise ValueError(
            "No selected footprints available"
        )

    if selection.get("coordinate_system", {}).get(
        "units"
    ) != "meters":
        raise ValueError(
            "Selection coordinate units are not meters"
        )

    if len(selected) != len({
        item["id"] for item in selected
    }):
        raise ValueError(
            "Duplicate selected footprint IDs"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, counts, exterior_count = make_figure(
        selection, cad, args.height
    )

    viewer_path = (
        OUTPUT_DIR / "reconstructed_twin.html"
    )

    report_path = (
        OUTPUT_DIR / "reconstruction_report.json"
    )

    fig.write_html(
        str(viewer_path),
        include_plotlyjs=True,
        full_html=True,
    )

    report = {
        "schema_version": "0.1",
        "source_selection_id": selection.get(
            "selection_id"
        ),
        "selected_wall_footprints": len(selected),
        "excluded_alternatives": len(
            selection.get("excluded_footprints", [])
        ),
        "selected_categories": dict(counts),
        "exterior_reference_segments": exterior_count,
        "assumed_height_m": args.height,
        "geometry_status": (
            "provisional_not_verified_BIM"
        ),
        "exterior_representation": (
            "source_CAD_wireframe_not_solid_walls"
        ),
        "openings_modeled": False,
        "floor_slab_modeled": False,
        "progress_status_estimated": False,
        "limitations": [
            "Only selected footprints are solid.",
            "Exterior wall thickness is not reconstructed.",
            "Wall height is an explicit assumption.",
            "Unmatched CAD chains remain unmodeled.",
            "No site imagery is registered to this model.",
            "No construction completion claims are made.",
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nDIGITAL TWIN V2 COMPLETE")
    print("-" * 52)
    print("Solid footprints:", len(selected))
    print("Selected categories:", dict(counts))
    print("Exterior reference segments:", exterior_count)
    print("Assumed height:", args.height, "m")
    print("Viewer:", viewer_path)
    print("Report:", report_path)

    if args.open:
        webbrowser.open(viewer_path.resolve().as_uri())


if __name__ == "__main__":
    main()

