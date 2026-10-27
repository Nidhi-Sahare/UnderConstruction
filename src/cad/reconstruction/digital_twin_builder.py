
"""
BuildProgress - Step 13
CAD Geometry to Interactive 3D Digital Twin

Purpose:
    Convert normalized CAD wall segments into a first
    interactive 3D representation of the planned building.

Important:
    This is a geometry prototype, not a validated BIM model.

    Wall height and thickness are assumptions unless
    supplied from authoritative architectural data.

Outputs:
    data/digital_twin/building_elements.json
    data/digital_twin/digital_twin.html
    data/digital_twin/build_report.json

Requirements:
    pip install numpy plotly

Usage:
    python digital_twin_builder.py --inspect

    python digital_twin_builder.py --build

    python digital_twin_builder.py --build --height 3.0 --thickness 0.15

    python digital_twin_builder.py --build --input path/to/normalized_segments.json
"""

import argparse
import hashlib
import json
import math
import webbrowser

from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


PROJECT_def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "digital_twin"
)

DEFAULT_SEARCH_ROOT = PROJECT_ROOT / "data"

CATEGORY_COLORS = {
    "exterior": "#D97706",
    "interior": "#2563EB",
    "partition": "#16A34A",
    "lining": "#8B5CF6",
    "unknown": "#64748B",
}

# These values are visualization assumptions,
# not verified architectural measurements.
DEFAULT_HEIGHT = 3.0
DEFAULT_THICKNESS = 0.15


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_geometry_file(search_root):
    candidates = sorted(
        search_root.rglob("normalized_segments.json")
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find normalized_segments.json.\n"
            "Pass --input with the correct file path."
        )

    if len(candidates) > 1:
        print("\nMultiple geometry files found:")

        for candidate in candidates:
            print(f"  {candidate}")

        raise RuntimeError(
            "Please select one using --input."
        )

    return candidates[0]


def is_point(value):
    if isinstance(value, dict):
        return (
            "x" in value
            and "y" in value
        )

    if isinstance(value, (list, tuple)):
        return (
            len(value) >= 2
            and all(
                isinstance(item, (int, float))
                for item in value[:2]
            )
        )

    return False


def parse_point(value):
    if isinstance(value, dict):
        x = float(value["x"])
        y = float(value["y"])
    else:
        x = float(value[0])
        y = float(value[1])

    if not (
        math.isfinite(x)
        and math.isfinite(y)
    ):
        raise ValueError("Non-finite coordinate.")

    return (x, y)


def find_endpoint_pair(record):
    """
    Supports several common segment JSON structures.

    This intentionally does not assume one specific
    CAD-export schema.
    """

    if isinstance(record, (list, tuple)):
        if (
            len(record) == 2
            and is_point(record[0])
            and is_point(record[1])
        ):
            return (
                parse_point(record[0]),
                parse_point(record[1]),
            )

    if not isinstance(record, dict):
        return None

    endpoint_pairs = [
        ("start", "end"),
        ("p1", "p2"),
        ("a", "b"),
        ("source", "target"),
        ("point1", "point2"),
    ]

    for first, second in endpoint_pairs:
        if (
            first in record
            and second in record
            and is_point(record[first])
            and is_point(record[second])
        ):
            return (
                parse_point(record[first]),
                parse_point(record[second]),
            )

    for key in (
        "points",
        "coordinates",
        "vertices",
        "geometry",
    ):
        value = record.get(key)

        if (
            isinstance(value, list)
            and len(value) == 2
            and is_point(value[0])
            and is_point(value[1])
        ):
            return (
                parse_point(value[0]),
                parse_point(value[1]),
            )

        if (
            isinstance(value, dict)
            and "coordinates" in value
        ):
            coordinates = value["coordinates"]

            if (
                isinstance(coordinates, list)
                and len(coordinates) == 2
                and is_point(coordinates[0])
                and is_point(coordinates[1])
            ):
                return (
                    parse_point(coordinates[0]),
                    parse_point(coordinates[1]),
                )

    if all(
        key in record
        for key in ("x1", "y1", "x2", "y2")
    ):
        return (
            parse_point(
                [record["x1"], record["y1"]]
            ),
            parse_point(
                [record["x2"], record["y2"]]
            ),
        )

    return None


def discover_segment_lists(data):
    """
    Recursively discover arrays of segment-like records.

    Returns candidate collections and their JSON paths.
    """

    candidates = []

    def visit(value, path):
        if isinstance(value, list):
            if value:
                sample = value[: min(20, len(value))]

                recognized = sum(
                    find_endpoint_pair(item) is not None
                    for item in sample
                )

                if recognized > 0:
                    candidates.append(
                        {
                            "path": path,
                            "records": value,
                            "sample_matches": recognized,
                            "sample_size": len(sample),
                        }
                    )

            for index, item in enumerate(value):
                if isinstance(item, (dict, list)):
                    visit(
                        item,
                        f"{path}[{index}]",
                    )

        elif isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    visit(
                        item,
                        f"{path}.{key}",
                    )

    visit(data, "$")

    # Prefer the largest recognized segment collection.
    candidates.sort(
        key=lambda item: (
            sum(
                find_endpoint_pair(record) is not None
                for record in item["records"]
            ),
            len(item["records"]),
        ),
        reverse=True,
    )

    return candidates


def infer_category(record):
    if not isinstance(record, dict):
        return "unknown"

    for key in (
        "category",
        "wall_category",
        "wall_type",
        "type",
        "layer",
        "source_layer",
    ):
        value = record.get(key)

        if not isinstance(value, str):
            continue

        normalized = value.lower()

        if (
            "exterior" in normalized
            or "aussenwand" in normalized
            or "auÃŸenwand" in normalized
        ):
            return "exterior"

        if (
            "interior" in normalized
            or "innenwand" in normalized
        ):
            return "interior"

        if (
            "partition" in normalized
            or "trennwand" in normalized
        ):
            return "partition"

        if (
            "lining" in normalized
            or "vormauerung" in normalized
        ):
            return "lining"

    return "unknown"


def stable_element_id(start, end, category):
    # Direction-independent geometry ID.
    points = sorted([
        [round(start[0], 6), round(start[1], 6)],
        [round(end[0], 6), round(end[1], 6)],
    ])

    payload = json.dumps(
        {
            "points": points,
            "category": category,
        },
        sort_keys=True,
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return f"wall_{digest}"


def extract_segments(data):
    collections = discover_segment_lists(data)

    if not collections:
        raise RuntimeError(
            "No supported segment collection found."
        )

    selected = collections[0]

    segments = []
    rejected = 0
    duplicates = 0
    seen = set()

    for record in selected["records"]:
        endpoints = find_endpoint_pair(record)

        if endpoints is None:
            rejected += 1
            continue

        start, end = endpoints

        length = math.dist(start, end)

        if length < 1e-6:
            rejected += 1
            continue

        category = infer_category(record)

        element_id = stable_element_id(
            start,
            end,
            category,
        )

        if element_id in seen:
            duplicates += 1
            continue

        seen.add(element_id)

        segments.append({
            "element_id": element_id,
            "category": category,
            "start": list(start),
            "end": list(end),
            "length_m": length,
        })

    if not segments:
        raise RuntimeError(
            "No valid wall segments could be extracted."
        )

    return {
        "segments": segments,
        "selected_path": selected["path"],
        "candidate_collections": [
            {
                "path": item["path"],
                "records": len(item["records"]),
                "sample_matches": item["sample_matches"],
            }
            for item in collections[:10]
        ],
        "rejected": rejected,
        "duplicates": duplicates,
    }


def make_wall_vertices(
    start,
    end,
    height,
    thickness,
):
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    length = math.hypot(dx, dy)

    if length <= 1e-9:
        raise ValueError(
            "Cannot extrude a zero-length segment."
        )

    nx = -dy / length
    ny = dx / length

    half = thickness / 2.0

    base = [
        (x1 + nx * half, y1 + ny * half),
        (x1 - nx * half, y1 - ny * half),
        (x2 - nx * half, y2 - ny * half),
        (x2 + nx * half, y2 + ny * half),
    ]

    vertices = []

    for z in (0.0, height):
        for x, y in base:
            vertices.append((x, y, z))

    return np.asarray(
        vertices,
        dtype=float,
    )


def add_wall_mesh(
    figure,
    vertices,
    color,
    element_id,
    category,
    length,
):
    # 12 triangles forming a rectangular prism.
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (1, 5, 6),
        (1, 6, 2),
        (2, 6, 7),
        (2, 7, 3),
        (3, 7, 4),
        (3, 4, 0),
    ]

    i, j, k = zip(*triangles)

    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color=color,
            opacity=0.82,
            flatshading=True,
            name=category,
            legendgroup=category,
            showlegend=False,
            customdata=[
                [element_id, category, length]
            ] * len(vertices),
            hovertemplate=(
                "Element: %{customdata[0]}"
                "<br>Category: %{customdata[1]}"
                "<br>Length: %{customdata[2]:.2f} m"
                "<extra></extra>"
            ),
        )
    )


def create_viewer(
    segments,
    output_path,
    height,
    thickness,
):
    figure = go.Figure()

    categories = Counter(
        segment["category"]
        for segment in segments
    )

    all_x = []
    all_y = []

    for segment in segments:
        start = segment["start"]
        end = segment["end"]

        vertices = make_wall_vertices(
            start,
            end,
            height,
            thickness,
        )

        color = CATEGORY_COLORS.get(
            segment["category"],
            CATEGORY_COLORS["unknown"],
        )

        add_wall_mesh(
            figure,
            vertices,
            color,
            segment["element_id"],
            segment["category"],
            segment["length_m"],
        )

        all_x.extend([
            start[0],
            end[0],
        ])

        all_y.extend([
            start[1],
            end[1],
        ])

    for category, count in sorted(categories.items()):
        figure.add_trace(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker={
                    "size": 8,
                    "color": CATEGORY_COLORS.get(
                        category,
                        CATEGORY_COLORS["unknown"],
                    ),
                },
                name=f"{category} ({count})",
                showlegend=True,
            )
        )

    x_center = (
        min(all_x) + max(all_x)
    ) / 2.0

    y_center = (
        min(all_y) + max(all_y)
    ) / 2.0

    # Center only the camera view.
    # Original CAD coordinates are preserved.
    figure.update_layout(
        title={
            "text": (
                "BuildProgress | Planned Building Geometry"
                "<br><sup>"
                "Assumed wall height/thickness â€” "
                "not verified BIM geometry"
                "</sup>"
            ),
            "x": 0.03,
        },
        scene={
            "xaxis_title": "CAD X (m)",
            "yaxis_title": "CAD Y (m)",
            "zaxis_title": "Height (m)",
            "aspectmode": "data",
            "camera": {
                "eye": {
                    "x": 1.5,
                    "y": -1.7,
                    "z": 1.2,
                },
                "center": {
                    "x": 0,
                    "y": 0,
                    "z": 0,
                },
            },
        },
        margin={
            "l": 0,
            "r": 0,
            "t": 90,
            "b": 0,
        },
        legend={
            "x": 0.02,
            "y": 0.94,
        },
        paper_bgcolor="#F8FAFC",
        font={
            "family": "Arial",
            "color": "#0F172A",
        },
    )

    figure.write_html(
        str(output_path),
        include_plotlyjs=True,
        full_html=True,
    )


def build_twin(
    input_path,
    output_dir,
    height,
    thickness,
    open_browser,
):
    print("\n" + "=" * 60)
    print("BUILDPROGRESS - DIGITAL TWIN BUILDER")
    print("=" * 60)

    print(f"Input: {input_path}")

    data = load_json(input_path)

    extracted = extract_segments(data)

    segments = extracted["segments"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    elements = []

    for segment in segments:
        elements.append({
            **segment,
            "geometry": {
                "representation": "extruded_centerline",
                "height_m": height,
                "thickness_m": thickness,
                "height_source": "user_assumption",
                "thickness_source": "user_assumption",
            },
            "progress": {
                "status": "unobserved",
                "confidence": None,
                "evidence": [],
            },
        })

    manifest = {
        "schema_version": "0.1",
        "coordinate_system": {
            "source": "DXF",
            "units": "meters",
            "axes": {
                "x": "CAD X",
                "y": "CAD Y",
                "z": "vertical",
            },
        },
        "source_geometry": str(
            input_path.resolve()
        ),
        "elements": elements,
    }

    manifest_path = (
        output_dir / "building_elements.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    html_path = (
        output_dir / "digital_twin.html"
    )

    print(f"Elements: {len(elements)}")
    print("Generating 3D viewer...")

    create_viewer(
        segments,
        html_path,
        height,
        thickness,
    )

    lengths = [
        segment["length_m"]
        for segment in segments
    ]

    report = {
        "input": str(input_path.resolve()),
        "selected_collection": extracted["selected_path"],
        "candidate_collections": extracted[
            "candidate_collections"
        ],
        "wall_elements": len(elements),
        "rejected_records": extracted["rejected"],
        "duplicate_records": extracted["duplicates"],
        "categories": dict(
            Counter(
                segment["category"]
                for segment in segments
            )
        ),
        "length_statistics_m": {
            "minimum": min(lengths),
            "median": float(np.median(lengths)),
            "maximum": max(lengths),
        },
        "assumptions": {
            "height_m": height,
            "thickness_m": thickness,
            "geometry": (
                "Every segment is represented by a "
                "rectangular prism. This does not "
                "reconstruct exact wall solids."
            ),
        },
    }

    report_path = (
        output_dir / "build_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)

    print(f"Wall elements: {len(elements)}")
    print(f"Manifest: {manifest_path}")
    print(f"Viewer: {html_path}")
    print(f"Report: {report_path}")

    print("\nCategory counts:")

    for category, count in sorted(
        report["categories"].items()
    ):
        print(f"  {category}: {count}")

    if open_browser:
        webbrowser.open(
            html_path.resolve().as_uri()
        )


def inspect_geometry(input_path):
    print("\n" + "=" * 60)
    print("CAD GEOMETRY INSPECTION")
    print("=" * 60)

    print(f"Input: {input_path}")

    data = load_json(input_path)

    collections = discover_segment_lists(data)

    if not collections:
        print("No supported segment collections found.")

        if isinstance(data, dict):
            print(
                "Top-level keys:",
                list(data.keys()),
            )

        return

    for collection in collections[:10]:
        print(
            f"\nPath: {collection['path']}"
            f"\nRecords: {len(collection['records'])}"
            f"\nSample matches: "
            f"{collection['sample_matches']}"
            f"/{collection['sample_size']}"
        )

        for record in collection["records"][:2]:
            print(
                json.dumps(
                    record,
                    indent=2,
                    default=str,
                )[:1000]
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    parser.add_argument(
        "--height",
        type=float,
        default=DEFAULT_HEIGHT,
    )

    parser.add_argument(
        "--thickness",
        type=float,
        default=DEFAULT_THICKNESS,
    )

    parser.add_argument(
        "--open",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.inspect and not args.build:
        parser.error(
            "Specify --inspect or --build."
        )

    if args.height <= 0:
        parser.error(
            "--height must be positive."
        )

    if args.thickness <= 0:
        parser.error(
            "--thickness must be positive."
        )

    input_path = args.input

    if input_path is None:
        input_path = find_geometry_file(
            DEFAULT_SEARCH_ROOT
        )

    if not input_path.is_file():
        parser.error(
            f"Input file not found: {input_path}"
        )

    if args.inspect:
        inspect_geometry(input_path)

    if args.build:
        build_twin(
            input_path=input_path,
            output_dir=args.output,
            height=args.height,
            thickness=args.thickness,
            open_browser=args.open,
        )


if __name__ == "__main__":
    main()

