
"""
BuildProgress - Step 18
Exterior Wall Reconstruction

Input:
    data/geometry/normalized_segments.json

Outputs:
    data/exterior_reconstruction/exterior_wall_candidates.json
    data/exterior_reconstruction/exterior_report.json
    data/exterior_reconstruction/exterior_preview.png

This script:
- Extracts exterior CAD segments.
- Snaps nearby endpoints conservatively.
- Builds connected linework.
- Finds closed polygon regions.
- Classifies nested polygon boundaries.
- Preserves open/unresolved segments.
- Does not invent wall thickness or openings.

Requires:
    pip install shapely matplotlib numpy

Usage:
    python exterior_wall_reconstructor.py --inspect
    python exterior_wall_reconstructor.py --build
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import (
    LineString,
    MultiLineString,
    Polygon,
)
from shapely.ops import (
    polygonize_full,
    unary_union,
)


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DEFAULT_INPUT = (
    ROOT / "data" / "geometry"
    / "normalized_segments.json"
)

DEFAULT_OUTPUT = (
    ROOT / "data" / "exterior_reconstruction"
)


def stable_id(prefix, payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(
        encoded
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


def read_segments(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("segments")

    if not isinstance(records, list):
        raise ValueError(
            "Expected 'segments' list."
        )

    exterior = []

    for index, record in enumerate(records):
        if record.get("category") != "exterior":
            continue

        start = record.get("start")
        end = record.get("end")

        if not (
            isinstance(start, list)
            and isinstance(end, list)
            and len(start) >= 2
            and len(end) >= 2
        ):
            continue

        a = (
            float(start[0]),
            float(start[1]),
        )

        b = (
            float(end[0]),
            float(end[1]),
        )

        if not all(
            math.isfinite(value)
            for value in (*a, *b)
        ):
            continue

        if math.dist(a, b) <= 1e-9:
            continue

        exterior.append({
            "id": str(
                record.get(
                    "id",
                    f"EXTERIOR_{index:06d}",
                )
            ),
            "start": a,
            "end": b,
            "layer": record.get("layer", ""),
            "source_handle": record.get(
                "source_handle"
            ),
        })

    return exterior


def snap_endpoints(segments, tolerance):
    """
    Cluster endpoints within tolerance using a
    spatial grid and union-find.

    Snapping is applied only to exterior endpoints.
    """
    points = []

    for item in segments:
        points.append(item["start"])
        points.append(item["end"])

    parent = list(range(len(points)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]

        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    grid = defaultdict(list)

    cell_size = max(tolerance, 1e-9)

    for index, point in enumerate(points):
        cell = (
            math.floor(point[0] / cell_size),
            math.floor(point[1] / cell_size),
        )

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                neighbor = (
                    cell[0] + dx,
                    cell[1] + dy,
                )

                for other in grid.get(neighbor, []):
                    if (
                        math.dist(
                            point,
                            points[other],
                        ) <= tolerance
                    ):
                        union(index, other)

        grid[cell].append(index)

    components = defaultdict(list)

    for index, point in enumerate(points):
        components[find(index)].append(point)

    representatives = {}

    for root, members in components.items():
        representatives[root] = (
            float(np.mean(
                [p[0] for p in members]
            )),
            float(np.mean(
                [p[1] for p in members]
            )),
        )

    snapped = []

    moved = 0

    for index, item in enumerate(segments):
        a = representatives[
            find(2 * index)
        ]

        b = representatives[
            find(2 * index + 1)
        ]

        if math.dist(a, item["start"]) > 1e-9:
            moved += 1

        if math.dist(b, item["end"]) > 1e-9:
            moved += 1

        if math.dist(a, b) <= 1e-9:
            continue

        snapped.append({
            **item,
            "start": a,
            "end": b,
        })

    return snapped, moved


def polygon_coordinates(polygon):
    return [
        [float(x), float(y)]
        for x, y in polygon.exterior.coords
    ]


def extract_polygons(segments):
    lines = [
        LineString([
            item["start"],
            item["end"],
        ])
        for item in segments
    ]

    if not lines:
        return [], [], [], []

    network = unary_union(lines)

    polygons, dangles, cuts, invalid = (
        polygonize_full(network)
    )

    regions = sorted(
        [
            geometry
            for geometry in polygons.geoms
            if (
                geometry.is_valid
                and geometry.area > 1e-8
            )
        ],
        key=lambda polygon: (
            -polygon.area,
            polygon.centroid.x,
            polygon.centroid.y,
        ),
    )

    return (
        regions,
        list(dangles.geoms),
        list(cuts.geoms),
        list(invalid.geoms),
    )


def classify_regions(regions):
    """
    Identify nesting relationships.

    This does not declare an enclosed region to be
    a physical wall. Regions remain hypotheses.
    """
    result = []

    for index, polygon in enumerate(regions):
        containers = []

        point = polygon.representative_point()

        for other_index, other in enumerate(regions):
            if index == other_index:
                continue

            if other.area <= polygon.area:
                continue

            if other.contains(point):
                containers.append(other_index)

        depth = len(containers)

        identifier = stable_id(
            "EXTERIOR_REGION",
            {
                "boundary": [
                    [round(x, 6), round(y, 6)]
                    for x, y in polygon.exterior.coords
                ]
            },
        )

        result.append({
            "id": identifier,
            "area_m2": float(polygon.area),
            "perimeter_m": float(polygon.length),
            "nesting_depth": depth,
            "polygon": polygon_coordinates(polygon),
            "status": "closed_cad_region",
            "verified_wall": False,
            "notes": (
                "Closed CAD region; architectural "
                "meaning not established"
            ),
        })

    return result


def plot_result(
    original,
    snapped,
    regions,
    output_path,
):
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 8)
    )

    for item in original:
        a = item["start"]
        b = item["end"]

        axes[0].plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#D97706",
            linewidth=0.9,
        )

    axes[0].set_title(
        f"Original exterior segments ({len(original)})"
    )

    for item in snapped:
        a = item["start"]
        b = item["end"]

        axes[1].plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#CBD5E1",
            linewidth=0.6,
        )

    for region in regions:
        points = np.asarray(
            region["polygon"]
        )

        axes[1].fill(
            points[:, 0],
            points[:, 1],
            color="#D97706",
            alpha=0.35,
        )

        axes[1].plot(
            points[:, 0],
            points[:, 1],
            color="#B45309",
            linewidth=1.0,
        )

    axes[1].set_title(
        f"Closed CAD regions ({len(regions)})"
    )

    for axis in axes:
        axis.set_aspect(
            "equal",
            adjustable="box",
        )
        axis.grid(alpha=0.15)
        axis.set_xlabel("CAD X (m)")
        axis.set_ylabel("CAD Y (m)")

    fig.suptitle(
        "BuildProgress | Exterior CAD Reconstruction"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)


def inspect(segments):
    print("\nEXTERIOR CAD INSPECTION")
    print("-" * 50)

    print("Exterior segments:", len(segments))

    lengths = [
        math.dist(
            item["start"],
            item["end"],
        )
        for item in segments
    ]

    if lengths:
        print(
            "Median segment length:",
            round(float(np.median(lengths)), 4),
            "m",
        )

        print(
            "Maximum segment length:",
            round(max(lengths), 4),
            "m",
        )

    layers = Counter(
        item["layer"]
        for item in segments
    )

    print("Layers:", dict(layers))


def build(segments, args):
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    snapped, moved = snap_endpoints(
        segments,
        args.snap_tolerance,
    )

    regions, dangles, cuts, invalid = (
        extract_polygons(snapped)
    )

    classified = classify_regions(regions)

    result = {
        "schema_version": "0.1",
        "source": str(args.input.resolve()),
        "coordinate_system": {
            "units": "meters",
            "source": "DXF",
        },
        "regions": classified,
        "source_exterior_segment_ids": [
            item["id"]
            for item in segments
        ],
        "notes": [
            "Closed regions are not confirmed walls.",
            "No wall thickness is invented.",
            "No openings are filled.",
            "No building-specific coordinates are used.",
        ],
    }

    report = {
        "source_exterior_segments": len(segments),
        "snapped_segments": len(snapped),
        "endpoints_moved": moved,
        "closed_regions": len(classified),
        "dangles": len(dangles),
        "cuts": len(cuts),
        "invalid_rings": len(invalid),
        "snap_tolerance_m": args.snap_tolerance,
    }

    result_path = (
        args.output / "exterior_wall_candidates.json"
    )

    report_path = (
        args.output / "exterior_report.json"
    )

    preview_path = (
        args.output / "exterior_preview.png"
    )

    result_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    plot_result(
        segments,
        snapped,
        classified,
        preview_path,
    )

    print("\nEXTERIOR RECONSTRUCTION COMPLETE")
    print("-" * 50)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nOutputs:")
    print(result_path)
    print(report_path)
    print(preview_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--snap-tolerance",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--inspect",
        action="store_true",
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"Input not found: {args.input}"
        )

    if args.snap_tolerance < 0:
        parser.error(
            "Snap tolerance cannot be negative."
        )

    if not (args.inspect or args.build):
        parser.error(
            "Specify --inspect or --build."
        )

    segments = read_segments(
        args.input
    )

    if args.inspect:
        inspect(segments)

    if args.build:
        build(segments, args)


if __name__ == "__main__":
    main()

