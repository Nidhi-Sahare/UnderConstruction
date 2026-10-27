
"""
BuildProgress - Step 19
Exterior Boundary Topology Audit

Input:
    data/geometry/normalized_segments.json

Outputs:
    data/exterior_audit/topology_report.json
    data/exterior_audit/gap_candidates.json
    data/exterior_audit/topology_preview.png

No source geometry is modified.

Requirements:
    pip install numpy matplotlib shapely

Usage:
    python exterior_boundary_audit.py --inspect
    python exterior_boundary_audit.py --build
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


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
    ROOT / "data" / "exterior_audit"
)


def load_exterior(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("segments")

    if not isinstance(records, list):
        raise ValueError("Missing segments list.")

    exterior = []

    for index, record in enumerate(records):
        if record.get("category") != "exterior":
            continue

        start = tuple(
            float(v) for v in record["start"][:2]
        )

        end = tuple(
            float(v) for v in record["end"][:2]
        )

        if not all(
            math.isfinite(v)
            for v in start + end
        ):
            continue

        if math.dist(start, end) <= 1e-9:
            continue

        exterior.append({
            "id": str(
                record.get(
                    "id", f"EXTERIOR_{index}"
                )
            ),
            "start": start,
            "end": end,
            "layer": record.get("layer", ""),
        })

    return exterior


def point_key(point, decimals=6):
    return (
        round(point[0], decimals),
        round(point[1], decimals),
    )


def build_graph(segments):
    """
    Endpoint graph based on exact coordinates rounded
    to six decimal places.

    No snapping or coordinate modification occurs.
    """
    adjacency = defaultdict(set)
    incident_segments = defaultdict(list)

    for index, segment in enumerate(segments):
        a = point_key(segment["start"])
        b = point_key(segment["end"])

        adjacency[a].add(b)
        adjacency[b].add(a)

        incident_segments[a].append(index)
        incident_segments[b].append(index)

    return adjacency, incident_segments


def connected_components(adjacency):
    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue

        stack = [node]
        visited.add(node)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components


def endpoint_candidates(
    endpoints,
    max_gap,
    min_gap,
):
    """
    Find nearby degree-1 endpoint pairs.

    These are candidates only, not confirmed gaps.
    """
    candidates = []

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            a = endpoints[i]
            b = endpoints[j]

            distance = math.dist(a, b)

            if not (
                min_gap <= distance <= max_gap
            ):
                continue

            candidates.append({
                "type": "endpoint_to_endpoint",
                "start": list(a),
                "end": list(b),
                "distance_m": round(distance, 6),
                "status": "unverified_gap_candidate",
            })

    return candidates


def endpoint_segment_candidates(
    endpoints,
    segments,
    incident_segments,
    max_gap,
    min_gap,
):
    """
    Find endpoints near non-incident line segments.

    Uses an STRtree for spatial lookup.
    """
    lines = [
        LineString([
            item["start"],
            item["end"],
        ])
        for item in segments
    ]

    tree = STRtree(lines)

    candidates = []

    for endpoint in endpoints:
        point = Point(endpoint)

        search_region = point.buffer(max_gap)

        possible_indices = tree.query(
            search_region
        )

        incident = set(
            incident_segments[endpoint]
        )

        for index in possible_indices:
            index = int(index)

            if index in incident:
                continue

            line = lines[index]

            distance = point.distance(line)

            if not (
                min_gap <= distance <= max_gap
            ):
                continue

            projection = line.interpolate(
                line.project(point)
            )

            candidates.append({
                "type": "endpoint_to_segment",
                "start": list(endpoint),
                "end": [
                    float(projection.x),
                    float(projection.y),
                ],
                "distance_m": round(
                    distance, 6
                ),
                "target_segment_id": (
                    segments[index]["id"]
                ),
                "status": (
                    "unverified_gap_candidate"
                ),
            })

    return candidates


def deduplicate_candidates(candidates):
    unique = {}

    for candidate in candidates:
        a = tuple(
            round(v, 6)
            for v in candidate["start"]
        )

        b = tuple(
            round(v, 6)
            for v in candidate["end"]
        )

        endpoints = sorted([a, b])

        key = (
            candidate["type"],
            tuple(endpoints),
            candidate.get(
                "target_segment_id"
            ),
        )

        if key not in unique:
            unique[key] = candidate

    result = list(unique.values())

    result.sort(
        key=lambda item: (
            item["distance_m"],
            item["type"],
        )
    )

    for index, candidate in enumerate(
        result, start=1
    ):
        candidate["id"] = f"GAP_{index:04d}"

    return result


def draw_preview(
    segments,
    endpoints,
    candidates,
    output_path,
):
    fig, axis = plt.subplots(
        figsize=(13, 10)
    )

    for item in segments:
        a = item["start"]
        b = item["end"]

        axis.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#D97706",
            linewidth=0.85,
            alpha=0.85,
        )

    if endpoints:
        array = np.asarray(endpoints)

        axis.scatter(
            array[:, 0],
            array[:, 1],
            color="#DC2626",
            s=20,
            zorder=5,
            label="Degree-1 endpoints",
        )

    for candidate in candidates:
        a = candidate["start"]
        b = candidate["end"]

        axis.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#2563EB",
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
        )

        midpoint = (
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2,
        )

        axis.annotate(
            candidate["id"],
            midpoint,
            fontsize=6,
            color="#1D4ED8",
        )

    axis.set_title(
        "BuildProgress | Exterior Boundary Topology"
    )

    axis.set_xlabel("CAD X (m)")
    axis.set_ylabel("CAD Y (m)")
    axis.set_aspect(
        "equal", adjustable="box"
    )
    axis.grid(alpha=0.15)

    if endpoints:
        axis.legend()

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def inspect(segments):
    adjacency, incident = build_graph(
        segments
    )

    components = connected_components(
        adjacency
    )

    degree_counts = Counter(
        len(neighbors)
        for neighbors in adjacency.values()
    )

    print("\nEXTERIOR TOPOLOGY INSPECTION")
    print("-" * 52)

    print("Segments:", len(segments))
    print("Unique nodes:", len(adjacency))
    print("Connected components:", len(components))
    print("Degree distribution:", dict(degree_counts))

    print(
        "Degree-1 endpoints:",
        sum(
            len(neighbors) == 1
            for neighbors in adjacency.values()
        ),
    )


def build(segments, args):
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    adjacency, incident = build_graph(
        segments
    )

    components = connected_components(
        adjacency
    )

    endpoints = [
        point
        for point, neighbors in adjacency.items()
        if len(neighbors) == 1
    ]

    candidates = endpoint_candidates(
        endpoints,
        args.max_gap,
        args.min_gap,
    )

    candidates.extend(
        endpoint_segment_candidates(
            endpoints,
            segments,
            incident,
            args.max_gap,
            args.min_gap,
        )
    )

    candidates = deduplicate_candidates(
        candidates
    )

    report = {
        "source_segments": len(segments),
        "unique_nodes": len(adjacency),
        "connected_components": len(components),
        "component_sizes": sorted(
            [len(c) for c in components],
            reverse=True,
        ),
        "degree_distribution": dict(
            Counter(
                len(neighbors)
                for neighbors in adjacency.values()
            )
        ),
        "degree_1_endpoints": len(endpoints),
        "gap_candidates": len(candidates),
        "endpoint_endpoint_candidates": sum(
            item["type"] == "endpoint_to_endpoint"
            for item in candidates
        ),
        "endpoint_segment_candidates": sum(
            item["type"] == "endpoint_to_segment"
            for item in candidates
        ),
        "parameters": {
            "min_gap_m": args.min_gap,
            "max_gap_m": args.max_gap,
        },
        "notes": [
            "No CAD geometry has been modified.",
            "Gap candidates are not confirmed errors.",
            "Some gaps may be intentional openings.",
            "Degree counts are based on rounded endpoints.",
            "Interior intersections are not automatically noded.",
        ],
    }

    candidates_path = (
        args.output / "gap_candidates.json"
    )

    report_path = (
        args.output / "topology_report.json"
    )

    preview_path = (
        args.output / "topology_preview.png"
    )

    candidates_path.write_text(
        json.dumps(candidates, indent=2),
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    draw_preview(
        segments,
        endpoints,
        candidates,
        preview_path,
    )

    print("\nEXTERIOR TOPOLOGY AUDIT COMPLETE")
    print("-" * 52)

    for key, value in report.items():
        if key not in ("notes", "component_sizes"):
            print(f"{key}: {value}")

    print("\nOutputs:")
    print(candidates_path)
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
        "--min-gap",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.30,
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

    if not (
        0 <= args.min_gap < args.max_gap
    ):
        parser.error(
            "Require 0 <= min-gap < max-gap."
        )

    if not (args.inspect or args.build):
        parser.error(
            "Specify --inspect or --build."
        )

    segments = load_exterior(args.input)

    if args.inspect:
        inspect(segments)

    if args.build:
        build(segments, args)


if __name__ == "__main__":
    main()

