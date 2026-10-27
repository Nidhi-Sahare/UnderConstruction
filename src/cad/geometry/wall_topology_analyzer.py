
"""
BuildProgress - Wall Topology Analyzer

Diagnoses DXF wall linework before reconstruction.

Outputs:
    data/geometry/topology_report.json
    data/geometry/topology_diagnostic.png

Run:
    python wall_topology_analyzer.py

Dependencies:
    ezdxf, shapely, matplotlib

Notes:
    - Coordinates are in meters.
    - Does not change the source DXF.
    - Does not close gaps or infer wall regions.
    - Parallel-edge distances are diagnostic estimates.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ezdxf.path import make_path
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

OUTPUT_DIR = ROOT / "data" / "geometry"
REPORT_PATH = OUTPUT_DIR / "topology_report.json"
IMAGE_PATH = OUTPUT_DIR / "topology_diagnostic.png"

WALL_LAYERS = {
    "010 AussenwÃ¤nde": "exterior",
    "015 InnenwÃ¤nde": "interior",
    "016 TrennwÃ¤nde": "partition",
    "019 Vormauerungen": "lining",
}

# Diagnostic tolerances in meters/degrees.
# They do not modify source geometry.
CURVE_FLATTENING_M = 0.01
ENDPOINT_NEIGHBOR_RADIUS_M = 0.10
ENDPOINT_COINCIDENCE_M = 0.001
PARALLEL_ANGLE_TOLERANCE_DEG = 3.0
PARALLEL_SEARCH_RADIUS_M = 0.60
MIN_PARALLEL_SEGMENT_LENGTH_M = 0.30
MIN_PARALLEL_OVERLAP_M = 0.20

# Avoid excessive diagnostic output on complex drawings.
MAX_EXAMPLES = 30
MAX_PLOT_ENDPOINTS = 2500


def xy(point):
    return (float(point[0]), float(point[1]))


def segment_records(entity):
    """
    Convert supported CAD geometry into primitive segments.

    For arcs and curved polylines, flatten to short
    line segments for diagnostic purposes.
    """

    kind = entity.dxftype()

    if kind == "LINE":
        return [(xy(entity.dxf.start), xy(entity.dxf.end))]

    if kind in ("LWPOLYLINE", "POLYLINE"):
        result = []

        for child in entity.virtual_entities():
            result.extend(segment_records(child))

        return result

    if kind in ("ARC", "CIRCLE"):
        path = make_path(entity)

        vertices = [
            xy(point)
            for point in path.flattening(
                distance=CURVE_FLATTENING_M
            )
        ]

        return list(zip(vertices, vertices[1:]))

    return []


def segment_length(start, end):
    return math.hypot(
        end[0] - start[0],
        end[1] - start[1],
    )


def segment_orientation(start, end):
    """
    Undirected orientation in [0, 180) degrees.
    """

    angle = math.degrees(
        math.atan2(
            end[1] - start[1],
            end[0] - start[0],
        )
    )

    return angle % 180.0


def orientation_difference(a, b):
    difference = abs(a - b) % 180.0
    return min(difference, 180.0 - difference)


def load_segments():
    document = ezdxf.readfile(DXF_PATH)

    if document.header.get("$INSUNITS", 0) != 6:
        raise ValueError(
            "This diagnostic expects DXF coordinates in meters."
        )

    segments = []
    layer_statistics = defaultdict(Counter)
    insert_examples = []

    for entity in document.modelspace():
        layer = entity.dxf.get("layer", "0")

        if layer not in WALL_LAYERS:
            continue

        category = WALL_LAYERS[layer]
        kind = entity.dxftype()
        handle = str(entity.dxf.get("handle", ""))

        layer_statistics[category][f"entity_{kind}"] += 1

        if kind in ("LWPOLYLINE", "POLYLINE"):
            key = (
                "closed_polylines"
                if entity.is_closed
                else "open_polylines"
            )
            layer_statistics[category][key] += 1

        if kind == "INSERT":
            if len(insert_examples) < MAX_EXAMPLES:
                insert_examples.append({
                    "handle": handle,
                    "layer": layer,
                    "block_name": entity.dxf.name,
                    "insert": list(entity.dxf.insert),
                })

            layer_statistics[category]["insert_count"] += 1
            continue

        for start, end in segment_records(entity):
            length = segment_length(start, end)

            if length < 1e-8:
                continue

            segments.append({
                "start": start,
                "end": end,
                "length_m": length,
                "orientation_deg": segment_orientation(
                    start, end
                ),
                "layer": layer,
                "category": category,
                "handle": handle,
            })

    return segments, layer_statistics, insert_examples


def summarize_lengths(segments):
    values = np.asarray(
        [s["length_m"] for s in segments],
        dtype=float,
    )

    if values.size == 0:
        return {}

    return {
        "count": int(values.size),
        "min_m": float(values.min()),
        "median_m": float(np.median(values)),
        "p90_m": float(np.percentile(values, 90)),
        "max_m": float(values.max()),
    }


def analyze_endpoints(segments):
    """
    Find endpoints with no other endpoint within
    coincidence tolerance, then inspect nearby gaps.

    Important:
        This is endpoint proximity, not a full
        graph-topology or intersection analysis.
    """

    if not segments:
        return {}, []

    endpoint_coordinates = []

    for segment in segments:
        endpoint_coordinates.extend([
            segment["start"],
            segment["end"],
        ])

    points = [
        Point(point)
        for point in endpoint_coordinates
    ]

    tree = STRtree(points)

    unmatched = []
    nearby_gaps = []

    for index, point in enumerate(points):
        candidate_indices = tree.query(
            point.buffer(ENDPOINT_NEIGHBOR_RADIUS_M)
        )

        distances = []

        for candidate_index in candidate_indices:
            candidate_index = int(candidate_index)

            if candidate_index == index:
                continue

            distance = point.distance(
                points[candidate_index]
            )

            distances.append(distance)

        if any(
            distance <= ENDPOINT_COINCIDENCE_M
            for distance in distances
        ):
            continue

        unmatched.append(endpoint_coordinates[index])

        nonzero = [
            distance
            for distance in distances
            if distance > ENDPOINT_COINCIDENCE_M
        ]

        if nonzero:
            nearby_gaps.append(min(nonzero))

    statistics = {
        "total_segment_endpoints": len(points),
        "unmatched_endpoints": len(unmatched),
        "nearby_gap_count": len(nearby_gaps),
        "median_nearby_gap_m": (
            float(np.median(nearby_gaps))
            if nearby_gaps else None
        ),
        "smallest_nearby_gaps_m": sorted(
            nearby_gaps
        )[:MAX_EXAMPLES],
    }

    return statistics, unmatched


def projected_overlap(a, b):
    """
    Estimate overlap along segment a's direction.
    """

    ax, ay = a["start"]
    bx, by = a["end"]

    dx = bx - ax
    dy = by - ay

    length = math.hypot(dx, dy)

    if length <= 0:
        return 0.0

    ux, uy = dx / length, dy / length

    projections = [
        (p[0] - ax) * ux + (p[1] - ay) * uy
        for p in (b["start"], b["end"])
    ]

    low = max(0.0, min(projections))
    high = min(length, max(projections))

    return max(0.0, high - low)


def analyze_parallel_edges(segments):
    """
    Identify nearby parallel segment pairs.

    Distances are not confirmed wall thicknesses.
    """

    eligible = [
        segment
        for segment in segments
        if segment["length_m"]
        >= MIN_PARALLEL_SEGMENT_LENGTH_M
    ]

    lines = [
        LineString([s["start"], s["end"]])
        for s in eligible
    ]

    if not lines:
        return {}

    tree = STRtree(lines)

    distances = []
    seen_pairs = set()

    for index, line in enumerate(lines):
        candidate_indices = tree.query(
            line.buffer(PARALLEL_SEARCH_RADIUS_M)
        )

        for candidate_index in candidate_indices:
            other_index = int(candidate_index)

            if other_index == index:
                continue

            pair = tuple(sorted((index, other_index)))

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            a = eligible[index]
            b = eligible[other_index]

            if a["category"] != b["category"]:
                continue

            angle_difference = orientation_difference(
                a["orientation_deg"],
                b["orientation_deg"],
            )

            if angle_difference > PARALLEL_ANGLE_TOLERANCE_DEG:
                continue

            overlap = min(
                projected_overlap(a, b),
                projected_overlap(b, a),
            )

            if overlap < MIN_PARALLEL_OVERLAP_M:
                continue

            distance = line.distance(lines[other_index])

            if (
                distance <= ENDPOINT_COINCIDENCE_M
                or distance > PARALLEL_SEARCH_RADIUS_M
            ):
                continue

            distances.append({
                "distance_m": float(distance),
                "category": a["category"],
                "overlap_m": float(overlap),
                "handles": [a["handle"], b["handle"]],
            })

    values = np.asarray(
        [item["distance_m"] for item in distances],
        dtype=float,
    )

    return {
        "pair_count": len(distances),
        "distance_median_m": (
            float(np.median(values))
            if values.size else None
        ),
        "distance_percentiles_m": (
            {
                "p10": float(np.percentile(values, 10)),
                "p25": float(np.percentile(values, 25)),
                "p50": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)),
                "p90": float(np.percentile(values, 90)),
            }
            if values.size else {}
        ),
        "examples": sorted(
            distances,
            key=lambda item: item["distance_m"],
        )[:MAX_EXAMPLES],
    }


def save_diagnostic_plot(segments, unmatched):
    fig, ax = plt.subplots(
        figsize=(14, 10),
        dpi=150,
    )

    by_category = defaultdict(list)

    for segment in segments:
        by_category[segment["category"]].append([
            segment["start"],
            segment["end"],
        ])

    colors = {
        "exterior": "#2563EB",
        "interior": "#E59A37",
        "partition": "#D65D8A",
        "lining": "#A16207",
    }

    for category, lines in by_category.items():
        ax.add_collection(
            LineCollection(
                lines,
                colors=colors.get(category, "#64748B"),
                linewidths=0.8,
                alpha=0.8,
                label=category,
            )
        )

    if unmatched:
        sample = unmatched[:MAX_PLOT_ENDPOINTS]

        ax.scatter(
            [p[0] for p in sample],
            [p[1] for p in sample],
            s=6,
            color="#DC2626",
            alpha=0.7,
            label="Unmatched endpoints (sample)",
            zorder=3,
        )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    ax.set_title(
        "BuildProgress | Floor 2 - "
        "Wall Topology Diagnostics"
    )

    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.1)

    fig.savefig(
        IMAGE_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DXF_PATH.exists():
        raise FileNotFoundError(
            f"Missing DXF: {DXF_PATH}"
        )

    print("Analyzing Floor 2 wall topology...")

    segments, layer_stats, inserts = load_segments()

    length_stats = summarize_lengths(segments)

    endpoint_stats, unmatched = analyze_endpoints(
        segments
    )

    parallel_stats = analyze_parallel_edges(
        segments
    )

    report = {
        "schema_version": "wall-topology-1.0",
        "source_dxf": str(DXF_PATH),
        "units": "meters",
        "parameters": {
            "curve_flattening_m": CURVE_FLATTENING_M,
            "endpoint_neighbor_radius_m":
                ENDPOINT_NEIGHBOR_RADIUS_M,
            "endpoint_coincidence_m":
                ENDPOINT_COINCIDENCE_M,
            "parallel_angle_tolerance_deg":
                PARALLEL_ANGLE_TOLERANCE_DEG,
            "parallel_search_radius_m":
                PARALLEL_SEARCH_RADIUS_M,
            "minimum_parallel_overlap_m":
                MIN_PARALLEL_OVERLAP_M,
        },
        "segment_count": len(segments),
        "length_statistics": length_stats,
        "endpoint_statistics": endpoint_stats,
        "parallel_edge_statistics": parallel_stats,
        "layer_statistics": {
            key: dict(value)
            for key, value in layer_stats.items()
        },
        "insert_examples": inserts,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    save_diagnostic_plot(segments, unmatched)

    print("\n" + "=" * 65)
    print("BUILDPROGRESS - TOPOLOGY ANALYSIS")
    print("=" * 65)

    print(f"Primitive segments: {len(segments)}")

    print("\nLINE LENGTHS")
    print(length_stats)

    print("\nENDPOINT CONNECTIVITY")
    print(endpoint_stats)

    print("\nPARALLEL EDGE ANALYSIS")
    print(
        "Candidate pairs:",
        parallel_stats.get("pair_count", 0),
    )
    print(
        "Median separation:",
        parallel_stats.get("distance_median_m"),
    )
    print(
        "Distance percentiles:",
        parallel_stats.get("distance_percentiles_m"),
    )

    print("\nWALL-LAYER BLOCK REFERENCES")
    print(f"INSERT examples: {len(inserts)}")

    print("\nSaved:")
    print(REPORT_PATH)
    print(IMAGE_PATH)

    print(
        "\nDiagnostic only: no walls were "
        "reconstructed or modified."
    )


if __name__ == "__main__":
    main()

