
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]

FACADE_FILE = ROOT / "outputs/cad_facade/facade_geometry.json"

OUTPUT_DIR = ROOT / "outputs/cad_facade"
OUTPUT_JSON = OUTPUT_DIR / "facade_runs.json"
OUTPUT_IMAGE = OUTPUT_DIR / "facade_runs_preview.png"


# Geometric grouping thresholds in CAD metres.
# These are provisional algorithm parameters,
# NOT verified construction tolerances.
MIN_SEGMENT_LENGTH_M = 0.20
MAX_DIRECTION_DIFFERENCE_DEG = 5.0
MAX_LATERAL_OFFSET_M = 0.10
MAX_JOIN_GAP_M = 1.00

EPS = 1e-9


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def norm(a):
    return math.hypot(a[0], a[1])


def unit(a):
    length = norm(a)

    if length <= EPS:
        return None

    return (a[0] / length, a[1] / length)


def rounded_point(point):
    return [
        round(float(point[0]), 6),
        round(float(point[1]), 6),
    ]


def angle_difference(a, b):
    value = abs(dot(a, b))
    value = max(-1.0, min(1.0, value))
    return math.degrees(math.acos(value))


def projection(point, origin, axis):
    return dot(subtract(point, origin), axis)


def load_facade():
    if not FACADE_FILE.exists():
        raise FileNotFoundError(
            f"Missing CAD façade geometry: {FACADE_FILE}"
        )

    data = json.loads(
        FACADE_FILE.read_text(encoding="utf-8")
    )

    if not isinstance(data.get("paths"), list):
        raise ValueError(
            "facade_geometry.json must contain a 'paths' list."
        )

    return data


def extract_curtain_segments(facade_data):
    """
    Extract actual straight edges from original
    CURTAIN_WALL CAD paths.

    No window positions are used to invent geometry.
    """
    segments = []
    skipped = Counter()
    source_paths = 0

    for path_index, path in enumerate(facade_data["paths"]):
        if path.get("category") != "CURTAIN_WALL":
            continue

        source_paths += 1
        points = path.get("points", [])

        if len(points) < 2:
            skipped["PATH_WITH_FEWER_THAN_TWO_POINTS"] += 1
            continue

        source_handle = path.get("source_handle")
        source_layer = path.get("source_layer")
        path_id = path.get("id", f"path_{path_index}")

        for edge_index in range(len(points) - 1):
            a = points[edge_index]
            b = points[edge_index + 1]

            try:
                start = [float(a[0]), float(a[1])]
                end = [float(b[0]), float(b[1])]
            except (IndexError, TypeError, ValueError):
                skipped["INVALID_POINT"] += 1
                continue

            if not all(
                math.isfinite(v)
                for v in (*start, *end)
            ):
                skipped["NONFINITE_POINT"] += 1
                continue

            vector = subtract(end, start)
            length = norm(vector)

            if length < MIN_SEGMENT_LENGTH_M:
                skipped["SHORT_SOURCE_EDGE"] += 1
                continue

            direction = unit(vector)

            segments.append({
                "id": f"curtain_edge_{len(segments) + 1:05d}",
                "start": start,
                "end": end,
                "direction": direction,
                "length_m": length,
                "source_path_id": path_id,
                "source_handle": source_handle,
                "source_layer": source_layer,
                "source_edge_index": edge_index,
            })

    return segments, source_paths, skipped


def compatible(first, second):
    """
    Test whether two CAD source edges are approximately
    collinear and separated by at most the join threshold.

    Overlapping edges are allowed, because multiple
    CAD paths can represent the same façade alignment.
    """
    angle = angle_difference(
        first["direction"],
        second["direction"],
    )

    if angle > MAX_DIRECTION_DIFFERENCE_DEG:
        return False

    origin = first["start"]
    axis = first["direction"]

    lateral = max(
        abs(cross(
            axis,
            subtract(second["start"], origin),
        )),
        abs(cross(
            axis,
            subtract(second["end"], origin),
        )),
    )

    if lateral > MAX_LATERAL_OFFSET_M:
        return False

    first_interval = sorted([
        projection(first["start"], origin, axis),
        projection(first["end"], origin, axis),
    ])

    second_interval = sorted([
        projection(second["start"], origin, axis),
        projection(second["end"], origin, axis),
    ])

    gap = max(
        first_interval[0] - second_interval[1],
        second_interval[0] - first_interval[1],
        0.0,
    )

    return gap <= MAX_JOIN_GAP_M


def connected_groups(segments):
    """
    Initial connected-component grouping.

    Each edge remains traceable to its CAD source.
    """
    count = len(segments)
    parent = list(range(count))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a = find(a)
        b = find(b)

        if a != b:
            parent[b] = a

    for i in range(count):
        for j in range(i + 1, count):
            if compatible(segments[i], segments[j]):
                union(i, j)

    groups = {}

    for index, segment in enumerate(segments):
        root = find(index)
        groups.setdefault(root, []).append(segment)

    return list(groups.values())


def split_nonstraight_groups(groups):
    """
    Prevent connected-component chaining from merging
    a bent perimeter into a single straight run.

    Every accepted edge must align with the selected
    reference edge, not merely its immediate neighbor.
    """
    output = []

    for group in groups:
        remaining = sorted(
            group,
            key=lambda item: item["id"],
        )

        while remaining:
            anchor = max(
                remaining,
                key=lambda item: item["length_m"],
            )

            axis = anchor["direction"]
            origin = anchor["start"]

            accepted = []
            deferred = []

            for segment in remaining:
                angle = angle_difference(
                    axis,
                    segment["direction"],
                )

                lateral = max(
                    abs(cross(
                        axis,
                        subtract(segment["start"], origin),
                    )),
                    abs(cross(
                        axis,
                        subtract(segment["end"], origin),
                    )),
                )

                if (
                    angle <= MAX_DIRECTION_DIFFERENCE_DEG
                    and lateral <= MAX_LATERAL_OFFSET_M
                ):
                    accepted.append(segment)
                else:
                    deferred.append(segment)

            output.append(accepted)
            remaining = deferred

    return output


def create_run(group, index):
    anchor = max(
        group,
        key=lambda item: item["length_m"],
    )

    axis = anchor["direction"]

    # Stable orientation for deterministic JSON output.
    if (
        axis[0] < -EPS
        or (abs(axis[0]) <= EPS and axis[1] < 0)
    ):
        axis = (-axis[0], -axis[1])

    origin = anchor["start"]
    ordered = []

    for segment in group:
        start = segment["start"]
        end = segment["end"]

        t_start = projection(start, origin, axis)
        t_end = projection(end, origin, axis)

        if t_start > t_end:
            start, end = end, start
            t_start, t_end = t_end, t_start

        ordered.append({
            **segment,
            "ordered_start": start,
            "ordered_end": end,
            "t_start": t_start,
            "t_end": t_end,
        })

    ordered.sort(
        key=lambda item: (
            item["t_start"],
            item["t_end"],
            item["id"],
        )
    )

    source_segments = []

    for segment in ordered:
        source_segments.append({
            "segment_id": segment["id"],
            "start": rounded_point(
                segment["ordered_start"]
            ),
            "end": rounded_point(
                segment["ordered_end"]
            ),
            "length_m": round(
                segment["length_m"], 6
            ),
            "source_path_id": segment["source_path_id"],
            "source_handle": segment["source_handle"],
            "source_layer": segment["source_layer"],
            "source_edge_index": segment["source_edge_index"],
        })

    # Track covered intervals to distinguish actual
    # source geometry from unsupported longitudinal gaps.
    intervals = []

    for segment in ordered:
        start_t = segment["t_start"]
        end_t = segment["t_end"]

        if not intervals or start_t > intervals[-1][1] + EPS:
            intervals.append([start_t, end_t])
        else:
            intervals[-1][1] = max(
                intervals[-1][1],
                end_t,
            )

    unverified_gaps = []

    for first, second in zip(intervals, intervals[1:]):
        gap_start_t = first[1]
        gap_end_t = second[0]

        if gap_end_t <= gap_start_t + EPS:
            continue

        start = [
            origin[0] + axis[0] * gap_start_t,
            origin[1] + axis[1] * gap_start_t,
        ]

        end = [
            origin[0] + axis[0] * gap_end_t,
            origin[1] + axis[1] * gap_end_t,
        ]

        unverified_gaps.append({
            "start": rounded_point(start),
            "end": rounded_point(end),
            "length_m": round(
                gap_end_t - gap_start_t, 6
            ),
            "status": "UNVERIFIED_GAP",
            "safe_to_fill": False,
        })

    # Endpoints are projections for run metadata only.
    # The original source edges remain unchanged.
    minimum_t = min(
        item["t_start"] for item in ordered
    )
    maximum_t = max(
        item["t_end"] for item in ordered
    )

    run_start = [
        origin[0] + axis[0] * minimum_t,
        origin[1] + axis[1] * minimum_t,
    ]

    run_end = [
        origin[0] + axis[0] * maximum_t,
        origin[1] + axis[1] * maximum_t,
    ]

    return {
        "run_id": f"facade_run_{index:03d}",
        "representation": "COLLINEAR_CURTAIN_WALL_CAD_EDGES",
        "start": rounded_point(run_start),
        "end": rounded_point(run_end),
        "direction": [
            round(axis[0], 8),
            round(axis[1], 8),
        ],
        "source_segment_count": len(ordered),
        "source_segments": source_segments,
        "source_handles": sorted({
            str(segment["source_handle"])
            for segment in ordered
            if segment["source_handle"] is not None
        }),
        "source_layers": sorted({
            str(segment["source_layer"])
            for segment in ordered
            if segment["source_layer"] is not None
        }),
        "unverified_gaps": unverified_gaps,
        "gap_count": len(unverified_gaps),
        "continuous_physical_facade_verified": False,
        "vertical_dimensions_verified": False,
        "safe_to_generate_solid": False,
        "construction_status": "UNKNOWN",
    }


def make_preview(runs, facade_data):
    fig, ax = plt.subplots(
        figsize=(16, 11),
        constrained_layout=True,
    )

    original_segments = []

    for path in facade_data["paths"]:
        if path.get("category") != "CURTAIN_WALL":
            continue

        points = path.get("points", [])

        for i in range(len(points) - 1):
            original_segments.append([
                points[i],
                points[i + 1],
            ])

    if original_segments:
        ax.add_collection(
            LineCollection(
                original_segments,
                colors="#b5a3e7",
                linewidths=0.8,
                alpha=0.45,
            )
        )

    colors = plt.get_cmap("tab20")
    gaps = []

    for index, run in enumerate(runs):
        segments = [
            [
                segment["start"],
                segment["end"],
            ]
            for segment in run["source_segments"]
        ]

        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=[colors(index % 20)],
                    linewidths=2.2,
                    alpha=0.95,
                )
            )

        for gap in run["unverified_gaps"]:
            gaps.append([
                gap["start"],
                gap["end"],
            ])

    if gaps:
        ax.add_collection(
            LineCollection(
                gaps,
                colors="#ef4444",
                linewidths=1.4,
                linestyles="dashed",
            )
        )

    ax.set_title(
        "Curtain-wall-driven façade runs\n"
        "Colored: source-supported CAD edges | "
        "Red dashed: unverified gaps | "
        "Purple: original curtain-wall geometry"
    )

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.05)
    ax.grid(alpha=0.15)
    ax.set_xlabel("CAD X (metres)")
    ax.set_ylabel("CAD Y (metres)")

    fig.savefig(
        OUTPUT_IMAGE,
        dpi=180,
        facecolor="white",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    facade_data = load_facade()

    segments, source_paths, skipped = (
        extract_curtain_segments(facade_data)
    )

    if not segments:
        raise RuntimeError(
            "No usable CURTAIN_WALL source edges were found. "
            "Check facade_geometry.json before continuing."
        )

    initial_groups = connected_groups(segments)
    groups = split_nonstraight_groups(initial_groups)

    groups.sort(
        key=lambda group: (
            -sum(item["length_m"] for item in group),
            min(item["id"] for item in group),
        )
    )

    runs = [
        create_run(group, index)
        for index, group in enumerate(groups, 1)
    ]

    total_gaps = sum(
        run["gap_count"] for run in runs
    )

    report = {
        "schema_version": "2.0",
        "coordinate_system": "DXF_METERS",
        "method": "CURTAIN_WALL_COLLINEAR_SOURCE_SEGMENTS",
        "source_facade": str(
            FACADE_FILE.relative_to(ROOT)
        ),
        "thresholds": {
            "min_segment_length_m":
                MIN_SEGMENT_LENGTH_M,
            "max_direction_difference_degrees":
                MAX_DIRECTION_DIFFERENCE_DEG,
            "max_lateral_offset_m":
                MAX_LATERAL_OFFSET_M,
            "max_join_gap_m":
                MAX_JOIN_GAP_M,
        },
        "curtain_wall_source_paths": source_paths,
        "usable_source_segments": len(segments),
        "skipped_source_edges": dict(skipped),
        "run_count": len(runs),
        "unverified_gap_count": total_gaps,
        "physical_facade_run_count_verified": False,
        "vertical_dimensions_verified": False,
        "solid_facade_geometry_generated": False,
        "wall_cutouts_generated": False,
        "construction_progress_verified": False,
        "runs": runs,
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    make_preview(runs, facade_data)

    distribution = Counter(
        run["source_segment_count"]
        for run in runs
    )

    print("\n=== CURTAIN-WALL FACADE RUNS ===")
    print("Original curtain-wall paths:", source_paths)
    print("Usable source edges:", len(segments))
    print("Skipped edges:", dict(skipped))
    print("Provisional run groups:", len(runs))
    print("Unverified gaps:", total_gaps)
    print(
        "Run-size distribution:",
        dict(sorted(distribution.items())),
    )

    print("\nLargest runs:")

    for run in runs[:10]:
        print(
            f"  {run['run_id']}: "
            f"{run['source_segment_count']} source edges, "
            f"{run['gap_count']} gaps"
        )

    print("\nPhysical façade runs verified: NO")
    print("Solid façade geometry generated: NO")
    print("Wall cutouts generated: NO")
    print("JSON:", OUTPUT_JSON.relative_to(ROOT))
    print("Preview:", OUTPUT_IMAGE.relative_to(ROOT))


if __name__ == "__main__":
    main()
