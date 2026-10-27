
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

WINDOW_FILE = (
    ROOT / "outputs/cad_openings/windows/window_assemblies.json"
)

VERIFICATION_FILE = (
    ROOT / "outputs/cad_facade/window_facade_verification.json"
)

FACADE_FILE = (
    ROOT / "outputs/cad_facade/facade_geometry.json"
)

OUTPUT_DIR = ROOT / "outputs/cad_facade"
OUTPUT_JSON = OUTPUT_DIR / "facade_runs.json"
OUTPUT_IMAGE = OUTPUT_DIR / "facade_runs_preview.png"


# Provisional grouping thresholds, in CAD metres.
# These are NOT construction tolerances.
MAX_DIRECTION_DIFFERENCE_DEG = 6.0
MAX_LATERAL_OFFSET_M = 0.12
MAX_JOIN_GAP_M = 0.65
MAX_JOIN_OVERLAP_M = 0.15

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
    if length < EPS:
        return None
    return (a[0] / length, a[1] / length)


def direction_angle(a, b):
    value = min(1.0, max(-1.0, abs(dot(a, b))))
    return math.degrees(math.acos(value))


def project(point, origin, direction):
    return dot(subtract(point, origin), direction)


def midpoint(a, b):
    return [
        (a[0] + b[0]) / 2.0,
        (a[1] + b[1]) / 2.0,
    ]


def rounded_point(point):
    return [
        round(float(point[0]), 6),
        round(float(point[1]), 6),
    ]


def load_inputs():
    for path in (
        WINDOW_FILE,
        VERIFICATION_FILE,
        FACADE_FILE,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing prerequisite: {path}"
            )

    windows_data = json.loads(
        WINDOW_FILE.read_text(encoding="utf-8")
    )
    verification_data = json.loads(
        VERIFICATION_FILE.read_text(encoding="utf-8")
    )
    facade_data = json.loads(
        FACADE_FILE.read_text(encoding="utf-8")
    )

    return (
        windows_data,
        verification_data,
        facade_data,
    )


def build_supported_spans(windows_data, verification_data):
    verification = {
        result["window_id"]: result
        for result in verification_data["results"]
    }

    spans = []
    rejected = []

    for item in windows_data["assemblies"]:
        if (
            item.get("status")
            != "PROVISIONAL_STRAIGHT_ASSEMBLY"
        ):
            continue

        identifier = item["assembly_id"]
        result = verification.get(identifier)

        if result is None:
            rejected.append({
                "id": identifier,
                "reason": "MISSING_SCREENING_RESULT",
            })
            continue

        if (
            result.get("status")
            != "GEOMETRICALLY_SUPPORTED_FACADE_REGION"
        ):
            rejected.append({
                "id": identifier,
                "reason": result.get("status", "UNKNOWN"),
            })
            continue

        start = item["start"]
        end = item["end"]

        direction = unit(subtract(end, start))

        if direction is None:
            rejected.append({
                "id": identifier,
                "reason": "DEGENERATE_SPAN",
            })
            continue

        selected = result.get("selected_region") or {}

        spans.append({
            "id": identifier,
            "start": start,
            "end": end,
            "center": midpoint(start, end),
            "direction": direction,
            "length_m": norm(subtract(end, start)),
            "source_window_handles": item.get(
                "source_handles", []
            ),
            "source_facade_handles": selected.get(
                "source_handles", []
            ),
            "source_facade_layers": selected.get(
                "source_layers", []
            ),
        })

    return spans, rejected


def pair_geometry(first, second):
    """
    Compare two window spans as undirected lines.

    A join is allowed only when the spans are
    parallel, laterally close, and separated by
    a small longitudinal gap.

    This does not assert that the gap contains
    actual glazing or a physical frame.
    """
    axis = first["direction"]

    angle = direction_angle(
        first["direction"],
        second["direction"],
    )

    if angle > MAX_DIRECTION_DIFFERENCE_DEG:
        return None

    origin = first["start"]

    first_interval = sorted([
        project(first["start"], origin, axis),
        project(first["end"], origin, axis),
    ])

    second_interval = sorted([
        project(second["start"], origin, axis),
        project(second["end"], origin, axis),
    ])

    lateral = max(
        abs(cross(axis, subtract(second["start"], origin))),
        abs(cross(axis, subtract(second["end"], origin))),
    )

    if lateral > MAX_LATERAL_OFFSET_M:
        return None

    gap = max(
        first_interval[0] - second_interval[1],
        second_interval[0] - first_interval[1],
        0.0,
    )

    overlap = max(
        0.0,
        min(first_interval[1], second_interval[1])
        - max(first_interval[0], second_interval[0]),
    )

    if gap > MAX_JOIN_GAP_M:
        return None

    if overlap > MAX_JOIN_OVERLAP_M:
        return None

    return {
        "gap_m": gap,
        "overlap_m": overlap,
        "angle_degrees": angle,
        "lateral_offset_m": lateral,
    }


def build_groups(spans):
    """
    Connected components of geometrically compatible
    window spans.

    Each resulting group is checked again for overall
    straightness to prevent transitive grouping from
    creating an artificially bent façade run.
    """
    count = len(spans)
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
            if pair_geometry(spans[i], spans[j]) is not None:
                union(i, j)

    groups = {}

    for index, span in enumerate(spans):
        root = find(index)
        groups.setdefault(root, []).append(span)

    return list(groups.values())


def split_nonstraight_group(group):
    """
    Greedily form groups with a common reference axis.

    This guards against chaining small angular or
    lateral differences into a curved run.
    """
    remaining = sorted(
        group,
        key=lambda item: item["id"],
    )

    output = []

    while remaining:
        anchor = max(
            remaining,
            key=lambda item: item["length_m"],
        )

        axis = anchor["direction"]
        origin = anchor["start"]

        accepted = []
        deferred = []

        for span in remaining:
            angle = direction_angle(
                axis,
                span["direction"],
            )

            lateral = max(
                abs(cross(
                    axis,
                    subtract(span["start"], origin),
                )),
                abs(cross(
                    axis,
                    subtract(span["end"], origin),
                )),
            )

            if (
                angle <= MAX_DIRECTION_DIFFERENCE_DEG
                and lateral <= MAX_LATERAL_OFFSET_M
            ):
                accepted.append(span)
            else:
                deferred.append(span)

        output.append(accepted)
        remaining = deferred

    return output


def create_run(group, index):
    anchor = max(
        group,
        key=lambda item: item["length_m"],
    )

    axis = anchor["direction"]

    # Deterministic orientation for stable output.
    if (
        axis[0] < -EPS
        or (abs(axis[0]) <= EPS and axis[1] < 0)
    ):
        axis = (-axis[0], -axis[1])

    origin = anchor["start"]

    ordered = []

    for span in group:
        start_t = project(
            span["start"],
            origin,
            axis,
        )

        end_t = project(
            span["end"],
            origin,
            axis,
        )

        a = span["start"]
        b = span["end"]

        if start_t > end_t:
            a, b = b, a
            start_t, end_t = end_t, start_t

        ordered.append({
            **span,
            "ordered_start": a,
            "ordered_end": b,
            "t_start": start_t,
            "t_end": end_t,
        })

    ordered.sort(
        key=lambda item: (
            item["t_start"],
            item["id"],
        )
    )

    # Preserve actual CAD span geometry.
    supported_segments = [
        {
            "window_id": span["id"],
            "start": rounded_point(span["ordered_start"]),
            "end": rounded_point(span["ordered_end"]),
            "length_m": round(span["length_m"], 6),
        }
        for span in ordered
    ]

    gaps = []

    for first, second in zip(
        ordered,
        ordered[1:],
    ):
        gap = second["t_start"] - first["t_end"]

        if gap > EPS:
            gaps.append({
                "from_window": first["id"],
                "to_window": second["id"],
                "start": rounded_point(first["ordered_end"]),
                "end": rounded_point(second["ordered_start"]),
                "length_m": round(
                    norm(subtract(
                        second["ordered_start"],
                        first["ordered_end"],
                    )),
                    6,
                ),
                "longitudinal_gap_m": round(gap, 6),
                "status": "UNVERIFIED_GAP",
                "safe_to_fill": False,
            })

    start = ordered[0]["ordered_start"]
    end = max(
        ordered,
        key=lambda item: item["t_end"],
    )["ordered_end"]

    return {
        "run_id": f"facade_run_{index:03d}",
        "representation": "GROUPED_WINDOW_SUPPORTED_CAD_SPANS",
        "start": rounded_point(start),
        "end": rounded_point(end),
        "direction": [
            round(axis[0], 8),
            round(axis[1], 8),
        ],
        "window_span_count": len(ordered),
        "window_ids": [
            item["id"] for item in ordered
        ],
        "source_window_handles": sorted({
            handle
            for item in ordered
            for handle in item["source_window_handles"]
        }),
        "source_facade_handles": sorted({
            handle
            for item in ordered
            for handle in item["source_facade_handles"]
        }),
        "source_facade_layers": sorted({
            layer
            for item in ordered
            for layer in item["source_facade_layers"]
        }),
        "supported_segments": supported_segments,
        "unverified_gaps": gaps,
        "gap_count": len(gaps),
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

    curtain_segments = []

    for path in facade_data["paths"]:
        if path.get("category") != "CURTAIN_WALL":
            continue

        points = path["points"]

        curtain_segments.extend(
            [points[i], points[i + 1]]
            for i in range(len(points) - 1)
        )

    if curtain_segments:
        ax.add_collection(
            LineCollection(
                curtain_segments,
                colors="#b5a3e7",
                linewidths=0.8,
                alpha=0.45,
            )
        )

    colors = plt.get_cmap("tab20")

    gap_segments = []

    for index, run in enumerate(runs):
        segments = [
            [segment["start"], segment["end"]]
            for segment in run["supported_segments"]
        ]

        if segments:
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=[colors(index % 20)],
                    linewidths=2.5,
                    alpha=0.95,
                )
            )

        for gap in run["unverified_gaps"]:
            gap_segments.append([
                gap["start"],
                gap["end"],
            ])

    if gap_segments:
        ax.add_collection(
            LineCollection(
                gap_segments,
                colors="#ef4444",
                linewidths=1.0,
                linestyles="dashed",
                alpha=0.9,
            )
        )

    ax.set_title(
        "Provisional façade runs\n"
        "Colored: grouped CAD-supported spans | "
        "Red dashed: unverified gaps | "
        "Purple: original curtain-wall paths"
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

    (
        windows_data,
        verification_data,
        facade_data,
    ) = load_inputs()

    spans, rejected = build_supported_spans(
        windows_data,
        verification_data,
    )

    initial_groups = build_groups(spans)

    groups = []

    for group in initial_groups:
        groups.extend(
            split_nonstraight_group(group)
        )

    # Larger runs first, with deterministic ordering.
    groups.sort(
        key=lambda group: (
            -len(group),
            min(item["id"] for item in group),
        )
    )

    runs = [
        create_run(group, index)
        for index, group in enumerate(groups, 1)
    ]

    total_gaps = sum(
        run["gap_count"]
        for run in runs
    )

    report = {
        "schema_version": "1.0",
        "coordinate_system": "DXF_METERS",
        "method": "COLLINEAR_ADJACENT_SUPPORTED_WINDOW_SPANS",
        "source_windows": str(
            WINDOW_FILE.relative_to(ROOT)
        ),
        "source_verification": str(
            VERIFICATION_FILE.relative_to(ROOT)
        ),
        "source_facade": str(
            FACADE_FILE.relative_to(ROOT)
        ),
        "thresholds": {
            "max_direction_difference_degrees":
                MAX_DIRECTION_DIFFERENCE_DEG,
            "max_lateral_offset_m":
                MAX_LATERAL_OFFSET_M,
            "max_join_gap_m":
                MAX_JOIN_GAP_M,
            "max_join_overlap_m":
                MAX_JOIN_OVERLAP_M,
        },
        "supported_window_spans": len(spans),
        "excluded_window_spans": len(rejected),
        "run_count": len(runs),
        "unverified_gap_count": total_gaps,
        "physical_facade_run_count_verified": False,
        "vertical_dimensions_verified": False,
        "solid_facade_geometry_generated": False,
        "wall_cutouts_generated": False,
        "construction_progress_verified": False,
        "excluded": rejected,
        "runs": runs,
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    make_preview(runs, facade_data)

    distribution = Counter(
        run["window_span_count"]
        for run in runs
    )

    print("\n=== FACADE RUN RECONSTRUCTION ===")
    print("Supported window spans:", len(spans))
    print("Excluded window spans:", len(rejected))
    print("Provisional run groups:", len(runs))
    print("Unverified gaps:", total_gaps)
    print("Run-size distribution:", dict(
        sorted(distribution.items())
    ))

    print("\nLargest runs:")

    for run in runs[:10]:
        print(
            f"  {run['run_id']}: "
            f"{run['window_span_count']} spans, "
            f"{run['gap_count']} gaps"
        )

    print("\nPhysical façade runs verified: NO")
    print("Solid façade geometry generated: NO")
    print("Wall cutouts generated: NO")
    print("JSON:", OUTPUT_JSON.relative_to(ROOT))
    print("Preview:", OUTPUT_IMAGE.relative_to(ROOT))


if __name__ == "__main__":
    main()
