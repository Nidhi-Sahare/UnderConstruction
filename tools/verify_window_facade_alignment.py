
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, box
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]

WINDOW_PATH = (
    ROOT / "outputs/cad_openings/windows/window_assemblies.json"
)

FACADE_PATH = (
    ROOT / "outputs/cad_facade/facade_geometry.json"
)

OUTPUT_DIR = ROOT / "outputs/cad_facade"
REPORT_PATH = OUTPUT_DIR / "window_facade_verification.json"
PREVIEW_PATH = OUTPUT_DIR / "window_facade_verification.png"


# All tolerances are provisional screening parameters.
SEARCH_RADIUS_M = 0.30
MAX_DISTANCE_M = 0.12
MAX_ANGLE_DEGREES = 10.0
MIN_OVERLAP_RATIO = 0.60

# Nearby CAD profiles describing one façade region.
REGION_OFFSET_TOLERANCE_M = 0.15
REGION_GAP_TOLERANCE_M = 0.12

# Two regions count as distinct only when their
# supporting geometry is spatially separated.
DISTINCT_REGION_SEPARATION_M = 0.25

MIN_SEGMENT_LENGTH_M = 0.025
MAX_REPORT_CANDIDATES = 5


def vec(a, b):
    return (b[0] - a[0], b[1] - a[1])


def magnitude(v):
    return math.hypot(v[0], v[1])


def unit(v):
    n = magnitude(v)
    return (v[0] / n, v[1] / n) if n > 1e-12 else None


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def angle_between(a, b):
    value = max(-1.0, min(1.0, abs(dot(a, b))))
    return math.degrees(math.acos(value))


def projection(point, origin, axis):
    return dot(vec(origin, point), axis)


def merge_intervals(intervals, gap=0.0):
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [list(intervals[0])]

    for start, end in intervals[1:]:
        if start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged


def overlap_length(intervals, span):
    total = 0.0

    for start, end in merge_intervals(intervals):
        total += max(
            0.0,
            min(span, end) - max(0.0, start),
        )

    return total


def load_inputs():
    windows_data = json.loads(
        WINDOW_PATH.read_text(encoding="utf-8")
    )
    facade_data = json.loads(
        FACADE_PATH.read_text(encoding="utf-8")
    )

    windows = [
        w for w in windows_data["assemblies"]
        if w.get("status") == "PROVISIONAL_STRAIGHT_ASSEMBLY"
    ]

    paths = facade_data["paths"]

    return windows, paths


def extract_segments(paths):
    segments = []

    for path in paths:
        points = path.get("points", [])

        for index in range(len(points) - 1):
            a = points[index]
            b = points[index + 1]

            direction = unit(vec(a, b))

            if direction is None:
                continue

            segment_length = magnitude(vec(a, b))

            if segment_length < MIN_SEGMENT_LENGTH_M:
                continue

            segments.append({
                "a": a,
                "b": b,
                "direction": direction,
                "length": segment_length,
                "source_handle": path["source_handle"],
                "source_layer": path["source_layer"],
                "category": path["category"],
                "geometry": LineString([a, b]),
            })

    return segments


def get_window_candidates(window, segments, tree):
    start = window["start"]
    end = window["end"]

    axis = unit(vec(start, end))
    span = magnitude(vec(start, end))

    if axis is None or span <= 1e-9:
        return [], span

    window_line = LineString([start, end])
    nearby = tree.query(
        window_line.buffer(SEARCH_RADIUS_M)
    )

    candidates = []

    for raw_index in nearby:
        index = int(raw_index)
        segment = segments[index]

        angle = angle_between(
            axis,
            segment["direction"],
        )

        if angle > MAX_ANGLE_DEGREES:
            continue

        distance = window_line.distance(
            segment["geometry"]
        )

        if distance > SEARCH_RADIUS_M:
            continue

        t1 = projection(
            segment["a"],
            start,
            axis,
        )
        t2 = projection(
            segment["b"],
            start,
            axis,
        )

        lo, hi = sorted((t1, t2))

        if hi < -REGION_GAP_TOLERANCE_M:
            continue

        if lo > span + REGION_GAP_TOLERANCE_M:
            continue

        # Signed lateral offset from window axis.
        midpoint = [
            (segment["a"][0] + segment["b"][0]) / 2,
            (segment["a"][1] + segment["b"][1]) / 2,
        ]

        offset = cross(
            axis,
            vec(start, midpoint),
        )

        candidates.append({
            "index": index,
            "lo": lo,
            "hi": hi,
            "offset": offset,
            "distance": distance,
            "angle": angle,
        })

    return candidates, span


def group_support_regions(candidates):
    """
    Group nearby parallel profiles along the same
    window axis.

    This is a geometric screening heuristic, not a
    reconstruction of physical façade assemblies.
    """
    if not candidates:
        return []

    parent = list(range(len(candidates)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri = find(i)
        rj = find(j)

        if ri != rj:
            parent[rj] = ri

    # Sort by projected interval start so we can stop
    # comparing when candidates are too far apart.
    order = sorted(
        range(len(candidates)),
        key=lambda i: candidates[i]["lo"],
    )

    for position, i in enumerate(order):
        first = candidates[i]

        for j in order[position + 1:]:
            second = candidates[j]

            if (
                second["lo"]
                > first["hi"] + REGION_GAP_TOLERANCE_M
            ):
                break

            offset_difference = abs(
                first["offset"] - second["offset"]
            )

            if (
                offset_difference
                <= REGION_OFFSET_TOLERANCE_M
            ):
                union(i, j)

    grouped = {}

    for i, candidate in enumerate(candidates):
        root = find(i)
        grouped.setdefault(root, []).append(candidate)

    return list(grouped.values())


def evaluate_region(region, span, segments):
    intervals = [
        (item["lo"], item["hi"])
        for item in region
    ]

    coverage = overlap_length(intervals, span)
    overlap_ratio = min(1.0, coverage / span)

    # Median is less sensitive to one unusual profile.
    offsets = sorted(item["offset"] for item in region)
    distances = sorted(item["distance"] for item in region)
    angles = sorted(item["angle"] for item in region)

    middle = len(region) // 2

    median_offset = offsets[middle]
    median_distance = distances[middle]
    median_angle = angles[middle]

    source_handles = sorted({
        segments[item["index"]]["source_handle"]
        for item in region
    })

    source_layers = sorted({
        segments[item["index"]]["source_layer"]
        for item in region
    })

    categories = sorted({
        segments[item["index"]]["category"]
        for item in region
    })

    # Use best observed local distance for proximity,
    # while retaining the median as a diagnostic.
    minimum_distance = min(distances)

    supported = (
        overlap_ratio >= MIN_OVERLAP_RATIO
        and minimum_distance <= MAX_DISTANCE_M
        and median_angle <= MAX_ANGLE_DEGREES
    )

    score = (
        0.65 * overlap_ratio
        + 0.25 * max(
            0.0,
            1.0 - minimum_distance / SEARCH_RADIUS_M,
        )
        + 0.10 * max(
            0.0,
            1.0 - median_angle / 90.0,
        )
    )

    return {
        "overlap_ratio": round(overlap_ratio, 4),
        "minimum_distance_m": round(
            minimum_distance, 6
        ),
        "median_distance_m": round(
            median_distance, 6
        ),
        "median_angle_degrees": round(
            median_angle, 3
        ),
        "median_signed_offset_m": round(
            median_offset, 6
        ),
        "source_segment_count": len(region),
        "source_handles": source_handles,
        "source_layers": source_layers,
        "categories": categories,
        "score": round(score, 5),
        "passes_screening": supported,
    }


def distinct_regions(passing):
    """
    Count clearly separated support regions.

    Regions within the separation tolerance may be
    duplicate CAD profiles of one façade assembly.
    """
    if not passing:
        return []

    ordered = sorted(
        passing,
        key=lambda r: r["score"],
        reverse=True,
    )

    distinct = []

    for region in ordered:
        offset = region["median_signed_offset_m"]

        if all(
            abs(
                offset
                - existing["median_signed_offset_m"]
            ) > DISTINCT_REGION_SEPARATION_M
            for existing in distinct
        ):
            distinct.append(region)

    return distinct


def verify(windows, segments):
    geometries = [
        segment["geometry"]
        for segment in segments
    ]

    tree = STRtree(geometries)

    results = []

    for number, window in enumerate(windows, 1):
        candidates, span = get_window_candidates(
            window,
            segments,
            tree,
        )

        groups = group_support_regions(candidates)

        regions = [
            evaluate_region(group, span, segments)
            for group in groups
        ]

        regions.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        passing = [
            region
            for region in regions
            if region["passes_screening"]
        ]

        distinct = distinct_regions(passing)

        if not passing:
            status = "NO_SUPPORTED_FACADE_REGION"
            selected = None

        elif len(distinct) > 1:
            status = "AMBIGUOUS_DISTINCT_FACADE_REGIONS"
            selected = None

        else:
            status = "GEOMETRICALLY_SUPPORTED_FACADE_REGION"
            selected = distinct[0]

        results.append({
            "window_id": window["assembly_id"],
            "window_source_group_id": window.get(
                "source_group_id"
            ),
            "window_source_handles": window.get(
                "source_handles", []
            ),
            "start": window["start"],
            "end": window["end"],
            "span_m": window["span_m"],
            "status": status,
            "candidate_segment_count": len(candidates),
            "support_region_count": len(regions),
            "passing_region_count": len(passing),
            "distinct_passing_region_count": len(distinct),
            "selected_region": selected,
            "top_regions": regions[:MAX_REPORT_CANDIDATES],
            "physical_window_verified": False,
            "window_opening_verified": False,
            "safe_to_cut": False,
            "construction_status": "UNKNOWN",
        })

        if number % 25 == 0 or number == len(windows):
            print(
                f"Screened {number}/{len(windows)} windows"
            )

    return results


def draw_lines(
    ax,
    segments,
    color,
    width,
    alpha,
    zorder,
):
    if not segments:
        return

    ax.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=width,
            alpha=alpha,
            zorder=zorder,
        )
    )


def create_preview(paths, results):
    fig, ax = plt.subplots(
        figsize=(16, 11),
        constrained_layout=True,
    )

    facade_segments = []
    curtain_segments = []

    for path in paths:
        points = path["points"]

        target = (
            curtain_segments
            if path["category"] == "CURTAIN_WALL"
            else facade_segments
        )

        target.extend(
            [points[i], points[i + 1]]
            for i in range(len(points) - 1)
        )

    draw_lines(
        ax,
        facade_segments,
        "#d97706",
        0.45,
        0.35,
        1,
    )

    draw_lines(
        ax,
        curtain_segments,
        "#8b5cf6",
        0.75,
        0.50,
        2,
    )

    colors = {
        "GEOMETRICALLY_SUPPORTED_FACADE_REGION": "#16a34a",
        "AMBIGUOUS_DISTINCT_FACADE_REGIONS": "#f59e0b",
        "NO_SUPPORTED_FACADE_REGION": "#dc2626",
    }

    for status, color in colors.items():
        matching = [
            [item["start"], item["end"]]
            for item in results
            if item["status"] == status
        ]

        draw_lines(
            ax,
            matching,
            color,
            2.2,
            0.95,
            4,
        )

    counts = Counter(
        item["status"] for item in results
    )

    ax.set_title(
        "Window-to-façade region screening\n"
        f"Supported: {counts['GEOMETRICALLY_SUPPORTED_FACADE_REGION']}  |  "
        f"Ambiguous: {counts['AMBIGUOUS_DISTINCT_FACADE_REGIONS']}  |  "
        f"Unsupported: {counts['NO_SUPPORTED_FACADE_REGION']}"
    )

    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.04)
    ax.grid(alpha=0.15)
    ax.set_xlabel("CAD X (metres)")
    ax.set_ylabel("CAD Y (metres)")

    fig.savefig(
        PREVIEW_PATH,
        dpi=180,
        facecolor="white",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows, paths = load_inputs()

    print("\n=== WINDOW-FACADE REGION SCREENING ===")
    print("Provisional window assemblies:", len(windows))
    print("Recovered facade paths:", len(paths))

    segments = extract_segments(paths)

    print("Usable facade segments:", len(segments))

    results = verify(windows, segments)

    counts = dict(
        Counter(item["status"] for item in results)
    )

    report = {
        "schema_version": "2.0",
        "coordinate_system": "DXF_METERS",
        "method": "LOCAL_FACADE_SUPPORT_REGION_SCREENING",
        "thresholds": {
            "search_radius_m": SEARCH_RADIUS_M,
            "maximum_distance_m": MAX_DISTANCE_M,
            "maximum_angle_degrees": MAX_ANGLE_DEGREES,
            "minimum_overlap_ratio": MIN_OVERLAP_RATIO,
            "region_offset_tolerance_m":
                REGION_OFFSET_TOLERANCE_M,
            "region_gap_tolerance_m":
                REGION_GAP_TOLERANCE_M,
            "distinct_region_separation_m":
                DISTINCT_REGION_SEPARATION_M,
        },
        "window_count": len(windows),
        "facade_path_count": len(paths),
        "usable_facade_segment_count": len(segments),
        "status_counts": counts,
        "physical_window_count_verified": False,
        "window_openings_verified": 0,
        "wall_cutouts_generated": False,
        "construction_progress_verified": False,
        "results": results,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    create_preview(paths, results)

    print("\nScreening results:")

    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")

    print("\nVerified physical window openings: 0")
    print("Wall cutouts generated: NO")
    print(
        "Report:",
        REPORT_PATH.relative_to(ROOT),
    )
    print(
        "Preview:",
        PREVIEW_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
