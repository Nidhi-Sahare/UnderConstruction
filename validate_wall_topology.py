
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RECOVERY_FILE = ROOT / "floor_2_wall_recovery.json"
ORIGINAL_FILE = ROOT / "data/geometry/floor_2_wall_meshes.json"
OUTPUT_FILE = ROOT / "outputs/wall_topology_validation.json"

# Metres. These are geometric screening tolerances,
# not evidence of architectural correctness.
CONNECT_TOLERANCE = 0.05
OVERLAP_TOLERANCE = 0.01
MIN_AREA = 0.0001


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_records(data, preferred_keys):
    if isinstance(data, list):
        return data

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object or array")

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            return value

    raise ValueError(
        f"Could not find any of these arrays: {preferred_keys}"
    )


def valid_polygon(points):
    if not isinstance(points, list):
        return False

    cleaned = []

    for point in points:
        if not isinstance(point, (list, tuple)):
            return False

        if len(point) < 2:
            return False

        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return False

        if not math.isfinite(x) or not math.isfinite(y):
            return False

        p = (x, y)

        if not cleaned or math.dist(p, cleaned[-1]) > 1e-9:
            cleaned.append(p)

    if len(cleaned) > 1:
        if math.dist(cleaned[0], cleaned[-1]) <= 1e-9:
            cleaned.pop()

    return len(cleaned) >= 3


def clean_polygon(points):
    cleaned = []

    for point in points:
        p = (float(point[0]), float(point[1]))

        if not cleaned or math.dist(p, cleaned[-1]) > 1e-9:
            cleaned.append(p)

    if len(cleaned) > 1:
        if math.dist(cleaned[0], cleaned[-1]) <= 1e-9:
            cleaned.pop()

    return cleaned


def polygon_area(points):
    return abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
    ) / 2


def polygon_edges(points):
    return [
        (points[i], points[(i + 1) % len(points)])
        for i in range(len(points))
    ]


def point_segment_distance(point, start, end):
    ax, ay = start
    bx, by = end
    px, py = point

    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy

    if length_sq <= 1e-18:
        return math.dist(point, start)

    t = max(
        0.0,
        min(
            1.0,
            ((px - ax) * dx + (py - ay) * dy) / length_sq
        )
    )

    nearest = (ax + t * dx, ay + t * dy)

    return math.dist(point, nearest)


def segments_intersect(a, b, c, d):
    def orientation(p, q, r):
        return (
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    o1 = orientation(a, b, c)
    o2 = orientation(a, b, d)
    o3 = orientation(c, d, a)
    o4 = orientation(c, d, b)

    eps = 1e-10

    if (
        ((o1 > eps and o2 < -eps) or
         (o1 < -eps and o2 > eps))
        and
        ((o3 > eps and o4 < -eps) or
         (o3 < -eps and o4 > eps))
    ):
        return True

    for p, s, e in (
        (c, a, b),
        (d, a, b),
        (a, c, d),
        (b, c, d),
    ):
        if point_segment_distance(p, s, e) <= eps:
            return True

    return False


def segment_distance(edge_a, edge_b):
    a, b = edge_a
    c, d = edge_b

    if segments_intersect(a, b, c, d):
        return 0.0

    return min(
        point_segment_distance(a, c, d),
        point_segment_distance(b, c, d),
        point_segment_distance(c, a, b),
        point_segment_distance(d, a, b),
    )


def polygon_distance(a, b):
    edges_a = polygon_edges(a)
    edges_b = polygon_edges(b)

    return min(
        segment_distance(e1, e2)
        for e1 in edges_a
        for e2 in edges_b
    )


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def bbox_distance(a, b):
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return math.hypot(dx, dy)


def prepare(items, name):
    prepared = []
    invalid = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            invalid.append(index)
            continue

        raw = item.get("outline_xy")

        if raw is None:
            raw = (
                (item.get("footprint") or {}).get("exterior")
            )

        if not valid_polygon(raw):
            invalid.append(index)
            continue

        points = clean_polygon(raw)
        area = polygon_area(points)

        if area < MIN_AREA:
            invalid.append(index)
            continue

        prepared.append({
            "id": str(item.get("id") or f"{name}_{index}"),
            "layer": item.get("layer"),
            "source_handle": item.get("source_handle"),
            "points": points,
            "bbox": bbox(points),
            "area_m2": area,
        })

    return prepared, invalid


def main():
    recovery_data = load_json(RECOVERY_FILE)
    original_data = load_json(ORIGINAL_FILE)

    recovery_raw = extract_records(
        recovery_data,
        ["candidate_walls", "recovered_wall_candidates", "candidates",
         "recovered_walls", "walls"]
    )

    original_raw = extract_records(
        original_data,
        ["wall_meshes", "meshes", "walls", "elements"]
    )

    recovered, invalid_recovered = prepare(
        recovery_raw, "recovered"
    )

    originals, invalid_original = prepare(
        original_raw, "original"
    )

    if not recovered or not originals:
        raise RuntimeError(
            "No usable recovered or original polygons. "
            "Inspect the JSON array names and geometry."
        )

    results = []
    connection_graph = defaultdict(set)

    for i, candidate in enumerate(recovered):
        original_connections = []
        recovered_connections = []

        for original in originals:
            if bbox_distance(
                candidate["bbox"], original["bbox"]
            ) > CONNECT_TOLERANCE:
                continue

            distance = polygon_distance(
                candidate["points"], original["points"]
            )

            if distance <= CONNECT_TOLERANCE:
                original_connections.append({
                    "id": original["id"],
                    "distance_m": round(distance, 6),
                })

        for j, other in enumerate(recovered):
            if i == j:
                continue

            if bbox_distance(
                candidate["bbox"], other["bbox"]
            ) > CONNECT_TOLERANCE:
                continue

            distance = polygon_distance(
                candidate["points"], other["points"]
            )

            if distance <= CONNECT_TOLERANCE:
                recovered_connections.append({
                    "id": other["id"],
                    "distance_m": round(distance, 6),
                })

                connection_graph[candidate["id"]].add(
                    other["id"]
                )

        if original_connections:
            classification = "CONTACTS_ORIGINAL_WALL"
        elif recovered_connections:
            classification = "CONTACTS_RECOVERED_WALL_ONLY"
        else:
            classification = "ISOLATED_CANDIDATE"

        results.append({
            "id": candidate["id"],
            "layer": candidate["layer"],
            "source_handle": candidate["source_handle"],
            "area_m2": round(candidate["area_m2"], 6),
            "classification": classification,
            "original_wall_contacts": original_connections,
            "recovered_wall_contacts": recovered_connections,
            "contact_count": (
                len(original_connections)
                + len(recovered_connections)
            ),
            "architectural_validity_verified": False,
            "opening_clearance_verified": False,
            "automatic_integration_approved": False,
        })

    summary = Counter(
        result["classification"] for result in results
    )

    report = {
        "step": 150,
        "method": "POLYGON_BOUNDARY_PROXIMITY_SCREENING",
        "units": "metres",
        "connection_tolerance_m": CONNECT_TOLERANCE,
        "original_polygons": len(originals),
        "recovered_polygons": len(recovered),
        "invalid_original_records": invalid_original,
        "invalid_recovered_records": invalid_recovered,
        "summary": dict(summary),
        "limitations": [
            "Contact does not prove architectural continuity.",
            "Touching and overlapping polygons are not distinguished.",
            "No source CAD edge consistency check performed.",
            "No door or window clearance verification performed.",
            "No recovered wall is approved for automatic merging.",
        ],
        "candidates": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    OUTPUT_FILE.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nSTEP 150 â€” WALL TOPOLOGY SCREENING")
    print("----------------------------------")
    print("Original wall polygons:", len(originals))
    print("Recovered wall polygons:", len(recovered))
    print("Invalid recovered:", len(invalid_recovered))

    for key in (
        "CONTACTS_ORIGINAL_WALL",
        "CONTACTS_RECOVERED_WALL_ONLY",
        "ISOLATED_CANDIDATE",
    ):
        print(f"{key}: {summary.get(key, 0)}")

    print("\nReport:", OUTPUT_FILE)
    print(
        "No geometry was merged or modified."
    )


if __name__ == "__main__":
    main()

