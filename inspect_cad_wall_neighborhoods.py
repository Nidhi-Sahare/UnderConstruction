
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

try:
    import ezdxf
except ImportError:
    raise SystemExit(
        "Missing dependency. Run: pip install ezdxf"
    )

ROOT = Path(__file__).resolve().parent
DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"
RECOVERY_PATH = ROOT / "floor_2_wall_recovery.json"
TOPOLOGY_PATH = ROOT / "outputs/wall_topology_validation.json"
OUTPUT_PATH = ROOT / "outputs/cad_wall_neighborhoods.json"

SEARCH_RADIUS_M = 1.0
MAX_NEIGHBORS = 30


def point_distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1]
    )


def segment_distance_to_point(point, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length_sq = dx * dx + dy * dy

    if length_sq <= 1e-18:
        return point_distance(point, a)

    t = (
        (point[0] - a[0]) * dx
        + (point[1] - a[1]) * dy
    ) / length_sq

    t = max(0.0, min(1.0, t))

    closest = (
        a[0] + t * dx,
        a[1] + t * dy
    )

    return point_distance(point, closest)


def valid_xy(point):
    try:
        x = float(point[0])
        y = float(point[1])
    except (IndexError, TypeError, ValueError):
        return None

    if not math.isfinite(x) or not math.isfinite(y):
        return None

    return (x, y)


def entity_segments(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        a = valid_xy(entity.dxf.start)
        b = valid_xy(entity.dxf.end)
        return [(a, b)] if a and b else []

    if kind == "LWPOLYLINE":
        points = [
            valid_xy(p)
            for p in entity.get_points("xy")
        ]

        points = [p for p in points if p is not None]

        segments = list(zip(points, points[1:]))

        if entity.closed and len(points) > 2:
            segments.append((points[-1], points[0]))

        # Bulge/arc segments are represented by their chords.
        return segments

    if kind == "POLYLINE":
        points = [
            valid_xy(v.dxf.location)
            for v in entity.vertices
        ]

        points = [p for p in points if p is not None]

        segments = list(zip(points, points[1:]))

        if entity.is_closed and len(points) > 2:
            segments.append((points[-1], points[0]))

        return segments

    if kind == "ARC":
        center = valid_xy(entity.dxf.center)
        radius = float(entity.dxf.radius)

        if not center or radius <= 0:
            return []

        start = math.radians(entity.dxf.start_angle)
        end = math.radians(entity.dxf.end_angle)

        while end <= start:
            end += 2 * math.pi

        steps = max(
            8,
            math.ceil((end - start) / (math.pi / 18))
        )

        points = [
            (
                center[0] + radius * math.cos(
                    start + (end - start) * i / steps
                ),
                center[1] + radius * math.sin(
                    start + (end - start) * i / steps
                )
            )
            for i in range(steps + 1)
        ]

        return list(zip(points, points[1:]))

    return []


def main():
    if not DXF_PATH.exists():
        raise SystemExit(f"Missing DXF: {DXF_PATH}")

    recovery = json.loads(
        RECOVERY_PATH.read_text(encoding="utf-8")
    )

    topology = json.loads(
        TOPOLOGY_PATH.read_text(encoding="utf-8")
    )

    isolated_ids = {
        item["id"]
        for item in topology["candidates"]
        if item["classification"] == "ISOLATED_CANDIDATE"
    }

    candidates = [
        item
        for item in recovery["candidate_walls"]
        if item["id"] in isolated_ids
    ]

    drawing = ezdxf.readfile(DXF_PATH)
    modelspace = drawing.modelspace()

    cad_segments = []
    unsupported = Counter()

    for entity in modelspace:
        kind = entity.dxftype()
        segments = entity_segments(entity)

        if not segments:
            unsupported[kind] += 1
            continue

        for a, b in segments:
            cad_segments.append({
                "handle": str(entity.dxf.handle),
                "layer": str(entity.dxf.layer),
                "entity_type": kind,
                "a": a,
                "b": b
            })

    results = []

    for candidate in candidates:
        line = candidate.get("source_line_xy")

        if not isinstance(line, list) or len(line) != 2:
            continue

        a, b = line
        midpoint = (
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2
        )

        neighbors = []

        for segment in cad_segments:
            if segment["handle"] == candidate["source_handle"]:
                continue

            distance = segment_distance_to_point(
                midpoint,
                segment["a"],
                segment["b"]
            )

            if distance > SEARCH_RADIUS_M:
                continue

            neighbors.append({
                "handle": segment["handle"],
                "layer": segment["layer"],
                "entity_type": segment["entity_type"],
                "distance_m": round(distance, 4),
                "start": segment["a"],
                "end": segment["b"]
            })

        neighbors.sort(
            key=lambda n: n["distance_m"]
        )

        results.append({
            "candidate_id": candidate["id"],
            "source_handle": candidate["source_handle"],
            "layer": candidate["layer"],
            "midpoint_xy": midpoint,
            "source_length_m": round(
                point_distance(a, b), 4
            ),
            "nearby_entity_count": len(neighbors),
            "nearby_layers": dict(
                Counter(n["layer"] for n in neighbors)
            ),
            "nearest_entities": neighbors[:MAX_NEIGHBORS],
            "architectural_interpretation_verified": False
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "step": 152,
        "source_dxf": str(DXF_PATH.relative_to(ROOT)),
        "search_radius_m": SEARCH_RADIUS_M,
        "method": "CAD_ENTITY_MIDPOINT_NEIGHBORHOOD",
        "isolated_candidates": len(candidates),
        "inspected_candidates": len(results),
        "indexed_segments": len(cad_segments),
        "unsupported_entity_types": dict(unsupported),
        "limitations": [
            "Entity proximity does not establish wall continuity.",
            "Only modelspace entities are indexed.",
            "Polyline bulges are approximated by chords.",
            "Nested INSERT geometry is not expanded.",
            "No openings are inferred or cut."
        ],
        "results": results
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nSTEP 152 - CAD NEIGHBORHOOD INSPECTION")
    print("-------------------------------------")
    print("Isolated candidates:", len(candidates))
    print("Inspected:", len(results))
    print("CAD segments indexed:", len(cad_segments))

    print("\nNORTHERN FACADE:")
    for result in results:
        if (
            abs(result["midpoint_xy"][1] - 33.993) < 0.01
            and result["layer"] == "010 Aussenwände"
        ):
            print(
                result["candidate_id"],
                "| neighbors:",
                result["nearby_entity_count"],
                "| layers:",
                result["nearby_layers"]
            )

    print("\nReport:", OUTPUT_PATH)
    print("Original CAD and wall geometry unchanged.")


if __name__ == "__main__":
    main()
