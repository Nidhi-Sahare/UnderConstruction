
"""
Under-construction — CAD Opening Extraction v2

Inputs:
    data/floorplans/floor_2.dxf
    data/digital_twin/building_elements.json

Output:
    outputs/cad_openings/opening_candidates.json

Improvements:
    - Door ARC entities become individual geometric seeds.
    - Arc centers are treated as possible hinge locations.
    - Arc endpoints define alternative possible closed-door directions.
    - Wall endpoint pairs are tested for doorway-sized gaps.
    - Gap direction and wall alignment contribute to matching.
    - Window grouping remains proximity-based.
    - Original wall element IDs are preserved.
    - Ambiguous results explicitly abstain.

No wall cutouts or construction progress are generated.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import ezdxf
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DXF = ROOT / "data/floorplans/floor_2.dxf"
DEFAULT_ELEMENTS = ROOT / "data/digital_twin/building_elements.json"
DEFAULT_OUTPUT = ROOT / "outputs/cad_openings/opening_candidates.json"

LAYERS = {
    "010 Türen": "door",
    "010 Fenster": "window",
}

CONNECT_TOLERANCE = 0.12
WINDOW_WALL_RADIUS = 0.50

MIN_SYMBOL_SIZE = 0.20
MAX_SYMBOL_SIZE = 5.00

MIN_DOOR_RADIUS = 0.45
MAX_DOOR_RADIUS = 1.50

# These are matching tolerances, not architectural facts.
HINGE_ENDPOINT_TOLERANCE = 0.38
FAR_ENDPOINT_TOLERANCE = 0.45
GAP_WIDTH_TOLERANCE = 0.30
MAX_WALL_ALIGNMENT_DEG = 22.5

# Difference needed to prefer one independent gap over another.
MIN_SCORE_MARGIN = 0.15

ARC_SAMPLE_STEP_DEG = 8.0


def xy(point):
    return float(point[0]), float(point[1])


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unit_vector(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return None

    return dx / length, dy / length


def parallel_angle_deg(a, b):
    """
    Smallest angle between two unoriented lines.
    Zero means parallel; 90 means perpendicular.
    """
    dot = abs(a[0] * b[0] + a[1] * b[1])
    dot = min(1.0, max(0.0, dot))
    return math.degrees(math.acos(dot))


def rounded_point(point):
    return [round(float(point[0]), 5),
            round(float(point[1]), 5)]


def arc_points(center, radius, start_angle, end_angle):
    start = float(start_angle)
    end = float(end_angle)

    while end < start:
        end += 360.0

    sweep = end - start
    count = max(
        2,
        math.ceil(sweep / ARC_SAMPLE_STEP_DEG) + 1,
    )

    cx, cy = xy(center)

    return [
        (
            cx + radius * math.cos(
                math.radians(start + sweep * i / (count - 1))
            ),
            cy + radius * math.sin(
                math.radians(start + sweep * i / (count - 1))
            ),
        )
        for i in range(count)
    ]


def entity_geometry(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        return LineString([
            xy(entity.dxf.start),
            xy(entity.dxf.end),
        ])

    if kind == "ARC":
        return LineString(arc_points(
            entity.dxf.center,
            float(entity.dxf.radius),
            entity.dxf.start_angle,
            entity.dxf.end_angle,
        ))

    if kind == "LWPOLYLINE":
        points = [
            (float(x), float(y))
            for x, y in entity.get_points("xy")
        ]

        if len(points) < 2:
            return None

        if entity.closed:
            points.append(points[0])

        return LineString(points)

    return None


def extract_primitives(doc):
    primitives = []
    skipped = Counter()

    for entity in doc.modelspace():
        layer = entity.dxf.layer

        if layer not in LAYERS:
            continue

        kind = entity.dxftype()

        if kind == "MTEXT":
            skipped["MTEXT"] += 1
            continue

        if kind == "INSERT":
            skipped["INSERT_UNEXPANDED"] += 1
            continue

        geometry = entity_geometry(entity)

        if geometry is None or geometry.is_empty:
            skipped[kind] += 1
            continue

        if geometry.length < 1e-8:
            skipped["DEGENERATE"] += 1
            continue

        record = {
            "type": LAYERS[layer],
            "layer": layer,
            "handle": entity.dxf.handle,
            "entity_type": kind,
            "geometry": geometry,
        }

        if kind == "ARC":
            record["arc"] = {
                "center": xy(entity.dxf.center),
                "radius": float(entity.dxf.radius),
                "start_angle": float(entity.dxf.start_angle),
                "end_angle": float(entity.dxf.end_angle),
            }

        primitives.append(record)

    return primitives, dict(skipped)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        a, b = self.find(i), self.find(j)

        if a == b:
            return

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a

        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def group_primitives(primitives, tolerance=CONNECT_TOLERANCE):
    if not primitives:
        return []

    geometries = [p["geometry"] for p in primitives]
    tree = STRtree(geometries)
    uf = UnionFind(len(primitives))

    for i, geometry in enumerate(geometries):
        for candidate in tree.query(geometry.buffer(tolerance)):
            j = int(candidate)

            if j <= i:
                continue

            if geometry.distance(geometries[j]) <= tolerance:
                uf.union(i, j)

    components = defaultdict(list)

    for i, primitive in enumerate(primitives):
        components[uf.find(i)].append(primitive)

    return list(components.values())


def load_walls(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    elements = data if isinstance(data, list) else data["elements"]
    walls = []

    for item in elements:
        start = item.get("start")
        end = item.get("end")

        if not start or not end:
            continue

        a, b = xy(start), xy(end)
        direction = unit_vector(a, b)

        if direction is None or distance(a, b) < 0.01:
            continue

        walls.append({
            "id": str(item["element_id"]),
            "category": str(item.get("category", "unknown")),
            "start": a,
            "end": b,
            "direction": direction,
            "line": LineString([a, b]),
        })

    return walls


def build_endpoint_index(walls):
    """
    Each CAD segment endpoint is retained separately.

    This matters because an opening can lie between two
    different wall segments rather than inside either one.
    """
    endpoints = []

    for wall in walls:
        for side in ("start", "end"):
            endpoints.append({
                "point": wall[side],
                "wall_id": wall["id"],
                "category": wall["category"],
                "direction": wall["direction"],
            })

    tree = STRtree([
        Point(item["point"]) for item in endpoints
    ])

    return endpoints, tree


def nearby_endpoints(position, endpoints, tree, radius):
    indices = tree.query(Point(position).buffer(radius))

    results = []

    for index in indices:
        item = endpoints[int(index)]
        d = distance(position, item["point"])

        if d <= radius:
            results.append((item, d))

    return sorted(results, key=lambda pair: pair[1])


def arc_endpoints(arc):
    points = arc_points(
        arc["center"],
        arc["radius"],
        arc["start_angle"],
        arc["end_angle"],
    )

    return points[0], points[-1]


def match_door_gap(arc, endpoints, endpoint_tree):
    """
    Evaluate possible doorway spans.

    For each swing arc:
      - its center is a possible hinge;
      - either arc endpoint may indicate a closed-door edge;
      - two CAD wall endpoints may delimit the doorway.

    This is a geometric hypothesis, not a verified door.
    """
    hinge = arc["center"]
    radius = arc["radius"]
    possible_edges = arc_endpoints(arc)

    hinge_candidates = nearby_endpoints(
        hinge,
        endpoints,
        endpoint_tree,
        HINGE_ENDPOINT_TOLERANCE,
    )

    gap_candidates = []

    for edge_index, edge in enumerate(possible_edges):
        far_candidates = nearby_endpoints(
            edge,
            endpoints,
            endpoint_tree,
            FAR_ENDPOINT_TOLERANCE,
        )

        for first, hinge_distance in hinge_candidates:
            for second, far_distance in far_candidates:
                if first["wall_id"] == second["wall_id"]:
                    continue

                gap_vector = unit_vector(
                    first["point"],
                    second["point"],
                )

                if gap_vector is None:
                    continue

                gap_width = distance(
                    first["point"],
                    second["point"],
                )

                width_error = abs(gap_width - radius)

                if width_error > GAP_WIDTH_TOLERANCE:
                    continue

                angle_first = parallel_angle_deg(
                    gap_vector,
                    first["direction"],
                )

                angle_second = parallel_angle_deg(
                    gap_vector,
                    second["direction"],
                )

                if max(
                    angle_first,
                    angle_second,
                ) > MAX_WALL_ALIGNMENT_DEG:
                    continue

                # Lower is better. All terms are normalized.
                score = (
                    0.30 * hinge_distance / HINGE_ENDPOINT_TOLERANCE
                    + 0.30 * far_distance / FAR_ENDPOINT_TOLERANCE
                    + 0.25 * width_error / GAP_WIDTH_TOLERANCE
                    + 0.15 * (
                        angle_first + angle_second
                    ) / (2 * MAX_WALL_ALIGNMENT_DEG)
                )

                gap_candidates.append({
                    "score": round(score, 5),
                    "gap_start": rounded_point(first["point"]),
                    "gap_end": rounded_point(second["point"]),
                    "gap_width": round(gap_width, 5),
                    "arc_radius": round(radius, 5),
                    "width_error": round(width_error, 5),
                    "hinge_distance": round(hinge_distance, 5),
                    "far_endpoint_distance": round(
                        far_distance, 5
                    ),
                    "wall_alignment_degrees": [
                        round(angle_first, 3),
                        round(angle_second, 3),
                    ],
                    "wall_element_ids": [
                        first["wall_id"],
                        second["wall_id"],
                    ],
                    "arc_endpoint_choice": edge_index,
                })

    # Deduplicate repeated endpoints, including alternative
    # interpretations of the same physical gap.
    unique = {}

    for candidate in gap_candidates:
        endpoints_key = tuple(sorted([
            tuple(candidate["gap_start"]),
            tuple(candidate["gap_end"]),
        ]))

        previous = unique.get(endpoints_key)

        if previous is None or candidate["score"] < previous["score"]:
            unique[endpoints_key] = candidate

    ranked = sorted(
        unique.values(),
        key=lambda candidate: candidate["score"],
    )

    if not ranked:
        status = "NO_SUPPORTED_GAP"

    elif ranked[0]["score"] > 0.75:
        status = "WEAK_GAP_CANDIDATE"

    elif (
        len(ranked) > 1
        and ranked[1]["score"] - ranked[0]["score"]
        < MIN_SCORE_MARGIN
    ):
        status = "AMBIGUOUS_GAPS"

    else:
        status = "PROVISIONAL_GAP_MATCH"

    return status, ranked[:8]


def door_candidate(index, primitive, endpoints, endpoint_tree):
    arc = primitive["arc"]
    radius = arc["radius"]

    start, end = arc_endpoints(arc)

    sweep = (
        arc["end_angle"] - arc["start_angle"]
    ) % 360.0

    if sweep == 0:
        sweep = 360.0

    if not MIN_DOOR_RADIUS <= radius <= MAX_DOOR_RADIUS:
        status = "ARC_RADIUS_OUT_OF_RANGE"
        matches = []
    else:
        status, matches = match_door_gap(
            arc,
            endpoints,
            endpoint_tree,
        )

    return {
        "candidate_id": f"door_arc_{index:05d}",
        "type": "door",
        "layer": primitive["layer"],
        "method": "SWING_ARC_AND_WALL_GAP",
        "entity_handles": [primitive["handle"]],
        "entity_count": 1,
        "hinge_candidate": rounded_point(arc["center"]),
        "arc_radius": round(radius, 5),
        "arc_sweep_degrees": round(sweep, 3),
        "arc_endpoints": [
            rounded_point(start),
            rounded_point(end),
        ],
        "association_status": status,
        "gap_candidates": matches,
        "nearby_walls": (
            [
                {
                    "element_id": wall_id,
                    "role": "gap_boundary_candidate",
                }
                for wall_id in matches[0]["wall_element_ids"]
            ]
            if matches else []
        ),
        "opening_verified": False,
        "safe_to_cut": False,
    }


def window_candidate(index, group, walls, wall_tree):
    geometry = unary_union([
        primitive["geometry"]
        for primitive in group
    ])

    min_x, min_y, max_x, max_y = geometry.bounds

    width = max_x - min_x
    height = max_y - min_y

    center = Point(
        (min_x + max_x) / 2,
        (min_y + max_y) / 2,
    )

    size = max(width, height)

    if size < MIN_SYMBOL_SIZE:
        geometry_status = "TOO_SMALL"
    elif size > MAX_SYMBOL_SIZE:
        geometry_status = "TOO_LARGE_OR_MERGED"
    else:
        geometry_status = "PLAUSIBLE_SIZE"

    matches = []

    for index_value in wall_tree.query(
        center.buffer(WINDOW_WALL_RADIUS)
    ):
        wall = walls[int(index_value)]
        d = center.distance(wall["line"])

        if d <= WINDOW_WALL_RADIUS:
            matches.append({
                "element_id": wall["id"],
                "category": wall["category"],
                "centerline_distance": round(d, 5),
            })

    matches.sort(
        key=lambda item: item["centerline_distance"]
    )

    if geometry_status != "PLAUSIBLE_SIZE":
        status = "GEOMETRY_UNCERTAIN"
    elif not matches:
        status = "NO_NEARBY_WALL"
    elif len(matches) == 1:
        status = "SINGLE_SPATIAL_CANDIDATE"
    else:
        status = "AMBIGUOUS_MULTIPLE_WALLS"

    return {
        "candidate_id": f"window_{index:05d}",
        "type": "window",
        "layer": group[0]["layer"],
        "method": "CONNECTED_PRIMITIVES_AND_WALL_PROXIMITY",
        "entity_count": len(group),
        "entity_handles": [
            primitive["handle"]
            for primitive in group
        ],
        "entity_types": dict(Counter(
            primitive["entity_type"]
            for primitive in group
        )),
        "center": rounded_point((center.x, center.y)),
        "bounds": [
            round(min_x, 5),
            round(min_y, 5),
            round(max_x, 5),
            round(max_y, 5),
        ],
        "width_coordinate_units": round(width, 5),
        "height_coordinate_units": round(height, 5),
        "geometry_status": geometry_status,
        "association_status": status,
        "nearby_walls": matches[:12],
        "opening_verified": False,
        "safe_to_cut": False,
    }


def unanchored_door_candidate(index, group):
    """
    Preserve door-layer geometry not explained by ARC entities.
    Do not mistake disconnected drafting lines for verified doors.
    """
    geometry = unary_union([
        primitive["geometry"]
        for primitive in group
    ])

    return {
        "candidate_id": f"door_unanchored_{index:05d}",
        "type": "door",
        "layer": group[0]["layer"],
        "method": "UNANCHORED_DRAWING_GROUP",
        "entity_count": len(group),
        "entity_handles": [
            primitive["handle"]
            for primitive in group
        ],
        "bounds": [
            round(value, 5)
            for value in geometry.bounds
        ],
        "association_status": "NO_SWING_ARC_ANCHOR",
        "nearby_walls": [],
        "opening_verified": False,
        "safe_to_cut": False,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dxf",
        type=Path,
        default=DEFAULT_DXF,
    )
    parser.add_argument(
        "--elements",
        type=Path,
        default=DEFAULT_ELEMENTS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    doc = ezdxf.readfile(args.dxf)
    primitives, skipped = extract_primitives(doc)

    walls = load_walls(args.elements)

    if not walls:
        raise RuntimeError("No usable CAD wall segments found")

    wall_tree = STRtree([
        wall["line"]
        for wall in walls
    ])

    endpoints, endpoint_tree = build_endpoint_index(walls)

    door_primitives = [
        p for p in primitives
        if p["type"] == "door"
    ]

    door_arcs = [
        p for p in door_primitives
        if p["entity_type"] == "ARC"
    ]

    windows = [
        p for p in primitives
        if p["type"] == "window"
    ]

    candidates = []

    # Arc-anchored doors.
    for i, primitive in enumerate(door_arcs):
        candidates.append(
            door_candidate(
                i,
                primitive,
                endpoints,
                endpoint_tree,
            )
        )

    # Preserve other door-layer drafting geometry as
    # unanchored groups. Exclude lines close to swing arcs
    # to avoid duplicating their associated symbols.
    other_door_primitives = [
        p for p in door_primitives
        if p["entity_type"] != "ARC"
    ]

    unanchored = []

    for primitive in other_door_primitives:
        near_arc = False

        for arc_primitive in door_arcs:
            if primitive["geometry"].distance(
                arc_primitive["geometry"]
            ) <= CONNECT_TOLERANCE:
                near_arc = True
                break

        if not near_arc:
            unanchored.append(primitive)

    unanchored_groups = group_primitives(unanchored)

    for i, group in enumerate(unanchored_groups):
        candidates.append(
            unanchored_door_candidate(i, group)
        )

    # Preserve existing window grouping.
    window_groups = group_primitives(windows)

    for i, group in enumerate(window_groups):
        candidates.append(
            window_candidate(
                i,
                group,
                walls,
                wall_tree,
            )
        )

    status_counts = dict(Counter(
        candidate["association_status"]
        for candidate in candidates
    ))

    report = {
        "schema_version": "2.0",
        "source_dxf": str(args.dxf),
        "source_elements": str(args.elements),
        "coordinate_units": "DXF_DRAWING_UNITS",
        "dxf_insunits": doc.header.get("$INSUNITS", 0),
        "units_verified": False,
        "primitive_count": len(primitives),
        "wall_element_count": len(walls),
        "wall_endpoint_count": len(endpoints),
        "door_arc_count": len(door_arcs),
        "unanchored_door_group_count": len(
            unanchored_groups
        ),
        "window_group_count": len(window_groups),
        "candidate_count": len(candidates),
        "candidate_types": dict(Counter(
            candidate["type"]
            for candidate in candidates
        )),
        "association_status_counts": status_counts,
        "skipped_entities": skipped,
        "matching_tolerances": {
            "hinge_endpoint": HINGE_ENDPOINT_TOLERANCE,
            "far_endpoint": FAR_ENDPOINT_TOLERANCE,
            "gap_width": GAP_WIDTH_TOLERANCE,
            "max_alignment_degrees": MAX_WALL_ALIGNMENT_DEG,
            "minimum_score_margin": MIN_SCORE_MARGIN,
        },
        "openings_verified": False,
        "wall_cutouts_generated": False,
        "notes": [
            "Door swing arcs are interpreted as possible hinge evidence.",
            "Wall endpoint pairs are tested as possible doorway gaps.",
            "Window associations remain provisional proximity matches.",
            "An unanchored drawing group is not necessarily one door.",
            "No candidate is automatically approved for wall subtraction.",
        ],
        "candidates": candidates,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== CAD OPENING EXTRACTION V2 ===")
    print("Wall elements:", len(walls))
    print("Wall endpoints:", len(endpoints))
    print("Drawing primitives:", len(primitives))
    print("Door swing arcs:", len(door_arcs))
    print("Unanchored door groups:", len(unanchored_groups))
    print("Window groups:", len(window_groups))
    print("Total candidates:", len(candidates))
    print("Candidate types:", report["candidate_types"])
    print("Associations:", status_counts)
    print("Skipped:", skipped)
    print("Output:", args.output)
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
