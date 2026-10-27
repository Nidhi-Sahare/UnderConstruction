
"""
Under-construction — CAD Door Assembly Analysis

Reads original DXF door-layer geometry and optional v3 candidates.

Outputs:
    outputs/cad_openings/assemblies/assembly_report.json
    outputs/cad_openings/assemblies/assembly_overview.png
    outputs/cad_openings/assemblies/assembly_XXX.png

Requires:
    ezdxf
    matplotlib

Read-only. Does not create or verify wall openings.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import ezdxf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DXF = ROOT / "data/floorplans/floor_2.dxf"
DEFAULT_V3 = ROOT / "outputs/cad_openings/opening_candidates_v3.json"
DEFAULT_OUT = ROOT / "outputs/cad_openings/assemblies"

DOOR_LAYER = "010 Türen"
WALL_LAYERS = {
    "010 Aussenwände",
    "015 Innenwände",
    "016 Trennwände",
    "019 Vormauerungen",
}

HINGE_TOL = 0.06
TIP_TOL = 0.06
ARC_PROXIMITY_TOL = 0.06
LINE_ASSOCIATION_TOL = 0.12
LOCAL_VIEW_MARGIN = 1.0
MAX_WALL_PRIMITIVES_PER_PLOT = 10000


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def as_xy(v):
    return [float(v[0]), float(v[1])]


def arc_sample(center, radius, start, end, n=80):
    sweep = (end - start) % 360.0

    # DXF arcs are counterclockwise.
    angles = [
        math.radians(start + sweep * i / (n - 1))
        for i in range(n)
    ]

    return [
        [
            center[0] + radius * math.cos(a),
            center[1] + radius * math.sin(a),
        ]
        for a in angles
    ]


def segment_distance(p, a, b):
    vx = b[0] - a[0]
    vy = b[1] - a[1]
    length_sq = vx * vx + vy * vy

    if length_sq < 1e-12:
        return distance(p, a)

    t = (
        (p[0] - a[0]) * vx +
        (p[1] - a[1]) * vy
    ) / length_sq

    t = max(0.0, min(1.0, t))

    q = [a[0] + t * vx, a[1] + t * vy]
    return distance(p, q)


def polyline_distance(first, second):
    """Minimum segment-to-sampled-point distance, symmetric."""

    best = float("inf")

    for poly_a, poly_b in (
        (first, second),
        (second, first),
    ):
        for p in poly_a:
            for a, b in zip(poly_b, poly_b[1:]):
                best = min(best, segment_distance(p, a, b))

    return best


def entity_paths(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        return [[
            as_xy(entity.dxf.start),
            as_xy(entity.dxf.end),
        ]]

    if kind == "ARC":
        return [arc_sample(
            as_xy(entity.dxf.center),
            float(entity.dxf.radius),
            float(entity.dxf.start_angle),
            float(entity.dxf.end_angle),
        )]

    if kind == "CIRCLE":
        center = as_xy(entity.dxf.center)
        radius = float(entity.dxf.radius)

        return [[
            [
                center[0] + radius * math.cos(t),
                center[1] + radius * math.sin(t),
            ]
            for t in [
                2 * math.pi * i / 79 for i in range(80)
            ]
        ]]

    if kind == "LWPOLYLINE":
        # Use virtual entities so bulged segments remain arcs.
        paths = []

        for sub in entity.virtual_entities():
            paths.extend(entity_paths(sub))

        return paths

    if kind == "POLYLINE":
        if not entity.is_2d_polyline:
            return []

        paths = []
        for sub in entity.virtual_entities():
            paths.extend(entity_paths(sub))
        return paths

    return []


def bounds(paths):
    points = [
        p for path in paths for p in path
    ]

    if not points:
        return None

    return [
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    ]


def boxes_overlap(a, b, margin=0):
    return not (
        a[2] + margin < b[0] or
        b[2] + margin < a[0] or
        a[3] + margin < b[1] or
        b[3] + margin < a[1]
    )


def make_record(entity):
    paths = entity_paths(entity)

    if not paths:
        return None

    return {
        "handle": entity.dxf.handle,
        "layer": entity.dxf.layer,
        "entity_type": entity.dxftype(),
        "paths": paths,
        "bounds": bounds(paths),
    }


def make_arc(entity):
    center = as_xy(entity.dxf.center)
    radius = float(entity.dxf.radius)
    start = float(entity.dxf.start_angle)
    end = float(entity.dxf.end_angle)
    paths = entity_paths(entity)
    points = paths[0]

    return {
        "handle": entity.dxf.handle,
        "center": center,
        "radius": radius,
        "start_angle": start,
        "end_angle": end,
        "sweep_degrees": (end - start) % 360.0,
        "tips": [points[0], points[-1]],
        "paths": paths,
        "bounds": bounds(paths),
    }


def arcs_related(a, b):
    hinge_distance = distance(a["center"], b["center"])

    if hinge_distance <= HINGE_TOL:
        return ["SHARED_HINGE"]

    for tip_a in a["tips"]:
        for tip_b in b["tips"]:
            if distance(tip_a, tip_b) <= TIP_TOL:
                return ["SHARED_ARC_TIP"]

    # Touching arcs alone are not sufficient to assert that
    # they form one architectural assembly. Keep that
    # observation separately rather than merging them.
    return []


def connected_components(arcs):
    parent = list(range(len(arcs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    relations = []

    for i in range(len(arcs)):
        for j in range(i + 1, len(arcs)):
            kinds = arcs_related(arcs[i], arcs[j])

            if kinds:
                union(i, j)
                relations.append({
                    "arc_handles": [
                        arcs[i]["handle"],
                        arcs[j]["handle"],
                    ],
                    "relations": kinds,
                })

    groups = {}

    for i, arc in enumerate(arcs):
        groups.setdefault(find(i), []).append(arc)

    result = list(groups.values())

    result.sort(
        key=lambda group: (
            min(a["center"][1] for a in group),
            min(a["center"][0] for a in group),
        )
    )

    return result, relations


def closest_path_distance(paths_a, paths_b):
    best = float("inf")

    for a in paths_a:
        for b in paths_b:
            best = min(best, polyline_distance(a, b))

    return best


def associate_lines(group, lines):
    arc_bounds = bounds([
        path
        for arc in group
        for path in arc["paths"]
    ])

    matches = []

    for line in lines:
        if not boxes_overlap(
            arc_bounds,
            line["bounds"],
            LINE_ASSOCIATION_TOL,
        ):
            continue

        best = min(
            closest_path_distance(
                arc["paths"],
                line["paths"],
            )
            for arc in group
        )

        if best <= LINE_ASSOCIATION_TOL:
            matches.append({
                "handle": line["handle"],
                "distance_to_arc": round(best, 6),
            })

    return sorted(
        matches,
        key=lambda x: x["distance_to_arc"],
    )


def load_v3(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return [
        item
        for item in data.get("candidates", [])
        if item.get("method") == "SWING_ARC_AND_WALL_GAP"
    ]


def match_v3(arc, candidates):
    matches = []

    for candidate in candidates:
        hinge = candidate.get("hinge_candidate")

        if hinge is None:
            continue

        error = distance(hinge, arc["center"])
        radius_error = abs(
            float(candidate.get("arc_radius", -1)) -
            arc["radius"]
        )

        if error <= 0.015 and radius_error <= 0.015:
            matches.append({
                "candidate_id": candidate.get("candidate_id"),
                "association_status": candidate.get(
                    "association_status"
                ),
                "hinge_error": round(error, 6),
                "radius_error": round(radius_error, 6),
            })

    return matches


def plot_paths(ax, records, color, width, alpha):
    for record in records:
        for path in record["paths"]:
            ax.plot(
                [p[0] for p in path],
                [p[1] for p in path],
                color=color,
                linewidth=width,
                alpha=alpha,
            )


def plot_assembly(group, lines, walls, output, title):
    group_bounds = bounds([
        path
        for arc in group
        for path in arc["paths"]
    ])

    xmin, ymin, xmax, ymax = group_bounds

    xmin -= LOCAL_VIEW_MARGIN
    ymin -= LOCAL_VIEW_MARGIN
    xmax += LOCAL_VIEW_MARGIN
    ymax += LOCAL_VIEW_MARGIN

    viewport = [xmin, ymin, xmax, ymax]

    visible_walls = [
        w for w in walls
        if boxes_overlap(w["bounds"], viewport)
    ][:MAX_WALL_PRIMITIVES_PER_PLOT]

    visible_lines = [
        line for line in lines
        if boxes_overlap(line["bounds"], viewport)
    ]

    fig, ax = plt.subplots(figsize=(9, 9))

    plot_paths(
        ax, visible_walls, "#94a3b8", 0.8, 0.65
    )
    plot_paths(
        ax, visible_lines, "#2563eb", 1.1, 0.7
    )

    colors = [
        "#e11d48",
        "#ea580c",
        "#9333ea",
        "#059669",
        "#ca8a04",
    ]

    for i, arc in enumerate(group):
        color = colors[i % len(colors)]
        plot_paths(ax, [arc], color, 2.4, 1.0)

        x, y = arc["center"]

        ax.scatter(
            [x], [y], color=color, marker="x", s=70
        )

        ax.annotate(
            arc["handle"],
            (x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )

        for tip in arc["tips"]:
            ax.scatter(
                [tip[0]], [tip[1]],
                color=color, s=16,
            )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("DXF X")
    ax.set_ylabel("DXF Y")

    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_overview(arcs, walls, output):
    fig, ax = plt.subplots(figsize=(16, 12))

    plot_paths(ax, walls, "#cbd5e1", 0.55, 0.65)

    for arc in arcs:
        x, y = arc["center"]

        ax.scatter(
            [x], [y],
            color="#e11d48",
            s=22,
            zorder=5,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("CAD door swing arcs — original DXF")
    ax.grid(alpha=0.1)

    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dxf", type=Path, default=DEFAULT_DXF
    )
    parser.add_argument(
        "--v3", type=Path, default=DEFAULT_V3
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT
    )

    args = parser.parse_args()

    if not args.dxf.exists():
        raise FileNotFoundError(args.dxf)

    args.out.mkdir(parents=True, exist_ok=True)

    doc = ezdxf.readfile(args.dxf)
    model = doc.modelspace()

    arcs = []
    door_lines = []
    walls = []
    skipped = Counter()

    for entity in model:
        layer = entity.dxf.layer
        kind = entity.dxftype()

        if layer == DOOR_LAYER and kind == "ARC":
            arcs.append(make_arc(entity))

        elif layer == DOOR_LAYER:
            record = make_record(entity)

            if record is None:
                skipped[f"DOOR_{kind}"] += 1
            else:
                door_lines.append(record)

        elif layer in WALL_LAYERS:
            record = make_record(entity)

            if record is not None:
                walls.append(record)
            else:
                skipped[f"WALL_{kind}"] += 1

    groups, relations = connected_components(arcs)
    v3_candidates = load_v3(args.v3)

    assemblies = []

    for index, group in enumerate(groups):
        assembly_id = f"assembly_{index:03d}"

        line_matches = associate_lines(
            group, door_lines
        )

        candidate_matches = {
            arc["handle"]: match_v3(
                arc, v3_candidates
            )
            for arc in group
        }

        # This is an arc-connectivity group, not a
        # verified physical door assembly.
        record = {
            "assembly_id": assembly_id,
            "arc_count": len(group),
            "arc_handles": [
                arc["handle"] for arc in group
            ],
            "arcs": [
                {
                    key: value
                    for key, value in arc.items()
                    if key != "paths"
                }
                for arc in group
            ],
            "nearby_door_line_handles": line_matches,
            "v3_candidate_matches": candidate_matches,
            "classification":
                "PROVISIONAL_ARC_CONNECTIVITY_GROUP",
            "verified_door_count": None,
            "opening_verified": False,
            "safe_to_cut": False,
        }

        assemblies.append(record)

        plot_assembly(
            group,
            door_lines,
            walls,
            args.out / f"{assembly_id}.png",
            (
                f"{assembly_id} — "
                f"{len(group)} connected arc(s)"
            ),
        )

    plot_overview(
        arcs, walls,
        args.out / "assembly_overview.png",
    )

    report = {
        "schema_version": "1.0",
        "source_dxf": str(args.dxf),
        "source_v3": (
            str(args.v3)
            if args.v3.exists()
            else None
        ),
        "arc_count": len(arcs),
        "connectivity_group_count": len(groups),
        "multi_arc_group_count": sum(
            len(group) > 1 for group in groups
        ),
        "door_line_primitive_count": len(door_lines),
        "wall_primitive_count": len(walls),
        "arc_relations": relations,
        "skipped_entities": dict(skipped),
        "assemblies": assemblies,
        "limitations": [
            "Groups use shared hinge or shared arc-tip geometry.",
            "Touching arcs alone are not merged.",
            "Nearby lines are descriptive, not verified door leaves.",
            "Unexpanded INSERT entities are not interpreted.",
            "Connectivity groups may not correspond one-to-one "
            "with physical door assemblies.",
            "No wall openings or construction states are inferred.",
        ],
        "wall_cutouts_generated": False,
    }

    output_json = args.out / "assembly_report.json"

    with output_json.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    print("\n=== CAD DOOR ASSEMBLY ANALYSIS ===")
    print("Original door arcs:", len(arcs))
    print("Arc connectivity groups:", len(groups))
    print(
        "Multi-arc groups:",
        report["multi_arc_group_count"],
    )
    print("Door line primitives:", len(door_lines))
    print("Wall primitives:", len(walls))
    print("Skipped:", dict(skipped))

    print("\nGroups:")

    for item in assemblies:
        print(
            f"  {item['assembly_id']}: "
            f"{item['arc_count']} arc(s), "
            f"{len(item['nearby_door_line_handles'])} "
            f"nearby door lines"
        )

    print("\nReport:", output_json)
    print("Images:", args.out)
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
