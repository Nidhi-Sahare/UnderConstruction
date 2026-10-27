
"""
Under-construction — Door Geometry Diagnostics

Inputs:
    data/digital_twin/building_elements.json
    outputs/cad_openings/opening_candidates_v3.json

Outputs:
    outputs/cad_openings/diagnostics/door_diagnostics.json
    outputs/cad_openings/diagnostics/door_overview.png
    outputs/cad_openings/diagnostics/door_arc_XXXXX.png

Requires:
    matplotlib

No geometry modifications, wall cutouts, or progress inference.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WALLS = (
    ROOT / "data/digital_twin/building_elements.json"
)
DEFAULT_OPENINGS = (
    ROOT / "outputs/cad_openings/opening_candidates_v3.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/cad_openings/diagnostics"
)

SEARCH_RADIUS = 2.0
ENDPOINT_RADIUS = 0.45
WALL_CONTACT_TOLERANCE = 0.10


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def segment_distance(p, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    denominator = dx * dx + dy * dy

    if denominator < 1e-12:
        return dist(p, a)

    t = (
        (p[0] - a[0]) * dx +
        (p[1] - a[1]) * dy
    ) / denominator

    t = max(0.0, min(1.0, t))

    closest = (
        a[0] + t * dx,
        a[1] + t * dy,
    )

    return dist(p, closest)


def load_walls(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    elements = (
        data if isinstance(data, list)
        else data["elements"]
    )

    walls = []

    for item in elements:
        a = item.get("start")
        b = item.get("end")

        if not a or not b:
            continue

        a = [float(a[0]), float(a[1])]
        b = [float(b[0]), float(b[1])]

        if dist(a, b) < 0.01:
            continue

        walls.append({
            "id": str(item["element_id"]),
            "category": item.get("category", "unknown"),
            "start": a,
            "end": b,
        })

    return walls


def arc_points(candidate, count=100):
    center = candidate["hinge_candidate"]
    radius = float(candidate["arc_radius"])

    first, last = candidate["arc_endpoints"]

    start = math.atan2(
        first[1] - center[1],
        first[0] - center[0],
    )

    end = math.atan2(
        last[1] - center[1],
        last[0] - center[0],
    )

    sweep = math.radians(
        float(candidate["arc_sweep_degrees"])
    )

    # DXF arcs run counterclockwise from start to end.
    # Preserve the original sweep, including wraparound.
    angles = [
        start + sweep * i / (count - 1)
        for i in range(count)
    ]

    return [
        [
            center[0] + radius * math.cos(t),
            center[1] + radius * math.sin(t),
        ]
        for t in angles
    ]


def nearby_walls(candidate, walls):
    center = candidate["hinge_candidate"]

    return [
        wall
        for wall in walls
        if segment_distance(
            center,
            wall["start"],
            wall["end"],
        ) <= SEARCH_RADIUS
    ]


def endpoints_near(point, walls, radius):
    results = []

    for wall in walls:
        for side in ("start", "end"):
            p = wall[side]
            d = dist(point, p)

            if d <= radius:
                results.append({
                    "element_id": wall["id"],
                    "side": side,
                    "point": p,
                    "distance": round(d, 5),
                })

    results.sort(key=lambda x: x["distance"])

    return results


def analyze(candidate, walls):
    hinge = candidate["hinge_candidate"]
    arc_ends = candidate["arc_endpoints"]

    local_walls = nearby_walls(candidate, walls)

    near_hinge = endpoints_near(
        hinge, local_walls, ENDPOINT_RADIUS
    )

    near_arc_ends = [
        endpoints_near(p, local_walls, ENDPOINT_RADIUS)
        for p in arc_ends
    ]

    # Distances from hinge and possible closed-door tips
    # to wall centerlines. These are descriptive only.
    wall_distances = sorted(
        (
            segment_distance(
                hinge,
                wall["start"],
                wall["end"],
            ),
            wall["id"],
        )
        for wall in local_walls
    )

    tip_distances = []

    for tip in arc_ends:
        tip_distances.append(sorted(
            (
                segment_distance(
                    tip,
                    wall["start"],
                    wall["end"],
                ),
                wall["id"],
            )
            for wall in local_walls
        )[:5])

    # Check whether wall centerlines pass close to the
    # middle of a possible closed-door span.
    # This is NOT a proof that a wall blocks the opening.
    span_observations = []

    for tip_index, tip in enumerate(arc_ends):
        midpoint = [
            (hinge[0] + tip[0]) / 2,
            (hinge[1] + tip[1]) / 2,
        ]

        close_walls = [
            wall["id"]
            for wall in local_walls
            if segment_distance(
                midpoint,
                wall["start"],
                wall["end"],
            ) <= WALL_CONTACT_TOLERANCE
        ]

        span_observations.append({
            "arc_endpoint_choice": tip_index,
            "midpoint": midpoint,
            "wall_centerlines_near_span_midpoint": close_walls,
        })

    if not local_walls:
        observation = "NO_WALLS_WITHIN_SEARCH_RADIUS"
    elif not near_hinge:
        observation = "NO_WALL_ENDPOINT_NEAR_HINGE"
    elif not any(near_arc_ends):
        observation = "NO_WALL_ENDPOINT_NEAR_EITHER_ARC_TIP"
    elif candidate["association_status"] == "AMBIGUOUS_PHYSICAL_GAPS":
        observation = "MULTIPLE_GAP_HYPOTHESES_REQUIRE_REVIEW"
    else:
        observation = "ENDPOINTS_EXIST_BUT_MATCHING_CONSTRAINTS_FAILED"

    return {
        "candidate_id": candidate["candidate_id"],
        "association_status": candidate["association_status"],
        "observation": observation,
        "hinge": hinge,
        "radius": candidate["arc_radius"],
        "arc_endpoints": arc_ends,
        "nearby_wall_count": len(local_walls),
        "nearby_wall_element_ids": [
            wall["id"] for wall in local_walls
        ],
        "hinge_endpoint_candidates": near_hinge[:20],
        "arc_tip_endpoint_candidates": [
            matches[:20] for matches in near_arc_ends
        ],
        "nearest_wall_centerlines_to_hinge": [
            {
                "distance": round(d, 5),
                "element_id": wall_id,
            }
            for d, wall_id in wall_distances[:5]
        ],
        "nearest_wall_centerlines_to_arc_tips": [
            [
                {
                    "distance": round(d, 5),
                    "element_id": wall_id,
                }
                for d, wall_id in matches
            ]
            for matches in tip_distances
        ],
        "closed_span_midpoint_observations":
            span_observations,
        "physical_gap_hypotheses":
            candidate.get("physical_gap_hypotheses", []),
        "physically_verified": False,
        "safe_to_cut": False,
    }


def draw_diagnostic(candidate, walls, output):
    hinge = candidate["hinge_candidate"]
    radius = float(candidate["arc_radius"])
    local = nearby_walls(candidate, walls)

    fig, ax = plt.subplots(figsize=(9, 9))

    for wall in local:
        a, b = wall["start"], wall["end"]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#64748b",
            linewidth=1.6,
            alpha=0.8,
            zorder=2,
        )

        ax.scatter(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#94a3b8",
            s=14,
            zorder=3,
        )

    points = arc_points(candidate)

    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        color="#f59e0b",
        linewidth=2.5,
        zorder=5,
        label="CAD swing arc",
    )

    ax.scatter(
        [hinge[0]],
        [hinge[1]],
        color="#ef4444",
        s=80,
        marker="x",
        zorder=7,
        label="Possible hinge",
    )

    for i, tip in enumerate(candidate["arc_endpoints"]):
        ax.plot(
            [hinge[0], tip[0]],
            [hinge[1], tip[1]],
            linestyle="--",
            color="#f59e0b",
            linewidth=1.2,
            alpha=0.7,
        )

        ax.scatter(
            [tip[0]],
            [tip[1]],
            color="#fbbf24",
            s=40,
            zorder=6,
        )

        ax.annotate(
            f"Tip {i}",
            (tip[0], tip[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    colors = [
        "#22c55e",
        "#38bdf8",
        "#c084fc",
        "#fb7185",
        "#eab308",
    ]

    hypotheses = candidate.get(
        "physical_gap_hypotheses", []
    )

    for i, hypothesis in enumerate(hypotheses):
        a = hypothesis["representative_gap_start"]
        b = hypothesis["representative_gap_end"]

        color = colors[i % len(colors)]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            linewidth=4,
            alpha=0.85,
            zorder=8,
            label=(
                f"Gap {i}: "
                f"score {hypothesis['representative_score']:.3f}"
            ),
        )

    ax.add_patch(
        Circle(
            tuple(hinge),
            SEARCH_RADIUS,
            fill=False,
            edgecolor="#475569",
            linestyle=":",
            linewidth=1,
        )
    )

    ax.set_xlim(
        hinge[0] - SEARCH_RADIUS - 0.3,
        hinge[0] + SEARCH_RADIUS + 0.3,
    )

    ax.set_ylim(
        hinge[1] - SEARCH_RADIUS - 0.3,
        hinge[1] + SEARCH_RADIUS + 0.3,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.15)

    ax.set_title(
        f"{candidate['candidate_id']}\n"
        f"{candidate['association_status']}",
        fontsize=12,
    )

    ax.set_xlabel("DXF X coordinate")
    ax.set_ylabel("DXF Y coordinate")

    ax.legend(
        loc="upper right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def draw_overview(candidates, walls, output):
    fig, ax = plt.subplots(figsize=(16, 12))

    for wall in walls:
        a, b = wall["start"], wall["end"]

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#94a3b8",
            linewidth=0.6,
            alpha=0.5,
        )

    status_colors = {
        "NO_SUPPORTED_GAP": "#ef4444",
        "AMBIGUOUS_PHYSICAL_GAPS": "#f59e0b",
    }

    for candidate in candidates:
        x, y = candidate["hinge_candidate"]
        color = status_colors.get(
            candidate["association_status"],
            "#64748b",
        )

        ax.scatter(
            [x], [y],
            color=color,
            s=45,
            zorder=5,
        )

        label = candidate["candidate_id"].split("_")[-1]

        ax.annotate(
            label,
            (x, y),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=7,
            color=color,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "Under-construction — unresolved CAD door arcs"
    )
    ax.set_xlabel("DXF X coordinate")
    ax.set_ylabel("DXF Y coordinate")
    ax.grid(alpha=0.1)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--walls", type=Path, default=DEFAULT_WALLS
    )
    parser.add_argument(
        "--openings", type=Path, default=DEFAULT_OPENINGS
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT
    )

    args = parser.parse_args()

    walls = load_walls(args.walls)

    with args.openings.open(
        "r", encoding="utf-8"
    ) as f:
        report = json.load(f)

    if report.get("schema_version") != "3.0":
        raise ValueError("Expected v3 opening candidates")

    unresolved = [
        candidate
        for candidate in report["candidates"]
        if candidate.get("method")
        == "SWING_ARC_AND_WALL_GAP"
        and candidate.get("association_status") in {
            "NO_SUPPORTED_GAP",
            "AMBIGUOUS_PHYSICAL_GAPS",
        }
    ]

    args.output.mkdir(
        parents=True, exist_ok=True
    )

    diagnostics = []

    for candidate in unresolved:
        result = analyze(candidate, walls)
        diagnostics.append(result)

        image_path = (
            args.output /
            f"{candidate['candidate_id']}.png"
        )

        draw_diagnostic(
            candidate, walls, image_path
        )

    draw_overview(
        unresolved,
        walls,
        args.output / "door_overview.png",
    )

    output_json = (
        args.output / "door_diagnostics.json"
    )

    payload = {
        "schema_version": "1.0",
        "source_openings": str(args.openings),
        "source_walls": str(args.walls),
        "unresolved_count": len(unresolved),
        "observation_counts": dict(Counter(
            item["observation"]
            for item in diagnostics
        )),
        "notes": [
            "Observations describe geometric proximity only.",
            "Wall centerline proximity does not prove blockage.",
            "The absence of endpoints does not prove the absence of a door.",
            "No geometry was modified.",
        ],
        "diagnostics": diagnostics,
    }

    with output_json.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(payload, f, indent=2)

    print("\n=== DOOR GEOMETRY DIAGNOSTICS ===")
    print("Unresolved door arcs:", len(unresolved))
    print("Observations:", payload["observation_counts"])

    for item in diagnostics:
        print(
            f"{item['candidate_id']}: "
            f"{item['observation']} "
            f"(walls={item['nearby_wall_count']})"
        )

    print("\nOverview:", args.output / "door_overview.png")
    print("JSON:", output_json)
    print("Individual images:", len(unresolved))
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
