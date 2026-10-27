
"""
Under-construction — CAD Door Span Reconstruction

Reads:
    data/floorplans/floor_2.dxf

Writes:
    outputs/cad_openings/spans/door_spans.json
    outputs/cad_openings/spans/door_spans.png

Reconstructs provisional opening spans using original CAD
door-swing geometry. Does not modify walls or infer progress.

Requires:
    ezdxf
    matplotlib
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


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DXF = ROOT / "data/floorplans/floor_2.dxf"
DEFAULT_OUT = ROOT / "outputs/cad_openings/spans"

DOOR_LAYER = "010 Türen"

# General geometric tolerances in drawing units.
MIN_SWEEP_DEG = 25.0
MIN_RADIUS = 0.50
MAX_RADIUS = 1.50

SAME_SPAN_ENDPOINT_TOL = 0.12
SAME_SPAN_ANGLE_TOL = 12.0
MAX_ASSOCIATED_LINE_DISTANCE = 0.10


def xy(value):
    return [float(value[0]), float(value[1])]


def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def midpoint(a, b):
    return [
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
    ]


def point_segment_distance(p, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    denominator = dx * dx + dy * dy

    if denominator < 1e-12:
        return distance(p, a)

    t = (
        (p[0] - a[0]) * dx +
        (p[1] - a[1]) * dy
    ) / denominator

    t = max(0.0, min(1.0, t))

    closest = [
        a[0] + t * dx,
        a[1] + t * dy,
    ]

    return distance(p, closest)


def arc_endpoints(center, radius, start, end):
    def at(angle):
        angle = math.radians(angle)
        return [
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        ]

    return [at(start), at(end)]


def arc_points(arc, count=64):
    sweep = arc["sweep_degrees"]

    return [
        [
            arc["center"][0] +
            arc["radius"] * math.cos(
                math.radians(
                    arc["start_angle"] +
                    sweep * i / (count - 1)
                )
            ),
            arc["center"][1] +
            arc["radius"] * math.sin(
                math.radians(
                    arc["start_angle"] +
                    sweep * i / (count - 1)
                )
            ),
        ]
        for i in range(count)
    ]


def extract_arcs(model):
    arcs = []

    for entity in model:
        if (
            entity.dxf.layer != DOOR_LAYER
            or entity.dxftype() != "ARC"
        ):
            continue

        center = xy(entity.dxf.center)
        radius = float(entity.dxf.radius)

        start = float(entity.dxf.start_angle)
        end = float(entity.dxf.end_angle)
        sweep = (end - start) % 360.0

        tips = arc_endpoints(
            center, radius, start, end
        )

        arcs.append({
            "handle": str(entity.dxf.handle),
            "center": center,
            "radius": radius,
            "start_angle": start,
            "end_angle": end,
            "sweep_degrees": sweep,
            "arc_endpoints": tips,
        })

    return arcs


def extract_door_lines(model):
    lines = []

    for entity in model:
        if (
            entity.dxf.layer != DOOR_LAYER
            or entity.dxftype() != "LINE"
        ):
            continue

        lines.append({
            "handle": str(entity.dxf.handle),
            "start": xy(entity.dxf.start),
            "end": xy(entity.dxf.end),
        })

    return lines


def candidate_spans(arc):
    """
    A swing arc has two radial directions.

    Each gives a possible closed/open door-leaf span:
        hinge center -> arc endpoint

    We retain both alternatives until the door-layer
    lines or wall footprint can distinguish them.
    """

    spans = []

    for index, tip in enumerate(
        arc["arc_endpoints"]
    ):
        spans.append({
            "alternative_index": index,
            "start": arc["center"],
            "end": tip,
            "length": round(
                distance(arc["center"], tip), 6
            ),
            "interpretation":
                "HINGE_TO_SWING_ENDPOINT",
        })

    return spans


def line_support(span, lines):
    """
    Finds door-layer lines that geometrically follow a
    proposed leaf span.

    This is evidence of a drawn leaf, not proof that
    the span is the wall opening.
    """

    a = span["start"]
    b = span["end"]

    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length = math.hypot(dx, dy)

    if length < 1e-9:
        return []

    ux = dx / length
    uy = dy / length

    matches = []

    for line in lines:
        p = line["start"]
        q = line["end"]

        lp = point_segment_distance(p, a, b)
        lq = point_segment_distance(q, a, b)

        if max(lp, lq) > MAX_ASSOCIATED_LINE_DISTANCE:
            continue

        vx = q[0] - p[0]
        vy = q[1] - p[1]
        line_length = math.hypot(vx, vy)

        if line_length < 0.20:
            continue

        alignment = abs(
            (vx / line_length) * ux +
            (vy / line_length) * uy
        )

        if alignment < math.cos(
            math.radians(SAME_SPAN_ANGLE_TOL)
        ):
            continue

        matches.append({
            "handle": line["handle"],
            "line_length": round(line_length, 6),
            "alignment": round(alignment, 6),
            "endpoint_distance": round(
                max(lp, lq), 6
            ),
        })

    return sorted(
        matches,
        key=lambda item: (
            -item["line_length"],
            item["endpoint_distance"],
        ),
    )


def equivalent_spans(a, b):
    """
    Equivalent spans must have matching physical
    endpoints, independent of direction.
    """

    direct = max(
        distance(a["start"], b["start"]),
        distance(a["end"], b["end"]),
    )

    reverse = max(
        distance(a["start"], b["end"]),
        distance(a["end"], b["start"]),
    )

    return min(direct, reverse) <= SAME_SPAN_ENDPOINT_TOL


def cluster_spans(spans):
    """
    Complete-link clustering prevents chaining
    unrelated neighboring door spans together.
    """

    groups = []

    for span in spans:
        compatible = [
            group
            for group in groups
            if all(
                equivalent_spans(span, member)
                for member in group
            )
        ]

        if len(compatible) == 1:
            compatible[0].append(span)
        else:
            groups.append([span])

    return groups


def reconstruct(arcs, lines):
    alternatives = []
    excluded = []

    for arc in arcs:
        if not (
            MIN_RADIUS <= arc["radius"] <= MAX_RADIUS
        ):
            excluded.append({
                "handle": arc["handle"],
                "reason": "RADIUS_OUTSIDE_RANGE",
            })
            continue

        if arc["sweep_degrees"] < MIN_SWEEP_DEG:
            excluded.append({
                "handle": arc["handle"],
                "reason": "SHORT_SWEEP",
                "sweep_degrees": arc["sweep_degrees"],
            })
            continue

        for span in candidate_spans(arc):
            support = line_support(span, lines)

            alternatives.append({
                "source_arc_handle": arc["handle"],
                "alternative_index":
                    span["alternative_index"],
                "start": span["start"],
                "end": span["end"],
                "length": span["length"],
                "door_line_support": support,
                "support_count": len(support),
                "interpretation":
                    span["interpretation"],
            })

    groups = cluster_spans(alternatives)

    reconstructed = []

    for index, group in enumerate(groups):
        representative = max(
            group,
            key=lambda item: (
                item["support_count"],
                -item["alternative_index"],
            ),
        )

        source_handles = sorted({
            item["source_arc_handle"]
            for item in group
        })

        line_handles = sorted({
            support["handle"]
            for item in group
            for support in item["door_line_support"]
        })

        reconstructed.append({
            "span_id": f"door_span_{index:03d}",
            "start": representative["start"],
            "end": representative["end"],
            "length": representative["length"],
            "source_arc_handles": source_handles,
            "source_line_handles": line_handles,
            "representation_count": len(group),
            "alternatives": group,
            "status": "PROVISIONAL_CAD_DOOR_LEAF_SPAN",
            "is_wall_opening": None,
            "opening_verified": False,
            "safe_to_cut": False,
        })

    return reconstructed, excluded


def draw(arcs, spans, output):
    fig, ax = plt.subplots(figsize=(16, 12))

    for arc in arcs:
        points = arc_points(arc)

        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            color="#94a3b8",
            linewidth=0.8,
            alpha=0.65,
        )

    for span in spans:
        a = span["start"]
        b = span["end"]

        supported = bool(
            span["source_line_handles"]
        )

        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=(
                "#2563eb" if supported
                else "#f59e0b"
            ),
            linewidth=1.8,
            alpha=0.8,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        "CAD-derived provisional door-leaf spans\n"
        "Blue: associated door-layer lines; "
        "Orange: no associated lines"
    )
    ax.set_xlabel("DXF X")
    ax.set_ylabel("DXF Y")
    ax.grid(alpha=0.12)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dxf",
        type=Path,
        default=DEFAULT_DXF,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    doc = ezdxf.readfile(args.dxf)
    model = doc.modelspace()

    arcs = extract_arcs(model)
    lines = extract_door_lines(model)

    spans, excluded = reconstruct(
        arcs, lines
    )

    args.out.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "schema_version": "1.0",
        "source_dxf": str(args.dxf),
        "drawing_units_code":
            doc.header.get("$INSUNITS"),
        "door_arc_count": len(arcs),
        "door_line_count": len(lines),
        "provisional_span_count": len(spans),
        "excluded_arcs": excluded,
        "spans": spans,
        "limitations": [
            "A radial door-leaf span is not necessarily "
            "the architectural opening span.",
            "Both arc-endpoint alternatives are retained.",
            "A line aligned with a span may belong to "
            "an adjacent door symbol.",
            "Overlapping arc chains are not automatically "
            "classified as one doorway.",
            "No wall footprint intersection has been tested.",
            "No wall cutouts or progress inference performed.",
        ],
        "wall_cutouts_generated": False,
    }

    output_json = (
        args.out / "door_spans.json"
    )

    with output_json.open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    draw(
        arcs,
        spans,
        args.out / "door_spans.png",
    )

    supported = sum(
        bool(span["source_line_handles"])
        for span in spans
    )

    print("\n=== CAD DOOR SPAN RECONSTRUCTION ===")
    print("Original door arcs:", len(arcs))
    print("Door-layer lines:", len(lines))
    print("Provisional spans:", len(spans))
    print("Spans with line support:", supported)
    print("Excluded arcs:", len(excluded))

    print(
        "Excluded reasons:",
        dict(Counter(
            item["reason"]
            for item in excluded
        )),
    )

    print("\nJSON:", output_json)
    print(
        "Preview:",
        args.out / "door_spans.png",
    )
    print("Verified wall openings: 0")
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
