
"""
Under-construction — CAD Window Geometry Reconstruction
Step 41: Geometry inventory and connectivity screening

Input:
    data/floorplans/floor_2.dxf

Outputs:
    outputs/cad_openings/windows/window_report.json
    outputs/cad_openings/windows/window_overview.png
    outputs/cad_openings/windows/groups/window_group_XXXXX.png

No wall cutouts or construction progress are inferred.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"
OUTPUT = ROOT / "outputs/cad_openings/windows"

WINDOW_LAYER = "010 Fenster"
FACADE_LAYERS = {
    "011 Fassadensystem",
    "012 Vorhangfassade",
}

# CAD drawing units are metres.
# This is a connectivity tolerance, not a window-size rule.
CONNECT_TOLERANCE = 0.012
MIN_LENGTH = 0.0001
ARC_SAMPLES = 48


def xy(point):
    return [float(point[0]), float(point[1])]


def distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def point_to_segment_distance(p, a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length2 = dx * dx + dy * dy

    if length2 < 1e-16:
        return distance(p, a)

    t = max(
        0.0,
        min(
            1.0,
            ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy)
            / length2,
        ),
    )

    return distance(
        p,
        [a[0] + t * dx, a[1] + t * dy],
    )


def polyline_segments(points):
    return [
        (a, b)
        for a, b in zip(points[:-1], points[1:])
        if distance(a, b) > MIN_LENGTH
    ]


def arc_points(entity):
    center = xy(entity.dxf.center)
    radius = float(entity.dxf.radius)

    start = math.radians(float(entity.dxf.start_angle))
    end = math.radians(float(entity.dxf.end_angle))

    sweep = (end - start) % (2 * math.pi)

    if sweep < 1e-9:
        sweep = 2 * math.pi

    angles = np.linspace(
        start,
        start + sweep,
        ARC_SAMPLES,
    )

    return [
        [
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
        ]
        for angle in angles
    ]


def extract_entity(entity):
    kind = entity.dxftype()
    handle = str(entity.dxf.handle)

    if kind == "LINE":
        points = [
            xy(entity.dxf.start),
            xy(entity.dxf.end),
        ]

    elif kind == "ARC":
        points = arc_points(entity)

    elif kind == "CIRCLE":
        center = xy(entity.dxf.center)
        radius = float(entity.dxf.radius)

        points = [
            [
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            ]
            for angle in np.linspace(
                0, 2 * math.pi, ARC_SAMPLES + 1
            )
        ]

    elif kind == "LWPOLYLINE":
        # Preserve straight segments only.
        # Bulged segments require curved-geometry handling.
        raw = list(entity.get_points("xyb"))

        if not raw:
            return None, "EMPTY_POLYLINE"

        if any(abs(float(p[2])) > 1e-9 for p in raw):
            return None, "BULGED_POLYLINE_NOT_SUPPORTED"

        points = [
            [float(p[0]), float(p[1])]
            for p in raw
        ]

        if entity.closed:
            points.append(points[0])

    elif kind == "POLYLINE":
        if not entity.is_2d_polyline:
            return None, "NON_2D_POLYLINE"

        vertices = list(entity.vertices)

        if any(
            abs(float(v.dxf.get("bulge", 0.0))) > 1e-9
            for v in vertices
        ):
            return None, "BULGED_POLYLINE_NOT_SUPPORTED"

        points = [
            xy(vertex.dxf.location)
            for vertex in vertices
        ]

        if entity.is_closed and points:
            points.append(points[0])

    else:
        return None, f"UNSUPPORTED_{kind}"

    if len(points) < 2:
        return None, "TOO_FEW_POINTS"

    segments = polyline_segments(points)

    if not segments:
        return None, "ZERO_LENGTH"

    return {
        "handle": handle,
        "entity_type": kind,
        "layer": entity.dxf.layer,
        "points": points,
        "start": points[0],
        "end": points[-1],
        "length_m": sum(
            distance(a, b)
            for a, b in segments
        ),
        "bbox": [
            min(p[0] for p in points),
            min(p[1] for p in points),
            max(p[0] for p in points),
            max(p[1] for p in points),
        ],
    }, None


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[
                self.parent[value]
            ]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return

        if self.rank[a] < self.rank[b]:
            a, b = b, a

        self.parent[b] = a

        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def endpoint_groups(primitives):
    """
    Group primitives by endpoint connectivity only.

    This deliberately avoids treating a line crossing
    another line as a confirmed connection.
    """

    uf = UnionFind(len(primitives))
    cells = defaultdict(list)

    def cell(point):
        return (
            math.floor(point[0] / CONNECT_TOLERANCE),
            math.floor(point[1] / CONNECT_TOLERANCE),
        )

    for index, primitive in enumerate(primitives):
        endpoints = [
            primitive["start"],
            primitive["end"],
        ]

        for point in endpoints:
            cx, cy = cell(point)

            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    for other_index, other_point in cells[
                        (cx + ox, cy + oy)
                    ]:
                        if distance(
                            point, other_point
                        ) <= CONNECT_TOLERANCE:
                            uf.union(index, other_index)

            cells[(cx, cy)].append((index, point))

    groups = defaultdict(list)

    for index, primitive in enumerate(primitives):
        groups[uf.find(index)].append(primitive)

    return list(groups.values())


def group_bbox(group):
    return [
        min(p["bbox"][0] for p in group),
        min(p["bbox"][1] for p in group),
        max(p["bbox"][2] for p in group),
        max(p["bbox"][3] for p in group),
    ]


def make_group_record(index, group):
    bbox = group_bbox(group)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return {
        "group_id": f"window_group_{index:05d}",
        "source_layer": WINDOW_LAYER,
        "entity_count": len(group),
        "entity_types": dict(
            Counter(p["entity_type"] for p in group)
        ),
        "source_handles": [
            p["handle"] for p in group
        ],
        "bbox": [
            round(value, 6) for value in bbox
        ],
        "bbox_width_m": round(width, 6),
        "bbox_height_m": round(height, 6),
        "total_drawn_length_m": round(
            sum(p["length_m"] for p in group), 6
        ),
        "physical_window_count": None,
        "window_opening_verified": False,
        "wall_association_verified": False,
        "safe_to_cut": False,
        "screening_status": "CONNECTED_CAD_PRIMITIVES_ONLY",
    }


def plot_primitives(ax, primitives, color, linewidth):
    for primitive in primitives:
        points = np.asarray(
            primitive["points"],
            dtype=float,
        )

        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=0.85,
        )


def save_overview(primitives, facade_primitives, groups):
    fig, ax = plt.subplots(figsize=(17, 11))

    plot_primitives(
        ax, facade_primitives,
        "#cbd5e1", 0.45
    )

    plot_primitives(
        ax, primitives,
        "#2563eb", 0.7
    )

    for index, group in enumerate(groups):
        bbox = group_bbox(group)

        x = (bbox[0] + bbox[2]) / 2
        y = (bbox[1] + bbox[3]) / 2

        ax.text(
            x, y, str(index),
            fontsize=6,
            color="#dc2626",
            ha="center",
            va="center",
        )

    ax.set_title(
        "CAD window-layer connectivity overview\n"
        "Blue: window primitives | "
        "Grey: facade primitives | "
        "Red: group IDs\n"
        "Connected groups are NOT verified physical windows"
    )

    ax.set_xlabel("DXF X (m)")
    ax.set_ylabel("DXF Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "window_overview.png",
        dpi=160,
    )
    plt.close(fig)


def save_closeups(groups):
    folder = OUTPUT / "groups"
    folder.mkdir(parents=True, exist_ok=True)

    for index, group in enumerate(groups):
        fig, ax = plt.subplots(figsize=(7, 6))

        plot_primitives(
            ax, group,
            "#2563eb", 1.5
        )

        bbox = group_bbox(group)

        span = max(
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            0.3,
        )

        margin = max(0.15, span * 0.15)

        ax.set_xlim(
            bbox[0] - margin,
            bbox[2] + margin,
        )

        ax.set_ylim(
            bbox[1] - margin,
            bbox[3] + margin,
        )

        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

        ax.set_title(
            f"Window group {index:05d} — "
            f"{len(group)} CAD entities\n"
            "Physical window count unverified"
        )

        fig.tight_layout()
        fig.savefig(
            folder / f"window_group_{index:05d}.png",
            dpi=120,
        )
        plt.close(fig)


def main():
    if not DXF_PATH.is_file():
        raise FileNotFoundError(DXF_PATH)

    OUTPUT.mkdir(
        parents=True, exist_ok=True
    )

    document = ezdxf.readfile(str(DXF_PATH))
    modelspace = document.modelspace()

    window_primitives = []
    facade_primitives = []

    skipped = Counter()
    layer_counts = Counter()

    for entity in modelspace:
        layer = entity.dxf.layer

        if (
            layer != WINDOW_LAYER
            and layer not in FACADE_LAYERS
        ):
            continue

        layer_counts[
            f"{layer} | {entity.dxftype()}"
        ] += 1

        primitive, reason = extract_entity(entity)

        if primitive is None:
            skipped[
                f"{layer} | {reason}"
            ] += 1
            continue

        if layer == WINDOW_LAYER:
            window_primitives.append(primitive)
        else:
            facade_primitives.append(primitive)

    groups = endpoint_groups(window_primitives)

    groups.sort(
        key=lambda group: (
            -len(group),
            group_bbox(group)[0],
            group_bbox(group)[1],
        )
    )

    records = [
        make_group_record(index, group)
        for index, group in enumerate(groups)
    ]

    report = {
        "schema_version": "1.0",
        "source_dxf": str(
            DXF_PATH.relative_to(ROOT)
        ),
        "coordinate_system": "DXF_METERS",
        "window_layer": WINDOW_LAYER,
        "facade_layers": sorted(FACADE_LAYERS),
        "layer_entity_counts": dict(layer_counts),
        "window_primitive_count": len(
            window_primitives
        ),
        "facade_primitive_count": len(
            facade_primitives
        ),
        "connected_window_groups": len(groups),
        "physical_window_count": None,
        "verified_window_openings": 0,
        "wall_cutouts_generated": False,
        "grouping_method": (
            "ENDPOINT_CONNECTIVITY_WITH_0.012_M_TOLERANCE"
        ),
        "skipped_entities": dict(skipped),
        "limitations": [
            "Connected CAD entities may describe several windows.",
            "Disconnected entities may describe one window.",
            "Crossing segments are not automatically connected.",
            "INSERT blocks are recorded but not expanded.",
            "MTEXT annotations are excluded.",
            "Facade systems are inventoried separately.",
            "No wall-opening verification has been performed.",
        ],
        "groups": records,
    }

    report_path = OUTPUT / "window_report.json"

    with report_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report, file,
            indent=2,
            allow_nan=False,
        )

    save_overview(
        window_primitives,
        facade_primitives,
        groups,
    )

    save_closeups(groups)

    print("\n=== CAD WINDOW RECONSTRUCTION ===")
    print("Window primitives:", len(window_primitives))
    print("Facade primitives:", len(facade_primitives))
    print("Connected window groups:", len(groups))
    print("Skipped entities:", dict(skipped))
    print("Physical window count: UNVERIFIED")
    print("Verified openings: 0")
    print("Wall cutouts generated: NO")
    print()
    print("JSON:", report_path)
    print(
        "Overview:",
        OUTPUT / "window_overview.png",
    )
    print(
        "Closeups:",
        OUTPUT / "groups",
    )


if __name__ == "__main__":
    main()
