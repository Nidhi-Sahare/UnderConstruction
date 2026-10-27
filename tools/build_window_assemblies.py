
"""
Under-construction — Build CAD Window Assemblies

Inputs:
    data/floorplans/floor_2.dxf
    outputs/cad_openings/windows/window_report.json

Outputs:
    outputs/cad_openings/windows/window_assemblies.json
    outputs/cad_openings/windows/window_assemblies.png

Purpose:
    Convert connected window-layer CAD primitives into
    provisional, straight, 3D-ready window assemblies.

Safety:
    - No wall cutouts
    - No inferred construction progress
    - No verified physical window counts
    - No invented sill or glazing heights
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"

REPORT_PATH = (
    ROOT
    / "outputs/cad_openings/windows/window_report.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs/cad_openings/windows"
)

OUTPUT_JSON = OUTPUT_DIR / "window_assemblies.json"
OUTPUT_PNG = OUTPUT_DIR / "window_assemblies.png"

WINDOW_LAYER = "010 Fenster"

# Screening parameters in DXF metres.
# These are algorithmic eligibility criteria, not
# assumed architectural window dimensions.
MIN_SPAN_M = 0.20
MAX_SPAN_M = 12.0
MIN_ASPECT_RATIO = 2.0
MAX_ORTHOGONAL_SPREAD_RATIO = 0.35

# The group report already establishes connectivity.
# This script does not merge neighboring groups.
ARC_SAMPLES = 48


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def sample_arc(entity):
    center = np.array(
        [entity.dxf.center.x, entity.dxf.center.y],
        dtype=float,
    )

    radius = float(entity.dxf.radius)

    start = math.radians(float(entity.dxf.start_angle))
    end = math.radians(float(entity.dxf.end_angle))

    sweep = (end - start) % (2 * math.pi)

    if sweep < 1e-10:
        sweep = 2 * math.pi

    angles = np.linspace(
        start,
        start + sweep,
        ARC_SAMPLES,
    )

    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])


def entity_points(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        return np.array([
            [entity.dxf.start.x, entity.dxf.start.y],
            [entity.dxf.end.x, entity.dxf.end.y],
        ], dtype=float)

    if kind == "ARC":
        return sample_arc(entity)

    if kind == "CIRCLE":
        center = entity.dxf.center
        radius = float(entity.dxf.radius)

        angles = np.linspace(
            0, 2 * math.pi,
            ARC_SAMPLES + 1,
        )

        return np.column_stack([
            center.x + radius * np.cos(angles),
            center.y + radius * np.sin(angles),
        ])

    if kind == "LWPOLYLINE":
        raw = list(entity.get_points("xyb"))

        if not raw:
            return None

        if any(abs(float(p[2])) > 1e-9 for p in raw):
            return None

        points = [
            [float(p[0]), float(p[1])]
            for p in raw
        ]

        if entity.closed and points:
            points.append(points[0])

        return np.asarray(points, dtype=float)

    if kind == "POLYLINE":
        if not entity.is_2d_polyline:
            return None

        vertices = list(entity.vertices)

        if any(
            abs(float(v.dxf.get("bulge", 0.0))) > 1e-9
            for v in vertices
        ):
            return None

        points = [
            [v.dxf.location.x, v.dxf.location.y]
            for v in vertices
        ]

        if entity.is_closed and points:
            points.append(points[0])

        return np.asarray(points, dtype=float)

    return None


def principal_axis(points):
    """
    PCA estimates the dominant direction of the
    connected CAD geometry.

    The result is a geometric orientation estimate,
    not a verified wall or window centerline.
    """

    center = np.mean(points, axis=0)

    centered = points - center

    covariance = (
        centered.T @ centered
    ) / max(len(points), 1)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(eigenvalues)[::-1]

    major = eigenvectors[:, order[0]]
    minor = eigenvectors[:, order[1]]

    # Canonicalize axis direction for reproducibility.
    if (
        major[0] < -1e-9
        or (
            abs(major[0]) <= 1e-9
            and major[1] < 0
        )
    ):
        major = -major

    along = centered @ major
    across = centered @ minor

    low = float(np.min(along))
    high = float(np.max(along))

    span = high - low

    spread = float(
        np.max(across) - np.min(across)
    )

    start = center + major * low
    end = center + major * high

    return {
        "center": center,
        "axis": major,
        "start": start,
        "end": end,
        "span": span,
        "spread": spread,
        "aspect_ratio": (
            span / spread
            if spread > 1e-9
            else None
        ),
        "spread_ratio": (
            spread / span
            if span > 1e-9
            else None
        ),
    }


def classify(axis):
    span = axis["span"]
    spread = axis["spread"]
    ratio = axis["aspect_ratio"]

    if not math.isfinite(span) or span < MIN_SPAN_M:
        return "REJECTED_TOO_SHORT"

    if span > MAX_SPAN_M:
        return "REJECTED_TOO_LONG"

    if (
        ratio is not None
        and ratio < MIN_ASPECT_RATIO
    ):
        return "REJECTED_IRREGULAR_GEOMETRY"

    if (
        axis["spread_ratio"] is not None
        and axis["spread_ratio"]
        > MAX_ORTHOGONAL_SPREAD_RATIO
    ):
        return "REJECTED_WIDE_GEOMETRY"

    return "PROVISIONAL_STRAIGHT_ASSEMBLY"


def round_point(point):
    return [
        round(float(point[0]), 6),
        round(float(point[1]), 6),
    ]


def load_group_points(group, entities):
    all_points = []
    recovered_handles = []
    missing_handles = []
    unsupported_handles = []
    entity_types = Counter()

    for handle in group.get("source_handles", []):
        entity = entities.get(str(handle).upper())

        if entity is None:
            missing_handles.append(str(handle))
            continue

        if entity.dxf.layer != WINDOW_LAYER:
            unsupported_handles.append(str(handle))
            continue

        points = entity_points(entity)

        if (
            points is None
            or len(points) < 2
            or not np.all(np.isfinite(points))
        ):
            unsupported_handles.append(str(handle))
            continue

        all_points.append(points)
        recovered_handles.append(str(handle))
        entity_types[entity.dxftype()] += 1

    if not all_points:
        return (
            None,
            recovered_handles,
            missing_handles,
            unsupported_handles,
            entity_types,
        )

    return (
        np.vstack(all_points),
        recovered_handles,
        missing_handles,
        unsupported_handles,
        entity_types,
    )


def make_record(group, entities):
    group_id = str(group["group_id"])

    (
        points,
        recovered,
        missing,
        unsupported,
        entity_types,
    ) = load_group_points(group, entities)

    base = {
        "assembly_id": group_id,
        "source_group_id": group_id,
        "source_layer": WINDOW_LAYER,
        "source_handles": recovered,
        "missing_handles": missing,
        "unsupported_handles": unsupported,
        "entity_types": dict(entity_types),
        "source_entity_count": len(recovered),
        "wall_association_verified": False,
        "window_opening_verified": False,
        "construction_status": "UNKNOWN",
        "safe_to_cut": False,
        "vertical_dimensions_verified": False,
        "physical_window_count": None,
    }

    if points is None or len(points) < 2:
        return {
            **base,
            "status": "REJECTED_NO_GEOMETRY",
            "start": None,
            "end": None,
        }

    axis = principal_axis(points)
    status = classify(axis)

    if missing or unsupported:
        status = "REJECTED_INCOMPLETE_SOURCE_GEOMETRY"

    angle = math.degrees(
        math.atan2(
            axis["axis"][1],
            axis["axis"][0],
        )
    )

    return {
        **base,
        "status": status,
        "start": round_point(axis["start"]),
        "end": round_point(axis["end"]),
        "center": round_point(axis["center"]),
        "axis": round_point(axis["axis"]),
        "orientation_degrees": round(angle, 4),
        "span_m": round(axis["span"], 6),
        "orthogonal_spread_m": round(
            axis["spread"], 6
        ),
        "aspect_ratio": (
            round(axis["aspect_ratio"], 5)
            if axis["aspect_ratio"] is not None
            else None
        ),
        "orthogonal_spread_ratio": (
            round(axis["spread_ratio"], 5)
            if axis["spread_ratio"] is not None
            else None
        ),
        "representation": (
            "PCA_ESTIMATED_PLAN_SPAN"
        ),
    }


def plot_results(records):
    fig, ax = plt.subplots(figsize=(16, 11))

    colors = {
        "PROVISIONAL_STRAIGHT_ASSEMBLY": "#16a34a",
        "REJECTED_TOO_SHORT": "#ef4444",
        "REJECTED_TOO_LONG": "#ef4444",
        "REJECTED_IRREGULAR_GEOMETRY": "#f59e0b",
        "REJECTED_WIDE_GEOMETRY": "#f59e0b",
        "REJECTED_NO_GEOMETRY": "#ef4444",
        "REJECTED_INCOMPLETE_SOURCE_GEOMETRY": "#ef4444",
    }

    for record in records:
        start = record.get("start")
        end = record.get("end")

        if not start or not end:
            continue

        color = colors.get(
            record["status"],
            "#64748b",
        )

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=2.3,
            alpha=0.85,
        )

    ax.set_title(
        "CAD window assembly screening\n"
        "Green: provisional straight assemblies | "
        "Orange: irregular | Red: rejected\n"
        "No window openings or installation states verified"
    )

    ax.set_xlabel("DXF X (m)")
    ax.set_ylabel("DXF Y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=160)
    plt.close(fig)


def main():
    if not DXF_PATH.is_file():
        raise FileNotFoundError(DXF_PATH)

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(REPORT_PATH)

    OUTPUT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    with REPORT_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        report = json.load(file)

    document = ezdxf.readfile(str(DXF_PATH))

    entities = {
        str(entity.dxf.handle).upper(): entity
        for entity in document.modelspace()
        if entity.dxf.hasattr("handle")
    }

    records = [
        make_record(group, entities)
        for group in report.get("groups", [])
    ]

    counts = Counter(
        record["status"]
        for record in records
    )

    provisional = [
        record
        for record in records
        if record["status"]
        == "PROVISIONAL_STRAIGHT_ASSEMBLY"
    ]

    output = {
        "schema_version": "1.0",
        "source_dxf": str(
            DXF_PATH.relative_to(ROOT)
        ),
        "source_report": str(
            REPORT_PATH.relative_to(ROOT)
        ),
        "coordinate_system": "DXF_METERS",
        "assembly_count": len(records),
        "provisional_assembly_count": len(provisional),
        "status_counts": dict(counts),
        "physical_window_count": None,
        "verified_window_openings": 0,
        "wall_cutouts_generated": False,
        "construction_progress_verified": False,
        "vertical_dimensions_verified": False,
        "method": (
            "PCA_DOMINANT_AXIS_OF_CONNECTED_WINDOW_LAYER_GEOMETRY"
        ),
        "limitations": [
            "Connectivity groups are not verified physical windows.",
            "PCA spans are geometric approximations.",
            "Wall association has not been verified.",
            "No sill height or glazing height is inferred.",
            "Curved facade systems are handled separately.",
            "No wall cutouts are authorized.",
        ],
        "assemblies": records,
    }

    with OUTPUT_JSON.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            output,
            file,
            indent=2,
            allow_nan=False,
        )

    plot_results(records)

    print("\n=== CAD WINDOW ASSEMBLIES ===")
    print("Connected groups:", len(records))
    print("Status counts:", dict(counts))
    print("Provisional straight assemblies:", len(provisional))
    print("Verified window openings: 0")
    print("Wall cutouts generated: NO")
    print()
    print("JSON:", OUTPUT_JSON)
    print("Preview:", OUTPUT_PNG)


if __name__ == "__main__":
    main()
