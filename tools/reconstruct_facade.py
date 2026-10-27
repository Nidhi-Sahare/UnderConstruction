
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"

WALL_PATH = ROOT / "data/digital_twin/building_elements.json"

WINDOW_PATH = (
    ROOT
    / "outputs/cad_openings/windows/window_assemblies.json"
)

OUTPUT_DIR = ROOT / "outputs/cad_facade"

REPORT_PATH = OUTPUT_DIR / "facade_geometry.json"
PREVIEW_PATH = OUTPUT_DIR / "facade_alignment.png"


FACADE_LAYERS = {
    "011 Fassadensystem": "FACADE_SYSTEM",
    "012 Vorhangfassade": "CURTAIN_WALL",
}

# Maximum distance between tessellated points along an arc.
# This is a visualization resolution, not a CAD tolerance.
MAX_ARC_STEP_M = 0.005

# Avoid extremely large output for tiny rounded CAD profiles.
MAX_ARC_SEGMENTS = 128


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def point2(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None

    x = finite(value[0])
    y = finite(value[1])

    if x is None or y is None:
        return None

    return [x, y]


def arc_points(start, end, bulge):
    """
    Tessellate a DXF bulge segment.

    bulge = tan(included_angle / 4)

    Positive bulge means counterclockwise traversal.
    Negative bulge means clockwise traversal.

    Returns points INCLUDING start and end.
    """
    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1

    chord = math.hypot(dx, dy)

    if chord <= 1e-12:
        return [start, end]

    if abs(bulge) <= 1e-12:
        return [start, end]

    angle = 4.0 * math.atan(bulge)

    radius = chord * (1.0 + bulge * bulge) / (
        4.0 * abs(bulge)
    )

    midpoint_x = (x1 + x2) / 2.0
    midpoint_y = (y1 + y2) / 2.0

    # Signed distance from chord midpoint to circle center.
    offset = chord * (1.0 - bulge * bulge) / (
        4.0 * bulge
    )

    normal_x = -dy / chord
    normal_y = dx / chord

    center_x = midpoint_x + normal_x * offset
    center_y = midpoint_y + normal_y * offset

    start_angle = math.atan2(
        y1 - center_y,
        x1 - center_x,
    )

    arc_length = abs(radius * angle)

    segments = max(
        2,
        min(
            MAX_ARC_SEGMENTS,
            math.ceil(arc_length / MAX_ARC_STEP_M),
        ),
    )

    points = []

    for index in range(segments + 1):
        fraction = index / segments
        current_angle = start_angle + fraction * angle

        points.append([
            center_x + radius * math.cos(current_angle),
            center_y + radius * math.sin(current_angle),
        ])

    # Keep exact original endpoints.
    points[0] = list(start)
    points[-1] = list(end)

    return points


def tessellate_polyline(entity):
    """
    Convert LWPOLYLINE to an ordered XY point path.

    The bulge on vertex i applies to segment i -> i+1.
    For closed polylines, the final bulge applies to the
    closing segment.
    """
    vertices = [
        (
            [float(x), float(y)],
            float(bulge),
        )
        for x, y, bulge in entity.get_points("xyb")
    ]

    if len(vertices) < 2:
        return []

    closed = bool(entity.closed)
    segment_count = (
        len(vertices)
        if closed
        else len(vertices) - 1
    )

    output = []

    for index in range(segment_count):
        start, bulge = vertices[index]
        end, _ = vertices[(index + 1) % len(vertices)]

        segment = arc_points(start, end, bulge)

        if output:
            output.extend(segment[1:])
        else:
            output.extend(segment)

    return output


def read_facade():
    doc = ezdxf.readfile(str(DXF_PATH))
    modelspace = doc.modelspace()

    paths = []
    stats = Counter()
    skipped = Counter()

    for entity in modelspace:
        layer = entity.dxf.layer

        if layer not in FACADE_LAYERS:
            continue

        kind = entity.dxftype()
        category = FACADE_LAYERS[layer]
        handle = str(entity.dxf.handle)

        stats[f"{category}_SOURCE_ENTITIES"] += 1
        stats[f"{category}_{kind}"] += 1

        try:
            if kind == "LINE":
                start = [
                    float(entity.dxf.start.x),
                    float(entity.dxf.start.y),
                ]
                end = [
                    float(entity.dxf.end.x),
                    float(entity.dxf.end.y),
                ]

                points = [start, end]
                closed = False
                has_bulge = False

            elif kind == "LWPOLYLINE":
                points = tessellate_polyline(entity)
                closed = bool(entity.closed)

                has_bulge = any(
                    abs(float(bulge)) > 1e-12
                    for _, _, bulge
                    in entity.get_points("xyb")
                )

            else:
                skipped[f"{layer}:{kind}"] += 1
                continue

            if len(points) < 2:
                skipped[f"{layer}:DEGENERATE"] += 1
                continue

            if not all(
                math.isfinite(value)
                for point in points
                for value in point
            ):
                skipped[f"{layer}:NONFINITE"] += 1
                continue

            paths.append({
                "id": f"facade_{handle}",
                "source_handle": handle,
                "source_layer": layer,
                "category": category,
                "entity_type": kind,
                "closed": closed,
                "has_bulge": has_bulge,
                "points": points,
                "construction_status": "UNKNOWN",
                "vertical_dimensions_verified": False,
                "physical_component_verified": False,
                "safe_to_extrude_as_wall": False,
            })

            stats[f"{category}_RECOVERED"] += 1

        except (
            ValueError,
            TypeError,
            ZeroDivisionError,
            OverflowError,
        ):
            skipped[f"{layer}:GEOMETRY_ERROR"] += 1

    return paths, dict(stats), dict(skipped)


def read_walls():
    if not WALL_PATH.exists():
        return []

    data = json.loads(WALL_PATH.read_text(encoding="utf-8"))

    items = (
        data
        if isinstance(data, list)
        else data.get("elements", [])
    )

    segments = []

    for item in items:
        start = point2(item.get("start"))
        end = point2(item.get("end"))

        if start is not None and end is not None:
            segments.append([start, end])

    return segments


def read_windows():
    if not WINDOW_PATH.exists():
        return []

    data = json.loads(WINDOW_PATH.read_text(encoding="utf-8"))

    segments = []

    for item in data.get("assemblies", []):
        if item.get("status") != "PROVISIONAL_STRAIGHT_ASSEMBLY":
            continue

        start = point2(item.get("start"))
        end = point2(item.get("end"))

        if start is not None and end is not None:
            segments.append([start, end])

    return segments


def path_segments(paths):
    segments = []

    for path in paths:
        points = path["points"]

        for index in range(len(points) - 1):
            segments.append([
                points[index],
                points[index + 1],
            ])

    return segments


def draw_collection(
    ax,
    segments,
    color,
    linewidth,
    alpha,
    zorder,
):
    if not segments:
        return

    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )

    ax.add_collection(collection)


def create_preview(paths, walls, windows):
    facade_paths = [
        item
        for item in paths
        if item["category"] == "FACADE_SYSTEM"
    ]

    curtain_paths = [
        item
        for item in paths
        if item["category"] == "CURTAIN_WALL"
    ]

    facade_segments = path_segments(facade_paths)
    curtain_segments = path_segments(curtain_paths)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(19, 9),
        constrained_layout=True,
    )

    # Panel 1: Existing wall reconstruction + windows.
    draw_collection(
        axes[0],
        walls,
        "#9ca3af",
        0.9,
        0.8,
        1,
    )

    draw_collection(
        axes[0],
        windows,
        "#06b6d4",
        1.6,
        0.95,
        3,
    )

    axes[0].set_title(
        "Current wall model + provisional windows"
    )

    # Panel 2: Original CAD facade + curtain-wall + windows.
    draw_collection(
        axes[1],
        walls,
        "#9ca3af",
        0.65,
        0.3,
        1,
    )

    draw_collection(
        axes[1],
        facade_segments,
        "#f59e0b",
        0.5,
        0.75,
        2,
    )

    draw_collection(
        axes[1],
        curtain_segments,
        "#a78bfa",
        0.9,
        0.85,
        3,
    )

    draw_collection(
        axes[1],
        windows,
        "#06b6d4",
        1.5,
        1.0,
        4,
    )

    axes[1].set_title(
        "Original facade + curtain wall + windows"
    )

    for ax in axes:
        ax.set_aspect("equal")
        ax.autoscale()
        ax.margins(0.04)
        ax.grid(alpha=0.15)
        ax.set_xlabel("CAD X (metres)")
        ax.set_ylabel("CAD Y (metres)")

    fig.suptitle(
        "Under-construction — CAD facade alignment diagnostic",
        fontsize=15,
    )

    fig.savefig(
        PREVIEW_PATH,
        dpi=180,
        facecolor="white",
    )

    plt.close(fig)


def main():
    if not DXF_PATH.exists():
        raise FileNotFoundError(DXF_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths, stats, skipped = read_facade()
    walls = read_walls()
    windows = read_windows()

    report = {
        "schema_version": "1.0",
        "source_dxf": str(DXF_PATH.relative_to(ROOT)),
        "coordinate_system": "DXF_METERS",
        "representation": "TESSELLATED_SOURCE_CAD_PATHS",
        "facade_path_count": len(paths),
        "source_stats": stats,
        "skipped": skipped,
        "wall_reference_count": len(walls),
        "provisional_window_reference_count": len(windows),
        "physical_facade_component_count": None,
        "facade_vertical_dimensions_verified": False,
        "window_to_facade_alignment_verified": False,
        "wall_cutouts_generated": False,
        "construction_progress_verified": False,
        "paths": paths,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    create_preview(paths, walls, windows)

    print("\n=== CAD FACADE RECONSTRUCTION ===")
    print("Recovered facade paths:", len(paths))
    print("Source statistics:", stats)
    print("Skipped:", skipped)
    print("Wall reference segments:", len(walls))
    print("Provisional window segments:", len(windows))
    print("Verified facade components: 0")
    print("Wall cutouts generated: NO")
    print("JSON:", REPORT_PATH.relative_to(ROOT))
    print("Preview:", PREVIEW_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
