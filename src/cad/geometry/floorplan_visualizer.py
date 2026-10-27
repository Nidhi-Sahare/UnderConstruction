
"""
BuildProgress - Floor Plan Visualizer

Reads an existing DXF and creates:
    1. Full architectural floor plan
    2. Walls-only floor plan
    3. Geometry statistics JSON

No video downloads or neural networks required.

Run:
    python floorplan_visualizer.py
"""

import json
from collections import Counter
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"
OUTPUT_DIR = ROOT / "data" / "floorplans" / "visualizations"

# Display configuration only. These are not extraction rules.
LAYER_STYLES = {
    "010 AussenwÃ¤nde": ("#2563EB", 1.8),
    "015 InnenwÃ¤nde": ("#E59A37", 1.4),
    "016 TrennwÃ¤nde": ("#D65D8A", 1.2),
    "010 TÃ¼ren": ("#299C73", 0.8),
    "010 Fenster": ("#9270C9", 0.7),
    "011 Fassadensystem": ("#64748B", 0.65),
    "012 Vorhangfassade": ("#0D9488", 0.9),
    "019 Vormauerungen": ("#A16207", 0.9),
    "020 StÃ¼tzen": ("#DC2626", 1.5),
}

WALL_LAYERS = {
    "010 AussenwÃ¤nde",
    "015 InnenwÃ¤nde",
    "016 TrennwÃ¤nde",
    "019 Vormauerungen",
}

DEFAULT_STYLE = ("#64748B", 0.7)
CURVE_SEGMENTS = 24


def xy(point):
    return float(point[0]), float(point[1])


def entity_segments(entity):
    """
    Convert supported DXF geometry into 2D line segments.

    Curved entities are approximated by short segments
    for visualization only.

    Unsupported entities are skipped and counted.
    """

    kind = entity.dxftype()

    if kind == "LINE":
        return [[xy(entity.dxf.start), xy(entity.dxf.end)]]

    if kind == "LWPOLYLINE":
        # virtual_entities resolves bulged polyline segments
        # into LINE and ARC primitives.
        result = []

        for child in entity.virtual_entities():
            result.extend(entity_segments(child))

        return result

    if kind == "POLYLINE":
        result = []

        for child in entity.virtual_entities():
            result.extend(entity_segments(child))

        return result

    if kind == "ARC":
        from ezdxf.path import make_path

        path = make_path(entity)
        vertices = [
            xy(point)
            for point in path.flattening(distance=0.03)
        ]

        return [
            [a, b]
            for a, b in zip(vertices, vertices[1:])
        ]

    if kind == "CIRCLE":
        from math import cos, sin, pi

        center = xy(entity.dxf.center)
        radius = float(entity.dxf.radius)

        vertices = [
            (
                center[0] + radius * cos(2 * pi * i / CURVE_SEGMENTS),
                center[1] + radius * sin(2 * pi * i / CURVE_SEGMENTS),
            )
            for i in range(CURVE_SEGMENTS + 1)
        ]

        return [
            [a, b]
            for a, b in zip(vertices, vertices[1:])
        ]

    if kind == "INSERT":
        # Expand block references for display.
        result = []

        for child in entity.virtual_entities():
            result.extend(entity_segments(child))

        return result

    return []


def load_geometry():
    if not DXF_PATH.exists():
        raise FileNotFoundError(
            f"Missing DXF: {DXF_PATH}\n"
            "Run floorplan_inspector.py first."
        )

    document = ezdxf.readfile(DXF_PATH)

    segments_by_layer = {}
    entity_counts = Counter()
    unsupported = Counter()
    closed_polylines = Counter()
    open_polylines = Counter()

    for entity in document.modelspace():
        layer = entity.dxf.get("layer", "0")
        kind = entity.dxftype()

        entity_counts[layer] += 1

        if kind in ("LWPOLYLINE", "POLYLINE"):
            if entity.is_closed:
                closed_polylines[layer] += 1
            else:
                open_polylines[layer] += 1

        segments = entity_segments(entity)

        if not segments and kind not in (
            "MTEXT",
            "TEXT",
            "ATTRIB",
            "ATTDEF",
        ):
            unsupported[kind] += 1

        if segments:
            segments_by_layer.setdefault(layer, []).extend(
                segments
            )

    statistics = {
        "dxf": str(DXF_PATH),
        "entity_counts_by_layer": dict(entity_counts),
        "closed_polylines_by_layer": dict(closed_polylines),
        "open_polylines_by_layer": dict(open_polylines),
        "unsupported_entity_types": dict(unsupported),
        "segments_by_layer": {
            layer: len(segments)
            for layer, segments in segments_by_layer.items()
        },
    }

    return segments_by_layer, statistics


def draw_plan(segments_by_layer, output_path, walls_only=False):
    fig, ax = plt.subplots(figsize=(13, 10), dpi=160)

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    handles = []

    for layer, segments in segments_by_layer.items():
        if walls_only and layer not in WALL_LAYERS:
            continue

        if not segments:
            continue

        color, width = LAYER_STYLES.get(
            layer,
            DEFAULT_STYLE,
        )

        collection = LineCollection(
            segments,
            colors=color,
            linewidths=width,
            alpha=0.9,
        )

        ax.add_collection(collection)

        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=layer,
            )
        )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    title = (
        "BuildProgress | Floor 2 - Wall Geometry"
        if walls_only
        else "BuildProgress | Floor 2 - Architectural Plan"
    )

    ax.set_title(title, fontsize=15, pad=18)

    ax.grid(
        True,
        alpha=0.12,
        linewidth=0.5,
    )

    if handles:
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=8,
            frameon=False,
        )

    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading Floor 2 geometry...")

    geometry, statistics = load_geometry()

    print("\nGenerating full floor plan...")

    draw_plan(
        geometry,
        OUTPUT_DIR / "floor_2_full.png",
    )

    print("\nGenerating walls-only floor plan...")

    draw_plan(
        geometry,
        OUTPUT_DIR / "floor_2_walls.png",
        walls_only=True,
    )

    report_path = OUTPUT_DIR / "geometry_statistics.json"

    report_path.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8",
    )

    print("\nPOLYLINE STATISTICS")
    print("-" * 55)

    layers = sorted(
        set(statistics["closed_polylines_by_layer"])
        | set(statistics["open_polylines_by_layer"])
    )

    for layer in layers:
        closed = statistics[
            "closed_polylines_by_layer"
        ].get(layer, 0)

        opened = statistics[
            "open_polylines_by_layer"
        ].get(layer, 0)

        print(
            f"{layer}: "
            f"{closed} closed, {opened} open"
        )

    print("\nGeometry inspection complete.")
    print(f"Statistics: {report_path}")


if __name__ == "__main__":
    main()

