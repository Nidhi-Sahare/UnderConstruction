
"""
BuildProgress - Wall Geometry Extractor

Extracts candidate wall regions from DXF wall layers.

Approach:
    1. Read wall-layer CAD geometry.
    2. Flatten supported curves.
    3. Build a planar line network.
    4. Polygonize closed boundaries.
    5. Preserve source provenance and uncertainty.
    6. Export JSON and a diagnostic PNG.

Important:
    Closed regions are geometric candidates, NOT
    independently verified physical wall elements.

Run:
    python wall_extractor.py

Dependencies:
    pip install ezdxf shapely matplotlib
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import ezdxf
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ezdxf.path import make_path
from matplotlib.patches import Polygon as PlotPolygon
from shapely.geometry import LineString
from shapely.ops import polygonize_full, unary_union
from shapely.strtree import STRtree


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

OUTPUT_DIR = ROOT / "data" / "geometry"

JSON_PATH = OUTPUT_DIR / "wall_candidates.json"
IMAGE_PATH = OUTPUT_DIR / "wall_candidates.png"

# These identify semantic CAD layers, not geometric coordinates.
WALL_LAYERS = {
    "010 AussenwÃ¤nde": "exterior",
    "015 InnenwÃ¤nde": "interior",
    "016 TrennwÃ¤nde": "partition",
    "019 Vormauerungen": "lining",
}

COLORS = {
    "exterior": "#2563EB",
    "interior": "#E59A37",
    "partition": "#D65D8A",
    "lining": "#A16207",
}

# Coordinates are in meters, as confirmed by DXF $INSUNITS.
# No endpoint snapping or gap closing is performed.
CURVE_TOLERANCE_M = 0.01
MIN_AREA_M2 = 0.0001
SOURCE_TOLERANCE_M = 0.000001


def coordinates(point):
    return float(point[0]), float(point[1])


def flatten_entity(entity):
    """
    Return supported entity geometry as line strings.

    Each returned line retains the parent entity's
    layer and handle through the caller.
    """

    kind = entity.dxftype()

    if kind == "LINE":
        points = [
            coordinates(entity.dxf.start),
            coordinates(entity.dxf.end),
        ]

        return [points]

    if kind in ("LWPOLYLINE", "POLYLINE"):
        result = []

        for child in entity.virtual_entities():
            result.extend(flatten_entity(child))

        return result

    if kind in ("ARC", "CIRCLE"):
        path = make_path(entity)

        points = [
            coordinates(point)
            for point in path.flattening(
                distance=CURVE_TOLERANCE_M
            )
        ]

        return [points] if len(points) >= 2 else []

    return []


def extract_source_lines(document):
    """
    Extract linework separately for each wall category.

    INSERT entities are intentionally not expanded here:
    the current dataset contains very few, and silently
    mixing transformed block geometry could compromise
    provenance. Unsupported types are reported.
    """

    lines_by_category = defaultdict(list)
    sources_by_category = defaultdict(list)
    unsupported = Counter()

    for entity in document.modelspace():
        layer = entity.dxf.get("layer", "0")

        if layer not in WALL_LAYERS:
            continue

        category = WALL_LAYERS[layer]
        handle = str(entity.dxf.get("handle", ""))

        if entity.dxftype() not in (
            "LINE",
            "LWPOLYLINE",
            "POLYLINE",
            "ARC",
            "CIRCLE",
        ):
            unsupported[entity.dxftype()] += 1
            continue

        for points in flatten_entity(entity):
            if len(points) < 2:
                continue

            line = LineString(points)

            if line.is_empty or line.length <= 0:
                continue

            lines_by_category[category].append(line)

            sources_by_category[category].append({
                "handle": handle,
                "layer": layer,
                "entity_type": entity.dxftype(),
            })

    return lines_by_category, sources_by_category, unsupported


def polygonize_category(lines, sources, category):
    """
    Build candidate polygons from a line network.

    unary_union nodes intersections before polygonization.
    polygonize_full also reports unresolved linework.
    """

    if not lines:
        return [], {
            "input_lines": 0,
            "dangles": 0,
            "cut_edges": 0,
            "invalid_rings": 0,
        }

    network = unary_union(lines)

    polygons, cuts, dangles, invalid = polygonize_full(
        network
    )

    source_tree = STRtree(lines)

    candidates = []

    for polygon in polygons.geoms:
        if polygon.is_empty:
            continue

        if not polygon.is_valid:
            continue

        if polygon.area < MIN_AREA_M2:
            continue

        boundary = polygon.boundary

        matching_sources = {}

        nearby_indices = source_tree.query(
            boundary.buffer(SOURCE_TOLERANCE_M)
        )

        for index in nearby_indices:
            index = int(index)

            source_line = lines[index]

            shared_length = source_line.intersection(
                boundary.buffer(SOURCE_TOLERANCE_M)
            ).length

            if shared_length <= SOURCE_TOLERANCE_M:
                continue

            source = sources[index]
            matching_sources[
                (source["layer"], source["handle"])
            ] = source

        candidates.append({
            "category": category,
            "area_m2": round(float(polygon.area), 6),
            "perimeter_m": round(float(polygon.length), 6),
            "bounds": [
                round(float(v), 6)
                for v in polygon.bounds
            ],
            "exterior": [
                [round(float(x), 6), round(float(y), 6)]
                for x, y in polygon.exterior.coords
            ],
            "holes": [
                [
                    [round(float(x), 6), round(float(y), 6)]
                    for x, y in ring.coords
                ]
                for ring in polygon.interiors
            ],
            "sources": sorted(
                matching_sources.values(),
                key=lambda item: (
                    item["layer"],
                    item["handle"],
                ),
            ),
            "status": "candidate_unverified",
        })

    statistics = {
        "input_lines": len(lines),
        "dangles": len(dangles.geoms),
        "cut_edges": len(cuts.geoms),
        "invalid_rings": len(invalid.geoms),
    }

    return candidates, statistics


def create_visualization(candidates):
    fig, ax = plt.subplots(
        figsize=(13, 10),
        dpi=150,
    )

    for candidate in candidates:
        color = COLORS[candidate["category"]]

        patch = PlotPolygon(
            candidate["exterior"],
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.40,
            linewidth=0.7,
        )

        ax.add_patch(patch)

        # Holes are intentionally left unfilled.
        # Their outlines are drawn for diagnostics.
        for hole in candidate["holes"]:
            xs = [p[0] for p in hole]
            ys = [p[1] for p in hole]

            ax.plot(
                xs,
                ys,
                color="black",
                linewidth=0.5,
                linestyle="--",
            )

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    ax.set_title(
        "BuildProgress | Floor 2 - "
        "Unverified Wall Region Candidates"
    )

    ax.grid(alpha=0.12)

    fig.savefig(
        IMAGE_PATH,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DXF_PATH.exists():
        raise FileNotFoundError(
            f"Missing DXF file: {DXF_PATH}"
        )

    print("Reading architectural drawing...")

    document = ezdxf.readfile(DXF_PATH)

    unit_code = document.header.get("$INSUNITS", 0)

    if unit_code != 6:
        raise ValueError(
            "Expected DXF units in meters (code 6). "
            f"Found unit code {unit_code}."
        )

    lines, sources, unsupported = extract_source_lines(
        document
    )

    all_candidates = []
    category_stats = {}

    print("\nEXTRACTION RESULTS")
    print("-" * 65)

    for category in WALL_LAYERS.values():
        candidates, statistics = polygonize_category(
            lines[category],
            sources[category],
            category,
        )

        category_stats[category] = {
            **statistics,
            "candidate_count": len(candidates),
        }

        all_candidates.extend(candidates)

        print(
            f"{category:12} "
            f"lines={statistics['input_lines']:5} "
            f"regions={len(candidates):5} "
            f"dangles={statistics['dangles']:5} "
            f"cuts={statistics['cut_edges']:5}"
        )

    for index, candidate in enumerate(
        all_candidates,
        start=1,
    ):
        candidate["id"] = f"WALL_CANDIDATE_{index:05d}"

    report = {
        "schema_version": "wall-candidates-1.0",
        "source_dxf": str(DXF_PATH),
        "coordinate_units": "meters",
        "method": "layerwise_polygonization",
        "curve_tolerance_m": CURVE_TOLERANCE_M,
        "minimum_area_m2": MIN_AREA_M2,
        "source_tolerance_m": SOURCE_TOLERANCE_M,
        "category_statistics": category_stats,
        "unsupported_entity_types": dict(unsupported),
        "candidates": all_candidates,
    }

    JSON_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    create_visualization(all_candidates)

    print("\nTOTAL CANDIDATES:", len(all_candidates))

    print("\nSaved:")
    print(JSON_PATH)
    print(IMAGE_PATH)

    print(
        "\nImportant: These are geometric candidates, "
        "not verified physical wall elements."
    )


if __name__ == "__main__":
    main()

