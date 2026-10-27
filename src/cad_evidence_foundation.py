
"""
Under-construction
CAD Evidence Foundation

Converts genuine Hilti DXF floor plans into metric wall maps
for subsequent camera localization and evidence ray casting.

Outputs:
    data/evidence_foundation/floor_1_walls.png
    data/evidence_foundation/floor_2_walls.png
    data/evidence_foundation/floor_3_walls.png
    data/evidence_foundation/metadata.json

Also inventories available reference floor-plan images,
camera trajectories and calibration files.

No registration, wall visibility, or construction progress
is claimed by this module.

Run:
    python src/cad_evidence_foundation.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import cv2
import ezdxf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLOORPLANS = DATA / "floorplans"
OUTPUT = DATA / "evidence_foundation"

FLOORS = (1, 2, 3)

PIXELS_PER_METRE = 100
MARGIN_M = 1.0

WALL_LAYERS = {
    "010 Aussenwände": "exterior",
    "015 Innenwände": "interior",
    "016 Trennwände": "partition",
    "019 Vormauerungen": "lining",
}

REFERENCE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".csv", ".txt", ".json",
    ".yaml", ".yml", ".tum",
    ".bag", ".db3", ".mcap",
}

REFERENCE_KEYWORDS = (
    "trajectory",
    "groundtruth",
    "ground_truth",
    "gt_pose",
    "poses",
    "calibration",
    "intrinsics",
    "extrinsics",
    "floorplan",
    "floor_plan",
    "reference",
)


def entity_segments(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        a = entity.dxf.start
        b = entity.dxf.end

        return [
            (
                (float(a.x), float(a.y)),
                (float(b.x), float(b.y)),
            )
        ]

    if kind == "LWPOLYLINE":
        vertices = list(entity.get_points("xyb"))

        # Arc segments require a different geometric treatment.
        # Never silently approximate them as straight lines.
        if any(abs(float(v[2])) > 1e-9 for v in vertices):
            return []

        points = [
            (float(v[0]), float(v[1]))
            for v in vertices
        ]

        if entity.closed and len(points) > 2:
            points.append(points[0])

        return list(zip(points, points[1:]))

    if kind == "POLYLINE" and entity.is_2d_polyline:
        vertices = list(entity.vertices)

        if any(
            abs(float(v.dxf.get("bulge", 0))) > 1e-9
            for v in vertices
        ):
            return []

        points = [
            (
                float(v.dxf.location.x),
                float(v.dxf.location.y),
            )
            for v in vertices
        ]

        if entity.is_closed and len(points) > 2:
            points.append(points[0])

        return list(zip(points, points[1:]))

    return []


def valid_segment(a, b):
    values = (*a, *b)

    if not all(math.isfinite(v) for v in values):
        return False

    length = math.dist(a, b)

    return 0.005 <= length <= 200.0


def read_floor(number):
    path = FLOORPLANS / f"floor_{number}.dxf"

    if not path.is_file():
        raise FileNotFoundError(path)

    drawing = ezdxf.readfile(str(path))

    units = int(drawing.header.get("$INSUNITS", 0))

    if units != 6:
        raise ValueError(
            f"{path.name}: expected metre units (6), got {units}"
        )

    segments = []
    layer_counts = Counter()
    skipped = Counter()

    for entity in drawing.modelspace():
        layer = str(entity.dxf.get("layer", ""))

        if layer not in WALL_LAYERS:
            continue

        extracted = entity_segments(entity)

        if not extracted:
            skipped[entity.dxftype()] += 1
            continue

        for index, (a, b) in enumerate(extracted):
            if not valid_segment(a, b):
                skipped["invalid_segment"] += 1
                continue

            segments.append({
                "id": (
                    f"F{number}_"
                    f"{entity.dxf.get('handle', 'unknown')}_"
                    f"{index}"
                ),
                "layer": layer,
                "category": WALL_LAYERS[layer],
                "start": list(a),
                "end": list(b),
            })

            layer_counts[layer] += 1

    if not segments:
        raise RuntimeError(
            f"No supported wall segments in {path.name}"
        )

    return {
        "number": number,
        "source": str(path.relative_to(ROOT)),
        "units": "metres",
        "segments": segments,
        "layer_counts": dict(layer_counts),
        "skipped": dict(skipped),
    }


def common_bounds(floors):
    xs = []
    ys = []

    for floor in floors:
        for segment in floor["segments"]:
            for point in (
                segment["start"],
                segment["end"],
            ):
                xs.append(point[0])
                ys.append(point[1])

    return {
        "xmin": min(xs) - MARGIN_M,
        "xmax": max(xs) + MARGIN_M,
        "ymin": min(ys) - MARGIN_M,
        "ymax": max(ys) + MARGIN_M,
    }


def image_dimensions(bounds):
    width = int(
        math.ceil(
            (bounds["xmax"] - bounds["xmin"])
            * PIXELS_PER_METRE
        )
    ) + 1

    height = int(
        math.ceil(
            (bounds["ymax"] - bounds["ymin"])
            * PIXELS_PER_METRE
        )
    ) + 1

    if width <= 0 or height <= 0:
        raise ValueError("Invalid raster dimensions")

    if width * height > 100_000_000:
        raise ValueError(
            "Raster would be too large for safe processing"
        )

    return width, height


def world_to_pixel(x, y, bounds):
    px = int(round(
        (x - bounds["xmin"]) * PIXELS_PER_METRE
    ))

    py = int(round(
        (bounds["ymax"] - y) * PIXELS_PER_METRE
    ))

    return px, py


def rasterize_floor(floor, bounds):
    width, height = image_dimensions(bounds)

    # White means a CAD wall segment.
    # Black means no wall segment was rasterized.
    # Black must NOT be interpreted as free physical space.
    wall_map = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    category_map = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    category_ids = {
        "exterior": 1,
        "interior": 2,
        "partition": 3,
        "lining": 4,
    }

    for segment in floor["segments"]:
        a = world_to_pixel(
            *segment["start"],
            bounds,
        )

        b = world_to_pixel(
            *segment["end"],
            bounds,
        )

        cv2.line(
            wall_map,
            a,
            b,
            255,
            thickness=2,
            lineType=cv2.LINE_8,
        )

        cv2.line(
            category_map,
            a,
            b,
            category_ids[segment["category"]],
            thickness=2,
            lineType=cv2.LINE_8,
        )

    return wall_map, category_map


def find_reference_candidates():
    candidates = []

    if not DATA.exists():
        return candidates

    for path in DATA.rglob("*"):
        if not path.is_file():
            continue

        if OUTPUT in path.parents:
            continue

        if path.suffix.lower() not in REFERENCE_EXTENSIONS:
            continue

        relative = path.relative_to(DATA)
        relative_lower = str(relative).casefold()

        if not any(
            keyword in relative_lower
            for keyword in REFERENCE_KEYWORDS
        ):
            continue

        # Avoid listing DXF source files as reference imagery.
        if path.suffix.lower() == ".dxf":
            continue

        candidates.append({
            "path": str(relative),
            "bytes": path.stat().st_size,
        })

    return sorted(
        candidates,
        key=lambda item: item["path"],
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("UNDER-CONSTRUCTION / CAD EVIDENCE FOUNDATION")
    print("=" * 62)

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Raster resolution: {PIXELS_PER_METRE} px/metre")

    floors = [
        read_floor(number)
        for number in FLOORS
    ]

    bounds = common_bounds(floors)
    width, height = image_dimensions(bounds)

    print()
    print("SHARED CAD COORDINATE FRAME")
    print(
        f"  X: {bounds['xmin']:.3f} "
        f"to {bounds['xmax']:.3f} m"
    )
    print(
        f"  Y: {bounds['ymin']:.3f} "
        f"to {bounds['ymax']:.3f} m"
    )
    print(f"  Raster: {width} x {height} pixels")

    floor_metadata = []

    for floor in floors:
        number = floor["number"]

        wall_map, category_map = rasterize_floor(
            floor,
            bounds,
        )

        wall_path = OUTPUT / f"floor_{number}_walls.png"
        category_path = (
            OUTPUT / f"floor_{number}_categories.png"
        )
        segments_path = (
            OUTPUT / f"floor_{number}_segments.json"
        )

        if not cv2.imwrite(str(wall_path), wall_map):
            raise RuntimeError(
                f"Could not save {wall_path}"
            )

        if not cv2.imwrite(
            str(category_path),
            category_map,
        ):
            raise RuntimeError(
                f"Could not save {category_path}"
            )

        segments_path.write_text(
            json.dumps(
                floor["segments"],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        floor_metadata.append({
            "floor": number,
            "source": floor["source"],
            "segment_count": len(floor["segments"]),
            "layer_counts": floor["layer_counts"],
            "skipped_entities": floor["skipped"],
            "wall_map": wall_path.name,
            "category_map": category_path.name,
            "segments_file": segments_path.name,
        })

        print()
        print(f"FLOOR {number}")
        print(
            f"  CAD wall segments: "
            f"{len(floor['segments'])}"
        )
        print(
            f"  Skipped unsupported/invalid: "
            f"{sum(floor['skipped'].values())}"
        )
        print(f"  Wall raster: {wall_path.name}")

    references = find_reference_candidates()

    metadata = {
        "project": "Under-construction",
        "dataset": "Hilti–Trimble–Oxford",
        "opencv_version": cv2.__version__,
        "pixels_per_metre": PIXELS_PER_METRE,
        "bounds_m": bounds,
        "image_width": width,
        "image_height": height,
        "pixel_coordinate_convention": (
            "column=(x-xmin)*pixels_per_metre; "
            "row=(ymax-y)*pixels_per_metre"
        ),
        "shared_coordinate_frame": True,
        "external_reference_registration_verified": False,
        "camera_to_cad_registration_verified": False,
        "construction_progress_verified": False,
        "map_semantics": (
            "White pixels represent rasterized CAD wall "
            "segments, not observed physical construction. "
            "Black pixels are not proof of free space."
        ),
        "category_ids": {
            "exterior": 1,
            "interior": 2,
            "partition": 3,
            "lining": 4,
        },
        "floors": floor_metadata,
        "reference_candidates": references,
    }

    metadata_path = OUTPUT / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("REFERENCE DATA CANDIDATES")

    if references:
        for item in references[:25]:
            print(f"  {item['path']}")

        if len(references) > 25:
            print(
                f"  ... {len(references) - 25} more"
            )
    else:
        print(
            "  No matching reference images or pose files "
            "found under data/"
        )

    print()
    print("RESULT")
    print(f"  {metadata_path}")
    print("  Three CAD wall maps generated")
    print("  Three segment catalogs generated")
    print("  No camera registration has been assumed")
    print("=" * 62)


if __name__ == "__main__":
    main()
