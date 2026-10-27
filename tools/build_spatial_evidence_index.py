
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ezdxf
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

ARCHIVE_DATA = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

MANIFEST_PATH = (
    ARCHIVE_DATA
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frame_manifest.csv"
)

FRAMES_DIR = MANIFEST_PATH.parent / "frames"

CANDIDATES_PATH = (
    ROOT
    / "outputs"
    / "automatic_registration_v2"
    / "registration_candidates.json"
)

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

OUTPUT_DIR = ROOT / "outputs" / "spatial_evidence"
OUTPUT_PATH = OUTPUT_DIR / "spatial_evidence_index.json"

NEARBY_RADIUS_M = 2.5
MAX_ELEMENTS_PER_CANDIDATE = 12
MIN_CONSENSUS_CANDIDATES = 4

# These are CAD layer classifications, not claims that
# physical construction has been verified.
LAYERS = {
    "015 Innenwände": "INTERIOR_WALL",
    "010 Aussenwände": "EXTERIOR_WALL",
    "016 Trennwände": "PARTITION_WALL",
    "010 Türen": "DOOR_SOURCE_GEOMETRY",
    "010 Fenster": "WINDOW_SOURCE_GEOMETRY",
    "012 Vorhangfassade": "CURTAIN_WALL_SOURCE_GEOMETRY",
}

# A CAD entity may contain multiple line segments.
# Grouping by source entity prevents those segments from
# being counted as independent construction elements.


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def load_frames():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    frames = []

    with MANIFEST_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row_index, row in enumerate(reader):
            matched = str(
                row.get("pose_matched", "")
            ).strip().lower()

            if matched not in ("true", "1", "yes"):
                continue

            x = finite(row.get("tx"))
            y = finite(row.get("ty"))
            z = finite(row.get("tz"))

            if x is None or y is None:
                continue

            filename = Path(
                str(row.get("filename", "")).replace("\\", "/")
            ).name

            if not filename:
                filename = f"cam0_{row_index:05d}.jpg"

            frame_path = FRAMES_DIR / filename

            if not frame_path.is_file():
                continue

            frames.append({
                "frame_index": row_index,
                "filename": filename,
                "image_path": str(
                    frame_path.relative_to(ROOT)
                ).replace("\\", "/"),
                "timestamp_s": finite(
                    row.get("header_timestamp_s")
                ),
                "trajectory_position": {
                    "x": x,
                    "y": y,
                    "z": z,
                },
            })

    if not frames:
        raise RuntimeError(
            "No matched frames with available images."
        )

    return frames


def load_candidates():
    payload = json.loads(
        CANDIDATES_PATH.read_text(encoding="utf-8")
    )

    raw = (
        payload["candidates"]
        if isinstance(payload, dict)
        else payload
    )

    candidates = []

    for index, item in enumerate(raw):
        angle = finite(item.get("angle_deg"))
        tx = finite(item.get("tx"))
        ty = finite(item.get("ty"))

        if None in (angle, tx, ty):
            continue

        candidates.append({
            "rank": int(item.get("rank", index + 1)),
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
            "original_score": finite(item.get("score")),
        })

    if not candidates:
        raise RuntimeError(
            "No valid registration candidates."
        )

    return candidates


def transform_point(position, candidate):
    angle = math.radians(candidate["angle_deg"])

    c = math.cos(angle)
    s = math.sin(angle)

    x = position["x"]
    y = position["y"]

    return (
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    )


def entity_segments(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        a = entity.dxf.start
        b = entity.dxf.end

        return [
            ((float(a.x), float(a.y)),
             (float(b.x), float(b.y)))
        ]

    if kind == "LWPOLYLINE":
        points = [
            (float(p[0]), float(p[1]))
            for p in entity.get_points("xy")
        ]

        segments = list(zip(points, points[1:]))

        if entity.closed and len(points) > 2:
            segments.append((points[-1], points[0]))

        return segments

    if kind == "ARC":
        center = entity.dxf.center
        radius = float(entity.dxf.radius)

        start = math.radians(
            float(entity.dxf.start_angle)
        )
        end = math.radians(
            float(entity.dxf.end_angle)
        )

        while end <= start:
            end += 2 * math.pi

        angles = np.linspace(start, end, 24)

        points = [
            (
                float(center.x) + radius * math.cos(a),
                float(center.y) + radius * math.sin(a),
            )
            for a in angles
        ]

        return list(zip(points, points[1:]))

    return []


def load_cad_elements():
    document = ezdxf.readfile(DXF_PATH)
    msp = document.modelspace()

    elements = []
    geometries = []

    for entity in msp:
        layer = entity.dxf.layer

        if layer not in LAYERS:
            continue

        segments = entity_segments(entity)

        valid = []

        for a, b in segments:
            if math.dist(a, b) < 0.03:
                continue

            valid.append(LineString([a, b]))

        if not valid:
            continue

        # Each source entity is treated as a provisional
        # CAD geometry group, not a verified physical element.
        handle = str(entity.dxf.handle)

        element_id = f"CAD_{handle}"

        for geometry in valid:
            elements.append({
                "element_id": element_id,
                "source_handle": handle,
                "layer": layer,
                "category": LAYERS[layer],
                "source_type": entity.dxftype(),
            })
            geometries.append(geometry)

    if not geometries:
        raise RuntimeError(
            "No usable CAD geometry found."
        )

    return elements, geometries


def query_nearby(
    point_xy,
    tree,
    elements,
    geometries,
):
    point = Point(point_xy)

    indices = tree.query(
        point.buffer(NEARBY_RADIUS_M)
    )

    grouped = {}

    for index in indices:
        index = int(index)

        geometry = geometries[index]
        distance = float(point.distance(geometry))

        if distance > NEARBY_RADIUS_M:
            continue

        element = elements[index]
        element_id = element["element_id"]

        previous = grouped.get(element_id)

        if (
            previous is None
            or distance < previous["distance_m"]
        ):
            grouped[element_id] = {
                **element,
                "distance_m": round(distance, 4),
            }

    return sorted(
        grouped.values(),
        key=lambda item: (
            item["distance_m"],
            item["element_id"],
        ),
    )[:MAX_ELEMENTS_PER_CANDIDATE]


def analyze_frame(
    frame,
    candidates,
    tree,
    elements,
    geometries,
):
    candidate_results = []
    candidate_sets = []

    positions = []

    for candidate in candidates:
        xy = transform_point(
            frame["trajectory_position"],
            candidate,
        )

        positions.append(xy)

        nearby = query_nearby(
            xy,
            tree,
            elements,
            geometries,
        )

        candidate_sets.append({
            item["element_id"]
            for item in nearby
        })

        candidate_results.append({
            "candidate_rank": candidate["rank"],
            "cad_position": {
                "x": round(xy[0], 5),
                "y": round(xy[1], 5),
            },
            "nearby_elements": nearby,
        })

    counts = defaultdict(int)
    details = {}

    for candidate_result in candidate_results:
        for item in candidate_result["nearby_elements"]:
            element_id = item["element_id"]

            counts[element_id] += 1

            if (
                element_id not in details
                or item["distance_m"]
                < details[element_id]["distance_m"]
            ):
                details[element_id] = item

    consensus_elements = []

    for element_id, count in counts.items():
        if count < MIN_CONSENSUS_CANDIDATES:
            continue

        consensus_elements.append({
            **details[element_id],
            "supporting_candidates": count,
            "candidate_count": len(candidates),
            "association_type": "SPATIAL_CONSENSUS",
            "visibility_verified": False,
            "construction_progress": "UNKNOWN",
        })

    consensus_elements.sort(
        key=lambda item: (
            -item["supporting_candidates"],
            item["distance_m"],
        )
    )

    separations = [
        math.dist(positions[i], positions[j])
        for i in range(len(positions))
        for j in range(i + 1, len(positions))
    ]

    max_separation = max(separations, default=0.0)

    union = set().union(*candidate_sets)
    intersection = (
        set.intersection(*candidate_sets)
        if candidate_sets
        else set()
    )

    if consensus_elements:
        classification = "SPATIAL_CONSENSUS"
    elif union:
        classification = "CANDIDATE_DEPENDENT"
    else:
        classification = "UNRESOLVED"

    return {
        "frame_index": frame["frame_index"],
        "filename": frame["filename"],
        "image_path": frame["image_path"],
        "timestamp_s": frame["timestamp_s"],
        "classification": classification,
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress": "UNKNOWN",
        "candidate_max_separation_m": round(
            max_separation, 4
        ),
        "all_candidate_element_intersection_count": len(
            intersection
        ),
        "candidate_results": candidate_results,
        "consensus_elements": consensus_elements,
    }


def build_reverse_index(frame_records):
    reverse = {}

    for frame in frame_records:
        frame_index = frame["frame_index"]

        for item in frame["consensus_elements"]:
            element_id = item["element_id"]

            reverse.setdefault(element_id, {
                "element_id": element_id,
                "category": item["category"],
                "source_layer": item["layer"],
                "registration_verified": False,
                "visibility_verified": False,
                "construction_progress": "UNKNOWN",
                "frames": [],
            })

            reverse[element_id]["frames"].append({
                "frame_index": frame_index,
                "filename": frame["filename"],
                "candidate_support": item[
                    "supporting_candidates"
                ],
                "distance_m": item["distance_m"],
            })

    return reverse


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading matched camera frames...")
    frames = load_frames()

    print("Loading registration candidates...")
    candidates = load_candidates()

    print("Loading CAD geometry...")
    elements, geometries = load_cad_elements()

    print("Building spatial search index...")
    tree = STRtree(geometries)

    print("Associating frames with nearby CAD geometry...")

    frame_records = [
        analyze_frame(
            frame,
            candidates,
            tree,
            elements,
            geometries,
        )
        for frame in frames
    ]

    reverse_index = build_reverse_index(frame_records)

    counts = defaultdict(int)

    for frame in frame_records:
        counts[frame["classification"]] += 1

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sequence": SEQUENCE,
        "method": "MULTI_CANDIDATE_SPATIAL_PROXIMITY",
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "parameters": {
            "nearby_radius_m": NEARBY_RADIUS_M,
            "minimum_consensus_candidates":
                MIN_CONSENSUS_CANDIDATES,
            "max_elements_per_candidate":
                MAX_ELEMENTS_PER_CANDIDATE,
        },
        "summary": {
            "matched_frames": len(frames),
            "registration_candidates": len(candidates),
            "cad_geometry_segments": len(geometries),
            "indexed_cad_source_entities": len({
                item["element_id"]
                for item in elements
            }),
            "classification_counts": dict(counts),
            "consensus_source_entities": len(
                reverse_index
            ),
        },
        "limitations": [
            "Registration candidates are not independently verified.",
            "Spatial proximity does not prove image visibility.",
            "Camera orientation is not used in this version.",
            "Fisheye projection is not used in this version.",
            "CAD source entities are not necessarily physical elements.",
            "Curved polyline bulges are approximated as straight chords.",
            "Nearby elements can be occluded by walls or other objects.",
            "No construction progress is inferred.",
        ],
        "candidates": candidates,
        "frames": frame_records,
        "elements": reverse_index,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== SPATIAL EVIDENCE INDEX ===")
    print("Frames:", len(frames))
    print("Candidates:", len(candidates))
    print("CAD segments:", len(geometries))
    print(
        "CAD source entities:",
        report["summary"]["indexed_cad_source_entities"],
    )

    for classification in (
        "SPATIAL_CONSENSUS",
        "CANDIDATE_DEPENDENT",
        "UNRESOLVED",
    ):
        print(
            f"{classification}: "
            f"{counts[classification]}"
        )

    print(
        "Consensus source entities:",
        len(reverse_index),
    )
    print("Registration verified: NO")
    print("Visibility verified: NO")
    print("Construction progress verified: NO")
    print(
        "Output:",
        OUTPUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
