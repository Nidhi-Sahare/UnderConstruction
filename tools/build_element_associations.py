
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

ELEMENTS_PATH = (
    ROOT / "data" / "digital_twin"
    / "building_elements.json"
)

CANDIDATES_PATH = (
    ROOT / "outputs" / "automatic_registration_v2"
    / "registration_candidates.json"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "element_associations"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "candidate_element_associations.json"
)

MAX_DISTANCE_M = 5.0
MAX_ELEMENTS_PER_FRAME = 5

EPSILON = 1e-12


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def finite_float(value):
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None

    return result if math.isfinite(result) else None


def xy_point(value):
    if isinstance(value, dict):
        x = finite_float(value.get("x"))
        y = finite_float(value.get("y"))

    elif isinstance(value, (list, tuple)) and len(value) >= 2:
        x = finite_float(value[0])
        y = finite_float(value[1])

    else:
        return None

    if x is None or y is None:
        return None

    return np.array([x, y], dtype=np.float64)


def find_manifest():
    matches = [
        path
        for path in ROOT.rglob("frame_manifest.csv")
        if SEQUENCE in str(path)
        and path.parent.name == "cam0"
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one cam0 manifest "
            f"for {SEQUENCE}; found {len(matches)}."
        )

    return matches[0]


def load_frames():
    manifest_path = find_manifest()

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    frame_dir = manifest_path.parent / "frames"

    records = []
    rejected = Counter()

    for row in rows:
        filename = row.get("filename", "")

        if str(row.get("pose_matched", "")).lower() != "true":
            rejected["unmatched_pose"] += 1
            continue

        x = finite_float(row.get("tx"))
        y = finite_float(row.get("ty"))
        z = finite_float(row.get("tz"))
        timestamp = finite_float(
            row.get("header_timestamp_s")
        )

        if None in (x, y, z):
            rejected["invalid_position"] += 1
            continue

        image_path = frame_dir / filename

        if not image_path.is_file():
            rejected["missing_image"] += 1
            continue

        records.append({
            "filename": filename,
            "timestamp_s": timestamp,
            "trajectory_xyz": [x, y, z],
            "image_path": str(
                image_path.relative_to(ROOT)
            ),
        })

    records.sort(key=lambda item: item["filename"])

    return records, dict(rejected)


def load_elements():
    data = load_json(ELEMENTS_PATH)

    if isinstance(data, dict):
        raw_elements = data.get("elements", [])
    elif isinstance(data, list):
        raw_elements = data
    else:
        raise ValueError(
            "Unsupported building-elements JSON structure."
        )

    if not isinstance(raw_elements, list):
        raise ValueError(
            "Expected 'elements' to be a list."
        )

    elements = []
    rejected = Counter()

    for index, item in enumerate(raw_elements):
        if not isinstance(item, dict):
            rejected["invalid_record"] += 1
            continue

        start = xy_point(item.get("start"))
        end = xy_point(item.get("end"))

        if start is None or end is None:
            rejected["missing_endpoints"] += 1
            continue

        if np.linalg.norm(end - start) < EPSILON:
            rejected["degenerate_geometry"] += 1
            continue

        elements.append({
            "element_id": str(
                item.get("element_id", f"element_{index}")
            ),
            "category": str(
                item.get("category", "unknown")
            ),
            "start": start,
            "end": end,
            "length_m": float(
                np.linalg.norm(end - start)
            ),
        })

    if not elements:
        raise RuntimeError(
            "No usable digital-twin elements found."
        )

    return elements, dict(rejected)


def load_candidates():
    data = load_json(CANDIDATES_PATH)

    raw_candidates = data.get("candidates", [])

    if not isinstance(raw_candidates, list):
        raise ValueError(
            "Registration candidates must be a list."
        )

    candidates = []

    for item in raw_candidates:
        angle = finite_float(item.get("angle_deg"))
        tx = finite_float(item.get("tx"))
        ty = finite_float(item.get("ty"))

        if None in (angle, tx, ty):
            continue

        candidates.append({
            "rank": int(item["rank"]),
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
        })

    if not candidates:
        raise RuntimeError(
            "No usable registration candidates found."
        )

    return sorted(
        candidates,
        key=lambda item: item["rank"],
    )


def transform_points(points, candidate):
    theta = math.radians(candidate["angle_deg"])

    c = math.cos(theta)
    s = math.sin(theta)

    rotation = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )

    translation = np.array(
        [candidate["tx"], candidate["ty"]],
        dtype=np.float64,
    )

    return points @ rotation.T + translation


def segment_distances(point, starts, ends):
    """
    Calculate the shortest 2D distance from a point
    to every finite line segment.

    Distances are measured in CAD metres.
    """
    directions = ends - starts

    squared_lengths = np.sum(
        directions * directions,
        axis=1,
    )

    relative = point - starts

    projections = np.sum(
        relative * directions,
        axis=1,
    )

    t = np.divide(
        projections,
        squared_lengths,
        out=np.zeros_like(projections),
        where=squared_lengths > EPSILON,
    )

    t = np.clip(t, 0.0, 1.0)

    closest_points = (
        starts + t[:, None] * directions
    )

    distances = np.linalg.norm(
        closest_points - point,
        axis=1,
    )

    return distances, closest_points


def build_associations(
    frames,
    elements,
    candidates,
):
    starts = np.stack(
        [element["start"] for element in elements]
    )

    ends = np.stack(
        [element["end"] for element in elements]
    )

    trajectory_points = np.array(
        [
            frame["trajectory_xyz"][:2]
            for frame in frames
        ],
        dtype=np.float64,
    )

    all_associations = []
    summary = []

    for candidate in candidates:
        transformed_points = transform_points(
            trajectory_points,
            candidate,
        )

        candidate_associations = []
        frames_with_nearby_elements = 0

        for frame, camera_xy in zip(
            frames,
            transformed_points,
        ):
            distances, closest_points = segment_distances(
                camera_xy,
                starts,
                ends,
            )

            eligible = np.where(
                distances <= MAX_DISTANCE_M
            )[0]

            eligible = sorted(
                eligible,
                key=lambda i: (
                    float(distances[i]),
                    elements[i]["element_id"],
                ),
            )

            eligible = eligible[
                :MAX_ELEMENTS_PER_FRAME
            ]

            nearby = []

            for element_index in eligible:
                element = elements[element_index]

                nearby.append({
                    "element_id": element["element_id"],
                    "category": element["category"],
                    "distance_m": round(
                        float(distances[element_index]),
                        4,
                    ),
                    "closest_cad_xy": [
                        round(float(value), 4)
                        for value in closest_points[
                            element_index
                        ]
                    ],
                    "element_start_xy": [
                        float(value)
                        for value in element["start"]
                    ],
                    "element_end_xy": [
                        float(value)
                        for value in element["end"]
                    ],
                    "association_type": (
                        "GEOMETRIC_PROXIMITY_ONLY"
                    ),
                    "visibility_verified": False,
                    "construction_state_verified": False,
                })

            if nearby:
                frames_with_nearby_elements += 1

            candidate_associations.append({
                "filename": frame["filename"],
                "timestamp_s": frame["timestamp_s"],
                "image_path": frame["image_path"],
                "trajectory_xyz": frame["trajectory_xyz"],
                "candidate_cad_xy": [
                    round(float(camera_xy[0]), 4),
                    round(float(camera_xy[1]), 4),
                ],
                "nearby_elements": nearby,
                "status": (
                    "UNVERIFIED_SPATIAL_ASSOCIATION"
                ),
            })

        all_associations.append({
            "candidate_rank": candidate["rank"],
            "transform": candidate,
            "registration_verified": False,
            "frames": candidate_associations,
        })

        summary.append({
            "candidate_rank": candidate["rank"],
            "frames_processed": len(frames),
            "frames_with_nearby_elements": (
                frames_with_nearby_elements
            ),
            "association_count": sum(
                len(record["nearby_elements"])
                for record in candidate_associations
            ),
        })

    return all_associations, summary


def main():
    print("\nUNDER-CONSTRUCTION")
    print("Candidate building-element associations")
    print("-" * 52)

    frames, frame_rejections = load_frames()
    elements, element_rejections = load_elements()
    candidates = load_candidates()

    print(f"Synchronized frames: {len(frames)}")
    print(f"Usable building elements: {len(elements)}")
    print(f"Registration candidates: {len(candidates)}")

    associations, summary = build_associations(
        frames,
        elements,
        candidates,
    )

    output = {
        "schema_version": "1.0",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "project": "Under-construction",
        "status": "CANDIDATE_ASSOCIATIONS_ONLY",
        "coordinate_system": "CAD_XY_METERS",
        "registration_verified": False,
        "configuration": {
            "max_distance_m": MAX_DISTANCE_M,
            "max_elements_per_frame": (
                MAX_ELEMENTS_PER_FRAME
            ),
            "distance_method": (
                "point_to_finite_segment_2d"
            ),
        },
        "sources": {
            "elements": str(
                ELEMENTS_PATH.relative_to(ROOT)
            ),
            "candidates": str(
                CANDIDATES_PATH.relative_to(ROOT)
            ),
        },
        "rejected_frames": frame_rejections,
        "rejected_elements": element_rejections,
        "summary": summary,
        "candidates": associations,
        "limitations": [
            "Registration candidates are unverified.",
            "Element association uses 2D proximity only.",
            "Nearby does not mean visible in the image.",
            "No camera orientation or field-of-view test.",
            "No occlusion or line-of-sight verification.",
            "No element-level construction state is inferred.",
            "No completion percentage is inferred.",
            "Distances depend on the candidate registration.",
            "Source digital-twin geometry is not BIM-verified.",
        ],
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("\nASSOCIATION SUMMARY")
    print("-" * 52)

    for item in summary:
        print(
            f"Candidate #{item['candidate_rank']}: "
            f"{item['frames_with_nearby_elements']}/"
            f"{item['frames_processed']} frames "
            f"with nearby elements; "
            f"{item['association_count']} associations"
        )

    print("\nStatus: CANDIDATE_ASSOCIATIONS_ONLY")
    print("Visibility verified: NO")
    print("Construction state verified: NO")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
