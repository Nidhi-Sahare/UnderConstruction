
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_archive" / "hilti_transition_20261023_004237" / "data"
SEQUENCE = "floor_2_2025-10-28_run_2"

ELEMENTS_FILE = ROOT / "data/digital_twin/building_elements.json"
MANIFEST_FILE = (
    ARCHIVE / "hilti_frames" / SEQUENCE
    / "cam0/frame_manifest.csv"
)
REGISTRATION_FILE = (
    ROOT / "outputs/automatic_registration_v2"
    / "registration_candidates.json"
)
OUTPUT_DIR = ROOT / "outputs/automatic_element_evidence"


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_frames():
    frames = []

    with MANIFEST_FILE.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if str(row.get("pose_matched", "")).lower() != "true":
                continue

            fields = (
                "tx", "ty", "tz",
                "qx", "qy", "qz", "qw",
            )
            values = [numeric(row.get(key)) for key in fields]

            if any(value is None for value in values):
                continue

            position = np.array(values[:3], dtype=float)
            quaternion = np.array(values[3:], dtype=float)

            norm = np.linalg.norm(quaternion)
            if not 0.9 <= norm <= 1.1:
                continue

            quaternion /= norm

            frames.append({
                "index": index,
                "filename": row["filename"],
                "position": position,
                "rotation": Rotation.from_quat(quaternion),
            })

    return frames


def load_elements():
    result = []

    for element in load_json(ELEMENTS_FILE)["elements"]:
        start = element.get("start")
        end = element.get("end")

        if not isinstance(start, list) or not isinstance(end, list):
            continue
        if len(start) < 2 or len(end) < 2:
            continue

        coords = [numeric(v) for v in start[:2] + end[:2]]
        if any(v is None for v in coords):
            continue

        result.append({
            "id": element["element_id"],
            "category": element.get("category", "unknown"),
            "start": np.array(coords[:2], dtype=float),
            "end": np.array(coords[2:], dtype=float),
        })

    return result


def load_candidate(rank):
    data = load_json(REGISTRATION_FILE)

    for candidate in data.get("candidates", []):
        if candidate.get("rank") == rank:
            return candidate

    raise ValueError(f"Registration candidate #{rank} not found")


def transform_xy(point, candidate):
    theta = math.radians(candidate["angle_deg"])
    c, s = math.cos(theta), math.sin(theta)

    x, y = point[:2]

    return np.array([
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    ])


def segment_distance(point, start, end):
    direction = end - start
    length_squared = float(direction @ direction)

    if length_squared < 1e-12:
        return float(np.linalg.norm(point - start))

    t = np.clip(
        float((point - start) @ direction) / length_squared,
        0.0,
        1.0,
    )

    nearest = start + t * direction
    return float(np.linalg.norm(point - nearest))


def provisional_direction(frame, candidate, axis, convention):
    """
    EXPERIMENTAL ONLY.

    The manifest quaternion's coordinate convention and
    camera-to-IMU extrinsics are not established.

    This function evaluates a named hypothesis; it does
    not produce a calibrated camera viewing direction.
    """
    local_axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }

    rotation = frame["rotation"]

    if convention == "inverse":
        rotation = rotation.inv()

    direction = rotation.apply(local_axes[axis])

    theta = math.radians(candidate["angle_deg"])
    c, s = math.cos(theta), math.sin(theta)

    dx = c * direction[0] - s * direction[1]
    dy = s * direction[0] + c * direction[1]

    norm = math.hypot(dx, dy)

    if norm < 1e-8:
        return None

    return np.array([dx / norm, dy / norm])


def compute_associations(
    elements,
    frames,
    candidate,
    radius,
    limit,
    axis,
    convention,
):
    transformed = []

    for frame in frames:
        position = transform_xy(frame["position"], candidate)
        direction = provisional_direction(
            frame, candidate, axis, convention
        )

        transformed.append((frame, position, direction))

    results = []

    for element in elements:
        midpoint = (element["start"] + element["end"]) / 2
        matches = []

        for frame, camera_xy, direction in transformed:
            distance = segment_distance(
                camera_xy,
                element["start"],
                element["end"],
            )

            if distance > radius:
                continue

            to_element = midpoint - camera_xy
            target_norm = float(np.linalg.norm(to_element))

            alignment = None
            angle_deg = None

            if direction is not None and target_norm > 1e-8:
                alignment = float(
                    np.clip(
                        direction @ (to_element / target_norm),
                        -1.0,
                        1.0,
                    )
                )
                angle_deg = math.degrees(math.acos(alignment))

            # Ranking deliberately uses distance alone.
            # Uncalibrated orientation is diagnostic only.
            matches.append({
                "frame_index": frame["index"],
                "filename": frame["filename"],
                "distance_m": round(distance, 3),
                "hypothesis_angle_deg": (
                    round(angle_deg, 2)
                    if angle_deg is not None else None
                ),
                "hypothesis_alignment": (
                    round(alignment, 4)
                    if alignment is not None else None
                ),
                "evidence_status": "GEOMETRIC_CANDIDATE_ONLY",
                "visibility_verified": False,
            })

        matches.sort(
            key=lambda item: (
                item["distance_m"],
                item["frame_index"],
            )
        )

        selected = matches[:limit]

        results.append({
            "element_id": element["id"],
            "category": element["category"],
            "construction_status": "UNKNOWN",
            "candidate_count_within_radius": len(matches),
            "candidate_frames": selected,
            "association_status": (
                "CANDIDATES_AVAILABLE"
                if selected else "NO_SPATIAL_CANDIDATES"
            ),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=int, default=1)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--axis",
        choices=["x", "y", "z", "-x", "-y", "-z"],
        default="z",
    )
    parser.add_argument(
        "--convention",
        choices=["direct", "inverse"],
        default="direct",
    )

    args = parser.parse_args()

    if not math.isfinite(args.radius) or args.radius <= 0:
        parser.error("--radius must be positive and finite")
    if args.limit < 1:
        parser.error("--limit must be at least 1")

    elements = load_elements()
    frames = load_frames()
    candidate = load_candidate(args.candidate)

    results = compute_associations(
        elements,
        frames,
        candidate,
        args.radius,
        args.limit,
        args.axis,
        args.convention,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "schema_version": "1.0",
        "registration_candidate_rank": args.candidate,
        "registration_verified": False,
        "camera_to_imu_extrinsics_verified": False,
        "pose_convention_verified": False,
        "orientation_hypothesis": {
            "axis": args.axis,
            "convention": args.convention,
            "usage": "DIAGNOSTIC_ONLY_NOT_RANKING",
        },
        "ranking_method": "CAMERA_TO_ELEMENT_DISTANCE",
        "search_radius_m": args.radius,
        "visibility_verified": False,
        "progress_inference_performed": False,
        "elements": results,
    }

    output_path = (
        OUTPUT_DIR
        / f"candidate_{args.candidate}_associations.json"
    )

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    with_candidates = sum(
        bool(item["candidate_frames"]) for item in results
    )

    print("\nUNDER-CONSTRUCTION — AUTOMATIC ASSOCIATION")
    print("-" * 55)
    print("Digital-twin elements:", len(elements))
    print("Pose-matched frames:", len(frames))
    print("Registration candidate:", args.candidate)
    print("Search radius:", args.radius, "m")
    print("Elements with candidates:", with_candidates)
    print("Elements without candidates:", len(results) - with_candidates)
    print("Orientation: experimental diagnostic only")
    print("Visibility: NOT VERIFIED")
    print("Construction progress: NOT INFERRED")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

