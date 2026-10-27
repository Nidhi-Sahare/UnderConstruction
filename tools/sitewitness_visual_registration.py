
"""
SiteWitness — Pose-Aware CAD Registration Diagnostics

Reuses existing BuildProgress artifacts.
Does not perform reconstruction or claim verified registration.

Run:
    python tools/sitewitness_visual_registration.py
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]

LANDMARKS = ROOT / (
    "outputs/multiview_tracks_v2/"
    "validated_landmarks_v2.json"
)

POSES = ROOT / (
    "outputs/multiview_tracks_v2/camera_poses_v2.json"
)

TRAJECTORY = ROOT / (
    "data/real_samples/floor_2_camera_trajectory.json"
)

CANDIDATES = ROOT / (
    "outputs/automatic_registration_v2/"
    "registration_candidates.json"
)

CAD = ROOT / "data/geometry/normalized_segments.json"

OUTPUT = ROOT / "outputs/sitewitness"
REPORT = OUTPUT / "visual_registration_report.json"
CSV = OUTPUT / "visual_registration_candidates.csv"

# Pose agreement is checked in source-coordinate units.
# This is not a verified physical measurement tolerance.
POSE_TOLERANCE = 1e-3

DISTANCE_THRESHOLDS = (0.25, 0.5, 1.0, 2.0)


def read(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8-sig"))


def vector3(value):
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            value = [value["x"], value["y"], value["z"]]
        elif all(k in value for k in ("tx", "ty", "tz")):
            value = [value["tx"], value["ty"], value["tz"]]
        else:
            return None

    try:
        a = np.asarray(value, dtype=float)
    except (ValueError, TypeError):
        return None

    if a.shape != (3,) or not np.isfinite(a).all():
        return None

    return a


def get_landmarks():
    data = read(LANDMARKS)

    if isinstance(data, dict):
        data = data.get("landmarks", [])

    points = []

    for item in data:
        xyz = vector3(item.get("xyz"))

        if xyz is not None and item.get("views", 0) >= 3:
            points.append(xyz)

    if len(points) < 30:
        raise RuntimeError("Insufficient validated landmarks")

    return np.asarray(points)


def get_v2_poses():
    data = read(POSES)

    poses = data.get("poses", {})
    result = {}

    for name, item in poses.items():
        center = vector3(item.get("center"))

        if center is not None:
            result[Path(name).name] = center

    if not result:
        raise RuntimeError("No usable V2 camera poses")

    return result, data


def get_trajectory_poses():
    data = read(TRAJECTORY)

    records = data.get("frames", [])
    result = {}

    if isinstance(records, dict):
        records = [
            {"filename": name, **value}
            for name, value in records.items()
            if isinstance(value, dict)
        ]

    for item in records:
        if not isinstance(item, dict):
            continue

        name = (
            item.get("filename")
            or item.get("frame")
            or item.get("image")
            or item.get("frame_filename")
        )

        pose = item.get("pose", {})

        sources = [item]

        if isinstance(pose, dict):
            sources.insert(0, pose)

        center = None

        for source in sources:
            for key in (
                "position",
                "center",
                "translation",
                "camera_position",
            ):
                center = vector3(source.get(key))

                if center is not None:
                    break

            if center is None:
                center = vector3(source)

            if center is not None:
                break

        if name and center is not None:
            result[Path(str(name)).name] = center

    return result, data


def compare_pose_frames(v2, trajectory):
    shared = sorted(set(v2) & set(trajectory))

    result = {
        "v2_pose_count": len(v2),
        "trajectory_pose_count": len(trajectory),
        "shared_frames": len(shared),
        "coordinate_frames_compatible": False,
        "median_center_difference": None,
        "maximum_center_difference": None,
        "reason": None,
    }

    if len(shared) < 10:
        result["reason"] = "INSUFFICIENT_SHARED_CAMERA_FRAMES"
        return result

    differences = np.asarray([
        np.linalg.norm(v2[name] - trajectory[name])
        for name in shared
    ])

    result["median_center_difference"] = float(
        np.median(differences)
    )

    result["maximum_center_difference"] = float(
        np.max(differences)
    )

    # Compare absolute centers, not only pairwise distances.
    # Pairwise agreement alone permits an unknown rigid transform.
    compatible = bool(
        np.max(differences) <= POSE_TOLERANCE
    )

    result["coordinate_frames_compatible"] = compatible

    result["reason"] = (
        "CAMERA_CENTERS_AGREE_NUMERICALLY"
        if compatible
        else "CAMERA_CENTER_COORDINATES_DISAGREE"
    )

    return result


def xy(value):
    if isinstance(value, dict):
        value = [value.get("x"), value.get("y")]

    try:
        a = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None

    if a.shape != (2,) or not np.isfinite(a).all():
        return None

    return a


def cad_segments():
    data = read(CAD)

    if isinstance(data, dict):
        for key in (
            "segments",
            "normalized_segments",
            "wall_segments",
            "lines",
        ):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise RuntimeError("Unsupported CAD segment structure")

    lines = []

    for item in data:
        if not isinstance(item, dict):
            continue

        endpoints = None

        for a, b in (
            ("start", "end"),
            ("p1", "p2"),
            ("a", "b"),
        ):
            if a in item and b in item:
                endpoints = (item[a], item[b])
                break

        if endpoints is None:
            continue

        a = xy(endpoints[0])
        b = xy(endpoints[1])

        if a is None or b is None:
            continue

        if np.linalg.norm(a - b) < 1e-8:
            continue

        lines.append(LineString([a, b]))

    if not lines:
        raise RuntimeError("No usable CAD segments")

    return unary_union(lines), len(lines)


def get_candidates():
    data = read(CANDIDATES)
    result = []

    for i, item in enumerate(data.get("candidates", [])):
        try:
            values = {
                "candidate_rank": int(item.get("rank", i + 1)),
                "angle_deg": float(item["angle_deg"]),
                "tx": float(item["tx"]),
                "ty": float(item["ty"]),
                "original_candidate_score": float(
                    item.get("score", 0)
                ),
            }
        except (KeyError, ValueError, TypeError):
            continue

        if all(
            math.isfinite(v)
            for k, v in values.items()
            if k != "candidate_rank"
        ):
            result.append(values)

    if not result:
        raise RuntimeError("No registration candidates")

    return result, data


def transform(points, candidate):
    angle = math.radians(candidate["angle_deg"])

    c = math.cos(angle)
    s = math.sin(angle)

    rotation = np.array([
        [c, -s],
        [s, c],
    ])

    return (
        points @ rotation.T
        + np.array([candidate["tx"], candidate["ty"]])
    )


def evaluate(points, geometry, candidate):
    # The candidate search uses a 2D trajectory plane.
    # XY is evaluated only after source pose-frame agreement.
    transformed = transform(points[:, :2], candidate)

    distances = np.asarray([
        Point(float(x), float(y)).distance(geometry)
        for x, y in transformed
    ])

    row = dict(candidate)

    row.update({
        "axis_hypothesis": "XY",
        "landmarks_evaluated": len(distances),
        "median_distance": float(np.median(distances)),
        "mean_distance": float(np.mean(distances)),
        "p90_distance": float(np.percentile(distances, 90)),
        "diagnostic_score": float(
            np.mean(np.exp(-distances))
        ),
    })

    for threshold in DISTANCE_THRESHOLDS:
        row[
            f"fraction_within_{threshold:g}_cad_units"
        ] = float(np.mean(distances <= threshold))

    return row


def save_csv(rows):
    if not rows:
        return

    with CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    print("\n=== SITEWITNESS POSE-AWARE REGISTRATION ===")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    landmarks = get_landmarks()
    v2_poses, pose_metadata = get_v2_poses()
    trajectory_poses, trajectory_data = (
        get_trajectory_poses()
    )

    candidates, candidate_metadata = get_candidates()
    geometry, segment_count = cad_segments()

    agreement = compare_pose_frames(
        v2_poses,
        trajectory_poses,
    )

    print("Visual landmarks:", len(landmarks))
    print("CAD segments:", segment_count)
    print("Candidates:", len(candidates))
    print("Shared camera frames:", agreement["shared_frames"])
    print("Pose-frame test:", agreement["reason"])

    results = []

    # Do not apply trajectory-derived candidate transforms
    # to landmarks in an incompatible coordinate frame.
    if agreement["coordinate_frames_compatible"]:
        for candidate in candidates:
            results.append(
                evaluate(landmarks, geometry, candidate)
            )

        results.sort(
            key=lambda x: x["diagnostic_score"],
            reverse=True,
        )

    # Even numerical pose agreement does not independently
    # verify horizontal axes, metric scale, or CAD identity.
    # Accordingly no candidate is promoted to verified.
    report = {
        "schema_version": "2.0",
        "system": "SiteWitness",
        "analysis": "POSE_AWARE_VISUAL_REGISTRATION",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "DIAGNOSTIC_ONLY",
        "registration_verified": False,
        "construction_compliance_verified": False,
        "construction_progress_verified": False,
        "inputs": {
            "landmarks": str(LANDMARKS.relative_to(ROOT)),
            "camera_poses": str(POSES.relative_to(ROOT)),
            "trajectory": str(TRAJECTORY.relative_to(ROOT)),
            "candidates": str(CANDIDATES.relative_to(ROOT)),
            "cad": str(CAD.relative_to(ROOT)),
        },
        "input_summary": {
            "landmark_count": len(landmarks),
            "cad_segment_count": segment_count,
            "candidate_count": len(candidates),
        },
        "pose_frame_validation": agreement,
        "source_verification": {
            "v2_camera_extrinsics_verified":
                pose_metadata.get(
                    "camera_extrinsics_verified", False
                ),
            "trajectory_coordinate_system":
                trajectory_data.get(
                    "coordinate_system", {}
                ),
            "candidate_status":
                candidate_metadata.get("status"),
        },
        "results": results,
        "best_diagnostic_candidate": (
            results[0]["candidate_rank"]
            if results else None
        ),
        "decision": {
            "selected_verified_candidate": None,
            "status": (
                "ABSTAIN_UNVERIFIED_VISUAL_CAD_CORRESPONDENCE"
                if results
                else "ABSTAIN_INCOMPATIBLE_OR_UNKNOWN_POSE_FRAMES"
            ),
        },
        "limitations": [
            "Pose-frame agreement is numerical, not "
            "independent camera calibration.",
            "XY is the trajectory candidate search plane; "
            "its physical horizontal orientation is unverified.",
            "Camera-to-IMU extrinsics are unverified.",
            "Metric scale is unverified.",
            "Visual landmarks are not classified as walls.",
            "CAD-line proximity is not an architectural "
            "correspondence test.",
            "Diagnostic scores are not probabilities.",
            "No verified CAD alignment or construction "
            "discrepancy is produced.",
        ],
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    save_csv(results)

    if results:
        print("\nDiagnostic candidate ranking:")

        for row in results:
            print(
                f"#{row['candidate_rank']} "
                f"score={row['diagnostic_score']:.4f} "
                f"median_distance="
                f"{row['median_distance']:.3f}"
            )

    print("\nRegistration verified: NO")
    print("Construction compliance verified: NO")
    print("Decision:", report["decision"]["status"])
    print("\nReport:", REPORT.relative_to(ROOT))
    print("CSV:", CSV.relative_to(ROOT))


if __name__ == "__main__":
    main()
