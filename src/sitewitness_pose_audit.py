
"""
SiteWitness — Trajectory and Synchronization Audit

Run:
    python -m src.sitewitness_pose_audit

Does not assume or certify a world/IMU pose convention.
Does not modify existing registration or dashboard files.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .sitewitness_camera_geometry import load_calibration


ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY = (
    ROOT / "data" / "real_samples"
    / "floor_2_camera_trajectory.json"
)

FRAME_DIR = (
    ROOT / "sample_gallery" / "real_floor_2" / "frames"
)

OUTPUT = (
    ROOT / "outputs" / "sitewitness"
    / "pose_audit.json"
)


def numeric_vector(value, length):
    if isinstance(value, dict):
        keys = (
            ("x", "y", "z")
            if length == 3
            else ("x", "y", "z", "w")
        )
        value = [value.get(k) for k in keys]

    try:
        array = np.asarray(value, dtype=float)
        if array.shape == (length,) and np.isfinite(array).all():
            return array
    except (TypeError, ValueError):
        pass

    return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def summarize(values):
    if not values:
        return None

    array = np.asarray(values, dtype=float)

    return {
        "count": len(values),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def main():
    data = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    frames = data.get("frames")

    if not isinstance(frames, list):
        raise ValueError("Expected trajectory.frames list")

    cam = load_calibration("cam0")

    records = []
    positions = []
    timestamps = []
    pose_time_differences = []
    unavailable = []
    missing_images = []
    invalid_poses = []

    for index, record in enumerate(frames):
        if not isinstance(record, dict):
            invalid_poses.append(index)
            continue

        filename = record.get("filename")
        image_exists = (
            isinstance(filename, str)
            and (FRAME_DIR / filename).is_file()
        )

        if not image_exists:
            missing_images.append({
                "index": index,
                "filename": filename,
            })

        position = numeric_vector(record.get("position"), 3)
        orientation = numeric_vector(record.get("orientation"), 4)

        pose_available = position is not None and orientation is not None

        timestamp = number(record.get("timestamp_s"))
        pose_delta = number(record.get("pose_time_difference_s"))

        if timestamp is not None:
            timestamps.append(timestamp)

        if pose_delta is not None:
            pose_time_differences.append(abs(pose_delta))

        if pose_available:
            positions.append((index, position))
        else:
            unavailable.append(index)

        records.append({
            "index": record.get("index", index),
            "filename": filename,
            "image_exists": image_exists,
            "timestamp_s": timestamp,
            "bag_timestamp_s": number(
                record.get("bag_timestamp_s")
            ),
            "pose_available": pose_available,
            "pose_time_difference_s": pose_delta,
            "position": (
                position.tolist()
                if position is not None else None
            ),
            "orientation_xyzw_assumed_for_arithmetic_only": (
                orientation.tolist()
                if orientation is not None else None
            ),
            "quaternion_norm": (
                float(np.linalg.norm(orientation))
                if orientation is not None else None
            ),
        })

    timestamp_steps = [
        b - a for a, b in zip(timestamps, timestamps[1:])
    ]

    position_steps = [
        float(np.linalg.norm(b[1] - a[1]))
        for a, b in zip(positions, positions[1:])
    ]

    pose_gaps = [
        b[0] - a[0]
        for a, b in zip(positions, positions[1:])
    ]

    # This is a consistency test only.
    # It does not prove which coordinate frame the poses use.
    quaternion_norms = [
        record["quaternion_norm"]
        for record in records
        if record["quaternion_norm"] is not None
    ]

    calibration_shift = cam.timeshift_cam_imu

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "module": "SiteWitness Pose Audit",
        "source": str(TRAJECTORY.relative_to(ROOT)),
        "trajectory_metadata": {
            "source": data.get("source"),
            "coordinate_system": data.get("coordinate_system"),
            "summary": data.get("summary"),
        },
        "frame_count": len(frames),
        "valid_pose_count": len(positions),
        "missing_pose_indices": unavailable,
        "missing_images": missing_images,
        "invalid_records": invalid_poses,
        "timestamp_step_seconds": summarize(timestamp_steps),
        "absolute_pose_time_difference_seconds": summarize(
            pose_time_differences
        ),
        "position_step_recorded_units": summarize(position_steps),
        "pose_index_gaps": summarize(pose_gaps),
        "quaternion_norms": summarize(quaternion_norms),
        "camera_calibration": {
            "camera": "cam0",
            "timeshift_cam_imu_seconds": calibration_shift,
            "extrinsic_direction": "T_cam_imu",
            "image_resolution": [cam.width, cam.height],
        },
        "sample_records": (
            records[:3]
            + records[max(0, len(records) // 2 - 1):
                      min(len(records), len(records) // 2 + 2)]
            + records[-3:]
        ),
        "checks": {
            "images_found_for_all_records": not missing_images,
            "at_least_one_valid_pose": bool(positions),
            "quaternions_near_unit_norm": bool(quaternion_norms)
            and all(abs(n - 1.0) < 0.02 for n in quaternion_norms),
            "timestamps_monotonic": all(
                delta > 0 for delta in timestamp_steps
            ),
        },
        "unverified": [
            "trajectory_position_coordinate_frame",
            "trajectory_quaternion_direction",
            "trajectory_quaternion_component_order",
            "trajectory_position_units",
            "camera_to_trajectory_timestamp_convention",
            "CAD_to_trajectory_registration",
        ],
        "registration_verified": False,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nSITEWITNESS — POSE AUDIT")
    print("=" * 48)
    print("Frames:", len(frames))
    print("Valid poses:", len(positions))
    print("Missing images:", len(missing_images))
    print("Quaternion norms:", report["quaternion_norms"])
    print("Frame timing:", report["timestamp_step_seconds"])
    print(
        "Pose association errors:",
        report["absolute_pose_time_difference_seconds"],
    )
    print("Trajectory metadata:", report["trajectory_metadata"])
    print("Pose convention: UNVERIFIED")
    print("CAD registration: UNVERIFIED")
    print("Report:", OUTPUT)


if __name__ == "__main__":
    main()
