
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
    / "frame_manifest.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "real_samples"
    / "floor_2_camera_trajectory.json"
)


def optional_float(value):
    if value is None or str(value).strip() == "":
        return None

    result = float(value)

    if not math.isfinite(result):
        raise ValueError(f"Non-finite numeric value: {value}")

    return result


def read_manifest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        required = {
            "filename",
            "bag_timestamp_s",
            "header_timestamp_s",
            "pose_matched",
            "pose_time_difference_s",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"Missing manifest columns: {sorted(missing)}"
            )

        return list(reader)


def build_trajectory(rows):
    frames = []

    for index, row in enumerate(rows):
        filename = (row["filename"] or "").strip()
        matched = (row["pose_matched"] or "").strip().lower() == "true"

        position_values = [
            optional_float(row[key])
            for key in ("tx", "ty", "tz")
        ]

        orientation_values = [
            optional_float(row[key])
            for key in ("qx", "qy", "qz", "qw")
        ]

        complete_pose = (
            matched
            and all(value is not None for value in position_values)
            and all(value is not None for value in orientation_values)
        )

        position = None
        orientation = None

        if complete_pose:
            position = {
                "x": position_values[0],
                "y": position_values[1],
                "z": position_values[2],
            }

            orientation = {
                "x": orientation_values[0],
                "y": orientation_values[1],
                "z": orientation_values[2],
                "w": orientation_values[3],
            }

        frames.append(
            {
                "index": index,
                "filename": filename,
                "timestamp_s": optional_float(
                    row["header_timestamp_s"]
                ),
                "bag_timestamp_s": optional_float(
                    row["bag_timestamp_s"]
                ),
                "pose_available": complete_pose,
                "pose_time_difference_s": optional_float(
                    row["pose_time_difference_s"]
                ),
                "position": position,
                "orientation": orientation,
            }
        )

    valid_frames = [
        frame
        for frame in frames
        if frame["pose_available"]
    ]

    if valid_frames:
        xs = [frame["position"]["x"] for frame in valid_frames]
        ys = [frame["position"]["y"] for frame in valid_frames]
        zs = [frame["position"]["z"] for frame in valid_frames]

        bounds = {
            "x": [min(xs), max(xs)],
            "y": [min(ys), max(ys)],
            "z": [min(zs), max(zs)],
        }
    else:
        bounds = None

    timestamps = [
        frame["timestamp_s"]
        for frame in frames
        if frame["timestamp_s"] is not None
    ]

    duration = (
        max(timestamps) - min(timestamps)
        if timestamps
        else None
    )

    return {
        "schema_version": 1,
        "source": "Hilti floor_2_2025-10-28_run_2 cam0",
        "coordinate_system": {
            "name": "recorded_pose_coordinates",
            "cad_registered": False,
            "reference_frame_verified": False,
            "horizontal_axes_verified": False,
            "units_verified": False,
            "notes": (
                "Raw pose translation and quaternion values from "
                "the frame manifest. Camera-to-CAD registration, "
                "reference-frame convention, and axis orientation "
                "have not been verified."
            ),
        },
        "summary": {
            "total_frames": len(frames),
            "frames_with_pose": len(valid_frames),
            "frames_without_pose": len(frames) - len(valid_frames),
            "pose_coverage_fraction": (
                len(valid_frames) / len(frames)
                if frames
                else 0.0
            ),
            "recording_duration_s": duration,
            "position_bounds": bounds,
        },
        "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare real recorded camera trajectory data."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    result = build_trajectory(rows)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            allow_nan=False,
        )

    summary = result["summary"]

    print("\nCAMERA TRAJECTORY PREPARATION")
    print("=" * 48)
    print(f"Source: {args.manifest}")
    print(f"Total frames: {summary['total_frames']}")
    print(f"Frames with poses: {summary['frames_with_pose']}")
    print(f"Frames without poses: {summary['frames_without_pose']}")
    print(
        "Pose coverage: "
        f"{summary['pose_coverage_fraction'] * 100:.1f}%"
    )

    if summary["recording_duration_s"] is not None:
        print(
            "Recording duration: "
            f"{summary['recording_duration_s']:.2f} s"
        )

    print(f"Position bounds: {summary['position_bounds']}")
    print("CAD registration: NOT VERIFIED")
    print(f"Output: {args.output}")
    print("\nPASS: Trajectory JSON created successfully.")


if __name__ == "__main__":
    main()
