
"""
Synchronize Hilti walkthrough frames with reference trajectories.

Inputs:
    data/hilti_groundtruth/floor_1.txt
    data/hilti_groundtruth/floor_3.txt

    sample_gallery/real_floor_1/manifest.json
    sample_gallery/real_floor_3/manifest.json

Outputs:
    sample_gallery/real_floor_1/trajectory.json
    sample_gallery/real_floor_3/trajectory.json

No CAD registration is performed.
No missing poses are fabricated.

Run:
    python src/synchronize_hilti_trajectories.py
"""

from __future__ import annotations

import bisect
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLOORS = (1, 3)

MAX_MATCH_ERROR_SECONDS = 0.10

COMMENT_PREFIXES = ("#", "//", "%")


def read_rows(path):
    rows = []
    header = []

    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw.strip()

        if not line:
            continue

        if line.startswith(COMMENT_PREFIXES):
            if not rows:
                header.append(line)
            continue

        parts = [
            part
            for part in re.split(r"[\s,;]+", line)
            if part
        ]

        try:
            values = [float(part) for part in parts]
        except ValueError:
            if not rows:
                header.append(line)
                continue
            raise ValueError(
                f"Non-numeric trajectory row at line {line_number}"
            )

        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                f"Non-finite trajectory row at line {line_number}"
            )

        rows.append(values)

    if not rows:
        raise ValueError(f"No numeric rows in {path}")

    widths = {len(row) for row in rows}

    if len(widths) != 1:
        raise ValueError(
            f"Inconsistent trajectory columns: {sorted(widths)}"
        )

    return rows, header


def detect_time_scale(values):
    """
    Convert a timestamp column into seconds.

    Candidate scales:
        seconds, milliseconds, microseconds, nanoseconds

    Select using the median positive sample interval.
    Require a plausible reference pose sampling rate.
    """
    if len(values) < 3:
        raise ValueError("Too few reference timestamps")

    diffs = sorted(
        b - a
        for a, b in zip(values, values[1:])
        if b > a
    )

    if not diffs:
        raise ValueError("Reference timestamps are not increasing")

    median = diffs[len(diffs) // 2]

    candidates = [
        ("seconds", 1.0),
        ("milliseconds", 1e-3),
        ("microseconds", 1e-6),
        ("nanoseconds", 1e-9),
    ]

    plausible = [
        (name, scale)
        for name, scale in candidates
        if 0.001 <= median * scale <= 0.2
    ]

    if len(plausible) != 1:
        raise ValueError(
            "Timestamp units ambiguous. "
            f"Median raw interval={median}, "
            f"candidate units={plausible}"
        )

    return plausible[0]


def nearest_index(sorted_values, value):
    index = bisect.bisect_left(sorted_values, value)

    candidates = []

    if index < len(sorted_values):
        candidates.append(index)

    if index > 0:
        candidates.append(index - 1)

    return min(
        candidates,
        key=lambda i: abs(sorted_values[i] - value),
    )


def synchronize_floor(floor):
    manifest_path = (
        ROOT
        / "sample_gallery"
        / f"real_floor_{floor}"
        / "manifest.json"
    )

    reference_path = (
        ROOT
        / "data"
        / "hilti_groundtruth"
        / f"floor_{floor}.txt"
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    rows, header = read_rows(reference_path)

    column_count = len(rows[0])

    if column_count != 8:
        raise ValueError(
            f"Floor {floor}: expected 8 trajectory columns "
            "(timestamp, tx, ty, tz, qx, qy, qz, qw), "
            f"found {column_count}. "
            "Inspect the source header before mapping columns."
        )

    reference_timestamps = [row[0] for row in rows]

    unit_name, time_scale = detect_time_scale(
        reference_timestamps
    )

    reference_times = [
        timestamp * time_scale
        for timestamp in reference_timestamps
    ]

    frames = manifest["frames"]

    if not frames:
        raise ValueError("Walkthrough manifest contains no frames")

    frame_header_times = [
        frame["header_timestamp_ns"] / 1e9
        for frame in frames
    ]

    frame_bag_times = [
        frame["bag_timestamp_ns"] / 1e9
        for frame in frames
    ]

    candidates = {
        "camera_header": frame_header_times,
        "rosbag_timestamp": frame_bag_times,
    }

    def evaluate_clock(clock_values):
        errors = [
            abs(
                reference_times[
                    nearest_index(reference_times, t)
                ] - t
            )
            for t in clock_values
        ]

        ordered = sorted(errors)

        return {
            "median_error_seconds": ordered[len(ordered) // 2],
            "maximum_error_seconds": max(errors),
            "matched_count": sum(
                error <= MAX_MATCH_ERROR_SECONDS
                for error in errors
            ),
        }

    clock_scores = {
        name: evaluate_clock(values)
        for name, values in candidates.items()
    }

    best_clock = min(
        candidates,
        key=lambda name: (
            -clock_scores[name]["matched_count"],
            clock_scores[name]["median_error_seconds"],
        ),
    )

    best_score = clock_scores[best_clock]

    if best_score["matched_count"] == 0:
        raise ValueError(
            f"Floor {floor}: no timestamps match within "
            f"{MAX_MATCH_ERROR_SECONDS}s.\n"
            f"Reference range: "
            f"{reference_times[0]:.6f} .. "
            f"{reference_times[-1]:.6f}\n"
            f"Camera header range: "
            f"{frame_header_times[0]:.6f} .. "
            f"{frame_header_times[-1]:.6f}\n"
            f"ROS bag range: "
            f"{frame_bag_times[0]:.6f} .. "
            f"{frame_bag_times[-1]:.6f}\n"
            "No arbitrary offset has been applied."
        )

    synchronized = []

    for frame, frame_time in zip(
        frames,
        candidates[best_clock],
    ):
        index = nearest_index(
            reference_times,
            frame_time,
        )

        error = abs(
            reference_times[index] - frame_time
        )

        matched = error <= MAX_MATCH_ERROR_SECONDS

        pose = rows[index] if matched else None

        synchronized.append({
            "frame_index": frame["index"],
            "filename": frame["filename"],
            "time_seconds": frame["time_seconds"],
            "camera_timestamp_ns": (
                frame["header_timestamp_ns"]
            ),
            "pose_available": matched,
            "pose_timestamp_seconds": (
                reference_times[index]
                if matched else None
            ),
            "time_error_seconds": round(error, 9),
            "position": (
                {
                    "x": pose[1],
                    "y": pose[2],
                    "z": pose[3],
                }
                if pose else None
            ),
            "orientation_xyzw": (
                {
                    "x": pose[4],
                    "y": pose[5],
                    "z": pose[6],
                    "w": pose[7],
                }
                if pose else None
            ),
        })

    matched_count = sum(
        item["pose_available"]
        for item in synchronized
    )

    result = {
        "schema_version": "1.0",
        "floor": floor,
        "source": "Hilti reference trajectory",
        "source_file": str(
            reference_path.relative_to(ROOT)
        ).replace("\\", "/"),
        "source_header": header,
        "reference_rows": len(rows),
        "reference_columns": column_count,
        "timestamp_unit": unit_name,
        "selected_camera_clock": best_clock,
        "clock_scores": clock_scores,
        "maximum_allowed_match_error_seconds": (
            MAX_MATCH_ERROR_SECONDS
        ),
        "total_frames": len(synchronized),
        "matched_frames": matched_count,
        "unmatched_frames": (
            len(synchronized) - matched_count
        ),
        "coordinate_system": {
            "reference": "Hilti trajectory reference frame",
            "cad_registered": False,
            "horizontal_axes_verified": False,
            "camera_extrinsics_applied": False,
            "construction_progress_verified": False,
        },
        "samples": synchronized,
    }

    output_path = (
        manifest_path.parent / "trajectory.json"
    )

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\nFloor {floor}")
    print(f"  Reference poses: {len(rows)}")
    print(f"  Columns: {column_count}")
    print(f"  Timestamp unit: {unit_name}")
    print(f"  Selected clock: {best_clock}")
    print(
        f"  Matched frames: "
        f"{matched_count}/{len(synchronized)}"
    )
    print(
        "  Median timestamp error:",
        round(
            best_score["median_error_seconds"],
            6,
        ),
        "seconds",
    )
    print(f"  Saved: {output_path}")

    return result


def main():
    print("Hilti trajectory synchronization")

    for floor in FLOORS:
        synchronize_floor(floor)

    print("\nSynchronization complete.")
    print("CAD registration remains unverified.")


if __name__ == "__main__":
    main()
