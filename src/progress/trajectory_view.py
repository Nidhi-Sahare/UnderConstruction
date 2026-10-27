
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

SEQUENCE = "floor_2_2025-10-28_run_2"

TRAJECTORY_CSV = (
    ARCHIVE
    / "floorplan_registration"
    / "aligned_trajectory.csv"
)

FRAME_MANIFEST = (
    ARCHIVE
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frame_manifest.csv"
)


def load_trajectory():
    if not TRAJECTORY_CSV.exists():
        raise FileNotFoundError(
            f"Trajectory missing: {TRAJECTORY_CSV}"
        )

    points = []

    with TRAJECTORY_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            try:
                points.append(
                    {
                        "timestamp": float(row["timestamp"]),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue

    if not points:
        raise RuntimeError("No valid trajectory points")

    points.sort(key=lambda p: p["timestamp"])

    return points


def load_frame_timestamps():
    if not FRAME_MANIFEST.exists():
        return []

    with FRAME_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return []

    # Manifest column names must be checked before claiming
    # exact frame-to-pose synchronization.
    possible_time_columns = (
        "timestamp",
        "timestamp_sec",
        "time",
        "time_sec",
    )

    timestamp_column = next(
        (
            key
            for key in possible_time_columns
            if key in rows[0]
        ),
        None,
    )

    if timestamp_column is None:
        return []

    timestamps = []

    for row in rows:
        try:
            timestamps.append(float(row[timestamp_column]))
        except (ValueError, TypeError):
            timestamps.append(None)

    return timestamps


def render_trajectory(
    trajectory,
    current_index=None,
    width=900,
    height=650,
):
    image = np.full(
        (height, width, 3),
        (24, 28, 34),
        dtype=np.uint8,
    )

    xy = np.array(
        [[p["x"], p["y"]] for p in trajectory],
        dtype=np.float64,
    )

    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)

    extent = np.maximum(maximum - minimum, 1e-6)

    padding = 65

    scale = min(
        (width - 2 * padding) / extent[0],
        (height - 2 * padding) / extent[1],
    )

    def project(point):
        x = padding + (point[0] - minimum[0]) * scale
        y = height - padding - (point[1] - minimum[1]) * scale

        return int(round(x)), int(round(y))

    pixels = np.array(
        [project(p) for p in xy],
        dtype=np.int32,
    )

    if len(pixels) >= 2:
        cv2.polylines(
            image,
            [pixels],
            False,
            (190, 145, 75),
            3,
            cv2.LINE_AA,
        )

    cv2.circle(
        image,
        tuple(pixels[0]),
        7,
        (90, 220, 120),
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        image,
        tuple(pixels[-1]),
        7,
        (80, 120, 240),
        -1,
        cv2.LINE_AA,
    )

    if current_index is not None:
        current_index = max(
            0,
            min(len(pixels) - 1, int(current_index)),
        )

        cv2.circle(
            image,
            tuple(pixels[current_index]),
            10,
            (0, 155, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            image,
            tuple(pixels[current_index]),
            16,
            (0, 155, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "HILTI CAMERA TRAJECTORY",
        (24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Trajectory coordinates only - CAD alignment unverified",
        (24, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 190, 200),
        1,
        cv2.LINE_AA,
    )

    return image


def nearest_pose_index(trajectory, timestamp):
    times = np.array(
        [p["timestamp"] for p in trajectory],
        dtype=np.float64,
    )

    index = int(
        np.argmin(np.abs(times - timestamp))
    )

    return index


if __name__ == "__main__":
    trajectory = load_trajectory()

    image = render_trajectory(trajectory)

    output = ROOT / "outputs" / "hilti_trajectory_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output), image)

    print(f"Trajectory points: {len(trajectory)}")
    print(f"Preview saved: {output}")

    timestamps = load_frame_timestamps()

    if timestamps:
        print(
            f"Frame timestamps found: {len(timestamps)}"
        )
    else:
        print(
            "Frame timestamp synchronization not yet verified."
        )
