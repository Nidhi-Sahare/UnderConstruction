
"""
BuildProgress - Step 23
Candidate floor-plan trajectory registration.

Inputs:
    data/geometry/normalized_segments.json
    data/hilti_trajectory/trajectory_inspection.json

Downloads the reference trajectory and initial-pose CSV
from the official Hilti dataset if necessary.

Outputs:
    data/floorplan_registration/
        registration_report.json
        candidate_alignment.png
        aligned_trajectory.csv

IMPORTANT:
    A candidate rigid transform is NOT proof of
    registration against architectural geometry.

Install:
    pip install numpy matplotlib huggingface_hub

Run:
    python floorplan_trajectory_registration.py
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DATASET = (
    "Hilti-Research/"
    "hilti-trimble-slam-challenge-2026"
)

DEFAULT_SEQUENCE = (
    "floor_2_2025-10-28_run_2"
)

DEFAULT_GEOMETRY = (
    ROOT / "data"
    / "geometry"
    / "normalized_segments.json"
)

OUTPUT = (
    ROOT / "data"
    / "floorplan_registration"
)


def quaternion_to_rotation(q):
    """
    Quaternion input order: qx, qy, qz, qw.
    Returns a 3x3 rotation matrix.
    """
    q = np.asarray(
        q, dtype=np.float64
    )

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError(
            "Invalid zero quaternion."
        )

    x, y, z, w = q / norm

    return np.array([
        [
            1 - 2 * (y*y + z*z),
            2 * (x*y - z*w),
            2 * (x*z + y*w),
        ],
        [
            2 * (x*y + z*w),
            1 - 2 * (x*x + z*z),
            2 * (y*z - x*w),
        ],
        [
            2 * (x*z - y*w),
            2 * (y*z + x*w),
            1 - 2 * (x*x + y*y),
        ],
    ], dtype=np.float64)


def pose_matrix(values):
    """
    values:
        tx ty tz qx qy qz qw
    """
    transform = np.eye(
        4, dtype=np.float64
    )

    transform[:3, :3] = (
        quaternion_to_rotation(
            values[3:7]
        )
    )

    transform[:3, 3] = (
        values[:3]
    )

    return transform


def download_file(path):
    return Path(
        hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            filename=path,
        )
    )


def load_reference(sequence):
    path = download_file(
        f"groundtruth/{sequence}.txt"
    )

    data = np.loadtxt(
        path,
        comments="#",
        dtype=np.float64,
    )

    data = np.atleast_2d(data)

    if data.shape[1] != 8:
        raise RuntimeError(
            "Expected TUM trajectory with "
            "8 columns."
        )

    if not np.all(np.isfinite(data)):
        raise RuntimeError(
            "Trajectory contains invalid values."
        )

    if np.any(
        np.diff(data[:, 0]) < 0
    ):
        raise RuntimeError(
            "Trajectory timestamps are not sorted."
        )

    return data


def load_initial_pose(sequence):
    path = download_file(
        "groundtruth/init_gt_poses.csv"
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    matches = [
        row
        for row in rows
        if row.get(
            "#Sequence Name", ""
        ).strip() == sequence
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Expected exactly one matching "
            f"initial pose, found {len(matches)}."
        )

    row = matches[0]

    values = np.array([
        float(row[key])
        for key in (
            "tx", "ty", "tz",
            "qx", "qy", "qz", "qw",
        )
    ], dtype=np.float64)

    timestamp = float(
        row["timestamp"]
    )

    return timestamp, values, row


def interpolate_reference_pose(
    trajectory,
    timestamp,
    tolerance,
):
    """
    Use the nearest reference pose only.
    Do not extrapolate or silently assume
    a timestamp match.
    """
    timestamps = trajectory[:, 0]

    index = int(
        np.argmin(
            np.abs(
                timestamps - timestamp
            )
        )
    )

    difference = abs(
        float(
            timestamps[index]
            - timestamp
        )
    )

    if difference > tolerance:
        return None, {
            "nearest_index": index,
            "nearest_timestamp": float(
                timestamps[index]
            ),
            "time_difference_s": (
                difference
            ),
            "within_tolerance": False,
        }

    return trajectory[index], {
        "nearest_index": index,
        "nearest_timestamp": float(
            timestamps[index]
        ),
        "time_difference_s": difference,
        "within_tolerance": True,
    }


def load_cad_segments(path):
    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    segments = data.get(
        "segments", []
    )

    parsed = []

    for segment in segments:
        start = segment.get(
            "start"
        )

        end = segment.get(
            "end"
        )

        if (
            not isinstance(start, list)
            or not isinstance(end, list)
            or len(start) < 2
            or len(end) < 2
        ):
            continue

        parsed.append({
            "start": np.array(
                start[:2],
                dtype=np.float64,
            ),
            "end": np.array(
                end[:2],
                dtype=np.float64,
            ),
            "category": segment.get(
                "category", "unknown"
            ),
        })

    if not parsed:
        raise RuntimeError(
            "No usable CAD segments."
        )

    return parsed


def transform_positions(
    trajectory,
    transform,
):
    positions = (
        trajectory[:, 1:4]
    )

    return (
        positions @
        transform[:3, :3].T
        + transform[:3, 3]
    )


def plot_alignment(
    segments,
    positions,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    colors = {
        "exterior": "#c65b37",
        "interior": "#444444",
        "partition": "#7b8da5",
        "lining": "#b1b1b1",
    }

    for segment in segments:
        start = segment["start"]
        end = segment["end"]

        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=colors.get(
                segment["category"],
                "#999999",
            ),
            linewidth=1.0,
            alpha=0.65,
        )

    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color="#1261a0",
        linewidth=1.5,
        label="Candidate camera path",
    )

    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        s=80,
        color="green",
        label="Start",
        zorder=5,
    )

    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        s=80,
        marker="x",
        color="red",
        label="End",
        zorder=5,
    )

    ax.set_title(
        "BuildProgress - Candidate CAD / "
        "Trajectory Alignment\n"
        "Not independently verified"
    )

    ax.set_xlabel(
        "Candidate floor X (m)"
    )

    ax.set_ylabel(
        "Candidate floor Y (m)"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        alpha=0.2
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence",
        default=DEFAULT_SEQUENCE,
    )

    parser.add_argument(
        "--geometry",
        type=Path,
        default=DEFAULT_GEOMETRY,
    )

    parser.add_argument(
        "--time-tolerance",
        type=float,
        default=0.05,
    )

    args = parser.parse_args()

    if not args.geometry.is_file():
        parser.error(
            f"Missing geometry: "
            f"{args.geometry}"
        )

    if args.time_tolerance < 0:
        parser.error(
            "Time tolerance must be nonnegative."
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading reference trajectory..."
    )

    trajectory = load_reference(
        args.sequence
    )

    (
        initial_timestamp,
        initial_values,
        initial_row,
    ) = load_initial_pose(
        args.sequence
    )

    (
        matched_pose,
        time_report,
    ) = interpolate_reference_pose(
        trajectory,
        initial_timestamp,
        args.time_tolerance,
    )

    report = {
        "sequence": args.sequence,
        "initial_pose": initial_row,
        "timestamp_match": time_report,
        "registration_status": (
            "NOT VERIFIED"
        ),
    }

    report_path = (
        OUTPUT
        / "registration_report.json"
    )

    if matched_pose is None:
        report[
            "registration_status"
        ] = "ABSTAIN_TIMESTAMP_MISMATCH"

        report_path.write_text(
            json.dumps(
                report, indent=2
            ),
            encoding="utf-8",
        )

        print(
            "\nRegistration abstained:"
        )

        print(
            "Initial pose timestamp:",
            initial_timestamp,
        )

        print(
            "Nearest reference timestamp:",
            time_report[
                "nearest_timestamp"
            ],
        )

        print(
            "Difference:",
            time_report[
                "time_difference_s"
            ],
            "seconds",
        )

        print(
            "\nReport:",
            report_path,
        )

        return

    initial_transform = (
        pose_matrix(
            initial_values
        )
    )

    reference_transform = (
        pose_matrix(
            matched_pose[1:8]
        )
    )

    candidate_transform = (
        initial_transform
        @ np.linalg.inv(
            reference_transform
        )
    )

    aligned_positions = (
        transform_positions(
            trajectory,
            candidate_transform,
        )
    )

    segments = load_cad_segments(
        args.geometry
    )

    preview_path = (
        OUTPUT
        / "candidate_alignment.png"
    )

    plot_alignment(
        segments,
        aligned_positions,
        preview_path,
    )

    csv_path = (
        OUTPUT
        / "aligned_trajectory.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow([
            "timestamp",
            "x",
            "y",
            "z",
        ])

        for pose, position in zip(
            trajectory,
            aligned_positions,
        ):
            writer.writerow([
                pose[0],
                *position.tolist(),
            ])

    report.update({
        "registration_status": (
            "CANDIDATE_ONLY"
        ),
        "candidate_transform": (
            candidate_transform.tolist()
        ),
        "aligned_position_bounds": {
            axis: [
                float(
                    np.min(
                        aligned_positions[:, index]
                    )
                ),
                float(
                    np.max(
                        aligned_positions[:, index]
                    )
                ),
            ]
            for index, axis in enumerate(
                ["x", "y", "z"]
            )
        },
        "limitations": [
            "No independent CAD landmark "
            "correspondences have been verified.",
            "Pose conventions must match "
            "the dataset documentation.",
            "Nearest timestamp matching "
            "does not guarantee exact sync.",
            "A plausible overlay is not "
            "proof of spatial registration.",
        ],
    })

    report_path.write_text(
        json.dumps(
            report, indent=2
        ),
        encoding="utf-8",
    )

    print(
        "\nFLOORPLAN REGISTRATION AUDIT"
    )

    print("-" * 55)

    print(
        "Sequence:",
        args.sequence,
    )

    print(
        "Timestamp difference:",
        round(
            time_report[
                "time_difference_s"
            ],
            6,
        ),
        "seconds",
    )

    print(
        "Registration status:",
        "CANDIDATE_ONLY",
    )

    print(
        "Candidate position bounds:"
    )

    for axis, bounds in report[
        "aligned_position_bounds"
    ].items():
        print(
            f"  {axis}: "
            f"{bounds[0]:.3f} "
            f"to {bounds[1]:.3f}"
        )

    print(
        "\nOutputs:"
    )

    print(report_path)
    print(preview_path)
    print(csv_path)

    print(
        "\nDo not interpret this as "
        "verified CAD registration."
    )


if __name__ == "__main__":
    main()

