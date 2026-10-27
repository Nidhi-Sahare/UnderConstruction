
"""
Under-construction — Step 82
Camera pose geometry audit.

Read-only, deterministic, classical geometry.

Audits:
- Camera intrinsics and distortion coefficients
- Manifest pose validity
- Rotation orthogonality and determinant
- Quaternion continuity
- Trajectory displacement and rotation continuity
- Camera optical-axis orientation
- Floor-plane ray intersection under two pose conventions
- Official camera/IMU extrinsic validity

Does NOT:
- Verify the trajectory coordinate convention
- Register the CAD to the trajectory
- Infer construction progress
- Choose a pose convention using image-edge scores

Output:
    outputs/camera_pose_geometry/geometry_audit.json
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

FRAME_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
)

MANIFEST = FRAME_ROOT / "frame_manifest.csv"

CALIBRATION = (
    ROOT
    / "data"
    / "hilti_calibration"
    / "kalibr_imucam_chain.yaml"
)

OUTPUT_DIR = ROOT / "outputs" / "camera_pose_geometry"
OUTPUT_FILE = OUTPUT_DIR / "geometry_audit.json"

K = np.array(
    [
        [465.3015482593691, 0.0, 730.0455886686005],
        [0.0, 465.32303798346413, 720.1427007671206],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

D = np.array(
    [
        0.025800718903376804,
        -0.010909240777406872,
        -0.0016899537986031076,
        0.00014766801645260894,
    ],
    dtype=np.float64,
)

IMAGE_WIDTH = 1472
IMAGE_HEIGHT = 1440

# Official YAML T_cam_imu, included here as a reference
# to audit. Do not automatically apply it to poses.
T_CAM_IMU = np.array(
    [
        [
            0.017214474772216132,
            -0.0008034642120502422,
            -0.9998514971252359,
            0.020670851120764513,
        ],
        [
            0.9998263174555488,
            -0.007128426214556394,
            0.017219769539067287,
            0.015539085669546057,
        ],
        [
            -0.007141203091335369,
            -0.9999742696614562,
            0.0006806125511055194,
            -0.01575188948566258,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# Diagnostic only. The map-frame floor elevation
# has not been independently established.
ASSUMED_FLOOR_Z = 0.0

MIN_DOWNWARD_RAY_Z = 0.05
MAX_FLOOR_INTERSECTION_M = 50.0


def finite_number(value):
    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Nonfinite numeric value")

    return number


def load_poses():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    poses = []
    skipped = {}

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            matched = str(
                row.get("pose_matched", "")
            ).strip().lower()

            if matched not in {
                "true",
                "1",
                "yes",
            }:
                skipped["pose_unmatched"] = (
                    skipped.get("pose_unmatched", 0) + 1
                )
                continue

            try:
                position = np.array(
                    [
                        finite_number(row[key])
                        for key in ("tx", "ty", "tz")
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        finite_number(row[key])
                        for key in ("qx", "qy", "qz", "qw")
                    ],
                    dtype=np.float64,
                )

                quaternion_norm = float(
                    np.linalg.norm(quaternion)
                )

                if not (
                    0.99 <= quaternion_norm <= 1.01
                ):
                    raise ValueError(
                        "Invalid quaternion norm"
                    )

                rotation = Rotation.from_quat(
                    quaternion
                ).as_matrix()

                poses.append(
                    {
                        "row": row_number,
                        "filename": row.get(
                            "filename",
                            f"row_{row_number}",
                        ),
                        "position": position,
                        "quaternion": quaternion,
                        "quaternion_norm": (
                            quaternion_norm
                        ),
                        "rotation": rotation,
                    }
                )

            except (
                KeyError,
                TypeError,
                ValueError,
            ):
                skipped["invalid_pose"] = (
                    skipped.get("invalid_pose", 0) + 1
                )

    if len(poses) < 2:
        raise RuntimeError(
            "Need at least two valid matched poses."
        )

    return poses, skipped


def rotation_metrics(matrix):
    identity = np.eye(3)

    error = float(
        np.linalg.norm(
            matrix.T @ matrix - identity,
            ord="fro",
        )
    )

    determinant = float(
        np.linalg.det(matrix)
    )

    return {
        "orthogonality_error": error,
        "determinant": determinant,
        "proper_rotation": bool(
            error < 1e-5
            and abs(determinant - 1.0) < 1e-5
        ),
    }


def summarize(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return None

    return {
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": float(np.max(values)),
        "p95": float(
            np.percentile(values, 95)
        ),
    }


def audit_calibration():
    focal_lengths = [
        float(K[0, 0]),
        float(K[1, 1]),
    ]

    principal_point = [
        float(K[0, 2]),
        float(K[1, 2]),
    ]

    extrinsic_rotation = T_CAM_IMU[:3, :3]

    extrinsic_metrics = rotation_metrics(
        extrinsic_rotation
    )

    yaml_exists = CALIBRATION.exists()

    return {
        "calibration_file": str(CALIBRATION),
        "calibration_file_exists": yaml_exists,
        "reference_values_cross_checked_with_yaml": False,
        "image_resolution": [
            IMAGE_WIDTH,
            IMAGE_HEIGHT,
        ],
        "focal_lengths_px": focal_lengths,
        "principal_point_px": principal_point,
        "principal_point_inside_image": bool(
            0 <= principal_point[0] < IMAGE_WIDTH
            and 0 <= principal_point[1] < IMAGE_HEIGHT
        ),
        "distortion_model": "equidistant",
        "distortion_coefficients": D.tolist(),
        "t_cam_imu": {
            "matrix": T_CAM_IMU.tolist(),
            "rotation": extrinsic_metrics,
            "translation_norm_m": float(
                np.linalg.norm(
                    T_CAM_IMU[:3, 3]
                )
            ),
            "homogeneous_last_row_valid": bool(
                np.allclose(
                    T_CAM_IMU[3],
                    [0, 0, 0, 1],
                    atol=1e-10,
                )
            ),
        },
    }


def audit_pose_rotations(poses):
    metrics = [
        rotation_metrics(pose["rotation"])
        for pose in poses
    ]

    return {
        "count": len(metrics),
        "invalid_rotation_count": sum(
            not item["proper_rotation"]
            for item in metrics
        ),
        "orthogonality_error": summarize(
            [
                item["orthogonality_error"]
                for item in metrics
            ]
        ),
        "determinant": summarize(
            [
                item["determinant"]
                for item in metrics
            ]
        ),
        "quaternion_norm": summarize(
            [
                pose["quaternion_norm"]
                for pose in poses
            ]
        ),
    }


def audit_trajectory(poses):
    positions = np.array(
        [
            pose["position"]
            for pose in poses
        ]
    )

    displacements = np.diff(
        positions,
        axis=0,
    )

    step_distances = np.linalg.norm(
        displacements,
        axis=1,
    )

    angular_steps = []
    quaternion_dots = []

    for previous, current in zip(
        poses[:-1],
        poses[1:],
    ):
        relative_rotation = (
            previous["rotation"].T
            @ current["rotation"]
        )

        angle = Rotation.from_matrix(
            relative_rotation
        ).magnitude()

        angular_steps.append(
            math.degrees(angle)
        )

        quaternion_dots.append(
            float(
                np.dot(
                    previous["quaternion"],
                    current["quaternion"],
                )
            )
        )

    return {
        "pose_count": len(poses),
        "path_length_m": float(
            np.sum(step_distances)
        ),
        "net_displacement_m": float(
            np.linalg.norm(
                positions[-1] - positions[0]
            )
        ),
        "position_bounds_m": {
            "min": positions.min(
                axis=0
            ).tolist(),
            "max": positions.max(
                axis=0
            ).tolist(),
        },
        "consecutive_translation_m": summarize(
            step_distances
        ),
        "consecutive_rotation_deg": summarize(
            angular_steps
        ),
        "negative_quaternion_dot_count": sum(
            value < 0
            for value in quaternion_dots
        ),
        "note": (
            "Quaternion sign changes can represent "
            "the same physical rotation. "
            "Large steps are diagnostic only because "
            "timestamps are not evaluated here."
        ),
    }


def optical_axis_for_pose(pose, convention):
    """
    Convention A:
        R is camera-to-map.
        Optical axis in map = R @ [0,0,1].

    Convention B:
        R is map-to-camera.
        Optical axis in map = R.T @ [0,0,1].

    Neither convention is selected by this audit.
    """

    axis_camera = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    rotation = pose["rotation"]

    if convention == "camera_to_map":
        return rotation @ axis_camera

    if convention == "map_to_camera":
        return rotation.T @ axis_camera

    raise ValueError(convention)


def floor_intersection(pose, axis):
    origin = pose["position"]

    if axis[2] >= -MIN_DOWNWARD_RAY_Z:
        return None

    distance = (
        ASSUMED_FLOOR_Z - origin[2]
    ) / axis[2]

    if not (
        0 < distance < MAX_FLOOR_INTERSECTION_M
    ):
        return None

    point = origin + distance * axis

    return {
        "distance_m": float(distance),
        "point_map": point.tolist(),
    }


def audit_convention(poses, convention):
    axes = np.array(
        [
            optical_axis_for_pose(
                pose,
                convention,
            )
            for pose in poses
        ]
    )

    downward = axes[:, 2] < (
        -MIN_DOWNWARD_RAY_Z
    )

    upward = axes[:, 2] > (
        MIN_DOWNWARD_RAY_Z
    )

    horizontal = ~(
        downward | upward
    )

    intersections = []

    for pose, axis in zip(
        poses,
        axes,
    ):
        intersection = floor_intersection(
            pose,
            axis,
        )

        if intersection is not None:
            intersections.append(
                intersection
            )

    return {
        "convention": convention,
        "optical_axis_z": summarize(
            axes[:, 2]
        ),
        "downward_fraction": float(
            np.mean(downward)
        ),
        "upward_fraction": float(
            np.mean(upward)
        ),
        "near_horizontal_fraction": float(
            np.mean(horizontal)
        ),
        "assumed_floor_z_m": ASSUMED_FLOOR_Z,
        "floor_intersection_count": len(
            intersections
        ),
        "floor_intersection_fraction": float(
            len(intersections) / len(poses)
        ),
        "floor_intersection_distance_m": (
            summarize(
                [
                    item["distance_m"]
                    for item in intersections
                ]
            )
        ),
        "warning": (
            "The assumed floor elevation and camera "
            "orientation are not independently verified. "
            "This is not a convention-selection score."
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    poses, skipped = load_poses()

    report = {
        "sequence": SEQUENCE,
        "method": "camera_pose_geometry_audit",
        "registration_verified": False,
        "camera_pose_convention_verified": False,
        "cad_alignment_verified": False,
        "construction_progress_verified": False,
        "manifest": {
            "path": str(MANIFEST),
            "valid_matched_poses": len(poses),
            "skipped": skipped,
        },
        "calibration": audit_calibration(),
        "pose_rotations": audit_pose_rotations(
            poses
        ),
        "trajectory": audit_trajectory(
            poses
        ),
        "pose_conventions": {
            "camera_to_map": audit_convention(
                poses,
                "camera_to_map",
            ),
            "map_to_camera": audit_convention(
                poses,
                "map_to_camera",
            ),
        },
        "decision": (
            "GEOMETRY_AUDITED_CONVENTION_UNVERIFIED"
        ),
        "limitations": [
            (
                "No authoritative trajectory frame "
                "metadata is parsed."
            ),
            (
                "Official YAML values are referenced "
                "but not parsed or compared."
            ),
            (
                "Floor z=0 is assumed for diagnostics."
            ),
            (
                "No CAD or image-edge agreement "
                "is evaluated."
            ),
            (
                "Camera-to-IMU extrinsic is audited "
                "but not composed with trajectory poses."
            ),
        ],
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== CAMERA POSE GEOMETRY AUDIT ==="
    )

    print(
        "Valid matched poses:",
        len(poses),
    )

    print(
        "Invalid rotations:",
        report["pose_rotations"][
            "invalid_rotation_count"
        ],
    )

    print(
        "Trajectory path length:",
        round(
            report["trajectory"][
                "path_length_m"
            ],
            3,
        ),
        "m",
    )

    for name, result in report[
        "pose_conventions"
    ].items():
        print(
            name,
            "| downward:",
            round(
                result["downward_fraction"],
                3,
            ),
            "| floor intersections:",
            result["floor_intersection_count"],
        )

    print(
        "Pose convention verified: NO"
    )

    print(
        "CAD registration verified: NO"
    )

    print(
        "Report:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
