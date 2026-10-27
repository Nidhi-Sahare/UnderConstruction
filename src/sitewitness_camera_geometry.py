
"""
SiteWitness — Calibrated Camera Geometry
BuildProgress / OpenCV 5

Run:
    python -m src.sitewitness_camera_geometry

Reads:
    data/hilti_calibration/kalibr_imucam_chain.yaml

Writes:
    outputs/sitewitness/camera_geometry_report.json

The calibration defines T_cam_imu, mapping IMU-frame
coordinates into camera-frame coordinates.

World pose conventions are NOT assumed or verified.
CAD-to-world alignment is NOT verified.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_PATH = (
    ROOT / "data" / "hilti_calibration"
    / "kalibr_imucam_chain.yaml"
)

OUTPUT_PATH = (
    ROOT / "outputs" / "sitewitness"
    / "camera_geometry_report.json"
)


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    T_cam_imu: np.ndarray
    timeshift_cam_imu: float

    @property
    def T_imu_cam(self) -> np.ndarray:
        return np.linalg.inv(self.T_cam_imu)


def read_camera_block(text: str, name: str) -> str:
    """
    Extract an active top-level Kalibr camera block.
    Ignore commented-out EUCM alternatives.
    """
    lines = text.splitlines()
    active = [
        line for line in lines
        if not line.lstrip().startswith("#")
    ]

    start = None

    for index, line in enumerate(active):
        if line.strip() == f"{name}:" and line == line.lstrip():
            start = index + 1
            break

    if start is None:
        raise ValueError(f"Active camera {name} not found")

    result = []

    for line in active[start:]:
        if (
            line.strip()
            and line == line.lstrip()
            and re.match(r"^[A-Za-z_]\w*:", line)
        ):
            break
        result.append(line)

    return "\n".join(result)


def parse_inline_array(block: str, field: str):
    match = re.search(
        rf"^\s+{re.escape(field)}:\s*(\[[^\n]+\])",
        block,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Missing {field}")

    return ast.literal_eval(match.group(1))


def parse_scalar(block: str, field: str) -> str:
    match = re.search(
        rf"^\s+{re.escape(field)}:\s*([^\n#]+)",
        block,
        re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Missing {field}")
    return match.group(1).strip()


def parse_transform(block: str, field: str) -> np.ndarray:
    lines = block.splitlines()

    start = None
    indentation = None

    for index, line in enumerate(lines):
        if line.strip() == f"{field}:":
            start = index + 1
            indentation = len(line) - len(line.lstrip())
            break

    if start is None:
        raise ValueError(f"Missing transform {field}")

    rows = []

    for line in lines[start:]:
        if not line.strip():
            continue

        current_indent = len(line) - len(line.lstrip())

        if current_indent <= indentation:
            break

        value = line.strip()

        if value.startswith("- ["):
            rows.append(ast.literal_eval(value[2:].strip()))

    matrix = np.asarray(rows, dtype=np.float64)

    if matrix.shape != (4, 4):
        raise ValueError(
            f"{field} must be 4x4, got {matrix.shape}"
        )

    return matrix


def load_calibration(name: str = "cam0") -> CameraCalibration:
    text = CALIBRATION_PATH.read_text(encoding="utf-8")
    block = read_camera_block(text, name)

    camera_model = parse_scalar(block, "camera_model")
    distortion_model = parse_scalar(block, "distortion_model")

    if camera_model != "pinhole":
        raise ValueError(
            f"Unsupported camera model: {camera_model}"
        )

    if distortion_model != "equidistant":
        raise ValueError(
            f"Unsupported distortion model: {distortion_model}"
        )

    fx, fy, cx, cy = map(
        float, parse_inline_array(block, "intrinsics")
    )

    width, height = map(
        int, parse_inline_array(block, "resolution")
    )

    distortion = np.asarray(
        parse_inline_array(block, "distortion_coeffs"),
        dtype=np.float64,
    ).reshape(4, 1)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    transform = parse_transform(block, "T_cam_imu")

    timeshift = float(
        parse_scalar(block, "timeshift_cam_imu")
    )

    return CameraCalibration(
        name=name,
        width=width,
        height=height,
        K=K,
        D=distortion,
        T_cam_imu=transform,
        timeshift_cam_imu=timeshift,
    )


def validate_calibration(cam: CameraCalibration) -> dict:
    T = cam.T_cam_imu
    R = T[:3, :3]

    orthogonality_error = float(
        np.linalg.norm(R.T @ R - np.eye(3))
    )
    determinant = float(np.linalg.det(R))

    bottom_row_error = float(
        np.linalg.norm(T[3] - [0, 0, 0, 1])
    )

    focal_positive = bool(
        cam.K[0, 0] > 0 and cam.K[1, 1] > 0
    )

    principal_inside = bool(
        0 <= cam.K[0, 2] < cam.width
        and 0 <= cam.K[1, 2] < cam.height
    )

    inverse_error = float(np.linalg.norm(
        T @ cam.T_imu_cam - np.eye(4)
    ))

    checks = {
        "matrix_finite": bool(np.isfinite(T).all()),
        "rotation_orthonormal": orthogonality_error < 1e-4,
        "rotation_determinant_positive": abs(determinant - 1) < 1e-4,
        "homogeneous_bottom_row": bottom_row_error < 1e-8,
        "inverse_consistent": inverse_error < 1e-8,
        "positive_focal_lengths": focal_positive,
        "principal_point_inside_image": principal_inside,
        "distortion_finite": bool(np.isfinite(cam.D).all()),
        "resolution_positive": cam.width > 0 and cam.height > 0,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "rotation_determinant": determinant,
        "rotation_orthogonality_error": orthogonality_error,
        "inverse_error": inverse_error,
        "camera_translation_from_imu_m": T[:3, 3].tolist(),
    }


def transform_points(
    points: np.ndarray,
    transform: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Expected Nx3 points")

    if transform.shape != (4, 4):
        raise ValueError("Expected 4x4 transform")

    return (
        points @ transform[:3, :3].T
        + transform[:3, 3]
    )


def project_camera_points(
    cam: CameraCalibration,
    points_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project camera-frame 3D points into the raw
    distorted fisheye image.

    Returns:
        pixels: Nx2; NaN for invalid points
        visible: boolean mask inside image with Z > 0
    """
    points = np.asarray(
        points_camera, dtype=np.float64
    ).reshape(-1, 3)

    pixels = np.full(
        (len(points), 2), np.nan, dtype=np.float64
    )
    visible = np.zeros(len(points), dtype=bool)

    valid = (
        np.isfinite(points).all(axis=1)
        & (points[:, 2] > 1e-6)
    )

    if not np.any(valid):
        return pixels, visible

    projected, _ = cv2.fisheye.projectPoints(
        np.ascontiguousarray(
            points[valid].reshape(-1, 1, 3)
        ),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        cam.K,
        cam.D,
    )

    uv = projected.reshape(-1, 2)
    pixels[valid] = uv

    inside = (
        np.isfinite(uv).all(axis=1)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < cam.width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < cam.height)
    )

    visible[np.flatnonzero(valid)] = inside
    return pixels, visible


def project_imu_points(
    cam: CameraCalibration,
    points_imu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform IMU-frame points to camera coordinates
    using the documented Kalibr T_cam_imu.
    """
    points_camera = transform_points(
        points_imu, cam.T_cam_imu
    )
    return project_camera_points(cam, points_camera)


def quaternion_xyzw_to_rotation(quaternion) -> np.ndarray:
    """
    Quaternion arithmetic only.

    The caller must separately establish what
    coordinate frames the quaternion relates.
    """
    q = np.asarray(quaternion, dtype=np.float64)

    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Expected finite xyzw quaternion")

    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Zero quaternion")

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


def world_to_imu(
    points_world: np.ndarray,
    position,
    quaternion_xyzw,
    convention: str,
) -> np.ndarray:
    """
    Two EXPLICIT possible pose interpretations.

    imu_to_world:
        p_world = R_world_imu @ p_imu + position

    world_to_imu:
        p_imu = R_imu_world @ p_world + position

    No convention is selected automatically.
    """
    points = np.asarray(
        points_world, dtype=np.float64
    ).reshape(-1, 3)

    position = np.asarray(
        position, dtype=np.float64
    ).reshape(3)

    R = quaternion_xyzw_to_rotation(quaternion_xyzw)

    if convention == "imu_to_world":
        return (points - position) @ R

    if convention == "world_to_imu":
        return points @ R.T + position

    raise ValueError(
        "convention must be 'imu_to_world' "
        "or 'world_to_imu'"
    )


def project_world_points(
    cam: CameraCalibration,
    points_world: np.ndarray,
    position,
    quaternion_xyzw,
    convention: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exploratory projection with an explicit pose
    interpretation.

    Does NOT verify that the supplied world frame
    is aligned to the architectural CAD frame.
    """
    points_imu = world_to_imu(
        points_world,
        position,
        quaternion_xyzw,
        convention,
    )

    return project_imu_points(cam, points_imu)


def run_tests(cam: CameraCalibration) -> dict:
    validation = validate_calibration(cam)

    if not validation["passed"]:
        raise RuntimeError(
            "Camera calibration validation failed: "
            + str(validation["checks"])
        )

    # A point on the optical axis should project
    # to the calibrated principal point.
    axis = np.array([[0.0, 0.0, 3.0]])
    uv, visible = project_camera_points(cam, axis)

    principal_error = float(np.linalg.norm(
        uv[0] - cam.K[:2, 2]
    ))

    # Inverting the IMU-to-camera transform should
    # recover the same point in the camera image.
    imu_axis = transform_points(
        axis, cam.T_imu_cam
    )

    uv_imu, visible_imu = project_imu_points(
        cam, imu_axis
    )

    roundtrip_error = float(np.linalg.norm(
        uv - uv_imu
    ))

    # A point behind the camera is never visible.
    behind_uv, behind_visible = project_camera_points(
        cam, np.array([[0.0, 0.0, -2.0]])
    )

    # Identity pose should preserve point coordinates
    # for both explicit pose conventions.
    test_point = np.array([[1.0, 2.0, 3.0]])
    identity_quat = [0, 0, 0, 1]

    identity_a = world_to_imu(
        test_point, [0, 0, 0],
        identity_quat, "imu_to_world"
    )
    identity_b = world_to_imu(
        test_point, [0, 0, 0],
        identity_quat, "world_to_imu"
    )

    tests = {
        "principal_point_projection": (
            principal_error < 1e-6
            and bool(visible[0])
        ),
        "extrinsic_inverse_roundtrip": (
            roundtrip_error < 1e-6
            and bool(visible_imu[0])
        ),
        "reject_point_behind_camera": (
            not bool(behind_visible[0])
        ),
        "identity_pose_conventions": bool(
            np.allclose(identity_a, test_point)
            and np.allclose(identity_b, test_point)
        ),
    }

    return {
        "passed": all(tests.values()),
        "tests": tests,
        "principal_point_error_px": principal_error,
        "extrinsic_roundtrip_error_px": roundtrip_error,
    }


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "module": "SiteWitness Camera Geometry",
        "opencv_version": cv2.__version__,
        "calibration_source": str(
            CALIBRATION_PATH.relative_to(ROOT)
        ),
        "cameras": {},
        "pose_convention_verified": False,
        "cad_registration_verified": False,
    }

    for name in ("cam0", "cam1"):
        cam = load_calibration(name)
        validation = validate_calibration(cam)
        tests = run_tests(cam)

        report["cameras"][name] = {
            "resolution": [
                cam.width, cam.height
            ],
            "camera_model": "pinhole",
            "distortion_model": "equidistant",
            "K": cam.K.tolist(),
            "D": cam.D.reshape(-1).tolist(),
            "T_cam_imu": cam.T_cam_imu.tolist(),
            "T_imu_cam": cam.T_imu_cam.tolist(),
            "timeshift_cam_imu_s": (
                cam.timeshift_cam_imu
            ),
            "validation": validation,
            "tests": tests,
        }

    report["all_tests_passed"] = all(
        item["tests"]["passed"]
        for item in report["cameras"].values()
    )

    report["next_action"] = {
        "type": "AUDIT_TRAJECTORY_POSE_CONVENTION",
        "reason": (
            "Camera calibration is available, but "
            "trajectory pose direction, timestamp "
            "association and CAD-to-world registration "
            "must be independently established."
        ),
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSITEWITNESS — CAMERA GEOMETRY")
    print("=" * 52)
    print("OpenCV:", cv2.__version__)

    for name, item in report["cameras"].items():
        print(
            name,
            "calibration:",
            "PASS" if item["validation"]["passed"]
            else "FAIL",
        )
        print(
            name,
            "projection tests:",
            "PASS" if item["tests"]["passed"]
            else "FAIL",
        )

    print(
        "Overall:",
        "PASS" if report["all_tests_passed"]
        else "FAIL",
    )
    print("Trajectory convention: UNVERIFIED")
    print("CAD registration: UNVERIFIED")
    print("Report:", OUTPUT_PATH)

    if not report["all_tests_passed"]:
        raise SystemExit(1)

    return report


if __name__ == "__main__":
    main()
