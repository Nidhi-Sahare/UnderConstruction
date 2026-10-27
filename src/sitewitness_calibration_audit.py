
"""
SiteWitness — Camera Calibration & Pose Audit
BuildProgress / OpenCV 5

Run from the BuildProgress project root:

    python -m src.sitewitness_calibration_audit

Purpose:
    Discover calibration and pose metadata.
    Validate available numerical calibration.
    Audit pose orientation and timestamp structure.
    Check actual image dimensions.
    Check Floor 2 DXF units and extents.
    Determine whether CAD-to-image projection is ready.

Safety:
    Read-only input processing.
    No dashboard modifications.
    No invented camera intrinsics or extrinsics.
    No automatic claims of verified registration.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "sitewitness"
OUTPUT_FILE = OUTPUT / "calibration_audit.json"

FRAME_DIR = (
    ROOT / "sample_gallery" / "real_floor_2" / "frames"
)

TRAJECTORY_CANDIDATES = [
    ROOT / "data" / "real_samples" / "floor_2_camera_trajectory.json",
    ROOT / "sample_gallery" / "real_floor_2" / "trajectory.json",
]

DXF_FILE = ROOT / "data" / "floorplans" / "floor_2.dxf"

SEARCH_ROOTS = [
    ROOT / "data",
    ROOT / "sample_gallery",
    ROOT / "config",
    ROOT / "calibration",
    ROOT / "outputs",
]

CALIBRATION_TERMS = (
    "calib",
    "intrinsic",
    "extrinsic",
    "camera_info",
    "camchain",
    "sensor",
    "imu",
)

SUPPORTED_METADATA = {
    ".json", ".yaml", ".yml", ".txt", ".toml"
}

MAX_METADATA_FILES = 150
MAX_METADATA_BYTES = 2_000_000
MAX_DISCOVERY_DEPTH = 5


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_frames() -> dict:
    if not FRAME_DIR.is_dir():
        return {
            "status": "MISSING",
            "directory": relative(FRAME_DIR),
        }

    files = sorted(
        p for p in FRAME_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    )

    if not files:
        return {
            "status": "EMPTY",
            "directory": relative(FRAME_DIR),
        }

    sample_indices = sorted(set([
        0,
        len(files) // 4,
        len(files) // 2,
        (3 * len(files)) // 4,
        len(files) - 1,
    ]))

    dimensions = Counter()
    samples = []

    for index in sample_indices:
        path = files[index]
        image = cv2.imread(str(path))

        if image is None:
            samples.append({
                "frame": relative(path),
                "readable": False,
            })
            continue

        height, width = image.shape[:2]
        dimensions[f"{width}x{height}"] += 1

        samples.append({
            "frame": relative(path),
            "readable": True,
            "width": width,
            "height": height,
            "channels": (
                image.shape[2]
                if image.ndim == 3 else 1
            ),
        })

    return {
        "status": "INSPECTED",
        "frame_count": len(files),
        "sampled_frame_count": len(samples),
        "sampled_dimensions": dict(dimensions),
        "samples": samples,
        "all_frames_same_dimensions": (
            len(dimensions) == 1
            and sum(dimensions.values()) == len(samples)
        ),
    }


def discover_calibration_files() -> list[dict]:
    found = []
    seen = set()

    for search_root in SEARCH_ROOTS:
        if not search_root.is_dir():
            continue

        for path in search_root.rglob("*"):
            if len(found) >= MAX_METADATA_FILES:
                break

            if not path.is_file():
                continue

            try:
                depth = len(
                    path.relative_to(search_root).parts
                )
            except ValueError:
                continue

            if depth > MAX_DISCOVERY_DEPTH:
                continue

            if path.suffix.lower() not in SUPPORTED_METADATA:
                continue

            if path.stat().st_size > MAX_METADATA_BYTES:
                continue

            name = path.name.lower()
            if not any(
                term in name
                for term in CALIBRATION_TERMS
            ):
                continue

            resolved = str(path.resolve()).lower()

            if resolved in seen:
                continue

            seen.add(resolved)

            found.append({
                "path": relative(path),
                "size_bytes": path.stat().st_size,
                "type": path.suffix.lower(),
                "status": "DISCOVERED_NOT_VALIDATED",
            })

    return found


def walk_keys(obj, prefix="", depth=0, limit=7):
    if depth > limit:
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = (
                f"{prefix}.{key}"
                if prefix else str(key)
            )
            yield name, value
            yield from walk_keys(
                value, name, depth + 1, limit
            )

    elif isinstance(obj, list) and obj:
        # Inspect the first few elements only.
        for index, value in enumerate(obj[:3]):
            if isinstance(value, (dict, list)):
                yield from walk_keys(
                    value,
                    f"{prefix}[{index}]",
                    depth + 1,
                    limit,
                )


def find_calibration_fields(obj) -> list[dict]:
    patterns = (
        "intrinsic", "extrinsic", "distortion",
        "camera_matrix", "projection_matrix",
        "camera_model", "cam0", "t_cam",
        "t_imu", "focal", "principal_point",
        "fx", "fy", "cx", "cy",
    )

    matches = []

    for path, value in walk_keys(obj):
        leaf = path.split(".")[-1].lower()

        if any(term in leaf for term in patterns):
            matches.append({
                "field": path,
                "value_preview": (
                    str(value)[:250]
                ),
            })

    return matches[:80]


def inspect_trajectory() -> dict:
    path = next(
        (
            candidate
            for candidate in TRAJECTORY_CANDIDATES
            if candidate.is_file()
        ),
        None,
    )

    if path is None:
        return {
            "status": "MISSING",
            "searched": [
                relative(p)
                for p in TRAJECTORY_CANDIDATES
            ],
        }

    data = load_json(path)

    result = {
        "status": "FOUND",
        "path": relative(path),
        "root_type": type(data).__name__,
        "root_keys": (
            list(data.keys())
            if isinstance(data, dict)
            else []
        ),
        "calibration_related_fields": (
            find_calibration_fields(data)
        ),
    }

    if not isinstance(data, dict):
        result["pose_status"] = (
            "UNRECOGNIZED_TRAJECTORY_FORMAT"
        )
        return result

    possible_pose_keys = (
        "poses",
        "trajectory",
        "frames",
        "samples",
        "positions",
    )

    pose_key = None
    poses = None

    for key in possible_pose_keys:
        value = data.get(key)
        if isinstance(value, list) and value:
            pose_key = key
            poses = value
            break

    if poses is None:
        result["pose_status"] = (
            "POSE_ARRAY_NOT_FOUND"
        )
        return result

    result["pose_key"] = pose_key
    result["pose_count"] = len(poses)

    first = poses[0]
    result["first_pose_preview"] = (
        first if isinstance(first, dict)
        else str(first)[:500]
    )

    if not isinstance(first, dict):
        result["pose_status"] = (
            "POSE_SCHEMA_REQUIRES_REVIEW"
        )
        return result

    position_keys = [
        "position",
        "translation",
        "xyz",
        "location",
    ]

    orientation_keys = [
        "orientation_xyzw",
        "orientation",
        "quaternion",
        "rotation",
    ]

    position_field = next(
        (
            key for key in position_keys
            if key in first
        ),
        None,
    )

    orientation_field = next(
        (
            key for key in orientation_keys
            if key in first
        ),
        None,
    )

    result["position_field"] = position_field
    result["orientation_field"] = orientation_field

    valid_positions = 0
    valid_quaternions = 0
    quaternion_norms = []
    missing_positions = 0
    missing_orientations = 0

    for pose in poses:
        if not isinstance(pose, dict):
            continue

        position = (
            pose.get(position_field)
            if position_field else None
        )

        if isinstance(position, dict):
            position = [
                position.get("x"),
                position.get("y"),
                position.get("z"),
            ]

        if isinstance(position, (list, tuple)):
            try:
                xyz = np.asarray(
                    position, dtype=float
                )
                if (
                    xyz.shape == (3,)
                    and np.isfinite(xyz).all()
                ):
                    valid_positions += 1
                else:
                    missing_positions += 1
            except (TypeError, ValueError):
                missing_positions += 1
        else:
            missing_positions += 1

        orientation = (
            pose.get(orientation_field)
            if orientation_field else None
        )

        if isinstance(orientation, dict):
            orientation = [
                orientation.get("x"),
                orientation.get("y"),
                orientation.get("z"),
                orientation.get("w"),
            ]

        if isinstance(orientation, (list, tuple)):
            try:
                quat = np.asarray(
                    orientation, dtype=float
                )
                if (
                    quat.shape == (4,)
                    and np.isfinite(quat).all()
                ):
                    norm = float(
                        np.linalg.norm(quat)
                    )
                    quaternion_norms.append(norm)

                    if abs(norm - 1.0) < 0.02:
                        valid_quaternions += 1
                else:
                    missing_orientations += 1
            except (TypeError, ValueError):
                missing_orientations += 1
        else:
            missing_orientations += 1

    result["pose_status"] = "STRUCTURE_INSPECTED"
    result["valid_positions"] = valid_positions
    result["valid_unit_quaternions"] = valid_quaternions
    result["missing_or_invalid_positions"] = (
        missing_positions
    )
    result["missing_or_invalid_orientations"] = (
        missing_orientations
    )

    if quaternion_norms:
        result["quaternion_norm_range"] = [
            min(quaternion_norms),
            max(quaternion_norms),
        ]

    result["pose_convention_verified"] = False
    result["sensor_to_camera_verified"] = False

    return result


def inspect_dxf() -> dict:
    if not DXF_FILE.is_file():
        return {
            "status": "MISSING",
            "path": relative(DXF_FILE),
        }

    try:
        import ezdxf
    except ImportError:
        return {
            "status": "EZDXF_NOT_INSTALLED",
            "path": relative(DXF_FILE),
        }

    try:
        document = ezdxf.readfile(str(DXF_FILE))
        modelspace = document.modelspace()

        units = int(
            document.header.get("$INSUNITS", 0)
        )

        counts = Counter(
            entity.dxftype()
            for entity in modelspace
        )

        return {
            "status": "INSPECTED",
            "path": relative(DXF_FILE),
            "insunits_code": units,
            "units_declared_meters": units == 6,
            "entity_count": sum(counts.values()),
            "entity_types": dict(counts),
            "cad_to_reference_verified": False,
        }

    except Exception as exc:
        return {
            "status": "READ_ERROR",
            "path": relative(DXF_FILE),
            "error": str(exc),
        }


def inspect_alignment() -> dict:
    candidate_file = (
        ROOT
        / "_archive"
        / "hilti_transition_20261023_004237"
        / "outputs"
        / "registration"
        / "candidate_alignments.json"
    )

    if not candidate_file.is_file():
        return {
            "status": "MISSING",
            "path": relative(candidate_file),
        }

    data = load_json(candidate_file)
    candidates = data.get("candidates", [])

    return {
        "status": "FOUND",
        "candidate_count": len(candidates),
        "registration_verified": False,
        "note": (
            "Candidate containment and wall-consistency "
            "scores do not independently validate "
            "camera-to-CAD registration."
        ),
    }


def projection_readiness(
    frames: dict,
    trajectory: dict,
    calibration_files: list,
    dxf: dict,
    alignment: dict,
) -> dict:
    checks = {
        "image_dimensions_available": (
            frames.get("status") == "INSPECTED"
            and bool(frames.get("sampled_dimensions"))
        ),
        "trajectory_found": (
            trajectory.get("status") == "FOUND"
        ),
        "pose_positions_parsed": (
            trajectory.get("valid_positions", 0) > 0
        ),
        "pose_orientations_parsed": (
            trajectory.get("valid_unit_quaternions", 0) > 0
        ),
        "camera_intrinsics_verified": False,
        "distortion_model_verified": False,
        "sensor_to_camera_extrinsics_verified": False,
        "pose_direction_verified": False,
        "cad_to_reference_registration_verified": False,
        "cad_units_declared_meters": (
            dxf.get("units_declared_meters", False)
        ),
    }

    blockers = [
        key for key, value in checks.items()
        if not value
    ]

    return {
        "ready_for_verified_projection": False,
        "checks": checks,
        "blockers": blockers,
        "calibration_file_candidates_found": len(
            calibration_files
        ),
        "explanation": (
            "A calibration filename or a valid quaternion "
            "does not establish the transform convention. "
            "The audit intentionally requires explicit "
            "camera calibration, extrinsics, and independent "
            "CAD registration validation."
        ),
    }


def main() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    frames = inspect_frames()
    calibration_files = discover_calibration_files()
    trajectory = inspect_trajectory()
    dxf = inspect_dxf()
    alignment = inspect_alignment()

    readiness = projection_readiness(
        frames,
        trajectory,
        calibration_files,
        dxf,
        alignment,
    )

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "BuildProgress",
        "module": "SiteWitness",
        "opencv_version": cv2.__version__,
        "frames": frames,
        "trajectory": trajectory,
        "calibration_file_candidates": (
            calibration_files
        ),
        "cad": dxf,
        "registration": alignment,
        "projection_readiness": readiness,
        "next_action": {
            "type": "RESOLVE_PROJECTION_INPUTS",
            "required": readiness["blockers"],
        },
    }

    OUTPUT_FILE.write_text(
        json.dumps(
            json_safe(report),
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSITEWITNESS — CALIBRATION AUDIT")
    print("=" * 55)

    print("OpenCV:", cv2.__version__)
    print(
        "Frames:",
        frames.get("frame_count", 0),
    )
    print(
        "Trajectory:",
        trajectory.get("status"),
    )
    print(
        "Parsed positions:",
        trajectory.get("valid_positions", 0),
    )
    print(
        "Parsed quaternions:",
        trajectory.get("valid_unit_quaternions", 0),
    )
    print(
        "Calibration file candidates:",
        len(calibration_files),
    )
    print(
        "DXF:",
        dxf.get("status"),
    )
    print(
        "Projection ready:",
        readiness["ready_for_verified_projection"],
    )

    print("\nBLOCKERS:")
    for blocker in readiness["blockers"]:
        print(" -", blocker)

    print("\nReport:", OUTPUT_FILE)

    return report


if __name__ == "__main__":
    main()
