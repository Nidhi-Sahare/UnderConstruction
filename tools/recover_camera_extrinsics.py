
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "camera_extrinsics"
OUTPUT_PATH = OUTPUT_DIR / "extrinsics_audit.json"

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".idea",
}

EXTENSIONS = {".yaml", ".yml", ".json"}

MATRIX_KEYS = {
    "T_cam_imu",
    "T_imu_cam",
    "T_cam_lidar",
    "T_lidar_cam",
    "T_cam_body",
    "T_body_cam",
}

CAMERA_KEYS = {
    "camera_model",
    "distortion_model",
    "intrinsics",
    "distortion_coeffs",
    "distortion_coefficients",
    "resolution",
    "cam_overlaps",
}


def parse_document(path):
    try:
        text = path.read_text(
            encoding="utf-8-sig"
        )
    except (UnicodeError, OSError):
        return None

    if path.suffix.lower() == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    if yaml is None:
        return None

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return None


def find_matrices(obj, location="$"):
    results = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            current = f"{location}.{key}"

            if key in MATRIX_KEYS:
                results.append({
                    "key": key,
                    "location": current,
                    "value": value,
                })

            results.extend(
                find_matrices(value, current)
            )

    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            results.extend(
                find_matrices(
                    value,
                    f"{location}[{index}]",
                )
            )

    return results


def validate_transform(value):
    try:
        matrix = np.asarray(
            value,
            dtype=np.float64,
        )
    except (ValueError, TypeError):
        return {
            "valid": False,
            "reason": "NON_NUMERIC_MATRIX",
        }

    if matrix.shape != (4, 4):
        return {
            "valid": False,
            "reason": "NOT_4X4",
        }

    if not np.isfinite(matrix).all():
        return {
            "valid": False,
            "reason": "NON_FINITE_VALUES",
        }

    rotation = matrix[:3, :3]

    orthogonality_error = float(
        np.linalg.norm(
            rotation.T @ rotation - np.eye(3)
        )
    )

    determinant = float(
        np.linalg.det(rotation)
    )

    bottom_row_error = float(
        np.linalg.norm(
            matrix[3] - np.array([0, 0, 0, 1])
        )
    )

    valid = (
        orthogonality_error < 1e-3
        and abs(determinant - 1.0) < 1e-3
        and bottom_row_error < 1e-6
    )

    return {
        "valid": valid,
        "orthogonality_error":
            round(orthogonality_error, 9),
        "rotation_determinant":
            round(determinant, 9),
        "bottom_row_error":
            round(bottom_row_error, 9),
        "translation_m": [
            float(v)
            for v in matrix[:3, 3]
        ],
    }


def camera_metadata(obj):
    if not isinstance(obj, dict):
        return {}

    found = {}

    for key in CAMERA_KEYS:
        if key in obj:
            found[key] = obj[key]

    for key, value in obj.items():
        if isinstance(value, dict):
            nested = camera_metadata(value)

            if nested:
                found[key] = nested

    return found


def relevant_paths():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        if any(
            part in SKIP_DIRS
            for part in path.relative_to(ROOT).parts
        ):
            continue

        if path.suffix.lower() not in EXTENSIONS:
            continue

        # Avoid parsing large generated reports
        # unrelated to calibration.
        try:
            if path.stat().st_size > 2_000_000:
                continue
        except OSError:
            continue

        yield path


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matches = []
    calibration_files = []
    inspected = 0

    for path in relevant_paths():
        inspected += 1

        document = parse_document(path)

        if document is None:
            continue

        matrices = find_matrices(document)

        metadata = camera_metadata(document)

        if not matrices and not metadata:
            continue

        relative_path = str(
            path.relative_to(ROOT)
        )

        calibration_files.append(relative_path)

        for item in matrices:
            validation = validate_transform(
                item["value"]
            )

            matches.append({
                "file": relative_path,
                "key": item["key"],
                "location": item["location"],
                "validation": validation,
                "matrix": (
                    item["value"]
                    if validation["valid"]
                    else None
                ),
            })

    valid_matches = [
        item
        for item in matches
        if item["validation"]["valid"]
    ]

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "files_inspected": inspected,
        "calibration_files": calibration_files,
        "transforms_found": matches,
        "valid_transform_count": len(
            valid_matches
        ),
        "camera_to_imu_transform_found": any(
            item["key"] in {
                "T_cam_imu",
                "T_imu_cam",
            }
            for item in valid_matches
        ),
        "registration_verified": False,
        "limitations": [
            "A valid 4x4 matrix does not establish "
            "its coordinate-frame convention.",
            "Camera and trajectory frames must be "
            "identified before projection.",
            "Finding extrinsics does not verify "
            "CAD registration.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== CAMERA EXTRINSICS AUDIT ===")
    print("Files inspected:", inspected)
    print(
        "Calibration-related files:",
        len(calibration_files),
    )
    print(
        "Valid transforms:",
        len(valid_matches),
    )
    print(
        "Camera-to-IMU found:",
        report["camera_to_imu_transform_found"],
    )

    for item in valid_matches:
        print(
            item["key"],
            "in",
            item["file"],
        )

    print("Registration verified: NO")
    print(
        "Report:",
        OUTPUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
