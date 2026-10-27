
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

OUTPUT_DIR = ROOT / "outputs" / "calibration_audit"
REPORT_PATH = OUTPUT_DIR / "calibration_audit.json"

SUPPORTED_METADATA = {
    ".json",
    ".yaml",
    ".yml",
    ".xml",
    ".txt",
    ".csv",
    ".toml",
    ".ini",
}

MAX_METADATA_BYTES = 5_000_000

CALIBRATION_TERMS = (
    "calib",
    "intrinsic",
    "extrinsic",
    "fisheye",
    "camera",
    "sensor",
    "transform",
    "projection",
    "distortion",
    "trajectory",
    "pose",
    "rig",
    "imu",
    "reference",
    "ground_truth",
)

ORIENTATION_FIELDS = (
    "qx",
    "qy",
    "qz",
    "qw",
    "quaternion",
    "rotation",
    "roll",
    "pitch",
    "yaw",
    "heading",
    "orientation",
    "r00",
    "r01",
    "r02",
    "r10",
    "r11",
    "r12",
    "r20",
    "r21",
    "r22",
)

POSITION_FIELDS = (
    "tx",
    "ty",
    "tz",
    "position",
    "translation",
    "pose",
)

INTRINSIC_FIELDS = (
    "fx",
    "fy",
    "cx",
    "cy",
    "camera_matrix",
    "intrinsic",
    "focal_length",
    "principal_point",
)

DISTORTION_FIELDS = (
    "distortion",
    "dist_coeff",
    "k1",
    "k2",
    "k3",
    "k4",
    "xi",
    "alpha",
    "kb4",
    "equidistant",
    "kannala",
)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def find_sequence_directories():
    matches = []

    for path in ROOT.rglob(SEQUENCE):
        if path.is_dir() and ".venv" not in path.parts:
            matches.append(path)

    return sorted(set(matches), key=str)


def find_manifests(sequence_dirs):
    matches = []

    for directory in sequence_dirs:
        matches.extend(directory.rglob("frame_manifest.csv"))

    if not matches:
        matches = [
            path
            for path in ROOT.rglob("frame_manifest.csv")
            if SEQUENCE in str(path)
            and ".venv" not in path.parts
        ]

    return sorted(set(matches), key=str)


def inspect_manifest(path: Path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        columns = reader.fieldnames or []
        rows = list(reader)

    matched = sum(
        str(row.get("pose_matched", "")).strip().lower()
        in ("true", "1", "yes")
        for row in rows
    )

    column_values = {}

    for column in columns:
        nonempty = [
            row.get(column)
            for row in rows
            if str(row.get(column, "")).strip()
        ]

        column_values[column] = {
            "nonempty_count": len(nonempty),
            "example": str(nonempty[0])[:160]
            if nonempty
            else None,
        }

    lower_columns = {
        column.lower(): column
        for column in columns
    }

    def matched_fields(terms):
        return [
            original
            for lower, original in lower_columns.items()
            if any(term in lower for term in terms)
        ]

    return {
        "path": relative(path),
        "row_count": len(rows),
        "pose_matched_count": matched,
        "columns": columns,
        "position_fields": matched_fields(
            POSITION_FIELDS
        ),
        "orientation_fields": matched_fields(
            ORIENTATION_FIELDS
        ),
        "intrinsic_fields": matched_fields(
            INTRINSIC_FIELDS
        ),
        "distortion_fields": matched_fields(
            DISTORTION_FIELDS
        ),
        "column_coverage": column_values,
    }


def inspect_images(sequence_dirs):
    image_paths = []

    for directory in sequence_dirs:
        image_paths.extend(directory.rglob("*.jpg"))
        image_paths.extend(directory.rglob("*.jpeg"))
        image_paths.extend(directory.rglob("*.png"))

    image_paths = sorted(set(image_paths))

    if not image_paths:
        return {
            "count": 0,
            "sample": None,
            "readable": False,
        }

    sample = image_paths[0]
    image = cv2.imread(str(sample))

    return {
        "count": len(image_paths),
        "sample": relative(sample),
        "readable": image is not None,
        "width": (
            int(image.shape[1])
            if image is not None
            else None
        ),
        "height": (
            int(image.shape[0])
            if image is not None
            else None
        ),
    }


def candidate_metadata_files(sequence_dirs):
    search_roots = set(sequence_dirs)

    for directory in sequence_dirs:
        search_roots.add(directory.parent)
        search_roots.add(directory.parent.parent)

    results = set()

    for directory in search_roots:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in SUPPORTED_METADATA:
                continue

            if path.stat().st_size > MAX_METADATA_BYTES:
                continue

            name = path.name.lower()

            if (
                any(term in name for term in CALIBRATION_TERMS)
                or path.name == "frame_manifest.csv"
            ):
                results.add(path)

    return sorted(results, key=str)


def inspect_metadata_text(path: Path):
    try:
        content = path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )
    except OSError as exc:
        return {
            "path": relative(path),
            "read_error": str(exc),
        }

    lines = content.splitlines()

    terms = (
        ORIENTATION_FIELDS
        + INTRINSIC_FIELDS
        + DISTORTION_FIELDS
        + (
            "camera_to",
            "body_to",
            "world_to",
            "transform",
            "extrinsic",
            "ground_truth",
            "reference",
        )
    )

    matches = []

    for line_number, line in enumerate(lines, 1):
        lower = line.lower()

        # Match field names rather than arbitrary substrings
        # inside unrelated words.
        found = [
            term
            for term in terms
            if re.search(
                rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
                lower,
            )
        ]

        if found:
            matches.append({
                "line": line_number,
                "terms": found[:8],
                "excerpt": line.strip()[:250],
            })

        if len(matches) >= 25:
            break

    return {
        "path": relative(path),
        "size_bytes": path.stat().st_size,
        "matching_lines": matches,
        "has_candidate_metadata": bool(matches),
    }


def inspect_project_metadata():
    roots = [
        ROOT / "data",
        ROOT / "outputs",
    ]

    results = []

    for directory in roots:
        if not directory.exists():
            continue

        for path in directory.rglob("*"):
            if not path.is_file():
                continue

            if ".venv" in path.parts:
                continue

            if path.suffix.lower() not in SUPPORTED_METADATA:
                continue

            name = path.name.lower()

            if any(
                term in name
                for term in (
                    "calib",
                    "intrinsic",
                    "extrinsic",
                    "fisheye",
                    "camera_model",
                    "ground_truth",
                    "reference_pose",
                )
            ):
                results.append(relative(path))

    return sorted(set(results))


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sequence_dirs = find_sequence_directories()

    if not sequence_dirs:
        raise RuntimeError(
            f"Cannot locate sequence: {SEQUENCE}"
        )

    manifests = find_manifests(sequence_dirs)

    manifest_reports = [
        inspect_manifest(path)
        for path in manifests
    ]

    image_report = inspect_images(sequence_dirs)

    metadata_files = candidate_metadata_files(
        sequence_dirs
    )

    metadata_reports = [
        inspect_metadata_text(path)
        for path in metadata_files
        if path.name != "frame_manifest.csv"
    ]

    project_metadata = inspect_project_metadata()

    orientation_fields = sorted({
        field
        for report in manifest_reports
        for field in report["orientation_fields"]
    })

    intrinsic_fields = sorted({
        field
        for report in manifest_reports
        for field in report["intrinsic_fields"]
    })

    distortion_fields = sorted({
        field
        for report in manifest_reports
        for field in report["distortion_fields"]
    })

    metadata_hits = [
        report
        for report in metadata_reports
        if report.get("has_candidate_metadata")
    ]

    report = {
        "sequence": SEQUENCE,
        "sequence_directories": [
            relative(path)
            for path in sequence_dirs
        ],
        "manifests": manifest_reports,
        "images": image_report,
        "manifest_orientation_fields": orientation_fields,
        "manifest_intrinsic_fields": intrinsic_fields,
        "manifest_distortion_fields": distortion_fields,
        "metadata_candidates": metadata_hits,
        "project_calibration_candidates": project_metadata,
        "calibration_verified": False,
        "cad_registration_verified": False,
        "notes": [
            "Field names are discovery clues, not verified calibration.",
            "Metadata presence does not establish its coordinate convention.",
            "Camera orientation must be interpreted in the correct frame.",
            "A valid camera model is required for CAD-to-image projection.",
            "Independent reference data is needed for stronger accuracy claims.",
        ],
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== WALKTHROUGH CALIBRATION AUDIT ===")
    print("Sequence:", SEQUENCE)
    print("Sequence directories:", len(sequence_dirs))
    print("Manifests:", len(manifest_reports))

    for item in manifest_reports:
        print(
            "Manifest:",
            item["path"],
        )
        print(
            "  Rows:",
            item["row_count"],
        )
        print(
            "  Matched poses:",
            item["pose_matched_count"],
        )
        print(
            "  Position fields:",
            item["position_fields"],
        )
        print(
            "  Orientation fields:",
            item["orientation_fields"],
        )

    print()
    print("Images:", image_report["count"])
    print(
        "Image dimensions:",
        image_report.get("width"),
        "x",
        image_report.get("height"),
    )

    print()
    print(
        "Manifest intrinsic fields:",
        intrinsic_fields,
    )
    print(
        "Manifest distortion fields:",
        distortion_fields,
    )
    print(
        "Metadata files with candidate fields:",
        len(metadata_hits),
    )

    for item in metadata_hits[:15]:
        print("  ", item["path"])

    print(
        "Other project calibration candidates:",
        len(project_metadata),
    )

    for path in project_metadata[:15]:
        print("  ", path)

    print()
    print("Calibration verified: NO")
    print("CAD registration verified: NO")
    print("Report:", relative(REPORT_PATH))


if __name__ == "__main__":
    main()
