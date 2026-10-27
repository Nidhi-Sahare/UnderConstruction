
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

DATA = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

FRAMES = (
    DATA
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frames"
)

MANIFEST = (
    DATA
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frame_manifest.csv"
)

LANDMARKS = (
    ROOT
    / "outputs"
    / "multiview_tracks"
    / "validated_landmarks.json"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "reconstruction_coverage"
)


def frame_number(path):
    match = re.search(
        r"(\d+)(?=\.[^.]+$)",
        path.name,
    )

    return int(match.group(1)) if match else -1


def read_manifest(path):
    result = {}

    if not path.exists():
        return result

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            filename = row.get("filename")

            if filename:
                result[filename] = row

    return result


def read_landmarks(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing landmark data: {path}"
        )

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, list):
        raise ValueError(
            "Expected a list of landmark records."
        )

    counts = Counter()
    unique_coordinates = set()
    duplicate_count = 0

    for landmark in data:
        xyz = landmark.get("xyz")

        if xyz is not None:
            key = tuple(
                round(float(v), 8)
                for v in xyz
            )

            if key in unique_coordinates:
                duplicate_count += 1
            else:
                unique_coordinates.add(key)

        for filename in set(
            landmark.get("frames", [])
        ):
            counts[filename] += 1

    return data, counts, duplicate_count


def image_metrics(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Central crop reduces the influence
    # of dark fisheye borders.
    h, w = gray.shape

    x1 = int(w * 0.15)
    x2 = int(w * 0.85)

    y1 = int(h * 0.15)
    y2 = int(h * 0.85)

    central = gray[y1:y2, x1:x2]

    if central.size == 0:
        central = gray

    laplacian = cv2.Laplacian(
        central,
        cv2.CV_64F,
    )

    sharpness = float(
        np.var(laplacian)
    )

    brightness = float(
        np.mean(central)
    )

    contrast = float(
        np.std(central)
    )

    dark_fraction = float(
        np.mean(central <= 20)
    )

    bright_fraction = float(
        np.mean(central >= 245)
    )

    return {
        "sharpness_laplacian_variance": sharpness,
        "mean_brightness": brightness,
        "contrast_std": contrast,
        "dark_fraction": dark_fraction,
        "bright_fraction": bright_fraction,
    }


def resize_image(image, max_width):
    h, w = image.shape[:2]

    scale = min(
        1.0,
        max_width / float(w),
    )

    if scale == 1.0:
        return image

    return cv2.resize(
        image,
        (
            max(1, int(round(w * scale))),
            max(1, int(round(h * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def extract_features(image, sift):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    keypoints, descriptors = (
        sift.detectAndCompute(
            gray,
            None,
        )
    )

    return keypoints, descriptors


def mutual_ratio_matches(
    descriptors_a,
    descriptors_b,
    matcher,
    ratio,
):
    if (
        descriptors_a is None
        or descriptors_b is None
        or len(descriptors_a) < 2
        or len(descriptors_b) < 2
    ):
        return []

    forward = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2,
    )

    backward = matcher.knnMatch(
        descriptors_b,
        descriptors_a,
        k=2,
    )

    def filtered(pairs):
        accepted = {}

        for pair in pairs:
            if len(pair) != 2:
                continue

            first, second = pair

            if (
                first.distance
                < ratio * second.distance
            ):
                accepted[first.queryIdx] = first

        return accepted

    fwd = filtered(forward)
    rev = filtered(backward)

    return [
        match
        for index, match in fwd.items()
        if (
            match.trainIdx in rev
            and rev[match.trainIdx].trainIdx
            == index
        )
    ]


def homography_inlier_count(
    keypoints_a,
    keypoints_b,
    matches,
):
    if len(matches) < 8:
        return 0

    points_a = np.float32(
        [
            keypoints_a[m.queryIdx].pt
            for m in matches
        ]
    )

    points_b = np.float32(
        [
            keypoints_b[m.trainIdx].pt
            for m in matches
        ]
    )

    try:
        _, mask = cv2.findHomography(
            points_a,
            points_b,
            cv2.RANSAC,
            3.0,
        )
    except cv2.error:
        return 0

    return (
        int(mask.sum())
        if mask is not None
        else 0
    )


def pose_displacement(row_a, row_b):
    if row_a is None or row_b is None:
        return None

    if (
        str(row_a.get("pose_matched", ""))
        .strip().lower() != "true"
        or str(row_b.get("pose_matched", ""))
        .strip().lower() != "true"
    ):
        return None

    try:
        a = np.array(
            [
                float(row_a[k])
                for k in ("tx", "ty", "tz")
            ],
            dtype=np.float64,
        )

        b = np.array(
            [
                float(row_b[k])
                for k in ("tx", "ty", "tz")
            ],
            dtype=np.float64,
        )

        if (
            not np.isfinite(a).all()
            or not np.isfinite(b).all()
        ):
            return None

        return float(
            np.linalg.norm(b - a)
        )

    except (KeyError, ValueError, TypeError):
        return None


def classify_frame(record):
    if record["read_error"]:
        return "UNREADABLE"

    if record["sift_features"] < 30:
        return "LOW_FEATURE_COUNT"

    if record["dark_fraction"] > 0.65:
        return "DARK_IMAGE"

    if record["bright_fraction"] > 0.65:
        return "OVEREXPOSED_IMAGE"

    if record["sharpness_laplacian_variance"] < 15:
        return "LOW_SHARPNESS_INDICATOR"

    if record["max_adjacent_mutual_matches"] < 15:
        return "LOW_ADJACENT_OVERLAP"

    if record["validated_landmark_support"] == 0:
        return "MATCHES_BUT_NO_VALIDATED_LANDMARKS"

    return "HAS_VALIDATED_LANDMARKS"


def color_for_status(status):
    mapping = {
        "HAS_VALIDATED_LANDMARKS": (55, 145, 75),
        "MATCHES_BUT_NO_VALIDATED_LANDMARKS": (
            30, 155, 220
        ),
        "LOW_ADJACENT_OVERLAP": (40, 85, 215),
        "LOW_FEATURE_COUNT": (75, 75, 190),
        "LOW_SHARPNESS_INDICATOR": (160, 85, 190),
        "DARK_IMAGE": (70, 70, 70),
        "OVEREXPOSED_IMAGE": (190, 155, 60),
        "UNREADABLE": (0, 0, 190),
    }

    return mapping.get(
        status,
        (120, 120, 120),
    )


def make_contact_sheet(
    paths,
    records,
    output_path,
    columns=8,
):
    tile_w = 200
    image_h = 160
    label_h = 65
    tile_h = image_h + label_h

    rows = math.ceil(
        len(paths) / columns
    )

    canvas = np.full(
        (
            rows * tile_h,
            columns * tile_w,
            3,
        ),
        246,
        dtype=np.uint8,
    )

    for index, path in enumerate(paths):
        record = records[index]

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            image = np.zeros(
                (image_h, tile_w, 3),
                dtype=np.uint8,
            )

        h, w = image.shape[:2]

        scale = min(
            tile_w / w,
            image_h / h,
        )

        resized = cv2.resize(
            image,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

        row = index // columns
        col = index % columns

        x0 = col * tile_w
        y0 = row * tile_h

        offset_x = (
            tile_w - resized.shape[1]
        ) // 2

        offset_y = (
            image_h - resized.shape[0]
        ) // 2

        canvas[
            y0 + offset_y:
            y0 + offset_y + resized.shape[0],
            x0 + offset_x:
            x0 + offset_x + resized.shape[1],
        ] = resized

        color = color_for_status(
            record["status"]
        )

        cv2.rectangle(
            canvas,
            (x0 + 1, y0 + 1),
            (
                x0 + tile_w - 2,
                y0 + image_h - 2,
            ),
            color,
            3,
        )

        lines = [
            path.stem,
            (
                f"SIFT {record['sift_features']} "
                f"| M {record['max_adjacent_mutual_matches']}"
            ),
            (
                f"3D {record['validated_landmark_support']} "
                f"| {record['status'][:20]}"
            ),
        ]

        for line_index, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (
                    x0 + 5,
                    y0 + image_h + 15
                    + line_index * 17,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (35, 35, 35),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def make_coverage_chart(
    records,
    output_path,
):
    width = 1600
    height = 650

    canvas = np.full(
        (height, width, 3),
        250,
        dtype=np.uint8,
    )

    left = 80
    right = 35
    top = 65
    bottom = 85

    chart_width = width - left - right
    chart_height = height - top - bottom

    support = np.array(
        [
            record["validated_landmark_support"]
            for record in records
        ],
        dtype=np.float64,
    )

    matches = np.array(
        [
            record["max_adjacent_mutual_matches"]
            for record in records
        ],
        dtype=np.float64,
    )

    maximum = max(
        1.0,
        float(
            np.max(
                np.concatenate(
                    [support, matches]
                )
            )
        ),
    )

    cv2.putText(
        canvas,
        "Under-construction | Reconstruction coverage",
        (left, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    for tick in range(6):
        value = maximum * tick / 5

        y = (
            top + chart_height
            - int(chart_height * tick / 5)
        )

        cv2.line(
            canvas,
            (left, y),
            (width - right, y),
            (225, 225, 225),
            1,
        )

        cv2.putText(
            canvas,
            str(int(round(value))),
            (15, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    n = len(records)

    def xy(index, value):
        x = left + int(
            index
            / max(1, n - 1)
            * chart_width
        )

        y = (
            top + chart_height
            - int(
                value / maximum
                * chart_height
            )
        )

        return x, y

    for index in range(n):
        x, y = xy(
            index, support[index]
        )

        cv2.line(
            canvas,
            (x, top + chart_height),
            (x, y),
            (55, 145, 75),
            3,
        )

    for index in range(n - 1):
        cv2.line(
            canvas,
            xy(index, matches[index]),
            xy(index + 1, matches[index + 1]),
            (210, 105, 40),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "Green bars: validated landmark support",
        (left, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (55, 145, 75),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Blue line: adjacent mutual feature matches",
        (left + 520, height - 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (210, 105, 40),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "Frame order",
        (width // 2 - 50, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(
        str(output_path),
        canvas,
    )


def write_csv(path, records):
    if not records:
        return

    fields = list(records[0].keys())

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--features",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.78,
    )

    args = parser.parse_args()

    if (
        args.width < 100
        or args.features < 100
        or not (0 < args.ratio < 1)
    ):
        parser.error(
            "Invalid width, feature count or ratio."
        )

    if not FRAMES.exists():
        raise FileNotFoundError(FRAMES)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        FRAMES.glob("*.jpg"),
        key=frame_number,
    )

    if not paths:
        raise RuntimeError(
            "No photographs found."
        )

    manifest = read_manifest(MANIFEST)

    landmarks, landmark_counts, duplicates = (
        read_landmarks(LANDMARKS)
    )

    sift = cv2.SIFT_create(
        nfeatures=args.features
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    records = []
    previous = None

    print("\nUNDER-CONSTRUCTION")
    print("Reconstruction coverage audit")
    print("-" * 58)

    print("Photographs:", len(paths))
    print("Validated landmarks:", len(landmarks))
    print("Duplicate landmark coordinates:", duplicates)

    for index, path in enumerate(paths):
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        record = {
            "frame_index": index,
            "filename": path.name,
            "pose_matched": (
                str(
                    manifest.get(
                        path.name, {}
                    ).get("pose_matched", "")
                ).strip().lower() == "true"
            ),
            "read_error": image is None,
            "sharpness_laplacian_variance": 0.0,
            "mean_brightness": 0.0,
            "contrast_std": 0.0,
            "dark_fraction": 0.0,
            "bright_fraction": 0.0,
            "sift_features": 0,
            "matches_to_previous": 0,
            "homography_inliers_to_previous": 0,
            "pose_displacement_from_previous": None,
            "max_adjacent_mutual_matches": 0,
            "validated_landmark_support": int(
                landmark_counts.get(path.name, 0)
            ),
            "status": "",
        }

        if image is not None:
            small = resize_image(
                image,
                args.width,
            )

            metrics = image_metrics(
                small
            )

            record.update(metrics)

            keypoints, descriptors = (
                extract_features(
                    small,
                    sift,
                )
            )

            record["sift_features"] = (
                len(keypoints)
                if keypoints is not None
                else 0
            )

            if previous is not None:
                matches = mutual_ratio_matches(
                    previous["descriptors"],
                    descriptors,
                    matcher,
                    args.ratio,
                )

                count = len(matches)

                record[
                    "matches_to_previous"
                ] = count

                record[
                    "homography_inliers_to_previous"
                ] = homography_inlier_count(
                    previous["keypoints"],
                    keypoints,
                    matches,
                )

                record[
                    "pose_displacement_from_previous"
                ] = pose_displacement(
                    manifest.get(
                        previous["filename"]
                    ),
                    manifest.get(path.name),
                )

                record[
                    "max_adjacent_mutual_matches"
                ] = count

                previous_record = records[-1]

                previous_record[
                    "max_adjacent_mutual_matches"
                ] = max(
                    previous_record[
                        "max_adjacent_mutual_matches"
                    ],
                    count,
                )

            previous = {
                "filename": path.name,
                "keypoints": keypoints,
                "descriptors": descriptors,
            }

        else:
            previous = None

        records.append(record)

        if (
            (index + 1) % 30 == 0
            or index + 1 == len(paths)
        ):
            print(
                f"Audited {index + 1}/{len(paths)} "
                "photographs"
            )

    for record in records:
        record["status"] = classify_frame(
            record
        )

    status_counts = Counter(
        record["status"]
        for record in records
    )

    supported_frames = sum(
        record["validated_landmark_support"] > 0
        for record in records
    )

    zero_feature_frames = sum(
        record["sift_features"] == 0
        for record in records
    )

    weak_overlap_frames = sum(
        record["max_adjacent_mutual_matches"] < 15
        for record in records
    )

    sharpness_values = [
        record["sharpness_laplacian_variance"]
        for record in records
        if not record["read_error"]
    ]

    csv_path = (
        OUTPUT
        / "frame_coverage.csv"
    )

    contact_path = (
        OUTPUT
        / "frame_contact_sheet.jpg"
    )

    chart_path = (
        OUTPUT
        / "coverage_chart.png"
    )

    report_path = (
        OUTPUT
        / "coverage_audit_report.json"
    )

    write_csv(
        csv_path,
        records,
    )

    make_contact_sheet(
        paths,
        records,
        contact_path,
    )

    make_coverage_chart(
        records,
        chart_path,
    )

    report = {
        "project": "Under-construction",
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "photographs": len(paths),
        "pose_matched_frames": sum(
            record["pose_matched"]
            for record in records
        ),
        "validated_landmarks": len(landmarks),
        "duplicate_landmark_coordinates": duplicates,
        "frames_with_validated_landmarks": supported_frames,
        "frames_with_zero_sift_features": zero_feature_frames,
        "frames_with_weak_adjacent_overlap": weak_overlap_frames,
        "median_sharpness_indicator": (
            float(
                np.median(
                    sharpness_values
                )
            )
            if sharpness_values else None
        ),
        "status_distribution": dict(
            status_counts
        ),
        "outputs": {
            "frame_csv": str(csv_path),
            "contact_sheet": str(contact_path),
            "coverage_chart": str(chart_path),
        },
        "limitations": [
            "Sharpness is measured by Laplacian variance.",
            "Sharpness threshold is heuristic and "
            "not calibrated to this camera.",
            "Adjacent mutual matches measure "
            "descriptor correspondence, not "
            "verified geometric overlap.",
            "Homography inliers are only a "
            "planar-consistency diagnostic.",
            "Homography results do not establish "
            "3D camera pose or scene depth.",
            "A frame may lack validated landmarks "
            "because its image pairs were not "
            "included in the earlier reconstruction.",
            "This audit does not establish "
            "metric reconstruction accuracy.",
            "Camera optical extrinsics remain unverified.",
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("-" * 58)

    print(
        "Frames with validated landmarks:",
        supported_frames,
    )

    print(
        "Frames with zero SIFT features:",
        zero_feature_frames,
    )

    print(
        "Frames with weak adjacent overlap:",
        weak_overlap_frames,
    )

    print(
        "Status distribution:",
        dict(status_counts),
    )

    print("\nFrame CSV:", csv_path)
    print("Contact sheet:", contact_path)
    print("Coverage chart:", chart_path)
    print("Report:", report_path)

    print(
        "\nNOTE: These diagnostics are observational. "
        "They do not verify camera calibration "
        "or metric 3D accuracy."
    )


if __name__ == "__main__":
    main()
