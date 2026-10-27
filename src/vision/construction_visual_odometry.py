
"""
BuildProgress - Step 20
Construction Walkthrough Motion Diagnostics

CPU-only OpenCV pipeline:
    Video / ordered frames
    -> ORB keypoints
    -> descriptor matching
    -> RANSAC fundamental matrix
    -> motion reliability report
    -> visual matching preview

IMPORTANT:
This is not metric visual odometry yet.
It does not estimate camera position in CAD coordinates.
It evaluates whether consecutive frames contain
sufficient geometric correspondences for future
camera-motion estimation.

Requirements:
    pip install opencv-python numpy matplotlib

Examples:
    python construction_visual_odometry.py --video data/walkthrough.mp4

    python construction_visual_odometry.py \
        --frames data/walkthrough_frames

    python construction_visual_odometry.py \
        --video data/walkthrough.mp4 \
        --frame-step 5 --max-frames 300
"""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()
DEFAULT_OUTPUT = ROOT / "data" / "visual_odometry"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def make_orb(features):
    return cv2.ORB_create(
        nfeatures=features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=15,
    )


def resize_frame(frame, max_width):
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width

    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )

    return cv2.resize(
        frame,
        new_size,
        interpolation=cv2.INTER_AREA,
    )


def load_video_frames(
    path,
    frame_step,
    max_frames,
    max_width,
):
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if not math.isfinite(fps) or fps <= 0:
        fps = None

    source_index = 0
    kept = 0

    try:
        while kept < max_frames:
            success, frame = capture.read()

            if not success:
                break

            if source_index % frame_step == 0:
                frame = resize_frame(
                    frame, max_width
                )

                timestamp = (
                    source_index / fps
                    if fps is not None
                    else None
                )

                yield {
                    "source_frame_index": source_index,
                    "timestamp_s": timestamp,
                    "frame": frame,
                }

                kept += 1

            source_index += 1

    finally:
        capture.release()


def load_image_frames(
    directory,
    frame_step,
    max_frames,
    max_width,
):
    paths = sorted(
        path
        for path in directory.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not paths:
        raise RuntimeError(
            f"No images found in: {directory}"
        )

    kept = 0

    for source_index, path in enumerate(paths):
        if source_index % frame_step != 0:
            continue

        if kept >= max_frames:
            break

        frame = cv2.imread(str(path))

        if frame is None:
            raise RuntimeError(
                f"Cannot decode image: {path}"
            )

        yield {
            "source_frame_index": source_index,
            "timestamp_s": None,
            "source_image": str(path),
            "frame": resize_frame(
                frame, max_width
            ),
        }

        kept += 1


def extract_features(frame, orb):
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY,
    )

    keypoints, descriptors = (
        orb.detectAndCompute(gray, None)
    )

    return keypoints, descriptors


def match_descriptors(
    first,
    second,
    ratio_threshold,
):
    if first is None or second is None:
        return []

    if len(first) < 2 or len(second) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        first,
        second,
        k=2,
    )

    good = []

    for pair in pairs:
        if len(pair) < 2:
            continue

        best, alternative = pair

        if (
            best.distance
            < ratio_threshold
            * alternative.distance
        ):
            good.append(best)

    good.sort(
        key=lambda match: (
            match.distance,
            match.queryIdx,
            match.trainIdx,
        )
    )

    return good


def estimate_geometry(
    first_keypoints,
    second_keypoints,
    matches,
    ransac_threshold,
):
    if len(matches) < 8:
        return {
            "inlier_mask": None,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_displacement_px": None,
            "fundamental_matrix": None,
        }

    first_points = np.float32([
        first_keypoints[m.queryIdx].pt
        for m in matches
    ])

    second_points = np.float32([
        second_keypoints[m.trainIdx].pt
        for m in matches
    ])

    displacement = np.linalg.norm(
        second_points - first_points,
        axis=1,
    )

    try:
        matrix, mask = (
            cv2.findFundamentalMat(
                first_points,
                second_points,
                cv2.FM_RANSAC,
                ransac_threshold,
                0.99,
            )
        )

    except cv2.error:
        matrix, mask = None, None

    if (
        matrix is None
        or mask is None
        or matrix.shape != (3, 3)
    ):
        return {
            "inlier_mask": None,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "median_displacement_px": float(
                np.median(displacement)
            ),
            "fundamental_matrix": None,
        }

    inlier_mask = (
        mask.reshape(-1).astype(bool)
    )

    inliers = int(
        np.count_nonzero(inlier_mask)
    )

    inlier_ratio = (
        inliers / len(matches)
        if matches else 0.0
    )

    median_displacement = (
        float(np.median(
            displacement[inlier_mask]
        ))
        if inliers > 0
        else None
    )

    return {
        "inlier_mask": inlier_mask,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "median_displacement_px": (
            median_displacement
        ),
        "fundamental_matrix": (
            matrix.tolist()
        ),
    }


def classify_tracking(
    matches,
    inliers,
    inlier_ratio,
    min_matches,
    min_inliers,
    min_inlier_ratio,
):
    if matches < min_matches:
        return "insufficient_matches"

    if inliers < min_inliers:
        return "insufficient_inliers"

    if inlier_ratio < min_inlier_ratio:
        return "weak_geometry"

    return "geometrically_consistent"


def save_match_preview(
    first_frame,
    second_frame,
    first_keypoints,
    second_keypoints,
    matches,
    inlier_mask,
    output_path,
    max_draw_matches,
):
    if inlier_mask is not None:
        display_matches = [
            match
            for match, accepted in zip(
                matches,
                inlier_mask,
            )
            if accepted
        ]
    else:
        display_matches = matches

    display_matches = (
        display_matches[:max_draw_matches]
    )

    preview = cv2.drawMatches(
        first_frame,
        first_keypoints,
        second_frame,
        second_keypoints,
        display_matches,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        ),
    )

    cv2.imwrite(
        str(output_path),
        preview,
    )


def process(args):
    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.video is not None:
        frames = load_video_frames(
            args.video,
            args.frame_step,
            args.max_frames,
            args.max_width,
        )

        source = str(
            args.video.resolve()
        )

        source_type = "video"

    else:
        frames = load_image_frames(
            args.frames,
            args.frame_step,
            args.max_frames,
            args.max_width,
        )

        source = str(
            args.frames.resolve()
        )

        source_type = "ordered_images"

    orb = make_orb(
        args.features
    )

    records = []
    previous = None

    processed_frames = 0
    preview_count = 0

    for current in frames:
        frame = current["frame"]

        keypoints, descriptors = (
            extract_features(
                frame, orb
            )
        )

        current["keypoints"] = keypoints
        current["descriptors"] = descriptors

        processed_frames += 1

        if previous is None:
            previous = current
            continue

        matches = match_descriptors(
            previous["descriptors"],
            current["descriptors"],
            args.ratio,
        )

        geometry = estimate_geometry(
            previous["keypoints"],
            current["keypoints"],
            matches,
            args.ransac_threshold,
        )

        status = classify_tracking(
            len(matches),
            geometry["inliers"],
            geometry["inlier_ratio"],
            args.min_matches,
            args.min_inliers,
            args.min_inlier_ratio,
        )

        record = {
            "from_frame": (
                previous["source_frame_index"]
            ),
            "to_frame": (
                current["source_frame_index"]
            ),
            "from_timestamp_s": (
                previous["timestamp_s"]
            ),
            "to_timestamp_s": (
                current["timestamp_s"]
            ),
            "first_keypoints": len(
                previous["keypoints"]
            ),
            "second_keypoints": len(
                current["keypoints"]
            ),
            "good_matches": len(matches),
            "ransac_inliers": (
                geometry["inliers"]
            ),
            "inlier_ratio": round(
                geometry["inlier_ratio"], 5
            ),
            "median_displacement_px": (
                geometry[
                    "median_displacement_px"
                ]
            ),
            "status": status,
            "fundamental_matrix": (
                geometry["fundamental_matrix"]
            ),
        }

        records.append(record)

        if (
            preview_count < args.previews
            and (
                status == "geometrically_consistent"
                or preview_count == 0
            )
        ):
            output_path = (
                args.output
                / f"matches_{preview_count:03d}.jpg"
            )

            save_match_preview(
                previous["frame"],
                current["frame"],
                previous["keypoints"],
                current["keypoints"],
                matches,
                geometry["inlier_mask"],
                output_path,
                args.max_draw_matches,
            )

            preview_count += 1

        previous = current

    if processed_frames < 2:
        raise RuntimeError(
            "Need at least two decoded frames."
        )

    status_counts = Counter(
        item["status"]
        for item in records
    )

    ratios = [
        item["inlier_ratio"]
        for item in records
    ]

    inlier_counts = [
        item["ransac_inliers"]
        for item in records
    ]

    reliable = status_counts[
        "geometrically_consistent"
    ]

    summary = {
        "schema_version": "0.1",
        "source": source,
        "source_type": source_type,
        "processed_frames": processed_frames,
        "frame_transitions": len(records),
        "status_counts": dict(status_counts),
        "geometrically_consistent_fraction": round(
            reliable / len(records), 5
        ),
        "median_inlier_ratio": float(
            np.median(ratios)
        ),
        "median_ransac_inliers": float(
            np.median(inlier_counts)
        ),
        "parameters": {
            "frame_step": args.frame_step,
            "max_width": args.max_width,
            "features": args.features,
            "ratio": args.ratio,
            "ransac_threshold": (
                args.ransac_threshold
            ),
            "min_matches": args.min_matches,
            "min_inliers": args.min_inliers,
            "min_inlier_ratio": (
                args.min_inlier_ratio
            ),
        },
        "limitations": [
            "Not a metric visual odometry system.",
            "No camera calibration is used.",
            "No camera pose is recovered.",
            "No CAD registration is performed.",
            "A consistent fundamental matrix does not "
            "guarantee correct camera motion.",
            "Repeated textures, moving objects, and "
            "planar scenes can cause degeneracies.",
        ],
    }

    summary_path = (
        args.output / "motion_summary.json"
    )

    transitions_path = (
        args.output / "frame_transitions.json"
    )

    csv_path = (
        args.output / "frame_transitions.csv"
    )

    summary_path.write_text(
        json.dumps(
            summary, indent=2
        ),
        encoding="utf-8",
    )

    transitions_path.write_text(
        json.dumps(
            records, indent=2
        ),
        encoding="utf-8",
    )

    csv_fields = [
        "from_frame",
        "to_frame",
        "from_timestamp_s",
        "to_timestamp_s",
        "first_keypoints",
        "second_keypoints",
        "good_matches",
        "ransac_inliers",
        "inlier_ratio",
        "median_displacement_px",
        "status",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(records)

    print("\nBUILDPROGRESS - MOTION DIAGNOSTICS")
    print("-" * 55)

    print("Source:", source)
    print("Processed frames:", processed_frames)
    print("Frame transitions:", len(records))

    for status, count in sorted(
        status_counts.items()
    ):
        print(f"  {status}: {count}")

    print(
        "Geometrically consistent:",
        f"{summary['geometrically_consistent_fraction']:.1%}",
    )

    print(
        "Median inlier ratio:",
        round(summary["median_inlier_ratio"], 3),
    )

    print(
        "Median RANSAC inliers:",
        round(summary["median_ransac_inliers"], 1),
    )

    print("\nOutputs:")
    print(summary_path)
    print(transitions_path)
    print(csv_path)

    print(
        "\nThis stage measures visual correspondence "
        "quality; it does not estimate metric camera pose."
    )


def main():
    parser = argparse.ArgumentParser()

    source = parser.add_mutually_exclusive_group(
        required=True
    )

    source.add_argument(
        "--video",
        type=Path,
    )

    source.add_argument(
        "--frames",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--frame-step",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--features",
        type=int,
        default=1800,
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=1.5,
    )

    parser.add_argument(
        "--min-matches",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--min-inliers",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--min-inlier-ratio",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--previews",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-draw-matches",
        type=int,
        default=70,
    )

    args = parser.parse_args()

    if args.video is not None:
        if not args.video.is_file():
            parser.error(
                f"Video not found: {args.video}"
            )

    if args.frames is not None:
        if not args.frames.is_dir():
            parser.error(
                f"Frames directory not found: "
                f"{args.frames}"
            )

    if args.frame_step < 1:
        parser.error(
            "--frame-step must be >= 1"
        )

    if args.max_frames < 2:
        parser.error(
            "--max-frames must be >= 2"
        )

    if args.max_width < 64:
        parser.error(
            "--max-width must be >= 64"
        )

    if args.features < 100:
        parser.error(
            "--features must be >= 100"
        )

    if not (0 < args.ratio < 1):
        parser.error(
            "--ratio must be between 0 and 1"
        )

    if args.ransac_threshold <= 0:
        parser.error(
            "--ransac-threshold must be positive"
        )

    if not (
        0 <= args.min_inlier_ratio <= 1
    ):
        parser.error(
            "--min-inlier-ratio must be between 0 and 1"
        )

    if args.previews < 0:
        parser.error(
            "--previews cannot be negative"
        )

    process(args)


if __name__ == "__main__":
    main()

