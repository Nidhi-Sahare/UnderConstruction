
"""
Under-construction: OpenCV 5 Walkthrough Analysis

Supported inputs:
    --frames DIRECTORY
    --video VIDEO_FILE

Uses OpenCV 5 for:
    - Image/video decoding
    - ORB feature detection
    - Descriptor matching
    - RANSAC homography estimation
    - Lucas-Kanade optical flow
    - Visual tracking diagnostics

Outputs:
    analysis.json
    tracking_preview.mp4 (when encoder available)
    tracking_preview_frames/*.jpg
    trajectory.png
    keyframes/*.jpg

Important:
    Image displacement is NOT a metric camera trajectory.
    No CAD alignment or construction progress is inferred.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs/opencv5_walkthrough"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


def numeric_sort_key(path):
    parts = re.split(r"(\d+)", path.stem.lower())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    )


def extract_frame_id(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def resize_frame(frame, max_width):
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width

    return cv2.resize(
        frame,
        (
            max_width,
            max(1, round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )


def image_sequence(directory):
    files = sorted(
        [
            path
            for path in directory.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and not path.stem.lower().endswith("_overlay")
            )
        ],
        key=numeric_sort_key,
    )

    if not files:
        raise RuntimeError(
            f"No source images found in {directory}"
        )

    for sequence_index, path in enumerate(files):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if frame is None:
            yield {
                "sequence_index": sequence_index,
                "source_frame_id": extract_frame_id(path),
                "source_filename": path.name,
                "frame": None,
                "timestamp_seconds": None,
            }
            continue

        yield {
            "sequence_index": sequence_index,
            "source_frame_id": extract_frame_id(path),
            "source_filename": path.name,
            "frame": frame,
            "timestamp_seconds": None,
        }


def video_sequence(path):
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video: {path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    valid_fps = math.isfinite(fps) and fps > 0

    index = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            yield {
                "sequence_index": index,
                "source_frame_id": index,
                "source_filename": None,
                "frame": frame,
                "timestamp_seconds": (
                    index / fps if valid_fps else None
                ),
            }

            index += 1
    finally:
        capture.release()


def create_orb():
    return cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=15,
    )


def orb_correspondences(
    detector,
    previous_gray,
    current_gray,
):
    kp_a, desc_a = detector.detectAndCompute(
        previous_gray, None
    )

    kp_b, desc_b = detector.detectAndCompute(
        current_gray, None
    )

    if (
        desc_a is None
        or desc_b is None
        or len(desc_a) < 2
        or len(desc_b) < 2
    ):
        return (
            len(kp_a),
            len(kp_b),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    pairs = matcher.knnMatch(
        desc_a, desc_b, k=2
    )

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < 0.75 * second.distance:
            good.append(first)

    points_a = np.float32([
        kp_a[match.queryIdx].pt
        for match in good
    ]).reshape(-1, 2)

    points_b = np.float32([
        kp_b[match.trainIdx].pt
        for match in good
    ]).reshape(-1, 2)

    return (
        len(kp_a),
        len(kp_b),
        points_a,
        points_b,
    )


def estimate_geometry(points_a, points_b):
    if len(points_a) < 8:
        return None, None

    try:
        matrix, mask = cv2.findHomography(
            points_a,
            points_b,
            cv2.RANSAC,
            3.0,
        )
    except cv2.error:
        return None, None

    if matrix is None or mask is None:
        return None, None

    if not np.all(np.isfinite(matrix)):
        return None, None

    return matrix, mask.ravel().astype(bool)


def track_optical_flow(
    previous_gray,
    current_gray,
    points,
):
    if len(points) < 8:
        return 0, None

    source = points.reshape(
        -1, 1, 2
    ).astype(np.float32)

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        source,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if tracked is None or status is None:
        return 0, None

    valid = status.ravel() == 1

    if not np.any(valid):
        return 0, None

    displacement = (
        tracked.reshape(-1, 2)[valid]
        - source.reshape(-1, 2)[valid]
    )

    return (
        int(valid.sum()),
        np.median(displacement, axis=0).tolist(),
    )


def analyze_pair(detector, previous_gray, gray):
    (
        features_a,
        features_b,
        points_a,
        points_b,
    ) = orb_correspondences(
        detector,
        previous_gray,
        gray,
    )

    matrix, mask = estimate_geometry(
        points_a, points_b
    )

    match_count = len(points_a)

    inliers = (
        int(mask.sum())
        if mask is not None
        else 0
    )

    ratio = (
        inliers / match_count
        if match_count
        else 0.0
    )

    flow_count, flow_motion = track_optical_flow(
        previous_gray,
        gray,
        points_a,
    )

    consistent = (
        mask is not None
        and inliers >= 12
        and ratio >= 0.35
    )

    displacement = None

    if consistent:
        displacement = np.median(
            points_b[mask] - points_a[mask],
            axis=0,
        ).tolist()

    return {
        "previous_feature_count": features_a,
        "current_feature_count": features_b,
        "orb_match_count": match_count,
        "homography_inliers": inliers,
        "homography_inlier_ratio": round(
            ratio, 5
        ),
        "optical_flow_tracks": flow_count,
        "optical_flow_median_displacement_px":
            flow_motion,
        "homography": (
            matrix.tolist()
            if matrix is not None
            else None
        ),
        "image_displacement_px": displacement,
        "motion_status": (
            "GEOMETRICALLY_CONSISTENT"
            if consistent
            else "INSUFFICIENT_GEOMETRIC_CONSISTENCY"
        ),
        "_points_a": points_a,
        "_points_b": points_b,
        "_inlier_mask": mask,
    }


def tracking_visual(frame, result):
    canvas = frame.copy()

    if result is not None:
        points_a = result["_points_a"]
        points_b = result["_points_b"]
        mask = result["_inlier_mask"]

        if mask is not None:
            for a, b in zip(
                points_a[mask],
                points_b[mask],
            ):
                start = tuple(
                    np.round(a).astype(int)
                )

                end = tuple(
                    np.round(b).astype(int)
                )

                cv2.line(
                    canvas,
                    start,
                    end,
                    (0, 180, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    canvas,
                    end,
                    2,
                    (0, 255, 0),
                    -1,
                )

    return canvas


def draw_trajectory(positions, output):
    canvas = np.full(
        (720, 960, 3),
        245,
        dtype=np.uint8,
    )

    points = np.asarray(
        positions,
        dtype=np.float64,
    )

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)

    extent = np.maximum(
        maximum - minimum,
        1.0,
    )

    scale = min(
        820 / extent[0],
        570 / extent[1],
    )

    pixels = np.round(
        (points - minimum) * scale
        + np.array([60, 100])
    ).astype(np.int32)

    for a, b in zip(
        pixels[:-1],
        pixels[1:],
    ):
        cv2.line(
            canvas,
            tuple(a),
            tuple(b),
            (170, 80, 30),
            2,
            cv2.LINE_AA,
        )

    cv2.circle(
        canvas,
        tuple(pixels[0]),
        7,
        (0, 160, 0),
        -1,
    )

    cv2.circle(
        canvas,
        tuple(pixels[-1]),
        7,
        (0, 0, 220),
        -1,
    )

    cv2.putText(
        canvas,
        "Cumulative image displacement - NOT camera position",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(output), canvas)


def run(args):
    if cv2.__version__.split(".")[0] != "5":
        raise RuntimeError(
            "This hackathon pipeline requires OpenCV 5."
        )

    if args.frames is not None:
        if not args.frames.is_dir():
            raise FileNotFoundError(args.frames)

        iterator = image_sequence(args.frames)
        source_type = "IMAGE_SEQUENCE"
        source_path = args.frames

    else:
        if not args.video.is_file():
            raise FileNotFoundError(args.video)

        iterator = video_sequence(args.video)
        source_type = "VIDEO"
        source_path = args.video

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    keyframes_dir = args.out / "keyframes"
    preview_frames_dir = (
        args.out / "tracking_preview_frames"
    )

    keyframes_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    preview_frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = create_orb()

    previous_gray = None
    previous_source_frame_id = None
    previous_sequence_index = None

    records = []
    failures = []
    positions = [[0.0, 0.0]]
    cumulative = np.zeros(
        2, dtype=np.float64
    )

    preview_writer = None
    preview_written = False

    processed = 0

    for item in iterator:
        if (
            args.max_frames > 0
            and item["sequence_index"] >= args.max_frames
        ):
            break

        if (
            item["sequence_index"]
            % args.sample_every != 0
        ):
            continue

        if item["frame"] is None:
            failures.append({
                "sequence_index": item["sequence_index"],
                "source_filename": item["source_filename"],
                "reason": "IMAGE_DECODE_FAILED",
            })

            previous_gray = None
            previous_source_frame_id = None
            previous_sequence_index = None
            continue

        frame = resize_frame(
            item["frame"],
            args.max_width,
        )

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        result = None

        if previous_gray is not None:
            result = analyze_pair(
                detector,
                previous_gray,
                gray,
            )

        record = {
            "sequence_index": item["sequence_index"],
            "source_frame_id": item["source_frame_id"],
            "source_filename": item["source_filename"],
            "timestamp_seconds": item["timestamp_seconds"],
            "previous_sequence_index":
                previous_sequence_index,
            "previous_source_frame_id":
                previous_source_frame_id,
            "motion_status": (
                "FIRST_FRAME_OR_TRACK_RESET"
                if result is None
                else result["motion_status"]
            ),
            "cumulative_image_displacement_px":
                cumulative.tolist(),
        }

        if result is not None:
            displacement = result[
                "image_displacement_px"
            ]

            if displacement is not None:
                cumulative += np.asarray(
                    displacement,
                    dtype=np.float64,
                )

            record.update({
                key: value
                for key, value in result.items()
                if not key.startswith("_")
            })

            record[
                "cumulative_image_displacement_px"
            ] = cumulative.tolist()

            positions.append(
                cumulative.tolist()
            )

        preview = tracking_visual(
            frame, result
        )

        label = (
            f"Frame {item['source_frame_id']} | "
            f"{record['motion_status']}"
        )

        cv2.putText(
            preview,
            label,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if processed % args.keyframe_every == 0:
            filename = (
                f"frame_{item['sequence_index']:06d}.jpg"
            )

            cv2.imwrite(
                str(keyframes_dir / filename),
                frame,
            )

            record["keyframe_path"] = str(
                keyframes_dir / filename
            )

        preview_name = (
            f"preview_{item['sequence_index']:06d}.jpg"
        )

        cv2.imwrite(
            str(preview_frames_dir / preview_name),
            preview,
        )

        if args.preview_fps is not None:
            if preview_writer is None:
                height, width = preview.shape[:2]

                preview_writer = cv2.VideoWriter(
                    str(
                        args.out / "tracking_preview.mp4"
                    ),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.preview_fps,
                    (width, height),
                )

                preview_written = (
                    preview_writer.isOpened()
                )

            if preview_written:
                preview_writer.write(preview)

        records.append(record)

        previous_gray = gray
        previous_source_frame_id = item[
            "source_frame_id"
        ]
        previous_sequence_index = item[
            "sequence_index"
        ]

        processed += 1

    if preview_writer is not None:
        preview_writer.release()

    if not records:
        raise RuntimeError(
            "No readable source frames processed."
        )

    pairs = [
        item
        for item in records
        if item["previous_sequence_index"] is not None
    ]

    consistent = [
        item
        for item in pairs
        if item["motion_status"]
        == "GEOMETRICALLY_CONSISTENT"
    ]

    report = {
        "schema_version": "1.0",
        "source_type": source_type,
        "source_path": str(
            source_path.resolve()
        ),
        "opencv_version": cv2.__version__,
        "source_timestamps_available": (
            source_type == "VIDEO"
            and any(
                item["timestamp_seconds"] is not None
                for item in records
            )
        ),
        "processed_frames": len(records),
        "attempted_pairs": len(pairs),
        "consistent_pairs": len(consistent),
        "consistent_pair_fraction": (
            round(
                len(consistent) / len(pairs),
                5,
            )
            if pairs else None
        ),
        "failed_decodes": failures,
        "coordinate_system":
            "RESIZED_SOURCE_IMAGE_PIXELS",
        "cad_alignment_verified": False,
        "metric_camera_trajectory_available": False,
        "construction_progress_inferred": False,
        "tracking_preview_video_written":
            preview_written,
        "limitations": [
            "Fisheye distortion is not corrected.",
            "Image motion is not camera trajectory.",
            "Frame numbers are not timestamps.",
            "Homography consistency does not establish "
            "metric 3D motion.",
            "No CAD registration is inferred.",
            "No construction status is inferred.",
        ],
        "frames": records,
    }

    json_path = args.out / "analysis.json"

    with json_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            allow_nan=False,
        )

    trajectory_path = (
        args.out / "trajectory.png"
    )

    draw_trajectory(
        positions,
        trajectory_path,
    )

    print("\n=== OPENCV 5 WALKTHROUGH ANALYSIS ===")
    print("OpenCV:", cv2.__version__)
    print("Source:", source_path)
    print("Source type:", source_type)
    print("Processed frames:", len(records))
    print("Attempted pairs:", len(pairs))
    print("Consistent pairs:", len(consistent))
    print(
        "Consistency fraction:",
        report["consistent_pair_fraction"],
    )
    print("Failed decodes:", len(failures))
    print("JSON:", json_path)
    print("Trajectory:", trajectory_path)
    print(
        "Tracking preview frames:",
        preview_frames_dir,
    )
    print("Preview video written:", preview_written)
    print("CAD alignment verified: NO")
    print("Construction progress inferred: NO")


def main():
    parser = argparse.ArgumentParser()

    sources = parser.add_mutually_exclusive_group(
        required=True
    )

    sources.add_argument(
        "--frames",
        type=Path,
    )

    sources.add_argument(
        "--video",
        type=Path,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--max-width",
        type=int,
        default=960,
    )

    parser.add_argument(
        "--sample-every",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--keyframe-every",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--preview-fps",
        type=float,
        default=None,
        help=(
            "Optional presentation playback rate. "
            "Not a source timestamp."
        ),
    )

    args = parser.parse_args()

    if args.sample_every < 1:
        parser.error("--sample-every must be >= 1")

    if args.keyframe_every < 1:
        parser.error("--keyframe-every must be >= 1")

    if args.max_width < 64:
        parser.error("--max-width must be >= 64")

    if (
        args.preview_fps is not None
        and args.preview_fps <= 0
    ):
        parser.error("--preview-fps must be positive")

    run(args)


if __name__ == "__main__":
    main()
