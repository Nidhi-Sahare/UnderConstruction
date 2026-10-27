
"""
Under-construction — Step 106
Corner-anchored line transfer and 3D hypothesis generation.

This is an image-first experiment. It does NOT reuse
previously reconstructed 3D lines or CAD correspondences.

Pipeline:
  1. Detect LSD segments globally in source frame.
  2. Detect Shi-Tomasi corners globally.
  3. Select corners near each source line, on both sides.
  4. Track corners through three frames with LK + FB checks.
  5. Estimate local affine transforms using RANSAC.
  6. Transfer the source line to frames 2 and 3.
  7. Require independent LSD support in those frames.
  8. Triangulate matched line endpoints using camera poses.
  9. Check 3D endpoints and reprojection in frame 3.

IMPORTANT:
  - Local affine transfer does not establish physical identity.
  - Corners may belong to a different depth surface.
  - The third-frame test is not fully independent because
    frame-3 features help estimate its affine transform.
  - All outputs are unverified hypotheses.

Run:
  .\\.venv\\Scripts\\python.exe tools\\reconstruct_corner_anchored_lines.py
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

CAM_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST = CAM_ROOT / "frame_manifest.csv"
FRAME_DIR = CAM_ROOT / "frames"

OUT = ROOT / "outputs" / "corner_anchored_lines"

SOURCE_WINDOWS = [
    {
        "label": "source_159_region",
        "frames": [
            "cam0_00164.jpg",
            "cam0_00165.jpg",
            "cam0_00166.jpg",
        ],
    },
    {
        "label": "source_159_later",
        "frames": [
            "cam0_00173.jpg",
            "cam0_00174.jpg",
            "cam0_00175.jpg",
        ],
    },
    {
        "label": "source_002_region",
        "frames": [
            "cam0_00007.jpg",
            "cam0_00008.jpg",
            "cam0_00009.jpg",
        ],
    },
]

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
).reshape(4, 1)

MAX_SOURCE_SEGMENTS = 100
MIN_SEGMENT_LENGTH_PX = 35.0

MAX_GLOBAL_CORNERS = 3000
CORNER_QUALITY = 0.01
CORNER_MIN_DISTANCE = 6

LINE_NEIGHBORHOOD_PX = 55.0
LINE_ENDPOINT_EXTENSION_PX = 25.0

MIN_NEARBY_CORNERS = 12
MIN_TRACKED_CORNERS = 8
MIN_RANSAC_INLIERS = 6

MAX_FB_ERROR_PX = 1.5
MAX_LK_STEP_PX = 55.0

AFFINE_RANSAC_THRESHOLD_PX = 3.0
MIN_AFFINE_INLIER_FRACTION = 0.55

MIN_LINE_SUPPORT_FRACTION = 0.50
MAX_LINE_EDGE_DISTANCE_PX = 7.0
MAX_LINE_ANGLE_DEG = 15.0

MIN_TRIANGULATION_BASELINE_M = 0.05
MIN_RAY_ANGLE_DEG = 0.5
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 60.0

MAX_RAY_GAP_M = 0.15
MAX_FRAME3_REPROJECTION_ERROR_PX = 6.0

MIN_RECONSTRUCTED_LENGTH_M = 0.10
MAX_RECONSTRUCTED_LENGTH_M = 15.0

LINE_SAMPLES = 21

cv2.setNumThreads(1)


def save_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def load_poses():
    poses = {}

    with MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        for row in csv.DictReader(handle):
            try:
                name = Path(row["filename"]).name

                position = np.array(
                    [
                        float(row["tx"]),
                        float(row["ty"]),
                        float(row["tz"]),
                    ],
                    dtype=np.float64,
                )

                quaternion = [
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ]

                rotation = Rotation.from_quat(
                    quaternion
                ).as_matrix()

                if not (
                    np.isfinite(position).all()
                    and np.isfinite(rotation).all()
                ):
                    continue

                poses[name] = {
                    "position": position,
                    "rotation": rotation,
                }

            except (
                KeyError,
                ValueError,
                TypeError,
            ):
                continue

    return poses


def read_gray(name):
    return cv2.imread(
        str(FRAME_DIR / name),
        cv2.IMREAD_GRAYSCALE,
    )


def detect_segments(gray, detector):
    raw = detector.detect(gray)[0]

    if raw is None:
        return []

    segments = raw.reshape(-1, 2, 2)

    lengths = np.linalg.norm(
        segments[:, 1] - segments[:, 0],
        axis=1,
    )

    indices = np.flatnonzero(
        lengths >= MIN_SEGMENT_LENGTH_PX
    )

    indices = sorted(
        indices,
        key=lambda i: -float(lengths[i]),
    )[:MAX_SOURCE_SEGMENTS]

    return [
        segments[index].astype(np.float64)
        for index in indices
    ]


def detect_corners(gray):
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=MAX_GLOBAL_CORNERS,
        qualityLevel=CORNER_QUALITY,
        minDistance=CORNER_MIN_DISTANCE,
        blockSize=7,
    )

    if corners is None:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return corners.reshape(-1, 2).astype(
        np.float32
    )


def points_near_line(corners, segment):
    a, b = segment

    direction = b - a

    length = float(
        np.linalg.norm(direction)
    )

    if length < 1e-8:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    unit = direction / length

    relative = corners - a

    along = relative @ unit

    perpendicular = np.abs(
        relative[:, 0] * unit[1]
        - relative[:, 1] * unit[0]
    )

    mask = (
        (along >= -LINE_ENDPOINT_EXTENSION_PX)
        & (
            along
            <= length + LINE_ENDPOINT_EXTENSION_PX
        )
        & (
            perpendicular
            <= LINE_NEIGHBORHOOD_PX
        )
    )

    return corners[mask]


def track_points(image_a, image_b, points):
    source = points.astype(
        np.float32
    ).reshape(-1, 1, 2)

    forward, status, _ = (
        cv2.calcOpticalFlowPyrLK(
            image_a,
            image_b,
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
    )

    if forward is None:
        return None, None

    backward, reverse_status, _ = (
        cv2.calcOpticalFlowPyrLK(
            image_b,
            image_a,
            forward,
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
    )

    if backward is None:
        return None, None

    forward = forward.reshape(-1, 2)
    backward = backward.reshape(-1, 2)

    fb_error = np.linalg.norm(
        points - backward,
        axis=1,
    )

    step = np.linalg.norm(
        forward - points,
        axis=1,
    )

    valid = (
        status.reshape(-1).astype(bool)
        & reverse_status.reshape(-1).astype(bool)
        & np.isfinite(forward).all(axis=1)
        & np.isfinite(backward).all(axis=1)
        & (fb_error <= MAX_FB_ERROR_PX)
        & (step <= MAX_LK_STEP_PX)
    )

    return forward, valid


def estimate_affine(source, target):
    if len(source) < MIN_TRACKED_CORNERS:
        return None

    matrix, inliers = (
        cv2.estimateAffinePartial2D(
            source.astype(np.float32),
            target.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=(
                AFFINE_RANSAC_THRESHOLD_PX
            ),
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )
    )

    if matrix is None or inliers is None:
        return None

    inlier_count = int(
        inliers.sum()
    )

    fraction = (
        inlier_count / len(source)
    )

    if (
        inlier_count < MIN_RANSAC_INLIERS
        or fraction
        < MIN_AFFINE_INLIER_FRACTION
    ):
        return None

    return {
        "matrix": matrix,
        "inliers": inlier_count,
        "inlier_fraction": fraction,
    }


def transform_segment(segment, matrix):
    homogeneous = np.column_stack(
        [
            segment,
            np.ones(2),
        ]
    )

    return (
        homogeneous @ matrix.T
    )


def point_segment_distance(point, a, b):
    ab = b - a

    denominator = float(
        ab @ ab
    )

    if denominator < 1e-12:
        return float(
            np.linalg.norm(point - a)
        )

    t = float(
        np.clip(
            ((point - a) @ ab)
            / denominator,
            0,
            1,
        )
    )

    nearest = a + t * ab

    return float(
        np.linalg.norm(point - nearest)
    )


def angle_difference(a, b):
    a = a / max(
        np.linalg.norm(a),
        1e-12,
    )

    b = b / max(
        np.linalg.norm(b),
        1e-12,
    )

    cosine = abs(
        float(a @ b)
    )

    return math.degrees(
        math.acos(
            float(
                np.clip(cosine, 0, 1)
            )
        )
    )


def line_support(segment, detected):
    t = np.linspace(
        0.05,
        0.95,
        LINE_SAMPLES,
    )

    points = (
        segment[0][None, :]
        + t[:, None]
        * (
            segment[1] - segment[0]
        )[None, :]
    )

    direction = (
        segment[1] - segment[0]
    )

    supported = []

    for point in points:
        found = False

        for other in detected:
            angle = angle_difference(
                direction,
                other[1] - other[0],
            )

            if angle > MAX_LINE_ANGLE_DEG:
                continue

            distance = point_segment_distance(
                point,
                other[0],
                other[1],
            )

            if distance <= MAX_LINE_EDGE_DISTANCE_PX:
                found = True
                break

        supported.append(found)

    return float(
        np.mean(supported)
    )


def camera_ray(pixel, pose):
    undistorted = (
        cv2.fisheye.undistortPoints(
            np.asarray(
                pixel,
                dtype=np.float64,
            ).reshape(1, 1, 2),
            K,
            D,
        ).reshape(2)
    )

    ray_camera = np.array(
        [
            undistorted[0],
            undistorted[1],
            1.0,
        ]
    )

    ray_camera /= np.linalg.norm(
        ray_camera
    )

    ray_map = (
        pose["rotation"].T
        @ ray_camera
    )

    return (
        ray_map
        / np.linalg.norm(ray_map)
    )


def triangulate(
    pixel_a,
    pixel_b,
    pose_a,
    pose_b,
):
    center_a = pose_a["position"]
    center_b = pose_b["position"]

    baseline = float(
        np.linalg.norm(
            center_a - center_b
        )
    )

    if baseline < MIN_TRIANGULATION_BASELINE_M:
        return None

    ray_a = camera_ray(
        pixel_a,
        pose_a,
    )

    ray_b = camera_ray(
        pixel_b,
        pose_b,
    )

    cosine = float(
        np.clip(
            ray_a @ ray_b,
            -1.0,
            1.0,
        )
    )

    angle = math.degrees(
        math.acos(cosine)
    )

    if angle < MIN_RAY_ANGLE_DEG:
        return None

    matrix = np.column_stack(
        [ray_a, -ray_b]
    )

    parameters, _, _, _ = (
        np.linalg.lstsq(
            matrix,
            center_b - center_a,
            rcond=None,
        )
    )

    depth_a = float(
        parameters[0]
    )

    depth_b = float(
        parameters[1]
    )

    if not (
        MIN_DEPTH_M <= depth_a <= MAX_DEPTH_M
        and MIN_DEPTH_M <= depth_b <= MAX_DEPTH_M
    ):
        return None

    point_a = (
        center_a
        + depth_a * ray_a
    )

    point_b = (
        center_b
        + depth_b * ray_b
    )

    gap = float(
        np.linalg.norm(
            point_a - point_b
        )
    )

    if gap > MAX_RAY_GAP_M:
        return None

    return (
        (point_a + point_b) / 2
    )


def project(point, pose):
    camera = (
        point - pose["position"]
    ) @ pose["rotation"]

    if camera[2] <= MIN_DEPTH_M:
        return None

    projected, _ = (
        cv2.fisheye.projectPoints(
            camera.reshape(1, 1, 3),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K,
            D,
        )
    )

    return projected.reshape(2)


def evaluate_segment(
    segment,
    images,
    poses,
    corners,
    segments_b,
    segments_c,
):
    nearby = points_near_line(
        corners,
        segment,
    )

    if len(nearby) < MIN_NEARBY_CORNERS:
        return None, "INSUFFICIENT_LOCAL_CORNERS"

    tracked_b, valid_ab = track_points(
        images[0],
        images[1],
        nearby,
    )

    if tracked_b is None:
        return None, "FIRST_TRACK_FAILED"

    source_b = nearby[valid_ab]
    target_b = tracked_b[valid_ab]

    affine_ab = estimate_affine(
        source_b,
        target_b,
    )

    if affine_ab is None:
        return None, "FIRST_AFFINE_FAILED"

    transferred_b = transform_segment(
        segment,
        affine_ab["matrix"],
    )

    support_b = line_support(
        transferred_b,
        segments_b,
    )

    if support_b < MIN_LINE_SUPPORT_FRACTION:
        return None, "SECOND_FRAME_EDGE_UNSUPPORTED"

    tracked_c, valid_bc = track_points(
        images[1],
        images[2],
        target_b.astype(np.float32),
    )

    if tracked_c is None:
        return None, "SECOND_TRACK_FAILED"

    source_c = target_b[valid_bc]
    target_c = tracked_c[valid_bc]

    affine_bc = estimate_affine(
        source_c,
        target_c,
    )

    if affine_bc is None:
        return None, "SECOND_AFFINE_FAILED"

    transferred_c = transform_segment(
        transferred_b,
        affine_bc["matrix"],
    )

    support_c = line_support(
        transferred_c,
        segments_c,
    )

    if support_c < MIN_LINE_SUPPORT_FRACTION:
        return None, "THIRD_FRAME_EDGE_UNSUPPORTED"

    # Endpoint triangulation is a tentative geometric
    # approximation. Affine-transferred endpoints are not
    # guaranteed physical point correspondences.
    reconstructed = []

    errors = []

    for index in range(2):
        point = triangulate(
            segment[index],
            transferred_b[index],
            poses[0],
            poses[1],
        )

        if point is None:
            return None, "TRIANGULATION_FAILED"

        predicted = project(
            point,
            poses[2],
        )

        if predicted is None:
            return None, "THIRD_PROJECTION_FAILED"

        error = float(
            np.linalg.norm(
                predicted - transferred_c[index]
            )
        )

        if error > MAX_FRAME3_REPROJECTION_ERROR_PX:
            return None, "THIRD_REPROJECTION_FAILED"

        reconstructed.append(
            point
        )

        errors.append(error)

    length = float(
        np.linalg.norm(
            reconstructed[1]
            - reconstructed[0]
        )
    )

    if not (
        MIN_RECONSTRUCTED_LENGTH_M
        <= length
        <= MAX_RECONSTRUCTED_LENGTH_M
    ):
        return None, "INVALID_3D_LINE_LENGTH"

    return {
        "source_segment_px": (
            segment.tolist()
        ),
        "transferred_second_px": (
            transferred_b.tolist()
        ),
        "transferred_third_px": (
            transferred_c.tolist()
        ),
        "endpoints_map": [
            point.tolist()
            for point in reconstructed
        ],
        "length_m": length,
        "second_frame_edge_support": support_b,
        "third_frame_edge_support": support_c,
        "first_affine_inliers": (
            affine_ab["inliers"]
        ),
        "second_affine_inliers": (
            affine_bc["inliers"]
        ),
        "median_third_reprojection_error_px": (
            float(np.median(errors))
        ),
        "physical_line_verified": False,
    }, "RECONSTRUCTED_HYPOTHESIS"


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    poses_by_frame = load_poses()

    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    results = []
    summaries = []

    for window in SOURCE_WINDOWS:
        names = window["frames"]

        images = [
            read_gray(name)
            for name in names
        ]

        if any(
            image is None
            for image in images
        ):
            raise FileNotFoundError(
                f"Missing image: {names}"
            )

        poses = [
            poses_by_frame[name]
            for name in names
        ]

        segments_a = detect_segments(
            images[0],
            detector,
        )

        segments_b = detect_segments(
            images[1],
            detector,
        )

        segments_c = detect_segments(
            images[2],
            detector,
        )

        corners = detect_corners(
            images[0]
        )

        counts = Counter()
        reconstructed = []

        for index, segment in enumerate(
            segments_a
        ):
            result, status = evaluate_segment(
                segment,
                images,
                poses,
                corners,
                segments_b,
                segments_c,
            )

            counts[status] += 1

            if result is None:
                continue

            result["track_id"] = (
                f"{window['label']}_{index:03d}"
            )

            result["source_window"] = (
                window["label"]
            )

            result["frames"] = names

            reconstructed.append(
                result
            )

        results.extend(
            reconstructed
        )

        summaries.append(
            {
                "label": window["label"],
                "frames": names,
                "source_segments": len(
                    segments_a
                ),
                "source_corners": len(
                    corners
                ),
                "reconstructed_hypotheses": len(
                    reconstructed
                ),
                "status_counts": dict(
                    counts
                ),
            }
        )

        print(
            f"{window['label']}: "
            f"segments={len(segments_a)} "
            f"corners={len(corners)} "
            f"hypotheses={len(reconstructed)}",
            flush=True,
        )

    save_json(
        OUT / "reconstructed_line_tracks.json",
        {
            "method": (
                "step106_corner_anchored_line_tracking"
            ),
            "tracks": results,
        },
    )

    report = {
        "method": (
            "step106_corner_anchored_line_tracking"
        ),
        "source_windows": summaries,
        "total_source_segments_tested": sum(
            row["source_segments"]
            for row in summaries
        ),
        "total_reconstructed_hypotheses": len(
            results
        ),
        "previous_3d_candidates_used": False,
        "camera_poses_modified": False,
        "geometry_verified": False,
        "physical_lines_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Nearby corners can lie on surfaces at different depths from the selected image line.",
            "Local affine transformations approximate image motion but do not model arbitrary 3D parallax.",
            "Line endpoints transferred by affine transformations are not guaranteed physical point correspondences.",
            "Third-frame reprojection is compared to an affine-transferred line rather than an independently identified physical line.",
            "LSD proximity support is not proof of physical line identity.",
            "Only three adjacent frames are evaluated per window.",
            "Camera calibration and poses are assumed correct.",
        ],
        "decision": (
            "CORNER_ANCHORED_HYPOTHESES_AVAILABLE"
            if results
            else "NO_ACCEPTED_HYPOTHESES"
        ),
    }

    save_json(
        OUT / "reconstruction_report.json",
        report,
    )

    print("\n=== STEP 106 COMPLETE ===")
    print(
        "Accepted hypotheses:",
        len(results)
    )
    print("Physical lines verified: NO")
    print("CAD registration verified: NO")
    print("Construction progress verified: NO")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
