
"""
Under-construction — Step 105
Image-first line tracking and 3D reconstruction.

Starts from globally detected image lines rather than
projected 3D candidates.

Uses:
    OpenCV LSD
    Lucas-Kanade optical flow
    Forward-backward tracking checks
    Calibrated fisheye undistortion
    Multi-view triangulation
    Held-out third-frame reprojection

Outputs:
    outputs/image_first_lines/
        reconstruction_report.json
        reconstructed_line_tracks.json

No AI, no manual annotation, no CAD alignment.

IMPORTANT:
    Reconstructed lines remain hypotheses.
    Three-frame reprojection is not independent physical
    identity verification.
"""

import csv
import json
import math
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

OUT = ROOT / "outputs" / "image_first_lines"

# These are experimental source windows, not geometric
# associations to existing candidate IDs.
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
        [465.3015482593691, 0, 730.0455886686005],
        [0, 465.32303798346413, 720.1427007671206],
        [0, 0, 1],
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

MIN_SEGMENT_LENGTH_PX = 35.0
MAX_SOURCE_SEGMENTS = 100

POINTS_PER_SEGMENT = 15
ENDPOINT_MARGIN = 0.08

LK_WINDOW = (21, 21)
LK_LEVELS = 3

MAX_FORWARD_BACKWARD_ERROR_PX = 1.5
MAX_TRACKING_STEP_PX = 45.0

MIN_TRACKED_POINTS = 6
MIN_TRACKED_FRACTION = 0.45

MIN_BASELINE_M = 0.05
MIN_TRIANGULATION_ANGLE_DEG = 0.5

MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 60.0

MAX_THIRD_FRAME_ERROR_PX = 6.0
MIN_VALID_TRIANGULATED_POINTS = 5

MIN_3D_LINE_LENGTH_M = 0.10
MAX_3D_LINE_LENGTH_M = 15.0

MAX_3D_LINE_RESIDUAL_M = 0.10

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
                name = Path(
                    row["filename"]
                ).name

                position = np.array(
                    [
                        float(row["tx"]),
                        float(row["ty"]),
                        float(row["tz"]),
                    ],
                    dtype=np.float64,
                )

                rotation = Rotation.from_quat(
                    [
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["qw"]),
                    ]
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
        segments[i].astype(np.float64)
        for i in indices
    ]


def sample_segment(segment):
    t = np.linspace(
        ENDPOINT_MARGIN,
        1.0 - ENDPOINT_MARGIN,
        POINTS_PER_SEGMENT,
    )[:, None]

    return (
        segment[0]
        + t * (segment[1] - segment[0])
    ).astype(np.float32)


def track_points(image_a, image_b, points):
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        image_a,
        image_b,
        points.reshape(-1, 1, 2),
        None,
        winSize=LK_WINDOW,
        maxLevel=LK_LEVELS,
        criteria=(
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if forward is None:
        return None, None

    backward, reverse_status, _ = (
        cv2.calcOpticalFlowPyrLK(
            image_b,
            image_a,
            forward,
            None,
            winSize=LK_WINDOW,
            maxLevel=LK_LEVELS,
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

    original = points.reshape(-1, 2)

    fb_error = np.linalg.norm(
        original - backward,
        axis=1,
    )

    step = np.linalg.norm(
        forward - original,
        axis=1,
    )

    valid = (
        status.reshape(-1).astype(bool)
        & reverse_status.reshape(-1).astype(bool)
        & np.isfinite(forward).all(axis=1)
        & np.isfinite(backward).all(axis=1)
        & (fb_error <= MAX_FORWARD_BACKWARD_ERROR_PX)
        & (step <= MAX_TRACKING_STEP_PX)
    )

    return forward, valid


def undistort_points(pixels):
    return cv2.fisheye.undistortPoints(
        np.asarray(
            pixels,
            dtype=np.float64,
        ).reshape(-1, 1, 2),
        K,
        D,
    ).reshape(-1, 2)


def camera_ray(pixel, pose):
    normalized = undistort_points(
        np.asarray(pixel).reshape(1, 2)
    )[0]

    direction_camera = np.array(
        [
            normalized[0],
            normalized[1],
            1.0,
        ],
        dtype=np.float64,
    )

    direction_camera /= np.linalg.norm(
        direction_camera
    )

    # Frozen convention: map-to-camera uses row @ R.
    # Therefore camera-to-map direction is R.T @ ray.
    direction_map = (
        pose["rotation"].T
        @ direction_camera
    )

    return (
        direction_map
        / np.linalg.norm(direction_map)
    )


def triangulate_two_rays(
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

    if baseline < MIN_BASELINE_M:
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

    if angle < MIN_TRIANGULATION_ANGLE_DEG:
        return None

    matrix = np.column_stack(
        [ray_a, -ray_b]
    )

    rhs = center_b - center_a

    parameters, _, _, _ = (
        np.linalg.lstsq(
            matrix,
            rhs,
            rcond=None,
        )
    )

    distance_a = float(
        parameters[0]
    )

    distance_b = float(
        parameters[1]
    )

    if not (
        MIN_DEPTH_M <= distance_a <= MAX_DEPTH_M
        and MIN_DEPTH_M <= distance_b <= MAX_DEPTH_M
    ):
        return None

    point_a = (
        center_a
        + distance_a * ray_a
    )

    point_b = (
        center_b
        + distance_b * ray_b
    )

    gap = float(
        np.linalg.norm(
            point_a - point_b
        )
    )

    if gap > 0.15:
        return None

    return {
        "point": (
            (point_a + point_b) / 2
        ),
        "ray_angle_deg": angle,
        "ray_gap_m": gap,
    }


def project_point(point, pose):
    camera = (
        point - pose["position"]
    ) @ pose["rotation"]

    if (
        not np.isfinite(camera).all()
        or camera[2] <= MIN_DEPTH_M
    ):
        return None

    pixel, _ = cv2.fisheye.projectPoints(
        camera.reshape(1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        D,
    )

    return pixel.reshape(2)


def fit_3d_line(points):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    if len(points) < MIN_VALID_TRIANGULATED_POINTS:
        return None

    center = points.mean(axis=0)

    centered = points - center

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    direction = vh[0]

    coordinates = (
        centered @ direction
    )

    endpoints = np.array([
        center + coordinates.min() * direction,
        center + coordinates.max() * direction,
    ])

    residuals = np.linalg.norm(
        centered
        - np.outer(
            coordinates,
            direction,
        ),
        axis=1,
    )

    length = float(
        np.linalg.norm(
            endpoints[1] - endpoints[0]
        )
    )

    median_residual = float(
        np.median(residuals)
    )

    if not (
        MIN_3D_LINE_LENGTH_M
        <= length
        <= MAX_3D_LINE_LENGTH_M
    ):
        return None

    if median_residual > MAX_3D_LINE_RESIDUAL_M:
        return None

    return {
        "endpoints_map": endpoints.tolist(),
        "length_m": length,
        "median_line_residual_m": median_residual,
    }


def evaluate_segment(
    segment,
    images,
    poses,
):
    first, second, third = images

    pose_a, pose_b, pose_c = poses

    points_a = sample_segment(segment)

    points_b, valid_ab = track_points(
        first,
        second,
        points_a,
    )

    if points_b is None:
        return None, "FIRST_TRACK_FAILED"

    if valid_ab.sum() < MIN_TRACKED_POINTS:
        return None, "FIRST_TRACK_FAILED"

    if valid_ab.mean() < MIN_TRACKED_FRACTION:
        return None, "FIRST_TRACK_FAILED"

    points_c, valid_bc = track_points(
        second,
        third,
        points_b.astype(np.float32),
    )

    if points_c is None:
        return None, "SECOND_TRACK_FAILED"

    valid = (
        valid_ab
        & valid_bc
    )

    if valid.sum() < MIN_TRACKED_POINTS:
        return None, "SECOND_TRACK_FAILED"

    triangulated = []
    reprojection_errors = []
    ray_angles = []
    ray_gaps = []

    for index in np.flatnonzero(valid):
        result = triangulate_two_rays(
            points_a[index],
            points_b[index],
            pose_a,
            pose_b,
        )

        if result is None:
            continue

        point = result["point"]

        predicted = project_point(
            point,
            pose_c,
        )

        if predicted is None:
            continue

        error = float(
            np.linalg.norm(
                predicted - points_c[index]
            )
        )

        if error > MAX_THIRD_FRAME_ERROR_PX:
            continue

        triangulated.append(point)
        reprojection_errors.append(error)
        ray_angles.append(
            result["ray_angle_deg"]
        )
        ray_gaps.append(
            result["ray_gap_m"]
        )

    if (
        len(triangulated)
        < MIN_VALID_TRIANGULATED_POINTS
    ):
        return None, "INSUFFICIENT_VALID_3D_POINTS"

    fitted = fit_3d_line(
        triangulated
    )

    if fitted is None:
        return None, "THREE_DIMENSIONAL_LINE_FIT_FAILED"

    return {
        "source_segment": (
            segment.tolist()
        ),
        "source_sample_count": len(
            points_a
        ),
        "tracked_three_frames": int(
            valid.sum()
        ),
        "triangulated_points": len(
            triangulated
        ),
        "median_heldout_reprojection_error_px": float(
            np.median(reprojection_errors)
        ),
        "median_ray_angle_deg": float(
            np.median(ray_angles)
        ),
        "median_ray_gap_m": float(
            np.median(ray_gaps)
        ),
        **fitted,
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

    all_tracks = []
    window_summaries = []

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
                f"Missing image in {names}"
            )

        try:
            poses = [
                poses_by_frame[name]
                for name in names
            ]
        except KeyError as exc:
            raise RuntimeError(
                f"Missing camera pose: {exc}"
            )

        segments = detect_segments(
            images[0],
            detector,
        )

        status_counts = {}

        reconstructed = []

        for index, segment in enumerate(segments):
            result, status = evaluate_segment(
                segment,
                images,
                poses,
            )

            status_counts[status] = (
                status_counts.get(status, 0)
                + 1
            )

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

        all_tracks.extend(
            reconstructed
        )

        summary = {
            "label": window["label"],
            "frames": names,
            "source_segments_tested": len(
                segments
            ),
            "reconstructed_hypotheses": len(
                reconstructed
            ),
            "status_counts": status_counts,
        }

        window_summaries.append(
            summary
        )

        print(
            f"{window['label']}: "
            f"segments={len(segments)} "
            f"reconstructed={len(reconstructed)}",
            flush=True,
        )

    save_json(
        OUT / "reconstructed_line_tracks.json",
        {
            "method": (
                "step105_image_first_line_tracking"
            ),
            "tracks": all_tracks,
        },
    )

    report = {
        "method": (
            "step105_image_first_line_tracking"
        ),
        "source_windows": window_summaries,
        "total_source_segments_tested": sum(
            item["source_segments_tested"]
            for item in window_summaries
        ),
        "total_reconstructed_hypotheses": len(
            all_tracks
        ),
        "existing_3d_candidates_used_for_selection": False,
        "existing_3d_candidates_used_for_fitting": False,
        "camera_poses_modified": False,
        "cad_registration_verified": False,
        "physical_lines_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Lucas-Kanade tracking may drift between adjacent frames.",
            "Tracking points sampled on an edge can suffer from the aperture problem and are not independently distinctive.",
            "The third frame is temporally adjacent, not an independent recording or independent physical identity check.",
            "The 3D line fit is performed after filtering triangulated points using third-frame reprojection error.",
            "The held-out reprojection error is therefore selection-conditioned, not an unbiased held-out evaluation.",
            "Camera poses and fisheye calibration are assumed correct.",
            "Reconstructed line hypotheses are not linked to CAD elements.",
        ],
        "decision": (
            "IMAGE_FIRST_RECONSTRUCTION_HYPOTHESES_AVAILABLE"
        ),
    }

    save_json(
        OUT / "reconstruction_report.json",
        report,
    )

    print("\n=== STEP 105 COMPLETE ===")
    print(
        "Reconstructed hypotheses:",
        len(all_tracks)
    )
    print("Physical lines verified: NO")
    print("CAD registration verified: NO")
    print("Construction progress verified: NO")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
