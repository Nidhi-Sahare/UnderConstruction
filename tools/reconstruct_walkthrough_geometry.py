
"""
Under-construction — Step 87
Independent sparse 3D reconstruction from walkthrough images.

Uses:
    - OpenCV SIFT or ORB
    - Recorded T_map_cam0 poses
    - Official cam0 fisheye calibration
    - Multi-view triangulation
    - Reprojection and parallax checks

Does NOT:
    - Use CAD registration candidates
    - Claim construction progress
    - Verify wall visibility
    - Use AI/LLMs or manual annotations

Run:
    python tools/reconstruct_walkthrough_geometry.py

Output:
    outputs/walkthrough_reconstruction/
        reconstruction_report.json
        reconstructed_points.ply
"""

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

FRAME_DIR = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
)

MANIFEST = FRAME_DIR / "frame_manifest.csv"

OUTPUT_DIR = ROOT / "outputs" / "walkthrough_reconstruction"

# Official Hilti cam0 fisheye calibration.
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

IMAGE_SCALE = 0.5

MAX_FRAMES = 60
MAX_FEATURES = 2500
MAX_PAIR_GAP = 3

MIN_BASELINE_M = 0.40
MAX_BASELINE_M = 8.0

RATIO_TEST = 0.75
MIN_PARALLAX_DEG = 1.5
MAX_REPROJECTION_PX = 3.0

MIN_DEPTH_M = 0.2
MAX_DEPTH_M = 60.0

VOXEL_SIZE_M = 0.10

# These are diagnostic thresholds, not registration verification.
MIN_RECONSTRUCTED_POINTS = 100
MIN_SPATIAL_CELLS = 8
SPATIAL_CELL_SIZE_M = 2.0


def is_true(value):
    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def load_frames():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST}"
        )

    frames = []

    with MANIFEST.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if not is_true(row.get("pose_matched")):
                continue

            filename = row["filename"]
            image_path = FRAME_DIR / "frames" / filename

            if not image_path.exists():
                continue

            try:
                position = np.array(
                    [
                        float(row["tx"]),
                        float(row["ty"]),
                        float(row["tz"]),
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                        float(row["qw"]),
                    ],
                    dtype=np.float64,
                )

                if (
                    not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(quaternion))
                    or np.linalg.norm(quaternion) < 1e-8
                ):
                    continue

                quaternion /= np.linalg.norm(quaternion)

                # Camera-to-map rotation.
                rotation = Rotation.from_quat(
                    quaternion
                ).as_matrix()

            except (ValueError, KeyError):
                continue

            frames.append(
                {
                    "filename": filename,
                    "image_path": image_path,
                    "position": position,
                    "rotation": rotation,
                }
            )

    if len(frames) > MAX_FRAMES:
        indices = np.linspace(
            0,
            len(frames) - 1,
            MAX_FRAMES,
        ).round().astype(int)

        frames = [frames[i] for i in indices]

    return frames


def make_detector():
    if hasattr(cv2, "SIFT_create"):
        return (
            cv2.SIFT_create(
                nfeatures=MAX_FEATURES
            ),
            cv2.NORM_L2,
            "SIFT",
        )

    return (
        cv2.ORB_create(
            nfeatures=MAX_FEATURES
        ),
        cv2.NORM_HAMMING,
        "ORB",
    )


def extract_features(frame, detector):
    image = cv2.imread(
        str(frame["image_path"]),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        return None

    image = cv2.resize(
        image,
        None,
        fx=IMAGE_SCALE,
        fy=IMAGE_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    keypoints, descriptors = (
        detector.detectAndCompute(image, None)
    )

    if descriptors is None or len(keypoints) == 0:
        return None

    pixels = np.array(
        [kp.pt for kp in keypoints],
        dtype=np.float64,
    )

    # Return points to original-resolution coordinates
    # before fisheye undistortion.
    pixels /= IMAGE_SCALE

    undistorted = cv2.fisheye.undistortPoints(
        pixels.reshape(-1, 1, 2),
        K,
        D,
    ).reshape(-1, 2)

    rays = np.column_stack(
        [
            undistorted,
            np.ones(len(undistorted)),
        ]
    )

    rays /= np.linalg.norm(
        rays,
        axis=1,
        keepdims=True,
    )

    return {
        "descriptors": descriptors,
        "pixels": pixels,
        "rays": rays,
        "keypoint_count": len(keypoints),
    }


def match_features(a, b, norm_type):
    matcher = cv2.BFMatcher(
        norm_type,
        crossCheck=False,
    )

    forward = matcher.knnMatch(
        a["descriptors"],
        b["descriptors"],
        k=2,
    )

    backward = matcher.knnMatch(
        b["descriptors"],
        a["descriptors"],
        k=2,
    )

    reverse = {}

    for neighbors in backward:
        if len(neighbors) < 2:
            continue

        first, second = neighbors

        if first.distance < RATIO_TEST * second.distance:
            reverse[first.queryIdx] = first.trainIdx

    matches = []

    for neighbors in forward:
        if len(neighbors) < 2:
            continue

        first, second = neighbors

        if first.distance >= RATIO_TEST * second.distance:
            continue

        if reverse.get(first.trainIdx) != first.queryIdx:
            continue

        matches.append(
            (first.queryIdx, first.trainIdx)
        )

    return matches


def triangulate_pair(frame_a, frame_b, features_a, features_b, matches):
    if not matches:
        return np.empty((0, 3), dtype=np.float64), {
            "raw_matches": 0,
            "geometrically_valid": 0,
            "accepted_points": 0,
            "median_reprojection_error_px": None,
            "median_parallax_deg": None,
        }

    idx_a = np.array(
        [m[0] for m in matches],
        dtype=np.int32,
    )

    idx_b = np.array(
        [m[1] for m in matches],
        dtype=np.int32,
    )

    ray_a = features_a["rays"][idx_a]
    ray_b = features_b["rays"][idx_b]

    ca = frame_a["position"]
    cb = frame_b["position"]

    ra = frame_a["rotation"]
    rb = frame_b["rotation"]

    # Convert camera-coordinate rays into map coordinates.
    world_a = (ra @ ray_a.T).T
    world_b = (rb @ ray_b.T).T

    cosine = np.clip(
        np.sum(world_a * world_b, axis=1),
        -1.0,
        1.0,
    )

    parallax = np.degrees(
        np.arccos(cosine)
    )

    # Closest approach of two world-space rays.
    # ca + s * world_a ~= cb + t * world_b
    # Solve for positive ray distances s and t.
    displacement = cb - ca

    dot = np.sum(
        world_a * world_b,
        axis=1,
    )

    rhs_a = np.sum(
        world_a * displacement,
        axis=1,
    )

    rhs_b = np.sum(
        world_b * displacement,
        axis=1,
    )

    denominator = 1.0 - dot * dot

    stable = denominator > 1e-8

    s = np.full(
        len(matches),
        np.nan,
        dtype=np.float64,
    )

    t = np.full_like(s, np.nan)

    s[stable] = (
        rhs_a[stable]
        - dot[stable] * rhs_b[stable]
    ) / denominator[stable]

    t[stable] = (
        dot[stable] * rhs_a[stable]
        - rhs_b[stable]
    ) / denominator[stable]

    pa = ca + s[:, None] * world_a
    pb = cb + t[:, None] * world_b

    points = (pa + pb) / 2.0

    # Project reconstructed points back to both cameras.
    camera_a = (ra.T @ (points - ca).T).T
    camera_b = (rb.T @ (points - cb).T).T

    depth_a = camera_a[:, 2]
    depth_b = camera_b[:, 2]

    valid = (
        stable
        & np.isfinite(points).all(axis=1)
        & (parallax >= MIN_PARALLAX_DEG)
        & (s >= MIN_DEPTH_M)
        & (t >= MIN_DEPTH_M)
        & (s <= MAX_DEPTH_M)
        & (t <= MAX_DEPTH_M)
        & (depth_a > 0)
        & (depth_b > 0)
    )

    errors = np.full(
        len(matches),
        np.inf,
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(valid)

    if len(valid_indices):
        projected_a, _ = cv2.fisheye.projectPoints(
            camera_a[valid_indices].reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            K,
            D,
        )

        projected_b, _ = cv2.fisheye.projectPoints(
            camera_b[valid_indices].reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            K,
            D,
        )

        projected_a = projected_a.reshape(-1, 2)
        projected_b = projected_b.reshape(-1, 2)

        observed_a = features_a["pixels"][
            idx_a[valid_indices]
        ]

        observed_b = features_b["pixels"][
            idx_b[valid_indices]
        ]

        error_a = np.linalg.norm(
            projected_a - observed_a,
            axis=1,
        )

        error_b = np.linalg.norm(
            projected_b - observed_b,
            axis=1,
        )

        errors[valid_indices] = np.maximum(
            error_a,
            error_b,
        )

    accepted = (
        valid
        & (errors <= MAX_REPROJECTION_PX)
    )

    accepted_errors = errors[accepted]

    return (
        points[accepted],
        {
            "raw_matches": len(matches),
            "geometrically_valid": int(
                np.count_nonzero(valid)
            ),
            "accepted_points": int(
                np.count_nonzero(accepted)
            ),
            "median_reprojection_error_px": (
                float(np.median(accepted_errors))
                if len(accepted_errors)
                else None
            ),
            "median_parallax_deg": (
                float(np.median(parallax[accepted]))
                if np.any(accepted)
                else None
            ),
        },
    )


def deduplicate_points(points):
    if len(points) == 0:
        return points, np.array([], dtype=np.int32)

    voxels = np.floor(
        points / VOXEL_SIZE_M
    ).astype(np.int64)

    unique_voxels, inverse = np.unique(
        voxels,
        axis=0,
        return_inverse=True,
    )

    sums = np.zeros(
        (len(unique_voxels), 3),
        dtype=np.float64,
    )

    counts = np.bincount(
        inverse,
        minlength=len(unique_voxels),
    )

    np.add.at(sums, inverse, points)

    return sums / counts[:, None], counts


def spatial_coverage(points):
    if not len(points):
        return 0

    cells = np.floor(
        points / SPATIAL_CELL_SIZE_M
    ).astype(np.int64)

    return int(
        len(np.unique(cells, axis=0))
    )


def write_ply(path, points):
    with path.open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(
            f"element vertex {len(points)}\n"
        )
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")

        for point in points:
            handle.write(
                f"{point[0]:.6f} "
                f"{point[1]:.6f} "
                f"{point[2]:.6f}\n"
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_frames()

    if len(frames) < 2:
        raise RuntimeError(
            "Not enough pose-matched walkthrough frames."
        )

    detector, norm_type, detector_name = (
        make_detector()
    )

    print("\n=== STEP 87: SPARSE RECONSTRUCTION ===")
    print("Frames:", len(frames))
    print("Feature detector:", detector_name)

    features = [
        extract_features(frame, detector)
        for frame in frames
    ]

    all_points = []
    pair_results = []

    for i in range(len(frames)):
        if features[i] is None:
            continue

        for gap in range(1, MAX_PAIR_GAP + 1):
            j = i + gap

            if j >= len(frames):
                continue

            if features[j] is None:
                continue

            baseline = float(
                np.linalg.norm(
                    frames[i]["position"]
                    - frames[j]["position"]
                )
            )

            if not (
                MIN_BASELINE_M
                <= baseline
                <= MAX_BASELINE_M
            ):
                continue

            matches = match_features(
                features[i],
                features[j],
                norm_type,
            )

            points, diagnostics = triangulate_pair(
                frames[i],
                frames[j],
                features[i],
                features[j],
                matches,
            )

            if len(points):
                all_points.append(points)

            pair_results.append(
                {
                    "frame_a": frames[i]["filename"],
                    "frame_b": frames[j]["filename"],
                    "baseline_m": baseline,
                    **diagnostics,
                }
            )

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/{len(frames)} frames",
                flush=True,
            )

    raw_points = (
        np.concatenate(all_points, axis=0)
        if all_points
        else np.empty(
            (0, 3),
            dtype=np.float64,
        )
    )

    unique_points, observation_counts = (
        deduplicate_points(raw_points)
    )

    occupied_cells = spatial_coverage(
        unique_points
    )

    reconstruction_usable = (
        len(unique_points)
        >= MIN_RECONSTRUCTED_POINTS
        and occupied_cells
        >= MIN_SPATIAL_CELLS
    )

    ply_path = (
        OUTPUT_DIR / "reconstructed_points.ply"
    )

    write_ply(
        ply_path,
        unique_points,
    )

    successful_pairs = [
        result
        for result in pair_results
        if result["accepted_points"] > 0
    ]

    errors = [
        result["median_reprojection_error_px"]
        for result in successful_pairs
        if result["median_reprojection_error_px"]
        is not None
    ]

    report = {
        "sequence": SEQUENCE,
        "method": (
            "pose_constrained_sparse_feature_triangulation"
        ),
        "detector": detector_name,
        "frames_used": len(frames),
        "evaluated_pairs": len(pair_results),
        "successful_pairs": len(successful_pairs),
        "raw_triangulated_observations": len(raw_points),
        "unique_voxel_points": len(unique_points),
        "occupied_spatial_cells": occupied_cells,
        "median_pair_reprojection_error_px": (
            float(np.median(errors))
            if errors
            else None
        ),
        "reconstruction_quality_gate_passed": (
            reconstruction_usable
        ),
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "max_frames": MAX_FRAMES,
            "max_pair_gap": MAX_PAIR_GAP,
            "image_scale": IMAGE_SCALE,
            "min_baseline_m": MIN_BASELINE_M,
            "max_baseline_m": MAX_BASELINE_M,
            "min_parallax_deg": MIN_PARALLAX_DEG,
            "max_reprojection_error_px": (
                MAX_REPROJECTION_PX
            ),
            "voxel_size_m": VOXEL_SIZE_M,
            "min_reconstructed_points": (
                MIN_RECONSTRUCTED_POINTS
            ),
            "min_spatial_cells": MIN_SPATIAL_CELLS,
        },
        "pair_results": pair_results,
        "limitations": [
            "Sparse features are not necessarily wall features.",
            "Voxel deduplication is not multi-view track fusion.",
            "Two-view matches may include moving objects.",
            "Camera trajectory uncertainty is not modeled.",
            "Passing the quality gate does not validate CAD registration.",
            "No explicit structural line reconstruction is performed.",
            "No construction element is classified as built or missing.",
        ],
        "decision": (
            "SPARSE_GEOMETRY_AVAILABLE_FOR_FURTHER_TESTING"
            if reconstruction_usable
            else "INSUFFICIENT_SPARSE_GEOMETRY"
        ),
    }

    report_path = (
        OUTPUT_DIR / "reconstruction_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print("Evaluated pairs:", len(pair_results))
    print("Successful pairs:", len(successful_pairs))
    print("Raw observations:", len(raw_points))
    print("Unique voxel points:", len(unique_points))
    print("Spatial cells:", occupied_cells)
    print(
        "Reconstruction quality gate:",
        "PASS" if reconstruction_usable else "FAIL",
    )
    print("Registration verified: NO")
    print("Report:", report_path)
    print("Point cloud:", ply_path)


if __name__ == "__main__":
    main()
