
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "floor_2_2025-10-28_run_2"

DATA = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

FRAMES = (
    DATA / "hilti_frames"
    / SEQUENCE / "cam0" / "frames"
)

MANIFEST = (
    DATA / "hilti_frames"
    / SEQUENCE / "cam0" / "frame_manifest.csv"
)

CALIBRATION = (
    DATA / "hilti_calibration"
    / "camera_calibration_report.json"
)

OUTPUT = (
    ROOT / "outputs"
    / "multiview_reconstruction"
)

IMAGE_WIDTH = 960
FEATURE_COUNT = 3000
MATCH_RATIO = 0.78

MIN_MATCHES = 30
MIN_EPIPOLAR_INLIERS = 20

MIN_BASELINE_M = 0.08
MAX_BASELINE_M = 5.0

EPIPOLAR_THRESHOLD_PX = 2.5
REPROJECTION_THRESHOLD_PX = 3.0

MIN_PARALLAX_DEG = 1.0
MAX_DEPTH_M = 100.0

VOXEL_SIZE_M = 0.05


def load_calibration():
    report = json.loads(
        CALIBRATION.read_text(
            encoding="utf-8"
        )
    )

    data = report["calibration"]

    fx, fy, cx, cy = map(
        float,
        data["intrinsics"],
    )

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    D = np.asarray(
        data["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(4, 1)

    return K, D


def quaternion_to_rotation(q):
    x, y, z, w = q

    return np.array(
        [
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
        ],
        dtype=np.float64,
    )


def load_poses():
    poses = {}

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if (
                row.get("pose_matched", "").lower()
                != "true"
            ):
                continue

            try:
                position = np.array(
                    [
                        float(row[key])
                        for key in (
                            "tx", "ty", "tz"
                        )
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        float(row[key])
                        for key in (
                            "qx", "qy", "qz", "qw"
                        )
                    ],
                    dtype=np.float64,
                )

                norm = np.linalg.norm(
                    quaternion
                )

                if (
                    not np.isfinite(position).all()
                    or not np.isfinite(quaternion).all()
                    or norm < 1e-10
                ):
                    continue

                quaternion /= norm

                poses[row["filename"]] = {
                    "center": position,
                    "rotation": (
                        quaternion_to_rotation(
                            quaternion
                        )
                    ),
                }

            except (
                KeyError,
                ValueError,
                TypeError,
            ):
                continue

    return poses


def skew(vector):
    x, y, z = vector

    return np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ],
        dtype=np.float64,
    )


def essential_from_poses(
    pose1,
    pose2,
):
    R1 = pose1["rotation"]
    R2 = pose2["rotation"]

    C1 = pose1["center"]
    C2 = pose2["center"]

    relative_R = R2.T @ R1

    relative_t = (
        R2.T @ (C1 - C2)
    )

    baseline = np.linalg.norm(
        relative_t
    )

    if baseline < 1e-10:
        return None

    return (
        skew(relative_t / baseline)
        @ relative_R
    )


def sampson_errors_px(
    points1,
    points2,
    E,
    focal,
):
    x1 = np.column_stack(
        [
            points1,
            np.ones(len(points1)),
        ]
    )

    x2 = np.column_stack(
        [
            points2,
            np.ones(len(points2)),
        ]
    )

    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T

    numerator = (
        np.sum(
            x2 * Ex1,
            axis=1,
        ) ** 2
    )

    denominator = (
        Ex1[:, 0] ** 2
        + Ex1[:, 1] ** 2
        + Etx2[:, 0] ** 2
        + Etx2[:, 1] ** 2
    )

    return (
        np.sqrt(
            numerator
            / np.maximum(
                denominator,
                1e-12,
            )
        )
        * focal
    )


def triangulate(
    points1,
    points2,
    pose1,
    pose2,
):
    R1 = pose1["rotation"]
    R2 = pose2["rotation"]

    C1 = pose1["center"]
    C2 = pose2["center"]

    world_to_camera_1 = R1.T
    world_to_camera_2 = R2.T

    t1 = (
        -world_to_camera_1 @ C1
    ).reshape(3, 1)

    t2 = (
        -world_to_camera_2 @ C2
    ).reshape(3, 1)

    P1 = np.hstack(
        [
            world_to_camera_1,
            t1,
        ]
    )

    P2 = np.hstack(
        [
            world_to_camera_2,
            t2,
        ]
    )

    homogeneous = cv2.triangulatePoints(
        P1,
        P2,
        points1.T,
        points2.T,
    )

    w = homogeneous[3]

    safe = np.abs(w) > 1e-10

    xyz = np.full(
        (len(w), 3),
        np.nan,
        dtype=np.float64,
    )

    xyz[safe] = (
        homogeneous[:3, safe]
        / w[safe]
    ).T

    return xyz


def validate_points(
    xyz,
    p1,
    p2,
    pose1,
    pose2,
    focal,
):
    R1 = pose1["rotation"]
    R2 = pose2["rotation"]

    C1 = pose1["center"]
    C2 = pose2["center"]

    local1 = (
        R1.T @ (xyz - C1).T
    ).T

    local2 = (
        R2.T @ (xyz - C2).T
    ).T

    depth1 = local1[:, 2]
    depth2 = local2[:, 2]

    finite = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(local1).all(axis=1)
        & np.isfinite(local2).all(axis=1)
    )

    positive_depth = (
        (depth1 > 0.05)
        & (depth2 > 0.05)
        & (depth1 < MAX_DEPTH_M)
        & (depth2 < MAX_DEPTH_M)
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        projected1 = (
            local1[:, :2]
            / local1[:, 2:3]
        )

        projected2 = (
            local2[:, :2]
            / local2[:, 2:3]
        )

    error1 = np.linalg.norm(
        projected1 - p1,
        axis=1,
    ) * focal

    error2 = np.linalg.norm(
        projected2 - p2,
        axis=1,
    ) * focal

    ray1 = xyz - C1
    ray2 = xyz - C2

    norm1 = np.linalg.norm(
        ray1, axis=1
    )

    norm2 = np.linalg.norm(
        ray2, axis=1
    )

    cosine = np.sum(
        ray1 * ray2,
        axis=1,
    ) / np.maximum(
        norm1 * norm2,
        1e-12,
    )

    parallax = np.degrees(
        np.arccos(
            np.clip(
                cosine,
                -1.0,
                1.0,
            )
        )
    )

    valid = (
        finite
        & positive_depth
        & np.isfinite(error1)
        & np.isfinite(error2)
        & np.isfinite(parallax)
        & (
            error1
            <= REPROJECTION_THRESHOLD_PX
        )
        & (
            error2
            <= REPROJECTION_THRESHOLD_PX
        )
        & (
            parallax
            >= MIN_PARALLAX_DEG
        )
    )

    return valid, error1, error2, parallax


def write_ply(path, points, colors):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write("ply\n")
        handle.write(
            "format ascii 1.0\n"
        )
        handle.write(
            "comment Pose-guided sparse reconstruction\n"
        )
        handle.write(
            "comment Camera extrinsics unverified\n"
        )
        handle.write(
            "comment Coordinates follow Hilti pose frame\n"
        )
        handle.write(
            f"element vertex {len(points)}\n"
        )
        handle.write(
            "property float x\n"
        )
        handle.write(
            "property float y\n"
        )
        handle.write(
            "property float z\n"
        )
        handle.write(
            "property uchar red\n"
        )
        handle.write(
            "property uchar green\n"
        )
        handle.write(
            "property uchar blue\n"
        )
        handle.write(
            "end_header\n"
        )

        for point, color in zip(
            points, colors
        ):
            x, y, z = map(
                float, point
            )

            r, g, b = map(
                int, color
            )

            handle.write(
                f"{x:.7f} "
                f"{y:.7f} "
                f"{z:.7f} "
                f"{r} {g} {b}\n"
            )


def voxel_fuse(
    points,
    colors,
    voxel_size,
):
    indices = np.floor(
        points / voxel_size
    ).astype(np.int64)

    unique, inverse = np.unique(
        indices,
        axis=0,
        return_inverse=True,
    )

    counts = np.bincount(
        inverse
    ).astype(np.float64)

    fused_points = np.zeros(
        (len(unique), 3),
        dtype=np.float64,
    )

    fused_colors = np.zeros(
        (len(unique), 3),
        dtype=np.float64,
    )

    for axis in range(3):
        fused_points[:, axis] = (
            np.bincount(
                inverse,
                weights=points[:, axis],
            ) / counts
        )

        fused_colors[:, axis] = (
            np.bincount(
                inverse,
                weights=colors[:, axis],
            ) / counts
        )

    return (
        fused_points,
        np.clip(
            np.rint(fused_colors),
            0,
            255,
        ).astype(np.uint8),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-gap",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--voxel",
        type=float,
        default=VOXEL_SIZE_M,
    )

    args = parser.parse_args()

    if (
        args.max_gap < 1
        or args.max_pairs < 1
        or args.voxel <= 0
    ):
        parser.error(
            "Arguments must be positive."
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        FRAMES.glob("*.jpg")
    )

    if not paths:
        raise RuntimeError(
            f"No photographs found: {FRAMES}"
        )

    poses = load_poses()
    K, D = load_calibration()

    detector = cv2.SIFT_create(
        nfeatures=FEATURE_COUNT
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    @lru_cache(maxsize=24)
    def features(index):
        image = cv2.imread(
            str(paths[index])
        )

        if image is None:
            return None

        h, w = image.shape[:2]

        scale = min(
            1.0,
            IMAGE_WIDTH / w,
        )

        if scale < 1:
            image = cv2.resize(
                image,
                (
                    int(round(w * scale)),
                    int(round(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        keypoints, descriptors = (
            detector.detectAndCompute(
                gray, None
            )
        )

        Ks = K.copy()
        Ks[0, :] *= scale
        Ks[1, :] *= scale

        return (
            image,
            keypoints,
            descriptors,
            Ks,
        )

    def match_pair(i, j):
        a = features(i)
        b = features(j)

        if a is None or b is None:
            return None

        image1, kp1, d1, K1 = a
        _, kp2, d2, K2 = b

        if (
            d1 is None
            or d2 is None
            or len(d1) < 2
            or len(d2) < 2
        ):
            return None

        forward = matcher.knnMatch(
            d1, d2, k=2
        )

        backward = matcher.knnMatch(
            d2, d1, k=2
        )

        def ratio_filter(pairs):
            result = {}

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if (
                    first.distance
                    < MATCH_RATIO
                    * second.distance
                ):
                    result[first.queryIdx] = first

            return result

        fwd = ratio_filter(forward)
        rev = ratio_filter(backward)

        matches = []

        for index, match in fwd.items():
            reverse = rev.get(
                match.trainIdx
            )

            if (
                reverse is not None
                and reverse.trainIdx == index
            ):
                matches.append(match)

        if len(matches) < MIN_MATCHES:
            return None

        pixels1 = np.float64(
            [
                kp1[m.queryIdx].pt
                for m in matches
            ]
        )

        pixels2 = np.float64(
            [
                kp2[m.trainIdx].pt
                for m in matches
            ]
        )

        p1 = cv2.fisheye.undistortPoints(
            pixels1.reshape(-1, 1, 2),
            K1,
            D,
        ).reshape(-1, 2)

        p2 = cv2.fisheye.undistortPoints(
            pixels2.reshape(-1, 1, 2),
            K2,
            D,
        ).reshape(-1, 2)

        finite = (
            np.isfinite(p1).all(axis=1)
            & np.isfinite(p2).all(axis=1)
        )

        if finite.sum() < MIN_MATCHES:
            return None

        return {
            "p1": p1[finite],
            "p2": p2[finite],
            "pixels1": pixels1[finite],
            "image1": image1,
            "focal": float(
                (
                    K1[0, 0]
                    + K1[1, 1]
                    + K2[0, 0]
                    + K2[1, 1]
                ) / 4
            ),
        }

    candidates = []

    for i in range(len(paths)):
        pose1 = poses.get(
            paths[i].name
        )

        if pose1 is None:
            continue

        for gap in range(
            1, args.max_gap + 1
        ):
            j = i + gap

            if j >= len(paths):
                continue

            pose2 = poses.get(
                paths[j].name
            )

            if pose2 is None:
                continue

            baseline = float(
                np.linalg.norm(
                    pose1["center"]
                    - pose2["center"]
                )
            )

            if not (
                MIN_BASELINE_M
                <= baseline
                <= MAX_BASELINE_M
            ):
                continue

            candidates.append(
                (i, j, baseline)
            )

    if len(candidates) > args.max_pairs:
        selected = np.linspace(
            0,
            len(candidates) - 1,
            args.max_pairs,
            dtype=int,
        )

        candidates = [
            candidates[int(index)]
            for index in selected
        ]

    print("\nUNDER-CONSTRUCTION")
    print("Pose-guided multi-view reconstruction")
    print("-" * 58)
    print(
        "Photographs:",
        len(paths),
    )
    print(
        "Frames with poses:",
        sum(
            p.name in poses
            for p in paths
        ),
    )
    print(
        "Candidate pairs:",
        len(candidates),
    )
    print(
        "Camera extrinsics: UNVERIFIED"
    )

    all_points = []
    all_colors = []

    pair_results = []
    rejection_counts = Counter()

    for number, (i, j, baseline) in enumerate(
        candidates, start=1
    ):
        name1 = paths[i].name
        name2 = paths[j].name

        pose1 = poses[name1]
        pose2 = poses[name2]

        record = {
            "frame_a": name1,
            "frame_b": name2,
            "baseline_m": baseline,
            "status": "REJECTED",
            "reason": None,
            "matches": 0,
            "epipolar_inliers": 0,
            "points_accepted": 0,
        }

        try:
            matched = match_pair(i, j)

            if matched is None:
                record["reason"] = (
                    "INSUFFICIENT_MATCHES"
                )
                rejection_counts[
                    record["reason"]
                ] += 1
                pair_results.append(record)
                continue

            p1 = matched["p1"]
            p2 = matched["p2"]

            record["matches"] = len(p1)

            E = essential_from_poses(
                pose1,
                pose2,
            )

            if E is None:
                record["reason"] = (
                    "INVALID_POSE"
                )
                rejection_counts[
                    record["reason"]
                ] += 1
                pair_results.append(record)
                continue

            errors = sampson_errors_px(
                p1,
                p2,
                E,
                matched["focal"],
            )

            inliers = (
                np.isfinite(errors)
                & (
                    errors
                    <= EPIPOLAR_THRESHOLD_PX
                )
            )

            record["epipolar_inliers"] = (
                int(inliers.sum())
            )

            if (
                inliers.sum()
                < MIN_EPIPOLAR_INLIERS
            ):
                record["reason"] = (
                    "INSUFFICIENT_EPIPOLAR_INLIERS"
                )
                rejection_counts[
                    record["reason"]
                ] += 1
                pair_results.append(record)
                continue

            filtered1 = p1[inliers]
            filtered2 = p2[inliers]

            xyz = triangulate(
                filtered1,
                filtered2,
                pose1,
                pose2,
            )

            (
                valid,
                error1,
                error2,
                parallax,
            ) = validate_points(
                xyz,
                filtered1,
                filtered2,
                pose1,
                pose2,
                matched["focal"],
            )

            xyz = xyz[valid]

            record["points_accepted"] = (
                len(xyz)
            )

            if len(xyz) < 10:
                record["reason"] = (
                    "INSUFFICIENT_3D_POINTS"
                )
                rejection_counts[
                    record["reason"]
                ] += 1
                pair_results.append(record)
                continue

            sample_pixels = np.rint(
                matched["pixels1"][inliers][valid]
            ).astype(int)

            image = matched["image1"]
            height, width = image.shape[:2]

            sample_pixels[:, 0] = np.clip(
                sample_pixels[:, 0],
                0,
                width - 1,
            )

            sample_pixels[:, 1] = np.clip(
                sample_pixels[:, 1],
                0,
                height - 1,
            )

            bgr = image[
                sample_pixels[:, 1],
                sample_pixels[:, 0],
            ]

            rgb = bgr[:, ::-1].copy()

            all_points.append(xyz)
            all_colors.append(rgb)

            record["status"] = "ACCEPTED"

            record["median_reprojection_error_px"] = (
                float(
                    np.median(
                        (
                            error1[valid]
                            + error2[valid]
                        ) / 2
                    )
                )
            )

            record["median_parallax_deg"] = (
                float(
                    np.median(
                        parallax[valid]
                    )
                )
            )

            pair_results.append(record)

        except (
            cv2.error,
            ValueError,
            FloatingPointError,
        ) as error:
            record["reason"] = (
                "GEOMETRY_ERROR"
            )

            record["error"] = (
                str(error).splitlines()[0]
            )

            rejection_counts[
                record["reason"]
            ] += 1

            pair_results.append(record)

        if (
            number % 25 == 0
            or number == len(candidates)
        ):
            print(
                f"Processed {number}/"
                f"{len(candidates)} pairs | "
                f"accepted="
                f"{sum(r['status'] == 'ACCEPTED' for r in pair_results)}"
            )

    accepted = [
        r for r in pair_results
        if r["status"] == "ACCEPTED"
    ]

    report = {
        "project": "Under-construction",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "reconstruction_type": (
            "POSE_GUIDED_SPARSE_MULTIVIEW"
        ),
        "status": "EXPERIMENTAL",
        "camera_extrinsics_verified": False,
        "metric_geometry_verified": False,
        "cad_registration_verified": False,
        "frames_available": len(paths),
        "frames_with_pose": len(poses),
        "candidate_pairs": len(candidates),
        "accepted_pairs": len(accepted),
        "rejected_pairs": (
            len(pair_results) - len(accepted)
        ),
        "rejection_counts": dict(
            rejection_counts
        ),
        "raw_points": 0,
        "fused_points": 0,
        "voxel_size_m": args.voxel,
        "pair_results": pair_results,
        "limitations": [
            "Camera-to-IMU extrinsics are unavailable.",
            "Identity camera-axis alignment is provisional.",
            "Positions are assumed to be camera centers.",
            "Recorded pose scale is not independently verified.",
            "Pairwise observations are voxel-averaged, "
            "not bundle-adjusted or track-fused.",
            "A point seen in multiple pairs may be "
            "triangulated more than once.",
            "No independent reconstruction ground truth.",
            "No CAD registration.",
            "No dense reconstruction.",
            "No construction progress inferred.",
        ],
    }

    if all_points:
        points = np.vstack(
            all_points
        )

        colors = np.vstack(
            all_colors
        )

        report["raw_points"] = (
            len(points)
        )

        finite = (
            np.isfinite(points).all(axis=1)
        )

        points = points[finite]
        colors = colors[finite]

        fused_points, fused_colors = (
            voxel_fuse(
                points,
                colors,
                args.voxel,
            )
        )

        report["fused_points"] = (
            len(fused_points)
        )

        ply_path = (
            OUTPUT
            / "multiview_sparse_cloud.ply"
        )

        write_ply(
            ply_path,
            fused_points,
            fused_colors,
        )

        print(
            "\nRaw triangulated points:",
            len(points),
        )

        print(
            "Voxel-fused points:",
            len(fused_points),
        )

        print(
            "Point cloud:",
            ply_path,
        )

    else:
        print(
            "\nNo reliable 3D points reconstructed."
        )

    report_path = (
        OUTPUT
        / "multiview_reconstruction_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("-" * 58)
    print(
        "Pairs accepted:",
        len(accepted),
        "/",
        len(candidates),
    )
    print(
        "Rejections:",
        dict(rejection_counts),
    )
    print(
        "Report:",
        report_path,
    )

    print(
        "\nWARNING: This is an experimental "
        "reconstruction. Camera extrinsics and "
        "metric accuracy remain unverified."
    )


if __name__ == "__main__":
    main()
