
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

ARCHIVE = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

FRAMES = (
    ARCHIVE / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0" / "frames"
)

POSE_FILE = (
    ROOT / "outputs"
    / "rotation_new_feature_validation"
    / "experimental_corrected_camera_poses.json"
)

LANDMARK_FILE = (
    ROOT / "outputs"
    / "multiview_tracks_v2"
    / "validated_landmarks_v2.json"
)

OUTPUT = (
    ROOT / "outputs"
    / "controlled_triangulation_v3"
)

# Previously reported equidistant fisheye calibration.
# These are calibration data, not estimated parameters.
K = np.array([
    [465.3015482593691, 0, 730.0455886686005],
    [0, 465.32303798346413, 720.1427007671206],
    [0, 0, 1],
], dtype=np.float64)

D = np.array([
    0.025800718903376804,
    -0.010909240777406872,
    -0.0016899537986031076,
    0.00014766801645260894,
], dtype=np.float64).reshape(4, 1)

FOCAL = float((K[0, 0] + K[1, 1]) / 2)


def frame_number(name):
    return int(Path(name).stem.rsplit("_", 1)[-1])


def load_poses():
    data = json.loads(
        POSE_FILE.read_text(encoding="utf-8")
    )

    if data.get("status") != "EXPERIMENTAL":
        raise RuntimeError(
            "Unexpected pose-file status."
        )

    result = {}

    for name, record in data["poses"].items():
        center = np.asarray(
            record["center"], dtype=float
        )

        original = np.asarray(
            record["original_rotation_camera_to_world"],
            dtype=float,
        )

        refined = np.asarray(
            record["rotation_camera_to_world"],
            dtype=float,
        )

        if (
            center.shape != (3,)
            or original.shape != (3, 3)
            or refined.shape != (3, 3)
        ):
            raise ValueError(
                f"Invalid camera pose: {name}"
            )

        for R in (original, refined):
            if (
                not np.isfinite(R).all()
                or np.linalg.det(R) < 0.99
                or np.linalg.det(R) > 1.01
                or not np.allclose(
                    R.T @ R, np.eye(3), atol=1e-3
                )
            ):
                raise ValueError(
                    f"Invalid rotation matrix: {name}"
                )

        result[name] = {
            "center": center,
            "original": original,
            "refined": refined,
        }

    return dict(
        sorted(
            result.items(),
            key=lambda item: frame_number(item[0]),
        )
    )


def load_exclusion_observations():
    """
    Extract original V2 observation coordinates.

    These coordinates are normalized undistorted
    image coordinates, as exported by V2.
    """
    document = json.loads(
        LANDMARK_FILE.read_text(encoding="utf-8")
    )

    if isinstance(document, list):
        landmarks = document

    elif isinstance(document, dict):
        landmarks = next(
            (
                document[key]
                for key in (
                    "landmarks",
                    "validated_landmarks",
                    "points",
                )
                if isinstance(document.get(key), list)
            ),
            None,
        )
    else:
        landmarks = None

    if landmarks is None:
        raise RuntimeError(
            "Cannot locate V2 landmark list. "
            "Do not run without exclusion observations."
        )

    excluded = defaultdict(list)
    count = 0

    for landmark in landmarks:
        observations = landmark.get(
            "observations", []
        )

        for obs in observations:
            name = (
                obs.get("frame")
                or obs.get("filename")
            )

            uv = obs.get("normalized_uv")

            if name is None or uv is None:
                continue

            name = Path(str(name)).name

            uv = np.asarray(
                uv, dtype=float
            )

            if (
                uv.shape != (2,)
                or not np.isfinite(uv).all()
            ):
                continue

            excluded[name].append(uv)
            count += 1

    if count < 100:
        raise RuntimeError(
            "Too few V2 observations found. "
            "Feature exclusion cannot be trusted."
        )

    return {
        name: np.asarray(values)
        for name, values in excluded.items()
    }, count


def extract_features(name, sift, width):
    image = cv2.imread(
        str(FRAMES / name),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise FileNotFoundError(FRAMES / name)

    scale = min(1.0, width / image.shape[1])

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                round(image.shape[1] * scale),
                round(image.shape[0] * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    keypoints, descriptors = (
        sift.detectAndCompute(image, None)
    )

    if descriptors is None:
        return {
            "uv": np.empty((0, 2)),
            "descriptors": None,
        }

    xy = np.asarray(
        [keypoint.pt for keypoint in keypoints],
        dtype=np.float64,
    )

    scaled_K = K.copy()
    scaled_K[:2, :] *= scale

    uv = cv2.fisheye.undistortPoints(
        xy.reshape(-1, 1, 2),
        scaled_K,
        D,
    ).reshape(-1, 2)

    return {
        "uv": uv,
        "descriptors": descriptors,
    }


def match_features(a, b, ratio):
    if (
        a["descriptors"] is None
        or b["descriptors"] is None
        or len(a["descriptors"]) < 2
        or len(b["descriptors"]) < 2
    ):
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
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

    accepted_forward = {}

    for pair in forward:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                accepted_forward[m.queryIdx] = m.trainIdx

    accepted_backward = {}

    for pair in backward:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                accepted_backward[m.queryIdx] = m.trainIdx

    return [
        (i, j)
        for i, j in accepted_forward.items()
        if accepted_backward.get(j) == i
    ]


def exclude_saved_features(
    uv, saved, radius
):
    if len(uv) == 0:
        return np.zeros(0, dtype=bool)

    if saved is None or len(saved) == 0:
        return np.ones(len(uv), dtype=bool)

    keep = np.ones(len(uv), dtype=bool)

    for start in range(0, len(uv), 256):
        end = min(start + 256, len(uv))

        delta = (
            uv[start:end, None, :]
            - saved[None, :, :]
        )

        distance_sq = np.sum(
            delta ** 2, axis=2
        )

        keep[start:end] = (
            distance_sq.min(axis=1)
            > radius ** 2
        )

    return keep


def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0],
    ])


def sampson_error(
    uv1, uv2, C1, C2, R1, R2
):
    relative_R = R2.T @ R1
    t = R2.T @ (C1 - C2)

    norm = np.linalg.norm(t)

    if norm < 1e-9:
        return np.full(len(uv1), np.inf)

    E = skew(t / norm) @ relative_R

    x1 = np.column_stack([
        uv1, np.ones(len(uv1))
    ])

    x2 = np.column_stack([
        uv2, np.ones(len(uv2))
    ])

    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T

    numerator = np.sum(
        x2 * Ex1, axis=1
    ) ** 2

    denominator = (
        Ex1[:, 0] ** 2
        + Ex1[:, 1] ** 2
        + Etx2[:, 0] ** 2
        + Etx2[:, 1] ** 2
    )

    return FOCAL * np.sqrt(
        numerator
        / np.maximum(denominator, 1e-15)
    )


def triangulate(
    uv1, uv2, C1, C2, R1, R2
):
    """
    Linear DLT with world-to-camera projections.
    The output uses the provisional pose scale.
    """
    t1 = -R1.T @ C1
    t2 = -R2.T @ C2

    P1 = np.column_stack([R1.T, t1])
    P2 = np.column_stack([R2.T, t2])

    homogeneous = cv2.triangulatePoints(
        P1,
        P2,
        uv1.T.copy(),
        uv2.T.copy(),
    )

    w = homogeneous[3]

    valid = np.abs(w) > 1e-12

    points = np.full(
        (len(uv1), 3),
        np.nan,
    )

    points[valid] = (
        homogeneous[:3, valid]
        / w[valid]
    ).T

    return points


def assess_points(
    points,
    uv1,
    uv2,
    C1,
    C2,
    R1,
    R2,
    min_parallax,
    max_error,
    max_range,
):
    x1 = (R1.T @ (points - C1).T).T
    x2 = (R2.T @ (points - C2).T).T

    depth_ok = (
        (x1[:, 2] > 1e-6)
        & (x2[:, 2] > 1e-6)
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        reproj1 = (
            x1[:, :2] / x1[:, 2:3]
        )

        reproj2 = (
            x2[:, :2] / x2[:, 2:3]
        )

    error1 = (
        np.linalg.norm(
            reproj1 - uv1, axis=1
        ) * FOCAL
    )

    error2 = (
        np.linalg.norm(
            reproj2 - uv2, axis=1
        ) * FOCAL
    )

    reproj_error = np.maximum(
        error1, error2
    )

    ray1 = points - C1
    ray2 = points - C2

    length1 = np.linalg.norm(
        ray1, axis=1
    )

    length2 = np.linalg.norm(
        ray2, axis=1
    )

    with np.errstate(
        invalid="ignore",
        divide="ignore",
    ):
        cosine = np.sum(
            ray1 * ray2, axis=1
        ) / (length1 * length2)

    parallax = np.degrees(
        np.arccos(
            np.clip(cosine, -1, 1)
        )
    )

    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(reproj_error)
        & np.isfinite(parallax)
    )

    valid = (
        finite
        & depth_ok
        & (parallax >= min_parallax)
        & (reproj_error <= max_error)
        & (length1 <= max_range)
        & (length2 <= max_range)
    )

    return {
        "valid": valid,
        "finite": finite,
        "positive_depth": depth_ok,
        "parallax_deg": parallax,
        "reprojection_px": reproj_error,
    }


def summarize_values(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
    }


def write_ply(path, points):
    points = np.asarray(points, dtype=float)

    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        for x, y, z in points:
            f.write(
                f"{x:.8f} {y:.8f} {z:.8f}\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width", type=int, default=960
    )

    parser.add_argument(
        "--features", type=int, default=3000
    )

    parser.add_argument(
        "--max-gap", type=int, default=4
    )

    parser.add_argument(
        "--ratio", type=float, default=0.78
    )

    parser.add_argument(
        "--exclude-radius-px",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--min-matches",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-original-inlier",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--epipolar-threshold",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--min-parallax",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--max-reprojection",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--max-range",
        type=float,
        default=100.0,
    )

    args = parser.parse_args()

    if (
        args.width < 100
        or args.features < 100
        or args.max_gap < 1
        or not 0 < args.ratio < 1
        or args.exclude_radius_px <= 0
        or args.min_matches < 2
        or not 0 <= args.min_original_inlier <= 1
        or args.epipolar_threshold <= 0
        or args.min_parallax <= 0
        or args.max_reprojection <= 0
        or args.max_range <= 0
    ):
        parser.error("Invalid parameters.")

    OUTPUT.mkdir(
        parents=True, exist_ok=True
    )

    poses = load_poses()

    excluded, observation_count = (
        load_exclusion_observations()
    )

    names = list(poses)

    sift = cv2.SIFT_create(
        nfeatures=args.features
    )

    features = {}

    print("\nUNDER-CONSTRUCTION")
    print("Controlled Triangulation V3")
    print("-" * 55)
    print("Supported cameras:", len(names))
    print(
        "Excluded V2 observations:",
        observation_count,
    )

    for index, name in enumerate(names):
        features[name] = extract_features(
            name, sift, args.width
        )

        if (index + 1) % 15 == 0:
            print(
                "Extracted:",
                index + 1,
                "/", len(names),
            )

    radius = (
        args.exclude_radius_px / FOCAL
    )

    clouds = {
        "original": [],
        "refined": [],
    }

    rows = []
    tested_pairs = 0
    supported_pairs = 0
    total_matches = 0

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            gap = (
                frame_number(name_b)
                - frame_number(name_a)
            )

            if gap > args.max_gap:
                break

            a = poses[name_a]
            b = poses[name_b]

            baseline = np.linalg.norm(
                b["center"] - a["center"]
            )

            if not 0.08 <= baseline <= 5.0:
                continue

            matches = match_features(
                features[name_a],
                features[name_b],
                args.ratio,
            )

            if not matches:
                continue

            ids1 = np.asarray(
                [m[0] for m in matches],
                dtype=int,
            )

            ids2 = np.asarray(
                [m[1] for m in matches],
                dtype=int,
            )

            uv1 = features[name_a]["uv"][ids1]
            uv2 = features[name_b]["uv"][ids2]

            keep = (
                exclude_saved_features(
                    uv1,
                    excluded.get(name_a),
                    radius,
                )
                & exclude_saved_features(
                    uv2,
                    excluded.get(name_b),
                    radius,
                )
            )

            uv1 = uv1[keep]
            uv2 = uv2[keep]

            finite = (
                np.isfinite(uv1).all(axis=1)
                & np.isfinite(uv2).all(axis=1)
                & (np.linalg.norm(uv1, axis=1) < 10)
                & (np.linalg.norm(uv2, axis=1) < 10)
            )

            uv1 = uv1[finite]
            uv2 = uv2[finite]

            if len(uv1) < args.min_matches:
                continue

            tested_pairs += 1

            original_epi = sampson_error(
                uv1,
                uv2,
                a["center"],
                b["center"],
                a["original"],
                b["original"],
            )

            original_inlier = np.mean(
                original_epi
                <= args.epipolar_threshold
            )

            # Fixed support selection using original
            # geometry only, not refined geometry.
            if original_inlier < args.min_original_inlier:
                continue

            supported_pairs += 1
            total_matches += len(uv1)

            results = {}

            for mode in ("original", "refined"):
                R1 = a[mode]
                R2 = b[mode]

                points = triangulate(
                    uv1,
                    uv2,
                    a["center"],
                    b["center"],
                    R1,
                    R2,
                )

                evaluation = assess_points(
                    points,
                    uv1,
                    uv2,
                    a["center"],
                    b["center"],
                    R1,
                    R2,
                    args.min_parallax,
                    args.max_reprojection,
                    args.max_range,
                )

                results[mode] = {
                    "points": points,
                    **evaluation,
                }

                clouds[mode].extend(
                    points[
                        evaluation["valid"]
                    ].tolist()
                )

            original_valid = results[
                "original"
            ]["valid"]

            refined_valid = results[
                "refined"
            ]["valid"]

            both = (
                original_valid & refined_valid
            )

            rows.append({
                "frame_a": name_a,
                "frame_b": name_b,
                "gap": gap,
                "matches": len(uv1),
                "original_inlier_fraction": (
                    float(original_inlier)
                ),
                "original_valid": int(
                    original_valid.sum()
                ),
                "refined_valid": int(
                    refined_valid.sum()
                ),
                "both_valid": int(
                    both.sum()
                ),
                "original_only": int(
                    (original_valid & ~refined_valid).sum()
                ),
                "refined_only": int(
                    (refined_valid & ~original_valid).sum()
                ),
                "original_common_reprojection_median": (
                    float(np.median(
                        results["original"][
                            "reprojection_px"
                        ][both]
                    ))
                    if both.any() else None
                ),
                "refined_common_reprojection_median": (
                    float(np.median(
                        results["refined"][
                            "reprojection_px"
                        ][both]
                    ))
                    if both.any() else None
                ),
                "original_common_parallax_median": (
                    float(np.median(
                        results["original"][
                            "parallax_deg"
                        ][both]
                    ))
                    if both.any() else None
                ),
                "refined_common_parallax_median": (
                    float(np.median(
                        results["refined"][
                            "parallax_deg"
                        ][both]
                    ))
                    if both.any() else None
                ),
            })

    if not rows:
        raise RuntimeError(
            "No supported camera pairs survived. "
            "No comparison can be made."
        )

    csv_path = (
        OUTPUT / "triangulation_pair_comparison.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    for mode in ("original", "refined"):
        write_ply(
            OUTPUT / f"{mode}_candidate_points.ply",
            clouds[mode],
        )

    def sum_field(key):
        return int(sum(
            row[key] for row in rows
        ))

    original_count = sum_field(
        "original_valid"
    )

    refined_count = sum_field(
        "refined_valid"
    )

    common_count = sum_field(
        "both_valid"
    )

    # Pool matched, shared-point reprojection
    # statistics using per-pair summaries only.
    # These are not pooled point-level medians.
    common_rows = [
        row for row in rows
        if row["both_valid"] > 0
    ]

    original_common = [
        row["original_common_reprojection_median"]
        for row in common_rows
    ]

    refined_common = [
        row["refined_common_reprojection_median"]
        for row in common_rows
    ]

    improved_common_pairs = sum(
        new < old
        for old, new in zip(
            original_common,
            refined_common,
        )
    )

    summary = {
        "project": "Under-construction",
        "experiment": (
            "CONTROLLED_TRIANGULATION_V3"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "parameters": vars(args),
        "supported_cameras": len(names),
        "excluded_v2_observations": (
            observation_count
        ),
        "tested_pairs": tested_pairs,
        "supported_pairs": supported_pairs,
        "candidate_correspondences": total_matches,
        "original_valid_pair_observations": (
            original_count
        ),
        "refined_valid_pair_observations": (
            refined_count
        ),
        "valid_in_both": common_count,
        "original_only": sum_field(
            "original_only"
        ),
        "refined_only": sum_field(
            "refined_only"
        ),
        "common_valid_pairs": len(
            common_rows
        ),
        "common_pair_median_reprojection_px": {
            "original": summarize_values(
                original_common
            ),
            "refined": summarize_values(
                refined_common
            ),
        },
        "pairs_with_improved_common_reprojection": (
            improved_common_pairs
        ),
        "screening": {
            "refined_more_valid": (
                refined_count > original_count
            ),
            "refined_common_pair_majority": (
                improved_common_pairs
                > len(common_rows) / 2
                if common_rows else False
            ),
        },
        "approved_for_application": False,
        "limitations": [
            "These are pairwise candidate 3D points, "
            "not unique multiview landmarks.",
            "A feature may be triangulated repeatedly "
            "across overlapping camera pairs.",
            "Point-cloud counts are not unique geometry "
            "coverage or building completeness.",
            "Same-image feature exclusion is not "
            "independent scene validation.",
            "The pose convention and camera-to-IMU "
            "extrinsics remain unverified.",
            "The pose scale is provisional.",
            "Parallax and reprojection gates can favor "
            "different candidate subsets.",
            "No CAD registration is validated.",
            "Original V2 reconstruction is unchanged.",
        ],
    }

    report_path = (
        OUTPUT / "triangulation_comparison_report.json"
    )

    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("-" * 55)
    print("Tested pairs:", tested_pairs)
    print("Supported pairs:", supported_pairs)
    print(
        "Candidate correspondences:",
        total_matches,
    )
    print(
        "Original accepted pair-points:",
        original_count,
    )
    print(
        "Refined accepted pair-points:",
        refined_count,
    )
    print(
        "Accepted by both:",
        common_count,
    )
    print(
        "Original only:",
        sum_field("original_only"),
    )
    print(
        "Refined only:",
        sum_field("refined_only"),
    )
    print(
        "Common-point pairs improved:",
        f"{improved_common_pairs}/{len(common_rows)}",
    )

    print("\nReport:", report_path)
    print("Pair comparison:", csv_path)
    print("Approved for application: NO")
    print(
        "\nThese are experimental pairwise clouds. "
        "V2 remains unchanged."
    )


if __name__ == "__main__":
    from datetime import datetime, timezone
    main()
