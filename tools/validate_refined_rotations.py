
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from refine_camera_poses import (
    ROOT,
    load_data,
    create_problem,
    projection_errors,
)

OUTPUT = ROOT / "outputs" / "rotation_new_feature_validation"

SEQUENCE = "floor_2_2025-10-28_run_2"

ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

FRAME_DIR = (
    ARCHIVE
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frames"
)

CALIBRATION_FILE = (
    ARCHIVE
    / "hilti_calibration"
    / "camera_calibration_report.json"
)


def find_calibration(node):
    """
    Locate the known calibration fields without assuming
    an exact number of JSON nesting levels.
    """
    if isinstance(node, dict):
        if (
            "intrinsics" in node
            and "distortion_coefficients" in node
        ):
            return node

        for value in node.values():
            result = find_calibration(value)
            if result is not None:
                return result

    elif isinstance(node, list):
        for value in node:
            result = find_calibration(value)
            if result is not None:
                return result

    return None


def load_calibration():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(CALIBRATION_FILE)

    document = json.loads(
        CALIBRATION_FILE.read_text(encoding="utf-8")
    )

    calibration = find_calibration(document)

    if calibration is None:
        raise RuntimeError(
            "Could not locate camera intrinsics and "
            "distortion coefficients."
        )

    intrinsics = np.asarray(
        calibration["intrinsics"],
        dtype=np.float64,
    ).reshape(-1)

    distortion = np.asarray(
        calibration["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(-1)

    if len(intrinsics) != 4 or len(distortion) != 4:
        raise RuntimeError(
            "Expected four intrinsics and four fisheye "
            "distortion coefficients."
        )

    fx, fy, cx, cy = intrinsics

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return K, distortion.reshape(4, 1)


def scaled_camera_matrix(K, scale):
    matrix = K.copy()
    matrix[0, :] *= scale
    matrix[1, :] *= scale
    matrix[2, :] = [0.0, 0.0, 1.0]
    return matrix


def make_rotation_sparsity(problem):
    ncam = len(problem["frames"])
    nobs = len(problem["obs_camera"])

    rows = 2 * nobs + 3 * ncam + 3
    cols = 3 * ncam

    S = lil_matrix((rows, cols), dtype=np.int8)

    for i, camera in enumerate(problem["obs_camera"]):
        camera = int(camera)

        S[
            2 * i:2 * i + 2,
            3 * camera:3 * camera + 3,
        ] = 1

    start = 2 * nobs

    for camera in range(ncam):
        S[
            start + 3 * camera:start + 3 * camera + 3,
            3 * camera:3 * camera + 3,
        ] = np.eye(3, dtype=np.int8)

    anchor = start + 3 * ncam

    S[
        anchor:anchor + 3,
        0:3,
    ] = np.eye(3, dtype=np.int8)

    return S.tocsr()


def fit_rotations(problem, args):
    ncam = len(problem["frames"])
    indices = np.arange(
        len(problem["obs_camera"]),
        dtype=int,
    )

    rotation_sigma = math.radians(
        args.rotation_sigma_deg
    )

    def corrected(offsets):
        correction = Rotation.from_rotvec(
            offsets.reshape(ncam, 3)
        ).as_matrix()

        return correction @ problem["rotations"]

    def residual(x):
        offsets = x.reshape(ncam, 3)
        rotations = corrected(x)

        errors = projection_errors(
            problem["centers"],
            rotations,
            problem["points"],
            problem,
            indices,
        ).ravel()

        prior = (offsets / rotation_sigma).ravel()
        anchor = offsets[0] / args.anchor_sigma

        return np.concatenate([
            errors,
            prior,
            anchor,
        ])

    print("\nFitting rotation-only corrections...")

    result = least_squares(
        residual,
        np.zeros(3 * ncam, dtype=np.float64),
        method="trf",
        jac="2-point",
        jac_sparsity=make_rotation_sparsity(problem),
        tr_solver="lsmr",
        loss="soft_l1",
        f_scale=args.loss_scale,
        x_scale="jac",
        max_nfev=args.max_nfev,
        verbose=0,
    )

    offsets = result.x.reshape(ncam, 3)

    angles = np.degrees(
        np.linalg.norm(offsets, axis=1)
    )

    bounded = bool(
        np.isfinite(result.x).all()
        and np.max(angles)
        <= 3.0 * args.rotation_sigma_deg
    )

    print("Converged:", bool(result.success))
    print("Evaluations:", result.nfev)
    print(
        "Median correction:",
        f"{np.median(angles):.4f} deg",
    )
    print(
        "Maximum correction:",
        f"{np.max(angles):.4f} deg",
    )

    if not result.success or not bounded:
        raise RuntimeError(
            "Rotation fitting did not converge safely. "
            "No new-feature validation will be run."
        )

    return corrected(result.x), offsets, result


def load_frame(name, width, K, D, sift):
    path = FRAME_DIR / name

    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(path)

    original_height, original_width = image.shape[:2]

    scale = min(1.0, width / original_width)

    if scale < 1.0:
        image = cv2.resize(
            image,
            (
                int(round(original_width * scale)),
                int(round(original_height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray, None
    )

    if descriptors is None or not keypoints:
        return {
            "xy": np.empty((0, 2), dtype=np.float64),
            "uv": np.empty((0, 2), dtype=np.float64),
            "descriptors": None,
        }

    xy = np.asarray(
        [keypoint.pt for keypoint in keypoints],
        dtype=np.float64,
    )

    scaled_K = scaled_camera_matrix(K, scale)

    normalized = cv2.fisheye.undistortPoints(
        xy.reshape(-1, 1, 2),
        scaled_K,
        D,
    ).reshape(-1, 2)

    return {
        "xy": xy,
        "uv": normalized,
        "descriptors": descriptors,
    }


def saved_observation_positions(problem):
    """
    Build normalized-coordinate exclusion centers for
    every frame used in the original optimization.
    """
    centers = defaultdict(list)

    for camera, uv in zip(
        problem["obs_camera"],
        problem["obs_uv"],
    ):
        centers[int(camera)].append(uv)

    return {
        camera: np.asarray(points, dtype=np.float64)
        for camera, points in centers.items()
    }


def independent_feature_mask(
    uv,
    excluded_uv,
    radius,
):
    """
    A candidate feature is excluded if it lies close
    to any previously optimized image observation.

    This is spatial exclusion, not a proof of semantic
    or scene-level independence.
    """
    if len(uv) == 0:
        return np.zeros(0, dtype=bool)

    if excluded_uv is None or len(excluded_uv) == 0:
        return np.ones(len(uv), dtype=bool)

    keep = np.ones(len(uv), dtype=bool)

    # Chunk to avoid large temporary distance matrices.
    for start in range(0, len(uv), 256):
        end = min(start + 256, len(uv))

        delta = (
            uv[start:end, None, :]
            - excluded_uv[None, :, :]
        )

        distances_sq = np.sum(
            delta * delta,
            axis=2,
        )

        keep[start:end] = (
            np.min(distances_sq, axis=1)
            > radius * radius
        )

    return keep


def mutual_matches(desc_a, desc_b, ratio):
    if (
        desc_a is None
        or desc_b is None
        or len(desc_a) < 2
        or len(desc_b) < 2
    ):
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=False,
    )

    forward = matcher.knnMatch(
        desc_a, desc_b, k=2
    )

    backward = matcher.knnMatch(
        desc_b, desc_a, k=2
    )

    forward_good = {}

    for candidates in forward:
        if len(candidates) != 2:
            continue

        first, second = candidates

        if first.distance < ratio * second.distance:
            forward_good[first.queryIdx] = (
                first.trainIdx,
                first.distance,
            )

    backward_good = {}

    for candidates in backward:
        if len(candidates) != 2:
            continue

        first, second = candidates

        if first.distance < ratio * second.distance:
            backward_good[first.queryIdx] = first.trainIdx

    result = []

    for query, (train, distance) in forward_good.items():
        if backward_good.get(train) == query:
            result.append((query, train, distance))

    return result


def skew(vector):
    x, y, z = vector

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def sampson_errors_px(
    uv_a,
    uv_b,
    center_a,
    center_b,
    rotation_a,
    rotation_b,
    focal,
):
    """
    Essential matrix for camera-to-world rotations:

        R_ba = R_b.T @ R_a
        t_ba = R_b.T @ (C_a - C_b)
        E = [t_ba]_x @ R_ba

    Coordinates are normalized, undistorted fisheye
    image coordinates. Sampson distance is expressed
    in approximate pixel units using focal.
    """

    relative_rotation = (
        rotation_b.T @ rotation_a
    )

    translation = (
        rotation_b.T @ (center_a - center_b)
    )

    norm = np.linalg.norm(translation)

    if norm < 1e-9:
        return np.full(
            len(uv_a),
            np.inf,
            dtype=np.float64,
        )

    translation /= norm

    E = skew(translation) @ relative_rotation

    x_a = np.column_stack([
        uv_a,
        np.ones(len(uv_a)),
    ])

    x_b = np.column_stack([
        uv_b,
        np.ones(len(uv_b)),
    ])

    Exa = (E @ x_a.T).T
    Etxb = (E.T @ x_b.T).T

    numerator = np.sum(
        x_b * Exa,
        axis=1,
    ) ** 2

    denominator = (
        Exa[:, 0] ** 2
        + Exa[:, 1] ** 2
        + Etxb[:, 0] ** 2
        + Etxb[:, 1] ** 2
    )

    distance = np.sqrt(
        numerator / np.maximum(
            denominator,
            1e-15,
        )
    )

    return distance * focal


def summarize(errors, threshold):
    errors = np.asarray(
        errors,
        dtype=np.float64,
    )

    finite = errors[np.isfinite(errors)]

    if len(finite) == 0:
        return {
            "count": 0,
            "median_px": None,
            "p90_px": None,
            "rmse_px": None,
            "inlier_fraction": None,
        }

    return {
        "count": int(len(finite)),
        "median_px": float(np.median(finite)),
        "p90_px": float(np.percentile(finite, 90)),
        "rmse_px": float(
            np.sqrt(np.mean(finite ** 2))
        ),
        "inlier_fraction": float(
            np.mean(finite <= threshold)
        ),
    }


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
        "--max-gap",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--ratio",
        type=float,
        default=0.78,
    )

    parser.add_argument(
        "--exclude-radius-px",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--epipolar-threshold-px",
        type=float,
        default=2.5,
    )

    parser.add_argument(
        "--min-pair-matches",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--loss-scale",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=40,
    )

    args = parser.parse_args()

    if (
        args.width < 100
        or args.features < 100
        or args.max_gap < 1
        or not (0 < args.ratio < 1)
        or args.exclude_radius_px <= 0
        or args.epipolar_threshold_px <= 0
        or args.min_pair_matches < 1
        or args.rotation_sigma_deg <= 0
        or args.anchor_sigma <= 0
        or args.loss_scale <= 0
        or args.max_nfev < 1
    ):
        parser.error("Invalid parameter values.")

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not FRAME_DIR.exists():
        raise FileNotFoundError(FRAME_DIR)

    landmarks, pose_records = load_data()

    problem = create_problem(
        landmarks,
        pose_records,
    )

    K, D = load_calibration()

    refined_rotations, offsets, fit = fit_rotations(
        problem,
        args,
    )

    original_rotations = problem["rotations"]

    saved_uv = saved_observation_positions(
        problem
    )

    frames = problem["frames"]

    # The image names contain their original frame
    # numbers; use those numbers to determine gaps.
    def frame_number(name):
        return int(Path(name).stem.rsplit("_", 1)[-1])

    frame_numbers = [
        frame_number(name)
        for name in frames
    ]

    sift = cv2.SIFT_create(
        nfeatures=args.features
    )

    # Store features for the supported cameras only.
    # This bounds the experiment to cameras whose
    # rotations were actually refined.
    feature_cache = {}

    for i, name in enumerate(frames):
        feature_cache[i] = load_frame(
            name,
            args.width,
            K,
            D,
            sift,
        )

        if (i + 1) % 15 == 0:
            print(
                f"Extracted features: {i + 1}/{len(frames)}"
            )

    focal = (
        float(K[0, 0] + K[1, 1]) / 2.0
    )

    # Convert approximate exclusion radius in original
    # image pixels to normalized-coordinate units.
    normalized_radius = (
        args.exclude_radius_px / focal
    )

    all_original = []
    all_refined = []
    pair_rows = []

    eligible_pairs = 0
    evaluated_pairs = 0

    print("\nEvaluating new feature correspondences...")

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            gap = abs(
                frame_numbers[j] - frame_numbers[i]
            )

            if gap > args.max_gap:
                continue

            baseline = np.linalg.norm(
                problem["centers"][j]
                - problem["centers"][i]
            )

            if not (0.08 <= baseline <= 5.0):
                continue

            eligible_pairs += 1

            a = feature_cache[i]
            b = feature_cache[j]

            matches = mutual_matches(
                a["descriptors"],
                b["descriptors"],
                args.ratio,
            )

            if not matches:
                continue

            ids_a = np.asarray(
                [m[0] for m in matches],
                dtype=int,
            )

            ids_b = np.asarray(
                [m[1] for m in matches],
                dtype=int,
            )

            uv_a = a["uv"][ids_a]
            uv_b = b["uv"][ids_b]

            keep_a = independent_feature_mask(
                uv_a,
                saved_uv.get(i),
                normalized_radius,
            )

            keep_b = independent_feature_mask(
                uv_b,
                saved_uv.get(j),
                normalized_radius,
            )

            keep = keep_a & keep_b

            uv_a = uv_a[keep]
            uv_b = uv_b[keep]

            finite = (
                np.isfinite(uv_a).all(axis=1)
                & np.isfinite(uv_b).all(axis=1)
                & (np.linalg.norm(uv_a, axis=1) < 10)
                & (np.linalg.norm(uv_b, axis=1) < 10)
            )

            uv_a = uv_a[finite]
            uv_b = uv_b[finite]

            if len(uv_a) < args.min_pair_matches:
                continue

            original_errors = sampson_errors_px(
                uv_a,
                uv_b,
                problem["centers"][i],
                problem["centers"][j],
                original_rotations[i],
                original_rotations[j],
                focal,
            )

            refined_errors = sampson_errors_px(
                uv_a,
                uv_b,
                problem["centers"][i],
                problem["centers"][j],
                refined_rotations[i],
                refined_rotations[j],
                focal,
            )

            finite_errors = (
                np.isfinite(original_errors)
                & np.isfinite(refined_errors)
            )

            original_errors = original_errors[finite_errors]
            refined_errors = refined_errors[finite_errors]

            if len(original_errors) < args.min_pair_matches:
                continue

            evaluated_pairs += 1

            all_original.extend(
                original_errors.tolist()
            )

            all_refined.extend(
                refined_errors.tolist()
            )

            old = summarize(
                original_errors,
                args.epipolar_threshold_px,
            )

            new = summarize(
                refined_errors,
                args.epipolar_threshold_px,
            )

            pair_rows.append({
                "frame_a": frames[i],
                "frame_b": frames[j],
                "gap": gap,
                "matches": len(original_errors),
                "original_median_px": old["median_px"],
                "refined_median_px": new["median_px"],
                "original_inlier_fraction": old[
                    "inlier_fraction"
                ],
                "refined_inlier_fraction": new[
                    "inlier_fraction"
                ],
                "original_rmse_px": old["rmse_px"],
                "refined_rmse_px": new["rmse_px"],
            })

    if not all_original:
        raise RuntimeError(
            "No feature-disjoint pairs were available. "
            "Try a smaller exclusion radius, but report "
            "the changed setting."
        )

    original_summary = summarize(
        all_original,
        args.epipolar_threshold_px,
    )

    refined_summary = summarize(
        all_refined,
        args.epipolar_threshold_px,
    )

    improved_pair_count = sum(
        row["refined_median_px"]
        < row["original_median_px"]
        for row in pair_rows
    )

    pair_improvement_fraction = (
        improved_pair_count / len(pair_rows)
    )

    # Exploratory screening, not independent validation.
    screening_pass = bool(
        refined_summary["median_px"]
        < original_summary["median_px"]
        and refined_summary["inlier_fraction"]
        > original_summary["inlier_fraction"]
        and refined_summary["rmse_px"]
        <= original_summary["rmse_px"]
        and pair_improvement_fraction >= 0.60
    )

    corrected_pose_document = {
        "status": "EXPERIMENTAL",
        "source": "ROTATION_ONLY_ALL_SAVED_OBSERVATIONS",
        "camera_extrinsics_verified": False,
        "metric_accuracy_verified": False,
        "cad_registration_verified": False,
        "approved_for_application": False,
        "poses": {},
    }

    for i, name in enumerate(frames):
        corrected_pose_document["poses"][name] = {
            "center": (
                problem["centers"][i].tolist()
            ),
            "rotation_camera_to_world": (
                refined_rotations[i].tolist()
            ),
            "original_rotation_camera_to_world": (
                original_rotations[i].tolist()
            ),
            "rotation_offset_rotvec": (
                offsets[i].tolist()
            ),
            "rotation_correction_deg": float(
                np.degrees(
                    np.linalg.norm(offsets[i])
                )
            ),
        }

    poses_path = (
        OUTPUT / "experimental_corrected_camera_poses.json"
    )

    poses_path.write_text(
        json.dumps(
            corrected_pose_document,
            indent=2,
        ),
        encoding="utf-8",
    )

    pairs_path = OUTPUT / "new_feature_pair_results.csv"

    with pairs_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_a",
                "frame_b",
                "gap",
                "matches",
                "original_median_px",
                "refined_median_px",
                "original_inlier_fraction",
                "refined_inlier_fraction",
                "original_rmse_px",
                "refined_rmse_px",
            ],
        )

        writer.writeheader()
        writer.writerows(pair_rows)

    report = {
        "project": "Under-construction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "ROTATION_NEW_FEATURE_VALIDATION_V1",
        "parameters": vars(args),
        "rotation_fit": {
            "converged": bool(fit.success),
            "function_evaluations": int(fit.nfev),
            "cameras_refined": len(frames),
            "saved_observations_used": int(
                len(problem["obs_camera"])
            ),
            "median_correction_deg": float(
                np.median(
                    np.degrees(
                        np.linalg.norm(offsets, axis=1)
                    )
                )
            ),
        },
        "eligible_pairs": eligible_pairs,
        "evaluated_pairs": evaluated_pairs,
        "feature_disjoint_correspondences": len(
            all_original
        ),
        "original": original_summary,
        "refined": refined_summary,
        "pairs_with_lower_median_error": (
            improved_pair_count
        ),
        "fraction_pairs_improved": float(
            pair_improvement_fraction
        ),
        "screening_pass": screening_pass,
        "approved_for_application": False,
        "limitations": [
            "New features come from the same photographs.",
            "Spatial exclusion does not guarantee scene-level independence.",
            "Feature matches have not been verified against ground truth.",
            "Some new SIFT matches may be incorrect.",
            "All pairs are restricted to previously supported cameras.",
            "Camera centers remain fixed.",
            "No camera-to-IMU extrinsic calibration is available.",
            "The same provisional pose convention is retained.",
            "The original V2 landmark positions are unchanged.",
            "No new landmarks are triangulated in this experiment.",
            "Metric accuracy and CAD alignment remain unverified.",
            "A passing result is not permission to deploy corrected poses.",
        ],
    }

    report_path = (
        OUTPUT / "new_feature_validation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("-" * 60)
    print("Eligible camera pairs:", eligible_pairs)
    print("Evaluated pairs:", evaluated_pairs)
    print(
        "New feature correspondences:",
        len(all_original),
    )
    print(
        "Original median epipolar error:",
        f"{original_summary['median_px']:.4f} px",
    )
    print(
        "Refined median epipolar error:",
        f"{refined_summary['median_px']:.4f} px",
    )
    print(
        "Original inlier fraction:",
        f"{original_summary['inlier_fraction']:.4f}",
    )
    print(
        "Refined inlier fraction:",
        f"{refined_summary['inlier_fraction']:.4f}",
    )
    print(
        "Original RMSE:",
        f"{original_summary['rmse_px']:.4f} px",
    )
    print(
        "Refined RMSE:",
        f"{refined_summary['rmse_px']:.4f} px",
    )
    print(
        "Pairs with improved median:",
        f"{improved_pair_count}/{evaluated_pairs}",
    )

    print(
        "\nSCREENING:",
        "PASS" if screening_pass else "FAIL",
    )

    print("Approved for application: NO")
    print("\nReport:", report_path)
    print("Pair results:", pairs_path)
    print("Experimental poses:", poses_path)

    print(
        "\nV2 reconstruction remains unchanged."
    )


if __name__ == "__main__":
    main()
