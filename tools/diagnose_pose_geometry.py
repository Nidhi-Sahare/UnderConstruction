
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
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

FRAMES = DATA / "hilti_frames" / SEQUENCE / "cam0" / "frames"
MANIFEST = DATA / "hilti_frames" / SEQUENCE / "cam0" / "frame_manifest.csv"
CALIBRATION = DATA / "hilti_calibration" / "camera_calibration_report.json"

OUTPUT = ROOT / "outputs" / "pose_geometry_diagnostic"

MATCH_RATIO = 0.78
MIN_MATCHES = 25
MIN_POSE_INLIERS = 15
POSE_EPIPOLAR_THRESHOLD_PX = 2.5
INDEPENDENT_THRESHOLD_PX = 2.0

MIN_BASELINE = 0.08
MAX_BASELINE = 5.0

MIN_PARALLAX_DEG = 1.0
MAX_DEPTH = 100.0


def rotation_from_quaternion(q):
    x, y, z, w = q

    return np.array([
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
    ], dtype=np.float64)


def load_poses():
    poses = {}

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("pose_matched", "").strip().lower() != "true":
                continue

            try:
                center = np.array(
                    [float(row[k]) for k in ("tx", "ty", "tz")],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [float(row[k]) for k in ("qx", "qy", "qz", "qw")],
                    dtype=np.float64,
                )

                norm = np.linalg.norm(quaternion)

                if (
                    norm < 1e-10
                    or not np.isfinite(center).all()
                    or not np.isfinite(quaternion).all()
                ):
                    continue

                poses[row["filename"]] = {
                    "center": center,
                    "rotation": rotation_from_quaternion(
                        quaternion / norm
                    ),
                }

            except (ValueError, KeyError, TypeError):
                continue

    return poses


def load_calibration():
    report = json.loads(
        CALIBRATION.read_text(encoding="utf-8")
    )

    data = report["calibration"]

    fx, fy, cx, cy = map(float, data["intrinsics"])

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    D = np.asarray(
        data["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(4, 1)

    return K, D


def skew(t):
    x, y, z = t

    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ], dtype=np.float64)


def relative_pose(a, b):
    R1 = a["rotation"]
    R2 = b["rotation"]

    C1 = a["center"]
    C2 = b["center"]

    R = R2.T @ R1
    t = R2.T @ (C1 - C2)

    baseline = float(np.linalg.norm(t))

    if baseline < 1e-10:
        return None

    return R, t / baseline, baseline


def sampson_error_px(p1, p2, E, focal):
    x1 = np.column_stack(
        (p1, np.ones(len(p1)))
    )

    x2 = np.column_stack(
        (p2, np.ones(len(p2)))
    )

    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T

    numerator = np.sum(x2 * Ex1, axis=1) ** 2

    denominator = (
        Ex1[:, 0] ** 2
        + Ex1[:, 1] ** 2
        + Etx2[:, 0] ** 2
        + Etx2[:, 1] ** 2
    )

    return focal * np.sqrt(
        numerator / np.maximum(denominator, 1e-12)
    )


def rotation_error_deg(Ra, Rb):
    delta = Ra @ Rb.T

    cosine = (np.trace(delta) - 1) / 2

    return float(np.degrees(
        np.arccos(np.clip(cosine, -1, 1))
    ))


def direction_error_deg(a, b):
    cosine = float(
        np.dot(a, b)
        / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)
    )

    return float(np.degrees(
        np.arccos(np.clip(cosine, -1, 1))
    ))


def triangulation_statistics(p1, p2, R, t):
    """
    Triangulate in camera-1 coordinates using
    the recorded relative pose.

    Translation scale is normalized here.
    The resulting depths are baseline-relative,
    not metric depth measurements.
    """
    P1 = np.hstack([
        np.eye(3),
        np.zeros((3, 1)),
    ])

    P2 = np.hstack([
        R,
        t.reshape(3, 1),
    ])

    homogeneous = cv2.triangulatePoints(
        P1,
        P2,
        p1.T.copy(),
        p2.T.copy(),
    )

    w = homogeneous[3]

    safe = np.abs(w) > 1e-10

    xyz = np.full(
        (len(w), 3),
        np.nan,
        dtype=np.float64,
    )

    xyz[safe] = (
        homogeneous[:3, safe] / w[safe]
    ).T

    local2 = (R @ xyz.T).T + t

    depth1 = xyz[:, 2]
    depth2 = local2[:, 2]

    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(local2).all(axis=1)
        & (depth1 > 0.05)
        & (depth2 > 0.05)
        & (depth1 < MAX_DEPTH)
        & (depth2 < MAX_DEPTH)
    )

    C2 = -R.T @ t

    ray1 = xyz
    ray2 = xyz - C2

    norms = (
        np.linalg.norm(ray1, axis=1)
        * np.linalg.norm(ray2, axis=1)
    )

    cosine = np.sum(ray1 * ray2, axis=1) / np.maximum(
        norms, 1e-12
    )

    parallax = np.degrees(
        np.arccos(np.clip(cosine, -1, 1))
    )

    valid &= np.isfinite(parallax)

    useful = valid & (parallax >= MIN_PARALLAX_DEG)

    return {
        "positive_depth_count": int(valid.sum()),
        "useful_parallax_count": int(useful.sum()),
        "median_parallax_deg": (
            float(np.median(parallax[valid]))
            if valid.any()
            else None
        ),
    }


def median_or_none(values):
    finite = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]

    return float(np.median(finite)) if finite else None


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


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
        default=0,
        help="0 means evaluate every eligible pair.",
    )

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
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        args.max_gap < 1
        or args.max_pairs < 0
        or args.width < 100
        or args.features < 100
    ):
        parser.error("Invalid arguments.")

    for path in (FRAMES, MANIFEST, CALIBRATION):
        if not path.exists():
            raise FileNotFoundError(path)

    cv2.setRNGSeed(args.seed)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(FRAMES.glob("*.jpg"))

    if not paths:
        raise RuntimeError("No photographs found.")

    poses = load_poses()
    K, D = load_calibration()

    sift = cv2.SIFT_create(nfeatures=args.features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    @lru_cache(maxsize=24)
    def features(index):
        image = cv2.imread(str(paths[index]))

        if image is None:
            return None

        h, w = image.shape[:2]
        scale = min(1.0, args.width / float(w))

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

        keypoints, descriptors = sift.detectAndCompute(
            gray, None
        )

        if descriptors is None or len(keypoints) < 2:
            return None

        scaled_K = K.copy()
        scaled_K[0, :] *= scale
        scaled_K[1, :] *= scale

        pixels = np.asarray(
            [kp.pt for kp in keypoints],
            dtype=np.float64,
        )

        normalized = cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2),
            scaled_K,
            D,
        ).reshape(-1, 2)

        focal = float(
            (scaled_K[0, 0] + scaled_K[1, 1]) / 2
        )

        return {
            "descriptors": descriptors,
            "normalized": normalized,
            "focal": focal,
            "feature_count": len(keypoints),
        }

    def mutual_matches(a, b):
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

        def ratio_filter(pairs):
            result = {}

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if first.distance < MATCH_RATIO * second.distance:
                    result[first.queryIdx] = first

            return result

        fwd = ratio_filter(forward)
        rev = ratio_filter(backward)

        return [
            m for idx, m in fwd.items()
            if (
                m.trainIdx in rev
                and rev[m.trainIdx].trainIdx == idx
            )
        ]

    candidates = []

    for i, path in enumerate(paths):
        a = poses.get(path.name)

        if a is None:
            continue

        for gap in range(1, args.max_gap + 1):
            j = i + gap

            if j >= len(paths):
                continue

            b = poses.get(paths[j].name)

            if b is None:
                continue

            relative = relative_pose(a, b)

            if relative is None:
                continue

            R, t, baseline = relative

            if MIN_BASELINE <= baseline <= MAX_BASELINE:
                candidates.append(
                    (i, j, gap, R, t, baseline)
                )

    if args.max_pairs and len(candidates) > args.max_pairs:
        indices = np.linspace(
            0,
            len(candidates) - 1,
            args.max_pairs,
            dtype=int,
        )

        candidates = [
            candidates[int(k)] for k in indices
        ]

    print("\nUNDER-CONSTRUCTION")
    print("Camera-pose geometry diagnostic")
    print("-" * 60)
    print("Photographs:", len(paths))
    print("Pose-matched frames:", len(poses))
    print("Eligible pairs:", len(candidates))
    print("Maximum frame gap:", args.max_gap)
    print("Camera extrinsics: UNVERIFIED")
    print()

    records = []
    reasons = Counter()

    for number, (i, j, gap, R, t, baseline) in enumerate(
        candidates, start=1
    ):
        record = {
            "frame_a": paths[i].name,
            "frame_b": paths[j].name,
            "gap": gap,
            "baseline_pose_units": baseline,
            "status": "UNTESTED",
            "mutual_matches": 0,
            "pose_epipolar_inliers": 0,
            "pose_epipolar_inlier_fraction": None,
            "median_pose_sampson_px": None,
            "independent_essential_inliers": 0,
            "independent_inlier_fraction": None,
            "independent_rotation_error_deg": None,
            "independent_translation_direction_error_deg": None,
            "independent_pose_cheirality_inliers": None,
            "positive_depth_count": 0,
            "useful_parallax_count": 0,
            "median_parallax_deg": None,
        }

        try:
            a = features(i)
            b = features(j)

            if a is None or b is None:
                record["status"] = "NO_FEATURES"
                records.append(record)
                reasons[record["status"]] += 1
                continue

            matches = mutual_matches(a, b)
            record["mutual_matches"] = len(matches)

            if len(matches) < MIN_MATCHES:
                record["status"] = "INSUFFICIENT_MATCHES"
                records.append(record)
                reasons[record["status"]] += 1
                continue

            idx1 = np.asarray(
                [m.queryIdx for m in matches],
                dtype=int,
            )

            idx2 = np.asarray(
                [m.trainIdx for m in matches],
                dtype=int,
            )

            p1 = a["normalized"][idx1]
            p2 = b["normalized"][idx2]

            finite = (
                np.isfinite(p1).all(axis=1)
                & np.isfinite(p2).all(axis=1)
            )

            p1 = p1[finite]
            p2 = p2[finite]

            if len(p1) < MIN_MATCHES:
                record["status"] = "INVALID_NORMALIZED_POINTS"
                records.append(record)
                reasons[record["status"]] += 1
                continue

            focal = (a["focal"] + b["focal"]) / 2

            recorded_E = skew(t) @ R

            errors = sampson_error_px(
                p1, p2, recorded_E, focal
            )

            pose_mask = (
                np.isfinite(errors)
                & (errors <= POSE_EPIPOLAR_THRESHOLD_PX)
            )

            pose_count = int(pose_mask.sum())

            record["pose_epipolar_inliers"] = pose_count
            record["pose_epipolar_inlier_fraction"] = (
                pose_count / len(p1)
            )
            record["median_pose_sampson_px"] = median_or_none(
                errors
            )

            # Estimate essential geometry independently
            # from image correspondences. This is a
            # diagnostic, not ground-truth calibration.
            independent_E, independent_mask = (
                cv2.findEssentialMat(
                    p1,
                    p2,
                    cameraMatrix=np.eye(3),
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=(
                        INDEPENDENT_THRESHOLD_PX / focal
                    ),
                )
            )

            if independent_E is not None and independent_mask is not None:
                # OpenCV can return multiple candidate
                # essential matrices stacked vertically.
                E_candidates = np.asarray(
                    independent_E,
                    dtype=np.float64,
                ).reshape(-1, 3, 3)

                best = None

                for candidate_E in E_candidates:
                    try:
                        count, recovered_R, recovered_t, mask = (
                            cv2.recoverPose(
                                candidate_E,
                                p1,
                                p2,
                                cameraMatrix=np.eye(3),
                                mask=independent_mask.copy(),
                            )
                        )

                        if best is None or count > best[0]:
                            best = (
                                int(count),
                                recovered_R,
                                recovered_t.reshape(3),
                            )
                    except cv2.error:
                        continue

                record["independent_essential_inliers"] = int(
                    np.count_nonzero(independent_mask)
                )

                record["independent_inlier_fraction"] = (
                    record["independent_essential_inliers"]
                    / len(p1)
                )

                if best is not None:
                    count, recovered_R, recovered_t = best

                    record[
                        "independent_pose_cheirality_inliers"
                    ] = count

                    record[
                        "independent_rotation_error_deg"
                    ] = rotation_error_deg(
                        recovered_R, R
                    )

                    record[
                        "independent_translation_direction_error_deg"
                    ] = direction_error_deg(
                        recovered_t, t
                    )

            if pose_count < MIN_POSE_INLIERS:
                record["status"] = "POSE_GEOMETRY_DISAGREEMENT"
                records.append(record)
                reasons[record["status"]] += 1
                continue

            geometry = triangulation_statistics(
                p1[pose_mask],
                p2[pose_mask],
                R,
                t,
            )

            record.update(geometry)

            if geometry["useful_parallax_count"] < 10:
                record["status"] = "INSUFFICIENT_TRIANGULATION"
            else:
                record["status"] = "USEFUL_POSE_GEOMETRY"

            records.append(record)
            reasons[record["status"]] += 1

        except (
            cv2.error,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            record["status"] = "GEOMETRY_EXCEPTION"
            record["error"] = str(error).splitlines()[0]
            records.append(record)
            reasons[record["status"]] += 1

        if number % 50 == 0 or number == len(candidates):
            print(
                f"Processed {number}/{len(candidates)} "
                f"| useful={reasons['USEFUL_POSE_GEOMETRY']}"
            )

    by_gap = {}

    for gap in range(1, args.max_gap + 1):
        group = [
            row for row in records
            if row["gap"] == gap
        ]

        if not group:
            continue

        by_gap[str(gap)] = {
            "pairs": len(group),
            "status_distribution": dict(Counter(
                row["status"] for row in group
            )),
            "median_mutual_matches": median_or_none(
                row["mutual_matches"] for row in group
            ),
            "median_pose_inlier_fraction": median_or_none(
                row["pose_epipolar_inlier_fraction"]
                for row in group
            ),
            "median_independent_inlier_fraction": median_or_none(
                row["independent_inlier_fraction"]
                for row in group
            ),
            "median_rotation_error_deg": median_or_none(
                row["independent_rotation_error_deg"]
                for row in group
            ),
            "median_translation_direction_error_deg": median_or_none(
                row["independent_translation_direction_error_deg"]
                for row in group
            ),
            "median_parallax_deg": median_or_none(
                row["median_parallax_deg"]
                for row in group
            ),
        }

    frame_stats = defaultdict(lambda: {
        "tested_pairs": 0,
        "useful_pairs": 0,
        "visual_match_pairs": 0,
        "pose_consistent_pairs": 0,
    })

    for row in records:
        for filename in (row["frame_a"], row["frame_b"]):
            item = frame_stats[filename]
            item["tested_pairs"] += 1

            if row["mutual_matches"] >= MIN_MATCHES:
                item["visual_match_pairs"] += 1

            if row["pose_epipolar_inliers"] >= MIN_POSE_INLIERS:
                item["pose_consistent_pairs"] += 1

            if row["status"] == "USEFUL_POSE_GEOMETRY":
                item["useful_pairs"] += 1

    frame_records = []

    for path in paths:
        stats = frame_stats[path.name]

        frame_records.append({
            "filename": path.name,
            "pose_available": path.name in poses,
            **stats,
        })

    report = {
        "project": "Under-construction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "DIAGNOSTIC_ONLY",
        "pose_convention": "PROVISIONAL_POSE_TO_WORLD",
        "camera_extrinsics_verified": False,
        "metric_accuracy_verified": False,
        "photographs": len(paths),
        "pose_matched_frames": len(poses),
        "eligible_pairs_evaluated": len(records),
        "maximum_frame_gap": args.max_gap,
        "status_distribution": dict(reasons),
        "results_by_frame_gap": by_gap,
        "frames_with_useful_pairs": sum(
            row["useful_pairs"] > 0
            for row in frame_records
        ),
        "frames_with_visual_matches_but_no_useful_pairs": sum(
            row["visual_match_pairs"] > 0
            and row["useful_pairs"] == 0
            for row in frame_records
        ),
        "limitations": [
            "Camera-to-IMU extrinsics are unavailable.",
            "Recorded positions are provisionally treated as camera centers.",
            "The image-derived essential matrix is not independent ground truth.",
            "Essential-matrix translation has direction but not metric scale.",
            "Rotation and translation comparisons can be unreliable "
            "for low-parallax or degenerate image pairs.",
            "Pose epipolar consistency alone does not establish "
            "correct camera-to-world calibration.",
            "Triangulation depths are baseline-relative.",
            "The diagnostic does not optimize camera poses.",
            "No CAD alignment or construction progress is inferred.",
        ],
    }

    pair_csv = OUTPUT / "pair_geometry.csv"
    frame_csv = OUTPUT / "frame_geometry.csv"
    report_path = OUTPUT / "pose_geometry_report.json"

    write_csv(pair_csv, records)
    write_csv(frame_csv, frame_records)

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("-" * 60)

    print("Pairs evaluated:", len(records))
    print("Status distribution:", dict(reasons))
    print(
        "Frames with useful pairs:",
        report["frames_with_useful_pairs"],
    )
    print(
        "Frames with visual matches but no useful pairs:",
        report[
            "frames_with_visual_matches_but_no_useful_pairs"
        ],
    )

    print("\nBY FRAME GAP")

    for gap, result in by_gap.items():
        print(
            f"Gap {gap}: "
            f"{result['pairs']} pairs | "
            f"median matches="
            f"{result['median_mutual_matches']} | "
            f"median pose inlier fraction="
            f"{result['median_pose_inlier_fraction']} | "
            f"median parallax="
            f"{result['median_parallax_deg']}"
        )

    print("\nPair diagnostics:", pair_csv)
    print("Frame diagnostics:", frame_csv)
    print("Report:", report_path)

    print(
        "\nWARNING: This experiment does not verify "
        "camera extrinsics or metric reconstruction accuracy."
    )


if __name__ == "__main__":
    main()
