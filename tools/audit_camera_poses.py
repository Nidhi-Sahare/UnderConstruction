
from __future__ import annotations

import argparse
import csv
import itertools
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

FRAMES_DIR = (
    DATA / "hilti_frames" / SEQUENCE
    / "cam0" / "frames"
)

MANIFEST = (
    DATA / "hilti_frames" / SEQUENCE
    / "cam0" / "frame_manifest.csv"
)

CALIBRATION = (
    DATA / "hilti_calibration"
    / "camera_calibration_report.json"
)

OUTPUT = ROOT / "outputs" / "camera_pose_audit"

IMAGE_WIDTH = 960
FEATURE_COUNT = 2500
RATIO = 0.78
MIN_MATCHES = 25
MIN_BASELINE = 0.03
MAX_BASELINE = 8.0
PIXEL_THRESHOLD = 2.5


def load_metadata():
    with MANIFEST.open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    by_name = {}

    for row in rows:
        name = row["filename"]
        matched = (
            row.get("pose_matched", "").lower()
            == "true"
        )

        record = {
            "filename": name,
            "matched": False,
        }

        if matched:
            try:
                position = np.array(
                    [
                        float(row[key])
                        for key in ("tx", "ty", "tz")
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        float(row[key])
                        for key in ("qx", "qy", "qz", "qw")
                    ],
                    dtype=np.float64,
                )

                norm = np.linalg.norm(quaternion)

                if (
                    np.isfinite(position).all()
                    and np.isfinite(quaternion).all()
                    and norm > 1e-8
                ):
                    record.update({
                        "matched": True,
                        "position": position,
                        "quaternion": quaternion / norm,
                    })

            except (KeyError, ValueError, TypeError):
                pass

        by_name[name] = record

    return by_name


def load_calibration():
    data = json.loads(
        CALIBRATION.read_text(encoding="utf-8")
    )["calibration"]

    fx, fy, cx, cy = map(
        float, data["intrinsics"]
    )

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    D = np.array(
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


def rotation_hypotheses():
    # All 24 proper signed axis-permutation rotations.
    # These are hypotheses, not known camera extrinsics.
    result = []

    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            A = np.zeros((3, 3), dtype=np.float64)

            for col, axis in enumerate(perm):
                A[axis, col] = signs[col]

            if round(np.linalg.det(A)) == 1:
                result.append({
                    "name": (
                        "perm_" + "".join(map(str, perm))
                        + "_sign_" + "".join(
                            "p" if s > 0 else "n"
                            for s in signs
                        )
                    ),
                    "matrix": A,
                })

    return result


def essential_from_pose(R1, R2, center1, center2):
    # R1 and R2 are camera-to-world rotations.
    # Relative world-to-camera transform:
    R = R2.T @ R1
    t = R2.T @ (center1 - center2)

    length = np.linalg.norm(t)

    if length < 1e-10:
        return None

    t /= length

    tx = np.array(
        [
            [0, -t[2], t[1]],
            [t[2], 0, -t[0]],
            [-t[1], t[0], 0],
        ],
        dtype=np.float64,
    )

    return tx @ R


def epipolar_errors(points1, points2, E, focal):
    ones = np.ones((len(points1), 1))

    x1 = np.hstack([points1, ones])
    x2 = np.hstack([points2, ones])

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

    sampson = numerator / np.maximum(
        denominator, 1e-12
    )

    return np.sqrt(sampson) * focal


def extract_pair_data(
    image1,
    image2,
    K,
    D,
    sift,
    matcher,
):
    def features(path):
        image = cv2.imread(str(path))

        if image is None:
            raise RuntimeError(
                f"Could not read {path}"
            )

        h, w = image.shape[:2]
        scale = min(1.0, IMAGE_WIDTH / w)

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
            image, cv2.COLOR_BGR2GRAY
        )

        keypoints, descriptors = (
            sift.detectAndCompute(gray, None)
        )

        scaled_K = K.copy()
        scaled_K[0, :] *= scale
        scaled_K[1, :] *= scale

        return keypoints, descriptors, scaled_K

    k1, d1, K1 = features(image1)
    k2, d2, K2 = features(image2)

    if (
        d1 is None or d2 is None
        or len(d1) < 2 or len(d2) < 2
    ):
        return None

    forward = matcher.knnMatch(d1, d2, k=2)
    backward = matcher.knnMatch(d2, d1, k=2)

    def filter_matches(pairs):
        good = {}

        for pair in pairs:
            if len(pair) != 2:
                continue

            a, b = pair

            if a.distance < RATIO * b.distance:
                good[a.queryIdx] = a

        return good

    forward = filter_matches(forward)
    backward = filter_matches(backward)

    matches = []

    for idx, match in forward.items():
        reverse = backward.get(match.trainIdx)

        if (
            reverse is not None
            and reverse.trainIdx == idx
        ):
            matches.append(match)

    if len(matches) < MIN_MATCHES:
        return None

    pixels1 = np.float64(
        [k1[m.queryIdx].pt for m in matches]
    )

    pixels2 = np.float64(
        [k2[m.trainIdx].pt for m in matches]
    )

    # Use the original equidistant fisheye model.
    # Undistort matched points directly into normalized
    # camera coordinates; avoid image rectification.
    normalized1 = cv2.fisheye.undistortPoints(
        pixels1.reshape(-1, 1, 2),
        K1,
        D,
    ).reshape(-1, 2)

    normalized2 = cv2.fisheye.undistortPoints(
        pixels2.reshape(-1, 1, 2),
        K2,
        D,
    ).reshape(-1, 2)

    finite = (
        np.isfinite(normalized1).all(axis=1)
        & np.isfinite(normalized2).all(axis=1)
    )

    if finite.sum() < MIN_MATCHES:
        return None

    return {
        "p1": normalized1[finite],
        "p2": normalized2[finite],
        "focal": float(
            (K1[0, 0] + K1[1, 1]
             + K2[0, 0] + K2[1, 1]) / 4
        ),
        "matches": int(finite.sum()),
    }


def evaluate_hypothesis(pairs, convention, axis):
    scores = []
    inlier_counts = []
    total_matches = 0
    total_inliers = 0

    for pair in pairs:
        a = pair["a"]
        b = pair["b"]

        Ra = quaternion_to_rotation(
            a["quaternion"]
        )

        Rb = quaternion_to_rotation(
            b["quaternion"]
        )

        if convention == "pose_to_world":
            R1 = Ra @ axis
            R2 = Rb @ axis
        else:
            R1 = Ra.T @ axis
            R2 = Rb.T @ axis

        E = essential_from_pose(
            R1,
            R2,
            a["position"],
            b["position"],
        )

        if E is None:
            continue

        errors = epipolar_errors(
            pair["p1"],
            pair["p2"],
            E,
            pair["focal"],
        )

        valid = np.isfinite(errors)

        if not valid.any():
            continue

        inliers = (
            valid
            & (errors <= PIXEL_THRESHOLD)
        )

        n = len(errors)
        count = int(inliers.sum())

        total_matches += n
        total_inliers += count

        scores.append(
            float(count / n)
        )

        inlier_counts.append(count)

    if not scores:
        return None

    return {
        "convention": convention,
        "axis_name": None,
        "axis_matrix": axis.tolist(),
        "pairs_evaluated": len(scores),
        "total_matches": total_matches,
        "total_inliers": total_inliers,
        "global_inlier_fraction": (
            total_inliers / max(total_matches, 1)
        ),
        "median_pair_inlier_fraction": (
            float(np.median(scores))
        ),
        "median_pair_inliers": (
            float(np.median(inlier_counts))
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--max-gap",
        type=int,
        default=6,
    )

    args = parser.parse_args()

    if args.max_pairs < 1 or args.max_gap < 1:
        parser.error(
            "Both arguments must be positive."
        )

    if not FRAMES_DIR.exists():
        raise FileNotFoundError(FRAMES_DIR)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    if not CALIBRATION.exists():
        raise FileNotFoundError(CALIBRATION)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = sorted(
        FRAMES_DIR.glob("*.jpg")
    )

    metadata = load_metadata()
    K, D = load_calibration()

    sift = cv2.SIFT_create(
        nfeatures=FEATURE_COUNT
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    @lru_cache(maxsize=16)
    def cached_features(index):
        image = cv2.imread(str(paths[index]))

        if image is None:
            return None

        h, w = image.shape[:2]
        scale = min(1.0, IMAGE_WIDTH / w)

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
            image, cv2.COLOR_BGR2GRAY
        )

        keypoints, descriptors = (
            sift.detectAndCompute(gray, None)
        )

        Ks = K.copy()
        Ks[0, :] *= scale
        Ks[1, :] *= scale

        return keypoints, descriptors, Ks

    def prepare_pair(i, j):
        fa = cached_features(i)
        fb = cached_features(j)

        if fa is None or fb is None:
            return None

        k1, d1, K1 = fa
        k2, d2, K2 = fb

        if (
            d1 is None or d2 is None
            or len(d1) < 2 or len(d2) < 2
        ):
            return None

        forward = matcher.knnMatch(
            d1, d2, k=2
        )

        backward = matcher.knnMatch(
            d2, d1, k=2
        )

        def good(pairs):
            result = {}

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if (
                    first.distance
                    < RATIO * second.distance
                ):
                    result[first.queryIdx] = first

            return result

        fwd = good(forward)
        rev = good(backward)

        matches = []

        for idx, match in fwd.items():
            reverse = rev.get(match.trainIdx)

            if (
                reverse is not None
                and reverse.trainIdx == idx
            ):
                matches.append(match)

        if len(matches) < MIN_MATCHES:
            return None

        p1 = np.float64(
            [k1[m.queryIdx].pt for m in matches]
        )

        p2 = np.float64(
            [k2[m.trainIdx].pt for m in matches]
        )

        p1 = cv2.fisheye.undistortPoints(
            p1.reshape(-1, 1, 2),
            K1,
            D,
        ).reshape(-1, 2)

        p2 = cv2.fisheye.undistortPoints(
            p2.reshape(-1, 1, 2),
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
            "focal": float(
                (
                    K1[0, 0] + K1[1, 1]
                    + K2[0, 0] + K2[1, 1]
                ) / 4
            ),
        }

    proposed_pairs = []

    for i in range(len(paths)):
        for gap in range(1, args.max_gap + 1):
            j = i + gap

            if j >= len(paths):
                continue

            a = metadata.get(paths[i].name)
            b = metadata.get(paths[j].name)

            if not a or not b:
                continue

            if not (
                a["matched"] and b["matched"]
            ):
                continue

            baseline = float(
                np.linalg.norm(
                    a["position"] - b["position"]
                )
            )

            if not (
                MIN_BASELINE
                <= baseline
                <= MAX_BASELINE
            ):
                continue

            proposed_pairs.append(
                (i, j, baseline)
            )

    # Cover the full walkthrough with a bounded
    # number of CPU feature-matching evaluations.
    if len(proposed_pairs) > args.max_pairs:
        chosen = np.linspace(
            0,
            len(proposed_pairs) - 1,
            args.max_pairs,
            dtype=int,
        )

        proposed_pairs = [
            proposed_pairs[int(index)]
            for index in chosen
        ]

    print("\nUNDER-CONSTRUCTION")
    print("Camera pose audit")
    print("-" * 55)
    print(f"Photographs: {len(paths)}")
    print(
        "Pose-matched photographs:",
        sum(
            metadata.get(p.name, {}).get(
                "matched", False
            )
            for p in paths
        ),
    )
    print(
        "Candidate pairs to inspect:",
        len(proposed_pairs),
    )

    usable = []
    rejection_counts = Counter()

    for n, (i, j, baseline) in enumerate(
        proposed_pairs, start=1
    ):
        try:
            geometry = prepare_pair(i, j)
        except cv2.error:
            geometry = None

        if geometry is None:
            rejection_counts[
                "insufficient_matches_or_geometry"
            ] += 1
            continue

        pair = {
            **geometry,
            "a": metadata[paths[i].name],
            "b": metadata[paths[j].name],
            "index_a": i,
            "index_b": j,
            "baseline_m": baseline,
        }

        usable.append(pair)

        print(
            f"[{n:03d}/{len(proposed_pairs):03d}] "
            f"{paths[i].name} -> "
            f"{paths[j].name} | "
            f"matches={len(pair['p1'])} | "
            f"baseline={baseline:.3f}"
        )

    print("\nGeometrically testable pairs:", len(usable))

    hypotheses = []

    if usable:
        axes = rotation_hypotheses()

        for convention in (
            "pose_to_world",
            "world_to_pose",
        ):
            for axis in axes:
                result = evaluate_hypothesis(
                    usable,
                    convention,
                    axis["matrix"],
                )

                if result is None:
                    continue

                result["axis_name"] = axis["name"]
                hypotheses.append(result)

        hypotheses.sort(
            key=lambda h: (
                h["global_inlier_fraction"],
                h["median_pair_inlier_fraction"],
            ),
            reverse=True,
        )

    best = hypotheses[0] if hypotheses else None

    report = {
        "project": "Under-construction",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "status": "DIAGNOSTIC_ONLY",
        "frames_available": len(paths),
        "frames_with_pose": sum(
            metadata.get(p.name, {}).get(
                "matched", False
            )
            for p in paths
        ),
        "pairs_considered": len(proposed_pairs),
        "pairs_testable": len(usable),
        "rejections": dict(rejection_counts),
        "threshold_px": PIXEL_THRESHOLD,
        "tested_conventions": len(hypotheses),
        "best_hypothesis": best,
        "top_hypotheses": hypotheses[:10],
        "usable_pairs": [
            {
                "frame_a": paths[p["index_a"]].name,
                "frame_b": paths[p["index_b"]].name,
                "baseline_m": p["baseline_m"],
                "matches": len(p["p1"]),
            }
            for p in usable
        ],
        "verification": {
            "camera_extrinsics_verified": False,
            "metric_reconstruction_verified": False,
            "cad_registration_verified": False,
        },
        "limitations": [
            "Only signed axis permutations are tested.",
            "An arbitrary camera-to-IMU rotation is not solved.",
            "Camera-to-IMU translation is not modeled.",
            "The position is assumed to be the camera center.",
            "No image correspondence ground truth exists.",
            "Epipolar consistency alone does not verify poses.",
            "The hypotheses are evaluated on the same pairs "
            "used for ranking; independent validation is required.",
            "The pixel threshold uses an approximate "
            "focal-length conversion for normalized coordinates.",
            "No metric point cloud is produced by this audit.",
        ],
    }

    report_path = (
        OUTPUT / "camera_pose_audit.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nAUDIT RESULT")
    print("-" * 55)

    if best:
        print(
            "Best convention:",
            best["convention"],
        )
        print(
            "Axis hypothesis:",
            best["axis_name"],
        )
        print(
            "Global epipolar inlier fraction:",
            f"{100 * best['global_inlier_fraction']:.2f}%",
        )
        print(
            "Median pair inlier fraction:",
            f"{100 * best['median_pair_inlier_fraction']:.2f}%",
        )
        print(
            "\nThese are hypothesis-ranking scores, "
            "NOT verified pose accuracy."
        )
    else:
        print(
            "No hypothesis could be evaluated."
        )
        print(
            "Insufficient overlapping matched-pose "
            "image pairs."
        )

    print("\nReport:", report_path)


if __name__ == "__main__":
    main()
