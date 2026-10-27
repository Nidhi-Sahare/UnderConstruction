
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "floor_2_2025-10-28_run_2"

FRAMES_DIR = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data" / "hilti_frames"
    / SEQUENCE / "cam0" / "frames"
)

CALIBRATION_PATH = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data" / "hilti_calibration"
    / "camera_calibration_report.json"
)

OUTPUT_DIR = ROOT / "outputs" / "reconstruction_3d"

MAX_IMAGE_WIDTH = 960
MAX_FEATURES = 3500
RATIO_THRESHOLD = 0.75
MIN_MATCHES = 35
MIN_POSE_INLIERS = 25
MIN_TRIANGULATED_POINTS = 25
MAX_REPROJECTION_ERROR_PX = 3.0
MIN_PARALLAX_DEG = 0.75


def load_calibration():
    data = json.loads(
        CALIBRATION_PATH.read_text(encoding="utf-8")
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

    D = np.asarray(
        data["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(-1)

    if D.size != 4:
        raise ValueError(
            "Expected four OpenCV fisheye coefficients."
        )

    return K, D


def load_image(path, K, D):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Cannot read: {path}")

    height, width = image.shape[:2]

    scale = min(1.0, MAX_IMAGE_WIDTH / width)

    if scale < 1.0:
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    scaled_K = K.copy()
    scaled_K[0, :] *= scale
    scaled_K[1, :] *= scale

    h, w = image.shape[:2]

    # The projection is intentionally fixed and documented.
    # Unlike the dashboard's viewing rectification, this
    # matrix is also used for reconstruction geometry.
    focal = 0.75 * min(w, h)

    rectified_K = np.array(
        [
            [focal, 0, w / 2],
            [0, focal, h / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    rectified = cv2.fisheye.undistortImage(
        image,
        scaled_K,
        D,
        Knew=rectified_K,
        new_size=(w, h),
    )

    return rectified, rectified_K


def extract_features(image, detector):
    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = (
        detector.detectAndCompute(gray, None)
    )

    return keypoints, descriptors


def match_features(desc_a, desc_b, matcher):
    if desc_a is None or desc_b is None:
        return []

    if len(desc_a) < 2 or len(desc_b) < 2:
        return []

    forward = matcher.knnMatch(
        desc_a, desc_b, k=2
    )

    reverse = matcher.knnMatch(
        desc_b, desc_a, k=2
    )

    def ratio_filter(pairs):
        accepted = {}

        for pair in pairs:
            if len(pair) != 2:
                continue

            first, second = pair

            if (
                first.distance
                < RATIO_THRESHOLD * second.distance
            ):
                accepted[first.queryIdx] = first

        return accepted

    forward_good = ratio_filter(forward)
    reverse_good = ratio_filter(reverse)

    mutual = []

    for query_idx, match in forward_good.items():
        reverse_match = reverse_good.get(
            match.trainIdx
        )

        if (
            reverse_match is not None
            and reverse_match.trainIdx == query_idx
        ):
            mutual.append(match)

    return sorted(
        mutual,
        key=lambda item: item.distance,
    )


def estimate_pair(
    image_a,
    image_b,
    K,
    detector,
    matcher,
):
    key_a, desc_a = extract_features(
        image_a, detector
    )
    key_b, desc_b = extract_features(
        image_b, detector
    )

    matches = match_features(
        desc_a, desc_b, matcher
    )

    result = {
        "keypoints_a": len(key_a),
        "keypoints_b": len(key_b),
        "mutual_matches": len(matches),
        "essential_inliers": 0,
        "pose_inliers": 0,
        "triangulated_points": 0,
        "status": "REJECTED",
    }

    if len(matches) < MIN_MATCHES:
        result["reason"] = "Insufficient feature matches"
        return result, None

    points_a = np.float64(
        [key_a[m.queryIdx].pt for m in matches]
    )

    points_b = np.float64(
        [key_b[m.trainIdx].pt for m in matches]
    )

    essential, essential_mask = (
        cv2.findEssentialMat(
            points_a,
            points_b,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
    )

    if essential is None or essential_mask is None:
        result["reason"] = "Essential matrix failed"
        return result, None

    if essential.shape != (3, 3):
        result["reason"] = (
            "Ambiguous essential matrix solution"
        )
        return result, None

    essential_inliers = (
        essential_mask.ravel() != 0
    )

    result["essential_inliers"] = int(
        essential_inliers.sum()
    )

    if result["essential_inliers"] < MIN_POSE_INLIERS:
        result["reason"] = (
            "Insufficient essential-matrix inliers"
        )
        return result, None

    pa = points_a[essential_inliers]
    pb = points_b[essential_inliers]

    count, R, t, pose_mask = cv2.recoverPose(
        essential,
        pa,
        pb,
        K,
    )

    result["pose_inliers"] = int(count)

    if count < MIN_POSE_INLIERS:
        result["reason"] = (
            "Insufficient relative-pose inliers"
        )
        return result, None

    valid_pose = pose_mask.ravel() != 0

    pa = pa[valid_pose]
    pb = pb[valid_pose]

    P1 = K @ np.hstack(
        [
            np.eye(3),
            np.zeros((3, 1)),
        ]
    )

    P2 = K @ np.hstack([R, t])

    homogeneous = cv2.triangulatePoints(
        P1,
        P2,
        pa.T,
        pb.T,
    )

    w = homogeneous[3]

    safe_w = np.abs(w) > 1e-10

    xyz = np.full(
        (len(w), 3),
        np.nan,
        dtype=np.float64,
    )

    xyz[safe_w] = (
        homogeneous[:3, safe_w]
        / w[safe_w]
    ).T

    xyz_b = (R @ xyz.T + t).T

    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(xyz_b).all(axis=1)
        & (xyz[:, 2] > 0)
        & (xyz_b[:, 2] > 0)
    )

    def project(points, rotation, translation):
        transformed = (
            rotation @ points.T
            + translation
        ).T

        pixels = (
            K @ transformed.T
        ).T

        return pixels[:, :2] / pixels[:, 2:3]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        reproj_a = project(
            xyz,
            np.eye(3),
            np.zeros((3, 1)),
        )

        reproj_b = project(
            xyz,
            R,
            t,
        )

    error_a = np.linalg.norm(
        reproj_a - pa, axis=1
    )

    error_b = np.linalg.norm(
        reproj_b - pb, axis=1
    )

    valid &= (
        (error_a < MAX_REPROJECTION_ERROR_PX)
        & (error_b < MAX_REPROJECTION_ERROR_PX)
    )

    # Camera 2 center expressed in camera 1 coordinates.
    center_b = (-R.T @ t).reshape(3)

    ray_a = xyz
    ray_b = xyz - center_b

    norms = (
        np.linalg.norm(ray_a, axis=1)
        * np.linalg.norm(ray_b, axis=1)
    )

    safe_norms = np.maximum(norms, 1e-12)

    cosine = np.sum(
        ray_a * ray_b,
        axis=1,
    ) / safe_norms

    parallax = np.degrees(
        np.arccos(np.clip(cosine, -1.0, 1.0))
    )

    valid &= (
        np.isfinite(parallax)
        & (parallax >= MIN_PARALLAX_DEG)
    )

    xyz = xyz[valid]
    pa = pa[valid]
    error_a = error_a[valid]
    error_b = error_b[valid]
    parallax = parallax[valid]

    result["triangulated_points"] = len(xyz)

    if len(xyz) < MIN_TRIANGULATED_POINTS:
        result["reason"] = (
            "Too few reliable triangulated points"
        )
        return result, None

    # Reject pairs with very weak overall parallax.
    median_parallax = float(
        np.median(parallax)
    )

    result["median_parallax_deg"] = (
        median_parallax
    )

    result["median_reprojection_error_px"] = (
        float(
            np.median(
                (error_a + error_b) / 2
            )
        )
    )

    # Keep colors from the first rectified photograph.
    sample_xy = np.rint(pa).astype(int)

    h, w = image_a.shape[:2]

    sample_xy[:, 0] = np.clip(
        sample_xy[:, 0], 0, w - 1
    )

    sample_xy[:, 1] = np.clip(
        sample_xy[:, 1], 0, h - 1
    )

    bgr = image_a[
        sample_xy[:, 1],
        sample_xy[:, 0],
    ]

    rgb = bgr[:, ::-1].copy()

    result["status"] = "ACCEPTED"
    result["reason"] = None

    return result, {
        "points": xyz,
        "colors": rgb,
        "R": R,
        "t": t,
    }


def write_ply(path, points, colors):
    if len(points) != len(colors):
        raise ValueError(
            "Point and color counts differ"
        )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(
            "comment Relative two-view reconstruction\n"
        )
        file.write(
            "comment Scale is arbitrary, not metres\n"
        )
        file.write(
            f"element vertex {len(points)}\n"
        )
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        for point, color in zip(points, colors):
            x, y, z = map(float, point)
            r, g, b = map(int, color)

            file.write(
                f"{x:.8f} {y:.8f} {z:.8f} "
                f"{r} {g} {b}\n"
            )


def save_preview(path, image_a, image_b, result):
    height = 380

    def fit(image):
        h, w = image.shape[:2]

        new_w = int(
            round(w * height / h)
        )

        return cv2.resize(
            image,
            (new_w, height),
            interpolation=cv2.INTER_AREA,
        )

    left = fit(image_a)
    right = fit(image_b)

    combined = np.hstack(
        [left, right]
    )

    label = (
        f"Matches: {result['mutual_matches']} | "
        f"3D points: {result['triangulated_points']}"
    )

    cv2.rectangle(
        combined,
        (0, 0),
        (combined.shape[1], 34),
        (20, 25, 30),
        -1,
    )

    cv2.putText(
        combined,
        label,
        (12, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imwrite(str(path), combined)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=100,
        help="Maximum pairs to evaluate",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Index difference between paired frames",
    )

    args = parser.parse_args()

    if args.max_pairs < 1 or args.stride < 1:
        parser.error(
            "--max-pairs and --stride must be positive"
        )

    if not FRAMES_DIR.is_dir():
        raise FileNotFoundError(FRAMES_DIR)

    paths = sorted(
        FRAMES_DIR.glob("*.jpg")
    )

    if len(paths) < 2:
        raise RuntimeError(
            "At least two photographs are required."
        )

    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "This OpenCV installation lacks SIFT."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    K, D = load_calibration()

    detector = cv2.SIFT_create(
        nfeatures=MAX_FEATURES
    )

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    @__import__("functools").lru_cache(maxsize=12)
    def prepared(index):
        return load_image(
            paths[index], K, D
        )

    attempts = []
    best = None

    available_pairs = [
        (i, i + args.stride)
        for i in range(
            0, len(paths) - args.stride
        )
    ]

    # Spread the test across the sequence rather than
    # examining only the beginning of the walkthrough.
    selected_indices = np.linspace(
        0,
        len(available_pairs) - 1,
        min(args.max_pairs, len(available_pairs)),
        dtype=int,
    )

    print("\nUNDER-CONSTRUCTION")
    print("Sparse 3D reconstruction")
    print("-" * 52)
    print(f"Photographs available: {len(paths)}")
    print(
        f"Pairs to evaluate: {len(selected_indices)}"
    )
    print(
        "Scale: arbitrary (not CAD metres)"
    )

    for attempt_no, pair_index in enumerate(
        selected_indices, start=1
    ):
        i, j = available_pairs[int(pair_index)]

        image_a, K_a = prepared(i)
        image_b, K_b = prepared(j)

        if not np.allclose(K_a, K_b):
            raise RuntimeError(
                "Paired images have incompatible "
                "rectified intrinsics."
            )

        try:
            result, cloud = estimate_pair(
                image_a,
                image_b,
                K_a,
                detector,
                matcher,
            )

        except cv2.error as error:
            result = {
                "status": "REJECTED",
                "reason": (
                    "OpenCV geometric estimation failed: "
                    + str(error).splitlines()[0]
                ),
                "triangulated_points": 0,
            }
            cloud = None

        record = {
            "frame_a": paths[i].name,
            "frame_b": paths[j].name,
            "index_a": i,
            "index_b": j,
            **result,
        }

        attempts.append(record)

        print(
            f"[{attempt_no:03d}/"
            f"{len(selected_indices):03d}] "
            f"{paths[i].name} -> "
            f"{paths[j].name} | "
            f"{result['status']} | "
            f"points="
            f"{result.get('triangulated_points', 0)}"
        )

        if cloud is None:
            continue

        # Prefer a larger geometrically filtered cloud.
        score = (
            len(cloud["points"]),
            result["median_parallax_deg"],
            -result["median_reprojection_error_px"],
        )

        if best is None or score > best["score"]:
            best = {
                "score": score,
                "record": record,
                "cloud": cloud,
                "image_a": image_a.copy(),
                "image_b": image_b.copy(),
            }

    report = {
        "schema_version": "1.0",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "project": "Under-construction",
        "reconstruction_type": (
            "RELATIVE_TWO_VIEW_SPARSE"
        ),
        "metric_scale_verified": False,
        "cad_alignment_verified": False,
        "camera_extrinsics_verified": False,
        "configuration": {
            "max_image_width": MAX_IMAGE_WIDTH,
            "max_features": MAX_FEATURES,
            "pair_stride": args.stride,
            "max_pairs": args.max_pairs,
            "minimum_parallax_deg": MIN_PARALLAX_DEG,
            "maximum_reprojection_error_px": (
                MAX_REPROJECTION_ERROR_PX
            ),
        },
        "frames_available": len(paths),
        "pairs_tested": len(attempts),
        "pairs_accepted": sum(
            record["status"] == "ACCEPTED"
            for record in attempts
        ),
        "best_pair": (
            best["record"]
            if best is not None
            else None
        ),
        "attempts": attempts,
        "limitations": [
            "Only one best pair is exported.",
            "This is not a fused multi-view reconstruction.",
            "Translation scale is arbitrary.",
            "Camera extrinsics are not applied.",
            "CAD registration is not applied.",
            "Image coverage does not imply 3D coverage.",
            "Reprojection filtering is not ground-truth validation.",
            "No construction completion is inferred.",
        ],
    }

    report_path = (
        OUTPUT_DIR / "reconstruction_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if best is None:
        print("\nNo geometrically acceptable pair found.")
        print(f"Report: {report_path}")
        print(
            "Try another --stride value; "
            "do not interpret failure as absence "
            "of reconstructable geometry."
        )
        return

    cloud_path = (
        OUTPUT_DIR / "sparse_reconstruction.ply"
    )

    preview_path = (
        OUTPUT_DIR / "best_pair_preview.jpg"
    )

    write_ply(
        cloud_path,
        best["cloud"]["points"],
        best["cloud"]["colors"],
    )

    save_preview(
        preview_path,
        best["image_a"],
        best["image_b"],
        best["record"],
    )

    print("\nBEST RECONSTRUCTION")
    print("-" * 52)
    print(
        f"Frames: "
        f"{best['record']['frame_a']} / "
        f"{best['record']['frame_b']}"
    )
    print(
        f"3D points: "
        f"{best['record']['triangulated_points']}"
    )
    print(
        f"Median parallax: "
        f"{best['record']['median_parallax_deg']:.2f}°"
    )
    print(
        f"Median reprojection error: "
        f"{best['record']['median_reprojection_error_px']:.3f} px"
    )
    print(f"\nPoint cloud: {cloud_path}")
    print(f"Preview: {preview_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
