
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
    / "multiview_tracks"
)

BASELINE_REPORT = (
    ROOT / "outputs"
    / "multiview_reconstruction"
    / "multiview_reconstruction_report.json"
)

IMAGE_WIDTH = 960
FEATURE_COUNT = 3000
MATCH_RATIO = 0.78

MIN_MATCHES = 25
MIN_EPIPOLAR_INLIERS = 15

MIN_BASELINE = 0.08
MAX_BASELINE = 5.0

EPIPOLAR_THRESHOLD_PX = 2.5
REPROJECTION_THRESHOLD_PX = 3.0
MIN_PARALLAX_DEG = 1.0
MAX_DEPTH = 100.0


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
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if (
                row.get("pose_matched", "")
                .strip().lower() != "true"
            ):
                continue

            try:
                center = np.array(
                    [
                        float(row[k])
                        for k in ("tx", "ty", "tz")
                    ],
                    dtype=np.float64,
                )

                q = np.array(
                    [
                        float(row[k])
                        for k in ("qx", "qy", "qz", "qw")
                    ],
                    dtype=np.float64,
                )

                norm = np.linalg.norm(q)

                if (
                    not np.isfinite(center).all()
                    or not np.isfinite(q).all()
                    or norm < 1e-10
                ):
                    continue

                R = quaternion_to_rotation(q / norm)

                poses[row["filename"]] = {
                    "center": center,
                    "rotation": R,
                }

            except (ValueError, KeyError, TypeError):
                continue

    return poses


def load_calibration():
    report = json.loads(
        CALIBRATION.read_text(encoding="utf-8")
    )

    calibration = report["calibration"]

    fx, fy, cx, cy = map(
        float, calibration["intrinsics"]
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
        calibration["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(4, 1)

    return K, D


def skew(t):
    x, y, z = t

    return np.array(
        [
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0],
        ],
        dtype=np.float64,
    )


def essential_from_poses(a, b):
    R1 = a["rotation"]
    R2 = b["rotation"]

    C1 = a["center"]
    C2 = b["center"]

    relative_R = R2.T @ R1
    relative_t = R2.T @ (C1 - C2)

    norm = np.linalg.norm(relative_t)

    if norm < 1e-10:
        return None

    return (
        skew(relative_t / norm)
        @ relative_R
    )


def sampson_errors(points1, points2, E, focal):
    x1 = np.column_stack(
        [points1, np.ones(len(points1))]
    )

    x2 = np.column_stack(
        [points2, np.ones(len(points2))]
    )

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

    return (
        np.sqrt(
            numerator / np.maximum(
                denominator, 1e-12
            )
        )
        * focal
    )


class DisjointSet:
    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, item):
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1

    def find(self, item):
        root = item

        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[item] != item:
            nxt = self.parent[item]
            self.parent[item] = root
            item = nxt

        return root

    def union(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def triangulate_multiview(observations, poses):
    # Solve for the world-space point closest
    # to all camera viewing rays.
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)

    eye = np.eye(3)

    for obs in observations:
        pose = poses[obs["frame"]]

        C = pose["center"]
        R = pose["rotation"]

        ray_camera = np.array(
            [
                obs["normalized"][0],
                obs["normalized"][1],
                1.0,
            ],
            dtype=np.float64,
        )

        ray_camera /= np.linalg.norm(ray_camera)

        direction = R @ ray_camera
        direction /= np.linalg.norm(direction)

        projector = (
            eye - np.outer(direction, direction)
        )

        A += projector
        b += projector @ C

    try:
        if np.linalg.cond(A) > 1e8:
            return None

        point = np.linalg.solve(A, b)

    except np.linalg.LinAlgError:
        return None

    if not np.isfinite(point).all():
        return None

    return point


def observation_error(point, obs, poses):
    pose = poses[obs["frame"]]

    local = (
        pose["rotation"].T
        @ (point - pose["center"])
    )

    depth = float(local[2])

    if (
        not np.isfinite(local).all()
        or depth <= 0.05
        or depth >= MAX_DEPTH
    ):
        return math.inf, depth

    predicted = local[:2] / depth

    error = (
        np.linalg.norm(
            predicted - obs["normalized"]
        )
        * obs["focal"]
    )

    return float(error), depth


def maximum_parallax(point, observations, poses):
    directions = []

    for obs in observations:
        C = poses[obs["frame"]]["center"]

        ray = point - C
        norm = np.linalg.norm(ray)

        if norm < 1e-10:
            return 0.0

        directions.append(ray / norm)

    maximum = 0.0

    for i in range(len(directions)):
        for j in range(i + 1, len(directions)):
            cosine = float(
                np.dot(
                    directions[i],
                    directions[j],
                )
            )

            angle = math.degrees(
                math.acos(
                    float(np.clip(cosine, -1, 1))
                )
            )

            maximum = max(maximum, angle)

    return maximum


def validate_track(observations, poses):
    if len(observations) < 3:
        return None, "LESS_THAN_THREE_VIEWS"

    point = triangulate_multiview(
        observations, poses
    )

    if point is None:
        return None, "DEGENERATE_GEOMETRY"

    errors = []

    for obs in observations:
        error, _ = observation_error(
            point, obs, poses
        )
        errors.append(error)

    errors = np.asarray(
        errors, dtype=np.float64
    )

    if not np.isfinite(errors).all():
        return None, "INVALID_DEPTH"

    if np.max(errors) > REPROJECTION_THRESHOLD_PX:
        return None, "HIGH_REPROJECTION_ERROR"

    parallax = maximum_parallax(
        point, observations, poses
    )

    if parallax < MIN_PARALLAX_DEG:
        return None, "LOW_PARALLAX"

    # Leave-one-view-out validation:
    # Fit using all other views, then predict
    # the excluded observation.
    held_out_errors = []

    for index in range(len(observations)):
        subset = (
            observations[:index]
            + observations[index + 1:]
        )

        candidate = triangulate_multiview(
            subset, poses
        )

        if candidate is None:
            return None, "HOLDOUT_DEGENERATE"

        error, _ = observation_error(
            candidate,
            observations[index],
            poses,
        )

        held_out_errors.append(error)

    held_out_errors = np.asarray(
        held_out_errors, dtype=np.float64
    )

    if (
        not np.isfinite(held_out_errors).all()
        or np.max(held_out_errors)
        > REPROJECTION_THRESHOLD_PX
    ):
        return None, "HOLDOUT_INCONSISTENT"

    colors = np.asarray(
        [obs["color"] for obs in observations],
        dtype=np.float64,
    )

    result = {
        "point": point,
        "color": np.clip(
            np.rint(np.median(colors, axis=0)),
            0, 255
        ).astype(np.uint8),
        "views": len(observations),
        "median_reprojection_error_px": float(
            np.median(errors)
        ),
        "max_reprojection_error_px": float(
            np.max(errors)
        ),
        "median_holdout_error_px": float(
            np.median(held_out_errors)
        ),
        "max_holdout_error_px": float(
            np.max(held_out_errors)
        ),
        "maximum_parallax_deg": parallax,
        "frames": [
            obs["frame"] for obs in observations
        ],
    }

    return result, None


def write_ply(path, landmarks):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(
            "comment Experimental multi-view landmarks\n"
        )
        handle.write(
            "comment Camera extrinsics unverified\n"
        )
        handle.write(
            f"element vertex {len(landmarks)}\n"
        )
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")

        for item in landmarks:
            x, y, z = item["point"]
            r, g, b = item["color"]

            handle.write(
                f"{x:.7f} {y:.7f} {z:.7f} "
                f"{int(r)} {int(g)} {int(b)}\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-gap", type=int, default=4
    )

    parser.add_argument(
        "--max-pairs", type=int, default=300
    )

    parser.add_argument(
        "--min-views", type=int, default=3
    )

    args = parser.parse_args()

    if (
        args.max_gap < 1
        or args.max_pairs < 1
        or args.min_views < 3
    ):
        parser.error(
            "max-gap and max-pairs must be positive; "
            "min-views must be at least 3."
        )

    for path in (FRAMES, MANIFEST, CALIBRATION):
        if not path.exists():
            raise FileNotFoundError(path)

    OUTPUT.mkdir(
        parents=True, exist_ok=True
    )

    paths = sorted(FRAMES.glob("*.jpg"))
    poses = load_poses()
    K, D = load_calibration()

    sift = cv2.SIFT_create(
        nfeatures=FEATURE_COUNT
    )

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    @lru_cache(maxsize=24)
    def features(index):
        image = cv2.imread(str(paths[index]))

        if image is None:
            return None

        h, w = image.shape[:2]
        scale = min(1.0, IMAGE_WIDTH / w)

        if scale < 1.0:
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

        if (
            descriptors is None
            or not keypoints
        ):
            return None

        pixels = np.asarray(
            [kp.pt for kp in keypoints],
            dtype=np.float64,
        )

        normalized = cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2),
            Ks,
            D,
        ).reshape(-1, 2)

        focal = float(
            (Ks[0, 0] + Ks[1, 1]) / 2
        )

        return {
            "image": image,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "normalized": normalized,
            "focal": focal,
        }

    def mutual_matches(a, b):
        d1 = a["descriptors"]
        d2 = b["descriptors"]

        if len(d1) < 2 or len(d2) < 2:
            return []

        forward = matcher.knnMatch(
            d1, d2, k=2
        )

        backward = matcher.knnMatch(
            d2, d1, k=2
        )

        def ratio_filter(pairs):
            good = {}

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if (
                    first.distance
                    < MATCH_RATIO * second.distance
                ):
                    good[first.queryIdx] = first

            return good

        fwd = ratio_filter(forward)
        rev = ratio_filter(backward)

        matches = []

        for idx, match in fwd.items():
            reverse = rev.get(match.trainIdx)

            if (
                reverse is not None
                and reverse.trainIdx == idx
            ):
                matches.append(match)

        return matches

    candidates = []

    for i, path in enumerate(paths):
        pose1 = poses.get(path.name)

        if pose1 is None:
            continue

        for gap in range(1, args.max_gap + 1):
            j = i + gap

            if j >= len(paths):
                continue

            pose2 = poses.get(paths[j].name)

            if pose2 is None:
                continue

            baseline = float(
                np.linalg.norm(
                    pose1["center"] - pose2["center"]
                )
            )

            if (
                MIN_BASELINE
                <= baseline
                <= MAX_BASELINE
            ):
                candidates.append((i, j))

    if len(candidates) > args.max_pairs:
        indices = np.linspace(
            0,
            len(candidates) - 1,
            args.max_pairs,
            dtype=int,
        )

        candidates = [
            candidates[int(k)]
            for k in indices
        ]

    print("\nUNDER-CONSTRUCTION")
    print("Multi-view landmark reconstruction")
    print("-" * 58)
    print("Photographs:", len(paths))
    print("Pose-matched frames:", len(poses))
    print("Candidate pairs:", len(candidates))
    print("Minimum views:", args.min_views)
    print("Camera extrinsics: UNVERIFIED")

    # Each node is (frame_index, keypoint_index).
    # A node refers to one particular feature observation.
    dsu = DisjointSet()
    observation_data = {}

    pair_counts = Counter()
    pair_records = []

    for number, (i, j) in enumerate(
        candidates, start=1
    ):
        a = features(i)
        b = features(j)

        if a is None or b is None:
            pair_counts["NO_FEATURES"] += 1
            continue

        matches = mutual_matches(a, b)

        if len(matches) < MIN_MATCHES:
            pair_counts["INSUFFICIENT_MATCHES"] += 1
            continue

        E = essential_from_poses(
            poses[paths[i].name],
            poses[paths[j].name],
        )

        if E is None:
            pair_counts["INVALID_POSE"] += 1
            continue

        indices1 = np.asarray(
            [m.queryIdx for m in matches],
            dtype=int,
        )

        indices2 = np.asarray(
            [m.trainIdx for m in matches],
            dtype=int,
        )

        p1 = a["normalized"][indices1]
        p2 = b["normalized"][indices2]

        errors = sampson_errors(
            p1,
            p2,
            E,
            (a["focal"] + b["focal"]) / 2,
        )

        valid = (
            np.isfinite(p1).all(axis=1)
            & np.isfinite(p2).all(axis=1)
            & np.isfinite(errors)
            & (errors <= EPIPOLAR_THRESHOLD_PX)
        )

        if valid.sum() < MIN_EPIPOLAR_INLIERS:
            pair_counts[
                "INSUFFICIENT_EPIPOLAR_INLIERS"
            ] += 1
            continue

        accepted_edges = 0

        for k in np.flatnonzero(valid):
            idx1 = int(indices1[k])
            idx2 = int(indices2[k])

            node1 = (i, idx1)
            node2 = (j, idx2)

            for node, data in (
                (node1, a),
                (node2, b),
            ):
                if node in observation_data:
                    continue

                frame_index, kp_index = node

                pixel = np.rint(
                    data["keypoints"][kp_index].pt
                ).astype(int)

                image = data["image"]
                h, w = image.shape[:2]

                px = int(np.clip(
                    pixel[0], 0, w - 1
                ))

                py = int(np.clip(
                    pixel[1], 0, h - 1
                ))

                bgr = image[py, px]
                rgb = bgr[::-1].copy()

                observation_data[node] = {
                    "frame": paths[frame_index].name,
                    "frame_index": frame_index,
                    "keypoint_index": kp_index,
                    "normalized": (
                        data["normalized"][kp_index]
                    ),
                    "focal": data["focal"],
                    "color": rgb,
                }

            dsu.union(node1, node2)
            accepted_edges += 1

        pair_counts["GEOMETRY_ACCEPTED"] += 1

        pair_records.append({
            "frame_a": paths[i].name,
            "frame_b": paths[j].name,
            "mutual_matches": len(matches),
            "epipolar_inliers": int(valid.sum()),
            "edges_added": accepted_edges,
        })

        if (
            number % 50 == 0
            or number == len(candidates)
        ):
            print(
                f"Processed {number}/{len(candidates)} "
                f"| accepted pairs="
                f"{pair_counts['GEOMETRY_ACCEPTED']}"
            )

    components = defaultdict(list)

    for node in observation_data:
        components[dsu.find(node)].append(
            observation_data[node]
        )

    print("\nFeature graph components:", len(components))

    rejection_counts = Counter()
    landmarks = []
    track_lengths = Counter()

    for observations in components.values():
        # A connected component with two keypoints
        # from the same frame is ambiguous. Reject it
        # rather than silently selecting one.
        frame_names = [
            obs["frame"] for obs in observations
        ]

        if len(frame_names) != len(set(frame_names)):
            rejection_counts[
                "DUPLICATE_FRAME_IN_TRACK"
            ] += 1
            continue

        if len(observations) < args.min_views:
            rejection_counts[
                "INSUFFICIENT_TRACK_LENGTH"
            ] += 1
            continue

        observations.sort(
            key=lambda obs: obs["frame_index"]
        )

        result, reason = validate_track(
            observations, poses
        )

        if result is None:
            rejection_counts[reason] += 1
            continue

        landmarks.append(result)
        track_lengths[result["views"]] += 1

    ply_path = (
        OUTPUT / "validated_multiview_landmarks.ply"
    )

    write_ply(ply_path, landmarks)

    landmark_records = []

    for index, landmark in enumerate(landmarks):
        landmark_records.append({
            "id": index,
            "xyz": landmark["point"].tolist(),
            "rgb": landmark["color"].tolist(),
            "views": landmark["views"],
            "frames": landmark["frames"],
            "median_reprojection_error_px": (
                landmark[
                    "median_reprojection_error_px"
                ]
            ),
            "max_reprojection_error_px": (
                landmark[
                    "max_reprojection_error_px"
                ]
            ),
            "median_holdout_error_px": (
                landmark[
                    "median_holdout_error_px"
                ]
            ),
            "max_holdout_error_px": (
                landmark[
                    "max_holdout_error_px"
                ]
            ),
            "maximum_parallax_deg": (
                landmark["maximum_parallax_deg"]
            ),
        })

    landmark_json_path = (
        OUTPUT / "validated_landmarks.json"
    )

    landmark_json_path.write_text(
        json.dumps(
            landmark_records, indent=2
        ),
        encoding="utf-8",
    )

    baseline = None

    if BASELINE_REPORT.exists():
        try:
            baseline = json.loads(
                BASELINE_REPORT.read_text(
                    encoding="utf-8"
                )
            )
        except (ValueError, OSError):
            pass

    reprojection_errors = [
        item["median_reprojection_error_px"]
        for item in landmarks
    ]

    holdout_errors = [
        item["median_holdout_error_px"]
        for item in landmarks
    ]

    parallaxes = [
        item["maximum_parallax_deg"]
        for item in landmarks
    ]

    supported_frames = sorted({
        frame
        for item in landmarks
        for frame in item["frames"]
    })

    report = {
        "project": "Under-construction",
        "generated_at_utc": (
            datetime.now(timezone.utc).isoformat()
        ),
        "status": "EXPERIMENTAL",
        "method": "MULTIVIEW_FEATURE_TRACKS",
        "frames_available": len(paths),
        "frames_with_pose": len(poses),
        "candidate_pairs": len(candidates),
        "pair_statistics": dict(pair_counts),
        "feature_graph_components": len(components),
        "landmarks_validated": len(landmarks),
        "frames_supporting_validated_landmarks": (
            len(supported_frames)
        ),
        "track_length_distribution": {
            str(k): v
            for k, v in sorted(
                track_lengths.items()
            )
        },
        "track_rejections": dict(
            rejection_counts
        ),
        "median_landmark_reprojection_error_px": (
            float(np.median(reprojection_errors))
            if reprojection_errors else None
        ),
        "median_landmark_holdout_error_px": (
            float(np.median(holdout_errors))
            if holdout_errors else None
        ),
        "median_landmark_maximum_parallax_deg": (
            float(np.median(parallaxes))
            if parallaxes else None
        ),
        "baseline": {
            "raw_pairwise_points": (
                baseline.get("raw_points")
                if baseline else None
            ),
            "voxel_fused_points": (
                baseline.get("fused_points")
                if baseline else None
            ),
        },
        "verification": {
            "camera_extrinsics_verified": False,
            "metric_geometry_verified": False,
            "cad_registration_verified": False,
        },
        "limitations": [
            "Camera-to-IMU extrinsics remain unknown.",
            "The pose-to-world convention is provisional.",
            "Recorded positions are treated as camera centers.",
            "The camera pose model is not bundle-adjusted.",
            "Feature graph merges are not globally optimized.",
            "Tracks with duplicate-frame observations "
            "are conservatively rejected.",
            "Leave-one-out testing uses the same "
            "recorded camera-pose model.",
            "This is not independent ground-truth "
            "validation of geometry.",
            "Landmark count cannot be directly compared "
            "with voxel-fused pairwise point count.",
            "No dense reconstruction or CAD alignment.",
        ],
    }

    report_path = (
        OUTPUT / "multiview_tracks_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("-" * 58)

    print(
        "Accepted feature-matching pairs:",
        pair_counts["GEOMETRY_ACCEPTED"],
    )

    print(
        "Feature graph components:",
        len(components),
    )

    print(
        "Validated multi-view landmarks:",
        len(landmarks),
    )

    print(
        "Frames supporting landmarks:",
        len(supported_frames),
    )

    print(
        "Track length distribution:",
        dict(track_lengths),
    )

    print(
        "Track rejections:",
        dict(rejection_counts),
    )

    if reprojection_errors:
        print(
            "Median landmark reprojection error:",
            f"{np.median(reprojection_errors):.3f} px",
        )

        print(
            "Median held-out observation error:",
            f"{np.median(holdout_errors):.3f} px",
        )

    print("\nPoint cloud:", ply_path)
    print("Landmarks:", landmark_json_path)
    print("Report:", report_path)

    print(
        "\nWARNING: Camera extrinsics and "
        "metric accuracy remain unverified."
    )


if __name__ == "__main__":
    main()
