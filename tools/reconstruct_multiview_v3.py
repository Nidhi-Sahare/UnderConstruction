
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

FRAMES = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
    / "frames"
)

POSE_FILE = (
    ROOT
    / "outputs"
    / "rotation_new_feature_validation"
    / "experimental_corrected_camera_poses.json"
)

LANDMARK_FILE = (
    ROOT
    / "outputs"
    / "multiview_tracks_v2"
    / "validated_landmarks_v2.json"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "multiview_v3_comparison"
)

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

    if (
        data.get("status") != "EXPERIMENTAL"
        or data.get("approved_for_application") is not False
    ):
        raise RuntimeError(
            "Expected an unapproved experimental pose file."
        )

    poses = {}

    for name, item in data["poses"].items():
        C = np.asarray(item["center"], dtype=float)

        rotations = {
            "original": np.asarray(
                item["original_rotation_camera_to_world"],
                dtype=float,
            ),
            "refined": np.asarray(
                item["rotation_camera_to_world"],
                dtype=float,
            ),
        }

        if C.shape != (3,) or not np.isfinite(C).all():
            raise ValueError(f"Invalid center: {name}")

        for mode, R in rotations.items():
            if (
                R.shape != (3, 3)
                or not np.isfinite(R).all()
                or not np.allclose(
                    R.T @ R,
                    np.eye(3),
                    atol=1e-3,
                )
                or abs(np.linalg.det(R) - 1) > 1e-3
            ):
                raise ValueError(
                    f"Invalid {mode} rotation: {name}"
                )

        poses[name] = {
            "center": C,
            **rotations,
        }

    return dict(
        sorted(
            poses.items(),
            key=lambda item: frame_number(item[0]),
        )
    )


def load_exclusions():
    data = json.loads(
        LANDMARK_FILE.read_text(encoding="utf-8")
    )

    if isinstance(data, list):
        landmarks = data
    elif isinstance(data, dict):
        landmarks = next(
            (
                data[key]
                for key in (
                    "landmarks",
                    "validated_landmarks",
                    "points",
                )
                if isinstance(data.get(key), list)
            ),
            None,
        )
    else:
        landmarks = None

    if landmarks is None:
        raise RuntimeError(
            "Cannot find V2 landmark observations."
        )

    result = defaultdict(list)

    for landmark in landmarks:
        for obs in landmark.get("observations", []):
            name = obs.get("frame") or obs.get("filename")
            uv = obs.get("normalized_uv")

            if name is None or uv is None:
                continue

            uv = np.asarray(uv, dtype=float)

            if uv.shape == (2,) and np.isfinite(uv).all():
                result[Path(str(name)).name].append(uv)

    count = sum(map(len, result.values()))

    if count < 100:
        raise RuntimeError(
            "Insufficient saved observations for exclusion."
        )

    return {
        name: np.asarray(values, dtype=float)
        for name, values in result.items()
    }, count


def extract_features(name, sift, width, excluded, radius):
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

    keypoints, descriptors = sift.detectAndCompute(
        image, None
    )

    if descriptors is None:
        return {
            "uv": np.empty((0, 2), dtype=float),
            "descriptors": None,
        }

    xy = np.asarray(
        [kp.pt for kp in keypoints],
        dtype=np.float64,
    )

    scaled_K = K.copy()
    scaled_K[:2, :] *= scale

    uv = cv2.fisheye.undistortPoints(
        xy.reshape(-1, 1, 2),
        scaled_K,
        D,
    ).reshape(-1, 2)

    keep = (
        np.isfinite(uv).all(axis=1)
        & (np.linalg.norm(uv, axis=1) < 10)
    )

    saved = excluded.get(name)

    if saved is not None and len(saved):
        for start in range(0, len(uv), 256):
            stop = min(start + 256, len(uv))

            delta = (
                uv[start:stop, None, :]
                - saved[None, :, :]
            )

            distances = np.sum(delta * delta, axis=2)

            keep[start:stop] &= (
                distances.min(axis=1) > radius * radius
            )

    return {
        "uv": uv[keep],
        "descriptors": descriptors[keep],
    }


def mutual_matches(a, b, ratio):
    da = a["descriptors"]
    db = b["descriptors"]

    if (
        da is None
        or db is None
        or len(da) < 2
        or len(db) < 2
    ):
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    forward = matcher.knnMatch(da, db, k=2)
    backward = matcher.knnMatch(db, da, k=2)

    accepted_forward = {}

    for pair in forward:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            accepted_forward[m.queryIdx] = (
                m.trainIdx,
                float(m.distance),
            )

    accepted_backward = {}

    for pair in backward:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            accepted_backward[m.queryIdx] = m.trainIdx

    return [
        (i, j, distance)
        for i, (j, distance) in accepted_forward.items()
        if accepted_backward.get(j) == i
    ]


def skew(t):
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0],
    ])


def epipolar_errors(uv1, uv2, pose1, pose2):
    R1 = pose1["original"]
    R2 = pose2["original"]

    C1 = pose1["center"]
    C2 = pose2["center"]

    relative_R = R2.T @ R1
    t = R2.T @ (C1 - C2)

    baseline = np.linalg.norm(t)

    if baseline < 1e-9:
        return np.full(len(uv1), np.inf)

    E = skew(t / baseline) @ relative_R

    x1 = np.column_stack([
        uv1,
        np.ones(len(uv1)),
    ])

    x2 = np.column_stack([
        uv2,
        np.ones(len(uv2)),
    ])

    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T

    numerator = np.sum(x2 * Ex1, axis=1) ** 2

    denominator = (
        Ex1[:, 0] ** 2
        + Ex1[:, 1] ** 2
        + Etx2[:, 0] ** 2
        + Etx2[:, 1] ** 2
    )

    return FOCAL * np.sqrt(
        numerator / np.maximum(denominator, 1e-15)
    )


class ConflictAwareUnionFind:
    def __init__(self):
        self.parent = {}
        self.size = {}
        self.frames = {}
        self.conflicts = 0

    def add(self, node):
        if node not in self.parent:
            self.parent[node] = node
            self.size[node] = 1
            self.frames[node] = {node[0]}

    def find(self, node):
        parent = self.parent[node]

        if parent != node:
            self.parent[node] = self.find(parent)

        return self.parent[node]

    def union(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return True

        if self.frames[ra] & self.frames[rb]:
            self.conflicts += 1
            return False

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.frames[ra].update(self.frames[rb])

        del self.frames[rb]

        return True

    def components(self):
        groups = defaultdict(list)

        for node in self.parent:
            groups[self.find(node)].append(node)

        return list(groups.values())


def triangulate_multiview(
    observations,
    poses,
    features,
    mode,
):
    """
    Linear multiview triangulation in normalized
    undistorted camera coordinates.
    """
    equations = []

    for name, feature_id in observations:
        C = poses[name]["center"]
        R = poses[name][mode]

        uv = features[name]["uv"][feature_id]

        rotation_wc = R.T
        translation_wc = -rotation_wc @ C

        P = np.column_stack([
            rotation_wc,
            translation_wc,
        ])

        equations.append(
            uv[0] * P[2] - P[0]
        )

        equations.append(
            uv[1] * P[2] - P[1]
        )

    A = np.asarray(equations)

    try:
        _, _, Vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None

    X = Vt[-1]

    if abs(X[3]) < 1e-12:
        return None

    point = X[:3] / X[3]

    if not np.isfinite(point).all():
        return None

    return point


def evaluate_observations(
    point,
    observations,
    poses,
    features,
    mode,
):
    errors = []
    depths = []
    rays = []
    ranges = []

    for name, feature_id in observations:
        C = poses[name]["center"]
        R = poses[name][mode]

        camera_point = R.T @ (point - C)
        depth = float(camera_point[2])

        uv = features[name]["uv"][feature_id]

        if depth > 1e-9:
            projected = (
                camera_point[:2] / depth
            )

            error = (
                float(np.linalg.norm(projected - uv))
                * FOCAL
            )
        else:
            error = float("inf")

        ray = point - C
        distance = float(np.linalg.norm(ray))

        if distance > 1e-12:
            ray /= distance

        errors.append(error)
        depths.append(depth)
        rays.append(ray)
        ranges.append(distance)

    maximum_parallax = 0.0

    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            cosine = float(
                np.clip(
                    np.dot(rays[i], rays[j]),
                    -1,
                    1,
                )
            )

            angle = float(
                np.degrees(np.arccos(cosine))
            )

            maximum_parallax = max(
                maximum_parallax,
                angle,
            )

    return {
        "errors": np.asarray(errors),
        "depths": np.asarray(depths),
        "ranges": np.asarray(ranges),
        "max_parallax_deg": maximum_parallax,
    }


def evaluate_track(
    observations,
    poses,
    features,
    mode,
    args,
):
    """
    For 4+ views, hold out one deterministic
    observation before triangulation.

    For exactly 3 views, use all three to fit;
    these tracks have no independent held-out test.
    """
    ordered = sorted(
        observations,
        key=lambda node: (
            frame_number(node[0]),
            node[1],
        ),
    )

    has_holdout = len(ordered) >= 4

    if has_holdout:
        holdout_index = len(ordered) // 2

        train = [
            node
            for i, node in enumerate(ordered)
            if i != holdout_index
        ]

        heldout = [ordered[holdout_index]]
    else:
        train = ordered
        heldout = []

    point = triangulate_multiview(
        train,
        poses,
        features,
        mode,
    )

    if point is None:
        return {
            "valid": False,
            "reason": "degenerate",
            "point": None,
            "heldout_px": None,
        }

    training = evaluate_observations(
        point,
        train,
        poses,
        features,
        mode,
    )

    if (
        not np.isfinite(training["errors"]).all()
        or not np.isfinite(training["depths"]).all()
        or not np.isfinite(training["ranges"]).all()
    ):
        reason = "nonfinite"

    elif np.any(training["depths"] <= 0):
        reason = "negative_depth"

    elif np.any(
        training["ranges"] > args.max_range
    ):
        reason = "excessive_range"

    elif (
        training["max_parallax_deg"]
        < args.min_parallax
    ):
        reason = "low_parallax"

    elif np.max(training["errors"]) > args.max_reprojection:
        reason = "training_reprojection"

    else:
        reason = "accepted"

    heldout_px = None

    if heldout:
        test = evaluate_observations(
            point,
            heldout,
            poses,
            features,
            mode,
        )

        heldout_px = float(test["errors"][0])

        if reason == "accepted":
            if (
                not np.isfinite(heldout_px)
                or test["depths"][0] <= 0
                or test["ranges"][0] > args.max_range
                or heldout_px > args.max_heldout
            ):
                reason = "heldout_failure"

    return {
        "valid": reason == "accepted",
        "reason": reason,
        "point": point,
        "training_median_px": float(
            np.median(training["errors"])
        ),
        "training_max_px": float(
            np.max(training["errors"])
        ),
        "heldout_px": heldout_px,
        "max_parallax_deg": float(
            training["max_parallax_deg"]
        ),
    }


def save_ply(path, points):
    points = np.asarray(points, dtype=float)

    if points.size == 0:
        points = np.empty((0, 3))

    with path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(
            f"element vertex {len(points)}\n"
        )
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        for x, y, z in points:
            file.write(
                f"{x:.9f} {y:.9f} {z:.9f}\n"
            )


def finite_median(values):
    values = np.asarray(
        [
            value
            for value in values
            if value is not None
            and np.isfinite(value)
        ],
        dtype=float,
    )

    return (
        float(np.median(values))
        if len(values)
        else None
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
        "--min-pair-matches",
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
        "--min-views",
        type=int,
        default=3,
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
        "--max-heldout",
        type=float,
        default=3.0,
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
        or args.min_pair_matches < 2
        or not 0 <= args.min_original_inlier <= 1
        or args.epipolar_threshold <= 0
        or args.min_views < 3
        or args.min_parallax <= 0
        or args.max_reprojection <= 0
        or args.max_heldout <= 0
        or args.max_range <= 0
    ):
        parser.error("Invalid parameters.")

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    poses = load_poses()
    excluded, excluded_count = load_exclusions()

    names = list(poses)

    print("\nUNDER-CONSTRUCTION")
    print("Multiview Reconstruction V3")
    print("-" * 60)
    print("Supported cameras:", len(names))
    print("V2 observations excluded:", excluded_count)

    sift = cv2.SIFT_create(
        nfeatures=args.features
    )

    features = {}

    for i, name in enumerate(names, start=1):
        features[name] = extract_features(
            name,
            sift,
            args.width,
            excluded,
            args.exclude_radius_px / FOCAL,
        )

        if i % 15 == 0 or i == len(names):
            print(
                f"Features extracted: {i}/{len(names)}"
            )

    # All edge decisions use ORIGINAL poses.
    # Refined poses are not used to select tracks.
    edges = []

    eligible_pairs = 0
    accepted_pairs = 0
    matched_correspondences = 0

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            gap = (
                frame_number(name_b)
                - frame_number(name_a)
            )

            if gap > args.max_gap:
                break

            pa = poses[name_a]
            pb = poses[name_b]

            baseline = float(
                np.linalg.norm(
                    pb["center"] - pa["center"]
                )
            )

            if not 0.08 <= baseline <= 5.0:
                continue

            eligible_pairs += 1

            matches = mutual_matches(
                features[name_a],
                features[name_b],
                args.ratio,
            )

            if len(matches) < args.min_pair_matches:
                continue

            ids_a = np.asarray(
                [match[0] for match in matches],
                dtype=int,
            )

            ids_b = np.asarray(
                [match[1] for match in matches],
                dtype=int,
            )

            errors = epipolar_errors(
                features[name_a]["uv"][ids_a],
                features[name_b]["uv"][ids_b],
                pa,
                pb,
            )

            inliers = (
                errors <= args.epipolar_threshold
            )

            fraction = float(
                np.mean(inliers)
            )

            if fraction < args.min_original_inlier:
                continue

            accepted_pairs += 1
            matched_correspondences += len(matches)

            for index in np.flatnonzero(inliers):
                fa, fb, distance = matches[index]

                edges.append((
                    float(errors[index]),
                    distance,
                    (name_a, int(fa)),
                    (name_b, int(fb)),
                ))

    # Process geometrically stronger edges first.
    edges.sort(
        key=lambda edge: (
            edge[0],
            edge[1],
            edge[2],
            edge[3],
        )
    )

    union = ConflictAwareUnionFind()

    for _, _, node_a, node_b in edges:
        union.union(node_a, node_b)

    components = union.components()

    tracks = [
        component
        for component in components
        if len(component) >= args.min_views
    ]

    tracks.sort(
        key=lambda track: (
            -len(track),
            min(frame_number(node[0]) for node in track),
        )
    )

    print("\nMATCHING")
    print("Eligible pairs:", eligible_pairs)
    print("Accepted pairs:", accepted_pairs)
    print(
        "Matches in accepted pairs:",
        matched_correspondences,
    )
    print("Accepted feature edges:", len(edges))
    print("Prevented conflicting merges:", union.conflicts)
    print("Feature components:", len(components))
    print(
        f"Tracks with {args.min_views}+ views:",
        len(tracks),
    )

    comparison = []
    point_clouds = {
        "original": [],
        "refined": [],
    }

    for track_id, track in enumerate(tracks):
        results = {}

        for mode in ("original", "refined"):
            results[mode] = evaluate_track(
                track,
                poses,
                features,
                mode,
                args,
            )

            if results[mode]["valid"]:
                point_clouds[mode].append(
                    results[mode]["point"]
                )

        original = results["original"]
        refined = results["refined"]

        comparison.append({
            "track_id": track_id,
            "views": len(track),
            "frames": "|".join(
                sorted(
                    [node[0] for node in track],
                    key=frame_number,
                )
            ),
            "original_valid": original["valid"],
            "refined_valid": refined["valid"],
            "original_reason": original["reason"],
            "refined_reason": refined["reason"],
            "original_training_median_px": original.get(
                "training_median_px"
            ),
            "refined_training_median_px": refined.get(
                "training_median_px"
            ),
            "original_heldout_px": original.get(
                "heldout_px"
            ),
            "refined_heldout_px": refined.get(
                "heldout_px"
            ),
            "original_parallax_deg": original.get(
                "max_parallax_deg"
            ),
            "refined_parallax_deg": refined.get(
                "max_parallax_deg"
            ),
        })

    csv_path = (
        OUTPUT / "multiview_track_comparison.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison[0])
            if comparison
            else [
                "track_id",
                "views",
                "frames",
                "original_valid",
                "refined_valid",
                "original_reason",
                "refined_reason",
                "original_training_median_px",
                "refined_training_median_px",
                "original_heldout_px",
                "refined_heldout_px",
                "original_parallax_deg",
                "refined_parallax_deg",
            ],
        )

        writer.writeheader()
        writer.writerows(comparison)

    for mode in ("original", "refined"):
        save_ply(
            OUTPUT / f"{mode}_multiview_landmarks.ply",
            point_clouds[mode],
        )

    original_valid = [
        row for row in comparison
        if row["original_valid"]
    ]

    refined_valid = [
        row for row in comparison
        if row["refined_valid"]
    ]

    both_valid = [
        row for row in comparison
        if row["original_valid"]
        and row["refined_valid"]
    ]

    heldout_both = [
        row for row in both_valid
        if row["views"] >= 4
        and row["original_heldout_px"] is not None
        and row["refined_heldout_px"] is not None
    ]

    supporting_frames = {}

    for mode in ("original", "refined"):
        frame_set = set()

        for row in comparison:
            if row[f"{mode}_valid"]:
                frame_set.update(
                    row["frames"].split("|")
                )

        supporting_frames[mode] = len(frame_set)

    report = {
        "project": "Under-construction",
        "experiment": (
            "CONTROLLED_MULTIVIEW_RECONSTRUCTION_V3"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "parameters": vars(args),
        "supported_cameras": len(names),
        "excluded_v2_observations": excluded_count,
        "eligible_pairs": eligible_pairs,
        "accepted_pairs": accepted_pairs,
        "matched_correspondences": (
            matched_correspondences
        ),
        "accepted_feature_edges": len(edges),
        "conflicting_merges_prevented": union.conflicts,
        "feature_components": len(components),
        "candidate_multiview_tracks": len(tracks),
        "original": {
            "validated_landmarks": len(original_valid),
            "supporting_frames": (
                supporting_frames["original"]
            ),
            "median_training_error_px": finite_median(
                [
                    row["original_training_median_px"]
                    for row in original_valid
                ]
            ),
            "median_heldout_error_px": finite_median(
                [
                    row["original_heldout_px"]
                    for row in original_valid
                    if row["views"] >= 4
                ]
            ),
        },
        "refined": {
            "validated_landmarks": len(refined_valid),
            "supporting_frames": (
                supporting_frames["refined"]
            ),
            "median_training_error_px": finite_median(
                [
                    row["refined_training_median_px"]
                    for row in refined_valid
                ]
            ),
            "median_heldout_error_px": finite_median(
                [
                    row["refined_heldout_px"]
                    for row in refined_valid
                    if row["views"] >= 4
                ]
            ),
        },
        "paired_comparison": {
            "valid_under_both": len(both_valid),
            "original_only": (
                len(original_valid) - len(both_valid)
            ),
            "refined_only": (
                len(refined_valid) - len(both_valid)
            ),
            "shared_valid_training_median_px": {
                "original": finite_median(
                    [
                        row["original_training_median_px"]
                        for row in both_valid
                    ]
                ),
                "refined": finite_median(
                    [
                        row["refined_training_median_px"]
                        for row in both_valid
                    ]
                ),
            },
            "shared_valid_heldout_tracks": len(
                heldout_both
            ),
            "shared_valid_heldout_median_px": {
                "original": finite_median(
                    [
                        row["original_heldout_px"]
                        for row in heldout_both
                    ]
                ),
                "refined": finite_median(
                    [
                        row["refined_heldout_px"]
                        for row in heldout_both
                    ]
                ),
            },
        },
        "original_v2_baseline_landmarks": 425,
        "approved_for_application": False,
        "limitations": [
            "The same tracks are compared under both poses.",
            "Original poses alone select feature edges.",
            "Track membership is not independent of "
            "the original pose model.",
            "Three-view tracks have no held-out view.",
            "Held-out observations are not independent "
            "of SIFT track construction.",
            "Spatial exclusion does not establish "
            "scene-level independence.",
            "Accepted landmarks are unique feature "
            "components, not necessarily distinct "
            "physical scene points.",
            "No metric ground truth is available.",
            "Camera-to-IMU extrinsics remain unverified.",
            "CAD registration remains unverified.",
            "No construction progress is inferred.",
            "Original V2 remains unchanged.",
        ],
    }

    report_path = (
        OUTPUT / "multiview_v3_report.json"
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
    print(
        "Original validated landmarks:",
        len(original_valid),
    )
    print(
        "Refined validated landmarks:",
        len(refined_valid),
    )
    print("Valid under both:", len(both_valid))
    print(
        "Original only:",
        len(original_valid) - len(both_valid),
    )
    print(
        "Refined only:",
        len(refined_valid) - len(both_valid),
    )
    print(
        "Supporting frames:",
        supporting_frames,
    )
    print(
        "Shared held-out tracks:",
        len(heldout_both),
    )

    print("\nReport:", report_path)
    print("Track comparison:", csv_path)
    print("Approved for application: NO")
    print("Original V2 remains unchanged.")


if __name__ == "__main__":
    main()
