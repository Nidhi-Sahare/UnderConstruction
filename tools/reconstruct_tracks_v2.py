
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

OUTPUT = ROOT / "outputs" / "multiview_tracks_v2"

BASELINE = (
    ROOT / "outputs"
    / "multiview_tracks"
    / "multiview_tracks_report.json"
)


def quaternion_to_rotation(q):
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def load_poses():
    poses = {}

    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("pose_matched", "").strip().lower() != "true":
                continue

            try:
                C = np.array(
                    [float(row[k]) for k in ("tx", "ty", "tz")],
                    dtype=np.float64,
                )

                q = np.array(
                    [float(row[k]) for k in ("qx", "qy", "qz", "qw")],
                    dtype=np.float64,
                )

                norm = np.linalg.norm(q)

                if (
                    norm < 1e-10
                    or not np.isfinite(C).all()
                    or not np.isfinite(q).all()
                ):
                    continue

                poses[row["filename"]] = {
                    "C": C,
                    "R": quaternion_to_rotation(q / norm),
                }

            except (KeyError, ValueError, TypeError):
                continue

    return poses


def load_calibration():
    data = json.loads(
        CALIBRATION.read_text(encoding="utf-8")
    )["calibration"]

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


def relative_essential(a, b):
    R = b["R"].T @ a["R"]
    t = b["R"].T @ (a["C"] - b["C"])

    baseline = np.linalg.norm(t)

    if baseline < 1e-10:
        return None

    return skew(t / baseline) @ R


def sampson_px(p1, p2, E, focal):
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])

    Ex1 = (E @ x1.T).T
    Etx2 = (E.T @ x2.T).T

    numerator = np.sum(x2 * Ex1, axis=1) ** 2

    denominator = (
        Ex1[:, 0]**2 + Ex1[:, 1]**2
        + Etx2[:, 0]**2 + Etx2[:, 1]**2
    )

    return focal * np.sqrt(
        numerator / np.maximum(denominator, 1e-12)
    )


class ConflictAwareTracks:
    """
    Union-find with one observation per frame per component.

    Prevents merging tracks that contain different keypoints
    from the same photograph.
    """

    def __init__(self):
        self.parent = {}
        self.size = {}
        self.frames = {}
        self.conflicts = 0
        self.redundant = 0

    def add(self, node):
        if node in self.parent:
            return

        self.parent[node] = node
        self.size[node] = 1
        self.frames[node] = {node[0]: node[1]}

    def find(self, node):
        parent = self.parent

        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]

        return node

    def merge(self, a, b):
        self.add(a)
        self.add(b)

        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            self.redundant += 1
            return False

        fa = self.frames[ra]
        fb = self.frames[rb]

        if any(
            frame in fa and fa[frame] != keypoint
            for frame, keypoint in fb.items()
        ):
            self.conflicts += 1
            return False

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.frames[ra].update(self.frames[rb])

        del self.frames[rb]

        return True


def triangulate(observations, poses):
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    I = np.eye(3)

    for obs in observations:
        pose = poses[obs["frame"]]

        ray = np.array([
            obs["uv"][0],
            obs["uv"][1],
            1.0,
        ])

        ray /= np.linalg.norm(ray)

        direction = pose["R"] @ ray
        direction /= np.linalg.norm(direction)

        projector = I - np.outer(direction, direction)

        A += projector
        b += projector @ pose["C"]

    try:
        if np.linalg.cond(A) > 1e8:
            return None

        point = np.linalg.solve(A, b)

        if not np.isfinite(point).all():
            return None

        return point

    except np.linalg.LinAlgError:
        return None


def reprojection_error(point, obs, poses, max_depth):
    pose = poses[obs["frame"]]

    local = pose["R"].T @ (point - pose["C"])

    if not np.isfinite(local).all():
        return math.inf

    depth = float(local[2])

    if depth <= 0.05 or depth >= max_depth:
        return math.inf

    predicted = local[:2] / depth

    return float(
        np.linalg.norm(predicted - obs["uv"])
        * obs["focal"]
    )


def maximum_parallax(point, observations, poses):
    rays = []

    for obs in observations:
        ray = point - poses[obs["frame"]]["C"]
        norm = np.linalg.norm(ray)

        if norm < 1e-10:
            return 0.0

        rays.append(ray / norm)

    maximum = 0.0

    for i in range(len(rays)):
        for j in range(i + 1, len(rays)):
            cosine = np.clip(
                np.dot(rays[i], rays[j]),
                -1.0,
                1.0,
            )

            maximum = max(
                maximum,
                math.degrees(math.acos(float(cosine))),
            )

    return maximum


def validate_track(observations, poses, args):
    """
    Iteratively remove inconsistent observations.

    A track is accepted only when at least three observations
    remain, all fit the provisional pose model, and every
    leave-one-view-out prediction passes.
    """

    active = list(observations)
    removed = 0

    while len(active) >= args.min_views:
        point = triangulate(active, poses)

        if point is None:
            return None, "DEGENERATE", removed

        errors = np.array([
            reprojection_error(
                point, obs, poses, args.max_depth
            )
            for obs in active
        ])

        if np.isfinite(errors).all() and (
            np.max(errors) <= args.reprojection
        ):
            break

        if removed >= args.max_removed:
            return None, "REPROJECTION_FAILURE", removed

        # Try deleting each observation and retain the
        # subset with the best robust reprojection fit.
        candidates = []

        for index in range(len(active)):
            subset = active[:index] + active[index + 1:]

            if len(subset) < args.min_views:
                continue

            candidate_point = triangulate(subset, poses)

            if candidate_point is None:
                continue

            subset_errors = np.array([
                reprojection_error(
                    candidate_point,
                    obs,
                    poses,
                    args.max_depth,
                )
                for obs in subset
            ])

            if not np.isfinite(subset_errors).all():
                continue

            candidates.append((
                float(np.max(subset_errors)),
                float(np.median(subset_errors)),
                index,
            ))

        if not candidates:
            return None, "CANNOT_PRUNE", removed

        _, _, drop_index = min(candidates)

        active.pop(drop_index)
        removed += 1

    if len(active) < args.min_views:
        return None, "TOO_FEW_AFTER_PRUNING", removed

    point = triangulate(active, poses)

    if point is None:
        return None, "DEGENERATE", removed

    errors = np.array([
        reprojection_error(point, obs, poses, args.max_depth)
        for obs in active
    ])

    if (
        not np.isfinite(errors).all()
        or np.max(errors) > args.reprojection
    ):
        return None, "REPROJECTION_FAILURE", removed

    parallax = maximum_parallax(point, active, poses)

    if parallax < args.min_parallax:
        return None, "LOW_PARALLAX", removed

    holdout = []

    for index in range(len(active)):
        subset = active[:index] + active[index + 1:]

        candidate = triangulate(subset, poses)

        if candidate is None:
            return None, "HOLDOUT_DEGENERATE", removed

        error = reprojection_error(
            candidate,
            active[index],
            poses,
            args.max_depth,
        )

        holdout.append(error)

    holdout = np.asarray(holdout, dtype=np.float64)

    if (
        not np.isfinite(holdout).all()
        or np.max(holdout) > args.holdout
    ):
        return None, "HOLDOUT_FAILURE", removed

    colors = np.asarray(
        [obs["color"] for obs in active],
        dtype=np.float64,
    )

    return {
        "xyz": point.tolist(),
        "observations": [
            {
                "frame": obs["frame"],
                "frame_index": int(obs["frame_index"]),
                "normalized_uv": [
                    float(obs["uv"][0]),
                    float(obs["uv"][1]),
                ],
                "focal_px": float(obs["focal"]),
            }
            for obs in active
        ],
        "rgb": np.clip(
            np.rint(np.median(colors, axis=0)),
            0,
            255,
        ).astype(int).tolist(),
        "views": len(active),
        "frames": [obs["frame"] for obs in active],
        "removed_observations": removed,
        "median_reprojection_error_px": float(
            np.median(errors)
        ),
        "max_reprojection_error_px": float(
            np.max(errors)
        ),
        "median_holdout_error_px": float(
            np.median(holdout)
        ),
        "max_holdout_error_px": float(
            np.max(holdout)
        ),
        "maximum_parallax_deg": float(parallax),
    }, None, removed


def write_ply(path, landmarks):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(
            "comment Experimental pose-guided landmarks\n"
        )
        f.write(
            "comment Camera extrinsics and metric accuracy unverified\n"
        )
        f.write(f"element vertex {len(landmarks)}\n")

        for name in ("x", "y", "z"):
            f.write(f"property float {name}\n")

        for name in ("red", "green", "blue"):
            f.write(f"property uchar {name}\n")

        f.write("end_header\n")

        for landmark in landmarks:
            x, y, z = landmark["xyz"]
            r, g, b = landmark["rgb"]

            f.write(
                f"{x:.7f} {y:.7f} {z:.7f} "
                f"{r} {g} {b}\n"
            )


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)


def median_or_none(values):
    values = [
        float(v)
        for v in values
        if v is not None and np.isfinite(v)
    ]

    return float(np.median(values)) if values else None


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--features", type=int, default=3000)

    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-matches", type=int, default=25)
    parser.add_argument("--min-pair-inliers", type=int, default=15)

    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--epipolar", type=float, default=2.5)
    parser.add_argument("--reprojection", type=float, default=3.0)
    parser.add_argument("--holdout", type=float, default=3.0)

    parser.add_argument("--min-parallax", type=float, default=1.0)
    parser.add_argument("--max-depth", type=float, default=100.0)

    parser.add_argument("--min-baseline", type=float, default=0.08)
    parser.add_argument("--max-baseline", type=float, default=5.0)

    parser.add_argument("--max-removed", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if (
        args.max_gap < 1
        or args.width < 100
        or args.features < 100
        or args.min_views < 3
        or args.min_matches < 8
        or args.min_pair_inliers < 8
        or args.max_removed < 0
        or not 0 < args.ratio < 1
        or args.epipolar <= 0
        or args.reprojection <= 0
        or args.holdout <= 0
        or args.min_parallax <= 0
        or args.min_baseline <= 0
        or args.max_baseline <= args.min_baseline
        or args.max_depth <= 0
    ):
        parser.error("Invalid reconstruction parameters.")

    for path in (FRAMES, MANIFEST, CALIBRATION):
        if not path.exists():
            raise FileNotFoundError(path)

    cv2.setRNGSeed(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    paths = sorted(FRAMES.glob("*.jpg"))
    poses = load_poses()
    K, D = load_calibration()

    sift = cv2.SIFT_create(nfeatures=args.features)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    @lru_cache(maxsize=32)
    def features(index):
        image = cv2.imread(str(paths[index]))

        if image is None:
            return None

        h, w = image.shape[:2]
        scale = min(1.0, args.width / w)

        if scale < 1:
            image = cv2.resize(
                image,
                (
                    max(1, round(w * scale)),
                    max(1, round(h * scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = sift.detectAndCompute(
            gray, None
        )

        if descriptors is None or len(keypoints) < 2:
            return None

        Ks = K.copy()
        Ks[0, :] *= scale
        Ks[1, :] *= scale

        pixels = np.array(
            [kp.pt for kp in keypoints],
            dtype=np.float64,
        )

        uv = cv2.fisheye.undistortPoints(
            pixels.reshape(-1, 1, 2),
            Ks,
            D,
        ).reshape(-1, 2)

        return {
            "image": image,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "uv": uv,
            "focal": float((Ks[0, 0] + Ks[1, 1]) / 2),
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
            good = {}

            for pair in pairs:
                if len(pair) != 2:
                    continue

                first, second = pair

                if first.distance < args.ratio * second.distance:
                    good[first.queryIdx] = first

            return good

        fwd = ratio_filter(forward)
        rev = ratio_filter(backward)

        return [
            m for index, m in fwd.items()
            if (
                m.trainIdx in rev
                and rev[m.trainIdx].trainIdx == index
            )
        ]

    def observation(node):
        frame_index, keypoint_index = node
        data = features(frame_index)

        pixel = np.rint(
            data["keypoints"][keypoint_index].pt
        ).astype(int)

        image = data["image"]
        h, w = image.shape[:2]

        x = int(np.clip(pixel[0], 0, w - 1))
        y = int(np.clip(pixel[1], 0, h - 1))

        return {
            "frame": paths[frame_index].name,
            "frame_index": frame_index,
            "uv": data["uv"][keypoint_index],
            "focal": data["focal"],
            "color": image[y, x, ::-1].copy(),
        }

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

            baseline = float(
                np.linalg.norm(a["C"] - b["C"])
            )

            if args.min_baseline <= baseline <= args.max_baseline:
                candidates.append((i, j, gap, baseline))

    print("\nUNDER-CONSTRUCTION")
    print("Multi-view tracks V2")
    print("-" * 60)
    print("Photographs:", len(paths))
    print("Pose-matched frames:", len(poses))
    print("Eligible pairs:", len(candidates))
    print("Minimum views:", args.min_views)
    print("Camera extrinsics: UNVERIFIED")

    # Store edges first, then process stronger matches before
    # weaker ones when constructing the feature-track graph.
    edges = []
    pair_records = []
    pair_rejections = Counter()

    for number, (i, j, gap, baseline) in enumerate(
        candidates, start=1
    ):
        record = {
            "frame_a": paths[i].name,
            "frame_b": paths[j].name,
            "gap": gap,
            "baseline_pose_units": baseline,
            "mutual_matches": 0,
            "epipolar_inliers": 0,
            "status": "UNTESTED",
        }

        try:
            a = features(i)
            b = features(j)

            if a is None or b is None:
                record["status"] = "NO_FEATURES"
            else:
                matches = mutual_matches(a, b)
                record["mutual_matches"] = len(matches)

                if len(matches) < args.min_matches:
                    record["status"] = "INSUFFICIENT_MATCHES"

                else:
                    E = relative_essential(
                        poses[paths[i].name],
                        poses[paths[j].name],
                    )

                    if E is None:
                        record["status"] = "INVALID_POSE"

                    else:
                        idx1 = np.array(
                            [m.queryIdx for m in matches],
                            dtype=int,
                        )

                        idx2 = np.array(
                            [m.trainIdx for m in matches],
                            dtype=int,
                        )

                        p1 = a["uv"][idx1]
                        p2 = b["uv"][idx2]

                        focal = (
                            a["focal"] + b["focal"]
                        ) / 2

                        errors = sampson_px(
                            p1, p2, E, focal
                        )

                        valid = (
                            np.isfinite(p1).all(axis=1)
                            & np.isfinite(p2).all(axis=1)
                            & np.isfinite(errors)
                            & (errors <= args.epipolar)
                        )

                        count = int(valid.sum())
                        record["epipolar_inliers"] = count

                        if count < args.min_pair_inliers:
                            record["status"] = "POSE_DISAGREEMENT"

                        else:
                            record["status"] = "ACCEPTED"

                            for k in np.flatnonzero(valid):
                                match = matches[int(k)]

                                # Lower cost = stronger evidence.
                                # Match distance and epipolar residual
                                # both influence edge ordering.
                                cost = (
                                    float(errors[k])
                                    / args.epipolar
                                    + float(match.distance) / 300.0
                                )

                                edges.append((
                                    cost,
                                    i,
                                    int(idx1[k]),
                                    j,
                                    int(idx2[k]),
                                ))

        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            record["status"] = "ERROR"
            record["error"] = str(exc).splitlines()[0]

        pair_records.append(record)
        pair_rejections[record["status"]] += 1

        if number % 50 == 0 or number == len(candidates):
            print(
                f"Pairs {number}/{len(candidates)} | "
                f"accepted={pair_rejections['ACCEPTED']}"
            )

    print("\nBuilding conflict-aware feature tracks...")

    tracks = ConflictAwareTracks()

    edges.sort(key=lambda item: (
        item[0], item[1], item[3], item[2], item[4]
    ))

    for _, i, ki, j, kj in edges:
        tracks.merge((i, ki), (j, kj))

    components = defaultdict(list)

    for node in tracks.parent:
        components[tracks.find(node)].append(node)

    print("Feature components:", len(components))
    print("Prevented conflicting merges:", tracks.conflicts)

    landmarks = []
    track_rejections = Counter()
    length_distribution = Counter()

    removed_total = 0

    for nodes in components.values():
        if len(nodes) < args.min_views:
            track_rejections["INSUFFICIENT_VIEWS"] += 1
            continue

        nodes.sort()

        observations = [
            observation(node)
            for node in nodes
        ]

        result, reason, removed = validate_track(
            observations,
            poses,
            args,
        )

        if result is None:
            track_rejections[reason] += 1
            continue

        landmarks.append(result)
        length_distribution[result["views"]] += 1
        removed_total += removed

    # Exact-coordinate deduplication is conservative:
    # it does not merge distinct nearby surfaces.
    unique = {}
    duplicates = 0

    for item in landmarks:
        key = tuple(
            round(float(v), 7)
            for v in item["xyz"]
        )

        previous = unique.get(key)

        if previous is None:
            unique[key] = item
        else:
            duplicates += 1

            if (
                item["max_holdout_error_px"]
                < previous["max_holdout_error_px"]
            ):
                unique[key] = item

    landmarks = list(unique.values())

    supported_frames = sorted({
        frame
        for item in landmarks
        for frame in item["frames"]
    })

    baseline = {}

    if BASELINE.exists():
        baseline = json.loads(
            BASELINE.read_text(encoding="utf-8")
        )

    ply_path = OUTPUT / "validated_landmarks_v2.ply"
    landmarks_path = OUTPUT / "validated_landmarks_v2.json"
    pairs_path = OUTPUT / "pair_results_v2.csv"
    report_path = OUTPUT / "tracks_v2_report.json"

    write_ply(ply_path, landmarks)

    landmarks_path.write_text(
        json.dumps(landmarks, indent=2),
        encoding="utf-8",
    )
    poses_path = OUTPUT / "camera_poses_v2.json"

    pose_records = {
        filename: {
            "center": pose["C"].tolist(),
            "rotation_camera_to_world": pose["R"].tolist(),
        }
        for filename, pose in sorted(poses.items())
    }

    poses_path.write_text(
        json.dumps(
            {
                "status": "PROVISIONAL",
                "camera_extrinsics_verified": False,
                "poses": pose_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(pairs_path, pair_records)

    baseline_landmarks = baseline.get("landmarks_validated")
    baseline_frames = baseline.get(
        "frames_supporting_validated_landmarks"
    )
    baseline_holdout = baseline.get(
        "median_landmark_holdout_error_px"
    )

    current_holdout = median_or_none(
        item["median_holdout_error_px"]
        for item in landmarks
    )

    report = {
        "project": "Under-construction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "EXPERIMENTAL",
        "method": "CONFLICT_AWARE_MULTIVIEW_TRACKS_V2",
        "parameters": vars(args),
        "photographs": len(paths),
        "pose_matched_frames": len(poses),
        "eligible_pairs": len(candidates),
        "pair_status_distribution": dict(pair_rejections),
        "accepted_feature_edges": len(edges),
        "prevented_conflicting_merges": tracks.conflicts,
        "redundant_consistent_edges": tracks.redundant,
        "feature_components": len(components),
        "track_rejections": dict(track_rejections),
        "removed_inconsistent_observations": removed_total,
        "duplicate_coordinates_removed": duplicates,
        "validated_landmarks": len(landmarks),
        "frames_supporting_landmarks": len(supported_frames),
        "supported_frame_names": supported_frames,
        "track_length_distribution": dict(
            sorted(length_distribution.items())
        ),
        "median_reprojection_error_px": median_or_none(
            item["median_reprojection_error_px"]
            for item in landmarks
        ),
        "median_holdout_error_px": current_holdout,
        "median_maximum_parallax_deg": median_or_none(
            item["maximum_parallax_deg"]
            for item in landmarks
        ),
        "baseline_comparison": {
            "v1_landmarks": baseline_landmarks,
            "v2_landmarks": len(landmarks),
            "v1_supported_frames": baseline_frames,
            "v2_supported_frames": len(supported_frames),
            "v1_median_holdout_error_px": baseline_holdout,
            "v2_median_holdout_error_px": current_holdout,
        },
        "verification": {
            "camera_extrinsics_verified": False,
            "metric_accuracy_verified": False,
            "cad_registration_verified": False,
        },
        "limitations": [
            "Recorded poses are treated as provisional camera poses.",
            "Camera-to-IMU extrinsics are unavailable.",
            "Pairwise epipolar filtering uses the same pose model "
            "later used for triangulation.",
            "Greedy conflict-aware track merging is not globally optimal.",
            "Pruning can remove genuine observations when poses are inaccurate.",
            "Leave-one-view-out validation is not independent "
            "ground-truth validation.",
            "No camera-pose refinement or bundle adjustment.",
            "No verified metric scale or CAD alignment.",
            "No dense surfaces or construction progress inferred.",
        ],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nRESULTS")
    print("-" * 60)
    print("Accepted pairs:", pair_rejections["ACCEPTED"])
    print("Feature graph components:", len(components))
    print("Prevented conflicting merges:", tracks.conflicts)
    print("Validated landmarks:", len(landmarks))
    print("Frames supporting landmarks:", len(supported_frames))
    print("Removed inconsistent observations:", removed_total)
    print("Duplicate coordinates removed:", duplicates)
    print("Track rejections:", dict(track_rejections))

    print(
        "Median reprojection error:",
        report["median_reprojection_error_px"],
    )

    print(
        "Median held-out error:",
        current_holdout,
    )

    print("\nBASELINE COMPARISON")
    print("-" * 60)

    print(
        "V1 landmarks:",
        baseline_landmarks,
        "| V2:",
        len(landmarks),
    )

    print(
        "V1 supported frames:",
        baseline_frames,
        "| V2:",
        len(supported_frames),
    )

    print(
        "V1 median held-out error:",
        baseline_holdout,
        "| V2:",
        current_holdout,
    )

    print("\nPoint cloud:", ply_path)
    print("Landmarks:", landmarks_path)
    print("Pair diagnostics:", pairs_path)
    print("Report:", report_path)

    print(
        "\nWARNING: Reconstruction remains experimental. "
        "Camera extrinsics and metric accuracy are unverified."
    )


if __name__ == "__main__":
    main()
