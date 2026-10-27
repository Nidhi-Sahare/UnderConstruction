
"""
Under-construction — Step 88
Automatic multi-view feature tracks and triangulation.

Requires:
    tools/reconstruct_walkthrough_geometry.py (Step 87)

Outputs:
    outputs/multiview_tracks/track_report.json
    outputs/multiview_tracks/tracks.json
    outputs/multiview_tracks/tracked_points.ply

No CAD registration, AI models, or manual annotations.

Run:
    python tools/build_multiview_feature_tracks.py
"""

import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STEP87 = ROOT / "tools" / "reconstruct_walkthrough_geometry.py"
OUT = ROOT / "outputs" / "multiview_tracks"

MAX_FRAMES = 90
MAX_PAIR_GAP = 4

MIN_MATCHES_FOR_PAIR = 8
MIN_TRACK_VIEWS = 3
MIN_BASELINE_M = 0.4
MAX_BASELINE_M = 8.0
MIN_PARALLAX_DEG = 1.5
MAX_REPROJECTION_PX = 3.0
MAX_POINT_RANGE_M = 60.0

# Candidate convention selection is exploratory only.
CONVENTIONS = ("camera_to_map", "map_to_camera")


def load_step87():
    if not STEP87.exists():
        raise FileNotFoundError(STEP87)

    spec = importlib.util.spec_from_file_location(
        "step87_reconstruction", STEP87
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.MAX_FRAMES = MAX_FRAMES
    return module


class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
        self.members = {}

    def add(self, key):
        if key not in self.parent:
            self.parent[key] = key
            self.rank[key] = 0
            self.members[key] = {key[0]: key[1]}

    def find(self, key):
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def join(self, a, b):
        self.add(a)
        self.add(b)

        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True

        # A track cannot contain two different keypoints
        # from the same frame.
        ma, mb = self.members[ra], self.members[rb]

        for frame_id in ma.keys() & mb.keys():
            if ma[frame_id] != mb[frame_id]:
                return False

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
            ma, mb = mb, ma

        self.parent[rb] = ra
        ma.update(mb)
        del self.members[rb]

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

        return True

    def tracks(self):
        return [
            sorted((int(f), int(k)) for f, k in members.items())
            for members in self.members.values()
        ]


def world_rotation(frame, convention):
    r = frame["rotation"]

    if convention == "camera_to_map":
        return r

    return r.T


def camera_projection(frame, convention):
    """
    Normalized pinhole projection after fisheye undistortion.
    P maps world/map coordinates to normalized camera rays.
    """
    r_wc = world_rotation(frame, convention)
    r_cw = r_wc.T
    c = frame["position"].reshape(3, 1)

    return np.column_stack((r_cw, -r_cw @ c))


def triangulate_track(observations, frames, features, convention, step87):
    """
    Multi-view homogeneous DLT triangulation.

    observations: [(frame_index, keypoint_index), ...]
    """

    rows = []

    for frame_id, keypoint_id in observations:
        normalized = features[frame_id]["rays"][keypoint_id]
        x = normalized[0] / normalized[2]
        y = normalized[1] / normalized[2]

        p = camera_projection(frames[frame_id], convention)

        rows.append(x * p[2] - p[0])
        rows.append(y * p[2] - p[1])

    a = np.asarray(rows, dtype=np.float64)

    try:
        _, _, vt = np.linalg.svd(a)
    except np.linalg.LinAlgError:
        return None

    homogeneous = vt[-1]

    if abs(homogeneous[3]) < 1e-10:
        return None

    point = homogeneous[:3] / homogeneous[3]

    if not np.all(np.isfinite(point)):
        return None

    errors = []
    viewing_rays = []
    centers = []

    for frame_id, keypoint_id in observations:
        frame = frames[frame_id]
        center = frame["position"]
        r_wc = world_rotation(frame, convention)

        camera_point = r_wc.T @ (point - center)

        if camera_point[2] <= 0:
            return None

        distance = float(np.linalg.norm(point - center))

        if not (
            0.2 <= distance <= MAX_POINT_RANGE_M
        ):
            return None

        projected, _ = cv2.fisheye.projectPoints(
            camera_point.reshape(1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            step87.K,
            step87.D,
        )

        projected = projected.reshape(2)
        observed = features[frame_id]["pixels"][keypoint_id]

        error = float(np.linalg.norm(projected - observed))

        if not np.isfinite(error):
            return None

        errors.append(error)
        viewing_rays.append((point - center) / distance)
        centers.append(center)

    viewing_rays = np.asarray(viewing_rays)
    centers = np.asarray(centers)

    largest_parallax = 0.0
    largest_baseline = 0.0

    for i in range(len(observations)):
        for j in range(i + 1, len(observations)):
            cosine = np.clip(
                np.dot(viewing_rays[i], viewing_rays[j]),
                -1.0,
                1.0,
            )

            parallax = float(
                np.degrees(np.arccos(cosine))
            )

            baseline = float(
                np.linalg.norm(centers[i] - centers[j])
            )

            largest_parallax = max(
                largest_parallax, parallax
            )
            largest_baseline = max(
                largest_baseline, baseline
            )

    if largest_parallax < MIN_PARALLAX_DEG:
        return None

    if largest_baseline < MIN_BASELINE_M:
        return None

    if max(errors) > MAX_REPROJECTION_PX:
        return None

    return {
        "point": point.tolist(),
        "view_count": len(observations),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "max_reprojection_error_px": float(max(errors)),
        "max_parallax_deg": largest_parallax,
        "max_baseline_m": largest_baseline,
        "observations": [
            {
                "frame": frames[f]["filename"],
                "frame_index": f,
                "keypoint_index": k,
                "pixel": features[f]["pixels"][k].tolist(),
                "reprojection_error_px": errors[i],
            }
            for i, (f, k) in enumerate(observations)
        ],
    }


def write_ply(path, tracks):
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(tracks)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")

        for track in tracks:
            x, y, z = track["point"]
            handle.write(f"{x:.8f} {y:.8f} {z:.8f}\n")


def evaluate_convention(
    tracks,
    frames,
    features,
    convention,
    step87,
):
    accepted = []

    for track_id, observations in enumerate(tracks):
        result = triangulate_track(
            observations,
            frames,
            features,
            convention,
            step87,
        )

        if result is None:
            continue

        result["track_id"] = track_id
        accepted.append(result)

    return accepted


def summarize(tracks):
    if not tracks:
        return {
            "accepted_tracks": 0,
            "median_reprojection_error_px": None,
            "median_parallax_deg": None,
            "median_track_views": None,
            "occupied_xy_cells_2m": 0,
        }

    points = np.array(
        [t["point"] for t in tracks],
        dtype=np.float64,
    )

    xy_cells = np.floor(points[:, :2] / 2.0).astype(int)

    return {
        "accepted_tracks": len(tracks),
        "median_reprojection_error_px": float(
            np.median([
                t["mean_reprojection_error_px"]
                for t in tracks
            ])
        ),
        "median_parallax_deg": float(
            np.median([
                t["max_parallax_deg"]
                for t in tracks
            ])
        ),
        "median_track_views": float(
            np.median([
                t["view_count"]
                for t in tracks
            ])
        ),
        "occupied_xy_cells_2m": int(
            len(np.unique(xy_cells, axis=0))
        ),
        "point_bounds": {
            "min": points.min(axis=0).tolist(),
            "max": points.max(axis=0).tolist(),
        },
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    step87 = load_step87()
    frames = step87.load_frames()

    detector, norm_type, detector_name = (
        step87.make_detector()
    )

    print("\n=== STEP 88: MULTI-VIEW FEATURE TRACKS ===")
    print("Frames:", len(frames))
    print("Detector:", detector_name)

    features = [
        step87.extract_features(frame, detector)
        for frame in frames
    ]

    union = UnionFind()
    pair_report = []

    for i in range(len(frames)):
        if features[i] is None:
            continue

        for gap in range(1, MAX_PAIR_GAP + 1):
            j = i + gap

            if j >= len(frames) or features[j] is None:
                continue

            baseline = float(
                np.linalg.norm(
                    frames[i]["position"] - frames[j]["position"]
                )
            )

            if not (
                MIN_BASELINE_M <= baseline <= MAX_BASELINE_M
            ):
                continue

            matches = step87.match_features(
                features[i],
                features[j],
                norm_type,
            )

            if len(matches) < MIN_MATCHES_FOR_PAIR:
                continue

            # Geometric epipolar consistency using the
            # known camera poses and undistorted rays.
            #
            # Test both conventions independently and
            # retain matches supported by either.
            #
            # The final triangulation tests the conventions
            # separately; this prefilter avoids discarding
            # one convention prematurely.
            valid_masks = []

            for convention in CONVENTIONS:
                r_a = world_rotation(frames[i], convention)
                r_b = world_rotation(frames[j], convention)

                c_a = frames[i]["position"]
                c_b = frames[j]["position"]

                relative_r = r_b.T @ r_a
                relative_t = r_b.T @ (c_a - c_b)

                skew = np.array(
                    [
                        [0, -relative_t[2], relative_t[1]],
                        [relative_t[2], 0, -relative_t[0]],
                        [-relative_t[1], relative_t[0], 0],
                    ],
                    dtype=np.float64,
                )

                essential = skew @ relative_r

                indices_a = np.array(
                    [a for a, _ in matches], dtype=int
                )
                indices_b = np.array(
                    [b for _, b in matches], dtype=int
                )

                rays_a = features[i]["rays"][indices_a]
                rays_b = features[j]["rays"][indices_b]

                residual = np.abs(
                    np.einsum(
                        "ni,ij,nj->n",
                        rays_b,
                        essential,
                        rays_a,
                    )
                )

                ex1 = (essential @ rays_a.T).T
                etx2 = (essential.T @ rays_b.T).T

                denominator = np.sqrt(
                    ex1[:, 0] ** 2
                    + ex1[:, 1] ** 2
                    + etx2[:, 0] ** 2
                    + etx2[:, 1] ** 2
                )

                normalized_error = residual / np.maximum(
                    denominator, 1e-12
                )

                # Roughly a 3-pixel normalized-image
                # epipolar tolerance.
                threshold = 3.0 / float(step87.K[0, 0])

                valid_masks.append(
                    normalized_error <= threshold
                )

            accepted_mask = (
                valid_masks[0] | valid_masks[1]
            )

            accepted_matches = [
                match
                for match, good in zip(matches, accepted_mask)
                if good
            ]

            rejected_conflicts = 0

            for a, b in accepted_matches:
                if not union.join((i, a), (j, b)):
                    rejected_conflicts += 1

            pair_report.append({
                "frame_a": frames[i]["filename"],
                "frame_b": frames[j]["filename"],
                "baseline_m": baseline,
                "descriptor_matches": len(matches),
                "epipolar_matches": len(accepted_matches),
                "track_merge_conflicts": rejected_conflicts,
            })

        if (i + 1) % 15 == 0:
            print(
                f"Processed {i + 1}/{len(frames)} frames",
                flush=True,
            )

    all_tracks = union.tracks()

    multiview_tracks = [
        track
        for track in all_tracks
        if len(track) >= MIN_TRACK_VIEWS
    ]

    print("Tracks formed:", len(all_tracks))
    print("Tracks with >=3 views:", len(multiview_tracks))

    results = {}

    for convention in CONVENTIONS:
        print(
            "Triangulating with convention:",
            convention,
            flush=True,
        )

        results[convention] = evaluate_convention(
            multiview_tracks,
            frames,
            features,
            convention,
            step87,
        )

    summaries = {
        convention: summarize(results[convention])
        for convention in CONVENTIONS
    }

    # Select only for exploratory reconstruction output.
    # A higher accepted-track count alone cannot prove
    # the pose convention is correct.
    selected = max(
        CONVENTIONS,
        key=lambda name: (
            summaries[name]["accepted_tracks"],
            -(
                summaries[name]["median_reprojection_error_px"]
                if summaries[name][
                    "median_reprojection_error_px"
                ] is not None
                else float("inf")
            ),
        ),
    )

    selected_tracks = results[selected]

    tracks_file = OUT / "tracks.json"
    ply_file = OUT / "tracked_points.ply"
    report_file = OUT / "track_report.json"

    tracks_file.write_text(
        json.dumps(
            {
                "selected_convention": selected,
                "tracks": selected_tracks,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    write_ply(ply_file, selected_tracks)

    # These gates are diagnostic, not registration proof.
    enough_tracks = len(selected_tracks) >= 30

    enough_coverage = (
        summaries[selected]["occupied_xy_cells_2m"] >= 8
    )

    report = {
        "method": "multi_view_feature_track_dlt",
        "sequence": step87.SEQUENCE,
        "detector": detector_name,
        "frames": len(frames),
        "evaluated_pairs": len(pair_report),
        "all_feature_tracks": len(all_tracks),
        "tracks_with_three_or_more_views": len(
            multiview_tracks
        ),
        "pose_convention_comparison": summaries,
        "exploratory_selected_convention": selected,
        "track_quality_gate_passed": (
            enough_tracks and enough_coverage
        ),
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "min_track_views": MIN_TRACK_VIEWS,
            "min_baseline_m": MIN_BASELINE_M,
            "max_baseline_m": MAX_BASELINE_M,
            "min_parallax_deg": MIN_PARALLAX_DEG,
            "max_reprojection_px": MAX_REPROJECTION_PX,
            "max_point_range_m": MAX_POINT_RANGE_M,
        },
        "pair_results": pair_report,
        "limitations": [
            "Descriptor chains can contain incorrect correspondences.",
            "Epipolar filtering is exploratory and accepts either pose convention.",
            "Camera pose uncertainty is not modeled.",
            "Accepted points may belong to clutter or moving objects.",
            "A successful track does not establish a CAD-wall correspondence.",
            "Track count alone cannot prove the camera pose convention.",
            "No CAD registration or construction progress is verified.",
        ],
    }

    report_file.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")

    for convention in CONVENTIONS:
        summary = summaries[convention]

        print(
            convention,
            "| accepted:", summary["accepted_tracks"],
            "| median error:",
            summary["median_reprojection_error_px"],
            "| XY cells:",
            summary["occupied_xy_cells_2m"],
        )

    print("\nExploratory convention:", selected)
    print(
        "Track quality gate:",
        "PASS" if enough_tracks and enough_coverage else "FAIL",
    )
    print("CAD registration verified: NO")
    print("Report:", report_file)
    print("Tracks:", tracks_file)
    print("Point cloud:", ply_file)


if __name__ == "__main__":
    main()
