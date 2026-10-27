
"""
Under-construction — Step 89
CAD-independent structural plane candidate extraction.

Input:
    outputs/multiview_tracks/tracks.json

Outputs:
    outputs/structural_planes/plane_report.json
    outputs/structural_planes/plane_candidates.json
    outputs/structural_planes/plane_points.ply

Methods:
    - Deterministic RANSAC plane hypotheses
    - PCA plane refinement
    - Point-to-plane residual checks
    - Spatial extent and support-density checks
    - Multi-frame observation provenance
    - Normal-vector orientation classification
    - Explicit abstention on insufficient evidence

No AI/LLMs, manual annotations, or CAD-driven fitting.

Run:
    python tools/extract_structural_planes.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "outputs"
    / "multiview_tracks"
    / "tracks.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "structural_planes"
)

RANDOM_SEED = 89

RANSAC_ITERATIONS = 4000
DISTANCE_THRESHOLD_M = 0.12

MIN_PLANE_POINTS = 12
MIN_DISTINCT_FRAMES = 3
MIN_FRAME_OBSERVATIONS = 20

MIN_MAJOR_EXTENT_M = 1.5
MIN_MINOR_EXTENT_M = 0.5
MIN_AREA_M2 = 1.0

MAX_PLANES = 8

HORIZONTAL_NORMAL_Z = 0.90
VERTICAL_NORMAL_Z = 0.20

# Candidate support is not a verification claim.
MIN_SUPPORTED_PLANES_FOR_GATE = 1

# Reject fits where points lie almost entirely along
# a line, even when the plane residual is small.
MIN_SECOND_EIGENVALUE_RATIO = 0.015


def load_tracks():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Step 88 tracks not found: {INPUT_PATH}"
        )

    data = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    tracks = data.get("tracks", [])

    valid = []

    for track in tracks:
        point = np.asarray(
            track.get("point", []),
            dtype=np.float64,
        )

        observations = track.get(
            "observations", []
        )

        if point.shape != (3,):
            continue

        if not np.all(np.isfinite(point)):
            continue

        frames = {
            observation.get("frame")
            for observation in observations
            if observation.get("frame")
        }

        if len(frames) < 3:
            continue

        valid.append(
            {
                "track_id": int(track["track_id"]),
                "point": point,
                "frames": frames,
                "view_count": len(frames),
                "mean_error": float(
                    track.get(
                        "mean_reprojection_error_px",
                        float("inf"),
                    )
                ),
                "max_error": float(
                    track.get(
                        "max_reprojection_error_px",
                        float("inf"),
                    )
                ),
                "parallax": float(
                    track.get(
                        "max_parallax_deg",
                        0.0,
                    )
                ),
            }
        )

    return valid, data.get(
        "selected_convention"
    )


def fit_plane_pca(points):
    """
    Return a least-squares plane through the centroid.

    The smallest-eigenvalue eigenvector is the normal.
    """

    if len(points) < 3:
        return None

    centroid = points.mean(axis=0)
    centered = points - centroid

    covariance = (
        centered.T @ centered
    ) / len(points)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    normal = eigenvectors[:, 0]

    norm = np.linalg.norm(normal)

    if norm < 1e-12:
        return None

    normal = normal / norm

    # Stable normal sign for repeatable output.
    dominant_axis = int(
        np.argmax(np.abs(normal))
    )

    if normal[dominant_axis] < 0:
        normal = -normal

    offset = -float(
        np.dot(normal, centroid)
    )

    return {
        "centroid": centroid,
        "normal": normal,
        "offset": offset,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
    }


def plane_distances(points, normal, offset):
    return np.abs(
        points @ normal + offset
    )


def plane_extent(points, model):
    """
    Compute robust plane-local dimensions using
    projected PCA coordinates.
    """

    centroid = model["centroid"]

    # Eigenvectors 2 and 1 span the fitted plane.
    basis = model["eigenvectors"][:, [2, 1]]

    coordinates = (
        points - centroid
    ) @ basis

    # Percentile extents reduce sensitivity to a
    # small number of faraway points.
    low = np.percentile(
        coordinates,
        5,
        axis=0,
    )

    high = np.percentile(
        coordinates,
        95,
        axis=0,
    )

    dimensions = np.sort(
        high - low
    )[::-1]

    major = float(dimensions[0])
    minor = float(dimensions[1])

    return {
        "major_extent_m": major,
        "minor_extent_m": minor,
        "bounding_area_m2": major * minor,
        "basis": basis,
        "bounds_low": low,
        "bounds_high": high,
    }


def orientation_class(normal):
    z = abs(float(normal[2]))

    if z >= HORIZONTAL_NORMAL_Z:
        return "HORIZONTAL_CANDIDATE"

    if z <= VERTICAL_NORMAL_Z:
        return "VERTICAL_CANDIDATE"

    return "OBLIQUE_CANDIDATE"


def evaluate_support(indices, tracks):
    frame_counts = Counter()

    for index in indices:
        frame_counts.update(
            tracks[index]["frames"]
        )

    return {
        "distinct_frames": len(frame_counts),
        "total_frame_observations": int(
            sum(frame_counts.values())
        ),
        "frames_with_multiple_points": int(
            sum(
                count >= 2
                for count in frame_counts.values()
            )
        ),
        "frame_counts": dict(
            sorted(frame_counts.items())
        ),
    }


def evaluate_candidate(
    indices,
    points,
    tracks,
):
    selected_points = points[indices]

    model = fit_plane_pca(
        selected_points
    )

    if model is None:
        return None

    extent = plane_extent(
        selected_points,
        model,
    )

    residuals = plane_distances(
        selected_points,
        model["normal"],
        model["offset"],
    )

    support = evaluate_support(
        indices,
        tracks,
    )

    eigenvalues = model["eigenvalues"]

    second_ratio = float(
        eigenvalues[1]
        / max(
            eigenvalues[2],
            1e-12,
        )
    )

    median_error = float(
        np.median(residuals)
    )

    p95_error = float(
        np.percentile(
            residuals,
            95,
        )
    )

    sufficient_support = (
        len(indices) >= MIN_PLANE_POINTS
        and support["distinct_frames"]
        >= MIN_DISTINCT_FRAMES
        and support["total_frame_observations"]
        >= MIN_FRAME_OBSERVATIONS
    )

    sufficient_extent = (
        extent["major_extent_m"]
        >= MIN_MAJOR_EXTENT_M
        and extent["minor_extent_m"]
        >= MIN_MINOR_EXTENT_M
        and extent["bounding_area_m2"]
        >= MIN_AREA_M2
    )

    not_collinear = (
        second_ratio
        >= MIN_SECOND_EIGENVALUE_RATIO
    )

    residual_pass = (
        p95_error
        <= DISTANCE_THRESHOLD_M
    )

    accepted = (
        sufficient_support
        and sufficient_extent
        and not_collinear
        and residual_pass
    )

    return {
        "point_count": len(indices),
        "normal": model["normal"].tolist(),
        "offset": model["offset"],
        "centroid": model["centroid"].tolist(),
        "orientation": orientation_class(
            model["normal"]
        ),
        "median_plane_error_m": median_error,
        "p95_plane_error_m": p95_error,
        "major_extent_m": extent["major_extent_m"],
        "minor_extent_m": extent["minor_extent_m"],
        "bounding_area_m2": extent["bounding_area_m2"],
        "second_eigenvalue_ratio": second_ratio,
        "support": support,
        "checks": {
            "sufficient_support": sufficient_support,
            "sufficient_extent": sufficient_extent,
            "not_collinear": not_collinear,
            "residual_pass": residual_pass,
        },
        "candidate_gate_passed": accepted,
        "track_ids": [
            tracks[index]["track_id"]
            for index in indices
        ],
        "indices": indices.tolist(),
    }


def find_best_plane(
    points,
    tracks,
    available_indices,
    rng,
):
    """
    Search for the best spatially supported plane.

    RANSAC scoring uses inlier count first, followed
    by spatial extent, without CAD information.
    """

    if len(available_indices) < MIN_PLANE_POINTS:
        return None

    available_points = points[
        available_indices
    ]

    best = None
    best_score = None

    for _ in range(RANSAC_ITERATIONS):
        sample_local = rng.choice(
            len(available_indices),
            size=3,
            replace=False,
        )

        sample = available_points[
            sample_local
        ]

        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]

        normal = np.cross(v1, v2)

        magnitude = np.linalg.norm(
            normal
        )

        if magnitude < 1e-8:
            continue

        normal /= magnitude

        offset = -float(
            np.dot(
                normal,
                sample[0],
            )
        )

        distances = plane_distances(
            available_points,
            normal,
            offset,
        )

        local_inliers = np.flatnonzero(
            distances <= DISTANCE_THRESHOLD_M
        )

        if len(local_inliers) < MIN_PLANE_POINTS:
            continue

        global_indices = available_indices[
            local_inliers
        ]

        candidate = evaluate_candidate(
            global_indices,
            points,
            tracks,
        )

        if candidate is None:
            continue

        if not candidate[
            "candidate_gate_passed"
        ]:
            continue

        score = (
            candidate["point_count"],
            candidate["bounding_area_m2"],
            -candidate["p95_plane_error_m"],
        )

        if best_score is None or score > best_score:
            best_score = score
            best = candidate

    if best is None:
        return None

    # Refine inliers against the fitted PCA plane.
    normal = np.asarray(
        best["normal"],
        dtype=np.float64,
    )

    distances = plane_distances(
        available_points,
        normal,
        best["offset"],
    )

    refined_local = np.flatnonzero(
        distances <= DISTANCE_THRESHOLD_M
    )

    refined_indices = available_indices[
        refined_local
    ]

    refined = evaluate_candidate(
        refined_indices,
        points,
        tracks,
    )

    if (
        refined is not None
        and refined["candidate_gate_passed"]
    ):
        return refined

    return best


def write_colored_ply(path, points, labels):
    """
    Assign deterministic colors to detected planes.
    Unassigned points remain grey.
    """

    palette = [
        (230, 95, 85),
        (75, 160, 220),
        (100, 190, 120),
        (220, 175, 70),
        (170, 110, 210),
        (70, 195, 190),
        (220, 115, 160),
        (150, 160, 75),
    ]

    with path.open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(
            f"element vertex {len(points)}\n"
        )
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")

        for point, label in zip(points, labels):
            color = (
                (150, 150, 150)
                if label < 0
                else palette[
                    label % len(palette)
                ]
            )

            handle.write(
                f"{point[0]:.8f} "
                f"{point[1]:.8f} "
                f"{point[2]:.8f} "
                f"{color[0]} "
                f"{color[1]} "
                f"{color[2]}\n"
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tracks, convention = load_tracks()

    if len(tracks) < 3:
        raise RuntimeError(
            "Insufficient Step 88 landmarks."
        )

    points = np.asarray(
        [track["point"] for track in tracks],
        dtype=np.float64,
    )

    print("\n=== STEP 89: STRUCTURAL PLANES ===")
    print("Input tracks:", len(tracks))
    print("Pose convention:", convention)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    remaining = np.arange(
        len(points),
        dtype=np.int32,
    )

    labels = np.full(
        len(points),
        -1,
        dtype=np.int32,
    )

    planes = []

    for plane_index in range(MAX_PLANES):
        candidate = find_best_plane(
            points,
            tracks,
            remaining,
            rng,
        )

        if candidate is None:
            break

        indices = np.asarray(
            candidate.pop("indices"),
            dtype=np.int32,
        )

        candidate["plane_id"] = (
            f"plane_{plane_index + 1:02d}"
        )

        candidate["semantic_identity_verified"] = False

        planes.append(candidate)

        labels[indices] = plane_index

        remaining = remaining[
            ~np.isin(
                remaining,
                indices,
            )
        ]

        print(
            candidate["plane_id"],
            "| points:",
            candidate["point_count"],
            "| orientation:",
            candidate["orientation"],
            "| extent:",
            round(
                candidate["major_extent_m"],
                2,
            ),
            "x",
            round(
                candidate["minor_extent_m"],
                2,
            ),
            "m",
        )

    orientation_counts = dict(
        Counter(
            plane["orientation"]
            for plane in planes
        )
    )

    plane_gate_passed = (
        len(planes)
        >= MIN_SUPPORTED_PLANES_FOR_GATE
    )

    plane_data = {
        "sequence": "floor_2_2025-10-28_run_2",
        "coordinate_frame": "trajectory_map",
        "source": "step88_multiview_tracks",
        "pose_convention": convention,
        "planes": planes,
        "unassigned_track_ids": [
            tracks[index]["track_id"]
            for index in remaining
        ],
    }

    candidates_path = (
        OUTPUT_DIR
        / "plane_candidates.json"
    )

    candidates_path.write_text(
        json.dumps(
            plane_data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    ply_path = (
        OUTPUT_DIR
        / "plane_points.ply"
    )

    write_colored_ply(
        ply_path,
        points,
        labels,
    )

    report = {
        "sequence": "floor_2_2025-10-28_run_2",
        "method": (
            "deterministic_ransac_pca_plane_extraction"
        ),
        "input_track_count": len(tracks),
        "candidate_plane_count": len(planes),
        "orientation_counts": orientation_counts,
        "assigned_track_count": int(
            np.count_nonzero(labels >= 0)
        ),
        "unassigned_track_count": int(
            np.count_nonzero(labels < 0)
        ),
        "plane_candidate_gate_passed": (
            plane_gate_passed
        ),
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "ransac_iterations": RANSAC_ITERATIONS,
            "distance_threshold_m": (
                DISTANCE_THRESHOLD_M
            ),
            "min_plane_points": MIN_PLANE_POINTS,
            "min_distinct_frames": (
                MIN_DISTINCT_FRAMES
            ),
            "min_frame_observations": (
                MIN_FRAME_OBSERVATIONS
            ),
            "min_major_extent_m": (
                MIN_MAJOR_EXTENT_M
            ),
            "min_minor_extent_m": (
                MIN_MINOR_EXTENT_M
            ),
            "min_area_m2": MIN_AREA_M2,
            "max_planes": MAX_PLANES,
            "random_seed": RANDOM_SEED,
        },
        "plane_summaries": [
            {
                "plane_id": plane["plane_id"],
                "orientation": plane["orientation"],
                "point_count": plane["point_count"],
                "major_extent_m": plane[
                    "major_extent_m"
                ],
                "minor_extent_m": plane[
                    "minor_extent_m"
                ],
                "bounding_area_m2": plane[
                    "bounding_area_m2"
                ],
                "median_plane_error_m": plane[
                    "median_plane_error_m"
                ],
                "p95_plane_error_m": plane[
                    "p95_plane_error_m"
                ],
                "distinct_frames": plane[
                    "support"
                ]["distinct_frames"],
                "candidate_gate_passed": plane[
                    "candidate_gate_passed"
                ],
            }
            for plane in planes
        ],
        "limitations": [
            "Sparse point planes may represent furniture or temporary objects.",
            "RANSAC can fit accidental coplanar point subsets.",
            "Point-cloud extent does not prove continuous surface coverage.",
            "Horizontal and vertical classifications are geometric, not semantic.",
            "The trajectory coordinate frame's vertical axis is assumed to be Z.",
            "No independent structural identity or CAD correspondence is established.",
            "No construction progress measurement is verified.",
        ],
        "decision": (
            "PLANE_CANDIDATES_AVAILABLE_FOR_VALIDATION"
            if plane_gate_passed
            else "INSUFFICIENT_SUPPORTED_PLANAR_GEOMETRY"
        ),
    }

    report_path = (
        OUTPUT_DIR
        / "plane_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print("Plane candidates:", len(planes))
    print("Orientations:", orientation_counts)
    print(
        "Assigned tracks:",
        report["assigned_track_count"],
    )
    print(
        "Unassigned tracks:",
        report["unassigned_track_count"],
    )
    print(
        "Plane candidate gate:",
        "PASS" if plane_gate_passed else "FAIL",
    )
    print("CAD registration verified: NO")
    print("Report:", report_path)
    print("Candidates:", candidates_path)
    print("Colored point cloud:", ply_path)


if __name__ == "__main__":
    main()
