
"""
Under-construction — Step 93
Pose-constrained multi-view 3D line reconstruction.

Inputs:
    outputs/line_evidence/line_observations.json
    Existing walkthrough images and camera poses

Outputs:
    outputs/multiview_lines/line_report.json
    outputs/multiview_lines/line_candidates.json

No CAD data is used.

Important:
    These are geometric line candidates, not
    verified walls or construction elements.

Run:
    python tools/reconstruct_multiview_lines.py
"""

import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "outputs" / "line_evidence"
    / "line_observations.json"
)

FRAME_DIR = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
    / "frames"
)

OUTPUT = ROOT / "outputs" / "multiview_lines"

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
], dtype=np.float64)

# Conservative exploratory settings.
MAX_FRAME_GAP = 6
MIN_BASELINE_M = 0.5
MAX_BASELINE_M = 8.0

MIN_PLANE_INTERSECTION_ANGLE_DEG = 3.0

MAX_LINE_REPROJECTION_ERROR_PX = 5.0
MIN_IMAGE_OVERLAP_FRACTION = 0.30

MIN_3D_LENGTH_M = 0.25
MAX_3D_LENGTH_M = 15.0

MIN_SUPPORTING_FRAMES = 3

# Held-out observations are never used to fit
# the initial pairwise 3D line.
MIN_HELDOUT_FRAMES = 1

# Appearance is an auxiliary check.
MIN_APPEARANCE_SIMILARITY = 0.25

MAX_PAIRS_PER_FRAME_PAIR = 400
MAX_CANDIDATES = 200

EPS = 1e-10


def unit(vector):
    v = np.asarray(vector, dtype=np.float64)
    n = np.linalg.norm(v)

    if n < EPS:
        return None

    return v / n


def load_observations():
    data = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    groups = defaultdict(list)

    for item in data["observations"]:
        normal = unit(
            item["interpretation_plane_normal_map"]
        )

        endpoints = np.asarray(
            item["endpoints_px"],
            dtype=np.float64,
        )

        position = np.asarray(
            item["camera_position_map"],
            dtype=np.float64,
        )

        if normal is None:
            continue

        if endpoints.shape != (2, 2):
            continue

        if not (
            np.all(np.isfinite(endpoints))
            and np.all(np.isfinite(position))
        ):
            continue

        item = dict(item)

        item["_normal"] = normal
        item["_endpoints"] = endpoints
        item["_position"] = position

        groups[item["frame_sample_index"]].append(
            item
        )

    return groups


def image_gray(frame_name, cache):
    if frame_name not in cache:
        path = FRAME_DIR / frame_name

        image = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(str(path))

        cache[frame_name] = image

    return cache[frame_name]


def sample_line_appearance(item, cache):
    """
    Extract an intensity profile along a
    detected segment.

    This is not a learned descriptor.
    """

    image = image_gray(item["frame"], cache)

    a, b = item["_endpoints"]

    samples = np.linspace(0.15, 0.85, 24)

    pixels = (
        a[None, :]
        + samples[:, None] * (b - a)[None, :]
    )

    h, w = image.shape

    x = np.clip(
        pixels[:, 0], 0, w - 1
    ).astype(np.float32)

    y = np.clip(
        pixels[:, 1], 0, h - 1
    ).astype(np.float32)

    values = cv2.remap(
        image,
        x.reshape(1, -1),
        y.reshape(1, -1),
        interpolation=cv2.INTER_LINEAR,
    ).reshape(-1).astype(np.float64)

    values -= np.mean(values)

    norm = np.linalg.norm(values)

    if norm < 1e-6:
        return None

    return values / norm


def appearance_similarity(a, b, cache):
    try:
        p = sample_line_appearance(a, cache)
        q = sample_line_appearance(b, cache)
    except FileNotFoundError:
        return None

    if p is None or q is None:
        return None

    # Endpoints have arbitrary orientation.
    return float(
        max(
            np.dot(p, q),
            np.dot(p, q[::-1]),
        )
    )


def undistort(pixels):
    p = np.asarray(
        pixels, dtype=np.float64
    ).reshape(-1, 1, 2)

    xy = cv2.fisheye.undistortPoints(
        p, K, D
    ).reshape(-1, 2)

    rays = np.column_stack([
        xy,
        np.ones(len(xy)),
    ])

    rays /= np.linalg.norm(
        rays, axis=1, keepdims=True
    )

    return rays


def closest_point_between_ray_and_line(
    ray_origin,
    ray_direction,
    line_point,
    line_direction,
):
    """
    Closest point on a 3D line to a
    camera ray, with ray parameter >= 0.

    Returns the 3D line point and ray-line
    separation, or None if degenerate.
    """

    d = unit(ray_direction)
    v = unit(line_direction)

    if d is None or v is None:
        return None

    w = ray_origin - line_point

    a = float(np.dot(d, d))
    b = float(np.dot(d, v))
    c = float(np.dot(v, v))

    rhs1 = -float(np.dot(d, w))
    rhs2 = float(np.dot(v, w))

    matrix = np.array([
        [a, -b],
        [-b, c],
    ])

    determinant = np.linalg.det(matrix)

    if abs(determinant) < 1e-6:
        return None

    try:
        ray_t, line_t = np.linalg.solve(
            matrix,
            np.array([rhs1, rhs2]),
        )
    except np.linalg.LinAlgError:
        return None

    if ray_t <= 0:
        return None

    ray_point = ray_origin + ray_t * d
    line_result = line_point + line_t * v

    separation = float(
        np.linalg.norm(ray_point - line_result)
    )

    return line_result, separation


def intersect_interpretation_planes(a, b):
    """
    Each interpretation plane satisfies:

        normal . (X - camera_position) = 0

    Their intersection defines an infinite
    3D line if normals are nonparallel.
    """

    n1 = a["_normal"]
    n2 = b["_normal"]

    direction = unit(np.cross(n1, n2))

    if direction is None:
        return None

    angle = math.degrees(
        math.asin(
            np.clip(
                np.linalg.norm(
                    np.cross(n1, n2)
                ),
                0, 1,
            )
        )
    )

    if angle < MIN_PLANE_INTERSECTION_ANGLE_DEG:
        return None

    A = np.vstack([
        n1,
        n2,
        direction,
    ])

    rhs = np.array([
        np.dot(n1, a["_position"]),
        np.dot(n2, b["_position"]),
        0.0,
    ])

    try:
        point = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(point)):
        return None

    return point, direction, angle


def recover_segment_endpoints(
    line_point,
    line_direction,
    observation,
    camera_rotation,
):
    """
    Intersect endpoint viewing rays with the
    candidate 3D line in a closest-point sense.

    Endpoint reconstruction is exploratory;
    it is rejected if ray-line separation
    is excessive.
    """

    rays_camera = undistort(
        observation["_endpoints"]
    )

    rays_map = (
        camera_rotation @ rays_camera.T
    ).T

    points = []
    separations = []

    for ray in rays_map:
        result = (
            closest_point_between_ray_and_line(
                observation["_position"],
                ray,
                line_point,
                line_direction,
            )
        )

        if result is None:
            return None

        p, error = result

        points.append(p)
        separations.append(error)

    return (
        np.asarray(points),
        np.asarray(separations),
    )


def project_points(points, position, rotation):
    """
    T_map_cam0: camera-to-map rotation.
    """

    camera = (
        np.asarray(points) - position
    ) @ rotation

    if np.any(camera[:, 2] <= 0.1):
        return None

    object_points = camera.reshape(
        -1, 1, 3
    ).astype(np.float64)

    projected, _ = cv2.fisheye.projectPoints(
        object_points,
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        D,
    )

    return projected.reshape(-1, 2)


def point_to_line_distance(p, a, b):
    """
    Perpendicular distance from a 2D point
    to an infinite 2D line through a and b.

    Compatible with NumPy 2.x.
    """
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    v = b - a
    w = p - a

    length = float(np.linalg.norm(v))

    if length < EPS:
        return float("inf")

    cross_2d = (
        v[0] * w[1]
        - v[1] * w[0]
    )

    return float(abs(cross_2d) / length)

def compare_projected_segment(
    projected,
    observed,
):
    """
    Compare infinite-line agreement AND
    finite segment overlap.
    """

    p0, p1 = projected
    q0, q1 = observed

    projected_length = np.linalg.norm(
        p1 - p0
    )

    observed_length = np.linalg.norm(
        q1 - q0
    )

    if (
        projected_length < 5
        or observed_length < 5
    ):
        return None

    error = 0.5 * (
        point_to_line_distance(q0, p0, p1)
        + point_to_line_distance(q1, p0, p1)
    )

    axis = (
        p1 - p0
    ) / projected_length

    projected_range = sorted([
        0.0,
        projected_length,
    ])

    observed_range = sorted([
        float(np.dot(q0 - p0, axis)),
        float(np.dot(q1 - p0, axis)),
    ])

    overlap = max(
        0.0,
        min(
            projected_range[1],
            observed_range[1],
        )
        - max(
            projected_range[0],
            observed_range[0],
        ),
    )

    overlap_fraction = overlap / max(
        min(
            projected_length,
            observed_length,
        ),
        EPS,
    )

    return {
        "error_px": float(error),
        "overlap_fraction": float(
            overlap_fraction
        ),
    }


def load_rotations():
    """
    Reuse the exact Step 92 manifest convention.
    """

    import csv
    from scipy.spatial.transform import Rotation

    manifest = FRAME_DIR.parent / "frame_manifest.csv"

    rotations = {}

    with manifest.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.DictReader(f):
            try:
                quaternion = [
                    float(row[k])
                    for k in (
                        "qx", "qy", "qz", "qw"
                    )
                ]

                rotations[row["filename"]] = (
                    Rotation.from_quat(
                        quaternion
                    ).as_matrix()
                )
            except (
                KeyError,
                ValueError,
            ):
                continue

    return rotations


def candidate_from_pair(a, b, rotations):
    frame_a = a["frame"]
    frame_b = b["frame"]

    if (
        frame_a not in rotations
        or frame_b not in rotations
    ):
        return None

    baseline = float(
        np.linalg.norm(
            a["_position"] - b["_position"]
        )
    )

    if not (
        MIN_BASELINE_M
        <= baseline
        <= MAX_BASELINE_M
    ):
        return None

    intersection = (
        intersect_interpretation_planes(a, b)
    )

    if intersection is None:
        return None

    point, direction, angle = intersection

    reconstructed = []

    for observation in (a, b):
        result = recover_segment_endpoints(
            point,
            direction,
            observation,
            rotations[observation["frame"]],
        )

        if result is None:
            return None

        endpoints, separations = result

        if np.max(separations) > 0.20:
            return None

        reconstructed.append(endpoints)

    # Use the common finite 3D extent, not
    # arbitrary extrapolation of infinite lines.
    intervals = []

    for endpoints in reconstructed:
        t = (endpoints - point) @ direction

        intervals.append(
            sorted(t.tolist())
        )

    lower = max(
        intervals[0][0],
        intervals[1][0],
    )

    upper = min(
        intervals[0][1],
        intervals[1][1],
    )

    length = upper - lower

    if not (
        MIN_3D_LENGTH_M
        <= length
        <= MAX_3D_LENGTH_M
    ):
        return None

    endpoints_3d = np.array([
        point + lower * direction,
        point + upper * direction,
    ])

    # Check the fitted observations as well.
    for observation in (a, b):
        projected = project_points(
            endpoints_3d,
            observation["_position"],
            rotations[observation["frame"]],
        )

        if projected is None:
            return None

        comparison = compare_projected_segment(
            projected,
            observation["_endpoints"],
        )

        if comparison is None:
            return None

        if (
            comparison["error_px"]
            > MAX_LINE_REPROJECTION_ERROR_PX
            or comparison["overlap_fraction"]
            < MIN_IMAGE_OVERLAP_FRACTION
        ):
            return None

    return {
        "point": point,
        "direction": direction,
        "endpoints": endpoints_3d,
        "length_m": float(length),
        "baseline_m": baseline,
        "intersection_angle_deg": angle,
    }


def evaluate_heldout(
    candidate,
    source_a,
    source_b,
    groups,
    rotations,
    cache,
):
    """
    Independently test the fitted 3D segment
    against observations from other frames.

    Note:
    Observations are not guaranteed to be
    independent physical correspondences.
    """

    matches = []

    fitted_frames = {
        source_a["frame"],
        source_b["frame"],
    }

    source_profiles = [
        sample_line_appearance(
            source_a, cache
        ),
        sample_line_appearance(
            source_b, cache
        ),
    ]

    if any(p is None for p in source_profiles):
        return []

    # Compare only nearby sampled frames.
    indices = sorted(groups)

    centre = 0.5 * (
        source_a["frame_sample_index"]
        + source_b["frame_sample_index"]
    )

    for index in indices:
        if abs(index - centre) > MAX_FRAME_GAP + 2:
            continue

        observations = groups[index]

        if not observations:
            continue

        frame_name = observations[0]["frame"]

        if frame_name in fitted_frames:
            continue

        if frame_name not in rotations:
            continue

        position = observations[0]["_position"]

        projected = project_points(
            candidate["endpoints"],
            position,
            rotations[frame_name],
        )

        if projected is None:
            continue

        best = None

        for observation in observations:
            comparison = compare_projected_segment(
                projected,
                observation["_endpoints"],
            )

            if comparison is None:
                continue

            if (
                comparison["error_px"]
                > MAX_LINE_REPROJECTION_ERROR_PX
            ):
                continue

            if (
                comparison["overlap_fraction"]
                < MIN_IMAGE_OVERLAP_FRACTION
            ):
                continue

            appearance = (
                sample_line_appearance(
                    observation, cache
                )
            )

            if appearance is None:
                continue

            similarity = max(
                max(
                    float(np.dot(p, appearance)),
                    float(np.dot(
                        p, appearance[::-1]
                    )),
                )
                for p in source_profiles
            )

            if (
                similarity
                < MIN_APPEARANCE_SIMILARITY
            ):
                continue

            match = {
                "frame": frame_name,
                "observation_id": (
                    observation["observation_id"]
                ),
                "reprojection_error_px": (
                    comparison["error_px"]
                ),
                "overlap_fraction": (
                    comparison["overlap_fraction"]
                ),
                "appearance_similarity": similarity,
            }

            if (
                best is None
                or match["reprojection_error_px"]
                < best["reprojection_error_px"]
            ):
                best = match

        if best is not None:
            matches.append(best)

    return matches


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    groups = load_observations()
    rotations = load_rotations()

    frame_indices = sorted(groups)

    image_cache = {}

    accepted = []
    tested_pairs = 0
    geometric_pairs = 0

    print(
        "\n=== STEP 93: MULTIVIEW LINE RECONSTRUCTION ==="
    )

    for position, index_a in enumerate(frame_indices):
        obs_a = groups[index_a]

        for index_b in frame_indices[position + 1:]:
            if index_b - index_a > MAX_FRAME_GAP:
                break

            obs_b = groups[index_b]

            # Preselect pairs by image appearance.
            # This reduces arbitrary geometric matches.
            proposals = []

            for a in obs_a:
                for b in obs_b:
                    # A similar interpretation-plane
                    # normal is not enough to match
                    # a physical line, but strongly
                    # dissimilar appearance is useful
                    # negative evidence.
                    similarity = appearance_similarity(
                        a, b, image_cache
                    )

                    if (
                        similarity is None
                        or similarity
                        < MIN_APPEARANCE_SIMILARITY
                    ):
                        continue

                    proposals.append(
                        (similarity, a, b)
                    )

            proposals.sort(
                key=lambda x: x[0],
                reverse=True,
            )

            for similarity, a, b in (
                proposals[:MAX_PAIRS_PER_FRAME_PAIR]
            ):
                tested_pairs += 1

                candidate = candidate_from_pair(
                    a, b, rotations
                )

                if candidate is None:
                    continue

                geometric_pairs += 1

                heldout = evaluate_heldout(
                    candidate,
                    a,
                    b,
                    groups,
                    rotations,
                    image_cache,
                )

                if (
                    len(heldout)
                    < MIN_HELDOUT_FRAMES
                ):
                    continue

                support_count = (
                    2 + len(heldout)
                )

                if (
                    support_count
                    < MIN_SUPPORTING_FRAMES
                ):
                    continue

                accepted.append({
                    "source_observations": [
                        a["observation_id"],
                        b["observation_id"],
                    ],
                    "source_frames": [
                        a["frame"],
                        b["frame"],
                    ],
                    "endpoints_map": (
                        candidate["endpoints"].tolist()
                    ),
                    "direction_map": (
                        candidate["direction"].tolist()
                    ),
                    "length_m": (
                        candidate["length_m"]
                    ),
                    "baseline_m": (
                        candidate["baseline_m"]
                    ),
                    "intersection_angle_deg": (
                        candidate[
                            "intersection_angle_deg"
                        ]
                    ),
                    "source_appearance_similarity": (
                        similarity
                    ),
                    "supporting_frame_count": (
                        support_count
                    ),
                    "heldout_matches": heldout,
                    "physical_line_verified": False,
                    "structural_identity_verified": False,
                    "cad_registration_verified": False,
                })

        print(
            f"[{position + 1}/{len(frame_indices)}] "
            f"tested={tested_pairs} "
            f"geometric={geometric_pairs} "
            f"supported={len(accepted)}"
        )

    accepted.sort(
        key=lambda x: (
            x["supporting_frame_count"],
            x["intersection_angle_deg"],
            x["length_m"],
        ),
        reverse=True,
    )

    # Keep a bounded candidate output.
    # These may contain duplicates from overlapping
    # source pairs; they are not distinct structures.
    accepted = accepted[:MAX_CANDIDATES]

    candidates_path = (
        OUTPUT / "line_candidates.json"
    )

    candidates_path.write_text(
        json.dumps(
            {
                "sequence": (
                    "floor_2_2025-10-28_run_2"
                ),
                "method": (
                    "two_view_interpretation_plane_"
                    "intersection_with_heldout_support"
                ),
                "candidates": accepted,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = {
        "sequence": (
            "floor_2_2025-10-28_run_2"
        ),
        "method": (
            "pose_constrained_multiview_line_candidates"
        ),
        "frames_available": len(groups),
        "line_observations": sum(
            len(v) for v in groups.values()
        ),
        "appearance_preselected_pairs_tested": (
            tested_pairs
        ),
        "geometrically_valid_pair_hypotheses": (
            geometric_pairs
        ),
        "retained_supported_candidates": (
            len(accepted)
        ),
        "max_candidate_supporting_frames": (
            max(
                (
                    x["supporting_frame_count"]
                    for x in accepted
                ),
                default=0,
            )
        ),
        "independent_physical_lines_verified": False,
        "distinct_structural_directions_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "The candidate list may contain duplicate physical edges.",
            "Intensity-profile similarity is not a unique line descriptor.",
            "Held-out frames may share appearance and pose correlations.",
            "The same incorrect hypothesis may attract unrelated parallel edges.",
            "Camera poses are treated as given and are not independently calibrated here.",
            "Reconstructed endpoints depend on finite image-segment overlap.",
            "No uniqueness test against competing 3D lines has been performed.",
            "No candidate is classified as a wall.",
            "No CAD information is used.",
        ],
        "decision": (
            "MULTIVIEW_LINE_CANDIDATES_AVAILABLE"
            if accepted
            else "NO_SUPPORTED_MULTIVIEW_LINES"
        ),
    }

    report_path = OUTPUT / "line_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print(
        "Tested appearance-selected pairs:",
        tested_pairs,
    )
    print(
        "Geometrically valid hypotheses:",
        geometric_pairs,
    )
    print(
        "Retained candidates:",
        len(accepted),
    )
    print("Physical lines verified: NO")
    print("CAD registration verified: NO")
    print("Report:", report_path)
    print("Candidates:", candidates_path)


if __name__ == "__main__":
    main()
