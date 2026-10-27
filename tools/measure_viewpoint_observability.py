
"""
Under-construction — Step 102
Viewpoint diversity and 3D geometric observability.

Uses frozen reconstructed lines, camera poses, and fisheye
calibration. No image-feature extraction and no optimization.

Inputs:
    outputs/line_ambiguity/resolved_lines.json
    outputs/geometric_consistency/per_candidate_consistency.json
    _archive/hilti_transition_20261023_004237/data/
        hilti_frames/floor_2_2025-10-28_run_2/cam0/
        frame_manifest.csv

Outputs:
    outputs/viewpoint_observability/
        observability_report.json
        per_candidate_observability.json

Measures:
    1. Candidate visibility across the camera trajectory.
    2. Camera translation baseline and viewing-ray diversity.
    3. Image displacement under frozen 3D perturbations.
    4. Local 4-parameter projection Jacobian singular values.
    5. Whether diverse camera viewpoints improve conditioning.

Cautions:
    - Visibility is predicted geometrically, not confirmed
      by an observed physical line.
    - Jacobian conditioning is local and model-based.
    - Pixel measurements do not include feature uncertainty.
    - Observability is not physical identity verification.

Run:
    .\\.venv\\Scripts\\python.exe tools\\measure_viewpoint_observability.py
"""

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

LINES_PATH = (
    ROOT / "outputs"
    / "line_ambiguity"
    / "resolved_lines.json"
)

STEP101_PATH = (
    ROOT / "outputs"
    / "geometric_consistency"
    / "per_candidate_consistency.json"
)

CAM_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST_PATH = CAM_ROOT / "frame_manifest.csv"
FRAME_DIR = CAM_ROOT / "frames"

OUT = ROOT / "outputs" / "viewpoint_observability"

TARGET_IDS = (159, 2, 47, 161, 108)

K = np.array(
    [
        [465.3015482593691, 0.0, 730.0455886686005],
        [0.0, 465.32303798346413, 720.1427007671206],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

D = np.array(
    [
        0.025800718903376804,
        -0.010909240777406872,
        -0.0016899537986031076,
        0.00014766801645260894,
    ],
    dtype=np.float64,
).reshape(4, 1)

# Native resolution of the supplied camera sequence.
IMAGE_WIDTH = 1472
IMAGE_HEIGHT = 1440

MIN_DEPTH_M = 0.20
IMAGE_MARGIN_PX = 8
MIN_VISIBLE_SAMPLES = 12
SAMPLE_COUNT = 41

SHIFT_AMOUNTS_M = (0.03, 0.08, 0.15)
ROTATION_AMOUNTS_DEG = (2.0, 5.0)

# Finite-difference steps for the four local parameters:
# two perpendicular translations (meters), two rotations
# about the midpoint (radians).
TRANSLATION_DIFFERENCE_M = 0.005
ROTATION_DIFFERENCE_RAD = math.radians(0.5)

MIN_PROJECTED_SHIFT_PX = 2.0

# Camera viewpoints sampled across the full manifest.
MAX_POSES_PER_CANDIDATE = 90

# A viewpoint must show a substantial portion of the line.
MIN_VISIBLE_FRACTION = 0.70

# Only compare a perturbation where enough of the same
# 3D samples are visible in both projections.
MIN_COMMON_SAMPLES = 12

# Conditioning is reported, never used to verify identity.
RELATIVE_SINGULAR_THRESHOLD = 0.01

cv2.setNumThreads(1)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def safe_float(value):
    value = float(value)

    if not math.isfinite(value):
        raise ValueError("Non-finite numeric value")

    return value


def load_lines():
    raw = read_json(LINES_PATH)
    lines = {}

    for item in raw.get("selected_lines", []):
        try:
            candidate_id = int(item["candidate_id"])

            points = np.asarray(
                item["endpoints_map"],
                dtype=np.float64,
            )

            if (
                points.shape != (2, 3)
                or not np.all(np.isfinite(points))
            ):
                continue

            lines[candidate_id] = {
                "candidate_id": candidate_id,
                "group_id": item.get("group_id"),
                "points": points,
            }

        except (KeyError, TypeError, ValueError):
            continue

    return lines


def load_previous():
    raw = read_json(STEP101_PATH)

    return {
        int(item["candidate_id"]): item
        for item in raw.get("results", [])
    }


def load_poses():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    poses = []

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        for row in csv.DictReader(handle):
            try:
                position = np.array(
                    [
                        safe_float(row["tx"]),
                        safe_float(row["ty"]),
                        safe_float(row["tz"]),
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        safe_float(row["qx"]),
                        safe_float(row["qy"]),
                        safe_float(row["qz"]),
                        safe_float(row["qw"]),
                    ],
                    dtype=np.float64,
                )

                rotation = Rotation.from_quat(
                    quaternion
                ).as_matrix()

                filename = Path(
                    row["filename"]
                ).name

                # Keep only camera poses with corresponding
                # image files, but do not read the images.
                if not (FRAME_DIR / filename).is_file():
                    continue

                poses.append(
                    {
                        "frame": filename,
                        "position": position,
                        "rotation": rotation,
                    }
                )

            except (
                KeyError,
                ValueError,
                TypeError,
            ):
                continue

    if not poses:
        raise RuntimeError(
            "No valid camera poses with image files."
        )

    # Preserve manifest order; use evenly distributed poses.
    if len(poses) > MAX_POSES_PER_CANDIDATE:
        indices = np.linspace(
            0,
            len(poses) - 1,
            MAX_POSES_PER_CANDIDATE,
        )

        selected = sorted(
            set(
                int(round(x))
                for x in indices
            )
        )

        poses = [
            poses[index]
            for index in selected
        ]

    return poses


def normalize(vector):
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        raise ValueError("Degenerate vector")

    return vector / norm


def local_axes(points):
    direction = normalize(
        points[1] - points[0]
    )

    reference = min(
        np.eye(3),
        key=lambda axis: abs(
            float(axis @ direction)
        ),
    )

    u = normalize(
        np.cross(direction, reference)
    )

    v = normalize(
        np.cross(direction, u)
    )

    return u, v


def transform_line(points, parameters):
    """
    Four local DOF:
        translation along u (meters)
        translation along v (meters)
        rotation around u (radians)
        rotation around v (radians)

    The u/v axes are frozen from the original line.
    """
    u, v = local_axes(points)

    du, dv, ru, rv = parameters

    midpoint = points.mean(axis=0)

    rotation_vector = (
        ru * u + rv * v
    )

    rotation = Rotation.from_rotvec(
        rotation_vector
    ).as_matrix()

    rotated = (
        (points - midpoint)
        @ rotation.T
        + midpoint
    )

    return (
        rotated
        + du * u
        + dv * v
    )


def sample_line(points):
    t = np.linspace(
        0.0,
        1.0,
        SAMPLE_COUNT,
    )[:, None]

    return (
        points[0]
        + t * (points[1] - points[0])
    )


def project_samples(points, pose):
    """
    Keep fixed sample identities, including invalid samples.

    The projective transform preserves the frozen convention
    used in Steps 93-101.
    """
    world = sample_line(points)

    camera = (
        world - pose["position"]
    ) @ pose["rotation"]

    depth_ok = (
        np.isfinite(camera).all(axis=1)
        & (camera[:, 2] > MIN_DEPTH_M)
    )

    pixels = np.full(
        (SAMPLE_COUNT, 2),
        np.nan,
        dtype=np.float64,
    )

    valid_indices = np.flatnonzero(depth_ok)

    if len(valid_indices):
        projected, _ = cv2.fisheye.projectPoints(
            camera[valid_indices].reshape(
                -1, 1, 3
            ),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K,
            D,
        )

        pixels[valid_indices] = (
            projected.reshape(-1, 2)
        )

    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= IMAGE_MARGIN_PX)
        & (
            pixels[:, 0]
            < IMAGE_WIDTH - IMAGE_MARGIN_PX
        )
        & (pixels[:, 1] >= IMAGE_MARGIN_PX)
        & (
            pixels[:, 1]
            < IMAGE_HEIGHT - IMAGE_MARGIN_PX
        )
    )

    return pixels, valid


def projected_line_length(pixels, valid):
    pairs = (
        valid[:-1] & valid[1:]
    )

    if not np.any(pairs):
        return 0.0

    differences = np.diff(
        pixels,
        axis=0,
    )

    return float(
        np.linalg.norm(
            differences[pairs],
            axis=1,
        ).sum()
    )


def visible_projection(points, pose):
    pixels, valid = project_samples(
        points,
        pose,
    )

    visible_count = int(
        valid.sum()
    )

    fraction = (
        visible_count / SAMPLE_COUNT
    )

    length = projected_line_length(
        pixels,
        valid,
    )

    accepted = (
        visible_count >= MIN_VISIBLE_SAMPLES
        and fraction >= MIN_VISIBLE_FRACTION
        and length >= 20.0
    )

    return {
        "pixels": pixels,
        "valid": valid,
        "visible_count": visible_count,
        "visible_fraction": fraction,
        "length_px": length,
        "accepted": accepted,
    }


def angle_degrees(a, b):
    cosine = float(
        np.clip(
            normalize(a) @ normalize(b),
            -1.0,
            1.0,
        )
    )

    return float(
        np.degrees(
            np.arccos(cosine)
        )
    )


def camera_diversity(poses, midpoint):
    """
    Camera translation and viewing-ray angular separation.

    Ray diversity is measured toward the candidate midpoint,
    rather than using camera orientation alone.
    """
    baselines = []
    ray_angles = []
    orientation_angles = []

    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            a = poses[i]
            b = poses[j]

            baseline = float(
                np.linalg.norm(
                    a["position"] - b["position"]
                )
            )

            ray_a = midpoint - a["position"]
            ray_b = midpoint - b["position"]

            if (
                np.linalg.norm(ray_a) < 1e-8
                or np.linalg.norm(ray_b) < 1e-8
            ):
                continue

            ray_angle = angle_degrees(
                ray_a,
                ray_b,
            )

            # In the frozen convention, camera-to-map
            # orientation is rotation.T.
            forward_a = (
                a["rotation"].T
                @ np.array([0.0, 0.0, 1.0])
            )

            forward_b = (
                b["rotation"].T
                @ np.array([0.0, 0.0, 1.0])
            )

            orientation_angle = angle_degrees(
                forward_a,
                forward_b,
            )

            baselines.append(baseline)
            ray_angles.append(ray_angle)
            orientation_angles.append(
                orientation_angle
            )

    if not baselines:
        return {
            "pair_count": 0,
            "median_baseline_m": None,
            "max_baseline_m": None,
            "median_ray_angle_deg": None,
            "max_ray_angle_deg": None,
            "median_orientation_angle_deg": None,
        }

    return {
        "pair_count": len(baselines),
        "median_baseline_m": float(
            np.median(baselines)
        ),
        "max_baseline_m": float(
            np.max(baselines)
        ),
        "median_ray_angle_deg": float(
            np.median(ray_angles)
        ),
        "max_ray_angle_deg": float(
            np.max(ray_angles)
        ),
        "median_orientation_angle_deg": float(
            np.median(orientation_angles)
        ),
    }


def perturbation_specs():
    specs = []

    for axis in ("u", "v"):
        for amount in SHIFT_AMOUNTS_M:
            for sign in (-1, 1):
                value = sign * amount

                parameters = np.zeros(
                    4,
                    dtype=np.float64,
                )

                parameters[
                    0 if axis == "u" else 1
                ] = value

                specs.append(
                    {
                        "name": (
                            f"shift_{axis}_{value:+.3f}m"
                        ),
                        "family": "translation",
                        "parameters": parameters,
                    }
                )

    for axis in ("u", "v"):
        for amount in ROTATION_AMOUNTS_DEG:
            for sign in (-1, 1):
                value = sign * amount

                parameters = np.zeros(
                    4,
                    dtype=np.float64,
                )

                parameters[
                    2 if axis == "u" else 3
                ] = math.radians(value)

                specs.append(
                    {
                        "name": (
                            f"rotate_{axis}_{value:+.1f}deg"
                        ),
                        "family": "rotation",
                        "parameters": parameters,
                    }
                )

    return specs


def displacement(original, alternative):
    common = (
        original["valid"]
        & alternative["valid"]
    )

    count = int(common.sum())

    if count < MIN_COMMON_SAMPLES:
        return None

    distances = np.linalg.norm(
        original["pixels"][common]
        - alternative["pixels"][common],
        axis=1,
    )

    return {
        "common_samples": count,
        "median_displacement_px": float(
            np.median(distances)
        ),
        "max_displacement_px": float(
            np.max(distances)
        ),
        "rms_displacement_px": float(
            np.sqrt(
                np.mean(distances ** 2)
            )
        ),
    }


def measure_perturbations(
    points,
    visible_poses,
):
    results = []

    original_projections = [
        visible_projection(points, pose)
        for pose in visible_poses
    ]

    for spec in perturbation_specs():
        transformed = transform_line(
            points,
            spec["parameters"],
        )

        per_view = []

        for pose, original in zip(
            visible_poses,
            original_projections,
        ):
            alternative = visible_projection(
                transformed,
                pose,
            )

            measurement = displacement(
                original,
                alternative,
            )

            if measurement is None:
                continue

            per_view.append(
                {
                    "frame": pose["frame"],
                    **measurement,
                }
            )

        shifts = [
            row["median_displacement_px"]
            for row in per_view
        ]

        if shifts:
            median_shift = float(
                np.median(shifts)
            )

            fraction_resolvable = float(
                np.mean(
                    np.asarray(shifts)
                    >= MIN_PROJECTED_SHIFT_PX
                )
            )

        else:
            median_shift = None
            fraction_resolvable = None

        results.append(
            {
                "name": spec["name"],
                "family": spec["family"],
                "views_compared": len(per_view),
                "median_displacement_px": median_shift,
                "max_view_displacement_px": (
                    float(np.max(shifts))
                    if shifts else None
                ),
                "fraction_views_above_2px": (
                    fraction_resolvable
                ),
                "per_view": per_view,
            }
        )

    return results


def jacobian_for_view(points, pose):
    """
    Differentiate predicted pixels with respect to four
    local 3D line parameters.

    Only samples valid for every +/- finite difference
    are retained. The same samples are used across columns.

    Parameter columns are normalized to represent:
        5 cm shift u
        5 cm shift v
        5 degree rotation u
        5 degree rotation v

    This prevents arbitrary meter/radian unit scales
    from dominating the singular-value calculation.
    """
    steps = np.array(
        [
            TRANSLATION_DIFFERENCE_M,
            TRANSLATION_DIFFERENCE_M,
            ROTATION_DIFFERENCE_RAD,
            ROTATION_DIFFERENCE_RAD,
        ],
        dtype=np.float64,
    )

    physical_scales = np.array(
        [
            0.05,
            0.05,
            math.radians(5.0),
            math.radians(5.0),
        ],
        dtype=np.float64,
    )

    original = visible_projection(
        points,
        pose,
    )

    if not original["accepted"]:
        return None

    derivatives = []
    common = original["valid"].copy()

    for parameter_index in range(4):
        plus = np.zeros(4)
        minus = np.zeros(4)

        plus[parameter_index] = (
            steps[parameter_index]
        )

        minus[parameter_index] = (
            -steps[parameter_index]
        )

        projected_plus = visible_projection(
            transform_line(points, plus),
            pose,
        )

        projected_minus = visible_projection(
            transform_line(points, minus),
            pose,
        )

        common &= (
            projected_plus["valid"]
            & projected_minus["valid"]
        )

        derivative = (
            projected_plus["pixels"]
            - projected_minus["pixels"]
        ) / (
            2.0 * steps[parameter_index]
        )

        derivatives.append(
            derivative
            * physical_scales[parameter_index]
        )

    if int(common.sum()) < MIN_COMMON_SAMPLES:
        return None

    # Matrix shape: 2 * visible samples by 4 parameters.
    matrix = np.column_stack(
        [
            derivative[common].reshape(-1)
            for derivative in derivatives
        ]
    )

    if not np.all(np.isfinite(matrix)):
        return None

    return {
        "frame": pose["frame"],
        "sample_count": int(common.sum()),
        "matrix": matrix,
    }


def singular_diagnostics(matrix):
    if matrix is None or matrix.shape[0] < 4:
        return None

    singular_values = np.linalg.svd(
        matrix,
        full_matrices=False,
        compute_uv=False,
    )

    if len(singular_values) != 4:
        return None

    largest = float(
        singular_values[0]
    )

    smallest = float(
        singular_values[-1]
    )

    relative = (
        smallest / largest
        if largest > 1e-12
        else 0.0
    )

    rank = int(
        np.sum(
            singular_values
            > largest
            * RELATIVE_SINGULAR_THRESHOLD
        )
    ) if largest > 1e-12 else 0

    return {
        "singular_values": [
            float(x)
            for x in singular_values
        ],
        "relative_smallest_singular_value": (
            float(relative)
        ),
        "effective_rank": rank,
        "condition_number": (
            float(largest / smallest)
            if smallest > 1e-12
            else None
        ),
    }


def measure_conditioning(points, visible_poses):
    """
    Evaluate one view, a locally clustered set of views,
    and a spatially spread set of views.

    The Jacobian is derived from predictions alone;
    it does not measure whether the line is real.
    """
    jacobians = []

    for pose in visible_poses:
        result = jacobian_for_view(
            points,
            pose,
        )

        if result is not None:
            jacobians.append(result)

    if not jacobians:
        return {
            "usable_jacobian_views": 0,
            "single_view": None,
            "local_views": None,
            "diverse_views": None,
            "diverse_frames": [],
        }

    single = jacobians[0]["matrix"]

    local = jacobians[
        :min(3, len(jacobians))
    ]

    local_matrix = np.vstack(
        [x["matrix"] for x in local]
    )

    # Deterministic farthest-point sampling in camera
    # position space, restricted to usable Jacobian views.
    pose_by_name = {
        pose["frame"]: pose
        for pose in visible_poses
    }

    selected = [0]
    desired = min(5, len(jacobians))

    while len(selected) < desired:
        best_index = None
        best_distance = -1.0

        for index, row in enumerate(jacobians):
            if index in selected:
                continue

            position = pose_by_name[
                row["frame"]
            ]["position"]

            minimum_distance = min(
                float(
                    np.linalg.norm(
                        position
                        - pose_by_name[
                            jacobians[j]["frame"]
                        ]["position"]
                    )
                )
                for j in selected
            )

            if minimum_distance > best_distance:
                best_distance = minimum_distance
                best_index = index

        if best_index is None:
            break

        selected.append(best_index)

    diverse = [
        jacobians[index]
        for index in selected
    ]

    diverse_matrix = np.vstack(
        [x["matrix"] for x in diverse]
    )

    return {
        "usable_jacobian_views": len(
            jacobians
        ),
        "single_view": singular_diagnostics(
            single
        ),
        "local_views": singular_diagnostics(
            local_matrix
        ),
        "diverse_views": singular_diagnostics(
            diverse_matrix
        ),
        "local_frames": [
            x["frame"]
            for x in local
        ],
        "diverse_frames": [
            x["frame"]
            for x in diverse
        ],
    }


def evaluate_candidate(line, previous, poses):
    points = line["points"]

    visible = []
    visibility_rows = []

    for pose in poses:
        projected = visible_projection(
            points,
            pose,
        )

        visibility_rows.append(
            {
                "frame": pose["frame"],
                "visible_fraction": (
                    projected["visible_fraction"]
                ),
                "projected_length_px": (
                    projected["length_px"]
                ),
                "accepted": projected["accepted"],
            }
        )

        if projected["accepted"]:
            visible.append(pose)

    midpoint = points.mean(axis=0)

    diversity = camera_diversity(
        visible,
        midpoint,
    )

    perturbations = measure_perturbations(
        points,
        visible,
    )

    conditioning = measure_conditioning(
        points,
        visible,
    )

    comparable = [
        item
        for item in perturbations
        if item["median_displacement_px"]
        is not None
    ]

    below_threshold = [
        item
        for item in comparable
        if item["median_displacement_px"]
        < MIN_PROJECTED_SHIFT_PX
    ]

    if not visible:
        status = "NO_GEOMETRICALLY_VISIBLE_VIEWS"

    elif len(visible) < 3:
        status = "LIMITED_VIEWPOINT_COVERAGE"

    elif not comparable:
        status = "PERTURBATION_OBSERVABILITY_UNAVAILABLE"

    else:
        status = "MODEL_OBSERVABILITY_MEASURED"

    return {
        "candidate_id": line["candidate_id"],
        "group_id": line["group_id"],
        "step101_decision_unchanged": (
            previous.get("decision")
        ),
        "line_length_m": float(
            np.linalg.norm(
                points[1] - points[0]
            )
        ),
        "camera_poses_tested": len(poses),
        "geometrically_visible_views": len(
            visible
        ),
        "visible_frames": [
            pose["frame"]
            for pose in visible
        ],
        "visibility": visibility_rows,
        "camera_diversity": diversity,
        "perturbations": perturbations,
        "comparable_perturbation_count": len(
            comparable
        ),
        "perturbations_below_2px": [
            item["name"]
            for item in below_threshold
        ],
        "conditioning": conditioning,
        "status": status,
        "observed_line_identity_verified": False,
        "physical_line_verified": False,
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = load_lines()
    previous = load_previous()
    poses = load_poses()

    results = []

    for candidate_id in TARGET_IDS:
        if candidate_id not in lines:
            raise RuntimeError(
                "Frozen candidate missing: "
                f"{candidate_id}"
            )

        result = evaluate_candidate(
            lines[candidate_id],
            previous.get(candidate_id, {}),
            poses,
        )

        results.append(result)

        diversity = result[
            "camera_diversity"
        ]

        conditioning = result[
            "conditioning"
        ]["diverse_views"]

        rank = (
            conditioning["effective_rank"]
            if conditioning is not None
            else None
        )

        print(
            f"Candidate {candidate_id:03d}: "
            f"visible={result['geometrically_visible_views']}"
            f"/{result['camera_poses_tested']} "
            f"max_baseline_m="
            f"{diversity['max_baseline_m']} "
            f"max_ray_angle_deg="
            f"{diversity['max_ray_angle_deg']} "
            f"diverse_rank={rank} "
            f"=> {result['status']}",
            flush=True,
        )

    details = {
        "method": (
            "step102_viewpoint_diversity_"
            "and_geometric_observability"
        ),
        "results": results,
    }

    write_json(
        OUT / "per_candidate_observability.json",
        details,
    )

    report = {
        "method": details["method"],
        "candidate_ids": list(
            TARGET_IDS
        ),
        "candidate_count": len(results),
        "camera_poses_loaded": len(poses),
        "candidate_summaries": [
            {
                "candidate_id": item["candidate_id"],
                "step101_decision": item[
                    "step101_decision_unchanged"
                ],
                "geometrically_visible_views": item[
                    "geometrically_visible_views"
                ],
                "max_baseline_m": item[
                    "camera_diversity"
                ]["max_baseline_m"],
                "max_ray_angle_deg": item[
                    "camera_diversity"
                ]["max_ray_angle_deg"],
                "perturbations_below_2px": item[
                    "perturbations_below_2px"
                ],
                "diverse_view_effective_rank": (
                    item["conditioning"][
                        "diverse_views"
                    ]["effective_rank"]
                    if item["conditioning"][
                        "diverse_views"
                    ] is not None
                    else None
                ),
                "status": item["status"],
            }
            for item in results
        ],
        "parameters": {
            "shift_amounts_m": list(
                SHIFT_AMOUNTS_M
            ),
            "rotation_amounts_deg": list(
                ROTATION_AMOUNTS_DEG
            ),
            "minimum_projected_shift_px": (
                MIN_PROJECTED_SHIFT_PX
            ),
            "minimum_visible_fraction": (
                MIN_VISIBLE_FRACTION
            ),
            "maximum_camera_poses": (
                MAX_POSES_PER_CANDIDATE
            ),
            "relative_singular_threshold": (
                RELATIVE_SINGULAR_THRESHOLD
            ),
            "jacobian_parameter_scales": {
                "translation_m": 0.05,
                "rotation_deg": 5.0,
            },
        },
        "geometry_modified": False,
        "camera_poses_modified": False,
        "step101_decisions_modified": False,
        "model_based_observability_measured": True,
        "empirical_feature_uncertainty_measured": False,
        "physical_lines_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Geometric visibility is predicted from the reconstructed line and camera pose, not confirmed by image evidence.",
            "Camera poses and calibration are assumed correct.",
            "Fisheye projections use fixed corresponding 3D sample points.",
            "Occlusion and appearance changes are not modeled.",
            "The projected-displacement threshold is not calibrated against measured tracking uncertainty.",
            "The four-parameter Jacobian covers only perpendicular translation and midpoint rotation, not endpoint position along the line or changes in line length.",
            "Jacobian rank is relative to an arbitrary 1 percent threshold and chosen parameter scales.",
            "The local and diverse Jacobian sets can differ in the number of observations; singular-value magnitudes should not be compared without normalization.",
            "Even a well-conditioned projection model cannot prove that an observed image feature belongs to this physical 3D line.",
            "No independent recording or CAD correspondence is verified.",
        ],
        "decision": (
            "MODEL_BASED_VIEWPOINT_OBSERVABILITY_AVAILABLE"
        ),
    }

    write_json(
        OUT / "observability_report.json",
        report,
    )

    print("\n=== STEP 102 ===")
    print(
        "Candidates:",
        len(results)
    )
    print(
        "Camera poses:",
        len(poses)
    )
    print(
        "Physical lines verified: NO"
    )
    print(
        "CAD registration verified: NO"
    )
    print(
        "Construction progress verified: NO"
    )
    print(
        "Reports:",
        OUT
    )


if __name__ == "__main__":
    main()
