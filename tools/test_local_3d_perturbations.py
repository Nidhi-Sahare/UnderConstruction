
"""
Under-construction — Step 100

Local 3D perturbation and cross-view prediction diagnostic.

Tests whether a frozen reconstructed 3D line fits tracked image
features better than nearby shifted or rotated 3D alternatives.

Candidate selection:
    159, 2, 47, 161, 108

Evidence:
    - Global LSD image segments
    - Shi-Tomasi features selected on associated LSD segments
    - Forward/backward Lucas-Kanade tracking
    - Two future frames for cross-view prediction

The source frame is used for feature association only.
Hypothesis scoring uses future-frame observations.

IMPORTANT:
    Source feature association remains conditioned on the original
    line projection. This is NOT a fully independent discovery test.
    Overlapping windows are NOT statistically independent.
    Candidate geometry, camera poses, and calibration remain frozen.

Outputs:
    outputs/line_perturbation_test/perturbation_report.json
    outputs/line_perturbation_test/per_candidate_perturbations.json

Run:
    .\\.venv\\Scripts\\python.exe tools\\test_local_3d_perturbations.py
"""

import csv
import json
from collections import Counter
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

STEP99_PATH = (
    ROOT / "outputs"
    / "line_identity_challenge"
    / "per_candidate_challenge.json"
)

CAMERA_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST_PATH = CAMERA_ROOT / "frame_manifest.csv"
FRAME_DIR = CAMERA_ROOT / "frames"

OUTPUT = ROOT / "outputs" / "line_perturbation_test"

TARGET_IDS = (159, 2, 47, 161, 108)

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

cv2.setNumThreads(1)
cv2.setRNGSeed(100)

MAX_WINDOWS_PER_CANDIDATE = 12
MAX_SEGMENT_DISTANCE_PX = 10.0
MAX_SEGMENT_ANGLE_DEG = 12.0
MIN_SEGMENT_LENGTH_PX = 25.0

MIN_FEATURES = 5
MIN_SURVIVORS = 5
MIN_TRACK_FRACTION = 0.25
MAX_FB_ERROR_PX = 1.5

MAX_ORIGINAL_ERROR_PX = 8.0
MIN_WIN_MARGIN_PX = 2.0

SHIFT_AMOUNTS_M = (0.03, 0.08, 0.15)
ROTATION_AMOUNTS_DEG = (2.0, 5.0)

SAMPLES_PER_LINE = 80
MIN_DEPTH_M = 0.20
IMAGE_MARGIN_PX = 8
MIN_VISIBLE_LENGTH_PX = 20.0

LK_PARAMS = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def load_lines():
    raw = load_json(LINES_PATH)

    lines = {}

    for item in raw.get("selected_lines", []):
        points = np.asarray(
            item.get("endpoints_map", []),
            dtype=np.float64,
        )

        if (
            points.shape != (2, 3)
            or not np.all(np.isfinite(points))
        ):
            continue

        cid = int(item["candidate_id"])

        lines[cid] = {
            "candidate_id": cid,
            "group_id": item.get("group_id"),
            "points": points,
        }

    return lines


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    poses = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            try:
                position = np.array([
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ])

                quaternion = np.array([
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ])

                if not (
                    np.all(np.isfinite(position))
                    and np.all(np.isfinite(quaternion))
                ):
                    continue

                poses.append({
                    "frame": Path(
                        row["filename"]
                    ).name,
                    "position": position,
                    "rotation": (
                        Rotation.from_quat(
                            quaternion
                        ).as_matrix()
                    ),
                })

            except (ValueError, KeyError):
                continue

    if not poses:
        raise RuntimeError(
            "No valid camera poses in manifest."
        )

    return poses


def frame_number(filename):
    try:
        return int(
            Path(filename).stem.split("_")[-1]
        )
    except ValueError:
        return None


def load_image(name, cache):
    if name not in cache:
        cache[name] = cv2.imread(
            str(FRAME_DIR / name),
            cv2.IMREAD_GRAYSCALE,
        )

    return cache[name]


def project_line(points, pose, shape):
    height, width = shape

    fractions = np.linspace(
        0.0,
        1.0,
        SAMPLES_PER_LINE,
    )[:, None]

    world = (
        points[0]
        + fractions
        * (points[1] - points[0])
    )

    camera = (
        world - pose["position"]
    ) @ pose["rotation"]

    if np.any(camera[:, 2] <= MIN_DEPTH_M):
        return None

    pixels, _ = cv2.fisheye.projectPoints(
        camera.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        D,
    )

    pixels = pixels.reshape(-1, 2)

    if not np.all(np.isfinite(pixels)):
        return None

    inside = (
        (pixels[:, 0] >= IMAGE_MARGIN_PX)
        & (
            pixels[:, 0]
            < width - IMAGE_MARGIN_PX
        )
        & (pixels[:, 1] >= IMAGE_MARGIN_PX)
        & (
            pixels[:, 1]
            < height - IMAGE_MARGIN_PX
        )
    )

    visible = pixels[inside]

    if len(visible) < 3:
        return None

    length = float(
        np.linalg.norm(
            np.diff(visible, axis=0),
            axis=1,
        ).sum()
    )

    if length < MIN_VISIBLE_LENGTH_PX:
        return None

    return visible


def point_curve_distance(points, curve):
    if curve is None or len(curve) < 2:
        return np.full(
            len(points),
            np.inf,
            dtype=np.float64,
        )

    a = curve[:-1]
    v = np.diff(curve, axis=0)

    denominator = np.maximum(
        np.sum(v * v, axis=1),
        1e-12,
    )

    relative = (
        points[:, None, :]
        - a[None, :, :]
    )

    t = np.clip(
        np.sum(
            relative * v[None, :, :],
            axis=2,
        )
        / denominator[None, :],
        0.0,
        1.0,
    )

    closest = (
        a[None, :, :]
        + t[:, :, None]
        * v[None, :, :]
    )

    return np.linalg.norm(
        points[:, None, :]
        - closest,
        axis=2,
    ).min(axis=1)


def detect_segments(gray):
    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    detected = detector.detect(gray)[0]

    if detected is None:
        return []

    segments = []

    for item in detected.reshape(-1, 4):
        a = item[:2].astype(np.float64)
        b = item[2:].astype(np.float64)

        length = float(
            np.linalg.norm(b - a)
        )

        if length >= MIN_SEGMENT_LENGTH_PX:
            segments.append({
                "a": a,
                "b": b,
                "length": length,
            })

    return segments


def associate_segments(segments, curve):
    direction = curve[-1] - curve[0]
    magnitude = np.linalg.norm(direction)

    if magnitude < 1e-9:
        return []

    direction /= magnitude

    matches = []

    for segment in segments:
        a = segment["a"]
        b = segment["b"]
        length = segment["length"]

        axis = (b - a) / length

        angle = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        abs(axis @ direction),
                        0.0,
                        1.0,
                    )
                )
            )
        )

        if angle > MAX_SEGMENT_ANGLE_DEG:
            continue

        fractions = np.linspace(
            0.0, 1.0, 7
        )[:, None]

        samples = (
            a + fractions * (b - a)
        )

        error = float(
            np.median(
                point_curve_distance(
                    samples,
                    curve,
                )
            )
        )

        if error <= MAX_SEGMENT_DISTANCE_PX:
            matches.append(segment)

    return matches


def detect_features(gray, matches):
    height, width = gray.shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for segment in matches:
        a = np.rint(
            segment["a"]
        ).astype(np.int32)

        b = np.rint(
            segment["b"]
        ).astype(np.int32)

        cv2.line(
            mask,
            tuple(a),
            tuple(b),
            255,
            7,
        )

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=150,
        qualityLevel=0.01,
        minDistance=6,
        mask=mask,
        blockSize=5,
    )

    if corners is None:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    return corners.reshape(
        -1, 2
    ).astype(np.float32)


def track_sequence(images, points):
    """
    Preserve point identities across both future frames.

    Returns only features valid throughout the sequence,
    with positions in each of the two target frames.
    """
    current = points.reshape(-1, 1, 2)
    count = len(points)

    valid = np.ones(
        count,
        dtype=bool,
    )

    future_positions = []

    for gray_a, gray_b in zip(
        images[:-1],
        images[1:],
    ):
        forward, status_f, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray_a,
                gray_b,
                current,
                None,
                **LK_PARAMS,
            )
        )

        if forward is None:
            return None

        backward, status_b, _ = (
            cv2.calcOpticalFlowPyrLK(
                gray_b,
                gray_a,
                forward,
                None,
                **LK_PARAMS,
            )
        )

        if backward is None:
            return None

        fb = np.linalg.norm(
            backward.reshape(-1, 2)
            - current.reshape(-1, 2),
            axis=1,
        )

        valid &= (
            (status_f.reshape(-1) == 1)
            & (status_b.reshape(-1) == 1)
            & np.isfinite(fb)
            & (fb <= MAX_FB_ERROR_PX)
            & np.all(
                np.isfinite(
                    forward.reshape(-1, 2)
                ),
                axis=1,
            )
        )

        future_positions.append(
            forward.reshape(-1, 2).copy()
        )

        current = forward

    if valid.sum() == 0:
        return None

    return [
        positions[valid].astype(np.float64)
        for positions in future_positions
    ]


def unit_vector(vector):
    norm = float(
        np.linalg.norm(vector)
    )

    if norm < 1e-10:
        return None

    return vector / norm


def build_perturbations(points):
    """
    Generate nearby 3D line hypotheses without changing
    the original geometry.

    Shifts are perpendicular to the line direction.
    Rotations are around its midpoint, about the same
    two perpendicular axes.

    All candidates receive the same perturbation grid.
    """
    midpoint = points.mean(axis=0)

    direction = unit_vector(
        points[1] - points[0]
    )

    if direction is None:
        return []

    # Deterministically choose a stable reference axis.
    coordinate_axes = np.eye(3)

    reference = min(
        coordinate_axes,
        key=lambda axis: abs(
            float(axis @ direction)
        ),
    )

    axis_u = unit_vector(
        np.cross(direction, reference)
    )

    axis_v = unit_vector(
        np.cross(direction, axis_u)
    )

    if axis_u is None or axis_v is None:
        return []

    alternatives = []

    for axis_name, axis in (
        ("u", axis_u),
        ("v", axis_v),
    ):
        for amount in SHIFT_AMOUNTS_M:
            for sign in (-1, 1):
                offset = sign * amount * axis

                alternatives.append({
                    "name": (
                        f"shift_{axis_name}_"
                        f"{sign * amount:+.3f}m"
                    ),
                    "family": "translation",
                    "magnitude": amount,
                    "points": points + offset,
                })

    for axis_name, axis in (
        ("u", axis_u),
        ("v", axis_v),
    ):
        for angle in ROTATION_AMOUNTS_DEG:
            for sign in (-1, 1):
                signed_angle = sign * angle

                rotation = Rotation.from_rotvec(
                    np.radians(
                        signed_angle
                    ) * axis
                ).as_matrix()

                rotated = (
                    (points - midpoint)
                    @ rotation.T
                    + midpoint
                )

                alternatives.append({
                    "name": (
                        f"rotate_{axis_name}_"
                        f"{signed_angle:+.1f}deg"
                    ),
                    "family": "rotation",
                    "magnitude": angle,
                    "points": rotated,
                })

    return alternatives


def select_windows(candidate, previous, poses):
    """
    Use Step 99's existing source-frame schedule,
    without using its target wins to select windows.
    """
    index = {
        pose["frame"]: i
        for i, pose in enumerate(poses)
    }

    source_names = []

    for triplet in previous.get(
        "triplets", []
    ):
        names = triplet.get("frames", [])

        if names:
            source_names.append(names[0])

    # Also allow the original Step 99 triplet schedule
    # to remain unchanged for candidate comparisons.
    source_names = list(
        dict.fromkeys(source_names)
    )

    windows = []

    for name in source_names:
        start = index.get(name)

        if (
            start is None
            or start + 2 >= len(poses)
        ):
            continue

        triple = poses[start:start + 3]

        numbers = [
            frame_number(pose["frame"])
            for pose in triple
        ]

        if any(n is None for n in numbers):
            continue

        if not (
            numbers[1] - numbers[0] == 1
            and numbers[2] - numbers[1] == 1
        ):
            continue

        windows.append(triple)

        if len(windows) >= MAX_WINDOWS_PER_CANDIDATE:
            break

    return windows


def score_hypothesis(
    points,
    target_poses,
    tracked_positions,
    shape,
):
    """
    Evaluate the same tracked features in BOTH future frames.
    Invalid projection means the hypothesis is not comparable.
    """
    errors = []

    for pose, observations in zip(
        target_poses,
        tracked_positions,
    ):
        curve = project_line(
            points,
            pose,
            shape,
        )

        if curve is None:
            return None

        error = float(
            np.median(
                point_curve_distance(
                    observations,
                    curve,
                )
            )
        )

        errors.append(error)

    return {
        "per_frame_median_errors_px": errors,
        "mean_future_frame_error_px": float(
            np.mean(errors)
        ),
        "worst_future_frame_error_px": float(
            np.max(errors)
        ),
    }


def evaluate_window(
    line,
    triple,
    alternatives,
    cache,
):
    names = [
        pose["frame"]
        for pose in triple
    ]

    images = [
        load_image(name, cache)
        for name in names
    ]

    if any(image is None for image in images):
        return {
            "status": "IMAGE_UNAVAILABLE",
            "frames": names,
        }

    if len({
        image.shape
        for image in images
    }) != 1:
        return {
            "status": "IMAGE_SHAPE_MISMATCH",
            "frames": names,
        }

    shape = images[0].shape

    original_curves = [
        project_line(
            line["points"],
            pose,
            shape,
        )
        for pose in triple
    ]

    if any(
        curve is None
        for curve in original_curves
    ):
        return {
            "status": "ORIGINAL_PROJECTION_UNAVAILABLE",
            "frames": names,
        }

    global_segments = detect_segments(
        images[0]
    )

    matches = associate_segments(
        global_segments,
        original_curves[0],
    )

    if not matches:
        return {
            "status": "NO_ASSOCIATED_LSD_SEGMENT",
            "frames": names,
            "global_segment_count": len(
                global_segments
            ),
        }

    features = detect_features(
        images[0],
        matches,
    )

    if len(features) < MIN_FEATURES:
        return {
            "status": "INSUFFICIENT_FEATURES",
            "frames": names,
            "feature_count": int(
                len(features)
            ),
        }

    tracked = track_sequence(
        images,
        features,
    )

    if tracked is None:
        surviving = 0
    else:
        surviving = len(tracked[0])

    fraction = (
        surviving / len(features)
    )

    if (
        surviving < MIN_SURVIVORS
        or fraction < MIN_TRACK_FRACTION
    ):
        return {
            "status": "TRACKING_FAILED",
            "frames": names,
            "feature_count": int(
                len(features)
            ),
            "surviving_count": int(
                surviving
            ),
            "survival_fraction": float(
                fraction
            ),
        }

    original_score = score_hypothesis(
        line["points"],
        triple[1:],
        tracked,
        shape,
    )

    if original_score is None:
        return {
            "status": "ORIGINAL_PROJECTION_UNAVAILABLE",
            "frames": names,
        }

    alternative_results = []

    for alternative in alternatives:
        score = score_hypothesis(
            alternative["points"],
            triple[1:],
            tracked,
            shape,
        )

        if score is None:
            alternative_results.append({
                "name": alternative["name"],
                "family": alternative["family"],
                "magnitude": alternative["magnitude"],
                "status": "PROJECTION_UNAVAILABLE",
            })
            continue

        original_error = original_score[
            "mean_future_frame_error_px"
        ]

        alternative_error = score[
            "mean_future_frame_error_px"
        ]

        difference = (
            alternative_error
            - original_error
        )

        if difference >= MIN_WIN_MARGIN_PX:
            result = "ORIGINAL_BETTER"

        elif difference <= -MIN_WIN_MARGIN_PX:
            result = "PERTURBATION_BETTER"

        else:
            result = "INDISTINGUISHABLE"

        alternative_results.append({
            "name": alternative["name"],
            "family": alternative["family"],
            "magnitude": alternative["magnitude"],
            "status": result,
            "mean_error_px": alternative_error,
            "per_frame_errors_px": score[
                "per_frame_median_errors_px"
            ],
            "error_difference_px": difference,
        })

    comparable = [
        item
        for item in alternative_results
        if item["status"]
        != "PROJECTION_UNAVAILABLE"
    ]

    if not comparable:
        status = "NO_COMPARABLE_PERTURBATIONS"

    else:
        status = "PERTURBATION_COMPARISON_COMPLETED"

    counts = Counter(
        item["status"]
        for item in alternative_results
    )

    return {
        "status": status,
        "frames": names,
        "global_segment_count": len(
            global_segments
        ),
        "associated_segment_count": len(
            matches
        ),
        "source_feature_count": int(
            len(features)
        ),
        "surviving_track_count": int(
            surviving
        ),
        "track_survival_fraction": float(
            fraction
        ),
        "original": original_score,
        "original_within_error_threshold": bool(
            original_score[
                "mean_future_frame_error_px"
            ] <= MAX_ORIGINAL_ERROR_PX
        ),
        "perturbation_status_counts": dict(
            counts
        ),
        "perturbations": alternative_results,
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = load_lines()
    poses = load_manifest()

    previous_rows = load_json(
        STEP99_PATH
    ).get("results", [])

    previous = {
        int(row["candidate_id"]): row
        for row in previous_rows
    }

    for cid in TARGET_IDS:
        if cid not in lines:
            raise RuntimeError(
                f"Missing frozen candidate: {cid}"
            )

        if cid not in previous:
            raise RuntimeError(
                f"Missing Step 99 history: {cid}"
            )

    cache = {}
    results = []
    global_window_statuses = Counter()
    global_control_statuses = Counter()

    for cid in TARGET_IDS:
        line = lines[cid]

        alternatives = build_perturbations(
            line["points"]
        )

        windows = select_windows(
            cid,
            previous[cid],
            poses,
        )

        evaluations = []

        for triple in windows:
            outcome = evaluate_window(
                line,
                triple,
                alternatives,
                cache,
            )

            evaluations.append(outcome)

            global_window_statuses[
                outcome["status"]
            ] += 1

            global_control_statuses.update(
                outcome.get(
                    "perturbation_status_counts",
                    {},
                )
            )

        completed = [
            item
            for item in evaluations
            if item["status"]
            == "PERTURBATION_COMPARISON_COMPLETED"
        ]

        candidate_control_counts = Counter()

        for item in completed:
            candidate_control_counts.update(
                item["perturbation_status_counts"]
            )

        original_better = candidate_control_counts[
            "ORIGINAL_BETTER"
        ]

        perturbed_better = candidate_control_counts[
            "PERTURBATION_BETTER"
        ]

        indistinguishable = candidate_control_counts[
            "INDISTINGUISHABLE"
        ]

        if not completed:
            decision = "INSUFFICIENT_PERTURBATION_EVIDENCE"

        elif perturbed_better > original_better:
            decision = "LOCAL_GEOMETRY_CHALLENGED"

        elif (
            original_better > perturbed_better
            and len(completed) >= 2
        ):
            decision = "LOCAL_PERTURBATION_RESISTANCE_OBSERVED"

        else:
            decision = "LOCAL_GEOMETRY_UNRESOLVED"

        results.append({
            "candidate_id": cid,
            "group_id": line["group_id"],
            "step96_status_unchanged": previous[cid][
                "step96_status_unchanged"
            ],
            "step98b_decision_unchanged": previous[cid][
                "step98b_decision_unchanged"
            ],
            "step99_decision_unchanged": previous[cid][
                "decision"
            ],
            "perturbation_count": len(alternatives),
            "windows_selected": len(windows),
            "windows_completed": len(completed),
            "original_better_count": original_better,
            "perturbation_better_count": perturbed_better,
            "indistinguishable_count": indistinguishable,
            "decision": decision,
            "physical_line_verified": False,
            "windows": evaluations,
        })

        print(
            f"Candidate {cid:03d}: "
            f"windows={len(windows)} "
            f"completed={len(completed)} "
            f"original_better={original_better} "
            f"perturbed_better={perturbed_better} "
            f"indistinguishable={indistinguishable} "
            f"=> {decision}",
            flush=True,
        )

    detail = {
        "method": (
            "step100_local_3d_perturbation_"
            "cross_view_diagnostic"
        ),
        "results": results,
    }

    save_json(
        OUTPUT / "per_candidate_perturbations.json",
        detail,
    )

    decision_counts = Counter(
        row["decision"]
        for row in results
    )

    report = {
        "method": (
            "step100_local_3d_perturbation_"
            "cross_view_diagnostic"
        ),
        "target_candidate_ids": list(TARGET_IDS),
        "candidate_count": len(results),
        "windows_selected": sum(
            row["windows_selected"]
            for row in results
        ),
        "windows_completed": sum(
            row["windows_completed"]
            for row in results
        ),
        "window_status_counts": dict(
            global_window_statuses
        ),
        "perturbation_status_counts": dict(
            global_control_statuses
        ),
        "candidate_decision_counts": dict(
            decision_counts
        ),
        "perturbation_resistant_candidate_ids": [
            row["candidate_id"]
            for row in results
            if row["decision"]
            == "LOCAL_PERTURBATION_RESISTANCE_OBSERVED"
        ],
        "geometry_challenged_candidate_ids": [
            row["candidate_id"]
            for row in results
            if row["decision"]
            == "LOCAL_GEOMETRY_CHALLENGED"
        ],
        "parameters": {
            "shift_amounts_m": list(
                SHIFT_AMOUNTS_M
            ),
            "rotation_amounts_deg": list(
                ROTATION_AMOUNTS_DEG
            ),
            "max_original_error_px": (
                MAX_ORIGINAL_ERROR_PX
            ),
            "minimum_win_margin_px": (
                MIN_WIN_MARGIN_PX
            ),
            "minimum_survivors": MIN_SURVIVORS,
            "maximum_forward_backward_error_px": (
                MAX_FB_ERROR_PX
            ),
        },
        "step96_decisions_modified": False,
        "step98b_decisions_modified": False,
        "step99_decisions_modified": False,
        "physical_lines_verified": False,
        "structural_identity_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Source LSD segment association is conditioned on the original 3D projection.",
            "Future-frame optical-flow positions are used for scoring, but their source selection is not independent of the original hypothesis.",
            "A single tracked feature set may contain multiple physical surfaces.",
            "Perturbations are tested against the same observations, but the source association can favor the original geometry.",
            "Some small 3D perturbations may be unobservable under the available camera motion.",
            "The comparison uses median image distance, not a calibrated probabilistic uncertainty model.",
            "The number of perturbation wins is correlated across controls and windows; it is not a count of independent evidence.",
            "Overlapping windows reuse video frames and feature tracks.",
            "Camera calibration and poses are treated as given.",
            "No fresh held-out recording is used.",
            "No CAD registration or construction-progress inference is performed.",
        ],
        "decision": (
            "PERTURBATION_DIAGNOSTICS_AVAILABLE"
        ),
    }

    save_json(
        OUTPUT / "perturbation_report.json",
        report,
    )

    print("\n=== STEP 100 RESULTS ===")
    print(
        "Candidates:",
        report["candidate_count"]
    )
    print(
        "Windows selected:",
        report["windows_selected"]
    )
    print(
        "Windows completed:",
        report["windows_completed"]
    )
    print(
        "Window outcomes:",
        dict(global_window_statuses)
    )
    print(
        "Perturbation outcomes:",
        dict(global_control_statuses)
    )
    print(
        "Candidate decisions:",
        dict(decision_counts)
    )
    print(
        "Physical identity verified: NO"
    )
    print(
        "CAD registration verified: NO"
    )
    print(
        "Construction progress verified: NO"
    )
    print(
        "Report:",
        OUTPUT / "perturbation_report.json"
    )


if __name__ == "__main__":
    main()
