
"""
Under-construction — Step 96

Independent image-level validation of frozen 3D
line hypotheses using previously unused frames.

Inputs:
    outputs/line_ambiguity/resolved_lines.json
    outputs/line_evidence/line_observations.json
    _archive/.../cam0/frame_manifest.csv
    _archive/.../cam0/frames/

Outputs:
    outputs/line_validation/validation_report.json
    outputs/line_validation/per_line_results.json

No CAD, AI models, manual labels, or geometry fitting.

A successful image test does not establish physical
structural identity or construction progress.
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

RESOLVED_PATH = (
    ROOT / "outputs" / "line_ambiguity"
    / "resolved_lines.json"
)

OBSERVATIONS_PATH = (
    ROOT / "outputs" / "line_evidence"
    / "line_observations.json"
)

CAMERA_ROOT = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data" / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

FRAME_DIR = CAMERA_ROOT / "frames"
MANIFEST_PATH = CAMERA_ROOT / "frame_manifest.csv"

OUTPUT = ROOT / "outputs" / "line_validation"

K = np.array([
    [465.3015482593691, 0.0, 730.0455886686005],
    [0.0, 465.32303798346413, 720.1427007671206],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D = np.array([
    0.025800718903376804,
    -0.010909240777406872,
    -0.0016899537986031076,
    0.00014766801645260894,
], dtype=np.float64)

# Validation configuration, fixed before inspecting
# any image-level validation results.
MIN_VISIBLE_LENGTH_PX = 35.0
MIN_VISIBLE_FRACTION = 0.40
MIN_VALIDATION_FRAMES = 3

EDGE_DISTANCE_TOLERANCE_PX = 3.0
CONTROL_OFFSETS_PX = (-48.0, -24.0, 24.0, 48.0)

MIN_EDGE_SUPPORT = 0.35
MIN_CONTROL_ADVANTAGE = 0.12
MIN_FRAME_WIN_FRACTION = 0.65

SAMPLES_PER_LINE = 80
IMAGE_MARGIN_PX = 8

# Prevent extremely distant or behind-camera
# geometry from contributing.
MIN_CAMERA_DEPTH_M = 0.20

# Frame selection is deterministic.
MAX_VALIDATION_FRAMES = 90

EPS = 1e-9


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def extract_training_frames(data):
    """
    Step 92 frames are excluded in their entirety,
    not just individual line observations.

    Fail closed if the inventory format is
    unexpected or the exclusion set is empty.
    """
    observations = data.get("observations")

    if not isinstance(observations, list):
        raise RuntimeError(
            "Expected 'observations' list in "
            "Step 92 line_observations.json."
        )

    frames = set()

    for observation in observations:
        frame = observation.get("frame")

        if frame:
            frames.add(Path(frame).name)

    if not frames:
        raise RuntimeError(
            "Cannot establish training-frame exclusion."
        )

    return frames


def load_poses():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing pose manifest: {MANIFEST_PATH}"
        )

    poses = {}

    with MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "filename",
            "tx", "ty", "tz",
            "qx", "qy", "qz", "qw",
        }

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise RuntimeError(
                "Pose manifest columns do not match "
                "the expected format."
            )

        for row in reader:
            try:
                name = Path(row["filename"]).name

                position = np.array([
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ], dtype=np.float64)

                quaternion = [
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ]

                rotation_map_camera = (
                    Rotation.from_quat(
                        quaternion
                    ).as_matrix()
                )

                if not np.all(
                    np.isfinite(position)
                ):
                    continue

                poses[name] = {
                    "position": position,
                    "R_map_camera": (
                        rotation_map_camera
                    ),
                }

            except (
                ValueError,
                KeyError,
            ):
                continue

    return poses


def select_validation_frames(training, poses):
    """
    Only image files not present in the Step 92
    inventory can be validation candidates.
    """
    available = sorted(
        path
        for path in FRAME_DIR.glob("*")
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
        and path.name in poses
        and path.name not in training
    )

    if len(available) > MAX_VALIDATION_FRAMES:
        indices = np.linspace(
            0,
            len(available) - 1,
            MAX_VALIDATION_FRAMES,
            dtype=int,
        )

        available = [
            available[int(i)]
            for i in np.unique(indices)
        ]

    return available


def project_points(points_map, pose):
    """
    World/map -> camera -> OpenCV fisheye pixels.

    Manifest quaternion convention:
        camera-to-map rotation.
    """
    R = pose["R_map_camera"]
    t = pose["position"]

    camera_points = (
        np.asarray(points_map, dtype=np.float64)
        - t
    ) @ R

    depths = camera_points[:, 2]

    if np.any(
        depths <= MIN_CAMERA_DEPTH_M
    ):
        return None

    object_points = camera_points.reshape(
        -1, 1, 3
    )

    pixels, _ = cv2.fisheye.projectPoints(
        object_points,
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K,
        D.reshape(4, 1),
    )

    pixels = pixels.reshape(-1, 2)

    if not np.all(np.isfinite(pixels)):
        return None

    return pixels


def sample_projected_line(
    endpoints_map,
    pose,
    width,
    height,
):
    """
    Sample the 3D segment, rather than only
    projecting endpoints. This also handles
    fisheye curvature more appropriately.
    """
    a, b = endpoints_map

    fractions = np.linspace(
        0.0, 1.0, SAMPLES_PER_LINE
    )

    points = (
        a[None, :]
        + fractions[:, None]
        * (b - a)[None, :]
    )

    projected = project_points(
        points, pose
    )

    if projected is None:
        return None

    inside = (
        (projected[:, 0] >= IMAGE_MARGIN_PX)
        & (
            projected[:, 0]
            < width - IMAGE_MARGIN_PX
        )
        & (
            projected[:, 1] >= IMAGE_MARGIN_PX
        )
        & (
            projected[:, 1]
            < height - IMAGE_MARGIN_PX
        )
    )

    visible_fraction = float(
        np.mean(inside)
    )

    if (
        visible_fraction
        < MIN_VISIBLE_FRACTION
    ):
        return None

    visible = projected[inside]

    if len(visible) < 3:
        return None

    lengths = np.linalg.norm(
        np.diff(visible, axis=0),
        axis=1,
    )

    visible_length = float(
        np.sum(lengths)
    )

    if (
        visible_length
        < MIN_VISIBLE_LENGTH_PX
    ):
        return None

    return {
        "points": visible,
        "visible_fraction": visible_fraction,
        "visible_length_px": visible_length,
    }


def edge_distance_map(gray):
    """
    Image-only edge detector.

    Parameters are fixed across all frames.
    """
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 1.0
    )

    edges = cv2.Canny(
        blurred,
        60,
        150,
        L2gradient=True,
    )

    inverse = (
        edges == 0
    ).astype(np.uint8)

    distance = cv2.distanceTransform(
        inverse,
        cv2.DIST_L2,
        3,
    )

    return distance


def point_normals(points):
    """
    Image-plane normals along a projected
    potentially curved fisheye line.
    """
    tangent = np.gradient(
        points,
        axis=0,
    )

    norms = np.linalg.norm(
        tangent,
        axis=1,
    )

    valid = norms > EPS

    normals = np.zeros_like(
        tangent
    )

    normals[valid, 0] = (
        -tangent[valid, 1]
        / norms[valid]
    )

    normals[valid, 1] = (
        tangent[valid, 0]
        / norms[valid]
    )

    return normals


def sample_distance(distance_map, points):
    """
    Bilinear distance-map sampling.
    Out-of-bounds points are not counted
    as successful edge support.
    """
    height, width = (
        distance_map.shape
    )

    x = points[:, 0]
    y = points[:, 1]

    inside = (
        (x >= 0)
        & (x <= width - 2)
        & (y >= 0)
        & (y <= height - 2)
    )

    distances = np.full(
        len(points),
        np.inf,
        dtype=np.float64,
    )

    if not np.any(inside):
        return distances

    xi = x[inside]
    yi = y[inside]

    x0 = np.floor(xi).astype(int)
    y0 = np.floor(yi).astype(int)

    dx = xi - x0
    dy = yi - y0

    a = distance_map[y0, x0]
    b = distance_map[y0, x0 + 1]
    c = distance_map[y0 + 1, x0]
    d = distance_map[y0 + 1, x0 + 1]

    distances[inside] = (
        (1 - dx) * (1 - dy) * a
        + dx * (1 - dy) * b
        + (1 - dx) * dy * c
        + dx * dy * d
    )

    return distances


def edge_support(distance_map, points):
    distances = sample_distance(
        distance_map, points
    )

    return float(
        np.mean(
            distances
            <= EDGE_DISTANCE_TOLERANCE_PX
        )
    )


def evaluate_projection(
    distance_map,
    projected_points,
):
    """
    Compare a frozen projected line with
    four locally displaced control curves.

    Controls preserve the projected shape.
    """
    normals = point_normals(
        projected_points
    )

    real_support = edge_support(
        distance_map,
        projected_points,
    )

    control_supports = []

    for offset in CONTROL_OFFSETS_PX:
        shifted = (
            projected_points
            + offset * normals
        )

        control_supports.append(
            edge_support(
                distance_map,
                shifted,
            )
        )

    control_mean = float(
        np.mean(control_supports)
    )

    control_best = float(
        np.max(control_supports)
    )

    advantage = (
        real_support - control_mean
    )

    win = (
        real_support >= MIN_EDGE_SUPPORT
        and advantage >= MIN_CONTROL_ADVANTAGE
        and real_support > control_best
    )

    return {
        "real_edge_support": real_support,
        "control_edge_supports": (
            control_supports
        ),
        "mean_control_support": control_mean,
        "best_control_support": control_best,
        "control_advantage": advantage,
        "frame_win": bool(win),
    }


def load_lines():
    data = load_json(
        RESOLVED_PATH
    )

    lines = []

    for item in data.get(
        "selected_lines", []
    ):
        endpoints = np.asarray(
            item.get("endpoints_map", []),
            dtype=np.float64,
        )

        if (
            endpoints.shape != (2, 3)
            or not np.all(
                np.isfinite(endpoints)
            )
        ):
            continue

        lines.append({
            "candidate_id": (
                item["candidate_id"]
            ),
            "group_id": item["group_id"],
            "endpoints": endpoints,
            "length_m": float(
                np.linalg.norm(
                    endpoints[1]
                    - endpoints[0]
                )
            ),
            "orientation_class": (
                item.get(
                    "orientation_class",
                    "unknown",
                )
            ),
        })

    return lines


def evaluate_line(
    line,
    frames,
    frame_data,
    poses,
):
    results = []

    for frame_path in frames:
        name = frame_path.name

        data = frame_data.get(name)

        if data is None:
            continue

        projected = sample_projected_line(
            line["endpoints"],
            poses[name],
            data["width"],
            data["height"],
        )

        if projected is None:
            continue

        evidence = evaluate_projection(
            data["distance_map"],
            projected["points"],
        )

        results.append({
            "frame": name,
            "visible_fraction": (
                projected["visible_fraction"]
            ),
            "visible_length_px": (
                projected["visible_length_px"]
            ),
            **evidence,
        })

    count = len(results)

    if count == 0:
        return {
            "candidate_id": line["candidate_id"],
            "group_id": line["group_id"],
            "length_m": line["length_m"],
            "orientation_class": (
                line["orientation_class"]
            ),
            "eligible_frame_count": 0,
            "winning_frame_count": 0,
            "winning_frame_fraction": 0.0,
            "median_real_edge_support": None,
            "median_control_edge_support": None,
            "median_control_advantage": None,
            "validation_status": (
                "INSUFFICIENT_VISIBILITY"
            ),
            "frames": [],
        }

    real = [
        r["real_edge_support"]
        for r in results
    ]

    control = [
        r["mean_control_support"]
        for r in results
    ]

    advantage = [
        r["control_advantage"]
        for r in results
    ]

    wins = sum(
        r["frame_win"]
        for r in results
    )

    win_fraction = wins / count

    median_real = float(
        np.median(real)
    )

    median_control = float(
        np.median(control)
    )

    median_advantage = float(
        np.median(advantage)
    )

    if count < MIN_VALIDATION_FRAMES:
        status = "INSUFFICIENT_VISIBILITY"

    elif (
        win_fraction >= MIN_FRAME_WIN_FRACTION
        and median_real >= MIN_EDGE_SUPPORT
        and median_advantage
        >= MIN_CONTROL_ADVANTAGE
    ):
        status = (
            "IMAGE_SUPPORT_PASSES_CONTROLS"
        )

    else:
        status = (
            "IMAGE_SUPPORT_NOT_ESTABLISHED"
        )

    return {
        "candidate_id": line["candidate_id"],
        "group_id": line["group_id"],
        "length_m": line["length_m"],
        "orientation_class": (
            line["orientation_class"]
        ),
        "eligible_frame_count": count,
        "winning_frame_count": wins,
        "winning_frame_fraction": (
            win_fraction
        ),
        "median_real_edge_support": (
            median_real
        ),
        "median_control_edge_support": (
            median_control
        ),
        "median_control_advantage": (
            median_advantage
        ),
        "validation_status": status,
        "frames": results,
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n=== STEP 96: UNSEEN-FRAME "
        "LINE VALIDATION ==="
    )

    training_data = load_json(
        OBSERVATIONS_PATH
    )

    training_frames = (
        extract_training_frames(
            training_data
        )
    )

    poses = load_poses()
    lines = load_lines()

    if not lines:
        raise RuntimeError(
            "No Step 95 lines available."
        )

    frames = select_validation_frames(
        training_frames,
        poses,
    )

    if not frames:
        raise RuntimeError(
            "No unused frames available. "
            "Cannot perform independent "
            "image validation."
        )

    overlap = (
        {path.name for path in frames}
        & training_frames
    )

    if overlap:
        raise AssertionError(
            "Training/validation frame leakage."
        )

    print(
        "Training frames excluded:",
        len(training_frames),
    )
    print(
        "Validation frames available:",
        len(frames),
    )
    print(
        "Frozen 3D lines:",
        len(lines),
    )

    frame_data = {}

    for path in frames:
        gray = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray is None:
            continue

        height, width = gray.shape

        frame_data[path.name] = {
            "width": width,
            "height": height,
            "distance_map": (
                edge_distance_map(gray)
            ),
        }

    usable_frames = [
        path
        for path in frames
        if path.name in frame_data
    ]

    results = []

    for index, line in enumerate(
        lines, start=1
    ):
        result = evaluate_line(
            line,
            usable_frames,
            frame_data,
            poses,
        )

        results.append(result)

        print(
            f"[{index}/{len(lines)}] "
            f"candidate={line['candidate_id']} "
            f"frames={result['eligible_frame_count']} "
            f"status={result['validation_status']}"
        )

    status_counts = Counter(
        r["validation_status"]
        for r in results
    )

    passing = [
        r
        for r in results
        if r["validation_status"]
        == "IMAGE_SUPPORT_PASSES_CONTROLS"
    ]

    sufficiently_visible = [
        r
        for r in results
        if r["eligible_frame_count"]
        >= MIN_VALIDATION_FRAMES
    ]

    orientation_counts = Counter(
        r["orientation_class"]
        for r in passing
    )

    frame_evaluations = sum(
        r["eligible_frame_count"]
        for r in results
    )

    winning_evaluations = sum(
        r["winning_frame_count"]
        for r in results
    )

    per_line_path = (
        OUTPUT / "per_line_results.json"
    )

    per_line_path.write_text(
        json.dumps(
            {
                "method": (
                    "frozen_line_unseen_image_"
                    "edge_control_test"
                ),
                "results": results,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = {
        "method": (
            "frozen_3d_line_unseen_frame_"
            "edge_discrimination"
        ),
        "training_frames_excluded": (
            len(training_frames)
        ),
        "validation_frames_selected": (
            len(frames)
        ),
        "validation_frames_readable": (
            len(usable_frames)
        ),
        "training_validation_overlap": 0,
        "frozen_line_count": len(lines),
        "lines_with_sufficient_visibility": (
            len(sufficiently_visible)
        ),
        "lines_passing_image_controls": (
            len(passing)
        ),
        "lines_failing_or_insufficient": (
            len(lines) - len(passing)
        ),
        "status_counts": dict(
            status_counts
        ),
        "passing_orientation_counts": dict(
            orientation_counts
        ),
        "passing_candidate_ids": [
            r["candidate_id"]
            for r in passing
        ],
        "total_visible_line_frame_evaluations": (
            frame_evaluations
        ),
        "total_winning_line_frame_evaluations": (
            winning_evaluations
        ),
        "thresholds": {
            "minimum_visible_length_px": (
                MIN_VISIBLE_LENGTH_PX
            ),
            "minimum_visible_fraction": (
                MIN_VISIBLE_FRACTION
            ),
            "minimum_validation_frames": (
                MIN_VALIDATION_FRAMES
            ),
            "edge_distance_tolerance_px": (
                EDGE_DISTANCE_TOLERANCE_PX
            ),
            "control_offsets_px": list(
                CONTROL_OFFSETS_PX
            ),
            "minimum_edge_support": (
                MIN_EDGE_SUPPORT
            ),
            "minimum_control_advantage": (
                MIN_CONTROL_ADVANTAGE
            ),
            "minimum_frame_win_fraction": (
                MIN_FRAME_WIN_FRACTION
            ),
        },
        "image_level_control_test_available": True,
        "physical_structural_identity_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Unused frames are independent of the Step 92 sampled-image inventory, assuming no other earlier stage used them for candidate selection.",
            "Adjacent video frames remain temporally correlated.",
            "Camera poses and calibration are shared with reconstruction and are not independently validated.",
            "Local displaced controls are not a complete geometric null model.",
            "Nearby unrelated edges can generate false support.",
            "Canny edge support does not prove correspondence or structural identity.",
            "Lines invisible in the validation frames cannot be evaluated.",
            "Thresholds are heuristic and not statistically calibrated.",
            "Only the 42 selected Step 95 hypotheses are evaluated.",
            "No CAD geometry is used.",
        ],
        "decision": (
            "IMAGE_LEVEL_SUPPORT_FOUND_FOR_SUBSET"
            if passing
            else "NO_LINE_PASSED_IMAGE_CONTROLS"
        ),
    }

    report_path = (
        OUTPUT / "validation_report.json"
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
    print(
        "Validation frames:",
        len(usable_frames),
    )
    print(
        "Lines evaluated:",
        len(lines),
    )
    print(
        "Sufficiently visible lines:",
        len(sufficiently_visible),
    )
    print(
        "Lines passing controls:",
        len(passing),
    )
    print(
        "Passing horizontal lines:",
        orientation_counts["near_horizontal"],
    )
    print(
        "Passing vertical lines:",
        orientation_counts["near_vertical"],
    )
    print(
        "Physical identity verified: NO"
    )
    print(
        "CAD registration verified: NO"
    )
    print(
        "Report:",
        report_path,
    )
    print(
        "Per-line results:",
        per_line_path,
    )


if __name__ == "__main__":
    main()
