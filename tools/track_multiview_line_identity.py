
"""
Under-construction — Step 98B

Repair temporal tracking of frozen 3D line hypotheses.

Starts from Step 96 validation frames and tracks into
their actual neighboring frames in the full manifest.

Uses:
    - Classical Lucas-Kanade optical flow
    - Forward-backward consistency
    - Spatially distributed corner tracks
    - Fisheye-aware projected 3D polylines
    - Displaced geometric negative controls

Does NOT:
    - Modify candidate geometry
    - Correct camera calibration or poses
    - Change Step 96 validation decisions
    - Claim physical structural identity
    - Perform CAD registration
    - Estimate construction progress

Inputs:
    outputs/line_ambiguity/resolved_lines.json
    outputs/line_validation/per_line_results.json
    cam0/frame_manifest.csv
    cam0/frames/

Outputs:
    outputs/line_temporal_tracking/tracking_report.json
    outputs/line_temporal_tracking/per_candidate_tracking.json

Run from project root:
    .\\.venv\\Scripts\\python.exe tools\\track_multiview_line_identity.py
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
    ROOT / "outputs" / "line_ambiguity"
    / "resolved_lines.json"
)

VALIDATION_PATH = (
    ROOT / "outputs" / "line_validation"
    / "per_line_results.json"
)

CAMERA_ROOT = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data" / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST_PATH = CAMERA_ROOT / "frame_manifest.csv"
FRAME_DIR = CAMERA_ROOT / "frames"

OUTPUT = (
    ROOT / "outputs" / "line_temporal_tracking"
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

cv2.setNumThreads(1)
cv2.setRNGSeed(98)

SAMPLES_PER_LINE = 80

MIN_DEPTH_M = 0.20
IMAGE_MARGIN_PX = 8
MIN_VISIBLE_LENGTH_PX = 20

MAX_SEED_FRAMES_PER_CANDIDATE = 12
MAX_NEIGHBORS_PER_SEED = 2
MAX_FRAME_NUMBER_GAP = 2

MIN_SEED_POINTS = 5
MIN_TRACKED_POINTS = 5
MIN_TRACKED_FRACTION = 0.40
MIN_SPREAD_FRACTION = 0.25

SEED_EDGE_DISTANCE_PX = 3.0
SEED_PROJECTION_DISTANCE_PX = 6.5

FORWARD_BACKWARD_TOLERANCE_PX = 1.5

TARGET_EDGE_DISTANCE_PX = 3.0
TARGET_PROJECTION_DISTANCE_PX = 8.0

MIN_TARGET_EDGE_FRACTION = 0.50

CONTROL_OFFSETS_PX = (
    -48.0, -24.0, 24.0, 48.0
)

CONTROL_MARGIN_PX = 2.0

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
            f"Missing required input: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    poses = []

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "filename",
            "tx", "ty", "tz",
            "qx", "qy", "qz", "qw",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise RuntimeError(
                "Unexpected pose manifest columns."
            )

        for row in reader:
            try:
                name = Path(
                    row["filename"]
                ).name

                position = np.array([
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ], dtype=np.float64)

                quaternion = np.array([
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ], dtype=np.float64)

                if not (
                    np.all(np.isfinite(position))
                    and np.all(np.isfinite(quaternion))
                ):
                    continue

                rotation = (
                    Rotation.from_quat(
                        quaternion
                    ).as_matrix()
                )

                poses.append({
                    "frame": name,
                    "position": position,
                    "rotation": rotation,
                })

            except (
                ValueError,
                KeyError,
            ):
                continue

    if not poses:
        raise RuntimeError(
            "No valid camera poses."
        )

    return poses


def frame_number(filename):
    try:
        return int(
            Path(filename).stem.split("_")[-1]
        )
    except ValueError:
        return None


def load_lines():
    raw = load_json(LINES_PATH)

    result = {}

    for item in raw.get(
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

        candidate_id = int(
            item["candidate_id"]
        )

        result[candidate_id] = {
            "candidate_id": candidate_id,
            "group_id": item.get(
                "group_id"
            ),
            "endpoints": endpoints,
            "length_m": float(
                np.linalg.norm(
                    endpoints[1]
                    - endpoints[0]
                )
            ),
        }

    return result


def project_line(
    endpoints,
    pose,
    width,
    height,
):
    fractions = np.linspace(
        0.0,
        1.0,
        SAMPLES_PER_LINE,
    )

    points_map = (
        endpoints[0][None, :]
        + fractions[:, None]
        * (
            endpoints[1]
            - endpoints[0]
        )[None, :]
    )

    camera_points = (
        points_map
        - pose["position"]
    ) @ pose["rotation"]

    if np.any(
        camera_points[:, 2]
        <= MIN_DEPTH_M
    ):
        return None

    pixels, _ = (
        cv2.fisheye.projectPoints(
            camera_points.reshape(
                -1, 1, 3
            ),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K,
            D,
        )
    )

    pixels = pixels.reshape(
        -1, 2
    )

    if not np.all(
        np.isfinite(pixels)
    ):
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

    visible_length = float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    visible,
                    axis=0,
                ),
                axis=1,
            )
        )
    )

    if (
        visible_length
        < MIN_VISIBLE_LENGTH_PX
    ):
        return None

    return visible


def point_to_polyline_distance(
    points,
    polyline,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    polyline = np.asarray(
        polyline,
        dtype=np.float64,
    )

    if (
        len(points) == 0
        or len(polyline) < 2
    ):
        return np.full(
            len(points),
            np.inf,
        )

    starts = polyline[:-1]
    ends = polyline[1:]

    vectors = ends - starts

    denominator = np.maximum(
        np.sum(
            vectors * vectors,
            axis=1,
        ),
        1e-12,
    )

    relative = (
        points[:, None, :]
        - starts[None, :, :]
    )

    t = (
        np.sum(
            relative
            * vectors[None, :, :],
            axis=2,
        )
        / denominator[None, :]
    )

    t = np.clip(
        t, 0.0, 1.0
    )

    closest = (
        starts[None, :, :]
        + t[:, :, None]
        * vectors[None, :, :]
    )

    distances = np.linalg.norm(
        points[:, None, :]
        - closest,
        axis=2,
    )

    return np.min(
        distances,
        axis=1,
    )


def polyline_normals(points):
    tangents = np.gradient(
        points,
        axis=0,
    )

    magnitudes = np.linalg.norm(
        tangents,
        axis=1,
    )

    normals = np.zeros_like(
        points
    )

    valid = (
        magnitudes > 1e-9
    )

    normals[valid, 0] = (
        -tangents[valid, 1]
        / magnitudes[valid]
    )

    normals[valid, 1] = (
        tangents[valid, 0]
        / magnitudes[valid]
    )

    return normals


def make_edge_distance(gray):
    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.0,
    )

    edges = cv2.Canny(
        blurred,
        60,
        150,
        L2gradient=True,
    )

    return cv2.distanceTransform(
        (edges == 0).astype(
            np.uint8
        ),
        cv2.DIST_L2,
        3,
    )


def sample_distance_map(
    distance,
    points,
):
    height, width = distance.shape

    x = points[:, 0]
    y = points[:, 1]

    inside = (
        (x >= 0)
        & (x <= width - 2)
        & (y >= 0)
        & (y <= height - 2)
    )

    values = np.full(
        len(points),
        np.inf,
        dtype=np.float64,
    )

    if not np.any(inside):
        return values

    xx = x[inside]
    yy = y[inside]

    x0 = np.floor(
        xx
    ).astype(int)

    y0 = np.floor(
        yy
    ).astype(int)

    dx = xx - x0
    dy = yy - y0

    values[inside] = (
        (1 - dx)
        * (1 - dy)
        * distance[y0, x0]
        + dx
        * (1 - dy)
        * distance[y0, x0 + 1]
        + (1 - dx)
        * dy
        * distance[y0 + 1, x0]
        + dx
        * dy
        * distance[y0 + 1, x0 + 1]
    )

    return values


def seed_corners(
    gray,
    projection,
):
    height, width = gray.shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    curve = np.rint(
        projection
    ).astype(np.int32)

    cv2.polylines(
        mask,
        [curve],
        False,
        255,
        thickness=13,
    )

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=400,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )

    if corners is None:
        return np.empty(
            (0, 2),
            dtype=np.float32,
        )

    points = corners.reshape(
        -1, 2
    )

    edge_distance = (
        make_edge_distance(gray)
    )

    near_edge = (
        sample_distance_map(
            edge_distance,
            points,
        )
        <= SEED_EDGE_DISTANCE_PX
    )

    near_projection = (
        point_to_polyline_distance(
            points,
            projection,
        )
        <= SEED_PROJECTION_DISTANCE_PX
    )

    return points[
        near_edge
        & near_projection
    ].astype(np.float32)


def spatial_spread(
    points,
    projection,
):
    if len(points) < 2:
        return 0.0

    # Spread is measured along the projected
    # polyline's approximate principal axis.
    centered = (
        projection
        - np.mean(
            projection,
            axis=0,
        )
    )

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    axis = vh[0]

    projected = (
        projection @ axis
    )

    tracked = (
        points @ axis
    )

    extent = (
        np.max(projected)
        - np.min(projected)
    )

    if extent <= 1e-9:
        return 0.0

    return float(
        (
            np.max(tracked)
            - np.min(tracked)
        )
        / extent
    )


def optical_flow(
    gray_a,
    gray_b,
    seeds,
):
    start = seeds.reshape(
        -1, 1, 2
    )

    forward, status_f, _ = (
        cv2.calcOpticalFlowPyrLK(
            gray_a,
            gray_b,
            start,
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

    p0 = start.reshape(-1, 2)
    p1 = forward.reshape(-1, 2)
    pback = backward.reshape(-1, 2)

    errors = np.linalg.norm(
        pback - p0,
        axis=1,
    )

    valid = (
        (status_f.reshape(-1) == 1)
        & (status_b.reshape(-1) == 1)
        & np.isfinite(errors)
        & np.all(
            np.isfinite(p1),
            axis=1,
        )
        & (
            errors
            <= FORWARD_BACKWARD_TOLERANCE_PX
        )
    )

    return (
        p0[valid],
        p1[valid],
        errors[valid],
    )


def compare_controls(
    tracked_points,
    projection,
):
    normals = polyline_normals(
        projection
    )

    real_distance = float(
        np.median(
            point_to_polyline_distance(
                tracked_points,
                projection,
            )
        )
    )

    controls = {}

    for offset in CONTROL_OFFSETS_PX:
        shifted = (
            projection
            + offset * normals
        )

        controls[str(offset)] = float(
            np.median(
                point_to_polyline_distance(
                    tracked_points,
                    shifted,
                )
            )
        )

    best_control = min(
        controls.values()
    )

    return {
        "median_real_distance_px": (
            real_distance
        ),
        "control_distances_px": controls,
        "best_control_distance_px": (
            best_control
        ),
        "real_beats_controls": bool(
            real_distance
            + CONTROL_MARGIN_PX
            < best_control
        ),
    }


def evaluate_pair(
    line,
    pose_a,
    pose_b,
    gray_a,
    gray_b,
):
    if gray_a.shape != gray_b.shape:
        return {
            "status": "IMAGE_SHAPE_MISMATCH"
        }

    height, width = gray_a.shape

    projection_a = project_line(
        line["endpoints"],
        pose_a,
        width,
        height,
    )

    projection_b = project_line(
        line["endpoints"],
        pose_b,
        width,
        height,
    )

    if (
        projection_a is None
        or projection_b is None
    ):
        return {
            "status": "PROJECTION_UNAVAILABLE"
        }

    seeds = seed_corners(
        gray_a,
        projection_a,
    )

    if len(seeds) < MIN_SEED_POINTS:
        return {
            "status": "INSUFFICIENT_SEEDS",
            "seed_count": int(
                len(seeds)
            ),
        }

    seed_spread = spatial_spread(
        seeds,
        projection_a,
    )

    if (
        seed_spread
        < MIN_SPREAD_FRACTION
    ):
        return {
            "status": "INSUFFICIENT_SEED_SPREAD",
            "seed_count": int(
                len(seeds)
            ),
            "seed_spread_fraction": (
                seed_spread
            ),
        }

    tracked = optical_flow(
        gray_a,
        gray_b,
        seeds,
    )

    if tracked is None:
        return {
            "status": "OPTICAL_FLOW_UNAVAILABLE",
            "seed_count": int(
                len(seeds)
            ),
        }

    p0, p1, fb = tracked

    tracked_count = len(p1)

    tracked_fraction = (
        tracked_count
        / len(seeds)
    )

    if (
        tracked_count < MIN_TRACKED_POINTS
        or tracked_fraction
        < MIN_TRACKED_FRACTION
    ):
        return {
            "status": "TRACKING_FAILED",
            "seed_count": int(
                len(seeds)
            ),
            "tracked_count": int(
                tracked_count
            ),
            "tracked_fraction": float(
                tracked_fraction
            ),
        }

    tracked_spread = spatial_spread(
        p0,
        projection_a,
    )

    if (
        tracked_spread
        < MIN_SPREAD_FRACTION
    ):
        return {
            "status": "INSUFFICIENT_TRACK_SPREAD",
            "seed_count": int(
                len(seeds)
            ),
            "tracked_count": int(
                tracked_count
            ),
            "tracked_spread_fraction": (
                tracked_spread
            ),
        }

    # Fisheye-aware comparison:
    # no straight-line fit is imposed.
    distances = (
        point_to_polyline_distance(
            p1,
            projection_b,
        )
    )

    median_distance = float(
        np.median(distances)
    )

    target_edge_distance = (
        make_edge_distance(gray_b)
    )

    edge_fraction = float(
        np.mean(
            sample_distance_map(
                target_edge_distance,
                p1,
            )
            <= TARGET_EDGE_DISTANCE_PX
        )
    )

    controls = compare_controls(
        p1,
        projection_b,
    )

    if (
        median_distance
        <= TARGET_PROJECTION_DISTANCE_PX
        and edge_fraction
        >= MIN_TARGET_EDGE_FRACTION
        and controls[
            "real_beats_controls"
        ]
    ):
        status = (
            "TRACK_AND_PROJECTION_AGREE"
        )

    elif (
        median_distance
        > TARGET_PROJECTION_DISTANCE_PX
    ):
        status = (
            "TRACK_DISAGREES_WITH_PROJECTION"
        )

    else:
        status = (
            "TRACK_IDENTITY_AMBIGUOUS"
        )

    return {
        "status": status,
        "seed_count": int(
            len(seeds)
        ),
        "tracked_count": int(
            tracked_count
        ),
        "tracked_fraction": float(
            tracked_fraction
        ),
        "seed_spread_fraction": (
            seed_spread
        ),
        "tracked_spread_fraction": (
            tracked_spread
        ),
        "median_forward_backward_error_px": float(
            np.median(fb)
        ),
        "median_target_projection_distance_px": (
            median_distance
        ),
        "target_edge_support_fraction": (
            edge_fraction
        ),
        **controls,
    }


def choose_seed_frames(
    validation_item,
):
    rows = validation_item.get(
        "frames", []
    )

    if not rows:
        return []

    # Prioritize frames with stronger existing
    # image evidence, but retain a deterministic
    # distribution through the candidate's
    # evaluated frames.
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["real_edge_support"],
            row["frame"],
        ),
    )

    selected = ranked[
        :MAX_SEED_FRAMES_PER_CANDIDATE
    ]

    return [
        row["frame"]
        for row in selected
    ]


def select_neighbor_pairs(
    seed_frames,
    manifest,
):
    lookup = {
        pose["frame"]: index
        for index, pose
        in enumerate(manifest)
    }

    pairs = []
    seen = set()

    for seed in seed_frames:
        index = lookup.get(seed)

        if index is None:
            continue

        source = manifest[index]
        source_number = frame_number(
            source["frame"]
        )

        if source_number is None:
            continue

        # Examine actual neighboring manifest
        # entries in both temporal directions.
        for delta in (
            -1, 1,
            -2, 2,
        ):
            if len(
                [
                    pair
                    for pair in pairs
                    if pair[0]["frame"] == seed
                ]
            ) >= MAX_NEIGHBORS_PER_SEED:
                break

            neighbor_index = (
                index + delta
            )

            if not (
                0 <= neighbor_index
                < len(manifest)
            ):
                continue

            target = manifest[
                neighbor_index
            ]

            target_number = frame_number(
                target["frame"]
            )

            if target_number is None:
                continue

            gap = abs(
                target_number
                - source_number
            )

            if not (
                0 < gap
                <= MAX_FRAME_NUMBER_GAP
            ):
                continue

            key = (
                source["frame"],
                target["frame"],
            )

            if key in seen:
                continue

            seen.add(key)

            pairs.append((
                source,
                target,
            ))

    return pairs


def load_gray(
    filename,
    cache,
):
    if filename not in cache:
        path = FRAME_DIR / filename

        cache[filename] = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

    return cache[filename]


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()
    lines = load_lines()

    validation = load_json(
        VALIDATION_PATH
    ).get("results", [])

    if not validation:
        raise RuntimeError(
            "No Step 96 validation results."
        )

    image_cache = {}

    candidate_results = []
    pair_status_counts = Counter()
    candidate_decisions = Counter()

    total_pairs = 0
    evaluated_tracking_pairs = 0

    for item in validation:
        candidate_id = int(
            item["candidate_id"]
        )

        if candidate_id not in lines:
            raise RuntimeError(
                f"Missing candidate {candidate_id}"
            )

        seed_frames = (
            choose_seed_frames(item)
        )

        pairs = select_neighbor_pairs(
            seed_frames,
            manifest,
        )

        pair_results = []

        for pose_a, pose_b in pairs:
            frame_a = pose_a["frame"]
            frame_b = pose_b["frame"]

            gray_a = load_gray(
                frame_a,
                image_cache,
            )

            gray_b = load_gray(
                frame_b,
                image_cache,
            )

            if (
                gray_a is None
                or gray_b is None
            ):
                outcome = {
                    "status": "IMAGE_UNAVAILABLE"
                }

            else:
                outcome = evaluate_pair(
                    lines[candidate_id],
                    pose_a,
                    pose_b,
                    gray_a,
                    gray_b,
                )

            outcome["frame_a"] = frame_a
            outcome["frame_b"] = frame_b

            pair_results.append(
                outcome
            )

            pair_status_counts[
                outcome["status"]
            ] += 1

            if outcome["status"] in {
                "TRACK_AND_PROJECTION_AGREE",
                "TRACK_DISAGREES_WITH_PROJECTION",
                "TRACK_IDENTITY_AMBIGUOUS",
            }:
                evaluated_tracking_pairs += 1

        counts = Counter(
            row["status"]
            for row in pair_results
        )

        agree = counts[
            "TRACK_AND_PROJECTION_AGREE"
        ]

        disagree = counts[
            "TRACK_DISAGREES_WITH_PROJECTION"
        ]

        ambiguous = counts[
            "TRACK_IDENTITY_AMBIGUOUS"
        ]

        tracked = (
            agree + disagree + ambiguous
        )

        if tracked == 0:
            decision = (
                "INSUFFICIENT_TRACKING_EVIDENCE"
            )

        elif disagree > agree:
            decision = (
                "TEMPORAL_PROJECTION_DISAGREEMENT"
            )

        elif (
            agree >= 3
            and disagree == 0
        ):
            decision = (
                "LOCAL_TEMPORAL_SUPPORT_OBSERVED"
            )

        else:
            decision = (
                "TEMPORAL_IDENTITY_UNRESOLVED"
            )

        candidate_decisions[
            decision
        ] += 1

        total_pairs += len(pairs)

        candidate_results.append({
            "candidate_id": candidate_id,
            "group_id": lines[
                candidate_id
            ]["group_id"],
            "length_m": lines[
                candidate_id
            ]["length_m"],
            "step96_status": item[
                "validation_status"
            ],
            "seed_frame_count": len(
                seed_frames
            ),
            "neighbor_pairs_selected": len(
                pairs
            ),
            "pair_status_counts": dict(
                counts
            ),
            "tracked_pairs": tracked,
            "agreement_pairs": agree,
            "disagreement_pairs": disagree,
            "ambiguous_pairs": ambiguous,
            "decision": decision,
            "physical_line_verified": False,
            "pairs": pair_results,
        })

        print(
            f"Candidate {candidate_id:03d}: "
            f"pairs={len(pairs)} "
            f"tracked={tracked} "
            f"agree={agree} "
            f"disagree={disagree} "
            f"=> {decision}"
        )

    detail_path = (
        OUTPUT / "per_candidate_tracking.json"
    )

    save_json(
        detail_path,
        {
            "method": (
                "step98b_neighbor_frame_"
                "optical_flow_diagnostic"
            ),
            "results": candidate_results,
        },
    )

    report = {
        "method": (
            "frozen_line_neighbor_frame_"
            "temporal_tracking"
        ),
        "candidate_count": len(
            candidate_results
        ),
        "step96_pass_count_unchanged": sum(
            item["validation_status"]
            == "IMAGE_SUPPORT_PASSES_CONTROLS"
            for item in validation
        ),
        "neighbor_pairs_selected": (
            total_pairs
        ),
        "pairs_reaching_tracking_comparison": (
            evaluated_tracking_pairs
        ),
        "pair_status_counts": dict(
            pair_status_counts
        ),
        "candidate_decision_counts": dict(
            candidate_decisions
        ),
        "local_temporal_support_candidate_ids": [
            row["candidate_id"]
            for row in candidate_results
            if row["decision"]
            == "LOCAL_TEMPORAL_SUPPORT_OBSERVED"
        ],
        "temporal_disagreement_candidate_ids": [
            row["candidate_id"]
            for row in candidate_results
            if row["decision"]
            == "TEMPORAL_PROJECTION_DISAGREEMENT"
        ],
        "parameters": {
            "max_seed_frames_per_candidate": (
                MAX_SEED_FRAMES_PER_CANDIDATE
            ),
            "max_neighbors_per_seed": (
                MAX_NEIGHBORS_PER_SEED
            ),
            "max_frame_number_gap": (
                MAX_FRAME_NUMBER_GAP
            ),
            "minimum_seed_points": (
                MIN_SEED_POINTS
            ),
            "minimum_tracked_points": (
                MIN_TRACKED_POINTS
            ),
            "minimum_tracked_fraction": (
                MIN_TRACKED_FRACTION
            ),
            "minimum_spread_fraction": (
                MIN_SPREAD_FRACTION
            ),
            "forward_backward_tolerance_px": (
                FORWARD_BACKWARD_TOLERANCE_PX
            ),
            "target_projection_distance_px": (
                TARGET_PROJECTION_DISTANCE_PX
            ),
            "target_edge_distance_px": (
                TARGET_EDGE_DISTANCE_PX
            ),
            "control_offsets_px": list(
                CONTROL_OFFSETS_PX
            ),
        },
        "physical_lines_verified": False,
        "structural_identity_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Additional neighboring frames are temporal diagnostic data, not an independent held-out validation set.",
            "Seeds are selected near frozen projections and are not independently associated with physical objects.",
            "Local optical-flow consistency does not prove unique physical identity.",
            "Nearby repetitive edges can produce false correspondences.",
            "Occlusion and moving workers are not explicitly modeled.",
            "Frame-to-frame tests are temporally correlated.",
            "Some lines may lack detectable corner features.",
            "The test does not optimize geometry or camera calibration.",
            "No CAD information or construction-progress inference is used.",
        ],
        "decision": (
            "TEMPORAL_DIAGNOSTICS_AVAILABLE"
            if total_pairs > 0
            else "NO_NEIGHBOR_PAIRS_AVAILABLE"
        ),
    }

    report_path = (
        OUTPUT / "tracking_report.json"
    )

    save_json(
        report_path,
        report,
    )

    print(
        "\n=== STEP 98B RESULTS ==="
    )
    print(
        "Candidates:",
        len(candidate_results)
    )
    print(
        "Neighbor pairs selected:",
        total_pairs
    )
    print(
        "Pairs reaching tracking comparison:",
        evaluated_tracking_pairs
    )
    print(
        "Pair outcomes:",
        dict(pair_status_counts)
    )
    print(
        "Candidate decisions:",
        dict(candidate_decisions)
    )
    print(
        "Step 96 passes unchanged:",
        report[
            "step96_pass_count_unchanged"
        ]
    )
    print(
        "Physical identity verified: NO"
    )
    print(
        "CAD registration verified: NO"
    )
    print(
        "Report:",
        report_path
    )
    print(
        "Details:",
        detail_path
    )


if __name__ == "__main__":
    main()
