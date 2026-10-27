
"""
Under-construction — Step 90
Independent reconstructed-plane vs CAD-wall alignment test.

Inputs:
    outputs/structural_planes/plane_candidates.json
    outputs/multiview_tracks/tracks.json
    outputs/automatic_registration_v2/registration_candidates.json
    data/floorplans/floor_2.dxf

Reuses the CAD parsing and transform functions in:
    tools/evaluate_cad_image_registration.py

Output:
    outputs/plane_cad_alignment/alignment_report.json

No image-edge scoring, AI models, manual annotations,
or construction progress inference.

Run:
    python tools/evaluate_plane_cad_alignment.py
"""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

PLANES_PATH = (
    ROOT / "outputs" / "structural_planes"
    / "plane_candidates.json"
)

TRACKS_PATH = (
    ROOT / "outputs" / "multiview_tracks"
    / "tracks.json"
)

STEP80_PATH = (
    ROOT / "tools"
    / "evaluate_cad_image_registration.py"
)

OUTPUT_DIR = (
    ROOT / "outputs"
    / "plane_cad_alignment"
)

OUTPUT_PATH = OUTPUT_DIR / "alignment_report.json"

TARGET_PLANE_ID = "plane_03"

# Distances are in metres.
DISTANCE_THRESHOLD_M = 0.30

# Difference between the horizontal normals of
# reconstructed and CAD wall planes.
ORIENTATION_THRESHOLD_DEG = 12.0

MIN_MATCHED_POINTS = 8
MIN_MATCHED_FRACTION = 0.50

# Minimum score advantage required over every
# deliberately incorrect transform.
MIN_CONTROL_ADVANTAGE = 0.15

# Even a strong result is single-plane evidence,
# not full registration verification.
RANDOM_SEED = 90


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_step80():
    if not STEP80_PATH.exists():
        raise FileNotFoundError(STEP80_PATH)

    spec = importlib.util.spec_from_file_location(
        "step80_registration",
        STEP80_PATH,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    required = (
        "load_candidates",
        "load_wall_segments",
        "cad_to_map",
    )

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]

    if missing:
        raise RuntimeError(
            "Step 80 is missing functions: "
            + ", ".join(missing)
        )

    return module


def normalize(vector):
    vector = np.asarray(
        vector, dtype=np.float64
    )

    magnitude = float(
        np.linalg.norm(vector)
    )

    if magnitude < 1e-12:
        return None

    return vector / magnitude


def get_vertical_plane(planes_data):
    matches = [
        plane
        for plane in planes_data["planes"]
        if plane["plane_id"] == TARGET_PLANE_ID
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {TARGET_PLANE_ID}; "
            f"found {len(matches)}"
        )

    plane = matches[0]

    if plane["orientation"] != "VERTICAL_CANDIDATE":
        raise RuntimeError(
            "Target is not classified as vertical."
        )

    return plane


def get_plane_points(plane, tracks_data):
    lookup = {
        int(track["track_id"]): track
        for track in tracks_data["tracks"]
    }

    ids = [
        int(value)
        for value in plane["track_ids"]
    ]

    missing = [
        value for value in ids
        if value not in lookup
    ]

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} track IDs."
        )

    points = np.asarray(
        [lookup[value]["point"] for value in ids],
        dtype=np.float64,
    )

    if not np.all(np.isfinite(points)):
        raise RuntimeError(
            "Nonfinite plane points."
        )

    return points


def normalize_wall_segments(raw_segments):
    """
    Step 80 wall segments are expected to be
    iterable pairs of XY endpoints.
    """

    result = []

    for index, segment in enumerate(raw_segments):
        if isinstance(segment, dict):
            a = segment.get("a")
            b = segment.get("b")

            if a is None or b is None:
                raise RuntimeError(
                    "Unexpected dictionary wall schema."
                )
        else:
            a, b = segment

        a = np.asarray(
            a, dtype=np.float64
        ).reshape(-1)

        b = np.asarray(
            b, dtype=np.float64
        ).reshape(-1)

        if len(a) < 2 or len(b) < 2:
            continue

        a = a[:2]
        b = b[:2]

        if not (
            np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
        ):
            continue

        length = float(
            np.linalg.norm(b - a)
        )

        if length < 0.10:
            continue

        result.append({
            "segment_id": index,
            "a": a,
            "b": b,
            "length_m": length,
        })

    return result


def transform_segments(step80, segments, candidate):
    """
    Use the established Step 80 CAD-to-map
    transformation without reimplementing it.
    """

    transformed = []

    for segment in segments:
        xyz = np.array(
            [
                [segment["a"][0], segment["a"][1], 0.0],
                [segment["b"][0], segment["b"][1], 0.0],
            ],
            dtype=np.float64,
        )

        mapped = np.asarray(
            step80.cad_to_map(
                xyz,
                candidate,
            ),
            dtype=np.float64,
        )

        if mapped.shape != (2, 3):
            raise RuntimeError(
                "Unexpected CAD transform output."
            )

        transformed.append({
            "segment_id": segment["segment_id"],
            "a": mapped[0, :2],
            "b": mapped[1, :2],
            "length_m": segment["length_m"],
        })

    return transformed


def angular_difference_deg(a, b):
    """
    Undirected 2D normal comparison.
    Parallel and antiparallel normals are equivalent.
    """

    cosine = float(
        np.clip(
            abs(np.dot(a, b)),
            0.0,
            1.0,
        )
    )

    return math.degrees(
        math.acos(cosine)
    )


def segment_point_distances(points_xy, a, b):
    """
    True point-to-finite-segment distance.

    This prevents an infinite extension of a CAD
    wall from receiving credit for distant points.
    """

    direction = b - a

    length_squared = float(
        np.dot(direction, direction)
    )

    if length_squared < 1e-12:
        return np.linalg.norm(
            points_xy - a,
            axis=1,
        )

    t = (
        (points_xy - a) @ direction
    ) / length_squared

    t = np.clip(t, 0.0, 1.0)

    closest = (
        a[None, :]
        + t[:, None] * direction[None, :]
    )

    return np.linalg.norm(
        points_xy - closest,
        axis=1,
    )


def score_candidate(
    candidate,
    points,
    plane_normal,
    segments,
    step80,
):
    transformed = transform_segments(
        step80,
        segments,
        candidate,
    )

    points_xy = points[:, :2]

    horizontal_normal = normalize(
        plane_normal[:2]
    )

    if horizontal_normal is None:
        raise RuntimeError(
            "Vertical plane has no XY normal."
        )

    # Each CAD segment is evaluated independently.
    # We do not combine unrelated walls to
    # artificially increase support.
    segment_results = []

    for segment in transformed:
        direction = normalize(
            segment["b"] - segment["a"]
        )

        if direction is None:
            continue

        wall_normal = np.array(
            [-direction[1], direction[0]],
            dtype=np.float64,
        )

        angle_error = angular_difference_deg(
            horizontal_normal,
            wall_normal,
        )

        if angle_error > ORIENTATION_THRESHOLD_DEG:
            continue

        distances = segment_point_distances(
            points_xy,
            segment["a"],
            segment["b"],
        )

        matched = (
            distances <= DISTANCE_THRESHOLD_M
        )

        matched_count = int(
            np.count_nonzero(matched)
        )

        matched_fraction = (
            matched_count / len(points_xy)
        )

        if matched_count:
            median_distance = float(
                np.median(
                    distances[matched]
                )
            )
        else:
            median_distance = None

        # Reward both fraction and distance quality.
        # The orientation term is a mild preference,
        # not an independent source of correspondence.
        if matched_count:
            distance_quality = float(
                np.mean(
                    1.0
                    - distances[matched]
                    / DISTANCE_THRESHOLD_M
                )
            )
        else:
            distance_quality = 0.0

        orientation_quality = (
            1.0
            - angle_error
            / ORIENTATION_THRESHOLD_DEG
        )

        score = (
            matched_fraction
            * (
                0.8 * distance_quality
                + 0.2 * orientation_quality
            )
        )

        segment_results.append({
            "segment_id": segment["segment_id"],
            "matched_points": matched_count,
            "matched_fraction": matched_fraction,
            "median_matched_distance_m": (
                median_distance
            ),
            "orientation_error_deg": (
                angle_error
            ),
            "score": float(score),
            "segment_length_m": (
                segment["length_m"]
            ),
            "segment_map_xy": [
                segment["a"].tolist(),
                segment["b"].tolist(),
            ],
        })

    segment_results.sort(
        key=lambda item: (
            item["score"],
            item["matched_points"],
            -item["orientation_error_deg"],
        ),
        reverse=True,
    )

    best = (
        segment_results[0]
        if segment_results
        else None
    )

    # Report multiple plausible matches:
    # single-wall ambiguity must remain visible.
    if best is not None:
        near_best = [
            item
            for item in segment_results
            if (
                item["score"]
                >= best["score"] - 0.05
                and item["matched_points"] >= 3
            )
        ]
    else:
        near_best = []

    return {
        "candidate_rank": candidate.get("rank"),
        "angle_deg": float(
            candidate["angle_deg"]
        ),
        "tx": float(candidate["tx"]),
        "ty": float(candidate["ty"]),
        "score": (
            best["score"]
            if best is not None
            else 0.0
        ),
        "best_segment": best,
        "near_best_segment_count": len(
            near_best
        ),
        "near_best_segments": near_best[:10],
        "orientation_compatible_segments": len(
            segment_results
        ),
    }


def make_controls(candidate):
    """
    Controls deliberately perturb the proposed
    CAD-to-map registration.

    The control uses the same transform schema as
    the original candidate. All perturbations are
    applied to CAD-frame transform parameters.
    """

    controls = []

    for delta in (45.0, 90.0, 135.0):
        modified = dict(candidate)
        modified["angle_deg"] = (
            float(candidate["angle_deg"]) + delta
        ) % 360.0

        controls.append((
            f"rotation_{int(delta)}",
            modified,
        ))

    for dx, dy in (
        (5.0, 0.0),
        (0.0, 5.0),
        (-5.0, -5.0),
    ):
        modified = dict(candidate)
        modified["tx"] = (
            float(candidate["tx"]) + dx
        )
        modified["ty"] = (
            float(candidate["ty"]) + dy
        )

        controls.append((
            f"translation_{dx:g}_{dy:g}",
            modified,
        ))

    return controls


def evaluate_controls(
    candidate,
    observed_score,
    points,
    plane_normal,
    segments,
    step80,
):
    results = []

    for name, modified in make_controls(candidate):
        result = score_candidate(
            modified,
            points,
            plane_normal,
            segments,
            step80,
        )

        control_score = result["score"]

        results.append({
            "name": name,
            "score": control_score,
            "observed_advantage": float(
                observed_score - control_score
            ),
            "control_outperforms_observed": (
                control_score >= observed_score
            ),
            "best_segment_id": (
                result["best_segment"]["segment_id"]
                if result["best_segment"]
                else None
            ),
        })

    return results


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    planes_data = load_json(
        PLANES_PATH
    )

    tracks_data = load_json(
        TRACKS_PATH
    )

    step80 = load_step80()

    plane = get_vertical_plane(
        planes_data
    )

    points = get_plane_points(
        plane,
        tracks_data,
    )

    plane_normal = np.asarray(
        plane["normal"],
        dtype=np.float64,
    )

    candidates = step80.load_candidates()

    segments = normalize_wall_segments(
        step80.load_wall_segments()
    )

    if not candidates:
        raise RuntimeError(
            "No CAD registration candidates."
        )

    if not segments:
        raise RuntimeError(
            "No CAD wall segments."
        )

    print("\n=== STEP 90: PLANE ↔ CAD TEST ===")
    print("Plane:", TARGET_PLANE_ID)
    print("Observed 3D landmarks:", len(points))
    print("CAD wall segments:", len(segments))
    print("Registration candidates:", len(candidates))

    evaluations = []

    for candidate in candidates:
        result = score_candidate(
            candidate,
            points,
            plane_normal,
            segments,
            step80,
        )

        controls = evaluate_controls(
            candidate,
            result["score"],
            points,
            plane_normal,
            segments,
            step80,
        )

        advantages = [
            item["observed_advantage"]
            for item in controls
        ]

        result["negative_controls"] = controls

        result["passes_absolute_support"] = bool(
            result["best_segment"] is not None
            and result["best_segment"]["matched_points"]
            >= MIN_MATCHED_POINTS
            and result["best_segment"]["matched_fraction"]
            >= MIN_MATCHED_FRACTION
        )

        result["passes_all_controls"] = bool(
            advantages
            and min(advantages)
            >= MIN_CONTROL_ADVANTAGE
        )

        result["single_plane_evidence_supported"] = bool(
            result["passes_absolute_support"]
            and result["passes_all_controls"]
        )

        evaluations.append(result)

        print(
            "Candidate",
            result["candidate_rank"],
            "| score",
            round(result["score"], 4),
            "| matched",
            (
                result["best_segment"]["matched_points"]
                if result["best_segment"]
                else 0
            ),
            "| controls",
            (
                "PASS"
                if result["passes_all_controls"]
                else "FAIL"
            ),
        )

    evaluations.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    supported = [
        item
        for item in evaluations
        if item["single_plane_evidence_supported"]
    ]

    report = {
        "sequence": planes_data.get("sequence"),
        "method": (
            "reconstructed_vertical_plane_to_finite_cad_wall"
        ),
        "source_plane_id": TARGET_PLANE_ID,
        "source_plane_orientation": plane["orientation"],
        "source_plane_point_count": len(points),
        "source_plane_normal": plane["normal"],
        "source_plane_centroid": plane["centroid"],
        "cad_wall_segment_count": len(segments),
        "candidate_count": len(candidates),
        "candidate_evaluations": evaluations,
        "single_plane_supported_candidate_ranks": [
            item["candidate_rank"]
            for item in supported
        ],
        "single_plane_evidence_gate_passed": bool(
            supported
        ),
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "distance_threshold_m": DISTANCE_THRESHOLD_M,
            "orientation_threshold_deg": (
                ORIENTATION_THRESHOLD_DEG
            ),
            "min_matched_points": MIN_MATCHED_POINTS,
            "min_matched_fraction": MIN_MATCHED_FRACTION,
            "min_control_advantage": (
                MIN_CONTROL_ADVANTAGE
            ),
        },
        "decision": (
            "SINGLE_PLANE_CAD_SUPPORT_ONLY"
            if supported
            else "SINGLE_PLANE_CAD_ALIGNMENT_UNVERIFIED"
        ),
        "limitations": [
            "Only one reconstructed vertical plane is evaluated.",
            "One wall cannot uniquely constrain a full 2D registration.",
            "A CAD segment match is not a verified physical wall identity.",
            "RANSAC plane points may include unrelated coplanar surfaces.",
            "Each transform searches many CAD segments, allowing chance matches.",
            "Negative controls are perturbations, not an exhaustive null distribution.",
            "The thresholds are exploratory and not calibrated on held-out sites.",
            "The source plane has not been verified as a continuous visible surface.",
            "No construction progress or visibility claim is established.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULT ===")
    print("Decision:", report["decision"])
    print("Registration verified: NO")
    print("Report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
