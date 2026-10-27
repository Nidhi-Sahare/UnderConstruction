
"""
Under-construction — Step 95

Resolve competing 3D line hypotheses with
exclusive observation ownership.

Inputs:
    outputs/multiview_lines/line_candidates.json
    outputs/line_consolidation/line_groups.json

Outputs:
    outputs/line_ambiguity/resolution_report.json
    outputs/line_ambiguity/resolved_lines.json

This is NOT independent geometric verification.
No CAD data, AI models or manual annotations.

Run:
    python tools/resolve_line_correspondence_ambiguity.py
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_PATH = (
    ROOT / "outputs"
    / "multiview_lines"
    / "line_candidates.json"
)

GROUPS_PATH = (
    ROOT / "outputs"
    / "line_consolidation"
    / "line_groups.json"
)

OUTPUT = ROOT / "outputs" / "line_ambiguity"

MIN_FRAMES = 3
MIN_LENGTH_M = 0.25
MIN_INTERSECTION_ANGLE_DEG = 3.0

VERTICAL_Z_THRESHOLD = 0.95
HORIZONTAL_Z_THRESHOLD = 0.30

# Score sensitivity is a diagnostic, not a
# statistically valid uncertainty interval.
SENSITIVITY_WEIGHTS = [
    (1.0, 1.0, 1.0),
    (1.5, 0.7, 1.0),
    (0.7, 1.5, 1.0),
    (1.0, 1.0, 1.5),
    (1.0, 1.0, 0.7),
]

# A group can contribute only one selected
# hypothesis, preventing duplicate candidates
# from consuming different observations.
MAX_SELECTED_PER_GROUP = 1

EPS = 1e-10


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}"
        )

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    return result if math.isfinite(result) else default


def observation_claims(item):
    """
    Build observation/frame associations.

    A source observation must have a source
    frame. Held-out matches must include both
    observation_id and frame.
    """
    claims = {}

    source_ids = item.get(
        "source_observations", []
    )
    source_frames = item.get(
        "source_frames", []
    )

    if len(source_ids) != len(source_frames):
        return None

    for obs_id, frame in zip(
        source_ids, source_frames
    ):
        if not obs_id or not frame:
            return None

        if (
            obs_id in claims
            and claims[obs_id] != frame
        ):
            return None

        claims[obs_id] = frame

    for match in item.get(
        "heldout_matches", []
    ):
        obs_id = match.get("observation_id")
        frame = match.get("frame")

        if not obs_id or not frame:
            continue

        if (
            obs_id in claims
            and claims[obs_id] != frame
        ):
            return None

        claims[obs_id] = frame

    # A physical line can be observed at most
    # once per frame in this conservative model.
    if len(set(claims.values())) != len(claims):
        return None

    return claims


def build_candidates(raw, group_map):
    result = []
    rejection_counts = Counter()

    for candidate_id, item in enumerate(
        raw.get("candidates", [])
    ):
        endpoints = np.asarray(
            item.get("endpoints_map", []),
            dtype=float,
        )

        if (
            endpoints.shape != (2, 3)
            or not np.all(np.isfinite(endpoints))
        ):
            rejection_counts[
                "invalid_geometry"
            ] += 1
            continue

        displacement = (
            endpoints[1] - endpoints[0]
        )

        length = float(
            np.linalg.norm(displacement)
        )

        if length < MIN_LENGTH_M:
            rejection_counts[
                "insufficient_length"
            ] += 1
            continue

        direction = displacement / length

        angle = finite_float(
            item.get(
                "intersection_angle_deg"
            )
        )

        if angle < MIN_INTERSECTION_ANGLE_DEG:
            rejection_counts[
                "insufficient_intersection_angle"
            ] += 1
            continue

        claims = observation_claims(item)

        if claims is None:
            rejection_counts[
                "invalid_observation_claims"
            ] += 1
            continue

        if len(claims) < MIN_FRAMES:
            rejection_counts[
                "insufficient_frames"
            ] += 1
            continue

        if candidate_id not in group_map:
            rejection_counts[
                "missing_consolidation_group"
            ] += 1
            continue

        errors = [
            finite_float(
                m.get("reprojection_error_px"),
                default=1e6,
            )
            for m in item.get(
                "heldout_matches", []
            )
        ]

        similarities = [
            finite_float(
                m.get("appearance_similarity"),
                default=-1.0,
            )
            for m in item.get(
                "heldout_matches", []
            )
        ]

        median_error = (
            float(np.median(errors))
            if errors
            else 1e6
        )

        median_similarity = (
            float(np.median(similarities))
            if similarities
            else -1.0
        )

        result.append({
            "id": candidate_id,
            "group": group_map[candidate_id],
            "endpoints": endpoints,
            "direction": direction,
            "length": length,
            "claims": claims,
            "frames": set(claims.values()),
            "angle": angle,
            "baseline": finite_float(
                item.get("baseline_m")
            ),
            "median_error": median_error,
            "median_similarity": median_similarity,
            "source_similarity": finite_float(
                item.get(
                    "source_appearance_similarity"
                ),
                default=-1.0,
            ),
        })

    return result, dict(rejection_counts)


def make_group_map(data):
    mapping = {}

    for group in data.get("groups", []):
        group_id = group["group_id"]

        for candidate_id in group.get(
            "member_candidate_ids", []
        ):
            if candidate_id in mapping:
                raise ValueError(
                    "Candidate appears in multiple "
                    f"groups: {candidate_id}"
                )

            mapping[candidate_id] = group_id

    return mapping


def score_components(candidate):
    """
    Conservative, bounded ranking features.

    More supporting frames help, but are not
    treated as independent evidence.
    """
    support = min(
        len(candidate["frames"]), 10
    ) / 10.0

    error = max(
        0.0,
        1.0 - candidate["median_error"] / 5.0
    )

    appearance = max(
        0.0,
        min(
            1.0,
            (
                candidate["median_similarity"]
                + 1.0
            ) / 2.0,
        ),
    )

    geometry = (
        min(candidate["length"], 3.0) / 3.0
        + min(candidate["angle"], 30.0) / 30.0
    ) / 2.0

    return {
        "support": support,
        "reprojection": error,
        "appearance": appearance,
        "geometry": geometry,
    }


def candidate_score(candidate, weights):
    support_weight, error_weight, geometry_weight = (
        weights
    )

    components = score_components(candidate)

    # Positive scores ensure the solver does
    # not prefer an empty solution.
    return (
        0.05
        + support_weight * components["support"]
        + error_weight * components["reprojection"]
        + 0.5 * components["appearance"]
        + geometry_weight * components["geometry"]
    )


def build_constraints(candidates):
    """
    Each observation belongs to at most one
    selected candidate.

    Each consolidated group contributes at
    most one selected candidate.
    """
    by_observation = defaultdict(list)
    by_group = defaultdict(list)

    for index, candidate in enumerate(candidates):
        for observation in candidate["claims"]:
            by_observation[observation].append(
                index
            )

        by_group[candidate["group"]].append(
            index
        )

    constraints = []

    for indices in by_observation.values():
        if len(indices) > 1:
            constraints.append(indices)

    for indices in by_group.values():
        if len(indices) > 1:
            constraints.append(indices)

    return (
        constraints,
        by_observation,
        by_group,
    )


def solve_greedy(candidates, weights):
    """
    Deterministic fallback.

    A feasible solution, not guaranteed
    globally optimal.
    """
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            -candidate_score(
                candidates[i], weights
            ),
            -len(candidates[i]["frames"]),
            candidates[i]["id"],
        ),
    )

    used_observations = set()
    used_groups = set()
    selected = []

    for index in order:
        candidate = candidates[index]

        if candidate["group"] in used_groups:
            continue

        if (
            set(candidate["claims"])
            & used_observations
        ):
            continue

        selected.append(index)

        used_groups.add(candidate["group"])
        used_observations.update(
            candidate["claims"]
        )

    return selected, "deterministic_greedy"


def solve_milp(candidates, constraints, weights):
    """
    Maximum-weight set packing.

    Requires scipy.optimize.milp, available
    in modern SciPy versions.
    """
    from scipy.optimize import (
        Bounds,
        LinearConstraint,
        milp,
    )
    from scipy.sparse import lil_matrix

    n = len(candidates)

    if n == 0:
        return [], "milp"

    matrix = lil_matrix(
        (len(constraints), n),
        dtype=float,
    )

    for row, indices in enumerate(constraints):
        for index in indices:
            matrix[row, index] = 1.0

    scores = np.array([
        candidate_score(candidate, weights)
        for candidate in candidates
    ])

    result = milp(
        c=-scores,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(
            np.zeros(n),
            np.ones(n),
        ),
        constraints=LinearConstraint(
            matrix.tocsr(),
            lb=np.zeros(len(constraints)),
            ub=np.ones(len(constraints)),
        ),
        options={
            "time_limit": 60.0,
            "mip_rel_gap": 0.001,
        },
    )

    if (
        result.x is None
        or not result.success
    ):
        return None, (
            "milp_unavailable_or_not_proven_optimal"
        )

    selected = [
        index
        for index, value in enumerate(result.x)
        if value > 0.5
    ]

    return selected, "milp_optimal_with_tolerance"


def solve(candidates, constraints, weights):
    try:
        selected, method = solve_milp(
            candidates,
            constraints,
            weights,
        )

        if selected is not None:
            return selected, method

    except (
        ImportError,
        ValueError,
        RuntimeError,
    ):
        pass

    return solve_greedy(
        candidates,
        weights,
    )


def orientation_class(direction):
    z = abs(float(direction[2]))

    if z >= VERTICAL_Z_THRESHOLD:
        return "near_vertical"

    if z <= HORIZONTAL_Z_THRESHOLD:
        return "near_horizontal"

    return "oblique"


def horizontal_diversity(selected):
    angles = []

    for candidate in selected:
        if (
            orientation_class(
                candidate["direction"]
            )
            != "near_horizontal"
        ):
            continue

        direction = candidate["direction"]

        angle = (
            math.degrees(
                math.atan2(
                    direction[1],
                    direction[0],
                )
            )
            % 180.0
        )

        angles.append({
            "candidate_id": candidate["id"],
            "azimuth_deg_mod_180": angle,
        })

    maximum = 0.0

    for i, a in enumerate(angles):
        for b in angles[i + 1:]:
            difference = abs(
                a["azimuth_deg_mod_180"]
                - b["azimuth_deg_mod_180"]
            )

            difference = min(
                difference,
                180.0 - difference,
            )

            maximum = max(
                maximum,
                difference,
            )

    return {
        "near_horizontal_selected": len(angles),
        "maximum_axial_separation_deg": maximum,
        "directions": angles,
        "structural_directions_verified": False,
    }


def serialize_candidate(candidate, weights):
    return {
        "candidate_id": candidate["id"],
        "group_id": candidate["group"],
        "endpoints_map": (
            candidate["endpoints"].tolist()
        ),
        "direction_map": (
            candidate["direction"].tolist()
        ),
        "length_m": candidate["length"],
        "supporting_frame_count": len(
            candidate["frames"]
        ),
        "observation_ids": sorted(
            candidate["claims"]
        ),
        "source_frames": sorted(
            candidate["frames"]
        ),
        "intersection_angle_deg": (
            candidate["angle"]
        ),
        "baseline_m": candidate["baseline"],
        "median_heldout_reprojection_error_px": (
            candidate["median_error"]
        ),
        "median_heldout_appearance_similarity": (
            candidate["median_similarity"]
        ),
        "selection_score": candidate_score(
            candidate, weights
        ),
        "orientation_class": orientation_class(
            candidate["direction"]
        ),
        "physical_line_verified": False,
        "cad_registration_verified": False,
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = read_json(CANDIDATES_PATH)
    group_data = read_json(GROUPS_PATH)

    group_map = make_group_map(
        group_data
    )

    candidates, rejected = build_candidates(
        raw, group_map
    )

    if not candidates:
        raise RuntimeError(
            "No valid hypotheses for resolution."
        )

    (
        constraints,
        by_observation,
        by_group,
    ) = build_constraints(candidates)

    baseline_weights = (
        SENSITIVITY_WEIGHTS[0]
    )

    selected_indices, method = solve(
        candidates,
        constraints,
        baseline_weights,
    )

    selected = [
        candidates[index]
        for index in selected_indices
    ]

    selected_ids = {
        candidate["id"]
        for candidate in selected
    }

    used_observations = set()

    for candidate in selected:
        claims = set(candidate["claims"])

        if claims & used_observations:
            raise AssertionError(
                "Observation ownership violation."
            )

        used_observations.update(claims)

    if len({
        candidate["group"]
        for candidate in selected
    }) != len(selected):
        raise AssertionError(
            "Duplicate group selection."
        )

    sensitivity = []

    selection_sets = []

    for weights in SENSITIVITY_WEIGHTS:
        indices, solver_method = solve(
            candidates,
            constraints,
            weights,
        )

        ids = {
            candidates[index]["id"]
            for index in indices
        }

        selection_sets.append(ids)

        sensitivity.append({
            "weights": list(weights),
            "solver": solver_method,
            "selected_count": len(ids),
            "selected_candidate_ids": sorted(ids),
        })

    always_selected = (
        set.intersection(*selection_sets)
        if selection_sets
        else set()
    )

    sometimes_selected = (
        set.union(*selection_sets)
        if selection_sets
        else set()
    )

    baseline_stability = (
        len(always_selected & selected_ids)
        / max(len(selected_ids), 1)
    )

    orientation_counts = Counter(
        orientation_class(
            candidate["direction"]
        )
        for candidate in selected
    )

    all_claimed_observations = set(
        by_observation
    )

    contested_observations = {
        obs
        for obs, indices in (
            by_observation.items()
        )
        if len(indices) > 1
    }

    selected_contested = (
        used_observations
        & contested_observations
    )

    # Important: ownership is exclusive only
    # within the selected hypothesis set.
    # It does not establish that a selected
    # hypothesis is physically correct.
    resolved_data = {
        "method": (
            "exclusive_observation_set_packing"
        ),
        "solver": method,
        "selected_lines": [
            serialize_candidate(
                candidate,
                baseline_weights,
            )
            for candidate in sorted(
                selected,
                key=lambda c: c["id"],
            )
        ],
    }

    resolved_path = (
        OUTPUT / "resolved_lines.json"
    )

    resolved_path.write_text(
        json.dumps(
            resolved_data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    horizontal = horizontal_diversity(
        selected
    )

    report = {
        "method": (
            "exclusive_line_hypothesis_selection"
        ),
        "input_hypotheses": len(
            raw.get("candidates", [])
        ),
        "eligible_hypotheses": len(
            candidates
        ),
        "rejection_counts": rejected,
        "input_consolidation_groups": len(
            group_data.get("groups", [])
        ),
        "solver": method,
        "selected_hypotheses": len(
            selected
        ),
        "selected_group_count": len({
            c["group"] for c in selected
        }),
        "excluded_eligible_hypotheses": (
            len(candidates) - len(selected)
        ),
        "distinct_observations_in_input": len(
            all_claimed_observations
        ),
        "contested_observations_in_input": len(
            contested_observations
        ),
        "distinct_observations_selected": len(
            used_observations
        ),
        "contested_observations_used_by_selection": (
            len(selected_contested)
        ),
        "selected_observation_ownership_conflicts": 0,
        "selected_orientation_counts": dict(
            orientation_counts
        ),
        "horizontal_direction_analysis": horizontal,
        "sensitivity": {
            "runs": sensitivity,
            "always_selected_candidate_count": len(
                always_selected
            ),
            "selected_in_any_run_count": len(
                sometimes_selected
            ),
            "baseline_selection_stability_fraction": (
                baseline_stability
            ),
            "always_selected_candidate_ids": sorted(
                always_selected
            ),
        },
        "independent_physical_lines_verified": False,
        "structural_directions_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Exclusive ownership does not prove correct physical correspondence.",
            "Selection scores are heuristic, not calibrated probabilities.",
            "Step 93 searched many candidate correspondences before this step.",
            "Step 93 held-out matches are not a truly untouched validation set.",
            "Only the top 200 Step 93 hypotheses are available.",
            "Selected hypotheses may still represent duplicate physical lines across different groups.",
            "One candidate per consolidated group is an imposed constraint.",
            "The sensitivity test changes scoring weights, not underlying measurements.",
            "MILP solves the stated set-packing objective, not the physical reconstruction problem.",
            "Camera pose and calibration errors are not assessed.",
            "No CAD data is used.",
        ],
        "decision": (
            "EXCLUSIVE_CANDIDATE_SET_AVAILABLE"
            if selected
            else "NO_UNAMBIGUOUS_CANDIDATE_SET"
        ),
    }

    report_path = (
        OUTPUT / "resolution_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== STEP 95: LINE AMBIGUITY RESOLUTION ==="
    )

    print(
        "Input hypotheses:",
        report["input_hypotheses"],
    )
    print(
        "Eligible hypotheses:",
        report["eligible_hypotheses"],
    )
    print(
        "Selected hypotheses:",
        report["selected_hypotheses"],
    )
    print(
        "Solver:",
        method,
    )
    print(
        "Selected observation conflicts:",
        0,
    )
    print(
        "Near-vertical:",
        orientation_counts["near_vertical"],
    )
    print(
        "Near-horizontal:",
        orientation_counts["near_horizontal"],
    )
    print(
        "Oblique:",
        orientation_counts["oblique"],
    )
    print(
        "Selection stability:",
        round(
            baseline_stability, 3
        ),
    )
    print(
        "Physical lines verified: NO"
    )
    print(
        "CAD registration verified: NO"
    )
    print(
        "Report:",
        report_path,
    )
    print(
        "Resolved lines:",
        resolved_path,
    )


if __name__ == "__main__":
    main()
