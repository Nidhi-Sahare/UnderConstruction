
"""
Under-construction — Step 94
3D line consolidation, conflict analysis and
registration-readiness diagnostics.

Input:
    outputs/multiview_lines/line_candidates.json

Outputs:
    outputs/line_consolidation/consolidation_report.json
    outputs/line_consolidation/line_groups.json

No images, CAD data, AI models or manual labels required.

Important:
    Geometric clustering is NOT physical-line
    verification. The script does not assert that
    grouped hypotheses represent real structures.

Run:
    python tools/consolidate_multiview_lines.py
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "outputs"
    / "multiview_lines"
    / "line_candidates.json"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "line_consolidation"
)

# Two hypotheses are potentially duplicates when
# their directions, separation and finite extents
# agree. These are exploratory grouping thresholds.
MAX_DIRECTION_DIFFERENCE_DEG = 8.0
MAX_LINE_SEPARATION_M = 0.30
MIN_OVERLAP_FRACTION = 0.25

# A shared observation supporting spatially
# incompatible lines is an ambiguity warning.
CONFLICT_DISTANCE_M = 0.75
CONFLICT_DIRECTION_DEG = 15.0

VERTICAL_MIN_ABS_Z = 0.95
HORIZONTAL_MAX_ABS_Z = 0.30

MIN_USEFUL_LENGTH_M = 1.0
MIN_GROUP_SUPPORT_FRAMES = 4

# Only for spatial-coverage diagnostics.
SPATIAL_CELL_SIZE_M = 2.0

# Two horizontal directions are independent
# when their axial angle separation is sufficient.
MIN_INDEPENDENT_HORIZONTAL_ANGLE_DEG = 25.0

EPS = 1e-10


def unit(v):
    v = np.asarray(v, dtype=np.float64)

    n = float(np.linalg.norm(v))

    if not np.isfinite(n) or n < EPS:
        return None

    return v / n


def axial_angle_deg(a, b):
    """Unoriented 3D line direction difference."""
    return math.degrees(
        math.acos(
            float(
                np.clip(
                    abs(np.dot(a, b)),
                    0.0,
                    1.0,
                )
            )
        )
    )


def point_line_distance(point, origin, direction):
    delta = point - origin

    return float(
        np.linalg.norm(
            delta
            - np.dot(delta, direction) * direction
        )
    )


def segment_distance(a, b):
    """
    Minimum distance between finite 3D segments.

    Based on the constrained closest-point
    calculation for two line segments.
    """
    p0, p1 = a
    q0, q1 = b

    u = p1 - p0
    v = q1 - q0
    w = p0 - q0

    aa = float(np.dot(u, u))
    bb = float(np.dot(u, v))
    cc = float(np.dot(v, v))
    dd = float(np.dot(u, w))
    ee = float(np.dot(v, w))

    if aa < EPS or cc < EPS:
        return float("inf")

    denominator = aa * cc - bb * bb

    s_n = denominator
    s_d = denominator
    t_n = denominator
    t_d = denominator

    if denominator < EPS:
        s_n = 0.0
        s_d = 1.0
        t_n = ee
        t_d = cc
    else:
        s_n = bb * ee - cc * dd
        t_n = aa * ee - bb * dd

        if s_n < 0.0:
            s_n = 0.0
            t_n = ee
            t_d = cc
        elif s_n > s_d:
            s_n = s_d
            t_n = ee + bb
            t_d = cc

    if t_n < 0.0:
        t_n = 0.0

        if -dd < 0.0:
            s_n = 0.0
        elif -dd > aa:
            s_n = s_d
        else:
            s_n = -dd
            s_d = aa

    elif t_n > t_d:
        t_n = t_d

        if -dd + bb < 0.0:
            s_n = 0.0
        elif -dd + bb > aa:
            s_n = s_d
        else:
            s_n = -dd + bb
            s_d = aa

    sc = (
        0.0
        if abs(s_n) < EPS
        else s_n / s_d
    )

    tc = (
        0.0
        if abs(t_n) < EPS
        else t_n / t_d
    )

    difference = w + sc * u - tc * v

    return float(np.linalg.norm(difference))


def overlap_fraction(a, b):
    """
    Approximate finite-segment overlap after
    projecting both segments onto a common axis.

    The result is meaningful only when directions
    are already approximately parallel.
    """
    direction_a = unit(a[1] - a[0])
    direction_b = unit(b[1] - b[0])

    if direction_a is None or direction_b is None:
        return 0.0

    if np.dot(direction_a, direction_b) < 0:
        direction_b = -direction_b

    axis = unit(direction_a + direction_b)

    if axis is None:
        return 0.0

    origin = 0.5 * (a.mean(axis=0) + b.mean(axis=0))

    ta = sorted(((a - origin) @ axis).tolist())
    tb = sorted(((b - origin) @ axis).tolist())

    intersection = max(
        0.0,
        min(ta[1], tb[1])
        - max(ta[0], tb[0]),
    )

    length_a = ta[1] - ta[0]
    length_b = tb[1] - tb[0]

    return float(
        intersection
        / max(min(length_a, length_b), EPS)
    )


def load_candidates():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing Step 93 output: {INPUT}"
        )

    raw = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    result = []

    for index, item in enumerate(
        raw.get("candidates", [])
    ):
        try:
            endpoints = np.asarray(
                item["endpoints_map"],
                dtype=np.float64,
            )

            if endpoints.shape != (2, 3):
                continue

            if not np.all(np.isfinite(endpoints)):
                continue

            direction = unit(
                endpoints[1] - endpoints[0]
            )

            if direction is None:
                continue

            length = float(
                np.linalg.norm(
                    endpoints[1] - endpoints[0]
                )
            )

            observation_ids = set(
                item.get("source_observations", [])
            )

            observation_ids.update(
                match["observation_id"]
                for match in item.get(
                    "heldout_matches", []
                )
                if "observation_id" in match
            )

            frames = set(
                item.get("source_frames", [])
            )

            frames.update(
                match["frame"]
                for match in item.get(
                    "heldout_matches", []
                )
                if "frame" in match
            )

            result.append({
                "candidate_id": index,
                "endpoints": endpoints,
                "midpoint": endpoints.mean(axis=0),
                "direction": direction,
                "length_m": length,
                "observations": observation_ids,
                "frames": frames,
                "baseline_m": float(
                    item.get("baseline_m", 0.0)
                ),
                "intersection_angle_deg": float(
                    item.get(
                        "intersection_angle_deg",
                        0.0,
                    )
                ),
                "reported_supporting_frames": int(
                    item.get(
                        "supporting_frame_count", 0
                    )
                ),
            })

        except (
            KeyError,
            ValueError,
            TypeError,
        ):
            continue

    return result


def pair_metrics(a, b):
    angle = axial_angle_deg(
        a["direction"],
        b["direction"],
    )

    distance = segment_distance(
        a["endpoints"],
        b["endpoints"],
    )

    overlap = overlap_fraction(
        a["endpoints"],
        b["endpoints"],
    )

    return angle, distance, overlap


def duplicate_compatible(a, b):
    angle, distance, overlap = pair_metrics(a, b)

    return (
        angle <= MAX_DIRECTION_DIFFERENCE_DEG
        and distance <= MAX_LINE_SEPARATION_M
        and overlap >= MIN_OVERLAP_FRACTION
    )


def build_duplicate_groups(candidates):
    """
    Complete-link agglomeration.

    Unlike connected-component clustering,
    this requires every member of a proposed
    merged group to be duplicate-compatible
    with every member of the other group.

    This prevents a chain of near-duplicates
    from merging widely separated structures.
    """
    n = len(candidates)

    compatible = np.eye(n, dtype=bool)

    for i in range(n):
        for j in range(i + 1, n):
            yes = duplicate_compatible(
                candidates[i],
                candidates[j],
            )

            compatible[i, j] = yes
            compatible[j, i] = yes

    groups = [[i] for i in range(n)]

    while True:
        best_pair = None
        best_score = None

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                members_a = groups[i]
                members_b = groups[j]

                if not all(
                    compatible[a, b]
                    for a in members_a
                    for b in members_b
                ):
                    continue

                # Prefer merges with larger
                # shared observation support.
                shared = 0

                for a in members_a:
                    for b in members_b:
                        shared += len(
                            candidates[a]["observations"]
                            & candidates[b]["observations"]
                        )

                score = (
                    shared,
                    -(len(members_a) + len(members_b)),
                )

                if (
                    best_score is None
                    or score > best_score
                ):
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair

        groups[i] = sorted(
            groups[i] + groups[j]
        )

        del groups[j]

    groups.sort(
        key=lambda group: (
            -len(group),
            min(group),
        )
    )

    return groups


def representative_index(members, candidates):
    """
    Select an existing reconstruction instead
    of averaging incompatible endpoints.
    """
    return max(
        members,
        key=lambda index: (
            len(candidates[index]["frames"]),
            candidates[index][
                "intersection_angle_deg"
            ],
            candidates[index]["length_m"],
        ),
    )


def orientation_class(direction):
    z = abs(float(direction[2]))

    if z >= VERTICAL_MIN_ABS_Z:
        return "near_vertical"

    if z <= HORIZONTAL_MAX_ABS_Z:
        return "near_horizontal"

    return "oblique"


def xy_cell(point):
    return (
        math.floor(
            float(point[0]) / SPATIAL_CELL_SIZE_M
        ),
        math.floor(
            float(point[1]) / SPATIAL_CELL_SIZE_M
        ),
    )


def build_group_records(groups, candidates):
    records = []

    for group_number, members in enumerate(
        groups, start=1
    ):
        representative = representative_index(
            members,
            candidates,
        )

        rep = candidates[representative]

        all_frames = set()
        all_observations = set()

        for index in members:
            all_frames.update(
                candidates[index]["frames"]
            )

            all_observations.update(
                candidates[index]["observations"]
            )

        orientation = orientation_class(
            rep["direction"]
        )

        record = {
            "group_id": f"line_group_{group_number:03d}",
            "member_candidate_ids": [
                candidates[index]["candidate_id"]
                for index in members
            ],
            "hypothesis_count": len(members),
            "representative_candidate_id": (
                rep["candidate_id"]
            ),
            "representative_endpoints_map": (
                rep["endpoints"].tolist()
            ),
            "representative_midpoint_map": (
                rep["midpoint"].tolist()
            ),
            "representative_direction_map": (
                rep["direction"].tolist()
            ),
            "representative_length_m": (
                rep["length_m"]
            ),
            "orientation_class": orientation,
            "distinct_supporting_frames": len(
                all_frames
            ),
            "distinct_observations": len(
                all_observations
            ),
            "spatial_cell_xy": list(
                xy_cell(rep["midpoint"])
            ),
            "useful_length_gate": (
                rep["length_m"]
                >= MIN_USEFUL_LENGTH_M
            ),
            "frame_support_gate": (
                len(all_frames)
                >= MIN_GROUP_SUPPORT_FRAMES
            ),
            "independent_physical_line_verified": False,
            "structural_identity_verified": False,
        }

        records.append(record)

    return records


def observation_conflicts(candidates, group_of):
    """
    Find observations claimed by hypotheses
    belonging to different geometric groups.

    Shared observations across groups are
    ambiguity signals, not automatic errors:
    some detections may contain overlapping
    edges or imperfect segmentation.
    """
    claims = defaultdict(list)

    for index, candidate in enumerate(candidates):
        for observation in candidate["observations"]:
            claims[observation].append(index)

    conflicting = []
    ambiguous = []

    for observation, indices in claims.items():
        group_ids = {
            group_of[index]
            for index in indices
        }

        if len(group_ids) < 2:
            continue

        examples = []

        for i_pos, i in enumerate(indices):
            for j in indices[i_pos + 1:]:
                if group_of[i] == group_of[j]:
                    continue

                a = candidates[i]
                b = candidates[j]

                angle, distance, _ = pair_metrics(
                    a, b
                )

                if (
                    distance >= CONFLICT_DISTANCE_M
                    or angle >= CONFLICT_DIRECTION_DEG
                ):
                    examples.append({
                        "candidate_a": (
                            a["candidate_id"]
                        ),
                        "candidate_b": (
                            b["candidate_id"]
                        ),
                        "distance_m": distance,
                        "direction_difference_deg": angle,
                    })

        ambiguous.append({
            "observation_id": observation,
            "claiming_candidate_count": len(indices),
            "claiming_group_count": len(group_ids),
        })

        if examples:
            examples.sort(
                key=lambda item: (
                    item["distance_m"],
                    item["direction_difference_deg"],
                ),
                reverse=True,
            )

            conflicting.append({
                "observation_id": observation,
                "claiming_candidate_count": len(indices),
                "claiming_group_count": len(group_ids),
                "conflicting_pair_count": len(examples),
                "worst_examples": examples[:3],
            })

    ambiguous.sort(
        key=lambda item: (
            item["claiming_group_count"],
            item["claiming_candidate_count"],
        ),
        reverse=True,
    )

    conflicting.sort(
        key=lambda item: (
            item["claiming_group_count"],
            item["conflicting_pair_count"],
        ),
        reverse=True,
    )

    return claims, ambiguous, conflicting


def horizontal_direction_analysis(records):
    """
    Axial horizontal orientations.

    This reports whether the representative
    directions span distinct angles, NOT whether
    those directions are structural or correct.
    """
    usable = []

    for record in records:
        if (
            record["orientation_class"]
            != "near_horizontal"
        ):
            continue

        if not (
            record["useful_length_gate"]
            and record["frame_support_gate"]
        ):
            continue

        direction = np.asarray(
            record["representative_direction_map"],
            dtype=np.float64,
        )

        azimuth = (
            math.degrees(
                math.atan2(
                    direction[1],
                    direction[0],
                )
            )
            % 180.0
        )

        usable.append({
            "group_id": record["group_id"],
            "azimuth_deg_mod_180": azimuth,
            "length_m": (
                record["representative_length_m"]
            ),
            "spatial_cell_xy": (
                record["spatial_cell_xy"]
            ),
        })

    max_separation = 0.0
    best_pair = None

    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            difference = abs(
                a["azimuth_deg_mod_180"]
                - b["azimuth_deg_mod_180"]
            )

            difference = min(
                difference,
                180.0 - difference,
            )

            if difference > max_separation:
                max_separation = difference
                best_pair = [
                    a["group_id"],
                    b["group_id"],
                ]

    return {
        "usable_near_horizontal_group_count": (
            len(usable)
        ),
        "maximum_pairwise_axial_separation_deg": (
            max_separation
        ),
        "most_separated_pair": best_pair,
        "two_distinct_directions_observed": (
            max_separation
            >= MIN_INDEPENDENT_HORIZONTAL_ANGLE_DEG
        ),
        "directions": usable,
        "structural_directions_verified": False,
    }


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = load_candidates()

    if not candidates:
        raise RuntimeError(
            "No valid Step 93 line candidates."
        )

    print(
        "\n=== STEP 94: LINE CONSOLIDATION ==="
    )
    print(
        "Input hypotheses:",
        len(candidates),
    )

    groups = build_duplicate_groups(
        candidates
    )

    group_of = {}

    for group_id, members in enumerate(groups):
        for index in members:
            group_of[index] = group_id

    records = build_group_records(
        groups,
        candidates,
    )

    claims, ambiguous, conflicting = (
        observation_conflicts(
            candidates,
            group_of,
        )
    )

    orientation_counts = Counter(
        record["orientation_class"]
        for record in records
    )

    usable_groups = [
        record
        for record in records
        if (
            record["useful_length_gate"]
            and record["frame_support_gate"]
        )
    ]

    spatial_cells = {
        tuple(record["spatial_cell_xy"])
        for record in usable_groups
    }

    horizontal = horizontal_direction_analysis(
        records
    )

    member_counts = [
        len(members)
        for members in groups
    ]

    all_observations = {
        observation
        for candidate in candidates
        for observation in candidate["observations"]
    }

    groups_data = {
        "method": (
            "complete_link_finite_3d_segment_"
            "consolidation"
        ),
        "groups": records,
    }

    groups_path = (
        OUTPUT / "line_groups.json"
    )

    groups_path.write_text(
        json.dumps(
            groups_data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = {
        "method": (
            "line_duplicate_and_ambiguity_diagnostic"
        ),
        "input_hypotheses": len(candidates),
        "consolidated_group_count": len(groups),
        "hypotheses_reduced_by_grouping": (
            len(candidates) - len(groups)
        ),
        "largest_group_hypothesis_count": (
            max(member_counts)
        ),
        "median_hypotheses_per_group": (
            float(np.median(member_counts))
        ),
        "orientation_group_counts": dict(
            orientation_counts
        ),
        "groups_with_length_and_support_gates": (
            len(usable_groups)
        ),
        "occupied_spatial_cells_2m": len(
            spatial_cells
        ),
        "distinct_observations_claimed": len(
            all_observations
        ),
        "observations_claimed_by_multiple_hypotheses": (
            sum(
                len(indices) > 1
                for indices in claims.values()
            )
        ),
        "observations_claimed_by_multiple_groups": (
            len(ambiguous)
        ),
        "observations_with_incompatible_claims": (
            len(conflicting)
        ),
        "top_ambiguous_observations": (
            ambiguous[:20]
        ),
        "top_conflicting_observations": (
            conflicting[:20]
        ),
        "horizontal_direction_analysis": horizontal,
        "thresholds": {
            "duplicate_max_direction_deg": (
                MAX_DIRECTION_DIFFERENCE_DEG
            ),
            "duplicate_max_separation_m": (
                MAX_LINE_SEPARATION_M
            ),
            "duplicate_min_overlap_fraction": (
                MIN_OVERLAP_FRACTION
            ),
            "conflict_distance_m": (
                CONFLICT_DISTANCE_M
            ),
            "conflict_direction_deg": (
                CONFLICT_DIRECTION_DEG
            ),
            "vertical_min_abs_z": (
                VERTICAL_MIN_ABS_Z
            ),
            "horizontal_max_abs_z": (
                HORIZONTAL_MAX_ABS_Z
            ),
            "min_useful_length_m": (
                MIN_USEFUL_LENGTH_M
            ),
            "min_group_support_frames": (
                MIN_GROUP_SUPPORT_FRAMES
            ),
        },
        "independent_physical_lines_verified": False,
        "independent_structural_directions_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Clustering does not establish physical identity.",
            "Complete-link clustering depends on chosen thresholds.",
            "Groups may still represent the same physical edge.",
            "Different physical edges may be incorrectly grouped.",
            "Aggregated supporting frames are not independent validation.",
            "Observation conflicts indicate ambiguity but do not prove which reconstruction is wrong.",
            "Only the top 200 Step 93 hypotheses were available.",
            "Horizontal directions have not been identified as building structure.",
            "Camera poses have not been independently verified.",
            "No CAD geometry was used.",
        ],
        "decision": (
            "CONSOLIDATION_DIAGNOSTIC_ONLY"
        ),
    }

    report_path = (
        OUTPUT / "consolidation_report.json"
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
        "Original hypotheses:",
        len(candidates),
    )
    print(
        "Consolidated groups:",
        len(groups),
    )
    print(
        "Near-vertical groups:",
        orientation_counts["near_vertical"],
    )
    print(
        "Near-horizontal groups:",
        orientation_counts["near_horizontal"],
    )
    print(
        "Oblique groups:",
        orientation_counts["oblique"],
    )
    print(
        "Useful length/support groups:",
        len(usable_groups),
    )
    print(
        "Occupied 2m spatial cells:",
        len(spatial_cells),
    )
    print(
        "Observations claimed by multiple groups:",
        len(ambiguous),
    )
    print(
        "Observations with incompatible claims:",
        len(conflicting),
    )
    print(
        "Usable horizontal groups:",
        horizontal[
            "usable_near_horizontal_group_count"
        ],
    )
    print(
        "Maximum horizontal angle separation:",
        round(
            horizontal[
                "maximum_pairwise_axial_separation_deg"
            ],
            2,
        ),
    )
    print("Registration verified: NO")
    print("Report:", report_path)
    print("Groups:", groups_path)


if __name__ == "__main__":
    main()
