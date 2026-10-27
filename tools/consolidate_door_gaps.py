
"""
Under-construction — Door Gap Consolidation v3

Reads:
    outputs/cad_openings/opening_candidates.json

Writes:
    outputs/cad_openings/opening_candidates_v3.json

Purpose:
    Collapse multiple wall-face representations of the same
    possible doorway into one physical gap hypothesis.

Preserves:
    - Every original candidate
    - Every original gap score and wall element ID
    - Unmatched door arcs
    - Window candidates
    - Explicit uncertainty and abstention

Does NOT modify the CAD, wall geometry, or 3D viewer.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    ROOT / "outputs/cad_openings/opening_candidates.json"
)

DEFAULT_OUTPUT = (
    ROOT / "outputs/cad_openings/opening_candidates_v3.json"
)

# Drawing units. These are conservative grouping heuristics,
# not independently verified architectural tolerances.
MAX_MIDPOINT_DISTANCE = 0.18
MAX_ENDPOINT_DISTANCE = 0.22
MAX_WIDTH_DIFFERENCE = 0.20
MAX_DIRECTION_DIFFERENCE_DEG = 12.0

MIN_SCORE_MARGIN = 0.15
MAX_ACCEPTED_SCORE = 0.75


def point(value):
    return float(value[0]), float(value[1])


def distance(a, b):
    return math.hypot(
        a[0] - b[0],
        a[1] - b[1],
    )


def midpoint(a, b):
    return (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
    )


def direction(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]

    length = math.hypot(dx, dy)

    if length < 1e-9:
        return None

    return dx / length, dy / length


def angle_between_lines(a, b):
    dot = abs(
        a[0] * b[0] +
        a[1] * b[1]
    )

    dot = min(1.0, max(0.0, dot))

    return math.degrees(math.acos(dot))


def endpoint_pair_distance(first, second):
    a1 = point(first["gap_start"])
    a2 = point(first["gap_end"])

    b1 = point(second["gap_start"])
    b2 = point(second["gap_end"])

    same_order = max(
        distance(a1, b1),
        distance(a2, b2),
    )

    reversed_order = max(
        distance(a1, b2),
        distance(a2, b1),
    )

    return min(same_order, reversed_order)


def equivalent_gaps(first, second):
    """
    Two candidate gaps are equivalent only if:
      - their midpoints are close,
      - their widths are similar,
      - their directions are similar,
      - both endpoint positions agree spatially.

    Different wall element IDs do not automatically
    mean different physical openings.
    """

    a1 = point(first["gap_start"])
    a2 = point(first["gap_end"])

    b1 = point(second["gap_start"])
    b2 = point(second["gap_end"])

    midpoint_error = distance(
        midpoint(a1, a2),
        midpoint(b1, b2),
    )

    if midpoint_error > MAX_MIDPOINT_DISTANCE:
        return False

    width_error = abs(
        float(first["gap_width"]) -
        float(second["gap_width"])
    )

    if width_error > MAX_WIDTH_DIFFERENCE:
        return False

    direction_a = direction(a1, a2)
    direction_b = direction(b1, b2)

    if direction_a is None or direction_b is None:
        return False

    angle_error = angle_between_lines(
        direction_a,
        direction_b,
    )

    if angle_error > MAX_DIRECTION_DIFFERENCE_DEG:
        return False

    if endpoint_pair_distance(
        first,
        second,
    ) > MAX_ENDPOINT_DISTANCE:
        return False

    return True


def group_equivalent_gaps(gaps):
    """
    Complete-link clustering:

    A candidate joins a group only if it is equivalent
    to EVERY existing member of that group.

    This prevents a chain of nearby but distinct gaps
    from being merged into one oversized cluster.
    """

    ordered = sorted(
        gaps,
        key=lambda item: float(item["score"]),
    )

    groups = []

    for gap in ordered:
        compatible_groups = []

        for group_index, group in enumerate(groups):
            if all(
                equivalent_gaps(gap, member)
                for member in group
            ):
                compatible_groups.append(group_index)

        if len(compatible_groups) == 1:
            groups[compatible_groups[0]].append(gap)

        elif not compatible_groups:
            groups.append([gap])

        else:
            # More than one compatible group is possible.
            # Do not bridge them automatically.
            groups.append([gap])

    return groups


def summarize_group(index, group):
    ranked = sorted(
        group,
        key=lambda item: float(item["score"]),
    )

    best = ranked[0]

    midpoints = [
        midpoint(
            point(item["gap_start"]),
            point(item["gap_end"]),
        )
        for item in group
    ]

    mean_midpoint = (
        sum(p[0] for p in midpoints) / len(midpoints),
        sum(p[1] for p in midpoints) / len(midpoints),
    )

    wall_ids = sorted({
        wall_id
        for item in group
        for wall_id in item.get("wall_element_ids", [])
    })

    return {
        "physical_gap_id": f"gap_{index:03d}",
        "representative_score": float(best["score"]),
        "representative_gap_start": best["gap_start"],
        "representative_gap_end": best["gap_end"],
        "representative_width": best["gap_width"],
        "mean_midpoint": [
            round(mean_midpoint[0], 5),
            round(mean_midpoint[1], 5),
        ],
        "wall_element_ids": wall_ids,
        "representation_count": len(group),
        "original_gap_candidates": ranked,
        "physically_verified": False,
    }


def consolidate_door(candidate):
    result = deepcopy(candidate)

    if result.get("type") != "door":
        return result

    if result.get("method") != "SWING_ARC_AND_WALL_GAP":
        result["consolidation_status"] = "NOT_ARC_ANCHORED"
        result["physical_gap_hypotheses"] = []
        result["safe_to_cut"] = False
        return result

    gaps = result.get("gap_candidates", [])

    if not gaps:
        result["consolidation_status"] = "NO_SUPPORTED_GAP"
        result["physical_gap_hypotheses"] = []
        result["safe_to_cut"] = False
        return result

    groups = group_equivalent_gaps(gaps)

    hypotheses = [
        summarize_group(i, group)
        for i, group in enumerate(groups)
    ]

    hypotheses.sort(
        key=lambda item: item["representative_score"]
    )

    best = hypotheses[0]

    if best["representative_score"] > MAX_ACCEPTED_SCORE:
        status = "WEAK_GAP_HYPOTHESIS"

    elif len(hypotheses) == 1:
        status = "SINGLE_PHYSICAL_GAP_HYPOTHESIS"

    else:
        second = hypotheses[1]

        score_margin = (
            second["representative_score"] -
            best["representative_score"]
        )

        if score_margin >= MIN_SCORE_MARGIN:
            status = "PREFERRED_PHYSICAL_GAP_HYPOTHESIS"
        else:
            status = "AMBIGUOUS_PHYSICAL_GAPS"

    result["original_association_status"] = result.get(
        "association_status"
    )

    result["association_status"] = status
    result["consolidation_status"] = status

    result["original_gap_candidate_count"] = len(gaps)
    result["physical_gap_hypothesis_count"] = len(hypotheses)
    result["physical_gap_hypotheses"] = hypotheses

    # A single preferred geometric hypothesis does not
    # establish that a doorway has been verified.
    result["opening_verified"] = False
    result["safe_to_cut"] = False

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    if args.input.resolve() == args.output.resolve():
        raise ValueError(
            "Input and output must be different files."
        )

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as handle:
        source = json.load(handle)

    if source.get("schema_version") != "2.0":
        raise ValueError(
            "Expected a v2 opening candidate report."
        )

    candidates = [
        consolidate_door(candidate)
        for candidate in source["candidates"]
    ]

    original_door_arcs = [
        candidate
        for candidate in source["candidates"]
        if candidate.get("method")
        == "SWING_ARC_AND_WALL_GAP"
    ]

    updated_door_arcs = [
        candidate
        for candidate in candidates
        if candidate.get("method")
        == "SWING_ARC_AND_WALL_GAP"
    ]

    before = Counter(
        candidate["association_status"]
        for candidate in original_door_arcs
    )

    after = Counter(
        candidate["association_status"]
        for candidate in updated_door_arcs
    )

    transitions = Counter(
        (
            old["association_status"],
            new["association_status"],
        )
        for old, new in zip(
            original_door_arcs,
            updated_door_arcs,
        )
    )

    report = deepcopy(source)

    report["schema_version"] = "3.0"
    report["source_candidate_report"] = str(args.input)
    report["candidates"] = candidates

    report["association_status_counts"] = dict(
        Counter(
            candidate["association_status"]
            for candidate in candidates
        )
    )

    report["door_gap_consolidation"] = {
        "original_door_arc_count": len(original_door_arcs),
        "original_status_counts": dict(before),
        "consolidated_status_counts": dict(after),
        "status_transitions": [
            {
                "from": old,
                "to": new,
                "count": count,
            }
            for (old, new), count in sorted(
                transitions.items()
            )
        ],
        "equivalence_tolerances": {
            "max_midpoint_distance": MAX_MIDPOINT_DISTANCE,
            "max_endpoint_distance": MAX_ENDPOINT_DISTANCE,
            "max_width_difference": MAX_WIDTH_DIFFERENCE,
            "max_direction_difference_degrees":
                MAX_DIRECTION_DIFFERENCE_DEG,
        },
        "method": "COMPLETE_LINK_SPATIAL_CLUSTERING",
        "physically_verified": False,
    }

    report["openings_verified"] = False
    report["wall_cutouts_generated"] = False

    report.setdefault("notes", []).append(
        "V3 clusters spatially equivalent gap representations; "
        "it does not validate or construct openings."
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print("\n=== CAD OPENING CONSOLIDATION V3 ===")
    print("Door arcs:", len(original_door_arcs))
    print("Original statuses:", dict(before))
    print("Consolidated statuses:", dict(after))

    print("\nStatus transitions:")

    for (old, new), count in sorted(
        transitions.items()
    ):
        print(f"  {old} -> {new}: {count}")

    print("\nOutput:", args.output)
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
