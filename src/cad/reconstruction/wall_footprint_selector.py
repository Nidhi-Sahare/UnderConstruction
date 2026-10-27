
"""
BuildProgress - Step 16
Conservative Wall Footprint Selector

Select a compatible subset of candidate wall footprints.

Important:
- Selection is geometric, not architectural verification.
- Perpendicular junction overlaps are allowed.
- Shared CAD boundaries alone do not cause rejection.
- Competing, substantially overlapping parallel walls
  are treated as alternatives.
- Unselected candidates remain in the output.
- No missing walls or openings are invented.

Run:
    python wall_footprint_selector.py --inspect
    python wall_footprint_selector.py --build

Inputs:
    data/wall_footprints/wall_footprints.json

Outputs:
    data/wall_selection/selected_wall_footprints.json
    data/wall_selection/selection_report.json
    data/wall_selection/selection_preview.png
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch

from shapely.geometry import Polygon


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DEFAULT_INPUT = (
    ROOT / "data" / "wall_footprints"
    / "wall_footprints.json"
)

DEFAULT_OUTPUT = (
    ROOT / "data" / "wall_selection"
)

COLORS = {
    "exterior": "#D97706",
    "interior": "#2563EB",
    "lining": "#8B5CF6",
    "partition": "#16A34A",
    "unknown": "#64748B",
}


def stable_id(prefix, values):
    payload = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


def load_input(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("footprints"), list):
        raise ValueError("Missing footprints list.")

    if not isinstance(data.get("conflicts"), list):
        raise ValueError("Missing conflicts list.")

    return data


def make_geometry(footprint):
    polygon = Polygon(footprint["polygon"])

    if (
        polygon.is_empty
        or not polygon.is_valid
        or polygon.area <= 0
    ):
        raise ValueError(
            f"Invalid polygon: {footprint['id']}"
        )

    return polygon


def direction(footprint):
    line = footprint["centerline"]
    a = line["start"]
    b = line["end"]

    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])

    length = math.hypot(dx, dy)

    if length <= 1e-9:
        raise ValueError(
            f"Zero-length centerline: {footprint['id']}"
        )

    return dx / length, dy / length


def orientation_similarity(first, second):
    ax, ay = direction(first)
    bx, by = direction(second)

    return abs(ax * bx + ay * by)


def candidate_score(footprint):
    """
    Relative evidence score, not calibrated confidence.

    Favor longer, well-supported boundary pairs.
    No architectural category is inherently preferred.
    """
    length = float(
        footprint.get("overlap_length_m", 0)
    )
    ratio = float(
        footprint.get("overlap_ratio", 0)
    )

    if not math.isfinite(length):
        length = 0.0

    if not math.isfinite(ratio):
        ratio = 0.0

    length = max(0.0, length)
    ratio = max(0.0, min(1.0, ratio))

    return round(
        ratio * math.log1p(length),
        8,
    )


def classify_pair(
    first,
    second,
    geometry_a,
    geometry_b,
    args,
):
    """
    Determine whether two footprints are competing
    interpretations or compatible geometry.

    Crossing walls and small junction overlaps are
    not automatically treated as alternatives.
    """
    shared_chains = sorted(
        set(first.get("boundary_chain_ids", []))
        & set(second.get("boundary_chain_ids", []))
    )

    intersection_area = 0.0

    if geometry_a.intersects(geometry_b):
        intersection_area = float(
            geometry_a.intersection(geometry_b).area
        )

    smaller_area = min(
        geometry_a.area,
        geometry_b.area,
    )

    overlap_fraction = (
        intersection_area / smaller_area
        if smaller_area > 0
        else 0.0
    )

    parallelism = orientation_similarity(
        first,
        second,
    )

    parallel = (
        parallelism >= args.parallel_cosine
    )

    same_category = (
        first.get("category")
        == second.get("category")
    )

    same_layer = (
        first.get("layer")
        == second.get("layer")
    )

    substantial_overlap = (
        intersection_area >= args.min_overlap_area
        and overlap_fraction >= args.min_overlap_fraction
    )

    # Only substantially overlapping, parallel,
    # same-category/layer hypotheses compete.
    #
    # This avoids deleting perpendicular wall junctions
    # and different architectural layers automatically.
    competing = (
        parallel
        and substantial_overlap
        and same_category
        and same_layer
    )

    if competing:
        classification = "competing_parallel_footprints"
    elif intersection_area > args.min_overlap_area:
        classification = "compatible_or_unresolved_overlap"
    elif shared_chains:
        classification = "shared_source_only"
    else:
        classification = "compatible"

    return {
        "first_id": first["id"],
        "second_id": second["id"],
        "classification": classification,
        "competing": competing,
        "shared_boundary_chain_ids": shared_chains,
        "intersection_area_m2": round(
            intersection_area, 9
        ),
        "overlap_fraction_of_smaller": round(
            overlap_fraction, 6
        ),
        "orientation_similarity": round(
            parallelism, 6
        ),
    }


def analyze_pairs(footprints, args):
    geometries = {
        item["id"]: make_geometry(item)
        for item in footprints
    }

    records = []
    competition = defaultdict(set)
    warnings = defaultdict(list)

    for i, first in enumerate(footprints):
        for second in footprints[i + 1:]:
            record = classify_pair(
                first,
                second,
                geometries[first["id"]],
                geometries[second["id"]],
                args,
            )

            if record["competing"]:
                a = first["id"]
                b = second["id"]

                competition[a].add(b)
                competition[b].add(a)

            if record["classification"] == (
                "compatible_or_unresolved_overlap"
            ):
                warnings[first["id"]].append(
                    second["id"]
                )
                warnings[second["id"]].append(
                    first["id"]
                )

            if record["classification"] != "compatible":
                records.append(record)

    return records, competition, warnings


def choose_candidates(footprints, competition):
    """
    Deterministic weighted greedy independent set,
    followed by single-candidate improvement.

    This is a heuristic, not a globally optimal solver.
    """
    lookup = {
        item["id"]: item
        for item in footprints
    }

    scores = {
        item["id"]: candidate_score(item)
        for item in footprints
    }

    ordered = sorted(
        lookup,
        key=lambda item_id: (
            -scores[item_id],
            item_id,
        ),
    )

    selected = set()

    for item_id in ordered:
        if not (competition[item_id] & selected):
            selected.add(item_id)

    # A deterministic local improvement pass.
    # Replace selected competitors when the alternative
    # has a higher total evidence score.
    changed = True

    while changed:
        changed = False

        for item_id in ordered:
            if item_id in selected:
                continue

            blockers = (
                competition[item_id] & selected
            )

            if not blockers:
                selected.add(item_id)
                changed = True
                continue

            blocker_score = sum(
                scores[blocker]
                for blocker in blockers
            )

            if scores[item_id] > blocker_score + 1e-9:
                selected.difference_update(blockers)
                selected.add(item_id)
                changed = True

    return selected, scores


def create_output(
    source,
    footprints,
    records,
    competition,
    warnings,
    selected,
    scores,
    args,
):
    selected_items = []
    excluded_items = []
    unresolved_items = []

    for item in footprints:
        item_id = item["id"]

        result = dict(item)
        result["selection_score"] = scores[item_id]
        result["verified_wall"] = False

        result["competing_candidate_ids"] = sorted(
            competition[item_id]
        )

        result["overlap_warning_ids"] = sorted(
            warnings[item_id]
        )

        if item_id in selected:
            result["selection_status"] = (
                "provisionally_selected"
            )
            selected_items.append(result)

            if warnings[item_id]:
                unresolved_items.append({
                    "footprint_id": item_id,
                    "reason": (
                        "overlap_not_resolved_by_selector"
                    ),
                    "related_ids": sorted(
                        warnings[item_id]
                    ),
                })
        else:
            blockers = sorted(
                competition[item_id] & selected
            )

            result["selection_status"] = (
                "excluded_competing_hypothesis"
            )

            result["excluded_by"] = blockers
            excluded_items.append(result)

    selected_items.sort(key=lambda x: x["id"])
    excluded_items.sort(key=lambda x: x["id"])

    selection_id = stable_id(
        "SELECTION",
        [x["id"] for x in selected_items],
    )

    return {
        "schema_version": "0.1",
        "selection_id": selection_id,
        "coordinate_system": source.get(
            "coordinate_system"
        ),
        "method": (
            "deterministic_conservative_"
            "weighted_independent_set"
        ),
        "parameters": {
            "parallel_cosine": args.parallel_cosine,
            "min_overlap_area_m2": (
                args.min_overlap_area
            ),
            "min_overlap_fraction": (
                args.min_overlap_fraction
            ),
        },
        "selected_footprints": selected_items,
        "excluded_footprints": excluded_items,
        "unresolved_overlaps": unresolved_items,
        "pair_analysis": records,
        "unmatched_chains": source.get(
            "unmatched_chains", []
        ),
        "rejected_candidates": source.get(
            "rejected_candidates", []
        ),
        "notes": [
            "Selected footprints are not verified BIM walls.",
            "Geometric junction overlaps may remain.",
            "Shared source boundaries alone do not exclude candidates.",
            "Exterior walls are not reconstructed by this selection.",
            "No door/window openings have been generated.",
            "Unmatched CAD chains are preserved.",
        ],
    }


def plot_polygon(axis, item, alpha, edge_width=0.6):
    polygon = item["polygon"]

    color = COLORS.get(
        item.get("category", "unknown"),
        COLORS["unknown"],
    )

    axis.add_patch(
        PolygonPatch(
            polygon,
            closed=True,
            facecolor=color,
            edgecolor=color,
            linewidth=edge_width,
            alpha=alpha,
        )
    )


def draw_preview(result, output_path):
    selected = result["selected_footprints"]
    excluded = result["excluded_footprints"]

    fig, axes = plt.subplots(
        1, 2, figsize=(17, 8)
    )

    all_items = selected + excluded

    for item in all_items:
        plot_polygon(
            axes[0],
            item,
            alpha=0.65,
        )

    axes[0].set_title(
        f"All footprint hypotheses ({len(all_items)})"
    )

    for item in excluded:
        polygon = Polygon(item["polygon"])
        x, y = polygon.exterior.xy

        axes[1].plot(
            x, y,
            color="#DC2626",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
        )

    for item in selected:
        plot_polygon(
            axes[1],
            item,
            alpha=0.75,
        )

    axes[1].set_title(
        f"Provisionally selected ({len(selected)}) | "
        f"Excluded ({len(excluded)})"
    )

    for axis in axes:
        axis.set_aspect(
            "equal",
            adjustable="box",
        )
        axis.set_xlabel("CAD X (m)")
        axis.set_ylabel("CAD Y (m)")
        axis.grid(alpha=0.12)

    # Match the same spatial extent in both panels.
    if all_items:
        bounds = [
            Polygon(item["polygon"]).bounds
            for item in all_items
        ]

        xmin = min(b[0] for b in bounds)
        ymin = min(b[1] for b in bounds)
        xmax = max(b[2] for b in bounds)
        ymax = max(b[3] for b in bounds)

        margin = 1.0

        for axis in axes:
            axis.set_xlim(
                xmin - margin,
                xmax + margin,
            )
            axis.set_ylim(
                ymin - margin,
                ymax + margin,
            )

    fig.suptitle(
        "BuildProgress | Conservative Wall Selection",
        fontsize=15,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(fig)


def inspect(data):
    footprints = data["footprints"]

    print("\nWALL SELECTION INPUT")
    print("-" * 52)

    print("Footprints:", len(footprints))
    print("Reported conflicts:", len(data["conflicts"]))

    print(
        "Unmatched chains:",
        len(data.get("unmatched_chains", [])),
    )

    counts = Counter(
        item.get("category", "unknown")
        for item in footprints
    )

    print("Categories:", dict(counts))

    reasons = Counter(
        tuple(record.get("reasons", []))
        for record in data["conflicts"]
    )

    print("\nOriginal conflict classifications:")

    for reason, count in sorted(reasons.items()):
        print(f"  {reason}: {count}")


def build(data, args):
    footprints = data["footprints"]

    ids = [item["id"] for item in footprints]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate footprint IDs in input."
        )

    records, competition, warnings = analyze_pairs(
        footprints,
        args,
    )

    selected, scores = choose_candidates(
        footprints,
        competition,
    )

    result = create_output(
        data,
        footprints,
        records,
        competition,
        warnings,
        selected,
        scores,
        args,
    )

    output_dir = args.output
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = (
        output_dir / "selected_wall_footprints.json"
    )

    report_path = (
        output_dir / "selection_report.json"
    )

    preview_path = (
        output_dir / "selection_preview.png"
    )

    selection_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    counts = Counter(
        record["classification"]
        for record in records
    )

    report = {
        "input_footprints": len(footprints),
        "selected_footprints": len(
            result["selected_footprints"]
        ),
        "excluded_footprints": len(
            result["excluded_footprints"]
        ),
        "unresolved_overlap_entries": len(
            result["unresolved_overlaps"]
        ),
        "pair_classifications": dict(counts),
        "unmatched_chains_preserved": len(
            result["unmatched_chains"]
        ),
        "selected_categories": dict(
            Counter(
                item["category"]
                for item in result["selected_footprints"]
            )
        ),
        "selection_id": result["selection_id"],
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    draw_preview(
        result,
        preview_path,
    )

    print("\nWALL FOOTPRINT SELECTION COMPLETE")
    print("-" * 52)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nOutputs:")
    print(selection_path)
    print(report_path)
    print(preview_path)

    print(
        "\nNote: selection is provisional; "
        "no architectural verification performed."
    )


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

    parser.add_argument(
        "--inspect",
        action="store_true",
    )

    parser.add_argument(
        "--build",
        action="store_true",
    )

    parser.add_argument(
        "--parallel-cosine",
        type=float,
        default=0.98,
    )

    parser.add_argument(
        "--min-overlap-area",
        type=float,
        default=0.0025,
    )

    parser.add_argument(
        "--min-overlap-fraction",
        type=float,
        default=0.15,
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"Input not found: {args.input}"
        )

    if not (args.inspect or args.build):
        parser.error(
            "Specify --inspect or --build."
        )

    if not (0 <= args.parallel_cosine <= 1):
        parser.error(
            "--parallel-cosine must be between 0 and 1."
        )

    if not (
        0 <= args.min_overlap_fraction <= 1
    ):
        parser.error(
            "--min-overlap-fraction must be between 0 and 1."
        )

    if args.min_overlap_area < 0:
        parser.error(
            "--min-overlap-area cannot be negative."
        )

    data = load_input(args.input)

    if args.inspect:
        inspect(data)

    if args.build:
        build(data, args)


if __name__ == "__main__":
    main()

