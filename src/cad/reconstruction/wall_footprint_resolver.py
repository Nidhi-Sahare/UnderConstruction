
"""
BuildProgress - Step 15
Conservative Wall Footprint Resolver

Input:
    data/wall_reconstruction/wall_candidates.json

Outputs:
    data/wall_footprints/wall_footprints.json
    data/wall_footprints/footprint_report.json
    data/wall_footprints/footprint_preview.png

Purpose:
    Convert parallel-boundary wall hypotheses into
    evidence-linked polygonal wall footprints.

    Detect conflicts and competing interpretations.

    Preserve unmatched CAD geometry and abstain
    from inventing missing architectural elements.

Requirements:
    pip install shapely numpy matplotlib

Usage:
    python wall_footprint_resolver.py --inspect
    python wall_footprint_resolver.py --build
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import (
    LineString,
    Polygon,
)
from shapely.ops import unary_union


PROJECT_def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "wall_reconstruction"
    / "wall_candidates.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "wall_footprints"
)

CATEGORY_COLORS = {
    "exterior": "#D97706",
    "interior": "#2563EB",
    "partition": "#16A34A",
    "lining": "#8B5CF6",
    "unknown": "#64748B",
}


def stable_id(prefix, payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hashlib.sha256(
        encoded
    ).hexdigest()[:16]

    return f"{prefix}_{digest}"


def load_reconstruction(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    chains = data.get("chains")
    candidates = data.get("wall_candidates")

    if not isinstance(chains, list):
        raise ValueError(
            "Missing 'chains' list."
        )

    if not isinstance(candidates, list):
        raise ValueError(
            "Missing 'wall_candidates' list."
        )

    return data, chains, candidates


def validate_point(point):
    if not isinstance(point, (list, tuple)):
        raise ValueError("Invalid coordinate.")

    if len(point) < 2:
        raise ValueError("Expected XY coordinate.")

    x = float(point[0])
    y = float(point[1])

    if not (
        math.isfinite(x)
        and math.isfinite(y)
    ):
        raise ValueError(
            "Non-finite coordinate."
        )

    return (x, y)


def candidate_polygon(candidate):
    centerline = candidate["centerline"]

    start = validate_point(
        centerline["start"]
    )

    end = validate_point(
        centerline["end"]
    )

    thickness = float(
        candidate["estimated_thickness_m"]
    )

    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError(
            "Invalid candidate thickness."
        )

    line = LineString([
        start,
        end,
    ])

    if line.length <= 1e-8:
        raise ValueError(
            "Zero-length centerline."
        )

    polygon = line.buffer(
        thickness / 2.0,
        cap_style="flat",
        join_style="mitre",
    )

    if (
        polygon.is_empty
        or not polygon.is_valid
        or polygon.area <= 1e-10
    ):
        raise ValueError(
            "Invalid footprint polygon."
        )

    return polygon


def polygon_coordinates(polygon):
    if polygon.geom_type != "Polygon":
        raise ValueError(
            "Expected single polygon."
        )

    return [
        [float(x), float(y)]
        for x, y in polygon.exterior.coords
    ]


def resolve_candidates(candidates):
    footprints = []
    rejected = []

    seen_ids = set()

    for candidate in candidates:
        candidate_id = str(
            candidate.get("id", "")
        )

        if not candidate_id:
            rejected.append({
                "candidate_id": None,
                "reason": "missing_id",
            })
            continue

        if candidate_id in seen_ids:
            rejected.append({
                "candidate_id": candidate_id,
                "reason": "duplicate_id",
            })
            continue

        seen_ids.add(candidate_id)

        try:
            polygon = candidate_polygon(
                candidate
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            rejected.append({
                "candidate_id": candidate_id,
                "reason": str(error),
            })
            continue

        boundary_ids = sorted(
            set(
                candidate.get(
                    "boundary_chain_ids",
                    [],
                )
            )
        )

        source_ids = sorted(
            set(
                candidate.get(
                    "source_segment_ids",
                    [],
                )
            )
        )

        footprint_id = stable_id(
            "FOOTPRINT",
            {
                "candidate_id": candidate_id,
                "boundary_ids": boundary_ids,
            },
        )

        footprints.append({
            "id": footprint_id,
            "candidate_id": candidate_id,
            "category": candidate.get(
                "category",
                "unknown",
            ),
            "layer": candidate.get(
                "layer",
                "",
            ),
            "boundary_chain_ids": boundary_ids,
            "source_segment_ids": source_ids,
            "centerline": candidate["centerline"],
            "estimated_thickness_m": float(
                candidate["estimated_thickness_m"]
            ),
            "overlap_length_m": float(
                candidate.get(
                    "overlap_length_m",
                    0.0,
                )
            ),
            "overlap_ratio": float(
                candidate.get(
                    "overlap_ratio",
                    0.0,
                )
            ),
            "area_m2": float(
                polygon.area
            ),
            "polygon": polygon_coordinates(
                polygon
            ),
            "status": "geometric_hypothesis",
            "verified_wall": False,
        })

    return footprints, rejected


def detect_conflicts(
    footprints,
    area_tolerance=0.0025,
):
    """
    Detect:
      - substantial polygon overlaps
      - shared source boundary chains

    Touching at an edge or point is not by itself
    considered an overlap conflict.
    """

    polygons = {
        footprint["id"]: Polygon(
            footprint["polygon"]
        )
        for footprint in footprints
    }

    conflicts = []

    for i in range(len(footprints)):
        first = footprints[i]

        first_polygon = polygons[
            first["id"]
        ]

        first_boundaries = set(
            first["boundary_chain_ids"]
        )

        for j in range(
            i + 1,
            len(footprints),
        ):
            second = footprints[j]

            second_polygon = polygons[
                second["id"]
            ]

            shared_boundaries = sorted(
                first_boundaries.intersection(
                    second["boundary_chain_ids"]
                )
            )

            intersection_area = 0.0

            if first_polygon.intersects(
                second_polygon
            ):
                intersection_area = float(
                    first_polygon.intersection(
                        second_polygon
                    ).area
                )

            reasons = []

            if (
                intersection_area
                > area_tolerance
            ):
                reasons.append(
                    "polygon_overlap"
                )

            if shared_boundaries:
                reasons.append(
                    "shared_boundary_chain"
                )

            if not reasons:
                continue

            conflicts.append({
                "first_id": first["id"],
                "second_id": second["id"],
                "reasons": reasons,
                "intersection_area_m2": (
                    intersection_area
                ),
                "shared_boundary_chain_ids": (
                    shared_boundaries
                ),
            })

    return conflicts


def group_compatible_footprints(
    footprints,
    conflicts,
    touch_tolerance=0.015,
):
    """
    Group touching footprints of the same category/layer.

    These groups are adjacency hypotheses only.

    Polygons are NOT automatically unioned into a
    single architectural wall.
    """

    conflicting_pairs = {
        frozenset([
            conflict["first_id"],
            conflict["second_id"],
        ])
        for conflict in conflicts
    }

    polygons = [
        Polygon(
            footprint["polygon"]
        )
        for footprint in footprints
    ]

    parent = list(
        range(len(footprints))
    )

    def find(index):
        while parent[index] != index:
            parent[index] = parent[
                parent[index]
            ]
            index = parent[index]

        return index

    def union(first, second):
        root_a = find(first)
        root_b = find(second)

        if root_a != root_b:
            parent[root_b] = root_a

    for i in range(len(footprints)):
        for j in range(
            i + 1,
            len(footprints),
        ):
            first = footprints[i]
            second = footprints[j]

            if (
                first["category"]
                != second["category"]
            ):
                continue

            if (
                first["layer"]
                != second["layer"]
            ):
                continue

            pair = frozenset([
                first["id"],
                second["id"],
            ])

            if pair in conflicting_pairs:
                continue

            distance = polygons[i].distance(
                polygons[j]
            )

            if distance <= touch_tolerance:
                union(i, j)

    components = defaultdict(list)

    for index in range(len(footprints)):
        components[
            find(index)
        ].append(index)

    groups = []

    for indices in components.values():
        members = sorted(
            footprints[index]["id"]
            for index in indices
        )

        group_id = stable_id(
            "GROUP",
            {
                "members": members,
            },
        )

        groups.append({
            "id": group_id,
            "footprint_ids": members,
            "category": footprints[
                indices[0]
            ]["category"],
            "layer": footprints[
                indices[0]
            ]["layer"],
            "status": (
                "geometric_adjacency_group"
            ),
            "verified_wall": False,
        })

    return groups


def find_unmatched_chains(
    chains,
    footprints,
):
    matched_ids = set()

    for footprint in footprints:
        matched_ids.update(
            footprint[
                "boundary_chain_ids"
            ]
        )

    unmatched = []

    for chain in chains:
        if chain["id"] in matched_ids:
            continue

        unmatched.append({
            "id": chain["id"],
            "category": chain.get(
                "category",
                "unknown",
            ),
            "layer": chain.get(
                "layer",
                "",
            ),
            "start": chain["start"],
            "end": chain["end"],
            "source_segment_ids": (
                chain.get(
                    "source_segment_ids",
                    [],
                )
            ),
            "status": (
                "unmatched_cad_boundary"
            ),
        })

    return unmatched


def render_preview(
    chains,
    footprints,
    unmatched,
    conflicts,
    output_path,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 9),
    )

    for chain in chains:
        start = chain["start"]
        end = chain["end"]

        axes[0].plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=CATEGORY_COLORS.get(
                chain.get(
                    "category",
                    "unknown",
                ),
                CATEGORY_COLORS["unknown"],
            ),
            linewidth=0.8,
            alpha=0.8,
        )

    axes[0].set_title(
        f"Source CAD chains ({len(chains)})"
    )

    for footprint in footprints:
        polygon = Polygon(
            footprint["polygon"]
        )

        x, y = polygon.exterior.xy

        color = CATEGORY_COLORS.get(
            footprint["category"],
            CATEGORY_COLORS["unknown"],
        )

        axes[1].fill(
            x,
            y,
            color=color,
            alpha=0.55,
            linewidth=0,
        )

        axes[1].plot(
            x,
            y,
            color=color,
            linewidth=0.5,
        )

    for chain in unmatched:
        start = chain["start"]
        end = chain["end"]

        axes[1].plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#94A3B8",
            linewidth=0.65,
            alpha=0.5,
        )

    footprint_lookup = {
        footprint["id"]: footprint
        for footprint in footprints
    }

    for conflict in conflicts:
        first = footprint_lookup[
            conflict["first_id"]
        ]

        second = footprint_lookup[
            conflict["second_id"]
        ]

        polygon_a = Polygon(
            first["polygon"]
        )

        polygon_b = Polygon(
            second["polygon"]
        )

        intersection = polygon_a.intersection(
            polygon_b
        )

        if intersection.is_empty:
            continue

        if intersection.area <= 0:
            continue

        representative = (
            intersection.representative_point()
        )

        axes[1].scatter(
            [representative.x],
            [representative.y],
            s=12,
            color="#DC2626",
            zorder=10,
        )

    axes[1].set_title(
        f"Candidate footprints ({len(footprints)}) | "
        f"Conflicts ({len(conflicts)})"
    )

    for axis in axes:
        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.set_xlabel(
            "CAD X (m)"
        )

        axis.set_ylabel(
            "CAD Y (m)"
        )

        axis.grid(
            True,
            alpha=0.12,
        )

    fig.suptitle(
        "BuildProgress | Wall Footprint Resolution",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


def inspect(
    chains,
    candidates,
):
    print("\nWALL FOOTPRINT INPUT INSPECTION")
    print("-" * 50)

    print(f"Chains: {len(chains)}")
    print(f"Boundary hypotheses: {len(candidates)}")

    category_counts = Counter(
        candidate.get(
            "category",
            "unknown",
        )
        for candidate in candidates
    )

    for category, count in sorted(
        category_counts.items()
    ):
        print(
            f"  {category}: {count}"
        )

    thicknesses = [
        float(
            candidate[
                "estimated_thickness_m"
            ]
        )
        for candidate in candidates
        if "estimated_thickness_m"
        in candidate
    ]

    if thicknesses:
        print(
            "Estimated thickness range: "
            f"{min(thicknesses):.3f} - "
            f"{max(thicknesses):.3f} m"
        )


def build(
    data,
    chains,
    candidates,
    output_dir,
    args,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    footprints, rejected = resolve_candidates(
        candidates
    )

    conflicts = detect_conflicts(
        footprints,
        area_tolerance=args.area_tolerance,
    )

    groups = group_compatible_footprints(
        footprints,
        conflicts,
        touch_tolerance=args.touch_tolerance,
    )

    unmatched = find_unmatched_chains(
        chains,
        footprints,
    )

    conflicting_ids = set()

    for conflict in conflicts:
        conflicting_ids.add(
            conflict["first_id"]
        )

        conflicting_ids.add(
            conflict["second_id"]
        )

    for footprint in footprints:
        footprint["has_conflict"] = (
            footprint["id"]
            in conflicting_ids
        )

        footprint["status"] = (
            "ambiguous_geometric_hypothesis"
            if footprint["has_conflict"]
            else "unverified_geometric_hypothesis"
        )

    result = {
        "schema_version": "0.1",
        "source_reconstruction": str(
            args.input.resolve()
        ),
        "coordinate_system": data.get(
            "coordinate_system",
            {
                "units": "meters",
                "source": "DXF",
            },
        ),
        "footprints": footprints,
        "adjacency_groups": groups,
        "conflicts": conflicts,
        "unmatched_chains": unmatched,
        "rejected_candidates": rejected,
        "notes": [
            "All footprints are geometric hypotheses.",
            "No footprint is a verified BIM wall.",
            "Unmatched CAD boundaries are retained.",
            "Conflicts are not silently resolved.",
            "Door and window openings are not inferred.",
        ],
    }

    result_path = (
        output_dir / "wall_footprints.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "source_chains": len(chains),
        "input_candidates": len(candidates),
        "resolved_footprints": len(footprints),
        "rejected_candidates": len(rejected),
        "conflict_pairs": len(conflicts),
        "footprints_with_conflicts": len(
            conflicting_ids
        ),
        "adjacency_groups": len(groups),
        "unmatched_chains": len(unmatched),
        "categories": dict(
            Counter(
                footprint["category"]
                for footprint in footprints
            )
        ),
        "parameters": {
            "area_tolerance_m2": (
                args.area_tolerance
            ),
            "touch_tolerance_m": (
                args.touch_tolerance
            ),
        },
    }

    report_path = (
        output_dir / "footprint_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    preview_path = (
        output_dir / "footprint_preview.png"
    )

    render_preview(
        chains,
        footprints,
        unmatched,
        conflicts,
        preview_path,
    )

    print("\nFOOTPRINT RESOLUTION COMPLETE")
    print("-" * 50)

    print(f"Input hypotheses: {len(candidates)}")
    print(f"Resolved footprints: {len(footprints)}")
    print(f"Rejected candidates: {len(rejected)}")
    print(f"Conflict pairs: {len(conflicts)}")
    print(f"Adjacency groups: {len(groups)}")
    print(f"Unmatched CAD chains: {len(unmatched)}")

    print("\nOutputs:")
    print(result_path)
    print(report_path)
    print(preview_path)


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
        "--area-tolerance",
        type=float,
        default=0.0025,
    )

    parser.add_argument(
        "--touch-tolerance",
        type=float,
        default=0.015,
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"Input file not found: {args.input}"
        )

    if not (
        args.inspect
        or args.build
    ):
        parser.error(
            "Specify --inspect or --build."
        )

    data, chains, candidates = (
        load_reconstruction(
            args.input
        )
    )

    if args.inspect:
        inspect(
            chains,
            candidates,
        )

    if args.build:
        build(
            data,
            chains,
            candidates,
            args.output,
            args,
        )


if __name__ == "__main__":
    main()

