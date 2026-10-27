
"""
BuildProgress - Step 14
Conservative CAD Wall Geometry Reconstruction

Input:
    data/geometry/normalized_segments.json

Outputs:
    data/wall_reconstruction/wall_candidates.json
    data/wall_reconstruction/reconstruction_report.json
    data/wall_reconstruction/reconstruction_preview.png

Purpose:
    1. Preserve source CAD segments and provenance.
    2. Merge connected collinear segments.
    3. Identify plausible parallel wall boundaries.
    4. Produce evidence-linked wall candidates.
    5. Abstain from declaring ambiguous candidates
       to be confirmed architectural walls.

No AI models or GPU required.

Usage:
    python wall_geometry_reconstructor.py --inspect
    python wall_geometry_reconstructor.py --build
"""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, Point
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
    / "geometry"
    / "normalized_segments.json"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "wall_reconstruction"
)

CATEGORY_COLORS = {
    "exterior": "#D97706",
    "interior": "#2563EB",
    "partition": "#16A34A",
    "lining": "#8B5CF6",
    "unknown": "#64748B",
}


def load_segments(path):
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("segments")

    if not isinstance(records, list):
        raise ValueError(
            "Expected top-level 'segments' list."
        )

    segments = []

    for index, record in enumerate(records):
        start = np.asarray(
            record["start"][:2],
            dtype=float,
        )

        end = np.asarray(
            record["end"][:2],
            dtype=float,
        )

        if not (
            np.all(np.isfinite(start))
            and np.all(np.isfinite(end))
        ):
            raise ValueError(
                f"Non-finite coordinates at record {index}."
            )

        length = float(
            np.linalg.norm(end - start)
        )

        if length <= 1e-8:
            continue

        segments.append({
            "id": str(
                record.get(
                    "id",
                    f"source_{index:06d}",
                )
            ),
            "start": start,
            "end": end,
            "category": str(
                record.get(
                    "category",
                    "unknown",
                )
            ),
            "layer": str(
                record.get("layer", "")
            ),
            "source_handle": record.get(
                "source_handle"
            ),
            "parent_insert_handle": record.get(
                "parent_insert_handle"
            ),
            "source_entity_type": record.get(
                "source_entity_type"
            ),
            "length": length,
        })

    return segments


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


def canonical_direction(vector):
    length = np.linalg.norm(vector)

    if length <= 1e-12:
        raise ValueError("Zero-length vector.")

    direction = vector / length

    if (
        direction[0] < -1e-12
        or (
            abs(direction[0]) <= 1e-12
            and direction[1] < 0
        )
    ):
        direction = -direction

    return direction


def quantized_point(point, tolerance):
    return tuple(
        int(round(float(value) / tolerance))
        for value in point
    )


def merge_collinear_segments(
    segments,
    angle_tolerance_degrees=1.0,
    distance_tolerance=0.015,
    endpoint_tolerance=0.015,
):
    """
    Conservatively merge connected collinear lines.

    Uses category + layer grouping to avoid combining
    unrelated CAD classes.

    Returns merged chains and their source IDs.
    """

    groups = defaultdict(list)

    for segment in segments:
        key = (
            segment["category"],
            segment["layer"],
        )

        groups[key].append(segment)

    merged = []

    for (category, layer), records in groups.items():
        count = len(records)

        parent = list(range(count))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]

            return index

        def union(first, second):
            root_a = find(first)
            root_b = find(second)

            if root_a != root_b:
                parent[root_b] = root_a

        directions = [
            canonical_direction(
                item["end"] - item["start"]
            )
            for item in records
        ]

        # Spatial endpoint index.
        endpoint_index = defaultdict(list)

        for index, record in enumerate(records):
            for endpoint in (
                record["start"],
                record["end"],
            ):
                key = quantized_point(
                    endpoint,
                    endpoint_tolerance,
                )

                endpoint_index[key].append(index)

        candidate_pairs = set()

        for indices in endpoint_index.values():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    a, b = sorted(
                        (indices[i], indices[j])
                    )

                    candidate_pairs.add((a, b))

        for first, second in sorted(candidate_pairs):
            a = records[first]
            b = records[second]

            dot = float(
                np.clip(
                    np.dot(
                        directions[first],
                        directions[second],
                    ),
                    -1.0,
                    1.0,
                )
            )

            angle = math.degrees(
                math.acos(abs(dot))
            )

            if angle > angle_tolerance_degrees:
                continue

            line_a = LineString([
                a["start"],
                a["end"],
            ])

            line_b = LineString([
                b["start"],
                b["end"],
            ])

            if line_a.distance(line_b) > distance_tolerance:
                continue

            endpoints_a = [
                a["start"],
                a["end"],
            ]

            endpoints_b = [
                b["start"],
                b["end"],
            ]

            min_endpoint_distance = min(
                float(np.linalg.norm(p - q))
                for p in endpoints_a
                for q in endpoints_b
            )

            if min_endpoint_distance > endpoint_tolerance:
                continue

            union(first, second)

        components = defaultdict(list)

        for index in range(count):
            components[find(index)].append(index)

        for indices in components.values():
            component = [
                records[index]
                for index in indices
            ]

            # Use the longest source segment to define
            # the reference direction.
            reference = max(
                component,
                key=lambda item: item["length"],
            )

            direction = canonical_direction(
                reference["end"]
                - reference["start"]
            )

            origin = reference["start"]

            projected = []

            all_points = []

            for item in component:
                for point in (
                    item["start"],
                    item["end"],
                ):
                    projected.append(
                        float(
                            np.dot(
                                point - origin,
                                direction,
                            )
                        )
                    )

                    all_points.append(point)

            minimum = min(projected)
            maximum = max(projected)

            start = origin + minimum * direction
            end = origin + maximum * direction

            # Residual is retained as a quality check.
            residuals = []

            for point in all_points:
                offset = point - origin

                perpendicular = (
                    offset
                    - np.dot(offset, direction)
                    * direction
                )

                residuals.append(
                    float(np.linalg.norm(perpendicular))
                )

            maximum_residual = max(residuals)

            source_ids = sorted(
                item["id"]
                for item in component
            )

            identifier = stable_id(
                "CHAIN",
                {
                    "sources": source_ids,
                    "category": category,
                    "layer": layer,
                },
            )

            merged.append({
                "id": identifier,
                "start": start.tolist(),
                "end": end.tolist(),
                "length_m": float(
                    np.linalg.norm(end - start)
                ),
                "category": category,
                "layer": layer,
                "source_segment_ids": source_ids,
                "source_count": len(component),
                "maximum_residual_m": maximum_residual,
                "geometry_status": (
                    "collinear_merge_candidate"
                    if len(component) > 1
                    else "original_segment"
                ),
            })

    return merged


def segment_projection(
    start,
    end,
    direction,
):
    a = float(
        np.dot(start, direction)
    )

    b = float(
        np.dot(end, direction)
    )

    return (
        min(a, b),
        max(a, b),
    )


def detect_parallel_boundaries(
    chains,
    angle_tolerance_degrees=2.0,
    minimum_separation=0.06,
    maximum_separation=0.45,
    minimum_overlap=0.50,
    minimum_overlap_ratio=0.50,
):
    """
    Identify parallel boundary pairs.

    These are geometric hypotheses, not confirmed walls.
    """

    candidates = []

    groups = defaultdict(list)

    for chain in chains:
        groups[
            (
                chain["category"],
                chain["layer"],
            )
        ].append(chain)

    for (category, layer), records in groups.items():
        for first in range(len(records)):
            a = records[first]

            a_start = np.asarray(
                a["start"],
                dtype=float,
            )

            a_end = np.asarray(
                a["end"],
                dtype=float,
            )

            direction_a = canonical_direction(
                a_end - a_start
            )

            normal = np.array([
                -direction_a[1],
                direction_a[0],
            ])

            a_min, a_max = segment_projection(
                a_start,
                a_end,
                direction_a,
            )

            for second in range(
                first + 1,
                len(records),
            ):
                b = records[second]

                b_start = np.asarray(
                    b["start"],
                    dtype=float,
                )

                b_end = np.asarray(
                    b["end"],
                    dtype=float,
                )

                direction_b = canonical_direction(
                    b_end - b_start
                )

                dot = float(
                    np.clip(
                        np.dot(
                            direction_a,
                            direction_b,
                        ),
                        -1.0,
                        1.0,
                    )
                )

                angle = math.degrees(
                    math.acos(abs(dot))
                )

                if angle > angle_tolerance_degrees:
                    continue

                b_min, b_max = segment_projection(
                    b_start,
                    b_end,
                    direction_a,
                )

                overlap_start = max(
                    a_min,
                    b_min,
                )

                overlap_end = min(
                    a_max,
                    b_max,
                )

                overlap = (
                    overlap_end - overlap_start
                )

                if overlap < minimum_overlap:
                    continue

                shorter_length = min(
                    a["length_m"],
                    b["length_m"],
                )

                if shorter_length <= 0:
                    continue

                overlap_ratio = (
                    overlap / shorter_length
                )

                if (
                    overlap_ratio
                    < minimum_overlap_ratio
                ):
                    continue

                offsets = [
                    float(
                        np.dot(
                            b_start - a_start,
                            normal,
                        )
                    ),
                    float(
                        np.dot(
                            b_end - a_start,
                            normal,
                        )
                    ),
                ]

                signed_separation = (
                    offsets[0] + offsets[1]
                ) / 2.0

                separation = abs(
                    signed_separation
                )

                if not (
                    minimum_separation
                    <= separation
                    <= maximum_separation
                ):
                    continue

                # Reject substantial lateral drift.
                drift = abs(
                    offsets[0] - offsets[1]
                )

                if drift > 0.03:
                    continue

                center_offset = (
                    signed_separation / 2.0
                )

                center_start = (
                    direction_a * overlap_start
                    + normal * (
                        float(
                            np.dot(
                                a_start,
                                normal,
                            )
                        )
                        + center_offset
                    )
                )

                center_end = (
                    direction_a * overlap_end
                    + normal * (
                        float(
                            np.dot(
                                a_start,
                                normal,
                            )
                        )
                        + center_offset
                    )
                )

                pair_ids = sorted([
                    a["id"],
                    b["id"],
                ])

                identifier = stable_id(
                    "WALL",
                    {
                        "boundaries": pair_ids,
                    },
                )

                candidates.append({
                    "id": identifier,
                    "category": category,
                    "layer": layer,
                    "boundary_chain_ids": pair_ids,
                    "source_segment_ids": sorted(
                        set(
                            a["source_segment_ids"]
                            + b["source_segment_ids"]
                        )
                    ),
                    "centerline": {
                        "start": center_start.tolist(),
                        "end": center_end.tolist(),
                    },
                    "estimated_thickness_m": separation,
                    "overlap_length_m": overlap,
                    "overlap_ratio": overlap_ratio,
                    "angle_difference_degrees": angle,
                    "lateral_drift_m": drift,
                    "geometry_status": (
                        "parallel_boundary_hypothesis"
                    ),
                    "verified_wall": False,
                })

    return candidates


def render_preview(
    segments,
    chains,
    candidates,
    output_path,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 9),
    )

    for segment in segments:
        start = segment["start"]
        end = segment["end"]

        color = CATEGORY_COLORS.get(
            segment["category"],
            CATEGORY_COLORS["unknown"],
        )

        axes[0].plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=0.7,
            alpha=0.8,
        )

    axes[0].set_title(
        f"Original CAD segments ({len(segments)})"
    )

    for chain in chains:
        start = chain["start"]
        end = chain["end"]

        color = CATEGORY_COLORS.get(
            chain["category"],
            CATEGORY_COLORS["unknown"],
        )

        axes[1].plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=1.0,
            alpha=0.65,
        )

    for candidate in candidates:
        start = candidate["centerline"]["start"]
        end = candidate["centerline"]["end"]

        axes[1].plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#DC2626",
            linewidth=2.2,
            alpha=0.9,
        )

    axes[1].set_title(
        f"Merged chains ({len(chains)}) | "
        f"Parallel hypotheses ({len(candidates)})"
    )

    for axis in axes:
        axis.set_aspect(
            "equal",
            adjustable="box",
        )

        axis.set_xlabel("CAD X (m)")
        axis.set_ylabel("CAD Y (m)")
        axis.grid(
            True,
            alpha=0.15,
        )

    fig.suptitle(
        "BuildProgress | Wall Geometry Reconstruction",
        fontsize=15,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


def inspect(segments):
    categories = Counter(
        item["category"]
        for item in segments
    )

    lengths = np.asarray(
        [
            item["length"]
            for item in segments
        ],
        dtype=float,
    )

    print("\nCAD WALL SEGMENT INSPECTION")
    print("-" * 50)

    print(f"Segments: {len(segments)}")

    for category, count in sorted(
        categories.items()
    ):
        print(f"  {category}: {count}")

    print(
        f"Length range: "
        f"{lengths.min():.4f} to "
        f"{lengths.max():.4f} m"
    )

    print(
        f"Median length: "
        f"{np.median(lengths):.4f} m"
    )


def build(
    segments,
    output_dir,
    args,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nMerging collinear segments...")

    chains = merge_collinear_segments(
        segments,
        angle_tolerance_degrees=args.angle,
        distance_tolerance=args.distance,
        endpoint_tolerance=args.endpoint,
    )

    print(
        f"Original segments: {len(segments)}"
    )

    print(
        f"Merged chains: {len(chains)}"
    )

    print("\nFinding parallel wall boundaries...")

    candidates = detect_parallel_boundaries(
        chains,
        angle_tolerance_degrees=args.parallel_angle,
        minimum_separation=args.min_thickness,
        maximum_separation=args.max_thickness,
        minimum_overlap=args.min_overlap,
        minimum_overlap_ratio=args.min_overlap_ratio,
    )

    print(
        f"Parallel boundary hypotheses: "
        f"{len(candidates)}"
    )

    result = {
        "schema_version": "0.1",
        "source_geometry": str(
            args.input.resolve()
        ),
        "coordinate_system": {
            "units": "meters",
            "source": "DXF",
        },
        "method": (
            "conservative_collinear_merging_"
            "and_parallel_boundary_detection"
        ),
        "chains": chains,
        "wall_candidates": candidates,
        "notes": [
            "Candidates are geometric hypotheses.",
            "No candidate is a verified BIM wall.",
            "Door/window semantics are not inferred.",
            "Original CAD coordinates are preserved.",
            "All source segment IDs are retained.",
        ],
    }

    candidates_path = (
        output_dir / "wall_candidates.json"
    )

    candidates_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "original_segments": len(segments),
        "merged_chains": len(chains),
        "parallel_boundary_hypotheses": len(candidates),
        "chains_with_multiple_sources": sum(
            chain["source_count"] > 1
            for chain in chains
        ),
        "category_counts": dict(
            Counter(
                chain["category"]
                for chain in chains
            )
        ),
        "parameters": {
            "merge_angle_degrees": args.angle,
            "merge_distance_m": args.distance,
            "endpoint_tolerance_m": args.endpoint,
            "parallel_angle_degrees": args.parallel_angle,
            "min_thickness_m": args.min_thickness,
            "max_thickness_m": args.max_thickness,
            "min_overlap_m": args.min_overlap,
            "min_overlap_ratio": args.min_overlap_ratio,
        },
    }

    report_path = (
        output_dir / "reconstruction_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    preview_path = (
        output_dir / "reconstruction_preview.png"
    )

    render_preview(
        segments,
        chains,
        candidates,
        preview_path,
    )

    print("\nRECONSTRUCTION COMPLETE")
    print("-" * 50)

    print(f"Candidates: {candidates_path}")
    print(f"Report: {report_path}")
    print(f"Preview: {preview_path}")


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
        "--angle",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--endpoint",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--parallel-angle",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--min-thickness",
        type=float,
        default=0.06,
    )

    parser.add_argument(
        "--max-thickness",
        type=float,
        default=0.45,
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(
            f"Input file not found: {args.input}"
        )

    if not args.inspect and not args.build:
        parser.error(
            "Specify --inspect or --build."
        )

    segments = load_segments(
        args.input
    )

    if args.inspect:
        inspect(segments)

    if args.build:
        build(
            segments,
            args.output,
            args,
        )


if __name__ == "__main__":
    main()

