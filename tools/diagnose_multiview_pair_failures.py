
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import reconstruct_multiview_v3 as v3


ROOT = Path(__file__).resolve().parent.parent

OUTPUT = (
    ROOT
    / "outputs"
    / "multiview_pair_diagnostics"
)


def number(name):
    return v3.frame_number(name)


def save_csv(path, rows, columns):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = np.asarray(values, dtype=float)

    if not len(values):
        return {
            "median": None,
            "p90": None,
            "max": None,
        }

    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--width", type=int, default=960
    )
    parser.add_argument(
        "--features", type=int, default=3000
    )
    parser.add_argument(
        "--max-gap", type=int, default=4
    )
    parser.add_argument(
        "--ratio", type=float, default=0.78
    )
    parser.add_argument(
        "--exclude-radius-px",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--min-pair-matches",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min-original-inlier",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--epipolar-threshold",
        type=float,
        default=2.5,
    )

    args = parser.parse_args()

    if (
        args.width < 100
        or args.features < 100
        or args.max_gap < 1
        or not 0 < args.ratio < 1
        or args.exclude_radius_px <= 0
        or args.min_pair_matches < 2
        or not 0 <= args.min_original_inlier <= 1
        or args.epipolar_threshold <= 0
    ):
        parser.error("Invalid parameters.")

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    poses = v3.load_poses()
    exclusions, excluded_count = (
        v3.load_exclusions()
    )

    names = list(poses)

    sift = cv2.SIFT_create(
        nfeatures=args.features
    )

    features = {}

    print("\nUNDER-CONSTRUCTION")
    print("Multiview Pair Failure Diagnostics")
    print("-" * 60)
    print("Supported cameras:", len(names))

    for index, name in enumerate(names, 1):
        features[name] = v3.extract_features(
            name,
            sift,
            args.width,
            exclusions,
            args.exclude_radius_px / v3.FOCAL,
        )

        if index % 15 == 0 or index == len(names):
            print(
                f"Features extracted: {index}/{len(names)}"
            )

    pair_rows = []
    accepted_edges = []

    frame_pair_stats = defaultdict(Counter)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            gap = number(b) - number(a)

            if gap > args.max_gap:
                break

            pa = poses[a]
            pb = poses[b]

            baseline = float(
                np.linalg.norm(
                    pb["center"] - pa["center"]
                )
            )

            if not 0.08 <= baseline <= 5.0:
                continue

            matches = v3.mutual_matches(
                features[a],
                features[b],
                args.ratio,
            )

            match_count = len(matches)

            inlier_count = 0
            inlier_fraction = 0.0
            median_epipolar = None
            accepted = False

            if match_count:
                ia = np.asarray(
                    [m[0] for m in matches],
                    dtype=int,
                )

                ib = np.asarray(
                    [m[1] for m in matches],
                    dtype=int,
                )

                errors = v3.epipolar_errors(
                    features[a]["uv"][ia],
                    features[b]["uv"][ib],
                    pa,
                    pb,
                )

                mask = (
                    errors
                    <= args.epipolar_threshold
                )

                inlier_count = int(
                    np.count_nonzero(mask)
                )

                inlier_fraction = float(
                    np.mean(mask)
                )

                finite = errors[
                    np.isfinite(errors)
                ]

                if len(finite):
                    median_epipolar = float(
                        np.median(finite)
                    )

                accepted = (
                    match_count >= args.min_pair_matches
                    and inlier_fraction
                    >= args.min_original_inlier
                )

                if accepted:
                    for index in np.flatnonzero(mask):
                        fa, fb, distance = matches[index]

                        accepted_edges.append((
                            float(errors[index]),
                            float(distance),
                            (a, int(fa)),
                            (b, int(fb)),
                        ))

            if match_count < args.min_pair_matches:
                reason = "insufficient_matches"
            elif (
                inlier_fraction
                < args.min_original_inlier
            ):
                reason = "pose_disagreement"
            else:
                reason = "accepted"

            pair_rows.append({
                "frame_a": a,
                "frame_b": b,
                "gap": gap,
                "baseline": baseline,
                "matches": match_count,
                "epipolar_inliers": inlier_count,
                "inlier_fraction": inlier_fraction,
                "median_epipolar_px": median_epipolar,
                "accepted": accepted,
                "reason": reason,
            })

            for name in (a, b):
                frame_pair_stats[name][
                    "eligible_pairs"
                ] += 1

                frame_pair_stats[name][
                    "descriptor_matches"
                ] += match_count

                frame_pair_stats[name][
                    "epipolar_inliers"
                ] += inlier_count

                if accepted:
                    frame_pair_stats[name][
                        "accepted_pairs"
                    ] += 1

                frame_pair_stats[name][
                    reason
                ] += 1

    accepted_edges.sort(
        key=lambda edge: (
            edge[0],
            edge[1],
            edge[2],
            edge[3],
        )
    )

    union = v3.ConflictAwareUnionFind()

    for _, _, a, b in accepted_edges:
        union.union(a, b)

    components = union.components()

    tracks = [
        component
        for component in components
        if len(component) >= 3
    ]

    candidate_counts = Counter()
    candidate_observations = Counter()

    for track in tracks:
        for name, _ in track:
            candidate_counts[name] += 1
            candidate_observations[name] += 1

    # Also measure which cameras participate in
    # two-view components but fail to form 3+ views.
    two_view_counts = Counter()

    for component in components:
        if len(component) == 2:
            for name, _ in component:
                two_view_counts[name] += 1

    # Read the previous V3 coverage report to identify
    # the exact cameras we are trying to recover.
    coverage_path = (
        ROOT
        / "outputs"
        / "multiview_v3_coverage"
        / "coverage_audit_report.json"
    )

    if coverage_path.exists():
        coverage = json.loads(
            coverage_path.read_text(
                encoding="utf-8"
            )
        )

        previously_uncovered = set(
            coverage["coverage"]["refined"][
                "zero_landmark_frames"
            ]
        )
    else:
        previously_uncovered = set()

    frame_rows = []

    for name in names:
        stats = frame_pair_stats[name]

        if candidate_counts[name] > 0:
            classification = "has_multiview_tracks"

        elif stats["accepted_pairs"] > 0:
            classification = "track_fragmentation"

        elif stats["pose_disagreement"] > 0:
            classification = "pose_disagreement"

        elif stats["insufficient_matches"] > 0:
            classification = "insufficient_matches"

        else:
            classification = "no_eligible_pairs"

        frame_rows.append({
            "frame": name,
            "frame_number": number(name),
            "previously_uncovered": (
                name in previously_uncovered
            ),
            "extracted_features": len(
                features[name]["uv"]
            ),
            "eligible_pairs": stats[
                "eligible_pairs"
            ],
            "accepted_pairs": stats[
                "accepted_pairs"
            ],
            "insufficient_match_pairs": stats[
                "insufficient_matches"
            ],
            "pose_disagreement_pairs": stats[
                "pose_disagreement"
            ],
            "descriptor_matches": stats[
                "descriptor_matches"
            ],
            "epipolar_inliers": stats[
                "epipolar_inliers"
            ],
            "two_view_components": (
                two_view_counts[name]
            ),
            "multiview_candidate_tracks": (
                candidate_counts[name]
            ),
            "classification": classification,
        })

    pair_csv = OUTPUT / "pair_diagnostics.csv"
    frame_csv = OUTPUT / "frame_diagnostics.csv"

    save_csv(
        pair_csv,
        pair_rows,
        [
            "frame_a",
            "frame_b",
            "gap",
            "baseline",
            "matches",
            "epipolar_inliers",
            "inlier_fraction",
            "median_epipolar_px",
            "accepted",
            "reason",
        ],
    )

    save_csv(
        frame_csv,
        frame_rows,
        list(frame_rows[0]),
    )

    reason_counts = Counter(
        row["reason"]
        for row in pair_rows
    )

    classification_counts = Counter(
        row["classification"]
        for row in frame_rows
    )

    uncovered_classification = Counter(
        row["classification"]
        for row in frame_rows
        if row["previously_uncovered"]
    )

    report = {
        "project": "Under-construction",
        "experiment": (
            "MULTIVIEW_PAIR_FAILURE_DIAGNOSTICS"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "parameters": vars(args),
        "supported_cameras": len(names),
        "excluded_v2_observations": (
            excluded_count
        ),
        "eligible_pairs": len(pair_rows),
        "pair_reasons": dict(reason_counts),
        "accepted_feature_edges": len(
            accepted_edges
        ),
        "prevented_conflicting_merges": (
            union.conflicts
        ),
        "feature_components": len(components),
        "candidate_three_view_tracks": len(
            tracks
        ),
        "frame_classifications": dict(
            classification_counts
        ),
        "previously_uncovered_count": len(
            previously_uncovered
        ),
        "previously_uncovered_classifications": dict(
            uncovered_classification
        ),
        "previously_uncovered_frames": [
            row
            for row in frame_rows
            if row["previously_uncovered"]
        ],
        "descriptor_match_statistics": summarize(
            [row["matches"] for row in pair_rows]
        ),
        "inlier_fraction_statistics": summarize(
            [
                row["inlier_fraction"]
                for row in pair_rows
            ]
        ),
        "limitations": [
            "Pair eligibility uses provisional "
            "camera centers and the V3 baseline gate.",
            "Only the original pose model is used "
            "for epipolar filtering.",
            "Pair classifications depend on the "
            "order of validation checks.",
            "Pose disagreement does not establish "
            "that the poses alone are incorrect.",
            "Camera-to-IMU extrinsics remain "
            "unverified.",
            "Only 69 previously supported cameras "
            "are evaluated.",
            "No reconstruction thresholds are "
            "relaxed.",
            "No camera poses or landmarks are "
            "modified.",
        ],
    }

    report_path = (
        OUTPUT
        / "pair_failure_diagnostics_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nPAIR DIAGNOSTICS")
    print("-" * 60)

    print("Eligible pairs:", len(pair_rows))

    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")

    print(
        "Accepted feature edges:",
        len(accepted_edges),
    )

    print(
        "Three-view candidate tracks:",
        len(tracks),
    )

    print(
        "Conflicting merges prevented:",
        union.conflicts,
    )

    print("\nPREVIOUSLY UNCOVERED CAMERAS")
    print(
        "Count:",
        len(previously_uncovered),
    )

    for classification, count in (
        uncovered_classification.most_common()
    ):
        print(
            f"  {classification}: {count}"
        )

    print("\nOutputs:")
    print(report_path)
    print(pair_csv)
    print(frame_csv)

    print(
        "\nOriginal V2/V3 reconstructions unchanged."
    )


if __name__ == "__main__":
    main()
