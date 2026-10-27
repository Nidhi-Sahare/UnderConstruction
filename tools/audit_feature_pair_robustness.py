
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    ROOT
    / "outputs"
    / "rotation_new_feature_validation"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "feature_pair_robustness"
)

REQUIRED_COLUMNS = [
    "frame_a",
    "frame_b",
    "gap",
    "matches",
    "original_median_px",
    "refined_median_px",
    "original_inlier_fraction",
    "refined_inlier_fraction",
    "original_rmse_px",
    "refined_rmse_px",
]


def load_pairs(path):
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        missing = set(REQUIRED_COLUMNS) - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                "Missing CSV columns: "
                + ", ".join(sorted(missing))
            )

        for line_number, raw in enumerate(reader, start=2):
            try:
                row = {
                    "frame_a": raw["frame_a"],
                    "frame_b": raw["frame_b"],
                    "gap": int(raw["gap"]),
                    "matches": int(raw["matches"]),
                }

                for key in REQUIRED_COLUMNS[4:]:
                    row[key] = float(raw[key])

            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Invalid data on CSV line {line_number}"
                ) from error

            numbers = [
                row[key]
                for key in REQUIRED_COLUMNS[4:]
            ]

            if not np.isfinite(numbers).all():
                raise ValueError(
                    f"Nonfinite data on line {line_number}"
                )

            if row["matches"] <= 0 or row["gap"] < 1:
                raise ValueError(
                    f"Invalid match count or gap "
                    f"on line {line_number}"
                )

            for key in (
                "original_inlier_fraction",
                "refined_inlier_fraction",
            ):
                if not 0.0 <= row[key] <= 1.0:
                    raise ValueError(
                        f"Invalid inlier fraction on "
                        f"line {line_number}"
                    )

            rows.append(row)

    if not rows:
        raise RuntimeError("The CSV contains no pairs.")

    return rows


def median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def percentile(values, q):
    return float(
        np.percentile(
            np.asarray(values, dtype=float),
            q,
        )
    )


def bootstrap_pair_delta(
    original,
    refined,
    seed,
    repetitions,
):
    """
    Bootstrap camera pairs, not individual matches.

    The statistic is the median of paired differences:
        refined - original

    Negative values indicate improvement.

    This is descriptive uncertainty only. Camera pairs
    share frames and are not statistically independent.
    """
    original = np.asarray(original, dtype=float)
    refined = np.asarray(refined, dtype=float)

    differences = refined - original

    if len(differences) < 2:
        return None

    rng = np.random.default_rng(seed)

    samples = np.empty(
        repetitions,
        dtype=np.float64,
    )

    for i in range(repetitions):
        indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )

        samples[i] = np.median(
            differences[indices]
        )

    return {
        "paired_median_delta_px": median(differences),
        "bootstrap_ci95_lower_px": percentile(
            samples, 2.5
        ),
        "bootstrap_ci95_upper_px": percentile(
            samples, 97.5
        ),
        "repetitions": repetitions,
        "interpretation": (
            "Negative deltas favor refined rotations. "
            "The interval is descriptive because pairs "
            "share cameras and observations."
        ),
    }


def summarize_group(
    rows,
    seed,
    repetitions,
):
    if not rows:
        return None

    original_medians = np.array([
        row["original_median_px"]
        for row in rows
    ])

    refined_medians = np.array([
        row["refined_median_px"]
        for row in rows
    ])

    original_inliers = np.array([
        row["original_inlier_fraction"]
        for row in rows
    ])

    refined_inliers = np.array([
        row["refined_inlier_fraction"]
        for row in rows
    ])

    original_rmse = np.array([
        row["original_rmse_px"]
        for row in rows
    ])

    refined_rmse = np.array([
        row["refined_rmse_px"]
        for row in rows
    ])

    weights = np.array([
        row["matches"]
        for row in rows
    ], dtype=float)

    median_improved = refined_medians < original_medians
    rmse_improved = refined_rmse < original_rmse
    inlier_improved = refined_inliers > original_inliers

    return {
        "pairs": len(rows),
        "total_correspondences": int(
            np.sum(weights)
        ),
        "median_error_px": {
            "original_pair_median": median(
                original_medians
            ),
            "refined_pair_median": median(
                refined_medians
            ),
            "median_paired_delta": median(
                refined_medians - original_medians
            ),
            "improved_pairs": int(
                np.sum(median_improved)
            ),
            "improved_fraction": float(
                np.mean(median_improved)
            ),
        },
        "inlier_fraction": {
            "original_pair_median": median(
                original_inliers
            ),
            "refined_pair_median": median(
                refined_inliers
            ),
            "median_paired_delta": median(
                refined_inliers - original_inliers
            ),
            "original_match_weighted_mean": float(
                np.average(
                    original_inliers,
                    weights=weights,
                )
            ),
            "refined_match_weighted_mean": float(
                np.average(
                    refined_inliers,
                    weights=weights,
                )
            ),
            "improved_pairs": int(
                np.sum(inlier_improved)
            ),
        },
        "rmse_px": {
            "original_pair_median": median(
                original_rmse
            ),
            "refined_pair_median": median(
                refined_rmse
            ),
            "median_paired_delta": median(
                refined_rmse - original_rmse
            ),
            "improved_pairs": int(
                np.sum(rmse_improved)
            ),
            "original_pair_p90": percentile(
                original_rmse, 90
            ),
            "refined_pair_p90": percentile(
                refined_rmse, 90
            ),
            "original_maximum": float(
                np.max(original_rmse)
            ),
            "refined_maximum": float(
                np.max(refined_rmse)
            ),
        },
        "bootstrap_pair_median_delta": (
            bootstrap_pair_delta(
                original_medians,
                refined_medians,
                seed,
                repetitions,
            )
        ),
    }


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pair-balanced robustness audit of "
            "rotation refinement."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=(
            INPUT_DIR
            / "new_feature_pair_results.csv"
        ),
    )

    parser.add_argument(
        "--min-matches",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--min-original-inlier",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.min_matches < 1:
        parser.error(
            "--min-matches must be positive"
        )

    if not 0 <= args.min_original_inlier <= 1:
        parser.error(
            "--min-original-inlier must be "
            "between 0 and 1"
        )

    if args.bootstrap < 100:
        parser.error(
            "--bootstrap must be at least 100"
        )

    rows = load_pairs(args.input)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    supported = [
        row
        for row in rows
        if (
            row["matches"] >= args.min_matches
            and row["original_inlier_fraction"]
            >= args.min_original_inlier
        )
    ]

    unsupported = [
        row
        for row in rows
        if (
            row["matches"] < args.min_matches
            or row["original_inlier_fraction"]
            < args.min_original_inlier
        )
    ]

    all_summary = summarize_group(
        rows,
        args.seed,
        args.bootstrap,
    )

    supported_summary = summarize_group(
        supported,
        args.seed + 1,
        args.bootstrap,
    )

    unsupported_summary = summarize_group(
        unsupported,
        args.seed + 2,
        args.bootstrap,
    )

    by_gap = {}

    supported_by_gap = {}

    for gap in sorted(
        set(row["gap"] for row in rows)
    ):
        group = [
            row
            for row in rows
            if row["gap"] == gap
        ]

        supported_group = [
            row
            for row in supported
            if row["gap"] == gap
        ]

        by_gap[str(gap)] = summarize_group(
            group,
            args.seed + 10 + gap,
            args.bootstrap,
        )

        supported_by_gap[str(gap)] = summarize_group(
            supported_group,
            args.seed + 20 + gap,
            args.bootstrap,
        )

    diagnostics = []

    for row in rows:
        record = dict(row)

        record["supported_by_original"] = (
            row in supported
        )

        record["median_delta_px"] = (
            row["refined_median_px"]
            - row["original_median_px"]
        )

        record["rmse_delta_px"] = (
            row["refined_rmse_px"]
            - row["original_rmse_px"]
        )

        record["inlier_delta"] = (
            row["refined_inlier_fraction"]
            - row["original_inlier_fraction"]
        )

        diagnostics.append(record)

    diagnostics.sort(
        key=lambda row: row["rmse_delta_px"],
        reverse=True,
    )

    diagnostic_path = (
        OUTPUT_DIR / "pair_diagnostics.csv"
    )

    save_csv(
        diagnostic_path,
        diagnostics,
        [
            *REQUIRED_COLUMNS,
            "supported_by_original",
            "median_delta_px",
            "rmse_delta_px",
            "inlier_delta",
        ],
    )

    worst_original = sorted(
        rows,
        key=lambda row: row["original_rmse_px"],
        reverse=True,
    )[:10]

    worst_regressions = sorted(
        diagnostics,
        key=lambda row: row["rmse_delta_px"],
        reverse=True,
    )[:10]

    # Exploratory diagnostic gate. It is not a
    # reconstruction or deployment approval.
    supported_pass = False

    if supported_summary is not None:
        metrics = supported_summary

        supported_pass = bool(
            metrics["pairs"] >= 20
            and metrics["median_error_px"][
                "improved_fraction"
            ] >= 0.60
            and metrics["median_error_px"][
                "median_paired_delta"
            ] < 0
            and metrics["inlier_fraction"][
                "median_paired_delta"
            ] > 0
            and metrics["rmse_px"][
                "median_paired_delta"
            ] <= 0
        )

    report = {
        "project": "Under-construction",
        "experiment": "FEATURE_PAIR_ROBUSTNESS_V1",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": str(args.input),
        "parameters": vars(args) | {
            "input": str(args.input)
        },
        "all_pairs": all_summary,
        "supported_pairs": supported_summary,
        "unsupported_pairs": unsupported_summary,
        "by_frame_gap": by_gap,
        "supported_by_frame_gap": supported_by_gap,
        "worst_original_rmse_pairs": worst_original,
        "largest_rmse_regressions": worst_regressions,
        "summary": {
            "total_pairs": len(rows),
            "supported_pairs": len(supported),
            "unsupported_pairs": len(unsupported),
            "supported_pair_screening_pass": (
                supported_pass
            ),
            "approved_for_application": False,
        },
        "limitations": [
            "Input contains pair-level summaries, "
            "not individual correspondence errors.",
            "Exact capped-error RMSE cannot be "
            "computed from this CSV.",
            "Median of pair RMSE values is not "
            "the pooled correspondence RMSE.",
            "Supported pairs are selected using "
            "original-pose inlier fraction and "
            "match count only.",
            "The support threshold is an "
            "exploratory choice.",
            "Camera pairs share frames and are "
            "not statistically independent.",
            "Bootstrap intervals do not account "
            "for all camera-level dependence.",
            "No new features are extracted.",
            "No new landmarks are triangulated.",
            "Camera extrinsics and CAD alignment "
            "remain unverified.",
            "V2 is not modified.",
        ],
    }

    report_path = (
        OUTPUT_DIR
        / "feature_pair_robustness_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nUNDER-CONSTRUCTION")
    print("Feature Pair Robustness Audit")
    print("-" * 60)

    print("Total pairs:", len(rows))
    print("Supported pairs:", len(supported))
    print("Unsupported pairs:", len(unsupported))

    def show(label, result):
        print(f"\n{label}")

        if result is None:
            print("No pairs available.")
            return

        med = result["median_error_px"]
        inliers = result["inlier_fraction"]
        rmse = result["rmse_px"]

        print(
            "Pairs:",
            result["pairs"],
            "| matches:",
            result["total_correspondences"],
        )

        print(
            "Pair-median error:",
            f"{med['original_pair_median']:.4f}",
            "->",
            f"{med['refined_pair_median']:.4f}",
            "px",
        )

        print(
            "Median paired error delta:",
            f"{med['median_paired_delta']:+.4f} px",
        )

        print(
            "Median paired inlier delta:",
            f"{inliers['median_paired_delta']:+.4f}",
        )

        print(
            "Median paired RMSE delta:",
            f"{rmse['median_paired_delta']:+.4f} px",
        )

        print(
            "Improved median-error pairs:",
            f"{med['improved_pairs']}/{result['pairs']}",
        )

        ci = result["bootstrap_pair_median_delta"]

        if ci:
            print(
                "Descriptive bootstrap interval:",
                f"[{ci['bootstrap_ci95_lower_px']:.4f},",
                f"{ci['bootstrap_ci95_upper_px']:.4f}] px",
            )

    show("ALL PAIRS", all_summary)
    show("SUPPORTED PAIRS", supported_summary)
    show("UNSUPPORTED PAIRS", unsupported_summary)

    print("\nBY FRAME GAP")

    for gap, result in by_gap.items():
        if result is None:
            continue

        med = result["median_error_px"]

        print(
            f"Gap {gap}:",
            f"{med['improved_pairs']}/{result['pairs']}",
            "pairs improved",
            "| median paired delta:",
            f"{med['median_paired_delta']:+.4f} px",
        )

    print("\nSCREENING")
    print(
        "Supported-pair result:",
        "PASS" if supported_pass else "FAIL",
    )

    print("Approved for application: NO")

    print("\nReport:", report_path)
    print("Diagnostics:", diagnostic_path)
    print(
        "\nOriginal V2 reconstruction remains unchanged."
    )


if __name__ == "__main__":
    main()
