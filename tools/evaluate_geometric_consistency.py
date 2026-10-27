
"""
Under-construction — Step 101
Observability-aware geometric consistency evaluation.

Reanalyses frozen Step 100 results. No OpenCV required.

Inputs:
    outputs/line_perturbation_test/
        per_candidate_perturbations.json
        perturbation_report.json

Outputs:
    outputs/geometric_consistency/
        consistency_report.json
        per_candidate_consistency.json

Principles:
    - No changes to geometry or previous results.
    - Absolute image error is a separate gate.
    - Evaluate perturbation direction consistency.
    - Distinguish error separability from true observability.
    - Count non-overlapping windows, not controls, as
      independent temporal evidence units.
    - Explicitly abstain when evidence is insufficient.

Run:
    .\\.venv\\Scripts\\python.exe tools\\evaluate_geometric_consistency.py
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = ROOT / "outputs" / "line_perturbation_test"
OUTPUT_DIR = ROOT / "outputs" / "geometric_consistency"

DETAIL_PATH = INPUT_DIR / "per_candidate_perturbations.json"
REPORT_PATH = INPUT_DIR / "perturbation_report.json"

ABSOLUTE_ERROR_LIMIT_PX = 8.0
IMPROVEMENT_MARGIN_PX = 2.0

MIN_NONOVERLAPPING_WINDOWS = 2
MIN_CONSISTENT_WINDOWS = 2

# A control must improve the error by at least this margin
# in multiple distinct, non-overlapping windows before it
# is called a repeatable local geometric challenge.
#
# The threshold is descriptive, not a significance test.

CONTROL_PATTERN = re.compile(
    r"^(shift|rotate)_([uv])_([+-]?[0-9.]+)(m|deg)$"
)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def finite_number(value):
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def frame_index(name):
    try:
        return int(
            Path(name).stem.split("_")[-1]
        )
    except (TypeError, ValueError, IndexError):
        return None


def window_interval(window):
    names = window.get("frames", [])

    indices = [
        frame_index(name)
        for name in names
    ]

    if (
        len(indices) < 2
        or any(x is None for x in indices)
    ):
        return None

    return (
        min(indices),
        max(indices),
    )


def nonoverlapping_selection(windows):
    """
    Select the maximum number of non-overlapping frame
    intervals using earliest-finish-time scheduling.

    A window spanning frames 168,169,170 overlaps a window
    spanning 170,171,172 because frame 170 is shared.
    """
    indexed = []

    for index, window in enumerate(windows):
        interval = window_interval(window)

        if interval is not None:
            indexed.append((
                interval[1],
                interval[0],
                index,
            ))

    indexed.sort()

    chosen = []
    last_end = -math.inf

    for end, start, index in indexed:
        if start > last_end:
            chosen.append(index)
            last_end = end

    return chosen


def parse_control(name):
    match = CONTROL_PATTERN.fullmatch(
        str(name)
    )

    if match is None:
        return None

    family, axis, signed, unit = match.groups()

    try:
        value = float(signed)
    except ValueError:
        return None

    return {
        "family": family,
        "axis": axis,
        "signed_amount": value,
        "unit": unit,
        "direction": (
            -1 if value < 0 else 1
        ),
        "key": (
            f"{family}_{axis}_{value:+.3f}{unit}"
        ),
    }


def evaluate_window(window):
    original = window.get(
        "original", {}
    )

    original_error = finite_number(
        original.get(
            "mean_future_frame_error_px"
        )
    )

    if original_error is None:
        return None

    controls = []

    for item in window.get(
        "perturbations", []
    ):
        alternative_error = finite_number(
            item.get("mean_error_px")
        )

        parsed = parse_control(
            item.get("name")
        )

        if (
            alternative_error is None
            or parsed is None
        ):
            continue

        improvement = (
            original_error - alternative_error
        )

        if improvement >= IMPROVEMENT_MARGIN_PX:
            classification = "ALTERNATIVE_BETTER"

        elif improvement <= -IMPROVEMENT_MARGIN_PX:
            classification = "ORIGINAL_BETTER"

        else:
            classification = "ERROR_INDISTINGUISHABLE"

        controls.append({
            **parsed,
            "alternative_error_px": alternative_error,
            "improvement_px": improvement,
            "classification": classification,
        })

    better = [
        item
        for item in controls
        if item["classification"]
        == "ALTERNATIVE_BETTER"
    ]

    original_wins = [
        item
        for item in controls
        if item["classification"]
        == "ORIGINAL_BETTER"
    ]

    best = (
        max(
            controls,
            key=lambda item: item[
                "improvement_px"
            ],
        )
        if controls
        else None
    )

    if original_error > ABSOLUTE_ERROR_LIMIT_PX:
        decision = "ABSOLUTE_FIT_FAILED"

    elif better:
        decision = "LOCAL_ALTERNATIVE_IMPROVES_FIT"

    elif controls:
        decision = "NO_DECISIVE_ALTERNATIVE"

    else:
        decision = "NO_COMPARABLE_CONTROLS"

    return {
        "frames": window.get(
            "frames", []
        ),
        "frame_interval": (
            window_interval(window)
        ),
        "original_error_px": original_error,
        "absolute_fit_pass": (
            original_error
            <= ABSOLUTE_ERROR_LIMIT_PX
        ),
        "control_count": len(controls),
        "alternative_better_count": len(better),
        "original_better_count": len(original_wins),
        "error_indistinguishable_count": (
            len(controls)
            - len(better)
            - len(original_wins)
        ),
        "best_alternative": best,
        "decision": decision,
        "controls": controls,
    }


def consistent_control_analysis(windows):
    """
    Analyze exact same perturbation across windows.

    This avoids incorrectly combining different directions
    or magnitudes into a single apparent trend.
    """
    grouped = defaultdict(list)

    for window in windows:
        for control in window["controls"]:
            grouped[
                control["key"]
            ].append({
                "frames": window["frames"],
                "interval": (
                    window["frame_interval"]
                ),
                "improvement_px": (
                    control["improvement_px"]
                ),
                "classification": (
                    control["classification"]
                ),
            })

    summaries = []

    for key, observations in grouped.items():
        better = [
            x for x in observations
            if x["classification"]
            == "ALTERNATIVE_BETTER"
        ]

        worse = [
            x for x in observations
            if x["classification"]
            == "ORIGINAL_BETTER"
        ]

        # Select a non-overlapping subset specifically among
        # the windows where this control improved fit.
        nonoverlap_indices = nonoverlapping_selection([
            {"frames": x["frames"]}
            for x in better
        ])

        nonoverlap_better = [
            better[i]
            for i in nonoverlap_indices
        ]

        improvements = [
            x["improvement_px"]
            for x in observations
        ]

        summaries.append({
            "control": key,
            "windows_observed": len(
                observations
            ),
            "windows_alternative_better": len(
                better
            ),
            "windows_original_better": len(
                worse
            ),
            "windows_error_indistinguishable": (
                len(observations)
                - len(better)
                - len(worse)
            ),
            "nonoverlapping_improvement_windows": (
                len(nonoverlap_better)
            ),
            "median_improvement_px": (
                median(improvements)
            ),
            "minimum_improvement_px": (
                min(improvements)
            ),
            "maximum_improvement_px": (
                max(improvements)
            ),
            "repeatable_challenge": (
                len(nonoverlap_better)
                >= MIN_CONSISTENT_WINDOWS
            ),
            "improving_frame_windows": [
                x["frames"]
                for x in better
            ],
        })

    summaries.sort(
        key=lambda item: (
            -item[
                "nonoverlapping_improvement_windows"
            ],
            -item[
                "median_improvement_px"
            ],
            item["control"],
        )
    )

    return summaries


def evaluate_candidate(candidate):
    completed = [
        item
        for item in candidate.get(
            "windows", []
        )
        if item.get("status")
        == "PERTURBATION_COMPARISON_COMPLETED"
    ]

    evaluated = []

    for window in completed:
        result = evaluate_window(window)

        if result is not None:
            evaluated.append(result)

    selected_indices = nonoverlapping_selection(
        evaluated
    )

    independent_windows = [
        evaluated[i]
        for i in selected_indices
    ]

    control_summaries = consistent_control_analysis(
        evaluated
    )

    repeatable = [
        item
        for item in control_summaries
        if item["repeatable_challenge"]
    ]

    absolute_failures = [
        item
        for item in evaluated
        if not item["absolute_fit_pass"]
    ]

    independent_absolute_failures = [
        item
        for item in independent_windows
        if not item["absolute_fit_pass"]
    ]

    windows_with_better_alternatives = [
        item
        for item in evaluated
        if item["alternative_better_count"] > 0
    ]

    independent_with_better = [
        item
        for item in independent_windows
        if item["alternative_better_count"] > 0
    ]

    total_controls = sum(
        item["control_count"]
        for item in evaluated
    )

    indistinguishable = sum(
        item["error_indistinguishable_count"]
        for item in evaluated
    )

    separability_fraction = (
        (total_controls - indistinguishable)
        / total_controls
        if total_controls
        else None
    )

    # Important: "separability" here means error difference
    # exceeds 2 px. It is NOT a geometric observability score.
    if not evaluated:
        decision = "INSUFFICIENT_EVIDENCE"

    elif independent_absolute_failures:
        decision = "ABSOLUTE_ALIGNMENT_CONCERN"

    elif repeatable:
        decision = "REPEATABLE_LOCAL_GEOMETRIC_CHALLENGE"

    elif (
        len(independent_windows)
        < MIN_NONOVERLAPPING_WINDOWS
    ):
        decision = "INSUFFICIENT_INDEPENDENT_WINDOWS"

    elif independent_with_better:
        decision = "LOCAL_GEOMETRY_UNRESOLVED"

    elif separability_fraction is None:
        decision = "INSUFFICIENT_EVIDENCE"

    else:
        decision = "NO_DECISIVE_CHALLENGE_OBSERVED"

    # This is a diagnostic decision, not a verified status.
    return {
        "candidate_id": candidate[
            "candidate_id"
        ],
        "group_id": candidate.get(
            "group_id"
        ),
        "step100_decision_unchanged": (
            candidate.get("decision")
        ),
        "windows_selected_step100": (
            candidate.get(
                "windows_selected", 0
            )
        ),
        "windows_completed": len(evaluated),
        "nonoverlapping_windows": len(
            independent_windows
        ),
        "original_error_median_px": (
            median([
                x["original_error_px"]
                for x in evaluated
            ])
            if evaluated
            else None
        ),
        "original_error_max_px": (
            max([
                x["original_error_px"]
                for x in evaluated
            ])
            if evaluated
            else None
        ),
        "absolute_fit_failure_windows": (
            len(absolute_failures)
        ),
        "nonoverlapping_absolute_fit_failures": (
            len(independent_absolute_failures)
        ),
        "windows_with_better_alternatives": (
            len(windows_with_better_alternatives)
        ),
        "nonoverlapping_windows_with_better_alternatives": (
            len(independent_with_better)
        ),
        "repeatable_challenge_controls": [
            x["control"]
            for x in repeatable
        ],
        "control_comparison_count": total_controls,
        "error_indistinguishable_fraction": (
            indistinguishable / total_controls
            if total_controls
            else None
        ),
        "error_separability_fraction": (
            separability_fraction
        ),
        "geometric_observability": (
            "NOT_MEASURED"
        ),
        "decision": decision,
        "physical_line_verified": False,
        "independent_windows": [
            x["frames"]
            for x in independent_windows
        ],
        "window_results": evaluated,
        "control_consistency": (
            control_summaries
        ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_report = read_json(
        REPORT_PATH
    )

    source_detail = read_json(
        DETAIL_PATH
    )

    candidates = source_detail.get(
        "results", []
    )

    results = [
        evaluate_candidate(candidate)
        for candidate in candidates
    ]

    decisions = Counter(
        row["decision"]
        for row in results
    )

    report = {
        "method": (
            "step101_observability_aware_"
            "geometric_consistency"
        ),
        "source_method": source_report.get(
            "method"
        ),
        "candidate_count": len(results),
        "candidate_decision_counts": dict(
            decisions
        ),
        "candidate_decisions": {
            str(row["candidate_id"]): row[
                "decision"
            ]
            for row in results
        },
        "total_completed_windows": sum(
            row["windows_completed"]
            for row in results
        ),
        "total_nonoverlapping_windows": sum(
            row["nonoverlapping_windows"]
            for row in results
        ),
        "total_control_comparisons": sum(
            row["control_comparison_count"]
            for row in results
        ),
        "candidates_with_absolute_fit_failures": [
            row["candidate_id"]
            for row in results
            if row[
                "absolute_fit_failure_windows"
            ] > 0
        ],
        "candidates_with_repeatable_challenges": [
            row["candidate_id"]
            for row in results
            if row[
                "repeatable_challenge_controls"
            ]
        ],
        "parameters": {
            "absolute_error_limit_px": (
                ABSOLUTE_ERROR_LIMIT_PX
            ),
            "improvement_margin_px": (
                IMPROVEMENT_MARGIN_PX
            ),
            "minimum_nonoverlapping_windows": (
                MIN_NONOVERLAPPING_WINDOWS
            ),
            "minimum_consistent_windows": (
                MIN_CONSISTENT_WINDOWS
            ),
        },
        "geometric_observability_measured": False,
        "error_separability_measured": True,
        "previous_decisions_modified": False,
        "geometry_modified": False,
        "physical_lines_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Reanalysis uses existing Step 100 results and introduces no new visual observations.",
            "A 2 px difference in reprojection error is an operational threshold, not a calibrated uncertainty bound.",
            "True geometric observability cannot be calculated because Step 100 did not save projected perturbation displacement or camera-baseline sensitivity.",
            "Non-overlapping windows do not guarantee independent evidence because they can track the same physical features and share camera-pose errors.",
            "The exact same control is compared across windows, but the source feature association is conditioned on the original hypothesis.",
            "A repeated alternative improvement suggests local geometric bias but does not prove the alternative corresponds to a physical line.",
            "Absolute image error alone cannot establish structural identity.",
            "No CAD registration or construction-progress measurement is performed.",
        ],
        "decision": (
            "GEOMETRIC_CONSISTENCY_DIAGNOSTICS_AVAILABLE"
        ),
    }

    write_json(
        OUTPUT_DIR / "per_candidate_consistency.json",
        {
            "method": report["method"],
            "results": results,
        },
    )

    write_json(
        OUTPUT_DIR / "consistency_report.json",
        report,
    )

    print("\n=== STEP 101 ===")
    print(
        "Candidates:",
        report["candidate_count"]
    )
    print(
        "Completed windows:",
        report["total_completed_windows"]
    )
    print(
        "Non-overlapping windows:",
        report["total_nonoverlapping_windows"]
    )
    print(
        "Decisions:",
        dict(decisions)
    )

    for row in results:
        print(
            f"Candidate {row['candidate_id']:03d}: "
            f"completed={row['windows_completed']} "
            f"nonoverlap={row['nonoverlapping_windows']} "
            f"median_error={row['original_error_median_px']} "
            f"repeatable_controls="
            f"{len(row['repeatable_challenge_controls'])} "
            f"=> {row['decision']}"
        )

    print(
        "True geometric observability measured: NO"
    )
    print(
        "Physical lines verified: NO"
    )
    print(
        "CAD registration verified: NO"
    )
    print(
        "Construction progress verified: NO"
    )
    print(
        "Output:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
