
"""
Under-construction — Step 84
Registration negative-control and coverage audit.

Reads the Step 81 evaluation report and produces:
- Candidate score and valid-frame coverage
- Pairwise comparisons with every negative control
- Missing-frame penalties
- Development-only candidate selection
- Held-out evaluation of that selected candidate
- Explicit abstention

No image processing, no manual annotation, no LLMs.
No registration is promoted to verified.

Run:
    python tools/audit_registration_controls.py
"""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "outputs"
    / "registration_nearest_edges"
    / "evaluation_report.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "registration_control_audit"
)

OUTPUT = OUTPUT_DIR / "control_audit.json"

MIN_PAIRED_FRAMES = 5
MIN_COVERAGE = 0.70
MIN_WIN_FRACTION = 0.70

# These are diagnostic thresholds, not calibrated
# probabilities or guarantees of correct registration.
MIN_MEAN_ADVANTAGE = 0.01


def frame_scores(partition):
    result = {}

    for row in partition.get("frame_results", []):
        name = row.get("frame")
        value = row.get("score")

        if name is None:
            continue

        if value is None:
            result[name] = None
            continue

        value = float(value)

        result[name] = (
            value
            if np.isfinite(value)
            else None
        )

    return result


def summarize_partition(partition):
    scores = frame_scores(partition)

    total = len(scores)

    valid = [
        score
        for score in scores.values()
        if score is not None
    ]

    coverage = (
        len(valid) / total
        if total
        else 0.0
    )

    mean_valid = (
        float(np.mean(valid))
        if valid
        else None
    )

    # Missing observations are not silently dropped.
    # Zero is a diagnostic failure penalty, not a
    # measured image-alignment score.
    coverage_adjusted = (
        float(sum(valid) / total)
        if total
        else None
    )

    return {
        "total_frames": total,
        "valid_frames": len(valid),
        "missing_frames": total - len(valid),
        "coverage_fraction": coverage,
        "mean_valid_score": mean_valid,
        "coverage_adjusted_score": (
            coverage_adjusted
        ),
    }


def compare_pair(candidate, control):
    a = frame_scores(candidate)
    b = frame_scores(control)

    names = sorted(set(a) & set(b))

    paired = [
        (
            name,
            a[name],
            b[name],
        )
        for name in names
        if a[name] is not None
        and b[name] is not None
    ]

    differences = np.array(
        [
            x - y
            for _, x, y in paired
        ],
        dtype=np.float64,
    )

    total_frames = len(names)
    paired_count = len(paired)

    coverage = (
        paired_count / total_frames
        if total_frames
        else 0.0
    )

    if paired_count:
        mean_difference = float(
            differences.mean()
        )

        win_fraction = float(
            np.mean(differences > 0)
        )

        median_difference = float(
            np.median(differences)
        )
    else:
        mean_difference = None
        win_fraction = None
        median_difference = None

    sufficient = (
        paired_count >= MIN_PAIRED_FRAMES
        and coverage >= MIN_COVERAGE
    )

    passed = (
        sufficient
        and mean_difference
        >= MIN_MEAN_ADVANTAGE
        and win_fraction
        >= MIN_WIN_FRACTION
    )

    if not sufficient:
        status = "INSUFFICIENT_PAIRED_COVERAGE"
    elif passed:
        status = "PASSES_EXPLORATORY_PAIR_GATE"
    else:
        status = "FAILS_PAIR_GATE"

    return {
        "status": status,
        "total_common_frames": total_frames,
        "paired_valid_frames": paired_count,
        "paired_coverage_fraction": coverage,
        "mean_score_advantage": mean_difference,
        "median_score_advantage": (
            median_difference
        ),
        "frame_win_fraction": win_fraction,
        "candidate_coverage": (
            summarize_partition(candidate)
        ),
        "control_coverage": (
            summarize_partition(control)
        ),
        "paired_frame_results": [
            {
                "frame": name,
                "candidate_score": x,
                "control_score": y,
                "advantage": x - y,
            }
            for name, x, y in paired
        ],
    }


def rank_candidates(candidates):
    ranked = []

    for candidate in candidates:
        development = summarize_partition(
            candidate["development"]
        )

        heldout = summarize_partition(
            candidate["heldout"]
        )

        ranked.append(
            {
                "rank": candidate["rank"],
                "development": development,
                "heldout": heldout,
            }
        )

    # Select using development data only.
    # This does not establish a correct registration.
    eligible = [
        item
        for item in ranked
        if item["development"][
            "coverage_fraction"
        ] >= MIN_COVERAGE
        and item["development"][
            "coverage_adjusted_score"
        ] is not None
    ]

    selected = (
        max(
            eligible,
            key=lambda item: item[
                "development"
            ]["coverage_adjusted_score"],
        )
        if eligible
        else None
    )

    return ranked, selected


def audit_selected(candidate, controls):
    comparisons = {
        "development": [],
        "heldout": [],
    }

    for partition in comparisons:
        for control in controls:
            comparisons[partition].append(
                {
                    "control_name": control.get(
                        "control_name",
                        "unnamed",
                    ),
                    "control_kind": (
                        control.get(
                            "transform",
                            {},
                        ).get("kind")
                    ),
                    **compare_pair(
                        candidate[partition],
                        control[partition],
                    ),
                }
            )

    heldout = comparisons["heldout"]

    all_sufficient = bool(heldout) and all(
        result["status"]
        != "INSUFFICIENT_PAIRED_COVERAGE"
        for result in heldout
    )

    all_pass = (
        all_sufficient
        and all(
            result["status"]
            == "PASSES_EXPLORATORY_PAIR_GATE"
            for result in heldout
        )
    )

    return {
        "comparisons": comparisons,
        "all_heldout_controls_sufficient": (
            all_sufficient
        ),
        "all_heldout_controls_pass": all_pass,
        "diagnostic_status": (
            "PASSES_EXPLORATORY_CONTROLS"
            if all_pass
            else "DOES_NOT_PASS_CONTROLS"
        ),
    }


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing Step 81 report: {INPUT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = json.loads(
        INPUT.read_text(
            encoding="utf-8"
        )
    )

    candidates = data.get(
        "candidates",
        [],
    )

    controls = data.get(
        "negative_controls",
        [],
    )

    ranked, selected = rank_candidates(
        candidates
    )

    selected_rank = (
        selected["rank"]
        if selected
        else None
    )

    report = {
        "source": str(INPUT),
        "method": (
            "independent_paired_control_audit"
        ),
        "registration_verified": False,
        "construction_progress_verified": False,
        "selected_candidate_rank": (
            selected_rank
        ),
        "candidate_summaries": ranked,
        "thresholds": {
            "min_paired_frames": (
                MIN_PAIRED_FRAMES
            ),
            "min_coverage": MIN_COVERAGE,
            "min_win_fraction": (
                MIN_WIN_FRACTION
            ),
            "min_mean_advantage": (
                MIN_MEAN_ADVANTAGE
            ),
        },
        "decision": (
            "UNVERIFIED"
        ),
        "limitations": [
            (
                "Reuses Step 81 projected-mask "
                "orientation scores."
            ),
            (
                "Does not verify occlusion or "
                "physical wall heights."
            ),
            (
                "Negative controls are perturbations, "
                "not independently labeled ground truth."
            ),
            (
                "Paired frames are temporally correlated."
            ),
            (
                "Passing exploratory gates would not "
                "establish correct CAD registration."
            ),
        ],
    }

    if selected is not None:
        selected_candidate = next(
            item
            for item in candidates
            if item["rank"] == selected_rank
        )

        related_controls = [
            control
            for control in controls
            if control[
                "source_candidate_rank"
            ] == selected_rank
        ]

        report["selected_candidate_audit"] = (
            audit_selected(
                selected_candidate,
                related_controls,
            )
        )

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== REGISTRATION CONTROL AUDIT ==="
    )

    for candidate in ranked:
        print(
            f"Candidate {candidate['rank']}: "
            f"dev adjusted="
            f"{candidate['development']['coverage_adjusted_score']}, "
            f"heldout adjusted="
            f"{candidate['heldout']['coverage_adjusted_score']}"
        )

    print(
        "\nDevelopment-selected candidate:",
        selected_rank,
    )

    if selected is not None:
        audit = report[
            "selected_candidate_audit"
        ]

        print(
            "Held-out control status:",
            audit["diagnostic_status"],
        )

        for comparison in audit[
            "comparisons"
        ]["heldout"]:
            print(
                comparison["control_name"],
                "|",
                comparison["status"],
                "| paired:",
                comparison[
                    "paired_valid_frames"
                ],
                "| advantage:",
                comparison[
                    "mean_score_advantage"
                ],
                "| wins:",
                comparison[
                    "frame_win_fraction"
                ],
            )

    print(
        "\nRegistration verified: NO"
    )

    print(
        "Report:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
