
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from refine_camera_poses import (
    ROOT,
    load_data,
    create_problem,
    select_holdout,
    projection_errors,
    error_summary,
)


OUTPUT = ROOT / "outputs" / "rotation_cross_validation"


def evaluate(problem, indices, rotations):
    residuals = projection_errors(
        problem["centers"],
        rotations,
        problem["points"],
        problem,
        indices,
    )

    errors = np.linalg.norm(residuals, axis=1)
    return error_summary(errors)


def apply_rotation_offsets(original_rotations, offsets):
    correction_matrices = Rotation.from_rotvec(
        offsets
    ).as_matrix()

    return correction_matrices @ original_rotations


def build_sparsity(problem, train_indices):
    n_cameras = len(problem["frames"])
    n_train = len(train_indices)

    # Reprojection: 2 per observation.
    # Rotation prior: 3 per camera.
    # Anchor: 3 for the first camera.
    n_rows = 2 * n_train + 3 * n_cameras + 3
    n_cols = 3 * n_cameras

    S = lil_matrix(
        (n_rows, n_cols),
        dtype=np.int8,
    )

    for row_index, observation_id in enumerate(train_indices):
        camera_id = int(
            problem["obs_camera"][observation_id]
        )

        S[
            2 * row_index:2 * row_index + 2,
            3 * camera_id:3 * camera_id + 3,
        ] = 1

    prior_start = 2 * n_train

    for camera_id in range(n_cameras):
        row = prior_start + 3 * camera_id
        col = 3 * camera_id

        S[
            row:row + 3,
            col:col + 3,
        ] = np.eye(3, dtype=np.int8)

    anchor_row = prior_start + 3 * n_cameras

    S[
        anchor_row:anchor_row + 3,
        0:3,
    ] = np.eye(3, dtype=np.int8)

    return S.tocsr()


def run_fold(problem, seed, args):
    train_mask, holdout_mask = select_holdout(
        problem,
        seed,
    )

    train_indices = np.flatnonzero(train_mask)
    holdout_indices = np.flatnonzero(holdout_mask)

    if not len(holdout_indices):
        raise RuntimeError(
            f"Seed {seed} produced no held-out observations."
        )

    original_rotations = problem["rotations"]
    n_cameras = len(problem["frames"])

    baseline_train = evaluate(
        problem,
        train_indices,
        original_rotations,
    )

    baseline_holdout = evaluate(
        problem,
        holdout_indices,
        original_rotations,
    )

    rotation_sigma = math.radians(
        args.rotation_sigma_deg
    )

    def residual(x):
        offsets = x.reshape(n_cameras, 3)

        corrected = apply_rotation_offsets(
            original_rotations,
            offsets,
        )

        reprojection = projection_errors(
            problem["centers"],
            corrected,
            problem["points"],
            problem,
            train_indices,
        ).ravel()

        priors = (
            offsets / rotation_sigma
        ).ravel()

        anchor = offsets[0] / args.anchor_sigma

        return np.concatenate([
            reprojection,
            priors,
            anchor,
        ])

    x0 = np.zeros(
        3 * n_cameras,
        dtype=np.float64,
    )

    sparsity = build_sparsity(
        problem,
        train_indices,
    )

    result = least_squares(
        residual,
        x0,
        method="trf",
        jac="2-point",
        jac_sparsity=sparsity,
        tr_solver="lsmr",
        loss="soft_l1",
        f_scale=args.loss_scale,
        x_scale="jac",
        max_nfev=args.max_nfev,
        verbose=0,
    )

    offsets = result.x.reshape(n_cameras, 3)

    refined_rotations = apply_rotation_offsets(
        original_rotations,
        offsets,
    )

    refined_train = evaluate(
        problem,
        train_indices,
        refined_rotations,
    )

    refined_holdout = evaluate(
        problem,
        holdout_indices,
        refined_rotations,
    )

    correction_angles = np.degrees(
        np.linalg.norm(offsets, axis=1)
    )

    finite = bool(
        np.isfinite(result.x).all()
        and np.isfinite(correction_angles).all()
        and np.isfinite(refined_holdout["rmse_px"])
    )

    bounded = bool(
        np.max(correction_angles)
        <= 3.0 * args.rotation_sigma_deg
    )

    improved_median = (
        refined_holdout["median_px"]
        < baseline_holdout["median_px"]
    )

    improved_rmse = (
        refined_holdout["rmse_px"]
        < baseline_holdout["rmse_px"]
    )

    improved_p90 = (
        refined_holdout["p90_px"]
        <= baseline_holdout["p90_px"]
    )

    passed = bool(
        result.success
        and finite
        and bounded
        and improved_median
        and improved_rmse
        and improved_p90
    )

    return {
        "seed": seed,
        "training_observations": int(len(train_indices)),
        "held_out_observations": int(len(holdout_indices)),
        "optimizer": {
            "converged": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev),
        },
        "baseline_training": baseline_train,
        "refined_training": refined_train,
        "baseline_holdout": baseline_holdout,
        "refined_holdout": refined_holdout,
        "corrections": {
            "median_rotation_deg": float(
                np.median(correction_angles)
            ),
            "maximum_rotation_deg": float(
                np.max(correction_angles)
            ),
        },
        "checks": {
            "finite": finite,
            "bounded": bounded,
            "improved_median": bool(improved_median),
            "improved_rmse": bool(improved_rmse),
            "nonworsening_p90": bool(improved_p90),
            "passed": passed,
        },
    }, offsets


def rotation_stability(offset_sets, frames):
    """
    Measure the angular difference between corrected
    camera orientations across cross-validation runs.

    Because all runs use the same original orientation,
    comparing correction matrices is sufficient.
    """

    stacked = np.asarray(
        offset_sets,
        dtype=np.float64,
    )

    n_folds, n_cameras, _ = stacked.shape

    per_camera = {}

    all_differences = []

    for camera_id, frame in enumerate(frames):
        rotations = Rotation.from_rotvec(
            stacked[:, camera_id, :]
        )

        matrices = rotations.as_matrix()

        pairwise_differences = []

        for i in range(n_folds):
            for j in range(i + 1, n_folds):
                relative = (
                    matrices[i] @ matrices[j].T
                )

                angle = np.degrees(
                    Rotation.from_matrix(
                        relative
                    ).magnitude()
                )

                pairwise_differences.append(
                    float(angle)
                )

        per_camera[frame] = {
            "median_pairwise_difference_deg": float(
                np.median(pairwise_differences)
            ),
            "max_pairwise_difference_deg": float(
                np.max(pairwise_differences)
            ),
        }

        all_differences.extend(
            pairwise_differences
        )

    return {
        "median_pairwise_difference_deg": float(
            np.median(all_differences)
        ),
        "p90_pairwise_difference_deg": float(
            np.percentile(all_differences, 90)
        ),
        "maximum_pairwise_difference_deg": float(
            np.max(all_differences)
        ),
        "per_camera": per_camera,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--rotation-sigma-deg",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--anchor-sigma",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--loss-scale",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--stability-threshold-deg",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    if args.folds < 2:
        parser.error("--folds must be at least 2")

    if args.max_nfev < 1:
        parser.error("--max-nfev must be positive")

    if (
        args.rotation_sigma_deg <= 0
        or args.anchor_sigma <= 0
        or args.loss_scale <= 0
        or args.stability_threshold_deg <= 0
    ):
        parser.error(
            "Optimization thresholds must be positive."
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    landmarks, poses = load_data()

    problem = create_problem(
        landmarks,
        poses,
    )

    print("\nUNDER-CONSTRUCTION")
    print("Rotation-only Cross-validation")
    print("-" * 60)
    print("Cameras:", len(problem["frames"]))
    print("Landmarks:", len(problem["points"]))
    print("Observations:", len(problem["obs_camera"]))
    print("Validation splits:", args.folds)
    print("Camera extrinsics: UNVERIFIED")

    records = []
    offsets_by_fold = []

    for fold_id in range(args.folds):
        seed = args.seed + fold_id

        print(
            f"\nFold {fold_id + 1}/{args.folds} "
            f"(seed={seed})"
        )

        record, offsets = run_fold(
            problem,
            seed,
            args,
        )

        records.append(record)
        offsets_by_fold.append(offsets)

        before = record["baseline_holdout"]
        after = record["refined_holdout"]

        print(
            "Converged:",
            record["optimizer"]["converged"],
            "| evaluations:",
            record["optimizer"]["evaluations"],
        )

        print(
            "Held-out median:",
            f"{before['median_px']:.4f}",
            "->",
            f"{after['median_px']:.4f}",
        )

        print(
            "Held-out p90:",
            f"{before['p90_px']:.4f}",
            "->",
            f"{after['p90_px']:.4f}",
        )

        print(
            "Held-out RMSE:",
            f"{before['rmse_px']:.4f}",
            "->",
            f"{after['rmse_px']:.4f}",
        )

        print(
            "Result:",
            "PASS"
            if record["checks"]["passed"]
            else "FAIL",
        )

    stability = rotation_stability(
        offsets_by_fold,
        problem["frames"],
    )

    passed_count = sum(
        record["checks"]["passed"]
        for record in records
    )

    converged_count = sum(
        record["optimizer"]["converged"]
        for record in records
    )

    median_before = float(np.median([
        r["baseline_holdout"]["rmse_px"]
        for r in records
    ]))

    median_after = float(np.median([
        r["refined_holdout"]["rmse_px"]
        for r in records
    ]))

    relative_improvement = (
        100.0
        * (median_before - median_after)
        / max(median_before, 1e-12)
    )

    stable = bool(
        stability["p90_pairwise_difference_deg"]
        <= args.stability_threshold_deg
    )

    required_passes = math.ceil(
        0.8 * args.folds
    )

    screening_pass = bool(
        passed_count >= required_passes
        and converged_count == args.folds
        and stable
        and median_after < median_before
    )

    report = {
        "project": "Under-construction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "ROTATION_CROSS_VALIDATION_V1",
        "parameters": vars(args),
        "cameras": len(problem["frames"]),
        "landmarks": len(problem["points"]),
        "observations": len(problem["obs_camera"]),
        "folds": records,
        "stability": stability,
        "summary": {
            "passed_folds": int(passed_count),
            "total_folds": int(args.folds),
            "converged_folds": int(converged_count),
            "required_passes": int(required_passes),
            "median_baseline_holdout_rmse_px": (
                median_before
            ),
            "median_refined_holdout_rmse_px": (
                median_after
            ),
            "relative_rmse_improvement_percent": (
                float(relative_improvement)
            ),
            "rotation_corrections_stable": stable,
            "screening_pass": screening_pass,
            "approved_for_application": False,
        },
        "limitations": [
            "The splits are repeated observation-level holdouts, "
            "not independent scene or trajectory folds.",
            "The same landmark tracks occur across splits.",
            "Holdout observations were selected from tracks "
            "constructed using the original camera poses.",
            "The rotation-only strategy was selected using "
            "an earlier holdout evaluation on this dataset.",
            "Cross-validation cannot remove that selection bias.",
            "Three-view tracks have no held-out observations.",
            "Camera-to-IMU extrinsics remain unavailable.",
            "Camera pose convention and metric scale are unverified.",
            "CAD registration is unverified.",
            "Passing screening does not validate new triangulations.",
            "The original V2 reconstruction remains unchanged.",
        ],
    }

    report_path = (
        OUTPUT / "rotation_cross_validation_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("-" * 60)
    print(
        "Passing folds:",
        f"{passed_count}/{args.folds}",
    )
    print(
        "Converged folds:",
        f"{converged_count}/{args.folds}",
    )
    print(
        "Median baseline RMSE:",
        f"{median_before:.4f} px",
    )
    print(
        "Median refined RMSE:",
        f"{median_after:.4f} px",
    )
    print(
        "Relative improvement:",
        f"{relative_improvement:.2f}%",
    )
    print(
        "Rotation stability p90:",
        f"{stability['p90_pairwise_difference_deg']:.4f} deg",
    )
    print(
        "Rotation stability threshold:",
        f"{args.stability_threshold_deg:.4f} deg",
    )

    print(
        "\nSCREENING:",
        "PASS" if screening_pass else "FAIL",
    )

    print("Approved for application: NO")
    print("Report:", report_path)

    print(
        "\nV2 remains unchanged. "
        "No camera corrections have been deployed."
    )


if __name__ == "__main__":
    main()
