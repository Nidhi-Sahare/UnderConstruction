
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from refine_camera_poses import (
    ROOT,
    load_data,
    create_problem,
    select_holdout,
    projection_errors,
    corrected_poses,
    error_summary,
)


OUTPUT = ROOT / "outputs" / "pose_refinement_ablation"

VARIANTS = {
    "landmarks_only": {
        "centers": False,
        "rotations": False,
        "points": True,
    },
    "rotation_only": {
        "centers": False,
        "rotations": True,
        "points": False,
    },
    "translation_only": {
        "centers": True,
        "rotations": False,
        "points": False,
    },
    "joint": {
        "centers": True,
        "rotations": True,
        "points": True,
    },
}


def make_layout(problem, variant):
    nc = len(problem["frames"])
    np_ = len(problem["points"])

    cursor = 0
    layout = {}

    for name, enabled, size in (
        ("centers", variant["centers"], 3 * nc),
        ("rotations", variant["rotations"], 3 * nc),
        ("points", variant["points"], 3 * np_),
    ):
        if enabled:
            layout[name] = slice(cursor, cursor + size)
            cursor += size
        else:
            layout[name] = None

    return layout, cursor


def decode(x, problem, layout):
    nc = len(problem["frames"])

    dc = np.zeros((nc, 3), dtype=np.float64)
    dr = np.zeros((nc, 3), dtype=np.float64)

    points = problem["points"].copy()

    if layout["centers"] is not None:
        dc = x[layout["centers"]].reshape(nc, 3)

    if layout["rotations"] is not None:
        dr = x[layout["rotations"]].reshape(nc, 3)

    if layout["points"] is not None:
        points = x[layout["points"]].reshape(-1, 3)

    centers, rotations = corrected_poses(
        problem, dc, dr
    )

    return centers, rotations, points, dc, dr


def make_sparsity(problem, train_indices, layout, nparams):
    nc = len(problem["frames"])
    ntrain = len(train_indices)

    center_enabled = layout["centers"] is not None
    rotation_enabled = layout["rotations"] is not None

    prior_count = (
        3 * nc * int(center_enabled)
        + 3 * nc * int(rotation_enabled)
    )

    anchor_count = (
        3 * int(center_enabled)
        + 3 * int(rotation_enabled)
    )

    rows = 2 * ntrain + prior_count + anchor_count

    S = lil_matrix((rows, nparams), dtype=np.int8)

    for i, obs_id in enumerate(train_indices):
        cam = int(problem["obs_camera"][obs_id])
        point = int(problem["obs_point"][obs_id])

        r = slice(2 * i, 2 * i + 2)

        for name, index in (
            ("centers", cam),
            ("rotations", cam),
            ("points", point),
        ):
            block = layout[name]
            if block is None:
                continue

            start = block.start + 3 * index
            S[r, start:start + 3] = 1

    row = 2 * ntrain

    for name in ("centers", "rotations"):
        block = layout[name]

        if block is None:
            continue

        for cam in range(nc):
            start = block.start + 3 * cam
            S[
                row:row + 3,
                start:start + 3,
            ] = np.eye(3, dtype=np.int8)
            row += 3 * 1

    for name in ("centers", "rotations"):
        block = layout[name]

        if block is None:
            continue

        S[
            row:row + 3,
            block.start:block.start + 3,
        ] = np.eye(3, dtype=np.int8)

        row += 3

    if row != rows:
        raise RuntimeError(
            f"Sparsity row mismatch: {row} != {rows}"
        )

    return S.tocsr()


def evaluate(problem, indices, centers, rotations, points):
    residuals = projection_errors(
        centers,
        rotations,
        points,
        problem,
        indices,
    )

    errors = np.linalg.norm(residuals, axis=1)

    return error_summary(errors)


def run_variant(
    name,
    variant,
    problem,
    train_indices,
    holdout_indices,
    baseline_train,
    baseline_holdout,
    args,
):
    nc = len(problem["frames"])

    layout, nparams = make_layout(problem, variant)

    x0 = np.zeros(nparams, dtype=np.float64)

    if layout["points"] is not None:
        x0[layout["points"]] = problem["points"].ravel()

    center_sigma = args.center_sigma
    rotation_sigma = math.radians(
        args.rotation_sigma_deg
    )

    def residual(x):
        centers, rotations, points, dc, dr = decode(
            x, problem, layout
        )

        parts = [
            projection_errors(
                centers,
                rotations,
                points,
                problem,
                train_indices,
            ).ravel()
        ]

        if variant["centers"]:
            parts.append((dc / center_sigma).ravel())

        if variant["rotations"]:
            parts.append((dr / rotation_sigma).ravel())

        if variant["centers"]:
            parts.append(dc[0] / args.anchor_sigma)

        if variant["rotations"]:
            parts.append(dr[0] / args.anchor_sigma)

        return np.concatenate(parts)

    sparsity = make_sparsity(
        problem,
        train_indices,
        layout,
        nparams,
    )

    print(f"\nRunning: {name}")
    print("Parameters:", nparams)
    print("Sparse Jacobian nonzeros:", sparsity.nnz)

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

    centers, rotations, points, dc, dr = decode(
        result.x, problem, layout
    )

    train = evaluate(
        problem,
        train_indices,
        centers,
        rotations,
        points,
    )

    holdout = evaluate(
        problem,
        holdout_indices,
        centers,
        rotations,
        points,
    )

    center_shifts = np.linalg.norm(dc, axis=1)

    rotation_shifts = np.degrees(
        np.linalg.norm(dr, axis=1)
    )

    finite = (
        np.isfinite(result.x).all()
        and np.isfinite(centers).all()
        and np.isfinite(rotations).all()
        and np.isfinite(points).all()
        and np.isfinite(train["rmse_px"])
        and np.isfinite(holdout["rmse_px"])
    )

    bounded = (
        float(np.max(center_shifts))
        <= 3.0 * center_sigma
        and float(np.max(rotation_shifts))
        <= 3.0 * args.rotation_sigma_deg
    )

    # Require all three holdout measures to improve.
    # This is a screening gate, not independent validation.
    improved = (
        holdout["median_px"]
        < baseline_holdout["median_px"]
        and holdout["p90_px"]
        < baseline_holdout["p90_px"]
        and holdout["rmse_px"]
        < baseline_holdout["rmse_px"]
    )

    # Convergence is required; an iteration-limited result
    # remains experimental even if its errors look better.
    eligible = bool(
        result.success
        and finite
        and bounded
        and improved
    )

    print(
        "Converged:",
        bool(result.success),
        "| evaluations:",
        result.nfev,
    )

    print(
        "Training median:",
        f"{baseline_train['median_px']:.4f}",
        "->",
        f"{train['median_px']:.4f}",
    )

    print(
        "Holdout median:",
        f"{baseline_holdout['median_px']:.4f}",
        "->",
        f"{holdout['median_px']:.4f}",
    )

    print(
        "Holdout p90:",
        f"{baseline_holdout['p90_px']:.4f}",
        "->",
        f"{holdout['p90_px']:.4f}",
    )

    print(
        "Holdout RMSE:",
        f"{baseline_holdout['rmse_px']:.4f}",
        "->",
        f"{holdout['rmse_px']:.4f}",
    )

    print(
        "Screening:",
        "PASS" if eligible else "FAIL",
    )

    return {
        "variant": name,
        "enabled_parameters": variant,
        "parameter_count": int(nparams),
        "optimization": {
            "converged": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "final_robust_cost": float(result.cost),
        },
        "training": train,
        "holdout": holdout,
        "camera_corrections": {
            "median_center_shift_pose_units": float(
                np.median(center_shifts)
            ),
            "max_center_shift_pose_units": float(
                np.max(center_shifts)
            ),
            "median_rotation_shift_deg": float(
                np.median(rotation_shifts)
            ),
            "max_rotation_shift_deg": float(
                np.max(rotation_shifts)
            ),
        },
        "screening": {
            "finite": bool(finite),
            "bounded_corrections": bool(bounded),
            "held_out_improved": bool(improved),
            "eligible": bool(eligible),
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=70,
    )

    parser.add_argument(
        "--center-sigma",
        type=float,
        default=0.05,
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
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if (
        args.max_nfev < 1
        or args.center_sigma <= 0
        or args.rotation_sigma_deg <= 0
        or args.anchor_sigma <= 0
        or args.loss_scale <= 0
    ):
        parser.error(
            "Optimization parameters must be positive."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    landmarks, pose_records = load_data()

    problem = create_problem(
        landmarks,
        pose_records,
    )

    train_mask, holdout_mask = select_holdout(
        problem,
        args.seed,
    )

    train_indices = np.flatnonzero(train_mask)
    holdout_indices = np.flatnonzero(holdout_mask)

    if len(holdout_indices) == 0:
        raise RuntimeError(
            "No held-out observations available."
        )

    baseline_train = evaluate(
        problem,
        train_indices,
        problem["centers"],
        problem["rotations"],
        problem["points"],
    )

    baseline_holdout = evaluate(
        problem,
        holdout_indices,
        problem["centers"],
        problem["rotations"],
        problem["points"],
    )

    print("\nUNDER-CONSTRUCTION")
    print("Pose Refinement Ablation")
    print("-" * 60)

    print("Cameras:", len(problem["frames"]))
    print("Landmarks:", len(problem["points"]))
    print(
        "Training observations:",
        len(train_indices),
    )
    print(
        "Held-out observations:",
        len(holdout_indices),
    )

    print(
        "Baseline held-out median:",
        baseline_holdout["median_px"],
    )

    results = {}

    for name, variant in VARIANTS.items():
        results[name] = run_variant(
            name,
            variant,
            problem,
            train_indices,
            holdout_indices,
            baseline_train,
            baseline_holdout,
            args,
        )

    eligible = [
        name
        for name, record in results.items()
        if record["screening"]["eligible"]
    ]

    # This selection is exploratory: choosing the best
    # variant on the same holdout set introduces selection
    # bias. A separate evaluation is required before use.
    exploratory_best = None

    if eligible:
        exploratory_best = min(
            eligible,
            key=lambda name: (
                results[name]["holdout"]["rmse_px"],
                results[name]["holdout"]["p90_px"],
            ),
        )

    report = {
        "project": "Under-construction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "POSE_REFINEMENT_ABLATION_V1",
        "parameters": vars(args),
        "cameras": len(problem["frames"]),
        "landmarks": len(problem["points"]),
        "training_observations": int(
            len(train_indices)
        ),
        "held_out_observations": int(
            len(holdout_indices)
        ),
        "baseline": {
            "training": baseline_train,
            "holdout": baseline_holdout,
        },
        "variants": results,
        "exploratory_best": exploratory_best,
        "approved_for_application": False,
        "limitations": [
            "Camera extrinsics are unverified.",
            "Metric scale and CAD registration are unverified.",
            "All variants share one observation-level holdout split.",
            "Three-view tracks have no held-out observation.",
            "Tracks were selected using the original pose model.",
            "Selecting a variant on this holdout set introduces selection bias.",
            "No independent physical ground truth is available.",
            "Passing the screening gate does not authorize deployment.",
            "No refined reconstruction replaces the V2 baseline.",
        ],
    }

    report_path = (
        OUTPUT / "pose_refinement_ablation_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print("-" * 60)

    for name, record in results.items():
        holdout = record["holdout"]

        status = (
            "PASS"
            if record["screening"]["eligible"]
            else "FAIL"
        )

        print(
            f"{name:20s} "
            f"median={holdout['median_px']:.4f} "
            f"p90={holdout['p90_px']:.4f} "
            f"rmse={holdout['rmse_px']:.4f} "
            f"{status}"
        )

    print("\nExploratory best:", exploratory_best)
    print("Approved for application: NO")
    print("Report:", report_path)

    print(
        "\nV2 remains the baseline. "
        "No refined geometry has been deployed."
    )


if __name__ == "__main__":
    main()
