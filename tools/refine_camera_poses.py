
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "outputs" / "multiview_tracks_v2"
OUTPUT = ROOT / "outputs" / "bundle_adjustment"

LANDMARK_FILE = INPUT / "validated_landmarks_v2.json"
POSE_FILE = INPUT / "camera_poses_v2.json"


def load_data():
    if not LANDMARK_FILE.exists():
        raise FileNotFoundError(LANDMARK_FILE)

    if not POSE_FILE.exists():
        raise FileNotFoundError(POSE_FILE)

    landmarks = json.loads(
        LANDMARK_FILE.read_text(encoding="utf-8")
    )

    pose_document = json.loads(
        POSE_FILE.read_text(encoding="utf-8")
    )

    poses = pose_document["poses"]

    if not landmarks:
        raise RuntimeError("No validated landmarks found.")

    if not poses:
        raise RuntimeError("No camera poses found.")

    return landmarks, poses


def median_or_none(values):
    values = np.asarray(values, dtype=np.float64)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return float(np.median(values))


def error_summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    errors = errors[np.isfinite(errors)]

    if len(errors) == 0:
        return {
            "count": 0,
            "median_px": None,
            "mean_px": None,
            "p90_px": None,
            "rmse_px": None,
        }

    return {
        "count": int(len(errors)),
        "median_px": float(np.median(errors)),
        "mean_px": float(np.mean(errors)),
        "p90_px": float(np.percentile(errors, 90)),
        "rmse_px": float(np.sqrt(np.mean(errors**2))),
    }


def create_problem(landmarks, pose_records):
    """
    Keep all landmarks and all exported observations.

    Camera rotations use camera-to-world conventions.
    The optimizer works with small world-frame
    rotation corrections to those recorded rotations.
    """

    point_positions = np.asarray(
        [item["xyz"] for item in landmarks],
        dtype=np.float64,
    )

    if (
        point_positions.ndim != 2
        or point_positions.shape[1] != 3
        or not np.isfinite(point_positions).all()
    ):
        raise RuntimeError("Invalid landmark coordinates.")

    used_frames = sorted({
        obs["frame"]
        for item in landmarks
        for obs in item["observations"]
    })

    missing = [
        name for name in used_frames
        if name not in pose_records
    ]

    if missing:
        raise RuntimeError(
            f"Missing camera poses: {missing[:5]}"
        )

    frame_index = {
        name: i
        for i, name in enumerate(used_frames)
    }

    centers = []
    rotations = []

    for name in used_frames:
        pose = pose_records[name]

        center = np.asarray(
            pose["center"],
            dtype=np.float64,
        )

        matrix = np.asarray(
            pose["rotation_camera_to_world"],
            dtype=np.float64,
        )

        if (
            center.shape != (3,)
            or matrix.shape != (3, 3)
            or not np.isfinite(center).all()
            or not np.isfinite(matrix).all()
        ):
            raise RuntimeError(
                f"Invalid camera pose for {name}"
            )

        if not np.allclose(
            matrix.T @ matrix,
            np.eye(3),
            atol=1e-3,
        ):
            raise RuntimeError(
                f"Non-orthogonal rotation for {name}"
            )

        centers.append(center)
        rotations.append(matrix)

    centers = np.asarray(centers)
    rotations = np.asarray(rotations)

    obs_camera = []
    obs_point = []
    obs_uv = []
    obs_focal = []

    for point_id, landmark in enumerate(landmarks):
        for obs in landmark["observations"]:
            uv = np.asarray(
                obs["normalized_uv"],
                dtype=np.float64,
            )

            focal = float(obs["focal_px"])

            if (
                uv.shape != (2,)
                or not np.isfinite(uv).all()
                or not np.isfinite(focal)
                or focal <= 0
            ):
                raise RuntimeError(
                    f"Invalid observation for landmark {point_id}"
                )

            obs_camera.append(frame_index[obs["frame"]])
            obs_point.append(point_id)
            obs_uv.append(uv)
            obs_focal.append(focal)

    return {
        "frames": used_frames,
        "centers": centers,
        "rotations": rotations,
        "points": point_positions,
        "obs_camera": np.asarray(obs_camera, dtype=int),
        "obs_point": np.asarray(obs_point, dtype=int),
        "obs_uv": np.asarray(obs_uv, dtype=np.float64),
        "obs_focal": np.asarray(obs_focal, dtype=np.float64),
    }


def select_holdout(problem, seed):
    """
    Withhold one observation from tracks having >=4 views.

    Three-view tracks remain entirely in training because
    removing one would leave only two observations.

    This is an internal held-out test, not independent
    camera calibration ground truth.
    """

    rng = np.random.default_rng(seed)

    n_obs = len(problem["obs_camera"])
    n_points = len(problem["points"])

    train = np.ones(n_obs, dtype=bool)
    holdout = np.zeros(n_obs, dtype=bool)

    for point_id in range(n_points):
        indices = np.flatnonzero(
            problem["obs_point"] == point_id
        )

        if len(indices) < 4:
            continue

        chosen = int(rng.choice(indices))

        train[chosen] = False
        holdout[chosen] = True

    return train, holdout


def projection_errors(
    centers,
    rotations,
    points,
    problem,
    observation_indices,
):
    cameras = problem["obs_camera"][observation_indices]
    point_ids = problem["obs_point"][observation_indices]

    vectors = (
        points[point_ids] - centers[cameras]
    )

    # Camera-to-world rotation -> world-to-camera.
    local = np.einsum(
        "nij,nj->ni",
        rotations[cameras].transpose(0, 2, 1),
        vectors,
    )

    depth = local[:, 2]

    valid = (
        np.isfinite(local).all(axis=1)
        & (depth > 0.05)
    )

    safe_depth = np.where(
        valid,
        depth,
        1.0,
    )

    projected = local[:, :2] / safe_depth[:, None]

    errors = (
        projected - problem["obs_uv"][observation_indices]
    ) * problem["obs_focal"][observation_indices, None]

    # Behind-camera observations are penalized.
    errors[~valid] = 1000.0

    return errors


def pack_parameters(
    center_offsets,
    rotation_offsets,
    points,
):
    return np.concatenate([
        center_offsets.ravel(),
        rotation_offsets.ravel(),
        points.ravel(),
    ])


def unpack_parameters(x, n_cameras, n_points):
    n = 3 * n_cameras

    center_offsets = x[:n].reshape(n_cameras, 3)

    rotation_offsets = x[n:2*n].reshape(
        n_cameras, 3
    )

    points = x[2*n:].reshape(n_points, 3)

    return center_offsets, rotation_offsets, points


def corrected_poses(
    problem,
    center_offsets,
    rotation_offsets,
):
    centers = problem["centers"] + center_offsets

    corrections = Rotation.from_rotvec(
        rotation_offsets
    ).as_matrix()

    rotations = corrections @ problem["rotations"]

    return centers, rotations


def residual_function(
    x,
    problem,
    train_indices,
    center_sigma,
    rotation_sigma,
    anchor_sigma,
):
    n_cameras = len(problem["frames"])
    n_points = len(problem["points"])

    dc, dr, points = unpack_parameters(
        x,
        n_cameras,
        n_points,
    )

    centers, rotations = corrected_poses(
        problem,
        dc,
        dr,
    )

    reprojection = projection_errors(
        centers,
        rotations,
        points,
        problem,
        train_indices,
    ).ravel()

    center_prior = (
        dc / center_sigma
    ).ravel()

    rotation_prior = (
        dr / rotation_sigma
    ).ravel()

    # Strongly anchor one camera to prevent
    # an unconstrained global coordinate drift.
    anchor = np.concatenate([
        dc[0] / anchor_sigma,
        dr[0] / anchor_sigma,
    ])

    return np.concatenate([
        reprojection,
        center_prior,
        rotation_prior,
        anchor,
    ])


def build_sparsity(problem, train_indices):
    n_cameras = len(problem["frames"])
    n_points = len(problem["points"])

    n_train = len(train_indices)

    parameter_count = (
        6 * n_cameras + 3 * n_points
    )

    residual_count = (
        2 * n_train
        + 6 * n_cameras
        + 6
    )

    S = lil_matrix(
        (residual_count, parameter_count),
        dtype=np.int8,
    )

    for k, obs_index in enumerate(train_indices):
        camera = problem["obs_camera"][obs_index]
        point = problem["obs_point"][obs_index]

        rows = slice(2*k, 2*k + 2)

        # Translation offset.
        S[rows, 3*camera:3*camera + 3] = 1

        # Rotation offset.
        rotation_start = 3*n_cameras + 3*camera
        S[rows, rotation_start:rotation_start + 3] = 1

        # Landmark position.
        point_start = 6*n_cameras + 3*point
        S[rows, point_start:point_start + 3] = 1

    prior_start = 2*n_train

    for camera in range(n_cameras):
        center_row = prior_start + 3*camera

        S[
            center_row:center_row + 3,
            3*camera:3*camera + 3,
        ] = np.eye(3, dtype=np.int8)

        rotation_row = (
            prior_start + 3*n_cameras + 3*camera
        )

        rotation_col = (
            3*n_cameras + 3*camera
        )

        S[
            rotation_row:rotation_row + 3,
            rotation_col:rotation_col + 3,
        ] = np.eye(3, dtype=np.int8)

    anchor_row = prior_start + 6*n_cameras

    S[
        anchor_row:anchor_row + 3,
        0:3,
    ] = np.eye(3, dtype=np.int8)

    S[
        anchor_row + 3:anchor_row + 6,
        3*n_cameras:3*n_cameras + 3,
    ] = np.eye(3, dtype=np.int8)

    return S.tocsr()


def write_ply(path, points, landmarks):
    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(
            "comment Experimental refined landmark positions\n"
        )
        f.write(
            "comment Metric accuracy and extrinsics unverified\n"
        )
        f.write(f"element vertex {len(points)}\n")

        for axis in ("x", "y", "z"):
            f.write(f"property float {axis}\n")

        for channel in ("red", "green", "blue"):
            f.write(f"property uchar {channel}\n")

        f.write("end_header\n")

        for point, item in zip(points, landmarks):
            rgb = item.get("rgb", [180, 180, 180])

            x, y, z = point
            r, g, b = rgb

            f.write(
                f"{x:.8f} {y:.8f} {z:.8f} "
                f"{int(r)} {int(g)} {int(b)}\n"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-nfev",
        type=int,
        default=35,
    )

    parser.add_argument(
        "--center-sigma",
        type=float,
        default=0.05,
        help="Position prior in recorded pose coordinate units.",
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
        parser.error("All optimization parameters must be positive.")

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
            "No holdout observations available."
        )

    n_cameras = len(problem["frames"])
    n_points = len(problem["points"])

    print("\nUNDER-CONSTRUCTION")
    print("Constrained Bundle Adjustment")
    print("-" * 60)
    print("Cameras in optimization:", n_cameras)
    print("Landmarks:", n_points)
    print("Total observations:", len(train_mask))
    print("Training observations:", len(train_indices))
    print("Held-out observations:", len(holdout_indices))
    print("Camera extrinsics: UNVERIFIED")

    zero_centers = np.zeros(
        (n_cameras, 3),
        dtype=np.float64,
    )

    zero_rotations = np.zeros(
        (n_cameras, 3),
        dtype=np.float64,
    )

    x0 = pack_parameters(
        zero_centers,
        zero_rotations,
        problem["points"],
    )

    before_train = np.linalg.norm(
        projection_errors(
            problem["centers"],
            problem["rotations"],
            problem["points"],
            problem,
            train_indices,
        ),
        axis=1,
    )

    before_holdout = np.linalg.norm(
        projection_errors(
            problem["centers"],
            problem["rotations"],
            problem["points"],
            problem,
            holdout_indices,
        ),
        axis=1,
    )

    sparsity = build_sparsity(
        problem,
        train_indices,
    )

    rotation_sigma = math.radians(
        args.rotation_sigma_deg
    )

    def residual(x):
        return residual_function(
            x,
            problem,
            train_indices,
            args.center_sigma,
            rotation_sigma,
            args.anchor_sigma,
        )

    print("\nOptimizing...")
    print(
        "Sparse Jacobian:",
        sparsity.shape,
        "nonzeros:",
        sparsity.nnz,
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
        max_nfev=args.max_nfev,
        x_scale="jac",
        verbose=1,
    )

    dc, dr, refined_points = unpack_parameters(
        result.x,
        n_cameras,
        n_points,
    )

    refined_centers, refined_rotations = corrected_poses(
        problem,
        dc,
        dr,
    )

    after_train = np.linalg.norm(
        projection_errors(
            refined_centers,
            refined_rotations,
            refined_points,
            problem,
            train_indices,
        ),
        axis=1,
    )

    after_holdout = np.linalg.norm(
        projection_errors(
            refined_centers,
            refined_rotations,
            refined_points,
            problem,
            holdout_indices,
        ),
        axis=1,
    )

    before_train_summary = error_summary(before_train)
    after_train_summary = error_summary(after_train)

    before_holdout_summary = error_summary(before_holdout)
    after_holdout_summary = error_summary(after_holdout)

    center_shifts = np.linalg.norm(dc, axis=1)
    rotation_shifts = np.degrees(
        np.linalg.norm(dr, axis=1)
    )

    # Require held-out median and p90 to improve.
    # This is a conservative internal acceptance test,
    # not independent physical validation.
    improved_holdout = (
        after_holdout_summary["median_px"]
        < before_holdout_summary["median_px"]
        and after_holdout_summary["p90_px"]
        <= before_holdout_summary["p90_px"]
    )

    finite_solution = (
        np.isfinite(result.x).all()
        and np.isfinite(after_train).all()
        and np.isfinite(after_holdout).all()
    )

    max_center_shift = float(np.max(center_shifts))
    max_rotation_shift = float(np.max(rotation_shifts))

    # Additional safety guard: no extreme corrections.
    bounded_corrections = (
        max_center_shift <= 3.0 * args.center_sigma
        and max_rotation_shift <= (
            3.0 * args.rotation_sigma_deg
        )
    )

    accepted = bool(
        finite_solution
        and improved_holdout
        and bounded_corrections
    )

    pose_results = {}

    for index, frame in enumerate(problem["frames"]):
        pose_results[frame] = {
            "original_center": (
                problem["centers"][index].tolist()
            ),
            "refined_center": (
                refined_centers[index].tolist()
            ),
            "original_rotation_camera_to_world": (
                problem["rotations"][index].tolist()
            ),
            "refined_rotation_camera_to_world": (
                refined_rotations[index].tolist()
            ),
            "center_shift_pose_units": float(
                center_shifts[index]
            ),
            "rotation_shift_deg": float(
                rotation_shifts[index]
            ),
        }

    report = {
        "project": "Under-construction",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "CONSTRAINED_BUNDLE_ADJUSTMENT_V1",
        "parameters": vars(args),
        "cameras": n_cameras,
        "landmarks": n_points,
        "total_observations": len(train_mask),
        "training_observations": len(train_indices),
        "held_out_observations": len(holdout_indices),
        "optimization": {
            "success": bool(result.success),
            "message": str(result.message),
            "function_evaluations": int(result.nfev),
            "initial_cost": float(
                0.5 * np.sum(residual(x0)**2)
            ),
            "final_robust_cost": float(result.cost),
        },
        "training_before": before_train_summary,
        "training_after": after_train_summary,
        "holdout_before": before_holdout_summary,
        "holdout_after": after_holdout_summary,
        "camera_corrections": {
            "median_center_shift_pose_units": float(
                np.median(center_shifts)
            ),
            "max_center_shift_pose_units": max_center_shift,
            "median_rotation_shift_deg": float(
                np.median(rotation_shifts)
            ),
            "max_rotation_shift_deg": max_rotation_shift,
        },
        "acceptance": {
            "accepted": accepted,
            "finite_solution": bool(finite_solution),
            "held_out_improved": bool(improved_holdout),
            "bounded_corrections": bool(bounded_corrections),
            "note": (
                "Internal validation only. Not verified "
                "against independently measured camera poses."
            ),
        },
        "limitations": [
            "Camera-to-IMU extrinsics are unavailable.",
            "The starting camera-to-world convention is provisional.",
            "Position prior units have not been independently verified.",
            "The holdout observations belong to tracks selected "
            "using the original camera poses.",
            "Three-view tracks cannot contribute held-out observations "
            "under this split.",
            "The holdout split is observation-level, not scene-level.",
            "The robust loss also applies to pose-prior residuals.",
            "No verified metric accuracy or CAD alignment.",
            "A lower reprojection error does not prove correct geometry.",
        ],
    }

    report_path = OUTPUT / "bundle_adjustment_report.json"
    poses_path = OUTPUT / "refined_camera_poses.json"
    points_path = OUTPUT / "refined_landmarks.json"
    ply_path = OUTPUT / "refined_landmarks.ply"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    poses_path.write_text(
        json.dumps(
            {
                "accepted": accepted,
                "poses": pose_results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    refined_landmarks = []

    for index, point in enumerate(refined_points):
        item = dict(landmarks[index])
        item["xyz"] = point.tolist()
        refined_landmarks.append(item)

    points_path.write_text(
        json.dumps(
            {
                "accepted": accepted,
                "landmarks": refined_landmarks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_ply(
        ply_path,
        refined_points,
        landmarks,
    )

    print("\nRESULTS")
    print("-" * 60)

    print(
        "Training median:",
        before_train_summary["median_px"],
        "->",
        after_train_summary["median_px"],
    )

    print(
        "Held-out median:",
        before_holdout_summary["median_px"],
        "->",
        after_holdout_summary["median_px"],
    )

    print(
        "Held-out p90:",
        before_holdout_summary["p90_px"],
        "->",
        after_holdout_summary["p90_px"],
    )

    print(
        "Median camera center correction:",
        float(np.median(center_shifts)),
    )

    print(
        "Maximum camera center correction:",
        max_center_shift,
    )

    print(
        "Maximum rotation correction:",
        max_rotation_shift,
        "degrees",
    )

    print(
        "\nACCEPTANCE:",
        "ACCEPTED" if accepted else "REJECTED",
    )

    print("\nReport:", report_path)
    print("Camera poses:", poses_path)
    print("Landmarks:", points_path)
    print("Point cloud:", ply_path)

    if not accepted:
        print(
            "\nThe refined geometry was saved for analysis "
            "but must NOT replace the V2 reconstruction."
        )

    print(
        "\nWARNING: This is an internal geometric "
        "consistency experiment, not metric validation."
    )


if __name__ == "__main__":
    main()
