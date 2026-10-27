
"""
SiteWitness — Real-Video Motion Feasibility Experiment

Run:
    python -m src.sitewitness_visual_feasibility

Outputs:
    outputs/sitewitness/visual_feasibility.json
    outputs/sitewitness/visual_matches.jpg

This experiment compares image-derived relative rotations
with pose-derived relative rotations.

It does NOT verify CAD registration, wall locations,
quaternion component order, or construction progress.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .sitewitness_camera_geometry import (
    load_calibration,
    quaternion_xyzw_to_rotation,
)


ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY = (
    ROOT / "data" / "real_samples"
    / "floor_2_camera_trajectory.json"
)

FRAMES = (
    ROOT / "sample_gallery"
    / "real_floor_2" / "frames"
)

OUTPUT = ROOT / "outputs" / "sitewitness"

MIN_MATCHES = 25
MIN_INLIERS = 15
MAX_PAIRS = 12


def vector(value, names):
    if isinstance(value, dict):
        value = [value.get(k) for k in names]

    try:
        result = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    if result.shape != (len(names),):
        return None

    if not np.isfinite(result).all():
        return None

    return result


def load_records():
    data = json.loads(TRAJECTORY.read_text(encoding="utf-8"))
    records = data["frames"]

    valid = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue

        if record.get("pose_available") is False:
            continue

        position = vector(
            record.get("position"), ("x", "y", "z")
        )
        orientation = vector(
            record.get("orientation"), ("x", "y", "z", "w")
        )

        filename = record.get("filename")

        if (
            position is None
            or orientation is None
            or not isinstance(filename, str)
            or not (FRAMES / filename).is_file()
        ):
            continue

        if abs(np.linalg.norm(orientation) - 1) > 0.02:
            continue

        valid[index] = {
            "position": position,
            "orientation": orientation,
            "filename": filename,
        }

    return valid


def choose_pairs(valid):
    candidates = [
        (i, i + 1)
        for i in sorted(valid)
        if i + 1 in valid
    ]

    if len(candidates) <= MAX_PAIRS:
        return candidates

    selected = np.linspace(
        0, len(candidates) - 1,
        MAX_PAIRS,
        dtype=int,
    )

    return [candidates[int(i)] for i in selected]


def detect_and_match(image_a, image_b):
    orb = cv2.ORB_create(
        nfeatures=3500,
        scaleFactor=1.2,
        nlevels=8,
    )

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)

    key_a, des_a = orb.detectAndCompute(gray_a, None)
    key_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    forward = matcher.knnMatch(des_a, des_b, k=2)
    backward = matcher.knnMatch(des_b, des_a, k=2)

    reverse_pairs = set()

    for matches in backward:
        if len(matches) != 2:
            continue
        first, second = matches
        if first.distance < 0.75 * second.distance:
            reverse_pairs.add(
                (first.trainIdx, first.queryIdx)
            )

    good = []

    for matches in forward:
        if len(matches) != 2:
            continue

        first, second = matches

        if (
            first.distance < 0.75 * second.distance
            and (first.queryIdx, first.trainIdx)
            in reverse_pairs
        ):
            good.append(first)

    good.sort(key=lambda match: match.distance)

    if len(good) < MIN_MATCHES:
        return {
            "matches": good,
            "key_a": key_a,
            "key_b": key_b,
            "points_a": None,
            "points_b": None,
        }

    points_a = np.array(
        [key_a[m.queryIdx].pt for m in good],
        dtype=np.float64,
    )

    points_b = np.array(
        [key_b[m.trainIdx].pt for m in good],
        dtype=np.float64,
    )

    return {
        "matches": good,
        "key_a": key_a,
        "key_b": key_b,
        "points_a": points_a,
        "points_b": points_b,
    }


def visual_rotation(cam, matches):
    if matches is None:
        return None

    p1 = matches["points_a"]
    p2 = matches["points_b"]

    if p1 is None or len(p1) < MIN_MATCHES:
        return None

    # Convert raw fisheye pixels into normalized rays.
    rays_a = cv2.fisheye.undistortPoints(
        p1.reshape(-1, 1, 2),
        cam.K,
        cam.D,
    ).reshape(-1, 2)

    rays_b = cv2.fisheye.undistortPoints(
        p2.reshape(-1, 1, 2),
        cam.K,
        cam.D,
    ).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        rays_a,
        rays_b,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.003,
    )

    if E is None or mask is None:
        return None

    # OpenCV may return multiple essential matrices.
    # Do not arbitrarily choose the first one.
    if E.shape != (3, 3):
        return None

    inliers, R, translation, recovered_mask = (
        cv2.recoverPose(
            E,
            rays_a,
            rays_b,
            cameraMatrix=np.eye(3),
            mask=mask.copy(),
        )
    )

    if inliers < MIN_INLIERS:
        return None

    return {
        "R": R,
        "inliers": int(inliers),
        "matches": len(p1),
        "inlier_fraction": float(inliers / len(p1)),
    }


def predicted_relative_rotation(
    cam,
    q_a,
    q_b,
    convention,
):
    R_a = quaternion_xyzw_to_rotation(q_a)
    R_b = quaternion_xyzw_to_rotation(q_b)

    if convention == "imu_to_world":
        # R_a and R_b map IMU -> world.
        R_imu_b_imu_a = R_b.T @ R_a

    elif convention == "world_to_imu":
        # R_a and R_b map world -> IMU.
        R_imu_b_imu_a = R_b @ R_a.T

    else:
        raise ValueError("Unknown convention")

    R_cam_imu = cam.T_cam_imu[:3, :3]

    return (
        R_cam_imu
        @ R_imu_b_imu_a
        @ R_cam_imu.T
    )


def rotation_error_degrees(R_est, R_pred):
    difference = R_est @ R_pred.T

    cosine = (np.trace(difference) - 1) / 2
    cosine = float(np.clip(cosine, -1, 1))

    return float(np.degrees(np.arccos(cosine)))


def rotation_angle_degrees(R):
    cosine = (np.trace(R) - 1) / 2
    return float(
        np.degrees(
            np.arccos(np.clip(cosine, -1, 1))
        )
    )


def create_match_panel(a, b, match_data, label):
    if match_data is None:
        panel = np.hstack([a, b])
    else:
        panel = cv2.drawMatches(
            a,
            match_data["key_a"],
            b,
            match_data["key_b"],
            match_data["matches"][:45],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

    target_width = 1100
    scale = target_width / panel.shape[1]

    panel = cv2.resize(
        panel,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1], 42),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        panel,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return panel


def summarize(values):
    if not values:
        return None

    return {
        "count": len(values),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cam = load_calibration("cam0")
    valid = load_records()
    pairs = choose_pairs(valid)

    results = []
    panels = []

    for index_a, index_b in pairs:
        a = valid[index_a]
        b = valid[index_b]

        image_a = cv2.imread(
            str(FRAMES / a["filename"])
        )
        image_b = cv2.imread(
            str(FRAMES / b["filename"])
        )

        result = {
            "frame_a": index_a,
            "frame_b": index_b,
            "status": "INCONCLUSIVE",
        }

        if image_a is None or image_b is None:
            result["reason"] = "IMAGE_READ_FAILURE"
            results.append(result)
            continue

        matches = detect_and_match(image_a, image_b)
        estimate = visual_rotation(cam, matches)

        label = f"Frames {index_a} -> {index_b}"

        if estimate is None:
            result["reason"] = (
                "INSUFFICIENT_RELIABLE_VISUAL_GEOMETRY"
            )
            result["mutual_matches"] = (
                len(matches["matches"])
                if matches is not None else 0
            )
        else:
            errors = {}

            for convention in (
                "imu_to_world",
                "world_to_imu",
            ):
                predicted = predicted_relative_rotation(
                    cam,
                    a["orientation"],
                    b["orientation"],
                    convention,
                )

                errors[convention] = (
                    rotation_error_degrees(
                        estimate["R"],
                        predicted,
                    )
                )

            result.update({
                "status": "MEASURED_NOT_VERIFIED",
                "mutual_matches": estimate["matches"],
                "essential_inliers": estimate["inliers"],
                "inlier_fraction": (
                    estimate["inlier_fraction"]
                ),
                "visual_rotation_degrees": (
                    rotation_angle_degrees(
                        estimate["R"]
                    )
                ),
                "rotation_error_degrees": errors,
            })

            label += (
                f" | {estimate['inliers']} inliers"
                f" | errors: "
                f"{errors['imu_to_world']:.1f} / "
                f"{errors['world_to_imu']:.1f} deg"
            )

        results.append(result)

        if len(panels) < 6:
            panels.append(
                create_match_panel(
                    image_a, image_b, matches, label
                )
            )

        print(label, "-", result["status"])

    error_a = []
    error_b = []

    for result in results:
        errors = result.get("rotation_error_degrees")
        if errors is None:
            continue

        error_a.append(errors["imu_to_world"])
        error_b.append(errors["world_to_imu"])

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "SITEWITNESS_VISUAL_POSE_FEASIBILITY",
        "opencv_version": cv2.__version__,
        "camera": "cam0",
        "available_valid_poses": len(valid),
        "sampled_pairs": len(pairs),
        "measured_pairs": len(error_a),
        "rotation_error_degrees": {
            "imu_to_world": summarize(error_a),
            "world_to_imu": summarize(error_b),
        },
        "pairs": results,
        "limitations": [
            "Essential-matrix rotation can be unreliable "
            "under low parallax, repeated patterns, "
            "moving objects, or planar scenes.",
            "The experiment assumes xyzw quaternion "
            "component order for arithmetic only.",
            "The two pose-direction hypotheses do not "
            "exhaust possible frame conventions.",
            "Frame-to-pose timestamp association "
            "and coordinate units are not independently verified.",
            "No architectural CAD geometry is projected.",
            "No camera-to-CAD registration is verified.",
            "No construction progress is inferred.",
        ],
        "pose_convention_verified": False,
        "cad_registration_verified": False,
        "next_action": (
            "Review image matches and pose rotation errors "
            "before attempting experimental CAD projection."
        ),
    }

    report_path = OUTPUT / "visual_feasibility.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if panels:
        sheet = np.vstack(panels)
        cv2.imwrite(
            str(OUTPUT / "visual_matches.jpg"),
            sheet,
        )

    print("\nSITEWITNESS — VISUAL FEASIBILITY")
    print("=" * 50)
    print("Valid poses:", len(valid))
    print("Pairs tested:", len(pairs))
    print("Pairs measured:", len(error_a))
    print("IMU-to-world errors:", summarize(error_a))
    print("World-to-IMU errors:", summarize(error_b))
    print("Pose convention: UNVERIFIED")
    print("CAD registration: UNVERIFIED")
    print("Report:", report_path)
    print("Visual sheet:", OUTPUT / "visual_matches.jpg")


if __name__ == "__main__":
    main()
