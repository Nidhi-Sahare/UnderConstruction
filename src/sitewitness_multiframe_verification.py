
"""
BuildProgress - SiteWitness Calibrated Multi-Frame Verification
==============================================================

Uses real Hilti cam0 fisheye calibration to test geometric
correspondence between construction walkthrough frames.

Pipeline:
    Selected anchor frames
    -> neighboring observations
    -> ORB matching
    -> fisheye coordinate undistortion
    -> essential matrix RANSAC
    -> inlier spatial coverage
    -> evidence images
    -> corroboration / abstention

Run:
    python -m src.sitewitness_multiframe_verification

This verifies image correspondences only.

It does NOT verify:
    CAD registration
    construction compliance
    architectural feature identity
    physical dimensions
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]

FRAMES = ROOT / "sample_gallery/real_floor_2/frames"

INSPECTION_REPORT = (
    ROOT
    / "outputs/agentic_inspection/floor_2_inspection_report.json"
)

CALIBRATION = (
    ROOT
    / "data/hilti_calibration/kalibr_imucam_chain.yaml"
)

OUTPUT = ROOT / "outputs/sitewitness/multiframe"

NEIGHBOR_OFFSETS = (-4, -2, -1, 1, 2, 4)

WORK_WIDTH = 900
MAX_FEATURES = 2500

MIN_GOOD_MATCHES = 35
MIN_RANSAC_INLIERS = 22
MIN_INLIER_RATIO = 0.35

MIN_SUPPORTING_NEIGHBORS = 2
MAX_VISUALIZATIONS_PER_ANCHOR = 2

MIN_INLIER_GRID_CELLS = 4
GRID_SIZE = 4

RATIO_TEST = 0.75

# Approximate normalized-camera threshold, corresponding
# to about 2.5 pixels at focal length ~465 px.
NORMALIZED_RANSAC_THRESHOLD = 2.5 / 465.0


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(str(path))

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)

    if not match:
        raise ValueError(
            f"Cannot determine frame number: {path}"
        )

    return int(match.group(1))


def frame_path(index: int) -> Path:
    return FRAMES / f"cam0_{index:05d}.jpg"


def load_calibration():
    raw = CALIBRATION.read_text(encoding="utf-8-sig")
    lines = raw.splitlines()

    # Kalibr/OpenCV-style YAML uses %YAML:1.0.
    if lines and lines[0].startswith("%YAML:"):
        lines = lines[1:]

    data = yaml.safe_load("\n".join(lines))

    cam = data["cam0"]

    fx, fy, cx, cy = map(
        float,
        cam["intrinsics"]
    )

    original_K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    D = np.asarray(
        cam["distortion_coeffs"],
        dtype=np.float64
    ).reshape(4, 1)

    resolution = cam["resolution"]

    return {
        "K": original_K,
        "D": D,
        "width": int(resolution[0]),
        "height": int(resolution[1]),
        "distortion_model": cam.get(
            "distortion_model"
        ),
    }


def prepare_image(path: Path):
    image = cv2.imread(str(path))

    if image is None:
        return None

    original_height, original_width = image.shape[:2]

    if original_width > WORK_WIDTH:
        scale = WORK_WIDTH / original_width

        image = cv2.resize(
            image,
            (
                WORK_WIDTH,
                round(original_height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return {
        "image": image,
        "gray": gray,
        "original_width": original_width,
        "original_height": original_height,
    }


def scaled_intrinsics(calibration, width, height):
    sx = width / calibration["width"]
    sy = height / calibration["height"]

    K = calibration["K"].copy()

    K[0, 0] *= sx
    K[0, 2] *= sx

    K[1, 1] *= sy
    K[1, 2] *= sy

    return K


def extract_features(gray):
    orb = cv2.ORB_create(
        nfeatures=MAX_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
    )

    keypoints, descriptors = orb.detectAndCompute(
        gray,
        None
    )

    return keypoints or [], descriptors


def match_features(desc_a, desc_b):
    if desc_a is None or desc_b is None:
        return []

    if len(desc_a) < 2 or len(desc_b) < 2:
        return []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    pairs = matcher.knnMatch(
        desc_a,
        desc_b,
        k=2
    )

    good = []

    for pair in pairs:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < RATIO_TEST * second.distance:
            good.append(first)

    return sorted(
        good,
        key=lambda match: match.distance
    )


def grid_coverage(points, width, height):
    if len(points) == 0:
        return 0

    occupied = set()

    for x, y in points:
        col = min(
            GRID_SIZE - 1,
            max(0, int(x * GRID_SIZE / width))
        )

        row = min(
            GRID_SIZE - 1,
            max(0, int(y * GRID_SIZE / height))
        )

        occupied.add((row, col))

    return len(occupied)


def calibrated_geometry(
    kp_a,
    kp_b,
    matches,
    K_a,
    K_b,
    D,
    width,
    height,
):
    empty = {
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "inlier_grid_cells": 0,
        "inlier_mask": None,
        "model": "CALIBRATED_ESSENTIAL_MATRIX",
    }

    if len(matches) < 8:
        return empty

    points_a = np.float64([
        kp_a[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    points_b = np.float64([
        kp_b[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    # Convert distorted fisheye pixels into normalized
    # pinhole camera coordinates.
    undistorted_a = cv2.fisheye.undistortPoints(
        points_a,
        K_a,
        D
    )

    undistorted_b = cv2.fisheye.undistortPoints(
        points_b,
        K_b,
        D
    )

    normalized_a = undistorted_a.reshape(-1, 2)
    normalized_b = undistorted_b.reshape(-1, 2)

    if (
        not np.all(np.isfinite(normalized_a))
        or not np.all(np.isfinite(normalized_b))
    ):
        return empty

    E, mask = cv2.findEssentialMat(
        normalized_a,
        normalized_b,
        cameraMatrix=np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=NORMALIZED_RANSAC_THRESHOLD,
    )

    if E is None or mask is None:
        return empty

    mask = mask.reshape(-1).astype(bool)

    if len(mask) != len(matches):
        return empty

    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / len(matches)

    original_pixels = points_a.reshape(-1, 2)

    coverage = grid_coverage(
        original_pixels[mask],
        width,
        height,
    )

    return {
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 4),
        "inlier_grid_cells": coverage,
        "inlier_mask": mask,
        "model": "CALIBRATED_ESSENTIAL_MATRIX",
    }


def draw_evidence(
    anchor,
    neighbor,
    matches,
    inlier_mask,
    label,
):
    selected = []

    if inlier_mask is not None:
        selected = [
            match
            for match, accepted in zip(
                matches, inlier_mask
            )
            if accepted
        ]

    selected = selected[:100]

    visual = cv2.drawMatches(
        anchor["image"],
        anchor["keypoints"],
        neighbor["image"],
        neighbor["keypoints"],
        selected,
        None,
        flags=(
            cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        ),
    )

    cv2.rectangle(
        visual,
        (0, 0),
        (visual.shape[1], 55),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        visual,
        label,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return visual


def verify_pair(
    anchor_index,
    neighbor_index,
    cache,
    calibration,
):
    def get(index):
        if index not in cache:
            prepared = prepare_image(
                frame_path(index)
            )

            if prepared is None:
                cache[index] = None
            else:
                kp, desc = extract_features(
                    prepared["gray"]
                )

                prepared["keypoints"] = kp
                prepared["descriptors"] = desc

                cache[index] = prepared

        return cache[index]

    anchor = get(anchor_index)
    neighbor = get(neighbor_index)

    if anchor is None or neighbor is None:
        return {
            "anchor": anchor_index,
            "neighbor": neighbor_index,
            "status": "ABSTAIN",
            "reason": "FRAME_UNREADABLE",
        }, None

    matches = match_features(
        anchor["descriptors"],
        neighbor["descriptors"]
    )

    height, width = anchor["gray"].shape

    neighbor_height, neighbor_width = (
        neighbor["gray"].shape
    )

    K_a = scaled_intrinsics(
        calibration, width, height
    )

    K_b = scaled_intrinsics(
        calibration,
        neighbor_width,
        neighbor_height,
    )

    geometry = calibrated_geometry(
        anchor["keypoints"],
        neighbor["keypoints"],
        matches,
        K_a,
        K_b,
        calibration["D"],
        width,
        height,
    )

    sufficient = (
        len(matches) >= MIN_GOOD_MATCHES
        and geometry["inlier_count"]
        >= MIN_RANSAC_INLIERS
        and geometry["inlier_ratio"]
        >= MIN_INLIER_RATIO
        and geometry["inlier_grid_cells"]
        >= MIN_INLIER_GRID_CELLS
    )

    status = (
        "CALIBRATED_CORRESPONDENCE_SUPPORTED"
        if sufficient
        else "ABSTAIN"
    )

    result = {
        "anchor": anchor_index,
        "neighbor": neighbor_index,
        "good_matches": len(matches),
        "ransac_inliers": geometry["inlier_count"],
        "inlier_ratio": geometry["inlier_ratio"],
        "inlier_grid_cells": geometry[
            "inlier_grid_cells"
        ],
        "status": status,
        "reason": (
            "CALIBRATED_GEOMETRY_THRESHOLDS_MET"
            if sufficient
            else "INSUFFICIENT_CALIBRATED_SUPPORT"
        ),
    }

    visual = None

    if sufficient:
        visual = draw_evidence(
            anchor,
            neighbor,
            matches,
            geometry["inlier_mask"],
            (
                f"Frames {anchor_index}/{neighbor_index}"
                f" | Inliers {geometry['inlier_count']}"
                f" | Cells {geometry['inlier_grid_cells']}"
            ),
        )

    return result, visual


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    inspection = load_json(
        INSPECTION_REPORT
    )

    selected = inspection.get(
        "selected_review_frames", []
    )

    if not selected:
        raise RuntimeError(
            "No selected review frames found. "
            "Run the adaptive inspection agent first."
        )

    calibration = load_calibration()

    anchors = [
        frame_number(Path(path))
        for path in selected
    ]

    cache = {}
    results = []

    for anchor_index in anchors:
        print(
            f"\nInspecting anchor frame {anchor_index}"
        )

        pair_results = []
        supporting = 0
        saved = 0

        for offset in NEIGHBOR_OFFSETS:
            neighbor_index = anchor_index + offset

            if not frame_path(
                neighbor_index
            ).is_file():
                continue

            result, visual = verify_pair(
                anchor_index,
                neighbor_index,
                cache,
                calibration,
            )

            pair_results.append(result)

            if (
                result["status"]
                == "CALIBRATED_CORRESPONDENCE_SUPPORTED"
            ):
                supporting += 1

                if (
                    visual is not None
                    and saved
                    < MAX_VISUALIZATIONS_PER_ANCHOR
                ):
                    destination = (
                        OUTPUT
                        / (
                            f"evidence_{anchor_index:05d}"
                            f"_{neighbor_index:05d}.jpg"
                        )
                    )

                    if not cv2.imwrite(
                        str(destination), visual
                    ):
                        raise RuntimeError(
                            f"Cannot save {destination}"
                        )

                    result["evidence_image"] = (
                        destination.relative_to(
                            ROOT
                        ).as_posix()
                    )

                    saved += 1

            print(
                f"  Neighbor {neighbor_index:03d}"
                f" | matches={result.get('good_matches', 0):4d}"
                f" | inliers={result.get('ransac_inliers', 0):4d}"
                f" | cells={result.get('inlier_grid_cells', 0):2d}"
                f" | {result['status']}"
            )

        anchor_status = (
            "CORROBORATED_VISUAL_CORRESPONDENCE"
            if supporting >= MIN_SUPPORTING_NEIGHBORS
            else "ABSTAIN"
        )

        results.append({
            "anchor_frame": anchor_index,
            "supporting_neighbors": supporting,
            "required_supporting_neighbors": (
                MIN_SUPPORTING_NEIGHBORS
            ),
            "status": anchor_status,
            "pair_results": pair_results,
        })

        print(
            "  Anchor result:",
            anchor_status
        )

    corroborated = sum(
        item["status"]
        == "CORROBORATED_VISUAL_CORRESPONDENCE"
        for item in results
    )

    output_report = {
        "schema_version": "2.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "agent": "SiteWitness",
        "floor": 2,
        "opencv_version": cv2.__version__,
        "method": (
            "ORB + calibrated fisheye undistortion "
            "+ essential matrix RANSAC "
            "+ spatial inlier coverage"
        ),
        "camera_calibration": {
            "camera": "cam0",
            "distortion_model": calibration[
                "distortion_model"
            ],
            "resolution": [
                calibration["width"],
                calibration["height"],
            ],
        },
        "anchor_count": len(anchors),
        "corroborated_anchor_count": corroborated,
        "abstained_anchor_count": (
            len(anchors) - corroborated
        ),
        "anchors": results,
        "cad_registration_verified": False,
        "construction_compliance_verified": False,
        "limitations": [
            "Feature correspondence does not prove architectural identity.",
            "Adjacent frames are correlated observations.",
            "A fitted essential matrix does not establish metric scale.",
            "The thresholds are heuristic and not benchmark-calibrated.",
            "Calibrated correspondence does not establish CAD registration.",
            "Construction compliance requires independent design evidence.",
        ],
    }

    destination = OUTPUT / "verification_report.json"

    destination.write_text(
        json.dumps(
            output_report,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSITEWITNESS - CALIBRATED VERIFICATION")
    print("=" * 55)
    print("Anchors:", len(anchors))
    print("Corroborated:", corroborated)
    print("Abstained:", len(anchors) - corroborated)
    print("CAD registration: UNVERIFIED")
    print("Construction compliance: UNVERIFIED")
    print("Report:", destination)


if __name__ == "__main__":
    main()
