
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

SEQUENCE = "floor_2_2025-10-28_run_2"

RAW_DIR = (
    ARCHIVE
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frames"
)

RECTIFIED_DIR = (
    ARCHIVE
    / "hilti_calibration"
    / "rectified_frames"
    / "cam0"
)

CALIBRATION_REPORT = (
    ARCHIVE
    / "hilti_calibration"
    / "camera_calibration_report.json"
)

TRAJECTORY_CSV = (
    ARCHIVE
    / "floorplan_registration"
    / "aligned_trajectory.csv"
)

FLOORPLAN = ROOT / "data" / "floorplans" / "floor_2.png"


def load_calibration():
    with CALIBRATION_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    calibration = report["calibration"]

    fx, fy, cx, cy = calibration["intrinsics"]

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    D = np.array(
        calibration["distortion_coefficients"],
        dtype=np.float64,
    )

    return K, D



def rectify_image(image, K, D):
    height, width = image.shape[:2]

    # Use a controlled perspective field of view rather than
    # the excessively wide automatic rectification matrix.
    focal = 0.75 * min(width, height)

    new_K = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        new_K,
        (width, height),
        cv2.CV_16SC2,
    )

    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def fit_image(image, width, height):
    ih, iw = image.shape[:2]

    scale = min(width / iw, height / ih)

    new_size = (
        max(1, int(iw * scale)),
        max(1, int(ih * scale)),
    )

    resized = cv2.resize(
        image,
        new_size,
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (height, width, 3),
        25,
        dtype=np.uint8,
    )

    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2

    canvas[
        y:y + resized.shape[0],
        x:x + resized.shape[1],
    ] = resized

    return canvas


def label_panel(image, title):
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 48),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        image,
        title,
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    return image


def load_trajectory():
    if not TRAJECTORY_CSV.exists():
        return []

    with TRAJECTORY_CSV.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def make_floorplan_panel(width, height, trajectory):
    if FLOORPLAN.exists():
        image = cv2.imread(str(FLOORPLAN))
    else:
        image = None

    if image is None:
        image = np.full(
            (height, width, 3),
            35,
            dtype=np.uint8,
        )

    image = fit_image(image, width, height)

    label_panel(
        image,
        "FLOOR PLAN / UNVERIFIED ALIGNMENT",
    )

    cv2.putText(
        image,
        f"Trajectory records: {len(trajectory)}",
        (18, height - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Camera-to-CAD registration not verified",
        (18, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 180, 255),
        1,
        cv2.LINE_AA,
    )

    return image


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Hilti image folder not found:\n{RAW_DIR}"
        )

    frames = sorted(RAW_DIR.glob("*.jpg"))

    if not frames:
        raise RuntimeError(
            "No Hilti JPEG frames found."
        )

    K, D = load_calibration()
    trajectory = load_trajectory()

    index = 0
    total = len(frames)

    print(f"Found {total} Hilti frames")
    print(f"Trajectory records: {len(trajectory)}")
    print("Controls: A = previous, D = next, Q = quit")

    while True:
        frame_path = frames[index]

        raw = cv2.imread(str(frame_path))

        if raw is None:
            raise RuntimeError(
                f"Could not read {frame_path}"
            )

        corrected_path = RECTIFIED_DIR / frame_path.name

        corrected = rectify_image(raw, K, D)

        panel_width = 640
        panel_height = 520

        raw_panel = fit_image(
            raw,
            panel_width,
            panel_height,
        )

        corrected_panel = fit_image(
            corrected,
            panel_width,
            panel_height,
        )

        label_panel(
            raw_panel,
            "ORIGINAL FISHEYE",
        )

        label_panel(
            corrected_panel,
            "RECTIFIED CAMERA",
        )

        floor_panel = make_floorplan_panel(
            panel_width,
            panel_height,
            trajectory,
        )

        combined = np.hstack(
            [
                raw_panel,
                corrected_panel,
                floor_panel,
            ]
        )

        cv2.putText(
            combined,
            f"Frame {index + 1}/{total} | {frame_path.name}",
            (18, panel_height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(
            "Under-construction | Hilti Viewer",
            combined,
        )

        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break

        if key in (ord("d"), 83):
            index = min(index + 1, total - 1)

        elif key in (ord("a"), 81):
            index = max(index - 1, 0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
