
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from progress.yolo_onnx import OnnxSegModel, render_overlay


ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

FRAMES_DIR = (
    ARCHIVE
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
    / "frames"
)

CALIBRATION_REPORT = (
    ARCHIVE
    / "hilti_calibration"
    / "camera_calibration_report.json"
)

OUTPUT_DIR = ROOT / "outputs" / "hilti_evaluation"

MODEL_PATH = ROOT / "models" / "underconstruction_seg_v0.onnx"
MODEL_CARD = ROOT / "models" / "model_card.json"

SAMPLE_COUNT = 12
CONFIDENCE = 0.45


def load_calibration():
    with CALIBRATION_REPORT.open("r", encoding="utf-8") as f:
        report = json.load(f)

    c = report["calibration"]
    fx, fy, cx, cy = c["intrinsics"]

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    D = np.array(
        c["distortion_coefficients"],
        dtype=np.float64,
    )

    return K, D


def make_rectification_maps(K, D, width, height):
    focal = 0.75 * min(width, height)

    new_K = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3, dtype=np.float64),
        new_K,
        (width, height),
        cv2.CV_16SC2,
    )


def fit_panel(image, width=480, height=480):
    ih, iw = image.shape[:2]

    scale = min(width / iw, height / ih)

    resized = cv2.resize(
        image,
        (
            max(1, int(iw * scale)),
            max(1, int(ih * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (height, width, 3),
        24,
        dtype=np.uint8,
    )

    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2

    canvas[
        y:y + resized.shape[0],
        x:x + resized.shape[1],
    ] = resized

    return canvas


def label(image, text):
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], 36),
        (15, 15, 15),
        -1,
    )

    cv2.putText(
        image,
        text,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )

    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames = sorted(FRAMES_DIR.glob("*.jpg"))

    if not frames:
        raise RuntimeError(
            f"No Hilti frames found at {FRAMES_DIR}"
        )

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"ONNX model not found: {MODEL_PATH}"
        )

    indices = np.linspace(
        0,
        len(frames) - 1,
        min(SAMPLE_COUNT, len(frames)),
        dtype=int,
    )

    K, D = load_calibration()

    model = OnnxSegModel(
        model_path=MODEL_PATH,
        card_path=MODEL_CARD,
    )

    records = []
    rows = []
    maps = None
    map_size = None

    for number, index in enumerate(indices, start=1):
        path = frames[int(index)]

        raw = cv2.imread(str(path))

        if raw is None:
            raise RuntimeError(
                f"Could not read {path}"
            )

        height, width = raw.shape[:2]

        if maps is None or map_size != (width, height):
            maps = make_rectification_maps(
                K, D, width, height
            )
            map_size = (width, height)

        corrected = cv2.remap(
            raw,
            maps[0],
            maps[1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        detections = model.infer(
            corrected,
            conf=CONFIDENCE,
            iou=0.7,
        )

        overlay = render_overlay(
            corrected.copy(),
            detections,
        )

        counts = Counter(
            d.class_name for d in detections
        )

        detection_data = [
            {
                "class_name": d.class_name,
                "score": float(d.score),
                "bbox": list(d.bbox),
                "area_fraction": float(d.area_fraction),
            }
            for d in detections
        ]

        records.append(
            {
                "frame": path.name,
                "frame_index": int(index),
                "detections": detection_data,
                "counts": dict(counts),
            }
        )

        cv2.imwrite(
            str(OUTPUT_DIR / f"{path.stem}_overlay.jpg"),
            overlay,
        )

        original_panel = label(
            fit_panel(corrected),
            f"{path.name} | RECTIFIED",
        )

        overlay_panel = label(
            fit_panel(overlay),
            f"MODEL CANDIDATES: {len(detections)}",
        )

        rows.append(
            np.hstack(
                [original_panel, overlay_panel]
            )
        )

        print(
            f"[{number}/{len(indices)}] "
            f"{path.name}: {dict(counts)}"
        )

    sheet = np.vstack(rows)

    sheet_path = OUTPUT_DIR / "evaluation_contact_sheet.jpg"

    cv2.imwrite(
        str(sheet_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )

    report = {
        "dataset": "Hilti floor_2_2025-10-28_run_2 cam0",
        "model": MODEL_PATH.name,
        "sampled_frames": len(records),
        "confidence_threshold": CONFIDENCE,
        "ground_truth_available": False,
        "accuracy_measured": False,
        "records": records,
    }

    report_path = OUTPUT_DIR / "evaluation_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation complete.")
    print(f"Contact sheet: {sheet_path}")
    print(f"Report: {report_path}")
    print(
        "Note: This is a qualitative model audit, "
        "not a measured accuracy evaluation."
    )


if __name__ == "__main__":
    main()
