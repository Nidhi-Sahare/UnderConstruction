
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

DATA_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

FRAME_DIR = (
    DATA_ROOT
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frames"
)

MANIFEST_PATH = FRAME_DIR.parent / "frame_manifest.csv"

OUTPUT_DIR = ROOT / "outputs" / "visual_quality"
OUTPUT_PATH = OUTPUT_DIR / "walkthrough_quality.json"

ANALYSIS_WIDTH = 480
DARK_THRESHOLD = 35
BRIGHT_THRESHOLD = 220


def safe_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_manifest():
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def image_metrics(image):
    height, width = image.shape[:2]

    scale = ANALYSIS_WIDTH / width

    resized = cv2.resize(
        image,
        (
            ANALYSIS_WIDTH,
            max(1, round(height * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    # Restrict measurements to the central circular
    # fisheye region. Black corners must not distort
    # exposure or detail statistics.
    h, w = gray.shape

    yy, xx = np.ogrid[:h, :w]

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    radius = min(w, h) * 0.43

    mask = (
        (xx - cx) ** 2 +
        (yy - cy) ** 2
    ) <= radius ** 2

    pixels = gray[mask]

    if pixels.size == 0:
        raise RuntimeError("Empty analysis mask")

    blur_map = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    laplacian_variance = float(
        np.var(blur_map[mask])
    )

    edges = cv2.Canny(
        gray,
        80,
        160,
    )

    edge_fraction = float(
        np.mean(edges[mask] > 0)
    )

    mean_brightness = float(
        np.mean(pixels)
    )

    dark_fraction = float(
        np.mean(pixels < DARK_THRESHOLD)
    )

    bright_fraction = float(
        np.mean(pixels > BRIGHT_THRESHOLD)
    )

    contrast = float(
        np.std(pixels)
    )

    return {
        "width": width,
        "height": height,
        "mean_brightness": round(mean_brightness, 3),
        "contrast_std": round(contrast, 3),
        "laplacian_variance": round(
            laplacian_variance, 3
        ),
        "edge_fraction": round(edge_fraction, 5),
        "dark_fraction": round(dark_fraction, 5),
        "bright_fraction": round(bright_fraction, 5),
    }, gray, mask


def similarity_to_previous(gray, previous, mask):
    if previous is None:
        return None

    if previous.shape != gray.shape:
        return None

    # Difference is measured on normalized intensity,
    # avoiding a misleading absolute-brightness-only
    # comparison.
    current = gray.astype(np.float32)
    earlier = previous.astype(np.float32)

    difference = np.abs(current - earlier)

    return round(
        float(np.mean(difference[mask])) / 255.0,
        5,
    )


def classify_quality(metrics):
    # Heuristic screening only. These thresholds are
    # not calibrated against a labeled quality dataset.
    if metrics["mean_brightness"] < 45:
        return "LOW_LIGHT"

    if metrics["bright_fraction"] > 0.55:
        return "OVEREXPOSED"

    if metrics["laplacian_variance"] < 25:
        return "LOW_DETAIL_OR_BLUR"

    if metrics["edge_fraction"] < 0.015:
        return "LOW_TEXTURE"

    return "USABLE_CANDIDATE"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_manifest()

    records = []
    previous_gray = None

    for index, row in enumerate(rows):
        filename = Path(
            str(row.get("filename", "")).replace("\\", "/")
        ).name

        if not filename:
            filename = f"cam0_{index:05d}.jpg"

        path = FRAME_DIR / filename

        image = cv2.imread(str(path))

        if image is None:
            records.append({
                "frame_index": index,
                "filename": filename,
                "quality_class": "IMAGE_UNAVAILABLE",
            })
            previous_gray = None
            continue

        metrics, gray, mask = image_metrics(image)

        difference = similarity_to_previous(
            gray,
            previous_gray,
            mask,
        )

        previous_gray = gray

        quality = classify_quality(metrics)

        records.append({
            "frame_index": index,
            "filename": filename,
            "timestamp_s": safe_float(
                row.get("header_timestamp_s")
            ),
            "quality_class": quality,
            "temporal_difference": difference,
            "metrics": metrics,
            "construction_progress": "UNKNOWN",
            "visibility_verified": False,
        })

        if index % 30 == 0:
            print(
                f"Processed {index + 1}/{len(rows)} frames"
            )

    counts = Counter(
        record["quality_class"]
        for record in records
    )

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "sequence": SEQUENCE,
        "method": "DETERMINISTIC_OPENCV_IMAGE_SCREENING",
        "summary": {
            "total_frames": len(records),
            "quality_counts": dict(counts),
        },
        "limitations": [
            "Thresholds are heuristic and not benchmark-calibrated.",
            "Blur and low texture cannot always be distinguished.",
            "Frame differences may reflect camera motion.",
            "No object recognition or CAD registration is performed.",
            "No construction completion is inferred.",
        ],
        "frames": records,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== WALKTHROUGH VISUAL QUALITY ===")
    print("Frames:", len(records))

    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")

    print("Construction progress verified: NO")
    print(
        "Report:",
        OUTPUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
