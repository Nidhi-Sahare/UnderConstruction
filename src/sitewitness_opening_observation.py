
"""
SiteWitness — Calibration-Aware Opening Hypothesis Detector
==========================================================

Classical CV only. No pretrained models or LLMs.

Uses:
- Hilti cam0 fisheye calibration
- Rectified perspective image
- LSD line segments
- Data-driven dominant direction estimation
- Parallel-side and crossbar geometry
- Candidate deduplication
- Visual evidence and structured abstention

Run:
    python -m src.sitewitness_opening_observation

Outputs:
    outputs/sitewitness/openings/observations.json
    outputs/sitewitness/openings/frame_XXXXX.jpg

A detected candidate is NOT a verified doorway, window,
or construction-compliance finding.
"""

from __future__ import annotations

import json
import math
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

OUTPUT = ROOT / "outputs/sitewitness/openings"

WORK_WIDTH = 900

# Retains a wide central field of view while limiting the
# extreme peripheral stretching of fisheye rectification.
RECTIFICATION_BALANCE = 0.55

MIN_LINE_LENGTH = 35.0
MIN_SIDE_LENGTH = 65.0

PARALLEL_TOLERANCE_DEG = 13.0
CROSSBAR_TOLERANCE_DEG = 20.0

MIN_SIDE_SEPARATION = 45.0
MAX_SIDE_SEPARATION_FRACTION = 0.72

MIN_OVERLAP_LENGTH = 65.0
MIN_CROSSBAR_SUPPORT = 0.35

MIN_ASPECT_RATIO = 0.45
MAX_ASPECT_RATIO = 6.0

MAX_CANDIDATES_PER_FRAME = 8

GRID_SIZE = 4


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def frame_number(path: Path):
    match = re.search(r"(\d+)$", path.stem)

    if match is None:
        raise ValueError(
            f"Cannot parse frame number: {path}"
        )

    return int(match.group(1))


def frame_path(index: int):
    return FRAMES / f"cam0_{index:05d}.jpg"


def load_camera():
    if not CALIBRATION.is_file():
        raise FileNotFoundError(CALIBRATION)

    text = CALIBRATION.read_text(
        encoding="utf-8-sig"
    )

    lines = text.splitlines()

    if lines and lines[0].startswith("%YAML:"):
        lines = lines[1:]

    data = yaml.safe_load("\n".join(lines))

    cam = data["cam0"]

    if cam.get("distortion_model") != "equidistant":
        raise ValueError(
            "Expected Kalibr equidistant fisheye model"
        )

    fx, fy, cx, cy = map(
        float,
        cam["intrinsics"]
    )

    K = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    D = np.asarray(
        cam["distortion_coeffs"],
        dtype=np.float64,
    ).reshape(4, 1)

    width, height = map(
        int,
        cam["resolution"]
    )

    return {
        "K": K,
        "D": D,
        "width": width,
        "height": height,
    }


def rectify_image(image, camera):
    height, width = image.shape[:2]

    sx = width / camera["width"]
    sy = height / camera["height"]

    K = camera["K"].copy()

    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy

    focal = 0.75 * min(width, height)

    new_K = np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K,
        camera["D"],
        np.eye(3, dtype=np.float64),
        new_K,
        (width, height),
        cv2.CV_32FC1,
    )

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    valid_mask = (
        (map_x >= 0)
        & (map_x < width - 1)
        & (map_y >= 0)
        & (map_y < height - 1)
    ).astype(np.uint8) * 255

    return rectified, valid_mask


def resize_image(image, mask):
    height, width = image.shape[:2]

    if width <= WORK_WIDTH:
        return image, mask

    scale = WORK_WIDTH / width

    size = (
        WORK_WIDTH,
        round(height * scale),
    )

    image = cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA,
    )

    mask = cv2.resize(
        mask,
        size,
        interpolation=cv2.INTER_NEAREST,
    )

    return image, mask


def segment_length(line):
    x1, y1, x2, y2 = line

    return math.hypot(
        x2 - x1,
        y2 - y1
    )


def segment_angle(line):
    x1, y1, x2, y2 = line

    return math.atan2(
        y2 - y1,
        x2 - x1
    ) % math.pi


def angular_difference(a, b):
    difference = abs(a - b) % math.pi

    return min(
        difference,
        math.pi - difference
    )


def extract_segments(image, valid_mask):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0.8
    )

    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    detected = detector.detect(gray)[0]

    if detected is None:
        return []

    height, width = gray.shape

    segments = []

    for raw in detected.reshape(-1, 4):
        line = tuple(
            float(value)
            for value in raw
        )

        length = segment_length(line)

        if length < MIN_LINE_LENGTH:
            continue

        x1, y1, x2, y2 = line

        mx = int((x1 + x2) / 2)
        my = int((y1 + y2) / 2)

        if not (
            0 <= mx < width
            and 0 <= my < height
        ):
            continue

        if valid_mask[my, mx] == 0:
            continue

        segments.append({
            "line": line,
            "length": length,
            "angle": segment_angle(line),
        })

    return segments


def estimate_dominant_directions(segments):
    """
    Estimate line orientations modulo 180 degrees.

    Uses a weighted circular histogram of doubled angles.
    No assumption that camera 'up' equals image 'up'.
    """

    if not segments:
        return []

    bins = 36

    histogram = np.zeros(
        bins,
        dtype=np.float64
    )

    for segment in segments:
        angle = segment["angle"]

        index = int(
            angle / math.pi * bins
        ) % bins

        histogram[index] += segment["length"]

    # Circular smoothing.
    smoothed = np.zeros_like(histogram)

    for offset, weight in (
        (-2, 1),
        (-1, 2),
        (0, 3),
        (1, 2),
        (2, 1),
    ):
        smoothed += (
            np.roll(histogram, offset) * weight
        )

    peaks = np.argsort(smoothed)[::-1]

    directions = []

    for index in peaks:
        angle = (
            (float(index) + 0.5)
            / bins
            * math.pi
        )

        if all(
            angular_difference(
                angle,
                existing
            ) > math.radians(20)
            for existing in directions
        ):
            directions.append(angle)

        if len(directions) >= 3:
            break

    return directions


def unit_vector(angle):
    return np.array(
        [math.cos(angle), math.sin(angle)],
        dtype=np.float64
    )


def line_endpoints(line):
    x1, y1, x2, y2 = line

    return (
        np.array([x1, y1], dtype=np.float64),
        np.array([x2, y2], dtype=np.float64),
    )


def project_segment(line, axis):
    p1, p2 = line_endpoints(line)

    values = (
        float(np.dot(p1, axis)),
        float(np.dot(p2, axis)),
    )

    return min(values), max(values)


def line_midpoint(line):
    p1, p2 = line_endpoints(line)

    return (p1 + p2) / 2.0


def point_line_distance(point, line):
    p1, p2 = line_endpoints(line)

    direction = p2 - p1

    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        return float(
            np.linalg.norm(point - p1)
        )

    delta = point - p1

    return abs(
        direction[0] * delta[1]
        - direction[1] * delta[0]
    ) / norm


def overlap_interval(a, b):
    low = max(a[0], b[0])
    high = min(a[1], b[1])

    return low, high, max(0.0, high - low)


def candidate_from_pair(
    side_a,
    side_b,
    crossbars,
    side_direction,
    image_width,
    image_height,
):
    axis = unit_vector(side_direction)

    normal = np.array(
        [-axis[1], axis[0]],
        dtype=np.float64,
    )

    line_a = side_a["line"]
    line_b = side_b["line"]

    midpoint_a = line_midpoint(line_a)
    midpoint_b = line_midpoint(line_b)

    separation = abs(
        np.dot(
            midpoint_b - midpoint_a,
            normal
        )
    )

    if separation < MIN_SIDE_SEPARATION:
        return None

    if separation > (
        MAX_SIDE_SEPARATION_FRACTION
        * image_width
    ):
        return None

    range_a = project_segment(
        line_a,
        axis
    )

    range_b = project_segment(
        line_b,
        axis
    )

    overlap_start, overlap_end, overlap = (
        overlap_interval(
            range_a,
            range_b
        )
    )

    if overlap < MIN_OVERLAP_LENGTH:
        return None

    aspect = overlap / max(
        separation,
        1.0
    )

    if not (
        MIN_ASPECT_RATIO
        <= aspect
        <= MAX_ASPECT_RATIO
    ):
        return None

    # A transverse segment must plausibly connect the
    # two parallel boundaries near one end.
    supporting_crossbars = []

    for crossbar in crossbars:
        cross_line = crossbar["line"]

        midpoint = line_midpoint(
            cross_line
        )

        along = float(
            np.dot(
                midpoint,
                axis
            )
        )

        if not (
            overlap_start - 35
            <= along
            <= overlap_end + 35
        ):
            continue

        cross_length = crossbar["length"]

        support_ratio = (
            cross_length / separation
        )

        if support_ratio < MIN_CROSSBAR_SUPPORT:
            continue

        # Crossbar should be near both side lines,
        # or span a meaningful portion between them.
        cross_p1, cross_p2 = line_endpoints(
            cross_line
        )

        d_a = min(
            point_line_distance(
                cross_p1,
                line_a
            ),
            point_line_distance(
                cross_p2,
                line_a
            ),
        )

        d_b = min(
            point_line_distance(
                cross_p1,
                line_b
            ),
            point_line_distance(
                cross_p2,
                line_b
            ),
        )

        if (
            d_a > separation * 0.5
            or d_b > separation * 0.5
        ):
            continue

        supporting_crossbars.append(
            crossbar
        )

    if not supporting_crossbars:
        return None

    points = []

    for line in (
        line_a,
        line_b,
    ):
        p1, p2 = line_endpoints(line)

        points.extend([p1, p2])

    points = np.asarray(
        points,
        dtype=np.float32
    )

    rectangle = cv2.minAreaRect(
        points
    )

    box = cv2.boxPoints(
        rectangle
    )

    x1 = max(
        0,
        int(np.floor(box[:, 0].min()))
    )

    y1 = max(
        0,
        int(np.floor(box[:, 1].min()))
    )

    x2 = min(
        image_width - 1,
        int(np.ceil(box[:, 0].max()))
    )

    y2 = min(
        image_height - 1,
        int(np.ceil(box[:, 1].max()))
    )

    if x2 <= x1 or y2 <= y1:
        return None

    support_score = min(
        len(supporting_crossbars) / 3.0,
        1.0
    )

    length_score = min(
        overlap / 300.0,
        1.0
    )

    separation_score = min(
        separation / 220.0,
        1.0
    )

    score = (
        0.4 * length_score
        + 0.3 * separation_score
        + 0.3 * support_score
    )

    return {
        "bbox": [
            x1, y1, x2, y2
        ],
        "polygon": (
            np.round(box)
            .astype(int)
            .tolist()
        ),
        "side_separation_px": round(
            separation, 2
        ),
        "overlap_length_px": round(
            overlap, 2
        ),
        "aspect_ratio": round(
            aspect, 3
        ),
        "crossbar_support_count": len(
            supporting_crossbars
        ),
        "geometric_score": round(
            score, 4
        ),
        "classification": (
            "POTENTIAL_ARCHITECTURAL_OPENING"
        ),
        "verification": "UNVERIFIED",
    }


def bbox_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])

    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area_a = max(
        0,
        (a[2] - a[0])
        * (a[3] - a[1])
    )

    area_b = max(
        0,
        (b[2] - b[0])
        * (b[3] - b[1])
    )

    union = (
        area_a + area_b - intersection
    )

    return (
        intersection / union
        if union > 0
        else 0.0
    )


def detect_candidates(image, valid_mask):
    segments = extract_segments(
        image,
        valid_mask
    )

    directions = (
        estimate_dominant_directions(
            segments
        )
    )

    height, width = image.shape[:2]

    candidates = []

    for direction in directions:
        sides = [
            segment
            for segment in segments
            if (
                segment["length"]
                >= MIN_SIDE_LENGTH
                and angular_difference(
                    segment["angle"],
                    direction
                )
                <= math.radians(
                    PARALLEL_TOLERANCE_DEG
                )
            )
        ]

        perpendicular = (
            direction + math.pi / 2
        ) % math.pi

        crossbars = [
            segment
            for segment in segments
            if angular_difference(
                segment["angle"],
                perpendicular
            )
            <= math.radians(
                CROSSBAR_TOLERANCE_DEG
            )
        ]

        # Avoid quadratic explosion on line-rich scenes.
        sides.sort(
            key=lambda item: item["length"],
            reverse=True
        )

        sides = sides[:100]

        for i, side_a in enumerate(sides):
            for side_b in sides[i + 1:]:
                candidate = candidate_from_pair(
                    side_a,
                    side_b,
                    crossbars,
                    direction,
                    width,
                    height,
                )

                if candidate is not None:
                    candidates.append(candidate)

    candidates.sort(
        key=lambda item: item[
            "geometric_score"
        ],
        reverse=True
    )

    filtered = []

    for candidate in candidates:
        if any(
            bbox_iou(
                candidate["bbox"],
                previous["bbox"]
            ) > 0.35
            for previous in filtered
        ):
            continue

        filtered.append(candidate)

        if len(filtered) >= MAX_CANDIDATES_PER_FRAME:
            break

    diagnostics = {
        "detected_line_segments": len(segments),
        "dominant_directions_deg": [
            round(math.degrees(angle), 2)
            for angle in directions
        ],
        "raw_candidate_count": len(candidates),
    }

    return filtered, diagnostics


def annotate(
    image,
    candidates,
    frame_index,
    diagnostics,
):
    visual = image.copy()

    for number, candidate in enumerate(
        candidates,
        1
    ):
        polygon = np.asarray(
            candidate["polygon"],
            dtype=np.int32
        )

        cv2.polylines(
            visual,
            [polygon],
            True,
            (0, 190, 255),
            2,
            cv2.LINE_AA
        )

        x1, y1, _, _ = candidate["bbox"]

        cv2.putText(
            visual,
            (
                f"H{number} "
                f"{candidate['geometric_score']:.2f}"
            ),
            (x1, max(80, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 190, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        visual,
        (0, 0),
        (visual.shape[1], 68),
        (20, 20, 20),
        -1,
    )

    cv2.putText(
        visual,
        (
            f"Frame {frame_index} | "
            f"Opening hypotheses: {len(candidates)}"
        ),
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        visual,
        (
            "FISHEYE RECTIFIED | "
            "UNVERIFIED HYPOTHESES"
        ),
        (15, 53),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 190, 255),
        1,
        cv2.LINE_AA,
    )

    return visual


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    camera = load_camera()

    report = load_json(
        INSPECTION_REPORT
    )

    selected = report.get(
        "selected_review_frames",
        []
    )

    if not selected:
        raise RuntimeError(
            "No selected frames found in inspection report"
        )

    observations = []

    for relative in selected:
        index = frame_number(
            Path(relative)
        )

        image = cv2.imread(
            str(frame_path(index))
        )

        if image is None:
            print(
                f"Frame {index}: unreadable, abstaining"
            )
            continue

        rectified, valid_mask = rectify_image(
            image,
            camera
        )

        rectified, valid_mask = resize_image(
            rectified,
            valid_mask
        )

        candidates, diagnostics = detect_candidates(
            rectified,
            valid_mask
        )

        visual = annotate(
            rectified,
            candidates,
            index,
            diagnostics,
        )

        destination = (
            OUTPUT
            / f"frame_{index:05d}.jpg"
        )

        if not cv2.imwrite(
            str(destination),
            visual
        ):
            raise RuntimeError(
                f"Could not write {destination}"
            )

        observations.append({
            "frame_index": index,
            "candidate_count": len(candidates),
            "candidates": candidates,
            "diagnostics": diagnostics,
            "evidence_image": (
                destination.relative_to(
                    ROOT
                ).as_posix()
            ),
            "status": (
                "UNVERIFIED_HYPOTHESES"
                if candidates
                else "NO_HYPOTHESIS_GENERATED"
            ),
        })

        print(
            f"Frame {index:03d}: "
            f"{len(candidates)} candidates | "
            f"{diagnostics['detected_line_segments']} "
            f"line segments | "
            f"directions="
            f"{diagnostics['dominant_directions_deg']}"
        )

    result = {
        "schema_version": "2.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "agent": "SiteWitness",
        "observation_type": (
            "POTENTIAL_ARCHITECTURAL_OPENING"
        ),
        "method": (
            "Calibrated fisheye rectification, "
            "LSD line detection, orientation-invariant "
            "parallel-boundary hypotheses"
        ),
        "observations": observations,
        "cad_registration_verified": False,
        "construction_compliance_verified": False,
        "architectural_identity_verified": False,
        "limitations": [
            "Candidates may represent recesses, rails, frames, or other structures.",
            "The detector does not infer physical opening identity.",
            "No multi-frame candidate correspondence has been performed.",
            "Rectification may crop peripheral scene content.",
            "Geometric scores are heuristics, not calibrated probabilities.",
            "No CAD alignment or metric dimension verification is claimed.",
        ],
    }

    destination = OUTPUT / "observations.json"

    destination.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\nSITEWITNESS — OPENING OBSERVATION V2")
    print("=" * 55)

    print(
        "Frames inspected:",
        len(observations)
    )

    print(
        "Total hypotheses:",
        sum(
            item["candidate_count"]
            for item in observations
        )
    )

    print("Architectural identity: UNVERIFIED")
    print("Construction compliance: UNVERIFIED")
    print("Report:", destination)


if __name__ == "__main__":
    main()
