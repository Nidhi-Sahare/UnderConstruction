
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import ezdxf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

DATA_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

MANIFEST = (
    DATA_ROOT
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frame_manifest.csv"
)

FRAMES_DIR = MANIFEST.parent / "frames"

CANDIDATES_PATH = (
    ROOT
    / "outputs"
    / "automatic_registration_v2"
    / "registration_candidates.json"
)

DXF_PATH = (
    ROOT
    / "data"
    / "floorplans"
    / "floor_2.dxf"
)

QUALITY_PATH = (
    ROOT
    / "outputs"
    / "visual_quality"
    / "walkthrough_quality.json"
)

OUTPUT_DIR = ROOT / "outputs" / "registration_image_geometry"
OUTPUT_PATH = OUTPUT_DIR / "image_geometry_report.json"

# Hilti cam0 fisheye calibration, original 1472x1440 images.
FX = 465.3015482593691
FY = 465.32303798346413
CX = 730.0455886686005
CY = 720.1427007671206

D = np.array([
    0.025800718903376804,
    -0.010909240777406872,
    -0.0016899537986031076,
    0.00014766801645260894,
], dtype=np.float64)

IMAGE_SIZE = (1472, 1440)

WALL_LAYERS = {
    "015 Innenwände",
    "010 Aussenwände",
    "016 Trennwände",
}

MAX_FRAMES = 60
MIN_LINE_LENGTH = 25
ANGLE_BINS = 36
HOLDOUT_MODULUS = 5


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_frames():
    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    quality = {}

    if QUALITY_PATH.exists():
        quality = {
            int(item["frame_index"]): item.get(
                "quality_class"
            )
            for item in load_json(QUALITY_PATH).get(
                "frames", []
            )
        }

    frames = []

    for index, row in enumerate(rows):
        if str(row.get("pose_matched", "")).lower() not in {
            "true", "1", "yes"
        }:
            continue

        if quality.get(index) not in (
            None,
            "USABLE_CANDIDATE",
        ):
            continue

        filename = Path(
            str(row.get("filename", "")).replace("\\", "/")
        ).name

        if not filename:
            filename = f"cam0_{index:05d}.jpg"

        path = FRAMES_DIR / filename

        if not path.is_file():
            continue

        frames.append({
            "frame_index": index,
            "path": path,
        })

    if len(frames) > MAX_FRAMES:
        selected = np.linspace(
            0,
            len(frames) - 1,
            MAX_FRAMES,
            dtype=int,
        )

        frames = [
            frames[int(i)]
            for i in np.unique(selected)
        ]

    return frames


def load_candidates():
    payload = load_json(CANDIDATES_PATH)

    items = (
        payload["candidates"]
        if isinstance(payload, dict)
        else payload
    )

    return [
        {
            "rank": int(
                item.get("rank", index + 1)
            ),
            "angle_deg": float(item["angle_deg"]),
            "tx": float(item["tx"]),
            "ty": float(item["ty"]),
        }
        for index, item in enumerate(items)
    ]


def extract_image_lines(path):
    image = cv2.imread(str(path))

    if image is None:
        return None

    height, width = image.shape[:2]

    if (width, height) != IMAGE_SIZE:
        return None

    K = np.array([
        [FX, 0.0, CX],
        [0.0, FY, CY],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Rectify to a limited central field of view.
    # This avoids using the extreme fisheye periphery.
    output_size = (720, 720)

    output_K = np.array([
        [460.0, 0.0, 360.0],
        [0.0, 460.0, 360.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        output_K,
        output_size,
        cv2.CV_32FC1,
    )

    rectified = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
    )

    gray = cv2.cvtColor(
        rectified,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    edges = cv2.Canny(
        gray,
        60,
        140,
    )

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=55,
        minLineLength=MIN_LINE_LENGTH,
        maxLineGap=15,
    )

    if lines is None:
        return {
            "angles": [],
            "weights": [],
            "line_count": 0,
        }

    angles = []
    weights = []

    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(float, line)

        dx = x2 - x1
        dy = y2 - y1

        length = math.hypot(dx, dy)

        if length < MIN_LINE_LENGTH:
            continue

        angle = math.atan2(dy, dx) % math.pi

        angles.append(angle)
        weights.append(length)

    return {
        "angles": angles,
        "weights": weights,
        "line_count": len(angles),
    }


def load_wall_orientations():
    document = ezdxf.readfile(DXF_PATH)

    angles = []
    weights = []

    for entity in document.modelspace():
        if entity.dxf.layer not in WALL_LAYERS:
            continue

        if entity.dxftype() == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end

            points = [
                (float(a.x), float(a.y)),
                (float(b.x), float(b.y)),
            ]

        elif entity.dxftype() == "LWPOLYLINE":
            points = [
                (float(p[0]), float(p[1]))
                for p in entity.get_points("xy")
            ]

        else:
            continue

        segments = list(zip(points, points[1:]))

        if (
            entity.dxftype() == "LWPOLYLINE"
            and entity.closed
        ):
            segments.append((points[-1], points[0]))

        for a, b in segments:
            dx = b[0] - a[0]
            dy = b[1] - a[1]

            length = math.hypot(dx, dy)

            if length < 0.1:
                continue

            angles.append(
                math.atan2(dy, dx) % math.pi
            )
            weights.append(length)

    return (
        np.asarray(angles, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
    )


def orientation_histogram(angles, weights):
    if not len(angles):
        return np.zeros(
            ANGLE_BINS,
            dtype=np.float64,
        )

    histogram, _ = np.histogram(
        np.asarray(angles) % math.pi,
        bins=ANGLE_BINS,
        range=(0, math.pi),
        weights=np.asarray(weights),
    )

    histogram = histogram.astype(np.float64)

    total = histogram.sum()

    if total > 0:
        histogram /= total

    return histogram


def circular_similarity(a, b):
    if not a.any() or not b.any():
        return None

    return float(
        np.dot(a, b)
        / (
            np.linalg.norm(a)
            * np.linalg.norm(b)
            + 1e-12
        )
    )


def score_candidate(
    candidate,
    frame_features,
    cad_angles,
    cad_weights,
    rotation_offset_deg=0.0,
):
    # IMPORTANT:
    # This is a planar orientation hypothesis,
    # NOT a calibrated 3D camera projection.
    #
    # A camera-to-body extrinsic is unavailable.
    # Results cannot verify registration.
    rotation = math.radians(
        candidate["angle_deg"]
        + rotation_offset_deg
    )

    predicted_angles = (
        cad_angles - rotation
    ) % math.pi

    cad_histogram = orientation_histogram(
        predicted_angles,
        cad_weights,
    )

    frame_scores = []

    for feature in frame_features:
        image_histogram = orientation_histogram(
            feature["angles"],
            feature["weights"],
        )

        similarity = circular_similarity(
            image_histogram,
            cad_histogram,
        )

        if similarity is None:
            continue

        frame_scores.append({
            "frame_index": feature["frame_index"],
            "similarity": round(similarity, 6),
            "partition": (
                "holdout"
                if feature["frame_index"]
                % HOLDOUT_MODULUS == 0
                else "development"
            ),
        })

    development = [
        record["similarity"]
        for record in frame_scores
        if record["partition"] == "development"
    ]

    holdout = [
        record["similarity"]
        for record in frame_scores
        if record["partition"] == "holdout"
    ]

    return {
        "candidate_rank": candidate["rank"],
        "rotation_offset_deg": rotation_offset_deg,
        "development_mean_similarity": (
            round(float(np.mean(development)), 6)
            if development
            else None
        ),
        "holdout_mean_similarity": (
            round(float(np.mean(holdout)), 6)
            if holdout
            else None
        ),
        "frame_scores": frame_scores,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_frames()
    candidates = load_candidates()
    cad_angles, cad_weights = load_wall_orientations()

    print("Extracting image geometry...")

    features = []

    for frame in frames:
        result = extract_image_lines(
            frame["path"]
        )

        if result is None:
            continue

        features.append({
            "frame_index": frame["frame_index"],
            **result,
        })

        if len(features) % 10 == 0:
            print(
                f"Processed {len(features)} frames"
            )

    print("Comparing orientation hypotheses...")

    candidate_scores = [
        score_candidate(
            candidate,
            features,
            cad_angles,
            cad_weights,
        )
        for candidate in candidates
    ]

    negative_controls = [
        score_candidate(
            candidate,
            features,
            cad_angles,
            cad_weights,
            rotation_offset_deg=45.0,
        )
        for candidate in candidates
    ]

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "method": "IMAGE_LINE_ORIENTATION_EXPLORATION",
        "registration_verified": False,
        "camera_to_imu_extrinsics_available": False,
        "calibrated_3d_projection_performed": False,
        "frames_analyzed": len(features),
        "cad_line_count": len(cad_angles),
        "candidate_scores": candidate_scores,
        "negative_controls": negative_controls,
        "limitations": [
            "No camera-to-IMU extrinsic transform.",
            "Image line angles are not equivalent to CAD plan angles.",
            "Camera pitch, roll, and heading are not modeled.",
            "Occlusion and depth are not modeled.",
            "The score is exploratory and cannot verify registration.",
            "A 45-degree rotation control is not guaranteed to be a hard negative.",
            "No construction progress is inferred.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== IMAGE GEOMETRY EXPLORATION ===")
    print("Frames analyzed:", len(features))
    print("CAD line segments:", len(cad_angles))

    for result in candidate_scores:
        print(
            f"Candidate #{result['candidate_rank']}: "
            f"development="
            f"{result['development_mean_similarity']}, "
            f"holdout="
            f"{result['holdout_mean_similarity']}"
        )

    print("Registration verified: NO")
    print("3D camera projection performed: NO")
    print(
        "Report:",
        OUTPUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
