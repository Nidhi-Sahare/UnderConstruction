
"""
Under-construction — Step 92
CAD-independent line evidence from real walkthrough frames.

Uses:
    - OpenCV Line Segment Detector (LSD)
    - Camera calibration and fisheye undistortion
    - Camera pose rotations
    - Length-weighted 3D interpretation-plane normals
    - Cross-frame direction repeatability diagnostics

Does NOT:
    - Label lines as walls
    - Claim 3D line triangulation
    - Register the CAD
    - Infer construction progress

Output:
    outputs/line_evidence/line_report.json
    outputs/line_evidence/line_observations.json

Run:
    python tools/extract_walkthrough_line_evidence.py
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

FRAME_DIR = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
)

MANIFEST = FRAME_DIR / "frame_manifest.csv"
IMAGE_DIR = FRAME_DIR / "frames"

OUTPUT_DIR = ROOT / "outputs" / "line_evidence"

K = np.array(
    [
        [465.3015482593691, 0, 730.0455886686005],
        [0, 465.32303798346413, 720.1427007671206],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

D = np.array(
    [
        0.025800718903376804,
        -0.010909240777406872,
        -0.0016899537986031076,
        0.00014766801645260894,
    ],
    dtype=np.float64,
)

MAX_FRAMES = 90
IMAGE_SCALE = 0.5

MIN_LINE_LENGTH_PX = 45.0
MAX_LINES_PER_FRAME = 120

# Direction repeatability is a diagnostic, not
# a structural verification criterion.
NORMAL_BIN_DEG = 10.0
MIN_DISTINCT_FRAMES_PER_BIN = 4

# Exclude lines close to image boundaries.
BORDER_MARGIN_PX = 12.0

# Numerical degeneracy threshold.
MIN_RAY_CROSS_NORM = 1e-5


def load_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing manifest: {MANIFEST}"
        )

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    valid = []

    for row in rows:
        try:
            name = row["filename"]

            position = np.array(
                [
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ],
                dtype=np.float64,
            )

            quaternion = np.array(
                [
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(position)):
                continue

            if not np.all(np.isfinite(quaternion)):
                continue

            if np.linalg.norm(quaternion) < 1e-8:
                continue

            # T_map_cam0 convention.
            R_map_cam = Rotation.from_quat(
                quaternion
            ).as_matrix()

            path = IMAGE_DIR / name

            if not path.exists():
                continue

            valid.append(
                {
                    "filename": name,
                    "path": path,
                    "position": position,
                    "rotation": R_map_cam,
                }
            )

        except (
            KeyError,
            ValueError,
            TypeError,
        ):
            continue

    if not valid:
        raise RuntimeError(
            "No usable frames with poses."
        )

    # Evenly sample the complete sequence rather
    # than concentrating on its beginning.
    if len(valid) > MAX_FRAMES:
        indices = np.linspace(
            0,
            len(valid) - 1,
            MAX_FRAMES,
            dtype=int,
        )

        valid = [valid[i] for i in indices]

    return valid


def undistort_rays(pixels):
    """
    Convert fisheye pixels into normalized
    camera-coordinate unit rays.
    """

    pixels = np.asarray(
        pixels,
        dtype=np.float64,
    ).reshape(-1, 1, 2)

    normalized = cv2.fisheye.undistortPoints(
        pixels,
        K,
        D,
    ).reshape(-1, 2)

    rays = np.column_stack(
        [
            normalized,
            np.ones(len(normalized)),
        ]
    )

    magnitudes = np.linalg.norm(
        rays,
        axis=1,
        keepdims=True,
    )

    rays /= np.maximum(
        magnitudes,
        1e-12,
    )

    return rays


def detect_lines(image, detector):
    height, width = image.shape[:2]

    resized = cv2.resize(
        image,
        None,
        fx=IMAGE_SCALE,
        fy=IMAGE_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    # CLAHE improves local contrast without
    # introducing a learned model.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    detected = detector.detect(gray)[0]

    if detected is None:
        return []

    segments = detected.reshape(-1, 4)

    segments /= IMAGE_SCALE

    results = []

    for x1, y1, x2, y2 in segments:
        endpoints = np.array(
            [[x1, y1], [x2, y2]],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(endpoints)):
            continue

        if (
            np.min(endpoints[:, 0]) < BORDER_MARGIN_PX
            or np.max(endpoints[:, 0])
            > width - BORDER_MARGIN_PX
            or np.min(endpoints[:, 1])
            < BORDER_MARGIN_PX
            or np.max(endpoints[:, 1])
            > height - BORDER_MARGIN_PX
        ):
            continue

        length = float(
            np.linalg.norm(
                endpoints[1] - endpoints[0]
            )
        )

        if length < MIN_LINE_LENGTH_PX:
            continue

        results.append(
            {
                "endpoints": endpoints,
                "length_px": length,
            }
        )

    results.sort(
        key=lambda item: item["length_px"],
        reverse=True,
    )

    return results[:MAX_LINES_PER_FRAME]


def interpretation_plane(line, rotation):
    """
    An image line backprojects to a plane through
    the camera centre.

    Its normal is expressed in trajectory-map
    coordinates.

    IMPORTANT:
    This normal is NOT a wall normal and is NOT
    a triangulated 3D line direction.
    """

    rays = undistort_rays(
        line["endpoints"]
    )

    camera_normal = np.cross(
        rays[0],
        rays[1],
    )

    magnitude = float(
        np.linalg.norm(camera_normal)
    )

    if magnitude < MIN_RAY_CROSS_NORM:
        return None

    camera_normal /= magnitude

    map_normal = (
        rotation @ camera_normal
    )

    map_normal /= max(
        np.linalg.norm(map_normal),
        1e-12,
    )

    # Sign-invariant normal representation.
    index = int(
        np.argmax(np.abs(map_normal))
    )

    if map_normal[index] < 0:
        map_normal *= -1

    return map_normal


def normal_bin(normal):
    """
    Bin unoriented interpretation-plane normals.

    This is only an exploratory repeatability
    diagnostic. Similar normals from different
    camera positions need not represent the
    same physical line.
    """

    normal = np.asarray(
        normal,
        dtype=np.float64,
    )

    azimuth = math.degrees(
        math.atan2(
            normal[1],
            normal[0],
        )
    ) % 360.0

    elevation = math.degrees(
        math.asin(
            float(np.clip(normal[2], -1, 1))
        )
    )

    az_bin = int(
        math.floor(azimuth / NORMAL_BIN_DEG)
    )

    el_bin = int(
        math.floor(
            (elevation + 90.0)
            / NORMAL_BIN_DEG
        )
    )

    return f"{az_bin}:{el_bin}"


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_manifest()

    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    observations = []
    per_frame = []

    print(
        "\n=== STEP 92: WALKTHROUGH LINE EVIDENCE ==="
    )

    for frame_index, frame in enumerate(frames):
        image = cv2.imread(
            str(frame["path"]),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        lines = detect_lines(
            image,
            detector,
        )

        accepted = 0

        for line_index, line in enumerate(lines):
            normal = interpretation_plane(
                line,
                frame["rotation"],
            )

            if normal is None:
                continue

            observation = {
                "observation_id": (
                    f"{frame_index:03d}_{line_index:03d}"
                ),
                "frame": frame["filename"],
                "frame_sample_index": frame_index,
                "camera_position_map": (
                    frame["position"].tolist()
                ),
                "endpoints_px": (
                    line["endpoints"].tolist()
                ),
                "length_px": line["length_px"],
                "interpretation_plane_normal_map": (
                    normal.tolist()
                ),
                "normal_bin": normal_bin(normal),
                "physical_line_verified": False,
                "structural_identity_verified": False,
            }

            observations.append(observation)
            accepted += 1

        per_frame.append(
            {
                "frame": frame["filename"],
                "line_count": accepted,
            }
        )

        print(
            f"[{frame_index + 1}/{len(frames)}] "
            f"{frame['filename']}: "
            f"{accepted} lines"
        )

    bins = {}

    for observation in observations:
        key = observation["normal_bin"]

        item = bins.setdefault(
            key,
            {
                "observation_count": 0,
                "total_line_length_px": 0.0,
                "frames": set(),
            },
        )

        item["observation_count"] += 1
        item["total_line_length_px"] += (
            observation["length_px"]
        )

        item["frames"].add(
            observation["frame"]
        )

    summaries = []

    for key, item in bins.items():
        summaries.append(
            {
                "normal_bin": key,
                "observation_count": (
                    item["observation_count"]
                ),
                "distinct_frames": len(
                    item["frames"]
                ),
                "total_line_length_px": (
                    item["total_line_length_px"]
                ),
                "repeatability_gate_passed": (
                    len(item["frames"])
                    >= MIN_DISTINCT_FRAMES_PER_BIN
                ),
            }
        )

    summaries.sort(
        key=lambda item: (
            item["distinct_frames"],
            item["total_line_length_px"],
        ),
        reverse=True,
    )

    line_counts = [
        item["line_count"]
        for item in per_frame
    ]

    repeatable_bins = [
        item
        for item in summaries
        if item["repeatability_gate_passed"]
    ]

    data = {
        "sequence": SEQUENCE,
        "method": (
            "opencv_lsd_fisheye_interpretation_planes"
        ),
        "observations": observations,
    }

    observations_path = (
        OUTPUT_DIR / "line_observations.json"
    )

    observations_path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    report = {
        "sequence": SEQUENCE,
        "method": (
            "cad_independent_line_evidence_inventory"
        ),
        "frames_sampled": len(frames),
        "frames_processed": len(per_frame),
        "total_line_observations": len(
            observations
        ),
        "median_lines_per_frame": (
            float(np.median(line_counts))
            if line_counts
            else 0.0
        ),
        "normal_bin_count": len(summaries),
        "repeatable_normal_bin_count": len(
            repeatable_bins
        ),
        "top_normal_bins": summaries[:25],
        "per_frame": per_frame,
        "line_inventory_available": bool(
            observations
        ),
        "multiview_line_correspondences_verified": False,
        "triangulated_3d_lines_verified": False,
        "independent_structural_landmarks_verified": False,
        "registration_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "max_frames": MAX_FRAMES,
            "image_scale": IMAGE_SCALE,
            "min_line_length_px": MIN_LINE_LENGTH_PX,
            "max_lines_per_frame": MAX_LINES_PER_FRAME,
            "normal_bin_deg": NORMAL_BIN_DEG,
            "min_distinct_frames_per_bin": (
                MIN_DISTINCT_FRAMES_PER_BIN
            ),
        },
        "limitations": [
            "LSD detects visual edges, not semantic walls.",
            "Repeated interpretation-plane normals do not prove physical correspondence.",
            "A camera-centred interpretation plane is not a reconstructed 3D wall.",
            "Frame observations are not independent when adjacent images overlap.",
            "Line observations have not been matched or triangulated across views.",
            "No CAD geometry influenced line extraction.",
            "No construction progress can be inferred from this inventory.",
        ],
        "decision": (
            "LINE_EVIDENCE_AVAILABLE_FOR_MULTIVIEW_MATCHING"
            if observations
            else "INSUFFICIENT_LINE_EVIDENCE"
        ),
    }

    report_path = OUTPUT_DIR / "line_report.json"

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")
    print("Frames processed:", len(per_frame))
    print("Line observations:", len(observations))
    print(
        "Median lines/frame:",
        report["median_lines_per_frame"],
    )
    print(
        "Repeatable normal bins:",
        len(repeatable_bins),
    )
    print("3D lines verified: NO")
    print("CAD registration verified: NO")
    print("Report:", report_path)
    print("Observations:", observations_path)


if __name__ == "__main__":
    main()
