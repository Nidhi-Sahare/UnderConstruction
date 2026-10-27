
"""
Under-construction — Step 97

Diagnose frozen 3D line projection failures.

Inputs:
  outputs/line_ambiguity/resolved_lines.json
  outputs/line_validation/per_line_results.json
  _archive/.../cam0/frame_manifest.csv
  _archive/.../cam0/frames/

Outputs:
  outputs/line_projection_diagnostic/diagnostic_report.json
  outputs/line_projection_diagnostic/per_line_diagnostics.json
  outputs/line_projection_diagnostic/overlays/*.jpg

No CAD, AI models, manual annotation, fitting,
or changes to reconstruction.

Run:
  python tools/diagnose_line_projection_failures.py
"""

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

LINES_PATH = (
    ROOT / "outputs" / "line_ambiguity"
    / "resolved_lines.json"
)

VALIDATION_PATH = (
    ROOT / "outputs" / "line_validation"
    / "per_line_results.json"
)

CAMERA_ROOT = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data" / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST_PATH = CAMERA_ROOT / "frame_manifest.csv"
FRAME_DIR = CAMERA_ROOT / "frames"

OUTPUT = (
    ROOT / "outputs"
    / "line_projection_diagnostic"
)

OVERLAY_DIR = OUTPUT / "overlays"

K = np.array([
    [465.3015482593691, 0.0, 730.0455886686005],
    [0.0, 465.32303798346413, 720.1427007671206],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

D = np.array([
    0.025800718903376804,
    -0.010909240777406872,
    -0.0016899537986031076,
    0.00014766801645260894,
], dtype=np.float64).reshape(4, 1)

SAMPLES = 80
EDGE_TOLERANCE_PX = 3.0

# Diagnostic offset sweep only.
# Never use the best offset to correct geometry.
OFFSETS_PX = np.arange(
    -64, 65, 4, dtype=np.float64
)

# Exactly the Step 96 control offsets.
CONTROL_OFFSETS_PX = (-48, -24, 24, 48)

MIN_CAMERA_DEPTH_M = 0.20
IMAGE_MARGIN_PX = 8

# Bounded image output.
MAX_LINES_PER_CATEGORY = 5
MAX_FRAMES_PER_LINE = 4

# Fixed diagnostic thresholds.
NEAR_ZERO_SUPPORT = 0.05
PARTIAL_SUPPORT_LOWER = 0.05
PARTIAL_SUPPORT_UPPER = 0.35
OFFSET_IMPROVEMENT_THRESHOLD = 0.20
MIN_MEANINGFUL_OFFSET_PX = 8.0

EPS = 1e-9


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input: {path}"
        )
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def write_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def load_poses():
    poses = {}

    with MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "filename", "tx", "ty", "tz",
            "qx", "qy", "qz", "qw",
        }

        if not required.issubset(
            reader.fieldnames or []
        ):
            raise RuntimeError(
                "Unexpected pose manifest schema."
            )

        for row in reader:
            try:
                name = Path(row["filename"]).name

                position = np.array([
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ])

                quaternion = np.array([
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ])

                if not (
                    np.all(np.isfinite(position))
                    and np.all(np.isfinite(quaternion))
                ):
                    continue

                rotation = Rotation.from_quat(
                    quaternion
                ).as_matrix()

                poses[name] = {
                    "position": position,
                    "rotation": rotation,
                }

            except (ValueError, KeyError):
                continue

    return poses


def load_lines():
    data = read_json(LINES_PATH)
    result = {}

    for item in data.get("selected_lines", []):
        endpoints = np.asarray(
            item["endpoints_map"],
            dtype=np.float64,
        )

        if (
            endpoints.shape != (2, 3)
            or not np.all(np.isfinite(endpoints))
        ):
            continue

        candidate_id = int(
            item["candidate_id"]
        )

        result[candidate_id] = {
            "endpoints": endpoints,
            "group_id": item["group_id"],
            "orientation": item.get(
                "orientation_class", "unknown"
            ),
            "length_m": float(
                np.linalg.norm(
                    endpoints[1] - endpoints[0]
                )
            ),
        }

    return result


def project_line(endpoints, pose, width, height):
    t = np.linspace(0, 1, SAMPLES)

    points_map = (
        endpoints[0][None, :]
        + t[:, None]
        * (endpoints[1] - endpoints[0])[None, :]
    )

    points_camera = (
        points_map - pose["position"]
    ) @ pose["rotation"]

    if np.any(
        points_camera[:, 2] <= MIN_CAMERA_DEPTH_M
    ):
        return None

    pixels, _ = cv2.fisheye.projectPoints(
        points_camera.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        D,
    )

    pixels = pixels.reshape(-1, 2)

    inside = (
        (pixels[:, 0] >= IMAGE_MARGIN_PX)
        & (pixels[:, 0] < width - IMAGE_MARGIN_PX)
        & (pixels[:, 1] >= IMAGE_MARGIN_PX)
        & (pixels[:, 1] < height - IMAGE_MARGIN_PX)
    )

    visible = pixels[inside]

    if len(visible) < 3:
        return None

    return visible


def calculate_normals(points):
    tangent = np.gradient(
        points, axis=0
    )

    length = np.linalg.norm(
        tangent, axis=1
    )

    normals = np.zeros_like(points)

    valid = length > EPS

    normals[valid, 0] = (
        -tangent[valid, 1] / length[valid]
    )

    normals[valid, 1] = (
        tangent[valid, 0] / length[valid]
    )

    return normals


def make_distance_map(gray):
    blurred = cv2.GaussianBlur(
        gray, (5, 5), 1.0
    )

    edges = cv2.Canny(
        blurred, 60, 150,
        L2gradient=True,
    )

    inverse = (
        edges == 0
    ).astype(np.uint8)

    distance = cv2.distanceTransform(
        inverse,
        cv2.DIST_L2,
        3,
    )

    return edges, distance


def sample_distance(distance, points):
    height, width = distance.shape

    x = points[:, 0]
    y = points[:, 1]

    inside = (
        (x >= 0)
        & (x <= width - 2)
        & (y >= 0)
        & (y <= height - 2)
    )

    result = np.full(
        len(points),
        np.inf,
        dtype=np.float64,
    )

    if not np.any(inside):
        return result

    xx = x[inside]
    yy = y[inside]

    x0 = np.floor(xx).astype(int)
    y0 = np.floor(yy).astype(int)

    dx = xx - x0
    dy = yy - y0

    result[inside] = (
        (1 - dx) * (1 - dy) * distance[y0, x0]
        + dx * (1 - dy) * distance[y0, x0 + 1]
        + (1 - dx) * dy * distance[y0 + 1, x0]
        + dx * dy * distance[y0 + 1, x0 + 1]
    )

    return result


def support(distance, points):
    distances = sample_distance(
        distance, points
    )

    return float(
        np.mean(
            distances <= EDGE_TOLERANCE_PX
        )
    )


def offset_profile(distance, points):
    normals = calculate_normals(points)

    scores = []

    for offset in OFFSETS_PX:
        shifted = (
            points + offset * normals
        )

        scores.append(
            support(distance, shifted)
        )

    scores = np.asarray(
        scores, dtype=np.float64
    )

    zero_index = int(
        np.argmin(np.abs(OFFSETS_PX))
    )

    real_score = float(
        scores[zero_index]
    )

    best_score = float(
        np.max(scores)
    )

    # If multiple offsets tie, prefer the
    # smallest absolute displacement.
    best_indices = np.flatnonzero(
        np.isclose(
            scores, best_score,
            atol=1e-10,
        )
    )

    best_index = min(
        best_indices,
        key=lambda i: (
            abs(OFFSETS_PX[i]),
            OFFSETS_PX[i],
        ),
    )

    best_offset = float(
        OFFSETS_PX[best_index]
    )

    improvement = (
        best_score - real_score
    )

    return {
        "real_support": real_score,
        "best_offset_px": best_offset,
        "best_offset_support": best_score,
        "offset_improvement": improvement,
        "offsets_px": OFFSETS_PX.tolist(),
        "support_profile": scores.tolist(),
        "possible_offset_error": bool(
            abs(best_offset)
            >= MIN_MEANINGFUL_OFFSET_PX
            and improvement
            >= OFFSET_IMPROVEMENT_THRESHOLD
        ),
    }


def frame_diagnostic(
    line,
    frame_name,
    pose,
    image_cache,
):
    if frame_name not in image_cache:
        path = FRAME_DIR / frame_name

        gray = cv2.imread(
            str(path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray is None:
            return None

        edges, distance = make_distance_map(
            gray
        )

        image_cache[frame_name] = {
            "gray": gray,
            "edges": edges,
            "distance": distance,
        }

    image = image_cache[frame_name]

    height, width = image["gray"].shape

    points = project_line(
        line["endpoints"],
        pose,
        width,
        height,
    )

    if points is None:
        return None

    profile = offset_profile(
        image["distance"],
        points,
    )

    normals = calculate_normals(points)

    controls = {}

    for offset in CONTROL_OFFSETS_PX:
        controls[str(offset)] = support(
            image["distance"],
            points + offset * normals,
        )

    real = profile["real_support"]

    if real <= NEAR_ZERO_SUPPORT:
        support_class = "NEAR_ZERO"

    elif real < PARTIAL_SUPPORT_UPPER:
        support_class = "PARTIAL"

    else:
        support_class = "SUBSTANTIAL"

    return {
        "frame": frame_name,
        "support_class": support_class,
        "control_supports": controls,
        **profile,
    }


def choose_overlay_frames(item):
    frames = item.get("frames", [])

    if not frames:
        return []

    # Fixed selection rule:
    # weakest real support, strongest real support,
    # strongest control advantage, weakest advantage.
    selectors = [
        lambda r: (
            r["real_edge_support"],
            r["frame"],
        ),
        lambda r: (
            -r["real_edge_support"],
            r["frame"],
        ),
        lambda r: (
            -r["control_advantage"],
            r["frame"],
        ),
        lambda r: (
            r["control_advantage"],
            r["frame"],
        ),
    ]

    chosen = []
    used = set()

    for key in selectors:
        for row in sorted(frames, key=key):
            name = row["frame"]

            if name not in used:
                chosen.append(name)
                used.add(name)
                break

        if len(chosen) >= MAX_FRAMES_PER_LINE:
            break

    return chosen


def draw_curve(image, points, color, thickness=2):
    height, width = image.shape[:2]

    finite = np.all(
        np.isfinite(points), axis=1
    )

    for i in range(len(points) - 1):
        if not (finite[i] and finite[i + 1]):
            continue

        a = tuple(
            np.rint(points[i]).astype(int)
        )

        b = tuple(
            np.rint(points[i + 1]).astype(int)
        )

        if (
            0 <= a[0] < width
            and 0 <= a[1] < height
            and 0 <= b[0] < width
            and 0 <= b[1] < height
        ):
            cv2.line(
                image, a, b,
                color, thickness,
                cv2.LINE_AA,
            )


def save_overlay(
    line,
    frame_name,
    pose,
    diagnostic,
    image_cache,
    candidate_id,
):
    image = image_cache[frame_name]

    gray = image["gray"]
    edges = image["edges"]

    height, width = gray.shape

    points = project_line(
        line["endpoints"],
        pose,
        width,
        height,
    )

    if points is None:
        return None

    normals = calculate_normals(points)

    original = cv2.cvtColor(
        gray, cv2.COLOR_GRAY2BGR
    )

    edge_view = cv2.cvtColor(
        edges, cv2.COLOR_GRAY2BGR
    )

    # Original projection: green.
    draw_curve(
        original,
        points,
        (0, 255, 0),
        3,
    )

    # Controls: orange.
    for offset in CONTROL_OFFSETS_PX:
        shifted = (
            points + offset * normals
        )

        draw_curve(
            original,
            shifted,
            (0, 165, 255),
            1,
        )

    # Best diagnostic offset: magenta.
    # It is not an accepted correction.
    best_offset = diagnostic[
        "best_offset_px"
    ]

    draw_curve(
        original,
        points + best_offset * normals,
        (255, 0, 255),
        2,
    )

    draw_curve(
        edge_view,
        points,
        (0, 255, 0),
        3,
    )

    draw_curve(
        edge_view,
        points + best_offset * normals,
        (255, 0, 255),
        2,
    )

    combined = np.hstack([
        original, edge_view
    ])

    # Downscale large output to reduce storage.
    target_width = 1600

    if combined.shape[1] > target_width:
        scale = (
            target_width
            / combined.shape[1]
        )

        combined = cv2.resize(
            combined,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    header_height = 100

    canvas = np.zeros(
        (
            combined.shape[0] + header_height,
            combined.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    canvas[header_height:] = combined

    labels = [
        (
            f"Candidate {candidate_id} | "
            f"{frame_name}"
        ),
        (
            "GREEN predicted | ORANGE controls | "
            "MAGENTA best diagnostic offset"
        ),
        (
            f"Real={diagnostic['real_support']:.3f} "
            f"Best={diagnostic['best_offset_support']:.3f} "
            f"Offset={best_offset:+.0f}px"
        ),
    ]

    for i, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (12, 25 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    filename = (
        f"candidate_{candidate_id:03d}_"
        f"{Path(frame_name).stem}.jpg"
    )

    path = OVERLAY_DIR / filename

    if not cv2.imwrite(
        str(path), canvas
    ):
        raise RuntimeError(
            f"Failed to write overlay: {path}"
        )

    return str(
        path.relative_to(ROOT)
    )


def summarize_line(item, diagnostics):
    if not diagnostics:
        return {
            "candidate_id": item["candidate_id"],
            "diagnosed_frames": 0,
            "status": "NO_DIAGNOSTIC_FRAMES",
        }

    real = np.array([
        d["real_support"]
        for d in diagnostics
    ])

    best = np.array([
        d["best_offset_support"]
        for d in diagnostics
    ])

    offsets = np.array([
        d["best_offset_px"]
        for d in diagnostics
    ])

    improvements = best - real

    classes = Counter(
        d["support_class"]
        for d in diagnostics
    )

    meaningful = [
        d
        for d in diagnostics
        if d["possible_offset_error"]
    ]

    offset_signs = [
        np.sign(d["best_offset_px"])
        for d in meaningful
        if abs(d["best_offset_px"]) > EPS
    ]

    sign_consistency = (
        max(
            offset_signs.count(-1),
            offset_signs.count(1),
        ) / len(offset_signs)
        if offset_signs
        else None
    )

    return {
        "candidate_id": item["candidate_id"],
        "group_id": item["group_id"],
        "orientation_class": (
            item["orientation_class"]
        ),
        "length_m": item["length_m"],
        "step96_status": (
            item["validation_status"]
        ),
        "diagnosed_frames": len(
            diagnostics
        ),
        "support_classes": dict(classes),
        "median_real_support": float(
            np.median(real)
        ),
        "median_best_offset_support": float(
            np.median(best)
        ),
        "median_offset_improvement": float(
            np.median(improvements)
        ),
        "median_best_offset_px": float(
            np.median(offsets)
        ),
        "meaningful_offset_frame_count": len(
            meaningful
        ),
        "meaningful_offset_sign_consistency": (
            sign_consistency
        ),
        "possible_systematic_offset": bool(
            len(meaningful) >= 3
            and sign_consistency is not None
            and sign_consistency >= 0.8
        ),
    }


def select_lines(validation_items):
    passing = sorted(
        [
            x for x in validation_items
            if x["validation_status"]
            == "IMAGE_SUPPORT_PASSES_CONTROLS"
        ],
        key=lambda x: x["candidate_id"],
    )

    failing = sorted(
        [
            x for x in validation_items
            if x["validation_status"]
            == "IMAGE_SUPPORT_NOT_ESTABLISHED"
        ],
        key=lambda x: (
            -x["eligible_frame_count"],
            x["candidate_id"],
        ),
    )

    return (
        passing[:MAX_LINES_PER_CATEGORY]
        + failing[:MAX_LINES_PER_CATEGORY]
    )


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    poses = load_poses()
    lines = load_lines()

    validation_data = read_json(
        VALIDATION_PATH
    )

    validation_items = (
        validation_data.get("results", [])
    )

    if not validation_items:
        raise RuntimeError(
            "No Step 96 per-line results."
        )

    image_cache = {}
    all_results = []
    summaries = []
    overlay_files = []

    # Numerical diagnostics cover all 42 lines.
    # Visual overlays cover a bounded sample.
    overlay_candidates = {
        x["candidate_id"]
        for x in select_lines(
            validation_items
        )
    }

    for item in validation_items:
        candidate_id = int(
            item["candidate_id"]
        )

        line = lines.get(candidate_id)

        if line is None:
            raise RuntimeError(
                "Step 96 candidate missing from "
                f"Step 95: {candidate_id}"
            )

        diagnostics = []

        for frame_result in item.get(
            "frames", []
        ):
            frame_name = frame_result["frame"]

            pose = poses.get(frame_name)

            if pose is None:
                continue

            result = frame_diagnostic(
                line,
                frame_name,
                pose,
                image_cache,
            )

            if result is None:
                continue

            # Include Step 96 outcome for comparison.
            result["step96_real_support"] = (
                frame_result[
                    "real_edge_support"
                ]
            )

            result["step96_frame_win"] = (
                frame_result["frame_win"]
            )

            diagnostics.append(result)

        summary = summarize_line(
            item, diagnostics
        )

        summaries.append(summary)

        all_results.append({
            "candidate_id": candidate_id,
            "summary": summary,
            "frames": diagnostics,
        })

        if candidate_id in overlay_candidates:
            chosen_frames = choose_overlay_frames(
                item
            )

            lookup = {
                d["frame"]: d
                for d in diagnostics
            }

            for frame_name in chosen_frames:
                if frame_name not in lookup:
                    continue

                overlay = save_overlay(
                    line,
                    frame_name,
                    poses[frame_name],
                    lookup[frame_name],
                    image_cache,
                    candidate_id,
                )

                if overlay:
                    overlay_files.append(overlay)

        print(
            f"Candidate {candidate_id}: "
            f"{len(diagnostics)} frames, "
            f"median real support="
            f"{summary.get('median_real_support', 0):.3f}"
        )

    frame_rows = [
        d
        for result in all_results
        for d in result["frames"]
    ]

    class_counts = Counter(
        d["support_class"]
        for d in frame_rows
    )

    systematic = [
        s["candidate_id"]
        for s in summaries
        if s.get(
            "possible_systematic_offset",
            False
        )
    ]

    # Consistency check against Step 96.
    differences = [
        abs(
            d["real_support"]
            - d["step96_real_support"]
        )
        for d in frame_rows
    ]

    maximum_difference = (
        max(differences)
        if differences
        else None
    )

    per_line_path = (
        OUTPUT / "per_line_diagnostics.json"
    )

    write_json(
        per_line_path,
        {
            "method": (
                "frozen_projection_offset_diagnostic"
            ),
            "results": all_results,
        },
    )

    report = {
        "method": (
            "frozen_line_projection_failure_diagnostic"
        ),
        "input_lines": len(
            validation_items
        ),
        "lines_diagnosed": sum(
            s["diagnosed_frames"] > 0
            for s in summaries
        ),
        "frame_level_diagnostics": len(
            frame_rows
        ),
        "support_class_counts": dict(
            class_counts
        ),
        "possible_systematic_offset_candidate_ids": (
            systematic
        ),
        "possible_systematic_offset_count": len(
            systematic
        ),
        "maximum_step96_support_discrepancy": (
            maximum_difference
        ),
        "overlay_count": len(
            overlay_files
        ),
        "overlay_files": overlay_files,
        "offset_sweep_px": (
            OFFSETS_PX.tolist()
        ),
        "diagnostic_thresholds": {
            "near_zero_support": (
                NEAR_ZERO_SUPPORT
            ),
            "partial_support_upper": (
                PARTIAL_SUPPORT_UPPER
            ),
            "offset_improvement": (
                OFFSET_IMPROVEMENT_THRESHOLD
            ),
            "minimum_meaningful_offset_px": (
                MIN_MEANINGFUL_OFFSET_PX
            ),
        },
        "step96_pass_count_unchanged": sum(
            x["validation_status"]
            == "IMAGE_SUPPORT_PASSES_CONTROLS"
            for x in validation_items
        ),
        "physical_lines_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Best offsets are diagnostic and must not be used to correct geometry.",
            "A high offset score can arise from a different nearby physical edge.",
            "A constant image-plane offset cannot represent all pose or calibration errors.",
            "Occlusion is not explicitly estimated.",
            "Frame observations are temporally correlated.",
            "A systematic offset pattern does not identify its physical cause.",
            "No CAD geometry or structural classification is used.",
            "No validation threshold is relaxed.",
        ],
        "decision": (
            "PROJECTION_FAILURE_DIAGNOSTICS_AVAILABLE"
        ),
    }

    report_path = (
        OUTPUT / "diagnostic_report.json"
    )

    write_json(
        report_path, report
    )

    print(
        "\n=== STEP 97 RESULTS ==="
    )
    print(
        "Lines diagnosed:",
        report["lines_diagnosed"]
    )
    print(
        "Frame diagnostics:",
        len(frame_rows)
    )
    print(
        "Support classes:",
        dict(class_counts)
    )
    print(
        "Possible systematic offsets:",
        len(systematic)
    )
    print(
        "Overlays saved:",
        len(overlay_files)
    )
    print(
        "Step 96 passing lines unchanged:",
        report["step96_pass_count_unchanged"]
    )
    print(
        "Report:",
        report_path
    )
    print(
        "Per-line diagnostics:",
        per_line_path
    )
    print(
        "Overlay directory:",
        OVERLAY_DIR
    )


if __name__ == "__main__":
    main()
