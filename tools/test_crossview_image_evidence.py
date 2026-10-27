
"""
Under-construction — Step 103
Cross-view image evidence for frozen reconstructed lines.

Inputs:
  outputs/line_ambiguity/resolved_lines.json
  outputs/viewpoint_observability/per_candidate_observability.json
  Camera frame manifest and actual images

Outputs:
  outputs/crossview_image_evidence/
    evidence_report.json
    per_candidate_evidence.json

No AI, no geometry optimization, no CAD registration.
"""

import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

LINES_PATH = (
    ROOT / "outputs" / "line_ambiguity"
    / "resolved_lines.json"
)

OBS_PATH = (
    ROOT / "outputs" / "viewpoint_observability"
    / "per_candidate_observability.json"
)

CAM_ROOT = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "data" / "hilti_frames"
    / "floor_2_2025-10-28_run_2" / "cam0"
)

MANIFEST = CAM_ROOT / "frame_manifest.csv"
FRAMES = CAM_ROOT / "frames"

OUT = ROOT / "outputs" / "crossview_image_evidence"

CANDIDATES = (159, 2, 47, 161, 108)

K = np.array([
    [465.3015482593691, 0, 730.0455886686005],
    [0, 465.32303798346413, 720.1427007671206],
    [0, 0, 1],
], dtype=np.float64)

D = np.array([
    0.025800718903376804,
    -0.010909240777406872,
    -0.0016899537986031076,
    0.00014766801645260894,
], dtype=np.float64).reshape(4, 1)

SAMPLES = 51
MIN_DEPTH = 0.2
MIN_VISIBLE_FRACTION = 0.7
MIN_PROJECTED_LENGTH = 20.0

MIN_FRAME_SEPARATION = 10
MAX_FRAMES_PER_CANDIDATE = 8

MAX_DISTANCE_PX = 8.0
MAX_ANGLE_DEG = 15.0
MIN_SEGMENT_LENGTH_PX = 12.0

CONTROL_SHIFTS_M = (-0.08, -0.03, 0.03, 0.08)

MIN_SEPARATED_SUPPORT_VIEWS = 3

cv2.setNumThreads(1)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def frame_number(filename):
    try:
        return int(Path(filename).stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def load_lines():
    raw = load_json(LINES_PATH)
    result = {}

    for item in raw["selected_lines"]:
        cid = int(item["candidate_id"])
        endpoints = np.asarray(
            item["endpoints_map"],
            dtype=np.float64,
        )

        if endpoints.shape != (2, 3):
            continue

        if not np.isfinite(endpoints).all():
            continue

        result[cid] = {
            "group_id": item.get("group_id"),
            "points": endpoints,
        }

    return result


def load_observability():
    raw = load_json(OBS_PATH)

    return {
        int(row["candidate_id"]): row
        for row in raw["results"]
    }


def load_poses():
    poses = {}

    with MANIFEST.open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        for row in csv.DictReader(handle):
            try:
                name = Path(row["filename"]).name

                position = np.array([
                    float(row["tx"]),
                    float(row["ty"]),
                    float(row["tz"]),
                ])

                quat = [
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ]

                rotation = Rotation.from_quat(
                    quat
                ).as_matrix()

                if not (
                    np.isfinite(position).all()
                    and np.isfinite(rotation).all()
                ):
                    continue

                if (FRAMES / name).is_file():
                    poses[name] = {
                        "position": position,
                        "rotation": rotation,
                    }

            except (KeyError, ValueError):
                continue

    return poses


def unit(vector):
    length = np.linalg.norm(vector)

    if length < 1e-10:
        raise ValueError("Degenerate direction")

    return vector / length


def perpendicular_axes(points):
    direction = unit(points[1] - points[0])

    reference = min(
        np.eye(3),
        key=lambda axis: abs(axis @ direction),
    )

    u = unit(np.cross(direction, reference))
    v = unit(np.cross(direction, u))

    return u, v


def project_line(points, pose, shape):
    height, width = shape[:2]

    t = np.linspace(0, 1, SAMPLES)[:, None]

    samples = (
        points[0]
        + t * (points[1] - points[0])
    )

    camera = (
        samples - pose["position"]
    ) @ pose["rotation"]

    valid_depth = (
        np.isfinite(camera).all(axis=1)
        & (camera[:, 2] > MIN_DEPTH)
    )

    pixels = np.full(
        (SAMPLES, 2), np.nan, dtype=np.float64
    )

    indices = np.flatnonzero(valid_depth)

    if len(indices):
        projected, _ = cv2.fisheye.projectPoints(
            camera[indices].reshape(-1, 1, 3),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K,
            D,
        )

        pixels[indices] = projected.reshape(-1, 2)

    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 8)
        & (pixels[:, 0] < width - 8)
        & (pixels[:, 1] >= 8)
        & (pixels[:, 1] < height - 8)
    )

    if valid.mean() < MIN_VISIBLE_FRACTION:
        return None

    # Only use consecutive valid samples.
    pair_valid = valid[:-1] & valid[1:]

    if not pair_valid.any():
        return None

    differences = np.diff(pixels, axis=0)

    length = float(
        np.linalg.norm(
            differences[pair_valid], axis=1
        ).sum()
    )

    if length < MIN_PROJECTED_LENGTH:
        return None

    return pixels, valid, length


def detect_segments(gray, detector):
    result = detector.detect(gray)[0]

    if result is None:
        return np.empty((0, 2, 2), dtype=np.float64)

    segments = result.reshape(-1, 2, 2).astype(
        np.float64
    )

    lengths = np.linalg.norm(
        segments[:, 1] - segments[:, 0],
        axis=1,
    )

    return segments[
        lengths >= MIN_SEGMENT_LENGTH_PX
    ]


def point_segment_distance(point, a, b):
    ab = b - a
    denominator = float(ab @ ab)

    if denominator < 1e-12:
        return float(np.linalg.norm(point - a))

    t = float(
        np.clip(
            ((point - a) @ ab) / denominator,
            0,
            1,
        )
    )

    closest = a + t * ab

    return float(np.linalg.norm(point - closest))


def angle_difference(a, b):
    a = unit(a)
    b = unit(b)

    cosine = abs(float(a @ b))
    cosine = float(np.clip(cosine, 0, 1))

    return math.degrees(math.acos(cosine))


def evaluate_projection(projection, segments):
    if projection is None:
        return {
            "status": "PROJECTION_UNAVAILABLE"
        }

    pixels, valid, projected_length = projection

    # Compare against every globally detected segment.
    # This is NOT a physical correspondence assignment.
    matches = []
    sampled_support = []

    for index in range(SAMPLES - 1):
        if not (valid[index] and valid[index + 1]):
            continue

        a = pixels[index]
        b = pixels[index + 1]

        tangent = b - a

        if np.linalg.norm(tangent) < 1e-9:
            continue

        midpoint = (a + b) / 2

        best = None

        for segment_index, segment in enumerate(segments):
            direction = segment[1] - segment[0]

            try:
                angle = angle_difference(
                    tangent, direction
                )
            except ValueError:
                continue

            if angle > MAX_ANGLE_DEG:
                continue

            distance = point_segment_distance(
                midpoint,
                segment[0],
                segment[1],
            )

            if best is None or distance < best[0]:
                best = (
                    distance,
                    angle,
                    segment_index,
                )

        if best is None:
            sampled_support.append(False)
            continue

        distance, angle, segment_index = best

        supported = distance <= MAX_DISTANCE_PX

        sampled_support.append(supported)

        if supported:
            matches.append({
                "segment_index": segment_index,
                "distance_px": distance,
                "angle_deg": angle,
            })

    fraction = (
        float(np.mean(sampled_support))
        if sampled_support
        else 0.0
    )

    distinct_segments = len({
        x["segment_index"]
        for x in matches
    })

    return {
        "status": "MEASURED",
        "projected_length_px": projected_length,
        "sampled_intervals": len(sampled_support),
        "supported_intervals": int(
            sum(sampled_support)
        ),
        "support_fraction": fraction,
        "distinct_matching_segments": distinct_segments,
        "median_match_distance_px": (
            float(np.median([
                x["distance_px"]
                for x in matches
            ]))
            if matches else None
        ),
    }


def choose_separated_frames(names):
    ordered = sorted(
        (
            (frame_number(name), name)
            for name in names
            if frame_number(name) is not None
        )
    )

    selected = []

    for number, name in ordered:
        if (
            not selected
            or number - selected[-1][0]
            >= MIN_FRAME_SEPARATION
        ):
            selected.append((number, name))

        if len(selected) >= MAX_FRAMES_PER_CANDIDATE:
            break

    return [name for _, name in selected]


def shifted_lines(points):
    u, v = perpendicular_axes(points)

    variants = []

    for axis_name, axis in (("u", u), ("v", v)):
        for amount in CONTROL_SHIFTS_M:
            variants.append((
                f"shift_{axis_name}_{amount:+.3f}m",
                points + amount * axis,
            ))

    return variants


def evaluate_candidate(
    candidate_id,
    line,
    observation,
    poses,
    detector,
    image_cache,
):
    names = choose_separated_frames(
        observation["visible_frames"]
    )

    controls = shifted_lines(line["points"])

    rows = []

    for name in names:
        pose = poses.get(name)

        if pose is None:
            rows.append({
                "frame": name,
                "status": "POSE_UNAVAILABLE",
            })
            continue

        if name not in image_cache:
            gray = cv2.imread(
                str(FRAMES / name),
                cv2.IMREAD_GRAYSCALE,
            )

            if gray is None:
                image_cache[name] = None
            else:
                image_cache[name] = (
                    gray,
                    detect_segments(gray, detector),
                )

        cached = image_cache[name]

        if cached is None:
            rows.append({
                "frame": name,
                "status": "IMAGE_UNAVAILABLE",
            })
            continue

        gray, segments = cached

        original_projection = project_line(
            line["points"], pose, gray.shape
        )

        original = evaluate_projection(
            original_projection, segments
        )

        control_results = []

        for control_name, shifted in controls:
            projection = project_line(
                shifted, pose, gray.shape
            )

            result = evaluate_projection(
                projection, segments
            )

            if result["status"] == "MEASURED":
                control_results.append({
                    "name": control_name,
                    "support_fraction": result[
                        "support_fraction"
                    ],
                })

        best_control = (
            max(
                control_results,
                key=lambda x: x["support_fraction"],
            )
            if control_results else None
        )

        if original["status"] != "MEASURED":
            status = "ORIGINAL_PROJECTION_UNAVAILABLE"

        elif original["support_fraction"] >= 0.60:
            status = "IMAGE_SUPPORT_OBSERVED"

        else:
            status = "IMAGE_SUPPORT_WEAK"

        control_advantage = None

        if (
            original["status"] == "MEASURED"
            and best_control is not None
        ):
            control_advantage = (
                original["support_fraction"]
                - best_control["support_fraction"]
            )

        rows.append({
            "frame": name,
            "status": status,
            "global_detected_segments": len(segments),
            "original": original,
            "best_shifted_control": best_control,
            "original_minus_best_control_support": (
                control_advantage
            ),
        })

    measured = [
        row for row in rows
        if row["status"] in (
            "IMAGE_SUPPORT_OBSERVED",
            "IMAGE_SUPPORT_WEAK",
        )
    ]

    supported = [
        row for row in measured
        if row["status"] == "IMAGE_SUPPORT_OBSERVED"
    ]

    controls_outperform = [
        row for row in measured
        if (
            row["original_minus_best_control_support"]
            is not None
            and row[
                "original_minus_best_control_support"
            ] < -0.15
        )
    ]

    if len(measured) < MIN_SEPARATED_SUPPORT_VIEWS:
        decision = "INSUFFICIENT_CROSSVIEW_EVIDENCE"

    elif len(controls_outperform) >= 2:
        decision = "SHIFTED_GEOMETRY_CHALLENGE"

    elif len(supported) >= MIN_SEPARATED_SUPPORT_VIEWS:
        decision = "MULTIVIEW_IMAGE_SUPPORT_OBSERVED"

    else:
        decision = "MULTIVIEW_IMAGE_SUPPORT_NOT_ESTABLISHED"

    return {
        "candidate_id": candidate_id,
        "group_id": line["group_id"],
        "step102_status_unchanged": observation["status"],
        "selected_separated_frames": names,
        "frames_measured": len(measured),
        "frames_with_image_support": len(supported),
        "frames_shifted_control_outperforms": len(
            controls_outperform
        ),
        "decision": decision,
        "frames": rows,
        "physical_line_verified": False,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    lines = load_lines()
    observations = load_observability()
    poses = load_poses()

    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    image_cache = {}
    results = []

    for candidate_id in CANDIDATES:
        if (
            candidate_id not in lines
            or candidate_id not in observations
        ):
            raise RuntimeError(
                f"Missing frozen candidate {candidate_id}"
            )

        result = evaluate_candidate(
            candidate_id,
            lines[candidate_id],
            observations[candidate_id],
            poses,
            detector,
            image_cache,
        )

        results.append(result)

        print(
            f"Candidate {candidate_id}: "
            f"measured={result['frames_measured']} "
            f"supported={result['frames_with_image_support']} "
            f"control_challenges="
            f"{result['frames_shifted_control_outperforms']} "
            f"=> {result['decision']}",
            flush=True,
        )

    details = {
        "method": "step103_crossview_image_evidence",
        "results": results,
    }

    report = {
        "method": details["method"],
        "candidate_count": len(results),
        "candidate_summaries": [
            {
                "candidate_id": row["candidate_id"],
                "frames_measured": row["frames_measured"],
                "frames_with_image_support": row[
                    "frames_with_image_support"
                ],
                "frames_shifted_control_outperforms": row[
                    "frames_shifted_control_outperforms"
                ],
                "decision": row["decision"],
            }
            for row in results
        ],
        "parameters": {
            "minimum_frame_separation": (
                MIN_FRAME_SEPARATION
            ),
            "maximum_frames_per_candidate": (
                MAX_FRAMES_PER_CANDIDATE
            ),
            "maximum_distance_px": MAX_DISTANCE_PX,
            "maximum_angle_deg": MAX_ANGLE_DEG,
            "minimum_support_fraction": 0.60,
            "control_challenge_margin": 0.15,
        },
        "geometry_modified": False,
        "camera_poses_modified": False,
        "physical_lines_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "LSD segments are detected globally, but correspondence is selected by proximity to each projected hypothesis.",
            "Global detection does not make the resulting correspondence independent of the tested geometry.",
            "The best shifted control is selected from multiple alternatives, which can increase its apparent advantage.",
            "Frame separation is temporal, not proof of statistically independent observations.",
            "Projected visibility does not account for occlusion.",
            "No measurement of camera-pose or image-feature uncertainty is included.",
            "Support fractions are diagnostics rather than calibrated probabilities.",
            "Image alignment cannot establish physical line identity or CAD correspondence.",
        ],
        "decision": "CROSSVIEW_IMAGE_DIAGNOSTICS_AVAILABLE",
    }

    save_json(
        OUT / "per_candidate_evidence.json",
        details,
    )

    save_json(
        OUT / "evidence_report.json",
        report,
    )

    print("\n=== STEP 103 COMPLETE ===")
    print("Physical lines verified: NO")
    print("CAD registration verified: NO")
    print("Construction progress verified: NO")
    print("Reports:", OUT)


if __name__ == "__main__":
    main()
