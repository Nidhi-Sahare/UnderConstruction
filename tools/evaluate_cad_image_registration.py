
"""
Under-construction — Step 80
Automatic CAD-to-image registration evaluation.

Classical OpenCV only. No manual annotations or learned models.

Inputs:
    outputs/automatic_registration_v2/registration_candidates.json
    data/floorplans/floor_2.dxf
    archived Hilti cam0 frame_manifest.csv and original JPGs

Output:
    outputs/cad_image_evaluation/evaluation_report.json

IMPORTANT:
    Scores measure image-edge agreement, NOT verified registration.
    Unmodelled occlusion, incomplete CAD, unknown wall heights,
    image clutter, and calibration uncertainty remain limitations.
"""

import csv
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "floor_2_2025-10-28_run_2"

FRAME_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
)

MANIFEST = FRAME_ROOT / "frame_manifest.csv"
DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

CANDIDATES_PATH = (
    ROOT
    / "outputs"
    / "automatic_registration_v2"
    / "registration_candidates.json"
)

OUTPUT_DIR = ROOT / "outputs" / "cad_image_evaluation"

WALL_LAYERS = {
    "015 Innenwände",
    "010 Aussenwände",
    "016 Trennwände",
}

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

# Evaluation settings. These are heuristics, not validated
# acceptance thresholds.
FRAME_COUNT = 30
SAMPLE_SPACING_M = 0.20
MAX_WALL_SEGMENTS = 800
IMAGE_SCALE = 0.5
MAX_EDGE_DISTANCE_PX = 6.0
ORIENTATION_TOLERANCE_DEG = 25.0
MIN_PROJECTED_PIXELS = 100
MIN_VALID_FRAMES = 5

WALL_HEIGHTS = (0.0, 3.0)

NEGATIVE_ROTATIONS_DEG = (45.0, 90.0, 135.0)
NEGATIVE_TRANSLATIONS_M = (
    (5.0, 0.0),
    (0.0, 5.0),
    (-5.0, -5.0),
)


def load_frames():
    frames = []

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("pose_matched", "").strip().lower() not in {
                "true", "1", "yes"
            }:
                continue

            try:
                position = np.array(
                    [
                        float(row[key])
                        for key in ("tx", "ty", "tz")
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        float(row[key])
                        for key in ("qx", "qy", "qz", "qw")
                    ],
                    dtype=np.float64,
                )

                if not (
                    np.all(np.isfinite(position))
                    and np.all(np.isfinite(quaternion))
                ):
                    continue

                norm = np.linalg.norm(quaternion)

                if not 0.99 <= norm <= 1.01:
                    continue

                image_path = (
                    FRAME_ROOT / "frames" / row["filename"]
                )

                if not image_path.is_file():
                    continue

                frames.append(
                    {
                        "filename": row["filename"],
                        "image_path": image_path,
                        "position": position,
                        "rotation": Rotation.from_quat(
                            quaternion
                        ).as_matrix(),
                    }
                )

            except (KeyError, ValueError):
                continue

    if len(frames) < MIN_VALID_FRAMES:
        raise RuntimeError("Insufficient matched frames.")

    indices = np.linspace(
        0,
        len(frames) - 1,
        min(FRAME_COUNT, len(frames)),
        dtype=int,
    )

    return [frames[i] for i in sorted(set(indices))]


def load_candidates():
    data = json.loads(
        CANDIDATES_PATH.read_text(encoding="utf-8")
    )

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        entries = None

        for key in (
            "candidates",
            "registration_candidates",
            "ranked_candidates",
            "results",
        ):
            if isinstance(data.get(key), list):
                entries = data[key]
                break

        if entries is None:
            raise ValueError(
                "Cannot find registration candidates."
            )
    else:
        raise ValueError("Invalid candidates JSON.")

    candidates = []

    for index, item in enumerate(entries):
        transform = item.get("transform", item)

        def field(*names):
            for name in names:
                if name in transform:
                    return float(transform[name])

            raise KeyError(
                f"Candidate {index + 1} missing {names}"
            )

        candidates.append(
            {
                "rank": index + 1,
                "angle_deg": field(
                    "angle_deg",
                    "rotation_deg",
                    "theta_deg",
                ),
                "tx": field("tx", "translation_x"),
                "ty": field("ty", "translation_y"),
                "kind": "candidate",
            }
        )

    return candidates


def load_wall_segments():
    doc = ezdxf.readfile(str(DXF_PATH))
    model = doc.modelspace()
    segments = []

    for entity in model:
        if entity.dxf.layer not in WALL_LAYERS:
            continue

        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end

            points = [
                (float(a.x), float(a.y)),
                (float(b.x), float(b.y)),
            ]

        elif kind == "LWPOLYLINE":
            points = [
                (float(x), float(y))
                for x, y in entity.get_points("xy")
            ]

        else:
            continue

        pairs = list(zip(points[:-1], points[1:]))

        if kind == "LWPOLYLINE" and entity.closed:
            pairs.append((points[-1], points[0]))

        for a, b in pairs:
            if 0.05 < math.dist(a, b) < 100.0:
                segments.append((a, b))

    if len(segments) > MAX_WALL_SEGMENTS:
        indices = np.linspace(
            0,
            len(segments) - 1,
            MAX_WALL_SEGMENTS,
            dtype=int,
        )

        segments = [segments[i] for i in indices]

    if not segments:
        raise RuntimeError("No CAD wall segments found.")

    return segments


def generate_controls(candidates):
    controls = []

    for candidate in candidates:
        for rotation in NEGATIVE_ROTATIONS_DEG:
            controls.append(
                {
                    **candidate,
                    "kind": "negative_rotation",
                    "angle_deg": (
                        candidate["angle_deg"] + rotation
                    ) % 360,
                    "control_name": f"rotation_{rotation:g}",
                }
            )

        for dx, dy in NEGATIVE_TRANSLATIONS_M:
            controls.append(
                {
                    **candidate,
                    "kind": "negative_translation",
                    "tx": candidate["tx"] + dx,
                    "ty": candidate["ty"] + dy,
                    "control_name": (
                        f"translation_{dx:g}_{dy:g}"
                    ),
                }
            )

    return controls


def cad_to_map(points, candidate):
    theta = math.radians(candidate["angle_deg"])

    c = math.cos(theta)
    s = math.sin(theta)

    rotation = np.array(
        [[c, -s], [s, c]],
        dtype=np.float64,
    )

    translation = np.array(
        [candidate["tx"], candidate["ty"]],
        dtype=np.float64,
    )

    output = points.copy()

    output[:, :2] = (
        points[:, :2] - translation
    ) @ rotation

    return output


def sample_segment(a, b):
    distance = math.dist(a, b)

    count = max(
        2,
        int(math.ceil(distance / SAMPLE_SPACING_M)) + 1,
    )

    alpha = np.linspace(0.0, 1.0, count)

    return (
        np.asarray(a)[None, :] * (1 - alpha[:, None])
        + np.asarray(b)[None, :] * alpha[:, None]
    )


def project_points(points_map, frame):
    camera_points = (
        points_map - frame["position"]
    ) @ frame["rotation"]

    visible = camera_points[:, 2] > 0.05

    projected = np.full(
        (len(points_map), 2),
        np.nan,
        dtype=np.float64,
    )

    if not np.any(visible):
        return projected, visible

    pixels, _ = cv2.fisheye.projectPoints(
        camera_points[visible].reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K,
        D,
    )

    projected[visible] = (
        pixels.reshape(-1, 2) * IMAGE_SCALE
    )

    return projected, visible


def make_projection_mask(frame, candidate, segments, shape):
    height, width = shape

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for a, b in segments:
        for z in WALL_HEIGHTS:
            samples = sample_segment(
                (a[0], a[1], z),
                (b[0], b[1], z),
            )

            points_map = cad_to_map(
                samples, candidate
            )

            pixels, valid = project_points(
                points_map, frame
            )

            for index in range(len(pixels) - 1):
                if not (
                    valid[index]
                    and valid[index + 1]
                ):
                    continue

                p0 = pixels[index]
                p1 = pixels[index + 1]

                if not (
                    np.all(np.isfinite(p0))
                    and np.all(np.isfinite(p1))
                ):
                    continue

                if np.max(
                    np.abs(np.concatenate([p0, p1]))
                ) > 100000:
                    continue

                a_px = tuple(
                    np.rint(p0).astype(int)
                )

                b_px = tuple(
                    np.rint(p1).astype(int)
                )

                visible_line, c0, c1 = cv2.clipLine(
                    (0, 0, width, height),
                    a_px,
                    b_px,
                )

                if visible_line:
                    cv2.line(
                        mask,
                        c0,
                        c1,
                        255,
                        1,
                        cv2.LINE_8,
                    )

    return mask


def prepare_image_evidence(frame):
    image = cv2.imread(str(frame["image_path"]))

    if image is None:
        raise RuntimeError(
            f"Cannot read {frame['image_path']}"
        )

    image = cv2.resize(
        image,
        None,
        fx=IMAGE_SCALE,
        fy=IMAGE_SCALE,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray, (5, 5), 1.0
    )

    edges = cv2.Canny(
        gray, 60, 150
    )

    gx = cv2.Sobel(
        gray, cv2.CV_32F, 1, 0, ksize=3
    )

    gy = cv2.Sobel(
        gray, cv2.CV_32F, 0, 1, ksize=3
    )

    distance = cv2.distanceTransform(
        255 - edges,
        cv2.DIST_L2,
        3,
    )

    return {
        "edges": edges,
        "gx": gx,
        "gy": gy,
        "distance": distance,
        "shape": gray.shape,
    }


def score_projection(mask, evidence):
    projected_count = int(
        np.count_nonzero(mask)
    )

    if projected_count < MIN_PROJECTED_PIXELS:
        return None

    ys, xs = np.nonzero(mask)

    distances = evidence["distance"][ys, xs]

    close = distances <= MAX_EDGE_DISTANCE_PX

    distance_score = np.exp(
        -np.minimum(
            distances,
            MAX_EDGE_DISTANCE_PX * 3,
        ) / MAX_EDGE_DISTANCE_PX
    )

    # Estimate projected-edge tangent directions
    # from the rasterized mask, then compare with
    # observed image-gradient normals.
    mask_float = mask.astype(np.float32)

    mx = cv2.Sobel(
        mask_float, cv2.CV_32F, 1, 0, ksize=3
    )

    my = cv2.Sobel(
        mask_float, cv2.CV_32F, 0, 1, ksize=3
    )

    projected_angle = np.arctan2(
        my[ys, xs],
        mx[ys, xs],
    )

    observed_angle = np.arctan2(
        evidence["gy"][ys, xs],
        evidence["gx"][ys, xs],
    )

    angle_difference = np.abs(
        np.arctan2(
            np.sin(
                projected_angle - observed_angle
            ),
            np.cos(
                projected_angle - observed_angle
            ),
        )
    )

    # Edge direction is unsigned, so opposite
    # gradient normals are equivalent.
    angle_difference = np.minimum(
        angle_difference,
        np.pi - angle_difference,
    )

    orientation_agreement = (
        angle_difference
        <= math.radians(
            ORIENTATION_TOLERANCE_DEG
        )
    )

    # Distance dominates, while orientation
    # provides an additional consistency check.
    score = (
        0.65 * float(np.mean(distance_score))
        + 0.35 * float(
            np.mean(close & orientation_agreement)
        )
    )

    return {
        "score": score,
        "projected_pixels": projected_count,
        "close_fraction": float(np.mean(close)),
        "orientation_fraction": float(
            np.mean(close & orientation_agreement)
        ),
        "mean_distance_px": float(
            np.mean(distances)
        ),
    }


def evaluate_transform(transform, frames, segments, evidence):
    results = []

    for frame, observed in zip(frames, evidence):
        mask = make_projection_mask(
            frame,
            transform,
            segments,
            observed["shape"],
        )

        score = score_projection(
            mask, observed
        )

        if score is None:
            continue

        results.append(
            {
                "frame": frame["filename"],
                **score,
            }
        )

    if len(results) < MIN_VALID_FRAMES:
        return {
            "status": "INSUFFICIENT_COVERAGE",
            "valid_frames": len(results),
            "mean_score": None,
            "frame_results": results,
        }

    scores = np.array(
        [item["score"] for item in results]
    )

    return {
        "status": "SCORED_UNVERIFIED",
        "valid_frames": len(results),
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "frame_results": results,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_frames()
    candidates = load_candidates()
    segments = load_wall_segments()

    evidence = [
        prepare_image_evidence(frame)
        for frame in frames
    ]

    split = len(frames) // 2

    development_frames = frames[:split]
    heldout_frames = frames[split:]

    development_evidence = evidence[:split]
    heldout_evidence = evidence[split:]

    controls = generate_controls(candidates)

    print("\n=== CAD IMAGE REGISTRATION EVALUATION ===")
    print("Frames:", len(frames))
    print("Development frames:", len(development_frames))
    print("Held-out frames:", len(heldout_frames))
    print("Candidates:", len(candidates))
    print("Negative controls:", len(controls))

    report = {
        "sequence": SEQUENCE,
        "pose_convention": "T_map_cam0",
        "registration_verified": False,
        "settings": {
            "frame_count": FRAME_COUNT,
            "sample_spacing_m": SAMPLE_SPACING_M,
            "image_scale": IMAGE_SCALE,
            "max_edge_distance_px": MAX_EDGE_DISTANCE_PX,
            "orientation_tolerance_deg": ORIENTATION_TOLERANCE_DEG,
            "min_projected_pixels": MIN_PROJECTED_PIXELS,
            "min_valid_frames": MIN_VALID_FRAMES,
            "wall_heights_m": list(WALL_HEIGHTS),
        },
        "candidates": [],
        "negative_controls": [],
    }

    for candidate in candidates:
        print(
            f"Evaluating candidate {candidate['rank']}..."
        )

        development = evaluate_transform(
            candidate,
            development_frames,
            segments,
            development_evidence,
        )

        heldout = evaluate_transform(
            candidate,
            heldout_frames,
            segments,
            heldout_evidence,
        )

        report["candidates"].append(
            {
                "rank": candidate["rank"],
                "transform": candidate,
                "development": development,
                "heldout": heldout,
            }
        )

    for control in controls:
        print(
            f"Negative control: "
            f"candidate {control['rank']} "
            f"{control['control_name']}"
        )

        development = evaluate_transform(
            control,
            development_frames,
            segments,
            development_evidence,
        )

        heldout = evaluate_transform(
            control,
            heldout_frames,
            segments,
            heldout_evidence,
        )

        report["negative_controls"].append(
            {
                "source_candidate_rank": control["rank"],
                "control_name": control["control_name"],
                "development": development,
                "heldout": heldout,
            }
        )

    # Selection uses development data ONLY.
    eligible = [
        item
        for item in report["candidates"]
        if item["development"]["mean_score"] is not None
    ]

    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["development"]["mean_score"]
            ),
        )

        report["development_selected_candidate"] = (
            selected["rank"]
        )

        report["selected_heldout_score"] = (
            selected["heldout"]["mean_score"]
        )
    else:
        report["development_selected_candidate"] = None
        report["selected_heldout_score"] = None

    report["decision"] = (
        "UNVERIFIED_REQUIRES_CONTROL_CALIBRATION"
    )

    destination = (
        OUTPUT_DIR / "evaluation_report.json"
    )

    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")

    for item in report["candidates"]:
        print(
            f"Candidate {item['rank']}: "
            f"dev={item['development']['mean_score']} "
            f"heldout={item['heldout']['mean_score']}"
        )

    print(
        "\nDevelopment selection:",
        report["development_selected_candidate"],
    )

    print(
        "Selected held-out score:",
        report["selected_heldout_score"],
    )

    print("Registration verified: NO")
    print("Report:", destination)


if __name__ == "__main__":
    main()
