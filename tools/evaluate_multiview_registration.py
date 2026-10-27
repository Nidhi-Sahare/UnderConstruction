
"""
Under-construction — Step 86
Multi-view CAD registration consistency.

No learned models, manual annotations, or frontend changes.

Uses Step 80 camera projection and CAD transforms.
Evaluates persistence of segment-level edge support across
different viewpoints.

Run:
    python tools/evaluate_multiview_registration.py

Output:
    outputs/registration_multiview/evaluation_report.json

IMPORTANT:
Multi-view edge support is exploratory evidence, not proof
of geometric correspondence, visibility, or registration.
"""

import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASE_FILE = (
    ROOT / "tools" / "evaluate_cad_image_registration.py"
)

OUTPUT_DIR = ROOT / "outputs" / "registration_multiview"
OUTPUT_FILE = OUTPUT_DIR / "evaluation_report.json"

FRAME_COUNT = 30

# Distances are in the resized image.
EDGE_DISTANCE_PX = 5.0
ORIENTATION_TOLERANCE_DEG = 20.0

# At least this many sampled projected pixels are required
# to score a CAD segment in one frame.
MIN_SEGMENT_PIXELS = 12

# A segment/frame observation must exceed this fraction
# of orientation-consistent nearby observed edges.
MIN_SEGMENT_SUPPORT = 0.12

# Minimum observations from different frames.
MIN_VIEW_COUNT = 3

# Require a nontrivial camera baseline for a multi-view test.
MIN_CAMERA_BASELINE_M = 0.50

# Avoid allowing one long CAD segment to dominate.
MAX_SAMPLES_PER_SEGMENT = 160

# Require useful spatial/temporal coverage.
MIN_ELIGIBLE_SEGMENTS = 5
MIN_PAIRED_SEGMENTS = 5

# Exploratory comparison thresholds.
MIN_CONTROL_ADVANTAGE = 0.02
MIN_CONTROL_WIN_FRACTION = 0.65


def load_base():
    spec = importlib.util.spec_from_file_location(
        "step80_geometry",
        BASE_FILE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load projection functions: {BASE_FILE}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.FRAME_COUNT = FRAME_COUNT
    return module


def prepare_image(frame, scale):
    image = cv2.imread(str(frame["image_path"]))

    if image is None:
        return None

    image = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.0,
    )

    edges = cv2.Canny(gray, 60, 150)

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    # OpenCV labels each zero-valued edge pixel.
    distance, labels = cv2.distanceTransformWithLabels(
        255 - edges,
        cv2.DIST_L2,
        3,
        labelType=cv2.DIST_LABEL_PIXEL,
    )

    edge_y, edge_x = np.nonzero(edges)

    if not len(edge_x):
        return None

    max_label = int(labels.max())

    lookup_x = np.full(
        max_label + 1,
        -1,
        dtype=np.int32,
    )

    lookup_y = np.full(
        max_label + 1,
        -1,
        dtype=np.int32,
    )

    edge_labels = labels[edge_y, edge_x]

    lookup_x[edge_labels] = edge_x
    lookup_y[edge_labels] = edge_y

    return {
        "shape": gray.shape,
        "distance": distance,
        "labels": labels,
        "lookup_x": lookup_x,
        "lookup_y": lookup_y,
        "gx": gx,
        "gy": gy,
    }


def project_segment(base, frame, transform, a, b, z):
    length = math.dist(a, b)

    count = min(
        MAX_SAMPLES_PER_SEGMENT,
        max(
            3,
            int(
                math.ceil(
                    length / base.SAMPLE_SPACING_M
                )
            ) + 1,
        ),
    )

    alpha = np.linspace(
        0.0,
        1.0,
        count,
    )

    start = np.array(
        [a[0], a[1], z],
        dtype=np.float64,
    )

    end = np.array(
        [b[0], b[1], z],
        dtype=np.float64,
    )

    points = (
        start[None, :] * (1 - alpha[:, None])
        + end[None, :] * alpha[:, None]
    )

    points_map = base.cad_to_map(
        points,
        transform,
    )

    return base.project_points(
        points_map,
        frame,
    )


def score_segment_view(
    base,
    frame,
    transform,
    segment,
    evidence,
):
    """
    Sample projected CAD subsegments.

    Normals are computed from projected endpoints.
    Compare each sampled pixel against the nearest
    observed Canny-edge gradient.
    """

    height, width = evidence["shape"]

    a, b = segment

    sampled = set()
    matched = set()

    for z in base.WALL_HEIGHTS:
        pixels, valid = project_segment(
            base,
            frame,
            transform,
            a,
            b,
            z,
        )

        for i in range(len(pixels) - 1):
            if not (valid[i] and valid[i + 1]):
                continue

            p0 = pixels[i]
            p1 = pixels[i + 1]

            if not (
                np.all(np.isfinite(p0))
                and np.all(np.isfinite(p1))
            ):
                continue

            if np.max(
                np.abs(np.concatenate((p0, p1)))
            ) > 100000:
                continue

            direction = p1 - p0
            norm = float(np.linalg.norm(direction))

            if norm < 0.5:
                continue

            normal = np.array(
                [
                    -direction[1] / norm,
                    direction[0] / norm,
                ],
                dtype=np.float64,
            )

            start = tuple(
                np.rint(p0).astype(int)
            )

            end = tuple(
                np.rint(p1).astype(int)
            )

            clipped, c0, c1 = cv2.clipLine(
                (0, 0, width, height),
                start,
                end,
            )

            if not clipped:
                continue

            # Sample every rasterized pixel on the clipped
            # subsegment, not just its endpoints.
            line_length = max(
                abs(c1[0] - c0[0]),
                abs(c1[1] - c0[1]),
            )

            steps = max(2, line_length + 1)

            xs = np.rint(
                np.linspace(
                    c0[0],
                    c1[0],
                    steps,
                )
            ).astype(np.int32)

            ys = np.rint(
                np.linspace(
                    c0[1],
                    c1[1],
                    steps,
                )
            ).astype(np.int32)

            inside = (
                (xs >= 0)
                & (xs < width)
                & (ys >= 0)
                & (ys < height)
            )

            xs = xs[inside]
            ys = ys[inside]

            if not len(xs):
                continue

            distances = evidence["distance"][ys, xs]
            labels = evidence["labels"][ys, xs]

            nearest_x = evidence["lookup_x"][labels]
            nearest_y = evidence["lookup_y"][labels]

            close = (
                (distances <= EDGE_DISTANCE_PX)
                & (nearest_x >= 0)
                & (nearest_y >= 0)
            )

            for j, (x, y) in enumerate(zip(xs, ys)):
                pixel_key = (int(x), int(y))
                sampled.add(pixel_key)

                if not close[j]:
                    continue

                ex = int(nearest_x[j])
                ey = int(nearest_y[j])

                gradient = np.array(
                    [
                        evidence["gx"][ey, ex],
                        evidence["gy"][ey, ex],
                    ],
                    dtype=np.float64,
                )

                magnitude = float(
                    np.linalg.norm(gradient)
                )

                if magnitude < 1e-5:
                    continue

                alignment = abs(
                    float(
                        np.dot(
                            normal,
                            gradient / magnitude,
                        )
                    )
                )

                if alignment >= math.cos(
                    math.radians(
                        ORIENTATION_TOLERANCE_DEG
                    )
                ):
                    matched.add(pixel_key)

    if len(sampled) < MIN_SEGMENT_PIXELS:
        return None

    fraction = len(matched) / len(sampled)

    return {
        "support_fraction": float(fraction),
        "supported": (
            fraction >= MIN_SEGMENT_SUPPORT
        ),
        "projected_pixels": len(sampled),
        "matched_pixels": len(matched),
    }


def baseline_ok(frame_a, frame_b):
    distance = np.linalg.norm(
        frame_a["position"] - frame_b["position"]
    )

    return distance >= MIN_CAMERA_BASELINE_M


def evaluate_transform(
    base,
    transform,
    frames,
    evidence,
    segments,
):
    """
    For each CAD segment, record the views in which its
    projected image evidence is supported.

    Repeated matches only count as multi-view support if
    at least two supporting camera positions have enough
    baseline separation.
    """

    segment_results = []

    for segment_id, segment in enumerate(segments):
        observations = []

        for frame_index, (frame, observed) in enumerate(
            zip(frames, evidence)
        ):
            if observed is None:
                continue

            result = score_segment_view(
                base,
                frame,
                transform,
                segment,
                observed,
            )

            if result is None:
                continue

            observations.append({
                "frame_index": frame_index,
                "frame": frame["filename"],
                **result,
            })

        supported = [
            item
            for item in observations
            if item["supported"]
        ]

        separated = False

        for i in range(len(supported)):
            if separated:
                break

            for j in range(i + 1, len(supported)):
                frame_a = frames[
                    supported[i]["frame_index"]
                ]

                frame_b = frames[
                    supported[j]["frame_index"]
                ]

                if baseline_ok(frame_a, frame_b):
                    separated = True
                    break

        # Require enough distinct supporting frames AND
        # a real camera baseline.
        persistent = (
            len(supported) >= MIN_VIEW_COUNT
            and separated
        )

        eligible = (
            len(observations) >= MIN_VIEW_COUNT
        )

        if not eligible:
            continue

        support_ratio = (
            len(supported) / len(observations)
        )

        # Persistence is evaluated over all eligible views,
        # including views where projected edges do not match.
        persistence_score = (
            support_ratio if persistent else 0.0
        )

        segment_results.append({
            "segment_id": segment_id,
            "eligible_views": len(observations),
            "supported_views": len(supported),
            "support_ratio": float(support_ratio),
            "baseline_verified": separated,
            "persistent": persistent,
            "persistence_score": float(
                persistence_score
            ),
            "supporting_frames": [
                item["frame"]
                for item in supported
            ],
        })

    scores = [
        item["persistence_score"]
        for item in segment_results
    ]

    persistent_count = sum(
        item["persistent"]
        for item in segment_results
    )

    return {
        "status": (
            "SCORED_UNVERIFIED"
            if len(segment_results)
            >= MIN_ELIGIBLE_SEGMENTS
            else "INSUFFICIENT_SEGMENT_COVERAGE"
        ),
        "total_cad_segments": len(segments),
        "eligible_segments": len(segment_results),
        "persistent_segments": persistent_count,
        "persistent_fraction_of_eligible": (
            persistent_count / len(segment_results)
            if segment_results
            else 0.0
        ),
        "coverage_adjusted_score": (
            float(sum(scores) / len(segments))
            if segments
            else 0.0
        ),
        "mean_eligible_score": (
            float(np.mean(scores))
            if scores
            else None
        ),
        "segment_results": segment_results,
    }


def compare_segments(candidate, control):
    candidate_map = {
        item["segment_id"]: item["persistence_score"]
        for item in candidate["segment_results"]
    }

    control_map = {
        item["segment_id"]: item["persistence_score"]
        for item in control["segment_results"]
    }

    common = sorted(
        set(candidate_map) & set(control_map)
    )

    differences = np.array(
        [
            candidate_map[index] - control_map[index]
            for index in common
        ],
        dtype=np.float64,
    )

    if len(differences):
        advantage = float(
            np.mean(differences)
        )

        win_fraction = float(
            np.mean(differences > 0)
        )
    else:
        advantage = None
        win_fraction = None

    sufficient = (
        len(common) >= MIN_PAIRED_SEGMENTS
    )

    passes = (
        sufficient
        and advantage >= MIN_CONTROL_ADVANTAGE
        and win_fraction >= MIN_CONTROL_WIN_FRACTION
    )

    return {
        "status": (
            "INSUFFICIENT_PAIRED_SEGMENTS"
            if not sufficient
            else (
                "PASSES_EXPLORATORY_PAIR_GATE"
                if passes
                else "FAILS_PAIR_GATE"
            )
        ),
        "paired_segments": len(common),
        "mean_advantage": advantage,
        "candidate_win_fraction": win_fraction,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = load_base()

    frames = base.load_frames()
    candidates = base.load_candidates()
    segments = base.load_wall_segments()

    controls = base.generate_controls(candidates)

    evidence = [
        prepare_image(
            frame,
            base.IMAGE_SCALE,
        )
        for frame in frames
    ]

    split = len(frames) // 2

    partitions = {
        "development": (
            frames[:split],
            evidence[:split],
        ),
        "heldout": (
            frames[split:],
            evidence[split:],
        ),
    }

    report = {
        "sequence": base.SEQUENCE,
        "method": "multiview_segment_persistence",
        "registration_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "frame_count": FRAME_COUNT,
            "edge_distance_px": EDGE_DISTANCE_PX,
            "orientation_tolerance_deg": (
                ORIENTATION_TOLERANCE_DEG
            ),
            "min_segment_pixels": MIN_SEGMENT_PIXELS,
            "min_segment_support": MIN_SEGMENT_SUPPORT,
            "min_view_count": MIN_VIEW_COUNT,
            "min_camera_baseline_m": (
                MIN_CAMERA_BASELINE_M
            ),
            "min_eligible_segments": (
                MIN_ELIGIBLE_SEGMENTS
            ),
        },
        "candidates": [],
        "negative_controls": [],
    }

    print("\n=== STEP 86: MULTI-VIEW CONSISTENCY ===")
    print("Frames:", len(frames))
    print("CAD segments:", len(segments))
    print("Candidates:", len(candidates))
    print("Controls:", len(controls))

    for candidate in candidates:
        print(
            f"Evaluating candidate {candidate['rank']}...",
            flush=True,
        )

        entry = {
            "rank": candidate["rank"],
            "transform": candidate,
        }

        for name, (part_frames, part_evidence) in (
            partitions.items()
        ):
            entry[name] = evaluate_transform(
                base,
                candidate,
                part_frames,
                part_evidence,
                segments,
            )

        report["candidates"].append(entry)

    for control in controls:
        print(
            f"Control {control['rank']} "
            f"{control['control_name']}...",
            flush=True,
        )

        entry = {
            "source_candidate_rank": control["rank"],
            "control_name": control["control_name"],
            "transform": control,
        }

        for name, (part_frames, part_evidence) in (
            partitions.items()
        ):
            entry[name] = evaluate_transform(
                base,
                control,
                part_frames,
                part_evidence,
                segments,
            )

        report["negative_controls"].append(entry)

    # Development-only selection.
    eligible = [
        item
        for item in report["candidates"]
        if item["development"]["status"]
        == "SCORED_UNVERIFIED"
    ]

    selected = (
        max(
            eligible,
            key=lambda item: item["development"][
                "coverage_adjusted_score"
            ],
        )
        if eligible
        else None
    )

    report["development_selected_candidate"] = (
        selected["rank"]
        if selected
        else None
    )

    if selected is not None:
        related_controls = [
            item
            for item in report["negative_controls"]
            if item["source_candidate_rank"]
            == selected["rank"]
        ]

        comparisons = {}

        for partition in (
            "development",
            "heldout",
        ):
            comparisons[partition] = [
                {
                    "control_name": item["control_name"],
                    **compare_segments(
                        selected[partition],
                        item[partition],
                    ),
                }
                for item in related_controls
            ]

        report["selected_candidate_controls"] = (
            comparisons
        )

    report["decision"] = (
        "UNVERIFIED_MULTIVIEW_EVIDENCE_ONLY"
    )

    report["limitations"] = [
        "Segment persistence does not establish correct correspondence.",
        "Repeated scene clutter may produce persistent false matches.",
        "CAD wall visibility and occlusion are not modeled.",
        "CAD wall heights are illustrative.",
        "Camera frames are temporally correlated.",
        "Segment sampling and acceptance thresholds are exploratory.",
        "Missing eligible segment observations are not proof of absence.",
        "No independent ground-truth CAD registration is available.",
    ]

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== RESULTS ===")

    for item in report["candidates"]:
        print(
            f"Candidate {item['rank']}: "
            f"dev={item['development']['coverage_adjusted_score']:.5f} "
            f"heldout={item['heldout']['coverage_adjusted_score']:.5f} "
            f"persistent_dev="
            f"{item['development']['persistent_segments']} "
            f"persistent_heldout="
            f"{item['heldout']['persistent_segments']}"
        )

    print(
        "\nDevelopment selection:",
        report["development_selected_candidate"],
    )

    if selected is not None:
        print("\nHeld-out negative controls:")

        for comparison in report[
            "selected_candidate_controls"
        ]["heldout"]:
            print(
                comparison["control_name"],
                comparison["status"],
                "paired=",
                comparison["paired_segments"],
                "advantage=",
                comparison["mean_advantage"],
                "win_fraction=",
                comparison["candidate_win_fraction"],
            )

    print("\nRegistration verified: NO")
    print("Report:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
