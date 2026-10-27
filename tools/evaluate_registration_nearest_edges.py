
"""
Under-construction — Step 85
Segment-tangent CAD-to-image registration evaluation.

Classical OpenCV, NumPy, SciPy and existing Step 80 geometry.
No manual annotations, learned models, or frontend changes.

Key corrections:
1. Projected normals come from projected segment endpoints.
2. Nearest observed Canny-edge gradients supply image normals.
3. Overlapping CAD segments are deduplicated per image pixel.
4. Every negative control is independently paired.
5. Missing frames penalize coverage-adjusted scores.
6. Candidate selection uses development frames only.
7. No registration is declared verified.

Run:
    python tools/evaluate_registration_nearest_edges.py
"""

import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT / "tools" / "evaluate_cad_image_registration.py"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "registration_segment_tangents"
)

OUTPUT_FILE = OUTPUT_DIR / "evaluation_report.json"

FRAME_COUNT = 20

MAX_DISTANCE_PX = 6.0
ANGLE_TOLERANCE_DEG = 25.0

MIN_PROJECTED_PIXELS = 150
MIN_MATCHED_PIXELS = 30

GRID_ROWS = 3
GRID_COLS = 4
MIN_OCCUPIED_CELLS = 3

MIN_VALID_FRAMES = 5
MIN_COVERAGE = 0.70
MIN_PAIRED_FRAMES = 5

# Exploratory thresholds only.
MIN_MEAN_ADVANTAGE = 0.01
MIN_FRAME_WIN_FRACTION = 0.70

# Used to distinguish ambiguous overlapping projected lines.
MIN_NORMAL_CONSISTENCY = 0.95


def load_base():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Missing Step 80 script: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "registration_step80",
        BASE_SCRIPT,
    )

    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    base.FRAME_COUNT = FRAME_COUNT

    return base


def prepare_evidence(frame, scale):
    image = cv2.imread(str(frame["image_path"]))

    if image is None:
        return {
            "status": "IMAGE_READ_FAILED",
        }

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

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        1.0,
    )

    edges = cv2.Canny(
        blurred,
        60,
        150,
    )

    gx = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        blurred,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    edge_y, edge_x = np.nonzero(edges)

    if len(edge_x) == 0:
        return {
            "status": "NO_OBSERVED_EDGES",
            "shape": gray.shape,
        }

    distance, labels = (
        cv2.distanceTransformWithLabels(
            255 - edges,
            cv2.DIST_L2,
            3,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
    )

    edge_labels = labels[edge_y, edge_x]

    max_label = int(labels.max())

    nearest_x = np.full(
        max_label + 1,
        -1,
        dtype=np.int32,
    )

    nearest_y = np.full(
        max_label + 1,
        -1,
        dtype=np.int32,
    )

    nearest_x[edge_labels] = edge_x
    nearest_y[edge_labels] = edge_y

    return {
        "status": "READY",
        "shape": gray.shape,
        "distance": distance,
        "labels": labels,
        "nearest_x": nearest_x,
        "nearest_y": nearest_y,
        "gx": gx,
        "gy": gy,
        "observed_edge_pixels": int(len(edge_x)),
    }


def rasterize_projected_segments(
    base,
    frame,
    transform,
    segments,
    shape,
):
    """
    Return projected pixels with segment-derived normals.

    Each projected line is rasterized separately. Its normal is
    computed from its two projected endpoints, not from the
    gradient of a combined binary mask.

    At overlapping pixels, orientation is represented using
    the doubled-angle vector (cos 2a, sin 2a). Opposite normals
    are equivalent. Conflicting orientations are rejected.
    """

    height, width = shape

    sum_cos2 = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    sum_sin2 = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    counts = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    visible_segments = 0

    for a, b in segments:
        for z in base.WALL_HEIGHTS:
            samples = base.sample_segment(
                (a[0], a[1], z),
                (b[0], b[1], z),
            )

            points_map = base.cad_to_map(
                samples,
                transform,
            )

            pixels, valid = base.project_points(
                points_map,
                frame,
            )

            for i in range(len(pixels) - 1):
                if not (
                    valid[i]
                    and valid[i + 1]
                ):
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

                delta = p1 - p0
                length = float(np.linalg.norm(delta))

                if length < 0.5:
                    continue

                # Segment tangent (dx,dy), rotated 90 degrees.
                nx = -delta[1] / length
                ny = delta[0] / length

                # Doubled angle makes n and -n equivalent.
                cos2 = nx * nx - ny * ny
                sin2 = 2.0 * nx * ny

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

                line_mask = np.zeros(
                    (height, width),
                    dtype=np.uint8,
                )

                cv2.line(
                    line_mask,
                    c0,
                    c1,
                    255,
                    1,
                    cv2.LINE_8,
                )

                ys, xs = np.nonzero(line_mask)

                if len(xs) == 0:
                    continue

                # Accumulate once per projected subsegment.
                # Overlap is resolved below, not double-scored.
                sum_cos2[ys, xs] += cos2
                sum_sin2[ys, xs] += sin2

                counts[ys, xs] = np.minimum(
                    counts[ys, xs].astype(np.uint32) + 1,
                    65535,
                ).astype(np.uint16)

                visible_segments += 1

    occupied = counts > 0

    consistency = np.zeros(
        shape,
        dtype=np.float32,
    )

    consistency[occupied] = (
        np.hypot(
            sum_cos2[occupied],
            sum_sin2[occupied],
        )
        / counts[occupied]
    )

    # Reject intersections where multiple projected line
    # directions disagree strongly.
    usable = (
        occupied
        & (consistency >= MIN_NORMAL_CONSISTENCY)
    )

    ys, xs = np.nonzero(usable)

    # Half-angle reconstructs an unsigned normal direction.
    angles = 0.5 * np.arctan2(
        sum_sin2[ys, xs],
        sum_cos2[ys, xs],
    )

    return {
        "xs": xs,
        "ys": ys,
        "nx": np.cos(angles).astype(np.float32),
        "ny": np.sin(angles).astype(np.float32),
        "visible_subsegments": visible_segments,
        "all_projected_pixels": int(
            np.count_nonzero(occupied)
        ),
        "ambiguous_overlap_pixels": int(
            np.count_nonzero(occupied & ~usable)
        ),
    }


def spatial_coverage(xs, ys, width, height):
    if len(xs) == 0:
        return 0

    cols = np.minimum(
        xs * GRID_COLS // width,
        GRID_COLS - 1,
    )

    rows = np.minimum(
        ys * GRID_ROWS // height,
        GRID_ROWS - 1,
    )

    return int(
        np.unique(rows * GRID_COLS + cols).size
    )


def score_projection(projected, evidence):
    if evidence["status"] != "READY":
        return {
            "status": evidence["status"],
            "score": None,
        }

    xs = projected["xs"]
    ys = projected["ys"]

    count = len(xs)

    diagnostics = {
        "projected_pixels": int(count),
        "all_projected_pixels": projected[
            "all_projected_pixels"
        ],
        "ambiguous_overlap_pixels": projected[
            "ambiguous_overlap_pixels"
        ],
        "visible_subsegments": projected[
            "visible_subsegments"
        ],
    }

    if count < MIN_PROJECTED_PIXELS:
        return {
            "status": "INSUFFICIENT_PROJECTION",
            "score": None,
            **diagnostics,
        }

    distances = evidence["distance"][ys, xs]

    labels = evidence["labels"][ys, xs]

    nearest_x = evidence["nearest_x"][labels]
    nearest_y = evidence["nearest_y"][labels]

    nearest_valid = (
        (nearest_x >= 0)
        & (nearest_y >= 0)
    )

    observed_nx = np.zeros(
        count,
        dtype=np.float32,
    )

    observed_ny = np.zeros(
        count,
        dtype=np.float32,
    )

    indices = np.flatnonzero(nearest_valid)

    if len(indices):
        xx = nearest_x[indices]
        yy = nearest_y[indices]

        observed_nx[indices] = evidence["gx"][
            yy, xx
        ]

        observed_ny[indices] = evidence["gy"][
            yy, xx
        ]

    observed_magnitude = np.hypot(
        observed_nx,
        observed_ny,
    )

    valid_orientation = (
        nearest_valid
        & (observed_magnitude > 1e-5)
    )

    cosine = np.zeros(
        count,
        dtype=np.float32,
    )

    good = valid_orientation

    cosine[good] = np.abs(
        (
            projected["nx"][good] * observed_nx[good]
            + projected["ny"][good] * observed_ny[good]
        )
        / observed_magnitude[good]
    )

    cosine = np.clip(
        cosine,
        0.0,
        1.0,
    )

    angle_ok = cosine >= math.cos(
        math.radians(ANGLE_TOLERANCE_DEG)
    )

    close = distances <= MAX_DISTANCE_PX

    matched = (
        close
        & valid_orientation
        & angle_ok
    )

    matched_count = int(
        np.count_nonzero(matched)
    )

    diagnostics["matched_pixels"] = matched_count

    if matched_count < MIN_MATCHED_PIXELS:
        return {
            "status": "INSUFFICIENT_MATCHES",
            "score": None,
            **diagnostics,
        }

    height, width = evidence["shape"]

    occupied_cells = spatial_coverage(
        xs[matched],
        ys[matched],
        width,
        height,
    )

    diagnostics["occupied_cells"] = occupied_cells

    if occupied_cells < MIN_OCCUPIED_CELLS:
        return {
            "status": "SPATIALLY_CONCENTRATED",
            "score": None,
            **diagnostics,
        }

    matched_fraction = matched_count / count

    closeness = np.exp(
        -distances[matched] / MAX_DISTANCE_PX
    )

    mean_closeness = float(
        np.mean(closeness)
    )

    coverage_fraction = (
        occupied_cells / (GRID_ROWS * GRID_COLS)
    )

    score = (
        matched_fraction
        * mean_closeness
        * (0.75 + 0.25 * coverage_fraction)
    )

    return {
        "status": "SCORED_UNVERIFIED",
        "score": float(score),
        **diagnostics,
        "matched_fraction": float(matched_fraction),
        "mean_closeness": mean_closeness,
        "coverage_fraction": float(coverage_fraction),
        "mean_nearest_distance_px": float(
            np.mean(distances)
        ),
    }


def evaluate_transform(
    base,
    transform,
    frames,
    evidence,
    segments,
):
    results = []

    for frame, observed in zip(frames, evidence):
        if observed["status"] != "READY":
            score = {
                "status": observed["status"],
                "score": None,
            }
        else:
            projected = rasterize_projected_segments(
                base,
                frame,
                transform,
                segments,
                observed["shape"],
            )

            score = score_projection(
                projected,
                observed,
            )

        results.append({
            "frame": frame["filename"],
            **score,
        })

    valid = [
        float(item["score"])
        for item in results
        if item["score"] is not None
    ]

    total = len(results)
    coverage = len(valid) / total if total else 0.0

    # Missing scores contribute zero to the adjusted mean.
    adjusted = (
        float(sum(valid) / total)
        if total
        else None
    )

    return {
        "status": (
            "SCORED_UNVERIFIED"
            if len(valid) >= MIN_VALID_FRAMES
            and coverage >= MIN_COVERAGE
            else "INSUFFICIENT_COVERAGE"
        ),
        "total_frames": total,
        "valid_frames": len(valid),
        "coverage_fraction": coverage,
        "mean_score": (
            float(np.mean(valid))
            if valid
            else None
        ),
        "coverage_adjusted_score": adjusted,
        "frame_results": results,
    }


def compare_pair(candidate, control):
    candidate_map = {
        row["frame"]: row["score"]
        for row in candidate["frame_results"]
    }

    control_map = {
        row["frame"]: row["score"]
        for row in control["frame_results"]
    }

    names = sorted(
        set(candidate_map) & set(control_map)
    )

    paired = [
        (
            name,
            float(candidate_map[name]),
            float(control_map[name]),
        )
        for name in names
        if candidate_map[name] is not None
        and control_map[name] is not None
    ]

    differences = np.array(
        [a - b for _, a, b in paired],
        dtype=np.float64,
    )

    total = len(names)
    n = len(paired)

    paired_coverage = (
        n / total if total else 0.0
    )

    sufficient = (
        n >= MIN_PAIRED_FRAMES
        and paired_coverage >= MIN_COVERAGE
    )

    advantage = (
        float(np.mean(differences))
        if n
        else None
    )

    win_fraction = (
        float(np.mean(differences > 0))
        if n
        else None
    )

    passes = (
        sufficient
        and advantage >= MIN_MEAN_ADVANTAGE
        and win_fraction >= MIN_FRAME_WIN_FRACTION
    )

    if not sufficient:
        status = "INSUFFICIENT_PAIRED_COVERAGE"
    elif passes:
        status = "PASSES_EXPLORATORY_PAIR_GATE"
    else:
        status = "FAILS_PAIR_GATE"

    return {
        "status": status,
        "total_common_frames": total,
        "paired_valid_frames": n,
        "paired_coverage_fraction": paired_coverage,
        "mean_score_advantage": advantage,
        "frame_win_fraction": win_fraction,
        "candidate_coverage_fraction": candidate[
            "coverage_fraction"
        ],
        "control_coverage_fraction": control[
            "coverage_fraction"
        ],
        "paired_frame_results": [
            {
                "frame": name,
                "candidate_score": a,
                "control_score": b,
                "advantage": a - b,
            }
            for name, a, b in paired
        ],
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
        prepare_evidence(frame, base.IMAGE_SCALE)
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
        "method": "projected_segment_tangent_nearest_edge",
        "pose_convention": "T_map_cam0",
        "registration_verified": False,
        "occlusion_verified": False,
        "visibility_verified": False,
        "construction_progress_verified": False,
        "settings": {
            "frame_count": FRAME_COUNT,
            "image_scale": base.IMAGE_SCALE,
            "sample_spacing_m": base.SAMPLE_SPACING_M,
            "wall_heights_m": list(base.WALL_HEIGHTS),
            "max_distance_px": MAX_DISTANCE_PX,
            "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
            "min_projected_pixels": MIN_PROJECTED_PIXELS,
            "min_matched_pixels": MIN_MATCHED_PIXELS,
            "min_occupied_cells": MIN_OCCUPIED_CELLS,
            "min_normal_consistency": MIN_NORMAL_CONSISTENCY,
            "min_paired_frames": MIN_PAIRED_FRAMES,
            "min_coverage": MIN_COVERAGE,
            "min_mean_advantage": MIN_MEAN_ADVANTAGE,
            "min_frame_win_fraction": MIN_FRAME_WIN_FRACTION,
        },
        "candidates": [],
        "negative_controls": [],
        "limitations": [
            "Wall heights are illustrative, not measured.",
            "CAD visibility and occlusion are not modeled.",
            "Canny edges include clutter and unrelated geometry.",
            "Nearest-edge matching does not establish correspondence.",
            "Negative controls are perturbations, not ground truth.",
            "Frame observations are temporally correlated.",
            "Thresholds are exploratory, not calibrated.",
        ],
    }

    print("\n=== STEP 85: SEGMENT-TANGENT EVALUATION ===")
    print("Frames:", len(frames))
    print("Candidates:", len(candidates))
    print("Controls:", len(controls))
    print("CAD wall segments:", len(segments))

    for candidate in candidates:
        print(f"Candidate {candidate['rank']}...")

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
            f"Control {control['rank']}: "
            f"{control['control_name']}"
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

    # Candidate selection is strictly development-only.
    eligible = [
        entry
        for entry in report["candidates"]
        if entry["development"]["status"]
        == "SCORED_UNVERIFIED"
        and entry["development"][
            "coverage_adjusted_score"
        ] is not None
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
        related = [
            item
            for item in report["negative_controls"]
            if item["source_candidate_rank"]
            == selected["rank"]
        ]

        comparisons = {}

        for partition in ("development", "heldout"):
            comparisons[partition] = [
                {
                    "control_name": item["control_name"],
                    "control_kind": item[
                        "transform"
                    ]["kind"],
                    **compare_pair(
                        selected[partition],
                        item[partition],
                    ),
                }
                for item in related
            ]

        report["selected_candidate_controls"] = comparisons

        heldout_results = comparisons["heldout"]

        report["heldout_control_status"] = (
            "PASSES_EXPLORATORY_CONTROLS"
            if heldout_results
            and all(
                item["status"]
                == "PASSES_EXPLORATORY_PAIR_GATE"
                for item in heldout_results
            )
            else "DOES_NOT_PASS_CONTROLS"
        )
    else:
        report["heldout_control_status"] = (
            "NO_ELIGIBLE_DEVELOPMENT_CANDIDATE"
        )

    report["decision"] = (
        "UNVERIFIED_NEEDS_INDEPENDENT_GEOMETRIC_VALIDATION"
    )

    OUTPUT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== CANDIDATE RESULTS ===")

    for item in report["candidates"]:
        dev = item["development"]
        held = item["heldout"]

        print(
            f"Candidate {item['rank']}: "
            f"dev_adjusted="
            f"{dev['coverage_adjusted_score']} "
            f"heldout_adjusted="
            f"{held['coverage_adjusted_score']}"
        )

    print(
        "\nDevelopment-selected candidate:",
        report["development_selected_candidate"],
    )

    if selected is not None:
        print("\nHeld-out paired controls:")

        for item in report[
            "selected_candidate_controls"
        ]["heldout"]:
            print(
                f"  {item['control_name']}: "
                f"{item['status']} | "
                f"paired={item['paired_valid_frames']} | "
                f"advantage={item['mean_score_advantage']} | "
                f"wins={item['frame_win_fraction']}"
            )

    print(
        "\nHeld-out control status:",
        report["heldout_control_status"],
    )

    print("Registration verified: NO")
    print("Report:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
