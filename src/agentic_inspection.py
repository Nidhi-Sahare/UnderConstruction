
"""
BuildProgress - SiteWitness Adaptive Visual Inspection
======================================================

Real footage -> quality assessment -> adaptive recovery
-> evidence-aware selection -> baseline comparison -> report.

Run:
    python -m src.agentic_inspection

Outputs:
    outputs/agentic_inspection/floor_2_inspection_report.json
    outputs/agentic_inspection/evidence_gallery.jpg

Requirements:
    OpenCV, NumPy

Important:
    Image quality and visual novelty are NOT proof of
    construction compliance or CAD registration.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_archive" / "hilti_transition_20261023_004237"
REGISTRATION = ARCHIVE / "outputs" / "registration"
FRAMES = ROOT / "sample_gallery" / "real_floor_2" / "frames"
OUTPUT = ROOT / "outputs" / "agentic_inspection"

INITIAL_SAMPLES = 12
MAX_ADDITIONAL_INSPECTIONS = 24
NEIGHBOR_OFFSETS = (-4, -2, -1, 1, 2, 4)
MAX_REVIEW_FRAMES = 6
WORK_WIDTH = 720

# Evidence-selection weights. These are heuristics,
# not calibrated construction-verification probabilities.
QUALITY_WEIGHT = 0.50
NOVELTY_WEIGHT = 0.30
TEMPORAL_WEIGHT = 0.20

ORB_REDUNDANCY_THRESHOLD = 0.30


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else -1


def discover_frames() -> list[Path]:
    if not FRAMES.is_dir():
        raise FileNotFoundError(
            f"Walkthrough frames not found: {FRAMES}"
        )

    frames = [
        p for p in FRAMES.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    frames.sort(key=lambda p: (frame_number(p), p.name))

    if not frames:
        raise RuntimeError("No Floor 2 walkthrough frames found")

    return frames


def registration_audit() -> dict:
    """
    Preserve existing geometric registration diagnostics.

    Never promote a geometric ranking to visually verified.
    """
    candidate_path = REGISTRATION / "candidate_alignments.json"
    wall_path = REGISTRATION / "wall_consistency.json"

    if not candidate_path.is_file() or not wall_path.is_file():
        return {
            "status": "UNVERIFIED",
            "preferred_candidate": None,
            "ranked_candidates": [],
            "reason": "Registration diagnostics unavailable",
        }

    raw = load_json(candidate_path)
    wall = load_json(wall_path)

    ranked = []

    for index, candidate in enumerate(
        raw.get("candidates", []), 1
    ):
        matches = [
            result
            for result in wall.get("candidates", [])
            if math.isclose(
                float(result["angle_degrees"]),
                float(candidate["angle_degrees"]),
                abs_tol=0.001,
            )
            and np.allclose(
                result["translation"],
                candidate["translation"],
                atol=0.001,
            )
        ]

        if not matches:
            continue

        result = matches[0]

        ranked.append({
            "original_candidate_number": index,
            "angle_degrees": candidate["angle_degrees"],
            "translation": candidate["translation"],
            "inside_fraction": candidate.get(
                "inside_fraction"
            ),
            "wall_crossings": result["wall_crossings"],
            "crossing_path_edges": result[
                "crossing_path_edges"
            ],
            "near_wall_fraction": result[
                "near_wall_fraction"
            ],
        })

    ranked.sort(
        key=lambda x: (
            x["crossing_path_edges"],
            x["wall_crossings"],
            x["near_wall_fraction"],
        )
    )

    return {
        "status": "UNVERIFIED",
        "ranking_basis": (
            "Geometric wall-consistency diagnostics only"
        ),
        "preferred_candidate": ranked[0] if ranked else None,
        "ranked_candidates": ranked,
    }


def prepare_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    if w > WORK_WIDTH:
        new_h = max(1, round(h * WORK_WIDTH / w))

        gray = cv2.resize(
            gray,
            (WORK_WIDTH, new_h),
            interpolation=cv2.INTER_AREA,
        )

    return gray


def inspect(
    path: Path,
) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    image = cv2.imread(str(path))
    relative = path.relative_to(ROOT).as_posix()

    if image is None:
        return ({
            "frame": relative,
            "readable": False,
            "quality": "UNREADABLE",
            "quality_reasons": ["IMAGE_DECODE_FAILED"],
            "screening_score": 0.0,
        }, None, None)

    gray = prepare_gray(image)

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )
    brightness = float(gray.mean())

    orb = cv2.ORB_create(nfeatures=1600)
    keypoints, descriptors = orb.detectAndCompute(
        gray, None
    )

    keypoints = keypoints or []

    h, w = gray.shape
    occupied = set()

    for kp in keypoints:
        col = min(5, int(kp.pt[0] * 6 / w))
        row = min(5, int(kp.pt[1] * 6 / h))
        occupied.add((row, col))

    coverage = len(occupied) / 36.0
    feature_count = len(keypoints)

    reasons = []

    if sharpness < 45:
        reasons.append("LOW_SHARPNESS")

    if brightness < 35:
        reasons.append("LOW_LIGHT")

    if brightness > 225:
        reasons.append("OVEREXPOSED")

    if feature_count < 80:
        reasons.append("FEW_FEATURES")

    if coverage < 0.25:
        reasons.append("POOR_FEATURE_DISTRIBUTION")

    sharpness_score = min(sharpness / 300, 1.0)

    brightness_score = max(
        0.0,
        1.0 - abs(brightness - 120) / 120,
    )

    feature_score = min(
        feature_count / 800, 1.0
    )

    score = (
        0.30 * sharpness_score
        + 0.20 * brightness_score
        + 0.20 * feature_score
        + 0.30 * coverage
    )

    result = {
        "frame": relative,
        "readable": True,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "sharpness": round(sharpness, 3),
        "mean_brightness": round(brightness, 3),
        "orb_feature_count": feature_count,
        "feature_grid_coverage": round(
            coverage, 4
        ),
        "quality": (
            "USABLE" if not reasons
            else "REVIEW_REQUIRED"
        ),
        "quality_reasons": reasons,
        "screening_score": round(score, 4),
    }

    return result, gray, descriptors


def feature_overlap(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
) -> dict:
    if descriptors_a is None or descriptors_b is None:
        return {
            "good_matches": 0,
            "overlap_ratio": None,
        }

    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return {
            "good_matches": 0,
            "overlap_ratio": None,
        }

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    pairs = matcher.knnMatch(
        descriptors_a,
        descriptors_b,
        k=2,
    )

    good = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.75 * second.distance
    ]

    ratio = len(good) / max(
        1,
        min(
            len(descriptors_a),
            len(descriptors_b),
        ),
    )

    return {
        "good_matches": len(good),
        "overlap_ratio": round(ratio, 4),
    }


def novelty_score(
    index: int,
    selected: list[int],
    cache: dict,
) -> tuple[float, float | None]:
    if not selected:
        return 1.0, None

    descriptor = cache[index][2]

    overlaps = [
        feature_overlap(
            descriptor,
            cache[other][2],
        )["overlap_ratio"]
        for other in selected
    ]

    valid = [
        value
        for value in overlaps
        if value is not None
    ]

    if not valid:
        return 0.0, None

    maximum = max(valid)

    return max(
        0.0,
        1.0 - maximum,
    ), maximum


def temporal_score(
    index: int,
    selected: list[int],
    total: int,
) -> float:
    if not selected:
        return 1.0

    distance = min(
        abs(index - other)
        for other in selected
    )

    # Distances of approximately 1/4 of the walkthrough
    # or more receive full temporal-coverage credit.
    scale = max(1.0, total / 4.0)

    return min(distance / scale, 1.0)


def select_evidence(
    candidate_indices: list[int],
    cache: dict,
    total_frames: int,
    limit: int = MAX_REVIEW_FRAMES,
) -> tuple[list[int], list[dict]]:
    """
    Greedy evidence-aware selection.

    Scores are recalculated after each selection because
    novelty and temporal coverage depend on the selected set.
    """
    remaining = set(candidate_indices)
    selected = []
    decisions = []

    while remaining and len(selected) < limit:
        scored = []

        for index in sorted(remaining):
            observation = cache[index][0]

            if observation["quality"] != "USABLE":
                continue

            quality = observation["screening_score"]

            novelty, overlap = novelty_score(
                index, selected, cache
            )

            temporal = temporal_score(
                index, selected, total_frames
            )

            utility = (
                QUALITY_WEIGHT * quality
                + NOVELTY_WEIGHT * novelty
                + TEMPORAL_WEIGHT * temporal
            )

            # Avoid treating high-overlap imagery as novel.
            if (
                overlap is not None
                and overlap >= ORB_REDUNDANCY_THRESHOLD
            ):
                utility *= 0.5

            scored.append((
                utility,
                index,
                quality,
                novelty,
                temporal,
                overlap,
            ))

        if not scored:
            break

        (
            utility,
            best,
            quality,
            novelty,
            temporal,
            overlap,
        ) = max(scored, key=lambda item: (
            item[0], -item[1]
        ))

        selected.append(best)
        remaining.remove(best)

        decisions.append({
            "frame": f"cam0_{frame_number_from_index(best, cache):05d}.jpg",
            "frame_index": best,
            "decision": "SELECT_FOR_EVIDENCE_REVIEW",
            "selection_rank": len(selected),
            "quality_score": round(quality, 4),
            "novelty_score": round(novelty, 4),
            "temporal_score": round(temporal, 4),
            "maximum_orb_overlap": overlap,
            "combined_utility": round(utility, 4),
        })

    for index in sorted(remaining):
        decisions.append({
            "frame": cache[index][0]["frame"],
            "frame_index": index,
            "decision": "NOT_SELECTED_WITHIN_REVIEW_BUDGET",
        })

    return selected, decisions


def frame_number_from_index(
    index: int,
    cache: dict,
) -> int:
    return frame_number(
        Path(cache[index][0]["frame"])
    )


def evaluate_selection(
    indices: list[int],
    cache: dict,
    total_frames: int,
) -> dict:
    if not indices:
        return {
            "selected_count": 0,
            "mean_quality": None,
            "mean_pairwise_overlap": None,
            "temporal_span_fraction": None,
            "usable_fraction": None,
        }

    observations = [
        cache[index][0]
        for index in indices
    ]

    qualities = [
        obs["screening_score"]
        for obs in observations
    ]

    overlaps = []

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            result = feature_overlap(
                cache[indices[i]][2],
                cache[indices[j]][2],
            )

            if result["overlap_ratio"] is not None:
                overlaps.append(
                    result["overlap_ratio"]
                )

    span = (
        max(indices) - min(indices)
    ) / max(1, total_frames - 1)

    usable = sum(
        obs["quality"] == "USABLE"
        for obs in observations
    )

    return {
        "selected_count": len(indices),
        "mean_quality": round(
            float(np.mean(qualities)), 4
        ),
        "mean_pairwise_overlap": (
            round(float(np.mean(overlaps)), 4)
            if overlaps else None
        ),
        "temporal_span_fraction": round(
            span, 4
        ),
        "usable_fraction": round(
            usable / len(indices), 4
        ),
    }


def fixed_interval_selection(
    total: int,
    count: int,
) -> list[int]:
    if count <= 0:
        return []

    return sorted(set(
        np.linspace(
            0,
            total - 1,
            min(count, total),
            dtype=int,
        ).tolist()
    ))


def make_gallery(
    frames: list[Path],
    selected: list[int],
    rejected: list[int],
    cache: dict,
) -> Path:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tiles = []

    entries = (
        [(i, "SELECTED") for i in selected]
        + [(i, "QUALITY REJECTED") for i in rejected]
    )

    if not entries:
        entries = [
            (0, "NO EVIDENCE SELECTED")
        ]

    tile_w = 480
    tile_h = 400
    image_h = 330

    for index, label in entries:
        image = cv2.imread(str(frames[index]))

        if image is None:
            image = np.zeros(
                (image_h, tile_w, 3),
                dtype=np.uint8,
            )

        h, w = image.shape[:2]

        scale = min(
            tile_w / w,
            image_h / h,
        )

        resized = cv2.resize(
            image,
            (
                max(1, int(w * scale)),
                max(1, int(h * scale)),
            ),
        )

        tile = np.full(
            (tile_h, tile_w, 3),
            25,
            dtype=np.uint8,
        )

        y = (image_h - resized.shape[0]) // 2
        x = (tile_w - resized.shape[1]) // 2

        tile[
            y:y + resized.shape[0],
            x:x + resized.shape[1],
        ] = resized

        observation = cache[index][0]

        color = (
            (75, 200, 110)
            if label == "SELECTED"
            else (60, 150, 255)
        )

        cv2.putText(
            tile,
            f"{label} | Frame {frame_number(frames[index]):03d}",
            (12, 352),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            tile,
            f"Quality: {observation['screening_score']:.3f}",
            (12, 380),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

        tiles.append(tile)

    columns = 3
    rows = math.ceil(len(tiles) / columns)

    gallery = np.full(
        (rows * tile_h, columns * tile_w, 3),
        25,
        dtype=np.uint8,
    )

    for index, tile in enumerate(tiles):
        row = index // columns
        col = index % columns

        gallery[
            row * tile_h:(row + 1) * tile_h,
            col * tile_w:(col + 1) * tile_w,
        ] = tile

    destination = OUTPUT / "evidence_gallery.jpg"

    if not cv2.imwrite(str(destination), gallery):
        raise RuntimeError("Could not save evidence gallery")

    return destination


def run_inspection() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    frames = discover_frames()
    registration = registration_audit()

    initial_indices = sorted(set(
        np.linspace(
            0,
            len(frames) - 1,
            min(INITIAL_SAMPLES, len(frames)),
            dtype=int,
        ).tolist()
    ))

    cache = {}
    trace = []
    candidate_indices = []
    rejected_indices = []
    additional_count = 0

    def perceive(index: int, reason: str):
        if index not in cache:
            cache[index] = inspect(frames[index])

        observation = cache[index][0]

        trace.append({
            "step": len(trace) + 1,
            "frame_index": index,
            "trigger": reason,
            "perception": observation,
        })

        return observation

    # Phase A: initial perception and adaptive recovery.
    for index in initial_indices:
        observation = perceive(
            index,
            "INITIAL_SAMPLE",
        )

        if observation["quality"] == "USABLE":
            trace[-1]["decision"] = (
                "ACCEPT_FOR_DIVERSITY_SCREENING"
            )
            trace[-1]["action"] = (
                "SCREEN_FOR_DUPLICATES"
            )

            candidate_indices.append(index)
            continue

        trace[-1]["decision"] = (
            "INSUFFICIENT_VISUAL_QUALITY"
        )
        trace[-1]["action"] = (
            "INSPECT_NEIGHBORING_FRAMES"
        )

        rejected_indices.append(index)
        alternatives = []

        for offset in NEIGHBOR_OFFSETS:
            neighbor = index + offset

            if not 0 <= neighbor < len(frames):
                continue

            if (
                neighbor not in cache
                and additional_count
                >= MAX_ADDITIONAL_INSPECTIONS
            ):
                break

            was_cached = neighbor in cache

            next_observation = perceive(
                neighbor,
                f"ADAPTIVE_RECOVERY_FROM_{index}",
            )

            if not was_cached:
                additional_count += 1

            if next_observation["quality"] == "USABLE":
                alternatives.append((
                    next_observation["screening_score"],
                    neighbor,
                ))

                trace[-1]["decision"] = (
                    "RECOVERY_CANDIDATE"
                )
                trace[-1]["action"] = (
                    "COMPARE_ALTERNATIVES"
                )
            else:
                trace[-1]["decision"] = (
                    "REJECT_LOW_QUALITY"
                )
                trace[-1]["action"] = (
                    "CONTINUE_NEIGHBOR_SEARCH"
                )

                rejected_indices.append(neighbor)

        if alternatives:
            _, best = max(
                alternatives,
                key=lambda item: (
                    item[0], -item[1]
                ),
            )

            candidate_indices.append(best)

            trace.append({
                "step": len(trace) + 1,
                "trigger": f"RECOVERY_FROM_{index}",
                "decision": "REPLACE_UNUSABLE_FRAME",
                "action": "SELECT_NEIGHBOR",
                "rejected_frame": frames[index].name,
                "replacement_frame": frames[best].name,
                "replacement_index": best,
            })
        else:
            trace.append({
                "step": len(trace) + 1,
                "trigger": f"RECOVERY_FROM_{index}",
                "decision": "NO_USABLE_NEIGHBOR",
                "action": "REQUEST_BETTER_VISUAL_EVIDENCE",
            })

    candidate_indices = sorted(set(candidate_indices))

    # Phase B: evidence-aware selection.
    review_indices, selection_decisions = select_evidence(
        candidate_indices,
        cache,
        len(frames),
    )

    # Phase C: fixed-interval baseline with the same final
    # review-frame budget. Baseline frames must also be inspected.
    fixed_indices = fixed_interval_selection(
        len(frames),
        len(review_indices),
    )

    for index in fixed_indices:
        if index not in cache:
            cache[index] = inspect(frames[index])

    adaptive_metrics = evaluate_selection(
        review_indices,
        cache,
        len(frames),
    )

    fixed_metrics = evaluate_selection(
        fixed_indices,
        cache,
        len(frames),
    )

    # Phase D: evidence gallery.
    gallery_path = make_gallery(
        frames,
        review_indices,
        sorted(set(rejected_indices)),
        cache,
    )

    preferred = registration["preferred_candidate"]

    if not review_indices:
        next_action = {
            "type": "REQUEST_NEW_WALKTHROUGH_FRAMES",
            "reason": "No usable review frames selected",
        }
    else:
        next_action = {
            "type": "REQUEST_ARCHITECTURAL_EVIDENCE_REVIEW",
            "candidate_to_test": (
                preferred["original_candidate_number"]
                if preferred else None
            ),
            "suggested_frames": [
                frames[i].relative_to(ROOT).as_posix()
                for i in review_indices
            ],
            "reason": (
                "Frames are suitable for visual review. "
                "CAD correspondence and construction "
                "compliance remain unverified."
            ),
        }

    report = {
        "schema_version": "3.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "project": "BuildProgress",
        "agent": "SiteWitness",
        "floor": 2,
        "opencv_version": cv2.__version__,
        "status": "EVIDENCE_SELECTION_COMPLETED",
        "construction_compliance_verified": False,
        "cad_registration_verified": False,
        "frame_count_found": len(frames),
        "initial_sample_count": len(initial_indices),
        "additional_inspection_count": additional_count,
        "unique_frames_inspected": len(cache),
        "usable_frames_inspected": sum(
            item[0]["quality"] == "USABLE"
            for item in cache.values()
        ),
        "selected_review_frames": [
            frames[i].relative_to(ROOT).as_posix()
            for i in review_indices
        ],
        "registration": registration,
        "agent_trace": trace,
        "evidence_selection_decisions": selection_decisions,
        "baseline_comparison": {
            "comparison_type": (
                "Equal final review-frame count; "
                "not equal perception-compute budget"
            ),
            "adaptive": adaptive_metrics,
            "fixed_interval": fixed_metrics,
        },
        "evidence_gallery": (
            gallery_path.relative_to(ROOT).as_posix()
        ),
        "next_action": next_action,
        "limitations": [
            "Quality scores are heuristic, not calibrated probabilities.",
            "ORB matching does not establish CAD registration.",
            "Low ORB overlap does not guarantee scene novelty.",
            "Temporal coverage does not establish spatial coverage.",
            "The baseline uses equal output size, not equal compute.",
            "Image quality does not prove construction compliance.",
            "Construction claims require independent verification.",
        ],
    }

    destination = (
        OUTPUT / "floor_2_inspection_report.json"
    )

    destination.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nBUILDPROGRESS - SITEWITNESS")
    print("=" * 54)
    print("OpenCV:", cv2.__version__)
    print("Available frames:", len(frames))
    print("Initial samples:", len(initial_indices))
    print("Adaptive additional inspections:", additional_count)
    print("Unique inspected frames:", len(cache))
    print("Selected evidence frames:", len(review_indices))

    print("\nAdaptive selection:")
    print(json.dumps(adaptive_metrics, indent=2))

    print("\nFixed-interval baseline:")
    print(json.dumps(fixed_metrics, indent=2))

    print("\nCAD registration: UNVERIFIED")
    print("Construction compliance: UNVERIFIED")
    print("Gallery:", gallery_path)
    print("Report:", destination)

    return report


if __name__ == "__main__":
    run_inspection()
