
"""
Under-construction — Step 99
Independent image-line and competing-hypothesis diagnostic.

Inputs:
  outputs/line_ambiguity/resolved_lines.json
  outputs/line_temporal_tracking/per_candidate_tracking.json
  camera manifest and frames

Outputs:
  outputs/line_identity_challenge/challenge_report.json
  outputs/line_identity_challenge/per_candidate_challenge.json

No CAD registration, geometry optimization, physical-line
verification, or construction-progress inference.

Run:
  .\\.venv\\Scripts\\python.exe tools\\challenge_line_identity.py
"""

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

LINE_PATH = ROOT / "outputs/line_ambiguity/resolved_lines.json"
TRACK_PATH = (
    ROOT / "outputs/line_temporal_tracking/"
    "per_candidate_tracking.json"
)

CAM_ROOT = (
    ROOT
    / "_archive/hilti_transition_20261023_004237"
    / "data/hilti_frames/floor_2_2025-10-28_run_2/cam0"
)

MANIFEST = CAM_ROOT / "frame_manifest.csv"
FRAMES = CAM_ROOT / "frames"

OUT = ROOT / "outputs/line_identity_challenge"

TARGET_IDS = (159, 108, 114, 2, 47, 61, 88, 122, 161, 188)

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

cv2.setNumThreads(1)
cv2.setRNGSeed(99)

MAX_SEEDS = 12
MAX_TRIPLETS = 8
MIN_SEGMENT_LENGTH = 25.0
MAX_SEGMENT_DISTANCE = 10.0
MAX_ANGLE_DEG = 12.0
MIN_TRACK_POINTS = 5
MAX_FB_ERROR = 1.5
MAX_PROJECTION_ERROR = 8.0
MIN_COMPETITOR_ADVANTAGE = 3.0

LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(
        cv2.TERM_CRITERIA_EPS
        | cv2.TERM_CRITERIA_COUNT,
        30,
        0.01,
    ),
)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_lines():
    raw = read_json(LINE_PATH)
    lines = {}

    for item in raw["selected_lines"]:
        points = np.asarray(
            item["endpoints_map"],
            dtype=np.float64,
        )
        if points.shape != (2, 3):
            continue
        if not np.all(np.isfinite(points)):
            continue

        cid = int(item["candidate_id"])
        lines[cid] = {
            "id": cid,
            "group_id": item.get("group_id"),
            "points": points,
        }

    return lines


def load_manifest():
    poses = []

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            try:
                p = np.array(
                    [float(row[k]) for k in ("tx", "ty", "tz")]
                )
                q = np.array(
                    [
                        float(row[k])
                        for k in ("qx", "qy", "qz", "qw")
                    ]
                )
                if not (
                    np.all(np.isfinite(p))
                    and np.all(np.isfinite(q))
                ):
                    continue

                poses.append({
                    "frame": Path(row["filename"]).name,
                    "p": p,
                    "R": Rotation.from_quat(q).as_matrix(),
                })
            except (ValueError, KeyError):
                continue

    if not poses:
        raise RuntimeError("No usable manifest poses")

    return poses


def number(name):
    try:
        return int(Path(name).stem.split("_")[-1])
    except ValueError:
        return None


def project(points, pose, shape):
    h, w = shape

    t = np.linspace(0, 1, 64)[:, None]
    world = points[0] + t * (points[1] - points[0])

    camera = (world - pose["p"]) @ pose["R"]

    # Conservative visibility rule: no behind-camera samples.
    if np.any(camera[:, 2] <= 0.2):
        return None

    pixels, _ = cv2.fisheye.projectPoints(
        camera.reshape(-1, 1, 3),
        np.zeros((3, 1)),
        np.zeros((3, 1)),
        K,
        D,
    )
    pixels = pixels.reshape(-1, 2)

    visible = (
        (pixels[:, 0] >= 8)
        & (pixels[:, 0] < w - 8)
        & (pixels[:, 1] >= 8)
        & (pixels[:, 1] < h - 8)
    )

    pixels = pixels[visible]

    if len(pixels) < 3:
        return None

    length = np.linalg.norm(
        np.diff(pixels, axis=0), axis=1
    ).sum()

    return pixels if length >= 20 else None


def distances(points, curve):
    if curve is None or len(curve) < 2:
        return np.full(len(points), np.inf)

    a = curve[:-1]
    v = np.diff(curve, axis=0)

    denom = np.maximum(
        np.sum(v * v, axis=1), 1e-12
    )

    rel = points[:, None, :] - a[None, :, :]

    t = np.clip(
        np.sum(rel * v[None, :, :], axis=2)
        / denom[None, :],
        0,
        1,
    )

    nearest = (
        a[None, :, :]
        + t[:, :, None] * v[None, :, :]
    )

    return np.linalg.norm(
        points[:, None, :] - nearest,
        axis=2,
    ).min(axis=1)


def detect_segments(gray):
    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )
    detected = detector.detect(gray)[0]

    if detected is None:
        return []

    result = []

    for raw in detected.reshape(-1, 4):
        a = raw[:2].astype(np.float64)
        b = raw[2:].astype(np.float64)
        length = float(np.linalg.norm(b - a))

        if length < MIN_SEGMENT_LENGTH:
            continue

        result.append((a, b, length))

    return result


def segment_matches(curve, segments):
    if curve is None:
        return []

    direction = curve[-1] - curve[0]
    direction /= max(np.linalg.norm(direction), 1e-9)

    matches = []

    for a, b, length in segments:
        axis = (b - a) / length

        angle = np.degrees(
            np.arccos(
                np.clip(abs(axis @ direction), 0, 1)
            )
        )

        if angle > MAX_ANGLE_DEG:
            continue

        samples = np.linspace(0, 1, 7)[:, None]
        pts = a + samples * (b - a)

        distance = float(
            np.median(distances(pts, curve))
        )

        if distance <= MAX_SEGMENT_DISTANCE:
            matches.append({
                "a": a,
                "b": b,
                "length": length,
                "distance": distance,
                "angle": float(angle),
            })

    return sorted(
        matches,
        key=lambda x: (x["distance"], -x["length"]),
    )


def independent_points(matches, gray):
    # Features originate from globally detected LSD segments,
    # not from a mask drawn around the predicted 3D line.
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for item in matches:
        a = np.rint(item["a"]).astype(np.int32)
        b = np.rint(item["b"]).astype(np.int32)
        cv2.line(mask, tuple(a), tuple(b), 255, 7)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=150,
        qualityLevel=0.01,
        minDistance=6,
        mask=mask,
        blockSize=5,
    )

    if corners is None:
        return np.empty((0, 2), dtype=np.float32)

    return corners.reshape(-1, 2).astype(np.float32)


def track_three(images, points):
    p = points.reshape(-1, 1, 2)
    valid = np.ones(len(points), dtype=bool)

    for a, b in zip(images[:-1], images[1:]):
        nxt, s1, _ = cv2.calcOpticalFlowPyrLK(
            a, b, p, None, **LK
        )
        if nxt is None:
            return np.empty((0, 2))

        back, s2, _ = cv2.calcOpticalFlowPyrLK(
            b, a, nxt, None, **LK
        )
        if back is None:
            return np.empty((0, 2))

        fb = np.linalg.norm(
            back.reshape(-1, 2) - p.reshape(-1, 2),
            axis=1,
        )

        valid &= (
            (s1.reshape(-1) == 1)
            & (s2.reshape(-1) == 1)
            & np.isfinite(fb)
            & (fb <= MAX_FB_ERROR)
            & np.all(
                np.isfinite(nxt.reshape(-1, 2)),
                axis=1,
            )
        )

        p = nxt

    return p.reshape(-1, 2)[valid]


def triplets_for(seed_names, poses):
    index = {
        pose["frame"]: i
        for i, pose in enumerate(poses)
    }

    result = []
    seen = set()

    for name in seed_names:
        i = index.get(name)
        if i is None or i < 0 or i + 2 >= len(poses):
            continue

        triple = poses[i:i + 3]
        nums = [number(x["frame"]) for x in triple]

        if any(x is None for x in nums):
            continue

        if not (
            nums[1] - nums[0] == 1
            and nums[2] - nums[1] == 1
        ):
            continue

        key = tuple(x["frame"] for x in triple)

        if key not in seen:
            seen.add(key)
            result.append(triple)

        if len(result) >= MAX_TRIPLETS:
            break

    return result


def load_image(name, cache):
    if name not in cache:
        cache[name] = cv2.imread(
            str(FRAMES / name),
            cv2.IMREAD_GRAYSCALE,
        )
    return cache[name]


def evaluate_triplet(cid, triple, lines, cache):
    images = [
        load_image(pose["frame"], cache)
        for pose in triple
    ]

    if any(im is None for im in images):
        return {"status": "IMAGE_UNAVAILABLE"}

    if len({im.shape for im in images}) != 1:
        return {"status": "IMAGE_SHAPE_MISMATCH"}

    shape = images[0].shape

    projections = [
        project(lines[cid]["points"], pose, shape)
        for pose in triple
    ]

    if any(p is None for p in projections):
        return {"status": "PROJECTION_UNAVAILABLE"}

    # Global LSD detection is performed before choosing evidence
    # relevant to any particular hypothesis.
    segments = detect_segments(images[0])
    matches = segment_matches(projections[0], segments)

    if not matches:
        return {
            "status": "NO_INDEPENDENT_SEGMENT",
            "global_segment_count": len(segments),
        }

    points = independent_points(matches, images[0])

    if len(points) < MIN_TRACK_POINTS:
        return {
            "status": "INSUFFICIENT_INDEPENDENT_FEATURES",
            "feature_count": len(points),
            "matching_segment_count": len(matches),
        }

    tracked = track_three(images, points)

    if len(tracked) < MIN_TRACK_POINTS:
        return {
            "status": "THREE_FRAME_TRACK_FAILED",
            "feature_count": len(points),
            "surviving_count": len(tracked),
        }

    real_error = float(
        np.median(distances(tracked, projections[2]))
    )

    competitors = []

    for other_id, line in lines.items():
        if other_id == cid:
            continue

        other_curve = project(
            line["points"], triple[2], shape
        )

        if other_curve is None:
            continue

        error = float(
            np.median(distances(tracked, other_curve))
        )

        competitors.append({
            "candidate_id": other_id,
            "error_px": error,
        })

    competitors.sort(key=lambda x: x["error_px"])

    best = competitors[0] if competitors else None

    if best is None:
        status = "NO_VISIBLE_COMPETITOR"

    elif (
        best["error_px"] + MIN_COMPETITOR_ADVANTAGE
        < real_error
    ):
        status = "COMPETITOR_FITS_BETTER"

    elif (
        real_error <= MAX_PROJECTION_ERROR
        and real_error + MIN_COMPETITOR_ADVANTAGE
        < best["error_px"]
    ):
        status = "TARGET_FITS_BETTER"

    else:
        status = "IDENTITY_AMBIGUOUS"

    return {
        "status": status,
        "global_segment_count": len(segments),
        "matching_segment_count": len(matches),
        "independent_feature_count": len(points),
        "surviving_three_frame_tracks": len(tracked),
        "target_median_error_px": real_error,
        "best_competitor": best,
        "closest_competitors": competitors[:5],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    lines = load_lines()
    poses = load_manifest()
    previous = read_json(TRACK_PATH)["results"]

    previous_by_id = {
        int(item["candidate_id"]): item
        for item in previous
    }

    missing = [
        cid for cid in TARGET_IDS
        if cid not in lines or cid not in previous_by_id
    ]

    if missing:
        raise RuntimeError(
            f"Required frozen candidates unavailable: {missing}"
        )

    image_cache = {}
    results = []

    for cid in TARGET_IDS:
        old = previous_by_id[cid]

        # Preserve the Step 98B selection history, but do not
        # use its tracked points or its success decisions
        # to construct independent image evidence.
        seeds = list(dict.fromkeys(
            row["frame_a"]
            for row in old["pairs"]
        ))

        triples = triplets_for(seeds, poses)
        evaluations = []

        for triple in triples:
            outcome = evaluate_triplet(
                cid, triple, lines, image_cache
            )

            outcome["frames"] = [
                pose["frame"] for pose in triple
            ]
            evaluations.append(outcome)

        counts = Counter(
            x["status"] for x in evaluations
        )

        target_wins = counts["TARGET_FITS_BETTER"]
        competitor_wins = counts["COMPETITOR_FITS_BETTER"]
        ambiguous = counts["IDENTITY_AMBIGUOUS"]

        compared = (
            target_wins + competitor_wins + ambiguous
        )

        if compared == 0:
            decision = "INSUFFICIENT_COMPARATIVE_EVIDENCE"
        elif competitor_wins > target_wins:
            decision = "COMPETING_HYPOTHESIS_PREFERRED"
        elif target_wins >= 2 and competitor_wins == 0:
            decision = "LOCAL_COMPARATIVE_SUPPORT"
        else:
            decision = "IDENTITY_UNRESOLVED"

        row = {
            "candidate_id": cid,
            "group_id": lines[cid]["group_id"],
            "step96_status_unchanged": old["step96_status"],
            "step98b_decision_unchanged": old["decision"],
            "triplets_selected": len(triples),
            "triplet_status_counts": dict(counts),
            "comparisons_completed": compared,
            "target_wins": target_wins,
            "competitor_wins": competitor_wins,
            "ambiguous_comparisons": ambiguous,
            "decision": decision,
            "physical_line_verified": False,
            "triplets": evaluations,
        }

        results.append(row)

        print(
            f"{cid:03d}: triplets={len(triples)} "
            f"compared={compared} "
            f"target={target_wins} "
            f"competitor={competitor_wins} "
            f"=> {decision}",
            flush=True,
        )

    write_json(
        OUT / "per_candidate_challenge.json",
        {
            "method": "step99_independent_lsd_three_frame_challenge",
            "results": results,
        },
    )

    decisions = Counter(x["decision"] for x in results)
    statuses = Counter()

    for row in results:
        statuses.update(row["triplet_status_counts"])

    report = {
        "method": "step99_independent_lsd_three_frame_challenge",
        "target_candidate_ids": list(TARGET_IDS),
        "candidate_count": len(results),
        "triplets_selected": sum(
            x["triplets_selected"] for x in results
        ),
        "comparisons_completed": sum(
            x["comparisons_completed"] for x in results
        ),
        "triplet_status_counts": dict(statuses),
        "candidate_decision_counts": dict(decisions),
        "local_support_candidate_ids": [
            x["candidate_id"] for x in results
            if x["decision"] == "LOCAL_COMPARATIVE_SUPPORT"
        ],
        "competing_hypothesis_preferred_ids": [
            x["candidate_id"] for x in results
            if x["decision"] == "COMPETING_HYPOTHESIS_PREFERRED"
        ],
        "step96_decisions_modified": False,
        "step98b_decisions_modified": False,
        "physical_lines_verified": False,
        "structural_identity_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "LSD segments are detected globally but associated with hypotheses using projected proximity.",
            "LSD segments may represent texture, shadows, or unrelated physical edges.",
            "Feature tracks are not guaranteed to belong to one physical line.",
            "The competing set contains only the frozen selected 3D line hypotheses.",
            "Near-duplicate hypotheses can make identity ambiguous.",
            "Three-frame tracks are short and temporally correlated.",
            "The experiment reuses frames and geometry from earlier development and is not independent held-out validation.",
            "Camera poses and fisheye calibration are treated as given.",
            "No geometry refinement or CAD information is used.",
        ],
        "decision": "COMPARATIVE_DIAGNOSTICS_AVAILABLE",
    }

    write_json(
        OUT / "challenge_report.json",
        report,
    )

    print("\n=== STEP 99 ===")
    print("Triplets:", report["triplets_selected"])
    print("Comparisons:", report["comparisons_completed"])
    print("Decisions:", dict(decisions))
    print("Statuses:", dict(statuses))
    print("Physical identity verified: NO")
    print("CAD registration verified: NO")
    print("Construction progress verified: NO")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
