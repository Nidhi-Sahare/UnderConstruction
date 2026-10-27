
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

import ezdxf
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "automatic_registration_validation"
DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"
REPORT_PATH = OUTPUT_DIR / "validation_report.json"

SEQUENCE = "floor_2_2025-10-28_run_2"

# Screening parameters, not measured accuracy guarantees.
BOUNDARY_MARGIN_M = 0.75
NEAR_WALL_M = 0.20
MAX_NEAR_WALL_FRACTION = 0.35
MIN_INSIDE_FRACTION = 0.90
MIN_FRAMES = 30
MAX_STEP_M = 4.0
CONSENSUS_DISTANCE_M = 1.50


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def find_file(filename, preferred_parts=()):
    matches = [
        p for p in ROOT.rglob(filename)
        if all(part in str(p) for part in preferred_parts)
        and ".venv" not in p.parts
    ]

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {ROOT}"
        )

    matches.sort(
        key=lambda p: (
            "outputs" not in p.parts,
            len(p.parts),
            str(p),
        )
    )
    return matches[0]


def load_frames():
    path = find_file(
        "frame_manifest.csv",
        (SEQUENCE, "cam0"),
    )

    frames = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            matched = str(row.get("pose_matched", "")).lower()
            if matched not in ("true", "1", "yes"):
                continue

            x = finite(row.get("tx"))
            y = finite(row.get("ty"))

            if x is None or y is None:
                continue

            frames.append({
                "frame_index": int(
                    row.get("frame_index") or row_index
                ),
                "filename": row.get("filename", ""),
                "x": x,
                "y": y,
            })

    if len(frames) < MIN_FRAMES:
        raise RuntimeError(
            f"Only {len(frames)} usable camera poses found."
        )

    return frames, path


def load_candidates():
    path = find_file("registration_candidates.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    raw = (
        payload.get("candidates", [])
        if isinstance(payload, dict)
        else payload
    )

    candidates = []
    for index, item in enumerate(raw):
        angle = finite(
            item.get("angle_deg", item.get("rotation_deg"))
        )

        translation = item.get("translation")
        if isinstance(translation, (list, tuple)):
            tx = finite(translation[0])
            ty = finite(translation[1])
        else:
            tx = finite(item.get("tx"))
            ty = finite(item.get("ty"))

        if None in (angle, tx, ty):
            continue

        candidates.append({
            "rank": int(item.get("rank", index + 1)),
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
            "source_score": finite(item.get("score")),
        })

    if not candidates:
        raise RuntimeError(
            "Candidate file contains no usable transformations."
        )

    return candidates, path


def load_cad_geometry():
    if not DXF_PATH.exists():
        raise FileNotFoundError(DXF_PATH)

    document = ezdxf.readfile(DXF_PATH)
    msp = document.modelspace()

    boundary_lines = []
    wall_lines = []

    for entity in msp:
        layer = entity.dxf.layer

        wall_layers = {
            "015 Innenwände",
            "010 Aussenwände",
            "016 Trennwände",
        }

        facade_layers = {
            "011 Fassadensystem",
            "012 Vorhangfassade",
        }

        if layer not in wall_layers | facade_layers:
            continue

        segments = []

        if entity.dxftype() == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end
            segments.append(
                ((a.x, a.y), (b.x, b.y))
            )

        elif entity.dxftype() == "LWPOLYLINE":
            points = [
                (float(p[0]), float(p[1]))
                for p in entity.get_points("xy")
            ]

            segments.extend(zip(points, points[1:]))

            if entity.closed and len(points) > 2:
                segments.append((points[-1], points[0]))

        for a, b in segments:
            if math.dist(a, b) < 0.05:
                continue

            line = LineString((a, b))

            if layer in wall_layers:
                wall_lines.append(line)
            else:
                boundary_lines.append(line)

    if not wall_lines and not boundary_lines:
        raise RuntimeError(
            "No supported CAD wall or facade geometry found."
        )

    all_lines = wall_lines + boundary_lines

    # A conservative bounding envelope, not a verified
    # walkable-floor polygon.
    coords = [
        coordinate
        for line in all_lines
        for coordinate in line.coords
    ]

    xmin = min(p[0] for p in coords)
    ymin = min(p[1] for p in coords)
    xmax = max(p[0] for p in coords)
    ymax = max(p[1] for p in coords)

    envelope = Polygon([
        (xmin, ymin),
        (xmax, ymin),
        (xmax, ymax),
        (xmin, ymax),
    ]).buffer(BOUNDARY_MARGIN_M)

    wall_geometry = unary_union(wall_lines) if wall_lines else None

    return {
        "envelope": envelope,
        "walls": wall_geometry,
        "bounds": [xmin, ymin, xmax, ymax],
        "wall_segments": len(wall_lines),
        "facade_segments": len(boundary_lines),
        "envelope_kind": "CAD_AXIS_ALIGNED_BOUNDING_BOX",
    }


def transform_frame(frame, candidate):
    theta = math.radians(candidate["angle_deg"])
    c = math.cos(theta)
    s = math.sin(theta)

    x = frame["x"]
    y = frame["y"]

    return (
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    )


def evaluate_candidate(frames, candidate, cad):
    positions = [
        transform_frame(frame, candidate)
        for frame in frames
    ]

    inside = [
        cad["envelope"].covers(Point(x, y))
        for x, y in positions
    ]

    wall_distances = []
    for x, y in positions:
        if cad["walls"] is None:
            continue

        wall_distances.append(
            Point(x, y).distance(cad["walls"])
        )

    steps = [
        math.dist(a, b)
        for a, b in zip(positions, positions[1:])
    ]

    inside_fraction = sum(inside) / len(inside)

    near_wall_fraction = (
        sum(d < NEAR_WALL_M for d in wall_distances)
        / len(wall_distances)
        if wall_distances
        else None
    )

    maximum_step = max(steps, default=0.0)

    # Rigid transforms preserve trajectory step lengths.
    # This diagnostic therefore cannot discriminate
    # candidates that differ only by rotation/translation.
    step_screen_pass = maximum_step <= MAX_STEP_M

    geometry_pass = (
        inside_fraction >= MIN_INSIDE_FRACTION
        and (
            near_wall_fraction is None
            or near_wall_fraction <= MAX_NEAR_WALL_FRACTION
        )
    )

    return {
        **candidate,
        "frame_count": len(positions),
        "inside_bounding_envelope_fraction": round(
            inside_fraction, 5
        ),
        "near_wall_fraction": (
            round(near_wall_fraction, 5)
            if near_wall_fraction is not None
            else None
        ),
        "median_wall_distance_m": (
            round(median(wall_distances), 4)
            if wall_distances
            else None
        ),
        "max_consecutive_pose_step_m": round(
            maximum_step, 4
        ),
        "trajectory_step_screen_pass": step_screen_pass,
        "geometry_screen_pass": geometry_pass,
        "positions": positions,
    }


def consensus_analysis(frames, evaluations):
    per_frame = []

    for i, frame in enumerate(frames):
        points = [
            evaluation["positions"][i]
            for evaluation in evaluations
        ]

        distances = [
            math.dist(points[a], points[b])
            for a in range(len(points))
            for b in range(a + 1, len(points))
        ]

        spread = max(distances, default=0.0)

        per_frame.append({
            "frame_index": frame["frame_index"],
            "filename": frame["filename"],
            "max_candidate_separation_m": round(
                spread, 4
            ),
            "candidate_consensus": (
                spread <= CONSENSUS_DISTANCE_M
            ),
        })

    consensus_fraction = (
        sum(
            item["candidate_consensus"]
            for item in per_frame
        ) / len(per_frame)
    )

    return {
        "consensus_threshold_m": CONSENSUS_DISTANCE_M,
        "consensus_frame_fraction": round(
            consensus_fraction, 5
        ),
        "frames": per_frame,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frames, manifest_path = load_frames()
    candidates, candidate_path = load_candidates()
    cad = load_cad_geometry()

    evaluations = [
        evaluate_candidate(frames, candidate, cad)
        for candidate in candidates
    ]

    consensus = consensus_analysis(frames, evaluations)

    passing = [
        item
        for item in evaluations
        if item["geometry_screen_pass"]
    ]

    if not passing:
        decision = "NO_CANDIDATE_PASSES_GEOMETRY_SCREEN"
    elif len(passing) == 1:
        decision = "ONE_GEOMETRICALLY_PLAUSIBLE_CANDIDATE"
    else:
        decision = "MULTIPLE_GEOMETRICALLY_PLAUSIBLE_CANDIDATES"

    report = {
        "method": "DETERMINISTIC_CAD_GEOMETRY_SCREENING_V1",
        "registration_verified": False,
        "construction_progress_verified": False,
        "decision": decision,
        "selected_verified_candidate": None,
        "inputs": {
            "camera_manifest": str(
                manifest_path.relative_to(ROOT)
            ),
            "registration_candidates": str(
                candidate_path.relative_to(ROOT)
            ),
            "cad": str(DXF_PATH.relative_to(ROOT)),
        },
        "cad": {
            "bounds": cad["bounds"],
            "wall_segments": cad["wall_segments"],
            "facade_segments": cad["facade_segments"],
            "envelope_kind": cad["envelope_kind"],
        },
        "limitations": [
            "CAD bounding box includes regions outside the building.",
            "Wall distance uses source linework, not verified wall solids.",
            "Trajectory continuity is invariant under rigid transforms.",
            "Candidate score is not calibrated registration confidence.",
            "No visual landmark correspondences are verified.",
            "No construction completion is inferred.",
        ],
        "candidates": [
            {
                key: value
                for key, value in item.items()
                if key != "positions"
            }
            for item in evaluations
        ],
        "consensus": consensus,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n=== AUTOMATIC REGISTRATION VALIDATION ===")
    print("Camera frames:", len(frames))
    print("Candidates:", len(candidates))
    print("CAD bounds:", cad["bounds"])
    print()

    for item in evaluations:
        print(
            f"Candidate #{item['rank']}: "
            f"inside={item['inside_bounding_envelope_fraction']:.1%}, "
            f"near_wall={item['near_wall_fraction']}, "
            f"geometry_pass={item['geometry_screen_pass']}"
        )

    print()
    print("Decision:", decision)
    print(
        "Consensus frame fraction:",
        consensus["consensus_frame_fraction"],
    )
    print("Registration verified: NO")
    print("Construction progress verified: NO")
    print("Report:", REPORT_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
