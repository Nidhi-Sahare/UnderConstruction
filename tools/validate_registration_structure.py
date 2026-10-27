
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import ezdxf
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

MANIFEST_PATH = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frame_manifest.csv"
)

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

OUTPUT_DIR = ROOT / "outputs" / "registration_structure"
OUTPUT_PATH = OUTPUT_DIR / "structure_validation.json"

WALL_LAYERS = {
    "015 Innenwände",
    "010 Aussenwände",
    "016 Trennwände",
}

RAY_LENGTH_M = 12.0
RAY_COUNT = 36
NEAR_WALL_THRESHOLD_M = 0.15
SAMPLE_EVERY_N_FRAMES = 3


def finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def load_frames():
    frames = []

    with MANIFEST_PATH.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))

    for index, row in enumerate(rows):
        if str(
            row.get("pose_matched", "")
        ).strip().lower() not in {"true", "1", "yes"}:
            continue

        x = finite_float(row.get("tx"))
        y = finite_float(row.get("ty"))

        if x is None or y is None:
            continue

        frames.append({
            "frame_index": index,
            "x": x,
            "y": y,
        })

    return frames


def load_candidates():
    data = json.loads(
        CANDIDATES_PATH.read_text(encoding="utf-8")
    )

    raw = (
        data["candidates"]
        if isinstance(data, dict)
        else data
    )

    candidates = []

    for index, item in enumerate(raw):
        angle = finite_float(item.get("angle_deg"))
        tx = finite_float(item.get("tx"))
        ty = finite_float(item.get("ty"))

        if None in (angle, tx, ty):
            continue

        candidates.append({
            "rank": int(item.get("rank", index + 1)),
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
        })

    return candidates


def transform(x, y, candidate):
    theta = math.radians(candidate["angle_deg"])

    c = math.cos(theta)
    s = math.sin(theta)

    return (
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    )


def wall_segments():
    document = ezdxf.readfile(DXF_PATH)

    segments = []

    for entity in document.modelspace():
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
                (float(p[0]), float(p[1]))
                for p in entity.get_points("xy")
            ]

        else:
            continue

        pairs = list(zip(points, points[1:]))

        if kind == "LWPOLYLINE" and entity.closed:
            pairs.append((points[-1], points[0]))

        for a, b in pairs:
            if math.dist(a, b) < 0.03:
                continue

            segments.append(LineString([a, b]))

    return segments


def intersection_distances(geometry, origin):
    if geometry.is_empty:
        return []

    if geometry.geom_type == "Point":
        return [float(origin.distance(geometry))]

    if geometry.geom_type == "MultiPoint":
        return [
            float(origin.distance(point))
            for point in geometry.geoms
        ]

    if geometry.geom_type in (
        "LineString",
        "LinearRing",
    ):
        coords = list(geometry.coords)

        if not coords:
            return []

        return [
            float(origin.distance(Point(coords[0]))),
            float(origin.distance(Point(coords[-1]))),
        ]

    if hasattr(geometry, "geoms"):
        distances = []

        for part in geometry.geoms:
            distances.extend(
                intersection_distances(part, origin)
            )

        return distances

    return []


def ray_profile(position, tree, segments):
    origin = Point(position)

    distances = []

    for ray_index in range(RAY_COUNT):
        angle = (
            2.0 * math.pi * ray_index / RAY_COUNT
        )

        endpoint = (
            position[0] + RAY_LENGTH_M * math.cos(angle),
            position[1] + RAY_LENGTH_M * math.sin(angle),
        )

        ray = LineString([
            position,
            endpoint,
        ])

        nearest = RAY_LENGTH_M

        for index in tree.query(ray):
            segment = segments[int(index)]

            if not ray.intersects(segment):
                continue

            intersection = ray.intersection(segment)

            hits = intersection_distances(
                intersection,
                origin,
            )

            if hits:
                nearest = min(nearest, min(hits))

        distances.append(nearest)

    return distances


def analyze_candidate(candidate, frames, tree, segments):
    records = []

    previous_profile = None
    previous_position = None

    for frame in frames[::SAMPLE_EVERY_N_FRAMES]:
        position = transform(
            frame["x"],
            frame["y"],
            candidate,
        )

        point = Point(position)

        nearby_indices = tree.query(
            point.buffer(1.0)
        )

        nearest_wall = min(
            (
                float(
                    point.distance(segments[int(index)])
                )
                for index in nearby_indices
            ),
            default=None,
        )

        profile = ray_profile(
            position,
            tree,
            segments,
        )

        hit_fraction = float(
            np.mean(
                np.asarray(profile) <
                RAY_LENGTH_M - 1e-6
            )
        )

        continuity_difference = None

        if previous_profile is not None:
            continuity_difference = float(
                np.mean(
                    np.abs(
                        np.asarray(profile)
                        - np.asarray(previous_profile)
                    )
                )
            )

        travel_distance = None

        if previous_position is not None:
            travel_distance = math.dist(
                position,
                previous_position,
            )

        records.append({
            "frame_index": frame["frame_index"],
            "cad_position": [
                round(position[0], 5),
                round(position[1], 5),
            ],
            "nearest_wall_m": (
                round(nearest_wall, 4)
                if nearest_wall is not None
                else None
            ),
            "ray_hit_fraction": round(
                hit_fraction, 4
            ),
            "mean_ray_distance_m": round(
                float(np.mean(profile)),
                4,
            ),
            "profile_change_m": (
                round(continuity_difference, 4)
                if continuity_difference is not None
                else None
            ),
            "travel_distance_m": (
                round(travel_distance, 4)
                if travel_distance is not None
                else None
            ),
            "ray_profile_m": [
                round(value, 3)
                for value in profile
            ],
        })

        previous_profile = profile
        previous_position = position

    wall_distances = [
        record["nearest_wall_m"]
        for record in records
        if record["nearest_wall_m"] is not None
    ]

    hits = [
        record["ray_hit_fraction"]
        for record in records
    ]

    changes = [
        record["profile_change_m"]
        for record in records
        if record["profile_change_m"] is not None
    ]

    summary = {
        "rank": candidate["rank"],
        "sampled_frames": len(records),
        "median_nearest_wall_m": (
            round(float(np.median(wall_distances)), 4)
            if wall_distances
            else None
        ),
        "near_wall_fraction": (
            round(float(np.mean(
                np.asarray(wall_distances)
                < NEAR_WALL_THRESHOLD_M
            )), 4)
            if wall_distances
            else None
        ),
        "mean_ray_hit_fraction": round(
            float(np.mean(hits)), 4
        ),
        "median_profile_change_m": (
            round(float(np.median(changes)), 4)
            if changes
            else None
        ),
    }

    return summary, records


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = load_frames()
    candidates = load_candidates()
    segments = wall_segments()

    if not frames or not candidates or not segments:
        raise RuntimeError(
            "Missing frames, candidates, or CAD walls."
        )

    tree = STRtree(segments)

    results = []

    for candidate in candidates:
        print(
            "Analyzing candidate",
            candidate["rank"],
        )

        summary, records = analyze_candidate(
            candidate,
            frames,
            tree,
            segments,
        )

        results.append({
            "candidate": candidate,
            "summary": summary,
            "frames": records,
        })

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "method": "CAD_WALL_RAYCAST_STRUCTURE",
        "registration_verified": False,
        "image_evidence_used": False,
        "parameters": {
            "ray_count": RAY_COUNT,
            "ray_length_m": RAY_LENGTH_M,
            "sample_every_n_frames":
                SAMPLE_EVERY_N_FRAMES,
            "near_wall_threshold_m":
                NEAR_WALL_THRESHOLD_M,
        },
        "results": results,
        "limitations": [
            "Ray casting uses CAD geometry only.",
            "No image observations are compared.",
            "Wall geometry may contain openings or errors.",
            "A ray intersection does not prove a visible wall.",
            "Profile continuity alone cannot establish alignment.",
            "Registration cannot be verified from these results.",
        ],
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== STRUCTURAL REGISTRATION SCREENING ===")

    for result in results:
        summary = result["summary"]

        print(
            f"Candidate #{summary['rank']}: "
            f"near_wall={summary['near_wall_fraction']}, "
            f"ray_hits={summary['mean_ray_hit_fraction']}, "
            f"profile_change={summary['median_profile_change_m']}"
        )

    print("Registration verified: NO")
    print(
        "Report:",
        OUTPUT_PATH.relative_to(ROOT),
    )


if __name__ == "__main__":
    main()
