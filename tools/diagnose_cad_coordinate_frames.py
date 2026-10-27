
"""
Under-construction — Step 91
CAD coordinate-frame and wall-geometry diagnostic.

Inputs:
    outputs/multiview_tracks/tracks.json
    outputs/structural_planes/plane_candidates.json
    outputs/automatic_registration_v2/registration_candidates.json
    data/floorplans/floor_2.dxf

Outputs:
    outputs/cad_coordinate_diagnostic/diagnostic_report.json

Does not change registration candidates, fit transforms,
or claim construction progress.

Run:
    python tools/diagnose_cad_coordinate_frames.py
"""

import csv
import json
import math
from pathlib import Path

import numpy as np
import ezdxf


ROOT = Path(__file__).resolve().parents[1]

TRACKS_PATH = (
    ROOT / "outputs" / "multiview_tracks" / "tracks.json"
)

PLANES_PATH = (
    ROOT / "outputs" / "structural_planes"
    / "plane_candidates.json"
)

CANDIDATES_PATH = (
    ROOT / "outputs" / "automatic_registration_v2"
    / "registration_candidates.json"
)

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

MANIFEST_PATH = (
    ROOT / "_archive" / "hilti_transition_20261023_004237"
    / "data" / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0" / "frame_manifest.csv"
)

OUT_DIR = ROOT / "outputs" / "cad_coordinate_diagnostic"
OUT_PATH = OUT_DIR / "diagnostic_report.json"

WALL_LAYERS = {
    "015 Innenwände",
    "010 Aussenwände",
    "016 Trennwände",
}

TARGET_PLANE = "plane_03"

# Diagnostics only, not acceptance thresholds.
DISTANCE_BANDS_M = (0.3, 1.0, 2.0, 5.0)
ROUNDTRIP_TOLERANCE_M = 1e-7


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def clean_number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def bounds(points):
    p = np.asarray(points, dtype=float)

    if p.size == 0:
        return None

    if p.ndim != 2:
        raise ValueError("Expected 2D point array")

    lo = p.min(axis=0)
    hi = p.max(axis=0)

    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "extent": (hi - lo).tolist(),
        "centroid": p.mean(axis=0).tolist(),
    }


def percentiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def rotation(angle_deg):
    a = math.radians(float(angle_deg))
    c, s = math.cos(a), math.sin(a)

    return np.array([
        [c, -s],
        [s, c],
    ], dtype=float)


def map_to_cad(points_map, candidate):
    """
    Documented candidate convention:

        cad = R(angle) * map + translation

    Row-vector equivalent:
        cad = map @ R.T + translation
    """
    p = np.asarray(points_map, dtype=float)
    r = rotation(candidate["angle_deg"])
    t = np.array([candidate["tx"], candidate["ty"]])

    return p @ r.T + t


def cad_to_map(points_cad, candidate):
    """
    Algebraic inverse of map_to_cad:

        map = R.T * (cad - translation)

    Row-vector equivalent:
        map = (cad - translation) @ R
    """
    p = np.asarray(points_cad, dtype=float)
    r = rotation(candidate["angle_deg"])
    t = np.array([candidate["tx"], candidate["ty"]])

    return (p - t) @ r


def extract_candidates(data):
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict):
        raw = None

        for key in (
            "candidates",
            "registration_candidates",
            "ranked_candidates",
        ):
            if isinstance(data.get(key), list):
                raw = data[key]
                break

        if raw is None:
            raise ValueError(
                "Cannot find candidate list in JSON. "
                "Inspect registration_candidates.json schema."
            )
    else:
        raise ValueError("Unexpected candidate JSON format")

    candidates = []

    for index, entry in enumerate(raw):
        # Accept either direct transform fields or
        # a nested transform object.
        obj = entry

        if not all(
            key in obj
            for key in ("angle_deg", "tx", "ty")
        ):
            for key in ("transform", "parameters"):
                nested = entry.get(key)

                if (
                    isinstance(nested, dict)
                    and all(
                        name in nested
                        for name in ("angle_deg", "tx", "ty")
                    )
                ):
                    obj = nested
                    break

        if not all(
            key in obj
            for key in ("angle_deg", "tx", "ty")
        ):
            raise ValueError(
                f"Candidate {index + 1} missing "
                "angle_deg, tx or ty"
            )

        candidates.append({
            "rank": entry.get("rank", index + 1),
            "angle_deg": float(obj["angle_deg"]),
            "tx": float(obj["tx"]),
            "ty": float(obj["ty"]),
        })

    return candidates


def load_cad_segments():
    if not DXF_PATH.exists():
        raise FileNotFoundError(str(DXF_PATH))

    doc = ezdxf.readfile(str(DXF_PATH))
    msp = doc.modelspace()

    segments = []
    entity_counts = {}
    layer_counts = {}

    def add(a, b, layer, entity_type):
        a = np.asarray(a, dtype=float)[:2]
        b = np.asarray(b, dtype=float)[:2]

        if not (
            np.all(np.isfinite(a))
            and np.all(np.isfinite(b))
        ):
            return

        length = float(np.linalg.norm(b - a))

        if length <= 1e-6:
            return

        segments.append({
            "a": a,
            "b": b,
            "length": length,
            "layer": layer,
            "entity_type": entity_type,
        })

    for entity in msp:
        layer = entity.dxf.layer

        if layer not in WALL_LAYERS:
            continue

        kind = entity.dxftype()
        entity_counts[kind] = entity_counts.get(kind, 0) + 1
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

        if kind == "LINE":
            add(
                entity.dxf.start,
                entity.dxf.end,
                layer,
                kind,
            )

        elif kind == "LWPOLYLINE":
            if entity.has_arc:
                # Bulged segments require curved geometry;
                # do not silently approximate them as lines.
                raise RuntimeError(
                    "Bulged wall LWPOLYLINE detected. "
                    "Explicit arc handling is required."
                )

            vertices = [
                np.asarray(v[:2], dtype=float)
                for v in entity.get_points("xy")
            ]

            for a, b in zip(vertices, vertices[1:]):
                add(a, b, layer, kind)

            if entity.closed and len(vertices) > 2:
                add(
                    vertices[-1],
                    vertices[0],
                    layer,
                    kind,
                )

        elif kind == "POLYLINE":
            if not entity.is_2d_polyline:
                continue

            vertices = [
                np.asarray(v.dxf.location[:2], dtype=float)
                for v in entity.vertices
            ]

            for a, b in zip(vertices, vertices[1:]):
                add(a, b, layer, kind)

            if entity.is_closed and len(vertices) > 2:
                add(
                    vertices[-1],
                    vertices[0],
                    layer,
                    kind,
                )

    if not segments:
        raise RuntimeError(
            "No supported CAD wall segments found."
        )

    return segments, entity_counts, layer_counts


def load_landmarks():
    data = read_json(TRACKS_PATH)
    tracks = data.get("tracks", [])

    lookup = {}

    for track in tracks:
        point = np.asarray(track["point"], dtype=float)

        if point.shape == (3,) and np.all(np.isfinite(point)):
            lookup[int(track["track_id"])] = point

    planes_data = read_json(PLANES_PATH)

    matches = [
        p for p in planes_data["planes"]
        if p["plane_id"] == TARGET_PLANE
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {TARGET_PLANE}"
        )

    plane = matches[0]

    ids = [int(x) for x in plane["track_ids"]]

    missing = [x for x in ids if x not in lookup]

    if missing:
        raise RuntimeError(
            f"{len(missing)} plane tracks missing"
        )

    all_points = np.array(list(lookup.values()))
    plane_points = np.array([lookup[i] for i in ids])

    return all_points, plane_points, plane


def load_trajectory():
    if not MANIFEST_PATH.exists():
        return {
            "available": False,
            "reason": "Manifest file not found",
        }

    with MANIFEST_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {
            "available": False,
            "reason": "Empty manifest",
        }

    possible_schemas = (
        ("tx", "ty", "tz"),
        ("position_x", "position_y", "position_z"),
        ("x", "y", "z"),
    )

    schema = next(
        (
            fields for fields in possible_schemas
            if all(k in rows[0] for k in fields)
        ),
        None,
    )

    if schema is None:
        return {
            "available": False,
            "reason": "Pose translation columns not recognized",
            "available_columns": list(rows[0]),
        }

    points = []

    for row in rows:
        try:
            p = np.array(
                [float(row[k]) for k in schema],
                dtype=float,
            )

            if np.all(np.isfinite(p)):
                points.append(p)

        except (TypeError, ValueError, KeyError):
            continue

    if not points:
        return {
            "available": False,
            "reason": "No valid pose translations",
        }

    return {
        "available": True,
        "pose_count": len(points),
        "bounds": bounds(np.array(points)),
    }


def endpoints(segments):
    return np.array([
        point
        for s in segments
        for point in (s["a"], s["b"])
    ], dtype=float)


def nearest_finite_segment_distances(points, segments):
    """
    Compute distance to finite CAD segments, not
    their infinitely extended supporting lines.
    """
    p = np.asarray(points, dtype=float)

    nearest = np.full(
        len(p),
        np.inf,
        dtype=float,
    )

    best_ids = np.full(
        len(p),
        -1,
        dtype=int,
    )

    for index, segment in enumerate(segments):
        a = segment["a"]
        b = segment["b"]
        v = b - a

        denominator = float(np.dot(v, v))

        if denominator <= 1e-12:
            continue

        t = np.clip(
            ((p - a) @ v) / denominator,
            0.0,
            1.0,
        )

        closest = a + t[:, None] * v
        distance = np.linalg.norm(p - closest, axis=1)

        improved = distance < nearest

        nearest[improved] = distance[improved]
        best_ids[improved] = index

    return nearest, best_ids


def orientation_error(plane_normal, segment):
    n = np.asarray(plane_normal[:2], dtype=float)
    n /= max(np.linalg.norm(n), 1e-12)

    v = segment["b"] - segment["a"]
    v /= max(np.linalg.norm(v), 1e-12)

    wall_normal = np.array([-v[1], v[0]])

    cosine = np.clip(
        abs(float(np.dot(n, wall_normal))),
        0.0,
        1.0,
    )

    return float(np.degrees(np.arccos(cosine)))


def transformed_segments(segments, candidate):
    transformed = []

    for segment in segments:
        transformed.append({
            **segment,
            "a": cad_to_map(
                segment["a"][None, :],
                candidate,
            )[0],
            "b": cad_to_map(
                segment["b"][None, :],
                candidate,
            )[0],
        })

    return transformed


def orientation_diagnostics(
    points,
    normal,
    segments,
):
    """
    Nearest segment with no orientation filter,
    followed by orientation error of that segment.

    Also compute the nearest segment among walls
    within 12 degrees of the plane orientation.
    """
    distances, ids = nearest_finite_segment_distances(
        points, segments
    )

    angles = [
        orientation_error(normal, s)
        for s in segments
    ]

    nearest_angles = [
        angles[i] if i >= 0 else None
        for i in ids
    ]

    compatible = [
        s for s, angle in zip(segments, angles)
        if angle <= 12.0
    ]

    if compatible:
        compatible_distances, _ = (
            nearest_finite_segment_distances(
                points, compatible
            )
        )
    else:
        compatible_distances = np.full(
            len(points), np.inf
        )

    return {
        "nearest_any_wall_m": percentiles(distances),
        "nearest_wall_orientation_error_deg": (
            percentiles([
                a for a in nearest_angles
                if a is not None
            ])
        ),
        "orientation_compatible_segment_count": len(
            compatible
        ),
        "nearest_orientation_compatible_wall_m": (
            percentiles(compatible_distances)
        ),
        "fractions_within_distance": {
            str(d): float(np.mean(distances <= d))
            for d in DISTANCE_BANDS_M
        },
        "nearest_segment_ids": ids.tolist(),
    }


def roundtrip_check(candidate, cad_xy, map_xy):
    cad_sample = cad_xy[: min(50, len(cad_xy))]
    map_sample = map_xy[: min(50, len(map_xy))]

    cad_recovered = map_to_cad(
        cad_to_map(cad_sample, candidate),
        candidate,
    )

    map_recovered = cad_to_map(
        map_to_cad(map_sample, candidate),
        candidate,
    )

    cad_error = float(
        np.max(np.linalg.norm(
            cad_recovered - cad_sample,
            axis=1,
        ))
    )

    map_error = float(
        np.max(np.linalg.norm(
            map_recovered - map_sample,
            axis=1,
        ))
    )

    return {
        "cad_roundtrip_max_error_m": cad_error,
        "map_roundtrip_max_error_m": map_error,
        "algebraic_inverse_passed": (
            max(cad_error, map_error)
            <= ROUNDTRIP_TOLERANCE_M
        ),
        "note": (
            "Algebraic inversion does not prove "
            "the candidate represents the true "
            "physical coordinate relationship."
        ),
    }


def bounding_box_gap(a, b):
    """
    Shortest distance between two axis-aligned XY
    bounding boxes. Zero means their boxes overlap.
    """
    a_min = np.array(a["min"][:2])
    a_max = np.array(a["max"][:2])

    b_min = np.array(b["min"][:2])
    b_max = np.array(b["max"][:2])

    separation = np.maximum(
        np.maximum(a_min - b_max, b_min - a_max),
        0.0,
    )

    return float(np.linalg.norm(separation))


def segment_fragmentation(segments):
    lengths = np.array(
        [s["length"] for s in segments]
    )

    return {
        "length_distribution_m": percentiles(lengths),
        "count_shorter_than_0_5m": int(
            np.sum(lengths < 0.5)
        ),
        "count_shorter_than_1m": int(
            np.sum(lengths < 1.0)
        ),
        "count_longer_than_3m": int(
            np.sum(lengths >= 3.0)
        ),
        "count_longer_than_10m": int(
            np.sum(lengths >= 10.0)
        ),
        "shorter_than_1m_fraction": float(
            np.mean(lengths < 1.0)
        ),
        "note": (
            "Short segments may be CAD fragments, "
            "but collinearity and connectivity must "
            "be tested before merging."
        ),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_points, plane_points, plane = load_landmarks()

    candidates = extract_candidates(
        read_json(CANDIDATES_PATH)
    )

    segments, entity_counts, layer_counts = (
        load_cad_segments()
    )

    cad_xy = endpoints(segments)
    map_xy = all_points[:, :2]

    cad_bounds = bounds(cad_xy)
    landmark_bounds = bounds(all_points)
    plane_bounds = bounds(plane_points)

    evaluations = []

    print("\n=== STEP 91: CAD COORDINATE DIAGNOSTIC ===")
    print("Landmarks:", len(all_points))
    print("Plane landmarks:", len(plane_points))
    print("CAD segments:", len(segments))
    print("Candidates:", len(candidates))

    for candidate in candidates:
        mapped_segments = transformed_segments(
            segments, candidate
        )

        mapped_bounds = bounds(
            endpoints(mapped_segments)
        )

        plane_distance = orientation_diagnostics(
            plane_points[:, :2],
            plane["normal"],
            mapped_segments,
        )

        all_distances, _ = (
            nearest_finite_segment_distances(
                all_points[:, :2],
                mapped_segments,
            )
        )

        result = {
            "candidate_rank": candidate["rank"],
            "transform": candidate,
            "roundtrip": roundtrip_check(
                candidate,
                cad_xy,
                map_xy,
            ),
            "transformed_cad_bounds_xy": mapped_bounds,
            "bounding_box_gap_to_landmarks_m": (
                bounding_box_gap(
                    mapped_bounds,
                    landmark_bounds,
                )
            ),
            "bounding_box_gap_to_plane_m": (
                bounding_box_gap(
                    mapped_bounds,
                    plane_bounds,
                )
            ),
            "all_landmark_nearest_wall_m": (
                percentiles(all_distances)
            ),
            "plane_diagnostics": plane_distance,
        }

        evaluations.append(result)

        print(
            "Candidate",
            candidate["rank"],
            "| box gap:",
            round(
                result[
                    "bounding_box_gap_to_plane_m"
                ], 3
            ),
            "m | nearest wall median:",
            (
                round(
                    plane_distance[
                        "nearest_any_wall_m"
                    ]["median"], 3
                )
                if plane_distance[
                    "nearest_any_wall_m"
                ]
                else None
            ),
        )

    report = {
        "method": "coordinate_frame_and_cad_geometry_diagnostic",
        "sequence": "floor_2_2025-10-28_run_2",
        "target_plane": TARGET_PLANE,
        "source_track_count": len(all_points),
        "plane_track_count": len(plane_points),
        "trajectory": load_trajectory(),
        "landmark_bounds_xyz": landmark_bounds,
        "plane_bounds_xyz": plane_bounds,
        "plane_normal": plane["normal"],
        "raw_cad_bounds_xy": cad_bounds,
        "cad_wall_segment_count": len(segments),
        "cad_entity_counts": entity_counts,
        "cad_layer_counts": layer_counts,
        "segment_fragmentation": (
            segment_fragmentation(segments)
        ),
        "candidate_diagnostics": evaluations,
        "registration_verified": False,
        "construction_progress_verified": False,
        "decision": "DIAGNOSTIC_ONLY",
        "limitations": [
            "The transform convention is inherited from earlier registration code.",
            "Round-trip consistency cannot establish real-world alignment.",
            "Bounding-box overlap is necessary but not sufficient for correspondence.",
            "Nearest-wall distances are not independent proof of CAD registration.",
            "Sparse reconstructed landmarks may belong to nonstructural objects.",
            "CAD polyline bulges are rejected rather than approximated.",
            "CAD inserts, blocks, splines and arcs are not expanded.",
            "The CAD wall extraction may therefore be incomplete.",
            "No candidate is optimized or automatically corrected.",
        ],
    }

    OUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\n=== COMPLETE ===")
    print("CAD registration verified: NO")
    print("Report:", OUT_PATH)


if __name__ == "__main__":
    main()
