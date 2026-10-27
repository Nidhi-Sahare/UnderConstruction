
"""
SiteWitness — Experimental CAD-to-Camera Overlay

Run:
    python -m src.sitewitness_cad_overlay

Reuses:
    Existing eight alignment candidates
    Existing Floor 2 wall meshes
    Existing Hilti trajectory
    Existing calibrated fisheye camera

Does NOT regenerate trajectories or modify the dashboard.

Outputs:
    outputs/sitewitness/cad_overlay/
        frame_00053_comparison.jpg
        frame_00068_comparison.jpg
        frame_00090_comparison.jpg
        frame_00129_comparison.jpg
        overlay_report.json

All overlays are EXPERIMENTAL / UNVERIFIED.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .sitewitness_camera_geometry import (
    load_calibration,
    project_camera_points,
    world_to_imu,
    transform_points,
)


ROOT = Path(__file__).resolve().parents[1]

CANDIDATES = (
    ROOT / "_archive"
    / "hilti_transition_20261023_004237"
    / "outputs" / "registration"
    / "candidate_alignments.json"
)

WALLS = ROOT / "data" / "geometry" / "floor_2_wall_meshes.json"

TRAJECTORY = (
    ROOT / "data" / "real_samples"
    / "floor_2_camera_trajectory.json"
)

FRAMES = ROOT / "sample_gallery" / "real_floor_2" / "frames"

OUTPUT = ROOT / "outputs" / "sitewitness" / "cad_overlay"

FRAME_INDICES = (53, 68, 90, 129)

# Existing preferred candidate, identified by its previously
# reported transform, not by an assumed array index.
EXPECTED_ANGLE = 317.9689499733972
EXPECTED_TRANSLATION = np.array([
    -47.29695344668649,
    26.84741454935609,
])

ANGLE_TOLERANCE_DEG = 0.01
TRANSLATION_TOLERANCE = 0.01

# Experimental display parameters, not verified measurements.
MAX_WALLS = 73
SAMPLES_PER_EDGE = 35


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_vector(value, count):
    if isinstance(value, dict):
        keys = (
            ("x", "y", "z")
            if count == 3
            else ("x", "y", "z", "w")
        )
        value = [value.get(k) for k in keys]

    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    if array.shape != (count,) or not np.isfinite(array).all():
        return None

    return array


def load_existing_candidate():
    data = read_json(CANDIDATES)
    candidates = data.get("candidates", [])

    matches = []

    for index, candidate in enumerate(candidates):
        try:
            angle = float(candidate["angle_degrees"])
            translation = np.asarray(
                candidate["translation"], dtype=np.float64
            )
        except (KeyError, TypeError, ValueError):
            continue

        if translation.shape != (2,):
            continue

        angle_error = abs(
            (angle - EXPECTED_ANGLE + 180) % 360 - 180
        )

        translation_error = float(
            np.linalg.norm(
                translation - EXPECTED_TRANSLATION
            )
        )

        if (
            angle_error < ANGLE_TOLERANCE_DEG
            and translation_error < TRANSLATION_TOLERANCE
        ):
            matches.append((index, candidate))

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely identify existing Candidate 04. "
            "Do not substitute another candidate automatically. "
            f"Matches found: {len(matches)}"
        )

    index, candidate = matches[0]
    return index, candidate


def cad_to_recorded_xy(points, candidate):
    """
    Invert the existing planar registration hypothesis:

        CAD_xy = R(angle) @ recorded_xy + translation

    The direction above is an explicit hypothesis that
    must be checked against the original registration code.
    """
    angle = math.radians(float(candidate["angle_degrees"]))

    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ], dtype=np.float64)

    translation = np.asarray(
        candidate["translation"], dtype=np.float64
    )

    return (points - translation) @ rotation


def load_wall_edges(candidate):
    data = read_json(WALLS)

    if data.get("units") != "metres":
        raise ValueError("Expected wall geometry in metres")

    walls = data.get("walls", [])
    edges = []

    for wall in walls[:MAX_WALLS]:
        mesh = wall.get("mesh", {})
        vertices = np.asarray(
            mesh.get("vertices_xyz", []),
            dtype=np.float64,
        )

        if vertices.shape != (8, 3):
            continue

        # Existing Three.js convention:
        # mesh X = CAD X
        # mesh Y = vertical height
        # mesh Z = CAD Y
        cad_xy = vertices[:, [0, 2]]
        height = vertices[:, 1]

        recorded_xy = cad_to_recorded_xy(
            cad_xy, candidate
        )

        recorded_xyz = np.column_stack([
            recorded_xy[:, 0],
            recorded_xy[:, 1],
            height,
        ])

        # Edges of the original eight-vertex wall prism.
        indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        for start, end in indices:
            edges.append({
                "wall_id": wall.get("id"),
                "start": recorded_xyz[start],
                "end": recorded_xyz[end],
            })

    return edges


def render_overlay(
    image,
    cam,
    record,
    edges,
    convention,
):
    canvas = image.copy()

    position = numeric_vector(
        record.get("position"), 3
    )
    orientation = numeric_vector(
        record.get("orientation"), 4
    )

    if position is None or orientation is None:
        raise ValueError("Frame has no valid pose")

    R_cam_imu = cam.T_cam_imu

    drawn_segments = 0
    visible_edges = 0

    for edge in edges:
        start = edge["start"]
        end = edge["end"]

        alpha = np.linspace(
            0.0, 1.0, SAMPLES_PER_EDGE
        )[:, None]

        points_world = (
            start[None, :] * (1.0 - alpha)
            + end[None, :] * alpha
        )

        points_imu = world_to_imu(
            points_world,
            position,
            orientation,
            convention,
        )

        points_camera = transform_points(
            points_imu,
            R_cam_imu,
        )

        pixels, visible = project_camera_points(
            cam, points_camera
        )

        edge_drawn = False

        for index in range(len(pixels) - 1):
            if not (
                visible[index]
                and visible[index + 1]
            ):
                continue

            p0 = tuple(
                np.rint(pixels[index]).astype(int)
            )
            p1 = tuple(
                np.rint(pixels[index + 1]).astype(int)
            )

            # Yellow: projected architectural hypothesis.
            cv2.line(
                canvas, p0, p1,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            drawn_segments += 1
            edge_drawn = True

        if edge_drawn:
            visible_edges += 1

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 105),
        (18, 18, 18),
        -1,
    )

    cv2.putText(
        canvas,
        f"Candidate 04 | {convention}",
        (22, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "EXPERIMENTAL - NOT VERIFIED",
        (22, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 190, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas, {
        "visible_edges": visible_edges,
        "drawn_segments": drawn_segments,
    }


def make_comparison(left, right):
    target_width = 1100

    combined = np.hstack([left, right])

    scale = target_width / combined.shape[1]

    return cv2.resize(
        combined,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    cam = load_calibration("cam0")

    candidate_index, candidate = (
        load_existing_candidate()
    )

    trajectory = read_json(TRAJECTORY)
    records = trajectory["frames"]

    edges = load_wall_edges(candidate)

    results = []

    for frame_index in FRAME_INDICES:
        if frame_index >= len(records):
            continue

        record = records[frame_index]
        filename = record.get("filename")

        if not filename:
            continue

        image = cv2.imread(
            str(FRAMES / filename)
        )

        if image is None:
            continue

        if (
            image.shape[1] != cam.width
            or image.shape[0] != cam.height
        ):
            raise ValueError(
                f"Frame {frame_index} resolution "
                "does not match calibration"
            )

        variants = []
        measurements = {}

        for convention in (
            "imu_to_world",
            "world_to_imu",
        ):
            rendered, metrics = render_overlay(
                image,
                cam,
                record,
                edges,
                convention,
            )

            variants.append(rendered)
            measurements[convention] = metrics

        comparison = make_comparison(
            variants[0], variants[1]
        )

        output_file = (
            OUTPUT
            / f"frame_{frame_index:05d}_comparison.jpg"
        )

        if not cv2.imwrite(
            str(output_file), comparison
        ):
            raise RuntimeError(
                f"Failed to write {output_file}"
            )

        results.append({
            "frame": frame_index,
            "image": filename,
            "output": str(
                output_file.relative_to(ROOT)
            ),
            "measurements": measurements,
            "status": "EXPERIMENTAL_UNVERIFIED",
        })

        print(
            f"Frame {frame_index}:",
            measurements,
        )

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "SITEWITNESS_CAD_IMAGE_PROJECTION",
        "candidate": {
            "label": "Candidate 04",
            "existing_array_index": candidate_index,
            "angle_degrees": candidate["angle_degrees"],
            "translation": candidate["translation"],
            "registration_verified": False,
        },
        "wall_edges_loaded": len(edges),
        "frames": results,
        "limitations": [
            "Uses the existing unverified planar alignment.",
            "Assumes the original candidate maps recorded "
            "XY coordinates to CAD XY.",
            "Assumes recorded Z and CAD height share a "
            "compatible vertical datum and metre scale.",
            "Assumes quaternion components are xyzw.",
            "Neither tested quaternion direction is verified.",
            "Uses provisional 3-metre wall geometry.",
            "Does not model visibility occlusion or "
            "construction completion.",
            "Projected lines are hypotheses, not detections.",
            "Visible-edge counts are not alignment scores.",
            "No independent image-to-CAD landmark "
            "correspondences have been verified.",
        ],
        "cad_registration_verified": False,
        "construction_progress_verified": False,
    }

    report_file = OUTPUT / "overlay_report.json"

    report_file.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSITEWITNESS — EXPERIMENTAL OVERLAY")
    print("=" * 50)
    print("Existing candidate index:", candidate_index)
    print("Wall edges:", len(edges))
    print("Frames rendered:", len(results))
    print("CAD registration: UNVERIFIED")
    print("Report:", report_file)


if __name__ == "__main__":
    main()
