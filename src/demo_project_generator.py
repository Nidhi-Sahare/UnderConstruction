
"""
Under-construction — Step 116C
Synthetic CAD + walkthrough dataset generator.

Creates:
    demo_project/floor_plan.dxf
    demo_project/walkthrough.mp4
    demo_project/ground_truth.json

Requirements:
    pip install ezdxf numpy opencv-python

Run from project root:
    .\\.venv\\Scripts\\python.exe src\\demo_project_generator.py

This dataset is synthetic. Its ground truth is known because
both the CAD and video are generated from the same model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "demo_project"

WIDTH = 640
HEIGHT = 360
FPS = 12
DURATION = 10
FRAME_COUNT = FPS * DURATION

WALL_HEIGHT = 2.8
CAMERA_HEIGHT = 1.55

# World coordinates are in metres for this synthetic dataset.
# Wall geometry is defined by endpoints in the horizontal plane.
# Each wall is a separate entity, with a stable ID.
WALLS = [
    ("W01", (0, 0), (12, 0), "EXTERIOR"),
    ("W02", (12, 0), (12, 9), "EXTERIOR"),
    ("W03", (12, 9), (0, 9), "EXTERIOR"),
    ("W04", (0, 9), (0, 0), "EXTERIOR"),

    # Living / corridor separation, with a doorway.
    ("W05", (4.5, 0), (4.5, 3.1), "INTERIOR"),
    ("W06", (4.5, 4.3), (4.5, 9), "INTERIOR"),

    # Kitchen partition and opening.
    ("W07", (4.5, 5.8), (7.3, 5.8), "INTERIOR"),
    ("W08", (8.5, 5.8), (12, 5.8), "INTERIOR"),

    # Bedroom partition and opening.
    ("W09", (8.2, 0), (8.2, 2.9), "INTERIOR"),
    ("W10", (8.2, 4.1), (8.2, 5.8), "INTERIOR"),

    # Bathroom.
    ("W11", (0, 5.7), (1.3, 5.7), "INTERIOR"),
    ("W12", (2.3, 5.7), (4.5, 5.7), "INTERIOR"),
]

COLUMNS = [
    ("C01", 4.5, 0),
    ("C02", 4.5, 9),
    ("C03", 8.2, 5.8),
    ("C04", 12, 5.8),
]

ROOMS = [
    ("Living room", 0.3, 0.3),
    ("Bathroom", 0.3, 6.0),
    ("Corridor", 4.8, 4.5),
    ("Bedroom", 8.5, 0.3),
    ("Kitchen", 5.0, 6.2),
]

# Illustrative construction state, not inferred from video.
WALL_STATES = {
    wall_id: (
        "BUILT" if int(wall_id[1:]) <= 8 else "PARTIALLY_BUILT"
    )
    for wall_id, *_ in WALLS
}


def generate_cad():
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6  # metres

    layers = [
        ("EXTERIOR", 7),
        ("INTERIOR", 8),
        ("COLUMNS", 3),
        ("ROOM_LABELS", 5),
        ("DOORS", 2),
    ]

    for name, color in layers:
        doc.layers.new(name, dxfattribs={"color": color})

    msp = doc.modelspace()

    for wall_id, a, b, layer in WALLS:
        entity = msp.add_line(
            a, b, dxfattribs={"layer": layer}
        )
        entity.set_xdata(
            "UNDERCONSTRUCTION",
            [(1000, wall_id)],
        ) if False else None

    for _, x, y in COLUMNS:
        half = 0.18
        msp.add_lwpolyline(
            [
                (x-half, y-half),
                (x+half, y-half),
                (x+half, y+half),
                (x-half, y+half),
            ],
            close=True,
            dxfattribs={"layer": "COLUMNS"},
        )

    for label, x, y in ROOMS:
        msp.add_text(
            label,
            dxfattribs={
                "height": 0.24,
                "layer": "ROOM_LABELS",
                "insert": (x, y),
            },
        )

    # Simple doorway threshold symbols.
    for a, b in [
        ((4.5, 3.1), (4.5, 4.3)),
        ((7.3, 5.8), (8.5, 5.8)),
        ((8.2, 2.9), (8.2, 4.1)),
        ((1.3, 5.7), (2.3, 5.7)),
    ]:
        msp.add_line(
            a, b,
            dxfattribs={"layer": "DOORS", "color": 2},
        )

    path = OUTPUT / "floor_plan.dxf"
    doc.saveas(path)
    return path


def camera_pose(t):
    """
    Camera follows an unobstructed illustrative path.
    Coordinates are in the same world frame as the DXF.
    """
    keyframes = [
        (0.0, 6.0, 3.5, 0.0),
        (2.5, 6.0, 4.3, 0.0),
        (5.0, 6.0, 4.8, 0.0),
        (7.5, 6.0, 5.0, 0.65),
        (10.0, 6.0, 5.0, 1.1),
    ]

    for i in range(len(keyframes) - 1):
        a = keyframes[i]
        b = keyframes[i + 1]

        if t <= b[0]:
            u = (t - a[0]) / (b[0] - a[0])
            u = max(0, min(1, u))
            u = u*u*(3-2*u)

            return tuple(
                a[k] + (b[k] - a[k])*u
                for k in (1, 2, 3)
            )

    return keyframes[-1][1:]


def to_camera(point, pose):
    """World 3D -> camera coordinates (right, up, forward)."""
    x, y, z = point
    cx, cy, yaw = pose

    dx = x - cx
    dy = y - cy

    # yaw=0 looks toward +Y.
    right = math.cos(yaw)*dx - math.sin(yaw)*dy
    forward = math.sin(yaw)*dx + math.cos(yaw)*dy
    up = z - CAMERA_HEIGHT

    return np.array([right, up, forward], dtype=float)


def clip_near(vertices, near=0.12):
    """Clip a polygon against the near camera plane."""
    result = []

    for i, current in enumerate(vertices):
        previous = vertices[i-1]

        ci = current[2] >= near
        pi = previous[2] >= near

        if ci != pi:
            fraction = (
                (near - previous[2]) /
                (current[2] - previous[2])
            )
            result.append(
                previous + fraction*(current-previous)
            )

        if ci:
            result.append(current)

    return result


def project(point):
    focal = 440
    return (
        int(WIDTH/2 + focal*point[0]/point[2]),
        int(HEIGHT/2 - focal*point[1]/point[2]),
    )


def wall_faces():
    faces = []

    for wall_id, a, b, _ in WALLS:
        height = (
            WALL_HEIGHT
            if WALL_STATES[wall_id] == "BUILT"
            else 1.35
        )

        ax, ay = a
        bx, by = b

        dx, dy = bx-ax, by-ay
        length = math.hypot(dx, dy)
        nx, ny = -dy/length*0.09, dx/length*0.09

        for sign in (-1, 1):
            p = [
                (ax+sign*nx, ay+sign*ny, 0),
                (bx+sign*nx, by+sign*ny, 0),
                (bx+sign*nx, by+sign*ny, height),
                (ax+sign*nx, ay+sign*ny, height),
            ]
            faces.append(
                (p, (154, 161, 166), wall_id)
            )

        faces.append((
            [
                (ax-nx, ay-ny, height),
                (bx-nx, by-ny, height),
                (bx+nx, by+ny, height),
                (ax+nx, ay+ny, height),
            ],
            (185, 191, 195),
            wall_id,
        ))

    for column_id, x, y in COLUMNS:
        h = WALL_HEIGHT
        s = 0.17

        for side in range(4):
            corners = [
                (x-s, y-s), (x+s, y-s),
                (x+s, y+s), (x-s, y+s),
            ]
            a = corners[side]
            b = corners[(side+1) % 4]

            faces.append((
                [
                    (a[0], a[1], 0),
                    (b[0], b[1], 0),
                    (b[0], b[1], h),
                    (a[0], a[1], h),
                ],
                (135, 144, 151),
                column_id,
            ))

    return faces


FACES = wall_faces()


def render_frame(pose, frame_number):
    image = np.full(
        (HEIGHT, WIDTH, 3),
        (37, 35, 32),
        dtype=np.uint8,
    )

    # Ceiling / sky and concrete floor.
    image[:HEIGHT//2] = (92, 91, 88)
    image[HEIGHT//2:] = (76, 79, 82)

    # Floor grid.
    floor_lines = []

    for x in range(-2, 16):
        floor_lines.append([
            (x, -2, 0.005),
            (x, 13, 0.005),
        ])

    for y in range(-2, 14):
        floor_lines.append([
            (-2, y, 0.005),
            (15, y, 0.005),
        ])

    for a, b in floor_lines:
        cam = [
            to_camera(a, pose),
            to_camera(b, pose),
        ]
        clipped = clip_near(cam)

        if len(clipped) == 2:
            cv2.line(
                image,
                project(clipped[0]),
                project(clipped[1]),
                (90, 92, 94),
                1,
                cv2.LINE_AA,
            )

    visible_faces = []

    for vertices, color, object_id in FACES:
        camera_vertices = [
            to_camera(p, pose)
            for p in vertices
        ]

        clipped = clip_near(camera_vertices)

        if len(clipped) < 3:
            continue

        depth = sum(
            v[2] for v in clipped
        ) / len(clipped)

        visible_faces.append((
            depth, clipped, color, object_id
        ))

    # Painter's algorithm, sufficient for this simple demo.
    visible_faces.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for _, vertices, color, _ in visible_faces:
        polygon = np.array(
            [project(p) for p in vertices],
            dtype=np.int32,
        )

        cv2.fillPoly(image, [polygon], color)
        cv2.polylines(
            image, [polygon], True,
            (76, 80, 84), 1, cv2.LINE_AA,
        )

    cv2.rectangle(
        image, (12, 12), (355, 72),
        (25, 25, 29), -1,
    )

    cv2.putText(
        image,
        "UNDER-CONSTRUCTION / SYNTHETIC SITE",
        (22, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43, (205, 191, 235), 1, cv2.LINE_AA,
    )

    cv2.putText(
        image,
        f"FRAME {frame_number:03d} / NOT REAL FOOTAGE",
        (22, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43, (125, 175, 230), 1, cv2.LINE_AA,
    )

    return image


def generate_video():
    path = OUTPUT / "walkthrough.mp4"

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "OpenCV could not initialize MP4 video output."
        )

    poses = []

    try:
        for frame in range(FRAME_COUNT):
            timestamp = frame / FPS
            pose = camera_pose(timestamp)

            image = render_frame(pose, frame)
            writer.write(image)

            poses.append({
                "frame_index": frame,
                "timestamp_seconds": round(timestamp, 5),
                "camera_x": round(pose[0], 5),
                "camera_y": round(pose[1], 5),
                "camera_height": CAMERA_HEIGHT,
                "camera_yaw_radians": round(pose[2], 6),
            })
    finally:
        writer.release()

    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError("MP4 output was not created.")

    return path, poses


def generate_ground_truth(poses):
    data = {
        "dataset": "underconstruction_synthetic_demo_v1",
        "synthetic": True,
        "real_world_validation": False,
        "coordinate_system": {
            "name": "SYNTHETIC_WORLD",
            "horizontal_axes": ["x", "y"],
            "vertical_axis": "z",
            "units": "metres",
            "cad_to_world_transform": "identity",
        },
        "video": {
            "filename": "walkthrough.mp4",
            "width": WIDTH,
            "height": HEIGHT,
            "fps": FPS,
            "frame_count": FRAME_COUNT,
            "camera_model": "ideal_pinhole",
            "focal_length_pixels": 440,
            "principal_point_pixels": [
                WIDTH/2, HEIGHT/2
            ],
        },
        "walls": [
            {
                "wall_id": wall_id,
                "start": list(a),
                "end": list(b),
                "layer": layer,
                "construction_state": WALL_STATES[wall_id],
            }
            for wall_id, a, b, layer in WALLS
        ],
        "columns": [
            {
                "column_id": column_id,
                "center": [x, y],
                "construction_state": "BUILT",
            }
            for column_id, x, y in COLUMNS
        ],
        "camera_poses": poses,
        "important_note": (
            "Ground-truth construction states and camera poses "
            "are generator-defined. They are not observations "
            "or predictions from the application."
        ),
    }

    path = OUTPUT / "ground_truth.json"

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("\nUNDER-CONSTRUCTION / DEMO GENERATOR")
    print("Generating synthetic DXF...")
    cad = generate_cad()

    print("Rendering synthetic walkthrough...")
    video, poses = generate_video()

    print("Writing ground truth...")
    truth = generate_ground_truth(poses)

    print("\nSUCCESS")
    print("Floor plan:", cad)
    print("Walkthrough:", video)
    print("Ground truth:", truth)
    print("Frames:", FRAME_COUNT)
    print("Duration:", DURATION, "seconds")
    print("\nThis is synthetic test data, not real site evidence.")


if __name__ == "__main__":
    main()
