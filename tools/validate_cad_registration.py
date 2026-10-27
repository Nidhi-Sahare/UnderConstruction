
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

REGISTRATION_PATH = (
    ROOT
    / "outputs"
    / "automatic_registration_v2"
    / "registration_candidates.json"
)

DXF_PATH = ROOT / "data" / "floorplans" / "floor_2.dxf"

ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

OUTPUT_DIR = ROOT / "outputs" / "registration_validation"

NUM_FRAMES = 12

MAP_SIZE = 520
PHOTO_WIDTH = 330
PHOTO_HEIGHT = 250

BACKGROUND = (23, 26, 31)
WHITE = (240, 240, 240)
MUTED = (150, 160, 170)
ORANGE = (45, 155, 255)
BLUE = (255, 175, 65)


def find_manifest():
    candidates = list(
        ROOT.rglob("frame_manifest.csv")
    )

    candidates = [
        path
        for path in candidates
        if SEQUENCE in str(path)
        and "cam0" in path.parts
    ]

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Hilti manifest, "
            f"found {len(candidates)}: {candidates}"
        )

    return candidates[0]


def load_frames():
    manifest = find_manifest()
    frame_dir = manifest.parent / "frames"

    records = []

    with manifest.open(
        newline="",
        encoding="utf-8-sig",
    ) as file:
        for row in csv.DictReader(file):
            if str(
                row.get("pose_matched", "")
            ).lower() != "true":
                continue

            try:
                x = float(row["tx"])
                y = float(row["ty"])
                timestamp = float(
                    row["header_timestamp_s"]
                )
            except (KeyError, TypeError, ValueError):
                continue

            image_path = (
                frame_dir / row["filename"]
            )

            if not image_path.exists():
                continue

            records.append(
                {
                    "filename": row["filename"],
                    "path": image_path,
                    "timestamp": timestamp,
                    "xy": np.array(
                        [x, y],
                        dtype=np.float64,
                    ),
                }
            )

    records.sort(
        key=lambda item: item["timestamp"]
    )

    if len(records) < NUM_FRAMES:
        raise RuntimeError(
            "Insufficient synchronized photographs."
        )

    return records


def select_frames(records, count):
    # Choose observations spread across time.
    indices = np.linspace(
        0,
        len(records) - 1,
        count,
    )

    indices = np.unique(
        np.rint(indices).astype(int)
    )

    return [records[i] for i in indices]


def load_candidates():
    data = json.loads(
        REGISTRATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    candidates = data.get(
        "candidates",
        []
    )

    if not candidates:
        raise RuntimeError(
            "No registration candidates found."
        )

    return candidates


def load_dxf():
    doc = ezdxf.readfile(DXF_PATH)

    segments = []

    for entity in doc.modelspace():
        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end

            segments.append(
                [[a.x, a.y], [b.x, b.y]]
            )

        elif kind == "LWPOLYLINE":
            points = [
                [p[0], p[1]]
                for p in entity.get_points("xy")
            ]

            for a, b in zip(
                points[:-1],
                points[1:],
            ):
                segments.append([a, b])

            if entity.closed and len(points) > 2:
                segments.append(
                    [points[-1], points[0]]
                )

    if not segments:
        raise RuntimeError(
            "No drawable CAD geometry found."
        )

    return np.asarray(
        segments,
        dtype=np.float64,
    )


class CADView:
    def __init__(self, segments):
        self.segments = segments

        points = segments.reshape(-1, 2)

        self.minimum = points.min(axis=0)
        maximum = points.max(axis=0)

        extent = maximum - self.minimum

        self.scale = (
            (MAP_SIZE - 60)
            / max(extent)
        )

        self.center = (
            self.minimum + maximum
        ) / 2

        self.base = self.draw_base()

    def project(self, points):
        points = np.asarray(
            points,
            dtype=np.float64,
        )

        centered = (
            points - self.center
        ) * self.scale

        pixels = np.empty_like(
            centered
        )

        pixels[..., 0] = (
            MAP_SIZE / 2 + centered[..., 0]
        )

        pixels[..., 1] = (
            MAP_SIZE / 2 - centered[..., 1]
        )

        return np.rint(
            pixels
        ).astype(np.int32)

    def draw_base(self):
        image = np.full(
            (MAP_SIZE, MAP_SIZE, 3),
            BACKGROUND,
            dtype=np.uint8,
        )

        projected = self.project(
            self.segments
        )

        for start, end in projected:
            cv2.line(
                image,
                tuple(start),
                tuple(end),
                MUTED,
                1,
                cv2.LINE_AA,
            )

        return image


def transform(points, candidate):
    theta = math.radians(
        candidate["angle_deg"]
    )

    c = math.cos(theta)
    s = math.sin(theta)

    rotation = np.array(
        [
            [c, -s],
            [s, c],
        ],
        dtype=np.float64,
    )

    return (
        points @ rotation.T
        + [
            candidate["tx"],
            candidate["ty"],
        ]
    )


def travel_direction(records, index):
    # Estimate direction from nearby camera positions.
    # This is NOT optical camera orientation.
    before = records[
        max(0, index - 2)
    ]["xy"]

    after = records[
        min(len(records) - 1, index + 2)
    ]["xy"]

    delta = after - before

    norm = np.linalg.norm(delta)

    if norm < 0.1:
        return None

    return delta / norm


def fit_image(image, width, height):
    canvas = np.full(
        (height, width, 3),
        BACKGROUND,
        dtype=np.uint8,
    )

    if image is None:
        return canvas

    ih, iw = image.shape[:2]

    scale = min(
        width / iw,
        height / ih,
    )

    resized = cv2.resize(
        image,
        (
            max(1, int(iw * scale)),
            max(1, int(ih * scale)),
        ),
    )

    rh, rw = resized.shape[:2]

    x = (width - rw) // 2
    y = (height - rh) // 2

    canvas[
        y:y + rh,
        x:x + rw,
    ] = resized

    return canvas


def draw_candidate_map(
    cad,
    candidate,
    selected,
    all_records,
):
    canvas = cad.base.copy()

    trajectory = np.asarray(
        [
            item["xy"]
            for item in all_records
        ],
        dtype=np.float64,
    )

    transformed = transform(
        trajectory,
        candidate,
    )

    pixels = cad.project(
        transformed
    )

    cv2.polylines(
        canvas,
        [pixels],
        False,
        BLUE,
        1,
        cv2.LINE_AA,
    )

    selected_points = np.asarray(
        [
            item["xy"]
            for item in selected
        ],
        dtype=np.float64,
    )

    transformed_selected = transform(
        selected_points,
        candidate,
    )

    projected = cad.project(
        transformed_selected
    )

    for index, point in enumerate(projected):
        cv2.circle(
            canvas,
            tuple(point),
            5,
            ORANGE,
            -1,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            str(index + 1),
            tuple(point + [8, -7]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    return canvas


def draw_photo_grid(selected):
    columns = 3
    rows = math.ceil(
        len(selected) / columns
    )

    tile_h = PHOTO_HEIGHT + 34

    grid = np.full(
        (
            rows * tile_h,
            columns * PHOTO_WIDTH,
            3,
        ),
        BACKGROUND,
        dtype=np.uint8,
    )

    for index, record in enumerate(selected):
        image = cv2.imread(
            str(record["path"])
        )

        fitted = fit_image(
            image,
            PHOTO_WIDTH,
            PHOTO_HEIGHT,
        )

        row = index // columns
        col = index % columns

        x = col * PHOTO_WIDTH
        y = row * tile_h

        grid[
            y:y + PHOTO_HEIGHT,
            x:x + PHOTO_WIDTH,
        ] = fitted

        label = (
            f"{index + 1:02d}  "
            f"{record['filename']}"
        )

        cv2.putText(
            grid,
            label,
            (x + 8, y + PHOTO_HEIGHT + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            WHITE,
            1,
            cv2.LINE_AA,
        )

    return grid


def main():
    print("Loading synchronized camera frames...")
    records = load_frames()

    selected = select_frames(
        records,
        NUM_FRAMES,
    )

    print(
        f"Available matched photographs: {len(records)}"
    )

    print(
        f"Selected validation photographs: {len(selected)}"
    )

    candidates = load_candidates()
    cad = CADView(load_dxf())

    photo_grid = draw_photo_grid(
        selected
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "status": "VISUAL_INSPECTION_REQUIRED",
        "verified": False,
        "frame_count": len(selected),
        "frames": [
            {
                "number": index + 1,
                "filename": item["filename"],
                "timestamp": item["timestamp"],
                "trajectory_xy": item["xy"].tolist(),
            }
            for index, item in enumerate(selected)
        ],
        "candidates": [],
    }

    for candidate in candidates:
        rank = candidate["rank"]

        map_image = draw_candidate_map(
            cad,
            candidate,
            selected,
            records,
        )

        # The contact sheet shares a single photo grid
        # because images are identical for every candidate.
        width = (
            MAP_SIZE + photo_grid.shape[1]
        )

        height = max(
            MAP_SIZE,
            photo_grid.shape[0],
        ) + 60

        sheet = np.full(
            (height, width, 3),
            BACKGROUND,
            dtype=np.uint8,
        )

        sheet[
            60:60 + MAP_SIZE,
            :MAP_SIZE,
        ] = map_image

        sheet[
            60:60 + photo_grid.shape[0],
            MAP_SIZE:,
        ] = photo_grid

        title = (
            f"Candidate #{rank} | "
            f"Rotation {candidate['angle_deg']:.1f} deg | "
            f"Score {candidate['score']:.3f}"
        )

        cv2.putText(
            sheet,
            title,
            (18, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        output_path = (
            OUTPUT_DIR
            / f"candidate_{rank}_contact_sheet.jpg"
        )

        cv2.imwrite(
            str(output_path),
            sheet,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                92,
            ],
        )

        predicted = transform(
            np.asarray(
                [item["xy"] for item in selected],
                dtype=np.float64,
            ),
            candidate,
        )

        summary["candidates"].append(
            {
                "rank": rank,
                "angle_deg": candidate["angle_deg"],
                "tx": candidate["tx"],
                "ty": candidate["ty"],
                "predicted_cad_positions": (
                    predicted.tolist()
                ),
                "contact_sheet": output_path.name,
                "verified": False,
            }
        )

        print(
            f"Saved candidate #{rank}: "
            f"{output_path.name}"
        )

    report_path = (
        OUTPUT_DIR
        / "validation_manifest.json"
    )

    report_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nValidation sheets saved to:")
    print(OUTPUT_DIR)
    print(
        "\nThese are visual inspection aids, "
        "not proof of registration."
    )


if __name__ == "__main__":
    main()
