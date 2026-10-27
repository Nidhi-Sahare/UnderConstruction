
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import ezdxf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"

TRAJECTORY_PATH = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "floorplan_registration"
    / "aligned_trajectory.csv"
)

WALLS_PATH = (
    ROOT
    / "data/wall_selection/selected_wall_footprints.json"
)

SEGMENTS_PATH = (
    ROOT
    / "data/geometry/normalized_segments.json"
)

OPENINGS_PATH = (
    ROOT
    / "data/geometry/opening_audit.json"
)

OUTPUT_DIR = ROOT / "outputs/automatic_registration_v2"

RESOLUTION_M = 0.10
MARGIN_M = 5.0

TRAJECTORY_SPACING_M = 0.20

COARSE_ROTATION_DEG = 5.0
COARSE_TRANSLATION_M = 1.0

FINE_ROTATION_DEG = 1.0
FINE_TRANSLATION_M = 0.25

TOP_K = 5

EXTERIOR_LAYER_TERMS = (
    "aussenwand",
    "fassadensystem",
    "vorhangfassade",
)

WALL_BUFFER_M = 0.15
CROSSING_SAMPLE_M = 0.10

# These weights are heuristic, not learned or calibrated.
WEIGHTS = {
    "outside": 8.0,
    "wall_occupancy": 12.0,
    "wall_crossing": 8.0,
    "clearance": 0.5,
}


@dataclass
class Candidate:
    angle_deg: float
    tx: float
    ty: float
    score: float
    inside_fraction: float
    wall_occupancy_fraction: float
    crossing_fraction: float
    mean_clearance_m: float


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_trajectory():
    points = []

    with TRAJECTORY_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.DictReader(f):
            try:
                x = float(row["x"])
                y = float(row["y"])
            except (KeyError, TypeError, ValueError):
                continue

            if math.isfinite(x) and math.isfinite(y):
                points.append([x, y])

    if len(points) < 10:
        raise RuntimeError("Insufficient trajectory points.")

    return np.asarray(points, dtype=np.float64)


def resample_trajectory(points, spacing):
    steps = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1,
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(steps)]
    )

    keep = np.concatenate(
        [[True], np.diff(cumulative) > 1e-9]
    )

    points = points[keep]
    cumulative = cumulative[keep]

    distances = np.arange(
        0,
        cumulative[-1],
        spacing,
    )

    distances = np.append(
        distances,
        cumulative[-1],
    )

    x = np.interp(
        distances,
        cumulative,
        points[:, 0],
    )

    y = np.interp(
        distances,
        cumulative,
        points[:, 1],
    )

    return np.column_stack([x, y])


def load_exterior_segments():
    doc = ezdxf.readfile(DXF_PATH)
    segments = []

    for entity in doc.modelspace():
        layer = entity.dxf.layer.lower()

        if not any(
            term in layer
            for term in EXTERIOR_LAYER_TERMS
        ):
            continue

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

            for a, b in zip(points[:-1], points[1:]):
                segments.append([a, b])

            if entity.closed and len(points) > 2:
                segments.append(
                    [points[-1], points[0]]
                )

    if not segments:
        raise RuntimeError(
            "No exterior segments extracted."
        )

    return np.asarray(
        segments,
        dtype=np.float64,
    )


def load_wall_polygons():
    data = read_json(WALLS_PATH)

    polygons = []

    for item in data["selected_footprints"]:
        polygon = np.asarray(
            item["polygon"],
            dtype=np.float64,
        )

        if (
            polygon.ndim == 2
            and polygon.shape[1] >= 2
            and len(polygon) >= 3
        ):
            polygons.append(
                polygon[:, :2]
            )

    if not polygons:
        raise RuntimeError(
            "No usable wall polygons found."
        )

    return polygons


def load_wall_segments():
    data = read_json(SEGMENTS_PATH)

    segments = []

    for item in data["segments"]:
        start = np.asarray(
            item["start"][:2],
            dtype=np.float64,
        )

        end = np.asarray(
            item["end"][:2],
            dtype=np.float64,
        )

        if np.linalg.norm(end - start) > 0.01:
            segments.append([start, end])

    return np.asarray(
        segments,
        dtype=np.float64,
    )


class GeometryGrid:
    def __init__(
        self,
        exterior_segments,
        wall_polygons,
        wall_segments,
    ):
        self.exterior_segments = exterior_segments
        self.wall_polygons = wall_polygons
        self.wall_segments = wall_segments

        points = exterior_segments.reshape(-1, 2)

        self.minimum = (
            points.min(axis=0) - MARGIN_M
        )

        self.maximum = (
            points.max(axis=0) + MARGIN_M
        )

        dimensions = np.ceil(
            (self.maximum - self.minimum)
            / RESOLUTION_M
        ).astype(int) + 1

        self.width = int(dimensions[0])
        self.height = int(dimensions[1])

        self.exterior = np.zeros(
            (self.height, self.width),
            dtype=np.uint8,
        )

        self.wall_occupancy = np.zeros_like(
            self.exterior
        )

        self.wall_lines = np.zeros_like(
            self.exterior
        )

        self.draw_exterior()
        self.draw_walls()

        self.inside = self.extract_inside()

        self.wall_distance = cv2.distanceTransform(
            255 - self.wall_occupancy,
            cv2.DIST_L2,
            5,
        ) * RESOLUTION_M

    def project(self, points):
        points = np.asarray(
            points,
            dtype=np.float64,
        )

        pixels = np.rint(
            (points - self.minimum)
            / RESOLUTION_M
        ).astype(np.int32)

        pixels[..., 1] = (
            self.height - 1 - pixels[..., 1]
        )

        return pixels

    def draw_exterior(self):
        projected = self.project(
            self.exterior_segments
        )

        for start, end in projected:
            cv2.line(
                self.exterior,
                tuple(start),
                tuple(end),
                255,
                1,
            )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (11, 11),
        )

        self.exterior = cv2.morphologyEx(
            self.exterior,
            cv2.MORPH_CLOSE,
            kernel,
        )

    def draw_walls(self):
        for polygon in self.wall_polygons:
            pixels = self.project(polygon)

            cv2.fillPoly(
                self.wall_occupancy,
                [pixels],
                255,
            )

        projected = self.project(
            self.wall_segments
        )

        for start, end in projected:
            cv2.line(
                self.wall_lines,
                tuple(start),
                tuple(end),
                255,
                1,
            )

        radius = max(
            1,
            round(
                WALL_BUFFER_M
                / RESOLUTION_M
            ),
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        )

        self.wall_occupancy = cv2.dilate(
            self.wall_occupancy,
            kernel,
        )

    def extract_inside(self):
        free = 255 - self.exterior

        count, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                free,
                connectivity=8,
            )
        )

        inside = np.zeros_like(
            free
        )

        for label in range(1, count):
            x, y, w, h, area = stats[label]

            touches_border = (
                x == 0
                or y == 0
                or x + w >= self.width
                or y + h >= self.height
            )

            if not touches_border:
                inside[labels == label] = 255

        return inside

    def sample_mask(self, points, mask):
        pixels = self.project(points)

        x = pixels[:, 0]
        y = pixels[:, 1]

        valid = (
            (x >= 0)
            & (x < self.width)
            & (y >= 0)
            & (y < self.height)
        )

        values = np.zeros(
            len(points),
            dtype=np.float64,
        )

        values[valid] = (
            mask[y[valid], x[valid]]
        )

        return values, valid

    def evaluate(self, points):
        inside, _ = self.sample_mask(
            points,
            self.inside,
        )

        walls, _ = self.sample_mask(
            points,
            self.wall_occupancy,
        )

        clearance, _ = self.sample_mask(
            points,
            self.wall_distance,
        )

        return (
            inside > 0,
            walls > 0,
            clearance,
        )

    def crossing_fraction(self, points):
        # Sample intermediate points along consecutive
        # trajectory segments to detect wall occupancy.
        crossings = 0
        tested = 0

        for start, end in zip(
            points[:-1],
            points[1:],
        ):
            length = np.linalg.norm(
                end - start
            )

            if length < 1e-6:
                continue

            count = max(
                2,
                int(
                    math.ceil(
                        length / CROSSING_SAMPLE_M
                    )
                ),
            )

            t = np.linspace(
                0,
                1,
                count,
            )

            samples = (
                start[None, :]
                + t[:, None]
                * (end - start)[None, :]
            )

            occupancy, _ = self.sample_mask(
                samples,
                self.wall_lines,
            )

            if np.any(occupancy > 0):
                crossings += 1

            tested += 1

        return (
            crossings / tested
            if tested
            else 0.0
        )


def transform_points(points, angle_deg, tx, ty):
    theta = math.radians(angle_deg)

    c = math.cos(theta)
    s = math.sin(theta)

    rotation = np.array(
        [[c, -s], [s, c]],
        dtype=np.float64,
    )

    return (
        points @ rotation.T
        + np.array([tx, ty])
    )


def score_candidate(grid, trajectory, angle, tx, ty):
    transformed = transform_points(
        trajectory,
        angle,
        tx,
        ty,
    )

    inside, occupied, clearance = (
        grid.evaluate(transformed)
    )

    outside_fraction = float(
        (~inside).mean()
    )

    occupancy_fraction = float(
        occupied.mean()
    )

    # Use a vectorized midpoint approximation during search.
    # Exact segment sampling is reserved for final candidates.
    midpoints = (
        transformed[:-1]
        + transformed[1:]
    ) / 2

    midpoint_wall, _ = grid.sample_mask(
        midpoints,
        grid.wall_lines,
    )

    crossing_fraction = float(
        (midpoint_wall > 0).mean()
    )

    mean_clearance = float(
        np.clip(
            clearance,
            0,
            2,
        ).mean()
    )

    score = (
        -WEIGHTS["outside"]
        * outside_fraction
        -WEIGHTS["wall_occupancy"]
        * occupancy_fraction
        -WEIGHTS["wall_crossing"]
        * crossing_fraction
        + WEIGHTS["clearance"]
        * mean_clearance
    )

    return Candidate(
        angle_deg=float(angle),
        tx=float(tx),
        ty=float(ty),
        score=float(score),
        inside_fraction=float(inside.mean()),
        wall_occupancy_fraction=occupancy_fraction,
        crossing_fraction=crossing_fraction,
        mean_clearance_m=mean_clearance,
    )


def search_candidates(grid, trajectory):
    building_center = (
        grid.minimum + grid.maximum
    ) / 2

    trajectory_center = (
        trajectory.min(axis=0)
        + trajectory.max(axis=0)
    ) / 2

    coarse = []

    print("Searching coarse transformations...")

    for angle in np.arange(
        0,
        360,
        COARSE_ROTATION_DEG,
    ):
        rotated_center = transform_points(
            trajectory_center.reshape(1, 2),
            angle,
            0,
            0,
        )[0]

        base = (
            building_center - rotated_center
        )

        for dx in np.arange(
            -12,
            12.01,
            COARSE_TRANSLATION_M,
        ):
            for dy in np.arange(
                -12,
                12.01,
                COARSE_TRANSLATION_M,
            ):
                t = base + [dx, dy]

                coarse.append(
                    score_candidate(
                        grid,
                        trajectory,
                        angle,
                        t[0],
                        t[1],
                    )
                )

    coarse.sort(
        key=lambda c: c.score,
        reverse=True,
    )

    print("Refining top transformations...")

    refined = []

    for seed in coarse[:20]:
        for da in np.arange(
            -5,
            5.01,
            FINE_ROTATION_DEG,
        ):
            for dx in np.arange(
                -1,
                1.01,
                FINE_TRANSLATION_M,
            ):
                for dy in np.arange(
                    -1,
                    1.01,
                    FINE_TRANSLATION_M,
                ):
                    refined.append(
                        score_candidate(
                            grid,
                            trajectory,
                            seed.angle_deg + da,
                            seed.tx + dx,
                            seed.ty + dy,
                        )
                    )

    refined.sort(
        key=lambda c: c.score,
        reverse=True,
    )

    selected = []

    for candidate in refined:
        distinct = all(
            abs(
                (
                    candidate.angle_deg
                    - existing.angle_deg
                    + 180
                ) % 360 - 180
            ) >= 5
            or math.hypot(
                candidate.tx - existing.tx,
                candidate.ty - existing.ty,
            ) >= 2
            for existing in selected
        )

        if distinct:
            selected.append(candidate)

        if len(selected) >= TOP_K:
            break

    return selected


def render_candidate(grid, trajectory, candidate, rank):
    canvas = np.zeros(
        (grid.height, grid.width, 3),
        dtype=np.uint8,
    )

    canvas[grid.inside > 0] = (
        35,
        38,
        43,
    )

    canvas[grid.wall_occupancy > 0] = (
        70,
        85,
        110,
    )

    canvas[grid.exterior > 0] = (
        180,
        185,
        195,
    )

    transformed = transform_points(
        trajectory,
        candidate.angle_deg,
        candidate.tx,
        candidate.ty,
    )

    pixels = grid.project(
        transformed
    )

    cv2.polylines(
        canvas,
        [pixels],
        False,
        (45, 165, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"Candidate #{rank}",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return canvas


def main():
    print("Loading existing CAD geometry...")

    exterior = load_exterior_segments()
    wall_polygons = load_wall_polygons()
    wall_segments = load_wall_segments()

    opening_audit = read_json(
        OPENINGS_PATH
    )

    print(
        "Selected wall polygons:",
        len(wall_polygons),
    )

    print(
        "Normalized wall segments:",
        len(wall_segments),
    )

    print(
        "Unverified opening-adjacent endpoints:",
        opening_audit.get(
            "endpoint_classifications",
            {},
        ).get("near_opening_geometry", 0),
    )

    original = load_trajectory()

    trajectory = resample_trajectory(
        original,
        TRAJECTORY_SPACING_M,
    )

    grid = GeometryGrid(
        exterior,
        wall_polygons,
        wall_segments,
    )

    candidates = search_candidates(
        grid,
        trajectory,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "status": "CANDIDATE_ONLY",
        "verified": False,
        "method": "interior_aware_rigid_search",
        "selected_wall_polygons": len(wall_polygons),
        "normalized_wall_segments": len(wall_segments),
        "candidates": [],
        "limitations": [
            "Wall footprints are reconstructed, not verified BIM.",
            "Door/window audit does not verify traversable openings.",
            "Scores are heuristic and not calibrated probabilities.",
            "No independent camera-to-CAD correspondences.",
        ],
    }

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        transformed = transform_points(
            trajectory,
            candidate.angle_deg,
            candidate.tx,
            candidate.ty,
        )

        exact_crossing_fraction = (
            grid.crossing_fraction(transformed)
        )

        print(
            f"#{rank}: "
            f"angle={candidate.angle_deg:.1f}deg "
            f"tx={candidate.tx:.2f} "
            f"ty={candidate.ty:.2f} "
            f"inside={candidate.inside_fraction:.1%} "
            f"wall={candidate.wall_occupancy_fraction:.1%} "
            f"crossings={exact_crossing_fraction:.1%} "
            f"score={candidate.score:.3f}"
        )

        report["candidates"].append(
            {
                "rank": rank,
                "angle_deg": candidate.angle_deg,
                "tx": candidate.tx,
                "ty": candidate.ty,
                "score": candidate.score,
                "inside_fraction": candidate.inside_fraction,
                "wall_occupancy_fraction": (
                    candidate.wall_occupancy_fraction
                ),
                "sampled_crossing_fraction": (
                    exact_crossing_fraction
                ),
                "mean_clearance_m": (
                    candidate.mean_clearance_m
                ),
            }
        )

        image = render_candidate(
            grid,
            original,
            candidate,
            rank,
        )

        cv2.imwrite(
            str(
                OUTPUT_DIR
                / f"candidate_{rank}.png"
            ),
            image,
        )

    report_path = (
        OUTPUT_DIR
        / "registration_candidates.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved results:")
    print(OUTPUT_DIR)
    print(
        "\nRegistration remains CANDIDATE_ONLY "
        "until independently validated."
    )


if __name__ == "__main__":
    main()
