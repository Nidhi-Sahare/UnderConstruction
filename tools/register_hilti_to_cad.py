
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
)

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"

TRAJECTORY_PATH = (
    ARCHIVE
    / "floorplan_registration"
    / "aligned_trajectory.csv"
)

def find_manifest():
    candidates = list(ROOT.rglob("frame_manifest.csv"))

    candidates = [
        p for p in candidates
        if "floor_2_2025-10-28_run_2" in str(p)
        and "cam0" in p.parts
    ]

    if not candidates:
        raise FileNotFoundError(
            "Cannot find the Hilti cam0 frame_manifest.csv "
            "anywhere inside BuildProgress."
        )

    if len(candidates) > 1:
        print("Multiple manifests found:")
        for p in candidates:
            print(" -", p)
        raise RuntimeError(
            "Multiple matching manifests found. "
            "Set MANIFEST_PATH explicitly."
        )

    print("Using manifest:", candidates[0])
    return candidates[0]


MANIFEST_PATH = find_manifest()

FRAME_DIR = MANIFEST_PATH.parent / "frames"

OUTPUT_DIR = ROOT / "outputs/cad_registration"

PANEL_W = 470
PANEL_H = 440
IMAGE_W = 360
HEADER_H = 48
FOOTER_H = 70
MARGIN = 24

CANVAS_W = PANEL_W * 2 + IMAGE_W
CANVAS_H = HEADER_H + PANEL_H + FOOTER_H

WINDOW = "Under-construction | Camera-assisted CAD registration"

BG = (24, 27, 32)
WHITE = (235, 235, 235)
MUTED = (155, 160, 168)
BLUE = (255, 170, 65)
ORANGE = (45, 155, 255)
GREEN = (100, 215, 120)
RED = (80, 90, 235)


def load_trajectory():
    points = []
    times = []

    with TRAJECTORY_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.DictReader(f):
            try:
                x = float(row["x"])
                y = float(row["y"])
                t = float(row["timestamp"])
            except (KeyError, TypeError, ValueError):
                continue

            if all(map(math.isfinite, (x, y, t))):
                points.append([x, y])
                times.append(t)

    if not points:
        raise RuntimeError("No trajectory positions loaded.")

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
    )


def load_manifest():
    records = []

    with MANIFEST_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        for row in csv.DictReader(f):
            try:
                matched = (
                    str(row.get("pose_matched", "")).lower()
                    == "true"
                )

                if not matched:
                    continue

                timestamp = float(row["header_timestamp_s"])
                x = float(row["tx"])
                y = float(row["ty"])

                path = FRAME_DIR / row["filename"]

                if path.exists():
                    records.append(
                        {
                            "timestamp": timestamp,
                            "xy": np.array(
                                [x, y],
                                dtype=np.float64,
                            ),
                            "path": path,
                            "filename": row["filename"],
                        }
                    )
            except (TypeError, ValueError, KeyError):
                continue

    if not records:
        raise RuntimeError(
            "No pose-matched camera photographs found."
        )

    return records


def load_dxf_segments():
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
            pts = [
                [p[0], p[1]]
                for p in entity.get_points("xy")
            ]

            for a, b in zip(pts[:-1], pts[1:]):
                segments.append([a, b])

            if entity.closed and len(pts) > 2:
                segments.append([pts[-1], pts[0]])

    if not segments:
        raise RuntimeError("No DXF geometry found.")

    return np.asarray(segments, dtype=np.float64)


class Viewport:
    def __init__(self, points, width, height):
        self.width = width
        self.height = height

        points = np.asarray(points, dtype=np.float64)

        self.minimum = points.min(axis=0)
        maximum = points.max(axis=0)

        extent = np.maximum(maximum - self.minimum, 0.001)

        self.scale = min(
            (width - 2 * MARGIN) / extent[0],
            (height - 2 * MARGIN) / extent[1],
        )

        used = extent * self.scale

        self.offset_x = (width - used[0]) / 2
        self.offset_y = (height - used[1]) / 2

    def project(self, points):
        p = np.asarray(points, dtype=np.float64)

        out = np.empty_like(p)

        out[..., 0] = (
            self.offset_x
            + (p[..., 0] - self.minimum[0]) * self.scale
        )

        out[..., 1] = (
            self.height
            - self.offset_y
            - (p[..., 1] - self.minimum[1]) * self.scale
        )

        return np.rint(out).astype(np.int32)

    def unproject(self, pixel):
        x, y = pixel

        return np.array(
            [
                self.minimum[0]
                + (x - self.offset_x) / self.scale,
                self.minimum[1]
                + (
                    self.height - self.offset_y - y
                ) / self.scale,
            ],
            dtype=np.float64,
        )


def rigid_fit(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if len(source) < 2:
        raise ValueError("At least two correspondences required.")

    src_mean = source.mean(axis=0)
    dst_mean = target.mean(axis=0)

    X = source - src_mean
    Y = target - dst_mean

    U, _, Vt = np.linalg.svd(X.T @ Y)

    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = dst_mean - rotation @ src_mean

    predicted = source @ rotation.T + translation

    errors = np.linalg.norm(predicted - target, axis=1)

    matrix = np.column_stack(
        [rotation, translation]
    )

    return matrix, errors


class RegistrationInspector:
    def __init__(self):
        self.trajectory, self.timestamps = load_trajectory()
        self.frames = load_manifest()
        self.segments = load_dxf_segments()

        self.trajectory_view = Viewport(
            self.trajectory,
            PANEL_W,
            PANEL_H,
        )

        self.cad_view = Viewport(
            self.segments.reshape(-1, 2),
            PANEL_W,
            PANEL_H,
        )

        self.selected_index = None
        self.selected_frame = None

        self.fit_pairs = []
        self.validation_pairs = []

        self.validation_mode = False

        self.matrix = None
        self.fit_errors = None
        self.validation_errors = None

        self.image_cache = {}

        self.status = "Select a point on the trajectory."

    def closest_frame(self, timestamp):
        differences = [
            abs(record["timestamp"] - timestamp)
            for record in self.frames
        ]

        index = int(np.argmin(differences))
        return self.frames[index]

    def get_image(self, path):
        key = str(path)

        if key not in self.image_cache:
            image = cv2.imread(key)

            if image is None:
                return None

            self.image_cache[key] = image

            if len(self.image_cache) > 12:
                first = next(iter(self.image_cache))
                del self.image_cache[first]

        return self.image_cache[key]

    def select_trajectory(self, pixel):
        xy = self.trajectory_view.unproject(pixel)

        distances = np.sum(
            (self.trajectory - xy) ** 2,
            axis=1,
        )

        index = int(np.argmin(distances))

        self.selected_index = index
        self.selected_frame = self.closest_frame(
            self.timestamps[index]
        )

        dt = abs(
            self.selected_frame["timestamp"]
            - self.timestamps[index]
        )

        self.status = (
            f"Trajectory #{index} | "
            f"Photo: {self.selected_frame['filename']} | "
            f"Time difference: {dt:.3f}s"
        )

        print(self.status)

    def add_correspondence(self, pixel):
        if self.selected_index is None:
            self.status = "Select a trajectory point first."
            return

        source = self.trajectory[self.selected_index].copy()
        target = self.cad_view.unproject(pixel)

        pair = {
            "trajectory_index": self.selected_index,
            "timestamp": float(
                self.timestamps[self.selected_index]
            ),
            "source": source.tolist(),
            "target": target.tolist(),
            "image": self.selected_frame["filename"],
        }

        if self.validation_mode:
            self.validation_pairs.append(pair)
            group = "validation"
        else:
            self.fit_pairs.append(pair)
            group = "fitting"

            self.matrix = None
            self.fit_errors = None
            self.validation_errors = None

        self.status = (
            f"Added {group} correspondence "
            f"#{len(self.validation_pairs if self.validation_mode else self.fit_pairs)}"
        )

        self.selected_index = None
        self.selected_frame = None

        print(self.status)

    def estimate(self):
        if len(self.fit_pairs) < 3:
            self.status = (
                "Select at least three fitting correspondences."
            )
            return

        source = [
            pair["source"]
            for pair in self.fit_pairs
        ]

        target = [
            pair["target"]
            for pair in self.fit_pairs
        ]

        self.matrix, self.fit_errors = rigid_fit(
            source,
            target,
        )

        if self.validation_pairs:
            validation_source = np.asarray(
                [p["source"] for p in self.validation_pairs]
            )

            validation_target = np.asarray(
                [p["target"] for p in self.validation_pairs]
            )

            predicted = (
                validation_source @ self.matrix[:, :2].T
                + self.matrix[:, 2]
            )

            self.validation_errors = np.linalg.norm(
                predicted - validation_target,
                axis=1,
            )
        else:
            self.validation_errors = None

        self.status = (
            f"Candidate fit RMSE: "
            f"{np.sqrt(np.mean(self.fit_errors ** 2)):.2f}m"
        )

        print("\nCandidate rigid transformation:")
        print(self.matrix)
        print(self.status)

        if self.validation_errors is not None:
            print(
                "Independent validation RMSE:",
                float(
                    np.sqrt(
                        np.mean(
                            self.validation_errors ** 2
                        )
                    )
                ),
            )

    def draw_trajectory(self, image):
        projected = self.trajectory_view.project(
            self.trajectory
        )

        cv2.polylines(
            image,
            [projected],
            False,
            BLUE,
            2,
            cv2.LINE_AA,
        )

        for i, pair in enumerate(self.fit_pairs):
            p = self.trajectory_view.project(
                np.asarray([pair["source"]])
            )[0]

            cv2.circle(image, tuple(p), 6, GREEN, -1)

            cv2.putText(
                image,
                str(i + 1),
                tuple(p + [9, -9]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                1,
            )

        if self.selected_index is not None:
            p = projected[self.selected_index]

            cv2.circle(
                image,
                tuple(p),
                10,
                ORANGE,
                2,
            )

    def draw_cad(self, image):
        projected = self.cad_view.project(
            self.segments
        )

        for segment in projected:
            cv2.line(
                image,
                tuple(segment[0]),
                tuple(segment[1]),
                MUTED,
                1,
                cv2.LINE_AA,
            )

        if self.matrix is not None:
            transformed = (
                self.trajectory @ self.matrix[:, :2].T
                + self.matrix[:, 2]
            )

            projected_path = self.cad_view.project(
                transformed
            )

            cv2.polylines(
                image,
                [projected_path],
                False,
                ORANGE,
                2,
                cv2.LINE_AA,
            )

        for i, pair in enumerate(self.fit_pairs):
            p = self.cad_view.project(
                np.asarray([pair["target"]])
            )[0]

            cv2.circle(image, tuple(p), 6, GREEN, -1)

            cv2.putText(
                image,
                str(i + 1),
                tuple(p + [9, -9]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                1,
            )

    def draw_camera(self, canvas):
        x0 = PANEL_W * 2
        y0 = HEADER_H

        panel = canvas[
            y0:y0 + PANEL_H,
            x0:x0 + IMAGE_W,
        ]

        if self.selected_frame is None:
            cv2.putText(
                panel,
                "Select trajectory point",
                (35, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                WHITE,
                1,
            )
            return

        image = self.get_image(
            self.selected_frame["path"]
        )

        if image is None:
            return

        ih, iw = image.shape[:2]

        scale = min(
            (IMAGE_W - 20) / iw,
            (PANEL_H - 65) / ih,
        )

        resized = cv2.resize(
            image,
            (
                max(1, int(iw * scale)),
                max(1, int(ih * scale)),
            ),
        )

        rh, rw = resized.shape[:2]

        px = (IMAGE_W - rw) // 2
        py = 45 + (PANEL_H - 65 - rh) // 2

        panel[py:py + rh, px:px + rw] = resized

        cv2.putText(
            panel,
            self.selected_frame["filename"],
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WHITE,
            1,
        )

    def draw(self):
        canvas = np.full(
            (CANVAS_H, CANVAS_W, 3),
            BG,
            dtype=np.uint8,
        )

        left = canvas[
            HEADER_H:HEADER_H + PANEL_H,
            :PANEL_W,
        ]

        middle = canvas[
            HEADER_H:HEADER_H + PANEL_H,
            PANEL_W:PANEL_W * 2,
        ]

        self.draw_trajectory(left)
        self.draw_cad(middle)
        self.draw_camera(canvas)

        titles = [
            ("TRAJECTORY", 18),
            ("CAD FLOOR PLAN", PANEL_W + 18),
            ("CAMERA PHOTO", PANEL_W * 2 + 18),
        ]

        for title, x in titles:
            cv2.putText(
                canvas,
                title,
                (x, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                WHITE,
                2,
            )

        cv2.putText(
            canvas,
            self.status[:110],
            (20, CANVAS_H - 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1,
        )

        mode = (
            "VALIDATION"
            if self.validation_mode
            else "FITTING"
        )

        controls = (
            f"Mode: {mode} | "
            "Click trajectory > CAD | "
            "R: fit | V: mode | "
            "U: undo | S: save | Q: quit"
        )

        cv2.putText(
            canvas,
            controls,
            (20, CANVAS_H - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            MUTED,
            1,
        )

        return canvas

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not (
            HEADER_H <= y < HEADER_H + PANEL_H
        ):
            return

        local_y = y - HEADER_H

        if 0 <= x < PANEL_W:
            self.select_trajectory(
                (x, local_y)
            )

        elif PANEL_W <= x < PANEL_W * 2:
            self.add_correspondence(
                (x - PANEL_W, local_y)
            )

    def undo(self):
        if self.selected_index is not None:
            self.selected_index = None
            self.selected_frame = None
            self.status = "Selection cancelled."
            return

        pairs = (
            self.validation_pairs
            if self.validation_mode
            else self.fit_pairs
        )

        if pairs:
            pairs.pop()

            self.matrix = None
            self.fit_errors = None
            self.validation_errors = None

            self.status = "Last correspondence removed."

    def save(self):
        if self.matrix is None:
            self.status = "Estimate registration first (R)."
            return

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "status": "CANDIDATE_ONLY",
            "verified": False,
            "transform": self.matrix.tolist(),
            "fit_landmarks": self.fit_pairs,
            "validation_landmarks": self.validation_pairs,
            "fit_errors_m": self.fit_errors.tolist(),
            "validation_errors_m": (
                self.validation_errors.tolist()
                if self.validation_errors is not None
                else None
            ),
            "limitations": [
                "Correspondences require human verification.",
                "A small fitting error does not prove correctness.",
                "No automatic camera-to-CAD localization.",
            ],
        }

        path = (
            OUTPUT_DIR
            / "camera_assisted_registration.json"
        )

        path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        cv2.imwrite(
            str(OUTPUT_DIR / "registration_preview.png"),
            self.draw(),
        )

        self.status = f"Candidate saved: {path.name}"

        print(self.status)

    def run(self):
        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_AUTOSIZE,
        )

        cv2.setMouseCallback(
            WINDOW,
            self.on_mouse,
        )

        print(
            "Click trajectory position, inspect camera image, "
            "then click corresponding CAD location."
        )

        while True:
            cv2.imshow(
                WINDOW,
                self.draw(),
            )

            key = cv2.waitKey(30) & 0xFF

            if key == ord("r"):
                self.estimate()

            elif key == ord("v"):
                self.validation_mode = (
                    not self.validation_mode
                )

                self.status = (
                    "Validation mode"
                    if self.validation_mode
                    else "Fitting mode"
                )

            elif key == ord("u"):
                self.undo()

            elif key == ord("s"):
                self.save()

            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    RegistrationInspector().run()
