
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

DXF_PATH = ROOT / "data/floorplans/floor_2.dxf"
CANDIDATES_PATH = (
    ROOT / "outputs/automatic_registration_v2"
    / "registration_candidates.json"
)
OUTPUT_DIR = ROOT / "outputs/landmark_validation"

MAP_W = 650
MAP_H = 570
PHOTO_W = 500
PHOTO_H = 570
HEADER_H = 55
FOOTER_H = 90
MARGIN = 35

WINDOW = "Under-construction | Independent registration validation"

BG = (24, 27, 32)
WHITE = (235, 235, 235)
GRAY = (125, 135, 145)
ORANGE = (45, 155, 255)
GREEN = (90, 210, 125)
BLUE = (255, 170, 65)

MAX_FRAME_TIME_DIFFERENCE_S = 0.30


def find_manifest():
    matches = [
        p for p in ROOT.rglob("frame_manifest.csv")
        if SEQUENCE in str(p) and "cam0" in p.parts
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one matching manifest, found {len(matches)}"
        )

    return matches[0]


def load_frames():
    manifest = find_manifest()
    frame_dir = manifest.parent / "frames"

    records = []

    with manifest.open(
        encoding="utf-8-sig",
        newline="",
    ) as f:
        for row in csv.DictReader(f):
            if row.get("pose_matched", "").lower() != "true":
                continue

            try:
                record = {
                    "filename": row["filename"],
                    "timestamp": float(row["header_timestamp_s"]),
                    "xy": np.array(
                        [float(row["tx"]), float(row["ty"])],
                        dtype=np.float64,
                    ),
                    "path": frame_dir / row["filename"],
                }
            except (KeyError, TypeError, ValueError):
                continue

            if record["path"].exists():
                records.append(record)

    records.sort(key=lambda r: r["timestamp"])

    if not records:
        raise RuntimeError("No matched camera frames found.")

    return records


def load_candidates():
    data = json.loads(
        CANDIDATES_PATH.read_text(encoding="utf-8")
    )

    candidates = data.get("candidates", [])

    if not candidates:
        raise RuntimeError("No registration candidates found.")

    return candidates


def load_dxf_segments():
    doc = ezdxf.readfile(DXF_PATH)
    segments = []

    for entity in doc.modelspace():
        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end
            segments.append([[a.x, a.y], [b.x, b.y]])

        elif kind == "LWPOLYLINE":
            pts = [
                [p[0], p[1]]
                for p in entity.get_points("xy")
            ]

            segments.extend(
                [[a, b] for a, b in zip(pts[:-1], pts[1:])]
            )

            if entity.closed and len(pts) > 2:
                segments.append([pts[-1], pts[0]])

    return np.asarray(segments, dtype=np.float64)


class CADViewport:
    def __init__(self, segments):
        self.segments = segments

        points = segments.reshape(-1, 2)
        self.minimum = points.min(axis=0)
        maximum = points.max(axis=0)

        extent = maximum - self.minimum

        self.scale = min(
            (MAP_W - 2 * MARGIN) / extent[0],
            (MAP_H - 2 * MARGIN) / extent[1],
        )

        used = extent * self.scale
        self.offset = np.array(
            [
                (MAP_W - used[0]) / 2,
                (MAP_H - used[1]) / 2,
            ]
        )

    def project(self, points):
        p = np.asarray(points, dtype=np.float64)
        out = np.empty_like(p)

        out[..., 0] = (
            self.offset[0]
            + (p[..., 0] - self.minimum[0]) * self.scale
        )
        out[..., 1] = (
            MAP_H - self.offset[1]
            - (p[..., 1] - self.minimum[1]) * self.scale
        )

        return np.rint(out).astype(np.int32)

    def unproject(self, x, y):
        return np.array(
            [
                self.minimum[0] + (x - self.offset[0]) / self.scale,
                self.minimum[1]
                + (MAP_H - self.offset[1] - y) / self.scale,
            ],
            dtype=np.float64,
        )


def transform(xy, candidate):
    theta = math.radians(candidate["angle_deg"])
    c = math.cos(theta)
    s = math.sin(theta)

    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    return np.asarray(xy) @ R.T + [
        candidate["tx"],
        candidate["ty"],
    ]


def fit_image(image, width, height):
    canvas = np.full((height, width, 3), BG, dtype=np.uint8)

    if image is None:
        return canvas

    h, w = image.shape[:2]
    scale = min(width / w, height / h)

    resized = cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
    )

    rh, rw = resized.shape[:2]
    x = (width - rw) // 2
    y = (height - rh) // 2

    canvas[y:y + rh, x:x + rw] = resized

    return canvas


class Validator:
    def __init__(self):
        self.frames = load_frames()
        self.candidates = load_candidates()
        self.segments = load_dxf_segments()
        self.viewport = CADViewport(self.segments)

        self.index = 0
        self.observations = []
        self.status = (
            "Select a camera location only if independently known."
        )

    def current_frame(self):
        return self.frames[self.index]

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if not (
            0 <= x < MAP_W
            and HEADER_H <= y < HEADER_H + MAP_H
        ):
            return

        point = self.viewport.unproject(
            x, y - HEADER_H
        )

        frame = self.current_frame()

        self.observations.append(
            {
                "filename": frame["filename"],
                "timestamp": frame["timestamp"],
                "trajectory_xy": frame["xy"].tolist(),
                "cad_xy": point.tolist(),
                "evidence_type": "human_verified_camera_location",
            }
        )

        self.status = (
            f"Recorded anchor #{len(self.observations)} "
            f"for {frame['filename']}"
        )

        print(self.status)

    def draw(self):
        width = MAP_W + PHOTO_W
        height = HEADER_H + MAP_H + FOOTER_H

        canvas = np.full(
            (height, width, 3),
            BG,
            dtype=np.uint8,
        )

        map_panel = canvas[
            HEADER_H:HEADER_H + MAP_H,
            :MAP_W,
        ]

        projected = self.viewport.project(self.segments)

        for a, b in projected:
            cv2.line(
                map_panel,
                tuple(a),
                tuple(b),
                GRAY,
                1,
                cv2.LINE_AA,
            )

        frame = self.current_frame()

        for candidate in self.candidates:
            predicted = transform(
                frame["xy"],
                candidate,
            )

            p = self.viewport.project(
                predicted.reshape(1, 2)
            )[0]

            color = (
                ORANGE if candidate["rank"] == 1 else BLUE
            )

            cv2.circle(
                map_panel,
                tuple(p),
                6,
                color,
                2,
            )

            cv2.putText(
                map_panel,
                f"#{candidate['rank']}",
                tuple(p + [8, -8]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                WHITE,
                1,
            )

        for observation in self.observations:
            p = self.viewport.project(
                np.asarray([observation["cad_xy"]])
            )[0]

            cv2.circle(
                map_panel,
                tuple(p),
                5,
                GREEN,
                -1,
            )

        photo = cv2.imread(
            str(frame["path"])
        )

        fitted = fit_image(
            photo,
            PHOTO_W,
            PHOTO_H,
        )

        canvas[
            HEADER_H:HEADER_H + PHOTO_H,
            MAP_W:MAP_W + PHOTO_W,
        ] = fitted

        cv2.putText(
            canvas,
            "CAD: predicted locations / independent anchor",
            (16, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            WHITE,
            2,
        )

        cv2.putText(
            canvas,
            frame["filename"],
            (MAP_W + 15, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            WHITE,
            2,
        )

        cv2.putText(
            canvas,
            self.status[:95],
            (16, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.53,
            WHITE,
            1,
        )

        cv2.putText(
            canvas,
            "N: next | P: previous | Click CAD: anchor | U: undo | E: evaluate | S: save | Q: quit",
            (16, height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            GRAY,
            1,
        )

        return canvas

    def evaluate(self):
        if not self.observations:
            self.status = "No independent anchors recorded."
            print(self.status)
            return

        print("\nINDEPENDENT ANCHOR ERRORS")

        for candidate in self.candidates:
            errors = []

            for observation in self.observations:
                predicted = transform(
                    observation["trajectory_xy"],
                    candidate,
                )

                actual = np.asarray(
                    observation["cad_xy"],
                    dtype=np.float64,
                )

                errors.append(
                    float(np.linalg.norm(predicted - actual))
                )

            rmse = math.sqrt(
                sum(e * e for e in errors) / len(errors)
            )

            print(
                f"Candidate #{candidate['rank']}: "
                f"RMSE={rmse:.3f} m | "
                f"errors={[round(e, 3) for e in errors]}"
            )

        self.status = "Evaluation printed to terminal."

    def save(self):
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = {
            "status": "INDEPENDENT_ANCHORS_UNREVIEWED",
            "verified": False,
            "observations": self.observations,
            "candidate_source": str(CANDIDATES_PATH),
            "limitations": [
                "Human-selected locations require external evidence.",
                "Predicted locations are not ground truth.",
                "No image-pixel-to-CAD geometric projection.",
            ],
        }

        path = OUTPUT_DIR / "independent_landmarks.json"

        path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        self.status = f"Saved {path.name}"
        print(self.status)

    def run(self):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        print("Navigate frames with N/P.")
        print(
            "Click CAD only when you can independently identify "
            "the camera's actual location."
        )

        while True:
            cv2.imshow(WINDOW, self.draw())

            key = cv2.waitKey(30) & 0xFF

            if key == ord("n"):
                self.index = min(
                    self.index + 1,
                    len(self.frames) - 1,
                )

            elif key == ord("p"):
                self.index = max(0, self.index - 1)

            elif key == ord("u"):
                if self.observations:
                    self.observations.pop()

            elif key == ord("e"):
                self.evaluate()

            elif key == ord("s"):
                self.save()

            elif key == ord("q"):
                break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    Validator().run()
