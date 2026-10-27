
from pathlib import Path
import json
import math

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY = ROOT / "data/real_samples/floor_2_camera_trajectory.json"
CANDIDATES = (
    ROOT
    / "_archive/hilti_transition_20261023_004237"
    / "outputs/registration/candidate_alignments.json"
)
WALLS = ROOT / "data/geometry/floor_2_wall_meshes.json"
CALIBRATION = ROOT / "data/hilti_calibration/kalibr_imucam_chain.yaml"
FRAMES = ROOT / "sample_gallery/real_floor_2/frames"

OUTPUT = ROOT / "outputs/sitewitness/pose_hypotheses"

FRAME_INDICES = [53, 68, 90, 129]
CANDIDATE_ANGLE = 317.9689499733972
CANDIDATE_TRANSLATION = [-47.29695344668649, 26.84741454935609]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quaternion_matrix(q):
    x, y, z, w = np.asarray(q, dtype=float)
    n = x*x + y*y + z*z + w*w

    if n < 1e-12:
        raise ValueError("Invalid quaternion")

    s = 2.0 / n

    return np.array([
        [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
        [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
        [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)]
    ], dtype=float)


def homogeneous(R, t):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform(T, points):
    points = np.asarray(points, dtype=float)
    return points @ T[:3, :3].T + T[:3, 3]



def load_camera():
    # OpenCV calibration files use %YAML:1.0,
    # which PyYAML cannot parse directly.
    raw = CALIBRATION.read_text(encoding="utf-8-sig")

    lines = raw.splitlines()

    if lines and lines[0].startswith("%YAML:"):
        lines = lines[1:]

    data = yaml.safe_load("\n".join(lines))

    if not isinstance(data, dict) or "cam0" not in data:
        raise ValueError("Camera calibration is missing cam0")

    cam = data["cam0"]

    fx, fy, cx, cy = cam["intrinsics"]

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    D = np.asarray(
        cam["distortion_coeffs"],
        dtype=np.float64
    ).reshape(4, 1)

    T_cam_imu = np.asarray(
        cam["T_cam_imu"],
        dtype=np.float64
    )

    if T_cam_imu.shape != (4, 4):
        raise ValueError("Invalid camera-to-IMU transformation")

    return K, D, T_cam_imu

def load_candidate():
    candidates = read_json(CANDIDATES)["candidates"]

    for candidate in candidates:
        angle = float(candidate["angle_degrees"])
        translation = np.asarray(candidate["translation"], dtype=float)

        difference = abs(
            (angle - CANDIDATE_ANGLE + 180) % 360 - 180
        )

        if (
            difference < 0.01
            and np.allclose(
                translation,
                CANDIDATE_TRANSLATION,
                atol=0.01
            )
        ):
            return candidate

    raise RuntimeError("Existing Candidate 04 was not found")


def cad_from_recorded(candidate):
    theta = math.radians(float(candidate["angle_degrees"]))
    c, s = math.cos(theta), math.sin(theta)

    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ], dtype=float)

    tx, ty = candidate["translation"]

    return homogeneous(R, [tx, ty, 0])


def load_wall_edges():
    data = read_json(WALLS)
    edges = []

    for wall in data["walls"]:
        mesh = wall.get("mesh", {})
        vertices = np.asarray(mesh.get("vertices_xyz", []), dtype=float)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            continue

        # Three.js coordinates: X=CAD X, Y=height, Z=CAD Y.
        vertices = vertices[:, [0, 2, 1]]

        # A provisional rectangular wall mesh normally has
        # eight corners. Connect pairs sharing two coordinates.
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                delta = np.abs(vertices[i] - vertices[j])

                if np.count_nonzero(delta > 1e-6) == 1:
                    edges.append((vertices[i], vertices[j]))

    if not edges:
        raise RuntimeError("No usable wall edges found")

    return edges



def load_poses():
    data = read_json(TRAJECTORY)
    frames = data["frames"]

    return {
        int(frame["index"]): frame
        for frame in frames
    }


def extract_pose(record):
    if not record.get("pose_available", False):
        return None

    position = record.get("position")
    orientation = record.get("orientation")

    if not isinstance(position, dict):
        return None

    if not isinstance(orientation, dict):
        return None

    try:
        p = np.array([
            position["x"],
            position["y"],
            position["z"]
        ], dtype=np.float64)

        q = np.array([
            orientation["x"],
            orientation["y"],
            orientation["z"],
            orientation["w"]
        ], dtype=np.float64)

    except (KeyError, TypeError, ValueError):
        return None

    if not np.all(np.isfinite(p)):
        return None

    if not np.all(np.isfinite(q)):
        return None

    if np.linalg.norm(q) < 1e-8:
        return None

    return p, q / np.linalg.norm(q)



def camera_pose(position, quaternion, T_cam_imu, hypothesis):
    R = quaternion_matrix(quaternion)

    if hypothesis.startswith("world_to"):
        R = R.T

    T_world_sensor = homogeneous(R, position)

    if hypothesis.endswith("_imu"):
        T_imu_cam = np.linalg.inv(T_cam_imu)
        return T_world_sensor @ T_imu_cam

    return T_world_sensor


def project_edges(image, edges, T_cam_cad, K, D):
    height, width = image.shape[:2]
    overlay = image.copy()

    visible = 0
    in_front = 0

    for a, b in edges:
        camera_points = transform(
            T_cam_cad,
            np.stack([a, b])
        )

        if np.any(camera_points[:, 2] <= 0.05):
            continue

        in_front += 1

        projected, _ = cv2.fisheye.projectPoints(
            camera_points.reshape(-1, 1, 3),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K,
            D
        )

        pixels = projected.reshape(2, 2)

        if not np.all(np.isfinite(pixels)):
            continue

        if np.any(np.abs(pixels) > 100000):
            continue

        p0, p1 = [
            tuple(np.round(p).astype(int))
            for p in pixels
        ]

        ok, p0, p1 = cv2.clipLine(
            (0, 0, width, height),
            p0,
            p1
        )

        if not ok:
            continue

        cv2.line(
            overlay,
            p0,
            p1,
            (0, 180, 255),
            2,
            cv2.LINE_AA
        )
        visible += 1

    result = cv2.addWeighted(
        overlay, 0.8, image, 0.2, 0
    )

    return result, in_front, visible


def label_image(image, label, visible):
    cv2.rectangle(
        image, (0, 0), (image.shape[1], 110),
        (20, 20, 20), -1
    )

    cv2.putText(
        image, label, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2
    )

    cv2.putText(
        image,
        f"Visible edges: {visible} | UNVERIFIED",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 190, 255), 2
    )

    return image


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    candidate = load_candidate()
    K, D, T_cam_imu = load_camera()
    poses = load_poses()
    edges = load_wall_edges()

    T_cad_recorded = cad_from_recorded(candidate)

    hypotheses = [
        "sensor_to_world_camera",
        "sensor_to_world_imu",
        "world_to_sensor_camera",
        "world_to_sensor_imu"
    ]

    results = []

    for frame_index in FRAME_INDICES:
        image_path = FRAMES / f"cam0_{frame_index:05d}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            print("Missing image:", image_path)
            continue

        record = poses.get(frame_index)

        if record is None:
            print("Missing pose:", frame_index)
            continue

        pose = extract_pose(record)

        if pose is None:
            print("Invalid pose:", frame_index)
            continue

        position, quaternion = pose
        panels = []

        for hypothesis in hypotheses:
            T_recorded_cam = camera_pose(
                position, quaternion, T_cam_imu, hypothesis
            )

            T_cad_cam = T_cad_recorded @ T_recorded_cam
            T_cam_cad = np.linalg.inv(T_cad_cam)

            rendered, in_front, visible = project_edges(
                image, edges, T_cam_cad, K, D
            )

            rendered = label_image(
                rendered, hypothesis, visible
            )

            panels.append(
                cv2.resize(rendered, (736, 720))
            )

            results.append({
                "frame": frame_index,
                "hypothesis": hypothesis,
                "edges_in_front": in_front,
                "edges_visible": visible,
                "status": "UNVERIFIED"
            })

            print(
                f"Frame {frame_index:03d} | "
                f"{hypothesis:24s} | "
                f"front={in_front:4d} | visible={visible:4d}"
            )

        comparison = np.vstack([
            np.hstack(panels[:2]),
            np.hstack(panels[2:])
        ])

        destination = (
            OUTPUT / f"frame_{frame_index:05d}_four_hypotheses.jpg"
        )

        cv2.imwrite(str(destination), comparison)
        print("Saved:", destination)

    report = {
        "candidate": "Candidate 04",
        "hypotheses": hypotheses,
        "results": results,
        "status": "UNVERIFIED",
        "note": (
            "Visible edge counts are diagnostic only. "
            "They do not establish CAD-to-camera alignment."
        )
    }

    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nCompleted four-hypothesis projection test.")
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
