
"""
BuildProgress - Step 29
Unified calibrated camera geometry.

Supports:
    - OpenCV equidistant fisheye cameras
    - Perspective cameras with no distortion
    - Pixel -> unit 3D ray
    - 3D camera point -> pixel
    - Camera pose -> world-space ray
    - Round-trip numerical verification

Coordinate convention:
    Camera: +X right, +Y down, +Z forward.
    Pose: T_world_camera maps camera coordinates
          into the world coordinate frame.

No CAD registration is assumed.

Install:
    pip install numpy opencv-python pyyaml huggingface_hub

Run:
    python camera_geometry.py --self-test
    python camera_geometry.py --hilti-test
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from huggingface_hub import hf_hub_download


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DATASET = (
    "Hilti-Research/"
    "hilti-trimble-slam-challenge-2026"
)

CALIBRATION_FILE = (
    "camera_calibration/"
    "kalibr_imucam_chain.yaml"
)

OUTPUT = (
    ROOT / "data"
    / "camera_geometry"
)


def normalize_vectors(vectors):
    vectors = np.asarray(
        vectors,
        dtype=np.float64,
    )

    lengths = np.linalg.norm(
        vectors,
        axis=-1,
        keepdims=True,
    )

    if np.any(lengths < 1e-12):
        raise ValueError(
            "Cannot normalize zero-length vector."
        )

    return vectors / lengths


@dataclass(frozen=True)
class CameraModel:
    model: str
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        if self.model not in (
            "fisheye",
            "pinhole",
        ):
            raise ValueError(
                f"Unsupported camera model: {self.model}"
            )

        K = np.asarray(
            self.K,
            dtype=np.float64,
        )

        D = np.asarray(
            self.D,
            dtype=np.float64,
        ).reshape(-1)

        if K.shape != (3, 3):
            raise ValueError(
                "K must have shape (3, 3)."
            )

        if not np.all(
            np.isfinite(K)
        ):
            raise ValueError(
                "K contains invalid values."
            )

        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError(
                "Focal lengths must be positive."
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "Invalid image dimensions."
            )

        if self.model == "fisheye":
            if D.size != 4:
                raise ValueError(
                    "Fisheye requires four "
                    "distortion coefficients."
                )
        elif D.size not in (0,):
            raise ValueError(
                "Pinhole mode currently supports "
                "undistorted images only."
            )

        if not np.all(
            np.isfinite(D)
        ):
            raise ValueError(
                "Invalid distortion coefficients."
            )

        object.__setattr__(
            self, "K", K.copy()
        )

        object.__setattr__(
            self, "D", D.copy()
        )

    def pixels_to_rays(self, pixels):
        pixels = np.asarray(
            pixels,
            dtype=np.float64,
        ).reshape(-1, 2)

        if not np.all(
            np.isfinite(pixels)
        ):
            raise ValueError(
                "Pixels must be finite."
            )

        points = pixels.reshape(
            -1, 1, 2
        )

        if self.model == "fisheye":
            undistorted = (
                cv2.fisheye.undistortPoints(
                    points,
                    self.K,
                    self.D.reshape(4, 1),
                )
            )
        else:
            undistorted = (
                cv2.undistortPoints(
                    points,
                    self.K,
                    None,
                )
            )

        xy = undistorted.reshape(
            -1, 2
        )

        rays = np.column_stack([
            xy,
            np.ones(len(xy)),
        ])

        return normalize_vectors(
            rays
        )

    def project_points(self, points):
        points = np.asarray(
            points,
            dtype=np.float64,
        ).reshape(-1, 3)

        if not np.all(
            np.isfinite(points)
        ):
            raise ValueError(
                "3D points must be finite."
            )

        if np.any(
            points[:, 2] <= 1e-9
        ):
            raise ValueError(
                "Points must be in front "
                "of the camera."
            )

        if self.model == "fisheye":
            projected, _ = (
                cv2.fisheye.projectPoints(
                    points.reshape(-1, 1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    self.K,
                    self.D.reshape(4, 1),
                )
            )
        else:
            projected, _ = (
                cv2.projectPoints(
                    points.reshape(-1, 1, 3),
                    np.zeros(3),
                    np.zeros(3),
                    self.K,
                    None,
                )
            )

        return projected.reshape(
            -1, 2
        )

    def world_rays(
        self,
        pixels,
        T_world_camera,
    ):
        T = np.asarray(
            T_world_camera,
            dtype=np.float64,
        )

        if T.shape != (4, 4):
            raise ValueError(
                "Pose must be a 4x4 matrix."
            )

        rotation = T[:3, :3]
        translation = T[:3, 3]

        if not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            atol=1e-5,
        ):
            raise ValueError(
                "Pose rotation is not orthonormal."
            )

        if not np.isclose(
            np.linalg.det(rotation),
            1.0,
            atol=1e-5,
        ):
            raise ValueError(
                "Pose rotation determinant "
                "must be +1."
            )

        rays_camera = (
            self.pixels_to_rays(pixels)
        )

        rays_world = (
            rays_camera @ rotation.T
        )

        origins = np.repeat(
            translation[None, :],
            len(rays_world),
            axis=0,
        )

        return origins, normalize_vectors(
            rays_world
        )


def load_hilti_camera(camera="cam0"):
    path = Path(
        hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            filename=CALIBRATION_FILE,
        )
    )

    text = path.read_text(
        encoding="utf-8-sig"
    )

    lines = text.splitlines()

    if (
        lines
        and lines[0].lstrip().startswith(
            "%YAML:"
        )
    ):
        lines = lines[1:]

    data = yaml.safe_load(
        "\n".join(lines)
    )

    entry = data[camera]

    camera_model = str(
        entry["camera_model"]
    ).lower()

    distortion_model = str(
        entry["distortion_model"]
    ).lower()

    if (
        camera_model != "pinhole"
        or distortion_model != "equidistant"
    ):
        raise ValueError(
            "Unsupported Hilti camera calibration: "
            f"{camera_model} / "
            f"{distortion_model}"
        )

    fx, fy, cx, cy = map(
        float,
        entry["intrinsics"],
    )

    width, height = map(
        int,
        entry["resolution"],
    )

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    D = np.asarray(
        entry["distortion_coeffs"],
        dtype=np.float64,
    )

    return CameraModel(
        model="fisheye",
        K=K,
        D=D,
        width=width,
        height=height,
    )


def run_self_test():
    print(
        "\nCAMERA GEOMETRY SELF-TEST"
    )

    print("-" * 55)

    models = [
        CameraModel(
            "pinhole",
            np.array([
                [600, 0, 320],
                [0, 600, 240],
                [0, 0, 1],
            ]),
            np.array([]),
            640,
            480,
        ),
        CameraModel(
            "fisheye",
            np.array([
                [420, 0, 320],
                [0, 420, 240],
                [0, 0, 1],
            ]),
            np.array([
                0.01,
                -0.005,
                0.001,
                0.0,
            ]),
            640,
            480,
        ),
    ]

    points = np.array([
        [0.0, 0.0, 2.0],
        [0.2, 0.1, 2.0],
        [-0.3, 0.2, 3.0],
        [0.4, -0.2, 4.0],
    ])

    for model in models:
        pixels = model.project_points(
            points
        )

        recovered = (
            model.pixels_to_rays(
                pixels
            )
        )

        expected = normalize_vectors(
            points
        )

        errors = np.linalg.norm(
            recovered - expected,
            axis=1,
        )

        max_error = float(
            np.max(errors)
        )

        print(
            model.model,
            "max ray error:",
            f"{max_error:.10f}",
        )

        if max_error > 1e-6:
            raise AssertionError(
                "Round-trip test failed."
            )

    print(
        "\nPASS: camera geometry self-test"
    )


def run_hilti_test():
    print(
        "\nHILTI CAMERA GEOMETRY TEST"
    )

    print("-" * 55)

    camera = load_hilti_camera(
        "cam0"
    )

    print(
        "Model:",
        camera.model,
    )

    print(
        "Resolution:",
        camera.width,
        "x",
        camera.height,
    )

    grid_x = np.linspace(
        camera.width * 0.2,
        camera.width * 0.8,
        7,
    )

    grid_y = np.linspace(
        camera.height * 0.2,
        camera.height * 0.8,
        7,
    )

    pixels = np.array([
        [x, y]
        for y in grid_y
        for x in grid_x
    ])

    rays = camera.pixels_to_rays(
        pixels
    )

    valid = rays[:, 2] > 0

    reconstructed = (
        camera.project_points(
            rays[valid] * 3.0
        )
    )

    errors = np.linalg.norm(
        reconstructed
        - pixels[valid],
        axis=1,
    )

    max_error = (
        float(np.max(errors))
        if len(errors)
        else None
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "camera": "cam0",
        "model": camera.model,
        "resolution": [
            camera.width,
            camera.height,
        ],
        "tested_pixels": len(
            pixels
        ),
        "forward_facing_rays": int(
            np.sum(valid)
        ),
        "max_reprojection_error_px": (
            max_error
        ),
        "status": (
            "PASS"
            if max_error is not None
            and max_error < 1e-4
            else "FAIL"
        ),
        "cad_registration": (
            "NOT VERIFIED"
        ),
    }

    path = (
        OUTPUT
        / "geometry_test_report.json"
    )

    path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "Tested pixels:",
        len(pixels),
    )

    print(
        "Forward-facing rays:",
        np.sum(valid),
    )

    print(
        "Maximum reprojection error:",
        max_error,
    )

    print(
        "Status:",
        report["status"],
    )

    print(
        "Report:",
        path,
    )

    if report["status"] != "PASS":
        raise AssertionError(
            "Hilti geometry verification failed."
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--hilti-test",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        not args.self_test
        and not args.hilti_test
    ):
        parser.print_help()
        return

    if args.self_test:
        run_self_test()

    if args.hilti_test:
        run_hilti_test()


if __name__ == "__main__":
    main()

