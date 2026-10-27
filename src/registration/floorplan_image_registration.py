
"""
BuildProgress - Step 25
Automatic PNG-to-DXF geometric registration.

Purpose:
    Estimate a similarity transformation from
    DXF world coordinates to floor-plan PNG pixels.

Method:
    - Extract exterior CAD line geometry
    - Detect raster architectural linework
    - Multi-start, multi-scale distance-transform
      optimization
    - Compare candidate solutions
    - Report coverage and residuals

Important:
    A good numerical fit is only a candidate.
    Dense annotations in architectural PNGs
    can create misleading matches.

Install:
    pip install numpy scipy opencv-python matplotlib huggingface_hub

Run:
    python floorplan_image_registration.py

Outputs:
    data/image_registration/
        registration_report.json
        registration_overlay.png
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
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

GEOMETRY = (
    ROOT / "data"
    / "geometry"
    / "normalized_segments.json"
)

OUTPUT = (
    ROOT / "data"
    / "image_registration"
)

MAX_DIMENSION = 1400
MAX_CAD_POINTS = 1800


def load_cad():
    data = json.loads(
        GEOMETRY.read_text(
            encoding="utf-8"
        )
    )

    lines = []

    for item in data.get("segments", []):
        if item.get("category") != "exterior":
            continue

        start = np.asarray(
            item["start"][:2],
            dtype=np.float64,
        )

        end = np.asarray(
            item["end"][:2],
            dtype=np.float64,
        )

        if (
            np.all(np.isfinite(start))
            and np.all(np.isfinite(end))
            and np.linalg.norm(end - start) > 1e-5
        ):
            lines.append((start, end))

    if not lines:
        raise RuntimeError(
            "No exterior CAD segments found."
        )

    return lines


def sample_lines(lines, spacing=0.10):
    samples = []

    for start, end in lines:
        length = np.linalg.norm(
            end - start
        )

        count = max(
            2,
            int(np.ceil(length / spacing)) + 1,
        )

        t = np.linspace(
            0.0, 1.0, count
        )

        points = (
            start[None, :] * (1 - t[:, None])
            + end[None, :] * t[:, None]
        )

        samples.append(points)

    points = np.concatenate(
        samples,
        axis=0,
    )

    if len(points) > MAX_CAD_POINTS:
        indices = np.linspace(
            0,
            len(points) - 1,
            MAX_CAD_POINTS,
            dtype=int,
        )

        points = points[indices]

    return points


def load_png():
    path = hf_hub_download(
        repo_id=DATASET,
        repo_type="dataset",
        filename=(
            "floorplans/png_format/"
            "floor_2.png"
        ),
    )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            "Cannot load floor-plan PNG."
        )

    height, width = image.shape[:2]

    factor = min(
        1.0,
        MAX_DIMENSION / max(
            width, height
        ),
    )

    if factor < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA,
        )

    return image, factor


def extract_edges(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0,
    )

    edges = cv2.Canny(
        gray,
        45,
        130,
    )

    return edges


def transform_points(
    points,
    params,
):
    """
    params:
        log_scale, angle, tx, ty

    Pixel Y inversion is included because
    image coordinates increase downward.
    """
    log_scale, angle, tx, ty = params

    scale = np.exp(
        log_scale
    )

    c = np.cos(angle)
    s = np.sin(angle)

    rotation = np.array([
        [c, -s],
        [s, c],
    ])

    result = (
        points @ rotation.T
    )

    result[:, 1] *= -1

    result *= scale

    result[:, 0] += tx
    result[:, 1] += ty

    return result


def score_transform(
    params,
    points,
    distance_map,
    penalty=35.0,
):
    height, width = (
        distance_map.shape
    )

    transformed = transform_points(
        points,
        params,
    )

    x = transformed[:, 0]
    y = transformed[:, 1]

    valid = (
        (x >= 0)
        & (x < width - 1)
        & (y >= 0)
        & (y < height - 1)
    )

    valid_fraction = (
        float(np.mean(valid))
    )

    if valid_fraction < 0.70:
        return (
            penalty
            + 100 * (
                0.70 - valid_fraction
            )
        )

    if not np.any(valid):
        return penalty + 100

    map_x = x[valid].astype(
        np.float32
    ).reshape(-1, 1)

    map_y = y[valid].astype(
        np.float32
    ).reshape(-1, 1)

    sampled = cv2.remap(
        distance_map,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=penalty,
    ).ravel()

    residual = np.minimum(
        sampled,
        penalty,
    )

    return (
        float(np.mean(residual))
        + 30.0 * (
            1.0 - valid_fraction
        )
    )


def optimize_registration(
    points,
    distance_map,
):
    height, width = (
        distance_map.shape
    )

    cad_extent = np.ptp(
        points,
        axis=0,
    )

    cad_size = max(
        float(np.max(cad_extent)),
        1e-6,
    )

    image_size = max(
        width, height
    )

    base_scale = (
        0.62 * image_size / cad_size
    )

    center = np.mean(
        points,
        axis=0,
    )

    candidates = []

    scale_factors = [
        0.65,
        0.85,
        1.0,
        1.15,
        1.35,
    ]

    angles = np.deg2rad(
        np.arange(
            0, 360, 45
        )
    )

    centers = [
        (0.35, 0.40),
        (0.50, 0.50),
        (0.65, 0.60),
    ]

    for factor in scale_factors:
        scale = (
            base_scale * factor
        )

        for angle in angles:
            for fx, fy in centers:
                target = np.array([
                    width * fx,
                    height * fy,
                ])

                initial = np.array([
                    np.log(scale),
                    angle,
                    0.0,
                    0.0,
                ])

                current_center = (
                    transform_points(
                        center[None, :],
                        initial,
                    )[0]
                )

                initial[2:4] = (
                    target - current_center
                )

                result = minimize(
                    score_transform,
                    initial,
                    args=(
                        points,
                        distance_map,
                    ),
                    method="Powell",
                    options={
                        "maxiter": 45,
                        "xtol": 1e-3,
                        "ftol": 1e-3,
                    },
                )

                candidates.append({
                    "score": float(
                        result.fun
                    ),
                    "params": (
                        result.x.tolist()
                    ),
                    "success": bool(
                        result.success
                    ),
                })

    candidates.sort(
        key=lambda item: item["score"]
    )

    return candidates


def evaluate(
    points,
    params,
    distance_map,
):
    transformed = transform_points(
        points,
        params,
    )

    height, width = (
        distance_map.shape
    )

    x = transformed[:, 0]
    y = transformed[:, 1]

    valid = (
        (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )

    distances = []

    for px, py in transformed[valid]:
        ix = min(
            int(round(px)),
            width - 1,
        )

        iy = min(
            int(round(py)),
            height - 1,
        )

        distances.append(
            float(
                distance_map[iy, ix]
            )
        )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )

    return {
        "valid_fraction": float(
            np.mean(valid)
        ),
        "median_edge_distance_px": (
            float(
                np.median(distances)
            )
            if len(distances)
            else None
        ),
        "within_3px_fraction": (
            float(
                np.mean(
                    distances <= 3
                )
            )
            if len(distances)
            else 0.0
        ),
        "within_8px_fraction": (
            float(
                np.mean(
                    distances <= 8
                )
            )
            if len(distances)
            else 0.0
        ),
    }


def draw_overlay(
    image,
    lines,
    params,
    output_path,
):
    overlay = image.copy()

    for start, end in lines:
        transformed = transform_points(
            np.stack(
                [start, end]
            ),
            params,
        )

        p1 = tuple(
            np.rint(
                transformed[0]
            ).astype(int)
        )

        p2 = tuple(
            np.rint(
                transformed[1]
            ).astype(int)
        )

        cv2.line(
            overlay,
            p1,
            p2,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        overlay,
    )


def main():
    if not GEOMETRY.is_file():
        raise FileNotFoundError(
            f"Missing CAD geometry: {GEOMETRY}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading CAD exterior geometry..."
    )

    lines = load_cad()

    points = sample_lines(
        lines
    )

    print(
        "Exterior segments:",
        len(lines),
    )

    print(
        "Sampled CAD points:",
        len(points),
    )

    print(
        "\nLoading official PNG..."
    )

    image, resize_factor = (
        load_png()
    )

    print(
        "Working image:",
        image.shape[1],
        "x",
        image.shape[0],
    )

    print(
        "\nExtracting raster edges..."
    )

    edges = extract_edges(
        image
    )

    distance_map = (
        distance_transform_edt(
            edges == 0
        ).astype(
            np.float32
        )
    )

    print(
        "\nOptimizing candidate transforms..."
    )

    candidates = (
        optimize_registration(
            points,
            distance_map,
        )
    )

    best = candidates[0]

    metrics = evaluate(
        points,
        best["params"],
        distance_map,
    )

    overlay_path = (
        OUTPUT
        / "registration_overlay.png"
    )

    draw_overlay(
        image,
        lines,
        best["params"],
        overlay_path,
    )

    report = {
        "status": (
            "CANDIDATE_ONLY"
        ),
        "method": (
            "CAD exterior-to-raster "
            "edge distance optimization"
        ),
        "working_image_size": [
            image.shape[1],
            image.shape[0],
        ],
        "image_resize_factor": (
            resize_factor
        ),
        "best_score": best["score"],
        "best_parameters": {
            "log_scale": (
                best["params"][0]
            ),
            "angle_radians": (
                best["params"][1]
            ),
            "tx": best["params"][2],
            "ty": best["params"][3],
        },
        "metrics": metrics,
        "top_candidates": (
            candidates[:10]
        ),
        "limitations": [
            "Raster edges include text, "
            "dimensions and furniture.",
            "Low edge distance alone "
            "does not verify registration.",
            "Trajectory-to-PNG registration "
            "has not been established.",
        ],
    }

    report_path = (
        OUTPUT
        / "registration_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nIMAGE REGISTRATION RESULTS"
    )

    print("-" * 55)

    print(
        "Best score:",
        round(
            best["score"],
            3,
        ),
    )

    print(
        "Valid fraction:",
        round(
            metrics["valid_fraction"],
            3,
        ),
    )

    print(
        "Median edge distance:",
        metrics[
            "median_edge_distance_px"
        ],
        "pixels",
    )

    print(
        "Within 3 pixels:",
        round(
            metrics[
                "within_3px_fraction"
            ],
            3,
        ),
    )

    print(
        "Within 8 pixels:",
        round(
            metrics[
                "within_8px_fraction"
            ],
            3,
        ),
    )

    print(
        "\nStatus: CANDIDATE_ONLY"
    )

    print(
        "\nOutputs:"
    )

    print(report_path)
    print(overlay_path)


if __name__ == "__main__":
    main()

