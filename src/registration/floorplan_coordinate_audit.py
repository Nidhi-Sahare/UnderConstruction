
"""
BuildProgress - Step 24
Floor-plan coordinate-system audit.

Compares:
  1. CAD vector geometry bounds
  2. Official floor-plan PNG
  3. Reference trajectory bounds
  4. Candidate transform from Step 23

No manual alignment, no hardcoded offsets.

Install:
    pip install numpy matplotlib pillow huggingface_hub

Run:
    python floorplan_coordinate_audit.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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

GEOMETRY_PATH = (
    ROOT / "data" / "geometry"
    / "normalized_segments.json"
)

TRAJECTORY_PATH = (
    ROOT / "data" / "floorplan_registration"
    / "aligned_trajectory.csv"
)

REGISTRATION_PATH = (
    ROOT / "data" / "floorplan_registration"
    / "registration_report.json"
)

OUTPUT = (
    ROOT / "data" / "coordinate_audit"
)


def load_geometry():
    data = json.loads(
        GEOMETRY_PATH.read_text(
            encoding="utf-8"
        )
    )

    segments = []

    for item in data.get("segments", []):
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
        ):
            segments.append({
                "start": start,
                "end": end,
                "category": item.get(
                    "category", "unknown"
                ),
            })

    if not segments:
        raise RuntimeError(
            "No valid CAD segments found."
        )

    points = np.vstack([
        point
        for segment in segments
        for point in (
            segment["start"],
            segment["end"],
        )
    ])

    bounds = {
        "min_x": float(points[:, 0].min()),
        "max_x": float(points[:, 0].max()),
        "min_y": float(points[:, 1].min()),
        "max_y": float(points[:, 1].max()),
    }

    return segments, bounds


def load_trajectory():
    data = np.genfromtxt(
        TRAJECTORY_PATH,
        delimiter=",",
        names=True,
        dtype=np.float64,
        encoding="utf-8",
    )

    data = np.atleast_1d(data)

    positions = np.column_stack([
        data["x"],
        data["y"],
        data["z"],
    ])

    if not np.all(
        np.isfinite(positions)
    ):
        raise RuntimeError(
            "Invalid trajectory positions."
        )

    return positions


def get_png():
    path = hf_hub_download(
        repo_id=DATASET,
        repo_type="dataset",
        filename=(
            "floorplans/png_format/"
            "floor_2.png"
        ),
    )

    return Path(path)


def bounds_from_points(points):
    return {
        "min_x": float(points[:, 0].min()),
        "max_x": float(points[:, 0].max()),
        "min_y": float(points[:, 1].min()),
        "max_y": float(points[:, 1].max()),
    }


def overlap_area(a, b):
    width = max(
        0.0,
        min(a["max_x"], b["max_x"])
        - max(a["min_x"], b["min_x"]),
    )

    height = max(
        0.0,
        min(a["max_y"], b["max_y"])
        - max(a["min_y"], b["min_y"]),
    )

    return width * height


def draw_comparison(
    segments,
    positions,
    image,
    output_path,
):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(19, 7),
    )

    colors = {
        "exterior": "#d46b47",
        "interior": "#424242",
        "partition": "#8093a8",
        "lining": "#aaaaaa",
    }

    for segment in segments:
        start = segment["start"]
        end = segment["end"]

        axes[0].plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=colors.get(
                segment["category"],
                "#999999",
            ),
            linewidth=0.8,
        )

    axes[0].set_title(
        "DXF CAD geometry"
    )

    axes[0].set_aspect(
        "equal",
        adjustable="box",
    )

    axes[0].set_xlabel(
        "CAD X (m)"
    )

    axes[0].set_ylabel(
        "CAD Y (m)"
    )

    axes[1].imshow(image)

    axes[1].set_title(
        "Official Floor 2 PNG"
    )

    axes[1].set_xlabel(
        "Image X (pixels)"
    )

    axes[1].set_ylabel(
        "Image Y (pixels)"
    )

    axes[2].plot(
        positions[:, 0],
        positions[:, 1],
        linewidth=1.2,
    )

    axes[2].scatter(
        positions[0, 0],
        positions[0, 1],
        s=60,
        label="Start",
    )

    axes[2].scatter(
        positions[-1, 0],
        positions[-1, 1],
        s=60,
        marker="x",
        label="End",
    )

    axes[2].set_title(
        "Candidate trajectory"
    )

    axes[2].set_aspect(
        "equal",
        adjustable="box",
    )

    axes[2].set_xlabel(
        "Candidate floor X (m)"
    )

    axes[2].set_ylabel(
        "Candidate floor Y (m)"
    )

    axes[2].legend()

    for ax in axes:
        ax.grid(
            alpha=0.15
        )

    fig.suptitle(
        "BuildProgress | Floor-plan "
        "coordinate audit\n"
        "Different coordinate systems "
        "must not be assumed equivalent"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)


def main():
    for path in (
        GEOMETRY_PATH,
        TRAJECTORY_PATH,
        REGISTRATION_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input missing: {path}"
            )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    segments, cad_bounds = (
        load_geometry()
    )

    positions = load_trajectory()

    trajectory_bounds = (
        bounds_from_points(
            positions[:, :2]
        )
    )

    registration = json.loads(
        REGISTRATION_PATH.read_text(
            encoding="utf-8"
        )
    )

    png_path = get_png()

    with Image.open(
        png_path
    ) as source:
        image = source.convert(
            "RGB"
        )

    image_width, image_height = (
        image.size
    )

    preview_path = (
        OUTPUT
        / "coordinate_comparison.png"
    )

    draw_comparison(
        segments,
        positions,
        image,
        preview_path,
    )

    transform = (
        registration.get(
            "candidate_transform"
        )
    )

    transform_identity_error = None

    if transform is not None:
        matrix = np.asarray(
            transform,
            dtype=np.float64,
        )

        if matrix.shape == (4, 4):
            transform_identity_error = (
                float(
                    np.linalg.norm(
                        matrix - np.eye(4)
                    )
                )
            )

    report = {
        "cad": {
            "segment_count": len(
                segments
            ),
            "bounds_m": cad_bounds,
            "width_m": (
                cad_bounds["max_x"]
                - cad_bounds["min_x"]
            ),
            "height_m": (
                cad_bounds["max_y"]
                - cad_bounds["min_y"]
            ),
        },
        "official_png": {
            "dataset_path": (
                "floorplans/png_format/"
                "floor_2.png"
            ),
            "width_px": image_width,
            "height_px": image_height,
        },
        "candidate_trajectory": {
            "pose_count": len(
                positions
            ),
            "bounds_m": (
                trajectory_bounds
            ),
        },
        "candidate_transform": {
            "frobenius_distance_to_identity": (
                transform_identity_error
            ),
        },
        "bounding_box_overlap_m2": (
            overlap_area(
                cad_bounds,
                trajectory_bounds,
            )
        ),
        "status": (
            "COORDINATE_RELATION_UNVERIFIED"
        ),
        "limitations": [
            "PNG pixels are not metres.",
            "No PNG-to-DXF transform estimated.",
            "No map-to-DXF transform verified.",
            "Bounding-box overlap is not "
            "a registration accuracy metric.",
            "The candidate trajectory must "
            "not be used for element-level "
            "progress attribution yet.",
        ],
    }

    report_path = (
        OUTPUT
        / "coordinate_audit_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nBUILDPROGRESS | COORDINATE AUDIT"
    )

    print("-" * 55)

    print(
        "CAD segments:",
        len(segments),
    )

    print(
        "CAD bounds:",
        cad_bounds,
    )

    print(
        "Official PNG:",
        f"{image_width} x "
        f"{image_height} pixels",
    )

    print(
        "Trajectory bounds:",
        trajectory_bounds,
    )

    print(
        "Candidate transform distance "
        "to identity:",
        transform_identity_error,
    )

    print(
        "Bounding-box overlap:",
        report[
            "bounding_box_overlap_m2"
        ],
        "mÂ²",
    )

    print(
        "\nStatus:",
        report["status"],
    )

    print(
        "\nOutputs:"
    )

    print(report_path)
    print(preview_path)


if __name__ == "__main__":
    main()

