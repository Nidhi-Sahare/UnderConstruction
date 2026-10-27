
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[1]

SEQUENCE = "floor_2_2025-10-28_run_2"

MANIFEST = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / SEQUENCE
    / "cam0"
    / "frame_manifest.csv"
)

CLOUD = (
    ROOT
    / "outputs"
    / "multiview_reconstruction"
    / "multiview_sparse_cloud.ply"
)

REPORT = (
    ROOT
    / "outputs"
    / "multiview_reconstruction"
    / "multiview_reconstruction_report.json"
)

OUTPUT = (
    ROOT
    / "outputs"
    / "reconstruction_viewer"
)


def quaternion_to_rotation(q):
    x, y, z, w = q

    return np.array(
        [
            [
                1 - 2 * (y*y + z*z),
                2 * (x*y - z*w),
                2 * (x*z + y*w),
            ],
            [
                2 * (x*y + z*w),
                1 - 2 * (x*x + z*z),
                2 * (y*z - x*w),
            ],
            [
                2 * (x*z - y*w),
                2 * (y*z + x*w),
                1 - 2 * (x*x + y*y),
            ],
        ],
        dtype=np.float64,
    )


def load_ply(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        header = []

        while True:
            line = handle.readline()

            if not line:
                raise ValueError(
                    "Unexpected end of PLY header."
                )

            line = line.strip()
            header.append(line)

            if line == "end_header":
                break

        if "format ascii 1.0" not in header:
            raise ValueError(
                "Only ASCII PLY is supported."
            )

        vertex_count = None
        properties = []
        in_vertex = False

        for line in header:
            if line.startswith("element "):
                parts = line.split()

                in_vertex = (
                    len(parts) >= 3
                    and parts[1] == "vertex"
                )

                if in_vertex:
                    vertex_count = int(parts[2])

            elif (
                line.startswith("property ")
                and in_vertex
            ):
                properties.append(
                    line.split()[-1]
                )

        if vertex_count is None:
            raise ValueError(
                "PLY vertex count not found."
            )

        required = (
            "x", "y", "z",
            "red", "green", "blue",
        )

        if any(
            name not in properties
            for name in required
        ):
            raise ValueError(
                "PLY lacks required XYZ/RGB fields."
            )

        rows = []

        for _ in range(vertex_count):
            line = handle.readline()

            if not line:
                raise ValueError(
                    "PLY ended before all vertices."
                )

            rows.append(
                [float(x) for x in line.split()]
            )

    if not rows:
        return (
            np.empty((0, 3)),
            np.empty((0, 3), dtype=np.uint8),
        )

    array = np.asarray(
        rows,
        dtype=np.float64,
    )

    xyz = array[
        :,
        [properties.index(k) for k in ("x", "y", "z")]
    ]

    rgb = array[
        :,
        [
            properties.index(k)
            for k in ("red", "green", "blue")
        ]
    ]

    return (
        xyz,
        np.clip(rgb, 0, 255).astype(np.uint8),
    )


def load_camera_poses():
    cameras = []

    with MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            if (
                row.get("pose_matched", "").lower()
                != "true"
            ):
                continue

            try:
                center = np.array(
                    [
                        float(row[k])
                        for k in ("tx", "ty", "tz")
                    ],
                    dtype=np.float64,
                )

                q = np.array(
                    [
                        float(row[k])
                        for k in (
                            "qx", "qy", "qz", "qw"
                        )
                    ],
                    dtype=np.float64,
                )

                norm = np.linalg.norm(q)

                if (
                    norm < 1e-10
                    or not np.isfinite(center).all()
                    or not np.isfinite(q).all()
                ):
                    continue

                R = quaternion_to_rotation(
                    q / norm
                )

                cameras.append(
                    {
                        "filename": row["filename"],
                        "center": center,
                        "forward": R[:, 2],
                    }
                )

            except (
                ValueError,
                KeyError,
                TypeError,
            ):
                continue

    return cameras


def robust_cloud_mask(
    xyz,
    threshold=5.0,
):
    if len(xyz) < 20:
        return np.ones(
            len(xyz),
            dtype=bool,
        )

    center = np.median(
        xyz,
        axis=0,
    )

    distances = np.linalg.norm(
        xyz - center,
        axis=1,
    )

    median = np.median(
        distances
    )

    mad = np.median(
        np.abs(distances - median)
    )

    if mad < 1e-9:
        return np.ones(
            len(xyz),
            dtype=bool,
        )

    limit = (
        median
        + threshold * 1.4826 * mad
    )

    return distances <= limit


def make_figure(
    xyz,
    rgb,
    cameras,
    show_outliers,
    arrow_length,
):
    fig = go.Figure()

    finite = np.isfinite(
        xyz
    ).all(axis=1)

    xyz = xyz[finite]
    rgb = rgb[finite]

    mask = robust_cloud_mask(
        xyz
    )

    if len(xyz):
        colors = [
            f"rgb({r},{g},{b})"
            for r, g, b in rgb[mask]
        ]

        fig.add_trace(
            go.Scatter3d(
                x=xyz[mask, 0],
                y=xyz[mask, 1],
                z=xyz[mask, 2],
                mode="markers",
                marker=dict(
                    size=2.5,
                    color=colors,
                    opacity=0.85,
                ),
                name="Reconstructed points",
                hovertemplate=(
                    "X=%{x:.3f}<br>"
                    "Y=%{y:.3f}<br>"
                    "Z=%{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        if show_outliers and (~mask).any():
            outliers = xyz[~mask]

            fig.add_trace(
                go.Scatter3d(
                    x=outliers[:, 0],
                    y=outliers[:, 1],
                    z=outliers[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                        color="red",
                        opacity=0.55,
                    ),
                    name="Statistical outliers",
                )
            )

    if cameras:
        centers = np.asarray(
            [c["center"] for c in cameras]
        )

        fig.add_trace(
            go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2],
                mode="lines+markers",
                line=dict(
                    color="#E58A40",
                    width=5,
                ),
                marker=dict(
                    color="#E58A40",
                    size=3,
                ),
                name="Recorded camera trajectory",
                text=[
                    c["filename"]
                    for c in cameras
                ],
                hovertemplate=(
                    "%{text}<br>"
                    "X=%{x:.3f}<br>"
                    "Y=%{y:.3f}<br>"
                    "Z=%{z:.3f}"
                    "<extra></extra>"
                ),
            )
        )

        sampled = cameras[::max(
            1,
            len(cameras) // 25,
        )]

        for camera in sampled:
            start = camera["center"]

            end = (
                start
                + arrow_length
                * camera["forward"]
            )

            fig.add_trace(
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line=dict(
                        color="#36A9AE",
                        width=3,
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        title=(
            "Under-construction | "
            "Experimental 3D Reconstruction"
        ),
        paper_bgcolor="#FAF8F4",
        plot_bgcolor="#FAF8F4",
        font=dict(
            family="Arial",
            color="#252525",
        ),
        scene=dict(
            xaxis_title="X (pose units)",
            yaxis_title="Y (pose units)",
            zaxis_title="Z (pose units)",
            aspectmode="data",
            bgcolor="#FAF8F4",
        ),
        margin=dict(
            l=0,
            r=0,
            t=60,
            b=0,
        ),
        legend=dict(
            orientation="h",
            y=1.02,
        ),
        annotations=[
            dict(
                text=(
                    "Experimental geometry — "
                    "camera extrinsics and CAD "
                    "alignment unverified"
                ),
                x=0.01,
                y=0.01,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(
                    size=12,
                    color="#666666",
                ),
            )
        ],
    )

    return fig, mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--show-outliers",
        action="store_true",
    )

    parser.add_argument(
        "--arrow-length",
        type=float,
        default=0.8,
    )

    args = parser.parse_args()

    if not CLOUD.exists():
        raise FileNotFoundError(CLOUD)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    xyz, rgb = load_ply(CLOUD)
    cameras = load_camera_poses()

    figure, mask = make_figure(
        xyz,
        rgb,
        cameras,
        args.show_outliers,
        args.arrow_length,
    )

    html_path = (
        OUTPUT
        / "site_reconstruction_3d.html"
    )

    figure.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True,
        auto_open=True,
    )

    summary = {
        "project": "Under-construction",
        "point_count": len(xyz),
        "points_inside_robust_mask": int(
            mask.sum()
        ),
        "statistical_outliers": int(
            (~mask).sum()
        ),
        "camera_pose_count": len(cameras),
        "bounds": {
            "minimum": (
                xyz.min(axis=0).tolist()
                if len(xyz) else None
            ),
            "maximum": (
                xyz.max(axis=0).tolist()
                if len(xyz) else None
            ),
        },
        "limitations": [
            "Point cloud is experimental.",
            "Camera optical extrinsics are unverified.",
            "The robust mask is statistical only.",
            "Statistical outliers are not proven errors.",
            "Camera forward vectors use the provisional "
            "camera-to-world quaternion convention.",
            "No CAD alignment has been verified.",
        ],
    }

    summary_path = (
        OUTPUT
        / "viewer_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nUNDER-CONSTRUCTION")
    print("3D reconstruction viewer")
    print("-" * 50)

    print(
        "Cloud points:",
        len(xyz),
    )

    print(
        "Camera poses:",
        len(cameras),
    )

    print(
        "Statistical outliers:",
        int((~mask).sum()),
    )

    print(
        "HTML viewer:",
        html_path,
    )

    print(
        "Summary:",
        summary_path,
    )


if __name__ == "__main__":
    main()
