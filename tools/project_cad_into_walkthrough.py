
"""
Under-construction — Step 79
Calibrated CAD-to-fisheye-image projection.

Uses the official Hilti reference trajectory, whose poses describe
cam0 in the reference map frame.

For each proposed map-to-CAD registration:
    1. Transform CAD wall coordinates into the reference map.
    2. Project 3D wall edges through the original fisheye camera.
    3. Save projection overlays and a machine-readable report.

IMPORTANT:
- No manual annotation.
- No camera-to-IMU transform is applied.
- Projected geometry is hypothetical until image evidence verifies it.
- Wall heights are illustrative geometry, NOT measured as-built heights.
"""

import csv
import json
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
SEQUENCE = "floor_2_2025-10-28_run_2"

ARCHIVE = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
)

FRAME_DIR = (
    ARCHIVE / "data" / "hilti_frames"
    / SEQUENCE / "cam0"
)

MANIFEST = FRAME_DIR / "frame_manifest.csv"
DXF = ROOT / "data" / "floorplans" / "floor_2.dxf"

CANDIDATES = (
    ROOT / "outputs"
    / "automatic_registration_v2"
    / "registration_candidates.json"
)

OUTPUT = ROOT / "outputs" / "cad_image_projection"

WALL_LAYERS = {
    "015 Innenwände",
    "010 Aussenwände",
    "016 Trennwände",
}

# These are visualization hypotheses, not measured wall heights.
WALL_BOTTOM_Z = 0.0
WALL_TOP_Z = 3.0

# Sample a manageable number of frames, including different
# positions along the walkthrough.
MAX_FRAMES = 12
MAX_SEGMENTS = 1200

K = np.array(
    [
        [465.3015482593691, 0, 730.0455886686005],
        [0, 465.32303798346413, 720.1427007671206],
        [0, 0, 1],
    ],
    dtype=np.float64,
)

D = np.array(
    [
        0.025800718903376804,
        -0.010909240777406872,
        -0.0016899537986031076,
        0.00014766801645260894,
    ],
    dtype=np.float64,
)


def load_frames():
    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    frames = []

    with MANIFEST.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            if row.get("pose_matched", "").lower() not in {
                "true", "1", "yes"
            }:
                continue

            try:
                position = np.array(
                    [
                        float(row[key])
                        for key in ("tx", "ty", "tz")
                    ],
                    dtype=np.float64,
                )

                quaternion = np.array(
                    [
                        float(row[key])
                        for key in ("qx", "qy", "qz", "qw")
                    ],
                    dtype=np.float64,
                )

                if (
                    not np.all(np.isfinite(position))
                    or not np.all(np.isfinite(quaternion))
                    or abs(np.linalg.norm(quaternion) - 1) > 0.01
                ):
                    continue

                image_path = (
                    FRAME_DIR / "frames" / row["filename"]
                )

                if image_path.exists():
                    frames.append(
                        {
                            "filename": row["filename"],
                            "path": image_path,
                            "position": position,
                            "rotation": Rotation.from_quat(
                                quaternion
                            ).as_matrix(),
                        }
                    )
            except (KeyError, ValueError):
                continue

    if not frames:
        raise RuntimeError("No valid pose-matched frames.")

    indices = np.linspace(
        0, len(frames) - 1,
        min(MAX_FRAMES, len(frames)),
        dtype=int,
    )

    return [frames[i] for i in sorted(set(indices))]


def extract_candidate_list(data):
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in (
            "candidates",
            "registration_candidates",
            "ranked_candidates",
            "results",
        ):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "Cannot identify candidate list in JSON. "
        "Inspect registration_candidates.json structure."
    )


def get_number(candidate, names):
    for name in names:
        if name in candidate:
            return float(candidate[name])
    raise KeyError(f"Missing candidate field: {names}")


def load_candidates():
    if not CANDIDATES.exists():
        raise FileNotFoundError(CANDIDATES)

    data = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    entries = extract_candidate_list(data)
    results = []

    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValueError(
                f"Candidate {index + 1} is not a dictionary."
            )

        transform = item.get("transform", item)

        if not isinstance(transform, dict):
            raise ValueError(
                f"Candidate {index + 1} has unsupported transform."
            )

        angle = get_number(
            transform,
            ("angle_deg", "rotation_deg", "theta_deg"),
        )
        tx = get_number(transform, ("tx", "translation_x"))
        ty = get_number(transform, ("ty", "translation_y"))

        if not all(map(math.isfinite, (angle, tx, ty))):
            raise ValueError("Nonfinite registration candidate.")

        results.append(
            {
                "rank": index + 1,
                "angle_deg": angle,
                "tx": tx,
                "ty": ty,
            }
        )

    if not results:
        raise RuntimeError("No registration candidates.")

    return results


def load_wall_segments():
    if not DXF.exists():
        raise FileNotFoundError(DXF)

    doc = ezdxf.readfile(str(DXF))
    model = doc.modelspace()
    segments = []

    for entity in model:
        if entity.dxf.layer not in WALL_LAYERS:
            continue

        kind = entity.dxftype()

        if kind == "LINE":
            a = entity.dxf.start
            b = entity.dxf.end
            points = [
                (float(a.x), float(a.y)),
                (float(b.x), float(b.y)),
            ]

        elif kind == "LWPOLYLINE":
            points = [
                (float(x), float(y))
                for x, y in entity.get_points("xy")
            ]
        else:
            continue

        if len(points) < 2:
            continue

        pairs = list(zip(points[:-1], points[1:]))

        if kind == "LWPOLYLINE" and entity.closed:
            pairs.append((points[-1], points[0]))

        for a, b in pairs:
            length = math.dist(a, b)

            if not 0.05 < length < 100.0:
                continue

            segments.append((a, b))

    if not segments:
        raise RuntimeError("No CAD wall segments extracted.")

    # Deterministic sampling, if necessary.
    if len(segments) > MAX_SEGMENTS:
        indices = np.linspace(
            0, len(segments) - 1,
            MAX_SEGMENTS,
            dtype=int,
        )
        segments = [segments[i] for i in indices]

    return segments


def cad_to_map(points, candidate):
    """
    Existing candidates encode:
        p_cad = R(theta) @ p_map + t

    Invert that transform to map CAD geometry into
    the reference trajectory's coordinate system.
    """
    theta = math.radians(candidate["angle_deg"])

    c = math.cos(theta)
    s = math.sin(theta)

    rotation = np.array(
        [[c, -s], [s, c]],
        dtype=np.float64,
    )

    translation = np.array(
        [candidate["tx"], candidate["ty"]],
        dtype=np.float64,
    )

    result = points.copy()

    result[:, :2] = (
        points[:, :2] - translation
    ) @ rotation

    return result


def project_polyline(points_map, frame, width, height):
    """
    T_map_cam maps cam0 coordinates into map coordinates.
    Therefore:
        p_cam = R_map_cam.T @ (p_map - t_map_cam)
    """
    camera_points = (
        points_map - frame["position"]
    ) @ frame["rotation"]

    # Fisheye projection requires positive optical Z.
    # Avoid drawing lines across points behind the camera.
    if np.any(camera_points[:, 2] <= 0.05):
        return None

    projected, _ = cv2.fisheye.projectPoints(
        camera_points.reshape(-1, 1, 3),
        np.zeros((3, 1), dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        K,
        D,
    )

    pixels = projected.reshape(-1, 2)

    if not np.all(np.isfinite(pixels)):
        return None

    # Reject very distant projected endpoints.
    if np.any(np.abs(pixels) > 100000):
        return None

    # Clip in image coordinates instead of demanding
    # that both endpoints already lie inside the image.
    p0 = tuple(np.rint(pixels[0]).astype(int))
    p1 = tuple(np.rint(pixels[1]).astype(int))

    visible, clipped_a, clipped_b = cv2.clipLine(
        (0, 0, width, height),
        p0,
        p1,
    )

    if not visible:
        return None

    return clipped_a, clipped_b


def render_candidate(frame, candidate, segments):
    image = cv2.imread(str(frame["path"]))

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {frame['path']}"
        )

    height, width = image.shape[:2]

    if (width, height) != (1472, 1440):
        raise RuntimeError(
            f"Unexpected image dimensions: {width}x{height}"
        )

    overlay = image.copy()

    counts = {
        "segments_considered": 0,
        "projected_visible_edges": 0,
    }

    for a, b in segments:
        counts["segments_considered"] += 1

        for z in (WALL_BOTTOM_Z, WALL_TOP_Z):
            points = np.array(
                [
                    [a[0], a[1], z],
                    [b[0], b[1], z],
                ],
                dtype=np.float64,
            )

            points_map = cad_to_map(points, candidate)

            line = project_polyline(
                points_map, frame, width, height
            )

            if line is None:
                continue

            color = (
                (0, 255, 255)
                if z == WALL_BOTTOM_Z
                else (255, 180, 0)
            )

            cv2.line(
                overlay,
                line[0],
                line[1],
                color,
                1,
                cv2.LINE_AA,
            )

            counts["projected_visible_edges"] += 1

    result = cv2.addWeighted(
        image, 0.55,
        overlay, 0.45,
        0,
    )

    label = (
        f"UNVERIFIED | Candidate {candidate['rank']} "
        f"| {frame['filename']}"
    )

    cv2.rectangle(
        result, (0, 0), (width, 48),
        (0, 0, 0), -1
    )

    cv2.putText(
        result,
        label,
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result, counts


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    frames = load_frames()
    candidates = load_candidates()
    segments = load_wall_segments()

    report = {
        "sequence": SEQUENCE,
        "trajectory_pose_convention": "T_map_cam0",
        "camera_model": "pinhole_equidistant",
        "camera_extrinsics_applied": False,
        "registration_verified": False,
        "wall_heights_measured": False,
        "wall_height_visualization_m": [
            WALL_BOTTOM_Z, WALL_TOP_Z
        ],
        "frame_count": len(frames),
        "candidate_count": len(candidates),
        "cad_wall_segments": len(segments),
        "projections": [],
    }

    print("\n=== CALIBRATED CAD IMAGE PROJECTION ===")
    print("Frames:", len(frames))
    print("Candidates:", len(candidates))
    print("CAD segments:", len(segments))
    print("Reference poses: cam0 -> map")

    for candidate in candidates:
        folder = OUTPUT / (
            f"candidate_{candidate['rank']:02d}"
        )
        folder.mkdir(parents=True, exist_ok=True)

        for frame in frames:
            image, counts = render_candidate(
                frame, candidate, segments
            )

            filename = frame["filename"]

            destination = folder / filename

            if not cv2.imwrite(str(destination), image):
                raise RuntimeError(
                    f"Cannot write {destination}"
                )

            report["projections"].append(
                {
                    "candidate_rank": candidate["rank"],
                    "frame": filename,
                    "output": str(
                        destination.relative_to(ROOT)
                    ),
                    **counts,
                }
            )

        print(
            f"Candidate {candidate['rank']}: "
            f"{len(frames)} projections generated"
        )

    report_path = OUTPUT / "projection_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nReport:", report_path)
    print("Registration verified: NO")
    print(
        "Projection images are geometric hypotheses, "
        "not evidence of correct registration."
    )


if __name__ == "__main__":
    main()
