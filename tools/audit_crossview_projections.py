
"""
Under-construction — Step 104
Automatic visual audit of cross-view line evidence.

Reads frozen Step 103 results and original camera images.
Generates overlays, contact sheets, and diagnostic reports.

Inputs:
    outputs/line_ambiguity/resolved_lines.json
    outputs/crossview_image_evidence/
        per_candidate_evidence.json
    Camera manifest and image frames

Outputs:
    outputs/crossview_visual_audit/
        audit_report.json
        per_candidate_audit.json
        candidate_XXX/
            *_audit.jpg
            contact_sheet.jpg

This step does not modify reconstruction or verify identity.

Run:
    .\\.venv\\Scripts\\python.exe tools\\audit_crossview_projections.py
"""

import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]

LINES_PATH = (
    ROOT / "outputs"
    / "line_ambiguity"
    / "resolved_lines.json"
)

EVIDENCE_PATH = (
    ROOT / "outputs"
    / "crossview_image_evidence"
    / "per_candidate_evidence.json"
)

CAM_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

MANIFEST = CAM_ROOT / "frame_manifest.csv"
FRAME_DIR = CAM_ROOT / "frames"

OUT = ROOT / "outputs" / "crossview_visual_audit"

TARGET_IDS = (159, 2, 47, 161, 108)

K = np.array(
    [
        [465.3015482593691, 0.0, 730.0455886686005],
        [0.0, 465.32303798346413, 720.1427007671206],
        [0.0, 0.0, 1.0],
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
).reshape(4, 1)

SAMPLE_COUNT = 101
MIN_DEPTH_M = 0.20

MAX_NEARBY_DISTANCE_PX = 14.0
MAX_NEARBY_ANGLE_DEG = 20.0

MAX_SEGMENTS_DRAWN = 120

PANEL_WIDTH = 720
PANEL_HEIGHT = 704

CONTACT_COLUMNS = 2

cv2.setNumThreads(1)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def save_json(path, value):
    path.write_text(
        json.dumps(
            value,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def load_lines():
    raw = load_json(LINES_PATH)
    lines = {}

    for item in raw.get("selected_lines", []):
        try:
            cid = int(item["candidate_id"])

            points = np.asarray(
                item["endpoints_map"],
                dtype=np.float64,
            )

            if (
                points.shape != (2, 3)
                or not np.isfinite(points).all()
            ):
                continue

            lines[cid] = points

        except (
            KeyError,
            ValueError,
            TypeError,
        ):
            continue

    return lines


def load_evidence():
    raw = load_json(EVIDENCE_PATH)

    return {
        int(item["candidate_id"]): item
        for item in raw.get("results", [])
    }


def load_poses():
    poses = {}

    with MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        for row in csv.DictReader(handle):
            try:
                filename = Path(
                    row["filename"]
                ).name

                position = np.array(
                    [
                        float(row["tx"]),
                        float(row["ty"]),
                        float(row["tz"]),
                    ],
                    dtype=np.float64,
                )

                quaternion = [
                    float(row["qx"]),
                    float(row["qy"]),
                    float(row["qz"]),
                    float(row["qw"]),
                ]

                rotation = Rotation.from_quat(
                    quaternion
                ).as_matrix()

                poses[filename] = {
                    "position": position,
                    "rotation": rotation,
                }

            except (
                KeyError,
                ValueError,
                TypeError,
            ):
                continue

    return poses


def normalize(vector):
    norm = float(np.linalg.norm(vector))

    if norm < 1e-12:
        raise ValueError("Degenerate vector")

    return vector / norm


def local_axes(points):
    direction = normalize(
        points[1] - points[0]
    )

    reference = min(
        np.eye(3),
        key=lambda axis: abs(
            float(axis @ direction)
        ),
    )

    u = normalize(
        np.cross(direction, reference)
    )

    v = normalize(
        np.cross(direction, u)
    )

    return u, v


def parse_shift(name):
    """
    Accepts Step 103 names such as:
        shift_u_-0.080m
        shift_v_+0.030m
    """
    if not name:
        return None

    parts = str(name).split("_")

    if len(parts) != 3:
        return None

    if parts[0] != "shift":
        return None

    if parts[1] not in ("u", "v"):
        return None

    if not parts[2].endswith("m"):
        return None

    try:
        amount = float(parts[2][:-1])
    except ValueError:
        return None

    return parts[1], amount


def shifted_line(points, name):
    parsed = parse_shift(name)

    if parsed is None:
        return None

    axis_name, amount = parsed

    u, v = local_axes(points)

    axis = (
        u if axis_name == "u"
        else v
    )

    return points + amount * axis


def project_line(points, pose, shape):
    height, width = shape[:2]

    t = np.linspace(
        0.0,
        1.0,
        SAMPLE_COUNT,
    )[:, None]

    samples = (
        points[0]
        + t * (points[1] - points[0])
    )

    camera = (
        samples - pose["position"]
    ) @ pose["rotation"]

    depth_valid = (
        np.isfinite(camera).all(axis=1)
        & (camera[:, 2] > MIN_DEPTH_M)
    )

    pixels = np.full(
        (SAMPLE_COUNT, 2),
        np.nan,
        dtype=np.float64,
    )

    indices = np.flatnonzero(depth_valid)

    if len(indices):
        projected, _ = cv2.fisheye.projectPoints(
            camera[indices].reshape(
                -1, 1, 3
            ),
            np.zeros((3, 1)),
            np.zeros((3, 1)),
            K,
            D,
        )

        pixels[indices] = (
            projected.reshape(-1, 2)
        )

    valid = (
        np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )

    return pixels, valid


def draw_projection(
    image,
    pixels,
    valid,
    color,
    thickness=3,
):
    segments = 0

    for i in range(len(pixels) - 1):
        if not (
            valid[i]
            and valid[i + 1]
        ):
            continue

        p0 = tuple(
            np.rint(pixels[i]).astype(int)
        )

        p1 = tuple(
            np.rint(pixels[i + 1]).astype(int)
        )

        cv2.line(
            image,
            p0,
            p1,
            color,
            thickness,
            cv2.LINE_AA,
        )

        segments += 1

    return segments


def detect_segments(gray, detector):
    detected = detector.detect(gray)[0]

    if detected is None:
        return np.empty(
            (0, 2, 2),
            dtype=np.float64,
        )

    return detected.reshape(
        -1, 2, 2
    ).astype(np.float64)


def distance_point_to_segment(point, a, b):
    direction = b - a

    denominator = float(
        direction @ direction
    )

    if denominator < 1e-12:
        return float(
            np.linalg.norm(point - a)
        )

    t = float(
        np.clip(
            ((point - a) @ direction)
            / denominator,
            0.0,
            1.0,
        )
    )

    nearest = a + t * direction

    return float(
        np.linalg.norm(point - nearest)
    )


def angular_difference(a, b):
    a = normalize(a)
    b = normalize(b)

    cosine = abs(
        float(a @ b)
    )

    return math.degrees(
        math.acos(
            float(
                np.clip(cosine, 0, 1)
            )
        )
    )


def nearby_segments(
    segments,
    projections,
):
    """
    Find globally detected segments near either projection.

    These are visualization associations only; not verified
    physical correspondences.
    """
    scored = []

    for index, segment in enumerate(segments):
        direction = (
            segment[1] - segment[0]
        )

        if np.linalg.norm(direction) < 1e-8:
            continue

        best_distance = math.inf
        best_angle = math.inf

        for pixels, valid in projections:
            for i in range(
                len(pixels) - 1
            ):
                if not (
                    valid[i]
                    and valid[i + 1]
                ):
                    continue

                tangent = (
                    pixels[i + 1]
                    - pixels[i]
                )

                if np.linalg.norm(tangent) < 1e-8:
                    continue

                angle = angular_difference(
                    direction,
                    tangent,
                )

                if angle > MAX_NEARBY_ANGLE_DEG:
                    continue

                midpoint = (
                    pixels[i]
                    + pixels[i + 1]
                ) / 2

                distance = distance_point_to_segment(
                    midpoint,
                    segment[0],
                    segment[1],
                )

                if distance < best_distance:
                    best_distance = distance
                    best_angle = angle

        if best_distance <= MAX_NEARBY_DISTANCE_PX:
            scored.append(
                (
                    best_distance,
                    best_angle,
                    index,
                )
            )

    scored.sort()

    return scored[
        :MAX_SEGMENTS_DRAWN
    ]


def add_text(
    image,
    lines,
    x=18,
    y=32,
):
    for index, line in enumerate(lines):
        position = (
            x,
            y + index * 27,
        )

        cv2.putText(
            image,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )

        cv2.putText(
            image,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def make_panel(image, title):
    resized = cv2.resize(
        image,
        (PANEL_WIDTH, PANEL_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )

    add_text(
        resized,
        [title],
    )

    return resized


def audit_frame(
    candidate_id,
    points,
    frame_record,
    poses,
    detector,
    destination,
):
    filename = frame_record["frame"]

    pose = poses.get(filename)

    if pose is None:
        return {
            "frame": filename,
            "status": "POSE_UNAVAILABLE",
        }

    image = cv2.imread(
        str(FRAME_DIR / filename),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return {
            "frame": filename,
            "status": "IMAGE_UNAVAILABLE",
        }

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    segments = detect_segments(
        gray,
        detector,
    )

    original_pixels, original_valid = (
        project_line(
            points,
            pose,
            image.shape,
        )
    )

    best_control = frame_record.get(
        "best_shifted_control"
    )

    control_name = (
        best_control.get("name")
        if isinstance(best_control, dict)
        else None
    )

    control_points = shifted_line(
        points,
        control_name,
    )

    control_projection = None

    if control_points is not None:
        control_projection = project_line(
            control_points,
            pose,
            image.shape,
        )

    projections = [
        (
            original_pixels,
            original_valid,
        )
    ]

    if control_projection is not None:
        projections.append(
            control_projection
        )

    nearby = nearby_segments(
        segments,
        projections,
    )

    annotated = image.copy()

    for _, _, segment_index in nearby:
        segment = segments[
            segment_index
        ]

        p0 = tuple(
            np.rint(
                segment[0]
            ).astype(int)
        )

        p1 = tuple(
            np.rint(
                segment[1]
            ).astype(int)
        )

        cv2.line(
            annotated,
            p0,
            p1,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    original_drawn = draw_projection(
        annotated,
        original_pixels,
        original_valid,
        (0, 255, 0),
        4,
    )

    control_drawn = 0

    if control_projection is not None:
        control_drawn = draw_projection(
            annotated,
            control_projection[0],
            control_projection[1],
            (0, 0, 255),
            3,
        )

    original_support = (
        frame_record.get(
            "original", {}
        ).get("support_fraction")
    )

    control_support = (
        best_control.get(
            "support_fraction"
        )
        if isinstance(best_control, dict)
        else None
    )

    info = [
        f"Candidate {candidate_id} | {filename}",
        f"Original support: {original_support}",
        f"Shifted support: {control_support}",
        f"Shift: {control_name}",
        f"LSD near projections: {len(nearby)}",
    ]

    left = make_panel(
        image,
        "Original camera frame",
    )

    right = make_panel(
        annotated,
        "Projection / LSD overlay",
    )

    comparison = np.hstack(
        [left, right]
    )

    # Place diagnostics in a separate footer so text does
    # not obscure the projected line itself.
    footer = np.zeros(
        (
            165,
            comparison.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    add_text(
        footer,
        info,
        x=18,
        y=30,
    )

    comparison = np.vstack(
        [comparison, footer]
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(destination),
        comparison,
    ):
        raise RuntimeError(
            f"Could not save {destination}"
        )

    return {
        "frame": filename,
        "status": "AUDIT_IMAGE_CREATED",
        "output": str(
            destination.relative_to(ROOT)
        ),
        "original_support": original_support,
        "shifted_support": control_support,
        "shift_name": control_name,
        "globally_detected_segments": len(
            segments
        ),
        "nearby_segments": len(nearby),
        "original_drawn_segments": original_drawn,
        "shifted_drawn_segments": control_drawn,
        "step103_status": frame_record.get(
            "status"
        ),
    }


def make_contact_sheet(
    images,
    destination,
):
    if not images:
        return False

    thumbnails = []

    target_width = 720

    for path in images:
        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            continue

        ratio = (
            target_width
            / image.shape[1]
        )

        height = int(
            round(
                image.shape[0]
                * ratio
            )
        )

        thumbnails.append(
            cv2.resize(
                image,
                (
                    target_width,
                    height,
                ),
                interpolation=cv2.INTER_AREA,
            )
        )

    if not thumbnails:
        return False

    tile_height = max(
        image.shape[0]
        for image in thumbnails
    )

    tile_width = target_width

    rows = math.ceil(
        len(thumbnails)
        / CONTACT_COLUMNS
    )

    sheet = np.full(
        (
            rows * tile_height,
            CONTACT_COLUMNS * tile_width,
            3,
        ),
        25,
        dtype=np.uint8,
    )

    for index, image in enumerate(
        thumbnails
    ):
        row = index // CONTACT_COLUMNS
        col = index % CONTACT_COLUMNS

        y = row * tile_height
        x = col * tile_width

        sheet[
            y:y + image.shape[0],
            x:x + image.shape[1],
        ] = image

    return bool(
        cv2.imwrite(
            str(destination),
            sheet,
        )
    )


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = load_lines()
    evidence = load_evidence()
    poses = load_poses()

    detector = cv2.createLineSegmentDetector(
        cv2.LSD_REFINE_STD
    )

    results = []

    for candidate_id in TARGET_IDS:
        if candidate_id not in lines:
            raise RuntimeError(
                f"Missing line {candidate_id}"
            )

        if candidate_id not in evidence:
            raise RuntimeError(
                f"Missing Step 103 evidence {candidate_id}"
            )

        candidate_dir = (
            OUT / f"candidate_{candidate_id:03d}"
        )

        candidate_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        audited = []
        paths = []

        for frame_record in evidence[
            candidate_id
        ].get("frames", []):
            filename = frame_record["frame"]

            destination = (
                candidate_dir
                / (
                    Path(filename).stem
                    + "_audit.jpg"
                )
            )

            result = audit_frame(
                candidate_id,
                lines[candidate_id],
                frame_record,
                poses,
                detector,
                destination,
            )

            audited.append(result)

            if result["status"] == "AUDIT_IMAGE_CREATED":
                paths.append(destination)

        sheet_path = (
            candidate_dir
            / "contact_sheet.jpg"
        )

        sheet_created = make_contact_sheet(
            paths,
            sheet_path,
        )

        results.append(
            {
                "candidate_id": candidate_id,
                "step103_decision_unchanged": (
                    evidence[
                        candidate_id
                    ]["decision"]
                ),
                "frames_requested": len(
                    evidence[
                        candidate_id
                    ].get("frames", [])
                ),
                "audit_images_created": len(
                    paths
                ),
                "contact_sheet_created": (
                    sheet_created
                ),
                "contact_sheet": (
                    str(
                        sheet_path.relative_to(ROOT)
                    )
                    if sheet_created
                    else None
                ),
                "frames": audited,
                "physical_line_verified": False,
            }
        )

        print(
            f"Candidate {candidate_id:03d}: "
            f"{len(paths)} audit images; "
            f"contact_sheet={sheet_created}",
            flush=True,
        )

    save_json(
        OUT / "per_candidate_audit.json",
        {
            "method": (
                "step104_visual_crossview_audit"
            ),
            "results": results,
        },
    )

    report = {
        "method": "step104_visual_crossview_audit",
        "candidate_count": len(results),
        "audit_images_created": sum(
            row["audit_images_created"]
            for row in results
        ),
        "contact_sheets_created": sum(
            int(row["contact_sheet_created"])
            for row in results
        ),
        "candidate_summaries": [
            {
                "candidate_id": row["candidate_id"],
                "audit_images_created": row[
                    "audit_images_created"
                ],
                "contact_sheet": row[
                    "contact_sheet"
                ],
            }
            for row in results
        ],
        "geometry_modified": False,
        "camera_poses_modified": False,
        "step103_decisions_modified": False,
        "physical_lines_verified": False,
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "limitations": [
            "Overlays visualize frozen hypotheses and nearby detected image segments.",
            "The shifted alternative is selected from Step 103 comparisons and is not independently validated.",
            "Nearby LSD segments are not verified physical correspondences.",
            "Occlusion, pose error, and structural identity remain unresolved.",
            "A contact sheet is a diagnostic visualization, not a verification result.",
            "No CAD matching or construction progress is measured.",
        ],
        "decision": "VISUAL_AUDIT_ARTIFACTS_CREATED",
    }

    save_json(
        OUT / "audit_report.json",
        report,
    )

    print("\n=== STEP 104 COMPLETE ===")
    print(
        "Audit images:",
        report["audit_images_created"]
    )
    print(
        "Contact sheets:",
        report["contact_sheets_created"]
    )
    print("Physical lines verified: NO")
    print("CAD registration verified: NO")
    print("Construction progress verified: NO")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
