
"""
Under-construction — Conservative CAD Wall Geometry Cleanup

Endpoint:
    GET /api/twin/cleaned

Purpose:
    Merge truly collinear overlapping/touching CAD wall centerlines
    without moving them onto an approximate reference line.

Safety:
    - No intentional gap bridging.
    - No source CAD modifications.
    - No wall cuts or opening inference.
    - Original element IDs are preserved.
    - Height/thickness remain assumptions unless source says otherwise.
    - Construction status remains UNKNOWN.

Designed to work with:
    digital_twin_3d.get_elements()
    architectural_geometry.get_reconstruction()
"""

from __future__ import annotations

import math
from collections import defaultdict

from flask import jsonify

from underconstruction_app import app
from digital_twin_3d import get_elements


MIN_LENGTH_M = 0.01

# Numerical tolerance only, not an architectural snapping distance.
COLLINEAR_TOLERANCE_M = 1e-7
ANGLE_TOLERANCE_RAD = 1e-8

# A small tolerance for floating-point endpoint equality.
JOIN_TOLERANCE_M = 1e-7


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return number if math.isfinite(number) else None


def valid_point(point):
    if not isinstance(point, (list, tuple)):
        return None

    if len(point) < 2:
        return None

    x = finite(point[0])
    y = finite(point[1])

    if x is None or y is None:
        return None

    return (x, y)


def normalize_segment(element):
    start = valid_point(element.get("start"))
    end = valid_point(element.get("end"))

    if start is None or end is None:
        return None

    dx = end[0] - start[0]
    dy = end[1] - start[1]

    length = math.hypot(dx, dy)

    if length < MIN_LENGTH_M:
        return None

    ux = dx / length
    uy = dy / length

    # Canonical orientation.
    if ux < 0 or (abs(ux) < 1e-12 and uy < 0):
        ux = -ux
        uy = -uy

    nx = -uy
    ny = ux

    offset = nx * start[0] + ny * start[1]

    t1 = ux * start[0] + uy * start[1]
    t2 = ux * end[0] + uy * end[1]

    return {
        "element": element,
        "start": start,
        "end": end,
        "ux": ux,
        "uy": uy,
        "offset": offset,
        "lo": min(t1, t2),
        "hi": max(t1, t2),
    }


def compatible(a, b):
    ea = a["element"]
    eb = b["element"]

    if ea.get("category") != eb.get("category"):
        return False

    if abs(
        float(ea.get("height_m", 0))
        - float(eb.get("height_m", 0))
    ) > 1e-9:
        return False

    if abs(
        float(ea.get("thickness_m", 0))
        - float(eb.get("thickness_m", 0))
    ) > 1e-9:
        return False

    cross = abs(
        a["ux"] * b["uy"] - a["uy"] * b["ux"]
    )

    if cross > ANGLE_TOLERANCE_RAD:
        return False

    # Check actual endpoints against the first segment's line.
    nx = -a["uy"]
    ny = a["ux"]

    for point in (b["start"], b["end"]):
        distance = abs(
            nx * (point[0] - a["start"][0])
            + ny * (point[1] - a["start"][1])
        )

        if distance > COLLINEAR_TOLERANCE_M:
            return False

    return True


def projected_interval(segment, reference):
    ux = reference["ux"]
    uy = reference["uy"]

    origin = reference["start"]

    values = []

    for point in (segment["start"], segment["end"]):
        values.append(
            ux * (point[0] - origin[0])
            + uy * (point[1] - origin[1])
        )

    return min(values), max(values)


def merge_group(group):
    reference = group[0]

    ux = reference["ux"]
    uy = reference["uy"]

    origin_x, origin_y = reference["start"]

    intervals = []

    for segment in group:
        lo, hi = projected_interval(segment, reference)

        intervals.append({
            "lo": lo,
            "hi": hi,
            "element": segment["element"],
        })

    intervals.sort(key=lambda item: (item["lo"], item["hi"]))

    runs = []
    current = []
    current_end = None

    for interval in intervals:
        if current_end is None:
            current = [interval]
            current_end = interval["hi"]
            continue

        if interval["lo"] <= current_end + JOIN_TOLERANCE_M:
            current.append(interval)
            current_end = max(
                current_end,
                interval["hi"],
            )
        else:
            runs.append(current)
            current = [interval]
            current_end = interval["hi"]

    if current:
        runs.append(current)

    output = []

    for run in runs:
        lo = min(item["lo"] for item in run)
        hi = max(item["hi"] for item in run)

        # The endpoints are on the original reference line.
        # Other members must already have passed the strict
        # collinearity check.
        start = [
            origin_x + ux * lo,
            origin_y + uy * lo,
        ]

        end = [
            origin_x + ux * hi,
            origin_y + uy * hi,
        ]

        source_elements = [
            item["element"]
            for item in run
        ]

        source_ids = []

        for element in source_elements:
            ids = element.get("source_element_ids")

            if not ids:
                ids = [element.get("id")]

            for source_id in ids:
                if source_id is not None:
                    source_ids.append(str(source_id))

        source_ids = list(dict.fromkeys(source_ids))

        representative = source_elements[0]

        output.append({
            "id": source_ids[0] if source_ids else "unknown",
            "source_element_ids": source_ids,
            "source_element_count": len(source_ids),
            "category": representative.get(
                "category", "unknown"
            ),
            "start": start,
            "end": end,
            "length_m": math.dist(start, end),
            "height_m": representative["height_m"],
            "thickness_m": representative["thickness_m"],
            "height_source": representative.get(
                "height_source", "ASSUMED"
            ),
            "thickness_source": representative.get(
                "thickness_source", "ASSUMED"
            ),
            "construction_status": "UNKNOWN",
            "progress_verified": False,
            "openings_verified": False,
        })

    return output


def cleanup(elements):
    segments = []
    rejected_ids = []

    for element in elements:
        segment = normalize_segment(element)

        if segment is None:
            rejected_ids.append(element.get("id"))
            continue

        segments.append(segment)

    groups = []

    for segment in segments:
        matched_group = None

        for group in groups:
            if compatible(group[0], segment):
                matched_group = group
                break

        if matched_group is None:
            groups.append([segment])
        else:
            matched_group.append(segment)

    cleaned = []

    for group in groups:
        cleaned.extend(merge_group(group))

    return cleaned, rejected_ids


def validate_cleanup(original, cleaned, rejected_ids):
    original_ids = set()

    for element in original:
        ids = element.get("source_element_ids")

        if not ids:
            ids = [element.get("id")]

        original_ids.update(
            str(value)
            for value in ids
            if value is not None
        )

    cleaned_ids = set()

    for element in cleaned:
        cleaned_ids.update(
            str(value)
            for value in element["source_element_ids"]
        )

    missing_ids = sorted(original_ids - cleaned_ids)
    unexpected_ids = sorted(cleaned_ids - original_ids)

    return {
        "original_element_count": len(original),
        "cleaned_element_count": len(cleaned),
        "rejected_element_count": len(rejected_ids),
        "rejected_element_ids": rejected_ids,
        "missing_source_ids": missing_ids,
        "unexpected_source_ids": unexpected_ids,
        "source_id_coverage_passed": (
            not missing_ids and not unexpected_ids
        ),
        "merged_mesh_count": sum(
            element["source_element_count"] > 1
            for element in cleaned
        ),
        "geometric_policy": (
            "STRICT_COLLINEAR_OVERLAP_OR_TOUCH"
        ),
        "gap_bridging_authorized": False,
        "opening_cuts_authorized": False,
        "construction_progress_verified": False,
    }


@app.route("/api/twin/cleaned")
def api_cleaned_twin():
    try:
        original = get_elements()

        cleaned, rejected_ids = cleanup(original)

        validation = validate_cleanup(
            original,
            cleaned,
            rejected_ids,
        )

        return jsonify({
            "elements": cleaned,
            "original_element_count": len(original),
            "rendered_mesh_count": len(cleaned),
            "merged_mesh_count": validation[
                "merged_mesh_count"
            ],
            "representation": (
                "STRICT_COLLINEAR_MERGED_CENTERLINES"
            ),
            "validation": validation,
            "source_element_ids_preserved": validation[
                "source_id_coverage_passed"
            ],
            "openings_verified": False,
            "construction_progress_verified": False,
            "camera_registration_verified": False,
        })

    except Exception:
        app.logger.exception(
            "Wall geometry cleanup failed"
        )

        return jsonify({
            "error": "Wall geometry cleanup failed"
        }), 500


if __name__ == "__main__":
    original = get_elements()

    cleaned, rejected = cleanup(original)

    report = validate_cleanup(
        original,
        cleaned,
        rejected,
    )

    print("\nUNDER-CONSTRUCTION WALL CLEANUP")
    print("=" * 45)

    for key, value in report.items():
        print(f"{key}: {value}")
