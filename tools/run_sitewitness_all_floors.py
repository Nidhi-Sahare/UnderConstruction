
"""
BuildProgress / SiteWitness
Three-floor multiview line reconstruction orchestrator.

Reuses tools/reconstruct_multiview_lines.py.
Does not invent missing evidence or verify CAD registration.

Run:
    python tools/run_sitewitness_all_floors.py
"""

from __future__ import annotations

import importlib.util
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "tools/reconstruct_multiview_lines.py"
OUTPUT_ROOT = ROOT / "outputs/sitewitness/all_floors"

FLOORS = (1, 2, 3)


def find_observations(floor):
    candidates = [
        ROOT / f"outputs/line_evidence/floor_{floor}/line_observations.json",
        ROOT / f"outputs/line_evidence/floor_{floor}_line_observations.json",
    ]

    if floor == 2:
        candidates.append(
            ROOT / "outputs/line_evidence/line_observations.json"
        )

    return next((p for p in candidates if p.is_file()), None)


def find_frame_directory(floor):
    candidates = []

    for root in (
        ROOT / "data",
        ROOT / "_archive",
        ROOT / "sample_gallery",
    ):
        if not root.exists():
            continue

        for path in root.rglob("frames"):
            if not path.is_dir():
                continue

            parts = [p.lower() for p in path.parts]

            if any(
                f"floor_{floor}" in part
                for part in parts
            ):
                if list(path.glob("*.jpg")):
                    candidates.append(path)

    # Avoid silently selecting between different walkthroughs.
    candidates = sorted(set(candidates))

    return candidates


def load_engine():
    spec = importlib.util.spec_from_file_location(
        "sitewitness_multiview_lines_engine",
        ENGINE,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load reconstruction engine")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def process_floor(floor):
    result = {
        "floor": floor,
        "status": "NOT_PROCESSED",
        "cad_registration_verified": False,
        "construction_progress_verified": False,
    }

    observations = find_observations(floor)
    frame_directories = find_frame_directory(floor)

    if observations is None:
        result["status"] = "MISSING_LINE_OBSERVATIONS"
        return result

    if not frame_directories:
        result["status"] = "MISSING_WALKTHROUGH_FRAMES"
        return result

    # Use observation frame names to resolve the walkthrough,
    # rather than guessing from directory names alone.
    data = json.loads(observations.read_text(encoding="utf-8"))

    records = data.get("observations", [])

    if not records:
        result["status"] = "EMPTY_LINE_OBSERVATIONS"
        return result

    frame_names = {
        str(item["frame"])
        for item in records
        if isinstance(item, dict) and item.get("frame")
    }

    matches = []

    for directory in frame_directories:
        if all((directory / name).is_file() for name in frame_names):
            matches.append(directory)

    # Prefer the original walkthrough used by the
    # existing Floor 2 reconstruction pipeline.
    if floor == 2:
        original = (
                ROOT
                / "_archive"
                / "hilti_transition_20261023_004237"
                / "data"
                / "hilti_frames"
                / "floor_2_2025-10-28_run_2"
                / "cam0"
                / "frames"
        )

        if original in matches:
            matches = [original]

    if len(matches) != 1:
        result["status"] = (
            "AMBIGUOUS_WALKTHROUGH"
            if matches
            else "WALKTHROUGH_FRAME_MISMATCH"
        )
        result["matching_directories"] = [
            str(p.relative_to(ROOT)) for p in matches
        ]
        return result

    frames = matches[0]
    output = OUTPUT_ROOT / f"floor_{floor}"
    output.mkdir(parents=True, exist_ok=True)

    engine = load_engine()

    # The engine's existing geometry, appearance matching,
    # triangulation and held-out verification remain unchanged.
    engine.INPUT = observations
    engine.FRAME_DIR = frames
    engine.OUTPUT = output

    try:
        engine.main()

        report_path = output / "line_report.json"
        candidates_path = output / "line_candidates.json"

        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        candidates = json.loads(
            candidates_path.read_text(encoding="utf-8")
        )

        # The original engine embeds a Floor 2 sequence label.
        # Correct metadata in these per-floor copies.
        sequence = frames.parent.parent.name

        report["floor"] = floor
        report["sequence"] = sequence

        candidates["floor"] = floor
        candidates["sequence"] = sequence

        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        candidates_path.write_text(
            json.dumps(candidates, indent=2),
            encoding="utf-8",
        )

        result.update({
            "status": "RECONSTRUCTION_COMPLETED",
            "sequence": sequence,
            "line_observations": report.get(
                "line_observations", 0
            ),
            "retained_line_candidates": report.get(
                "retained_supported_candidates", 0
            ),
            "report": str(report_path.relative_to(ROOT)),
            "candidates": str(candidates_path.relative_to(ROOT)),
        })

    except Exception as exc:
        result["status"] = "RECONSTRUCTION_FAILED"
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n=== SITEWITNESS: THREE-FLOOR RECONSTRUCTION ===")

    results = []

    for floor in FLOORS:
        print(f"\n{'=' * 50}")
        print(f"FLOOR {floor}")
        print("=" * 50)

        result = process_floor(floor)
        results.append(result)

        print("Status:", result["status"])

        if "retained_line_candidates" in result:
            print(
                "3D line candidates:",
                result["retained_line_candidates"],
            )

        if "error" in result:
            print("Error:", result["error"])

    summary = {
        "system": "SiteWitness",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "requested_floors": list(FLOORS),
        "completed_floors": [
            r["floor"] for r in results
            if r["status"] == "RECONSTRUCTION_COMPLETED"
        ],
        "cad_registration_verified": False,
        "construction_progress_verified": False,
        "floors": results,
    }

    path = OUTPUT_ROOT / "building_inspection_summary.json"

    path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n=== BUILDING SUMMARY ===")

    for result in results:
        print(
            f"Floor {result['floor']}: {result['status']}"
        )

    print("\nSummary:", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
