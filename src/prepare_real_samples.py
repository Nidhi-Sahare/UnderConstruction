
"""
Under-construction — Step 118B
Prepare real construction dataset samples.

Run:
    .\\.venv\\Scripts\\python.exe src\\prepare_real_samples.py

Creates:
    sample_gallery/real_floor_2/
        project.json
        frame_index.json
        floor_plan.dxf
        frames/
            cam0_00000.jpg
            ...
            cam0_00179.jpg

Uses original real dataset imagery.
Does not fabricate CAD-video alignment or construction progress.
"""

from pathlib import Path
import csv
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]

CAD_SOURCE = ROOT / "data" / "floorplans" / "floor_2.dxf"

RECORDING_ROOT = (
    ROOT
    / "_archive"
    / "hilti_transition_20261023_004237"
    / "data"
    / "hilti_frames"
    / "floor_2_2025-10-28_run_2"
    / "cam0"
)

FRAME_SOURCE = RECORDING_ROOT / "frames"
MANIFEST_SOURCE = RECORDING_ROOT / "frame_manifest.csv"

OUTPUT = ROOT / "sample_gallery" / "real_floor_2"


def copy_if_needed(source, destination):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        if (
            destination.stat().st_size
            == source.stat().st_size
        ):
            return False

    shutil.copy2(source, destination)
    return True


def main():
    required = [
        CAD_SOURCE,
        FRAME_SOURCE,
        MANIFEST_SOURCE,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing source files:\n"
            + "\n".join(missing)
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_output = OUTPUT / "frames"
    frames_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MANIFEST_SOURCE.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        manifest = list(csv.DictReader(file))

    if not manifest:
        raise RuntimeError(
            "Frame manifest is empty."
        )

    frame_index = []
    copied = 0
    unmatched = 0

    for index, row in enumerate(manifest):
        filename = Path(
            row.get("filename", "")
        ).name

        if not filename:
            raise RuntimeError(
                f"Missing filename at row {index}."
            )

        source = FRAME_SOURCE / filename

        if not source.is_file():
            raise FileNotFoundError(
                f"Missing frame: {source}"
            )

        destination = frames_output / filename

        if copy_if_needed(source, destination):
            copied += 1

        pose_matched = (
            str(row.get("pose_matched", ""))
            .strip()
            .lower()
            == "true"
        )

        if not pose_matched:
            unmatched += 1

        frame_index.append({
            "index": index,
            "filename": filename,
            "image": f"frames/{filename}",
            "timestamp_s": row.get(
                "header_timestamp_s"
            ) or row.get("bag_timestamp_s"),
            "pose_matched": pose_matched,
        })

    copy_if_needed(
        CAD_SOURCE,
        OUTPUT / "floor_plan.dxf",
    )

    metadata = {
        "id": "real_floor_2",
        "name": "Floor 2 Construction Walkthrough",
        "subtitle": "Real construction dataset",
        "description": (
            "Explore the Floor 2 CAD drawing alongside "
            "real construction-site camera frames."
        ),
        "source": "Hilti construction dataset",
        "data_type": "real",
        "floor_plan": "floor_plan.dxf",
        "frame_index": "frame_index.json",
        "frame_count": len(frame_index),
        "camera": "cam0",
        "recording": "floor_2_2025-10-28_run_2",
        "cad_video_alignment_verified": False,
        "construction_progress_assessed": False,
        "viewer_mode": "image_sequence",
    }

    (OUTPUT / "frame_index.json").write_text(
        json.dumps(
            {
                "sample_id": "real_floor_2",
                "frames": frame_index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (OUTPUT / "project.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print(" STEP 118B — REAL SAMPLE PREPARED")
    print("=" * 55)

    print("Sample:", metadata["name"])
    print("Source:", metadata["source"])
    print("CAD:", "floor_plan.dxf")
    print("Frames:", len(frame_index))
    print("Copied this run:", copied)
    print("Frames without matched poses:", unmatched)
    print("Output:", OUTPUT)
    print()
    print("Original dataset files were not modified.")
    print("CAD-video alignment has not been verified.")
    print("Construction progress has not been assessed.")


if __name__ == "__main__":
    main()
