
"""
BuildProgress - Floor Plan Inspector

Purpose:
    Download one small DXF floor plan and inspect
    its CAD structure without processing large videos.

Requirements:
    pip install huggingface_hub ezdxf

Run:
    python floorplan_inspector.py
"""

import json
from collections import Counter
from pathlib import Path

import ezdxf
from ezdxf import bbox
from ezdxf import units
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DATA_DIR = PROJECT_ROOT / "data"
FLOORPLAN_DIR = DATA_DIR / "floorplans"

DATASET_ID = (
    "Hilti-Research/"
    "hilti-trimble-slam-challenge-2026"
)

REMOTE_FILE = "floorplans/dxf_format/floor_2.dxf"

LOCAL_FILE = FLOORPLAN_DIR / "floor_2.dxf"

REPORT_FILE = FLOORPLAN_DIR / "floor_2_report.json"

MAX_DISPLAY_LAYERS = 40
MAX_DISPLAY_BLOCKS = 20


# --------------------------------------------------
# Download
# --------------------------------------------------

def download_floorplan():
    """
    Download the DXF using the official Hugging Face client.

    The client may use its own cache during transfer.
    The final file is copied into our project folder.
    """

    FLOORPLAN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if LOCAL_FILE.exists():
        print("Floor plan already exists.")
        print(f"Using: {LOCAL_FILE}")
        return LOCAL_FILE

    print("Downloading Floor 2 DXF...")

    cached_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=REMOTE_FILE,
        repo_type="dataset",
        local_dir=DATA_DIR / "hilti_download",
    )

    cached_path = Path(cached_path)

    # Move the downloaded file into our chosen location.
    # The Hugging Face client may also retain metadata
    # in its local download directory.
    import shutil

    shutil.copy2(
        cached_path,
        LOCAL_FILE,
    )

    print(f"Saved floor plan: {LOCAL_FILE}")

    return LOCAL_FILE


# --------------------------------------------------
# CAD Inspection
# --------------------------------------------------

def inspect_layers(document):
    """
    Read CAD layers.

    A layer groups drawing entities, potentially
    separating walls, dimensions, text, etc.
    """

    layers = []

    for layer in document.layers:
        layers.append({
            "name": layer.dxf.name,
            "color": layer.dxf.color,
            "is_off": layer.is_off(),
            "is_frozen": layer.is_frozen(),
        })

    return layers


def inspect_entities(modelspace):
    """
    Count entity types and layers in modelspace.

    Examples:
        LINE
        LWPOLYLINE
        POLYLINE
        INSERT
        TEXT
        MTEXT
    """

    type_counts = Counter()
    layer_counts = Counter()

    examples = []

    for entity in modelspace:
        entity_type = entity.dxftype()

        type_counts[entity_type] += 1

        layer = entity.dxf.get(
            "layer",
            "0",
        )

        layer_counts[layer] += 1

        if len(examples) < 12:
            examples.append({
                "type": entity_type,
                "layer": layer,
                "handle": entity.dxf.get(
                    "handle",
                    None,
                ),
            })

    return {
        "entity_types": dict(type_counts),
        "entities_per_layer": dict(layer_counts),
        "examples": examples,
        "total_entities": sum(type_counts.values()),
    }


def inspect_blocks(document):
    """
    Inspect block definitions.

    Blocks are reusable collections of CAD entities.
    Some drawings represent doors, windows or
    repeated architectural elements as blocks.
    """

    blocks = []

    for block in document.blocks:
        blocks.append({
            "name": block.name,
            "entity_count": len(block),
        })

    return blocks


def inspect_bounds(modelspace):
    """
    Estimate the geometric bounding box of modelspace.

    The result is expressed in the DXF drawing's
    coordinate units.
    """

    try:
        bounds = bbox.extents(
            modelspace,
            fast=True,
        )

        if not bounds.has_data:
            return None

        return {
            "minimum": list(bounds.extmin),
            "maximum": list(bounds.extmax),
            "size": list(bounds.size),
        }

    except Exception as error:
        return {
            "error": str(error),
        }


def inspect_floorplan(path):
    """Open and analyze the DXF drawing."""

    print("\nOpening DXF drawing...")

    document = ezdxf.readfile(path)
    modelspace = document.modelspace()

    unit_code = document.header.get(
        "$INSUNITS",
        0,
    )

    unit_name = units.unit_name(unit_code)

    report = {
        "file": str(path),
        "dxf_version": document.dxfversion,
        "units_code": unit_code,
        "units_name": unit_name,
        "layers": inspect_layers(document),
        "entities": inspect_entities(modelspace),
        "blocks": inspect_blocks(document),
        "bounds": inspect_bounds(modelspace),
    }

    return report


# --------------------------------------------------
# Reporting
# --------------------------------------------------

def print_report(report):
    """Print the most important CAD information."""

    print("\n" + "=" * 65)
    print("BUILDPROGRESS - FLOOR PLAN INSPECTION")
    print("=" * 65)

    print(f"\nFile: {report['file']}")
    print(f"DXF version: {report['dxf_version']}")
    print(f"Units: {report['units_name']}")
    print(f"Unit code: {report['units_code']}")

    entities = report["entities"]

    print("\nENTITY TYPES")
    print("-" * 65)

    for name, count in sorted(
        entities["entity_types"].items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{name:20} {count:8}")

    print(
        f"\nTotal modelspace entities: "
        f"{entities['total_entities']}"
    )

    print("\nMOST POPULATED LAYERS")
    print("-" * 65)

    for name, count in sorted(
        entities["entities_per_layer"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:MAX_DISPLAY_LAYERS]:
        print(f"{count:8}  {name}")

    print("\nBLOCK DEFINITIONS")
    print("-" * 65)

    for block in report["blocks"][:MAX_DISPLAY_BLOCKS]:
        print(
            f"{block['entity_count']:8}  "
            f"{block['name']}"
        )

    print("\nDRAWING BOUNDS")
    print("-" * 65)

    bounds = report["bounds"]

    if bounds is None:
        print("No geometric bounds available.")
    elif "error" in bounds:
        print(f"Bounds calculation failed: {bounds['error']}")
    else:
        print(f"Minimum: {bounds['minimum']}")
        print(f"Maximum: {bounds['maximum']}")
        print(f"Size:    {bounds['size']}")

    print("\n" + "=" * 65)


def save_report(report):
    """Save the full inspection results as JSON."""

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nReport saved: {REPORT_FILE}")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

def main():
    try:
        path = download_floorplan()

        report = inspect_floorplan(path)

        print_report(report)

        save_report(report)

        print("\nFloor plan inspection complete.")

    except Exception as error:
        print("\nInspection failed.")
        print(f"Error: {type(error).__name__}")
        print(f"Details: {error}")

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

