
"""
BuildProgress - Dataset Inspector

Lists public dataset files without downloading recordings.

Run:
    python dataset_inspector.py

Dependency:
    pip install huggingface_hub
"""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


DATASET_ID = "Hilti-Research/hilti-trimble-slam-challenge-2026"

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = PROJECT_DIR / "data" / "dataset_inventory.json"

RECORDING_EXTENSIONS = {".mcap", ".bag", ".db3", ".mp4", ".mov"}
PLAN_EXTENSIONS = {".dxf", ".ifc", ".dwg"}


def format_size(size_bytes):
    """Convert byte counts into readable sizes."""

    if size_bytes is None:
        return "Unknown"

    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024

    return "Unknown"


def classify_file(path):
    """Classify a file by its extension."""

    extension = Path(path).suffix.lower()

    if extension in RECORDING_EXTENSIONS:
        return "Recording"

    if extension in PLAN_EXTENSIONS:
        return "Floor plan / BIM"

    if extension in {".png", ".jpg", ".jpeg"}:
        return "Image"

    if extension in {".json", ".yaml", ".yml"}:
        return "Metadata"

    if extension in {".csv", ".txt"}:
        return "Text / tabular"

    return "Other"


def retrieve_inventory():
    """
    Query the dataset repository using the official client.

    This retrieves file metadata only.
    It does not download dataset contents.
    """

    api = HfApi()

    print("Retrieving dataset information...")
    dataset = api.dataset_info(DATASET_ID)

    print("Listing repository files...")

    files = []

    for entry in api.list_repo_tree(
        repo_id=DATASET_ID,
        repo_type="dataset",
        recursive=True,
        expand=False,
    ):
        # Folders have no file size.
        # Keep only entries representing files.
        if not hasattr(entry, "size"):
            continue

        path = entry.path
        size = entry.size

        files.append({
            "path": path,
            "size_bytes": size,
            "category": classify_file(path),
        })

    return {
        "dataset_id": DATASET_ID,
        "revision": dataset.sha,
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "files": files,
    }


def display_inventory(inventory):
    """Print a summary of the available files."""

    files = inventory["files"]

    print("\n" + "=" * 65)
    print("BUILDPROGRESS - DATASET INSPECTOR")
    print("=" * 65)

    print(f"Dataset: {inventory['dataset_id']}")
    print(f"Files discovered: {len(files)}")

    categories = Counter(
        item["category"] for item in files
    )

    print("\nFILE CATEGORIES")
    print("-" * 65)

    for category, count in sorted(categories.items()):
        print(f"{category:20} {count:5}")

    known_sizes = [
        item["size_bytes"]
        for item in files
        if isinstance(item["size_bytes"], int)
    ]

    print("\nTotal known repository size:")
    print(format_size(sum(known_sizes)))

    unknown_count = len(files) - len(known_sizes)

    if unknown_count:
        print(f"Unknown file sizes: {unknown_count}")

    print("\nFLOOR PLAN / BIM FILES")
    print("-" * 65)

    plans = [
        item for item in files
        if item["category"] == "Floor plan / BIM"
    ]

    if not plans:
        print("No matching vector plan files found.")

    for item in plans[:30]:
        print(
            f"{format_size(item['size_bytes']):>12}  "
            f"{item['path']}"
        )

    print("\nSMALLEST RECORDING FILES")
    print("-" * 65)

    recordings = [
        item for item in files
        if item["category"] == "Recording"
    ]

    recordings.sort(
        key=lambda item: (
            item["size_bytes"] is None,
            item["size_bytes"]
            if item["size_bytes"] is not None
            else float("inf"),
        )
    )

    if not recordings:
        print("No recording files found.")

    for item in recordings[:20]:
        print(
            f"{format_size(item['size_bytes']):>12}  "
            f"{item['path']}"
        )

    print("\n" + "=" * 65)


def save_inventory(inventory):
    """Write the metadata inventory to a JSON file."""

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            inventory,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved inventory: {OUTPUT_FILE}")


def main():
    try:
        inventory = retrieve_inventory()

        display_inventory(inventory)
        save_inventory(inventory)

        print("\nInspection completed successfully.")
        print("No dataset recordings were downloaded.")

    except Exception as error:
        print("\nDataset inspection failed.")
        print(f"Error type: {type(error).__name__}")
        print(f"Details: {error}")

        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

