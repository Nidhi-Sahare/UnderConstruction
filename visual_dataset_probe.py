
"""
BuildProgress - Step 8: Visual Dataset Probe

Inspect small image assets from the Hilti dataset without
downloading large ROS recordings.

Run:
    python visual_dataset_probe.py

Optional:
    python visual_dataset_probe.py --inspect-only
    python visual_dataset_probe.py --max-mb 100

Outputs:
    data/visual_probe/visual_inventory.json
    data/visual_probe/contact_sheet.png
    data/visual_probe/images/

Dependencies:
    huggingface_hub, pillow
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image, ImageDraw, ImageOps


REPO_ID = "Hilti-Research/hilti-trimble-slam-challenge-2026"
REPO_TYPE = "dataset"

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "data" / "visual_probe"
IMAGE_DIR = OUTPUT / "images"

INVENTORY_PATH = OUTPUT / "visual_inventory.json"
CONTACT_SHEET_PATH = OUTPUT / "contact_sheet.png"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp",
    ".bmp", ".tif", ".tiff",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv",
    ".webm",
}

RECORDING_EXTENSIONS = {
    ".db3", ".mcap", ".bag",
}

MAX_SINGLE_FILE_MB = 25
DEFAULT_BUDGET_MB = 100

THUMBNAIL_WIDTH = 260
THUMBNAIL_HEIGHT = 180


def size_mb(size):
    if size is None:
        return None

    return round(size / (1024 * 1024), 3)


def classify_file(path):
    extension = Path(path).suffix.lower()

    if extension in IMAGE_EXTENSIONS:
        return "image"

    if extension in VIDEO_EXTENSIONS:
        return "video"

    if extension in RECORDING_EXTENSIONS:
        return "sensor_recording"

    return None


def scan_repository():
    print("Fetching remote dataset file metadata...")

    api = HfApi()

    files = api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        recursive=True,
        expand=True,
    )

    results = []

    for item in files:
        if getattr(item, "type", None) == "directory":
            continue

        path = getattr(item, "path", None)

        if not path:
            continue

        category = classify_file(path)

        if category is None:
            continue

        size = getattr(item, "size", None)

        results.append({
            "path": path,
            "category": category,
            "size_bytes": size,
            "size_mb": size_mb(size),
        })

    return sorted(
        results,
        key=lambda item: (
            item["category"],
            item["size_bytes"]
            if item["size_bytes"] is not None
            else float("inf"),
            item["path"],
        ),
    )


def safe_image_name(remote_path):
    """
    Generate a local name that avoids collisions between
    files with identical basenames in different folders.
    """

    import hashlib

    digest = hashlib.sha256(
        remote_path.encode("utf-8")
    ).hexdigest()[:12]

    original = Path(remote_path)

    return f"{original.stem[:60]}_{digest}{original.suffix.lower()}"


def download_images(candidates, budget_mb):
    budget_bytes = int(budget_mb * 1024 * 1024)
    single_limit = MAX_SINGLE_FILE_MB * 1024 * 1024

    used_bytes = 0
    downloaded = []
    skipped = []
    errors = []

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for item in candidates:
        size = item["size_bytes"]

        if size is None:
            skipped.append({
                "path": item["path"],
                "reason": "Unknown remote file size",
            })
            continue

        if size > single_limit:
            skipped.append({
                "path": item["path"],
                "reason": "Exceeds single-file limit",
            })
            continue

        if used_bytes + size > budget_bytes:
            skipped.append({
                "path": item["path"],
                "reason": "Exceeds total download budget",
            })
            continue

        print(
            f"Downloading: {item['path']} "
            f"({item['size_mb']} MB)"
        )

        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                filename=item["path"],
            )

            target = IMAGE_DIR / safe_image_name(
                item["path"]
            )

            if not target.exists():
                import shutil
                shutil.copy2(cached_path, target)

            with Image.open(target) as image:
                image.verify()

            with Image.open(target) as image:
                width, height = image.size
                image_format = image.format

            actual_size = target.stat().st_size

            used_bytes += actual_size

            downloaded.append({
                "remote_path": item["path"],
                "local_path": str(target),
                "width": width,
                "height": height,
                "format": image_format,
                "size_mb": size_mb(actual_size),
            })

        except Exception as exc:
            errors.append({
                "path": item["path"],
                "error": str(exc),
            })

    return downloaded, skipped, errors


def make_contact_sheet(downloaded):
    if not downloaded:
        return False

    columns = 3
    rows = math.ceil(len(downloaded) / columns)

    cell_width = 290
    cell_height = 235

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, item in enumerate(downloaded):
        column = index % columns
        row = index // columns

        x = column * cell_width
        y = row * cell_height

        try:
            with Image.open(item["local_path"]) as image:
                image = ImageOps.exif_transpose(
                    image
                ).convert("RGB")

                thumb = ImageOps.contain(
                    image,
                    (
                        THUMBNAIL_WIDTH,
                        THUMBNAIL_HEIGHT,
                    ),
                )

            sheet.paste(
                thumb,
                (
                    x + 15,
                    y + 10,
                ),
            )

            filename = Path(
                item["remote_path"]
            ).name

            label = filename[:38]

            draw.text(
                (x + 15, y + 195),
                label,
                fill="black",
            )

            draw.text(
                (x + 15, y + 211),
                f"{item['width']} x {item['height']}",
                fill="#555555",
            )

        except Exception as exc:
            draw.text(
                (x + 15, y + 20),
                f"Error: {str(exc)[:30]}",
                fill="red",
            )

    sheet.save(CONTACT_SHEET_PATH)
    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="List remote files without downloading.",
    )

    parser.add_argument(
        "--max-mb",
        type=float,
        default=DEFAULT_BUDGET_MB,
        help="Maximum total image download size in MB.",
    )

    args = parser.parse_args()

    if args.max_mb < 0:
        parser.error("--max-mb cannot be negative")

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = scan_repository()

    images = [
        item for item in files
        if item["category"] == "image"
    ]

    videos = [
        item for item in files
        if item["category"] == "video"
    ]

    recordings = [
        item for item in files
        if item["category"] == "sensor_recording"
    ]

    print("\n" + "=" * 65)
    print("BUILDPROGRESS - VISUAL DATASET PROBE")
    print("=" * 65)

    print(f"Image files: {len(images)}")
    print(f"Direct video files: {len(videos)}")
    print(f"Sensor recording files: {len(recordings)}")

    print("\nIMAGE CANDIDATES")

    for item in images:
        print(
            f"  {item['size_mb']} MB | "
            f"{item['path']}"
        )

    print("\nSMALLEST RECORDINGS (NOT DOWNLOADED)")

    known_recordings = sorted(
        [
            item for item in recordings
            if item["size_bytes"] is not None
        ],
        key=lambda item: item["size_bytes"],
    )

    for item in known_recordings[:5]:
        print(
            f"  {item['size_mb']} MB | "
            f"{item['path']}"
        )

    downloaded = []
    skipped = []
    errors = []

    if not args.inspect_only:
        print(
            f"\nImage download budget: "
            f"{args.max_mb:.1f} MB"
        )

        downloaded, skipped, errors = download_images(
            images,
            args.max_mb,
        )

    report = {
        "dataset": REPO_ID,
        "inspect_only": args.inspect_only,
        "download_budget_mb": args.max_mb,
        "counts": dict(
            Counter(
                item["category"]
                for item in files
            )
        ),
        "image_candidates": images,
        "direct_video_candidates": videos,
        "sensor_recording_candidates": recordings,
        "downloaded_images": downloaded,
        "skipped_images": skipped,
        "download_errors": errors,
    }

    INVENTORY_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    contact_created = make_contact_sheet(
        downloaded
    )

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    print(f"Images downloaded: {len(downloaded)}")
    print(f"Images skipped: {len(skipped)}")
    print(f"Download errors: {len(errors)}")

    if downloaded:
        print("\nDOWNLOADED IMAGE DETAILS")

        for item in downloaded:
            print(
                f"  {item['width']}x{item['height']} | "
                f"{item['remote_path']}"
            )

    print(f"\nInventory: {INVENTORY_PATH}")

    if contact_created:
        print(f"Contact sheet: {CONTACT_SHEET_PATH}")
    else:
        print(
            "No contact sheet created: "
            "no usable images were downloaded."
        )

    print(
        "\nNo large sensor recordings were downloaded."
    )


if __name__ == "__main__":
    main()
