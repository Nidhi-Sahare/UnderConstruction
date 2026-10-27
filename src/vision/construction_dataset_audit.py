
"""
BuildProgress - Step 10
Construction Dataset Quality Audit

Usage:
    python construction_dataset_audit.py

Or specify the extracted dataset directory:
    python construction_dataset_audit.py --dataset "data/datasets/construction_progress"

Outputs:
    data/dataset_audit/audit_report.json
    data/dataset_audit/annotation_samples.png
"""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()
DEFAULT_DATASET_ROOT = ROOT / "data" / "datasets"
OUTPUT_DIR = ROOT / "data" / "dataset_audit"

SPLIT_NAMES = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "validation": "valid",
    "test": "test",
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp"
}


def locate_annotations(dataset_root):
    found = {}

    for path in dataset_root.rglob("*.json"):
        if path.name not in {
            "_annotations.coco.json",
            "annotations.coco.json",
            "instances_default.json",
        }:
            continue

        split_name = path.parent.name.lower()
        normalized = SPLIT_NAMES.get(split_name)

        if normalized is None:
            continue

        if normalized in found:
            raise RuntimeError(
                f"Multiple annotation files found for {normalized}:\n"
                f"  {found[normalized]}\n"
                f"  {path}\n"
                "Pass --dataset with the exact extracted dataset directory."
            )

        found[normalized] = path

    return found


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_image(image_root, file_name):
    candidate = image_root / file_name

    if candidate.is_file():
        return candidate

    candidate = image_root / Path(file_name).name

    if candidate.is_file():
        return candidate

    return None


def inspect_segmentation(segmentation):
    if isinstance(segmentation, list):
        if not segmentation:
            return False, "Empty polygon list"

        for polygon in segmentation:
            if not isinstance(polygon, list):
                return False, "Invalid polygon type"

            if len(polygon) < 6:
                return False, "Polygon has fewer than 3 points"

            if len(polygon) % 2 != 0:
                return False, "Polygon has an odd coordinate count"

            try:
                coordinates = np.asarray(
                    polygon, dtype=float
                )
            except (TypeError, ValueError):
                return False, "Non-numeric polygon coordinates"

            if not np.all(np.isfinite(coordinates)):
                return False, "Non-finite polygon coordinates"

        return True, None

    if isinstance(segmentation, dict):
        if "counts" in segmentation and "size" in segmentation:
            return True, None

        return False, "Invalid RLE structure"

    return False, "Missing or unsupported segmentation"


def inspect_split(split, annotation_path):
    print(f"\nInspecting {split}...")

    data = json.loads(
        annotation_path.read_text(encoding="utf-8")
    )

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    category_map = {
        category["id"]: category["name"]
        for category in categories
    }

    image_map = {
        image["id"]: image
        for image in images
    }

    annotations_by_image = defaultdict(list)

    issues = []
    class_counts = Counter()
    missing_images = []
    invalid_segmentations = 0
    invalid_boxes = 0

    for annotation in annotations:
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")

        if image_id not in image_map:
            issues.append(
                f"Annotation references unknown image: {image_id}"
            )
            continue

        if category_id not in category_map:
            issues.append(
                f"Unknown category ID: {category_id}"
            )
            continue

        class_counts[category_map[category_id]] += 1

        bbox = annotation.get("bbox")

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(
                isinstance(value, (int, float))
                and np.isfinite(value)
                for value in bbox
            )
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            invalid_boxes += 1

        valid, reason = inspect_segmentation(
            annotation.get("segmentation")
        )

        if not valid:
            invalid_segmentations += 1

            if len(issues) < 100:
                issues.append(
                    f"Annotation {annotation.get('id')}: {reason}"
                )

        annotations_by_image[image_id].append(annotation)

    resolved_images = {}
    image_hashes = {}

    for image in images:
        path = resolve_image(
            annotation_path.parent,
            image["file_name"],
        )

        if path is None:
            missing_images.append(image["file_name"])
            continue

        resolved_images[image["id"]] = path

        try:
            with Image.open(path) as picture:
                picture.verify()

            image_hashes[image["id"]] = sha256_file(path)

        except Exception as exc:
            issues.append(
                f"Unreadable image {path.name}: {exc}"
            )

    empty_images = sum(
        1 for image in images
        if not annotations_by_image[image["id"]]
    )

    print(f"  Images: {len(images)}")
    print(f"  Annotations: {len(annotations)}")
    print(f"  Missing images: {len(missing_images)}")
    print(f"  Empty images: {empty_images}")
    print(f"  Invalid segmentations: {invalid_segmentations}")
    print(f"  Invalid boxes: {invalid_boxes}")

    print("  Classes:")

    for name, count in class_counts.most_common():
        print(f"    {name}: {count}")

    report = {
        "split": split,
        "annotation_file": str(annotation_path),
        "images": len(images),
        "annotations": len(annotations),
        "categories": category_map,
        "class_counts": dict(class_counts),
        "missing_images": missing_images[:100],
        "missing_image_count": len(missing_images),
        "empty_images": empty_images,
        "invalid_segmentations": invalid_segmentations,
        "invalid_boxes": invalid_boxes,
        "issues": issues[:100],
    }

    return {
        "report": report,
        "images": images,
        "annotations_by_image": annotations_by_image,
        "resolved_images": resolved_images,
        "image_hashes": image_hashes,
        "category_map": category_map,
    }


def draw_annotations(image, annotations, category_map):
    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    colors = [
        (230, 80, 60, 100),
        (40, 170, 110, 100),
        (55, 125, 220, 100),
        (225, 165, 35, 100),
        (145, 90, 200, 100),
    ]

    labels = []

    for annotation in annotations:
        category_id = annotation.get("category_id")
        name = category_map.get(category_id, "unknown")

        color = colors[
            list(category_map).index(category_id)
            % len(colors)
        ]

        segmentation = annotation.get("segmentation")

        if isinstance(segmentation, list):
            for polygon in segmentation:
                if not isinstance(polygon, list):
                    continue

                if len(polygon) < 6 or len(polygon) % 2:
                    continue

                points = [
                    (polygon[i], polygon[i + 1])
                    for i in range(0, len(polygon), 2)
                ]

                draw.polygon(
                    points,
                    fill=color,
                    outline=color[:3] + (255,),
                )

        labels.append(name)

    combined = Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    )

    return combined.convert("RGB"), sorted(set(labels))


def make_contact_sheet(results):
    random_generator = random.Random(42)

    samples = []

    for split, result in results.items():
        image_ids = list(result["resolved_images"])

        random_generator.shuffle(image_ids)

        for image_id in image_ids[:4]:
            samples.append((
                split,
                image_id,
                result,
            ))

    if not samples:
        return False

    columns = 3
    rows = (len(samples) + columns - 1) // columns

    cell_width = 340
    cell_height = 280

    sheet = Image.new(
        "RGB",
        (
            columns * cell_width,
            rows * cell_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    for index, (split, image_id, result) in enumerate(samples):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height

        path = result["resolved_images"][image_id]

        try:
            with Image.open(path) as picture:
                picture = ImageOps.exif_transpose(
                    picture
                ).convert("RGB")

            annotations = result[
                "annotations_by_image"
            ].get(image_id, [])

            picture, labels = draw_annotations(
                picture,
                annotations,
                result["category_map"],
            )

            picture.thumbnail((320, 220))

            sheet.paste(picture, (x + 10, y + 10))

            draw.text(
                (x + 10, y + 234),
                f"{split} | {path.name[:35]}",
                fill="black",
            )

            draw.text(
                (x + 10, y + 251),
                ", ".join(labels)[:45] or "No annotations",
                fill="#555555",
            )

        except Exception as exc:
            draw.text(
                (x + 10, y + 10),
                str(exc)[:45],
                fill="red",
            )

    sheet.save(OUTPUT_DIR / "annotation_samples.png")

    return True


def find_cross_split_duplicates(results):
    hashes = defaultdict(list)

    for split, result in results.items():
        for image_id, digest in result["image_hashes"].items():
            hashes[digest].append({
                "split": split,
                "image_id": image_id,
            })

    duplicates = []

    for digest, entries in hashes.items():
        splits = {entry["split"] for entry in entries}

        if len(splits) > 1:
            duplicates.append({
                "sha256": digest,
                "occurrences": entries,
            })

    return duplicates


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing the extracted dataset.",
    )

    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset directory not found: {args.dataset}"
        )

    annotation_files = locate_annotations(args.dataset)

    if not annotation_files:
        raise SystemExit(
            "No COCO annotation files found.\n"
            "Expected train/valid/test folders containing "
            "_annotations.coco.json.\n"
            "Check the extraction location or use --dataset."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("BUILDPROGRESS - CONSTRUCTION DATASET AUDIT")
    print("=" * 60)

    results = {}

    for split, annotation_path in sorted(
        annotation_files.items()
    ):
        results[split] = inspect_split(
            split,
            annotation_path,
        )

    duplicates = find_cross_split_duplicates(results)

    contact_created = make_contact_sheet(results)

    report = {
        "dataset_root": str(args.dataset.resolve()),
        "splits": {
            split: result["report"]
            for split, result in results.items()
        },
        "cross_split_exact_duplicates": duplicates[:100],
        "cross_split_exact_duplicate_count": len(duplicates),
        "contact_sheet_created": contact_created,
    }

    report_path = OUTPUT_DIR / "audit_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(f"Splits found: {', '.join(results)}")
    print(f"Cross-split exact duplicates: {len(duplicates)}")

    print(f"\nSaved report: {report_path}")

    if contact_created:
        print(
            "Saved contact sheet: "
            f"{OUTPUT_DIR / 'annotation_samples.png'}"
        )

    print("\nAudit complete. No model trained.")


if __name__ == "__main__":
    main()

