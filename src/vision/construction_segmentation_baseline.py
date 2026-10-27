
"""
BuildProgress - Step 12
Construction Element Segmentation Baseline

Converts Roboflow COCO segmentation data to YOLO segmentation
format and optionally fine-tunes a pretrained YOLO model.

Commands:
    python construction_segmentation_baseline.py --prepare
    python construction_segmentation_baseline.py --train --epochs 1
    python construction_segmentation_baseline.py --train --epochs 15

The test split is prepared but never used for training
or checkpoint selection.

Requirements:
    pip install ultralytics pyyaml pillow numpy
"""

import argparse
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image


PROJECT_def find_project_root():
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        if (candidate / "data").is_dir() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("BuildProgress project root not found")

ROOT = find_project_root()

DATASET_SEARCH_ROOT = (
    PROJECT_ROOT / "data" / "datasets"
)

OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "model_training"
)

YOLO_ROOT = OUTPUT_ROOT / "yolo_dataset"

EXPECTED_CLASSES = (
    "concrete",
    "formwork",
    "rebar",
    "tiles",
)

SPLIT_ALIASES = {
    "train": "train",
    "valid": "val",
    "validation": "val",
    "val": "val",
    "test": "test",
}

ANNOTATION_FILENAMES = {
    "_annotations.coco.json",
    "annotations.coco.json",
    "instances_default.json",
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


def discover_dataset(root):
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n{root}"
        )

    found = {}

    for path in root.rglob("*.json"):
        if path.name not in ANNOTATION_FILENAMES:
            continue

        split = SPLIT_ALIASES.get(
            path.parent.name.lower()
        )

        if split is None:
            continue

        if split in found:
            raise RuntimeError(
                f"Multiple annotation files found for {split}:\n"
                f"{found[split]}\n{path}\n"
                "Use --dataset to select one extracted dataset."
            )

        found[split] = path

    if "train" not in found or "val" not in found:
        raise RuntimeError(
            "Expected COCO annotations in train and valid folders."
        )

    return found


def read_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)



def get_category_mapping(annotation_files):
    """
    Map the four expected construction classes to YOLO IDs.

    Extra COCO categories are allowed only if they have
    no annotations. This avoids silently discarding data.
    """

    mappings = {}

    for split, path in annotation_files.items():
        data = read_json(path)

        categories = {
            int(category["id"]): str(category["name"]).strip()
            for category in data.get("categories", [])
        }

        annotated_category_ids = {
            int(annotation["category_id"])
            for annotation in data.get("annotations", [])
        }

        unknown_ids = (
            annotated_category_ids - set(categories)
        )

        if unknown_ids:
            raise RuntimeError(
                f"{split}: annotations reference unknown "
                f"category IDs: {sorted(unknown_ids)}"
            )

        actual_classes = {
            categories[category_id]
            for category_id in annotated_category_ids
        }

        unexpected = (
            actual_classes - set(EXPECTED_CLASSES)
        )

        if unexpected:
            raise RuntimeError(
                f"{split}: unexpected annotated classes: "
                f"{sorted(unexpected)}"
            )

        missing = (
            set(EXPECTED_CLASSES)
            - set(categories.values())
        )

        if missing:
            raise RuntimeError(
                f"{split}: missing expected class definitions: "
                f"{sorted(missing)}"
            )

        unused_categories = {
            name
            for category_id, name in categories.items()
            if category_id not in annotated_category_ids
        }

        if unused_categories:
            print(
                f"{split}: ignoring unannotated COCO "
                f"categories: {sorted(unused_categories)}"
            )

        mappings[split] = {
            category_id: EXPECTED_CLASSES.index(name)
            for category_id, name in categories.items()
            if name in EXPECTED_CLASSES
        }

    return mappings

def resolve_image(folder, filename):
    relative = Path(filename)

    if relative.is_absolute() or ".." in relative.parts:
        return None

    candidate = folder / relative

    if candidate.is_file():
        return candidate

    candidate = folder / relative.name

    if candidate.is_file():
        return candidate

    return None


def normalize_polygon(polygon, width, height):
    if not isinstance(polygon, list):
        return None

    if len(polygon) < 6 or len(polygon) % 2:
        return None

    if width <= 0 or height <= 0:
        return None

    points = []

    for index in range(0, len(polygon), 2):
        try:
            x = float(polygon[index])
            y = float(polygon[index + 1])
        except (ValueError, TypeError):
            return None

        if not math.isfinite(x) or not math.isfinite(y):
            return None

        x = min(1.0, max(0.0, x / width))
        y = min(1.0, max(0.0, y / height))

        points.extend((x, y))

    unique_points = {
        (points[i], points[i + 1])
        for i in range(0, len(points), 2)
    }

    if len(unique_points) < 3:
        return None

    return points


def convert_split(
    split,
    annotation_path,
    category_mapping,
):
    print(f"\nConverting {split}...")

    data = read_json(annotation_path)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    image_lookup = {
        image["id"]: image
        for image in images
    }

    annotations_by_image = {
        image_id: []
        for image_id in image_lookup
    }

    unknown_image_refs = 0

    for annotation in annotations:
        image_id = annotation.get("image_id")

        if image_id not in annotations_by_image:
            unknown_image_refs += 1
            continue

        annotations_by_image[image_id].append(annotation)

    images_dir = YOLO_ROOT / "images" / split
    labels_dir = YOLO_ROOT / "labels" / split

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter()

    missing_images = 0
    invalid_polygons = 0
    unsupported_rle = 0
    converted_objects = 0
    copied_images = 0
    empty_label_files = 0

    for image_id, image_info in image_lookup.items():
        source = resolve_image(
            annotation_path.parent,
            image_info["file_name"],
        )

        if source is None:
            missing_images += 1
            continue

        if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            missing_images += 1
            continue

        with Image.open(source) as picture:
            actual_width, actual_height = picture.size

        width = int(image_info["width"])
        height = int(image_info["height"])

        if (actual_width, actual_height) != (width, height):
            raise RuntimeError(
                f"COCO dimensions differ from image: {source.name}"
            )

        # Deterministic collision-resistant filenames.
        digest = hashlib.sha256(
            (
                str(annotation_path.resolve())
                + "|"
                + str(image_id)
                + "|"
                + image_info["file_name"]
            ).encode("utf-8")
        ).hexdigest()[:16]

        output_name = (
            f"{source.stem[:60]}_{digest}{source.suffix.lower()}"
        )

        destination = images_dir / output_name
        label_path = labels_dir / (
            Path(output_name).stem + ".txt"
        )

        if not destination.exists():
            shutil.copy2(source, destination)

        copied_images += 1

        label_lines = []

        for annotation in annotations_by_image[image_id]:
            category_id = annotation.get("category_id")

            if category_id not in category_mapping:
                invalid_polygons += 1
                continue

            class_id = category_mapping[category_id]

            segmentation = annotation.get("segmentation")

            if isinstance(segmentation, dict):
                unsupported_rle += 1
                continue

            if not isinstance(segmentation, list):
                invalid_polygons += 1
                continue

            # COCO can contain multiple polygon components for
            # one instance. YOLO polygon labels cannot preserve
            # disjoint components as one exact instance.
            # We write each component separately and record
            # the conversion limitation in the report.
            for polygon in segmentation:
                normalized = normalize_polygon(
                    polygon,
                    width,
                    height,
                )

                if normalized is None:
                    invalid_polygons += 1
                    continue

                coordinates = " ".join(
                    f"{value:.6f}"
                    for value in normalized
                )

                label_lines.append(
                    f"{class_id} {coordinates}"
                )

                converted_objects += 1
                counts[EXPECTED_CLASSES[class_id]] += 1

        if not label_lines:
            empty_label_files += 1

        label_path.write_text(
            "\n".join(label_lines)
            + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

    if unknown_image_refs:
        raise RuntimeError(
            f"{unknown_image_refs} annotations reference "
            "unknown image IDs."
        )

    if missing_images:
        raise RuntimeError(
            f"{missing_images} images could not be resolved "
            f"in split {split}."
        )

    print(f"  Images: {copied_images}")
    print(f"  Polygon labels: {converted_objects}")
    print(f"  Empty label files: {empty_label_files}")
    print(f"  Invalid polygons: {invalid_polygons}")
    print(f"  Unsupported RLE masks: {unsupported_rle}")

    return {
        "images": copied_images,
        "polygon_labels": converted_objects,
        "empty_label_files": empty_label_files,
        "invalid_polygons": invalid_polygons,
        "unsupported_rle": unsupported_rle,
        "class_counts": dict(counts),
    }


def prepare_dataset(dataset_root):
    annotation_files = discover_dataset(dataset_root)

    category_mappings = get_category_mapping(
        annotation_files
    )

    YOLO_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = {}

    for split in ("train", "val", "test"):
        if split not in annotation_files:
            continue

        reports[split] = convert_split(
            split,
            annotation_files[split],
            category_mappings[split],
        )

    config = {
        "path": str(YOLO_ROOT.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            index: name
            for index, name in enumerate(EXPECTED_CLASSES)
        },
    }

    if "test" in reports:
        config["test"] = "images/test"

    config_path = YOLO_ROOT / "data.yaml"

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )

    report = {
        "source_dataset": str(dataset_root.resolve()),
        "classes": list(EXPECTED_CLASSES),
        "splits": reports,
        "notes": [
            "COCO polygon components are exported separately.",
            "RLE masks are not converted.",
            "Validation is for checkpoint selection.",
            "Test data is reserved for final evaluation.",
            "The source COCO files remain unchanged.",
        ],
    }

    report_path = OUTPUT_ROOT / "conversion_report.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Report: {report_path}")

    total_invalid = sum(
        item["invalid_polygons"]
        + item["unsupported_rle"]
        for item in reports.values()
    )

    if total_invalid:
        raise RuntimeError(
            f"Conversion has {total_invalid} unsupported "
            "or invalid masks. Review the report before training."
        )

    return config_path


def train_model(epochs, image_size, batch, workers):
    from ultralytics import YOLO

    config_path = YOLO_ROOT / "data.yaml"

    if not config_path.exists():
        raise RuntimeError(
            "Dataset has not been prepared.\n"
            "Run: python construction_segmentation_baseline.py --prepare"
        )

    print("\n" + "=" * 60)
    print("CPU SEGMENTATION BASELINE")
    print("=" * 60)

    print("Model: yolo11n-seg.pt")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch}")
    print("Device: CPU")
    print("Test split: reserved")

    model = YOLO("yolo11n-seg.pt")

    results = model.train(
        data=str(config_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device="cpu",
        workers=workers,
        project=str(OUTPUT_ROOT / "runs"),
        name=f"baseline_cpu_{epochs}epochs",
        exist_ok=False,
        pretrained=True,
        optimizer="auto",
        patience=10,
        seed=42,
        deterministic=True,
        plots=True,
        verbose=True,
    )

    print("\nTraining finished.")
    print(f"Results directory: {results.save_dir}")

    print(
        "\nDo not evaluate the test set yet. "
        "We will first inspect validation metrics."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_SEARCH_ROOT,
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
    )

    parser.add_argument(
        "--train",
        action="store_true",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if not args.prepare and not args.train:
        parser.error(
            "Specify --prepare or --train."
        )

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.batch < 1:
        parser.error("--batch must be at least 1.")

    if args.imgsz < 64:
        parser.error("--imgsz must be at least 64.")

    if args.prepare:
        prepare_dataset(args.dataset)

    if args.train:
        train_model(
            epochs=args.epochs,
            image_size=args.imgsz,
            batch=args.batch,
            workers=args.workers,
        )


if __name__ == "__main__":
    # Required for reliable multiprocessing behavior
    # when running PyTorch on Windows.
    import multiprocessing

    multiprocessing.freeze_support()
    main()

