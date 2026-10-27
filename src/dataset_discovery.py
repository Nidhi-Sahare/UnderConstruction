
"""
Under-construction — Step 118A
Discover real CAD plans and construction walkthrough data.

Run from BuildProgress:
    .\\.venv\\Scripts\\python.exe src\\dataset_discovery.py

Outputs:
    data/dataset_discovery_report.json

Does not modify the dataset.
"""

from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data" / "dataset_discovery_report.json"

SEARCH_ROOTS = [
    ROOT / "data",
    ROOT / "_archive",
    ROOT / "datasets",
    ROOT / "dataset",
]

CAD_EXTENSIONS = {".dxf", ".dwg", ".ifc"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MANIFEST_NAMES = {
    "frame_manifest.csv",
    "frames.csv",
    "manifest.csv",
    "timestamps.csv",
}

SKIP_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules",
    "__pycache__", ".pytest_cache",
    "sample_gallery", "demo_project",
    "user_projects",
}


def relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def natural_key(value):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\\d+)", str(value))
    ]


def file_size(path):
    try:
        return path.stat().st_size
    except OSError:
        return 0


def walk_files(base):
    """Walk without following symlinks; avoid known generated folders."""
    import os

    for current, dirs, files in os.walk(base, followlinks=False):
        dirs[:] = [
            d for d in dirs
            if d not in SKIP_DIRECTORIES
        ]

        for name in files:
            yield Path(current) / name


def inspect_manifest(path):
    info = {
        "path": relative(path),
        "columns": [],
        "rows": 0,
        "first_row": None,
    }

    try:
        with path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            info["columns"] = reader.fieldnames or []

            for row in reader:
                info["rows"] += 1
                if info["first_row"] is None:
                    info["first_row"] = row

    except (OSError, UnicodeError, csv.Error) as exc:
        info["error"] = str(exc)

    return info


def inspect_cad(path):
    result = {
        "path": relative(path),
        "extension": path.suffix.lower(),
        "size_mb": round(file_size(path) / 1048576, 2),
    }

    if path.suffix.lower() != ".dxf":
        result["inspection"] = "Metadata only"
        return result

    try:
        import ezdxf

        doc = ezdxf.readfile(path)
        modelspace = doc.modelspace()
        counts = Counter(
            entity.dxftype()
            for entity in modelspace
        )

        result["entity_count"] = sum(counts.values())
        result["entity_types"] = dict(counts.most_common(12))
        result["layers"] = [
            layer.dxf.name
            for layer in doc.layers
        ][:30]
        result["drawing_units"] = int(
            doc.header.get("$INSUNITS", 0)
        )

    except ImportError:
        result["inspection"] = "Install ezdxf for DXF details"
    except Exception as exc:
        result["inspection_error"] = str(exc)

    return result


def inspect_video(path):
    result = {
        "path": relative(path),
        "size_mb": round(file_size(path) / 1048576, 2),
    }

    try:
        import cv2

        capture = cv2.VideoCapture(str(path))

        if capture.isOpened():
            fps = capture.get(cv2.CAP_PROP_FPS)
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            result.update({
                "fps": round(fps, 2),
                "frames": frames,
                "duration_seconds": (
                    round(frames / fps, 2)
                    if fps > 0 else None
                ),
                "width": int(
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                ),
                "height": int(
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                ),
            })
        else:
            result["inspection_error"] = "Could not open video"

        capture.release()

    except ImportError:
        result["inspection"] = "Install opencv-python"
    except Exception as exc:
        result["inspection_error"] = str(exc)

    return result


def inspect_frame_folder(folder, images):
    images = sorted(images, key=natural_key)

    return {
        "folder": relative(folder),
        "frame_count": len(images),
        "first_frame": relative(images[0]),
        "last_frame": relative(images[-1]),
        "total_mb": round(
            sum(file_size(path) for path in images) / 1048576,
            2,
        ),
    }


def discover():
    cad_files = []
    video_files = []
    manifest_files = []
    image_folders = {}
    scanned_roots = []

    seen_files = set()

    for base in SEARCH_ROOTS:
        if not base.is_dir():
            continue

        scanned_roots.append(relative(base))

        for path in walk_files(base):
            try:
                identity = str(path.resolve())
            except OSError:
                identity = str(path)

            if identity in seen_files:
                continue

            seen_files.add(identity)
            extension = path.suffix.lower()

            if extension in CAD_EXTENSIONS:
                cad_files.append(path)

            elif extension in VIDEO_EXTENSIONS:
                video_files.append(path)

            elif extension in IMAGE_EXTENSIONS:
                image_folders.setdefault(
                    path.parent, []
                ).append(path)

            elif path.name.lower() in MANIFEST_NAMES:
                manifest_files.append(path)

    frame_folders = [
        inspect_frame_folder(folder, images)
        for folder, images in image_folders.items()
        if len(images) >= 5
    ]

    frame_folders.sort(
        key=lambda item: item["frame_count"],
        reverse=True,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(ROOT),
        "scanned_roots": scanned_roots,
        "summary": {
            "cad_files": len(cad_files),
            "video_files": len(video_files),
            "frame_folders": len(frame_folders),
            "manifest_files": len(manifest_files),
        },
        "cad_files": [
            inspect_cad(path)
            for path in sorted(cad_files, key=natural_key)
        ],
        "video_files": [
            inspect_video(path)
            for path in sorted(video_files, key=natural_key)
        ],
        "frame_folders": frame_folders,
        "manifests": [
            inspect_manifest(path)
            for path in sorted(manifest_files, key=natural_key)
        ],
    }

    return report


def main():
    report = discover()

    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    summary = report["summary"]

    print("\n" + "=" * 55)
    print(" UNDER-CONSTRUCTION — REAL DATASET DISCOVERY")
    print("=" * 55)

    print("\nScanned:")
    for root in report["scanned_roots"]:
        print(" ", root)

    print("\nFOUND")
    print(" CAD drawings:       ", summary["cad_files"])
    print(" Video recordings:   ", summary["video_files"])
    print(" Frame sequences:    ", summary["frame_folders"])
    print(" CSV manifests:      ", summary["manifest_files"])

    print("\nCAD DRAWINGS")
    for item in report["cad_files"][:20]:
        print(" -", item["path"])
        if "entity_count" in item:
            print("   Entities:", item["entity_count"])

    print("\nVIDEO RECORDINGS")
    for item in report["video_files"][:20]:
        print(" -", item["path"])
        if item.get("duration_seconds") is not None:
            print("   Duration:", item["duration_seconds"], "s")

    print("\nFRAME SEQUENCES")
    for item in report["frame_folders"][:25]:
        print(
            " -", item["folder"],
            "|", item["frame_count"], "frames"
        )

    print("\nMANIFESTS")
    for item in report["manifests"][:20]:
        print(
            " -", item["path"],
            "|", item["rows"], "rows"
        )

    print("\nReport saved to:")
    print(" ", REPORT)
    print("\nNo dataset files were modified.")


if __name__ == "__main__":
    main()
