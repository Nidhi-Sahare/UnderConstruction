
import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS = (
    ROOT
    / "runs"
    / "segment"
    / "data"
    / "model_training"
    / "runs"
    / "baseline_cpu_20epochs"
    / "weights"
    / "best.pt"
)

DEFAULT_OUTPUT = ROOT / "models" / "underconstruction_seg_v0.onnx"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Export Under-construction YOLO segmentation to ONNX."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    weights = args.weights.resolve()
    output = args.out.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    if output.suffix.lower() != ".onnx":
        raise ValueError("Output filename must end in .onnx")

    if args.imgsz < 32 or args.imgsz % 32 != 0:
        raise ValueError("Image size must be >= 32 and divisible by 32")

    output.parent.mkdir(parents=True, exist_ok=True)
    card_path = output.parent / "model_card.json"

    if output.exists() or card_path.exists():
        raise FileExistsError(
            "Output ONNX or model card already exists. "
            "Refusing to overwrite. Move or rename existing outputs first."
        )

    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    print("=" * 65)
    print("UNDER-CONSTRUCTION | ONNX EXPORT")
    print("=" * 65)
    print("Python:", sys.version.split()[0])
    print("OpenCV:", cv2.__version__)
    print("NumPy:", np.__version__)
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("Weights:", weights)
    print("Image size:", args.imgsz)
    print("ONNX opset:", args.opset)

    model = YOLO(str(weights))

    if model.task != "segment":
        raise ValueError(
            f"Expected segmentation model, found task={model.task!r}"
        )

    names = model.names
    if isinstance(names, dict):
        class_names = [
            str(names[index])
            for index in sorted(names, key=lambda value: int(value))
        ]
    else:
        class_names = [str(name) for name in names]

    print("\nModel task:", model.task)
    print("Classes:", class_names)

    expected = ["concrete", "formwork", "rebar", "tiles"]
    if class_names != expected:
        raise ValueError(
            "Model classes differ from inspected dataset.\n"
            f"Expected: {expected}\n"
            f"Actual:   {class_names}\n"
            "Verify the checkpoint before exporting."
        )

    print("\nStarting ONNX export...")

    exported = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            dynamic=False,
            simplify=False,
            nms=False,
            half=False,
            device="cpu",
        )
    ).resolve()

    if not exported.is_file():
        raise RuntimeError(
            f"Exporter did not create the expected file: {exported}"
        )

    if exported != output:
        shutil.copy2(exported, output)

    size_bytes = output.stat().st_size
    if size_bytes < 100_000:
        raise RuntimeError(
            f"Exported ONNX looks unexpectedly small: {size_bytes} bytes"
        )

    card = {
        "model_id": "underconstruction-seg-v0",
        "task": "segment",
        "source_weights": str(weights),
        "source_weights_sha256": sha256_file(weights),
        "onnx_filename": output.name,
        "onnx_sha256": sha256_file(output),
        "onnx_size_bytes": size_bytes,
        "imgsz": args.imgsz,
        "num_classes": len(class_names),
        "class_names": class_names,
        "export": {
            "format": "onnx",
            "opset": args.opset,
            "dynamic": False,
            "simplify": False,
            "nms": False,
            "half": False,
            "device": "cpu",
        },
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "evaluation": {
            "independent_accuracy_verified": False,
            "note": (
                "Original dataset has confirmed cross-split base-frame "
                "overlap. Historical validation metrics are not evidence "
                "of independent-site generalization."
            ),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    card_path.write_text(
        json.dumps(card, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EXPORT COMPLETED")
    print("=" * 65)
    print("ONNX:", output)
    print("Size: %.2f MB" % (size_bytes / 1_000_000))
    print("SHA-256:", card["onnx_sha256"])
    print("Model card:", card_path)
    print("Classes:", ", ".join(class_names))
    print("\nNEXT GATE: Test loading and inference using OpenCV 5 DNN.")


if __name__ == "__main__":
    main()
