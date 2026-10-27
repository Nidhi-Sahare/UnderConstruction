
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "models" / "underconstruction_seg_v0.onnx"
DEFAULT_CARD = ROOT / "models" / "model_card.json"

COLORS = (
    (198, 142, 80),
    (87, 174, 230),
    (100, 205, 137),
    (190, 113, 210),
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    score: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    area_fraction: float


class OnnxSegModel:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        card_path: str | Path = DEFAULT_CARD,
    ):
        self.model_path = Path(model_path)
        self.card_path = Path(card_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)

        if not self.card_path.is_file():
            raise FileNotFoundError(self.card_path)

        self.card = json.loads(
            self.card_path.read_text(encoding="utf-8")
        )

        self.class_names = list(self.card["class_names"])
        self.imgsz = int(self.card["imgsz"])
        self.nc = len(self.class_names)
        self.backend = "cv2.dnn"

        self.net = cv2.dnn.readNetFromONNX(
            str(self.model_path)
        )
        self.net.setPreferableBackend(
            cv2.dnn.DNN_BACKEND_OPENCV
        )

        self.output_names = (
            self.net.getUnconnectedOutLayersNames()
        )

    def _letterbox(self, image):
        h, w = image.shape[:2]
        size = self.imgsz

        ratio = min(size / w, size / h)

        new_w = min(size, round(w * ratio))
        new_h = min(size, round(h * ratio))

        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_LINEAR,
        )

        left = (size - new_w) // 2
        top = (size - new_h) // 2
        right = size - new_w - left
        bottom = size - new_h - top

        padded = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return padded, ratio, left, top, new_w, new_h

    @staticmethod
    def _sigmoid(values):
        values = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-values))

    def infer(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.70,
    ) -> list[Detection]:
        if image is None or image.ndim != 3:
            raise ValueError("Expected a BGR image")

        h, w = image.shape[:2]

        padded, ratio, pad_x, pad_y, new_w, new_h = (
            self._letterbox(image)
        )

        blob = cv2.dnn.blobFromImage(
            padded,
            scalefactor=1.0 / 255.0,
            size=(self.imgsz, self.imgsz),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        self.net.setInput(blob)
        outputs = self.net.forward(self.output_names)

        prediction = None
        prototypes = None

        for output in outputs:
            if output.ndim == 4:
                prototypes = output
            elif output.ndim == 3:
                prediction = output

        if prediction is None or prototypes is None:
            raise RuntimeError(
                "Unexpected ONNX segmentation outputs: "
                + str([x.shape for x in outputs])
            )

        proto = prototypes[0]
        mask_channels, proto_h, proto_w = proto.shape

        expected_channels = 4 + self.nc + mask_channels

        if prediction.shape[1] == expected_channels:
            rows = prediction[0].T
        elif prediction.shape[2] == expected_channels:
            rows = prediction[0]
        else:
            raise RuntimeError(
                "Unexpected prediction tensor shape: "
                + str(prediction.shape)
            )

        class_scores = rows[:, 4:4 + self.nc]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[
            np.arange(len(rows)), class_ids
        ]

        selected = np.flatnonzero(scores >= conf)

        if len(selected) == 0:
            return []

        rows = rows[selected]
        class_ids = class_ids[selected]
        scores = scores[selected]

        boxes_xywh = []
        for row, class_id in zip(rows, class_ids):
            cx, cy, bw, bh = row[:4]

            x1 = float(cx - bw / 2)
            y1 = float(cy - bh / 2)

            # Class offsets prevent cross-class suppression.
            offset = int(class_id) * (self.imgsz + 1)

            boxes_xywh.append([
                x1 + offset,
                y1 + offset,
                float(bw),
                float(bh),
            ])

        keep = cv2.dnn.NMSBoxes(
            boxes_xywh,
            scores.astype(float).tolist(),
            conf,
            iou,
        )

        if len(keep) == 0:
            return []

        detections = []
        proto_flat = proto.reshape(mask_channels, -1)

        for index in np.asarray(keep).reshape(-1):
            index = int(index)
            row = rows[index]
            class_id = int(class_ids[index])
            score = float(scores[index])

            cx, cy, bw, bh = row[:4]

            lx1 = float(cx - bw / 2)
            ly1 = float(cy - bh / 2)
            lx2 = float(cx + bw / 2)
            ly2 = float(cy + bh / 2)

            x1 = int(np.clip(
                round((lx1 - pad_x) / ratio), 0, w
            ))
            y1 = int(np.clip(
                round((ly1 - pad_y) / ratio), 0, h
            ))
            x2 = int(np.clip(
                round((lx2 - pad_x) / ratio), 0, w
            ))
            y2 = int(np.clip(
                round((ly2 - pad_y) / ratio), 0, h
            ))

            if x2 <= x1 or y2 <= y1:
                continue

            coefficients = row[
                4 + self.nc:
                4 + self.nc + mask_channels
            ]

            mask_logits = (
                coefficients @ proto_flat
            ).reshape(proto_h, proto_w)

            mask_prob = self._sigmoid(mask_logits)

            # Crop the mask to its detection box in prototype space.
            px1 = int(np.clip(
                np.floor(lx1 * proto_w / self.imgsz),
                0, proto_w
            ))
            py1 = int(np.clip(
                np.floor(ly1 * proto_h / self.imgsz),
                0, proto_h
            ))
            px2 = int(np.clip(
                np.ceil(lx2 * proto_w / self.imgsz),
                0, proto_w
            ))
            py2 = int(np.clip(
                np.ceil(ly2 * proto_h / self.imgsz),
                0, proto_h
            ))

            cropped_proto = np.zeros_like(mask_prob)
            cropped_proto[py1:py2, px1:px2] = (
                mask_prob[py1:py2, px1:px2]
            )

            mask_512 = cv2.resize(
                cropped_proto,
                (self.imgsz, self.imgsz),
                interpolation=cv2.INTER_LINEAR,
            )

            mask_unpadded = mask_512[
                pad_y:pad_y + new_h,
                pad_x:pad_x + new_w,
            ]

            mask_original = cv2.resize(
                mask_unpadded,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

            binary_mask = (
                mask_original > 0.5
            ).astype(np.uint8)

            # Enforce the original-image bounding box.
            binary_mask[:y1, :] = 0
            binary_mask[y2:, :] = 0
            binary_mask[:, :x1] = 0
            binary_mask[:, x2:] = 0

            area_fraction = float(
                np.count_nonzero(binary_mask) / (w * h)
            )

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    score=score,
                    bbox=(x1, y1, x2, y2),
                    mask=binary_mask,
                    area_fraction=area_fraction,
                )
            )

        return detections


def render_overlay(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    overlay = image.copy()

    for detection in detections:
        color = COLORS[detection.class_id % len(COLORS)]
        overlay[detection.mask > 0] = color

    result = cv2.addWeighted(
        overlay, 0.45, image, 0.55, 0
    )

    for detection in detections:
        color = COLORS[detection.class_id % len(COLORS)]

        contours, _ = cv2.findContours(
            detection.mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            result, contours, -1, color, 2
        )

        x1, y1, x2, y2 = detection.bbox

        label = (
            f"{detection.class_name} "
            f"{detection.score:.2f}"
        )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text_y = max(20, y1 - 8)

        cv2.putText(
            result,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test OpenCV 5 construction segmentation."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "visualizations"
        / "construction_segmentation.jpg",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(args.image)

    image = cv2.imread(str(args.image))

    if image is None:
        raise RuntimeError(
            f"Could not read image: {args.image}"
        )

    model = OnnxSegModel()

    start = time.perf_counter()
    detections = model.infer(
        image,
        conf=args.conf,
    )
    elapsed = (time.perf_counter() - start) * 1000

    result = render_overlay(image, detections)

    args.output.parent.mkdir(
        parents=True, exist_ok=True
    )

    if not cv2.imwrite(str(args.output), result):
        raise RuntimeError(
            "Failed to save overlay"
        )

    print("=" * 60)
    print("UNDER-CONSTRUCTION | REAL IMAGE INFERENCE")
    print("=" * 60)
    print("OpenCV:", cv2.__version__)
    print("Backend:", model.backend)
    print("Image:", args.image)
    print("Image size:", image.shape)
    print("Inference + decode: %.2f ms" % elapsed)
    print("Detections:", len(detections))

    for detection in detections:
        print(
            f"  {detection.class_name:<12} "
            f"score={detection.score:.3f} "
            f"coverage={detection.area_fraction:.4f} "
            f"bbox={detection.bbox}"
        )

    print("Overlay:", args.output)


if __name__ == "__main__":
    main()
