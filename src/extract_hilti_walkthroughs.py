
"""
Extract real Hilti Camera 0 walkthrough frames from ROS 2 bags.

Input:
    data/hilti_recordings/floor_1/rosbag.db3
    data/hilti_recordings/floor_3/rosbag.db3

Output:
    sample_gallery/real_floor_1/frames/
    sample_gallery/real_floor_1/manifest.json

    sample_gallery/real_floor_3/frames/
    sample_gallery/real_floor_3/manifest.json

Uses actual compressed ROS camera messages.
No synthetic images, trajectories, or CAD registration.

Run:
    python src/extract_hilti_walkthroughs.py
"""

from __future__ import annotations

import json
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

INPUT_ROOT = ROOT / "data" / "hilti_recordings"
OUTPUT_ROOT = ROOT / "sample_gallery"

FLOORS = (1, 3)

CAMERA_TOPIC = "/cam0/image_raw/compressed"

SAMPLE_INTERVAL_SECONDS = 0.5

JPEG_QUALITY = 88


class CDRReader:
    """
    Minimal reader for the fields used by
    sensor_msgs/msg/CompressedImage.

    ROS 2 CDR layout:
        std_msgs/Header header
        string format
        uint8[] data
    """

    def __init__(self, payload: bytes):
        if len(payload) < 4:
            raise ValueError("CDR payload is too short")

        identifier = payload[:2]

        if identifier in (b"\x00\x01", b"\x00\x03"):
            self.endian = "<"
        elif identifier in (b"\x00\x00", b"\x00\x02"):
            self.endian = ">"
        else:
            raise ValueError(
                f"Unsupported CDR encapsulation: "
                f"{identifier.hex()}"
            )

        self.payload = memoryview(payload)
        self.offset = 4

    def align(self, size: int):
        # CDR alignment is relative to the serialized body,
        # which begins immediately after the encapsulation.
        relative = self.offset - 4
        padding = (-relative) % size
        self.offset += padding

    def read(self, fmt: str, alignment: int):
        self.align(alignment)

        size = struct.calcsize(fmt)

        if self.offset + size > len(self.payload):
            raise ValueError("Truncated CDR field")

        value = struct.unpack_from(
            self.endian + fmt,
            self.payload,
            self.offset,
        )[0]

        self.offset += size
        return value

    def int32(self):
        return self.read("i", 4)

    def uint32(self):
        return self.read("I", 4)

    def string(self):
        length = self.uint32()

        if length > len(self.payload) - self.offset:
            raise ValueError("Invalid CDR string length")

        raw = bytes(
            self.payload[
                self.offset:self.offset + length
            ]
        )

        self.offset += length

        return raw.rstrip(b"\x00").decode(
            "utf-8",
            errors="replace",
        )

    def byte_array(self):
        length = self.uint32()

        if length > len(self.payload) - self.offset:
            raise ValueError("Invalid CDR array length")

        raw = bytes(
            self.payload[
                self.offset:self.offset + length
            ]
        )

        self.offset += length
        return raw


def decode_compressed_message(payload):
    reader = CDRReader(payload)

    seconds = reader.int32()
    nanoseconds = reader.uint32()

    frame_id = reader.string()
    image_format = reader.string()
    image_bytes = reader.byte_array()

    if not image_bytes:
        raise ValueError("Compressed image is empty")

    array = np.frombuffer(
        image_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not decode image: "
            f"{image_format}"
        )

    header_timestamp_ns = (
        seconds * 1_000_000_000 + nanoseconds
    )

    return {
        "image": image,
        "header_timestamp_ns": header_timestamp_ns,
        "frame_id": frame_id,
        "format": image_format,
    }


def extract_floor(floor: int):
    database = (
        INPUT_ROOT
        / f"floor_{floor}"
        / "rosbag.db3"
    )

    if not database.is_file():
        raise FileNotFoundError(
            f"Missing recording: {database}"
        )

    destination = (
        OUTPUT_ROOT
        / f"real_floor_{floor}"
    )

    frames_dir = destination / "frames"
    frames_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        f"file:{database.resolve().as_posix()}?mode=ro",
        uri=True,
    )

    try:
        topic = connection.execute(
            """
            SELECT id, type, serialization_format
            FROM topics
            WHERE name = ?
            """,
            (CAMERA_TOPIC,),
        ).fetchone()

        if topic is None:
            raise RuntimeError(
                f"Camera topic not found: {CAMERA_TOPIC}"
            )

        topic_id, message_type, serialization = topic

        if message_type != (
            "sensor_msgs/msg/CompressedImage"
        ):
            raise RuntimeError(
                f"Unexpected message type: {message_type}"
            )

        if serialization != "cdr":
            raise RuntimeError(
                f"Unsupported serialization: {serialization}"
            )

        total_messages = connection.execute(
            """
            SELECT COUNT(*)
            FROM messages
            WHERE topic_id = ?
            """,
            (topic_id,),
        ).fetchone()[0]

        cursor = connection.execute(
            """
            SELECT timestamp, data
            FROM messages
            WHERE topic_id = ?
            ORDER BY timestamp, id
            """,
            (topic_id,),
        )

        print()
        print("=" * 62)
        print(f"FLOOR {floor} / CAMERA 0")
        print("=" * 62)
        print("Source:", database)
        print("Camera messages:", total_messages)

        first_timestamp_ns = None
        next_sample_time = 0.0

        records = []
        decode_failures = 0
        image_width = None
        image_height = None

        for source_index, row in enumerate(cursor):
            bag_timestamp_ns, payload = row

            if first_timestamp_ns is None:
                first_timestamp_ns = bag_timestamp_ns

            elapsed_seconds = (
                bag_timestamp_ns - first_timestamp_ns
            ) / 1_000_000_000

            if (
                elapsed_seconds + 1e-9
                < next_sample_time
            ):
                continue

            try:
                decoded = decode_compressed_message(
                    payload
                )
            except Exception as exc:
                decode_failures += 1
                print(
                    f"  Decode failed at source "
                    f"message {source_index}: {exc}"
                )
                continue

            image = decoded["image"]

            if image_width is None:
                image_height, image_width = (
                    image.shape[:2]
                )

            frame_number = len(records) + 1

            filename = (
                f"frame_{frame_number:04d}.jpg"
            )

            output_path = frames_dir / filename

            success = cv2.imwrite(
                str(output_path),
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if not success:
                raise IOError(
                    f"Unable to save {output_path}"
                )

            records.append({
                "index": frame_number - 1,
                "filename": filename,
                "relative_path": (
                    f"frames/{filename}"
                ),
                "time_seconds": round(
                    elapsed_seconds, 6
                ),
                "bag_timestamp_ns": (
                    bag_timestamp_ns
                ),
                "header_timestamp_ns": (
                    decoded["header_timestamp_ns"]
                ),
                "camera_frame_id": (
                    decoded["frame_id"]
                ),
                "source_message_index": (
                    source_index
                ),
            })

            next_sample_time = (
                elapsed_seconds
                + SAMPLE_INTERVAL_SECONDS
            )

            if frame_number % 25 == 0:
                print(
                    f"  Saved {frame_number} frames "
                    f"({elapsed_seconds:.1f}s)"
                )

        if not records:
            raise RuntimeError(
                f"No frames extracted for Floor {floor}"
            )

        manifest = {
            "schema_version": "1.0",
            "floor": floor,
            "source": "Hilti real ROS 2 recording",
            "source_database": str(
                database.relative_to(ROOT)
            ).replace("\\\\", "/"),
            "camera_topic": CAMERA_TOPIC,
            "camera": "cam0",
            "message_type": message_type,
            "recording_start_timestamp_ns": (
                first_timestamp_ns
            ),
            "sample_interval_target_seconds": (
                SAMPLE_INTERVAL_SECONDS
            ),
            "sampled_frame_count": len(records),
            "source_camera_message_count": (
                total_messages
            ),
            "image_width": image_width,
            "image_height": image_height,
            "duration_seconds_sampled": (
                records[-1]["time_seconds"]
            ),
            "decode_failures": decode_failures,
            "photographs_real": True,
            "camera_trajectory_connected": False,
            "cad_registration_verified": False,
            "component_correspondence_verified": False,
            "construction_progress_verified": False,
            "generated_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "frames": records,
        }

        manifest_path = (
            destination / "manifest.json"
        )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print(
            f"Extracted: {len(records)} real images"
        )
        print(
            "Sampled duration:",
            round(
                records[-1]["time_seconds"], 2
            ),
            "seconds",
        )
        print(
            "Resolution:",
            f"{image_width} x {image_height}",
        )
        print(
            "Decode failures:",
            decode_failures,
        )
        print(
            "Manifest:",
            manifest_path,
        )

        return manifest

    finally:
        connection.close()


def main():
    print(
        "Hilti real walkthrough extraction"
    )
    print(
        "Sampling interval:",
        SAMPLE_INTERVAL_SECONDS,
        "seconds",
    )

    results = {}

    for floor in FLOORS:
        results[floor] = extract_floor(floor)

    print()
    print("=" * 62)
    print("EXTRACTION SUMMARY")
    print("=" * 62)

    for floor, manifest in results.items():
        print(
            f"Floor {floor}: "
            f"{manifest['sampled_frame_count']} "
            f"real camera frames"
        )

    print()
    print(
        "Original recordings remain unchanged."
    )
    print(
        "Camera trajectories and CAD alignment "
        "have not yet been attached."
    )


if __name__ == "__main__":
    main()
