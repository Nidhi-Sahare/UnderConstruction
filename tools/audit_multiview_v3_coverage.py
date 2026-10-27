
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

V3_DIR = ROOT / "outputs" / "multiview_v3_comparison"

TRACK_FILE = V3_DIR / "multiview_track_comparison.csv"
V3_REPORT = V3_DIR / "multiview_v3_report.json"

POSE_FILE = (
    ROOT
    / "outputs"
    / "rotation_new_feature_validation"
    / "experimental_corrected_camera_poses.json"
)

OUTPUT = ROOT / "outputs" / "multiview_v3_coverage"


def frame_number(name):
    return int(Path(name).stem.rsplit("_", 1)[-1])


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(path.read_text(encoding="utf-8"))


def load_tracks(path):
    if not path.exists():
        raise FileNotFoundError(path)

    tracks = []

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "track_id",
            "views",
            "frames",
            "original_valid",
            "refined_valid",
            "original_reason",
            "refined_reason",
        }

        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                "Missing CSV columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            frames = [
                name.strip()
                for name in row["frames"].split("|")
                if name.strip()
            ]

            if len(frames) != len(set(frames)):
                raise RuntimeError(
                    f"Duplicate frame in track {row['track_id']}"
                )

            if len(frames) != int(row["views"]):
                raise RuntimeError(
                    f"View count mismatch in track {row['track_id']}"
                )

            tracks.append({
                "id": int(row["track_id"]),
                "views": int(row["views"]),
                "frames": frames,
                "original_valid": (
                    row["original_valid"].lower() == "true"
                ),
                "refined_valid": (
                    row["refined_valid"].lower() == "true"
                ),
                "original_reason": row["original_reason"],
                "refined_reason": row["refined_reason"],
            })

    return tracks


class UnionFind:
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.size = {node: 1 for node in nodes}

    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(
                self.parent[node]
            )

        return self.parent[node]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self):
        groups = defaultdict(list)

        for node in self.parent:
            groups[self.find(node)].append(node)

        return sorted(
            [
                sorted(
                    members,
                    key=frame_number,
                )
                for members in groups.values()
            ],
            key=lambda members: (
                -len(members),
                frame_number(members[0]),
            ),
        )


def build_components(frames, tracks, mode):
    union = UnionFind(frames)

    for track in tracks:
        if not track[f"{mode}_valid"]:
            continue

        members = track["frames"]

        for name in members[1:]:
            union.union(members[0], name)

    return union.components()


def summarize_components(components, counts):
    output = []

    for index, component in enumerate(components, start=1):
        active = [
            frame
            for frame in component
            if counts[frame] > 0
        ]

        if not active:
            continue

        output.append({
            "component_id": index,
            "frame_count": len(active),
            "first_frame": active[0],
            "last_frame": active[-1],
            "frames": active,
            "total_landmark_observations": int(
                sum(counts[name] for name in active)
            ),
        })

    return output


def save_csv(path, rows, columns):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()
        writer.writerows(rows)


def make_coverage_plot(rows, output_path):
    x = [
        row["frame_number"]
        for row in rows
    ]

    original = [
        row["original_landmarks"]
        for row in rows
    ]

    refined = [
        row["refined_landmarks"]
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        x,
        original,
        linewidth=1.6,
        label="Original rotations",
    )

    ax.plot(
        x,
        refined,
        linewidth=1.6,
        label="Refined rotations",
    )

    ax.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
    )

    ax.set_xlabel("Frame number")
    ax.set_ylabel("Validated landmarks observed")
    ax.set_title(
        "Multiview V3 — Camera Coverage"
    )
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def make_reason_plot(tracks, output_path):
    modes = ["original", "refined"]

    counters = {
        mode: Counter(
            track[f"{mode}_reason"]
            for track in tracks
            if not track[f"{mode}_valid"]
        )
        for mode in modes
    }

    reasons = sorted(
        set(counters["original"])
        | set(counters["refined"])
    )

    if not reasons:
        return

    x = np.arange(len(reasons))
    width = 0.38

    fig, ax = plt.subplots(
        figsize=(max(10, len(reasons) * 1.6), 5)
    )

    ax.bar(
        x - width / 2,
        [counters["original"][r] for r in reasons],
        width,
        label="Original",
    )

    ax.bar(
        x + width / 2,
        [counters["refined"][r] for r in reasons],
        width,
        label="Refined",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        reasons,
        rotation=25,
        ha="right",
    )

    ax.set_ylabel("Rejected candidate tracks")
    ax.set_title("Multiview V3 — Rejection Reasons")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weak-threshold",
        type=int,
        default=5,
        help=(
            "Fewer than this many validated landmarks "
            "means weak frame coverage."
        ),
    )

    args = parser.parse_args()

    if args.weak_threshold < 1:
        parser.error(
            "--weak-threshold must be positive"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    poses = load_json(POSE_FILE)
    v3_report = load_json(V3_REPORT)
    tracks = load_tracks(TRACK_FILE)

    if "poses" not in poses:
        raise RuntimeError(
            "Pose file does not contain poses."
        )

    frames = sorted(
        poses["poses"],
        key=frame_number,
    )

    frame_set = set(frames)

    for track in tracks:
        unknown = set(track["frames"]) - frame_set

        if unknown:
            raise RuntimeError(
                f"Track references unknown frames: {unknown}"
            )

    counts = {
        mode: Counter()
        for mode in ("original", "refined")
    }

    candidate_counts = Counter()

    for track in tracks:
        for frame in track["frames"]:
            candidate_counts[frame] += 1

            for mode in ("original", "refined"):
                if track[f"{mode}_valid"]:
                    counts[mode][frame] += 1

    rows = []

    for name in frames:
        original_count = counts["original"][name]
        refined_count = counts["refined"][name]

        rows.append({
            "frame": name,
            "frame_number": frame_number(name),
            "candidate_tracks": candidate_counts[name],
            "original_landmarks": original_count,
            "refined_landmarks": refined_count,
            "landmark_difference": (
                refined_count - original_count
            ),
            "original_status": (
                "zero"
                if original_count == 0
                else "weak"
                if original_count < args.weak_threshold
                else "covered"
            ),
            "refined_status": (
                "zero"
                if refined_count == 0
                else "weak"
                if refined_count < args.weak_threshold
                else "covered"
            ),
        })

    frame_csv = (
        OUTPUT / "frame_coverage.csv"
    )

    save_csv(
        frame_csv,
        rows,
        list(rows[0]),
    )

    component_reports = {}

    for mode in ("original", "refined"):
        components = build_components(
            frames,
            tracks,
            mode,
        )

        component_reports[mode] = summarize_components(
            components,
            counts[mode],
        )

    reason_counts = {}

    for mode in ("original", "refined"):
        reason_counts[mode] = dict(
            Counter(
                track[f"{mode}_reason"]
                for track in tracks
                if not track[f"{mode}_valid"]
            )
        )

    zero_frames = {
        mode: [
            row["frame"]
            for row in rows
            if row[f"{mode}_landmarks"] == 0
        ]
        for mode in ("original", "refined")
    }

    weak_frames = {
        mode: [
            row["frame"]
            for row in rows
            if (
                0 < row[f"{mode}_landmarks"]
                < args.weak_threshold
            )
        ]
        for mode in ("original", "refined")
    }

    recovered_frames = [
        row["frame"]
        for row in rows
        if (
            row["original_landmarks"] == 0
            and row["refined_landmarks"] > 0
        )
    ]

    lost_frames = [
        row["frame"]
        for row in rows
        if (
            row["original_landmarks"] > 0
            and row["refined_landmarks"] == 0
        )
    ]

    zero_with_candidates = [
        row["frame"]
        for row in rows
        if (
            row["refined_landmarks"] == 0
            and row["candidate_tracks"] > 0
        )
    ]

    zero_without_candidates = [
        row["frame"]
        for row in rows
        if (
            row["refined_landmarks"] == 0
            and row["candidate_tracks"] == 0
        )
    ]

    make_coverage_plot(
        rows,
        OUTPUT / "frame_coverage.png",
    )

    make_reason_plot(
        tracks,
        OUTPUT / "rejection_reasons.png",
    )

    report = {
        "project": "Under-construction",
        "experiment": (
            "MULTIVIEW_V3_COVERAGE_AUDIT"
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "weak_threshold": args.weak_threshold,
        "supported_camera_count": len(frames),
        "candidate_track_count": len(tracks),
        "source_v3_landmark_counts": {
            "original": v3_report["original"][
                "validated_landmarks"
            ],
            "refined": v3_report["refined"][
                "validated_landmarks"
            ],
        },
        "coverage": {
            mode: {
                "frames_with_landmarks": int(
                    sum(
                        counts[mode][name] > 0
                        for name in frames
                    )
                ),
                "zero_landmark_frames": zero_frames[mode],
                "weak_landmark_frames": weak_frames[mode],
                "component_count": len(
                    component_reports[mode]
                ),
                "components": component_reports[mode],
            }
            for mode in ("original", "refined")
        },
        "recovered_by_refinement": recovered_frames,
        "lost_after_refinement": lost_frames,
        "refined_zero_with_candidate_tracks": (
            zero_with_candidates
        ),
        "refined_zero_without_candidate_tracks": (
            zero_without_candidates
        ),
        "rejection_reasons": reason_counts,
        "limitations": [
            "Only cameras in the experimental pose "
            "file are audited.",
            "Candidate tracks were built from "
            "original-pose-selected feature matches.",
            "A zero-candidate frame is not proof "
            "that no reconstructable geometry exists.",
            "This audit does not inspect pair-level "
            "rejections because V3 did not export "
            "a pair-level rejection ledger.",
            "Connected components are based on "
            "shared validated feature tracks, not "
            "physical site connectivity.",
            "Track rejection reasons reflect the "
            "first failed validation check.",
            "No new feature extraction or "
            "triangulation is performed.",
            "Camera extrinsics and CAD registration "
            "remain unverified.",
        ],
    }

    report_path = (
        OUTPUT / "coverage_audit_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nUNDER-CONSTRUCTION")
    print("Multiview V3 Coverage Audit")
    print("-" * 60)

    print("Supported cameras:", len(frames))
    print("Candidate tracks:", len(tracks))

    for mode in ("original", "refined"):
        print(f"\n{mode.upper()}")

        covered = len(frames) - len(zero_frames[mode])

        print(
            "Frames with landmarks:",
            f"{covered}/{len(frames)}",
        )
        print(
            "Zero-landmark frames:",
            len(zero_frames[mode]),
        )
        print(
            "Weak-landmark frames:",
            len(weak_frames[mode]),
        )
        print(
            "Connected components:",
            len(component_reports[mode]),
        )

        print("Rejection reasons:")

        for reason, count in sorted(
            reason_counts[mode].items(),
            key=lambda item: -item[1],
        ):
            print(f"  {reason}: {count}")

    print("\nREFINEMENT COVERAGE")
    print("Newly covered frames:", len(recovered_frames))
    print("Lost frames:", len(lost_frames))

    print(
        "Refined zero frames with candidate tracks:",
        len(zero_with_candidates),
    )

    print(
        "Refined zero frames without candidate tracks:",
        len(zero_without_candidates),
    )

    print("\nReport:", report_path)
    print("Frame CSV:", frame_csv)
    print(
        "Coverage plot:",
        OUTPUT / "frame_coverage.png",
    )
    print(
        "Rejection plot:",
        OUTPUT / "rejection_reasons.png",
    )
    print("\nV2 and V3 reconstructions unchanged.")


if __name__ == "__main__":
    main()
