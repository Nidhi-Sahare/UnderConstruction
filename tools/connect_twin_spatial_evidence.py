
from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "digital_twin_3d.py"
BACKUP = ROOT / "src" / "digital_twin_3d.before_spatial_bridge.py"

MARKER = "SPATIAL_EVIDENCE_BRIDGE_V1"

NEW_LOADER = r"""
// SPATIAL_EVIDENCE_BRIDGE_V1

function cadEvidenceIds(kind, item) {
    const raw = [];

    function add(value) {
        if (Array.isArray(value)) {
            value.forEach(add);
        } else if (typeof value === "string" ||
                   typeof value === "number") {
            raw.push(String(value));
        }
    }

    if (kind === "wall") {
        add(item.source_element_ids);
        add(item.source_handles);
        add(item.source_handle);
        add(item.id);
    } else if (kind === "door") {
        add(item.source_arc_handles);
        add(item.source_line_handles);
        add(item.source_handles);
    } else if (kind === "window") {
        add(item.source_handles);
    }

    const ids = new Set();

    for (const value of raw) {
        const cleaned = value.trim().toUpperCase();

        // Only exact DXF-style hexadecimal source handles
        // can be joined to the spatial evidence index.
        const handle = cleaned.startsWith("CAD_")
            ? cleaned.slice(4)
            : cleaned;

        if (/^[0-9A-F]+$/.test(handle)) {
            ids.add("CAD_" + handle);
        }
    }

    return [...ids];
}

async function loadEvidence(kind, item) {
    const token = ++requestGeneration;
    const container = $("evidence");

    container.textContent = "Looking up spatial evidence...";

    const ids = cadEvidenceIds(kind, item);

    if (!ids.length) {
        container.textContent =
            "No exact DXF source handle is available for " +
            "this rendered object. Spatial evidence cannot " +
            "be safely linked.";
        return;
    }

    try {
        const responses = await Promise.all(
            ids.map(async id => {
                const url =
                    "/api/spatial-evidence/element/" +
                    encodeURIComponent(id);

                const response = await fetch(url);

                if (response.status === 404) {
                    return null;
                }

                if (!response.ok) {
                    throw new Error(
                        "Spatial evidence API: HTTP " +
                        response.status
                    );
                }

                return {
                    id,
                    data: await response.json()
                };
            })
        );

        if (token !== requestGeneration) return;

        const matches = responses.filter(Boolean);
        const unique = new Map();

        for (const match of matches) {
            const frames = Array.isArray(match.data.frames)
                ? match.data.frames
                : [];

            for (const frame of frames) {
                const index = Number(frame.frame_index);

                if (!Number.isInteger(index) || index < 0) {
                    continue;
                }

                const distance = Number(frame.distance_m);
                const safeDistance = Number.isFinite(distance)
                    ? distance
                    : Infinity;

                const existing = unique.get(index);

                if (!existing) {
                    unique.set(index, {
                        index,
                        distance: safeDistance,
                        sources: new Set([match.id])
                    });
                } else {
                    existing.sources.add(match.id);
                    existing.distance = Math.min(
                        existing.distance,
                        safeDistance
                    );
                }
            }
        }

        const ranked = [...unique.values()]
            .sort((a, b) =>
                a.distance - b.distance ||
                a.index - b.index
            )
            .slice(0, 6);

        container.replaceChildren();

        const note = document.createElement("p");
        note.className = "muted";
        note.textContent =
            `${ids.length} exact CAD source references checked; ` +
            `${matches.length} indexed. ` +
            "These are proximity associations only. " +
            "Image visibility, registration, and construction " +
            "progress are NOT verified.";

        container.appendChild(note);

        if (!ranked.length) {
            const empty = document.createElement("p");
            empty.className = "muted";
            empty.textContent =
                "No indexed candidate frames for this " +
                "CAD source entity.";
            container.appendChild(empty);
            return;
        }

        for (const frame of ranked) {
            const block = document.createElement("div");
            block.className = "evidence";

            const label = document.createElement("div");
            label.className = "muted";

            const distanceText = Number.isFinite(frame.distance)
                ? frame.distance.toFixed(2) + " m"
                : "distance unavailable";

            label.textContent =
                `Frame ${frame.index} · ` +
                `proximity ${distanceText} · ` +
                `CAD references: ${[...frame.sources].join(", ")}`;

            const link = document.createElement("a");
            link.href = "/spatial-evidence";
            link.title = "Open the spatial evidence browser";

            const image = document.createElement("img");
            image.className = "photo";
            image.loading = "lazy";
            image.alt =
                `Unverified walkthrough frame ${frame.index}`;

            image.src =
                "/api/spatial-evidence/image/" +
                frame.index;

            link.appendChild(image);
            block.append(label, link);
            container.appendChild(block);
        }

    } catch (error) {
        if (token === requestGeneration) {
            container.textContent =
                "Unable to retrieve spatial evidence: " +
                error.message;
        }
    }
}

// END_SPATIAL_EVIDENCE_BRIDGE_V1
"""


def replace_once(source, old, new, label):
    count = source.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: expected exactly one match, found {count}. "
            "No changes have been written."
        )

    return source.replace(old, new, 1)


def main():
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    source = TARGET.read_text(encoding="utf-8")

    if MARKER in source:
        print("Spatial evidence bridge already installed.")
        return

    # Update the wall selection branch.
    source = replace_once(
        source,
        "loadEvidence(sourceIds);",
        'loadEvidence("wall", item);',
        "Wall evidence call",
    )

    # Update the door selection branch.
    old_door = (
        '        $("evidence").textContent=\n'
        '            "Door-specific video evidence is not established. "+\n'
        '            "This geometry is derived from the CAD drawing, "+\n'
        '            "not from verified construction footage.";'
    )

    source = replace_once(
        source,
        old_door,
        '        loadEvidence("door", item);',
        "Door evidence call",
    )

    # Update the window selection branch.
    old_window = (
        '        $("evidence").textContent=\n'
        '            "Window-specific visual evidence has not been verified.";'
    )

    source = replace_once(
        source,
        old_window,
        '        loadEvidence("window", item);',
        "Window evidence call",
    )

    # Replace only the existing evidence loader.
    pattern = re.compile(
        r"function frameIndex\(frame\) \{.*?"
        r"(?=const pointerDown=\{x:0,y:0,button:0\};)",
        flags=re.DOTALL,
    )

    source, count = pattern.subn(
        lambda _: NEW_LOADER + "\n\n",
        source,
    )

    if count != 1:
        raise RuntimeError(
            "Could not uniquely locate the existing evidence "
            "loader. No changes have been written."
        )

    # Check that the modified Python source is syntactically valid.
    ast.parse(source, filename=str(TARGET))

    # Keep an untouched copy of the original.
    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(source, encoding="utf-8")

    print("Spatial evidence bridge installed successfully.")
    print("Updated:", TARGET)
    print("Backup:", BACKUP)
    print("Wall, door, and window evidence lookup enabled.")
    print("Construction progress remains UNKNOWN.")


if __name__ == "__main__":
    main()
