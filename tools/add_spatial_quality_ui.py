
from __future__ import annotations

import ast
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "spatial_evidence_browser.py"
BACKUP = ROOT / "src" / "spatial_evidence_browser.before_quality.py"

MARKER = "SPATIAL_QUALITY_INTEGRATION_V1"


def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one match, found {count}. "
            "No files were modified."
        )
    return source.replace(old, new, 1)


def main():
    source = TARGET.read_text(encoding="utf-8")

    if MARKER in source:
        print("Quality integration already installed.")
        return

    # Add quality report location and cached reader.
    anchor = 'spatial_evidence_bp = Blueprint('

    addition = '''
# SPATIAL_QUALITY_INTEGRATION_V1
QUALITY_PATH = (
    ROOT / "outputs" / "visual_quality"
    / "walkthrough_quality.json"
)

_quality_cache = {"mtime": None, "frames": {}}


def load_quality():
    if not QUALITY_PATH.is_file():
        return {}

    mtime = QUALITY_PATH.stat().st_mtime_ns

    if _quality_cache["mtime"] != mtime:
        data = json.loads(
            QUALITY_PATH.read_text(encoding="utf-8")
        )
        _quality_cache["frames"] = {
            int(frame["frame_index"]): frame
            for frame in data.get("frames", [])
        }
        _quality_cache["mtime"] = mtime

    return _quality_cache["frames"]


'''

    source = replace_once(
        source, anchor, addition + anchor,
        "Quality reader",
    )

    # Extend the frame-list API.
    source = replace_once(
        source,
        '    results = []\n\n    for frame in frames[:limit]:',
        '    results = []\n'
        '    quality = load_quality()\n\n'
        '    for frame in frames[:limit]:',
        "Frame quality lookup",
    )

    source = replace_once(
        source,
        '            "filename": frame["filename"],',
        '            "filename": frame["filename"],\n'
        '            "quality_class": quality.get(\n'
        '                int(frame["frame_index"]), {}\n'
        '            ).get("quality_class", "NOT_SCREENED"),',
        "Frame list response",
    )

    # Extend the frame-detail API without mutating
    # the original spatial index.
    source = replace_once(
        source,
        '    return jsonify(frame)\n',
        '    result = dict(frame)\n'
        '    result["visual_quality"] = load_quality().get(\n'
        '        frame_index\n'
        '    )\n'
        '    return jsonify(result)\n',
        "Frame detail response",
    )

    # Add a quality filter to the existing controls.
    source = replace_once(
        source,
        '        <input\n            id="search"',
        '''        <label for="qualityFilter">Image quality:</label>
        <select id="qualityFilter">
            <option value="ALL">All quality levels</option>
            <option value="USABLE_CANDIDATE">
                Usable candidate
            </option>
            <option value="LOW_LIGHT">Low light</option>
            <option value="LOW_TEXTURE">Low texture</option>
            <option value="LOW_DETAIL_OR_BLUR">
                Low detail / blur
            </option>
            <option value="OVEREXPOSED">Overexposed</option>
            <option value="NOT_SCREENED">Not screened</option>
        </select>

        <input
            id="search"''',
        "Quality filter control",
    )

    # Apply the quality filter alongside the existing
    # association and frame-number filters.
    source = replace_once(
        source,
        '    const search = $("search").value.trim();',
        '    const search = $("search").value.trim();\n'
        '    const qualityFilter = $("qualityFilter").value;',
        "Quality filter state",
    )

    source = replace_once(
        source,
        '        return matchesFilter && matchesSearch;',
        '''        const matchesQuality =
            qualityFilter === "ALL" ||
            frame.quality_class === qualityFilter;

        return matchesFilter && matchesSearch && matchesQuality;''',
        "Quality filter application",
    )

    source = replace_once(
        source,
        '            <small>${escapeHtml(frame.classification)}</small>',
        '''            <small>${escapeHtml(frame.classification)}</small>
            <small>
                Quality: ${escapeHtml(
                    frame.quality_class || "NOT_SCREENED"
                )}
            </small>''',
        "Frame card quality label",
    )

    # Add quality information to the selected frame.
    source = replace_once(
        source,
        '        const candidates = frame.candidate_results || [];',
        '''        const quality = frame.visual_quality || null;
        const metrics = quality?.metrics || {};

        const qualityHtml = quality
            ? `
                <h3>Visual quality screening</h3>
                <p>
                    <strong>${escapeHtml(
                        quality.quality_class
                    )}</strong>
                </p>
                <table>
                    <tbody>
                        <tr>
                            <th>Mean brightness</th>
                            <td>${escapeHtml(
                                metrics.mean_brightness
                            )}</td>
                        </tr>
                        <tr>
                            <th>Contrast (std)</th>
                            <td>${escapeHtml(
                                metrics.contrast_std
                            )}</td>
                        </tr>
                        <tr>
                            <th>Laplacian variance</th>
                            <td>${escapeHtml(
                                metrics.laplacian_variance
                            )}</td>
                        </tr>
                        <tr>
                            <th>Edge fraction</th>
                            <td>${escapeHtml(
                                metrics.edge_fraction
                            )}</td>
                        </tr>
                    </tbody>
                </table>
                <p class="empty">
                    Heuristic image-quality screening only.
                    This does not verify CAD alignment,
                    element visibility, or construction progress.
                </p>
              `
            : `<p class="empty">
                   No image-quality screening available.
               </p>`;

        const candidates = frame.candidate_results || [];''',
        "Frame detail quality rendering",
    )

    source = replace_once(
        source,
        '            <h3>Registration candidates</h3>',
        '''            ${qualityHtml}

            <h3>Registration candidates</h3>''',
        "Quality detail insertion",
    )

    source = replace_once(
        source,
        '$("filter").addEventListener("change", renderFrames);',
        '$("filter").addEventListener("change", renderFrames);\n'
        '$("qualityFilter").addEventListener("change", renderFrames);',
        "Quality filter listener",
    )

    ast.parse(source, filename=str(TARGET))

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(source, encoding="utf-8")

    print("Quality integration installed successfully.")
    print("Updated:", TARGET.relative_to(ROOT))
    print("Backup:", BACKUP.relative_to(ROOT))
    print("Quality report: outputs/visual_quality/walkthrough_quality.json")


if __name__ == "__main__":
    main()
