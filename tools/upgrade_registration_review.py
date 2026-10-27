
from __future__ import annotations

import ast
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TARGET = ROOT / "src" / "registration_review.py"

BACKUP = (
    ROOT / "src"
    / "registration_review.before_comparison.py"
)

START = "// REGISTRATION_COMPARISON_V1_START"
END = "// REGISTRATION_COMPARISON_V1_END"


COMPARISON_JS = r"""
// REGISTRATION_COMPARISON_V1_START

const comparisonColors = [
  "#ffb45a",
  "#a78bfa",
  "#fb7185",
  "#34d399",
  "#facc15"
];

let showAllCandidates = true;
let mapZoom = 1;
let mapPanX = 0;
let mapPanY = 0;
let mapDragging = false;
let dragStart = null;
let suppressMapClick = false;

function candidatePosition(f, c) {
  return transform([f.x, f.y], c);
}

function frameDisagreement(f) {
  const positions = data.candidates.map(
    c => candidatePosition(f, c)
  );

  let maximum = 0;

  for (let i = 0; i < positions.length; i++) {
    for (let j = i + 1; j < positions.length; j++) {
      maximum = Math.max(
        maximum,
        Math.hypot(
          positions[i][0] - positions[j][0],
          positions[i][1] - positions[j][1]
        )
      );
    }
  }

  return maximum;
}

function svgCoordinates(event) {
  const svg = $("map");
  const point = svg.createSVGPoint();

  point.x = event.clientX;
  point.y = event.clientY;

  const matrix = svg.getScreenCTM();

  if (!matrix) {
    return null;
  }

  const result = point.matrixTransform(
    matrix.inverse()
  );

  return [result.x, result.y];
}

function mapToCad(event) {
  const coordinates = svgCoordinates(event);

  if (!coordinates) {
    return null;
  }

  const x = (
    coordinates[0] - 350 - mapPanX
  ) / mapZoom + 350;

  const y = (
    coordinates[1] - 350 - mapPanY
  ) / mapZoom + 350;

  return unproject(x, y);
}

function installComparisonControls() {
  const map = $("map");

  if (!map || $("registrationComparisonControls")) {
    return;
  }

  const controls = document.createElement("div");
  controls.id = "registrationComparisonControls";
  controls.className = "controls";

  const toggle = document.createElement("label");
  const checkbox = document.createElement("input");

  checkbox.type = "checkbox";
  checkbox.checked = true;

  checkbox.addEventListener("change", () => {
    showAllCandidates = checkbox.checked;
    drawMap();
  });

  toggle.append(
    checkbox,
    document.createTextNode(
      " Show all candidate positions"
    )
  );

  const zoomIn = document.createElement("button");
  zoomIn.textContent = "Zoom +";

  zoomIn.onclick = () => {
    mapZoom = Math.min(8, mapZoom * 1.35);
    drawMap();
  };

  const zoomOut = document.createElement("button");
  zoomOut.textContent = "Zoom −";

  zoomOut.onclick = () => {
    mapZoom = Math.max(1, mapZoom / 1.35);

    if (mapZoom === 1) {
      mapPanX = 0;
      mapPanY = 0;
    }

    drawMap();
  };

  const reset = document.createElement("button");
  reset.textContent = "Reset map";

  reset.onclick = () => {
    mapZoom = 1;
    mapPanX = 0;
    mapPanY = 0;
    drawMap();
  };

  controls.append(
    toggle,
    zoomIn,
    zoomOut,
    reset
  );

  map.parentElement.insertBefore(
    controls,
    map
  );

  const info = document.createElement("div");
  info.id = "registrationDisagreement";
  info.className = "muted";
  info.style.margin = "10px 0";

  map.insertAdjacentElement(
    "afterend",
    info
  );

  const ranking = document.createElement("div");
  ranking.id = "registrationFrameRanking";
  ranking.style.marginTop = "14px";

  info.insertAdjacentElement(
    "afterend",
    ranking
  );

  const instructions = document.createElement("small");
  instructions.textContent =
    "Drag to pan. Use Zoom +/− to inspect the map. " +
    "Click without dragging to propose a CAD camera position.";

  controls.insertAdjacentElement(
    "afterend",
    instructions
  );

  map.style.touchAction = "none";

  map.addEventListener("pointerdown", event => {
    if (event.button !== 0) return;

    mapDragging = true;
    dragStart = {
      x: event.clientX,
      y: event.clientY,
      panX: mapPanX,
      panY: mapPanY,
      moved: false
    };

    map.setPointerCapture(event.pointerId);
  });

  map.addEventListener("pointermove", event => {
    if (!mapDragging || !dragStart) return;

    const rect = map.getBoundingClientRect();

    const dx = (
      event.clientX - dragStart.x
    ) * 700 / rect.width;

    const dy = (
      event.clientY - dragStart.y
    ) * 700 / rect.height;

    if (Math.hypot(dx, dy) > 3) {
      dragStart.moved = true;
    }

    if (!dragStart.moved) return;

    mapPanX = dragStart.panX + dx;
    mapPanY = dragStart.panY + dy;

    drawMap();
  });

  map.addEventListener("pointerup", event => {
    if (!mapDragging) return;

    suppressMapClick = Boolean(
      dragStart && dragStart.moved
    );

    mapDragging = false;
    dragStart = null;

    if (map.hasPointerCapture(event.pointerId)) {
      map.releasePointerCapture(event.pointerId);
    }
  });

  map.addEventListener("pointercancel", () => {
    mapDragging = false;
    dragStart = null;
  });
}

function drawComparisonOverlay() {
  if (!data) return;

  const svg = $("map");
  const f = frame();

  const overlay = svgElement("g", {
    transform:
      `translate(${350 + mapPanX},${350 + mapPanY}) ` +
      `scale(${mapZoom}) translate(-350,-350)`
  });

  const candidates = showAllCandidates
    ? data.candidates
    : [candidate()];

  candidates.forEach((c, index) => {
    const position = project(
      ...candidatePosition(f, c)
    );

    const selected = (
      c.rank === candidate().rank
    );

    const color = comparisonColors[
      (c.rank - 1) % comparisonColors.length
    ];

    overlay.appendChild(
      svgElement("circle", {
        cx: position[0],
        cy: position[1],
        r: selected ? 8 : 6,
        fill: color,
        stroke: selected ? "#ffffff" : "#0c1118",
        "stroke-width": selected ? 2 : 1.5,
        opacity: selected ? 1 : 0.9
      })
    );

    const label = svgElement("text", {
      x: position[0] + 10,
      y: position[1] - 10,
      fill: color,
      "font-size": 14,
      "font-weight": "bold",
      "paint-order": "stroke",
      stroke: "#0c1118",
      "stroke-width": 3
    });

    label.textContent = `#${c.rank}`;
    overlay.appendChild(label);
  });

  svg.appendChild(overlay);

  const disagreement = frameDisagreement(f);

  const info = $("registrationDisagreement");

  if (info) {
    info.textContent =
      `Candidate position disagreement: ` +
      `${disagreement.toFixed(2)} m maximum pairwise distance. ` +
      `This is uncertainty between hypotheses, ` +
      `not measured registration error.`;
  }
}

function renderFrameRanking() {
  if (!data) return;

  const container = $("registrationFrameRanking");

  if (!container) return;

  container.replaceChildren();

  const heading = document.createElement("h3");
  heading.textContent =
    "Frames with greatest candidate disagreement";

  container.appendChild(heading);

  const ranked = data.frames.map(
    (f, position) => ({
      position,
      frameIndex: f.frame_index,
      disagreement: frameDisagreement(f)
    })
  );

  ranked.sort(
    (a, b) => b.disagreement - a.disagreement
  );

  for (const item of ranked.slice(0, 8)) {
    const button = document.createElement("button");

    button.style.margin = "3px";
    button.textContent =
      `Frame ${item.frameIndex} · ` +
      `${item.disagreement.toFixed(1)} m`;

    button.onclick = () => {
      framePosition = item.position;
      updateFrame();
    };

    container.appendChild(button);
  }
}

function activateComparisonUpgrade() {
  if (!data) return;

  installComparisonControls();

  // Preserve the existing drawing logic while
  // adding comparison markers afterward.
  if (!drawMap._comparisonWrapped) {
    const originalDrawMap = drawMap;

    const upgradedDrawMap = function() {
      originalDrawMap();
      drawComparisonOverlay();

      const svg = $("map");
      const nodes = Array.from(svg.children);

      for (const node of nodes) {
        if (node.tagName.toLowerCase() === "g") {
          continue;
        }

        if (node.getAttribute("data-base-group")) {
          continue;
        }

        const group = svgElement("g", {
          transform:
            `translate(${350 + mapPanX},${350 + mapPanY}) ` +
            `scale(${mapZoom}) translate(-350,-350)`
        });

        group.setAttribute(
          "data-base-group",
          "true"
        );

        svg.insertBefore(group, node);
        group.appendChild(node);
      }
    };

    upgradedDrawMap._comparisonWrapped = true;
    drawMap = upgradedDrawMap;
  }

  // Replace only the old click listener's behavior
  // by capturing the event before it reaches the
  // original handler.
  $("map").addEventListener(
    "click",
    event => {
      event.stopImmediatePropagation();

      if (suppressMapClick) {
        suppressMapClick = false;
        return;
      }

      const point = mapToCad(event);

      if (!point) return;

      proposed = point;

      $("selection").textContent =
        `Proposed CAD camera position: ` +
        `(${point[0].toFixed(3)}, ` +
        `${point[1].toFixed(3)}) metres`;

      $("save").disabled = false;
      drawMap();
    },
    true
  );

  renderFrameRanking();
  drawMap();
}

// Wait until the original page has loaded its data.
// This wrapper does not modify the Flask API.
const originalInitialize = initialize;

initialize = async function() {
  await originalInitialize();

  if (data) {
    activateComparisonUpgrade();
  }
};

// REGISTRATION_COMPARISON_V1_END
"""


def main():
    if not TARGET.is_file():
        raise FileNotFoundError(TARGET)

    original = TARGET.read_text(encoding="utf-8")

    if START in original:
        print("Comparison upgrade already installed.")
        return

    if "PAGE_HTML = r" not in original:
        raise RuntimeError(
            "Expected registration review HTML not found."
        )

    if "initialize();" not in original:
        raise RuntimeError(
            "Expected page initialization not found."
        )

    # Insert before the existing initialize call.
    # The injected script wraps initialize() before
    # the original call executes.
    updated = original.replace(
        "initialize();",
        COMPARISON_JS + "\ninitialize();",
        1,
    )

    try:
        ast.parse(updated)
    except SyntaxError as exc:
        raise RuntimeError(
            f"Python syntax validation failed: {exc}"
        ) from exc

    shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(
        updated,
        encoding="utf-8"
    )

    print("\n=== REGISTRATION REVIEW UPGRADE ===")
    print("Updated:", TARGET.relative_to(ROOT))
    print("Backup:", BACKUP.relative_to(ROOT))
    print("All five candidate positions: enabled")
    print("Zoom and pan: enabled")
    print("Candidate disagreement: enabled")
    print("Frame ranking: enabled")


if __name__ == "__main__":
    main()
