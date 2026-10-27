
"""
Under-construction — Digital Twin Component Inspector

Adds component selection and an evidence inspector to the
existing Hilti three-storey viewer without modifying the
CAD reconstruction or the original Three.js module.

Photographs are walkthrough references, NOT verified
component-specific evidence.
"""

from __future__ import annotations

from pathlib import Path
from flask import Blueprint, jsonify, send_file

ROOT = Path(__file__).resolve().parents[1]
FRAMES_DIR = ROOT / "sample_gallery" / "real_floor_2" / "frames"

digital_twin_inspector_bp = Blueprint(
    "digital_twin_inspector",
    __name__,
)


def available_frames():
    if not FRAMES_DIR.is_dir():
        return []

    return sorted(
        path
        for path in FRAMES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


@digital_twin_inspector_bp.get(
    "/api/digital-twin/floor-2-frames"
)
def floor_2_frames():
    frames = available_frames()

    return jsonify({
        "floor": 2,
        "count": len(frames),
        "registration_verified": False,
        "component_correspondence_verified": False,
        "frames": [
            {
                "filename": path.name,
                "url": (
                    "/api/digital-twin/frame/"
                    + path.name
                ),
            }
            for path in frames
        ],
    })


@digital_twin_inspector_bp.get(
    "/api/digital-twin/frame/<filename>"
)
def floor_2_frame(filename):
    # Only serve files present in the approved directory.
    frames = {
        path.name: path
        for path in available_frames()
    }

    path = frames.get(filename)

    if path is None:
        return jsonify({
            "error": "Frame not found"
        }), 404

    return send_file(path, conditional=True)


INSPECTOR_STYLE = """
<style id="uc-twin-inspector-style">
#uc-building-panel .uc-twin-workspace {
  display:grid;
  grid-template-columns:minmax(0,1fr) 350px;
  min-height:620px;
}
#uc-building-panel .uc-twin-model {
  min-width:0;
  position:relative;
  overflow:hidden;
}
#uc-building-panel #uc-building-canvas {
  width:100%;
  height:620px;
  display:block;
  cursor:crosshair;
}
#uc-twin-hint {
  position:absolute;
  left:18px;
  bottom:18px;
  padding:9px 12px;
  border:1px solid #464154;
  border-radius:7px;
  background:rgba(20,19,28,.88);
  color:#c9b7ef;
  font-size:11px;
  pointer-events:none;
}
#uc-twin-inspector {
  min-width:0;
  border-left:1px solid #373440;
  background:#17161f;
  color:#f3effa;
  display:flex;
  flex-direction:column;
}
.uc-ti-header {
  padding:20px;
  border-bottom:1px solid #35313f;
}
.uc-ti-kicker {
  color:#b89aff;
  font:10px Consolas,monospace;
  letter-spacing:1.3px;
  margin-bottom:10px;
}
.uc-ti-header h3 {
  font-size:20px;
  line-height:1.3;
  margin:0 0 7px;
}
.uc-ti-subtitle {
  font-size:12px;
  line-height:1.6;
  color:#aaa4b7;
}
.uc-ti-body {
  padding:19px;
  overflow:auto;
  flex:1;
}
.uc-ti-status {
  display:inline-block;
  padding:6px 9px;
  border:1px solid #79573c;
  border-radius:6px;
  background:#35261d;
  color:#f2b87a;
  font:10px Consolas,monospace;
  letter-spacing:.5px;
  margin-bottom:16px;
}
.uc-ti-section-title {
  font-size:11px;
  color:#b89aff;
  letter-spacing:1px;
  text-transform:uppercase;
  margin:18px 0 11px;
}
.uc-ti-detail {
  display:flex;
  justify-content:space-between;
  gap:14px;
  padding:10px 0;
  border-bottom:1px solid #302d38;
  font-size:12px;
}
.uc-ti-detail span:first-child {
  color:#9992a6;
  flex-shrink:0;
}
.uc-ti-detail span:last-child {
  color:#f0eaf8;
  text-align:right;
  overflow-wrap:anywhere;
}
.uc-ti-photo {
  width:100%;
  max-height:230px;
  object-fit:contain;
  background:#0b0a10;
  border:1px solid #363140;
  border-radius:8px;
  display:block;
}
.uc-ti-photo-controls {
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-top:10px;
}
.uc-ti-photo-controls button {
  background:#292534;
  color:#eae0f7;
  border:1px solid #524765;
  border-radius:6px;
  padding:8px 11px;
  cursor:pointer;
}
.uc-ti-photo-controls button:disabled {
  opacity:.4;
  cursor:default;
}
#uc-ti-frame-label {
  color:#aaa3b7;
  font:10px Consolas,monospace;
}
.uc-ti-notice {
  margin-top:13px;
  padding:12px;
  border-left:2px solid #c58b56;
  background:#28221f;
  color:#c8bcb5;
  font-size:11px;
  line-height:1.7;
}
.uc-ti-empty {
  color:#a9a1b7;
  font-size:12px;
  line-height:1.8;
}
.uc-ti-empty strong {
  color:#f0eaf8;
}
.uc-ti-clear {
  background:transparent;
  color:#bda5f5;
  border:1px solid #514466;
  padding:8px 11px;
  border-radius:6px;
  cursor:pointer;
  font-size:11px;
  margin-top:15px;
}
@media(max-width:1050px) {
  #uc-building-panel .uc-twin-workspace {
    grid-template-columns:1fr;
  }
  #uc-twin-inspector {
    border-left:0;
    border-top:1px solid #373440;
  }
  #uc-building-panel #uc-building-canvas {
    height:480px;
  }
}
</style>
"""


INSPECTOR_HTML = """
<aside id="uc-twin-inspector">
  <div class="uc-ti-header">
    <div class="uc-ti-kicker">
      COMPONENT INSPECTOR / HILTI
    </div>
    <h3 id="uc-ti-title">Select a component</h3>
    <div class="uc-ti-subtitle" id="uc-ti-subtitle">
      Click a wall in the digital twin to inspect
      its CAD geometry and available site photographs.
    </div>
  </div>
  <div class="uc-ti-body" id="uc-ti-content">
    <div class="uc-ti-empty">
      <strong>No component selected.</strong>
      <p>
        Select an exterior wall, interior wall,
        partition, or lining in the 3D model.
      </p>
      <p>
        The selected component will be highlighted
        and its details will appear here.
      </p>
    </div>
  </div>
</aside>
"""


# This script runs inside the existing Three.js module.
# It is inserted after build() and before selectFloor().
# It therefore has direct access to scene, camera, model,
# floorGroups, renderer, canvas and THREE.
INSPECTOR_MODULE_JS = r"""
const inspectorRaycaster = new THREE.Raycaster();
const inspectorPointer = new THREE.Vector2();

let inspectorSelectedMesh = null;
let inspectorOriginalMaterial = null;
let inspectorFrames = [];
let inspectorFrameIndex = 0;
let inspectorSelection = null;
let inspectorPointerDown = null;

function inspectorEscape(value) {
  return String(value ?? "—")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function inspectorDetail(label, value) {
  return (
    '<div class="uc-ti-detail">' +
    '<span>' + inspectorEscape(label) + '</span>' +
    '<span>' + inspectorEscape(value) + '</span>' +
    '</div>'
  );
}

function inspectorClearHighlight() {
  if (inspectorSelectedMesh && inspectorOriginalMaterial) {
    inspectorSelectedMesh.material.dispose();
    inspectorSelectedMesh.material = inspectorOriginalMaterial;
  }

  inspectorSelectedMesh = null;
  inspectorOriginalMaterial = null;
}

function inspectorReset() {
  inspectorClearHighlight();
  inspectorSelection = null;

  document.getElementById("uc-ti-title").textContent =
    "Select a component";

  document.getElementById("uc-ti-subtitle").textContent =
    "Click a wall in the digital twin to inspect its details.";

  document.getElementById("uc-ti-content").innerHTML =
    '<div class="uc-ti-empty">' +
    '<strong>No component selected.</strong>' +
    '<p>Click a wall surface in the 3D model.</p>' +
    '</div>';
}

function inspectorPhotoSection(floorNumber) {
  if (floorNumber !== 2) {
    return (
      '<div class="uc-ti-section-title">Visual evidence</div>' +
      '<div class="uc-ti-empty">' +
      'No floor-specific walkthrough is connected ' +
      'for Floor ' + floorNumber + '.' +
      '</div>'
    );
  }

  if (!inspectorFrames.length) {
    return (
      '<div class="uc-ti-section-title">Visual evidence</div>' +
      '<div class="uc-ti-empty">' +
      'No Floor 2 walkthrough frames were found.' +
      '</div>'
    );
  }

  return (
    '<div class="uc-ti-section-title">' +
    'Floor 2 walkthrough references' +
    '</div>' +
    '<img class="uc-ti-photo" id="uc-ti-photo" ' +
    'alt="Real Floor 2 construction walkthrough frame">' +
    '<div class="uc-ti-photo-controls">' +
    '<button type="button" id="uc-ti-prev">Previous</button>' +
    '<span id="uc-ti-frame-label"></span>' +
    '<button type="button" id="uc-ti-next">Next</button>' +
    '</div>' +
    '<div class="uc-ti-notice">' +
    '<strong>Component correspondence unverified.</strong> ' +
    'This photograph belongs to the Floor 2 walkthrough, ' +
    'but the selected CAD wall has not been reliably ' +
    'matched to the image. It must not be interpreted ' +
    'as proof of construction completion.' +
    '</div>'
  );
}

function inspectorUpdatePhoto() {
  const image = document.getElementById("uc-ti-photo");
  const label = document.getElementById("uc-ti-frame-label");

  if (!image || !label || !inspectorFrames.length) return;

  const frame = inspectorFrames[inspectorFrameIndex];

  image.src = frame.url;

  label.textContent =
    (inspectorFrameIndex + 1) +
    " / " +
    inspectorFrames.length;
}

function inspectorSelect(mesh) {
  const data = mesh.userData.inspector;

  if (!data) return;

  inspectorClearHighlight();

  inspectorSelectedMesh = mesh;
  inspectorOriginalMaterial = mesh.material;

  mesh.material = mesh.material.clone();
  mesh.material.color.setHex(0xffd36f);
  mesh.material.emissive.setHex(0x614310);
  mesh.material.emissiveIntensity = 0.45;

  inspectorSelection = data;
  inspectorFrameIndex = 0;

  const surface = data.surface;
  const floor = data.floor;

  const category = surface.category || "wall";
  const label =
    category.charAt(0).toUpperCase() +
    category.slice(1) +
    " wall";

  document.getElementById("uc-ti-title").textContent = label;

  document.getElementById("uc-ti-subtitle").textContent =
    "Floor " + floor.floor_number +
    " · CAD-derived wall surface";

  const sourceIds = Array.isArray(surface.source_element_ids)
    ? surface.source_element_ids
    : [];

  const footprint = surface.footprint || {};
  const exterior = footprint.exterior || [];

  let area = null;

  if (exterior.length >= 3) {
    let signedArea = 0;

    for (let i = 0; i < exterior.length; i++) {
      const a = exterior[i];
      const b = exterior[(i + 1) % exterior.length];

      signedArea += a[0] * b[1] - b[0] * a[1];
    }

    area = Math.abs(signedArea) / 2;
  }

  const content =
    '<span class="uc-ti-status">CONSTRUCTION UNVERIFIED</span>' +
    '<div class="uc-ti-section-title">Component details</div>' +
    inspectorDetail("Floor", floor.floor_number) +
    inspectorDetail("Category", category) +
    inspectorDetail(
      "Surface ID",
      surface.id || "Not supplied"
    ) +
    inspectorDetail(
      "Source CAD elements",
      sourceIds.length
        ? sourceIds.join(", ")
        : "Not supplied"
    ) +
    inspectorDetail(
      "Height",
      surface.height_m != null
        ? Number(surface.height_m).toFixed(2) + " m (model)"
        : "Not supplied"
    ) +
    inspectorDetail(
      "Footprint area",
      area != null
        ? area.toFixed(2) + " m² (model)"
        : "Not available"
    ) +
    inspectorDetail(
      "CAD source",
      floor.geometry_source || "Unknown"
    ) +
    inspectorDetail(
      "Reconstruction",
      floor.reconstruction_method || "Unknown"
    ) +
    inspectorDetail(
      "Progress verified",
      "No"
    ) +
    inspectorPhotoSection(floor.floor_number) +
    '<button class="uc-ti-clear" id="uc-ti-clear">' +
    'Clear selection' +
    '</button>';

  document.getElementById("uc-ti-content").innerHTML = content;

  document.getElementById("uc-ti-clear")
    .addEventListener("click", inspectorReset);

  const previous = document.getElementById("uc-ti-prev");
  const next = document.getElementById("uc-ti-next");

  if (previous && next) {
    previous.addEventListener("click", () => {
      inspectorFrameIndex =
        (inspectorFrameIndex - 1 + inspectorFrames.length) %
        inspectorFrames.length;

      inspectorUpdatePhoto();
    });

    next.addEventListener("click", () => {
      inspectorFrameIndex =
        (inspectorFrameIndex + 1) %
        inspectorFrames.length;

      inspectorUpdatePhoto();
    });

    inspectorUpdatePhoto();
  }
}

function inspectorPick(event) {
  const rect = canvas.getBoundingClientRect();

  inspectorPointer.x =
    ((event.clientX - rect.left) / rect.width) * 2 - 1;

  inspectorPointer.y =
    -((event.clientY - rect.top) / rect.height) * 2 + 1;

  inspectorRaycaster.setFromCamera(inspectorPointer, camera);

  const meshes = [];

  for (const floor of floorGroups) {
    if (!floor.group.visible) continue;

    floor.group.traverse(object => {
      if (object.isMesh && object.userData.inspector) {
        meshes.push(object);
      }
    });
  }

  const intersections = inspectorRaycaster.intersectObjects(
    meshes,
    false
  );

  if (intersections.length) {
    inspectorSelect(intersections[0].object);
  }
}

canvas.addEventListener("pointerdown", event => {
  inspectorPointerDown = {
    x: event.clientX,
    y: event.clientY
  };
});

canvas.addEventListener("pointerup", event => {
  if (!inspectorPointerDown) return;

  const distance = Math.hypot(
    event.clientX - inspectorPointerDown.x,
    event.clientY - inspectorPointerDown.y
  );

  inspectorPointerDown = null;

  // Don't select walls while orbiting the model.
  if (distance > 5) return;

  inspectorPick(event);
});

fetch("/api/digital-twin/floor-2-frames")
  .then(response => {
    if (!response.ok) throw new Error("Frames unavailable");
    return response.json();
  })
  .then(data => {
    inspectorFrames = Array.isArray(data.frames)
      ? data.frames
      : [];

    if (inspectorSelection) {
      inspectorSelect(inspectorSelectedMesh);
    }
  })
  .catch(error => {
    console.warn("Floor 2 frame listing unavailable:", error);
  });
"""


def enhance_digital_twin_inspector(html: str) -> str:
    """
    Enhance the existing sample_three_storey HTML.

    Must run AFTER enhance_sample_html().
    """

    if 'id="uc-twin-inspector"' in html:
        return html

    canvas_markup = '<canvas id="uc-building-canvas"></canvas>'

    if canvas_markup not in html:
        raise ValueError(
            "Existing three-storey canvas not found"
        )

    # Keep the original canvas and put the inspector beside it.
    replacement = (
        '<div class="uc-twin-workspace">'
        '<div class="uc-twin-model">'
        + canvas_markup +
        '<div id="uc-twin-hint">'
        'CLICK A WALL TO INSPECT'
        '</div>'
        '</div>'
        + INSPECTOR_HTML +
        '</div>'
    )

    html = html.replace(
        canvas_markup,
        replacement,
        1,
    )

    # Attach source metadata when original meshes are created.
    original_mesh_line = (
        "group.add(new THREE.Mesh(geometry,material));"
    )

    replacement_mesh_line = """
      const wallMesh = new THREE.Mesh(geometry,material);

      wallMesh.userData.inspector = {
        surface: surface,
        floor: floor
      };

      group.add(wallMesh);
"""

    if original_mesh_line not in html:
        raise ValueError(
            "Existing wall mesh construction not found"
        )

    html = html.replace(
        original_mesh_line,
        replacement_mesh_line,
        1,
    )

    # Insert the selection logic inside the existing ES module,
    # where THREE, camera, canvas and floorGroups are accessible.
    marker = "function selectFloor(mode) {"

    if marker not in html:
        raise ValueError(
            "Existing floor selection function not found"
        )

    html = html.replace(
        marker,
        INSPECTOR_MODULE_JS + "\n" + marker,
        1,
    )

    # Clear a selection when the selected floor is hidden.
    floor_visibility_marker = (
        "floor.group.position.y = floor.elevation;"
    )

    # We do not need to modify floor switching; the selected
    # component can remain selected while another floor is viewed.

    if "</head>" not in html:
        raise ValueError("HTML head closing tag not found")

    html = html.replace(
        "</head>",
        INSPECTOR_STYLE + "\n</head>",
        1,
    )

    return html
