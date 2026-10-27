
"""
Under-construction — Hilti Three-Storey Digital Twin

Three real DXF plans:
    data/floorplans/floor_1.dxf
    data/floorplans/floor_2.dxf
    data/floorplans/floor_3.dxf

Adds optional, CAD-derived illustrative floor slabs.
Preserves the existing Floor 2 wall reconstruction.

Slab outlines, thicknesses and storey elevations are
visualization assumptions, not verified structural geometry.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import math

from flask import Blueprint, jsonify

sample_building_bp = Blueprint("sample_three_storey", __name__)

ROOT = Path(__file__).resolve().parents[1]
FLOOR_DIR = ROOT / "data" / "floorplans"

STOREY_HEIGHT_M = 3.0
WALL_HEIGHT_M = 3.0
SLAB_THICKNESS_M = 0.16

CATEGORY_LAYERS = {
    "aussenw": "exterior",
    "innenw": "interior",
    "trennw": "partition",
    "vormauer": "lining",
}

THICKNESS_M = {
    "exterior": 0.20,
    "interior": 0.15,
    "partition": 0.12,
    "lining": 0.10,
}


def segments(entity):
    kind = entity.dxftype()

    if kind == "LINE":
        return [(entity.dxf.start, entity.dxf.end)]

    if kind == "LWPOLYLINE":
        vertices = list(entity.get_points("xyb"))
        if any(abs(float(v[2])) > 1e-9 for v in vertices):
            return []

        points = [(float(v[0]), float(v[1])) for v in vertices]

        if entity.closed and len(points) > 2:
            points.append(points[0])

        return list(zip(points, points[1:]))

    if kind == "POLYLINE" and entity.is_2d_polyline:
        vertices = list(entity.vertices)

        if any(
            abs(float(v.dxf.get("bulge", 0))) > 1e-9
            for v in vertices
        ):
            return []

        points = [
            (float(v.dxf.location.x), float(v.dxf.location.y))
            for v in vertices
        ]

        if entity.is_closed and len(points) > 2:
            points.append(points[0])

        return list(zip(points, points[1:]))

    return []


def extract_wall_elements(floor_number):
    import ezdxf

    path = FLOOR_DIR / f"floor_{floor_number}.dxf"

    if not path.is_file():
        raise FileNotFoundError(str(path))

    drawing = ezdxf.readfile(str(path))

    if drawing.header.get("$INSUNITS", 0) != 6:
        raise ValueError(
            f"Floor {floor_number} is not declared in metres"
        )

    elements = []
    skipped = 0

    for entity in drawing.modelspace():
        layer = str(entity.dxf.get("layer", "")).casefold()

        category = next(
            (
                value
                for key, value in CATEGORY_LAYERS.items()
                if key in layer
            ),
            None,
        )

        if category is None:
            continue

        handle = str(entity.dxf.get("handle", "unknown"))
        entity_segments = segments(entity)

        if not entity_segments:
            skipped += 1
            continue

        for index, (a, b) in enumerate(entity_segments):
            coords = [
                float(a[0]), float(a[1]),
                float(b[0]), float(b[1]),
            ]

            if not all(math.isfinite(v) for v in coords):
                skipped += 1
                continue

            x1, y1, x2, y2 = coords
            length = math.hypot(x2 - x1, y2 - y1)

            if not 0.05 <= length <= 150:
                skipped += 1
                continue

            elements.append({
                "id": f"F{floor_number}_CAD_{handle}_{index}",
                "source_element_ids": [
                    f"F{floor_number}_CAD_{handle}"
                ],
                "start": [x1, y1],
                "end": [x2, y2],
                "category": category,
                "height_m": WALL_HEIGHT_M,
                "thickness_m": THICKNESS_M[category],
                "source_layer": entity.dxf.layer,
                "source_handle": handle,
                "construction_status": "UNKNOWN",
                "progress_verified": False,
            })

    return elements, skipped


def approximate_slab_outline(surfaces):
    """
    Derive an illustrative envelope from exterior CAD walls.

    A convex hull is deliberately used as a stable approximation.
    It may bridge concavities, courtyards and openings; therefore
    it must not be represented as a structural slab footprint.
    """
    from shapely.geometry import MultiPoint

    points = []

    for surface in surfaces:
        if surface.get("category") != "exterior":
            continue

        footprint = surface.get("footprint") or {}
        ring = footprint.get("exterior") or []

        for point in ring:
            if (
                isinstance(point, (list, tuple))
                and len(point) >= 2
            ):
                try:
                    x, y = float(point[0]), float(point[1])
                except (TypeError, ValueError):
                    continue

                if math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))

    if len(points) < 3:
        return None

    hull = MultiPoint(points).convex_hull

    if hull.geom_type != "Polygon" or hull.area <= 0:
        return None

    return [
        [round(float(x), 5), round(float(y), 5)]
        for x, y in hull.exterior.coords
    ]


@lru_cache(maxsize=1)
def building_geometry():
    from architectural_geometry import (
        reconstruct,
        get_reconstruction,
    )

    floors = []

    for number in (1, 2, 3):
        if number == 2:
            result = get_reconstruction()
            method = "EXISTING_WORKING_FLOOR_2_RECONSTRUCTION"
            skipped = None
        else:
            elements, skipped = extract_wall_elements(number)
            result = reconstruct(elements)
            method = "CAD_CENTERLINE_RECONSTRUCTION"

        surfaces = result.get("surfaces", [])

        if not surfaces:
            raise ValueError(
                f"Floor {number} has no reconstructed surfaces"
            )

        slab_outline = approximate_slab_outline(surfaces)

        floors.append({
            "floor_number": number,
            "floor_name": f"Floor {number}",
            "elevation_m": (number - 1) * STOREY_HEIGHT_M,
            "elevation_verified": False,
            "height_m": WALL_HEIGHT_M,
            "geometry_source": (
                f"data/floorplans/floor_{number}.dxf"
            ),
            "reconstruction_method": method,
            "surface_count": len(surfaces),
            "original_element_count": result.get(
                "original_element_count"
            ),
            "skipped_source_entities": skipped,
            "surfaces": surfaces,
            "illustrative_slab": {
                "outline": slab_outline,
                "thickness_m": SLAB_THICKNESS_M,
                "method": "CONVEX_HULL_OF_EXTERIOR_WALL_FOOTPRINTS",
                "verified": False,
                "structural_geometry": False,
                "openings_modelled": False,
            },
        })

    return {
        "name": "Hilti Construction Site",
        "sample_id": "sample_01",
        "floor_count": len(floors),
        "surface_count": sum(
            floor["surface_count"] for floor in floors
        ),
        "floors": floors,
        "coordinate_system": "DXF_METRES",
        "horizontal_alignment": "SHARED_DXF_COORDINATES",
        "horizontal_alignment_verified": False,
        "storey_height_m": STOREY_HEIGHT_M,
        "storey_height_verified": False,
        "construction_progress_verified": False,
        "camera_registration_verified": False,
        "representation": "CAD_DERIVED_3D_WALL_GEOMETRY",
        "notes": [
            "Each floor originates from a distinct real DXF.",
            "Floor 2 preserves the existing wall reconstruction.",
            "Slabs are optional illustrative envelopes.",
            "Slab boundaries and voids are not structurally verified.",
            "Storey elevations and slab thicknesses are assumed.",
            "No construction completion is inferred.",
        ],
    }


@sample_building_bp.get("/api/sample/building")
def sample_building_api():
    try:
        return jsonify(building_geometry())
    except Exception as exc:
        return jsonify({
            "error": "Three-storey reconstruction failed",
            "detail": str(exc),
        }), 503


STYLE = """
<style>
.uc-building-panel {
  margin-top:14px;
  background:#191a22;
  border:1px solid #343542;
  border-radius:11px;
  overflow:hidden;
}
.uc-building-head {
  padding:16px;
  display:flex;
  justify-content:space-between;
  align-items:center;
  gap:12px;
  flex-wrap:wrap;
  border-bottom:1px solid #343542;
}
.uc-building-head strong {font-size:13px}
.uc-building-controls {
  display:flex;
  flex-wrap:wrap;
  align-items:center;
  gap:7px;
}
.uc-building-controls button {
  padding:9px 11px;
  background:#292a35;
  color:#f0eef7;
  border:1px solid #514e61;
  border-radius:7px;
  font-size:11px;
  cursor:pointer;
}
.uc-building-controls button.active {
  background:#b89aff;
  color:#17131f;
}
.uc-slab-toggle {
  display:flex;
  align-items:center;
  gap:7px;
  color:#c5bdd5;
  font-size:11px;
  padding:6px 4px;
  cursor:pointer;
}
.uc-slab-toggle input {accent-color:#b89aff}
#uc-building-canvas {
  width:100%;
  height:620px;
  display:block;
  touch-action:none;
}
.uc-building-foot {
  padding:13px 16px;
  color:#aaa5b9;
  font-size:11px;
  line-height:1.7;
  border-top:1px solid #343542;
}
#uc-building-message {
  color:#b89aff;
  font:11px Consolas,monospace;
}
@media(max-width:700px) {
  #uc-building-canvas {height:420px}
}
</style>
"""

PANEL = """
<div class="uc-building-panel hidden" id="uc-building-panel">
  <div class="uc-building-head">
    <div>
      <strong>01 / HILTI · THREE-STOREY DIGITAL TWIN</strong>
      <div id="uc-building-message">
        THREE REAL CAD FLOOR PLANS
      </div>
    </div>
    <div class="uc-building-controls">
      <button data-floor="all" class="active">Whole Building</button>
      <button data-floor="1">Floor 1</button>
      <button data-floor="2">Floor 2</button>
      <button data-floor="3">Floor 3</button>
      <button data-floor="exploded">Exploded View</button>
      <label class="uc-slab-toggle">
        <input type="checkbox" id="uc-show-slabs" checked>
        Show slabs
      </label>
    </div>
  </div>
  <canvas id="uc-building-canvas"></canvas>
  <div class="uc-building-foot">
    Three real Hilti CAD floor plans.
    Floor slabs are semi-transparent illustrative envelopes
    derived from exterior wall geometry; slab boundaries,
    voids and thicknesses are not verified.
    Vertical spacing is assumed at 3 m per storey.
    Construction progress remains unverified.
  </div>
</div>
"""

SCRIPT = """
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from "three";
import {OrbitControls} from
  "three/addons/controls/OrbitControls.js";

const $ = id => document.getElementById(id);
const panel = $("uc-building-panel");
const canvas = $("uc-building-canvas");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14151d);

const camera = new THREE.PerspectiveCamera(
  50,1,0.1,100000
);
camera.position.set(55,55,55);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias:true
});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));

const controls = new OrbitControls(
  camera,renderer.domElement
);
controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(
  0xffffff,0x526078,2.1
));

const sunlight = new THREE.DirectionalLight(
  0xffffff,2
);
sunlight.position.set(30,70,25);
scene.add(sunlight);

const model = new THREE.Group();
scene.add(model);

const colors = {
  exterior:0xb89aff,
  interior:0x8593b2,
  partition:0xf0784b,
  lining:0x79b7a9
};

let floorGroups = [];
let loaded = false;
let loading = false;
let activeMode = "all";

function makeShape(footprint) {
  const ring = footprint?.exterior;

  if (!Array.isArray(ring) || ring.length < 4) {
    return null;
  }

  const shape = new THREE.Shape();
  shape.moveTo(ring[0][0],ring[0][1]);

  for (let i=1;i<ring.length;i++) {
    shape.lineTo(ring[i][0],ring[i][1]);
  }

  for (const hole of footprint.holes || []) {
    if (!Array.isArray(hole) || hole.length < 4) continue;

    const path = new THREE.Path();
    path.moveTo(hole[0][0],hole[0][1]);

    for (let i=1;i<hole.length;i++) {
      path.lineTo(hole[i][0],hole[i][1]);
    }

    shape.holes.push(path);
  }

  return shape;
}

function makeSlab(data) {
  const ring = data?.outline;

  if (!Array.isArray(ring) || ring.length < 4) {
    return null;
  }

  const shape = makeShape({
    exterior:ring,
    holes:[]
  });

  if (!shape) return null;

  const thickness = Number(data.thickness_m) || 0.16;

  const geometry = new THREE.ExtrudeGeometry(
    shape,{
      depth:thickness,
      bevelEnabled:false
    }
  );

  geometry.rotateX(-Math.PI/2);

  const material = new THREE.MeshStandardMaterial({
    color:0x85849a,
    transparent:true,
    opacity:0.25,
    depthWrite:false,
    roughness:0.92,
    side:THREE.DoubleSide
  });

  const mesh = new THREE.Mesh(geometry,material);

  // Geometry extends downward from local Y=0.
  // Put the top face at the floor elevation.
  mesh.position.y = 0;

  mesh.renderOrder = 2;
  return mesh;
}

function updateSlabs() {
  const visible = $("uc-show-slabs").checked;

  for (const floor of floorGroups) {
    if (floor.slab) {
      floor.slab.visible = visible;
    }
  }
}

function frameModel() {
  const box = new THREE.Box3().setFromObject(model);

  if (box.isEmpty()) return;

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());

  const radius = Math.max(size.x,size.y,size.z,10);

  camera.position.set(
    radius*1.05,
    center.y+radius*.85,
    radius*1.05
  );

  controls.target.copy(center);
  controls.update();
}

function build(project) {
  floorGroups = [];

  while (model.children.length) {
    const child = model.children[0];

    child.traverse(object => {
      object.geometry?.dispose();
      if (Array.isArray(object.material)) {
        object.material.forEach(m => m.dispose());
      } else {
        object.material?.dispose();
      }
    });

    model.remove(child);
  }

  for (const floor of project.floors) {
    const group = new THREE.Group();

    for (const surface of floor.surfaces) {
      const shape = makeShape(surface.footprint);
      if (!shape) continue;

      const height = Number(surface.height_m);
      if (!(height > 0)) continue;

      const geometry = new THREE.ExtrudeGeometry(
        shape,{
          depth:height,
          bevelEnabled:false,
          curveSegments:1
        }
      );

      geometry.rotateX(-Math.PI/2);

      const material = new THREE.MeshStandardMaterial({
        color:colors[surface.category] || 0xaaaaaa,
        roughness:0.82,
        side:THREE.DoubleSide
      });

      group.add(new THREE.Mesh(geometry,material));
    }

    const slab = makeSlab(floor.illustrative_slab);

    if (slab) {
      group.add(slab);
    }

    const elevation = Number(floor.elevation_m);
    group.position.y = elevation;
    model.add(group);

    floorGroups.push({
      number:floor.floor_number,
      elevation,
      group,
      slab
    });
  }

  const box = new THREE.Box3().setFromObject(model);
  const center = box.getCenter(new THREE.Vector3());

  // Centre the model horizontally, preserving storey heights.
  model.position.x = -center.x;
  model.position.z = -center.z;

  updateSlabs();
  frameModel();

  const slabCount = floorGroups.filter(f=>f.slab).length;

  $("uc-building-message").textContent =
    `${project.floor_count} FLOORS / `+
    `${project.surface_count} CAD WALL SURFACES / `+
    `${slabCount} ILLUSTRATIVE SLABS`;
}

function selectFloor(mode) {
  activeMode = mode;

  for (const floor of floorGroups) {
    floor.group.visible = (
      mode==="all" ||
      mode==="exploded" ||
      String(floor.number)===mode
    );

    floor.group.position.y = floor.elevation;

    if (mode==="exploded") {
      floor.group.position.y +=
        (floor.number-1)*2.5;
    }
  }

  document.querySelectorAll(
    "#uc-building-panel [data-floor]"
  ).forEach(button=>{
    button.classList.toggle(
      "active",
      button.dataset.floor===mode
    );
  });

  // Reframe only the currently visible geometry.
  frameModel();
}

document.querySelectorAll(
  "#uc-building-panel [data-floor]"
).forEach(button=>{
  button.addEventListener("click",()=>{
    selectFloor(button.dataset.floor);
  });
});

$("uc-show-slabs").addEventListener(
  "change",updateSlabs
);

function resize() {
  if (panel.classList.contains("hidden")) return;

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  if (!width || !height) return;

  renderer.setSize(width,height,false);
  camera.aspect = width/height;
  camera.updateProjectionMatrix();
}

async function openBuilding() {
  panel.classList.remove("hidden");

  if (!loaded && !loading) {
    loading = true;

    $("uc-building-message").textContent =
      "LOADING THREE-STOREY CAD MODEL...";

    try {
      const response = await fetch(
        "/api/sample/building"
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.error || "API failed"
        );
      }

      build(data);
      loaded = true;
      selectFloor("all");

    } catch(error) {
      console.error(error);
      $("uc-building-message").textContent =
        "ERROR / "+error.message;
    } finally {
      loading = false;
    }
  }

  requestAnimationFrame(resize);

  panel.scrollIntoView({
    behavior:"smooth",
    block:"start"
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  resize();
  renderer.render(scene,camera);
}
animate();

window.addEventListener("resize",resize);

const header = document.querySelector(
  "#uc-3d-panel .uc-3d-head"
);

if (header) {
  const button = document.createElement("button");

  button.type = "button";
  button.textContent = "Explore 3 floors →";
  button.id = "uc-open-three-floors";

  button.addEventListener("click",openBuilding);

  header.append(button);
}
</script>
"""


def enhance_sample_html(html: str) -> str:
    """Add the sample viewer to the existing dashboard."""

    if 'id="uc-building-panel"' in html:
        return html

    anchor = "<!-- UC_STEP124_3D_END -->"

    if anchor not in html:
        raise ValueError(
            "Existing Sample 01 3D anchor was not found"
        )

    html = html.replace(
        anchor,
        anchor + "\n" + PANEL,
        1,
    )

    html = html.replace(
        "</head>",
        STYLE + "\n</head>",
        1,
    )

    html = html.replace(
        "</body>",
        SCRIPT + "\n</body>",
        1,
    )

    return html
