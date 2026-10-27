
"""
Under-construction — Dashboard Upload Integration

Adds user DXF reconstruction to the existing dashboard.
Does not change Sample 01, its CAD geometry, or its 3D viewer.

Requires uploaded_building_api.py to be registered.
"""

from __future__ import annotations


EXTRA_STYLE = """
<style>
.uc-upload-settings {
  margin-top:20px;
  border:1px solid #343542;
  border-radius:11px;
  background:#191a22;
  padding:20px;
}
.uc-upload-settings h3 {
  margin:0 0 12px;
  font-size:15px;
}
.uc-upload-floor {
  border:1px solid #454353;
  background:#20212b;
  padding:15px;
  margin:12px 0;
  border-radius:9px;
}
.uc-upload-floor strong {
  display:block;
  font-size:12px;
  margin-bottom:12px;
}
.uc-upload-field {
  display:block;
  font-size:11px;
  color:#aaa5b9;
  margin:12px 0 5px;
}
.uc-upload-input {
  background:#282936;
  color:#f0eef7;
  border:1px solid #514e61;
  border-radius:7px;
  padding:9px;
  width:100%;
}
.uc-upload-row {
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:10px;
}
.uc-upload-layer {
  display:flex;
  align-items:center;
  gap:9px;
  padding:7px 0;
  font-size:11px;
}
.uc-upload-layer input {
  accent-color:#b89aff;
}
.uc-upload-layer span {
  flex:1;
  overflow-wrap:anywhere;
}
.uc-upload-layer select {
  background:#282936;
  color:#f0eef7;
  border:1px solid #514e61;
  padding:6px;
  border-radius:5px;
  max-width:120px;
}
.uc-upload-result {
  margin-top:20px;
  border:1px solid #343542;
  border-radius:11px;
  overflow:hidden;
  background:#15161e;
}
.uc-upload-result-head {
  padding:16px;
  border-bottom:1px solid #343542;
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:12px;
  flex-wrap:wrap;
}
.uc-upload-result-head strong {
  font-size:13px;
}
.uc-upload-floor-buttons {
  display:flex;
  gap:7px;
  flex-wrap:wrap;
}
.uc-upload-floor-buttons button {
  background:#282936;
  border:1px solid #514e61;
  color:#f0eef7;
  border-radius:6px;
  padding:8px 11px;
  font-size:11px;
}
#uc-upload-canvas {
  display:block;
  width:100%;
  height:570px;
  touch-action:none;
}
.uc-upload-caption {
  padding:12px 16px;
  font-size:11px;
  line-height:1.7;
  color:#aaa5b9;
}
@media(max-width:750px) {
  .uc-upload-row {grid-template-columns:1fr}
  #uc-upload-canvas {height:400px}
}
</style>
"""

EXTRA_HTML = """
<div class="uc-upload-settings" id="uc-upload-settings">
  <h3>Building configuration</h3>
  <p class="note">
    Select one or more DXF floor plans. Choose the wall layers
    and confirm floor elevations before reconstruction.
  </p>

  <label class="uc-upload-field" for="uc-project-name">
    Project name
  </label>
  <input class="uc-upload-input"
         id="uc-project-name"
         value="My Building"
         maxlength="120">

  <div id="uc-upload-floors"></div>
</div>

<div class="uc-upload-result hidden" id="uc-upload-result">
  <div class="uc-upload-result-head">
    <strong id="uc-upload-result-title">
      CAD-derived 3D building
    </strong>
    <div class="uc-upload-floor-buttons"
         id="uc-upload-floor-buttons"></div>
  </div>
  <canvas id="uc-upload-canvas"></canvas>
  <div class="uc-upload-caption">
    Geometry comes from the uploaded DXF files.
    Heights and thicknesses are configured assumptions.
    Walkthrough registration and construction progress are unverified.
  </div>
</div>
"""

EXTRA_SCRIPT = """
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
import { OrbitControls } from
  "three/addons/controls/OrbitControls.js";

const el = id => document.getElementById(id);

let files = [];
let readers = [];
let model = null;

const canvas = el("uc-upload-canvas");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x15161e);

const camera = new THREE.PerspectiveCamera(
  55, 1, 0.1, 1000000
);

camera.position.set(20, 20, 20);

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias:true
});

renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const controls = new OrbitControls(
  camera, renderer.domElement
);

controls.enableDamping = true;

scene.add(new THREE.HemisphereLight(
  0xffffff, 0x667080, 2.4
));

const sunlight = new THREE.DirectionalLight(
  0xffffff, 1.6
);

sunlight.position.set(30, 60, 20);
scene.add(sunlight);

const building = new THREE.Group();
scene.add(building);

const floorGroups = [];

const colors = {
  exterior:0xb89aff,
  interior:0x8c95b4,
  partition:0xe49a75,
  lining:0x7cb8b1
};

function status(text) {
  el("upload-status").textContent = text;
}

function field(label, value, type="number") {
  const wrapper = document.createElement("div");

  const caption = document.createElement("label");
  caption.className = "uc-upload-field";
  caption.textContent = label;

  const input = document.createElement("input");
  input.className = "uc-upload-input";
  input.type = type;
  input.value = value;

  if (type === "number") {
    input.step = "any";
  }

  wrapper.append(caption, input);
  return {wrapper, input};
}

async function inspectFiles() {
  files = Array.from(el("plan-input").files || []);
  readers = [];

  const container = el("uc-upload-floors");
  container.replaceChildren();

  el("open-uploads").disabled = true;

  if (!files.length) {
    status("SELECT FLOOR PLANS TO BEGIN");
    return;
  }

  if (files.length > 20) {
    status("MAXIMUM 20 FLOORS");
    return;
  }

  try {
    for (let i=0; i<files.length; i++) {
      status(`INSPECTING FLOOR ${i+1} / ${files.length}`);

      const form = new FormData();
      form.append("file", files[i]);

      const response = await fetch(
        "/api/projects/inspect-dxf",
        {method:"POST",body:form}
      );

      const inspection = await response.json();

      if (!response.ok) {
        throw new Error(inspection.error || "DXF inspection failed");
      }

      const card = document.createElement("div");
      card.className = "uc-upload-floor";

      const heading = document.createElement("strong");
      heading.textContent = `${i+1} / ${files[i].name}`;
      card.append(heading);

      const name = field("Floor name", `Floor ${i+1}`, "text");
      const elevation = field("Elevation (metres)", i*3);
      const height = field("Wall height (metres)", 3);
      const thickness = field("Wall thickness (metres)", 0.2);

      card.append(name.wrapper);

      const row = document.createElement("div");
      row.className = "uc-upload-row";

      row.append(
        elevation.wrapper,
        height.wrapper,
        thickness.wrapper
      );

      card.append(row);

      let units = null;

      if (inspection.units_need_confirmation) {
        units = field(
          "Metres per drawing unit (required)",
          "",
          "number"
        );
        card.append(units.wrapper);
      }

      const layerTitle = document.createElement("label");
      layerTitle.className = "uc-upload-field";
      layerTitle.textContent = "Wall centreline layers";
      card.append(layerTitle);

      const layers = [];

      for (const layer of inspection.layers) {
        const item = document.createElement("div");
        item.className = "uc-upload-layer";

        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";

        const label = document.createElement("span");
        label.textContent =
          `${layer.name} (${layer.entity_count})`;

        const category = document.createElement("select");

        for (const type of [
          "exterior", "interior", "partition", "lining"
        ]) {
          const option = document.createElement("option");
          option.value = type;
          option.textContent = type;
          category.append(option);
        }

        const lower = layer.name.toLowerCase();

        const hints = [
          ["aussen", "exterior"],
          ["exterior", "exterior"],
          ["innen", "interior"],
          ["interior", "interior"],
          ["trenn", "partition"],
          ["partition", "partition"],
          ["vormauer", "lining"]
        ];

        for (const [word, type] of hints) {
          if (lower.includes(word)) {
            checkbox.checked = true;
            category.value = type;
            break;
          }
        }

        item.append(checkbox,label,category);
        card.append(item);

        layers.push({
          name:layer.name,
          checkbox,
          category
        });
      }

      container.append(card);

      readers.push(() => {
        const wallLayers = {};

        for (const layer of layers) {
          if (layer.checkbox.checked) {
            wallLayers[layer.name] = {
              category:layer.category.value
            };
          }
        }

        if (!Object.keys(wallLayers).length) {
          throw new Error(
            `Select wall layers for ${files[i].name}`
          );
        }

        const result = {
          name:name.input.value,
          elevation_m:Number(elevation.input.value),
          height_m:Number(height.input.value),
          thickness_m:Number(thickness.input.value),
          wall_layers:wallLayers
        };

        if (units) {
          if (!units.input.value) {
            throw new Error(
              `Confirm drawing units for ${files[i].name}`
            );
          }
          result.metres_per_unit =
            Number(units.input.value);
        }

        if (
          !Number.isFinite(result.elevation_m) ||
          !(result.height_m > 0) ||
          !(result.thickness_m > 0) ||
          (units && !(result.metres_per_unit > 0))
        ) {
          throw new Error(
            `Invalid dimensions for ${files[i].name}`
          );
        }

        return result;
      });
    }

    el("plan-file").textContent =
      `SELECTED / ${files.length} FLOOR PLAN(S)`;

    el("open-uploads").disabled = false;
    el("open-uploads").textContent =
      "Generate 3D building →";

    status("READY / CONFIRM WALL LAYERS");
  } catch (error) {
    status("ERROR / " + error.message);
  }
}

function makeShape(footprint) {
  const ring = footprint.exterior;

  if (!ring || ring.length < 4) return null;

  const shape = new THREE.Shape();

  shape.moveTo(ring[0][0],ring[0][1]);

  for (let i=1; i<ring.length; i++) {
    shape.lineTo(ring[i][0],ring[i][1]);
  }

  for (const hole of footprint.holes || []) {
    if (hole.length < 4) continue;

    const path = new THREE.Path();

    path.moveTo(hole[0][0],hole[0][1]);

    for (let i=1; i<hole.length; i++) {
      path.lineTo(hole[i][0],hole[i][1]);
    }

    shape.holes.push(path);
  }

  return shape;
}

function renderBuilding(project) {
  while (building.children.length) {
    const group = building.children[0];

    group.traverse(object => {
      object.geometry?.dispose();
      object.material?.dispose();
    });

    building.remove(group);
  }

  floorGroups.length = 0;

  const bounds = new THREE.Box3();

  for (const floor of project.floors) {
    const group = new THREE.Group();

    for (const surface of floor.surfaces) {
      const shape = makeShape(surface.footprint);
      if (!shape) continue;

      const height = Number(surface.height_m);
      if (!(height > 0)) continue;

      const geometry = new THREE.ExtrudeGeometry(
        shape,
        {depth:height,bevelEnabled:false}
      );

      geometry.rotateX(-Math.PI/2);

      const material = new THREE.MeshStandardMaterial({
        color:colors[surface.category] || 0xb0b0b0,
        roughness:0.82,
        side:THREE.DoubleSide
      });

      const mesh = new THREE.Mesh(geometry,material);
      mesh.position.y = Number(floor.elevation_m);

      group.add(mesh);
    }

    building.add(group);

    floorGroups.push({
      name:floor.floor_name,
      group
    });
  }

  bounds.setFromObject(building);

  if (!bounds.isEmpty()) {
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());

    building.position.x = -center.x;
    building.position.z = -center.z;

    const radius = Math.max(
      size.x,size.y,size.z,5
    );

    camera.position.set(
      radius*.85,
      center.y + radius*.8,
      radius*.9
    );

    controls.target.set(0,center.y,0);
    controls.update();
  }

  el("uc-upload-result").classList.remove("hidden");

  el("uc-upload-result-title").textContent =
    `${project.name} / ${project.floor_count} FLOORS`;

  const buttons = el("uc-upload-floor-buttons");
  buttons.replaceChildren();

  function button(label,handler) {
    const element = document.createElement("button");
    element.textContent = label;
    element.onclick = handler;
    buttons.append(element);
  }

  button("All floors", () => {
    floorGroups.forEach(f => f.group.visible = true);
  });

  floorGroups.forEach((floor,index) => {
    button(floor.name, () => {
      floorGroups.forEach((f,i) => {
        f.group.visible = i === index;
      });
    });
  });

  requestAnimationFrame(resize);
  el("uc-upload-result").scrollIntoView({
    behavior:"smooth",
    block:"start"
  });
}

async function createBuilding() {
  const button = el("open-uploads");
  button.disabled = true;

  try {
    if (!files.length || !readers.length) {
      throw new Error("Select DXF floor plans first");
    }

    const form = new FormData();

    form.append("config",JSON.stringify({
      name:el("uc-project-name").value,
      floors:readers.map(read => read())
    }));

    for (const file of files) {
      form.append("floorplans",file);
    }

    const video = el("video-input").files[0];

    if (video) {
      if (!["mp4","mov"].includes(
        video.name.split(".").pop().toLowerCase()
      )) {
        throw new Error("Use MP4 or MOV walkthroughs");
      }
      form.append("walkthroughs",video);
    }

    status("RECONSTRUCTING CAD GEOMETRY...");

    const response = await fetch(
      "/api/projects/create",
      {method:"POST",body:form}
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(
        result.error || "Reconstruction failed"
      );
    }

    const geometryResponse = await fetch(
      result.geometry_url
    );

    if (!geometryResponse.ok) {
      throw new Error("Cannot retrieve project geometry");
    }

    model = await geometryResponse.json();
    renderBuilding(model);

    status(
      `RECONSTRUCTED / ${result.floor_count} FLOORS / ` +
      `${result.surface_count} SURFACES`
    );

  } catch (error) {
    console.error(error);
    status("ERROR / " + error.message);
  } finally {
    button.disabled = false;
  }
}

function resize() {
  if (el("uc-upload-result").classList.contains("hidden")) {
    return;
  }

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;

  if (!width || !height) return;

  renderer.setSize(width,height,false);
  camera.aspect = width/height;
  camera.updateProjectionMatrix();
}

window.addEventListener("resize",resize);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  resize();
  renderer.render(scene,camera);
}
animate();

// Override only the personal-upload workflow.
// Sample 01 continues using the original handlers.
el("plan-input").multiple = true;
el("plan-input").onchange = inspectFiles;
el("open-uploads").onclick = createBuilding;

// Support dropping multiple DXFs onto the original dropzone.
el("plan-drop").addEventListener("drop", event => {
  event.preventDefault();
  event.stopImmediatePropagation();

  const dropped = Array.from(event.dataTransfer.files);

  if (!dropped.every(f => f.name.toLowerCase().endsWith(".dxf"))) {
    status("ONLY DXF FLOOR PLANS ARE SUPPORTED");
    return;
  }

  const transfer = new DataTransfer();
  dropped.forEach(file => transfer.items.add(file));

  el("plan-input").files = transfer.files;
  inspectFiles();
}, true);

</script>
"""


def enhance_dashboard_html(html: str) -> str:
    """Inject the upload feature into the existing dashboard."""

    anchor = '<div class="upload-actions">'
    if anchor not in html:
        raise ValueError("Dashboard upload actions not found")

    html = html.replace(
        anchor,
        EXTRA_HTML + "\n" + anchor,
        1,
    )

    html = html.replace(
        'accept=".dxf" class="hidden"',
        'accept=".dxf" multiple class="hidden"',
        1,
    )

    html = html.replace(
        "Preview selected files →",
        "Generate 3D building →",
        1,
    )

    html = html.replace(
        "Local file selection does not submit your files to",
        "Selected floor plans are submitted to",
        1,
    )

    html = html.replace(
        "the analysis backend. DXF parsing and progress",
        "the local reconstruction backend. Construction progress",
        1,
    )

    html = html.replace(
        "verification are not performed in this step.",
        "verification is not performed by CAD reconstruction.",
        1,
    )

    html = html.replace(
        "</head>",
        EXTRA_STYLE + "\n</head>",
        1,
    )

    html = html.replace(
        "</body>",
        EXTRA_SCRIPT + "\n</body>",
        1,
    )

    return html
