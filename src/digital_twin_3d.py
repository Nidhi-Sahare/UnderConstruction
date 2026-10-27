
"""
Under-construction — Architectural 3D Digital Twin

Routes:
    GET /twin
    GET /api/twin/elements
    GET /api/twin/doors

Walls:
    Prefer /api/twin/footprints, then /api/twin/cleaned,
    then /api/twin/elements.

Doors:
    Read the original CAD-derived Step 39 screening report.
    Display provisional door-leaf spans at CAD coordinates.

Important:
    No wall cutouts.
    No verified physical door counts.
    No construction-completion inference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from flask import jsonify, render_template_string

from underconstruction_app import app


ROOT = Path(__file__).resolve().parents[1]
TWIN_PATH = ROOT / "data/digital_twin/building_elements.json"
DOOR_PATH = (
    ROOT
    / "outputs/cad_openings/verification"
    / "door_span_verification.json"
)

DOOR_STATUSES = {
    "GEOMETRICALLY_SUPPORTED_CANDIDATE",
    "AMBIGUOUS_GEOMETRY",
    "INSUFFICIENT_WALL_EVIDENCE",
}


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def valid_point(point):
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        return None

    x = finite_number(point[0])
    y = finite_number(point[1])

    if x is None or y is None:
        return None

    return [x, y]


def get_elements():
    with TWIN_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    raw = data if isinstance(data, list) else data.get("elements", [])
    elements = []

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue

        start = valid_point(item.get("start"))
        end = valid_point(item.get("end"))

        if start is None or end is None:
            continue

        geometry = item.get("geometry") or {}

        height = finite_number(geometry.get("height_m", 3.0))
        thickness = finite_number(
            geometry.get("thickness_m", 0.15)
        )

        if height is None or thickness is None:
            continue

        length = math.dist(start, end)

        if length < 0.01 or height <= 0 or thickness <= 0:
            continue

        element_id = str(
            item.get("element_id") or f"cad_element_{index}"
        )

        elements.append({
            "id": element_id,
            "source_element_ids": [element_id],
            "source_element_count": 1,
            "category": str(item.get("category") or "unknown"),
            "start": start,
            "end": end,
            "length_m": length,
            "height_m": height,
            "thickness_m": thickness,
            "construction_status": "UNKNOWN",
            "progress_verified": False,
        })

    return elements


def get_doors():
    if not DOOR_PATH.is_file():
        return {
            "doors": [],
            "count": 0,
            "status": "VERIFICATION_REPORT_NOT_FOUND",
            "wall_cutouts_generated": False,
        }

    with DOOR_PATH.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    doors = []

    for index, item in enumerate(report.get("results", [])):
        if not isinstance(item, dict):
            continue

        start = valid_point(item.get("start"))
        end = valid_point(item.get("end"))

        if start is None or end is None:
            continue

        length = math.dist(start, end)

        if not 0.05 <= length <= 10.0:
            continue

        status = item.get("status")

        if status not in DOOR_STATUSES:
            status = "INSUFFICIENT_WALL_EVIDENCE"

        evidence = item.get("evidence") or {}

        doors.append({
            "id": str(
                item.get("span_id") or f"door_span_{index:03d}"
            ),
            "start": start,
            "end": end,
            "length_m": round(length, 6),
            "status": status,
            "source_arc_handles": [
                str(value)
                for value in item.get("source_arc_handles", [])
            ],
            "source_line_handles": [
                str(value)
                for value in item.get("source_line_handles", [])
            ],
            "endpoint_a_candidates": len(
                evidence.get("endpoint_a_candidates") or []
            ),
            "endpoint_b_candidates": len(
                evidence.get("endpoint_b_candidates") or []
            ),
            "wall_boundary_proximity_fraction":
                evidence.get(
                    "interior_wall_boundary_proximity_fraction"
                ),
            "opening_verified": False,
            "safe_to_cut": False,
            "construction_status": "UNKNOWN",
            "representation": "PROVISIONAL_CAD_DOOR_LEAF",
        })

    return {
        "doors": doors,
        "count": len(doors),
        "status": "PROVISIONAL_GEOMETRY",
        "source": str(DOOR_PATH.relative_to(ROOT)),
        "wall_cutouts_generated": False,
        "construction_progress_verified": False,
        "note": (
            "Each record is a provisional radial leaf span. "
            "Multiple records may describe one physical doorway."
        ),
    }




# UNDER_CONSTRUCTION_FACADE_INTEGRATION_V2
@app.route("/api/twin/facade")
def twin_facade_api():
    import json
    from pathlib import Path
    from flask import jsonify

    project_root = Path(__file__).resolve().parents[1]

    source_path = (
        project_root
        / "outputs"
        / "cad_facade"
        / "facade_geometry.json"
    )

    if not source_path.exists():
        return jsonify({
            "error": "Facade geometry has not been generated",
            "count": 0,
            "paths": [],
        }), 404

    data = json.loads(
        source_path.read_text(encoding="utf-8")
    )

    paths = []

    for item in data.get("paths", []):
        if item.get("category") != "CURTAIN_WALL":
            continue

        points = item.get("points", [])

        if len(points) < 2:
            continue

        paths.append({
            "id": item.get("id"),
            "source_handle": item.get("source_handle"),
            "source_layer": item.get("source_layer"),
            "category": item.get("category"),
            "closed": item.get("closed", False),
            "points": points,
            "construction_status": "UNKNOWN",
            "vertical_dimensions_verified": False,
            "physical_component_verified": False,
        })

    return jsonify({
        "coordinate_system": "DXF_METERS",
        "representation": "PROVISIONAL_PLAN_FRAMING",
        "count": len(paths),
        "paths": paths,
        "physical_components_verified": 0,
        "vertical_dimensions_verified": False,
        "construction_progress_verified": False,
    })


@app.route("/api/twin/elements")
def api_twin_elements():
    try:
        elements = get_elements()
        return jsonify({
            "elements": elements,
            "original_element_count": len(elements),
            "rendered_mesh_count": len(elements),
            "representation": "EXTRUDED_CENTERLINES",
            "construction_progress_verified": False,
        })
    except Exception:
        app.logger.exception("Digital twin geometry failed")
        return jsonify({"error": "Unable to load geometry"}), 500


@app.route("/api/twin/doors")
def api_twin_doors():
    try:
        return jsonify(get_doors())
    except Exception:
        app.logger.exception("Door geometry loading failed")
        return jsonify({"error": "Unable to load door geometry"}), 500


@app.route("/twin")
def twin_page():
    return render_template_string(HTML)


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Under-construction — Digital Twin</title>

<style>
:root {
    --bg:#0c1117;
    --panel:#151c24;
    --border:#293541;
    --text:#edf1f4;
    --muted:#9aa9b7;
    --accent:#ff7045;
    --blue:#4aa8ff;
}
* {box-sizing:border-box}
html,body {margin:0;width:100%;height:100%}
body {
    background:var(--bg);
    color:var(--text);
    font-family:Inter,Arial,sans-serif;
}
header {
    height:62px;
    padding:0 22px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    border-bottom:1px solid var(--border);
    gap:12px;
}
.brand {font-size:22px;font-weight:750}
.brand span {color:var(--accent)}
nav {display:flex;gap:18px;flex-wrap:wrap}
nav a {
    color:var(--muted);
    text-decoration:none;
    font-size:13px;
}
nav a.active {color:var(--accent)}
.layout {
    display:grid;
    grid-template-columns:260px minmax(0,1fr) 300px;
    height:calc(100vh - 62px);
}
aside {
    background:var(--panel);
    padding:18px;
    overflow-y:auto;
}
.left {border-right:1px solid var(--border)}
.right {border-left:1px solid var(--border)}
h2 {font-size:16px;margin:0 0 16px}
h3 {font-size:14px;margin:20px 0 10px}
.muted {
    color:var(--muted);
    font-size:12px;
    line-height:1.6;
}
input[type="search"] {
    width:100%;
    background:#202b36;
    color:var(--text);
    padding:11px;
    border:1px solid var(--border);
    border-radius:7px;
    margin:12px 0;
}
.toggle {
    display:flex;
    align-items:center;
    gap:9px;
    font-size:13px;
    margin:14px 0;
    cursor:pointer;
}
.item {
    border-bottom:1px solid var(--border);
    padding:11px 7px;
    cursor:pointer;
    overflow-wrap:anywhere;
}
.item:hover {background:#202b36}
.item.active {
    background:#20374b;
    border-left:3px solid var(--blue);
}
.item strong {font-size:12px}
.item small {
    display:block;
    margin-top:5px;
    color:var(--muted);
}
.viewport {
    position:relative;
    overflow:hidden;
    background:#101821;
    min-width:0;
}
#scene {display:block;width:100%;height:100%}
.legend,.hint {
    position:absolute;
    padding:12px;
    background:#111a23ed;
    border:1px solid var(--border);
    border-radius:8px;
    font-size:12px;
    line-height:1.8;
}
.legend {top:15px;right:15px}
.hint {left:18px;bottom:18px;color:var(--muted)}
.dot {
    display:inline-block;
    width:9px;
    height:9px;
    border-radius:50%;
    margin-right:8px;
}
.toolbar {
    position:absolute;
    bottom:18px;
    right:18px;
    display:flex;
    gap:8px;
}
button {
    background:#202b36;
    border:1px solid var(--border);
    color:var(--text);
    padding:9px 12px;
    border-radius:7px;
    cursor:pointer;
}
button:hover {border-color:var(--blue)}
.value {
    margin:5px 0 15px;
    font-size:14px;
    overflow-wrap:anywhere;
}
.warning {
    border:1px solid #76502f;
    background:#30251d;
    color:#ffd3a4;
    padding:12px;
    border-radius:8px;
    margin-top:18px;
    font-size:12px;
    line-height:1.6;
}
.photo {
    width:100%;
    border-radius:8px;
    border:1px solid var(--border);
    margin-top:12px;
}
.evidence {
    padding:10px 0;
    border-bottom:1px solid var(--border);
}
#elementList {max-height:55vh;overflow-y:auto}
@media(max-width:1000px) {
    .layout {
        grid-template-columns:210px minmax(0,1fr);
        height:auto;
    }
    .viewport {height:65vh}
    .right {
        grid-column:1/-1;
        border-top:1px solid var(--border);
    }
}
</style>

<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.1/examples/jsm/"
  }
}
</script>
</head>

<body>
<header>
    <div class="brand">Under<span>-construction</span></div>
    <nav>
        <a href="/">Dashboard</a>
        <a href="/twin" class="active">3D Model</a>
        <a href="/evidence">Evidence</a>
        <a href="/screening">Screening</a>
    </nav>
</header>

<div class="layout">
<aside class="left">
    <h2>Architectural Elements</h2>
    <div class="muted" id="count">Loading...</div>
    <div class="muted" id="geometrySource"></div>
    <div class="muted">Default: CAD-derived provisional wall footprints only. Doors, windows and façade interpretations are hidden until explicitly enabled.</div>

    <label class="toggle">
        <input type="checkbox" id="showDoors">
        Show provisional door geometry
    </label>

    <div class="muted" id="doorCount"></div>

    <label class="toggle">
        <input type="checkbox" id="showWindows">
        Show provisional window candidates
    </label>
    <div class="muted" id="windowCount"></div>

    <h3>Find an element</h3>
    <input type="search" id="search"
           placeholder="Search walls, doors, windows, CAD handles">
    <div id="elementList"></div>
</aside>

<main class="viewport">
    <canvas id="scene"></canvas>

    <div class="legend">
        <div>
            <span class="dot" style="background:#61c5d8"></span>
            Walls — translucent CAD geometry
        </div>
        <div>
            <span class="dot" style="background:#22c55e"></span>
            Door — geometric support
        </div>
        <div>
            <span class="dot" style="background:#f59e0b"></span>
            Door — ambiguous
        </div>
        <div>
            <span class="dot" style="background:#ef4444"></span>
            Door — insufficient evidence
        </div>
        <div>
            <span class="dot" style="background:#22d3ee"></span>
            Window — provisional candidate
        </div>
        <div>
            <span class="dot" style="background:#4aa8ff"></span>
            Selected
        </div>
    </div>

    <div class="hint">
        3D perspective by default · Left drag: orbit · Right drag: pan · Scroll: zoom
        <div id="renderStatus"></div>
    </div>

    <div class="toolbar">
        <button id="reset">3D perspective</button>
        <button id="top">Top view</button>
    </div>
</main>

<aside class="right">
    <h2>Selected Element</h2>

    <div class="muted">Element ID</div>
    <div class="value" id="elementId">Select a wall, door, or window</div>

    <div class="muted">Type</div>
    <div class="value" id="category">—</div>

    <div class="muted">Geometry</div>
    <div class="value" id="dimensions">—</div>

    <div class="muted">Original CAD references</div>
    <div class="value" id="sourceCount">—</div>

    <div class="muted">Geometric screening</div>
    <div class="value" id="geometryStatus">—</div>

    <div class="muted">Construction status</div>
    <div class="value">UNKNOWN</div>

    <h3>Visual Evidence Candidates</h3>
    <div id="evidence" class="muted">
        Select a wall to retrieve nearby frames.
    </div>

    <div class="warning">
        Wall height and thickness are assumed.
        Door leaves are provisional CAD interpretations.
        Door and window openings, CAD/video alignment,
        and construction completion remain unverified.
        Window heights and sill levels are display assumptions.
        No walls have been cut.
    </div>
</aside>
</div>

<script type="module">
import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";

const $ = id => document.getElementById(id);

const canvas = $("scene");
const viewport = canvas.parentElement;

const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias:true
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2));

const scene = new THREE.Scene();
scene.background = new THREE.Color("#101821");

const camera = new THREE.PerspectiveCamera(50,1,0.1,2000);
const controls = new OrbitControls(camera,canvas);
controls.enableDamping = true;
controls.screenSpacePanning = true;

scene.add(new THREE.HemisphereLight(0xffffff,0x465466,2));

const sun = new THREE.DirectionalLight(0xffffff,2);
sun.position.set(30,80,40);
scene.add(sun);

const buildingGroup = new THREE.Group();
const doorGroup = new THREE.Group();
const windowGroup = new THREE.Group();
scene.add(buildingGroup,doorGroup,windowGroup);

// Visual styling only: wall categories receive different translucent colours.
// CAD footprints, coordinates and wall counts are unchanged.
const wallPalette = {
    exterior: 0x61c5d8,    // cyan
    interior: 0xa78bfa,    // violet
    partition: 0xf2a766,   // amber
    lining: 0x7cc9a5,      // mint
    other: 0x9fb8d1       // blue-grey
};

function wallCategory(surface) {
    const name = String(surface.category || "").toLowerCase();
    if (/exterior|aussen|außen|outer/.test(name)) return "exterior";
    if (/partition|trenn/.test(name)) return "partition";
    if (/lining|vormauer|cladding/.test(name)) return "lining";
    if (/interior|innen|internal/.test(name)) return "interior";
    return "other";
}

const wallMaterials = Object.fromEntries(
    Object.entries(wallPalette).map(([name, color]) => [
        name,
        new THREE.MeshStandardMaterial({
            color,
            roughness: 0.58,
            metalness: 0.04,
            transparent: true,
            opacity: 0.44,
            depthWrite: false,
            side: THREE.DoubleSide
        })
    ])
);

function materialForWall(surface) {
    return wallMaterials[wallCategory(surface)];
}

function addWallEdges(mesh, surface) {
    // Edges add legibility without changing the underlying CAD geometry.
    const edges = new THREE.EdgesGeometry(mesh.geometry, 28);
    const lines = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({
            color: wallPalette[wallCategory(surface)],
            transparent: true,
            opacity: 0.75,
            depthWrite: false
        })
    );
    lines.raycast = () => {}; // selection stays on the wall mesh
    mesh.add(lines);
}

const selectedMaterial = new THREE.MeshStandardMaterial({
    color:0x4aa8ff,
    roughness:0.65,
    emissive:0x14334b,
    transparent:true,
    opacity:0.82,
    depthWrite:false,
    side:THREE.DoubleSide
});

const doorColors = {
    GEOMETRICALLY_SUPPORTED_CANDIDATE:0x22c55e,
    AMBIGUOUS_GEOMETRY:0xf59e0b,
    INSUFFICIENT_WALL_EVIDENCE:0xef4444
};

const doorMaterials = Object.fromEntries(
    Object.entries(doorColors).map(([status,color])=>[
        status,
        new THREE.MeshStandardMaterial({
            color,
            transparent:true,
            opacity:0.70,
            side:THREE.DoubleSide,
            depthWrite:false
        })
    ])
);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

let surfaces = [];
let doors = [];
let windows = [];
let wallMeshes = [];
let doorMeshes = [];
let windowMeshes = [];
let selectedMesh = null;
let modelCenter = new THREE.Vector3();
let grid = null;
let requestGeneration = 0;

function resize() {
    const width=viewport.clientWidth;
    const height=viewport.clientHeight;
    if (!width || !height) return;

    renderer.setSize(width,height,false);
    camera.aspect=width/height;
    camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(viewport);

function elementId(item) {
    return item.surface_id || item.id || "unknown";
}

function boundsForSurfaces(items) {
    const box=new THREE.Box3();

    for (const item of items) {
        if (item.footprint) {
            const rings=[
                item.footprint.exterior || [],
                ...(item.footprint.holes || [])
            ];

            for (const ring of rings) {
                for (const p of ring) {
                    if (Number.isFinite(p[0]) && Number.isFinite(p[1])) {
                        box.expandByPoint(
                            new THREE.Vector3(p[0],0,p[1])
                        );
                    }
                }
            }
        } else if (item.start && item.end) {
            for (const p of [item.start,item.end]) {
                box.expandByPoint(
                    new THREE.Vector3(p[0],0,p[1])
                );
            }
        }
    }
    return box;
}

function makeShape(footprint,center) {
    const exterior=footprint.exterior || [];
    if (exterior.length<4) return null;

    const shape=new THREE.Shape();

    exterior.forEach((p,i)=>{
        const x=p[0]-center.x;
        // DXF Y becomes world Z. rotateX(-PI/2) maps shape Y to -Z,
        // so reverse the planar Y coordinate BEFORE extrusion.
        const y=center.z-p[1];
        if (i===0) shape.moveTo(x,y);
        else shape.lineTo(x,y);
    });
    shape.closePath();

    for (const ring of footprint.holes || []) {
        if (ring.length<4) continue;
        const hole=new THREE.Path();

        ring.forEach((p,i)=>{
            const x=p[0]-center.x;
            // DXF Y becomes world Z. rotateX(-PI/2) maps shape Y to -Z,
        // so reverse the planar Y coordinate BEFORE extrusion.
        const y=center.z-p[1];
            if (i===0) hole.moveTo(x,y);
            else hole.lineTo(x,y);
        });

        hole.closePath();
        shape.holes.push(hole);
    }

    return shape;
}

function createWallMesh(surface,center) {
    let geometry;
    const height=Number(surface.height_m || 3);

    if (!Number.isFinite(height) || height<=0) return null;

    if (surface.footprint) {
        const shape=makeShape(surface.footprint,center);
        if (!shape) return null;

        geometry=new THREE.ExtrudeGeometry(shape,{
            depth:height,
            bevelEnabled:false,
            steps:1,
            curveSegments:1
        });
        geometry.rotateX(-Math.PI/2);

    } else {
        if (!surface.start || !surface.end) return null;

        const [x1,y1]=surface.start;
        const [x2,y2]=surface.end;
        const length=Math.hypot(x2-x1,y2-y1);
        // Display minimum for thin centerline geometry only.
        // CAD footprint solids above retain their original width.
        const thickness=Math.max(Number(surface.thickness_m || 0.15),0.28);

        if (length<0.01 || thickness<=0) return null;

        geometry=new THREE.BoxGeometry(
            length,height,thickness
        );

        const mesh=new THREE.Mesh(geometry,materialForWall(surface));
        addWallEdges(mesh,surface);
        mesh.position.set(
            (x1+x2)/2-center.x,
            height/2,
            (y1+y2)/2-center.z
        );
        mesh.rotation.y=-Math.atan2(y2-y1,x2-x1);
        mesh.userData={kind:"wall",item:surface};
        return mesh;
    }

    const mesh=new THREE.Mesh(geometry,materialForWall(surface));
    addWallEdges(mesh,surface);
    mesh.userData={kind:"wall",item:surface};
    return mesh;
}

function createDoorMesh(door,center) {
    const [x1,y1]=door.start;
    const [x2,y2]=door.end;

    const dx=x2-x1;
    const dy=y2-y1;
    const length=Math.hypot(dx,dy);

    if (length<0.05) return null;

    // Visualization dimensions, not verified door dimensions.
    const displayHeight=2.1;
    const displayThickness=0.035;

    const geometry=new THREE.BoxGeometry(
        length,displayHeight,displayThickness
    );

    const material=doorMaterials[door.status] ||
        doorMaterials.INSUFFICIENT_WALL_EVIDENCE;

    const mesh=new THREE.Mesh(geometry,material);

    mesh.position.set(
        (x1+x2)/2-center.x,
        displayHeight/2,
        (y1+y2)/2-center.z
    );

    mesh.rotation.y=-Math.atan2(dy,dx);
    mesh.userData={kind:"door",item:door};

    return mesh;
}


// Window spans originate from CAD plan geometry.
// Vertical dimensions are ONLY visualization assumptions.
const windowMaterial = new THREE.MeshStandardMaterial({
    color:0x22d3ee,
    transparent:true,
    opacity:0.52,
    roughness:0.25,
    metalness:0.1,
    side:THREE.DoubleSide,
    depthWrite:false
});

function spanDistance(a,b) {
    return Math.hypot(a[0]-b[0],a[1]-b[1]);
}

function validWindow(w) {
    return w &&
        Array.isArray(w.start) &&
        Array.isArray(w.end) &&
        w.start.length===2 &&
        w.end.length===2 &&
        [...w.start,...w.end].every(Number.isFinite) &&
        spanDistance(w.start,w.end)>0.05;
}

function windowDirection(w) {
    const dx=w.end[0]-w.start[0];
    const dy=w.end[1]-w.start[1];
    const len=Math.hypot(dx,dy);
    return [dx/len,dy/len];
}

function windowProjection(w,axis) {
    const a=w.start[0]*axis[0]+w.start[1]*axis[1];
    const b=w.end[0]*axis[0]+w.end[1]*axis[1];
    return [Math.min(a,b),Math.max(a,b)];
}

function windowPerpendicular(w,axis) {
    const center=[
        (w.start[0]+w.end[0])/2,
        (w.start[1]+w.end[1])/2
    ];
    return -axis[1]*center[0]+axis[0]*center[1];
}

function overlapFraction(a,b) {
    const overlap=Math.max(
        0,Math.min(a[1],b[1])-Math.max(a[0],b[0])
    );
    const smaller=Math.min(a[1]-a[0],b[1]-b[0]);
    return smaller>0 ? overlap/smaller : 0;
}

function sameWindowLocation(a,b) {
    const axis=windowDirection(a);
    const other=windowDirection(b);
    const parallel=Math.abs(
        axis[0]*other[0]+axis[1]*other[1]
    );

    if (parallel<Math.cos(5*Math.PI/180)) return false;

    const offset=Math.abs(
        windowPerpendicular(a,axis)-
        windowPerpendicular(b,axis)
    );

    // Conservative 8 cm offset allowance for nearby CAD
    // frame components, not a wall association threshold.
    if (offset>0.08) return false;

    return overlapFraction(
        windowProjection(a,axis),
        windowProjection(b,axis)
    )>=0.80;
}

function consolidateWindows(input) {
    const candidates=input.filter(validWindow);
    const parent=candidates.map((_,i)=>i);

    function find(i) {
        while(parent[i]!==i) {
            parent[i]=parent[parent[i]];
            i=parent[i];
        }
        return i;
    }

    function union(a,b) {
        a=find(a);
        b=find(b);
        if(a!==b) parent[b]=a;
    }

    // Group only components sharing essentially the same
    // plan location. Do not join adjacent glazing modules.
    for(let i=0;i<candidates.length;i++) {
        for(let j=i+1;j<candidates.length;j++) {
            if(sameWindowLocation(candidates[i],candidates[j])) {
                union(i,j);
            }
        }
    }

    const groups=new Map();
    candidates.forEach((w,i)=>{
        const root=find(i);
        if(!groups.has(root)) groups.set(root,[]);
        groups.get(root).push(w);
    });

    const output=[];

    for(const group of groups.values()) {
        // Prefer the longest source span as the representative.
        // This avoids averaging full-frame and inner-pane spans.
        const representative=[...group].sort(
            (a,b)=>spanDistance(b.start,b.end)-
                   spanDistance(a.start,a.end)
        )[0];

        const handles=[...new Set(
            group.flatMap(w=>w.source_handles||[])
        )];

        output.push({
            ...representative,
            id:group.map(w=>w.id).sort()[0],
            source_group_ids:group.map(w=>w.id),
            source_handles:handles,
            component_count:group.length,
            status:"PROVISIONAL_WINDOW_CANDIDATE",
            opening_verified:false,
            wall_association_verified:false,
            vertical_dimensions_verified:false,
            construction_status:"UNKNOWN",
            safe_to_cut:false
        });
    }

    return output;
}

function createWindowMesh(item,center) {
    const [x1,y1]=item.start;
    const [x2,y2]=item.end;
    const dx=x2-x1;
    const dy=y2-y1;
    const length=Math.hypot(dx,dy);

    if(length<=0.05) return null;

    // Visualization only: NOT recovered from the DXF.
    const assumedSill=0.90;
    const assumedGlazingHeight=1.20;
    const assumedThickness=0.035;

    const geometry=new THREE.BoxGeometry(
        length,assumedGlazingHeight,assumedThickness
    );

    const mesh=new THREE.Mesh(geometry,windowMaterial);

    mesh.position.set(
        (x1+x2)/2-center.x,
        assumedSill+assumedGlazingHeight/2,
        (y1+y2)/2-center.z
    );

    mesh.rotation.y=-Math.atan2(dy,dx);
    mesh.userData={kind:"window",item};
    return mesh;
}

function buildWindows(items) {
    for(const item of items) {
        const mesh=createWindowMesh(item,modelCenter);
        if(!mesh) continue;
        windowGroup.add(mesh);
        windowMeshes.push(mesh);
    }
    windowGroup.visible=$("showWindows").checked;
}

function buildWalls(items) {
    const bounds=boundsForSurfaces(items);

    if (bounds.isEmpty()) {
        throw new Error("No valid wall geometry");
    }

    modelCenter=bounds.getCenter(new THREE.Vector3());

    for (const surface of items) {
        const mesh=createWallMesh(surface,modelCenter);
        if (!mesh) continue;

        buildingGroup.add(mesh);
        wallMeshes.push(mesh);
    }

    const size=bounds.getSize(new THREE.Vector3());
    const gridSize=Math.max(size.x,size.z,10);

    grid=new THREE.GridHelper(
        gridSize*1.2,30,0x354454,0x25313e
    );
    grid.position.y=-0.02;
    scene.add(grid);

    fitCamera(false);
}

function buildDoors(items) {
    for (const door of items) {
        const mesh=createDoorMesh(door,modelCenter);
        if (!mesh) continue;

        doorGroup.add(mesh);
        doorMeshes.push(mesh);
    }

    doorGroup.visible=$("showDoors").checked;
}

function fitCamera(top=false) {
    // Fit the ACTUAL rendered wall solids, not unverified façade runs.
    // Use the camera field-of-view and viewport aspect ratio, so the
    // default view is a readable 3D building instead of a tiny plan.
    buildingGroup.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(buildingGroup);
    if (box.isEmpty()) return;

    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const width = Math.max(size.x, 0.1);
    const depth = Math.max(size.z, 0.1);
    const height = Math.max(size.y, 0.1);
    const horizontal = Math.hypot(width, depth);
    const aspect = Math.max(0.3, viewport.clientWidth / Math.max(1, viewport.clientHeight));
    const vfov = THREE.MathUtils.degToRad(camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect);
    const limitingFov = Math.min(vfov, hfov);
    const radius = Math.max(horizontal / 2, height);
    const distance = radius / Math.sin(limitingFov / 2) * 1.32;

    controls.target.copy(center);
    camera.up.set(0, 1, 0);
    if (top) {
        camera.up.set(0, 0, -1);
        camera.position.set(center.x, center.y + distance, center.z + 0.0001);
    } else {
        // High enough to inspect the plan, low enough to see wall height.
        const azimuth = Math.PI / 4;
        const elevation = THREE.MathUtils.degToRad(31);
        camera.position.set(
            center.x + distance * Math.cos(elevation) * Math.cos(azimuth),
            center.y + distance * Math.sin(elevation),
            center.z + distance * Math.cos(elevation) * Math.sin(azimuth)
        );
    }
    camera.near = Math.max(0.01, distance / 5000);
    camera.far = Math.max(1000, distance * 10);
    camera.updateProjectionMatrix();
    controls.minDistance = Math.max(0.1, Math.min(width, depth) * 0.02);
    controls.maxDistance = Math.max(100, distance * 10);
    controls.update();
}

function allItems() {
    return [
        ...surfaces.map(item=>({kind:"wall",item})),
        ...($("showDoors").checked
            ? doors.map(item=>({kind:"door",item}))
            : []),
        ...($("showWindows").checked
            ? windows.map(item=>({kind:"window",item}))
            : [])
    ];
}

function renderList() {
    const query=$("search").value.trim().toLowerCase();
    const container=$("elementList");
    container.replaceChildren();

    const matches=allItems().filter(({kind,item})=>{
        const searchable=[
            kind,
            elementId(item),
            item.category,
            item.status,
            ...(item.source_element_ids || []),
            ...(item.source_arc_handles || []),
            ...(item.source_handles || []),
            ...(item.source_group_ids || [])
        ].join(" ").toLowerCase();

        return searchable.includes(query);
    });

    for (const {kind,item} of matches.slice(0,180)) {
        const row=document.createElement("div");
        row.className="item";

        if (
            selectedMesh &&
            selectedMesh.userData.kind===kind &&
            elementId(selectedMesh.userData.item)===elementId(item)
        ) {
            row.classList.add("active");
        }

        const title=document.createElement("strong");
        title.textContent=elementId(item);

        const details=document.createElement("small");
        details.textContent=kind==="door"
            ? `Door-leaf candidate · ${item.status}`
            : kind==="window"
                ? `Window candidate · ${item.component_count} CAD groups`
                : `${item.category || "wall"} · CAD wall`;

        row.append(title,details);

        row.addEventListener("click",()=>{
            selectElement(kind,elementId(item));
        });

        container.appendChild(row);
    }
}

function restoreMaterial(mesh) {
    if (!mesh) return;

    if (mesh.userData.kind==="window") {
        mesh.material=windowMaterial;
    } else if (mesh.userData.kind==="door") {
        mesh.material=doorMaterials[mesh.userData.item.status] ||
            doorMaterials.INSUFFICIENT_WALL_EVIDENCE;
    } else {
        mesh.material=materialForWall(mesh.userData.item);
    }
}

function selectElement(kind,id) {
    const candidates=kind==="door" ? doorMeshes :
        kind==="window" ? windowMeshes : wallMeshes;

    const mesh=candidates.find(
        candidate=>elementId(candidate.userData.item)===id
    );

    if (!mesh) return;

    restoreMaterial(selectedMesh);

    selectedMesh=mesh;
    selectedMesh.material=selectedMaterial;

    const item=mesh.userData.item;

    $("elementId").textContent=elementId(item);

    if (kind==="window") {
        ++requestGeneration;
        $("category").textContent="Provisional CAD window candidate";
        $("dimensions").textContent=
            `${Number(item.span_m).toFixed(2)} m plan span · `+
            "0.90 m sill and 1.20 m glazing height (assumed)";
        $("sourceCount").textContent=
            `${item.component_count} grouped CAD components · `+
            `${item.source_handles.length} DXF handles: `+
            item.source_handles.join(", ");
        $("geometryStatus").textContent=
            "Plan geometry only; wall opening unverified";
        $("evidence").textContent=
            "Window-specific visual evidence has not been verified.";
    } else if (kind==="door") {
        ++requestGeneration;
        $("category").textContent="Provisional CAD door leaf";

        $("dimensions").textContent=
            `${Number(item.length_m).toFixed(2)} m span · `+
            "2.10 m display height (assumed)";

        $("sourceCount").textContent=
            item.source_arc_handles.join(", ") || "Unknown";

        $("geometryStatus").textContent=item.status;

        $("evidence").textContent=
            "Door-specific video evidence is not established. "+
            "This geometry is derived from the CAD drawing, "+
            "not from verified construction footage.";

    } else {
        $("category").textContent=item.category || "Wall";

        const area=item.area_m2;
        $("dimensions").textContent=
            area !== undefined
                ? `${Number(area).toFixed(2)} m² footprint · `+
                  `${Number(item.height_m).toFixed(2)} m assumed height`
                : `${Number(item.height_m).toFixed(2)} m assumed height`;

        const sourceIds=item.source_element_ids || [item.id];

        $("sourceCount").textContent=
            `${sourceIds.length} CAD elements`;

        $("geometryStatus").textContent=
            "CAD-derived provisional footprint; height and openings unverified";

        loadEvidence(sourceIds);
    }

    renderList();
}

function frameIndex(frame) {
    for (const key of ["frame_index","index","frame_id"]) {
        const value=frame[key];
        if (
            value!==undefined &&
            value!==null &&
            Number.isInteger(Number(value))
        ) {
            return Number(value);
        }
    }
    return null;
}

async function loadEvidence(sourceIds) {
    const token=++requestGeneration;
    const container=$("evidence");

    container.textContent="Loading spatial evidence candidates...";

    try {
        const requests=sourceIds.slice(0,12).map(async id=>{
            const response=await fetch(
                `/api/evidence/element/${encodeURIComponent(id)}?candidate=1`
            );

            if (!response.ok) return [];

            const data=await response.json();

            return data.candidate_frames ||
                data.frames ||
                data.candidates ||
                [];
        });

        const responses=await Promise.allSettled(requests);

        if (token!==requestGeneration) return;

        const unique=new Map();

        for (const response of responses) {
            if (response.status!=="fulfilled") continue;

            for (const frame of response.value) {
                const index=frameIndex(frame);
                if (index===null) continue;

                const distance=Number(
                    frame.distance_m ??
                    frame.camera_distance_m ??
                    Infinity
                );

                const existing=unique.get(index);

                if (!existing || distance<existing.distance) {
                    unique.set(index,{index,distance});
                }
            }
        }

        const ranked=[...unique.values()]
            .sort((a,b)=>a.distance-b.distance)
            .slice(0,4);

        container.replaceChildren();

        if (!ranked.length) {
            container.textContent="No nearby candidate frames.";
            return;
        }

        const note=document.createElement("p");
        note.className="muted";
        note.textContent=
            "Spatial candidates only. Visibility and CAD alignment "+
            "are not verified.";
        container.appendChild(note);

        for (const frame of ranked) {
            const block=document.createElement("div");
            block.className="evidence";

            const label=document.createElement("div");
            label.className="muted";
            label.textContent=
                `Frame ${frame.index} · `+
                `${Number.isFinite(frame.distance)
                    ? frame.distance.toFixed(2)+" m"
                    : "distance unknown"} · Unverified`;

            const image=document.createElement("img");
            image.className="photo";
            image.loading="lazy";
            image.alt=`Construction frame ${frame.index}`;
            image.src=`/api/frame/${frame.index}?mode=raw`;

            block.append(label,image);
            container.appendChild(block);
        }

    } catch (error) {
        if (token===requestGeneration) {
            container.textContent=error.message;
        }
    }
}

const pointerDown={x:0,y:0,button:0};

canvas.addEventListener("pointerdown",event=>{
    pointerDown.x=event.clientX;
    pointerDown.y=event.clientY;
    pointerDown.button=event.button;
});

canvas.addEventListener("pointerup",event=>{
    if (event.button!==0 || pointerDown.button!==0) return;

    if (
        Math.hypot(
            event.clientX-pointerDown.x,
            event.clientY-pointerDown.y
        )>5
    ) return;

    const rect=canvas.getBoundingClientRect();

    pointer.x=((event.clientX-rect.left)/rect.width)*2-1;
    pointer.y=-((event.clientY-rect.top)/rect.height)*2+1;

    raycaster.setFromCamera(pointer,camera);

    const candidates=[
        ...(windowGroup.visible ? windowMeshes : []),
        ...(doorGroup.visible ? doorMeshes : []),
        ...wallMeshes
    ];

    const hits=raycaster.intersectObjects(candidates,false);

    if (hits.length) {
        const {kind,item}=hits[0].object.userData;
        selectElement(kind,elementId(item));
    }
});

$("search").addEventListener("input",renderList);

$("showDoors").addEventListener("change",event=>{
    doorGroup.visible=event.target.checked;

    if (
        !event.target.checked &&
        selectedMesh?.userData.kind==="door"
    ) {
        restoreMaterial(selectedMesh);
        selectedMesh=null;
    }

    renderList();
});

$("showWindows").addEventListener("change",event=>{
    windowGroup.visible=event.target.checked;
    if(!event.target.checked &&
       selectedMesh?.userData.kind==="window") {
        restoreMaterial(selectedMesh);
        selectedMesh=null;
    }
    renderList();
});

$("reset").addEventListener("click",()=>fitCamera(false));
$("top").addEventListener("click",()=>fitCamera(true));

async function fetchGeometry(url) {
    const response=await fetch(url);
    const data=await response.json();

    if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}`);
    }

    const items=data.surfaces || data.elements;

    if (!Array.isArray(items)) {
        throw new Error("Geometry response missing surfaces/elements");
    }

    return {data,items};
}

async function initialize() {
    const sources=[
        ["/api/twin/footprints","Architectural footprints"],
        ["/api/twin/cleaned","Cleaned centerlines"],
        ["/api/twin/elements","Original CAD centerlines"]
    ];

    let result=null;
    let sourceLabel="";

    for (const [url,label] of sources) {
        try {
            result=await fetchGeometry(url);
            if (!result.items.length) throw new Error("Empty geometry");
            sourceLabel=label;
            break;
        } catch (error) {
            console.warn(`${label} unavailable:`,error);
        }
    }

    if (!result) {
        $("count").textContent="Wall geometry unavailable";
        $("renderStatus").textContent="No usable geometry endpoint";
        return;
    }

    try {
        surfaces=result.items;
        buildWalls(surfaces);

        $("count").textContent=
            `${result.data.original_element_count ?? surfaces.length} `+
            "CAD wall elements";

        $("geometrySource").textContent=
            `${wallMeshes.length} rendered wall surfaces · ${sourceLabel}`;

    } catch (error) {
        console.error(error);
        $("renderStatus").textContent=error.message;
        return;
    }

    try {
        const response=await fetch("/api/twin/doors");

        if (!response.ok) {
            throw new Error(`Door API returned ${response.status}`);
        }

        const data=await response.json();
        doors=Array.isArray(data.doors) ? data.doors : [];

        buildDoors(doors);

        $("doorCount").textContent=
            `${doorMeshes.length} provisional leaf spans · `+
            "0 verified wall openings";

    } catch (error) {
        console.warn("Door geometry unavailable:",error);

        $("doorCount").textContent=
            "Door geometry unavailable; walls remain visible";
    }

    try {
        const response=await fetch("/api/twin/windows");
        if(!response.ok) {
            throw new Error(`Window API returned ${response.status}`);
        }

        const data=await response.json();
        const raw=Array.isArray(data.windows) ? data.windows : [];
        windows=consolidateWindows(raw);
        buildWindows(windows);

        $("windowCount").textContent=
            `${windowMeshes.length} consolidated window candidates · `+
            `${raw.length} source groups · 0 verified openings`;
    } catch(error) {
        console.warn("Window geometry unavailable:",error);
        $("windowCount").textContent=
            "Window geometry unavailable; walls and doors remain visible";
    }

    renderList();

    $("renderStatus").textContent=
        `${wallMeshes.length} wall meshes · `+
        `${doorMeshes.length} provisional door meshes · `+
        `${windowMeshes.length} provisional window meshes`;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene,camera);
}

resize();
animate();
initialize();



// UNDER_CONSTRUCTION_FACADE_INTEGRATION_V2

const facadeGroup = new THREE.Group();
facadeGroup.name = "Provisional curtain-wall framing";
scene.add(facadeGroup);

let facadePaths = [];
let facadeVisible = false;
let facadeLoadStarted = false;


function buildFacadeControl() {
    if (document.getElementById("showFacade")) {
        return;
    }

    const windowToggle = document.getElementById(
        "showWindows"
    );

    if (!windowToggle) {
        console.warn(
            "Facade toggle unavailable: window toggle missing."
        );
        return;
    }

    const parent = windowToggle.closest("label")
        || windowToggle.parentElement;

    if (!parent || !parent.parentElement) {
        console.warn(
            "Facade toggle unavailable: no suitable parent."
        );
        return;
    }

    const control = document.createElement("label");

    control.style.display = "flex";
    control.style.alignItems = "center";
    control.style.gap = "8px";
    control.style.marginTop = "8px";
    control.style.cursor = "pointer";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "showFacade";
    checkbox.checked = false;

    const label = document.createElement("span");
    label.textContent = "Facade framing (provisional)";

    control.appendChild(checkbox);
    control.appendChild(label);

    parent.parentElement.insertBefore(
        control,
        parent.nextSibling
    );

    checkbox.addEventListener("change", () => {
        facadeVisible = checkbox.checked;
        facadeGroup.visible = facadeVisible;
    });
}


function clearFacadeGeometry() {
    const materials = new Set();

    while (facadeGroup.children.length > 0) {
        const child = facadeGroup.children[0];

        facadeGroup.remove(child);

        if (child.geometry) {
            child.geometry.dispose();
        }

        if (child.material) {
            if (Array.isArray(child.material)) {
                child.material.forEach(
                    material => materials.add(material)
                );
            } else {
                materials.add(child.material);
            }
        }
    }

    materials.forEach(material => material.dispose());
}


function buildFacadeGeometry(paths) {
    clearFacadeGeometry();

    facadePaths = paths;

    // This is an assumed visualization elevation.
    // It is NOT a verified CAD height.
    const visualizationHeight = 2.4;

    const material = new THREE.LineBasicMaterial({
        color: 0x9b87f5,
        transparent: true,
        opacity: 0.9,
        depthTest: true
    });

    let rendered = 0;

    for (const path of paths) {
        if (
            !Array.isArray(path.points)
            || path.points.length < 2
        ) {
            continue;
        }

        const points = [];

        for (const point of path.points) {
            if (
                !Array.isArray(point)
                || point.length < 2
            ) {
                continue;
            }

            const x = Number(point[0]);
            const y = Number(point[1]);

            if (
                !Number.isFinite(x)
                || !Number.isFinite(y)
            ) {
                continue;
            }

            points.push(
                new THREE.Vector3(
                    x - modelCenter.x,
                    visualizationHeight,
                    y - modelCenter.z
                )
            );
        }

        if (points.length < 2) {
            continue;
        }

        const geometry = new THREE.BufferGeometry()
            .setFromPoints(points);

        const line = new THREE.Line(
            geometry,
            material
        );

        line.userData = {
            id: path.id,
            type: "PROVISIONAL_FACADE",
            sourceHandle: path.source_handle,
            sourceLayer: path.source_layer,
            constructionStatus: "UNKNOWN",
            verticalDimensionsVerified: false,
            physicalComponentVerified: false
        };

        facadeGroup.add(line);
        rendered++;
    }

    facadeGroup.visible = facadeVisible;

    console.log(
        "Provisional facade paths rendered:",
        rendered
    );
}


async function loadFacadeGeometry() {
    if (facadeLoadStarted) {
        return;
    }

    facadeLoadStarted = true;

    try {
        const response = await fetch(
            "/api/twin/facade"
        );

        if (!response.ok) {
            throw new Error(
                "Facade API returned HTTP "
                + response.status
            );
        }

        const data = await response.json();

        buildFacadeGeometry(
            data.paths || []
        );

        console.log(
            "Facade API path count:",
            data.count
        );

    } catch (error) {
        facadeLoadStarted = false;

        console.warn(
            "Facade visualization unavailable:",
            error
        );
    }
}


function waitForFacadeViewerReady() {
    buildFacadeControl();

    let attempts = 0;
    const maxAttempts = 100;

    const interval = setInterval(() => {
        attempts++;

        // The wall model establishes modelCenter.
        // Checking that walls have been rendered avoids
        // loading facade coordinates prematurely.
        const centerReady = (
            typeof modelCenter !== "undefined"
            && modelCenter !== null
            && Number.isFinite(modelCenter.x)
            && Number.isFinite(modelCenter.y)
            && Number.isFinite(modelCenter.z)
        );

        const wallsReady = (
            typeof wallMeshes !== "undefined"
            && Array.isArray(wallMeshes)
            && wallMeshes.length > 0
        );

        if (centerReady && wallsReady) {
            clearInterval(interval);
            loadFacadeGeometry();
            return;
        }

        if (attempts >= maxAttempts) {
            clearInterval(interval);

            console.warn(
                "Facade loading skipped: "
                + "viewer geometry was not ready."
            );
        }
    }, 100);
}


window.addEventListener(
    "load",
    waitForFacadeViewerReady
);




// UNDER_CONSTRUCTION_FACADE_PANELS_START

const provisionalFacadePanels = new THREE.Group();
provisionalFacadePanels.name = "Provisional CAD Facade Panels";
// Source run alignment is not validated: hide this overlay by default.
provisionalFacadePanels.visible = false;
scene.add(provisionalFacadePanels);

const FACADE_PANEL_HEIGHT_ASSUMED_M = 3.0;
const FACADE_PANEL_THICKNESS_VISUAL_M = 0.035;

const facadePanelMaterial = new THREE.MeshBasicMaterial({
    color: 0x67d6d0,
    transparent: true,
    opacity: 0.23,
    side: THREE.DoubleSide,
    depthWrite: false
});

const facadePanelEdgesMaterial = new THREE.LineBasicMaterial({
    color: 0x17a8a4,
    transparent: true,
    opacity: 0.75
});

function buildProvisionalFacadePanel(segment) {
    if (!segment || !segment.start || !segment.end) {
        return null;
    }

    const x1 = Number(segment.start[0]);
    const z1 = Number(segment.start[1]);
    const x2 = Number(segment.end[0]);
    const z2 = Number(segment.end[1]);

    if (![x1, z1, x2, z2].every(Number.isFinite)) {
        return null;
    }

    const dx = x2 - x1;
    const dz = z2 - z1;
    const length = Math.hypot(dx, dz);

    if (length < 0.20) {
        return null;
    }

    const geometry = new THREE.BoxGeometry(
        length,
        FACADE_PANEL_HEIGHT_ASSUMED_M,
        FACADE_PANEL_THICKNESS_VISUAL_M
    );

    const mesh = new THREE.Mesh(
        geometry,
        facadePanelMaterial
    );

    mesh.position.set(
        (x1 + x2) / 2 - modelCenter.x,
        FACADE_PANEL_HEIGHT_ASSUMED_M / 2,
        (z1 + z2) / 2 - modelCenter.z
    );

    mesh.rotation.y = -Math.atan2(dz, dx);

    mesh.userData = {
        type: "PROVISIONAL_FACADE_PANEL",
        sourceSegmentId: segment.source_segment_id,
        sourceHandle: segment.source_handle,
        constructionStatus: "UNKNOWN",
        heightVerified: false,
        openingsVerified: false
    };

    const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        facadePanelEdgesMaterial
    );

    mesh.add(edges);

    return mesh;
}

async function loadProvisionalFacadePanels() {
    try {
        const response = await fetch(
            "/api/twin/facade/panels"
        );

        if (!response.ok) {
            throw new Error(
                "Façade panel API returned " + response.status
            );
        }

        const data = await response.json();

        if (!Array.isArray(data.segments)) {
            throw new Error(
                "Façade panel response has no segments"
            );
        }

        for (const segment of data.segments) {
            const panel = buildProvisionalFacadePanel(
                segment
            );

            if (panel) {
                provisionalFacadePanels.add(panel);
            }
        }

        console.log(
            "Provisional CAD façade panels:",
            provisionalFacadePanels.children.length
        );

    } catch (error) {
        console.warn(
            "Unable to load provisional façade panels:",
            error
        );
    }
}

function installFacadePanelToggle() {
    const existing = document.getElementById(
        "showFacadePanels"
    );

    if (existing) {
        return;
    }

    const label = document.createElement("label");

    label.style.display = "block";
    label.style.marginTop = "8px";
    label.style.cursor = "pointer";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "showFacadePanels";
    checkbox.checked = false;

    checkbox.addEventListener("change", () => {
        provisionalFacadePanels.visible =
            checkbox.checked;
    });

    label.appendChild(checkbox);
    label.appendChild(
        document.createTextNode(
            " Unaligned façade source runs (review only; OFF by default)"
        )
    );

    const windowsToggle = document.getElementById(
        "showWindows"
    );

    if (windowsToggle && windowsToggle.parentElement) {
        windowsToggle.parentElement.insertAdjacentElement(
            "afterend",
            label
        );
    } else {
        document.body.appendChild(label);
    }
}

function initializeProvisionalFacadePanels() {
    installFacadePanelToggle();
    // Wait until wall geometry establishes the shared CAD origin.
    // Starting the request before initialize() completes previously
    // allowed façade meshes to be built around the origin (0, 0, 0).
    let attempts = 0;
    const ready = setInterval(() => {
        if (wallMeshes.length > 0) {
            clearInterval(ready);
            loadProvisionalFacadePanels();
        } else if (++attempts >= 100) {
            clearInterval(ready);
            console.warn("Facade panels skipped: wall geometry unavailable");
        }
    }, 100);
}

initializeProvisionalFacadePanels();

// UNDER_CONSTRUCTION_FACADE_PANELS_END

</script>
</body>
</html>
"""


# UNDER_CONSTRUCTION_FACADE_PANELS_START

@app.route("/api/twin/facade/panels")
def twin_facade_panels_api():
    """CAD-supported provisional façade panel segments."""
    from flask import jsonify

    import json
    import math

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/cad_facade/facade_runs.json"

    if not source.exists():
        return jsonify({
            "error": "FACADE_RUNS_NOT_FOUND",
            "segments": [],
        }), 404

    data = json.loads(
        source.read_text(encoding="utf-8")
    )

    if data.get("method") != (
        "CURTAIN_WALL_COLLINEAR_SOURCE_SEGMENTS"
    ):
        return jsonify({
            "error": "UNEXPECTED_FACADE_RUN_METHOD",
            "segments": [],
        }), 409

    segments = []
    seen = set()

    for run in data.get("runs", []):
        for item in run.get("source_segments", []):
            a = item.get("start")
            b = item.get("end")

            if not (
                isinstance(a, list)
                and isinstance(b, list)
                and len(a) >= 2
                and len(b) >= 2
            ):
                continue

            try:
                x1, y1 = float(a[0]), float(a[1])
                x2, y2 = float(b[0]), float(b[1])
            except (ValueError, TypeError):
                continue

            if not all(
                math.isfinite(v)
                for v in (x1, y1, x2, y2)
            ):
                continue

            length = math.hypot(x2 - x1, y2 - y1)

            if length < 0.20:
                continue

            key = tuple(sorted((
                (round(x1, 5), round(y1, 5)),
                (round(x2, 5), round(y2, 5)),
            )))

            if key in seen:
                continue

            seen.add(key)

            segments.append({
                "start": [x1, y1],
                "end": [x2, y2],
                "length_m": round(length, 6),
                "source_segment_id": item.get(
                    "segment_id"
                ),
                "source_handle": item.get(
                    "source_handle"
                ),
            })

    return jsonify({
        "representation":
            "PROVISIONAL_CAD_SUPPORTED_FACADE_PANELS",
        "coordinate_system": "DXF_METERS",
        "assumed_panel_height_m": 3.0,
        "height_verified": False,
        "physical_facade_verified": False,
        "wall_cutouts_generated": False,
        "construction_status": "UNKNOWN",
        "count": len(segments),
        "segments": segments,
    })

# UNDER_CONSTRUCTION_FACADE_PANELS_END
