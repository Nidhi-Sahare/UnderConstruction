
"""
Under-construction — Step 109
Spatial Studio: Dark Edition

Run backend:
    .\\.venv\\Scripts\\python.exe src\\cad_progress_api.py

Run dashboard:
    .\\.venv\\Scripts\\python.exe src\\construction_dashboard.py

Open:
    http://127.0.0.1:5052

Notes:
- Sample 01 uses the real Step 107 CAD and walkthrough API.
- Sample 02 is a placeholder until a second dataset exists.
- Local uploads are previewed in the browser, not processed.
- No CAD/video registration or construction verification is claimed.
"""

import os
import urllib.error
import urllib.request

from flask import Flask, Response, request

app = Flask(__name__)
from clean_verification_labels import install_clean_verification_labels
install_clean_verification_labels(app)
from wall_recovery_preview import wall_recovery_bp

app.register_blueprint(wall_recovery_bp)
from uploaded_building_api import upload_blueprint

app.register_blueprint(upload_blueprint)

from sample_three_storey import sample_building_bp
app.register_blueprint(sample_building_bp)

from sample_floor_inspection import floor_inspection_bp
app.register_blueprint(floor_inspection_bp)

from registration_review_panel import registration_review_bp
app.register_blueprint(registration_review_bp)

from digital_twin_inspector import digital_twin_inspector_bp
app.register_blueprint(digital_twin_inspector_bp)

from floor_walkthrough_upgrade import bp as floor_walkthrough_bp
app.register_blueprint(floor_walkthrough_bp)


PORT = int(os.environ.get("DASHBOARD_PORT", "5052"))
BACKEND = os.environ.get(
    "CAD_PROGRESS_API",
    "http://127.0.0.1:5051",
).rstrip("/")


@app.get("/api/<path:path>")
def proxy_api(path):
    """Forward read-only requests to the Step 107 backend."""
    if not (
        path == "project"
        or path == "cad/elements"
        or path.startswith("walkthrough/")
    ):
        return {"error": "Endpoint not permitted"}, 404

    target = BACKEND + "/api/" + path

    if request.query_string:
        target += "?" + request.query_string.decode("ascii")

    try:
        with urllib.request.urlopen(target, timeout=25) as upstream:
            content = upstream.read()
            content_type = upstream.headers.get(
                "Content-Type", "application/octet-stream"
            )
            return Response(content, content_type=content_type)

    except urllib.error.HTTPError as exc:
        return Response(
            exc.read(),
            status=exc.code,
            content_type=exc.headers.get(
                "Content-Type", "application/json"
            ),
        )

    except Exception as exc:
        return {
            "error": "Backend unavailable",
            "detail": str(exc),
        }, 502



# STEP_111_LOCAL_DXF
@app.post("/api/local/dxf")
def parse_local_dxf():
    """Parse an uploaded DXF for visualization only.

    This does not register CAD to video or infer construction progress.
    """
    import math
    import tempfile
    from pathlib import Path

    try:
        import ezdxf
    except ImportError:
        return {
            "error": "ezdxf is not installed. Run: pip install ezdxf"
        }, 503

    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return {"error": "No DXF file provided."}, 400

    if not uploaded.filename.lower().endswith(".dxf"):
        return {"error": "Only .dxf files are accepted."}, 400

    # Avoid loading arbitrarily large files.
    MAX_BYTES = 20 * 1024 * 1024
    payload = uploaded.stream.read(MAX_BYTES + 1)

    if len(payload) > MAX_BYTES:
        return {"error": "DXF exceeds the 20 MB limit."}, 413

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".dxf", delete=False
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)

        drawing = ezdxf.readfile(str(temporary_path))
        modelspace = drawing.modelspace()

        elements = []
        skipped = {}
        MAX_ENTITIES = 20000

        def add_polyline(entity, points, closed=False):
            if len(elements) >= MAX_ENTITIES:
                return

            cleaned = []

            for point in points:
                x = float(point[0])
                y = float(point[1])

                if not math.isfinite(x) or not math.isfinite(y):
                    return

                cleaned.append([x, y])

            if len(cleaned) < 2:
                return

            entity_type = entity.dxftype()

            elements.append({
                "element_id": (
                    f"LOCAL-{entity.dxf.handle}"
                    if entity.dxf.hasattr("handle")
                    else f"LOCAL-{len(elements) + 1}"
                ),
                "layer": entity.dxf.get("layer", "0"),
                "source_entity_type": entity_type,
                "geometry": {
                    "type": "polyline",
                    "coordinates": cleaned,
                    "closed": bool(closed),
                },
                "verification": "unverified",
                "source": "local_dxf",
            })

        def sample_arc(cx, cy, radius, start, end):
            if radius <= 0 or not math.isfinite(radius):
                return []

            sweep = end - start
            if sweep <= 0:
                sweep += 360

            steps = max(8, min(128, math.ceil(sweep / 5)))

            return [
                [
                    cx + radius * math.cos(
                        math.radians(start + sweep * i / steps)
                    ),
                    cy + radius * math.sin(
                        math.radians(start + sweep * i / steps)
                    ),
                ]
                for i in range(steps + 1)
            ]

        for entity in modelspace:
            if len(elements) >= MAX_ENTITIES:
                break

            kind = entity.dxftype()

            try:
                if kind == "LINE":
                    add_polyline(
                        entity,
                        [entity.dxf.start, entity.dxf.end],
                    )

                elif kind == "LWPOLYLINE":
                    # Bulged segments are not converted here.
                    # Skip them rather than drawing false straight walls.
                    vertices = list(entity.get_points("xyb"))

                    if any(abs(v[2]) > 1e-10 for v in vertices):
                        skipped["LWPOLYLINE_BULGE"] = (
                            skipped.get("LWPOLYLINE_BULGE", 0) + 1
                        )
                        continue

                    add_polyline(
                        entity,
                        [(v[0], v[1]) for v in vertices],
                        entity.closed,
                    )

                elif kind == "POLYLINE":
                    if not entity.is_2d_polyline:
                        skipped[kind] = skipped.get(kind, 0) + 1
                        continue

                    vertices = list(entity.vertices)

                    if any(
                        abs(float(v.dxf.get("bulge", 0))) > 1e-10
                        for v in vertices
                    ):
                        skipped["POLYLINE_BULGE"] = (
                            skipped.get("POLYLINE_BULGE", 0) + 1
                        )
                        continue

                    add_polyline(
                        entity,
                        [v.dxf.location for v in vertices],
                        entity.is_closed,
                    )

                elif kind == "CIRCLE":
                    center = entity.dxf.center
                    radius = float(entity.dxf.radius)

                    add_polyline(
                        entity,
                        sample_arc(
                            center.x, center.y, radius, 0, 360
                        )[:-1],
                        closed=True,
                    )

                elif kind == "ARC":
                    center = entity.dxf.center

                    add_polyline(
                        entity,
                        sample_arc(
                            center.x,
                            center.y,
                            float(entity.dxf.radius),
                            float(entity.dxf.start_angle),
                            float(entity.dxf.end_angle),
                        ),
                    )

                else:
                    skipped[kind] = skipped.get(kind, 0) + 1

            except (ValueError, TypeError, AttributeError, OverflowError):
                skipped[kind] = skipped.get(kind, 0) + 1

        return {
            "elements": elements,
            "count": len(elements),
            "skipped": skipped,
            "truncated": len(elements) >= MAX_ENTITIES,
            "source": "local_dxf",
            "verified": False,
            "registered_to_video": False,
        }

    except Exception as exc:
        return {
            "error": "Could not parse this DXF.",
            "detail": str(exc)[:300],
        }, 422

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)




# Reuse the original digital_twin_3d.py HTML/Three.js implementation,
# without importing its Flask app or showing its separate navigation.
@app.get("/embedded-twin")
def embedded_twin():
    import ast
    from pathlib import Path
    source = Path(__file__).with_name("digital_twin_3d.py")
    if not source.is_file():
        return Response("Original src/digital_twin_3d.py not found.", status=404)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        template = next(
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "HTML" for t in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    except (OSError, SyntaxError, StopIteration) as exc:
        return Response("Unable to load original 3D twin: " + str(exc), status=500)
    theme = """<style>
    :root{--bg:#101117!important;--panel:#191a22!important;--border:#343542!important;
    --text:#f0eef7!important;--muted:#9898aa!important;--accent:#b89aff!important;
    --blue:#f0784b!important}
    header{display:none!important}
    .layout{height:100vh!important;min-height:500px!important}
    .viewport{min-width:0}
    </style>"""
    template = template.replace('id="showDoors" checked', 'id="showDoors"')
    template = template.replace('id="showWindows" checked', 'id="showWindows"')
    theme += """<style>.legend{max-width:220px;font-size:10px!important}
    .viewport{min-height:650px}</style>"""
    template = template.replace("</head>", theme + "</head>", 1)
    template = template.replace(
        "Select a wall to retrieve nearby frames.",
        "Floor 2 geometry is sourced from the current DXF. Earlier twin evidence is not registered to this floor."
    )
    # Remove the actual source HTML block, including nested markup.
    # This is performed on the template returned by /embedded-twin,
    # not on the parent dashboard HTML.
    start = template.find('<h3>Visual Evidence Candidates</h3>')
    end = template.find('<div class="warning">', start)
    if start < 0 or end < 0:
        return Response("Unexpected 3D inspector template: evidence section not found", status=500)
    template = template[:start] + template[end:]

    # Disable the obsolete lookup itself, not merely its visible heading.
    # Keep a no-op function so element selection remains functional.
    template = template.replace(
        'async function loadEvidence(sourceIds) {',
        'async function loadEvidence(sourceIds) { return; // Disabled for Floor 2',
        1,
    )
    # Step 132: presentation-only changes in the embedded 3D viewer.
    # Keep the original dashboard hero, logo, CAD and walkthrough intact.
    polish = """<style>
      .layout {grid-template-columns: 250px minmax(0,1fr) 270px!important}
      .sidebar,.inspector {overflow-y:auto!important}
      .warning {display:none!important}
      .viewport {min-width:0!important}
      @media(max-width:1100px){
        .layout {grid-template-columns:210px minmax(0,1fr) 225px!important}
      }
    </style>"""
    template = template.replace("</head>", polish + "</head>", 1)
    template = template.replace(
        "INTERACTIVE 3D / ORIGINAL DIGITAL TWIN",
        "FLOOR 2 / INTERACTIVE 3D MODEL"
    )
    template = template.replace(
        "Show provisional door geometry", "Show door CAD references"
    ).replace(
        "Show provisional window candidates", "Show window CAD references"
    )
    template = template.replace(
        "Architectural Elements", "Building Elements"
    ).replace(
        "Geometric screening", "CAD classification"
    ).replace(
        "Construction status", "Site progress"
    )
    # The optional facade overlays are not part of the Floor 2 model.
    # Remove their injected toggle labels when created, without touching
    # the underlying source module.
    template = template.replace(
        'checkbox.checked = true;',
        'checkbox.checked = false;'
    )
    template = template.replace(
        'checkbox.checked=true;',
        'checkbox.checked=false;'
    )
    # Remove redundant "unverified" copy only from the inspector UI.
    # Technical provenance is preserved in API responses.
    template = template.replace(
        'class="warning"', 'class="warning"'
    )
    # Avoid cached copies of the old embedded viewer.
    response = Response(template, content_type="text/html; charset=utf-8")
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response

# STEP 128: All displayed architectural geometry is derived from the
# SAME Floor 2 DXF used by the 2D CAD viewer. No legacy twin geometry.
from functools import lru_cache
from flask import jsonify

@lru_cache(maxsize=1)
def floor2_twin_geometry():
    import ezdxf
    import math
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = root / "data" / "floorplans" / "floor_2.dxf"
    if not source.is_file():
        raise FileNotFoundError(str(source))

    drawing = ezdxf.readfile(str(source))
    walls = []
    doors = []
    windows = []

    def segments(entity):
        kind = entity.dxftype()
        if kind == "LINE":
            return [(entity.dxf.start, entity.dxf.end)]
        if kind == "LWPOLYLINE":
            vertices = list(entity.get_points("xyb"))
            if any(abs(float(v[2])) > 1e-9 for v in vertices):
                return []  # do not flatten curved CAD geometry
            coords = [(float(v[0]), float(v[1])) for v in vertices]
            if entity.closed and len(coords) > 2:
                coords.append(coords[0])
            return list(zip(coords, coords[1:]))
        return []

    for entity in drawing.modelspace():
        layer = entity.dxf.get("layer", "").lower()
        category = (
            "exterior" if "aussenw" in layer else
            "interior" if "innenw" in layer else
            "partition" if "trennw" in layer else
            "lining" if "vormauer" in layer else None
        )
        is_door = "türen" in layer or "tueren" in layer
        is_window = "fenster" in layer
        if not (category or is_door or is_window):
            continue
        handle = str(entity.dxf.get("handle", "unknown"))
        for index, (a, b) in enumerate(segments(entity)):
            coords = [float(a[0]), float(a[1]), float(b[0]), float(b[1])]
            if not all(math.isfinite(v) for v in coords):
                continue
            x1, y1, x2, y2 = coords
            length = math.hypot(x2-x1, y2-y1)
            if length < 0.05 or length > 150:
                continue
            item = {
                "id": f"CAD_{handle}_{index}",
                "start": [x1,y1], "end": [x2,y2],
                "source_element_ids": [f"CAD_{handle}"],
                "source_handles": [handle],
                "source_layer": entity.dxf.layer,
                "construction_status": "UNKNOWN",
            }
            if category:
                walls.append({
                    **item, "category": category, "length_m": length,
                    "height_m": 3.0, "thickness_m": 0.15,
                    "progress_verified": False,
                })
            elif is_door:
                doors.append({
                    **item, "length_m": length,
                    "status": "INSUFFICIENT_WALL_EVIDENCE",
                    "source_arc_handles": [handle],
                    "source_line_handles": [handle],
                    "opening_verified": False, "safe_to_cut": False,
                })
            elif is_window:
                windows.append({
                    **item, "span_m": length,
                    "status": "PROVISIONAL_GEOMETRY",
                    "component_count": 1,
                    "source_group_ids": [f"CAD_{handle}_{index}"],
                    "opening_verified": False, "safe_to_cut": False,
                })
    return {"walls": walls, "doors": doors, "windows": windows, "source": str(source)}

@app.get("/api/twin/footprints")
@app.get("/api/twin/cleaned")
@app.get("/api/twin/elements")
def floor2_twin_walls():
    """Use the same in-process CAD reconstruction as the working /twin viewer.

    This does NOT make an HTTP request to port 5000. The architectural_geometry
    module is imported as a Python library and its reconstruction is returned
    directly by this dashboard process.
    """
    try:
        from architectural_geometry import get_reconstruction
        data = get_reconstruction()
        if not data.get("surfaces"):
            return jsonify({"error": "CAD reconstruction returned no wall surfaces"}), 503
        return jsonify(data)
    except Exception as exc:
        app.logger.exception("Working CAD wall reconstruction unavailable")
        return jsonify({
            "error": "Could not load the same reconstruction as the working twin",
            "detail": str(exc),
        }), 503


@app.get("/api/twin/doors")
def floor2_twin_doors():
    try:
        data = floor2_twin_geometry()
        return jsonify({"doors": data["doors"], "count": len(data["doors"]),
                        "status": "CAD_LAYER_SEGMENTS_NOT_VERIFIED_DOOR_LEAVES",
                        "wall_cutouts_generated": False})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

@app.get("/api/twin/windows")
def floor2_twin_windows():
    try:
        data = floor2_twin_geometry()
        return jsonify({"windows": data["windows"], "count": len(data["windows"]),
                        "status": "CAD_LAYER_SEGMENTS_NOT_VERIFIED_WINDOWS",
                        "wall_cutouts_generated": False})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

@app.get("/api/twin/facade")
@app.get("/api/twin/facade/panels")
def floor2_twin_facade():
    """
    Serve provisional Floor 2 facade geometry locally.

    Source:
        data/floorplans/floor_2.dxf
        outputs/cad_facade/facade_geometry.json
        outputs/cad_facade/facade_runs.json

    This endpoint is visualization-only.
    It does not verify physical facade components,
    construction progress, heights, or wall openings.
    """
    import json
    import math
    from pathlib import Path
    from flask import jsonify

    root = Path(__file__).resolve().parents[1]

    if request.path.endswith("/panels"):
        source = root / "outputs/cad_facade/facade_runs.json"

        if not source.is_file():
            return jsonify({
                "error": "FACADE_RUNS_NOT_FOUND",
                "count": 0,
                "segments": [],
            }), 404

        data = json.loads(source.read_text(encoding="utf-8"))

        if data.get("method") != "CURTAIN_WALL_COLLINEAR_SOURCE_SEGMENTS":
            return jsonify({
                "error": "UNEXPECTED_FACADE_RUN_METHOD",
                "count": 0,
                "segments": [],
            }), 409

        segments = []
        seen = set()

        for run in data.get("runs", []):
            for item in run.get("source_segments", []):
                a = item.get("start")
                b = item.get("end")

                if not (
                    isinstance(a, (list, tuple))
                    and isinstance(b, (list, tuple))
                    and len(a) >= 2
                    and len(b) >= 2
                ):
                    continue

                try:
                    x1, y1 = float(a[0]), float(a[1])
                    x2, y2 = float(b[0]), float(b[1])
                except (TypeError, ValueError):
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
                    "source_segment_id": item.get("segment_id"),
                    "source_handle": item.get("source_handle"),
                })

        return jsonify({
            "representation": "PROVISIONAL_CAD_SUPPORTED_FACADE_PANELS",
            "coordinate_system": "DXF_METERS",
            "assumed_panel_height_m": 3.0,
            "height_verified": False,
            "physical_facade_verified": False,
            "wall_cutouts_generated": False,
            "construction_status": "UNKNOWN",
            "count": len(segments),
            "segments": segments,
        })

    source = root / "outputs/cad_facade/facade_geometry.json"

    if not source.is_file():
        return jsonify({
            "error": "FACADE_GEOMETRY_NOT_FOUND",
            "count": 0,
            "paths": [],
        }), 404

    data = json.loads(source.read_text(encoding="utf-8"))

    paths = []

    for item in data.get("paths", []):
        if item.get("category") != "CURTAIN_WALL":
            continue

        points = item.get("points", [])

        if not isinstance(points, list) or len(points) < 2:
            continue

        valid = True

        for point in points:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                valid = False
                break

            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                valid = False
                break

            if not (math.isfinite(x) and math.isfinite(y)):
                valid = False
                break

        if not valid:
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
# All 3D geometry routes are served by THIS dashboard Flask app.
# Unknown twin routes must fail locally, never proxy to port 5000.
@app.get("/api/twin/<path:subpath>")
def unknown_twin_route(subpath):
    return jsonify({"error": "3D route not implemented in the integrated dashboard",
                    "path": subpath}), 404

@app.get("/api/spatial-evidence/<path:subpath>")
@app.get("/api/evidence/<path:subpath>")
@app.get("/api/frame/<path:subpath>")
def unavailable_legacy_evidence(subpath):
    return jsonify({
        "error": "Legacy evidence is not registered to Sample 1 Floor 2."
    }), 404


HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>under-construction — Spatial Studio</title>
<style>
:root{
  --bg:#101117;
  --surface:#191a22;
  --surface2:#20212b;
  --surface3:#282936;
  --line:#343542;
  --ink:#f0eef7;
  --muted:#9898aa;
  --violet:#b89aff;
  --orange:#f0784b;
  --green:#80cbb0;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font-family:Inter,"Segoe UI",Arial,sans-serif
}
button,input{font:inherit}
button{cursor:pointer}
button:focus-visible,input:focus-visible{
  outline:2px solid var(--violet);outline-offset:3px
}
button:disabled{opacity:.4;cursor:not-allowed}
a{color:inherit;text-decoration:none}
.hidden{display:none!important}
.mono{
  font-family:"Cascadia Code",Consolas,monospace;
  letter-spacing:.05em
}
.eyebrow{
  font-size:11px;letter-spacing:.17em;
  text-transform:uppercase;color:var(--violet)
}
.topbar{
  height:72px;display:flex;align-items:center;
  justify-content:space-between;padding:0 5%;
  border-bottom:1px solid #292a34
}
.brand{display:flex;align-items:center;gap:12px;font-weight:750}
.brand-mark{
  width:36px;height:36px;border:2px dashed var(--violet);
  border-radius:50%;display:grid;place-items:center;
  color:var(--violet);font-size:20px
}
.nav-right{display:flex;gap:20px;align-items:center}
.nav-right a{font-size:13px;color:#c8c5d2}
.primary{
  background:var(--violet);color:#17131f;
  border:0;border-radius:9px;padding:12px 17px;
  font-weight:700
}
.secondary{
  background:transparent;color:var(--ink);
  border:1px solid #646272;border-radius:9px;
  padding:12px 17px
}
.primary:hover{background:#cab4ff}
.secondary:hover{background:#272733}
.hero{
  min-height:580px;
  display:grid;grid-template-columns:40% 60%;
  align-items:center;padding:65px 5% 70px;
  overflow:hidden;
  background:
    radial-gradient(ellipse at 82% 58%,#35232e 0%,transparent 43%),
    radial-gradient(ellipse at 53% 38%,#25213a 0%,transparent 42%);
}
.hero-copy{position:relative;z-index:1}
.hero h1{
  font-size:clamp(49px,5.6vw,89px);
  line-height:1.01;letter-spacing:-.065em;
  font-weight:680;margin:23px 0
}
.hero h1 em{font-style:normal;color:var(--violet)}
.hero p{
  max-width:410px;color:#b9b6c5;
  font-size:16px;line-height:1.75
}
.actions{display:flex;gap:12px;margin-top:28px;flex-wrap:wrap}
.hero-art{
  height:450px;position:relative;
  display:grid;place-items:center;
  perspective:1000px
}
.hero-grid{
  position:absolute;inset:7%;
  transform:rotateX(60deg) rotateZ(-30deg);
  background-image:
    linear-gradient(#6d618733 1px,transparent 1px),
    linear-gradient(90deg,#6d618733 1px,transparent 1px);
  background-size:32px 32px;
  mask-image:radial-gradient(ellipse,#000 35%,transparent 75%)
}
.model{
  position:relative;width:360px;height:320px;
  transform-style:preserve-3d;
  transform:rotateX(54deg) rotateZ(-38deg);
  transition:transform .3s ease-out
}
.floor{
  position:absolute;width:270px;height:195px;
  left:45px;top:50px;
  border:2px solid #c2adff;
  background:linear-gradient(130deg,#9a76e42b,#f0784b18);
  box-shadow:0 18px 42px #0005;
  transform-style:preserve-3d
}
.floor:nth-child(1){transform:translateZ(0px)}
.floor:nth-child(2){transform:translateZ(65px)}
.floor:nth-child(3){transform:translateZ(130px)}
.floor:nth-child(4){transform:translateZ(195px)}
.floor:before{
  content:"";position:absolute;
  left:35%;top:0;height:100%;width:2px;
  background:#b89affad
}
.floor:after{
  content:"";position:absolute;
  top:45%;left:0;width:100%;height:2px;
  background:#f0784b99
}
.wall{
  position:absolute;width:120px;height:65px;
  border:2px solid #bda5ff;
  background:#b89aff25;
  transform-origin:bottom;
  transform:rotateX(-90deg)
}
.wall.one{left:0;bottom:0}
.wall.two{right:0;top:0;border-color:var(--orange)}
.model-label{
  position:absolute;right:3%;top:18%;
  color:var(--violet);font-size:11px
}
.model-caption{
  position:absolute;bottom:0;right:5%;
  color:var(--muted);font-size:11px;
  text-align:right;line-height:1.8
}
.hero-divider{
  border-top:1px solid #34303c;
  padding:14px 5%;display:flex;
  justify-content:space-between;color:var(--muted);
  font-size:11px
}
.workspace{
  min-height:780px;
  display:grid;grid-template-columns:235px minmax(0,1fr);
  border-top:1px solid var(--line)
}
.sidebar{
  background:#15161e;border-right:1px solid var(--line);
  padding:24px 14px;display:flex;flex-direction:column;gap:22px
}
.sidebar-heading{
  color:var(--muted);font-size:10px;
  letter-spacing:.18em;margin:0 10px 12px
}
.project{
  display:flex;align-items:center;gap:11px;
  width:100%;padding:12px;border:1px solid transparent;
  border-radius:9px;background:transparent;
  color:var(--ink);text-align:left;margin-bottom:7px
}
.project.active{
  background:#2a2638;border-color:#514469
}
.project:hover{background:#242530}
.project-thumb{
  width:43px;height:43px;flex:none;
  border:1px solid #7e718e;
  background:
    linear-gradient(90deg,transparent 48%,#b89aff80 49%,#b89aff80 51%,transparent 52%),
    linear-gradient(transparent 47%,#b89aff80 48%,#b89aff80 51%,transparent 52%);
  border-radius:4px
}
.project-name{font-size:12px;font-weight:650}
.project-note{font-size:10px;color:var(--muted);margin-top:5px}
.new-project{
  width:100%;padding:12px;border:1px dashed #706b80;
  background:transparent;color:var(--ink);
  border-radius:8px;text-align:left
}
.sidebar-bottom{
  margin-top:auto;padding:12px;
  border-top:1px solid var(--line);
  font-size:10px;line-height:1.8;color:var(--muted)
}
.main{min-width:0;padding:26px}
.workspace-top{
  display:flex;justify-content:space-between;
  align-items:center;gap:14px;margin-bottom:20px
}
.workspace-title{
  font-size:25px;font-weight:680;letter-spacing:-.04em
}
.status{
  color:#f4a47f;font-size:10px;
  border:1px solid #7b4c40;
  border-radius:5px;padding:7px 9px;
  white-space:nowrap
}
.intro{
  margin:12px 0 24px;
  font-size:14px;line-height:1.7;color:var(--muted)
}
.upload-grid{
  display:grid;grid-template-columns:1fr 1fr;
  gap:16px
}
.dropzone{
  min-height:290px;
  border:1px dashed #665c85;
  border-radius:12px;
  background:
    radial-gradient(circle at 50% 30%,#55407930,transparent 50%),
    #1b1b25;
  display:flex;flex-direction:column;
  justify-content:center;align-items:center;
  padding:28px;text-align:center;
  transition:.2s;cursor:pointer
}
.dropzone.video{border-color:#885744}
.dropzone.dragging,.dropzone:hover{
  border-color:var(--violet);
  background-color:#292437;
  transform:translateY(-2px)
}
.dropzone.video.dragging,.dropzone.video:hover{
  border-color:var(--orange)
}
.drop-icon{font-size:39px;color:var(--violet)}
.dropzone.video .drop-icon{color:var(--orange)}
.drop-title{font-size:18px;font-weight:650;margin-top:20px}
.drop-note{font-size:12px;color:var(--muted);margin-top:10px}
.file-name{
  font-size:11px;color:var(--green);
  margin-top:15px;overflow-wrap:anywhere
}
.upload-actions{margin-top:20px;display:flex;gap:12px;align-items:center}
.note{
  margin-top:22px;color:var(--muted);
  font-size:12px;line-height:1.7
}
.viewer-grid{
  display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);
  gap:12px
}
.viewer{
  border:1px solid var(--line);
  background:var(--surface);border-radius:11px;
  overflow:hidden;min-width:0
}
.viewer-head{
  min-height:54px;padding:12px 15px;
  display:flex;align-items:center;justify-content:space-between;
  gap:10px;border-bottom:1px solid var(--line)
}
.viewer-head strong{font-size:12px}
.viewer-tools{display:flex;gap:6px}
.tool{
  background:#292a35;color:var(--ink);
  border:1px solid #42414e;border-radius:6px;
  padding:7px 10px;font-size:11px
}
.cad-stage{
  position:relative;height:410px;overflow:hidden;
  background-color:#14151d;
  background-image:radial-gradient(#5e5c752e 1px,transparent 1px);
  background-size:18px 18px;
  touch-action:none;cursor:grab
}
.cad-stage:active{cursor:grabbing}
#cad-svg{width:100%;height:100%;display:block}
.cad-line{
  fill:none;stroke:#d5d2df;stroke-width:1.35;
  vector-effect:non-scaling-stroke;
  cursor:pointer;pointer-events:stroke
}
.cad-line:hover{stroke:var(--violet);stroke-width:3}
.cad-line.selected{stroke:var(--orange);stroke-width:3.5}
.viewer-foot{
  min-height:48px;padding:13px 15px;
  border-top:1px solid var(--line);
  font-size:11px;color:var(--muted)
}
.video-stage{
  height:410px;background:#111116;
  display:grid;place-items:center;position:relative
}
.video-stage img,.video-stage video{
  width:100%;height:100%;object-fit:contain
}
.video-empty{
  color:var(--muted);font-size:12px;
  text-align:center;padding:20px;line-height:1.8
}
.video-controls{
  display:flex;align-items:center;gap:10px;
  padding:12px 15px;border-top:1px solid var(--line)
}
.video-controls input{width:100%;accent-color:var(--violet)}
.inspector{
  margin-top:14px;background:var(--surface);
  border:1px solid var(--line);border-radius:10px;
  padding:18px;display:flex;justify-content:space-between;
  align-items:flex-start;gap:18px;flex-wrap:wrap
}
.inspector h3{font-size:14px;margin:0 0 9px}
.inspector p{
  font-size:12px;line-height:1.7;
  color:var(--muted);margin:0
}
.inspector-data{
  display:grid;grid-template-columns:repeat(3,minmax(100px,1fr));
  gap:18px;font-size:12px
}
.inspector-data small{
  display:block;color:var(--muted);
  margin-bottom:7px;font-size:10px
}
.notice{
  border-left:2px solid var(--orange);
  background:#f0784b0b;
  padding:12px 15px;margin-top:16px;
  color:#c3b5b2;font-size:11px;line-height:1.7
}
@media(max-width:1100px){
  .hero{grid-template-columns:1fr}
  .hero-art{height:360px}
  .workspace{grid-template-columns:190px minmax(0,1fr)}
  .viewer-grid{grid-template-columns:1fr}
}
@media(max-width:700px){
  .topbar{padding:0 18px}
  .nav-right a{display:none}
  .hero{padding:45px 20px}
  .hero h1{font-size:52px}
  .model{transform:scale(.8) rotateX(54deg) rotateZ(-38deg)}
  .workspace{grid-template-columns:1fr}
  .sidebar{border-right:0;border-bottom:1px solid var(--line)}
  .main{padding:18px}
  .upload-grid{grid-template-columns:1fr}
  .workspace-top{align-items:flex-start;flex-direction:column}
  .inspector-data{grid-template-columns:1fr 1fr}
}
@media(prefers-reduced-motion:reduce){
  *,*:before,*:after{
    scroll-behavior:auto!important;
    transition:none!important;animation:none!important
  }
}
</style>
</head>
<body>

<header class="topbar">
  <div class="brand">
    <div class="brand-mark" aria-hidden="true"><svg viewBox="0 0 50 40" xmlns="http://www.w3.org/2000/svg" focusable="false"><path fill="currentColor" d="M10 30C10 17 16.5 9 25 9s15 8 15 21z"/><path fill="currentColor" d="M20.5 10.5C20.5 6 22.3 3.5 25 3.5s4.5 2.5 4.5 7z"/><path fill="currentColor" d="M3 29.5h44a2.5 2.5 0 0 1 2.5 2.5v1.5a3 3 0 0 1-3 3h-43a3 3 0 0 1-3-3V32a2.5 2.5 0 0 1 2.5-2.5z"/><path d="M20.5 10.5V30M29.5 10.5V30" stroke="#15141c" stroke-opacity=".45" stroke-width="1.4" fill="none"/><path d="M14 29C14 21 17 15 21 12" stroke="#ffffff" stroke-opacity=".28" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg></div>
    <span>under-construction</span>
  </div>
  <nav class="nav-right">
    <a href="#studio">Workspace</a>
    <a href="#studio" id="nav-samples">Samples</a>
    <a href="#studio" class="primary">Get started →</a>
  </nav>
</header>

<section class="hero">
  <div class="hero-copy">
  <div class="eyebrow mono">
    UNDER-CONSTRUCTION / FOLLOW THE BUILD
  </div>

  <h1>
    Be there<br>
    as it <em>takes shape.</em>
  </h1>

  <p>
    Bring your floor plans and site walkthroughs together.
    Explore the building, follow its progress, and see
    how the plans take shape on site.
  </p>

  <div class="actions">
    <button class="primary" id="hero-sample">
      Explore a sample →
    </button>

    <button class="secondary" id="hero-own">
      Start your project
    </button>
  </div>
</div>

  <div class="hero-art" id="hero-art"
       aria-label="Illustrative layered architectural model">
    <div class="hero-grid"></div>
    <div class="model" id="hero-model">
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
      <div class="floor">
        <div class="wall one"></div>
        <div class="wall two"></div>
      </div>
    </div>
    <div class="model-label mono">ILLUSTRATIVE / SPATIAL STUDY</div>
    <div class="model-caption mono">
      MOVE TO EXPLORE<br>
      A BUILDING, NOT A CLAIM
    </div>
  </div>
</section>

<div class="hero-divider mono">
  <span>01 / THE IDEA</span>
  <span>↓ OPEN THE WORKSPACE</span>
</div>

<section class="workspace" id="studio">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark" aria-hidden="true"><svg viewBox="0 0 50 40" xmlns="http://www.w3.org/2000/svg" focusable="false"><path fill="currentColor" d="M10 30C10 17 16.5 9 25 9s15 8 15 21z"/><path fill="currentColor" d="M20.5 10.5C20.5 6 22.3 3.5 25 3.5s4.5 2.5 4.5 7z"/><path fill="currentColor" d="M3 29.5h44a2.5 2.5 0 0 1 2.5 2.5v1.5a3 3 0 0 1-3 3h-43a3 3 0 0 1-3-3V32a2.5 2.5 0 0 1 2.5-2.5z"/><path d="M20.5 10.5V30M29.5 10.5V30" stroke="#15141c" stroke-opacity=".45" stroke-width="1.4" fill="none"/><path d="M14 29C14 21 17 15 21 12" stroke="#ffffff" stroke-opacity=".28" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg></div>
      <span style="font-size:12px">under-construction</span>
    </div>

    <div>
      <div class="sidebar-heading mono">SAMPLE PROJECTS</div>

      <button class="project" id="sample-one">
        <span class="project-thumb"></span>
        <span>
          <span class="project-name">Floor 2 Walkthrough</span>
          <span class="project-note" style="display:block">
            Available local dataset
          </span>
        </span>
      </button>

      <button class="project" id="sample-two">
        <span class="project-thumb"></span>
        <span>
          <span class="project-name">Sample 02</span>
          <span class="project-note" style="display:block">
            Dataset not installed
          </span>
        </span>
      </button>
    </div>

    <div>
      <div class="sidebar-heading mono">YOUR PROJECTS</div>
      <button class="new-project" id="new-project">
        + New project
      </button>
    </div>

    <div class="sidebar-bottom mono">
      PLANNED ≠ BUILT<br>
      VERIFICATION IS EARNED.
    </div>
  </aside>

  <main class="main">
    <div class="workspace-top">
      <div>
        <div class="eyebrow mono" id="workspace-eyebrow">
          PROJECT / NEW
        </div>
        <div class="workspace-title" id="workspace-title">
          Start with your files.
        </div>
      </div>
      <span class="status mono" id="workspace-status">
        NOT ANALYZED
      </span>
    </div>

    <div id="upload-view">
      <p class="intro">
        Two views of the same place. Drop in your plan and
        walkthrough to begin. We'll keep the difference
        between uploaded, processed, and verified clear.
      </p>

      <div class="upload-grid">
        <div class="dropzone" id="plan-drop"
             role="button" tabindex="0">
          <div class="drop-icon">▱</div>
          <div class="drop-title">Your floor plan.</div>
          <div class="drop-note">
            Drop a DXF file or click to browse.
          </div>
          <div class="file-name" id="plan-file"></div>
        </div>

        <div class="dropzone video" id="video-drop"
             role="button" tabindex="0">
          <div class="drop-icon">▷</div>
          <div class="drop-title">Your walkthrough.</div>
          <div class="drop-note">
            Drop an MP4, MOV, or WebM video.
          </div>
          <div class="file-name" id="video-file"></div>
        </div>
      </div>

      <input type="file" id="plan-input"
             accept=".dxf" class="hidden">
      <input type="file" id="video-input"
             accept=".mp4,.mov,.webm,video/*" class="hidden">

      <div class="upload-actions">
        <button class="primary" id="open-uploads" disabled>
          Preview selected files →
        </button>
        <span class="mono" id="upload-status"
              style="color:var(--muted);font-size:11px">
          WAITING FOR FILES
        </span>
      </div>

      <p class="note">
        Local file selection does not submit your files to
        the analysis backend. DXF parsing and progress
        verification are not performed in this step.
      </p>
    </div>

    <div id="viewer-view" class="hidden">
      <div class="viewer-grid">
        <div class="viewer">
          <div class="viewer-head">
            <strong>01 / Floor plan</strong>
            <div class="viewer-tools">
              <button class="tool" id="zoom-in">+</button>
              <button class="tool" id="zoom-out">−</button>
              <button class="tool" id="reset-cad">Fit</button>
            </div>
          </div>
          <div class="cad-stage" id="cad-stage">
            <svg id="cad-svg" xmlns="http://www.w3.org/2000/svg">
              <g id="cad-lines"></g>
              <g id="uc-align-layer" visibility="hidden"><path id="uc-align-track"/><path id="uc-align-travelled"/><circle id="uc-align-marker" r="2.8" visibility="hidden"/></g>
            </svg>
            <div class="video-empty" id="cad-empty">
              Loading floor plan...
            </div>
          </div>
          <div class="uc-align-controls hidden" id="uc-align-controls">
            <label for="uc-align-select">Camera alignment</label>
            <select id="uc-align-select" aria-label="Registration candidate"><option value="">Loading candidates…</option></select>
            <label><input type="checkbox" id="uc-align-visible" checked> Show path</label>
            <span class="uc-align-note" id="uc-align-note">UNVERIFIED · GEOMETRIC CANDIDATE ONLY</span>
          </div>
          <div class="viewer-foot mono" id="cad-footer">
            CAD / LOADING
          </div>
        </div>

        <div class="viewer">
          <div class="viewer-head">
            <strong>02 / Walkthrough</strong>
            <div class="viewer-tools">
              <button class="tool" id="replace-project">
                New project
              </button>
            </div>
          </div>

          <div class="video-stage">
            <img id="frame-image" class="hidden"
                 alt="Walkthrough frame">
            <video id="local-video" class="hidden"
                   controls playsinline></video>
            <div class="video-empty" id="video-empty">
              No walkthrough loaded.
            </div>
          </div>

          <div class="video-controls" id="frame-controls">
            <button class="tool" id="prev-frame">◀</button>
            <input type="range" id="frame-slider"
                   min="0" max="0" value="0">
            <button class="tool" id="next-frame">▶</button>
            <span class="mono" id="frame-counter"
                  style="font-size:10px;white-space:nowrap">
              0 / 0
            </span>
          </div>
        </div>
      </div>


<!-- UC_STEP123_ALIGNMENT_START -->
<style>
.uc-align-controls{padding:12px 15px;border-top:1px solid #343542;background:#1b1b25;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.uc-align-controls label{font-size:11px;color:#c8bfd9}
#uc-align-select{background:#292a35;color:#f0eef7;border:1px solid #625576;border-radius:6px;padding:7px 9px;max-width:210px}
.uc-align-note{font:10px Consolas,monospace;color:#f0a47f;flex:1;min-width:175px}
#uc-align-layer{pointer-events:none}
#uc-align-track{fill:none;stroke:#b89aff;stroke-width:2.7;vector-effect:non-scaling-stroke;stroke-linecap:round;stroke-linejoin:round}
#uc-align-travelled{fill:none;stroke:#f0784b;stroke-width:3;vector-effect:non-scaling-stroke;stroke-linecap:round}
#uc-align-marker{fill:#f0784b;stroke:#fff0e8;stroke-width:1.6;vector-effect:non-scaling-stroke}
</style>
<script>
// Alignment UI is initialized by the isolated Step 123 script below.
</script>
<!-- UC_STEP123_ALIGNMENT_END -->

<!-- UC_STEP124_3D_START -->
<style>
.uc-3d-panel{margin-top:14px;background:#191a22;border:1px solid #343542;border-radius:11px;overflow:hidden}
.uc-3d-head{padding:16px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
.uc-3d-head strong{font-size:12px}
.uc-3d-head button{background:#b89aff;color:#17131f;border:0;border-radius:7px;padding:10px 15px;font-size:12px;font-weight:700}
.uc-3d-foot{padding:0 16px 15px;font-size:11px;color:#aaa5b9;line-height:1.7}
</style>
<div class="uc-3d-panel hidden" id="uc-3d-panel">
 <div class="uc-3d-head"><strong>03 / Floor 2 · Interactive 3D CAD</strong><button id="uc-3d-open" type="button">Open 3D →</button></div>
 <div class="uc-3d-foot">The 3D geometry now comes from the same Floor 2 DXF as the plan. Door/window CAD-layer segments are hidden by default because they are not verified openings. Video evidence is not yet registered to these 3D elements.</div>
</div>
<div id="uc-3d-workspace" class="hidden" style="border-top:1px solid #343542">
 <div style="padding:12px 16px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap">
  <span class="mono" style="font-size:11px;color:#b89aff">INTERACTIVE 3D / ORIGINAL DIGITAL TWIN</span>
  <button type="button" class="tool" id="uc-3d-back">← Back to floor plan</button>
 </div>
 <iframe id="uc-3d-frame" title="Interactive architectural 3D twin" loading="lazy" style="width:100%;height:690px;min-height:65vh;border:0;display:block;background:#101117" referrerpolicy="same-origin"></iframe>
</div>
<!-- UC_STEP124_3D_END -->



      <div class="inspector">
        <div>
          <h3 id="inspector-title">Element inspector</h3>
          <p id="inspector-description">
            Select a CAD line to inspect its source data.
          </p>
        </div>
        <div class="inspector-data">
          <div>
            <small>CAD layer</small>
            <span id="detail-layer">—</span>
          </div>
          <div>
            <small>Entity</small>
            <span id="detail-type">—</span>
          </div>
          <div>
            <small>Verification</small>
            <span style="color:var(--orange)">
              UNVERIFIED
            </span>
          </div>
        </div>
      </div>

      <div class="notice">
        This workspace shows planned geometry and recorded
        footage. No physical correspondence, CAD registration,
        or construction completion has been verified.
      </div>
    </div>
  </main>
</section>

<script>
"use strict";

const $ = id => document.getElementById(id);

const state = {
  elements: [],
  frames: [],
  frameIndex: 0,
  box: null,
  initialBox: null,
  selected: null,
  planFile: null,
  videoFile: null,
  videoURL: null,
  dragging: null,
  moved: false
};

function setText(id,value){
  $(id).textContent = String(value);
}

function scrollToStudio(){
  $("studio").scrollIntoView({
    behavior: matchMedia("(prefers-reduced-motion:reduce)")
      .matches ? "instant" : "smooth"
  });
}

function showUpload(){
  window.uc3DHide?.();
  $("upload-view").classList.remove("hidden");
  $("viewer-view").classList.add("hidden");
  $("sample-one").classList.remove("active");
  $("sample-two").classList.remove("active");
  setText("workspace-eyebrow","PROJECT / NEW");
  setText("workspace-title","Start with your files.");
  setText("workspace-status","NOT ANALYZED");
  scrollToStudio();
}

async function getJSON(path){
  const response=await fetch(path);
  if(!response.ok){
    throw new Error(
      `${path}: HTTP ${response.status}`
    );
  }
  return response.json();
}

function getGeometry(element){
  const g=element.geometry;
  if(!g || g.type!=="polyline" ||
     !Array.isArray(g.coordinates)) return null;

  const points=g.coordinates
    .filter(p=>Array.isArray(p)&&p.length>=2)
    .map(p=>[Number(p[0]),Number(p[1])])
    .filter(p=>p.every(Number.isFinite));

  return points.length>=2 ? points : null;
}

function setBox(b){
  state.box={...b};
  $("cad-svg").setAttribute(
    "viewBox",
    [b.x,b.y,b.w,b.h].join(" ")
  );
}

function renderCAD(){
  const group=$("cad-lines");
  group.replaceChildren();

  let minX=Infinity,minY=Infinity;
  let maxX=-Infinity,maxY=-Infinity;
  let count=0;

  const fragment=document.createDocumentFragment();

  for(const element of state.elements){
    const points=getGeometry(element);
    if(!points)continue;

    const path=document.createElementNS(
      "http://www.w3.org/2000/svg","path"
    );

    const d=points.map((p,i)=>{
      const x=p[0],y=-p[1];
      minX=Math.min(minX,x);
      maxX=Math.max(maxX,x);
      minY=Math.min(minY,y);
      maxY=Math.max(maxY,y);
      return `${i===0?"M":"L"} ${x} ${y}`;
    }).join(" ");

    path.setAttribute(
      "d",d+(element.geometry.closed?" Z":"")
    );
    path.setAttribute("class","cad-line");

    path.addEventListener("click",event=>{
      event.stopPropagation();
      if(state.moved)return;
      group.querySelectorAll(".selected")
        .forEach(n=>n.classList.remove("selected"));
      path.classList.add("selected");
      state.selected=element;

      setText(
        "inspector-title",
        element.element_id||"CAD element"
      );
      setText(
        "inspector-description",
        "Planned CAD geometry. No physical match established."
      );
      setText("detail-layer",element.layer||"—");
      setText(
        "detail-type",
        element.source_entity_type||"—"
      );
    });

    fragment.appendChild(path);
    count++;
  }

  group.appendChild(fragment);
  $("cad-empty").classList.toggle("hidden",count>0);

  if(count){
    const w=Math.max(1,maxX-minX);
    const h=Math.max(1,maxY-minY);
    const pad=Math.max(w,h)*.07;
    const b={
      x:minX-pad,y:minY-pad,
      w:w+2*pad,h:h+2*pad
    };
    state.initialBox=b;
    setBox(b);
  }

  setText(
    "cad-footer",
    `${count} drawable elements / ${state.elements.length} loaded`
  );
}

function zoom(factor,cx,cy){
  if(!state.box)return;
  const b=state.box;
  cx??=b.x+b.w/2;
  cy??=b.y+b.h/2;
  setBox({
    x:cx-(cx-b.x)*factor,
    y:cy-(cy-b.y)*factor,
    w:b.w*factor,h:b.h*factor
  });
}

function svgPoint(event){
  const svg=$("cad-svg");
  const matrix=svg.getScreenCTM();
  if(!matrix)return null;
  const p=svg.createSVGPoint();
  p.x=event.clientX;
  p.y=event.clientY;
  return p.matrixTransform(matrix.inverse());
}

function setupCAD(){
  $("zoom-in").onclick=()=>zoom(.8);
  $("zoom-out").onclick=()=>zoom(1.25);
  $("reset-cad").onclick=()=>{
    if(state.initialBox)setBox(state.initialBox);
  };

  const stage=$("cad-stage");

  stage.addEventListener("wheel",event=>{
    if(!state.box)return;
    event.preventDefault();
    const p=svgPoint(event);
    if(p)zoom(
      event.deltaY>0?1.12:.89,p.x,p.y
    );
  },{passive:false});

  stage.addEventListener("pointerdown",event=>{
    if(!state.box)return;
    state.moved=false;
    state.dragging={
      x:event.clientX,y:event.clientY,
      box:{...state.box}
    };
    stage.setPointerCapture(event.pointerId);
  });

  stage.addEventListener("pointermove",event=>{
    if(!state.dragging)return;
    const dx=event.clientX-state.dragging.x;
    const dy=event.clientY-state.dragging.y;
    if(Math.abs(dx)+Math.abs(dy)>5){
      state.moved=true;
    }
    const rect=stage.getBoundingClientRect();
    const b=state.dragging.box;
    setBox({
      x:b.x-dx*b.w/rect.width,
      y:b.y-dy*b.h/rect.height,
      w:b.w,h:b.h
    });
  });

  stage.addEventListener("pointerup",()=>{
    state.dragging=null;
  });
  stage.addEventListener("pointercancel",()=>{
    state.dragging=null;
  });
}

function showFrame(index){
  if(!state.frames.length)return;
  state.frameIndex=Math.max(
    0,Math.min(index,state.frames.length-1)
  );
  const frame=state.frames[state.frameIndex];
  const img=$("frame-image");
  img.src=frame.image_url;
  img.classList.remove("hidden");
  $("video-empty").classList.add("hidden");
  $("frame-slider").value=state.frameIndex;
  setText(
    "frame-counter",
    `${state.frameIndex+1} / ${state.frames.length}`
  );
}

async function loadFrames(){
  const result=[];
  let offset=0;

  for(let page=0;page<30;page++){
    const response=await getJSON(
      `/api/walkthrough/frames?offset=${offset}&limit=250`
    );
    const batch=response.frames||[];
    result.push(...batch);
    offset+=batch.length;
    if(!batch.length || offset>=response.total)break;
  }

  return result;
}

async function openSampleOne(){
  scrollToStudio();
  $("upload-view").classList.add("hidden");
  $("viewer-view").classList.remove("hidden");
  $("sample-one").classList.add("active");
  $("sample-two").classList.remove("active");

  setText("workspace-eyebrow","SAMPLE / LOCAL DATASET");
  setText("workspace-title","Floor 2 Walkthrough");
  setText("workspace-status","UNVERIFIED");
  setText("cad-footer","Loading real CAD geometry...");

  $("frame-image").classList.add("hidden");
  $("local-video").classList.add("hidden");
  $("video-empty").classList.remove("hidden");
  $("frame-controls").classList.remove("hidden");

  try{
    const [cad,frames]=await Promise.all([
      getJSON("/api/cad/elements"),
      loadFrames()
    ]);

    state.elements=cad.elements||[];
    state.frames=frames;
    renderCAD();

    $("frame-slider").max=Math.max(
      0,frames.length-1
    );

    if(frames.length){
      showFrame(0);
    }else{
      setText("video-empty","No recorded frames found.");
    }

  }catch(error){
    console.error(error);
    state.elements=[];
    renderCAD();
    setText(
      "cad-footer",
      "Backend unavailable. Start Step 107 on port 5051."
    );
    setText(
      "video-empty",
      "Could not load walkthrough frames."
    );
  }
}

function openSampleTwo(){
  showUpload();
  $("sample-two").classList.add("active");
  setText("workspace-eyebrow","SAMPLE / NOT INSTALLED");
  setText("workspace-title","Second sample coming soon.");
  setText("workspace-status","NO DATASET");
  setText(
    "upload-status",
    "A SECOND REAL DATASET IS REQUIRED"
  );
}

function bindDrop(zoneId,inputId,kind){
  const zone=$(zoneId);
  const input=$(inputId);

  zone.onclick=()=>input.click();
  zone.onkeydown=event=>{
    if(event.key==="Enter"||event.key===" "){
      event.preventDefault();
      input.click();
    }
  };

  function receive(file){
    if(!file)return;

    const extension=file.name.split(".").pop().toLowerCase();
    const valid=kind==="plan"
      ? ["dxf"].includes(extension)
      : ["mp4","mov","webm"].includes(extension);

    if(!valid){
      alert(`Unsupported ${kind} file type.`);
      return;
    }

    state[kind+"File"]=file;
    setText(
      kind==="plan"?"plan-file":"video-file",
      `SELECTED / ${file.name}`
    );

    $("open-uploads").disabled=!(
      state.planFile&&state.videoFile
    );

    setText(
      "upload-status",
      state.planFile&&state.videoFile
        ?"FILES READY FOR LOCAL PREVIEW"
        :"ONE MORE FILE NEEDED"
    );
  }

  input.onchange=()=>receive(input.files[0]);

  zone.addEventListener("dragover",event=>{
    event.preventDefault();
    zone.classList.add("dragging");
  });

  zone.addEventListener("dragleave",()=>{
    zone.classList.remove("dragging");
  });

  zone.addEventListener("drop",event=>{
    event.preventDefault();
    zone.classList.remove("dragging");
    receive(event.dataTransfer.files[0]);
  });
}

// STEP_111_LOCAL_DXF
async function openLocalPreview(){
  window.uc3DHide?.();
  if(!state.planFile || !state.videoFile) return;

  $("upload-view").classList.add("hidden");
  $("viewer-view").classList.remove("hidden");

  $("sample-one").classList.remove("active");
  $("sample-two").classList.remove("active");

  setText("workspace-eyebrow", "PROJECT / LOCAL FILES");
  setText("workspace-title", "Your project");
  setText("workspace-status", "UNVERIFIED");

  setText("cad-footer", "Reading uploaded DXF...");
  setText("cad-empty", "Parsing your floor plan...");
  $("cad-empty").classList.remove("hidden");

  state.elements = [];
  state.frames = [];
  state.selected = null;
  state.box = null;
  state.initialBox = null;

  $("cad-lines").replaceChildren();
  $("cad-svg").removeAttribute("viewBox");

  setText("inspector-title", "Element inspector");
  setText(
    "inspector-description",
    "Select a CAD element to inspect its source data."
  );
  setText("detail-layer", "—");
  setText("detail-type", "—");

  // Display the actual uploaded video in the browser.
  $("frame-image").classList.add("hidden");
  $("frame-controls").classList.add("hidden");
  $("video-empty").classList.add("hidden");

  const video = $("local-video");
  video.pause();

  if(state.videoURL){
    URL.revokeObjectURL(state.videoURL);
  }

  state.videoURL = URL.createObjectURL(state.videoFile);
  video.src = state.videoURL;
  video.classList.remove("hidden");
  video.load();

  // Upload the DXF only to the local Flask server for parsing.
  // No analysis, progress detection, or video registration occurs.
  const form = new FormData();
  form.append("file", state.planFile);

  try{
    const response = await fetch("/api/local/dxf", {
      method: "POST",
      body: form
    });

    const result = await response.json();

    if(!response.ok){
      throw new Error(
        result.detail || result.error || "DXF parsing failed."
      );
    }

    state.elements = result.elements || [];

    renderCAD();

    if(!state.elements.length){
      setText(
        "cad-empty",
        "No supported drawable CAD entities were found."
      );
      $("cad-empty").classList.remove("hidden");
    }

    const skippedCount = Object.values(result.skipped || {})
      .reduce((sum, n) => sum + Number(n || 0), 0);

    setText(
      "cad-footer",
      `${state.elements.length} drawable CAD entities` +
      (skippedCount ? ` / ${skippedCount} unsupported or skipped` : "") +
      (result.truncated ? " / LIMIT REACHED" : "") +
      " / UNVERIFIED"
    );

    setText(
      "inspector-description",
      "This element comes from your uploaded DXF. " +
      "Its physical location and construction status " +
      "have not been established from the walkthrough."
    );

  }catch(error){
    console.error("Local DXF error:", error);

    state.elements = [];
    state.box = null;
    state.initialBox = null;

    $("cad-lines").replaceChildren();

    setText(
      "cad-empty",
      "Could not display DXF: " + error.message
    );

    $("cad-empty").classList.remove("hidden");

    setText("cad-footer", "DXF / PARSING FAILED");
  }

  scrollToStudio();
}


function setupHero(){
  const art=$("hero-art");
  const model=$("hero-model");
  if(matchMedia("(prefers-reduced-motion:reduce)").matches){
    return;
  }

  art.addEventListener("pointermove",event=>{
    const r=art.getBoundingClientRect();
    const x=(event.clientX-r.left)/r.width-.5;
    const y=(event.clientY-r.top)/r.height-.5;
    model.style.transform=
      `rotateX(${54-y*13}deg) rotateZ(${-38+x*16}deg)`;
  });

  art.addEventListener("pointerleave",()=>{
    model.style.transform=
      "rotateX(54deg) rotateZ(-38deg)";
  });
}

function initialize(){
  setupHero();
  setupCAD();

  bindDrop("plan-drop","plan-input","plan");
  bindDrop("video-drop","video-input","video");

  $("hero-sample").onclick=openSampleOne;
  $("hero-own").onclick=showUpload;
  $("sample-one").onclick=openSampleOne;
  $("sample-two").onclick=openSampleTwo;
  $("new-project").onclick=showUpload;
  $("replace-project").onclick=showUpload;
  $("open-uploads").onclick=openLocalPreview;

  $("prev-frame").onclick=()=>showFrame(
    state.frameIndex-1
  );
  $("next-frame").onclick=()=>showFrame(
    state.frameIndex+1
  );
  $("frame-slider").oninput=event=>showFrame(
    Number(event.target.value)
  );

  $("nav-samples").onclick=event=>{
    event.preventDefault();
    openSampleOne();
  };
}

initialize();
</script>









<!-- UC_STEP114_START -->

<style>
/* ==========================================================
   UNDER-CONSTRUCTION  hero: architectural cutaway animation
   Replaces the old "Living Floor Plan" block (Step 114).
   ========================================================== */

.hero-art {
  position: relative !important;
  min-height: 600px;
  height: 600px;
  overflow: hidden !important;
  perspective: none !important;
}

/* hide the legacy CSS-3D model, grid and captions */
.hero-art > :not(.living-stage) {
  display: none !important;
}

.living-stage {
  position: absolute;
  inset: 0;
  overflow: hidden;
  isolation: isolate;
  background: radial-gradient(ellipse at 52% 55%, rgba(133, 104, 184, .10), transparent 62%);
}

.living-stage canvas.living-gl {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
}

/* fine orange leader lines (drawn by JS from 3D anchors) */
.living-leaders {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 2;
  pointer-events: none;
  overflow: visible;
}

.living-leaders path {
  fill: none;
  stroke: #ff8a4c;
  stroke-width: 1.3;
  stroke-linejoin: round;
  stroke-linecap: round;
  stroke-dasharray: 1;
  stroke-dashoffset: 1;
  opacity: 0;
}

.living-leaders circle {
  fill: #ff8a4c;
  opacity: 0;
}

.living-card {
  position: absolute;
  z-index: 3;
  width: clamp(120px, 17.5%, 198px);
  background: #17161d;
  border: 1.5px solid #ff8a4c;
  border-radius: 9px;
  padding: 7px 7px 8px;
  box-shadow: 0 16px 34px rgba(0, 0, 0, .55);
  opacity: 0;
  transform: translateY(8px) scale(.97);
  transition: opacity .45s ease, transform .45s ease;
  pointer-events: none;
}

.living-card.active {
  opacity: 1;
  transform: none;
}

.living-card-one { left: 1.5%; top: 3%; }
.living-card-two { right: 1%; bottom: 11%; }

.living-card-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  color: #e4dbf7;
  font: 600 9px/1 Consolas, "Courier New", monospace;
  letter-spacing: .09em;
}

.living-card-label i {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #ff8a4c;
}

.living-view {
  display: block;
  width: 100%;
  height: auto;
  aspect-ratio: 360 / 224;
  border-radius: 3px;
  background: #34333a;
}

.living-caption {
  position: absolute;
  z-index: 3;
  bottom: 6%;
  left: 4%;
  color: #a99bbb;
  font: 10px/1.7 Consolas, "Courier New", monospace;
  letter-spacing: .1em;
  text-transform: uppercase;
  pointer-events: none;
}

.living-caption strong {
  color: #ff9b68;
  font-weight: normal;
}

.living-error {
  display: none;
  position: absolute;
  inset: 28% 10%;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: #c9bce0;
  font: 13px/1.7 Consolas, "Courier New", monospace;
}

@media (max-width: 900px) {
  .hero-art { height: 450px; min-height: 450px; }
  .living-card { width: 135px; }
}

@media (max-width: 560px) {
  .hero-art { height: 340px; min-height: 340px; }
  .living-card { width: 104px; padding: 5px; }
  .living-card-label { font-size: 7px; margin-bottom: 4px; }
  .living-caption { font-size: 8px; }
}

/* ---- hard-hat logo (replaces the dashed circle) ---- */
.brand-mark {
  width: auto !important;
  height: auto !important;
  border: 0 !important;
  border-radius: 0 !important;
  display: flex !important;
  align-items: center;
  justify-content: center;
  font-size: 0 !important;
  color: var(--violet);
}

.brand-mark svg {
  display: block;
  width: 42px;
  height: 34px;
}
</style>

<script type="module">
import * as THREE from
  "https://cdn.jsdelivr.net/npm/three@0.160.1/build/three.module.js";

/* ==========================================================
   HELPERS
   ========================================================== */
const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smooth = (v) => { v = clamp(v); return v * v * (3 - 2 * v); };

function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ==========================================================
   ONE COORDINATE SYSTEM (metres).  x = east, z = south, y = up.
   The plan texture, 3D walls, route and card anchors all
   derive from the data below.
   ========================================================== */
const EXT = 0.30;   // exterior wall thickness
const INT = 0.20;   // interior wall thickness

const door = (u0, u1, side = 1, hinge = 0) =>
  ({ k: "door", u0, u1, y0: 0, y1: 2.1, side, hinge });
const win = (u0, u1) =>
  ({ k: "win", u0, u1, y0: 0.95, y1: 2.15 });

/* a = start [x,z], b = end [x,z] (axis aligned, a < b),
   t thickness, h final height, zone = reveal group,
   open = openings measured from the start along the wall. */
const WALLS = [
  /* north (back) exterior, full height */
  { id: "N1", a: [0, 0],  b: [7, 0],  t: EXT, h: 3.0, zone: 0, open: [win(0.9, 2.1), win(4.4, 5.8)] },
  { id: "N2", a: [7, 0],  b: [13, 0], t: EXT, h: 3.0, zone: 1, open: [win(1.0, 2.6), win(3.4, 5.0)] },
  { id: "N3", a: [13, 0], b: [20, 0], t: EXT, h: 3.0, zone: 2, open: [win(1.2, 2.6), win(4.4, 5.8)] },
  /* west exterior, full height */
  { id: "W1", a: [0, 0],  b: [0, 7],  t: EXT, h: 3.0, zone: 0, open: [win(4.4, 5.8)] },
  { id: "W2", a: [0, 7],  b: [0, 12], t: EXT, h: 3.0, zone: 3, open: [win(1.2, 2.8)] },
  /* front / east exterior, partially built (low) so the plan reads */
  { id: "S1", a: [0, 12], b: [13, 12], t: EXT, h: 1.4, zone: 3, open: [door(4.0, 5.2), door(8.4, 9.8, -1)] },
  { id: "SE1", a: [13, 7], b: [13, 12], t: EXT, h: 1.3, zone: 3, open: [door(2.2, 3.4)] },
  { id: "Z7", a: [13, 7], b: [20, 7],  t: EXT, h: 1.5, zone: 4, open: [door(2.4, 3.6, -1)] },
  { id: "E1", a: [20, 0], b: [20, 4],  t: EXT, h: 1.7, zone: 2, open: [] },
  { id: "E2", a: [20, 4], b: [20, 7],  t: EXT, h: 1.3, zone: 4, open: [], rebar: true },
  /* interior partitions at different stages */
  { id: "P1", a: [7, 0],  b: [7, 7],   t: INT, h: 2.7, zone: 1, open: [door(2.0, 3.2)] },
  { id: "P2", a: [13, 0], b: [13, 7],  t: INT, h: 2.4, zone: 2, open: [door(2.0, 3.2, -1)] },
  { id: "P3", a: [0, 7],  b: [13, 7],  t: INT, h: 1.9, zone: 3, open: [door(4.0, 5.2), door(9.4, 10.6, -1)] },
  { id: "P4", a: [13, 4], b: [20, 4],  t: INT, h: 2.2, zone: 2, open: [door(2.5, 3.7)] },
  { id: "P5", a: [0, 3.5], b: [3.5, 3.5], t: INT, h: 2.6, zone: 0, open: [] },
  { id: "P6", a: [3.5, 0], b: [3.5, 3.5], t: INT, h: 2.6, zone: 0, open: [door(1.9, 2.8, -1)] },
  { id: "P7", a: [6.5, 7], b: [6.5, 12], t: INT, h: 2.3, zone: 3, open: [door(2.0, 3.1)] },
  { id: "K1", a: [16.8, 0], b: [16.8, 1.6], t: INT, h: 1.9, zone: 2, open: [], rebar: true },
  { id: "D1", a: [10, 5],  b: [13, 5],  t: INT, h: 1.2, zone: 1, open: [], rebar: true }
];

const COLUMNS = [
  { x: 9.0,  z: 9.5,  h: 2.9, zone: 3 },
  { x: 11.5, z: 9.5,  h: 2.0, zone: 3, rebar: true },
  { x: 11.0, z: 1.2,  h: 3.0, zone: 1 },
  { x: 18.6, z: 5.9,  h: 1.4, zone: 4, rebar: true },
  { x: 15.2, z: 9.2,  h: 3.0, zone: 4 },
  { x: 18.6, z: 9.2,  h: 2.2, zone: 4 },
  { x: 15.2, z: 11.2, h: 1.5, zone: 4, rebar: true },
  { x: 18.6, z: 11.2, h: 3.0, zone: 4 },
  { x: -3.4, z: 4.2,  h: 2.0, zone: 0, pad: true },
  { x: -4.4, z: 9.0,  h: 1.5, zone: 3, pad: true, rebar: true }
];

/* floor-glow rectangles (inner clear area of each room) */
const ROOMS = [
  { x0: 0.15, z0: 0.1,  x1: 3.4,  z1: 3.4,  zone: 0, col: 0xa98cff, gain: 0.7 },
  { x0: 3.6,  z0: 0.1,  x1: 6.9,  z1: 6.9,  zone: 0, col: 0xff8a4c, gain: 1.0 },
  { x0: 0.15, z0: 3.6,  x1: 3.4,  z1: 6.9,  zone: 0, col: 0xa98cff, gain: 0.7 },
  { x0: 7.1,  z0: 0.1,  x1: 12.9, z1: 6.9,  zone: 1, col: 0xff8a4c, gain: 1.0 },
  { x0: 13.1, z0: 0.1,  x1: 19.9, z1: 3.9,  zone: 2, col: 0xff8a4c, gain: 1.0 },
  { x0: 13.1, z0: 4.1,  x1: 19.9, z1: 6.9,  zone: 4, col: 0xff8a4c, gain: 1.0 },
  { x0: 0.15, z0: 7.1,  x1: 6.4,  z1: 11.9, zone: 3, col: 0xa98cff, gain: 0.8 },
  { x0: 6.6,  z0: 7.1,  x1: 12.9, z1: 11.9, zone: 3, col: 0xff8a4c, gain: 0.9 },
  { x0: 13.3, z0: 7.3,  x1: 20.2, z1: 12.2, zone: 4, col: 0xa98cff, gain: 0.8 }
];

/* the walk: start outside, in through the front door, across the
   plan to the east wing.  Doors above are placed on this line. */
const ROUTE = [
  [4.6, 14.3], [4.6, 12.0], [4.6, 7.0], [4.6, 2.6], [16.1, 2.6], [16.1, 5.7]
];
const STATION_1 = [10.0, 2.6];   /* anchor of SITE VIEW / 01 */
const STATION_2 = ROUTE[5];      /* anchor of SITE VIEW / 02 */

const PLAN = { x0: -6, z0: -5, w: 32, h: 22 };       /* plan texture extent */
const PODIUM = { x0: -0.4, z0: -0.4, x1: 20.4, z1: 12.4 };
const CENTER = { x: 10, z: 6 };
const FIT = { x0: -4.6, x1: 21.0, y0: -0.4, y1: 3.2, z0: -0.8, z1: 14.9 };

/* ---- route arc length + zone entry points ---- */
const CUM = [0];
for (let i = 0; i < ROUTE.length - 1; i++) {
  CUM.push(CUM[i] + Math.hypot(ROUTE[i + 1][0] - ROUTE[i][0], ROUTE[i + 1][1] - ROUTE[i][1]));
}
const ROUTE_LEN = CUM[CUM.length - 1];
const S1_LEN = CUM[3] + (STATION_1[0] - ROUTE[3][0]);
const ZONE_ENTRY = {
  3: CUM[1],
  0: CUM[2],
  1: CUM[3] + (7 - ROUTE[3][0]),
  2: CUM[3] + (13 - ROUTE[3][0]),
  4: CUM[4] + (4.0 - ROUTE[4][1])
};

function posAt(s) {
  s = clamp(s, 0, ROUTE_LEN);
  for (let i = 0; i < ROUTE.length - 1; i++) {
    if (s <= CUM[i + 1] || i === ROUTE.length - 2) {
      const len = CUM[i + 1] - CUM[i];
      const u = len > 0 ? (s - CUM[i]) / len : 0;
      return [lerp(ROUTE[i][0], ROUTE[i + 1][0], u), lerp(ROUTE[i][1], ROUTE[i + 1][1], u)];
    }
  }
  return ROUTE[ROUTE.length - 1];
}

/* ==========================================================
   TIMELINE (seconds, 15 s loop)
   A  0.0-1.4  plan only
   B  1.4-6.8  marker walks, route draws
   C  rooms rise as the marker enters them
   D  6.8 / 10.4  site views appear + leader lines
   E  hold, then 13.3-14.7 restrained reset
   ========================================================== */
const LOOP = 15;
const T_INTRO = 1.4;
const T_MOVE0 = 2.2;
const T_S1 = 6.8;
const T_S1_OUT = 7.5;
const T_END = 10.4;
const T_RESET0 = 13.3;
const T_RESET1 = 14.7;
const STATIC_T = 12.4;

const KEYS = [[T_MOVE0, 0], [T_S1, S1_LEN], [T_S1_OUT, S1_LEN], [T_END, ROUTE_LEN]];

function sAt(t) {
  if (t <= KEYS[0][0]) return 0;
  for (let i = 0; i < KEYS.length - 1; i++) {
    const [t0, s0] = KEYS[i];
    const [t1, s1] = KEYS[i + 1];
    if (t <= t1) {
      const u = (t - t0) / (t1 - t0);
      return lerp(s0, s1, lerp(u, smooth(u), 0.65));
    }
  }
  return ROUTE_LEN;
}

function timeAtS(s) {
  if (s <= 0) return T_MOVE0;
  let lo = T_MOVE0, hi = T_END;
  for (let i = 0; i < 28; i++) {
    const mid = (lo + hi) / 2;
    if (sAt(mid) >= s - 1e-6) hi = mid; else lo = mid;
  }
  return hi;
}

const ZONE_T0 = {};
Object.keys(ZONE_ENTRY).forEach((z) => {
  ZONE_T0[z] = timeAtS(Math.max(0, ZONE_ENTRY[z] - 1.4));
});

/* ==========================================================
   ARCHITECTURAL BUILDERS
   ========================================================== */

/* Splits a wall into solid plan intervals and 3D box pieces
   around its openings. */
function wallInfo(w) {
  const horiz = Math.abs(w.b[1] - w.a[1]) < 1e-6;
  const L = horiz ? w.b[0] - w.a[0] : w.b[1] - w.a[1];
  const h2 = w.t / 2;
  const ops = (w.open || [])
    .filter((o) => o.k === "door" || w.h >= o.y1 + 0.25)
    .map((o) => Object.assign({}, o, { y1: Math.min(o.y1, w.h) }))
    .sort((p, q) => p.u0 - q.u0);

  const solids = [];
  const pieces = [];
  let cur = -h2;
  ops.forEach((o) => {
    if (o.u0 > cur) {
      solids.push([cur, o.u0]);
      pieces.push({ u0: cur, u1: o.u0, y0: 0, y1: w.h });
    }
    if (o.y0 > 0.001) pieces.push({ u0: o.u0, u1: o.u1, y0: 0, y1: o.y0 });
    if (o.y1 < w.h - 0.001) pieces.push({ u0: o.u0, u1: o.u1, y0: o.y1, y1: w.h });
    cur = o.u1;
  });
  solids.push([cur, L + h2]);
  pieces.push({ u0: cur, u1: L + h2, y0: 0, y1: w.h });
  return { horiz, L, h2, ops, solids, pieces };
}

/* plan point of a wall-local coordinate (u along, v across) */
function wallPt(w, info, u, v) {
  return info.horiz ? [w.a[0] + u, w.a[1] + v] : [w.a[0] + v, w.a[1] + u];
}

/* rectangle (plan) of a wall solid interval */
function wallRect(w, info, u0, u1) {
  const p = wallPt(w, info, u0, -info.h2);
  const q = wallPt(w, info, u1, info.h2);
  return [Math.min(p[0], q[0]), Math.min(p[1], q[1]), Math.max(p[0], q[0]), Math.max(p[1], q[1])];
}

/* merged-box geometry builder with baked ambient occlusion */
class Geo {
  constructor() { this.p = []; this.n = []; this.u = []; this.c = []; this.i = []; }

  quad(v0, v1, v2, v3, nrm, uvs, cols) {
    const b = this.p.length / 3;
    [v0, v1, v2, v3].forEach((v, k) => {
      this.p.push(v[0], v[1], v[2]);
      this.n.push(nrm[0], nrm[1], nrm[2]);
      this.u.push(uvs[k][0], uvs[k][1]);
      this.c.push(cols[k][0], cols[k][1], cols[k][2]);
    });
    this.i.push(b, b + 1, b + 2, b, b + 2, b + 3);
  }

  box(x0, y0, z0, x1, y1, z1, o) {
    o = o || {};
    if (x1 - x0 < 1e-4 || y1 - y0 < 1e-4 || z1 - z0 < 1e-4) return;
    const tint = o.tint || [1, 1, 1];
    const ao = (y) => (o.flat ? 1 : 0.56 + 0.44 * smooth(y / 1.6));
    const col = (y, k) => {
      const a = ao(y) * (k || 1);
      return [tint[0] * a, tint[1] * a, tint[2] * a];
    };
    const S = 1 / 3;
    const uv = (a, b) => [a * S, b * S];
    const cb = col(y0), ct = col(y1);

    this.quad([x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [1, 0, 0],
      [uv(z1, y0), uv(z0, y0), uv(z0, y1), uv(z1, y1)], [cb, cb, ct, ct]);
    this.quad([x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0], [-1, 0, 0],
      [uv(z0, y0), uv(z1, y0), uv(z1, y1), uv(z0, y1)], [cb, cb, ct, ct]);
    this.quad([x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1], [0, 0, 1],
      [uv(x0, y0), uv(x1, y0), uv(x1, y1), uv(x0, y1)], [cb, cb, ct, ct]);
    this.quad([x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0], [0, 0, -1],
      [uv(x1, y0), uv(x0, y0), uv(x0, y1), uv(x1, y1)], [cb, cb, ct, ct]);
    const top = o.flat ? col(0, 1) : [tint[0] * 1.08, tint[1] * 1.08, tint[2] * 1.08];
    this.quad([x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0], [0, 1, 0],
      [uv(x0, z1), uv(x1, z1), uv(x1, z0), uv(x0, z0)], [top, top, top, top]);
  }

  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.p, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.n, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(this.u, 2));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.c, 3));
    g.setIndex(this.i);
    g.computeBoundingSphere();
    return g;
  }
}

/* wall with door / window openings + optional rebar stubs */
function buildWallGeometry(w, idx) {
  const info = wallInfo(w);
  const geo = new Geo();
  const jitter = idx * 0.0006;               /* avoids coplanar top faces */
  const tint = [0.97 + (idx % 3) * 0.01, 0.94 + (idx % 2) * 0.012, 0.96];
  const hh = w.t / 2;

  info.pieces.forEach((pc) => {
    const y1 = pc.y1 >= w.h - 0.001 ? pc.y1 + jitter : pc.y1;
    if (info.horiz) {
      geo.box(w.a[0] + pc.u0, pc.y0, w.a[1] - hh, w.a[0] + pc.u1, y1, w.a[1] + hh, { tint });
    } else {
      geo.box(w.a[0] - hh, pc.y0, w.a[1] + pc.u0, w.a[0] + hh, y1, w.a[1] + pc.u1, { tint });
    }
  });

  if (w.rebar) {
    let k = 0;
    for (let u = 0.35; u < info.L - 0.1; u += 0.85, k++) {
      const off = (k % 2 ? 1 : -1) * hh * 0.35;
      const p = wallPt(w, info, u, off);
      geo.box(p[0] - 0.02, w.h, p[1] - 0.02, p[0] + 0.02, w.h + 0.45, p[1] + 0.02,
        { flat: true, tint: [0.22, 0.2, 0.22] });
    }
  }
  return geo.build();
}

/* structural column on a plinth, optional exposed rebar */
function buildColumnGeometry(c) {
  const geo = new Geo();
  const s = 0.21;
  geo.box(c.x - s - 0.07, 0, c.z - s - 0.07, c.x + s + 0.07, 0.1, c.z + s + 0.07, { tint: [1, 0.97, 0.99] });
  geo.box(c.x - s, 0.1, c.z - s, c.x + s, c.h, c.z + s, { tint: [0.98, 0.95, 0.97] });
  if (c.rebar) {
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach((q) => {
      const x = c.x + q[0] * (s - 0.06), z = c.z + q[1] * (s - 0.06);
      geo.box(x - 0.02, c.h, z - 0.02, x + 0.02, c.h + 0.5, z + 0.02,
        { flat: true, tint: [0.22, 0.2, 0.22] });
    });
  }
  return geo.build();
}

/* ==========================================================
   TEXTURES
   ========================================================== */
function makeConcreteTexture(maxAniso) {
  const N = 512;
  const c = document.createElement("canvas");
  c.width = c.height = N;
  const g = c.getContext("2d");
  const rnd = mulberry32(7);
  g.fillStyle = "#cdc6ca";
  g.fillRect(0, 0, N, N);

  for (let i = 0; i < 46; i++) {
    const x = rnd() * N, y = rnd() * N, r = 40 + rnd() * 110;
    const dark = rnd() > 0.5;
    for (let dx = -N; dx <= N; dx += N) {
      for (let dy = -N; dy <= N; dy += N) {
        const gr = g.createRadialGradient(x + dx, y + dy, 0, x + dx, y + dy, r);
        gr.addColorStop(0, dark ? "rgba(90,80,95,0.10)" : "rgba(255,250,255,0.10)");
        gr.addColorStop(1, "rgba(0,0,0,0)");
        g.fillStyle = gr;
        g.fillRect(x + dx - r, y + dy - r, r * 2, r * 2);
      }
    }
  }
  for (let i = 0; i < 9000; i++) {
    g.fillStyle = rnd() > 0.5 ? "rgba(255,255,255,0.05)" : "rgba(60,50,65,0.06)";
    g.fillRect(rnd() * N, rnd() * N, 1 + rnd() * 2, 1 + rnd() * 2);
  }
  for (let i = 0; i < 240; i++) {
    g.fillStyle = "rgba(55,48,60,0.22)";
    g.beginPath();
    g.ellipse(rnd() * N, rnd() * N, 0.6 + rnd() * 1.4, 0.5 + rnd() * 1.1, 0, 0, Math.PI * 2);
    g.fill();
  }
  /* formwork tie holes */
  for (let ix = 0; ix < 4; ix++) {
    for (let iy = 0; iy < 2; iy++) {
      const x = 64 + ix * 128, y = 90 + iy * 256;
      g.fillStyle = "rgba(50,44,56,0.30)";
      g.beginPath(); g.arc(x, y, 3, 0, Math.PI * 2); g.fill();
      g.strokeStyle = "rgba(255,255,255,0.10)";
      g.lineWidth = 1;
      g.beginPath(); g.arc(x, y, 4.5, 0, Math.PI * 2); g.stroke();
    }
  }
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = Math.min(8, maxAniso);
  return t;
}

function makeGlowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  gr.addColorStop(0, "rgba(255,255,255,1)");
  gr.addColorStop(0.35, "rgba(255,255,255,0.45)");
  gr.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = gr;
  g.fillRect(0, 0, 128, 128);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

/* Blueprint linework, drawn from the SAME wall data as the 3D model. */
function buildPlanTexture(maxAniso) {
  const small = window.innerWidth < 700;
  const PW = small ? 1280 : 2048;
  const S = PW / PLAN.w;
  const PH = Math.round(PLAN.h * S);
  const cv = document.createElement("canvas");
  cv.width = PW; cv.height = PH;
  const g = cv.getContext("2d");
  const LAV = "#a98cff", LAV_HI = "#cdbcff";
  const setM = (ctx) => ctx.setTransform(S, 0, 0, S, -PLAN.x0 * S, -PLAN.z0 * S);
  setM(g);

  const xMax = PLAN.x0 + PLAN.w, zMax = PLAN.z0 + PLAN.h;
  const seg = (a, b) => { g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); };
  const txt = (str, x, z, px, col, align) => {
    g.save();
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.font = (px * S / 64) + "px Consolas, 'Courier New', monospace";
    g.fillStyle = col;
    g.textAlign = align || "center";
    g.textBaseline = "middle";
    g.fillText(str, (x - PLAN.x0) * S, (z - PLAN.z0) * S);
    g.restore();
  };

  /* 1 m grid + 5 m major grid */
  g.lineWidth = 0.012;
  g.strokeStyle = "rgba(169,140,255,0.13)";
  g.beginPath();
  for (let x = Math.ceil(PLAN.x0); x <= xMax; x++) seg([x, PLAN.z0], [x, zMax]);
  for (let z = Math.ceil(PLAN.z0); z <= zMax; z++) seg([PLAN.x0, z], [xMax, z]);
  g.stroke();
  g.lineWidth = 0.02;
  g.strokeStyle = "rgba(169,140,255,0.28)";
  g.beginPath();
  for (let x = Math.ceil(PLAN.x0 / 5) * 5; x <= xMax; x += 5) seg([x, PLAN.z0], [x, zMax]);
  for (let z = Math.ceil(PLAN.z0 / 5) * 5; z <= zMax; z += 5) seg([PLAN.x0, z], [xMax, z]);
  g.stroke();

  /* ghost wing (planned, not yet drawn in 3D) */
  g.save();
  g.setLineDash([0.5, 0.25]);
  g.lineWidth = 0.025;
  g.strokeStyle = "rgba(169,140,255,0.38)";
  g.strokeRect(20.9, 0.6, 4.3, 7.6);
  g.setLineDash([]);
  g.lineWidth = 0.015;
  g.beginPath();
  seg([20.9, 3.4], [25.2, 3.4]); seg([23.0, 0.6], [23.0, 3.4]); seg([22.4, 3.4], [22.4, 8.2]);
  g.stroke();
  g.restore();

  /* collect wall rectangles */
  const infos = WALLS.map((w) => wallInfo(w));
  const rects = [];
  WALLS.forEach((w, i) => infos[i].solids.forEach((s) => rects.push(wallRect(w, infos[i], s[0], s[1]))));
  COLUMNS.forEach((c) => rects.push([c.x - 0.21, c.z - 0.21, c.x + 0.21, c.z + 0.21]));

  /* contact shadow band under every wall */
  g.save();
  g.shadowColor = "rgba(0,0,0,0.7)";
  g.shadowBlur = 0.45 * S;
  g.fillStyle = "rgba(0,0,0,0.55)";
  rects.forEach((r) => g.fillRect(r[0], r[1], r[2] - r[0], r[3] - r[1]));
  g.restore();

  /* union mask -> tinted fill + clean outline ring */
  const mask = document.createElement("canvas");
  mask.width = PW; mask.height = PH;
  const mg = mask.getContext("2d");
  setM(mg);
  mg.fillStyle = "#fff";
  rects.forEach((r) => mg.fillRect(r[0], r[1], r[2] - r[0], r[3] - r[1]));

  const tint = document.createElement("canvas");
  tint.width = PW; tint.height = PH;
  const tg = tint.getContext("2d");
  tg.drawImage(mask, 0, 0);
  tg.globalCompositeOperation = "source-in";
  tg.fillStyle = LAV;
  tg.fillRect(0, 0, PW, PH);

  const ring = document.createElement("canvas");
  ring.width = PW; ring.height = PH;
  const rg = ring.getContext("2d");
  const d = Math.max(1.6, 0.03 * S);
  for (let k = 0; k < 8; k++) {
    const a = (k / 8) * Math.PI * 2;
    rg.drawImage(tint, Math.cos(a) * d, Math.sin(a) * d);
  }
  rg.globalCompositeOperation = "destination-out";
  rg.drawImage(mask, 0, 0);

  g.save();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.globalAlpha = 0.26;
  g.drawImage(tint, 0, 0);
  g.globalAlpha = 0.95;
  g.drawImage(ring, 0, 0);
  g.restore();

  /* window + door symbols */
  WALLS.forEach((w, i) => {
    const info = infos[i];
    info.ops.forEach((o) => {
      if (o.k === "win") {
        g.strokeStyle = "rgba(205,188,255,0.9)";
        g.lineWidth = 0.018;
        g.beginPath();
        [-1, 0, 1].forEach((m) => seg(wallPt(w, info, o.u0, m * info.h2), wallPt(w, info, o.u1, m * info.h2)));
        g.stroke();
      } else {
        const hu = o.hinge === 1 ? o.u1 : o.u0;
        const ou = o.hinge === 1 ? o.u0 : o.u1;
        const wd = o.u1 - o.u0;
        const H = wallPt(w, info, hu, 0), J = wallPt(w, info, ou, 0);
        const E = info.horiz ? [H[0], H[1] + o.side * wd] : [H[0] + o.side * wd, H[1]];
        g.strokeStyle = "rgba(205,188,255,0.75)";
        g.lineWidth = 0.018;
        g.beginPath(); seg(H, E); g.stroke();
        const a0 = Math.atan2(E[1] - H[1], E[0] - H[0]);
        const a1 = Math.atan2(J[1] - H[1], J[0] - H[0]);
        let dd = a1 - a0;
        while (dd > Math.PI) dd -= Math.PI * 2;
        while (dd < -Math.PI) dd += Math.PI * 2;
        g.strokeStyle = "rgba(169,140,255,0.6)";
        g.lineWidth = 0.014;
        g.beginPath(); g.arc(H[0], H[1], wd, a0, a1, dd < 0); g.stroke();
      }
    });
  });

  /* fixtures */
  const rc = (x0, z0, x1, z1) => g.strokeRect(x0, z0, x1 - x0, z1 - z0);
  const el = (x, z, rx, rz) => { g.beginPath(); g.ellipse(x, z, rx, rz, 0, 0, Math.PI * 2); g.stroke(); };
  g.strokeStyle = "rgba(235,226,255,0.92)";
  g.lineWidth = 0.02;
  rc(1.55, 0.3, 2.05, 0.55); el(1.8, 0.82, 0.19, 0.27);                 /* WC */
  rc(2.55, 0.3, 3.15, 0.78); el(2.85, 0.54, 0.16, 0.12);                /* basin */
  rc(0.3, 1.2, 1.05, 3.2);   el(0.675, 2.2, 0.28, 0.8);                 /* bath */
  rc(7.1, 11.2, 12.7, 11.75);                                           /* kitchen run */
  el(8.3, 11.48, 0.2, 0.14); el(8.9, 11.48, 0.2, 0.14);
  [10.6, 11.2, 11.8, 12.4].forEach((x) => el(x, 11.48, 0.12, 0.12));
  rc(0.4, 7.6, 1.6, 11.5);                                              /* stair */
  g.beginPath();
  for (let z = 7.93; z < 11.5; z += 0.33) seg([0.4, z], [1.6, z]);
  seg([1.0, 11.2], [1.0, 7.9]); seg([0.82, 8.2], [1.0, 7.9]); seg([1.18, 8.2], [1.0, 7.9]);
  g.stroke();
  rc(5.2, 0.5, 6.7, 1.25); rc(5.4, 2.1, 6.4, 2.45);                     /* west lounge */
  rc(17.6, 1.0, 19.3, 3.0);                                             /* long table */
  rc(18.0, 4.5, 19.8, 6.6); rc(18.1, 4.6, 18.8, 5.0); rc(19.0, 4.6, 19.7, 5.0);

  /* structural column symbols */
  COLUMNS.forEach((c) => {
    g.fillStyle = "rgba(169,140,255,0.45)";
    g.fillRect(c.x - 0.21, c.z - 0.21, 0.42, 0.42);
    g.strokeStyle = LAV_HI;
    g.lineWidth = 0.025;
    rc(c.x - 0.21, c.z - 0.21, c.x + 0.21, c.z + 0.21);
    g.beginPath();
    seg([c.x - 0.21, c.z - 0.21], [c.x + 0.21, c.z + 0.21]);
    seg([c.x + 0.21, c.z - 0.21], [c.x - 0.21, c.z + 0.21]);
    g.stroke();
  });

  /* structural grid axes with bubbles */
  g.save();
  g.setLineDash([0.9, 0.12, 0.12, 0.12]);
  g.lineWidth = 0.014;
  g.strokeStyle = "rgba(169,140,255,0.42)";
  [0, 7, 13, 20].forEach((x) => { g.beginPath(); seg([x, -2.2], [x, 13.2]); g.stroke(); });
  [0, 7, 12].forEach((z) => { g.beginPath(); seg([-2.2, z], [22.4, z]); g.stroke(); });
  g.restore();
  g.lineWidth = 0.02;
  g.strokeStyle = "rgba(205,188,255,0.8)";
  ["A", "B", "C", "D"].forEach((n, i) => {
    const x = [0, 7, 13, 20][i];
    g.beginPath(); g.arc(x, -2.6, 0.4, 0, Math.PI * 2); g.stroke();
    txt(n, x, -2.6, 22, "rgba(205,188,255,0.9)");
  });
  ["1", "2", "3"].forEach((n, i) => {
    const z = [0, 7, 12][i];
    g.beginPath(); g.arc(-2.6, z, 0.4, 0, Math.PI * 2); g.stroke();
    txt(n, -2.6, z, 22, "rgba(205,188,255,0.9)");
  });

  /* dimension lines */
  g.strokeStyle = "rgba(205,188,255,0.55)";
  g.lineWidth = 0.014;
  g.beginPath();
  seg([0, 13.5], [20, 13.5]);
  [0, 7, 13, 20].forEach((x) => seg([x, 13.25], [x, 13.75]));
  seg([21.9, 0], [21.9, 12]);
  [0, 7, 12].forEach((z) => seg([21.65, z], [22.15, z]));
  g.stroke();
  txt("7000", 3.5, 13.85, 18, "rgba(205,188,255,0.65)");
  txt("6000", 10, 13.85, 18, "rgba(205,188,255,0.65)");
  txt("7000", 16.5, 13.85, 18, "rgba(205,188,255,0.65)");

  /* room tags */
  [["R-01", 1.7, 5.2], ["R-02", 5.2, 5.0], ["R-03", 9.9, 5.8], ["R-04", 17.4, 0.5],
   ["R-05", 14.6, 5.5], ["R-06", 3.4, 9.8], ["R-07", 9.8, 8.1]].forEach((r) => {
    txt(r[0], r[1], r[2], 16, "rgba(205,188,255,0.45)");
  });

  /* site boundary + glowing footprint */
  g.save();
  g.setLineDash([0.6, 0.3]);
  g.lineWidth = 0.03;
  g.strokeStyle = "rgba(169,140,255,0.5)";
  g.strokeRect(-2.2, -1.8, 24.6, 15.6);
  g.restore();

  g.save();
  g.shadowColor = "rgba(150,110,255,0.65)";
  g.shadowBlur = 0.4 * S;
  g.lineWidth = 0.1;
  g.strokeStyle = "rgba(141,108,240,0.9)";
  g.strokeRect(PODIUM.x0, PODIUM.z0, PODIUM.x1 - PODIUM.x0, PODIUM.z1 - PODIUM.z0);
  g.restore();
  g.lineWidth = 0.03;
  g.strokeStyle = LAV_HI;
  g.strokeRect(PODIUM.x0, PODIUM.z0, PODIUM.x1 - PODIUM.x0, PODIUM.z1 - PODIUM.z0);

  /* soft elliptical fade at the sheet edges */
  g.save();
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.globalCompositeOperation = "destination-in";
  g.translate(PW / 2, PH / 2);
  g.scale(1, PH / PW);
  const fade = g.createRadialGradient(0, 0, 0, 0, 0, PW / 2);
  fade.addColorStop(0, "rgba(0,0,0,1)");
  fade.addColorStop(0.88, "rgba(0,0,0,1)");
  fade.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = fade;
  g.fillRect(-PW / 2, -PW / 2, PW, PW);
  g.restore();

  const t = new THREE.CanvasTexture(cv);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = Math.min(8, maxAniso);
  return t;
}

/* ==========================================================
   SITE-VIEW FRAMES: one-point-perspective concrete interiors
   ========================================================== */
function drawSiteView(canvas, kind) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = 360, H = 224;
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  const g = canvas.getContext("2d");
  if (!g) return;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const rnd = mulberry32(kind * 7919 + 13);

  const cx = W * 0.5, cy = H * 0.46, f = W * 0.78, eye = 1.55;
  const P = (x, y, z) => [cx + f * x / z, cy - f * (y - eye) / z];
  const poly = (pts, fill) => {
    g.beginPath();
    pts.forEach((p, i) => (i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1])));
    g.closePath();
    g.fillStyle = fill;
    g.fill();
  };
  const grad = (a, b, stops) => {
    const gr = g.createLinearGradient(a[0], a[1], b[0], b[1]);
    stops.forEach((s) => gr.addColorStop(s[0], s[1]));
    return gr;
  };
  const line = (a, b, col, lw) => {
    g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]);
    g.strokeStyle = col; g.lineWidth = lw || 1; g.stroke();
  };

  const R = kind === 1
    ? { hw: 3.2, h: 3.1, zn: 1.1, zf: 13 }
    : { hw: 2.6, h: 3.0, zn: 1.1, zf: 17 };
  const hw = R.hw, h = R.h, zn = R.zn, zf = R.zf;
  const WL = (x, z0, z1, y0, y1) => [P(x, y0, z0), P(x, y0, z1), P(x, y1, z1), P(x, y1, z0)];

  g.fillStyle = "#26242b";
  g.fillRect(0, 0, W, H);

  poly([P(-hw, 0, zf), P(hw, 0, zf), P(hw, h, zf), P(-hw, h, zf)], "#8c8588");
  poly([P(-hw, 0, zn), P(hw, 0, zn), P(hw, 0, zf), P(-hw, 0, zf)],
    grad(P(0, 0, zn), P(0, 0, zf), [[0, "#3f3b43"], [1, "#8d8688"]]));
  poly([P(-hw, h, zn), P(hw, h, zn), P(hw, h, zf), P(-hw, h, zf)],
    grad(P(0, h, zn), P(0, h, zf), [[0, "#4a464d"], [1, "#7b757a"]]));
  poly(WL(-hw, zn, zf, 0, h),
    grad(P(-hw, 1.5, zn), P(-hw, 1.5, zf), [[0, "#59545a"], [1, "#8f8789"]]));
  poly(WL(hw, zn, zf, 0, h),
    grad(P(hw, 1.5, zn), P(hw, 1.5, zf), [[0, "#6b666b"], [1, "#9b9391"]]));

  /* formwork seams and tie holes */
  [1.05, 2.1].forEach((y) => {
    line(P(-hw, y, zn), P(-hw, y, zf), "rgba(0,0,0,0.16)", 1);
    line(P(hw, y, zn), P(hw, y, zf), "rgba(0,0,0,0.16)", 1);
    line(P(-hw, y, zf), P(hw, y, zf), "rgba(0,0,0,0.12)", 1);
  });
  for (let z = zn + 1.3; z < zf; z += 2.4) {
    line(P(-hw, 0, z), P(-hw, h, z), "rgba(0,0,0,0.10)", 1);
    line(P(hw, 0, z), P(hw, h, z), "rgba(0,0,0,0.10)", 1);
  }
  for (let x = -hw + 0.8; x < hw; x += 1.6) line(P(x, 0, zf), P(x, h, zf), "rgba(0,0,0,0.10)", 1);
  g.fillStyle = "rgba(0,0,0,0.25)";
  for (let x = -hw + 0.4; x < hw; x += 0.8) {
    [0.5, 1.5, 2.5].forEach((y) => { const p = P(x, y, zf); g.fillRect(p[0] - 0.6, p[1] - 0.6, 1.2, 1.2); });
  }

  const elems = [];
  const drawBeam = (z, d) => {
    const by = h - 0.5;
    poly([P(-hw, by, z + d), P(hw, by, z + d), P(hw, by, z), P(-hw, by, z)], "#5f5a60");
    poly([P(-hw, by, z), P(hw, by, z), P(hw, h, z), P(-hw, h, z)], "#8a8386");
  };
  const drawCol = (x, z, w, hgt, lit) => {
    const x0 = x - w / 2, x1 = x + w / 2;
    const sx = Math.max(-hw + 0.05, x0 - 1.7);
    poly([P(x0, 0, z), P(sx, 0, z + 0.3), P(sx, 0, z + 0.3 + w), P(x0, 0, z + w)], "rgba(10,8,14,0.28)");
    if (x < 0) {
      poly([P(x1, 0, z), P(x1, 0, z + w), P(x1, hgt, z + w), P(x1, hgt, z)], lit ? "#bdb3ad" : "#7d767a");
    } else {
      poly([P(x0, 0, z), P(x0, 0, z + w), P(x0, hgt, z + w), P(x0, hgt, z)], lit ? "#a79d98" : "#4f4a50");
    }
    poly([P(x0, 0, z), P(x1, 0, z), P(x1, hgt, z), P(x0, hgt, z)], x < 0 ? "#9a9294" : "#837c80");
  };

  if (kind === 1) {
    /* window on the east wall, doorway at the back, recess on the west wall */
    poly(WL(hw, 4, 8.5, 0.9, 2.5),
      grad(P(hw, 2.5, 6), P(hw, 0.9, 6), [[0, "#e6eef3"], [1, "#f1dfc4"]]));
    line(P(hw, 0.9, 6.25), P(hw, 2.5, 6.25), "#9d9490", 2);
    poly(WL(hw, 4, 8.5, 0.82, 0.9), "#c7bdb6");
    poly([P(hw, 0, 4.2), P(hw, 0, 8.4), P(0.6, 0, 10.6), P(0.6, 0, 6.4)], "rgba(255,236,205,0.20)");
    poly(WL(-hw, 9, 10.6, 0, 2.2), "#1f1d23");
    poly([P(-0.8, 0, zf), P(0.8, 0, zf), P(0.8, 2.2, zf), P(-0.8, 2.2, zf)],
      grad(P(0, 2.2, zf), P(0, 0, zf), [[0, "#25232a"], [1, "#34313a"]]));
    poly([P(-0.25, 0.9, zf), P(0.25, 0.9, zf), P(0.25, 1.9, zf), P(-0.25, 1.9, zf)], "rgba(190,204,214,0.75)");
    elems.push({ z: 8.6, fn: () => drawCol(1.9, 8.6, 0.5, h, false) });
    elems.push({ z: 6.5, fn: () => drawBeam(6.5, 0.6) });
    elems.push({ z: 5.4, fn: () => drawCol(-1.6, 5.4, 0.5, h, true) });
  } else {
    /* corridor with bright opening, side doorways and pilasters */
    const ox0 = P(-1.6, 0, zf)[0], ox1 = P(1.6, 0, zf)[0];
    const oy0 = P(0, 2.6, zf)[1], oy1 = P(0, 0, zf)[1];
    g.save();
    g.beginPath(); g.rect(ox0, oy0, ox1 - ox0, oy1 - oy0); g.clip();
    g.fillStyle = grad([0, oy0], [0, cy], [[0, "#dfe9ef"], [1, "#f2dcc0"]]);
    g.fillRect(ox0, oy0, ox1 - ox0, cy - oy0);
    g.fillStyle = grad([0, cy], [0, oy1], [[0, "#8b8878"], [1, "#5a5a50"]]);
    g.fillRect(ox0, cy, ox1 - ox0, oy1 - cy);
    g.fillStyle = "rgba(120,134,146,0.55)";
    for (let i = 0; i < 7; i++) {
      const bw = 6 + rnd() * 10, bh = 3 + rnd() * 7;
      g.fillRect(ox0 + rnd() * (ox1 - ox0 - bw), cy - bh, bw, bh);
    }
    g.restore();
    poly([P(-0.17, 0, zf), P(0.17, 0, zf), P(0.17, 2.6, zf), P(-0.17, 2.6, zf)], "#8a8386");
    poly(WL(-hw, 4.5, 6, 0, 2.2), "#242229");
    poly(WL(-hw, 9, 10.5, 0, 2.2), "#2b2830");
    poly([P(-1.5, 0, zf), P(1.5, 0, zf), P(1.0, 0, 9), P(-1.0, 0, 9)], "rgba(255,232,196,0.22)");
    [3.6, 7, 10.4, 13.8].forEach((z) => {
      elems.push({ z: z, fn: () => {
        const x0 = hw - 0.35;
        poly([P(x0, 0, z), P(x0, 0, z + 0.5), P(x0, h, z + 0.5), P(x0, h, z)], "#6f696e");
        poly([P(x0, 0, z), P(hw, 0, z), P(hw, h, z), P(x0, h, z)], "#8f888b");
        drawBeam(z, 0.4);
      } });
    });
  }

  elems.sort((a, b) => b.z - a.z).forEach((e) => e.fn());

  /* grain, tint, vignette */
  for (let i = 0; i < 1500; i++) {
    g.fillStyle = rnd() > 0.5 ? "rgba(255,255,255,0.035)" : "rgba(0,0,0,0.05)";
    g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 2, 1 + rnd() * 2);
  }
  g.fillStyle = "rgba(120,90,150,0.07)";
  g.fillRect(0, 0, W, H);
  const vg = g.createRadialGradient(cx, cy, H * 0.2, cx, cy, W * 0.7);
  vg.addColorStop(0, "rgba(0,0,0,0)");
  vg.addColorStop(1, "rgba(10,8,14,0.55)");
  g.fillStyle = vg;
  g.fillRect(0, 0, W, H);
}

/* ==========================================================
   HERO INSTANCE
   ========================================================== */
function createHero(host) {
  host.querySelectorAll(".living-stage").forEach((el) => el.remove());

  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stage = document.createElement("div");
  stage.className = "living-stage";
  stage.innerHTML = `
    <svg class="living-leaders" aria-hidden="true">
      <path class="leader-1" pathLength="1"></path>
      <path class="leader-2" pathLength="1"></path>
      <circle class="dot-1" r="3.2"></circle>
      <circle class="dot-2" r="3.2"></circle>
    </svg>
    <div class="living-card living-card-one">
      <div class="living-card-label"><span>SITE VIEW / 01</span><i></i></div>
      <canvas class="living-view" aria-hidden="true"></canvas>
    </div>
    <div class="living-card living-card-two">
      <div class="living-card-label"><span>SITE VIEW / 02</span><i></i></div>
      <canvas class="living-view" aria-hidden="true"></canvas>
    </div>
    <div class="living-caption">
      PLAN &rarr; WALKTHROUGH &rarr; <strong>EXPLORE</strong>
      <div>ILLUSTRATIVE SPATIAL STUDY</div>
    </div>
    <div class="living-error">
      3D animation unavailable.<br>
      Check your internet connection and refresh.
    </div>`;
  host.appendChild(stage);

  const q = (s) => stage.querySelector(s);
  const cards = [q(".living-card-one"), q(".living-card-two")];
  const viewCanvases = stage.querySelectorAll(".living-view");
  drawSiteView(viewCanvases[0], 1);
  drawSiteView(viewCanvases[1], 2);

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  } catch (err) {
    q(".living-error").style.display = "flex";
    return { dispose() { stage.remove(); } };
  }
  const small = window.innerWidth < 700;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.75));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.classList.add("living-gl");
  stage.prepend(renderer.domElement);

  const maxAniso = renderer.capabilities.getMaxAnisotropy();
  const disposables = [];
  const track = (o) => { disposables.push(o); return o; };

  /* ---- scene, camera (orthographic isometric) ---- */
  const scene = new THREE.Scene();
  const target = new THREE.Vector3(CENTER.x, 0, CENTER.z);
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 220);
  camera.position.copy(target).addScaledVector(new THREE.Vector3(1, 1.2, 1).normalize(), 90);
  camera.lookAt(target);
  camera.updateMatrixWorld(true);

  /* ---- lighting ---- */
  scene.add(new THREE.HemisphereLight(0xcfc3ff, 0x2a2433, 1.5));
  const key = new THREE.DirectionalLight(0xffe2c2, 3.2);
  key.position.set(CENTER.x + 6, 14, CENTER.z + 16);
  key.target.position.copy(target);
  key.castShadow = true;
  key.shadow.mapSize.set(small ? 1024 : 2048, small ? 1024 : 2048);
  key.shadow.camera.left = -20; key.shadow.camera.right = 20;
  key.shadow.camera.top = 20; key.shadow.camera.bottom = -20;
  key.shadow.camera.near = 1; key.shadow.camera.far = 70;
  key.shadow.bias = -0.0004;
  key.shadow.normalBias = 0.03;
  scene.add(key, key.target);
  const fill = new THREE.DirectionalLight(0x8f7bff, 1.3);
  fill.position.set(CENTER.x + 22, 7, CENTER.z + 3);
  fill.target.position.copy(target);
  scene.add(fill, fill.target);

  const markerLight = new THREE.PointLight(0xff9d5c, 9, 8, 2);
  scene.add(markerLight);

  /* ---- materials ---- */
  const concreteTex = track(makeConcreteTexture(maxAniso));
  const concreteMat = track(new THREE.MeshStandardMaterial({
    map: concreteTex, vertexColors: true, roughness: 0.95, metalness: 0
  }));
  const slabMat = track(new THREE.MeshStandardMaterial({ color: 0x2b2836, roughness: 0.92, metalness: 0 }));
  const glowTex = track(makeGlowTexture());

  /* ---- podium slab ---- */
  const slabGeo = track(new THREE.BoxGeometry(PODIUM.x1 - PODIUM.x0, 0.35, PODIUM.z1 - PODIUM.z0));
  const slab = new THREE.Mesh(slabGeo, slabMat);
  slab.position.set((PODIUM.x0 + PODIUM.x1) / 2, -0.175, (PODIUM.z0 + PODIUM.z1) / 2);
  slab.receiveShadow = true;
  scene.add(slab);

  /* ---- blueprint plane (same metres as the walls) ---- */
  const planTex = track(buildPlanTexture(maxAniso));
  const planGeo = track(new THREE.PlaneGeometry(PLAN.w, PLAN.h));
  planGeo.rotateX(-Math.PI / 2);
  const planMat = track(new THREE.MeshBasicMaterial({
    map: planTex, transparent: true, depthWrite: false, toneMapped: false
  }));
  const planMesh = new THREE.Mesh(planGeo, planMat);
  planMesh.position.set(PLAN.x0 + PLAN.w / 2, 0.012, PLAN.z0 + PLAN.h / 2);
  planMesh.renderOrder = 1;
  scene.add(planMesh);

  /* ---- room glow (warm bounce) ---- */
  const rooms = ROOMS.map((r) => {
    const geo = track(new THREE.PlaneGeometry(r.x1 - r.x0, r.z1 - r.z0));
    geo.rotateX(-Math.PI / 2);
    const mat = track(new THREE.MeshBasicMaterial({
      map: glowTex, color: r.col, transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
    }));
    const m = new THREE.Mesh(geo, mat);
    m.position.set((r.x0 + r.x1) / 2, 0.022, (r.z0 + r.z1) / 2);
    m.renderOrder = 2;
    scene.add(m);
    return Object.assign({ mat: mat, cur: 0 }, r);
  });

  /* ---- walls and columns (rise from their base) ---- */
  const risers = [];
  const zoneCount = {};
  const addRiser = (mesh, zone, H) => {
    const k = zoneCount[zone] = (zoneCount[zone] || 0) + 1;
    risers.push({
      mesh: mesh, zone: zone, foot: Math.min(1, 0.1 / H),
      t0: ZONE_T0[zone] + ((k - 1) % 5) * 0.14, dur: 2.1
    });
    mesh.scale.y = Math.min(1, 0.1 / H);
  };

  WALLS.forEach((w, i) => {
    const geo = track(buildWallGeometry(w, i));
    const mesh = new THREE.Mesh(geo, concreteMat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    scene.add(mesh);
    addRiser(mesh, w.zone, w.h);
  });

  const padGeo = track(new THREE.BoxGeometry(1, 0.35, 1));
  COLUMNS.forEach((c) => {
    const geo = track(buildColumnGeometry(c));
    const mesh = new THREE.Mesh(geo, concreteMat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    scene.add(mesh);
    addRiser(mesh, c.zone, c.h);
    if (c.pad) {
      const pad = new THREE.Mesh(padGeo, slabMat);
      pad.position.set(c.x, -0.175, c.z);
      pad.receiveShadow = true;
      scene.add(pad);
    }
  });

  /* ---- route: dashed orange path drawn progressively ---- */
  const dashPos = [];
  const dashEnd = [];
  (function () {
    let acc = 0;
    const y = 0.045, hw = 0.075;
    for (let i = 0; i < ROUTE.length - 1; i++) {
      const a = ROUTE[i], b = ROUTE[i + 1];
      const dx = b[0] - a[0], dz = b[1] - a[1];
      const len = Math.hypot(dx, dz);
      const ux = dx / len, uz = dz / len, nx = -uz, nz = ux;
      for (let d = 0.08; d + 0.55 <= len; d += 0.95) {
        const p0 = [a[0] + ux * d, a[1] + uz * d];
        const p1 = [a[0] + ux * (d + 0.55), a[1] + uz * (d + 0.55)];
        const qd = [
          [p0[0] + nx * hw, p0[1] + nz * hw], [p0[0] - nx * hw, p0[1] - nz * hw],
          [p1[0] - nx * hw, p1[1] - nz * hw], [p1[0] + nx * hw, p1[1] + nz * hw]
        ];
        [0, 1, 2, 0, 2, 3].forEach((k) => dashPos.push(qd[k][0], y, qd[k][1]));
        dashEnd.push(acc + d + 0.55);
      }
      acc += len;
    }
  })();
  const dashGeo = track(new THREE.BufferGeometry());
  dashGeo.setAttribute("position", new THREE.Float32BufferAttribute(dashPos, 3));
  const dashMat = track(new THREE.MeshBasicMaterial({
    color: 0xff8a3c, side: THREE.DoubleSide, transparent: true, toneMapped: false
  }));
  const dashMesh = new THREE.Mesh(dashGeo, dashMat);
  dashMesh.frustumCulled = false;
  dashMesh.renderOrder = 3;
  dashGeo.setDrawRange(0, 0);
  scene.add(dashMesh);

  /* ---- route markers ---- */
  const orangeMat = track(new THREE.MeshBasicMaterial({ color: 0xff8a4c, toneMapped: false }));
  const hotMat = track(new THREE.MeshBasicMaterial({ color: 0xffd2a8, toneMapped: false }));
  const ringMat = track(new THREE.MeshBasicMaterial({
    color: 0xff9b68, transparent: true, opacity: 0.75, toneMapped: false, side: THREE.DoubleSide
  }));
  const hazeMat = track(new THREE.MeshBasicMaterial({
    map: glowTex, color: 0xff8a4c, transparent: true, opacity: 0.8,
    blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false
  }));

  function makeMarker(r) {
    const grp = new THREE.Group();
    const disc = track(new THREE.CircleGeometry(r, 48)); disc.rotateX(-Math.PI / 2);
    const core = track(new THREE.CircleGeometry(r * 0.52, 40)); core.rotateX(-Math.PI / 2);
    const ringG = track(new THREE.RingGeometry(r * 1.28, r * 1.46, 56)); ringG.rotateX(-Math.PI / 2);
    const haze = track(new THREE.PlaneGeometry(r * 7, r * 7)); haze.rotateX(-Math.PI / 2);
    const m1 = new THREE.Mesh(disc, orangeMat); m1.position.y = 0.06; m1.renderOrder = 4;
    const m2 = new THREE.Mesh(core, hotMat);    m2.position.y = 0.065; m2.renderOrder = 5;
    const m3 = new THREE.Mesh(ringG, ringMat);  m3.position.y = 0.062; m3.renderOrder = 4;
    const m4 = new THREE.Mesh(haze, hazeMat);   m4.position.y = 0.05; m4.renderOrder = 3;
    grp.add(m4, m1, m3, m2);
    grp.userData.ring = m3;
    scene.add(grp);
    return grp;
  }

  const walker = makeMarker(0.42);
  const stations = [
    { grp: makeMarker(0.3), at: ROUTE[0],    t: T_INTRO },
    { grp: makeMarker(0.3), at: STATION_1,   t: T_S1 },
    { grp: makeMarker(0.3), at: STATION_2,   t: T_END }
  ];
  stations.forEach((s) => { s.grp.position.set(s.at[0], 0, s.at[1]); s.grp.scale.setScalar(0.001); });
  walker.scale.setScalar(0.001);

  /* ---- leader lines (2D overlay driven by the 3D anchors) ---- */
  const svg = q(".living-leaders");
  const leaders = [
    { path: q(".leader-1"), dot: q(".dot-1"), card: cards[0], anchor: STATION_1, t: T_S1 + 0.15 },
    { path: q(".leader-2"), dot: q(".dot-2"), card: cards[1], anchor: STATION_2, t: T_END + 0.15 }
  ];
  const tmpV = new THREE.Vector3();
  function project(x, y, z, w, h) {
    tmpV.set(x, y, z).project(camera);
    return { x: (tmpV.x * 0.5 + 0.5) * w, y: (-tmpV.y * 0.5 + 0.5) * h };
  }

  /* ---- layout: camera fit + leader geometry ---- */
  const fwd = new THREE.Vector3();
  const rightV = new THREE.Vector3();
  const upV = new THREE.Vector3();

  function fitCamera(aspect) {
    fwd.subVectors(target, camera.position).normalize();
    rightV.crossVectors(fwd, new THREE.Vector3(0, 1, 0)).normalize();
    upV.crossVectors(rightV, fwd).normalize();
    let minR = 1e9, maxR = -1e9, minU = 1e9, maxU = -1e9;
    [FIT.x0, FIT.x1].forEach((x) => [FIT.y0, FIT.y1].forEach((y) => [FIT.z0, FIT.z1].forEach((z) => {
      tmpV.set(x, y, z).sub(target);
      const r = tmpV.dot(rightV), u = tmpV.dot(upV);
      minR = Math.min(minR, r); maxR = Math.max(maxR, r);
      minU = Math.min(minU, u); maxU = Math.max(maxU, u);
    })));
    const pad = 1.0;
    let halfW = ((maxR - minR) / 2) * pad;
    let halfH = ((maxU - minU) / 2) * pad;
    if (halfW / halfH > aspect) halfH = halfW / aspect; else halfW = halfH * aspect;
    const cR = (minR + maxR) / 2, cU = (minU + maxU) / 2;
    camera.left = cR - halfW; camera.right = cR + halfW;
    camera.top = cU + halfH;  camera.bottom = cU - halfH;
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
  }

  function layout() {
    const w = stage.clientWidth, h = stage.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    fitCamera(w / h);
    svg.setAttribute("viewBox", "0 0 " + w + " " + h);

    leaders.forEach((L) => {
      const c = L.card;
      const x = c.offsetLeft, y = c.offsetTop, cw = c.offsetWidth, ch = c.offsetHeight;
      const a = project(L.anchor[0], 0.06, L.anchor[1], w, h);
      let sx, sy, d;
      if (a.x > x + cw) {
        sx = x + cw; sy = y + ch * 0.5;
        d = "M" + sx + " " + sy + " L" + (sx + 26) + " " + sy + " L" + a.x + " " + a.y;
      } else if (a.x < x) {
        sx = x; sy = y + ch * 0.5;
        d = "M" + sx + " " + sy + " L" + (sx - 26) + " " + sy + " L" + a.x + " " + a.y;
      } else if (a.y > y + ch) {
        sx = x + cw * 0.5; sy = y + ch;
        d = "M" + sx + " " + sy + " L" + sx + " " + (sy + 22) + " L" + a.x + " " + a.y;
      } else {
        sx = x + cw * 0.5; sy = y;
        d = "M" + sx + " " + sy + " L" + sx + " " + (sy - 22) + " L" + a.x + " " + a.y;
      }
      L.path.setAttribute("d", d);
      L.dot.setAttribute("cx", a.x);
      L.dot.setAttribute("cy", a.y);
    });
  }

  /* ---- per-frame state ---- */
  const cardOn = [false, false];
  function setCard(i, on) {
    if (cardOn[i] === on) return;
    cardOn[i] = on;
    cards[i].classList.toggle("active", on);
  }

  function update(t, dt) {
    const reset = smooth((t - T_RESET0) / (T_RESET1 - T_RESET0));
    const live = 1 - reset;
    const s = sAt(Math.min(t, T_END + 1));
    const p = posAt(s);

    risers.forEach((r) => {
      const k = smooth((t - r.t0) / r.dur) * live;
      r.mesh.scale.y = Math.max(0.001, lerp(r.foot, 1, k));
    });

    const zoneK = {};
    Object.keys(ZONE_T0).forEach((z) => { zoneK[z] = smooth((t - ZONE_T0[z]) / 1.6) * live; });

    const intro = smooth((t - T_INTRO) / 0.8) * live;
    walker.position.set(p[0], 0, p[1]);
    walker.scale.setScalar(Math.max(0.001, intro));
    markerLight.position.set(p[0], 1.1, p[1]);
    markerLight.intensity = 9 * intro;

    stations.forEach((st) => {
      st.grp.scale.setScalar(Math.max(0.001, smooth((t - st.t) / 0.5) * live));
    });

    let n = 0;
    while (n < dashEnd.length && dashEnd[n] <= s + 0.02) n++;
    if (t < T_MOVE0) n = 0;
    dashGeo.setDrawRange(0, n * 6);
    dashMat.opacity = live;

    rooms.forEach((r) => {
      const inside = intro > 0.5 && p[0] >= r.x0 && p[0] <= r.x1 && p[1] >= r.z0 && p[1] <= r.z1;
      const goal = (0.2 * zoneK[r.zone] + (inside ? 0.55 : 0)) * r.gain;
      r.cur += (goal - r.cur) * (1 - Math.exp(-dt * 5));
      r.mat.opacity = r.cur;
    });

    setCard(0, t >= T_S1 && reset < 0.05);
    setCard(1, t >= T_END + 0.1 && reset < 0.05);
    leaders.forEach((L) => {
      const k = smooth((t - L.t) / 0.9);
      L.path.style.strokeDashoffset = String(1 - k);
      L.path.style.opacity = String(live * (k > 0 ? 1 : 0));
      L.dot.style.opacity = String(live * smooth((t - L.t - 0.6) / 0.3));
    });

    const pulse = reduced ? 1 : 1 + 0.09 * Math.sin(t * 4);
    walker.userData.ring.scale.set(pulse, 1, pulse);
  }

  /* ---- single render loop ---- */
  let raf = 0, running = false, last = 0, clock = 0;
  function frame(ts) {
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.05, Math.max(0.001, (ts - last) / 1000));
    last = ts;
    clock = (clock + dt) % LOOP;
    update(clock, dt);
    renderer.render(scene, camera);
  }
  function start() {
    if (running || reduced) return;
    running = true;
    last = performance.now();
    raf = requestAnimationFrame(frame);
  }
  function stop() {
    running = false;
    cancelAnimationFrame(raf);
  }
  function drawStatic() {
    update(STATIC_T, 1);
    renderer.render(scene, camera);
  }

  const resizeObs = new ResizeObserver(() => {
    layout();
    if (!running) drawStatic();
  });
  resizeObs.observe(stage);

  const visObs = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) start(); else stop(); });
  }, { threshold: 0.05 });
  visObs.observe(host);

  const onVis = () => { if (document.hidden) stop(); else if (!reduced) start(); };
  document.addEventListener("visibilitychange", onVis);

  layout();
  if (reduced) drawStatic(); else { update(0, 1); renderer.render(scene, camera); start(); }

  function dispose() {
    stop();
    resizeObs.disconnect();
    visObs.disconnect();
    document.removeEventListener("visibilitychange", onVis);
    disposables.forEach((o) => { if (o && o.dispose) o.dispose(); });
    renderer.dispose();
    if (renderer.forceContextLoss) renderer.forceContextLoss();
    stage.remove();
  }
  window.addEventListener("pagehide", dispose, { once: true });
  return { dispose: dispose };
}

/* ==========================================================
   MOUNT (kept last so every constant above is initialised)
   ========================================================== */
if (window.__ucHero && typeof window.__ucHero.dispose === "function") {
  try { window.__ucHero.dispose(); } catch (e) { /* ignore */ }
  window.__ucHero = null;
}
const heroHost = document.getElementById("hero-art");
if (heroHost) {
  try {
    window.__ucHero = createHero(heroHost);
  } catch (err) {
    console.error("[under-construction hero]", err);
  }
}
</script>

<!-- UC_STEP114_END -->



<!-- UC_STEP117_UI_START -->
<style>
/* Cleaner, quieter workspace presentation */

.workspace .status {
  color: #b9a6ed;
  border-color: #554b70;
  background: #b89aff0b;
  font-size: 10px;
  letter-spacing: .07em;
  padding: 8px 12px;
  border-radius: 7px;
}

.workspace .sidebar-bottom {
  display: none;
}

.workspace .notice {
  margin-top: 18px;
  padding: 12px 15px;
  border: 1px solid #343542;
  border-left: 1px solid #343542;
  border-radius: 8px;
  background: #1b1c25;
  color: #9898aa;
  font-size: 11px;
  line-height: 1.65;
}

.workspace .inspector {
  border-color: #353543;
  padding: 20px;
}

.workspace .inspector h3 {
  font-size: 14px;
  font-weight: 650;
}

.workspace .inspector-data {
  grid-template-columns:
    repeat(3, minmax(85px, 1fr));
}

.workspace .viewer-foot {
  color: #a4a0b5;
  font-size: 10px;
  letter-spacing: .025em;
}

.workspace .project.active {
  background: #292538;
  border-color: #685586;
}

.workspace .viewer {
  box-shadow: 0 8px 24px #00000013;
}

@media(max-width:700px) {
  .workspace .inspector-data {
    grid-template-columns: repeat(2, minmax(85px, 1fr));
  }
}
</style>

<script>
(function () {
  "use strict";

  const get = (id) => document.getElementById(id);

  const status = get("workspace-status");
  const footer = get("cad-footer");
  const inspector = get("inspector-description");

  const notice = document.querySelector(
    ".workspace .notice"
  );

  if (notice) {
    notice.textContent =
      "The floor plan and walkthrough are separate views. " +
      "Construction progress will appear here when analysis is available.";
  }

  const inspectorData = document.querySelector(
    ".workspace .inspector-data"
  );

  if (inspectorData) {
    const fields = inspectorData.children;

    if (fields.length >= 3) {
      const third = fields[2];

      const label = third.querySelector("small");
      const value = third.querySelector("span");

      if (label) label.textContent = "Source";

      if (value) {
        value.textContent = "CAD drawing";
        value.style.color = "var(--ink)";
      }
    }
  }

  const translations = {
    "UNVERIFIED": "READY TO EXPLORE",
    "NOT ANALYZED": "READY TO START",
    "NO DATASET": "SAMPLE UNAVAILABLE",
    "FILES_SAVED_NOT_ANALYZED": "FILES SAVED"
  };

  function tidyStatus() {
    if (!status) return;

    const original = status.textContent.trim();

    if (Object.prototype.hasOwnProperty.call(
      translations, original
    )) {
      status.textContent = translations[original];
    }
  }

  function tidyFooter() {
    if (!footer) return;

    const original = footer.textContent;

    const cleaned = original
      .replace(/\s*\/\s*UNVERIFIED\b/g, "")
      .replace(
        /unsupported or skipped/g,
        "not displayed"
      )
      .replace(
        /drawable CAD entities/g,
        "CAD elements"
      )
      .replace(
        /drawable elements/g,
        "CAD elements"
      );

    if (cleaned !== original) {
      footer.textContent = cleaned;
    }
  }

  function tidyInspector() {
    if (!inspector) return;

    const original = inspector.textContent.trim();

    if (
      original.includes(
        "Its physical location and construction status"
      ) ||
      original.includes(
        "No physical match established"
      )
    ) {
      inspector.textContent =
        "Select another element to explore its CAD details.";
    }
  }

  function update() {
    tidyStatus();
    tidyFooter();
    tidyInspector();
  }

  // Existing viewer code changes these labels as files load.
  // Keep the presentation consistent without changing
  // the original CAD/video logic.
  for (const node of [status, footer, inspector]) {
    if (!node) continue;

    const observer = new MutationObserver(update);

    observer.observe(node, {
      childList: true,
      characterData: true,
      subtree: true
    });
  }

  update();
})();
</script>
<!-- UC_STEP117_UI_END -->


<!-- UC_STEP119C_PLAYER_LAYOUT_START -->
<style>
/* Player layout: explicit rows prevent the camera image from covering controls. */
#viewer-view .viewer-grid > .viewer:nth-child(2) {
  display: grid !important;
  grid-template-columns: minmax(0, 1fr);
  grid-template-rows: 54px 480px auto auto;
  height: auto !important;
  min-height: 0 !important;
  overflow: hidden;
}
#viewer-view .viewer-grid > .viewer:nth-child(2) > .viewer-head {
  grid-row: 1;
}
#viewer-view .viewer-grid > .viewer:nth-child(2) > .video-stage {
  grid-row: 2;
  position: relative !important;
  display: block !important;
  height: 480px !important;
  min-height: 0 !important;
  max-height: 480px !important;
  overflow: hidden !important;
  isolation: isolate;
}
#viewer-view .video-stage > #frame-image,
#viewer-view .video-stage > #local-video {
  position: absolute !important;
  inset: 0 !important;
  width: 100% !important;
  height: 100% !important;
  max-height: 100% !important;
  object-fit: contain !important;
}
#viewer-view #frame-controls {
  grid-row: 3;
  position: relative !important;
  z-index: 10 !important;
  display: flex !important;
  min-height: 64px;
  padding: 13px 16px;
  background: #20212b !important;
  border-top: 1px solid #444052;
  align-items: center;
  gap: 12px;
}
#viewer-view #frame-controls.hidden { display: none !important; }
#viewer-view #frame-slider { flex: 1 1 auto; min-width: 60px; }
#viewer-view .uc-frame-info {
  grid-row: 4;
  position: relative !important;
  z-index: 10 !important;
  background: #20212b !important;
  padding: 0 16px 12px;
}
@media (max-width: 700px) {
  #viewer-view .viewer-grid > .viewer:nth-child(2) {
    grid-template-rows: 54px 340px auto auto;
  }
  #viewer-view .viewer-grid > .viewer:nth-child(2) > .video-stage {
    height: 340px !important;
    max-height: 340px !important;
  }
}
</style>
<!-- UC_STEP119C_PLAYER_LAYOUT_END -->

<!-- UC_STEP118C_START -->
<style>
/* Real sample gallery additions */
.uc-real-sample .project-thumb {
  border-color: #b89aff;
  background:
    linear-gradient(90deg,transparent 47%,#b89aff88 48%,#b89aff88 50%,transparent 51%),
    linear-gradient(transparent 48%,#b89aff88 49%,#b89aff88 51%,transparent 52%),
    #2a2538;
}

.uc-sample-source {
  display: inline-block;
  margin-top: 7px;
  color: #80cbb0;
  font-size: 9px;
  letter-spacing: .045em;
}

/* STEP 119B — Keep the image inside its stage, with controls below. */
.video-stage {
  height: 410px !important;
  min-height: 410px;
  position: relative;
  display: block !important;
  overflow: hidden !important;
  background: #0d0e13;
}

.video-stage > #frame-image,
.video-stage > #local-video {
  position: absolute;
  inset: 0;
  display: block;
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
}

.video-stage > #video-empty {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
}

.video-controls {
  position: relative;
  z-index: 2;
  display: flex !important;
  visibility: visible !important;
  opacity: 1 !important;
  align-items: center;
  gap: 10px;
  min-height: 58px;
  padding: 12px 15px;
  background: #191a22;
  border-top: 1px solid #343542;
}

.video-controls.hidden { display: none !important; }
#frame-slider { flex: 1; min-width: 70px; accent-color: #b89aff; }
#uc-frame-play { flex-shrink: 0; }
.uc-frame-info {
  position: relative;
  z-index: 2;
  display: block;
  min-height: 30px;
  padding: 7px 15px 12px;
  background: #191a22;
}

#uc-frame-play {
  min-width: 62px;
  border-color: #75628e;
  color: #e6d9ff;
}

#frame-counter {
  min-width: 66px;
  text-align: right;
}

.uc-frame-info {
  padding: 0 15px 12px;
  color: #9898aa;
  font-size: 10px;
  line-height: 1.5;
}

#sample-two:disabled {
  opacity: .42;
  cursor: default;
}
</style>

<script>
(function () {
  "use strict";

  const byId = id => document.getElementById(id);

  const sampleButton = byId("sample-one");
  const placeholder = byId("sample-two");
  const controls = byId("frame-controls");

  if (!sampleButton || !controls) return;

  sampleButton.classList.add("uc-real-sample");

  const sampleNote = sampleButton.querySelector(".project-note");
  if (sampleNote) {
    sampleNote.textContent = "Real CAD + 180 camera frames";
  }

  if (!sampleButton.querySelector(".uc-sample-source")) {
    const source = document.createElement("span");
    source.className = "uc-sample-source";
    source.textContent = "REAL DATASET";
    sampleButton.querySelector(".project-name")
      ?.parentElement.appendChild(source);
  }

  if (placeholder) {
    placeholder.disabled = true;
    const name = placeholder.querySelector(".project-name");
    const note = placeholder.querySelector(".project-note");
    if (name) name.textContent = "More samples";
    if (note) note.textContent = "Additional dataset needed";
  }

  const playButton = document.createElement("button");
  playButton.id = "uc-frame-play";
  playButton.className = "tool";
  playButton.type = "button";
  playButton.textContent = "Play";

  controls.insertBefore(playButton, byId("prev-frame"));

  const info = document.createElement("div");
  info.className = "uc-frame-info mono";
  info.textContent = "REAL SITE FRAMES · CAMERA 0";
  controls.insertAdjacentElement("afterend", info);

  let timer = null;
  let loadingToken = 0;
  const FRAME_RATE = 6;

  function stopPlayback() {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
    playButton.textContent = "Play";
  }

  function showRealFrame(index) {
    if (!state.frames.length) return;

    const clamped = Math.max(
      0,
      Math.min(index, state.frames.length - 1)
    );

    state.frameIndex = clamped;

    const frame = state.frames[clamped];
    const image = byId("frame-image");

    image.src = frame.image_url;
    image.classList.remove("hidden");

    byId("video-empty").classList.add("hidden");
    byId("frame-slider").value = clamped;
    window.ucAlignmentUpdate?.(frame.filename);

    setText(
      "frame-counter",
      `${clamped + 1} / ${state.frames.length}`
    );
  }

  function play() {
    if (!state.frames.length) return;

    if (state.frameIndex >= state.frames.length - 1) {
      showRealFrame(0);
    }

    stopPlayback();
    playButton.textContent = "Pause";

    timer = setInterval(() => {
      if (state.frameIndex >= state.frames.length - 1) {
        stopPlayback();
        return;
      }

      showRealFrame(state.frameIndex + 1);
    }, 1000 / FRAME_RATE);
  }

  playButton.onclick = () => {
    if (timer !== null) {
      stopPlayback();
    } else {
      play();
    }
  };

  function leaveSample() {
    window.uc3DHide?.();
    window.ucAlignmentHide?.();
    loadingToken++;
    stopPlayback();
  }

  async function openRealSample() {
    window.uc3DHide?.();
    window.ucAlignmentHide?.();
    const token = ++loadingToken;
    stopPlayback();

    scrollToStudio();

    byId("upload-view").classList.add("hidden");
    byId("viewer-view").classList.remove("hidden");

    sampleButton.classList.add("active");
    placeholder?.classList.remove("active");

    setText("workspace-eyebrow", "SAMPLE / REAL DATASET");
    setText(
      "workspace-title",
      "Floor 2 Construction Walkthrough"
    );
    setText("workspace-status", "READY TO EXPLORE");

    setText("cad-footer", "Loading real floor plan...");
    setText("cad-empty", "Loading floor plan...");

    byId("cad-empty").classList.remove("hidden");
    byId("frame-image").classList.add("hidden");
    byId("local-video").pause();
    byId("local-video").classList.add("hidden");
    byId("video-empty").classList.remove("hidden");
    byId("frame-controls").classList.remove("hidden");

    setText("video-empty", "Loading real camera frames...");

    state.elements = [];
    state.frames = [];
    state.frameIndex = 0;
    state.selected = null;

    setText("inspector-title", "Element inspector");
    setText(
      "inspector-description",
      "Select a CAD element to inspect its drawing details."
    );
    setText("detail-layer", "—");
    setText("detail-type", "—");

    try {
      // The existing local DXF parser already understands
      // LINE, LWPOLYLINE, ARC and other supported geometry.
      const [dxfResponse, framesResponse] = await Promise.all([
        fetch("/api/real-samples/floor-2/dxf"),
        fetch("/api/real-samples/floor-2/frames"),
      ]);

      if (!dxfResponse.ok || !framesResponse.ok) {
        throw new Error(
          "Real sample files could not be loaded."
        );
      }

      const [dxfBlob, frameData] = await Promise.all([
        dxfResponse.blob(),
        framesResponse.json(),
      ]);

      const form = new FormData();
      form.append(
        "file",
        dxfBlob,
        "floor_2.dxf"
      );

      const parsedResponse = await fetch("/api/local/dxf", {
        method: "POST",
        body: form,
      });

      const parsed = await parsedResponse.json();

      if (!parsedResponse.ok) {
        throw new Error(
          parsed.detail || parsed.error || "DXF parsing failed."
        );
      }

      if (token !== loadingToken) return;

      state.elements = parsed.elements || [];
      state.frames = frameData.frames || [];

      renderCAD();
      window.uc3DShow?.(state.elements);

      const skipped = Object.values(parsed.skipped || {})
        .reduce((total, value) => total + Number(value || 0), 0);

      setText(
        "cad-footer",
        `${state.elements.length.toLocaleString()} CAD elements displayed` +
        (skipped ? ` · ${skipped} not displayed` : "")
      );

      byId("frame-slider").min = 0;
      byId("frame-slider").max = Math.max(
        0,
        state.frames.length - 1
      );
      byId("frame-slider").value = 0;

      if (state.frames.length) {
        window.ucAlignmentLoad?.(state.frames);
        showRealFrame(0);
        info.textContent =
          `REAL SITE FRAMES · CAMERA 0 · ${state.frames.length} IMAGES`;
      } else {
        setText(
          "video-empty",
          "No camera frames available."
        );
      }

    } catch (error) {
      if (token !== loadingToken) return;

      console.error("Real sample:", error);

      state.elements = [];
      state.frames = [];
      renderCAD();

      setText(
        "cad-footer",
        "Could not load sample: " + error.message
      );
      setText(
        "video-empty",
        "The real sample could not be opened."
      );
    }
  }

  // Override only the sample navigation handlers.
  // Existing uploads, CAD interaction and hero remain intact.
  sampleButton.onclick = openRealSample;

  const heroSample = byId("hero-sample");
  if (heroSample) heroSample.onclick = openRealSample;

  const navSamples = byId("nav-samples");
  if (navSamples) {
    navSamples.onclick = event => {
      event.preventDefault();
      openRealSample();
    };
  }

  for (const id of [
    "hero-own",
    "new-project",
    "replace-project",
    "open-uploads",
  ]) {
    const button = byId(id);
    if (!button) continue;

    button.addEventListener("click", leaveSample);
  }

  const previous = byId("prev-frame");
  const next = byId("next-frame");
  const slider = byId("frame-slider");

  if (previous) {
    previous.onclick = () => {
      stopPlayback();
      showRealFrame(state.frameIndex - 1);
    };
  }

  if (next) {
    next.onclick = () => {
      stopPlayback();
      showRealFrame(state.frameIndex + 1);
    };
  }

  if (slider) {
    slider.oninput = event => {
      stopPlayback();
      showRealFrame(Number(event.target.value));
    };
  }

  // Stop playback when the browser tab is hidden.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stopPlayback();
  });
})();
</script>
<!-- UC_STEP118C_END -->


<!-- UC_STEP124_3D_SCRIPT_START -->
<script>
(() => {
  "use strict";
  const panel = document.getElementById("uc-3d-panel");
  const button = document.getElementById("uc-3d-open");
  window.uc3DShow = elements => panel.classList.toggle("hidden", !Array.isArray(elements) || !elements.length);
  window.uc3DHide = () => { panel.classList.add("hidden"); document.getElementById("uc-3d-workspace").classList.add("hidden"); };
  button.addEventListener("click", () => {
    // The corrected 3D viewer is served directly by this dashboard.
    const workspace = document.getElementById("uc-3d-workspace");
    const frame = document.getElementById("uc-3d-frame");
    if (!frame.getAttribute("src")) frame.setAttribute("src", "/embedded-twin");
    workspace.classList.remove("hidden");
    button.textContent = "3D open ✓";
    workspace.scrollIntoView({behavior:"smooth",block:"start"});
  });
  document.getElementById("uc-3d-back").addEventListener("click", () => {
    document.getElementById("uc-3d-workspace").classList.add("hidden");
    button.textContent = "Open 3D →";
    document.getElementById("cad-stage").scrollIntoView({behavior:"smooth",block:"center"});
  });
  if (typeof state !== "undefined" && Array.isArray(state.elements) && state.elements.length && document.getElementById("sample-one")?.classList.contains("active")) window.uc3DShow(state.elements);
})();
</script>
<!-- UC_STEP124_3D_SCRIPT_END -->

<!-- UC_STEP123_ALIGNMENT_SCRIPT_START -->
<script>
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const controls = $("uc-align-controls"), layer = $("uc-align-layer");
  const select = $("uc-align-select"), toggle = $("uc-align-visible");
  const track = $("uc-align-track"), travelled = $("uc-align-travelled");
  const marker = $("uc-align-marker"), note = $("uc-align-note");
  let poses = new Map(), candidates = [], current = "", epoch = 0;
  let activePoints = [], selected = 0;
  const finite = v => typeof v === "number" && Number.isFinite(v);
  const valid = f => f?.pose_available === true &&
    finite(f.position?.x) && finite(f.position?.y);
  const project = (pose, candidate) => {
    const theta = candidate.angle_degrees * Math.PI / 180;
    const c = Math.cos(theta), s = Math.sin(theta);
    const [tx,ty] = candidate.translation;
    const x = pose.position.x, y = pose.position.y;
    // The original registration uses points @ rotation.T + translation.
    // Existing CAD renderer maps CAD (x,y) to SVG (x,-y).
    return [x*c-y*s+tx, -(x*s+y*c+ty)];
  };
  const path = pts => pts.map((p,i) =>
    `${i ? "L" : "M"}${p.x.toFixed(4)},${p.y.toFixed(4)}`).join(" ");
  function hideMarker(){marker.setAttribute("visibility","hidden");travelled.setAttribute("d","");}
  function draw(){
    const candidate = candidates[selected];
    activePoints = candidate ? [...poses.values()].filter(valid)
      .sort((a,b) => a.index-b.index)
      .map(f => { const [x,y] = project(f,candidate); return {filename:f.filename,index:f.index,x,y}; }) : [];
    track.setAttribute("d",path(activePoints));
    layer.setAttribute("visibility",candidate && toggle.checked ? "visible":"hidden");
    hideMarker();
    if(candidate){
      note.textContent = `UNVERIFIED · ${candidate.angle_degrees.toFixed(2)}° · containment ${(candidate.inside_fraction*100).toFixed(0)}%`;
      window.ucAlignmentUpdate?.(current);
    }
  }
  window.ucAlignmentHide = () => {
    epoch++; poses.clear(); candidates=[]; activePoints=[]; current="";
    controls.classList.add("hidden");layer.setAttribute("visibility","hidden");
    track.setAttribute("d","");hideMarker();
  };
  window.ucAlignmentUpdate = filename => {
    current=filename;
    if(!toggle.checked || !candidates[selected]){hideMarker();return;}
    const f=poses.get(filename);
    if(!valid(f)){hideMarker();return;}
    const point=activePoints.find(p=>p.filename===filename);
    if(!point){hideMarker();return;}
    marker.setAttribute("cx",point.x);marker.setAttribute("cy",point.y);
    marker.setAttribute("visibility","visible");
    travelled.setAttribute("d",path(activePoints.filter(p=>p.index<=f.index)));
  };
  select.addEventListener("change",()=>{selected=Number(select.value)||0;draw();});
  toggle.addEventListener("change",()=>draw());
  window.ucAlignmentLoad = async frames => {
    const mine=++epoch;
    poses=new Map();candidates=[];activePoints=[];
    controls.classList.remove("hidden");
    note.textContent="LOADING · UNVERIFIED CANDIDATES";
    try {
      const [r1,r2]=await Promise.all([
        fetch("/api/real-samples/floor-2/trajectory"),
        fetch("/api/real-samples/floor-2/alignments")
      ]);
      if(!r1.ok || !r2.ok)throw Error("Trajectory or candidate file unavailable");
      const [trajectory,alignment]=await Promise.all([r1.json(),r2.json()]);
      if(mine!==epoch)return;
      if(alignment.status!=="UNVERIFIED_CANDIDATES" || !Array.isArray(alignment.candidates))
        throw Error("Unexpected alignment data");
      if(!Array.isArray(trajectory.frames))throw Error("Invalid trajectory");
      const filenames=new Set(frames.map(f=>f.filename));
      poses=new Map(trajectory.frames.filter(f=>filenames.has(f.filename)).map(f=>[f.filename,f]));
      candidates=alignment.candidates.filter(c=>finite(c.angle_degrees) &&
        Array.isArray(c.translation) && c.translation.length===2 && c.translation.every(finite));
      if(!candidates.length || ![...poses.values()].some(valid))throw Error("No usable candidates or poses");
      select.replaceChildren();
      candidates.forEach((c,i)=>{
        const option=document.createElement("option");option.value=String(i);
        option.textContent=`Candidate ${String(i+1).padStart(2,"0")} · ${c.angle_degrees.toFixed(1)}°`;
        select.appendChild(option);
      });
      const preferredIndex = candidates.findIndex(
  c => Math.abs(c.angle_degrees - 317.9689499733972) < 0.001
);

selected = preferredIndex >= 0 ? preferredIndex : 0;
select.value = String(selected);
toggle.checked = true;
      current=state.frames[state.frameIndex]?.filename || frames[0]?.filename || "";
      draw();
    }catch(error){
      if(mine!==epoch)return;
      note.textContent="ALIGNMENT UNAVAILABLE · "+error.message;
      layer.setAttribute("visibility","hidden");
      select.replaceChildren();
    }
  };
})();
</script>
<!-- UC_STEP123_ALIGNMENT_SCRIPT_END -->



</body>
</html>
"""



# UC_STEP118C_ROUTES
@app.get("/api/real-samples/floor-2")
def real_sample_metadata():
    from pathlib import Path
    from flask import jsonify

    folder = (
        Path(__file__).resolve().parents[1]
        / "sample_gallery"
        / "real_floor_2"
    )

    path = folder / "project.json"
    if not path.is_file():
        return jsonify({
            "error": "Real sample has not been prepared."
        }), 404

    return Response(
        path.read_bytes(),
        mimetype="application/json",
    )


@app.get("/api/real-samples/floor-2/frames")
def real_sample_frames():
    from pathlib import Path
    from flask import jsonify

    folder = (
        Path(__file__).resolve().parents[1]
        / "sample_gallery"
        / "real_floor_2"
    )

    index_path = folder / "frame_index.json"
    if not index_path.is_file():
        return jsonify({
            "error": "Frame index not found."
        }), 404

    import json
    data = json.loads(index_path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])

    result = []

    for frame in frames:
        filename = Path(
            str(frame.get("filename", ""))
        ).name

        if not filename or filename != frame.get("filename"):
            continue

        if not (folder / "frames" / filename).is_file():
            continue

        result.append({
            "index": len(result),
            "filename": filename,
            "image_url": (
                "/api/real-samples/floor-2/frame/"
                + filename
            ),
            "pose_matched": bool(
                frame.get("pose_matched", False)
            ),
        })

    return jsonify({
        "frames": result,
        "total": len(result),
        "source": "real_dataset",
    })


@app.get("/api/real-samples/floor-2/frame/<filename>")
def real_sample_frame(filename):
    from pathlib import Path
    from flask import abort, send_file

    # Never accept paths or arbitrary filenames.
    if (
        not filename.startswith("cam0_")
        or not filename.endswith(".jpg")
        or not filename[5:-4].isdigit()
    ):
        abort(404)

    folder = (
        Path(__file__).resolve().parents[1]
        / "sample_gallery"
        / "real_floor_2"
        / "frames"
    )

    path = folder / filename
    if not path.is_file():
        abort(404)

    return send_file(
        path,
        mimetype="image/jpeg",
        conditional=True,
    )


@app.get("/api/real-samples/floor-2/dxf")
def real_sample_dxf():
    from pathlib import Path
    from flask import abort, send_file

    path = (
        Path(__file__).resolve().parents[1]
        / "sample_gallery"
        / "real_floor_2"
        / "floor_plan.dxf"
    )

    if not path.is_file():
        abort(404)

    return send_file(
        path,
        mimetype="application/dxf",
        as_attachment=False,
    )




# UC_STEP121B_TRAJECTORY_ROUTE
@app.get("/api/real-samples/floor-2/trajectory")
def real_sample_trajectory():
    from pathlib import Path
    from flask import jsonify

    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "real_samples"
        / "floor_2_camera_trajectory.json"
    )
    if not path.is_file():
        return jsonify({"error": "Trajectory JSON not found."}), 404
    return Response(path.read_bytes(), mimetype="application/json")



# UC_STEP123_ALIGNMENT_ROUTE
@app.get("/api/real-samples/floor-2/alignments")
def real_sample_alignments():
    from pathlib import Path
    from flask import abort, send_file

    root = Path(__file__).resolve().parents[1]
    path = (root / "_archive" / "hilti_transition_20261023_004237"
            / "outputs" / "registration" / "candidate_alignments.json")
    if not path.is_file():
        abort(404, description="Registration candidate file not found")
    return send_file(path, mimetype="application/json", conditional=True)

@app.get("/")
def index():
    from dashboard_upload_bridge import enhance_dashboard_html
    from sample_three_storey import enhance_sample_html
    from sample_dashboard_layout import enhance_sample_layout
    from sample_floor_inspection import enhance_floor_inspection
    from registration_review_panel import enhance_registration_review

    html = enhance_dashboard_html(HTML)
    html = enhance_sample_html(html)
    html = enhance_sample_layout(html)
    html = enhance_floor_inspection(html)

    from floor_plan_route_overlay import enhance_floor_plan_route
    html = enhance_floor_plan_route(html)

    from floor_walkthrough_upgrade import enhance_floor_walkthrough
    html = enhance_floor_walkthrough(html)

    from digital_twin_inspector import enhance_digital_twin_inspector
    html = enhance_digital_twin_inspector(html)


    return Response(html, mimetype="text/html")
if __name__ == "__main__":
    print("\nUnder-construction — Step 109")
    print("Backend:", BACKEND)
    print(f"Dashboard: http://127.0.0.1:{PORT}")
    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        use_reloader=False,
    )
