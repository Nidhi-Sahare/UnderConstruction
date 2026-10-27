
from __future__ import annotations

import csv
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, Response, abort, jsonify, render_template_string, request


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "_archive" / "hilti_transition_20261023_004237" / "data"
SEQUENCE = "floor_2_2025-10-28_run_2"

ELEMENTS_PATH = ROOT / "data" / "digital_twin" / "building_elements.json"
CANDIDATES_PATH = (
    ROOT / "outputs" / "automatic_registration_v2"
    / "registration_candidates.json"
)
FRAMES_DIR = ARCHIVE / "hilti_frames" / SEQUENCE / "cam0" / "frames"
MANIFEST_PATH = (
    ARCHIVE / "hilti_frames" / SEQUENCE
    / "cam0" / "frame_manifest.csv"
)
REVIEWS_PATH = ROOT / "data" / "digital_twin" / "element_evidence_reviews.json"

app = Flask(__name__)
write_lock = threading.Lock()

VALID_LABELS = {"visible", "not_visible", "uncertain"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_elements():
    raw = read_json(ELEMENTS_PATH)
    result = {}

    for element in raw.get("elements", []):
        element_id = element.get("element_id")
        start = element.get("start")
        end = element.get("end")

        if not isinstance(element_id, str):
            continue

        if not isinstance(start, list) or not isinstance(end, list):
            continue

        if len(start) < 2 or len(end) < 2:
            continue

        values = [
            finite(start[0]),
            finite(start[1]),
            finite(end[0]),
            finite(end[1]),
        ]

        if any(value is None for value in values):
            continue

        result[element_id] = {
            "id": element_id,
            "category": str(element.get("category", "unknown")),
            "start": values[:2],
            "end": values[2:],
            "length_m": finite(element.get("length_m")),
            "progress": element.get("progress") or {},
        }

    return result


def load_candidates():
    if not CANDIDATES_PATH.exists():
        return []

    result = []

    for raw in read_json(CANDIDATES_PATH).get("candidates", []):
        try:
            item = {
                "rank": int(raw["rank"]),
                "angle_deg": float(raw["angle_deg"]),
                "tx": float(raw["tx"]),
                "ty": float(raw["ty"]),
            }
        except (KeyError, TypeError, ValueError):
            continue

        if all(math.isfinite(value) for value in item.values()):
            result.append(item)

    return sorted(result, key=lambda item: item["rank"])


def load_positions(frame_files):
    with MANIFEST_PATH.open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        rows = {
            row["filename"]: row
            for row in csv.DictReader(file)
        }

    result = []

    for index, frame in enumerate(frame_files):
        row = rows.get(frame.name, {})

        x = finite(row.get("tx"))
        y = finite(row.get("ty"))
        z = finite(row.get("tz"))

        matched = (
            str(row.get("pose_matched", "")).lower() == "true"
            and all(v is not None for v in (x, y, z))
        )

        result.append({
            "index": index,
            "matched": matched,
            "x": x if matched else None,
            "y": y if matched else None,
        })

    return result


elements = load_elements()
candidates = load_candidates()
frames = sorted(FRAMES_DIR.glob("*.jpg"))
positions = load_positions(frames)


def transform(x, y, candidate):
    theta = math.radians(candidate["angle_deg"])
    c = math.cos(theta)
    s = math.sin(theta)

    return (
        c * x - s * y + candidate["tx"],
        s * x + c * y + candidate["ty"],
    )


def distance_to_segment(point, a, b):
    px, py = point
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    denominator = dx * dx + dy * dy

    if denominator < 1e-12:
        return math.hypot(px - ax, py - ay)

    t = max(
        0.0,
        min(
            1.0,
            ((px - ax) * dx + (py - ay) * dy)
            / denominator,
        ),
    )

    return math.hypot(
        px - (ax + t * dx),
        py - (ay + t * dy),
    )


def nearest_frames(element, candidate, limit=16):
    result = []

    for position in positions:
        if not position["matched"]:
            continue

        x, y = transform(
            position["x"],
            position["y"],
            candidate,
        )

        distance = distance_to_segment(
            (x, y),
            element["start"],
            element["end"],
        )

        result.append({
            "frame_index": position["index"],
            "distance_m": round(distance, 3),
        })

    result.sort(
        key=lambda item: (
            item["distance_m"],
            item["frame_index"],
        )
    )

    return result[:limit]


def read_reviews():
    if not REVIEWS_PATH.exists():
        return {
            "schema_version": "1.0",
            "reviews": [],
        }

    data = read_json(REVIEWS_PATH)

    if not isinstance(data, dict):
        raise ValueError("Review database must be an object.")

    if not isinstance(data.get("reviews"), list):
        raise ValueError("Review database must contain a reviews list.")

    return data


def save_reviews(data):
    REVIEWS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = REVIEWS_PATH.with_suffix(".json.tmp")

    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temporary, REVIEWS_PATH)


def reviews_for_element(element_id):
    with write_lock:
        reviews = read_reviews()["reviews"]

        return [
            review
            for review in reviews
            if review.get("element_id") == element_id
        ]


@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/api/elements")
def api_elements():
    return jsonify({
        "elements": list(elements.values()),
        "candidates": candidates,
        "frame_count": len(frames),
        "registration": "UNVERIFIED",
    })


@app.route("/api/element/<element_id>")
def api_element(element_id):
    element = elements.get(element_id)

    if element is None:
        abort(404)

    rank = request.args.get("candidate", type=int)
    candidate = next(
        (item for item in candidates if item["rank"] == rank),
        None,
    )

    return jsonify({
        "element": element,
        "candidate": candidate,
        "candidate_frames": (
            nearest_frames(element, candidate)
            if candidate else []
        ),
        "reviews": reviews_for_element(element_id),
        "registration": "UNVERIFIED",
    })


@app.route("/api/review", methods=["POST"])
def api_review():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({"error": "Expected JSON object"}), 400

    element_id = payload.get("element_id")
    frame_index = payload.get("frame_index")
    label = payload.get("label")
    rank = payload.get("candidate_rank")

    if element_id not in elements:
        return jsonify({"error": "Unknown element"}), 400

    if type(frame_index) is not int or not 0 <= frame_index < len(frames):
        return jsonify({"error": "Invalid frame index"}), 400

    if label not in VALID_LABELS:
        return jsonify({"error": "Invalid review label"}), 400

    if type(rank) is not int or not any(
        item["rank"] == rank for item in candidates
    ):
        return jsonify({"error": "Invalid candidate rank"}), 400

    review = {
        "element_id": element_id,
        "frame_index": frame_index,
        "frame_filename": frames[frame_index].name,
        "label": label,
        "candidate_rank": rank,
        "registration_verified": False,
        "source": "manual_visual_review",
        "reviewed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with write_lock:
        data = read_reviews()

        data["reviews"] = [
            item
            for item in data["reviews"]
            if not (
                item.get("element_id") == element_id
                and item.get("frame_index") == frame_index
                and item.get("candidate_rank") == rank
            )
        ]

        data["reviews"].append(review)
        save_reviews(data)

    return jsonify({
        "saved": True,
        "review": review,
    })


@app.route("/api/frame/<int:index>")
def api_frame(index):
    if not 0 <= index < len(frames):
        abort(404)

    image = cv2.imread(str(frames[index]))

    if image is None:
        abort(500)

    height, width = image.shape[:2]

    if width > 960:
        scale = 960 / width
        image = cv2.resize(
            image,
            (960, max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 83],
    )

    if not ok:
        abort(500)

    return Response(
        encoded.tobytes(),
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Under-construction | Evidence Review</title>
<style>
:root {
  --bg:#101317;
  --panel:#191e24;
  --panel2:#252c34;
  --line:#343e48;
  --muted:#aab3bc;
  --accent:#ff8a43;
}
* { box-sizing:border-box; }
body {
  margin:0;
  background:var(--bg);
  color:#eef0f2;
  font-family:Inter,Arial,sans-serif;
}
header {
  padding:20px 28px;
  display:flex;
  justify-content:space-between;
  gap:16px;
  border-bottom:1px solid var(--line);
}
.brand {
  font-size:23px;
  font-weight:750;
  letter-spacing:-1px;
}
.brand span { color:var(--accent); }
.muted { color:var(--muted); font-size:12px; }
main {
  padding:22px;
  display:grid;
  grid-template-columns:minmax(0,1.1fr) minmax(340px,1fr);
  gap:20px;
}
.panel {
  background:var(--panel);
  border:1px solid var(--line);
  border-radius:14px;
  overflow:hidden;
}
.title {
  padding:17px 20px;
  border-bottom:1px solid var(--line);
  display:flex;
  justify-content:space-between;
  gap:12px;
  font-size:14px;
  font-weight:650;
}
.toolbar {
  padding:15px 20px;
  display:flex;
  gap:10px;
  flex-wrap:wrap;
}
select,button {
  background:var(--panel2);
  border:1px solid #4b5662;
  border-radius:8px;
  color:white;
  padding:10px 12px;
  cursor:pointer;
}
select { max-width:100%; }
#elementSelect { flex:1; min-width:200px; }
.viewer {
  height:490px;
  background:#080a0d;
  display:flex;
  justify-content:center;
  align-items:center;
}
.viewer img {
  max-width:100%;
  max-height:100%;
  object-fit:contain;
}
.actions {
  padding:18px 20px;
  display:flex;
  flex-wrap:wrap;
  gap:10px;
}
.actions button:hover { filter:brightness(1.2); }
.actions button:disabled {
  opacity:.45;
  cursor:not-allowed;
}
.note {
  padding:14px 20px;
  color:var(--muted);
  font-size:12px;
  line-height:1.65;
  border-top:1px solid var(--line);
}
.warning { color:#ffb47f; }
.gallery {
  padding:12px;
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:8px;
  max-height:400px;
  overflow-y:auto;
}
.thumb {
  background:var(--panel2);
  color:var(--muted);
  border:1px solid var(--line);
  border-radius:8px;
  overflow:hidden;
  text-align:left;
  padding:0;
  cursor:pointer;
}
.thumb.active { border:2px solid var(--accent); }
.thumb img {
  width:100%;
  aspect-ratio:1.45;
  object-fit:cover;
  display:block;
}
.thumb span {
  padding:7px;
  display:block;
  font-size:11px;
}
.status {
  padding:12px 20px;
  color:var(--muted);
  font-size:12px;
}
.review-list {
  padding:12px 20px 20px;
  max-height:260px;
  overflow-y:auto;
}
.review {
  padding:10px 0;
  border-bottom:1px solid var(--line);
  font-size:12px;
  line-height:1.6;
}
@media(max-width:950px) {
  main { grid-template-columns:1fr; }
}
</style>
</head>
<body>

<header>
  <div class="brand">Under<span>-construction</span></div>
  <div class="muted">BUILDING ELEMENT · VISUAL EVIDENCE REVIEW</div>
</header>

<main>
  <section class="panel">
    <div class="title">
      <span>Walkthrough evidence</span>
      <span class="muted" id="frameLabel">No frame selected</span>
    </div>

    <div class="toolbar">
      <select id="elementSelect"></select>
      <select id="candidateSelect"></select>
    </div>

    <div class="viewer">
      <img id="photo" alt="Construction site photograph">
    </div>

    <div class="actions">
      <button id="visibleButton" onclick="saveReview('visible')">
        Visible
      </button>
      <button id="notVisibleButton" onclick="saveReview('not_visible')">
        Not visible
      </button>
      <button id="uncertainButton" onclick="saveReview('uncertain')">
        Uncertain
      </button>
    </div>

    <div class="status" id="status">
      Select an element and photograph.
    </div>

    <div class="note">
      Reviews record whether an element is visually
      identifiable in a photograph. They do not establish
      whether the element is completed.
    </div>
  </section>

  <section class="panel">
    <div class="title">
      <span>Candidate photographs</span>
      <span class="muted warning">UNVERIFIED REGISTRATION</span>
    </div>

    <div class="gallery" id="gallery"></div>

    <div class="title">
      <span>Saved observations</span>
      <span class="muted" id="reviewCount">0 reviews</span>
    </div>

    <div class="review-list" id="reviewList"></div>

    <div class="note">
      Candidate photographs are ranked by spatial
      proximity using an experimental alignment.
      This ranking does not guarantee visibility.
    </div>
  </section>
</main>

<script>
let data=null;
let selectedFrame=null;
let candidateFrames=[];
let requestVersion=0;
let saving=false;

const elementSelect=document.getElementById("elementSelect");
const candidateSelect=document.getElementById("candidateSelect");

function setStatus(message) {
  document.getElementById("status").textContent=message;
}

function setActionsEnabled(enabled) {
  for (const id of [
    "visibleButton","notVisibleButton","uncertainButton"
  ]) {
    document.getElementById(id).disabled=!enabled;
  }
}

function selectFrame(item) {
  selectedFrame=item.frame_index;

  document.getElementById("photo").src=
    "/api/frame/"+selectedFrame;

  document.getElementById("frameLabel").textContent=
    "Frame "+(selectedFrame+1)+
    " · candidate distance "+item.distance_m.toFixed(2)+" m";

  document.querySelectorAll(".thumb").forEach(button => {
    button.classList.toggle(
      "active",
      Number(button.dataset.frame)===selectedFrame
    );
  });

  setActionsEnabled(!saving);
  setStatus("Inspect the image, then select a visibility label.");
}

function renderGallery() {
  const gallery=document.getElementById("gallery");
  gallery.replaceChildren();

  for (const item of candidateFrames) {
    const button=document.createElement("button");
    button.className="thumb";
    button.dataset.frame=item.frame_index;

    const image=document.createElement("img");
    image.src="/api/frame/"+item.frame_index;
    image.loading="lazy";

    const label=document.createElement("span");
    label.textContent=
      "Frame "+(item.frame_index+1)+
      " · "+item.distance_m.toFixed(1)+" m";

    button.append(image,label);
    button.addEventListener("click",()=>selectFrame(item));
    gallery.appendChild(button);
  }

  if (candidateFrames.length) {
    selectFrame(candidateFrames[0]);
  } else {
    selectedFrame=null;
    document.getElementById("photo").removeAttribute("src");
    document.getElementById("frameLabel").textContent=
      "No candidate frames";
    setActionsEnabled(false);
    setStatus("No candidate frames available.");
  }
}

function renderReviews(reviews) {
  const container=document.getElementById("reviewList");
  container.replaceChildren();

  document.getElementById("reviewCount").textContent=
    reviews.length+" reviews";

  if (!reviews.length) {
    container.textContent="No saved observations yet.";
    return;
  }

  for (const review of reviews.slice().reverse()) {
    const row=document.createElement("div");
    row.className="review";

    row.textContent=
      "Frame "+(review.frame_index+1)+
      " · "+review.label.replace("_"," ")+
      " · candidate #"+review.candidate_rank+
      " · "+review.reviewed_at_utc;

    container.appendChild(row);
  }
}

async function loadElement() {
  if (!data) return;

  const version=++requestVersion;
  selectedFrame=null;
  setActionsEnabled(false);
  setStatus("Loading element observations...");

  const elementId=elementSelect.value;
  const rank=candidateSelect.value;

  try {
    const response=await fetch(
      "/api/element/"+encodeURIComponent(elementId)+
      "?candidate="+encodeURIComponent(rank)
    );

    if (!response.ok) throw new Error("HTTP "+response.status);

    const result=await response.json();

    if (version!==requestVersion) return;

    candidateFrames=result.candidate_frames||[];
    renderGallery();
    renderReviews(result.reviews||[]);

  } catch(error) {
    if (version!==requestVersion) return;
    setStatus("Unable to load element observations.");
    console.error(error);
  }
}

async function saveReview(label) {
  if (saving || selectedFrame===null) return;

  const elementId=elementSelect.value;
  const candidateRank=Number(candidateSelect.value);
  const frameIndex=selectedFrame;

  saving=true;
  setActionsEnabled(false);
  setStatus("Saving review...");

  try {
    const response=await fetch("/api/review",{
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        element_id:elementId,
        frame_index:frameIndex,
        candidate_rank:candidateRank,
        label:label
      })
    });

    const result=await response.json();

    if (!response.ok)
      throw new Error(result.error||"Save failed");

    setStatus("Observation saved.");
    await loadElement();

  } catch(error) {
    setStatus("Could not save observation: "+error.message);
  } finally {
    saving=false;
    setActionsEnabled(selectedFrame!==null);
  }
}

async function initialize() {
  try {
    const response=await fetch("/api/elements");

    if (!response.ok) throw new Error("HTTP "+response.status);

    data=await response.json();

    const elements=data.elements.slice().sort(
      (a,b)=>a.category.localeCompare(b.category)
    );

    for (const element of elements) {
      const option=document.createElement("option");
      option.value=element.id;
      option.textContent=
        element.category+" · "+element.id.slice(-8);
      elementSelect.appendChild(option);
    }

    for (const candidate of data.candidates) {
      const option=document.createElement("option");
      option.value=candidate.rank;
      option.textContent="Candidate #"+candidate.rank;
      candidateSelect.appendChild(option);
    }

    if (!elements.length || !data.candidates.length) {
      setStatus("Missing building elements or registration candidates.");
      return;
    }

    await loadElement();

  } catch(error) {
    setStatus("Initialization failed: "+error.message);
  }
}

elementSelect.addEventListener("change",loadElement);
candidateSelect.addEventListener("change",loadElement);

setActionsEnabled(false);
initialize();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    if not frames:
        raise RuntimeError(f"No frames at {FRAMES_DIR}")

    if not elements:
        raise RuntimeError("No digital-twin elements loaded.")

    print("\nUNDER-CONSTRUCTION · ELEMENT EVIDENCE")
    print("Building elements:", len(elements))
    print("Camera frames:", len(frames))
    print("Registration candidates:", len(candidates))
    print("Review database:", REVIEWS_PATH)
    print("Open http://127.0.0.1:5003")

    app.run(
        host="127.0.0.1",
        port=5003,
        debug=False,
        threaded=True,
    )
