
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "outputs"
    / "registration_validation"
    / "validation_manifest.json"
)

PROJECTION_REPORT_PATH = (
    ROOT
    / "outputs"
    / "cad_image_projection"
    / "projection_report.json"
)

REVIEW_PATH = (
    ROOT
    / "outputs"
    / "registration_validation"
    / "review_decisions.json"
)

PROJECTION_ROOT = (
    ROOT
    / "outputs"
    / "cad_image_projection"
)

registration_review_bp = Blueprint(
    "registration_review",
    __name__,
)


def read_json(path: Path, default=None):
    if not path.is_file():
        return default

    try:
        return json.loads(
            path.read_text(encoding="utf-8-sig")
        )
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_name = tempfile.mkstemp(
        prefix=".registration_review_",
        suffix=".json",
        dir=str(path.parent),
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                payload,
                stream,
                indent=2,
                ensure_ascii=False,
            )

        os.replace(temp_name, path)

    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def load_candidates():
    manifest = read_json(MANIFEST_PATH, {})

    if not isinstance(manifest, dict):
        return []

    candidates = manifest.get("candidates", [])

    if not isinstance(candidates, list):
        return []

    result = []

    for candidate in candidates:
        try:
            rank = int(candidate["rank"])
            angle = float(candidate["angle_deg"])
            tx = float(candidate["tx"])
            ty = float(candidate["ty"])
        except (KeyError, TypeError, ValueError):
            continue

        if rank < 1 or not all(
            math.isfinite(value)
            for value in (angle, tx, ty)
        ):
            continue

        result.append({
            "rank": rank,
            "angle_deg": angle,
            "tx": tx,
            "ty": ty,
            "verified": False,
        })

    return sorted(
        result,
        key=lambda candidate: candidate["rank"],
    )


def load_frames():
    manifest = read_json(MANIFEST_PATH, {})

    if not isinstance(manifest, dict):
        return []

    frames = manifest.get("frames", [])

    if not isinstance(frames, list):
        return []

    result = []
    seen = set()

    for frame in frames:
        filename = str(frame.get("filename", ""))

        if (
            not filename.startswith("cam0_")
            or not filename.endswith(".jpg")
            or "/" in filename
            or "\\" in filename
        ):
            continue

        if filename in seen:
            continue

        seen.add(filename)

        result.append({
            "filename": filename,
            "timestamp": frame.get("timestamp"),
        })

    return result


def projection_lookup():
    report = read_json(PROJECTION_REPORT_PATH, {})

    if not isinstance(report, dict):
        return {}

    result = {}

    for projection in report.get("projections", []):
        try:
            rank = int(projection["candidate_rank"])
            filename = str(projection["frame"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            "/" in filename
            or "\\" in filename
            or not filename.endswith(".jpg")
        ):
            continue

        path = (
            PROJECTION_ROOT
            / f"candidate_{rank:02d}"
            / filename
        )

        if path.is_file():
            result[(rank, filename)] = True

    return result


def load_reviews():
    payload = read_json(REVIEW_PATH, {})

    if not isinstance(payload, dict):
        return {}

    reviews = payload.get("reviews", {})

    return reviews if isinstance(reviews, dict) else {}


@registration_review_bp.get(
    "/api/registration-review/manifest"
)
def review_manifest():
    candidates = load_candidates()
    frames = load_frames()
    projections = projection_lookup()
    reviews = load_reviews()

    return jsonify({
        "status": "REVIEW_REQUIRED",
        "registration_verified": False,
        "construction_progress_verified": False,
        "candidates": candidates,
        "frames": frames,
        "projections": [
            {
                "rank": rank,
                "filename": filename,
            }
            for rank, filename in sorted(projections)
        ],
        "reviews": reviews,
        "note": (
            "These transforms are candidate hypotheses. "
            "A plausible reviewer decision is not "
            "independent registration validation."
        ),
    })


@registration_review_bp.get(
    "/api/registration-review/image/<int:rank>/<filename>"
)
def review_image(rank, filename):
    valid_ranks = {
        candidate["rank"]
        for candidate in load_candidates()
    }

    if rank not in valid_ranks:
        return jsonify({
            "error": "Unknown candidate"
        }), 404

    if filename not in {
        frame["filename"]
        for frame in load_frames()
    }:
        return jsonify({
            "error": "Unknown inspection frame"
        }), 404

    path = (
        PROJECTION_ROOT
        / f"candidate_{rank:02d}"
        / filename
    )

    if not path.is_file():
        return jsonify({
            "error": "Projection image unavailable"
        }), 404

    return send_file(
        path,
        mimetype="image/jpeg",
        conditional=True,
    )


@registration_review_bp.post(
    "/api/registration-review/decision"
)
def review_decision():
    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify({
            "error": "Expected JSON object"
        }), 400

    try:
        rank = int(payload.get("rank"))
    except (TypeError, ValueError):
        return jsonify({
            "error": "Invalid candidate rank"
        }), 400

    valid_ranks = {
        candidate["rank"]
        for candidate in load_candidates()
    }

    if rank not in valid_ranks:
        return jsonify({
            "error": "Unknown candidate"
        }), 404

    decision = str(
        payload.get("decision", "")
    ).upper()

    if decision not in {
        "UNREVIEWED",
        "PLAUSIBLE",
        "REJECTED",
    }:
        return jsonify({
            "error": "Invalid review decision"
        }), 400

    notes = str(
        payload.get("notes", "")
    ).strip()

    reviewer = str(
        payload.get("reviewer", "local_reviewer")
    ).strip()

    if len(notes) > 2000:
        return jsonify({
            "error": "Notes must be at most 2000 characters"
        }), 400

    if not reviewer or len(reviewer) > 100:
        return jsonify({
            "error": "Invalid reviewer identifier"
        }), 400

    timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    reviews = load_reviews()

    record = {
        "rank": rank,
        "decision": decision,
        "notes": notes,
        "reviewer": reviewer,
        "timestamp_utc": timestamp,
        "registration_verified": False,
    }

    reviews[str(rank)] = record

    write_json_atomic(REVIEW_PATH, {
        "schema_version": 1,
        "registration_verified": False,
        "reviews": reviews,
    })

    return jsonify({
        "saved": True,
        "record": record,
    })


PANEL_HTML = r"""
<section id="uc-registration-review">
  <div class="uc-reg-header">
    <div>
      <div class="uc-reg-eyebrow">
        SPATIAL EVIDENCE / FLOOR 02
      </div>
      <h2>Registration review</h2>
      <p>
        Inspect candidate CAD-to-camera alignments
        against real construction footage.
      </p>
    </div>
    <span class="uc-reg-status">
      NOT VERIFIED
    </span>
  </div>

  <div class="uc-reg-controls">
    <label>
      Candidate alignment
      <select id="uc-reg-candidate"></select>
    </label>

    <label>
      Inspection frame
      <select id="uc-reg-frame"></select>
    </label>

    <label>
      Compare against
      <select id="uc-reg-compare"></select>
    </label>
  </div>

  <div class="uc-reg-images">
    <div class="uc-reg-image-card">
      <div class="uc-reg-image-heading">
        <strong id="uc-reg-left-title">
          Candidate
        </strong>
        <span>PROJECTED CAD / CAMERA IMAGE</span>
      </div>
      <img
        id="uc-reg-left-image"
        alt="Candidate CAD projection"
      >
      <div
        class="uc-reg-image-empty"
        id="uc-reg-left-empty"
      >
        Projection unavailable
      </div>
    </div>

    <div class="uc-reg-image-card">
      <div class="uc-reg-image-heading">
        <strong id="uc-reg-right-title">
          Comparison
        </strong>
        <span>ALTERNATIVE REGISTRATION</span>
      </div>
      <img
        id="uc-reg-right-image"
        alt="Comparison CAD projection"
      >
      <div
        class="uc-reg-image-empty"
        id="uc-reg-right-empty"
      >
        Projection unavailable
      </div>
    </div>
  </div>

  <div class="uc-reg-bottom">
    <div class="uc-reg-metrics">
      <div>
        <span>ROTATION</span>
        <strong id="uc-reg-angle">—</strong>
      </div>
      <div>
        <span>TRANSLATION X</span>
        <strong id="uc-reg-tx">—</strong>
      </div>
      <div>
        <span>TRANSLATION Y</span>
        <strong id="uc-reg-ty">—</strong>
      </div>
      <div>
        <span>REVIEW STATE</span>
        <strong id="uc-reg-review-state">
          Unreviewed
        </strong>
      </div>
    </div>

    <div class="uc-reg-review">
      <div class="uc-reg-review-title">
        Reviewer assessment
      </div>
      <p>
        Does this candidate appear spatially plausible
        across the inspected frames?
      </p>

      <div class="uc-reg-decision-buttons">
        <button
          type="button"
          data-uc-reg-decision="PLAUSIBLE"
        >
          Mark plausible
        </button>

        <button
          type="button"
          data-uc-reg-decision="REJECTED"
        >
          Reject candidate
        </button>

        <button
          type="button"
          data-uc-reg-decision="UNREVIEWED"
        >
          Clear review
        </button>
      </div>

      <textarea
        id="uc-reg-notes"
        placeholder="Record visible correspondences, mismatches, or uncertainty..."
        rows="3"
      ></textarea>

      <div
        id="uc-reg-feedback"
        aria-live="polite"
      ></div>
    </div>
  </div>

  <div class="uc-reg-disclaimer">
    <strong>Interpretation boundary.</strong>
    Projected lines represent hypotheses generated
    using unverified camera-to-CAD transforms.
    Reviewer approval does not establish registration
    accuracy, physical wall identity, or construction
    completion.
  </div>
</section>
"""


PANEL_CSS = r"""
<style id="uc-registration-review-style">
#uc-registration-review {
  margin: 24px 0;
  padding: 26px;
  border: 1px solid rgba(180, 170, 220, .20);
  border-radius: 18px;
  background: #15131c;
  color: #eeeaf4;
  font-family: inherit;
}

#uc-registration-review * {
  box-sizing: border-box;
}

.uc-reg-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 18px;
  margin-bottom: 24px;
}

.uc-reg-eyebrow {
  font-size: 10px;
  letter-spacing: .17em;
  color: #b8a3f3;
  font-weight: 700;
  margin-bottom: 8px;
}

.uc-reg-header h2 {
  margin: 0 0 8px;
  font-size: 25px;
  font-weight: 650;
}

.uc-reg-header p {
  margin: 0;
  color: #a9a3b3;
  font-size: 13px;
  line-height: 1.6;
}

.uc-reg-status {
  flex-shrink: 0;
  padding: 8px 11px;
  border: 1px solid #705744;
  border-radius: 6px;
  color: #f2bc83;
  background: #322519;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .08em;
}

.uc-reg-controls {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.uc-reg-controls label {
  display: flex;
  flex-direction: column;
  gap: 8px;
  font-size: 11px;
  color: #aaa4b5;
}

.uc-reg-controls select {
  width: 100%;
  min-height: 39px;
  border: 1px solid #403b4a;
  border-radius: 8px;
  background: #211e2a;
  color: #f0edf6;
  padding: 8px;
  font: inherit;
  font-size: 12px;
}

.uc-reg-images {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.uc-reg-image-card {
  min-width: 0;
  border: 1px solid #34303e;
  border-radius: 10px;
  overflow: hidden;
  background: #0b0b0e;
}

.uc-reg-image-heading {
  display: flex;
  justify-content: space-between;
  gap: 8px;
  align-items: center;
  padding: 11px 13px;
  border-bottom: 1px solid #34303e;
}

.uc-reg-image-heading strong {
  font-size: 12px;
}

.uc-reg-image-heading span {
  font-size: 9px;
  color: #8f879d;
  letter-spacing: .05em;
}

.uc-reg-image-card img {
  display: block;
  width: 100%;
  height: 360px;
  object-fit: contain;
  background: #08080a;
}

.uc-reg-image-empty {
  display: none;
  padding: 80px 12px;
  text-align: center;
  color: #a59cae;
  font-size: 12px;
}

.uc-reg-bottom {
  margin-top: 16px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr);
  gap: 16px;
}

.uc-reg-metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.uc-reg-metrics > div {
  border: 1px solid #34303e;
  background: #1e1a26;
  border-radius: 9px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 9px;
}

.uc-reg-metrics span {
  color: #958da2;
  font-size: 10px;
  letter-spacing: .06em;
}

.uc-reg-metrics strong {
  font-size: 16px;
  font-weight: 600;
}

.uc-reg-review {
  border: 1px solid #34303e;
  border-radius: 9px;
  padding: 15px;
  background: #1b1822;
}

.uc-reg-review-title {
  font-size: 14px;
  font-weight: 650;
}

.uc-reg-review p {
  font-size: 12px;
  line-height: 1.6;
  color: #aaa4b5;
  margin: 8px 0 12px;
}

.uc-reg-decision-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 12px;
}

.uc-reg-decision-buttons button {
  cursor: pointer;
  border: 1px solid #554e61;
  background: #292431;
  color: #e8e1f2;
  border-radius: 7px;
  padding: 9px 12px;
  font: inherit;
  font-size: 11px;
}

.uc-reg-decision-buttons button:hover {
  border-color: #ad93e4;
}

.uc-reg-decision-buttons button.active {
  border-color: #bc9cf8;
  background: #45345d;
}

#uc-reg-notes {
  display: block;
  width: 100%;
  resize: vertical;
  border: 1px solid #403b4a;
  border-radius: 7px;
  padding: 10px;
  background: #100e15;
  color: #eeeaf4;
  font: inherit;
  font-size: 12px;
}

#uc-reg-feedback {
  margin-top: 9px;
  color: #a9a3b3;
  font-size: 11px;
  min-height: 15px;
}

.uc-reg-disclaimer {
  margin-top: 18px;
  padding: 13px 15px;
  border-left: 2px solid #e0a15c;
  background: #251f20;
  color: #c8bdbe;
  font-size: 11px;
  line-height: 1.7;
}

.uc-reg-disclaimer strong {
  color: #f1c395;
}

@media (max-width: 1000px) {
  .uc-reg-bottom {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 720px) {
  .uc-reg-controls,
  .uc-reg-images {
    grid-template-columns: 1fr;
  }

  #uc-registration-review {
    padding: 16px;
  }

  .uc-reg-image-card img {
    height: 280px;
  }
}
</style>
"""


PANEL_JS = r"""
<script id="uc-registration-review-script">
(function () {
  "use strict";

  const root = document.getElementById(
    "uc-registration-review"
  );

  if (!root || root.dataset.initialized === "true") {
    return;
  }

  root.dataset.initialized = "true";

  const $ = id => document.getElementById(id);

  const candidateSelect = $("uc-reg-candidate");
  const frameSelect = $("uc-reg-frame");
  const compareSelect = $("uc-reg-compare");
  const notes = $("uc-reg-notes");
  const feedback = $("uc-reg-feedback");

  let manifest = null;
  let reviews = {};
  let available = new Set();

  function setOptions(select, options) {
    select.replaceChildren();

    options.forEach(option => {
      const element = document.createElement("option");
      element.value = String(option.value);
      element.textContent = option.label;
      select.appendChild(element);
    });
  }

  function currentCandidate() {
    if (!manifest) return null;

    return manifest.candidates.find(
      candidate =>
        candidate.rank === Number(candidateSelect.value)
    );
  }

  function showImage(rank, filename, imageId, emptyId) {
    const image = $(imageId);
    const empty = $(emptyId);
    const key = rank + "|" + filename;

    if (!available.has(key)) {
      image.style.display = "none";
      image.removeAttribute("src");
      empty.style.display = "block";
      return;
    }

    image.style.display = "block";
    empty.style.display = "none";

    image.onerror = function () {
      image.style.display = "none";
      empty.style.display = "block";
    };

    image.src =
      "/api/registration-review/image/" +
      encodeURIComponent(rank) + "/" +
      encodeURIComponent(filename);
  }

  function refresh() {
    const candidate = currentCandidate();

    if (!candidate) return;

    const frame = frameSelect.value;
    const comparisonRank = Number(compareSelect.value);

    $("uc-reg-left-title").textContent =
      "Candidate " + candidate.rank;

    $("uc-reg-right-title").textContent =
      "Candidate " + comparisonRank;

    $("uc-reg-angle").textContent =
      candidate.angle_deg.toFixed(2) + "°";

    $("uc-reg-tx").textContent =
      candidate.tx.toFixed(3) + " m";

    $("uc-reg-ty").textContent =
      candidate.ty.toFixed(3) + " m";

    showImage(
      candidate.rank,
      frame,
      "uc-reg-left-image",
      "uc-reg-left-empty"
    );

    showImage(
      comparisonRank,
      frame,
      "uc-reg-right-image",
      "uc-reg-right-empty"
    );

    const review = reviews[String(candidate.rank)] || {};

    $("uc-reg-review-state").textContent =
      review.decision || "UNREVIEWED";

    notes.value = review.notes || "";

    root.querySelectorAll(
      "[data-uc-reg-decision]"
    ).forEach(button => {
      button.classList.toggle(
        "active",
        button.dataset.ucRegDecision === review.decision
      );
    });

    feedback.textContent =
      review.timestamp_utc
        ? "Last reviewed: " + review.timestamp_utc
        : "No reviewer assessment recorded.";
  }

  async function saveDecision(decision) {
    const candidate = currentCandidate();

    if (!candidate) return;

    feedback.textContent = "Saving review...";

    try {
      const response = await fetch(
        "/api/registration-review/decision",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            rank: candidate.rank,
            decision: decision,
            notes: notes.value,
            reviewer: "local_reviewer"
          })
        }
      );

      const result = await response.json();

      if (!response.ok || !result.saved) {
        throw new Error(
          result.error || "Unable to save review"
        );
      }

      reviews[String(candidate.rank)] = result.record;
      refresh();

    } catch (error) {
      feedback.textContent = error.message;
    }
  }

  async function initialize() {
    feedback.textContent = "Loading existing candidates...";

    try {
      const response = await fetch(
        "/api/registration-review/manifest"
      );

      if (!response.ok) {
        throw new Error("Could not load registration data");
      }

      manifest = await response.json();
      reviews = manifest.reviews || {};

      available = new Set(
        manifest.projections.map(
          item => item.rank + "|" + item.filename
        )
      );

      if (!manifest.candidates.length) {
        throw new Error(
          "No registration candidates were found."
        );
      }

      setOptions(
        candidateSelect,
        manifest.candidates.map(candidate => ({
          value: candidate.rank,
          label: "Candidate " + candidate.rank
        }))
      );

      setOptions(
        compareSelect,
        manifest.candidates.map(candidate => ({
          value: candidate.rank,
          label: "Candidate " + candidate.rank
        }))
      );

      setOptions(
        frameSelect,
        manifest.frames.map(frame => ({
          value: frame.filename,
          label: frame.filename
        }))
      );

      if (manifest.candidates.length > 1) {
        compareSelect.value = String(
          manifest.candidates[1].rank
        );
      }

      candidateSelect.addEventListener(
        "change",
        refresh
      );

      frameSelect.addEventListener(
        "change",
        refresh
      );

      compareSelect.addEventListener(
        "change",
        refresh
      );

      root.querySelectorAll(
        "[data-uc-reg-decision]"
      ).forEach(button => {
        button.addEventListener("click", () => {
          saveDecision(button.dataset.ucRegDecision);
        });
      });

      refresh();

    } catch (error) {
      feedback.textContent = error.message;
    }
  }

  initialize();
})();
</script>
"""


def enhance_registration_review(html: str) -> str:
    """
    Add the review panel to the existing dashboard.

    Prefer placing it immediately after the three-storey
    viewer. Fall back to the end of the dashboard body.

    This function does not modify existing viewer code.
    """
    if 'id="uc-registration-review"' in html:
        return html

    panel = PANEL_HTML + PANEL_CSS + PANEL_JS

    marker = '<div id="uc-building-panel"'

    position = html.find(marker)

    if position >= 0:
        # Insert immediately before the building panel.
        # This avoids attempting to parse nested HTML.
        return html[:position] + panel + html[position:]

    closing_body = html.lower().rfind("</body>")

    if closing_body >= 0:
        return (
            html[:closing_body]
            + panel
            + html[closing_body:]
        )

    return html + panel
