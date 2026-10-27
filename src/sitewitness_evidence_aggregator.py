
"""
BuildProgress — SiteWitness Evidence Aggregator
================================================

Combines existing CAD and real-video inspection outputs into
a single evidence-backed inspection report.

Does not:
- invent CAD-to-image correspondences
- claim construction compliance
- treat CAD proximity as a verified opening
- treat visual feature matches as architectural identification

Run:
    python -m src.sitewitness_evidence_aggregator

Outputs:
    outputs/sitewitness/inspection_summary.json
    outputs/sitewitness/inspection_summary.html

Dependencies:
    Python standard library only
"""

from __future__ import annotations

import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CAD_CANDIDATES = (
    ROOT / "outputs/cad_openings/opening_candidates.json"
)

CAD_AUDIT = (
    ROOT / "data/geometry/opening_audit.json"
)

VISUAL_SELECTION = (
    ROOT
    / "outputs/agentic_inspection"
    / "floor_2_inspection_report.json"
)

VISUAL_VERIFICATION = (
    ROOT
    / "outputs/sitewitness/multiframe"
    / "verification_report.json"
)

OUTPUT = ROOT / "outputs/sitewitness"


def load_optional(path: Path):
    if not path.is_file():
        return None

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        print(f"WARNING: Cannot read {path}: {exc}")
        return None


def relative_path(path: Path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def prioritize_cad_candidate(candidate):
    status = candidate.get(
        "association_status", "UNKNOWN"
    )

    candidate_type = candidate.get(
        "type", "unknown"
    )

    # Priority reflects review usefulness, not confidence
    # that the candidate is a real physical opening.
    priorities = {
        "PROVISIONAL_GAP_MATCH": (
            1, "REVIEW_PROVISIONAL_WALL_GAP"
        ),
        "AMBIGUOUS_GAPS": (
            2, "RESOLVE_MULTIPLE_GAP_CANDIDATES"
        ),
        "NO_SUPPORTED_GAP": (
            3, "CHECK_MISSING_WALL_GAP"
        ),
        "NO_SWING_ARC_ANCHOR": (
            4, "CHECK_DOOR_ANCHOR"
        ),
        "SINGLE_SPATIAL_CANDIDATE": (
            5, "REVIEW_WINDOW_ASSOCIATION"
        ),
    }

    priority, action = priorities.get(
        status,
        (6, "MANUAL_REVIEW")
    )

    return {
        "candidate_id": candidate.get("candidate_id"),
        "type": candidate_type,
        "association_status": status,
        "priority": priority,
        "recommended_action": action,
        "opening_verified": bool(
            candidate.get("opening_verified", False)
        ),
        "safe_to_cut": bool(
            candidate.get("safe_to_cut", False)
        ),
        "source_layer": candidate.get("layer"),
        "source_handles": candidate.get(
            "entity_handles", []
        ),
    }


def summarize_cad(candidates_data, audit_data):
    if candidates_data is None:
        return {
            "status": "MISSING_EVIDENCE",
            "candidate_count": 0,
            "candidates": [],
        }

    raw_candidates = candidates_data.get(
        "candidates", []
    )

    candidates = [
        prioritize_cad_candidate(candidate)
        for candidate in raw_candidates
    ]

    candidates.sort(
        key=lambda item: (
            item["priority"],
            str(item["candidate_id"])
        )
    )

    status_counts = Counter(
        item["association_status"]
        for item in candidates
    )

    type_counts = Counter(
        item["type"]
        for item in candidates
    )

    audit = {
        "status": "MISSING_EVIDENCE"
    }

    if audit_data is not None:
        audit = {
            "status": "AVAILABLE",
            "unmatched_wall_endpoint_count": (
                audit_data.get(
                    "unmatched_wall_endpoint_count"
                )
            ),
            "endpoint_classifications": (
                audit_data.get(
                    "endpoint_classifications", {}
                )
            ),
            "interpretation": (
                audit_data.get("interpretation")
            ),
        }

    return {
        "status": "CAD_HYPOTHESES_AVAILABLE",
        "candidate_count": len(candidates),
        "type_counts": dict(type_counts),
        "association_status_counts": dict(
            status_counts
        ),
        "units_verified": candidates_data.get(
            "units_verified", False
        ),
        "openings_verified": candidates_data.get(
            "openings_verified", False
        ),
        "wall_cutouts_generated": (
            candidates_data.get(
                "wall_cutouts_generated", False
            )
        ),
        "candidates": candidates,
        "opening_audit": audit,
    }


def summarize_visual(selection_data, verification_data):
    if verification_data is None:
        return {
            "status": "MISSING_VERIFICATION",
            "anchors": [],
            "corroborated_anchor_count": 0,
            "abstained_anchor_count": 0,
        }

    anchors = []

    for item in verification_data.get(
        "anchors", []
    ):
        pair_results = item.get(
            "pair_results", []
        )

        supported_pairs = [
            pair
            for pair in pair_results
            if pair.get("status")
            == "CALIBRATED_CORRESPONDENCE_SUPPORTED"
        ]

        evidence_images = [
            pair["evidence_image"]
            for pair in supported_pairs
            if pair.get("evidence_image")
        ]

        anchors.append({
            "frame_index": item.get(
                "anchor_frame"
            ),
            "status": item.get(
                "status", "ABSTAIN"
            ),
            "supporting_neighbors": (
                item.get(
                    "supporting_neighbors", 0
                )
            ),
            "evidence_images": evidence_images,
            "best_inlier_count": max(
                (
                    pair.get(
                        "ransac_inliers", 0
                    )
                    for pair in pair_results
                ),
                default=0
            ),
            "best_spatial_coverage_cells": max(
                (
                    pair.get(
                        "inlier_grid_cells", 0
                    )
                    for pair in pair_results
                ),
                default=0
            ),
        })

    corroborated = sum(
        item["status"]
        == "CORROBORATED_VISUAL_CORRESPONDENCE"
        for item in anchors
    )

    selection_status = (
        selection_data.get("status")
        if selection_data is not None
        else "MISSING_SELECTION"
    )

    return {
        "status": "VISUAL_EVIDENCE_AVAILABLE",
        "selection_status": selection_status,
        "method": verification_data.get(
            "method"
        ),
        "anchor_count": len(anchors),
        "corroborated_anchor_count": corroborated,
        "abstained_anchor_count": (
            len(anchors) - corroborated
        ),
        "anchors": anchors,
        "architectural_identity_verified": False,
    }


def make_findings(cad, visual):
    findings = []

    for candidate in cad.get(
        "candidates", []
    ):
        findings.append({
            "finding_id": (
                "CAD_" + str(
                    candidate["candidate_id"]
                )
            ),
            "domain": "CAD",
            "priority": candidate["priority"],
            "status": "REQUIRES_REVIEW",
            "description": (
                f"{candidate['type']} candidate "
                f"with association status "
                f"{candidate['association_status']}"
            ),
            "recommended_action": (
                candidate["recommended_action"]
            ),
            "construction_compliance_verified": False,
        })

    for anchor in visual.get(
        "anchors", []
    ):
        supported = (
            anchor["status"]
            == "CORROBORATED_VISUAL_CORRESPONDENCE"
        )

        findings.append({
            "finding_id": (
                "VIDEO_FRAME_"
                + str(anchor["frame_index"])
            ),
            "domain": "VIDEO",
            "priority": (
                5 if supported else 2
            ),
            "status": (
                "VISUAL_CORRESPONDENCE_SUPPORTED"
                if supported
                else "INSUFFICIENT_EVIDENCE"
            ),
            "description": (
                f"Frame {anchor['frame_index']}: "
                f"{anchor['supporting_neighbors']} "
                f"supporting neighboring observations"
            ),
            "recommended_action": (
                "REVIEW_CORROBORATED_VISUAL_EVIDENCE"
                if supported
                else "REQUEST_ADDITIONAL_VISUAL_EVIDENCE"
            ),
            "evidence_images": anchor[
                "evidence_images"
            ],
            "construction_compliance_verified": False,
        })

    findings.sort(
        key=lambda item: (
            item["priority"],
            item["finding_id"]
        )
    )

    return findings


def render_html(report):
    cad = report["cad_evidence"]
    visual = report["visual_evidence"]

    def safe(value):
        return html.escape(
            str(value), quote=True
        )

    rows = []

    for finding in report["findings"]:
        rows.append(
            "<tr>"
            f"<td>{safe(finding['finding_id'])}</td>"
            f"<td>{safe(finding['domain'])}</td>"
            f"<td>{safe(finding['priority'])}</td>"
            f"<td>{safe(finding['status'])}</td>"
            f"<td>{safe(finding['description'])}</td>"
            f"<td>{safe(finding['recommended_action'])}</td>"
            "</tr>"
        )

    anchor_cards = []

    for anchor in visual.get("anchors", []):
        images = []

        for image_path in anchor.get(
            "evidence_images", []
        ):
            # Paths are generated from the local pipeline,
            # not arbitrary remote URLs.
            resolved = (
                ROOT / image_path
            ).resolve()

            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                continue

            if not resolved.is_file():
                continue

            image_url = (
                resolved.relative_to(OUTPUT)
                .as_posix()
                if resolved.is_relative_to(OUTPUT)
                else Path(
                    "..",
                    image_path
                ).as_posix()
            )

            images.append(
                f'<img src="{safe(image_url)}" '
                'loading="lazy">'
            )

        anchor_cards.append(
            '<article class="anchor">'
            f"<h3>Frame {safe(anchor['frame_index'])}</h3>"
            f"<p>{safe(anchor['status'])}</p>"
            f"<p>Supporting neighbors: "
            f"{safe(anchor['supporting_neighbors'])}</p>"
            + "".join(images)
            + "</article>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SiteWitness — Inspection Evidence</title>
<style>
:root {{
    color-scheme: dark;
    font-family: Inter, Segoe UI, Arial, sans-serif;
}}
* {{ box-sizing: border-box; }}
body {{
    margin: 0;
    background: #101418;
    color: #e7edf2;
    line-height: 1.5;
}}
main {{
    max-width: 1400px;
    margin: auto;
    padding: 36px;
}}
h1 {{ font-size: 38px; margin-bottom: 4px; }}
h2 {{ margin-top: 44px; }}
.subtitle {{ color: #a6b3be; }}
.metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin-top: 30px;
}}
.metric {{
    background: #1c242b;
    border: 1px solid #35414a;
    border-radius: 12px;
    padding: 20px;
}}
.metric strong {{
    display: block;
    font-size: 32px;
}}
.metric span {{ color: #a6b3be; }}
.notice {{
    border-left: 3px solid #d4a35a;
    background: #29251d;
    padding: 16px;
    margin-top: 26px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}}
th, td {{
    padding: 11px;
    text-align: left;
    border-bottom: 1px solid #35414a;
    vertical-align: top;
}}
th {{ color: #a6b3be; }}
.table-wrap {{ overflow-x: auto; }}
.gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 16px;
}}
.anchor {{
    background: #1c242b;
    border: 1px solid #35414a;
    border-radius: 12px;
    padding: 16px;
}}
.anchor img {{
    width: 100%;
    border-radius: 8px;
    margin-top: 12px;
}}
</style>
</head>
<body>
<main>
<h1>SiteWitness</h1>
<p class="subtitle">Construction Inspection Evidence — Floor 2</p>
<p class="subtitle">Generated {safe(report['generated_at'])}</p>

<section class="metrics">
<div class="metric">
<strong>{cad['candidate_count']}</strong>
<span>CAD opening hypotheses</span>
</div>
<div class="metric">
<strong>{visual['corroborated_anchor_count']}</strong>
<span>Corroborated video anchors</span>
</div>
<div class="metric">
<strong>{visual['abstained_anchor_count']}</strong>
<span>Abstained video anchors</span>
</div>
<div class="metric">
<strong>0</strong>
<span>Verified CAD-to-video associations</span>
</div>
</section>

<div class="notice">
<strong>Verification boundary</strong>
<p>CAD opening hypotheses and calibrated video correspondence
are separate evidence streams. This report does not establish
architectural identity, metric opening dimensions, CAD alignment,
or construction compliance.</p>
</div>

<h2>Prioritized Inspection Findings</h2>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>ID</th><th>Domain</th><th>Priority</th>
<th>Status</th><th>Evidence</th><th>Next action</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</div>

<h2>Visual Evidence</h2>
<section class="gallery">
{''.join(anchor_cards)}
</section>

</main>
</body>
</html>"""


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    cad_data = load_optional(
        CAD_CANDIDATES
    )

    audit_data = load_optional(
        CAD_AUDIT
    )

    selection_data = load_optional(
        VISUAL_SELECTION
    )

    verification_data = load_optional(
        VISUAL_VERIFICATION
    )

    cad = summarize_cad(
        cad_data,
        audit_data
    )

    visual = summarize_visual(
        selection_data,
        verification_data
    )

    findings = make_findings(
        cad,
        visual
    )

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "system": "BuildProgress",
        "agent": "SiteWitness",
        "floor": 2,
        "status": "EVIDENCE_AGGREGATED",
        "cad_evidence": cad,
        "visual_evidence": visual,
        "findings": findings,
        "cross_modal_associations": [],
        "cad_registration_verified": False,
        "construction_compliance_verified": False,
        "limitations": [
            "CAD candidates are not verified physical openings.",
            "Calibrated feature matches are not architectural identification.",
            "CAD and video evidence have not been spatially registered.",
            "Review priorities are heuristic workflow ordering.",
            "No construction compliance finding is established.",
        ],
    }

    json_path = (
        OUTPUT / "inspection_summary.json"
    )

    html_path = (
        OUTPUT / "inspection_summary.html"
    )

    json_path.write_text(
        json.dumps(
            report,
            indent=2
        ),
        encoding="utf-8"
    )

    html_path.write_text(
        render_html(report),
        encoding="utf-8"
    )

    print("\nSITEWITNESS — EVIDENCE AGGREGATION")
    print("=" * 55)
    print(
        "CAD candidates:",
        cad["candidate_count"]
    )
    print(
        "Video anchors:",
        visual.get("anchor_count", 0)
    )
    print(
        "Corroborated:",
        visual["corroborated_anchor_count"]
    )
    print(
        "Abstained:",
        visual["abstained_anchor_count"]
    )
    print(
        "Inspection findings:",
        len(findings)
    )
    print("CAD-to-video matches: 0 (unverified)")
    print("JSON:", json_path)
    print("HTML:", html_path)


if __name__ == "__main__":
    main()
