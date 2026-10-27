
"""
SiteWitness — BuildProgress Dashboard Integration
==================================================

Native SiteWitness inspection interface for BuildProgress.

Features:
    - CAD opening hypothesis summary
    - Multi-frame visual evidence summary
    - Searchable and filterable inspection findings
    - Clickable finding details
    - Persistent human review decisions
    - Reviewer notes
    - Source evidence metadata
    - Clear verification limitations

Existing dependencies:
    outputs/sitewitness/inspection_summary.json
    src/sitewitness_review.py

No additional Flask server or detection pipeline.

Usage in construction_dashboard.py:

    from sitewitness_dashboard import enhance_sitewitness
    html = enhance_sitewitness(html)
"""

from __future__ import annotations

import html as html_lib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    ROOT
    / "outputs"
    / "sitewitness"
    / "inspection_summary.json"
)


def _escape(value):
    if value is None:
        return ""

    return html_lib.escape(
        str(value),
        quote=True,
    )


def _load_report():
    if not REPORT_PATH.is_file():
        return None

    try:
        return json.loads(
            REPORT_PATH.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


def _friendly_status(value):
    labels = {
        "PROVISIONAL_GAP_MATCH":
            "Provisional wall-gap match",
        "AMBIGUOUS_GAPS":
            "Multiple possible wall gaps",
        "NO_SUPPORTED_GAP":
            "No supported wall gap",
        "NO_SWING_ARC_ANCHOR":
            "Door swing anchor unresolved",
        "SINGLE_SPATIAL_CANDIDATE":
            "Single spatial candidate",
        "REQUIRES_REVIEW":
            "Requires review",
        "VISUAL_CORRESPONDENCE_SUPPORTED":
            "Visual correspondence supported",
        "INSUFFICIENT_EVIDENCE":
            "Insufficient evidence",
        "CORROBORATED_VISUAL_CORRESPONDENCE":
            "Corroborated visual correspondence",
        "ABSTAIN":
            "Abstained",
        "NEEDS_REVIEW":
            "Needs review",
        "REVIEWED":
            "Reviewed",
        "DISMISSED":
            "Dismissed",
        "REVIEW_PROVISIONAL_WALL_GAP":
            "Inspect provisional wall-gap association",
        "RESOLVE_MULTIPLE_GAP_CANDIDATES":
            "Resolve ambiguous wall gaps",
        "CHECK_MISSING_WALL_GAP":
            "Check unsupported wall gap",
        "CHECK_DOOR_ANCHOR":
            "Review door swing anchor",
        "REVIEW_WINDOW_ASSOCIATION":
            "Review window association",
        "REVIEW_CORROBORATED_VISUAL_EVIDENCE":
            "Inspect supporting video observations",
        "REQUEST_ADDITIONAL_VISUAL_EVIDENCE":
            "Collect additional visual evidence",
    }

    value = str(value or "UNKNOWN")

    return labels.get(
        value,
        value.replace("_", " ").capitalize(),
    )


def _render_findings(findings):
    rows = []

    for finding in findings:
        domain = str(
            finding.get(
                "domain",
                "UNKNOWN",
            )
        ).upper()

        finding_id = _escape(
            finding.get(
                "finding_id",
                "UNKNOWN",
            )
        )

        status = _escape(
            _friendly_status(
                finding.get(
                    "status",
                    "UNKNOWN",
                )
            )
        )

        description = _escape(
            finding.get(
                "description",
                "",
            )
        )

        action = _escape(
            _friendly_status(
                finding.get(
                    "recommended_action",
                    "",
                )
            )
        )

        try:
            priority = int(
                finding.get(
                    "priority",
                    6,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            priority = 6

        rows.append(
            f"""
            <tr
                data-sw-domain="{_escape(domain)}"
                data-sw-priority="{priority}"
                data-sw-id="{finding_id}"
                tabindex="0"
                aria-label="Review finding {finding_id}"
            >
                <td class="sw-id">
                    {finding_id}
                </td>

                <td>
                    {_escape(domain)}
                </td>

                <td>
                    {priority}
                </td>

                <td>
                    {status}
                </td>

                <td>
                    {description}
                </td>

                <td>
                    {action}
                </td>

                <td class="sw-review-state">
                    Not reviewed
                </td>
            </tr>
            """
        )

    return "\n".join(rows)


def _render_anchors(anchors):
    cards = []

    for anchor in anchors:
        frame = anchor.get(
            "frame_index",
            "Unknown",
        )

        status = str(
            anchor.get(
                "status",
                "ABSTAIN",
            )
        )

        supported = (
            status
            == "CORROBORATED_VISUAL_CORRESPONDENCE"
        )

        state_class = (
            "sw-supported"
            if supported
            else "sw-abstained"
        )

        state_label = (
            "Correspondence supported"
            if supported
            else "Insufficient evidence"
        )

        neighbors = anchor.get(
            "supporting_neighbors",
            0,
        )

        inliers = anchor.get(
            "best_inlier_count",
            0,
        )

        cells = anchor.get(
            "best_spatial_coverage_cells",
            0,
        )

        cards.append(
            f"""
            <article class="sw-anchor">

                <div class="sw-anchor-top">
                    <strong>
                        Frame {_escape(frame)}
                    </strong>

                    <span class="{state_class}">
                        {_escape(state_label)}
                    </span>
                </div>

                <div class="sw-anchor-stats">

                    <div>
                        <strong>
                            {_escape(neighbors)}
                        </strong>
                        <span>
                            Supporting neighbors
                        </span>
                    </div>

                    <div>
                        <strong>
                            {_escape(inliers)}
                        </strong>
                        <span>
                            Best RANSAC inliers
                        </span>
                    </div>

                    <div>
                        <strong>
                            {_escape(cells)}
                        </strong>
                        <span>
                            Spatial coverage cells
                        </span>
                    </div>

                </div>

                <button
                    type="button"
                    class="sw-anchor-review"
                    data-sw-finding="VIDEO_FRAME_{_escape(frame)}"
                >
                    Review this frame
                </button>

            </article>
            """
        )

    return "\n".join(cards)


def enhance_sitewitness(page_html):
    """
    Insert the SiteWitness section into the existing dashboard.

    Safe against duplicate insertion.
    """

    if 'id="sitewitness-inspection"' in page_html:
        return page_html

    report = _load_report()

    if report is None:
        cad_count = 0
        corroborated = 0
        abstained = 0
        findings = []
        anchors = []

        availability = (
            "Inspection report unavailable. "
            "Run the evidence aggregator first."
        )

    else:
        cad = report.get(
            "cad_evidence",
            {},
        )

        visual = report.get(
            "visual_evidence",
            {},
        )

        cad_count = cad.get(
            "candidate_count",
            0,
        )

        corroborated = visual.get(
            "corroborated_anchor_count",
            0,
        )

        abstained = visual.get(
            "abstained_anchor_count",
            0,
        )

        findings = report.get(
            "findings",
            [],
        )

        anchors = visual.get(
            "anchors",
            [],
        )

        availability = (
            "Evidence report loaded"
        )

    findings_html = _render_findings(
        findings
    )

    anchors_html = _render_anchors(
        anchors
    )

    # Data is embedded as JSON rather than reconstructed
    # from visible table text.
    finding_data = {
        str(item.get("finding_id")): item
        for item in findings
        if item.get("finding_id")
    }

    finding_json = json.dumps(
        finding_data,
        ensure_ascii=True,
    ).replace(
        "<",
        "\\u003c",
    ).replace(
        ">",
        "\\u003e",
    ).replace(
        "&",
        "\\u0026",
    )

    # Keep HTML/CSS/JS as a normal string so JavaScript
    # braces do not conflict with Python f-string syntax.
    section = """
<!-- SITEWITNESS INTEGRATION START -->

<style>

#sitewitness-inspection {
    padding: 76px 5%;
    background: #14151d;
    border-top: 1px solid #343542;
    color: #f0eef7;
}

#sitewitness-inspection * {
    box-sizing: border-box;
}

.sw-container {
    max-width: 1500px;
    margin: 0 auto;
}

.sw-eyebrow {
    color: #b89aff;
    font-size: 11px;
    letter-spacing: .17em;
    text-transform: uppercase;
    font-weight: 700;
}

.sw-heading {
    font-size: clamp(34px, 4vw, 55px);
    letter-spacing: -.045em;
    margin: 12px 0;
    font-weight: 700;
}

.sw-subtitle {
    color: #a6a4b6;
    max-width: 780px;
    line-height: 1.7;
    font-size: 15px;
}

.sw-metrics {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 30px 0;
}

.sw-metric {
    padding: 23px;
    background: #20212b;
    border: 1px solid #343542;
    border-radius: 13px;
}

.sw-metric strong {
    display: block;
    font-size: 33px;
    letter-spacing: -.04em;
}

.sw-metric span {
    display: block;
    color: #a6a4b6;
    font-size: 12px;
    margin-top: 5px;
}

.sw-notice {
    padding: 20px 24px;
    border-left: 3px solid #f0a84b;
    background: #28251f;
    border-radius: 0 10px 10px 0;
    margin-bottom: 40px;
}

.sw-notice strong {
    font-size: 14px;
}

.sw-notice p {
    margin: 8px 0 0;
    color: #c8c1b6;
    line-height: 1.65;
    font-size: 13px;
}

.sw-section-title {
    font-size: 24px;
    letter-spacing: -.025em;
    margin: 36px 0 18px;
}

.sw-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}

.sw-controls input,
.sw-controls select {
    background: #20212b;
    border: 1px solid #444552;
    color: #f0eef7;
    border-radius: 9px;
    padding: 12px 14px;
    font-size: 13px;
}

.sw-controls input {
    flex: 1;
    min-width: 230px;
}

.sw-controls select {
    min-width: 160px;
}

.sw-table-wrap {
    overflow-x: auto;
    max-height: 520px;
    overflow-y: auto;
    border: 1px solid #343542;
    border-radius: 12px;
    background: #1b1c25;
}

.sw-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}

.sw-table th,
.sw-table td {
    padding: 14px 13px;
    text-align: left;
    border-bottom: 1px solid #343542;
    vertical-align: top;
}

.sw-table th {
    color: #a6a4b6;
    background: #22232e;
    position: sticky;
    top: 0;
    z-index: 1;
}

.sw-table td {
    color: #e0dce8;
}

.sw-table .sw-id {
    font-family: Consolas, monospace;
    white-space: nowrap;
    color: #c5b1ff;
}

.sw-table tbody tr {
    cursor: pointer;
}

.sw-table tbody tr:hover,
.sw-table tbody tr:focus-visible,
.sw-table tbody tr.sw-selected {
    background: #302b40;
    outline: 1px solid #b89aff;
    outline-offset: -1px;
}

.sw-review-state {
    white-space: nowrap;
    color: #a6a4b6;
}

.sw-review-panel {
    margin-top: 22px;
    padding: 26px;
    border: 1px solid #625478;
    border-radius: 14px;
    background: #20212b;
}

.sw-review-panel[hidden] {
    display: none;
}

.sw-review-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 20px;
}

.sw-review-header h3 {
    margin: 8px 0;
    font-size: 22px;
    overflow-wrap: anywhere;
}

.sw-review-description {
    color: #c8c5d2;
    line-height: 1.65;
}

.sw-review-details {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-top: 20px;
}

.sw-review-detail {
    padding: 14px;
    background: #181922;
    border: 1px solid #343542;
    border-radius: 9px;
}

.sw-review-detail span {
    display: block;
    font-size: 11px;
    color: #a6a4b6;
}

.sw-review-detail strong {
    display: block;
    font-size: 13px;
    margin-top: 7px;
    overflow-wrap: anywhere;
}

.sw-review-fields {
    display: grid;
    gap: 12px;
    max-width: 720px;
    margin-top: 24px;
}

.sw-review-fields label {
    font-size: 13px;
    color: #c8c5d2;
}

.sw-review-fields select,
.sw-review-fields textarea {
    background: #15161d;
    color: #f0eef7;
    border: 1px solid #555565;
    border-radius: 8px;
    padding: 12px;
    font: inherit;
}

.sw-review-fields textarea {
    width: 100%;
    resize: vertical;
    min-height: 100px;
}

#sw-review-save {
    width: fit-content;
    padding: 12px 20px;
    background: #b89aff;
    color: #17131f;
    border: none;
    border-radius: 8px;
    font-weight: 700;
}

#sw-review-save:disabled {
    opacity: .5;
    cursor: wait;
}

#sw-review-close {
    background: transparent;
    color: #c8c5d2;
    border: 1px solid #555565;
    border-radius: 8px;
    padding: 9px 14px;
}

.sw-review-disclaimer {
    margin-top: 22px;
    color: #a6a4b6;
    font-size: 12px;
    line-height: 1.6;
}

.sw-anchor-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
}

.sw-anchor {
    padding: 20px;
    background: #20212b;
    border: 1px solid #343542;
    border-radius: 12px;
}

.sw-anchor-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
}

.sw-anchor-top strong {
    font-size: 17px;
}

.sw-supported {
    color: #80cbb0;
    font-size: 11px;
}

.sw-abstained {
    color: #f0a84b;
    font-size: 11px;
}

.sw-anchor-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 22px;
}

.sw-anchor-stats strong {
    display: block;
    font-size: 20px;
}

.sw-anchor-stats span {
    display: block;
    font-size: 10px;
    color: #a6a4b6;
    margin-top: 5px;
    line-height: 1.5;
}

.sw-anchor-review {
    margin-top: 22px;
    background: transparent;
    color: #c5b1ff;
    border: 1px solid #625478;
    border-radius: 8px;
    padding: 10px 13px;
    font-size: 12px;
    cursor: pointer;
}

.sw-anchor-review:hover {
    background: #302b40;
}

.sw-footer {
    margin-top: 22px;
    color: #9898aa;
    font-size: 12px;
}

@media (max-width: 1050px) {
    .sw-metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .sw-anchor-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 650px) {
    .sw-metrics,
    .sw-anchor-grid,
    .sw-review-details {
        grid-template-columns: 1fr;
    }

    #sitewitness-inspection {
        padding: 48px 5%;
    }

    .sw-review-header {
        flex-direction: column;
    }
}

</style>

<section id="sitewitness-inspection">
    <div class="sw-container">

        <div class="sw-eyebrow">
            Autonomous Evidence Review
        </div>

        <h2 class="sw-heading">
            SiteWitness
        </h2>

        <p class="sw-subtitle">
            Evidence-backed construction inspection combining
            CAD opening hypotheses and calibrated multi-frame
            walkthrough analysis. Select any finding to inspect
            its evidence and record a human review decision.
        </p>

        <div class="sw-metrics">

            <div class="sw-metric">
                <strong>__SW_CAD_COUNT__</strong>
                <span>CAD opening hypotheses</span>
            </div>

            <div class="sw-metric">
                <strong>__SW_CORROBORATED__</strong>
                <span>Corroborated video anchors</span>
            </div>

            <div class="sw-metric">
                <strong>__SW_ABSTAINED__</strong>
                <span>Abstained video anchors</span>
            </div>

            <div class="sw-metric">
                <strong>0</strong>
                <span>Verified CAD-to-video associations</span>
            </div>

        </div>

        <div class="sw-notice">
            <strong>Verification boundary</strong>
            <p>
                CAD opening candidates and calibrated video
                correspondences are separate evidence streams.
                A human review decision does not establish
                architectural identity, CAD registration,
                physical opening dimensions, or construction
                compliance.
            </p>
        </div>

        <h3 class="sw-section-title">
            Inspection findings
        </h3>

        <div class="sw-controls">

            <input
                id="sw-search"
                type="search"
                placeholder="Search findings, frames, or statuses"
                aria-label="Search SiteWitness findings"
            >

            <select
                id="sw-domain"
                aria-label="Filter evidence domain"
            >
                <option value="ALL">All evidence</option>
                <option value="CAD">CAD hypotheses</option>
                <option value="VIDEO">Video evidence</option>
            </select>

            <select
                id="sw-priority"
                aria-label="Filter review priority"
            >
                <option value="ALL">All priorities</option>
                <option value="1">Priority 1</option>
                <option value="2">Priority 2</option>
                <option value="3">Priority 3</option>
                <option value="4">Priority 4</option>
                <option value="5">Priority 5</option>
                <option value="6">Priority 6</option>
            </select>

        </div>

        <div class="sw-table-wrap">

            <table class="sw-table">

                <thead>
                    <tr>
                        <th>Evidence ID</th>
                        <th>Domain</th>
                        <th>Priority</th>
                        <th>Finding status</th>
                        <th>Observation</th>
                        <th>Recommended action</th>
                        <th>Human review</th>
                    </tr>
                </thead>

                <tbody id="sw-findings-body">
                    __SW_FINDINGS__
                </tbody>

            </table>

        </div>

        <div
            id="sw-review-panel"
            class="sw-review-panel"
            hidden
        >

            <div class="sw-review-header">

                <div>
                    <div class="sw-eyebrow">
                        Evidence review
                    </div>

                    <h3 id="sw-review-title">
                        Selected finding
                    </h3>
                </div>

                <button
                    type="button"
                    id="sw-review-close"
                >
                    Close
                </button>

            </div>

            <p
                id="sw-review-description"
                class="sw-review-description"
            ></p>

            <div class="sw-review-details">

                <div class="sw-review-detail">
                    <span>Evidence domain</span>
                    <strong id="sw-detail-domain">—</strong>
                </div>

                <div class="sw-review-detail">
                    <span>Finding status</span>
                    <strong id="sw-detail-status">—</strong>
                </div>

                <div class="sw-review-detail">
                    <span>Review priority</span>
                    <strong id="sw-detail-priority">—</strong>
                </div>

                <div class="sw-review-detail">
                    <span>Recommended action</span>
                    <strong id="sw-detail-action">—</strong>
                </div>

            </div>

            <div class="sw-review-fields">

                <label for="sw-review-status">
                    Review decision
                </label>

                <select id="sw-review-status">
                    <option value="NEEDS_REVIEW">
                        Needs review
                    </option>
                    <option value="REVIEWED">
                        Reviewed
                    </option>
                    <option value="DISMISSED">
                        Dismissed
                    </option>
                </select>

                <label for="sw-review-note">
                    Reviewer notes
                </label>

                <textarea
                    id="sw-review-note"
                    rows="4"
                    maxlength="2000"
                    placeholder="What was checked? What remains uncertain?"
                ></textarea>

                <button
                    type="button"
                    id="sw-review-save"
                >
                    Save review
                </button>

                <p
                    id="sw-review-message"
                    role="status"
                    aria-live="polite"
                ></p>

            </div>

            <p class="sw-review-disclaimer">
                Review status records a human workflow decision.
                It does not verify construction compliance,
                architectural identity, or CAD-to-video alignment.
            </p>

        </div>

        <h3 class="sw-section-title">
            Multi-frame visual evidence
        </h3>

        <div class="sw-anchor-grid">
            __SW_ANCHORS__
        </div>

        <p class="sw-footer">
            __SW_AVAILABILITY__.
            Visual correspondence is not equivalent
            to architectural verification.
        </p>

    </div>
</section>

<script id="sw-finding-data" type="application/json">
__SW_FINDING_JSON__
</script>

<script>
(function () {
    "use strict";

    const section = document.getElementById(
        "sitewitness-inspection"
    );

    if (!section) return;

    const search = document.getElementById("sw-search");
    const domain = document.getElementById("sw-domain");
    const priority = document.getElementById("sw-priority");

    const rows = Array.from(
        document.querySelectorAll(
            "#sw-findings-body tr"
        )
    );

    const reviewPanel = document.getElementById(
        "sw-review-panel"
    );

    const reviewTitle = document.getElementById(
        "sw-review-title"
    );

    const reviewDescription = document.getElementById(
        "sw-review-description"
    );

    const reviewStatus = document.getElementById(
        "sw-review-status"
    );

    const reviewNote = document.getElementById(
        "sw-review-note"
    );

    const reviewMessage = document.getElementById(
        "sw-review-message"
    );

    const reviewSave = document.getElementById(
        "sw-review-save"
    );

    const reviewClose = document.getElementById(
        "sw-review-close"
    );

    const detailDomain = document.getElementById(
        "sw-detail-domain"
    );

    const detailStatus = document.getElementById(
        "sw-detail-status"
    );

    const detailPriority = document.getElementById(
        "sw-detail-priority"
    );

    const detailAction = document.getElementById(
        "sw-detail-action"
    );

    const findingDataElement = document.getElementById(
        "sw-finding-data"
    );

    let findings = {};

    try {
        findings = JSON.parse(
            findingDataElement.textContent
        );
    } catch (error) {
        console.error(
            "SiteWitness finding data could not be parsed",
            error
        );
    }

    let selectedFindingId = null;
    let savedReviews = {};

    function friendly(value) {
        if (!value) return "Unknown";

        return String(value)
            .replace(/_/g, " ")
            .toLowerCase()
            .replace(/^./, function (character) {
                return character.toUpperCase();
            });
    }

    function filterFindings() {
        const query = search.value
            .toLowerCase()
            .trim();

        const selectedDomain = domain.value;
        const selectedPriority = priority.value;

        rows.forEach(function (row) {
            const matchesText = row.textContent
                .toLowerCase()
                .includes(query);

            const matchesDomain =
                selectedDomain === "ALL" ||
                row.dataset.swDomain === selectedDomain;

            const matchesPriority =
                selectedPriority === "ALL" ||
                row.dataset.swPriority === selectedPriority;

            row.hidden = !(
                matchesText &&
                matchesDomain &&
                matchesPriority
            );
        });
    }

    search.addEventListener(
        "input",
        filterFindings
    );

    domain.addEventListener(
        "change",
        filterFindings
    );

    priority.addEventListener(
        "change",
        filterFindings
    );

    function updateReviewBadges() {
        rows.forEach(function (row) {
            const id = row.dataset.swId;
            const review = savedReviews[id];

            const cell = row.querySelector(
                ".sw-review-state"
            );

            if (!cell) return;

            if (review) {
                cell.textContent = friendly(
                    review.status
                );
            } else {
                cell.textContent = "Not reviewed";
            }
        });
    }

    async function loadReviews() {
        try {
            const response = await fetch(
                "/api/sitewitness/reviews",
                {
                    cache: "no-store"
                }
            );

            if (!response.ok) {
                throw new Error(
                    "Review API unavailable"
                );
            }

            const data = await response.json();

            savedReviews = data.decisions || {};

            updateReviewBadges();

            if (selectedFindingId) {
                populateSavedReview(
                    selectedFindingId
                );
            }

        } catch (error) {
            console.error(
                "SiteWitness review loading failed:",
                error
            );

            rows.forEach(function (row) {
                const cell = row.querySelector(
                    ".sw-review-state"
                );

                if (cell) {
                    cell.textContent =
                        "Review status unavailable";
                }
            });
        }
    }

    function populateSavedReview(id) {
        const review = savedReviews[id];

        reviewStatus.value = review
            ? review.status
            : "NEEDS_REVIEW";

        reviewNote.value = review
            ? review.note || ""
            : "";
    }

    function openReview(id) {
        const finding = findings[id];

        if (!finding) {
            console.warn(
                "Unknown SiteWitness finding:",
                id
            );
            return;
        }

        selectedFindingId = id;

        rows.forEach(function (row) {
            row.classList.toggle(
                "sw-selected",
                row.dataset.swId === id
            );
        });

        reviewTitle.textContent = id;

        reviewDescription.textContent =
            finding.description || "";

        detailDomain.textContent =
            finding.domain || "Unknown";

        detailStatus.textContent =
            friendly(finding.status);

        detailPriority.textContent =
            String(finding.priority ?? "Unknown");

        detailAction.textContent =
            friendly(finding.recommended_action);

        populateSavedReview(id);

        reviewMessage.textContent = "";

        reviewPanel.hidden = false;

        reviewPanel.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    }

    rows.forEach(function (row) {
        row.addEventListener(
            "click",
            function () {
                openReview(
                    row.dataset.swId
                );
            }
        );

        row.addEventListener(
            "keydown",
            function (event) {
                if (
                    event.key === "Enter" ||
                    event.key === " "
                ) {
                    event.preventDefault();

                    openReview(
                        row.dataset.swId
                    );
                }
            }
        );
    });

    document.querySelectorAll(
        ".sw-anchor-review"
    ).forEach(function (button) {
        button.addEventListener(
            "click",
            function () {
                openReview(
                    button.dataset.swFinding
                );
            }
        );
    });

    reviewClose.addEventListener(
        "click",
        function () {
            reviewPanel.hidden = true;
            selectedFindingId = null;

            rows.forEach(function (row) {
                row.classList.remove(
                    "sw-selected"
                );
            });
        }
    );

    reviewSave.addEventListener(
        "click",
        async function () {
            if (!selectedFindingId) return;

            const idToSave = selectedFindingId;

            reviewSave.disabled = true;
            reviewMessage.textContent =
                "Saving review...";

            try {
                const response = await fetch(
                    "/api/sitewitness/reviews/" +
                    encodeURIComponent(idToSave),
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body: JSON.stringify({
                            status: reviewStatus.value,
                            note: reviewNote.value
                        })
                    }
                );

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(
                        data.error ||
                        "Review could not be saved"
                    );
                }

                savedReviews[idToSave] = data.review;

                updateReviewBadges();

                reviewMessage.textContent =
                    "Review saved successfully.";

            } catch (error) {
                reviewMessage.textContent =
                    error.message;
            } finally {
                reviewSave.disabled = false;
            }
        }
    );

    loadReviews();

})();
</script>

<!-- SITEWITNESS INTEGRATION END -->
"""

    replacements = {
        "__SW_CAD_COUNT__": _escape(
            cad_count
        ),
        "__SW_CORROBORATED__": _escape(
            corroborated
        ),
        "__SW_ABSTAINED__": _escape(
            abstained
        ),
        "__SW_FINDINGS__": findings_html,
        "__SW_ANCHORS__": anchors_html,
        "__SW_AVAILABILITY__": _escape(
            availability
        ),
        "__SW_FINDING_JSON__": finding_json,
    }

    for marker, value in replacements.items():
        section = section.replace(
            marker,
            value,
        )

    closing_body = "</body>"

    if closing_body not in page_html:
        raise RuntimeError(
            "Dashboard HTML has no closing body tag"
        )

    return page_html.replace(
        closing_body,
        section + "\n" + closing_body,
        1,
    )

# ============================================================
# SITEWITNESS — THREE-FLOOR DASHBOARD INTEGRATION
# ============================================================

_SITEWITNESS_ORIGINAL_ENHANCER = enhance_sitewitness


def _sw_read_json(path):
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return {}


def _sw_floor_state():
    summary = _sw_read_json(
        ROOT
        / "outputs/sitewitness/all_floors"
        / "building_inspection_summary.json"
    )

    floor_records = {
        int(item["floor"]): item
        for item in summary.get("floors", [])
        if isinstance(item, dict)
        and str(item.get("floor", "")).isdigit()
    }

    states = {}

    for floor in (1, 2, 3):
        record = floor_records.get(floor, {})

        line_report = _sw_read_json(
            ROOT
            / "outputs/sitewitness/all_floors"
            / f"floor_{floor}/line_report.json"
        )

        completed = (
            record.get("status") == "RECONSTRUCTION_COMPLETED"
            and bool(line_report)
        )

        states[floor] = {
            "reconstruction_completed": completed,
            "status": record.get(
                "status", "NOT_PROCESSED"
            ),
            "lines": (
                int(line_report.get(
                    "retained_supported_candidates", 0
                ))
                if completed else 0
            ),
            "frames": (
                int(line_report.get(
                    "frames_available", 0
                ))
                if completed else 0
            ),
        }

    return states


def _sw_floor_interface():
    states = _sw_floor_state()

    cards = []

    for floor in (1, 2, 3):
        state = states[floor]

        if state["reconstruction_completed"]:
            status = "Visual reconstruction available"
            detail = (
                f'{state["frames"]} sampled frames · '
                f'{state["lines"]} retained 3D line candidates'
            )
        elif state["status"] == "MISSING_LINE_OBSERVATIONS":
            status = "Awaiting walkthrough evidence"
            detail = (
                "No usable multiview line observations "
                "have been supplied for this floor."
            )
        else:
            status = "Inspection evidence unavailable"
            detail = (
                "Visual reconstruction has not been "
                "completed for this floor."
            )

        cards.append(
            f"""
            <button
                type="button"
                class="sw-floor-card"
                data-floor="{floor}"
                aria-pressed="{'true' if floor == 2 else 'false'}"
            >
                <span class="sw-floor-number">
                    FLOOR {floor:02d}
                </span>
                <strong>{_escape(status)}</strong>
                <small>{_escape(detail)}</small>
            </button>
            """
        )

    return """
    <div class="sw-floor-switch">
        <div class="sw-floor-intro">
            <span class="sw-eyebrow">
                BUILDING-WIDE INSPECTION
            </span>
            <h3>Choose a floor</h3>
            <p>
                One building. Three floors.
                Inspection evidence is reported independently
                for each floor.
            </p>
        </div>

        <div class="sw-floor-cards">
            __FLOOR_CARDS__
        </div>

        <div class="sw-floor-message" id="sw-floor-message"
             hidden>
            <strong id="sw-floor-message-title"></strong>
            <p id="sw-floor-message-text"></p>
        </div>
    </div>
    """.replace("__FLOOR_CARDS__", "\n".join(cards))


def enhance_sitewitness(page_html):
    """
    Add three-floor navigation to the existing SiteWitness
    dashboard without changing its routes or evidence APIs.
    """

    if 'id="sitewitness-inspection"' in page_html:
        return page_html

    page_html = _SITEWITNESS_ORIGINAL_ENHANCER(page_html)

    marker = '<div class="sw-container">'

    section_start = page_html.find(
        '<section id="sitewitness-inspection">'
    )

    if section_start < 0:
        return page_html

    container_position = page_html.find(
        marker, section_start
    )

    if container_position < 0:
        return page_html

    insertion = (
        marker
        + "\n"
        + _sw_floor_interface()
    )

    page_html = (
        page_html[:container_position]
        + insertion
        + page_html[container_position + len(marker):]
    )

    enhancement = """
<style>
.sw-floor-switch {
    margin-bottom: 38px;
}

.sw-floor-intro h3 {
    margin: 10px 0;
    font-size: 27px;
    letter-spacing: -.035em;
}

.sw-floor-intro p {
    color: #aaa8b9;
    font-size: 14px;
    line-height: 1.6;
}

.sw-floor-cards {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 22px;
}

.sw-floor-card {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    text-align: left;
    gap: 12px;
    padding: 23px;
    border: 1px solid #41414f;
    border-radius: 13px;
    background: #20212b;
    color: #f0eef7;
    cursor: pointer;
    min-height: 170px;
    font: inherit;
    transition: border-color .2s, background .2s;
}

.sw-floor-card:hover {
    border-color: #b89aff;
}

.sw-floor-card[aria-pressed="true"] {
    border-color: #b89aff;
    background: #30283e;
}

.sw-floor-number {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .15em;
    color: #b89aff;
}

.sw-floor-card strong {
    font-size: 16px;
    line-height: 1.4;
}

.sw-floor-card small {
    color: #aaa8b9;
    line-height: 1.6;
    font-size: 12px;
}

.sw-floor-message {
    margin-top: 20px;
    padding: 25px;
    background: #20212b;
    border: 1px solid #41414f;
    border-radius: 12px;
}

.sw-floor-message[hidden] {
    display: none;
}

.sw-floor-message strong {
    font-size: 19px;
}

.sw-floor-message p {
    color: #aaa8b9;
    line-height: 1.7;
}

/* Keep existing inspection evidence scoped to Floor 2. */
#sitewitness-inspection.sw-other-floor
.sw-container > :not(.sw-floor-switch) {
    display: none !important;
}

@media(max-width: 760px) {
    .sw-floor-cards {
        grid-template-columns: 1fr;
    }
}
</style>

<script>
(function () {
    "use strict";

    const section = document.getElementById(
        "sitewitness-inspection"
    );

    if (!section) return;

    const cards = section.querySelectorAll(
        ".sw-floor-card"
    );

    const message = document.getElementById(
        "sw-floor-message"
    );

    const title = document.getElementById(
        "sw-floor-message-title"
    );

    const text = document.getElementById(
        "sw-floor-message-text"
    );

    function selectFloor(floor) {
        cards.forEach(function (card) {
            card.setAttribute(
                "aria-pressed",
                String(card.dataset.floor === String(floor))
            );
        });

        const isFloorTwo = floor === 2;

        section.classList.toggle(
            "sw-other-floor",
            !isFloorTwo
        );

        message.hidden = isFloorTwo;

        if (!isFloorTwo) {
            title.textContent =
                "Floor " + floor + " — Awaiting visual evidence";

            text.textContent =
                "The building's existing 3D viewer remains " +
                "available. SiteWitness cannot verify this " +
                "floor against the CAD until walkthrough " +
                "frames, camera poses, and usable visual " +
                "observations are available.";
        }
    }

    cards.forEach(function (card) {
        card.addEventListener("click", function () {
            selectFloor(Number(card.dataset.floor));
        });
    });

    selectFloor(2);
})();
</script>
"""

    closing_body = "</body>"

    return page_html.replace(
        closing_body,
        enhancement + "\n" + closing_body,
        1,
    )

# ============================================================
# SITEWITNESS — REAL RECONSTRUCTION EVIDENCE PANEL
# ============================================================

_SW_PREVIOUS_ENHANCER = enhance_sitewitness


def _sw_real_evidence_panel():
    path = (
        ROOT
        / "outputs/sitewitness/all_floors/floor_2"
        / "line_candidates.json"
    )

    report_path = (
        ROOT
        / "outputs/sitewitness/all_floors/floor_2"
        / "line_report.json"
    )

    candidates_data = _sw_read_json(path)
    report = _sw_read_json(report_path)

    candidates = candidates_data.get("candidates", [])

    if not isinstance(candidates, list):
        candidates = []

    candidate_count = len(candidates)

    tested = report.get(
        "appearance_preselected_pairs_tested", 0
    )

    geometric = report.get(
        "geometrically_valid_pair_hypotheses", 0
    )

    frames = report.get("frames_available", 0)

    rows = []

    for index, candidate in enumerate(candidates[:30], 1):
        source_frames = candidate.get("source_frames", [])
        heldout = candidate.get("heldout_matches", [])

        frames_text = ", ".join(
            str(name) for name in source_frames[:2]
        )

        supporting = candidate.get(
            "supporting_frame_count", 0
        )

        length = candidate.get("length_m")

        try:
            length_text = f"{float(length):.2f}"
        except (TypeError, ValueError):
            length_text = "—"

        rows.append(
            f"""
            <tr>
                <td>{index:03d}</td>
                <td>{_escape(frames_text)}</td>
                <td>{_escape(supporting)}</td>
                <td>{_escape(length_text)}</td>
                <td>{len(heldout)}</td>
                <td>Unverified</td>
            </tr>
            """
        )

    return """
    <section id="sw-reconstruction-evidence">
        <div class="sw-evidence-heading">
            <span class="sw-eyebrow">
                FLOOR 02 · RECONSTRUCTED EVIDENCE
            </span>
            <h3>From walkthrough to geometry</h3>
            <p>
                These measurements come from the existing
                pose-constrained multiview reconstruction.
                They represent candidate visual lines,
                not confirmed structural elements.
            </p>
        </div>

        <div class="sw-evidence-stats">
            <div>
                <strong>__FRAMES__</strong>
                <span>Sampled frames</span>
            </div>
            <div>
                <strong>__TESTED__</strong>
                <span>Appearance-selected pairs tested</span>
            </div>
            <div>
                <strong>__GEOMETRIC__</strong>
                <span>Geometric hypotheses</span>
            </div>
            <div>
                <strong>__CANDIDATES__</strong>
                <span>Retained line candidates</span>
            </div>
        </div>

        <div class="sw-evidence-boundary">
            <strong>CAD registration: Not verified</strong>
            <p>
                Visual reconstruction and CAD geometry
                are not yet uniquely aligned.
                No construction discrepancy is claimed.
            </p>
        </div>

        <h4>Reconstructed line evidence</h4>

        <div class="sw-table-wrap">
            <table class="sw-table">
                <thead>
                    <tr>
                        <th>Candidate</th>
                        <th>Source frames</th>
                        <th>Supporting frames</th>
                        <th>3D length (provisional units)</th>
                        <th>Held-out matches</th>
                        <th>Structural identity</th>
                    </tr>
                </thead>
                <tbody>
                    __ROWS__
                </tbody>
            </table>
        </div>

        <p class="sw-footer">
            Showing up to 30 of __CANDIDATES__
            retained candidates. Duplicate geometric
            hypotheses may exist.
        </p>
    </section>
    """.replace(
        "__FRAMES__", _escape(frames)
    ).replace(
        "__TESTED__", _escape(f"{tested:,}")
    ).replace(
        "__GEOMETRIC__", _escape(f"{geometric:,}")
    ).replace(
        "__CANDIDATES__", _escape(candidate_count)
    ).replace(
        "__ROWS__", "\n".join(rows)
    )


def enhance_sitewitness(page_html):
    """
    Preserve the existing three-floor SiteWitness interface
    and insert real Floor 2 reconstruction evidence.
    """

    if 'id="sitewitness-inspection"' in page_html:
        return page_html

    page_html = _SW_PREVIOUS_ENHANCER(page_html)

    if 'id="sw-reconstruction-evidence"' in page_html:
        return page_html

    marker = '<h3 class="sw-section-title">'

    section_start = page_html.find(
        '<section id="sitewitness-inspection">'
    )

    insertion_point = page_html.find(
        marker, section_start
    )

    if insertion_point < 0:
        return page_html

    panel = _sw_real_evidence_panel()

    page_html = (
        page_html[:insertion_point]
        + panel
        + "\n"
        + page_html[insertion_point:]
    )

    styling = """
<style>
#sw-reconstruction-evidence {
    margin: 44px 0;
    padding: 28px;
    background: #1d1e28;
    border: 1px solid #41414f;
    border-radius: 16px;
}

.sw-evidence-heading h3 {
    font-size: 28px;
    letter-spacing: -.035em;
    margin: 10px 0;
}

.sw-evidence-heading p {
    color: #aaa8b9;
    line-height: 1.7;
    max-width: 760px;
}

.sw-evidence-stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 26px 0;
}

.sw-evidence-stats > div {
    padding: 18px;
    border: 1px solid #41414f;
    border-radius: 10px;
    background: #252633;
}

.sw-evidence-stats strong {
    display: block;
    font-size: 27px;
    letter-spacing: -.04em;
}

.sw-evidence-stats span {
    display: block;
    color: #aaa8b9;
    font-size: 12px;
    margin-top: 6px;
}

.sw-evidence-boundary {
    border-left: 3px solid #f0a84b;
    background: #2b271f;
    padding: 18px;
    margin-bottom: 28px;
}

.sw-evidence-boundary p {
    color: #c9c1b6;
    font-size: 13px;
    line-height: 1.6;
}

@media(max-width: 760px) {
    .sw-evidence-stats {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
"""

    return page_html.replace(
        "</body>",
        styling + "\n</body>",
        1,
    )
