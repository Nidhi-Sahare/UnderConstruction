
"""
SiteWitness — Human Evidence Review

Adds a persistent, auditable review workflow to BuildProgress.

Read-only evidence:
    outputs/sitewitness/inspection_summary.json

Reviewer decisions:
    outputs/sitewitness/review_decisions.json

Endpoints:
    GET  /api/sitewitness/reviews
    POST /api/sitewitness/reviews/<finding_id>

Review status does NOT imply construction verification.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading

from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, request


ROOT = Path(__file__).resolve().parents[1]

REPORT = (
    ROOT / "outputs/sitewitness/inspection_summary.json"
)

DECISIONS = (
    ROOT / "outputs/sitewitness/review_decisions.json"
)

review_bp = Blueprint(
    "sitewitness_reviews",
    __name__
)

LOCK = threading.Lock()

VALID_STATUSES = {
    "NEEDS_REVIEW",
    "REVIEWED",
    "DISMISSED",
}

MAX_NOTE_LENGTH = 2000


def _read_json(path, default):
    if not path.exists():
        return default

    with path.open(
        "r",
        encoding="utf-8"
    ) as handle:
        return json.load(handle)


def _write_atomic(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    descriptor, temporary = tempfile.mkstemp(
        prefix=".sitewitness_",
        suffix=".json",
        dir=path.parent,
    )

    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8"
        ) as handle:
            json.dump(
                data,
                handle,
                indent=2,
                ensure_ascii=False,
            )

            handle.flush()
            os.fsync(handle.fileno())

        os.replace(
            temporary,
            path
        )

    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _known_findings():
    report = _read_json(
        REPORT,
        {}
    )

    return {
        item["finding_id"]
        for item in report.get(
            "findings",
            []
        )
        if item.get("finding_id")
    }


def _load_decisions():
    data = _read_json(
        DECISIONS,
        {}
    )

    return data.get(
        "decisions",
        {}
    )


@review_bp.get("/api/sitewitness/reviews")
def get_reviews():
    with LOCK:
        try:
            decisions = _load_decisions()
        except (OSError, ValueError):
            return jsonify({
                "error": "Unable to read review decisions"
            }), 500

    return jsonify({
        "schema_version": "1.0",
        "decisions": decisions,
        "review_is_not_verification": True,
    })


@review_bp.post(
    "/api/sitewitness/reviews/<finding_id>"
)
def save_review(finding_id):
    if not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}",
        finding_id
    ):
        return jsonify({
            "error": "Invalid finding ID"
        }), 400

    payload = request.get_json(
        silent=True
    )

    if not isinstance(
        payload,
        dict
    ):
        return jsonify({
            "error": "Expected a JSON object"
        }), 400

    status = payload.get(
        "status"
    )

    note = payload.get(
        "note",
        ""
    )

    if status not in VALID_STATUSES:
        return jsonify({
            "error": "Invalid review status"
        }), 400

    if not isinstance(
        note,
        str
    ):
        return jsonify({
            "error": "Note must be text"
        }), 400

    note = note.strip()

    if len(note) > MAX_NOTE_LENGTH:
        return jsonify({
            "error": "Note exceeds 2000 characters"
        }), 400

    with LOCK:
        try:
            known = _known_findings()

            if finding_id not in known:
                return jsonify({
                    "error": "Unknown finding ID"
                }), 404

            decisions = _load_decisions()

            record = {
                "finding_id": finding_id,
                "status": status,
                "note": note,
                "updated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "construction_verified": False,
            }

            decisions[finding_id] = record

            _write_atomic(
                DECISIONS,
                {
                    "schema_version": "1.0",
                    "decisions": decisions,
                }
            )

        except (OSError, ValueError):
            return jsonify({
                "error": "Could not save review"
            }), 500

    return jsonify({
        "saved": True,
        "review": record,
        "review_is_not_verification": True,
    })
