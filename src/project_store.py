
"""
Under-construction — Step 116A
Persistent local project storage.

Run:
    .\\.venv\\Scripts\\python.exe src\\project_store.py

API:
    GET  /api/projects
    POST /api/projects
    GET  /api/projects/<project_id>
    GET  /api/projects/<project_id>/files/<kind>
    GET  /api/projects/health

Storage:
    data/user_projects/<project_id>/
        project.json
        floor_plan.dxf
        walkthrough.<extension>

No CAD-to-video registration or progress verification.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "user_projects"

HOST = "127.0.0.1"
PORT = int(os.environ.get("PROJECT_STORE_PORT", "5053"))

MAX_DXF = 20 * 1024 * 1024
MAX_VIDEO = 500 * 1024 * 1024

VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
PROJECT_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = (
    MAX_DXF + MAX_VIDEO + 1024 * 1024
)

write_lock = Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def error(message, status=400):
    return jsonify({"error": message}), status


def valid_id(project_id):
    return bool(PROJECT_ID_PATTERN.fullmatch(project_id))


def project_directory(project_id):
    if not valid_id(project_id):
        return None
    return STORE / project_id


def read_metadata(project_id):
    folder = project_directory(project_id)
    if folder is None:
        return None

    path = folder / "project.json"

    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    if data.get("project_id") != project_id:
        return None

    return data


def save_metadata(folder, metadata):
    destination = folder / "project.json"

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=folder,
        prefix=".metadata_",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)

        json.dump(
            metadata,
            handle,
            indent=2,
            allow_nan=False,
        )

    try:
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def save_upload(upload, destination, limit):
    total = 0

    with destination.open("xb") as output:
        while True:
            chunk = upload.stream.read(1024 * 1024)

            if not chunk:
                break

            total += len(chunk)

            if total > limit:
                raise ValueError(
                    f"{destination.name} exceeds its size limit."
                )

            output.write(chunk)

    if total == 0:
        raise ValueError(
            f"{destination.name} is empty."
        )

    return total


def public_metadata(metadata):
    result = dict(metadata)

    result["file_urls"] = {
        "plan": (
            f"/api/projects/{metadata['project_id']}"
            "/files/plan"
        ),
        "video": (
            f"/api/projects/{metadata['project_id']}"
            "/files/video"
        ) if metadata["files"]["video"] else None,
    }

    return result


@app.get("/api/projects/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "underconstruction_project_store",
        "storage": "local",
    })


@app.get("/api/projects")
def list_projects():
    STORE.mkdir(parents=True, exist_ok=True)

    projects = []

    for folder in STORE.iterdir():
        if not folder.is_dir():
            continue

        metadata = read_metadata(folder.name)

        if metadata is not None:
            projects.append(public_metadata(metadata))

    projects.sort(
        key=lambda item: item.get("created_at", ""),
        reverse=True,
    )

    return jsonify({
        "count": len(projects),
        "projects": projects,
    })


@app.post("/api/projects")
def create_project():
    plan = request.files.get("plan")
    video = request.files.get("video")

    if plan is None or not plan.filename:
        return error("A DXF floor plan is required.")

    if Path(plan.filename).suffix.lower() != ".dxf":
        return error("Floor plan must be a .dxf file.")

    video_ext = None

    if video is not None and video.filename:
        video_ext = Path(video.filename).suffix.lower()

        if video_ext not in VIDEO_EXTENSIONS:
            return error(
                "Video must be MP4, MOV, or WebM."
            )
    else:
        video = None

    name = str(
        request.form.get("name", "")
    ).strip()

    if not name:
        name = Path(plan.filename).stem

    if len(name) > 100:
        return error(
            "Project name must be at most 100 characters."
        )

    project_id = uuid.uuid4().hex
    folder = STORE / project_id

    with write_lock:
        folder.mkdir(parents=True, exist_ok=False)

    try:
        plan_size = save_upload(
            plan,
            folder / "floor_plan.dxf",
            MAX_DXF,
        )

        video_name = None
        video_size = None

        if video is not None:
            video_name = "walkthrough" + video_ext
            video_size = save_upload(
                video,
                folder / video_name,
                MAX_VIDEO,
            )

        metadata = {
            "schema_version": "1.0",
            "project_id": project_id,
            "name": name,
            "created_at": now(),
            "project_type": "USER_UPLOAD",
            "files": {
                "plan": {
                    "filename": "floor_plan.dxf",
                    "original_name": Path(
                        plan.filename
                    ).name,
                    "size_bytes": plan_size,
                },
                "video": {
                    "filename": video_name,
                    "original_name": Path(
                        video.filename
                    ).name,
                    "size_bytes": video_size,
                } if video_name else None,
            },
            "analysis": {
                "cad_parsed": False,
                "video_processed": False,
                "cad_video_registered": False,
                "physical_correspondence_verified": False,
                "construction_progress_verified": False,
                "completion_percentage": None,
            },
            "status": "FILES_SAVED_NOT_ANALYZED",
        }

        save_metadata(folder, metadata)

    except Exception:
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        raise

    return jsonify(
        public_metadata(metadata)
    ), 201


@app.get("/api/projects/<project_id>")
def get_project(project_id):
    metadata = read_metadata(project_id)

    if metadata is None:
        return error("Project not found.", 404)

    return jsonify(public_metadata(metadata))


@app.get("/api/projects/<project_id>/files/<kind>")
def get_project_file(project_id, kind):
    metadata = read_metadata(project_id)

    if metadata is None:
        return error("Project not found.", 404)

    if kind not in {"plan", "video"}:
        return error("Unknown file type.", 404)

    record = metadata["files"].get(kind)

    if record is None:
        return error("File not available.", 404)

    filename = record.get("filename")

    # Never trust arbitrary filenames from metadata.
    allowed = (
        filename == "floor_plan.dxf"
        if kind == "plan"
        else filename in {
            "walkthrough.mp4",
            "walkthrough.mov",
            "walkthrough.webm",
        }
    )

    if not allowed:
        return error("Invalid stored filename.", 500)

    path = project_directory(project_id) / filename

    if not path.is_file():
        return error("Stored file is missing.", 404)

    mimetypes = {
        ".dxf": "application/dxf",
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".webm": "video/webm",
    }

    return send_file(
        path,
        mimetype=mimetypes.get(
            path.suffix.lower(),
            "application/octet-stream",
        ),
        conditional=True,
    )


@app.errorhandler(413)
def too_large(_):
    return error(
        "Upload exceeds the server request-size limit.",
        413,
    )


@app.errorhandler(HTTPException)
def http_error(exc):
    return error(exc.description, exc.code)


if __name__ == "__main__":
    STORE.mkdir(parents=True, exist_ok=True)

    print("\nUNDER-CONSTRUCTION / PROJECT STORE")
    print("Storage:", STORE)
    print(f"API: http://{HOST}:{PORT}/api/projects")
    print("Mode: local storage, no progress verification\n")

    app.run(
        host=HOST,
        port=PORT,
        debug=False,
        use_reloader=False,
    )
