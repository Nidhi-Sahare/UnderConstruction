
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "src" / "digital_twin_3d.py"
BACKUP = ROOT / "src" / "digital_twin_3d.before_recovery.py"

MARKER = "// RECOVERED_WALL_PREVIEW_V1"

INTEGRATION = r"""
// RECOVERED_WALL_PREVIEW_V1
// CAD-edge hypotheses, not verified construction geometry.

const recoveredWallGroup = new THREE.Group();
recoveredWallGroup.name = "Unverified recovered wall hypotheses";
scene.add(recoveredWallGroup);

const recoveredWallMaterial = new THREE.MeshStandardMaterial({
    color: 0xf59e0b,
    transparent: true,
    opacity: 0.32,
    roughness: 0.9,
    side: THREE.DoubleSide,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1
});

const recoveredWallEdgeMaterial = new THREE.LineBasicMaterial({
    color: 0xfbbf24,
    transparent: true,
    opacity: 0.65,
    depthTest: true
});

recoveredWallGroup.visible = false;

let recoveredWallCount = 0;

function installRecoveredWallControl() {
    if (document.getElementById("showRecoveredWalls")) {
        return;
    }

    const label = document.createElement("label");
    label.className = "toggle";
    label.style.marginTop = "14px";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.id = "showRecoveredWalls";
    checkbox.checked = false;

    const text = document.createElement("span");
    text.textContent =
        "Recovered wall hypotheses (unverified)";

    label.append(checkbox, text);

    const description = document.createElement("div");
    description.id = "recoveredWallCount";
    description.className = "muted";
    description.textContent = "Loading recovery candidates...";

    const anchor = document.getElementById("geometrySource");

    if (anchor && anchor.parentElement) {
        anchor.insertAdjacentElement("afterend", description);
        description.insertAdjacentElement("afterend", label);
    } else {
        document.querySelector("aside.left")?.append(
            description, label
        );
    }

    checkbox.addEventListener("change", () => {
        recoveredWallGroup.visible = checkbox.checked;

        if (checkbox.checked) {
            const box = new THREE.Box3()
                .setFromObject(recoveredWallGroup);

            if (!box.isEmpty()) {
                const originalBox = new THREE.Box3()
                    .setFromObject(buildingGroup);

                originalBox.union(box);

                const center = originalBox.getCenter(
                    new THREE.Vector3()
                );

                const size = originalBox.getSize(
                    new THREE.Vector3()
                );

                const extent = Math.max(
                    size.x, size.y, size.z, 1
                );

                controls.target.copy(center);

                camera.position.set(
                    center.x + extent * 0.85,
                    center.y + extent * 0.75,
                    center.z + extent * 0.95
                );

                controls.update();
            }
        }
    });
}

function validRecoveryOutline(points) {
    return Array.isArray(points)
        && points.length === 4
        && points.every(p =>
            Array.isArray(p)
            && p.length === 2
            && p.every(Number.isFinite)
        );
}

function makeRecoveredWallMesh(item) {
    const outline = item.outline_xy;

    if (!validRecoveryOutline(outline)) {
        return null;
    }

    const height = 3.0; // Display assumption only

    const shape = new THREE.Shape();

    outline.forEach((point, index) => {
        const x = point[0] - modelCenter.x;
        const y = point[1] - modelCenter.z;

        if (index === 0) {
            shape.moveTo(x, y);
        } else {
            shape.lineTo(x, y);
        }
    });

    shape.closePath();

    const geometry = new THREE.ExtrudeGeometry(shape, {
        depth: height,
        bevelEnabled: false,
        steps: 1,
        curveSegments: 1
    });

    geometry.rotateX(-Math.PI / 2);

    const mesh = new THREE.Mesh(
        geometry,
        recoveredWallMaterial
    );

    mesh.userData = {
        kind: "recovered_wall",
        item,
        verified: false,
        previewOnly: true,
        constructionStatus: "UNKNOWN"
    };

    const edges = new THREE.LineSegments(
        new THREE.EdgesGeometry(geometry),
        recoveredWallEdgeMaterial
    );

    mesh.add(edges);

    return mesh;
}

async function loadRecoveredWalls() {
    installRecoveredWallControl();

    const status = document.getElementById(
        "recoveredWallCount"
    );

    try {
        const response = await fetch(
            "/api/twin/recovered-walls"
        );

        if (!response.ok) {
            throw new Error(
                "Recovery API returned HTTP " + response.status
            );
        }

        const data = await response.json();

        if (data.mode !== "INFERRED_PREVIEW_ONLY") {
            throw new Error(
                "Unexpected recovery mode: " + data.mode
            );
        }

        if (!Array.isArray(data.walls)) {
            throw new Error(
                "Recovery response has no walls array"
            );
        }

        for (const item of data.walls) {
            if (item.preview_only !== true) {
                continue;
            }

            const mesh = makeRecoveredWallMesh(item);

            if (mesh) {
                recoveredWallGroup.add(mesh);
                recoveredWallCount++;
            }
        }

        if (status) {
            status.textContent =
                recoveredWallCount +
                " inferred CAD wall candidates Â· " +
                "not verified or merged";
        }

        console.log(
            "Recovered wall hypotheses rendered:",
            recoveredWallCount
        );

    } catch (error) {
        console.error(
            "Recovered wall preview unavailable:",
            error
        );

        if (status) {
            status.textContent =
                "Recovered wall preview unavailable: " +
                error.message;
        }
    }
}
"""

def main():
    if not TARGET.is_file():
        sys.exit(f"ERROR: Cannot find {TARGET}")

    source = TARGET.read_text(encoding="utf-8-sig")

    if MARKER in source:
        sys.exit(
            "Recovery preview already installed. "
            "No changes made."
        )

    if source.count("</script>") < 2:
        sys.exit(
            "ERROR: Expected exactly one closing script tag."
        )

    anchor = "        buildWalls(surfaces);"

    if source.count(anchor) != 1:
        sys.exit(
            "ERROR: Could not locate the unique buildWalls call."
        )

    if "const modelCenter" in source:
        sys.exit(
            "ERROR: Unexpected modelCenter declaration."
        )

    # Keep existing geometry and camera initialization.
    # Load hypotheses only after original walls establish
    # the CAD model center.
    source = source.replace(
        anchor,
        anchor + "\n        await loadRecoveredWalls();",
        1
    )

    source = source.replace(
        "</script>",
        INTEGRATION + "\n</script>",
        1
    ) if False else source.rsplit("</script>", 1)[0] + INTEGRATION + "\n</script>" + source.rsplit("</script>", 1)[1]

    # Check Python syntax before overwriting the original.
    try:
        compile(source, str(TARGET), "exec")
    except SyntaxError as exc:
        sys.exit(f"ERROR: Generated Python is invalid: {exc}")

    shutil.copy2(TARGET, BACKUP)
    TARGET.write_text(source, encoding="utf-8")

    print("SUCCESS: Recovered wall preview installed.")
    print("Updated:", TARGET)
    print("Backup:", BACKUP)
    print("Original walls preserved.")
    print("Recovered hypotheses are hidden by default.")
    print("Restart the dashboard and refresh the browser.")

if __name__ == "__main__":
    main()


