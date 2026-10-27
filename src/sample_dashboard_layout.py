
"""
Under-construction — Sample 01 dashboard layout

Moves the existing working three-storey viewer to the top of
the Sample 01 workspace.

Preserves:
- Three-storey geometry and floor controls
- Original Floor 2 CAD and walkthrough
- Original Floor 2 3D viewer
- Personal project uploads
- Existing dashboard styling
"""

LAYOUT_CSS = """
<style>
#uc-building-panel {
    margin-top: 20px;
    margin-bottom: 26px;
}

#uc-building-panel #uc-building-canvas {
    height: min(65vh, 680px);
    min-height: 430px;
}

#sample-one .project-name {
    font-weight: 700;
}

#sample-one .project-note {
    line-height: 1.5;
}

.uc-sample-intro {
    margin: 8px 0 18px;
    color: #aaa5b9;
    font-size: 13px;
    line-height: 1.7;
}

.uc-sample-intro strong {
    color: #f0eef7;
}

@media (max-width: 700px) {
    #uc-building-panel #uc-building-canvas {
        height: 420px;
        min-height: 320px;
    }
}
</style>
"""

LAYOUT_SCRIPT = """
<script>
(function () {
    "use strict";

    const byId = id => document.getElementById(id);

    function updateSidebar() {
        const sample = byId("sample-one");
        if (!sample) return;

        const title = sample.querySelector(".project-name");
        const description = sample.querySelector(".project-note");

        if (title) {
            title.textContent = "Hilti Construction Site";
        }

        if (description) {
            description.textContent =
                "3 real CAD floors + walkthrough";
        }

        sample.setAttribute(
            "aria-label",
            "Explore Hilti three-storey construction site"
        );
    }

    function moveBuildingToTop() {
        const viewer = byId("viewer-view");
        const building = byId("uc-building-panel");

        if (!viewer || !building) {
            console.warn(
                "[Sample Layout] Viewer or building panel missing"
            );
            return;
        }

        // Keep the original viewer DOM intact.
        // Move the actual Three.js canvas rather than copying it.
        // This preserves its geometry, renderer and controls.

        const intro = viewer.querySelector(".intro");

        if (intro && intro.parentElement === viewer) {
            intro.insertAdjacentElement(
                "afterend",
                building
            );
        } else {
            viewer.insertBefore(
                building,
                viewer.firstChild
            );
        }

        // Add a short explanation immediately above the model.
        if (!byId("uc-sample-intro")) {
            const introduction = document.createElement("div");

            introduction.id = "uc-sample-intro";
            introduction.className = "uc-sample-intro";

            introduction.innerHTML =
                "<strong>Three floors. One building.</strong> " +
                "Explore the real Hilti architectural drawings " +
                "as a CAD-derived 3D model. Select individual " +
                "floors or separate them in the exploded view.";

            building.parentElement.insertBefore(
                introduction,
                building
            );
        }

        // This panel is now the first main visualization.
        const buildingTitle = building.querySelector(
            ".uc-building-head strong"
        );

        if (buildingTitle) {
            buildingTitle.textContent =
                "01 / HILTI · THREE-STOREY DIGITAL TWIN";
        }

        // Keep the existing detailed Floor 2 panel.
        const oldPanel = byId("uc-3d-panel");

        if (oldPanel) {
            const heading = oldPanel.querySelector(
                ".uc-3d-head"
            );

            if (heading) {
                for (const node of heading.childNodes) {
                    if (
                        node.nodeType === Node.TEXT_NODE &&
                        node.textContent.includes("03 /")
                    ) {
                        node.textContent =
                            node.textContent.replace(
                                "03 /",
                                "04 /"
                            );
                    }
                }
            }
        }
    }

    let sampleOpened = false;
    let opening = false;

    function sampleIsVisible() {
        const viewer = byId("viewer-view");
        const sample = byId("sample-one");

        return Boolean(
            viewer &&
            !viewer.classList.contains("hidden") &&
            sample &&
            sample.classList.contains("active")
        );
    }

    function activateBuilding() {
        if (!sampleIsVisible()) {
            sampleOpened = false;
            return;
        }

        if (sampleOpened || opening) return;

        const button = byId("uc-open-three-floors");

        if (!button) {
            console.warn(
                "[Sample Layout] Three-floor button missing"
            );
            return;
        }

        opening = true;

        // Reuse the existing, tested reconstruction loader.
        // No duplicate fetch, no duplicate Three.js renderer.
        requestAnimationFrame(() => {
            button.click();

            sampleOpened = true;
            opening = false;
        });
    }

    function updateSampleHeading() {
        if (!sampleIsVisible()) return;

        const eyebrow = byId("workspace-eyebrow");
        const title = byId("workspace-title");

        if (eyebrow) {
            eyebrow.textContent =
                "SAMPLE 01 / HILTI CONSTRUCTION SITE";
        }

        if (title) {
            title.textContent = "Explore the building.";
        }
    }

    function sync() {
        updateSampleHeading();
        activateBuilding();
    }

    function initialize() {
        updateSidebar();
        moveBuildingToTop();

        // The original dashboard controls which workspace is visible.
        // Observe that existing state rather than overriding handlers.
        const viewer = byId("viewer-view");
        const sample = byId("sample-one");

        if (viewer) {
            const observer = new MutationObserver(sync);

            observer.observe(viewer, {
                attributes: true,
                attributeFilter: ["class"]
            });
        }

        if (sample) {
            const observer = new MutationObserver(sync);

            observer.observe(sample, {
                attributes: true,
                attributeFilter: ["class"]
            });
        }

        // Allow reopening Sample 01 after switching to uploads.
        const upload = byId("upload-view");

        if (upload) {
            const observer = new MutationObserver(() => {
                if (
                    !upload.classList.contains("hidden")
                ) {
                    sampleOpened = false;
                }
            });

            observer.observe(upload, {
                attributes: true,
                attributeFilter: ["class"]
            });
        }

        sync();

        console.info(
            "[Sample Layout] Three-storey viewer promoted"
        );
    }

    if (document.readyState === "loading") {
        document.addEventListener(
            "DOMContentLoaded",
            initialize,
            {once: true}
        );
    } else {
        initialize();
    }
})();
</script>
"""


def enhance_sample_layout(html: str) -> str:
    """Apply the Sample 01 layout without modifying the viewer."""

    if "uc-sample-intro" in html:
        return html

    if "</head>" not in html or "</body>" not in html:
        raise ValueError(
            "Expected a complete dashboard HTML document"
        )

    html = html.replace(
        "</head>",
        LAYOUT_CSS + "\n</head>",
        1,
    )

    html = html.replace(
        "</body>",
        LAYOUT_SCRIPT + "\n</body>",
        1,
    )

    return html
