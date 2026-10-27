
"""
Floor 1 / Floor 3 walkthrough trajectory overlay.

Adds a route and moving camera marker directly over the
existing CAD plan preview.

Uses the existing /api/floor-walkthrough/<floor> endpoint.

IMPORTANT:
The trajectory-to-CAD placement is an approximate,
bounds-fitted preview, NOT verified CAD registration.

Floor 2 remains unchanged.
"""

CSS = """
<style>
#uc-floor-plan-overlay {
    position: absolute !important;
    inset: 0;
    overflow: hidden;
}

#uc-floor-route-svg {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 5;
}

#uc-floor-route-track {
    fill: none;
    stroke: #f0784b;
    stroke-width: 0.16;
    stroke-linecap: round;
    stroke-linejoin: round;
    opacity: .95;
}

#uc-floor-route-progress {
    fill: none;
    stroke: #c9a5ff;
    stroke-width: .21;
    stroke-linecap: round;
    stroke-linejoin: round;
}

#uc-floor-route-marker {
    fill: #f0784b;
    stroke: white;
    stroke-width: .13;
}

#uc-floor-route-halo {
    fill: #f0784b;
    opacity: .16;
}

#uc-floor-route-note {
    position: absolute;
    left: 14px;
    bottom: 14px;
    z-index: 7;
    padding: 9px 12px;
    border: 1px solid #735c4c;
    background: #17151bdf;
    color: #f3c19d;
    font: 10px Consolas, monospace;
    border-radius: 7px;
    max-width: 85%;
    line-height: 1.5;
}

/* The route is now on the floor plan, not over the video. */
.uc-fw-route {
    display: none !important;
}
</style>
"""

SCRIPT = r"""
<script>
(() => {
    "use strict";

    const $ = id => document.getElementById(id);

    let activeFloor = 2;
    let activeData = null;
    let points = [];
    let positions = [];
    let loadToken = 0;
    let routeSVG = null;
    let lastIndex = -1;

    const NS = "http://www.w3.org/2000/svg";

    function element(name, attrs = {}) {
        const node = document.createElementNS(NS, name);

        for (const [key, value] of Object.entries(attrs)) {
            node.setAttribute(key, value);
        }

        return node;
    }

    function clear() {
        points = [];
        positions = [];
        activeData = null;
        lastIndex = -1;

        if (routeSVG) {
            routeSVG.remove();
            routeSVG = null;
        }

        $("uc-floor-route-note")?.remove();
    }

    function finitePoint(point) {
        return Array.isArray(point) &&
            point.length >= 2 &&
            Number.isFinite(Number(point[0])) &&
            Number.isFinite(Number(point[1]));
    }

    function createOverlay(viewBox) {
        const parent = $("uc-floor-plan-overlay");
        if (!parent) return null;

        routeSVG = element("svg", {
            id: "uc-floor-route-svg",
            viewBox: viewBox.join(" "),
            preserveAspectRatio: "xMidYMid meet"
        });

        const track = element("path", {
            id: "uc-floor-route-track"
        });

        const progress = element("path", {
            id: "uc-floor-route-progress"
        });

        const halo = element("circle", {
            id: "uc-floor-route-halo",
            r: "0.8"
        });

        const marker = element("circle", {
            id: "uc-floor-route-marker",
            r: "0.35"
        });

        routeSVG.append(track, progress, halo, marker);
        parent.append(routeSVG);

        const note = document.createElement("div");
        note.id = "uc-floor-route-note";
        note.textContent =
            "APPROXIMATE ROUTE PLACEMENT · CAD ALIGNMENT UNVERIFIED";

        parent.append(note);

        return routeSVG;
    }

    function pathString(list) {
        if (!list.length) return "";

        return list.map((p, i) =>
            `${i ? "L" : "M"}${p[0].toFixed(4)},${p[1].toFixed(4)}`
        ).join(" ");
    }

    function projectToPreview(route, box) {
        /*
         * Fit the measured trajectory inside the visible
         * CAD drawing area.
         *
         * This is NOT a registration algorithm.
         * It does not establish where the camera actually
         * was relative to walls or rooms.
         */
        const xs = route.map(p => p[0]);
        const ys = route.map(p => p[1]);

        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);

        const spanX = Math.max(maxX - minX, 0.001);
        const spanY = Math.max(maxY - minY, 0.001);

        const [vx, vy, vw, vh] = box;

        const margin = Math.min(vw, vh) * 0.12;

        const availableW = Math.max(vw - margin * 2, 1);
        const availableH = Math.max(vh - margin * 2, 1);

        const scale = Math.min(
            availableW / spanX,
            availableH / spanY
        );

        const offsetX =
            vx + (vw - spanX * scale) / 2;

        const offsetY =
            vy + (vh - spanY * scale) / 2;

        return point => [
            offsetX + (point[0] - minX) * scale,
            offsetY + (maxY - point[1]) * scale
        ];
    }

    function currentFrameIndex() {
        const slider = $("uc-fw-slider");

        if (!slider) return 0;

        const value = Number(slider.value);

        return Number.isFinite(value)
            ? Math.max(0, Math.floor(value))
            : 0;
    }

    function updateMarker() {
        if (!routeSVG || !points.length) return;

        const index = currentFrameIndex();

        if (index === lastIndex) return;
        lastIndex = index;

        const frames = activeData?.frames || [];

        let position = null;
        let progressIndex = 0;

        // Use a timestamp-matched camera position when
        // the existing API provides one.
        const frame = frames[index];

        if (frame && finitePoint(frame.position) &&
            typeof positions.project === "function") {

            position = positions.project(frame.position);

            const fraction = frames.length > 1
                ? index / (frames.length - 1)
                : 0;

            progressIndex = Math.round(
                fraction * (points.length - 1)
            );

        } else {
            /*
             * When gallery timestamps are unavailable,
             * only animate approximate progress through
             * the recorded route.
             *
             * This is NOT frame-accurate localization.
             */
            const fraction = frames.length > 1
                ? Math.min(1, index / (frames.length - 1))
                : 0;

            progressIndex = Math.round(
                fraction * (points.length - 1)
            );

            position = points[progressIndex];
        }

        if (!position) return;

        const [x, y] = position;

        $("uc-floor-route-marker")
            ?.setAttribute("cx", x);

        $("uc-floor-route-marker")
            ?.setAttribute("cy", y);

        $("uc-floor-route-halo")
            ?.setAttribute("cx", x);

        $("uc-floor-route-halo")
            ?.setAttribute("cy", y);

        $("uc-floor-route-progress")
            ?.setAttribute(
                "d",
                pathString(points.slice(0, progressIndex + 1))
            );

        const note = $("uc-floor-route-note");

        if (note) {
            note.textContent =
                frame && finitePoint(frame.position)
                ? "TIMESTAMP-MATCHED POSITION · CAD PLACEMENT APPROXIMATE"
                : "ROUTE PROGRESS ESTIMATED · CAD ALIGNMENT UNVERIFIED";
        }
    }

    async function load(floor) {
        const token = ++loadToken;

        activeFloor = floor;
        clear();

        if (floor === 2) return;
        if (![1, 3].includes(floor)) return;

        try {
            const [routeResponse, planResponse] =
                await Promise.all([
                    fetch(`/api/floor-walkthrough/${floor}`),
                    fetch(`/api/sample/floor/${floor}/plan.svg`)
                ]);

            if (!routeResponse.ok || !planResponse.ok) {
                throw new Error("Floor evidence unavailable");
            }

            const data = await routeResponse.json();
            const svgText = await planResponse.text();

            if (token !== loadToken) return;

            const xml = new DOMParser().parseFromString(
                svgText,
                "image/svg+xml"
            );

            const sourceSVG =
                xml.documentElement;

            if (
                !sourceSVG ||
                sourceSVG.localName !== "svg"
            ) {
                throw new Error("Invalid CAD SVG");
            }

            const box = sourceSVG
                .getAttribute("viewBox")
                ?.split(/[,\s]+/)
                .map(Number);

            if (
                !box ||
                box.length !== 4 ||
                !box.every(Number.isFinite)
            ) {
                throw new Error("CAD viewBox unavailable");
            }

            const route = (data.trajectory || [])
                .filter(finitePoint)
                .map(p => [
                    Number(p[0]),
                    Number(p[1])
                ]);

            if (route.length < 2) {
                throw new Error("Recorded route unavailable");
            }

            const project = projectToPreview(route, box);

            positions = [];
            positions.project = project;

            points = route.map(project);
            activeData = data;

            if (!createOverlay(box)) return;

            $("uc-floor-route-track")
                .setAttribute("d", pathString(points));

            updateMarker();

        } catch (error) {
            console.warn(
                "[Floor route overlay]",
                error.message
            );
        }
    }

    function initialize() {
        const selector = $("uc-inspection-floor");

        if (!selector) return;

        selector.addEventListener("change", () => {
            load(Number(selector.value));
        });

        // The existing player controls its own playback.
        // Observe its slider so we don't introduce a second
        // competing playback loop.
        let previousFloor = -1;

        function tick() {
            const floor = Number(selector.value);

            if (floor !== previousFloor) {
                previousFloor = floor;
                load(floor);
            }

            if (floor === activeFloor && floor !== 2) {
                updateMarker();
            }

            requestAnimationFrame(tick);
        }

        requestAnimationFrame(tick);
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


def enhance_floor_plan_route(html: str) -> str:
    if "uc-floor-route-svg" in html:
        return html

    html = html.replace(
        "</head>",
        CSS + "</head>",
        1,
    )

    html = html.replace(
        "</body>",
        SCRIPT + "</body>",
        1,
    )

    return html
