
from pathlib import Path
from datetime import datetime
import re

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "src" / "construction_dashboard.py"

if not DASHBOARD.exists():
    raise SystemExit(f"Dashboard not found: {DASHBOARD}")

source = DASHBOARD.read_text(encoding="utf-8")

START = "<!-- UNDER_CONSTRUCTION_HERO_V2_START -->"
END = "<!-- UNDER_CONSTRUCTION_HERO_V2_END -->"

# Make the patch repeatable without stacking copies.
if START in source and END in source:
    source = re.sub(
        re.escape(START) + r".*?" + re.escape(END),
        "",
        source,
        flags=re.S,
    )

upgrade = r"""
<!-- UNDER_CONSTRUCTION_HERO_V2_START -->
<style>
/* STEP 110: HERO VISUAL UPGRADE */

:root {
  --uc-purple: #b597ff;
  --uc-orange: #ff8752;
}

.brand-mark,
.logo-mark {
  display: inline-flex !important;
  align-items: center;
  justify-content: center;
  flex: 0 0 43px;
  width: 43px !important;
  height: 43px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: transparent !important;
  color: transparent !important;
  font-size: 0 !important;
  overflow: visible !important;
}

.uc-helmet-logo {
  width: 43px;
  height: 43px;
  display: block;
  flex-shrink: 0;
}

.hero-art {
  position: relative !important;
  isolation: isolate;
  overflow: visible !important;
  min-height: 610px;
}

.hero-art > :not(.uc-hero-scene) {
  display: none !important;
}

.uc-hero-scene {
  position: absolute;
  inset: -38px -4% -20px -12%;
  pointer-events: none;
  overflow: hidden;
  background:
    radial-gradient(ellipse at 56% 52%, #b597ff14, transparent 46%),
    radial-gradient(ellipse at 82% 58%, #ff87521b, transparent 42%);
}

.uc-hero-scene::after {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(90deg, #101117 0%, transparent 16%),
    linear-gradient(0deg, #101117 0%, transparent 16%);
  pointer-events: none;
}

.uc-building-svg {
  position: absolute;
  left: -3%;
  top: 1%;
  width: 109%;
  height: 99%;
  overflow: visible;
  filter: drop-shadow(0 25px 45px #0009);
}

.uc-grid {
  position: absolute;
  inset: 22% 0 0;
  opacity: .24;
  transform: perspective(650px) rotateX(65deg) scale(1.7);
  transform-origin: center 72%;
  background-image:
    linear-gradient(#aa92ff20 1px, transparent 1px),
    linear-gradient(90deg, #aa92ff20 1px, transparent 1px);
  background-size: 35px 35px;
  mask-image: radial-gradient(ellipse, black 10%, transparent 68%);
}

.uc-building {
  transform-origin: 51% 64%;
  animation: uc-float 8s ease-in-out infinite;
}

.uc-structure-level {
  opacity: 0;
  animation: uc-build-in .9s ease-out forwards;
}

.uc-structure-level:nth-of-type(1) { animation-delay: .15s; }
.uc-structure-level:nth-of-type(2) { animation-delay: .45s; }
.uc-structure-level:nth-of-type(3) { animation-delay: .75s; }
.uc-structure-level:nth-of-type(4) { animation-delay: 1.05s; }

.uc-ghost {
  animation: uc-ghost 6s ease-in-out infinite;
}

.uc-crane-hook {
  animation: uc-hook 5s ease-in-out infinite;
}

.uc-beam {
  animation: uc-beam 4s ease-in-out infinite;
}

.uc-hero-note {
  position: absolute;
  z-index: 2;
  font: 10px/1.6 ui-monospace, Consolas, monospace;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #b7a3e7;
}

.uc-note-top {
  right: 1%;
  top: 15%;
  color: var(--uc-orange);
}

.uc-note-bottom {
  right: 1%;
  bottom: 10%;
}

.uc-hero-card {
  position: absolute;
  z-index: 3;
  right: 1%;
  width: 24%;
  max-width: 170px;
  min-width: 115px;
  border: 1px solid #ff875270;
  border-radius: 9px;
  overflow: hidden;
  background: #1a1618e8;
  box-shadow: 0 15px 35px #0009;
}

.uc-video-card {
  top: 23%;
  aspect-ratio: 1.2;
}

.uc-video-card svg,
.uc-plan-card svg {
  width: 100%;
  height: 100%;
  display: block;
}

.uc-video-card::after {
  content: "▶";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: grid;
  place-items: center;
  width: 39px;
  height: 39px;
  border-radius: 50%;
  border: 1px solid #ff8752;
  background: #141218d9;
  color: #ff8752;
  font-size: 14px;
}

.uc-plan-card {
  top: 67%;
  aspect-ratio: 1.7;
  border-color: #b597ff70;
}

.uc-plan-label {
  top: 62%;
  right: 1%;
}

.uc-hero-disclaimer {
  position: absolute;
  left: 27%;
  bottom: 3%;
  z-index: 3;
  color: #a6a0b7;
  font: 9px ui-monospace, Consolas, monospace;
  letter-spacing: .13em;
  text-transform: uppercase;
}

@keyframes uc-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-9px); }
}

@keyframes uc-build-in {
  from { opacity: 0; transform: translateY(-28px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes uc-ghost {
  0%, 100% { opacity: .15; }
  50% { opacity: .65; }
}

@keyframes uc-hook {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(20px); }
}

@keyframes uc-beam {
  0%, 100% { opacity: .15; }
  50% { opacity: .8; }
}

@media (max-width: 900px) {
  .hero-art {
    min-height: 420px;
  }

  .uc-hero-scene {
    inset: 0 -3%;
  }

  .uc-building-svg {
    width: 100%;
    left: -7%;
  }
}

@media (max-width: 560px) {
  .hero-art {
    min-height: 330px;
  }

  .uc-hero-card,
  .uc-note-top,
  .uc-plan-label,
  .uc-note-bottom {
    display: none;
  }

  .uc-building-svg {
    width: 118%;
    left: -10%;
  }
}

@media (prefers-reduced-motion: reduce) {
  .uc-building,
  .uc-ghost,
  .uc-crane-hook,
  .uc-beam,
  .uc-structure-level {
    animation: none !important;
    opacity: 1 !important;
  }
}
</style>

<script>
(() => {
  const helmet = `
    <svg class="uc-helmet-logo" viewBox="0 0 64 64"
         fill="none" xmlns="http://www.w3.org/2000/svg"
         role="img" aria-label="Construction hard hat">
      <defs>
        <linearGradient id="ucHatGradient" x1="10" y1="8" x2="54" y2="54">
          <stop stop-color="#D0B8FF"/>
          <stop offset="1" stop-color="#8B65E6"/>
        </linearGradient>
      </defs>
      <path d="M12 39C12 25 21 15 32 15S52 25 52 39"
            fill="url(#ucHatGradient)"/>
      <path d="M28 16V39M36 16V39"
            stroke="#3A255B" stroke-width="2.8"/>
      <path d="M32 11V39"
            stroke="#E7D8FF" stroke-width="5"
            stroke-linecap="round"/>
      <path d="M8 39H56V45H8Z" fill="#B597FF"/>
      <path d="M6 45H58V50H6Z" fill="#7751CB"/>
      <path d="M15 50L21 55H43L49 50"
            stroke="#C7B0FF" stroke-width="2.5"
            stroke-linejoin="round"/>
      <path d="M14 35C15 28 19 23 24 20"
            stroke="#F2E8FF" stroke-width="2"
            stroke-linecap="round" opacity=".8"/>
    </svg>`;

  // Replace the temporary logo icon wherever it appears.
  document.querySelectorAll(".brand-mark, .logo-mark").forEach(el => {
    el.innerHTML = helmet;
  });

  const art = document.querySelector(".hero-art");
  if (!art) {
    console.warn("Hero art container not found; logo was updated.");
    return;
  }

  const NS = "http://www.w3.org/2000/svg";

  const svgElement = (tag, attrs = {}) => {
    const el = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => {
      el.setAttribute(key, String(value));
    });
    return el;
  };

  const add = (parent, tag, attrs = {}) => {
    const el = svgElement(tag, attrs);
    parent.appendChild(el);
    return el;
  };

  const line = (parent, x1, y1, x2, y2, stroke, width = 1, extra = {}) =>
    add(parent, "line", {
      x1, y1, x2, y2,
      stroke,
      "stroke-width": width,
      ...extra
    });

  const polygon = (parent, points, fill, stroke = "none", width = 1) =>
    add(parent, "polygon", {
      points: points.map(p => p.join(",")).join(" "),
      fill,
      stroke,
      "stroke-width": width,
      "stroke-linejoin": "round"
    });

  const group = (parent, cls = "") =>
    add(parent, "g", {class: cls});

  // Isometric architectural illustration:
  // Four concrete levels, columns, glazing, slabs and plan overlays.
  const scene = document.createElement("div");
  scene.className = "uc-hero-scene";
  scene.innerHTML = `
    <div class="uc-grid"></div>
    <div class="uc-hero-note uc-note-top">
      CONSTRUCTION<br>WALKTHROUGH<br>(VIDEO)
    </div>
    <div class="uc-hero-card uc-video-card">
      <svg viewBox="0 0 160 125" aria-hidden="true">
        <defs>
          <linearGradient id="ucRoom" x1="0" x2="1" y1="0" y2="1">
            <stop stop-color="#574139"/>
            <stop offset="1" stop-color="#16141A"/>
          </linearGradient>
        </defs>
        <rect width="160" height="125" fill="url(#ucRoom)"/>
        <polygon points="0,0 160,0 123,37 36,37"
                 fill="#8A7164"/>
        <polygon points="0,125 160,125 123,70 36,70"
                 fill="#40332F"/>
        <path d="M36 37V70M123 37V70M80 37V92"
              stroke="#C4A58E" stroke-width="9"/>
        <path d="M0 0L36 37M160 0L123 37"
              stroke="#D0A789" stroke-width="3"/>
        <rect x="128" y="34" width="24" height="43"
              fill="#C98B59"/>
        <path d="M140 34V77M128 55H152"
              stroke="#40322D" stroke-width="2"/>
        <path d="M0 107L65 77M160 108L97 78"
              stroke="#B18A6D" opacity=".5"/>
      </svg>
    </div>
    <div class="uc-hero-note uc-plan-label">
      FLOOR PLAN<br>(DXF)
    </div>
    <div class="uc-hero-card uc-plan-card">
      <svg viewBox="0 0 170 100" aria-hidden="true">
        <rect width="170" height="100" fill="#10121B"/>
        <g stroke="#B597FF" fill="none" stroke-width="1"
           opacity=".85">
          <rect x="15" y="12" width="140" height="75"/>
          <path d="M15 45H80V12M80 45H155M105 45V87
                   M15 68H105M50 45V68M130 45V87"/>
          <path d="M40 12V28M40 33V45M80 61H96
                   M115 12V27M115 32V45"/>
          <path d="M20 20H35V37H20Z
                   M85 18H109V38H85Z"/>
        </g>
        <g stroke="#B597FF" opacity=".22">
          <path d="M0 25H170M0 50H170M0 75H170
                   M25 0V100M50 0V100M75 0V100
                   M100 0V100M125 0V100M150 0V100"/>
        </g>
      </svg>
    </div>
    <div class="uc-hero-note uc-note-bottom">
      SAME SPACE.<br>DIFFERENT PERSPECTIVES.
    </div>
    <div class="uc-hero-disclaimer">
      Illustrative architectural study
    </div>
  `;

  const svg = svgElement("svg", {
    class: "uc-building-svg",
    viewBox: "0 0 850 650",
    preserveAspectRatio: "xMidYMid meet",
    "aria-hidden": "true"
  });

  const defs = add(svg, "defs");

  const concrete = add(defs, "linearGradient", {
    id: "ucConcrete",
    x1: "0%", y1: "0%",
    x2: "100%", y2: "100%"
  });
  add(concrete, "stop", {
    offset: "0%", "stop-color": "#C8B3A9"
  });
  add(concrete, "stop", {
    offset: "55%", "stop-color": "#857C80"
  });
  add(concrete, "stop", {
    offset: "100%", "stop-color": "#3D3B47"
  });

  const glow = add(defs, "linearGradient", {
    id: "ucGlass",
    x1: "0%", y1: "0%",
    x2: "100%", y2: "100%"
  });
  add(glow, "stop", {
    offset: "0%", "stop-color": "#A786FF",
    "stop-opacity": ".5"
  });
  add(glow, "stop", {
    offset: "100%", "stop-color": "#FF9458",
    "stop-opacity": ".18"
  });

  const building = group(svg, "uc-building");

  // Floor-plan ground plane.
  polygon(
    building,
    [[110,510],[430,390],[705,505],[384,626]],
    "#B597FF08",
    "#B597FF",
    1.4
  );

  for (let i = 0; i < 10; i++) {
    const t = i / 10;
    line(
      building,
      110 + 320*t, 510 - 120*t,
      384 + 321*t, 626 - 121*t,
      "#B597FF35"
    );
    line(
      building,
      110 + 274*t, 510 + 116*t,
      430 + 275*t, 390 + 115*t,
      "#B597FF35"
    );
  }

  // Crane tower and boom.
  const crane = group(building);
  line(crane, 225, 95, 225, 474, "#A783F3", 4);
  line(crane, 245, 95, 245, 474, "#A783F3", 4);

  for (let y = 105; y < 460; y += 23) {
    line(crane, 225, y, 245, y+23, "#B597FF95", 1);
    line(crane, 245, y, 225, y+23, "#B597FF95", 1);
    line(crane, 225, y, 245, y, "#B597FF75", 1);
  }

  polygon(
    crane,
    [[165,96],[470,55],[480,65],[165,106]],
    "#9B74EA20",
    "#B597FF",
    1.2
  );

  for (let x = 175; x < 470; x += 22) {
    const y = 95 - (x-165)*41/305;
    line(crane, x, y, x+11, y+7, "#B597FF85");
  }

  line(crane, 300, 77, 300, 154, "#B597FF90", 1, {
    "stroke-dasharray": "3 5"
  });

  const hook = group(crane, "uc-crane-hook");
  line(hook, 300, 154, 300, 180, "#D3B9FF", 1.3);
  add(hook, "path", {
    d: "M294 180 Q300 193 307 180",
    stroke: "#D3B9FF",
    fill: "none",
    "stroke-width": 2
  });

  // Building coordinates, in isometric projection.
  const A = [290, 400]; // front-left
  const B = [525, 314]; // back-left
  const C = [682, 383]; // back-right
  const D = [447, 469]; // front-right

  const point = (p, dy) => [p[0], p[1] + dy];
  const top = (dy) => [
    point(A, dy), point(B, dy),
    point(C, dy), point(D, dy)
  ];

  const slabThickness = 13;
  const levelHeight = 84;
  const baseShift = 0;

  // Warm window lighting and vertical structural members.
  for (let level = 0; level < 4; level++) {
    const dy = baseShift - level * levelHeight;
    const floor = group(building, "uc-structure-level");

    const corners = top(dy);

    // Slab top.
    polygon(
      floor,
      corners,
      "url(#ucConcrete)",
      "#D4B6A7",
      1.2
    );

    // Front slab fascia.
    polygon(
      floor,
      [
        corners[0], corners[3],
        point(D,dy+slabThickness),
        point(A,dy+slabThickness)
      ],
      "#665C61",
      "#D7B0A0",
      .7
    );

    // Right slab fascia.
    polygon(
      floor,
      [
        corners[3], corners[2],
        point(C,dy+slabThickness),
        point(D,dy+slabThickness)
      ],
      "#947467",
      "#FFAB78",
      .8
    );

    // Columns rising to the next floor.
    if (level < 3) {
      const columnPoints = [
        A, B, C, D,
        [368,371], [445,342],
        [523,400], [600,354]
      ];

      columnPoints.forEach((p, idx) => {
        const x = p[0], y = p[1]+dy;
        const warm = idx >= 3;
        polygon(
          floor,
          [
            [x-4,y],
            [x+4,y-1],
            [x+4,y-levelHeight+5],
            [x-4,y-levelHeight+6]
          ],
          warm ? "#BA9180" : "#9B9297",
          warm ? "#E5A179" : "#C7B4BE",
          .55
        );
      });

      // Interior glass and walls.
      polygon(
        floor,
        [
          [370,371+dy],
          [445,343+dy],
          [445,343+dy-levelHeight+12],
          [370,371+dy-levelHeight+12]
        ],
        "url(#ucGlass)",
        "#B597FF80",
        .7
      );

      polygon(
        floor,
        [
          [523,400+dy],
          [600,354+dy],
          [600,354+dy-levelHeight+12],
          [523,400+dy-levelHeight+12]
        ],
        "#FF87521A",
        "#FFB18A80",
        .8
      );

      // Repeating facade mullions.
      for (let t = .2; t < 1; t += .2) {
        const x = A[0] + (D[0]-A[0])*t;
        const y = A[1] + (D[1]-A[1])*t + dy;

        line(
          floor,
          x, y,
          x, y-levelHeight+13,
          "#E0C5C1A0",
          1.4
        );
      }

      for (let t = .25; t < 1; t += .25) {
        const x = D[0] + (C[0]-D[0])*t;
        const y = D[1] + (C[1]-D[1])*t + dy;

        line(
          floor,
          x, y,
          x, y-levelHeight+13,
          "#F6A87C",
          1.2
        );
      }

      // Horizontal rails.
      line(
        floor,
        A[0], A[1]+dy-40,
        D[0], D[1]+dy-40,
        "#BBA6B3",
        1
      );

      line(
        floor,
        D[0], D[1]+dy-40,
        C[0], C[1]+dy-40,
        "#FFB183",
        1
      );
    }
  }

  // Semi-transparent purple section: the 'drawing' side.
  const ghost = group(building, "uc-ghost");

  polygon(
    ghost,
    [[265,185],[400,137],[400,475],[265,525]],
    "#A07AFF18",
    "#B597FF",
    1.4
  );

  for (let y = 215; y <= 475; y += 85) {
    line(
      ghost,
      265,y,
      400,y-48,
      "#C2A8FF",
      1.1
    );
  }

  for (let x = 288; x <= 390; x += 34) {
    line(
      ghost,
      x,177-(x-265)*48/135,
      x,517-(x-265)*48/135,
      "#B597FF70",
      .9
    );
  }

  // Animated section scanning line.
  const scan = group(building, "uc-beam");
  line(
    scan,
    270, 300,
    650, 300,
    "#B597FF",
    1.5,
    {"stroke-dasharray": "8 7"}
  );

  // Level markers, with no claim about real progress.
  const levels = [
    ["ROOF", 157],
    ["L3", 242],
    ["L2", 327],
    ["L1", 412],
    ["GROUND", 497]
  ];

  levels.forEach(([name, y], idx) => {
    add(svg, "text", {
      x: 94, y,
      fill: "#CBBCE6",
      "font-size": 12,
      "font-family": "Consolas, monospace",
      "letter-spacing": 1
    }).textContent = name;

    add(svg, "circle", {
      cx: 170,
      cy: y-4,
      r: idx === 2 ? 7 : 3.2,
      fill: idx === 2 ? "#B597FF" : "#101117",
      stroke: "#B597FF",
      "stroke-width": 1.4
    });

    line(
      svg,
      178,y-4,
      248,y-4,
      "#B597FF45",
      1,
      {"stroke-dasharray": "3 6"}
    );
  });

  // Technical guide lines.
  for (const x of [290,447,525,682]) {
    line(
      svg,
      x, 85,
      x, 560,
      "#B597FF30",
      .7,
      {"stroke-dasharray": "3 8"}
    );
  }

  scene.appendChild(svg);
  art.appendChild(scene);

  // A subtle pointer response adds depth without requiring WebGL.
  const reducedMotion = window.matchMedia(
    "(prefers-reduced-motion: reduce)"
  ).matches;

  if (!reducedMotion) {
    art.addEventListener("pointermove", e => {
      const rect = art.getBoundingClientRect();
      const x = (e.clientX - rect.left) / rect.width - .5;
      const y = (e.clientY - rect.top) / rect.height - .5;

      svg.style.transform =
        `translate(${x*7}px, ${y*5}px)`;
    });

    art.addEventListener("pointerleave", () => {
      svg.style.transform = "";
    });
  }
})();
</script>
<!-- UNDER_CONSTRUCTION_HERO_V2_END -->
"""

# Inject before </body>, preserving existing dashboard HTML and JS.
closing_body = re.search(r"</body\s*>", source, flags=re.I)

if not closing_body:
    raise SystemExit(
        "Could not locate </body> in construction_dashboard.py. "
        "No changes made."
    )

backup = DASHBOARD.with_name(
    f"construction_dashboard.backup_"
    f"{datetime.now():%Y%m%d_%H%M%S}.py"
)
backup.write_text(
    DASHBOARD.read_text(encoding="utf-8"),
    encoding="utf-8"
)

source = (
    source[:closing_body.start()]
    + upgrade
    + "\n"
    + source[closing_body.start():]
)

DASHBOARD.write_text(source, encoding="utf-8")

print("Hero visual upgrade installed.")
print("Construction helmet logo installed.")
print("Existing dashboard logic preserved.")
print(f"Backup: {backup.name}")
print("Restart src/construction_dashboard.py to view.")
