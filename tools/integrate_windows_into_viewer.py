
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "digital_twin_3d.py"
BACKUP = ROOT / "src" / "digital_twin_3d.before_windows.py"

def replace_once(source, old, new, label):
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected one insertion point, found {count}. "
            "The viewer has not been modified."
        )
    return source.replace(old, new, 1)

def main():
    if not TARGET.exists():
        raise FileNotFoundError(TARGET)

    source = TARGET.read_text(encoding="utf-8-sig")

    if "function consolidateWindows(" in source:
        print("Window integration already present. No changes made.")
        return

    source = replace_once(
        source,
        '    <div class="muted" id="doorCount"></div>',
        '''    <div class="muted" id="doorCount"></div>

    <label class="toggle">
        <input type="checkbox" id="showWindows" checked>
        Show provisional window candidates
    </label>
    <div class="muted" id="windowCount"></div>''',
        "Window visibility control"
    )

    source = replace_once(
        source,
        'placeholder="Search walls, doors, CAD handles"',
        'placeholder="Search walls, doors, windows, CAD handles"',
        "Search placeholder"
    )

    source = replace_once(
        source,
        '''        <div>
            <span class="dot" style="background:#4aa8ff"></span>
            Selected
        </div>''',
        '''        <div>
            <span class="dot" style="background:#22d3ee"></span>
            Window — provisional candidate
        </div>
        <div>
            <span class="dot" style="background:#4aa8ff"></span>
            Selected
        </div>''',
        "Window legend"
    )

    source = replace_once(
        source,
        'Select a wall or door',
        'Select a wall, door, or window',
        "Selection label"
    )

    source = replace_once(
        source,
        '''        Door openings, CAD/video alignment, and
        construction completion remain unverified.''',
        '''        Door and window openings, CAD/video alignment,
        and construction completion remain unverified.
        Window heights and sill levels are display assumptions.''',
        "Uncertainty notice"
    )

    source = replace_once(
        source,
        '''const doorGroup = new THREE.Group();
scene.add(buildingGroup,doorGroup);''',
        '''const doorGroup = new THREE.Group();
const windowGroup = new THREE.Group();
scene.add(buildingGroup,doorGroup,windowGroup);''',
        "Three.js window group"
    )

    source = replace_once(
        source,
        '''let doors = [];
let wallMeshes = [];
let doorMeshes = [];''',
        '''let doors = [];
let windows = [];
let wallMeshes = [];
let doorMeshes = [];
let windowMeshes = [];''',
        "Window state"
    )

    window_functions = r'''
// Window spans originate from CAD plan geometry.
// Vertical dimensions are ONLY visualization assumptions.
const windowMaterial = new THREE.MeshStandardMaterial({
    color:0x22d3ee,
    transparent:true,
    opacity:0.52,
    roughness:0.25,
    metalness:0.1,
    side:THREE.DoubleSide,
    depthWrite:false
});

function spanDistance(a,b) {
    return Math.hypot(a[0]-b[0],a[1]-b[1]);
}

function validWindow(w) {
    return w &&
        Array.isArray(w.start) &&
        Array.isArray(w.end) &&
        w.start.length===2 &&
        w.end.length===2 &&
        [...w.start,...w.end].every(Number.isFinite) &&
        spanDistance(w.start,w.end)>0.05;
}

function windowDirection(w) {
    const dx=w.end[0]-w.start[0];
    const dy=w.end[1]-w.start[1];
    const len=Math.hypot(dx,dy);
    return [dx/len,dy/len];
}

function windowProjection(w,axis) {
    const a=w.start[0]*axis[0]+w.start[1]*axis[1];
    const b=w.end[0]*axis[0]+w.end[1]*axis[1];
    return [Math.min(a,b),Math.max(a,b)];
}

function windowPerpendicular(w,axis) {
    const center=[
        (w.start[0]+w.end[0])/2,
        (w.start[1]+w.end[1])/2
    ];
    return -axis[1]*center[0]+axis[0]*center[1];
}

function overlapFraction(a,b) {
    const overlap=Math.max(
        0,Math.min(a[1],b[1])-Math.max(a[0],b[0])
    );
    const smaller=Math.min(a[1]-a[0],b[1]-b[0]);
    return smaller>0 ? overlap/smaller : 0;
}

function sameWindowLocation(a,b) {
    const axis=windowDirection(a);
    const other=windowDirection(b);
    const parallel=Math.abs(
        axis[0]*other[0]+axis[1]*other[1]
    );

    if (parallel<Math.cos(5*Math.PI/180)) return false;

    const offset=Math.abs(
        windowPerpendicular(a,axis)-
        windowPerpendicular(b,axis)
    );

    // Conservative 8 cm offset allowance for nearby CAD
    // frame components, not a wall association threshold.
    if (offset>0.08) return false;

    return overlapFraction(
        windowProjection(a,axis),
        windowProjection(b,axis)
    )>=0.80;
}

function consolidateWindows(input) {
    const candidates=input.filter(validWindow);
    const parent=candidates.map((_,i)=>i);

    function find(i) {
        while(parent[i]!==i) {
            parent[i]=parent[parent[i]];
            i=parent[i];
        }
        return i;
    }

    function union(a,b) {
        a=find(a);
        b=find(b);
        if(a!==b) parent[b]=a;
    }

    // Group only components sharing essentially the same
    // plan location. Do not join adjacent glazing modules.
    for(let i=0;i<candidates.length;i++) {
        for(let j=i+1;j<candidates.length;j++) {
            if(sameWindowLocation(candidates[i],candidates[j])) {
                union(i,j);
            }
        }
    }

    const groups=new Map();
    candidates.forEach((w,i)=>{
        const root=find(i);
        if(!groups.has(root)) groups.set(root,[]);
        groups.get(root).push(w);
    });

    const output=[];

    for(const group of groups.values()) {
        // Prefer the longest source span as the representative.
        // This avoids averaging full-frame and inner-pane spans.
        const representative=[...group].sort(
            (a,b)=>spanDistance(b.start,b.end)-
                   spanDistance(a.start,a.end)
        )[0];

        const handles=[...new Set(
            group.flatMap(w=>w.source_handles||[])
        )];

        output.push({
            ...representative,
            id:group.map(w=>w.id).sort()[0],
            source_group_ids:group.map(w=>w.id),
            source_handles:handles,
            component_count:group.length,
            status:"PROVISIONAL_WINDOW_CANDIDATE",
            opening_verified:false,
            wall_association_verified:false,
            vertical_dimensions_verified:false,
            construction_status:"UNKNOWN",
            safe_to_cut:false
        });
    }

    return output;
}

function createWindowMesh(item,center) {
    const [x1,y1]=item.start;
    const [x2,y2]=item.end;
    const dx=x2-x1;
    const dy=y2-y1;
    const length=Math.hypot(dx,dy);

    if(length<=0.05) return null;

    // Visualization only: NOT recovered from the DXF.
    const assumedSill=0.90;
    const assumedGlazingHeight=1.20;
    const assumedThickness=0.035;

    const geometry=new THREE.BoxGeometry(
        length,assumedGlazingHeight,assumedThickness
    );

    const mesh=new THREE.Mesh(geometry,windowMaterial);

    mesh.position.set(
        (x1+x2)/2-center.x,
        assumedSill+assumedGlazingHeight/2,
        (y1+y2)/2-center.z
    );

    mesh.rotation.y=-Math.atan2(dy,dx);
    mesh.userData={kind:"window",item};
    return mesh;
}

function buildWindows(items) {
    for(const item of items) {
        const mesh=createWindowMesh(item,modelCenter);
        if(!mesh) continue;
        windowGroup.add(mesh);
        windowMeshes.push(mesh);
    }
    windowGroup.visible=$("showWindows").checked;
}
'''

    source = replace_once(
        source,
        'function buildWalls(items) {',
        window_functions + '\nfunction buildWalls(items) {',
        "Window consolidation and rendering"
    )

    source = replace_once(
        source,
        '''        ...($("showDoors").checked
            ? doors.map(item=>({kind:"door",item}))
            : [])''',
        '''        ...($("showDoors").checked
            ? doors.map(item=>({kind:"door",item}))
            : []),
        ...($("showWindows").checked
            ? windows.map(item=>({kind:"window",item}))
            : [])''',
        "Element listing"
    )

    source = replace_once(
        source,
        '''            ...(item.source_arc_handles || [])''',
        '''            ...(item.source_arc_handles || []),
            ...(item.source_handles || []),
            ...(item.source_group_ids || [])''',
        "Window search"
    )

    source = replace_once(
        source,
        '''        details.textContent=kind==="door"
            ? `Door-leaf candidate · ${item.status}`
            : `${item.category || "wall"} · CAD wall`;''',
        '''        details.textContent=kind==="door"
            ? `Door-leaf candidate · ${item.status}`
            : kind==="window"
                ? `Window candidate · ${item.component_count} CAD groups`
                : `${item.category || "wall"} · CAD wall`;''',
        "Window list details"
    )

    source = replace_once(
        source,
        '''    if (mesh.userData.kind==="door") {''',
        '''    if (mesh.userData.kind==="window") {
        mesh.material=windowMaterial;
    } else if (mesh.userData.kind==="door") {''',
        "Window material restoration"
    )

    source = replace_once(
        source,
        '''    const candidates=kind==="door" ? doorMeshes : wallMeshes;''',
        '''    const candidates=kind==="door" ? doorMeshes :
        kind==="window" ? windowMeshes : wallMeshes;''',
        "Window selection lookup"
    )

    source = replace_once(
        source,
        '''    if (kind==="door") {
        $("category").textContent="Provisional CAD door leaf";''',
        '''    if (kind==="window") {
        ++requestGeneration;
        $("category").textContent="Provisional CAD window candidate";
        $("dimensions").textContent=
            `${Number(item.span_m).toFixed(2)} m plan span · `+
            "0.90 m sill and 1.20 m glazing height (assumed)";
        $("sourceCount").textContent=
            `${item.component_count} grouped CAD components · `+
            `${item.source_handles.length} DXF handles: `+
            item.source_handles.join(", ");
        $("geometryStatus").textContent=
            "Plan geometry only; wall opening unverified";
        $("evidence").textContent=
            "Window-specific visual evidence has not been verified.";
    } else if (kind==="door") {
        ++requestGeneration;
        $("category").textContent="Provisional CAD door leaf";''',
        "Window details panel"
    )

    source = replace_once(
        source,
        '''    const candidates=[
        ...(doorGroup.visible ? doorMeshes : []),
        ...wallMeshes
    ];''',
        '''    const candidates=[
        ...(windowGroup.visible ? windowMeshes : []),
        ...(doorGroup.visible ? doorMeshes : []),
        ...wallMeshes
    ];''',
        "Window raycasting"
    )

    source = replace_once(
        source,
        '''$("reset").addEventListener("click",()=>fitCamera(false));''',
        '''$("showWindows").addEventListener("change",event=>{
    windowGroup.visible=event.target.checked;
    if(!event.target.checked &&
       selectedMesh?.userData.kind==="window") {
        restoreMaterial(selectedMesh);
        selectedMesh=null;
    }
    renderList();
});

$("reset").addEventListener("click",()=>fitCamera(false));''',
        "Window toggle handler"
    )

    source = replace_once(
        source,
        '''    renderList();

    $("renderStatus").textContent=
        `${wallMeshes.length} wall meshes · `+
        `${doorMeshes.length} provisional door meshes`;''',
        '''    try {
        const response=await fetch("/api/twin/windows");
        if(!response.ok) {
            throw new Error(`Window API returned ${response.status}`);
        }

        const data=await response.json();
        const raw=Array.isArray(data.windows) ? data.windows : [];
        windows=consolidateWindows(raw);
        buildWindows(windows);

        $("windowCount").textContent=
            `${windowMeshes.length} consolidated window candidates · `+
            `${raw.length} source groups · 0 verified openings`;
    } catch(error) {
        console.warn("Window geometry unavailable:",error);
        $("windowCount").textContent=
            "Window geometry unavailable; walls and doors remain visible";
    }

    renderList();

    $("renderStatus").textContent=
        `${wallMeshes.length} wall meshes · `+
        `${doorMeshes.length} provisional door meshes · `+
        `${windowMeshes.length} provisional window meshes`;''',
        "Window API initialization"
    )

    # Compile before modifying the working source file.
    compile(source, str(TARGET), "exec")

    if not BACKUP.exists():
        shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(source, encoding="utf-8")

    print("\n=== WINDOW VIEWER INTEGRATION ===")
    print("Updated:", TARGET)
    print("Backup:", BACKUP)
    print("Window endpoint: /api/twin/windows")
    print("Consolidation: ENABLED")
    print("Window visibility toggle: ENABLED")
    print("Window selection: ENABLED")
    print("Verified wall openings: 0")
    print("Wall cutouts generated: NO")


if __name__ == "__main__":
    main()
