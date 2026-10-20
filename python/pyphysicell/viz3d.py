"""Interactive 3-D view of a population, in the notebook.

Cells are drawn as instanced spheres and fibres as segments, in a WebGL canvas
you can orbit, zoom and slice -- the same idea as ipyspeck for molecules.  The
scene is a self-contained HTML document: it carries its own data, so it can be
displayed inline, written to a file, or sent to someone who has no Python.
"""

from __future__ import annotations

import base64
import json

import numpy as np

THREE = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
ORBIT = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"

_PALETTE = np.array([
    [0.25, 0.47, 0.85], [0.90, 0.38, 0.30], [0.30, 0.69, 0.45], [0.80, 0.62, 0.20],
    [0.60, 0.40, 0.75], [0.35, 0.72, 0.78], [0.85, 0.52, 0.70], [0.50, 0.50, 0.52],
])


def _b64(a: np.ndarray, dtype="float32") -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def _colors(sim, color_by: str, idx: np.ndarray) -> np.ndarray:
    c = sim.cells
    if color_by == "cell_type":
        return _PALETTE[c.type_idx[idx] % len(_PALETTE)]
    values = sim.dictionary.signal_values(color_by, c, idx, sim.t).astype(float)
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    t = np.zeros_like(values) if hi - lo < 1e-15 else (values - lo) / (hi - lo)
    # viridis-ish: dark blue -> teal -> yellow
    stops = np.array([[0.27, 0.00, 0.33], [0.13, 0.57, 0.55], [0.99, 0.91, 0.14]])
    out = np.empty((t.size, 3))
    for k in range(3):
        out[:, k] = np.interp(t, [0.0, 0.5, 1.0], stops[:, k])
    return out


def scene(sim, color_by: str = "cell_type", include_fibres: bool = True,
          max_cells: int = 400_000, title: str | None = None,
          theme: str = "light", still: bool = False) -> str:
    """Return a self-contained HTML document showing the current state."""
    c = sim.cells
    n = c.n
    idx = np.arange(n)
    if n > max_cells:                      # keep the browser responsive
        idx = sim.rng.choice(n, max_cells, replace=False)
    pos = c.position[idx]
    rad = c.radius[idx]
    col = _colors(sim, color_by, idx)
    dead = c.dead[idx].astype(bool)
    col[dead] = [0.35, 0.35, 0.37]

    d = sim.model.domain
    data = {
        "n": int(idx.size),
        "pos": _b64(pos.ravel()),
        "rad": _b64(rad),
        "col": _b64(col.ravel()),
        "bounds": [d.x_min, d.x_max, d.y_min, d.y_max, d.z_min, d.z_max],
        "is2d": bool(d.use_2D),
        "t": float(sim.t),
        "title": title or f"{sim.model.name} — t = {sim.t:.0f} min — {n} cells",
        "colorBy": color_by,
        "types": sim.model.cell_type_names,
        "typeIdx": _b64(c.type_idx[idx], "int32"),
        "fibres": None,
        "theme": theme,
        "still": bool(still),       # draw a few frames and stop, for screenshots
    }
    f = getattr(sim, "fibres", None)
    if include_fibres and f is not None and f.n:
        ends = f.endpoints
        data["fibres"] = {"n": int(f.n), "a": _b64(ends[:, 0, :].ravel()),
                          "b": _b64(ends[:, 1, :].ravel()),
                          "rad": _b64(f.view("radius"))}
    return _HTML.replace("__DATA__", json.dumps(data)).replace("__THREE__", THREE) \
                .replace("__ORBIT__", ORBIT)


def show(sim, height: int = 600, backend: str = "mpl", **kwargs):
    """Show the population in 3-D.

    `backend="mpl"` (the default) is matplotlib: plain, dependency-free and it
    renders anywhere, including a notebook executed headlessly.
    `backend="speck"` is ipyspeck, the best looking and needs a live kernel;
    `backend="html"` is the self-contained WebGL viewer, which can be saved or
    shared as one file.
    """
    if backend == "mpl":
        kwargs.pop("theme", None)
        return mpl(sim, **kwargs)
    if backend in ("auto", "speck"):
        try:
            return speck(sim, height=f"{height}px", **kwargs)
        except ImportError:
            if backend == "speck":
                raise
    from IPython.display import HTML, display

    html = scene(sim, **kwargs).replace('"', "&quot;")
    return display(HTML(f'<iframe srcdoc="{html}" width="100%" height="{height}" '
                        'style="border:1px solid #ddd;border-radius:8px"></iframe>'))


def save(sim, path, **kwargs):
    """Write the scene as a standalone .html file."""
    from pathlib import Path

    p = Path(path)
    p.write_text(scene(sim, **kwargs))
    return p




def mpl(sim, color_by: str = "cell_type", ax=None, max_cells: int = 20_000,
        include_fibres: bool = True, elev: float = 22.0, azim: float = -58.0,
        figsize=(7.5, 6.5), title: str | None = None):
    """A matplotlib 3-D view: no WebGL, no widget, works anywhere.

    Slower and plainer than `speck()` or the WebGL `scene()`, but it renders in
    any backend, survives nbconvert, and goes straight into a paper figure.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    c = sim.cells
    n = c.n
    idx = np.arange(n)
    if n > max_cells:
        idx = sim.rng.choice(n, max_cells, replace=False)
    pos = c.position[idx]
    colours = _colors(sim, color_by, idx)
    colours[c.dead[idx] == 1] = [0.35, 0.35, 0.37]

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    d = sim.model.domain
    ax.set_xlim(d.x_min, d.x_max)
    ax.set_ylim(d.y_min, d.y_max)
    ax.set_zlim(d.z_min, d.z_max)
    try:
        ax.set_box_aspect((d.x_max - d.x_min, d.y_max - d.y_min,
                           max(d.z_max - d.z_min, 1e-9)))
    except Exception:                                    # older matplotlib
        pass
    ax.view_init(elev=elev, azim=azim)

    # marker area in points^2, from the cell radius in data units
    fig = ax.get_figure()
    span = max(d.x_max - d.x_min, d.y_max - d.y_min, 1e-9)
    pts_per_unit = (fig.get_size_inches()[0] * 72.0 * 0.62) / span
    sizes = (2.0 * c.radius[idx] * pts_per_unit) ** 2
    order = np.argsort(-pos[:, 2])                       # draw far cells first
    ax.scatter(pos[order, 0], pos[order, 1], pos[order, 2],
               s=sizes[order], c=colours[order], depthshade=True,
               edgecolors="none", alpha=0.95)

    f = getattr(sim, "fibres", None)
    if include_fibres and f is not None and f.n:
        ax.add_collection3d(Line3DCollection(list(f.endpoints), colors="0.35",
                                             linewidths=0.8, alpha=0.6))

    if color_by == "cell_type":
        from matplotlib.lines import Line2D

        seen = sorted(set(c.type_idx[idx]))
        ax.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=7,
                                  markerfacecolor=_PALETTE[t % len(_PALETTE)],
                                  markeredgecolor="none",
                                  label=sim.model.cell_type_names[t]) for t in seen],
                  loc="upper left", frameon=False, fontsize=8)
    ax.set_xlabel("x [um]", fontsize=8)
    ax.set_ylabel("y [um]", fontsize=8)
    ax.set_zlabel("z [um]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title or f"{sim.model.name} — t = {sim.t:.0f} min — {n} cells", fontsize=10)
    return ax


def snapshot(sim, path, width: int = 1000, height: int = 760, browser: str | None = None,
             timeout: int = 120, **kwargs):
    """Render the 3-D view to a PNG with headless Chrome.

    Unlike a widget, this works in a script and survives nbconvert, so an
    executed notebook keeps its 3-D figure.  Needs Chrome, Chromium or Edge.
    """
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    candidates = [browser] if browser else [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        shutil.which("chromium"), shutil.which("chromium-browser"),
        shutil.which("google-chrome"), shutil.which("microsoft-edge"),
    ]
    exe = next((c for c in candidates if c and Path(c).exists()), None)
    if exe is None:
        raise RuntimeError("no Chrome/Chromium/Edge found for headless rendering; "
                           "pass browser=... or use viz3d.save() and open it yourself")

    out = Path(path).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "scene.html"
        page.write_text(scene(sim, still=True, **kwargs))
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--enable-unsafe-swiftshader",
                        "--hide-scrollbars", f"--window-size={width},{height}",
                        "--virtual-time-budget=8000", f"--screenshot={out}", page.as_uri()],
                       capture_output=True, timeout=timeout, cwd=tmp)
    if not out.exists():
        raise RuntimeError(f"{exe} produced no image; try a longer timeout")
    return out


def speck_snapshot(sim, path, **kwargs):
    """Save a Speck-rendered PNG (ambient occlusion, outlines) of the population.

    Speck renders in the browser, so this needs a live notebook frontend: build
    the widget, display it, then call this.  For a script or an nbconvert run,
    use `snapshot()`, which drives headless Chrome instead.
    """
    view = kwargs.pop("view", None) or speck(sim, **kwargs)
    view.save_image(str(path))
    return view


# ---------------------------------------------------------------------------
# ipyspeck interoperability
#
# Speck renders XYZ: one element symbol and a position per atom.  A cell type
# maps to an element (which gives Speck a radius and a colour slot), and the
# colours are overridden per element so the view matches the rest of the
# library.  What this cannot carry is a per-cell radius or a continuous colour
# scale -- Speck sizes by element -- so `scene()` stays the richer view.
# ---------------------------------------------------------------------------

# elements picked for distinct default Speck colours, in cell-type order
SPECK_ELEMENTS = ["C", "N", "O", "S", "P", "Fe", "Cl", "Na", "Mg", "Zn", "Si", "B"]


def to_xyz(sim, scale: float | None = None, comment: str | None = None,
           elements: list[str] | None = None) -> tuple[str, dict[str, str]]:
    """Export the population as XYZ for Speck / ipyspeck / any molecular viewer.

    Returns (xyz text, {element: cell type}).  Positions are scaled so that one
    cell diameter is about one Speck "bond length", which is what makes the
    default atom sizing look right.
    """
    c = sim.cells
    n = c.n
    syms = elements or SPECK_ELEMENTS
    names = sim.model.cell_type_names
    mapping = {syms[i % len(syms)]: name for i, name in enumerate(names)}
    inverse = {name: syms[i % len(syms)] for i, name in enumerate(names)}

    radius = float(np.median(c.view("radius"))) if n else 8.4
    k = scale if scale is not None else 1.0 / max(radius, 1e-9)
    pos = c.view("position") * k
    types = c.view("type_idx")
    head = comment or f"{sim.model.name} t={sim.t:.0f} min ({n} cells, 1 unit = {radius:.1f} um)"
    lines = [str(n), head]
    for i in range(n):
        el = inverse[names[types[i]]]
        lines.append(f"{el} {pos[i, 0]:.3f} {pos[i, 1]:.3f} {pos[i, 2]:.3f}")
    return "\n".join(lines) + "\n", mapping


def speck(sim, color_by: str = "cell_type", width: str = "100%", height: str = "560px",
          bands: int = 8, atom_scale: float = 1.4, outline: float = 0.0, **traits):
    """An ipyspeck widget showing the population (`pip install ipyspeck`).

    Speck colours and sizes by element, so a cell type becomes an element and
    the element colours are overridden with this library's palette.  A
    continuous signal (`color_by="pressure"`) is binned into `bands` elements
    shaded along the same scale `scene()` uses, which gives Speck a colour
    ramp it could not otherwise show.  Bonds are off: cells are not bonded.
    """
    import ipyspeck

    c = sim.cells
    n = c.n
    idx = np.arange(n)
    radius = float(np.median(c.radius[idx])) if n else 8.4
    pos = c.position[idx] / max(radius, 1e-9)

    if color_by == "cell_type":
        labels = c.type_idx[idx]
        names = sim.model.cell_type_names
        colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(names))]
        titles = list(names)
    else:
        values = sim.dictionary.signal_values(color_by, c, idx, sim.t).astype(float)
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
        t = np.zeros_like(values) if hi - lo < 1e-15 else (values - lo) / (hi - lo)
        labels = np.clip((t * bands).astype(int), 0, bands - 1)
        stops = np.array([[0.27, 0.00, 0.33], [0.13, 0.57, 0.55], [0.99, 0.91, 0.14]])
        colours, titles = [], []
        for b in range(bands):
            f = b / max(bands - 1, 1)
            colours.append([np.interp(f, [0, 0.5, 1], stops[:, k]) for k in range(3)])
            titles.append(f"{color_by} {lo + f * (hi - lo):.3g}")

    syms = (SPECK_ELEMENTS * (1 + len(colours) // len(SPECK_ELEMENTS)))[: len(colours)]
    dead = c.dead[idx] == 1
    lines = [str(n), f"{sim.model.name} t={sim.t:.0f} min "
                     f"({n} cells, 1 unit = {radius:.1f} um, coloured by {color_by})"]
    for i in range(n):
        el = syms[int(labels[i])]
        lines.append(f"{el} {pos[i, 0]:.3f} {pos[i, 1]:.3f} {pos[i, 2]:.3f}")
    view = ipyspeck.Speck(data="\n".join(lines) + "\n")

    colour_map = {syms[i]: "#%02x%02x%02x" % tuple(int(255 * v) for v in rgb)
                  for i, rgb in enumerate(colours)}
    # cells are spheres of one size here: no bonds, no cartoon, no outline, and
    # the radius is uniform (relativeAtomScale=0) because Speck would otherwise
    # size them by element
    defaults = dict(bonds=False, cartoon=False, atomScale=atom_scale,
                    relativeAtomScale=0.0, ao=0.9, outline=outline, brightness=0.6,
                    atomShade=0.5, atomColor="element", atomColors=colour_map)
    defaults.update(traits)
    for key, value in defaults.items():
        setattr(view, key, value)
    view.layout.width = width
    view.layout.height = height
    view.physicell_legend = dict(zip(syms, titles))    # what each element stands for
    if dead.any():
        view.physicell_legend["(dead cells keep their type colour)"] = ""
    return view


def save_xyz(sim, path, **kwargs):
    """Write the population as an .xyz file (Speck, VMD, Ovito, ASE all read it)."""
    from pathlib import Path

    text, _mapping = to_xyz(sim, **kwargs)
    p = Path(path)
    p.write_text(text)
    return p


_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body { margin:0; height:100%; overflow:hidden;
              font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; }
  #app { position:fixed; inset:0; }
  .light { background:#eef1f6; color:#1b2733; }
  .dark  { background:#0b1020; color:#e8eef7; }
  #hud { position:absolute; top:12px; left:14px; pointer-events:none; font-weight:600; }
  #panel { position:absolute; top:12px; right:14px; padding:10px 12px; border-radius:10px;
           min-width:170px; backdrop-filter:blur(6px); }
  .light #panel { background:#ffffffcc; border:1px solid #d3dae6;
                  box-shadow:0 2px 10px #0b102014; }
  .dark  #panel { background:#131a2dcc; border:1px solid #2a3350; }
  #panel label { display:block; margin:8px 0 2px; font-size:10.5px; letter-spacing:.06em;
                 text-transform:uppercase; opacity:.65; }
  #panel input[type=range] { width:100%; }
  #legend { position:absolute; bottom:14px; left:14px; display:flex; gap:8px; flex-wrap:wrap; }
  .key { display:flex; align-items:center; gap:6px; padding:3px 9px; border-radius:999px; }
  .light .key { background:#ffffffcc; border:1px solid #d3dae6; }
  .dark  .key { background:#131a2dcc; border:1px solid #2a3350; }
  .dot { width:10px; height:10px; border-radius:50%; }
  #hint { position:absolute; bottom:14px; right:14px; opacity:.5; }
</style></head>
<body class="light">
<div id="app"></div>
<div id="hud" class="light"><span id="title"></span></div>
<div id="panel">
  <label>slice (z)</label><input id="clip" type="range" min="0" max="100" value="100">
  <label>cell size</label><input id="size" type="range" min="20" max="160" value="100">
  <label>brightness</label><input id="bright" type="range" min="40" max="180" value="100">
  <label style="display:flex;gap:6px;align-items:center;text-transform:none;font-size:12px">
    <input id="spin" type="checkbox"> rotate</label>
  <label style="display:flex;gap:6px;align-items:center;text-transform:none;font-size:12px">
    <input id="dark" type="checkbox"> dark</label>
</div>
<div id="legend"></div>
<div id="hint">drag orbit · scroll zoom · right-drag pan</div>
<script src="__THREE__"></script>
<script src="__ORBIT__"></script>
<script>
const D = __DATA__;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const pos = dec(D.pos, Float32Array), rad = dec(D.rad, Float32Array),
      col = dec(D.col, Float32Array), typeIdx = dec(D.typeIdx, Int32Array);
document.getElementById('title').textContent = D.title;

const [x0, x1, y0, y1, z0, z1] = D.bounds;
const cx = (x0+x1)/2, cy = (y0+y1)/2, cz = (z0+z1)/2;
const span = Math.max(x1-x0, y1-y0, z1-z0);

function fail(msg) {
  const el = document.createElement('pre');
  el.style.cssText = 'position:absolute;inset:16px;white-space:pre-wrap;overflow:auto;'
                   + 'background:#fff3f3;color:#8a1f1f;border:1px solid #e0b4b4;'
                   + 'padding:12px;border-radius:8px;font:12px/1.4 ui-monospace,monospace';
  el.textContent = 'pyphysicell 3-D view could not start:\n\n' + msg;
  document.body.appendChild(el);
}
if (!window.THREE) { fail('three.js did not load (offline, or the CDN is blocked).'); }
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false,
  preserveDrawingBuffer: D.still });   // a screenshot reads the buffer later
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
document.getElementById('app').appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 1, span/1000, span*20);
camera.position.set(cx + span*0.75, cy - span*0.95, cz + span*0.65);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy, cz);
controls.enableDamping = true; controls.dampingFactor = 0.08;

// ---- impostor spheres: one quad per cell, the sphere solved in the fragment
// shader.  No sphere geometry, one draw call, so 10^5 cells stay interactive.
const quad = new THREE.InstancedBufferGeometry();
quad.setAttribute('corner', new THREE.Float32BufferAttribute(
  [-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1], 2));
// three needs a real `position` attribute to exist; the shader ignores it
quad.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(18), 3));
quad.setAttribute('iPos', new THREE.InstancedBufferAttribute(pos, 3));
quad.setAttribute('iRadius', new THREE.InstancedBufferAttribute(rad, 1));
quad.setAttribute('iColor', new THREE.InstancedBufferAttribute(col, 3));
quad.instanceCount = D.n;

const uniforms = {
  uScale:   { value: 1.0 },
  uClipZ:   { value: z1 + 1.0 },
  uBright:  { value: 1.0 },
  uLightDir:{ value: new THREE.Vector3(0.4, -0.6, 0.8).normalize() },
  uAmbient: { value: new THREE.Color(0.42, 0.46, 0.52) },
  uRim:     { value: 0.35 },
};

const material = new THREE.ShaderMaterial({
  uniforms,
  side: THREE.DoubleSide,       // the billboard's winding depends on the view
  extensions: { fragDepth: true },
  vertexShader: `
    attribute vec2 corner;
    attribute vec3 iPos;
    attribute float iRadius;
    attribute vec3 iColor;
    uniform float uScale, uClipZ;
    varying vec3 vColor;
    varying vec2 vCorner;
    varying vec3 vCentreView;
    varying float vRadius;
    void main() {
      vColor = iColor;
      vCorner = corner;
      vRadius = iRadius * uScale;
      if (iPos.z > uClipZ) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
      vec4 centre = modelViewMatrix * vec4(iPos, 1.0);
      vCentreView = centre.xyz;
      // billboard the quad in view space, so the impostor always faces the eye
      vec4 view = centre + vec4(corner * vRadius, 0.0, 0.0);
      gl_Position = projectionMatrix * view;
    }`,
  fragmentShader: `
    precision highp float;
    uniform mat4 projectionMatrix;   // three injects this into the vertex stage only
    uniform vec3 uLightDir, uAmbient;
    uniform float uBright, uRim;
    varying vec3 vColor;
    varying vec2 vCorner;
    varying vec3 vCentreView;
    varying float vRadius;
    void main() {
      float d2 = dot(vCorner, vCorner);
      if (d2 > 1.0) discard;                       // outside the sphere's disc
      float z = sqrt(1.0 - d2);
      vec3 normal = vec3(vCorner, z);              // view-space normal
      vec3 viewPos = vCentreView + normal * vRadius;

      #ifdef GL_EXT_frag_depth
      // the true sphere depth, so impostors interpenetrate like real spheres
      vec4 clip = projectionMatrix * vec4(viewPos, 1.0);
      gl_FragDepthEXT = 0.5 + 0.5 * (clip.z / clip.w);
      #endif

      vec3 L = normalize(uLightDir);
      float diff = max(dot(normal, L), 0.0);
      vec3 V = normalize(-viewPos);
      vec3 H = normalize(L + V);
      float spec = pow(max(dot(normal, H), 0.0), 48.0) * 0.35;
      float rim = pow(1.0 - max(dot(normal, V), 0.0), 2.5) * uRim;
      vec3 c = vColor * (uAmbient + vec3(0.95) * diff) + vec3(spec) + vColor * rim;
      c *= uBright;
      c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));   // gamma
      gl_FragColor = vec4(c, 1.0);
    }`,
});
const cellMesh = new THREE.Mesh(quad, material);
cellMesh.frustumCulled = false;      // the impostors are built in the shader
scene.add(cellMesh);

if (D.fibres) {
  const a = dec(D.fibres.a, Float32Array), b = dec(D.fibres.b, Float32Array);
  const pts = new Float32Array(D.fibres.n * 6);
  for (let i = 0; i < D.fibres.n; i++) {
    pts[i*6+0]=a[i*3]; pts[i*6+1]=a[i*3+1]; pts[i*6+2]=a[i*3+2];
    pts[i*6+3]=b[i*3]; pts[i*6+4]=b[i*3+1]; pts[i*6+5]=b[i*3+2];
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pts, 3));
  scene.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial(
      { color: 0x5a6b85, transparent: true, opacity: 0.75 })));
}

const boxMat = new THREE.LineBasicMaterial({ color: 0x9aa7bd, transparent: true, opacity: 0.5 });
const box = new THREE.Box3Helper(
  new THREE.Box3(new THREE.Vector3(x0,y0,z0), new THREE.Vector3(x1,y1,z1)));
box.material = boxMat; scene.add(box);

// legend
if (D.colorBy === 'cell_type') {
  const seen = new Map(), legend = document.getElementById('legend');
  for (let i = 0; i < D.n; i++) if (!seen.has(typeIdx[i])) seen.set(typeIdx[i], i);
  [...seen.entries()].sort((a,b)=>a[0]-b[0]).forEach(([t, i]) => {
    const el = document.createElement('div'); el.className = 'key';
    const dot = document.createElement('span'); dot.className = 'dot';
    dot.style.background = `rgb(${col[i*3]*255|0},${col[i*3+1]*255|0},${col[i*3+2]*255|0})`;
    el.append(dot, document.createTextNode(D.types[t] || ('type ' + t)));
    legend.appendChild(el);
  });
} else {
  document.getElementById('legend').innerHTML =
    '<div class="key">coloured by ' + D.colorBy + ' &mdash; dark low, yellow high</div>';
}

function theme(dark) {
  document.body.className = dark ? 'dark' : 'light';
  document.getElementById('hud').className = dark ? 'dark' : 'light';
  scene.background = new THREE.Color(dark ? 0x0b1020 : 0xeef1f6);
  uniforms.uAmbient.value.setRGB(...(dark ? [0.30,0.34,0.42] : [0.42,0.46,0.52]));
  boxMat.color.setHex(dark ? 0x3a4866 : 0x9aa7bd);
}
const darkBox = document.getElementById('dark');
darkBox.checked = (D.theme === 'dark');
darkBox.onchange = () => theme(darkBox.checked);
theme(darkBox.checked);

const clip = document.getElementById('clip'), size = document.getElementById('size'),
      bright = document.getElementById('bright'), spin = document.getElementById('spin');
function apply() {
  uniforms.uScale.value = size.value / 100;
  uniforms.uBright.value = bright.value / 100;
  uniforms.uClipZ.value = z0 + (z1 - z0 + 1e-9) * (clip.value / 100) + 1e-6;
}
clip.oninput = size.oninput = bright.oninput = apply; apply();

function resize() {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();
// surface a shader or WebGL failure instead of leaving an empty canvas
renderer.debug.onShaderError = function (gl, program, vs, fs) {
  const log = gl.getProgramInfoLog(program) || '';
  const v = gl.getShaderInfoLog(vs) || '', f = gl.getShaderInfoLog(fs) || '';
  fail('the impostor shader failed to compile.\n\n' + log + '\n' + v + '\n' + f
       + '\n\nUse pyphysicell.viz3d.speck(sim) (ipyspeck) instead, or report this.');
};
let framesDrawn = 0;
if (D.still) {
  // draw synchronously: in headless Chrome requestAnimationFrame may never run
  // before the screenshot is taken
  for (let i = 0; i < 3; i++) { controls.update(); renderer.render(scene, camera); }
  // then bake the frame into an <img>, which a screenshot always captures
  try {
    const url = renderer.domElement.toDataURL('image/png');
    const img = new Image();
    img.src = url;
    img.style.cssText = 'position:fixed;inset:0;width:100%;height:100%';
    renderer.domElement.replaceWith(img);
  } catch (e) { /* keep the canvas if the buffer cannot be read */ }
}
(function animate() {
  if (D.still) return;
  requestAnimationFrame(animate);
  if (spin.checked) {
    const a = 0.0018, dx = camera.position.x - cx, dy = camera.position.y - cy;
    camera.position.x = cx + dx*Math.cos(a) - dy*Math.sin(a);
    camera.position.y = cy + dx*Math.sin(a) + dy*Math.cos(a);
  }
  controls.update();
  renderer.render(scene, camera);
  if (++framesDrawn === 30 && renderer.info.render.triangles === 0) {
    fail('nothing was drawn: ' + D.n + ' cells were uploaded but no triangles '
         + 'reached the screen. This is a bug in the viewer, not in your model.');
  }
})();
</script></body></html>
"""
