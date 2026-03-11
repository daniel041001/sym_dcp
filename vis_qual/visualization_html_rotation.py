from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Sequence, Union

import numpy as np


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / (n + eps)


def _pick_pointclouds(pcs: Any) -> List[np.ndarray]:
    if hasattr(pcs, "detach"):
        pcs = pcs.detach().cpu().numpy()

    if isinstance(pcs, np.ndarray):
        if pcs.ndim == 2 and pcs.shape[1] == 3:
            return [pcs.astype(np.float32)]
        if pcs.ndim == 3 and pcs.shape[-1] == 3:
            return [pcs[i].astype(np.float32) for i in range(pcs.shape[0])]
        raise ValueError(f"pcs must be (N,3) or (B,N,3). Got {pcs.shape}")

    if isinstance(pcs, (list, tuple)):
        out = []
        for i, p in enumerate(pcs):
            p_np = _as_numpy(p)
            if p_np.ndim != 2 or p_np.shape[1] != 3:
                raise ValueError(f"pcs[{i}] must be (N,3). Got {p_np.shape}")
            out.append(p_np.astype(np.float32))
        return out

    raise ValueError(f"Unsupported pcs type: {type(pcs)}")


def _pick_axis_dirs(axis_labels: Any) -> List[np.ndarray]:
    """
    axis_labels:
      - (M,3) or (B,M,3) numpy/torch
      - list of (M,3) or (3,)
    """
    if hasattr(axis_labels, "detach"):
        axis_labels = axis_labels.detach().cpu().numpy()

    if isinstance(axis_labels, np.ndarray):
        if axis_labels.ndim == 2 and axis_labels.shape[1] == 3:
            return [axis_labels.astype(np.float32)]
        if axis_labels.ndim == 3 and axis_labels.shape[-1] == 3:
            return [axis_labels[i].astype(np.float32) for i in range(axis_labels.shape[0])]
        raise ValueError(f"axis_labels must be (M,3) or (B,M,3). Got {axis_labels.shape}")

    if isinstance(axis_labels, (list, tuple)):
        out = []
        for i, arr in enumerate(axis_labels):
            a = _as_numpy(arr)
            if a.ndim == 1 and a.shape[0] == 3:
                a = a.reshape(1, 3)
            if a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"axis_labels[{i}] must be (M,3). Got {a.shape}")
            out.append(a.astype(np.float32))
        return out

    raise ValueError(f"Unsupported axis_labels type: {type(axis_labels)}")


def _pick_axis_points(point_labels: Any) -> List[np.ndarray]:
    """
    point_labels:
      - (M,3) or (B,M,3) numpy/torch
      - list of (M,3) or (3,)
    """
    if hasattr(point_labels, "detach"):
        point_labels = point_labels.detach().cpu().numpy()

    if isinstance(point_labels, np.ndarray):
        if point_labels.ndim == 2 and point_labels.shape[1] == 3:
            return [point_labels.astype(np.float32)]
        if point_labels.ndim == 3 and point_labels.shape[-1] == 3:
            return [point_labels[i].astype(np.float32) for i in range(point_labels.shape[0])]
        raise ValueError(f"point_labels must be (M,3) or (B,M,3). Got {point_labels.shape}")

    if isinstance(point_labels, (list, tuple)):
        out = []
        for i, arr in enumerate(point_labels):
            a = _as_numpy(arr)
            if a.ndim == 1 and a.shape[0] == 3:
                a = a.reshape(1, 3)
            if a.ndim != 2 or a.shape[1] != 3:
                raise ValueError(f"point_labels[{i}] must be (M,3). Got {a.shape}")
            out.append(a.astype(np.float32))
        return out

    raise ValueError(f"Unsupported point_labels type: {type(point_labels)}")


def _make_single_html(
    points: np.ndarray,
    axis_dirs: np.ndarray,
    axis_points: np.ndarray,
    *,
    axis_opacity: float,
    point_size: float,
    prefix: str = "",
) -> str:
    points = points.astype(np.float32)
    axis_dirs = axis_dirs.astype(np.float32)
    axis_points = axis_points.astype(np.float32)

    if axis_dirs.shape != axis_points.shape:
        raise ValueError(f"axis_dirs and axis_points must have same shape (M,3). "
                         f"Got {axis_dirs.shape} vs {axis_points.shape}")

    # bbox / center / radius
    if points.shape[0] > 0:
        bb_min = points.min(axis=0)
        bb_max = points.max(axis=0)
        center = ((bb_min + bb_max) * 0.5).astype(np.float32)
        diag = float(np.linalg.norm(bb_max - bb_min))
    else:
        center = np.zeros(3, dtype=np.float32)
        diag = 1.0
    if diag < 1e-9:
        diag = 1.0

    radius = 0.5 * diag
    cam_dist = 3.0 * radius

    # point size slider range
    point_size = float(point_size)
    point_size_min = max(point_size * 0.25, 1e-4)
    point_size_max = point_size * 50.0

    # axis segments (each: [p0, p1])
    # length scales with pointcloud size
    axis_half_len = 0.6 * diag
    axis_segments: List[List[List[float]]] = []
    for k in range(axis_dirs.shape[0]):
        d = axis_dirs[k]
        p = axis_points[k]
        dn = float(np.linalg.norm(d))
        if not np.isfinite(dn) or dn < 1e-12:
            continue
        u = (d / dn).astype(np.float32)
        p0 = (p - axis_half_len * u).astype(np.float32)
        p1 = (p + axis_half_len * u).astype(np.float32)
        axis_segments.append([p0.tolist(), p1.tolist()])

    # color by prefix (fallback included)
    if prefix == "gt":
        axis_color = "#5EA7DF"  # blue-ish
    elif prefix == "pred":
        axis_color = "#DF5E5E"  # red-ish
    else:
        axis_color = "#2ECC71"  # green fallback

    data = {
        "points": points.tolist(),
        "axis_segments": axis_segments,
        "center": center.tolist(),
        "camera": {
            "pos": [float(center[0]), float(center[1] - cam_dist), float(center[2] + 0.6 * cam_dist)],
            "lookAt": center.tolist(),
        },
        "style": {
            "bg": "#FFFFFF",
            "point": "#A2A0A0",
            "axis": axis_color,
            "axisOpacity": float(axis_opacity),
            "axisLineWidth": 7.0,  # 화면 픽셀 단위 (Line2)
            "axisLineWidthMin": 0.5,
            "axisLineWidthMax": 20.0,
            "pointSize": float(point_size),
            "pointSizeMin": float(point_size_min),
            "pointSizeMax": float(point_size_max),
        },
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Qualitative Visualization</title>
  <style>
    html, body {{ margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#fff; }}
    #wrap {{ position:fixed; inset:0; }}
    #hud {{
      position:fixed; left:12px; top:12px;
      background:rgba(255,255,255,0.90);
      border:1px solid rgba(0,0,0,0.10);
      border-radius:10px;
      padding:10px 12px;
      font: 12px/1.2 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Inter,Arial,sans-serif;
      color:#222;
      user-select:none;
      z-index: 10;
      min-width: 260px;
    }}
    #hud .row {{ display:flex; align-items:center; gap:10px; margin-top:8px; }}
    #hud .row span.label {{ width: 92px; }}
    #hud input[type="range"] {{ width:160px; }}
    #err {{
      position:fixed; left:12px; bottom:12px; right:12px;
      background:rgba(255,230,230,0.95);
      border:1px solid rgba(200,0,0,0.35);
      border-radius:10px;
      padding:10px 12px;
      font: 12px/1.35 ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono","Courier New",monospace;
      color:#700;
      white-space:pre-wrap;
      display:none;
      z-index: 20;
    }}
  </style>

  <script type="importmap">
  {{
    "imports": {{
      "three": "https://unpkg.com/three@0.158.0/build/three.module.js",
      "three/addons/": "https://unpkg.com/three@0.158.0/examples/jsm/"
    }}
  }}
  </script>
</head>
<body>
  <div id="wrap"></div>

  <div id="hud">
    <div><b>Viewer</b></div>

    <div class="row">
      <span class="label">Axis opacity</span>
      <input id="op" type="range" min="0" max="1" step="0.01" value="{axis_opacity}">
      <span id="opv">{axis_opacity:.2f}</span>
    </div>

    <div class="row">
      <span class="label">Point size</span>
      <input id="ps" type="range"
        min="{data["style"]["pointSizeMin"]}"
        max="{data["style"]["pointSizeMax"]}"
        step="{max(data["style"]["pointSizeMin"] * 0.02, 1e-5)}"
        value="{data["style"]["pointSize"]}">
      <span id="psv">{data["style"]["pointSize"]:.6f}</span>
    </div>

    <div class="row">
      <span class="label">Axis line width</span>
      <input id="alw" type="range"
        min="{data["style"]["axisLineWidthMin"]}"
        max="{data["style"]["axisLineWidthMax"]}"
        step="{max(data["style"]["axisLineWidthMin"] * 0.02, 1e-5)}"
        value="{data["style"]["axisLineWidth"]}">
      <span id="alwv">{data["style"]["axisLineWidth"]:.6f}</span>
    </div>
  </div>

  <div id="err"></div>

  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
    import {{ Line2 }} from "three/addons/lines/Line2.js";
    import {{ LineMaterial }} from "three/addons/lines/LineMaterial.js";
    import {{ LineGeometry }} from "three/addons/lines/LineGeometry.js";

    const DATA = {json.dumps(data)};

    function showError(e) {{
      const el = document.getElementById("err");
      el.style.display = "block";
      el.textContent = String(e?.stack || e);
      console.error(e);
    }}

    try {{
      const wrap = document.getElementById("wrap");
      const scene = new THREE.Scene();
      scene.background = new THREE.Color(DATA.style.bg);

      const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.001, 1e6);
      camera.position.set(...DATA.camera.pos);

      const renderer = new THREE.WebGLRenderer({{ antialias:true }});
      renderer.setSize(window.innerWidth, window.innerHeight);
      renderer.setPixelRatio(window.devicePixelRatio || 1);
      wrap.appendChild(renderer.domElement);

      const controls = new OrbitControls(camera, renderer.domElement);
      controls.target.set(...DATA.camera.lookAt);
      controls.update();

      // lights
      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const dl = new THREE.DirectionalLight(0xffffff, 0.8);
      dl.position.set(1, 1, 2);
      scene.add(dl);

      // point cloud
      const pts = DATA.points;
      const pcGeom = new THREE.BufferGeometry();
      const flat = new Float32Array(pts.length * 3);
      for (let i=0;i<pts.length;i++) {{
        flat[3*i+0] = pts[i][0];
        flat[3*i+1] = pts[i][1];
        flat[3*i+2] = pts[i][2];
      }}
      pcGeom.setAttribute("position", new THREE.BufferAttribute(flat, 3));

      // circular point texture (with black border)
      function createCircleTexture() {{
        const size = 128;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const stroke = "black";
        const lineWidth = 4;

        const ctx = canvas.getContext("2d");

        ctx.beginPath();
        ctx.arc(size/2, size/2, size/2, 0, Math.PI * 2);
        ctx.closePath();
        ctx.fillStyle = "white";
        ctx.fill();

        if (lineWidth > 0) {{
          ctx.lineWidth = lineWidth;
          ctx.strokeStyle = stroke;
          ctx.stroke();
        }}

        const tex = new THREE.CanvasTexture(canvas);
        tex.needsUpdate = true;
        return tex;
      }}
      const circleTexture = createCircleTexture();

      const pcMat = new THREE.PointsMaterial({{
        color: new THREE.Color(DATA.style.point),
        size: DATA.style.pointSize,
        sizeAttenuation: true,
        map: circleTexture,
        alphaTest: 0.5,
        transparent: false,
        depthWrite: true,
      }});
      const pcObj = new THREE.Points(pcGeom, pcMat);
      scene.add(pcObj);

      // axes (thick lines via Line2)
      const axisLines = [];
      for (const seg of DATA.axis_segments) {{
        const p0 = seg[0];
        const p1 = seg[1];

        const geom = new LineGeometry();
        // LineGeometry expects a flat array of xyzxyz...
        geom.setPositions([p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]]);

        const mat = new LineMaterial({{
          color: new THREE.Color(DATA.style.axis),
          linewidth: DATA.style.axisLineWidth, // in pixels
          transparent: true,
          opacity: DATA.style.axisOpacity,
          depthTest: true,
        }});
        mat.resolution.set(window.innerWidth, window.innerHeight);

        const line = new Line2(geom, mat);
        line.computeLineDistances();
        scene.add(line);
        axisLines.push(line);
      }}

      // axis opacity slider
      const op = document.getElementById("op");
      const opv = document.getElementById("opv");
      op.addEventListener("input", () => {{
        const val = parseFloat(op.value);
        opv.textContent = val.toFixed(2);
        for (const l of axisLines) {{
          l.material.opacity = val;
          l.material.needsUpdate = true;
        }}
      }});

      // point size slider
      const ps = document.getElementById("ps");
      const psv = document.getElementById("psv");
      const fmt = (x) => {{
        const ax = Math.abs(x);
        return (ax >= 0.01) ? x.toFixed(4) : x.toExponential(2);
      }};
      ps.addEventListener("input", () => {{
        const val = parseFloat(ps.value);
        psv.textContent = fmt(val);
        pcMat.size = val;
        pcMat.needsUpdate = true;
      }});

      // axis line width slider
      const alw = document.getElementById("alw");
      const alwv = document.getElementById("alwv");

      if (alw) {{
        alw.addEventListener("input", () => {{
            const val = parseFloat(alw.value);
            alwv.textContent = val.toFixed(1);

            for (const l of axisLines) {{
                l.material.linewidth = val;
                l.material.needsUpdate = true;
            }}
        }}); 
      }}

      function onResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);

        // update LineMaterial resolution for thick lines
        for (const l of axisLines) {{
          l.material.resolution.set(window.innerWidth, window.innerHeight);
        }}
      }}
      window.addEventListener("resize", onResize);

      function animate() {{
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
      }}
      animate();

    }} catch (e) {{
      showError(e);
    }}
  </script>
</body>
</html>
"""


def save_visualization(
    pcs: Any,
    axis_labels: Any,
    point_labels: Any,
    save_dir: Union[str, Path],
    *,
    prefix: str = "sample",
    start_index: int = 0,
    axis_opacity: float = 0.70,
    point_size: float = 0.0266,
) -> List[Path]:
    """
    Save HTML per sample: pcs[i] + (axis_labels[i], point_labels[i]).

    axis_labels[i]: (M,3) direction vectors (each can be non-unit; normalized internally)
    point_labels[i]: (M,3) a point on each axis
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pcs_list = _pick_pointclouds(pcs)
    axis_dirs_list = _pick_axis_dirs(axis_labels)
    axis_pts_list = _pick_axis_points(point_labels)

    if not (len(pcs_list) == len(axis_dirs_list) == len(axis_pts_list)):
        raise ValueError(
            f"Batch size mismatch: pcs={len(pcs_list)}, axis_labels={len(axis_dirs_list)}, point_labels={len(axis_pts_list)}"
        )

    out_paths: List[Path] = []
    for i, (pc, ad, ap) in enumerate(zip(pcs_list, axis_dirs_list, axis_pts_list)):
        html = _make_single_html(
            pc,
            ad,
            ap,
            axis_opacity=axis_opacity,
            point_size=point_size,
            prefix=prefix,
        )
        out_path = save_dir / f"{start_index + i:04d}_{prefix}.html"
        out_path.write_text(html, encoding="utf-8")
        out_paths.append(out_path)

    return out_paths