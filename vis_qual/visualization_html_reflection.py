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


def _pick_planes(labels: Any) -> List[np.ndarray]:
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    if isinstance(labels, np.ndarray):
        if labels.ndim == 2 and labels.shape[1] == 4:
            return [labels.astype(np.float32)]
        if labels.ndim == 3 and labels.shape[-1] == 4:
            return [labels[i].astype(np.float32) for i in range(labels.shape[0])]
        raise ValueError(f"labels must be (M,4) or (B,M,4). Got {labels.shape}")

    if isinstance(labels, (list, tuple)):
        out = []
        for i, arr in enumerate(labels):
            a = _as_numpy(arr)
            if a.ndim == 1 and a.shape[0] == 4:
                a = a.reshape(1, 4)
            if a.ndim != 2 or a.shape[1] != 4:
                raise ValueError(f"labels[{i}] must be (M,4). Got {a.shape}")
            out.append(a.astype(np.float32))
        return out

    raise ValueError(f"Unsupported labels type: {type(labels)}")


def _plane_square_corners(normal: np.ndarray, offset: float, size: float) -> np.ndarray:
    # plane: n·x + d = 0
    n = _normalize(normal)
    x0 = (-float(offset)) * n

    a = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(n, a))
    v = _normalize(np.cross(n, u))

    s = float(size) * 0.5
    p00 = x0 - s * u - s * v
    p10 = x0 + s * u - s * v
    p11 = x0 + s * u + s * v
    p01 = x0 - s * u + s * v
    return np.stack([p00, p10, p11, p01], axis=0).astype(np.float32)


def _make_single_html(points: np.ndarray, planes: np.ndarray, *, plane_opacity: float, point_size: float, prefix: str = "") -> str:
    points = points.astype(np.float32)

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

    # point size: 데이터 스케일에 따라 자동
    # point_size = max(0.003 * diag, 0.002)  # 기본값
    point_size = float(point_size)  # 인자로 받도록 변경
    # ✅ 슬라이더 범위(너무 작으면 안 보이고, 너무 크면 뭉개짐)
    point_size_min = max(point_size * 0.25, 1e-4)
    point_size_max = point_size * 50.0

    # plane quad size
    square_scale = 0.8  # bbox의 80% 정도 크기로
    square_size = square_scale * diag

    plane_quads = []
    for i in range(planes.shape[0]):
        corners = _plane_square_corners(planes[i, :3], float(planes[i, 3]), square_size)
        plane_quads.append(corners.tolist())

    if prefix == 'gt':
        plane_color = "#5EA7DF"  # blue-ish
    elif prefix == 'pred':
        # plane_color = "#B05EDF"  # purple-ish
        plane_color = "#DF5E5E"  # red-ish
    else:
        raise ValueError(f"Unsupported prefix: {prefix}. Expected 'gt' or 'pred'.")

    data = {
        "points": points.tolist(),
        "plane_quads": plane_quads,
        "center": center.tolist(),
        "camera": {
            "pos": [float(center[0]), float(center[1] - cam_dist), float(center[2] + 0.6 * cam_dist)],
            "lookAt": center.tolist(),
        },
        "style": {
            "bg": "#FFFFFF",
            "point": "#A2A0A0",
            "plane": plane_color,
            "planeOpacity": float(plane_opacity),
            "pointSize": float(point_size),
            "pointSizeMin": float(point_size_min),
            "pointSizeMax": float(point_size_max),
            "planeScale": 1.0,
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
      <span class="label">Plane opacity</span>
      <input id="op" type="range" min="0" max="1" step="0.01" value="{plane_opacity}">
      <span id="opv">{plane_opacity:.2f}</span>
    </div>

    <!-- ✅ point size 슬라이더 추가 -->
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
      <span class="label">Plane size</span>
      <input id="pscale" type="range" min="0.1" max="2.0" step="0.01" value="{data["style"]["planeScale"]}">
      <span id="pscalev">{data["style"]["planeScale"]:.2f}</span>
    </div>
  </div>

  <div id="err"></div>

  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";
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

      // 원형 point texture 생성
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

        // stroke 추가 (선명하게 보이도록)
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

      // planes
      const planeMeshes = [];
      const planeQuadsRaw = DATA.plane_quads;

      function makeTriVerticesFromQuad(quad, scale) {{
        const c = [
          (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0]) * 0.25,
          (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1]) * 0.25,
          (quad[0][2] + quad[1][2] + quad[2][2] + quad[3][2]) * 0.25,
        ];

        const q = quad.map((p) => [
          c[0] + (p[0] - c[0]) * scale,
          c[1] + (p[1] - c[1]) * scale,
          c[2] + (p[2] - c[2]) * scale,
        ]);

        return new Float32Array([
          q[0][0], q[0][1], q[0][2],
          q[1][0], q[1][1], q[1][2],
          q[2][0], q[2][1], q[2][2],
          q[0][0], q[0][1], q[0][2],
          q[2][0], q[2][1], q[2][2],
          q[3][0], q[3][1], q[3][2],
        ]);
      }}

      const initialPlaneScale = parseFloat(DATA.style.planeScale || 1.0);
      for (const quad of DATA.plane_quads) {{
        const g = new THREE.BufferGeometry();
        const v = makeTriVerticesFromQuad(quad, initialPlaneScale);
        g.setAttribute("position", new THREE.BufferAttribute(v, 3));
        g.computeVertexNormals();

        const m = new THREE.Mesh(
          g,
          new THREE.MeshBasicMaterial({{
            color: new THREE.Color(DATA.style.plane),
            transparent: true,
            opacity: DATA.style.planeOpacity,
            side: THREE.DoubleSide,
            depthWrite: false
          }})
        );
        scene.add(m);
        planeMeshes.push(m);
      }}

      // plane opacity slider
      const op = document.getElementById("op");
      const opv = document.getElementById("opv");
      op.addEventListener("input", () => {{
        const val = parseFloat(op.value);
        opv.textContent = val.toFixed(2);
        for (const m of planeMeshes) {{
          m.material.opacity = val;
          m.material.needsUpdate = true;
        }}
      }});

      // ✅ point size slider
      const ps = document.getElementById("ps");
      const psv = document.getElementById("psv");
      const fmt = (x) => {{
        // 너무 긴 소수는 보기 힘드니 적당히
        const ax = Math.abs(x);
        return (ax >= 0.01) ? x.toFixed(4) : x.toExponential(2);
      }};
      ps.addEventListener("input", () => {{
        const val = parseFloat(ps.value);
        psv.textContent = fmt(val);
        pcMat.size = val;
        pcMat.needsUpdate = true;
      }});

      // ✅ plane scale slider
      const pscale = document.getElementById("pscale");
      const pscalev = document.getElementById("pscalev");
      pscale.addEventListener("input", () => {{
        const val = parseFloat(pscale.value);
        pscalev.textContent = val.toFixed(2);
        for (let i = 0; i < planeMeshes.length; i++) {{
          const m = planeMeshes[i];
          const geom = m.geometry;
          const verts = makeTriVerticesFromQuad(planeQuadsRaw[i], val);
          geom.setAttribute("position", new THREE.BufferAttribute(verts, 3));
          geom.computeVertexNormals();
          geom.attributes.position.needsUpdate = true;
        }}
      }});

      function onResize() {{
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
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
    labels: Sequence[Any],
    save_dir: Union[str, Path],
    *,
    prefix: str = "sample",
    start_index: int = 0,
    plane_opacity: float = 0.44,
    point_size: float = 0.0266,
) -> List[Path]:
    """
    Save HTML per sample: pcs[i] + labels[i] (Mi,4).
    labels[i] contains plane normals & offsets: [nx,ny,nz,d] with n·x + d = 0
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pcs_list = _pick_pointclouds(pcs)
    planes_list = _pick_planes(labels)

    if len(pcs_list) != len(planes_list):
        raise ValueError(f"pcs has {len(pcs_list)} samples but labels has {len(planes_list)} samples")

    out_paths: List[Path] = []
    for i, (pc, pl) in enumerate(zip(pcs_list, planes_list)):
      html = _make_single_html(pc, pl, plane_opacity=plane_opacity, point_size=point_size, prefix=prefix)
      out_path = save_dir / f"{start_index + i:04d}_{prefix}.html"
      out_path.write_text(html, encoding="utf-8")
      out_paths.append(out_path)

    return out_paths