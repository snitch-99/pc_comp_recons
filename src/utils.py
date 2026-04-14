"""
utils.py — Shared helpers for the Rock Reconstruction Pipeline
==============================================================
Centralises:
  - Superquadric math  (signed_pow, surface point/normal, parse txt)
  - Open3D visualization helpers (_shift, _make_grid_panel, event_loop)
  - Existing SQ geometry / analysis functions
"""

import copy
import os
import numpy as np
import open3d as o3d
import scipy.signal

# =============================================================================
# SUPERQUADRIC MATH  (pure — no globals)
# =============================================================================

def signed_pow(x: np.ndarray, e: float) -> np.ndarray:
    """Signed power: sign(x) * |x|^e — safe for negative x."""
    return np.sign(x) * (np.abs(x) ** e)


def sq_point_canonical(eta, omega, a, e1, e2) -> np.ndarray:
    """
    Evaluate a Superquadric surface point in canonical frame.
    eta   : longitude angle array
    omega : latitude angle array
    a     : [ax, ay, az]
    Returns (..., 3) array.
    """
    ce = signed_pow(np.cos(eta), e2)
    se = signed_pow(np.sin(eta), e2)
    cw = signed_pow(np.cos(omega), e1)
    sw = signed_pow(np.sin(omega), e1)
    x  = a[0] * cw * ce
    y  = a[1] * cw * se
    z  = a[2] * sw
    return np.stack([x, y, z], axis=-1)


def sq_normal_canonical(p: np.ndarray, a, e1, e2) -> np.ndarray:
    """
    Outward surface normal of the implicit SQ function (normalised).
    p : (..., 3) canonical-frame points on the surface
    Returns (..., 3) unit normals.
    """
    eps = 1e-12
    x, y, z       = p[..., 0], p[..., 1], p[..., 2]
    a1, a2, a3    = a[0], a[1], a[2]
    e1 = max(e1, 0.1); e2 = max(e2, 0.1)
    p_xy = 2.0 / e2;  p_z = 2.0 / e1;  q = e2 / e1
    X    = np.abs(x / (a1 + eps))
    Y    = np.abs(y / (a2 + eps))
    Z    = np.abs(z / (a3 + eps))
    A    = X**p_xy + Y**p_xy
    dF_dx = q * (A**(q-1)) * p_xy * (X**(p_xy-1)) * (np.sign(x) / (a1+eps))
    dF_dy = q * (A**(q-1)) * p_xy * (Y**(p_xy-1)) * (np.sign(y) / (a2+eps))
    dF_dz = p_z * (Z**(p_z-1)) * (np.sign(z) / (a3+eps))
    n  = np.stack([dF_dx, dF_dy, dF_dz], axis=-1)
    nn = np.linalg.norm(n, axis=-1, keepdims=True) + eps
    return n / nn


def parse_sq_fit_txt(txt_path: str) -> dict:
    """
    Parse a sq_fit_Cluster_N.txt file produced by Step 1.
    Returns dict with keys: params, center, R
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    result: dict = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("Best-fit Superquadric parameters"):
            result["params"] = list(map(float, lines[i+1].strip().split()))
            i += 2; continue
        if line == "Center (world):":
            result["center"] = np.array(list(map(float, lines[i+1].strip().split())))
            i += 2; continue
        if line.startswith("Rotation R_init"):
            rows = [list(map(float, lines[i+k].strip().split())) for k in range(1, 4)]
            result["R"] = np.array(rows, dtype=np.float64)
            i += 4; continue
        i += 1
    return result


# =============================================================================
# OPEN3D VISUALIZATION HELPERS
# =============================================================================

def viz_shift(geom, dx: float):
    """Return a deep copy of geom translated by dx along X."""
    g = copy.deepcopy(geom)
    g.translate([dx, 0, 0])
    return g


def viz_make_grid_panel(center_x: float,
                        bb_min: np.ndarray, bb_max: np.ndarray,
                        y_floor: float,
                        axis_size: float,
                        grid_spacing: float) -> list:
    """
    Build an XZ reference grid + coordinate frame for one display panel.
    All grids share the same y_floor so they are co-planar.
    Returns a list of Open3D geometries.
    """
    geoms = []

    # Flat XZ grid
    x0 = center_x + float(bb_min[0]); x1 = center_x + float(bb_max[0])
    z0 = float(bb_min[2]);             z1 = float(bb_max[2])
    pts, lines, idx = [], [], 0
    for x in np.arange(x0, x1 + grid_spacing, grid_spacing):
        pts += [[x, y_floor, z0], [x, y_floor, z1]]
        lines.append([idx, idx+1]); idx += 2
    for z in np.arange(z0, z1 + grid_spacing, grid_spacing):
        pts += [[x0, y_floor, z], [x1, y_floor, z]]
        lines.append([idx, idx+1]); idx += 2

    if pts:
        grid = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(pts, dtype=float)),
            lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)),
        )
        grid.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))
        geoms.append(grid)

    # Coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[center_x + float(bb_min[0]), y_floor, float(bb_min[2])],
    )
    geoms.append(frame)
    return geoms


def viz_event_loop(visualizers: list):
    """Run the Open3D poll/update event loop until all windows are closed.

    If the environment variable PC_HEADLESS=1 is set (e.g. when launched from
    the GUI), the windows are skipped entirely so the step exits immediately.
    """
    if os.environ.get("PC_HEADLESS") == "1":
        for vis in visualizers:
            vis.destroy_window()
        return
    open_flags = [True] * len(visualizers)
    while any(open_flags):
        for i, vis in enumerate(visualizers):
            if open_flags[i]:
                if not vis.poll_events():
                    open_flags[i] = False
                else:
                    vis.update_renderer()
    for vis in visualizers:
        vis.destroy_window()


# =============================================================================
# EXISTING SQ / MESH HELPERS  (kept from original utils.py)
# =============================================================================

def world_to_parametric(model, points: np.ndarray):
    """Approximate inverse mapping for a canonical-frame Superquadric."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    def sgn_pow_inv(val, e):
        return np.sign(val) * (np.abs(val) ** (1.0 / e))

    sin_eta = np.clip(sgn_pow_inv(z / model.az, model.e1), -1.0, 1.0)
    eta     = np.arcsin(sin_eta)

    x_base  = sgn_pow_inv(x / model.ax, model.e2)
    y_base  = sgn_pow_inv(y / model.ay, model.e2)
    omega   = np.arctan2(y_base, x_base)
    return eta, omega


def get_mesh(model, resolution: int = 50) -> o3d.geometry.TriangleMesh:
    """Generate a full SQ triangle mesh for visualization."""
    eta   = np.linspace(-np.pi/2, np.pi/2, resolution)
    omega = np.linspace(-np.pi, np.pi, resolution)
    ETA, OMEGA = np.meshgrid(eta, omega)

    x = model.ax * signed_pow(np.cos(ETA), model.e1) * signed_pow(np.cos(OMEGA), model.e2)
    y = model.ay * signed_pow(np.cos(ETA), model.e1) * signed_pow(np.sin(OMEGA), model.e2)
    z = model.az * signed_pow(np.sin(ETA), model.e1)

    vertices  = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            i0 = i*resolution + j
            i1 = i*resolution + j+1
            i2 = (i+1)*resolution + j+1
            i3 = (i+1)*resolution + j
            triangles += [[i0, i2, i1], [i0, i3, i2]]

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


def analyze_inliers(model, points: np.ndarray,
                    distance_threshold: float = 0.01):
    """Count inliers within a radial distance threshold."""
    mu          = model.radial_distance_approximation(points)
    distances   = np.linalg.norm(points - mu, axis=1)
    inlier_mask = distances <= distance_threshold
    num_inliers = int(np.sum(inlier_mask))
    return num_inliers, len(points) - num_inliers, inlier_mask


def get_model_cloud(model, width: int = 360,
                    height: int = 180) -> o3d.geometry.PointCloud:
    """Generate a point cloud sampled from the SQ surface."""
    u, v = np.linspace(0, width-1, width), np.linspace(0, height-1, height)
    U, V = np.meshgrid(u, v)
    lon  = (U / (width-1))  * 2*np.pi - np.pi
    lat  = (V / (height-1)) * np.pi   - np.pi/2

    x = model.ax * signed_pow(np.cos(lat), model.e1) * signed_pow(np.cos(lon), model.e2)
    y = model.ay * signed_pow(np.cos(lat), model.e1) * signed_pow(np.sin(lon), model.e2)
    z = model.az * signed_pow(np.sin(lat), model.e1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.dstack((x, y, z)).reshape(-1, 3))
    pcd.estimate_normals()
    return pcd
