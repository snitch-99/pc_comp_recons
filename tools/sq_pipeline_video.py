#!/usr/bin/env python3
"""
sq_pipeline_video.py
====================
Generates a step-by-step MP4 video of the SQ+DEM compression pipeline
on Apollo 15 rock_1.

The rock rotates continuously. Each pipeline step overlays sequentially:
  1. Original rock.obj with photogrammetry texture/color
  2. Point cloud sampling
  3. SQ EMS fitting — OBB init → EM loops → S-step candidates + rejection
  4. Final SQ gridded at 720×360
  5. Raycasting from SQ grid → rock surface
  6. Poisson reconstruction from displaced points
  7. Split screen: Original vs Reconstructed

Output: /home/kanav/workspaces/pc_comp_recons/sq_pipeline_video.mp4

Rendering backend: Open3D OffscreenRenderer (GPU/EGL).
"""

import os
import sys
import copy
import math
import contextlib
import warnings

import numpy as np
import open3d as o3d
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from ems_core import Superquadric, EMSFitter, MAX_EMS_LOOPS, CONVERGENCE_TOL, SAFE_MIN_VAL
import utils

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROCK_DIR       = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1'
OBJ_PATH       = os.path.join(ROCK_DIR, 'rock.obj')
PLY_PATH       = os.path.join(ROCK_DIR, 'Cluster_1.ply')
POISSON_PATH   = os.path.join(ROCK_DIR, 'poisson_Cluster_1.ply')
RECON_PTS_PATH = os.path.join(ROCK_DIR, 'recon_pts_Cluster_1.ply')
OUT_VIDEO      = '/home/kanav/workspaces/pc_comp_recons/sq_pipeline_video.mp4'

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO CONFIG
# ─────────────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1920, 1080
FPS           = 30
BG_COLOR      = [0.07, 0.07, 0.12, 1.0]   # dark blue-gray RGBA 0-1

DEM_W, DEM_H  = 720, 360
N_RAYS_VIZ    = 200     # rays to animate in section 5
EMS_DOWNSAMPLE = 5000   # points used for EMS recording

# Display geometry limits (keep rendering fast)
MAX_DISPLAY_PTS   = 6000   # max scatter points per frame
MAX_DISPLAY_LINES = 1500   # max line segments per frame

# Section durations in frames (30 fps)  — total = 1350 = 45s
N_ROCK       = 150   # 5s  — rock alone
N_SAMPLE     = 120   # 4s  — point sampling
N_INIT       = 90    # 3s  — OBB initial guess
N_EMS        = 300   # 10s — EMS loops + S-step candidates
N_OUTSCRIBE  = 60    # 2s  — best-fit → outscribed expansion
N_GRID       = 120   # 4s  — 720×360 grid appears
N_RAYCAST    = 150   # 5s  — ray animation
N_RECON      = 120   # 4s  — reconstruction
N_COMPARE    = 240   # 8s  — split comparison

# ─────────────────────────────────────────────────────────────────────────────
# COLORS  (RGB 0–1)
# ─────────────────────────────────────────────────────────────────────────────
C_ROCK_FALLBACK = [0.72, 0.60, 0.45]
C_PC            = [0.25, 0.60, 1.00]
C_SQ            = [1.00, 0.50, 0.15]
C_SQ_INIT       = [0.55, 0.55, 0.90]
C_SQ_WIRE       = [1.00, 0.78, 0.30]
C_GRID          = [0.30, 0.90, 0.80]
C_RAY           = [1.00, 0.95, 0.30]
C_HIT           = [1.00, 0.50, 0.10]
C_RECON         = [0.45, 0.85, 0.55]

CAND_COLORS = [
    [0.35, 0.85, 0.35],
    [0.85, 0.35, 0.85],
    [0.35, 0.85, 0.85],
]

# ─────────────────────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────────────────────
_FONT_PATHS = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
]
_FONT_PATHS_REG = [p.replace('Bold', '').replace('-Bold', '') for p in _FONT_PATHS]


def _load_font(paths, size):
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT_TITLE = _load_font(_FONT_PATHS,     50)
FONT_BODY  = _load_font(_FONT_PATHS_REG, 30)
FONT_LABEL = _load_font(_FONT_PATHS_REG, 26)
FONT_SMALL = _load_font(_FONT_PATHS_REG, 22)


# ─────────────────────────────────────────────────────────────────────────────
# RECORDING EMS FITTER
# ─────────────────────────────────────────────────────────────────────────────
class RecordingEMSFitter(EMSFitter):
    """
    Subclasses EMSFitter to capture fitting history:
      - initial: OBB params before any EM
      - em_done: converged params + inlier weights after each EM run
      - s_step:  all candidates with their losses and winner flag
    """

    def __init__(self, pcd, **kwargs):
        super().__init__(pcd, **kwargs)
        self.history   = []
        self._loop_num = 0

    def fit(self, max_iters=100, external_pbar=None):
        self.history.append({
            'type':   'initial',
            'params': list(self.params),
            'center': self.center.copy(),
            'R':      self.R_init.copy(),
            'extent': self.extent.copy(),
        })

        ems_converged = False
        loop_count    = 0

        while not ems_converged and loop_count < MAX_EMS_LOOPS:
            loop_count    += 1
            self._loop_num = loop_count

            self._run_em_to_convergence(max_iters, None)

            z_prob, _ = self.e_step()
            self.history.append({
                'type':   'em_done',
                'loop':   loop_count,
                'params': list(self.params),
                'z_prob': z_prob.copy(),
                'center': self.center.copy(),
                'R':      self.R_init.copy(),
                'loss':   self._loss(self.params),
            })

            if not self._s_step_recording():
                ems_converged = True

        return Superquadric(self.params)

    def _s_step_recording(self):
        best_loss   = self._loss(self.params)
        best_params = copy.deepcopy(self.params)
        ax, ay, az, e1, e2 = self.params

        candidates = [
            [az, ay, ax, e2, e1],
            [ax, az, ay, e2, e1],
        ]
        if e2 < 1.0:
            candidates.append([ax * math.sqrt(2), ay * math.sqrt(2), az, e1, 2.0 - e2])
        elif e2 > 1.0:
            candidates.append([ax / math.sqrt(2), ay / math.sqrt(2), az, e1, 2.0 - e2])
        else:
            candidates.append([ax, ay, az, e1, 1.0])

        cand_data = []
        found     = False

        for cand in candidates:
            cand_clipped = [max(c, SAFE_MIN_VAL) for c in cand]
            loss         = self._loss(cand_clipped)
            is_better    = loss < best_loss
            cand_data.append({
                'params':  cand_clipped,
                'loss':    loss,
                'better':  is_better,
            })
            if is_better:
                best_loss   = loss
                best_params = cand_clipped
                found       = True

        self.history.append({
            'type':           's_step',
            'loop':           self._loop_num,
            'current_params': list(self.params),
            'current_loss':   self._loss(self.params),
            'candidates':     cand_data,
            'switched':       found,
        })

        if found:
            self.params = best_params
        return found


# ─────────────────────────────────────────────────────────────────────────────
# MATERIAL HELPER
# ─────────────────────────────────────────────────────────────────────────────
def mkmat(color, shader='defaultUnlit', alpha=1.0, pt_size=3.5):
    """Return a simple material dict. (shader arg kept for call-site compat.)"""
    return {'color': list(color), 'alpha': float(alpha), 'pt_size': float(pt_size)}


# ─────────────────────────────────────────────────────────────────────────────
# FILAMENT NOISE SUPPRESSION
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def suppress_filament():
    """Redirect fd 1+2 to /dev/null — silences all Filament/EGL console output."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_out, 1); os.close(old_out)
        os.dup2(old_err, 2); os.close(old_err)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
def orbit_cam(angle_deg, elev_deg=22.0):
    """Return (azim_deg, elev_deg) camera angles."""
    return float(angle_deg % 360), float(elev_deg)


def _cam_eye(azim_deg, elev_deg, center, dist):
    az = np.radians(azim_deg)
    el = np.radians(elev_deg)
    return center + dist * np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# GPU RENDERING  (Open3D OffscreenRenderer)
# ─────────────────────────────────────────────────────────────────────────────
def _render_scene(geoms, cam_params, scene_center, scene_radius, width, height):
    """
    Render one scene with a fresh OffscreenRenderer.
    geoms : list of (name, o3d_geometry, mat_dict)
    Returns HxWx3 uint8 RGB image.
    """
    azim, elev = cam_params
    center = np.array(scene_center, dtype=float)
    eye    = _cam_eye(azim, elev, center, scene_radius * 2.2)

    with suppress_filament():
        r = o3d.visualization.rendering.OffscreenRenderer(width, height)
        r.scene.set_background(BG_COLOR)
        r.scene.scene.set_sun_light([0.4, -0.7, -0.5], [1.0, 0.97, 0.92], 120000)
        r.scene.scene.enable_sun_light(True)
        r.scene.scene.enable_indirect_light(True)
        r.scene.scene.set_indirect_light_intensity(50000)

        for name, geom, mat in geoms:
            if geom is None:
                continue
            color = list(mat['color'])
            alpha = float(np.clip(mat.get('alpha', 1.0), 0.0, 1.0))
            if alpha < 0.02:
                continue

            m = o3d.visualization.rendering.MaterialRecord()

            if isinstance(geom, o3d.geometry.TriangleMesh):
                m.shader     = 'defaultLit'
                m.base_color = color + [alpha]
                geom = o3d.geometry.TriangleMesh(geom)
                geom.vertex_colors = o3d.utility.Vector3dVector([])
                if not geom.has_vertex_normals():
                    geom.compute_vertex_normals()

            elif isinstance(geom, o3d.geometry.PointCloud):
                m.shader     = 'defaultUnlit'
                m.base_color = color + [alpha]
                m.point_size = float(mat.get('pt_size', 3.5))

            elif isinstance(geom, o3d.geometry.LineSet):
                m.shader     = 'unlitLine'
                m.base_color = color + [alpha]
                m.line_width = 1.5
            else:
                continue

            r.scene.add_geometry(name, geom, m)

        r.setup_camera(55.0, center.tolist(), eye.tolist(), [0.0, 1.0, 0.0])
        img = np.asarray(r.render_to_image()).copy()
        del r

    return img


def render_frame(geoms, cam_params, scene_center, scene_radius):
    """Render one full-resolution frame on GPU."""
    return _render_scene(geoms, cam_params, scene_center, scene_radius, WIDTH, HEIGHT)


def render_split_frame(geoms_l, geoms_r, cam_params, scene_center, scene_radius):
    """Render two half-width panels side by side."""
    half_w = WIDTH // 2
    left  = _render_scene(geoms_l, cam_params, scene_center, scene_radius, half_w, HEIGHT)
    right = _render_scene(geoms_r, cam_params, scene_center, scene_radius, half_w, HEIGHT)
    return np.hstack([left, right])


# ─────────────────────────────────────────────────────────────────────────────
# TEXT OVERLAY
# ─────────────────────────────────────────────────────────────────────────────
def overlay(frame, title, body=None, labels=None):
    """
    Add text overlays to a uint8 RGB frame.
    labels : list of (x, y, text, rgb_tuple_0_255)
    Returns annotated frame.
    """
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, 'RGBA')

    draw.rectangle([(0, 0), (WIDTH, 80)], fill=(5, 5, 20, 210))
    draw.text((36, 14), title, font=FONT_TITLE, fill=(255, 215, 70, 255))

    if body:
        draw.rectangle([(0, HEIGHT - 58), (WIDTH, HEIGHT)], fill=(5, 5, 20, 190))
        draw.text((36, HEIGHT - 48), body, font=FONT_BODY, fill=(200, 200, 210, 255))

    if labels:
        for (x, y, text, col) in labels:
            bb = draw.textbbox((x, y), text, font=FONT_LABEL)
            draw.rectangle([bb[0]-6, bb[1]-4, bb[2]+6, bb[3]+4], fill=(0, 0, 0, 170))
            draw.text((x, y), text, font=FONT_LABEL, fill=(*col, 255))

    return np.array(img)[:, :, :3]


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY BUILDERS  (Open3D objects — used as data containers, not rendered directly)
# ─────────────────────────────────────────────────────────────────────────────
def build_sq_mesh(params, R, center, color=None):
    color = color or C_SQ
    sq    = Superquadric(params)
    V, F  = sq.sample_surface(nu=40, nv=20)    # smaller for speed; still looks smooth
    mesh  = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(V),
        triangles=o3d.utility.Vector3iVector(F),
    )
    mesh.compute_vertex_normals()
    T = np.eye(4)
    T[:3, :3] = R.T
    T[:3, 3]  = center
    mesh.transform(T)
    mesh.paint_uniform_color(color)
    return mesh


def build_sq_wireframe(params, R, center, color=None):
    color = color or C_SQ_WIRE
    m     = build_sq_mesh(params, R, center, color)
    wire  = o3d.geometry.LineSet.create_from_triangle_mesh(m)
    wire.paint_uniform_color(color)
    return wire


def build_pc_weighted(points_can, z_prob, R, center):
    """Point cloud colored blue→red by inlier probability."""
    z         = np.clip(z_prob, 0, 1)
    colors    = np.zeros((len(z), 3))
    colors[:, 0] = 1.0 - z
    colors[:, 2] = z
    pts_world = points_can @ R.T + center
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_world)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_dem_grid(params, R, center):
    """720×360 wireframe on the SQ surface (subsampled for clarity)."""
    a      = np.array(params[:3])
    e1, e2 = params[3], params[4]

    eta   = -np.pi     + (np.arange(DEM_W) + 0.5) * (2.0 * np.pi / DEM_W)
    omega = -0.5*np.pi + (np.arange(DEM_H) + 0.5) * (np.pi  / DEM_H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = utils.sq_point_canonical(ETA, OMEGA, a, e1, e2)
    P_w   = (P_can @ R.T) + center

    # Every 20th ring/meridian keeps the grid recognisable but light
    step = 20
    pts, lines, idx = [], [], 0

    for j in range(0, DEM_H, step):
        row = P_w[j, :, :]
        for i in range(DEM_W):
            pts.append(row[i])
        for i in range(DEM_W):
            lines.append([idx + i, idx + (i + 1) % DEM_W])
        idx += DEM_W

    for i in range(0, DEM_W, step):
        col = P_w[:, i, :]
        for j in range(DEM_H):
            pts.append(col[j])
        for j in range(DEM_H - 1):
            lines.append([idx + j, idx + j + 1])
        idx += DEM_H

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.paint_uniform_color(C_GRID)
    return ls


def build_ray_data(params, R, center, poisson_mesh, n=N_RAYS_VIZ):
    """Cast n rays from SQ surface to rock. Returns (origins, hit_pts) valid pairs."""
    a      = np.array(params[:3])
    e1, e2 = params[3], params[4]

    rng   = np.random.default_rng(42)
    idxs  = rng.choice(DEM_W * DEM_H, n, replace=False)
    rows, cols = np.unravel_index(idxs, (DEM_H, DEM_W))

    eta_s   = -np.pi     + (cols + 0.5) * (2.0 * np.pi / DEM_W)
    omega_s = -0.5*np.pi + (rows + 0.5) * (np.pi  / DEM_H)

    P_can = utils.sq_point_canonical(eta_s, omega_s, a, e1, e2)
    N_can = utils.sq_normal_canonical(P_can, a, e1, e2)
    dots  = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[:, None] < 0.0, -N_can, N_can)

    P_w = (P_can @ R.T) + center
    N_w = (N_can @ R.T)
    N_w = N_w / (np.linalg.norm(N_w, axis=-1, keepdims=True) + 1e-12)

    rock_t = o3d.t.geometry.TriangleMesh.from_legacy(poisson_mesh)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(rock_t)

    rays_pos = np.concatenate([P_w.astype(np.float32),  N_w.astype(np.float32)], axis=1)
    rays_neg = np.concatenate([P_w.astype(np.float32), -N_w.astype(np.float32)], axis=1)
    t_pos = scene.cast_rays(
        o3d.core.Tensor(rays_pos, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()
    t_neg = scene.cast_rays(
        o3d.core.Tensor(rays_neg, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()

    hit_p, hit_n = np.isfinite(t_pos), np.isfinite(t_neg)
    valid = hit_p | hit_n

    t_signed = np.where(hit_p, t_pos, -t_neg)
    sign     = np.where(hit_p, 1.0, -1.0)
    hit_pts  = P_w + t_signed[:, None] * N_w * sign[:, None]

    return P_w[valid], hit_pts[valid]


def make_ray_lineset(origins, hit_pts, n_shown):
    n  = min(n_shown, len(origins))
    if n == 0:
        return None
    pts   = np.vstack([origins[:n], hit_pts[:n]])
    lines = [[i, i + n] for i in range(n)]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.paint_uniform_color(C_RAY)
    return ls


def make_outscribed(params, points_canonical):
    """
    Scale the SQ axes uniformly so that all canonical-frame points
    satisfy F(p) <= 1  (i.e. the rock is fully inside the SQ).

    For a SQ with axes [ax, ay, az] the inside-outside function is:
      F(x,y,z) = ((|x/ax|^(2/e2) + |y/ay|^(2/e2))^(e2/e1) + |z/az|^(2/e1))
    Scaling all axes by s multiplies F by s^(-2/e1), so we need:
      s = max_F^(e1/2)
    """
    ax, ay, az, e1, e2 = params
    x = points_canonical[:, 0]
    y = points_canonical[:, 1]
    z = points_canonical[:, 2]

    eps = 1e-12
    term_xy = (np.abs(x / (ax + eps)) ** (2.0 / e2) +
               np.abs(y / (ay + eps)) ** (2.0 / e2)) ** (e2 / e1)
    term_z  = np.abs(z / (az + eps)) ** (2.0 / e1)
    max_F   = float(np.max(term_xy + term_z))

    if max_F <= 1.0:
        return list(params)          # already contains all points

    s = max_F ** (e1 / 2.0)
    return [ax * s, ay * s, az * s, e1, e2]


def lerp_params(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return [a[k] + (b[k] - a[k]) * t for k in range(len(a))]


def smooth(t):
    """Smoothstep easing."""
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Load data ────────────────────────────────────────────────────────────
    print('Loading meshes and point clouds...')
    rock_mesh = o3d.io.read_triangle_mesh(OBJ_PATH, enable_post_processing=True)
    rock_mesh.compute_vertex_normals()
    if not rock_mesh.has_vertex_colors():
        rock_mesh.paint_uniform_color(C_ROCK_FALLBACK)

    pcd = o3d.io.read_point_cloud(PLY_PATH)
    pcd.paint_uniform_color(C_PC)

    poisson = o3d.io.read_triangle_mesh(POISSON_PATH)
    poisson.compute_vertex_normals()
    poisson.paint_uniform_color(C_RECON)

    recon_pcd = o3d.io.read_point_cloud(RECON_PTS_PATH)
    recon_pcd.paint_uniform_color(C_RECON)

    # Strip UVs before decimation — quadric decimation can't handle them
    # and produces a holey mesh if they're present
    rock_clean = o3d.geometry.TriangleMesh(rock_mesh)
    rock_clean.triangle_uvs   = o3d.utility.Vector2dVector([])
    rock_clean.triangle_material_ids = o3d.utility.IntVector([])
    rock_clean.remove_degenerate_triangles()
    rock_clean.remove_duplicated_triangles()
    rock_clean.remove_duplicated_vertices()
    rock_clean.remove_non_manifold_edges()
    display_rock = rock_clean.simplify_quadric_decimation(5_000)
    display_rock.compute_vertex_normals()
    display_rock.paint_uniform_color(C_ROCK_FALLBACK)

    display_poisson = poisson.simplify_quadric_decimation(5_000)
    display_poisson.compute_vertex_normals()
    display_poisson.paint_uniform_color(C_RECON)

    # ── Run recording EMS ────────────────────────────────────────────────────
    print('Running recording EMS fitter...')
    pts_all = np.asarray(pcd.points)
    n_pts   = len(pts_all)
    if n_pts > EMS_DOWNSAMPLE:
        idx     = np.random.default_rng(0).choice(n_pts, EMS_DOWNSAMPLE, replace=False)
        pcd_fit = pcd.select_by_index(idx.tolist())
    else:
        pcd_fit = pcd

    fitter      = RecordingEMSFitter(pcd_fit)
    final_model = fitter.fit(max_iters=60)
    history     = fitter.history
    R, center   = fitter.R_init, fitter.center

    final_params = [final_model.ax, final_model.ay, final_model.az,
                    final_model.e1, final_model.e2]

    print(f'  History events: {len(history)}')
    for ev in history:
        if ev['type'] == 's_step':
            n_better = sum(1 for c in ev['candidates'] if c['better'])
            print(f"    S-Step loop {ev['loop']}: {len(ev['candidates'])} candidates, "
                  f"{n_better} better, switched={ev['switched']}")

    # ── Outscribed SQ: scale up so all rock points are inside ────────────────
    outscribed_params = make_outscribed(final_params, fitter.points)
    scale = outscribed_params[0] / final_params[0]   # uniform scale factor
    print(f'  Best-fit axes:    {[f"{v:.4f}" for v in final_params[:3]]}')
    print(f'  Outscribed axes:  {[f"{v:.4f}" for v in outscribed_params[:3]]}  (×{scale:.3f})')

    # ── Pre-compute heavy geometry ───────────────────────────────────────────
    print('Pre-computing DEM grid...')
    dem_grid = build_dem_grid(outscribed_params, R, center)

    print('Pre-computing raycasting...')
    origins, hit_pts = build_ray_data(outscribed_params, R, center, poisson)
    print(f'  {len(origins)} valid rays')

    # ── Scene bounds (from original full-res rock mesh) ──────────────────────
    bb          = rock_mesh.get_axis_aligned_bounding_box()
    rock_center = np.array(bb.get_center())
    rock_extent = float(np.max(bb.get_extent()))
    cam_radius  = rock_extent * 1.6

    total_frames = (N_ROCK + N_SAMPLE + N_INIT + N_EMS + N_OUTSCRIBE +
                    N_GRID + N_RAYCAST + N_RECON + N_COMPARE)
    # 1.5 full rotations over the entire video
    deg_per_frame = (1.5 * 360.0) / total_frames

    frame_num = [0]

    def cam():
        return orbit_cam(frame_num[0] * deg_per_frame)

    def rf(geoms):
        """Shorthand: render_frame with shared scene bounds."""
        return render_frame(geoms, cam(), rock_center, cam_radius)

    # ── Video writer + progress bar ──────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (WIDTH, HEIGHT))

    _tty  = open('/dev/tty', 'w')
    _pbar = tqdm(total=total_frames, unit='frame', dynamic_ncols=True,
                 file=_tty,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    def write(frame):
        bgr = cv2.cvtColor(
            cv2.resize(frame.astype(np.uint8), (WIDTH, HEIGHT)),
            cv2.COLOR_RGB2BGR
        )
        writer.write(bgr)
        frame_num[0] += 1
        _pbar.update(1)

    # Reusable materials
    m_pc       = mkmat(C_PC,            pt_size=3.5)
    m_sq_solid = mkmat(C_SQ,            alpha=0.85)
    m_sq_wire  = mkmat(C_SQ_WIRE)
    m_grid     = mkmat(C_GRID)
    m_ray      = mkmat(C_RAY)
    m_hit      = mkmat(C_HIT,           pt_size=5.0)
    m_recon    = mkmat(C_RECON)
    m_recon_pc = mkmat(C_RECON,         pt_size=3.5)
    m_rock     = mkmat(C_ROCK_FALLBACK)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Original Rock Mesh
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[1/7] Rock mesh')
    for _ in range(N_ROCK):
        f = rf([('rock', display_rock, m_rock)])
        f = overlay(f, 'Apollo 15 Rock — Original Photogrammetry Mesh')
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Point Cloud Sampling
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[2/7] Point sampling')
    n_pcd = len(np.asarray(pcd.points))
    for i in range(N_SAMPLE):
        t    = smooth(i / N_SAMPLE)
        m_rf = mkmat(C_ROCK_FALLBACK, alpha=max(0.05, 1.0 - t))
        m_pf = mkmat(C_PC, alpha=t, pt_size=3.5)
        f    = rf([('rock', display_rock, m_rf), ('pc', pcd, m_pf)])
        f    = overlay(f,
                       'Step 1 — Surface Point Sampling',
                       body=f'{int(n_pcd * t):,} / {n_pcd:,} points sampled')
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — Initial SQ guess (OBB)
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[3/7] Initial SQ guess')
    init_ev     = history[0]
    init_params = init_ev['params']
    init_mesh   = build_sq_mesh(init_params, R, center, C_SQ_INIT)

    for i in range(N_INIT):
        t  = smooth(i / N_INIT)
        mi = mkmat(C_SQ_INIT, alpha=t * 0.75)
        f  = rf([('pc', pcd, m_pc), ('init', init_mesh, mi)])
        ax, ay, az, e1, e2 = init_params
        f  = overlay(f,
                     'Step 2 — EMS Superquadric Fitting: Initial Guess (OBB)',
                     body=(f'Initial: ax={ax:.4f}  ay={ay:.4f}  az={az:.4f}  '
                           f'e\u2081={e1:.2f}  e\u2082={e2:.2f}  '
                           f'(sphere — e\u2081=e\u2082=1)'))
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — EMS fitting: EM loops + S-step candidates
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[4/7] EMS fitting')

    em_events = [ev for ev in history if ev['type'] in ('em_done', 's_step')]
    n_events  = max(len(em_events), 1)
    base_f    = max(20, N_EMS // n_events)
    extra     = N_EMS - base_f * n_events
    ev_frames = [base_f + (1 if i < extra else 0) for i in range(n_events)]

    cur_params = init_params

    for ev_idx, (event, nf) in enumerate(zip(em_events, ev_frames)):

        if event['type'] == 'em_done':
            target_p = event['params']
            z_prob   = event['z_prob']
            loop_num = event['loop']
            loss     = event['loss']
            ax, ay, az, e1, e2 = target_p

            for i in range(nf):
                t      = smooth(i / max(nf - 1, 1))
                interp = lerp_params(cur_params, target_p, t)
                sq_i   = build_sq_mesh(interp, R, center, C_SQ)
                pc_w   = build_pc_weighted(fitter.points, z_prob, R, center)
                geoms  = [
                    ('pc', pc_w,  mkmat([1, 1, 1], pt_size=3.5)),
                    ('sq', sq_i,  mkmat(C_SQ, alpha=0.82)),
                ]
                f = rf(geoms)
                f = overlay(f,
                            f'Step 2 — EMS Loop {loop_num}: E-step + M-step Optimization',
                            body=(f'ax={ax:.4f}  ay={ay:.4f}  az={az:.4f}  '
                                  f'e\u2081={e1:.3f}  e\u2082={e2:.3f}  '
                                  f'loss={loss:.5f}   '
                                  f'\U0001f534 outlier  \U0001f535 inlier'))
                write(f)

            cur_params = target_p

        elif event['type'] == 's_step':
            cands    = event['candidates']
            loop_num = event['loop']
            cur_loss = event['current_loss']

            cur_sq      = build_sq_mesh(event['current_params'], R, center, C_SQ)
            cand_meshes = [build_sq_mesh(c['params'], R, center, CAND_COLORS[ci % 3])
                           for ci, c in enumerate(cands)]

            for i in range(nf):
                t       = i / max(nf - 1, 1)
                alpha_c = math.sin(t * math.pi) * 0.65 + 0.05

                geoms = [
                    ('pc',     pcd,    m_pc),
                    ('cur_sq', cur_sq, mkmat(C_SQ, alpha=0.85)),
                ]
                for ci, (cm, cand) in enumerate(zip(cand_meshes, cands)):
                    geoms.append((
                        f'cand_{ci}', cm,
                        mkmat(CAND_COLORS[ci % 3], alpha=alpha_c)
                    ))

                f = rf(geoms)

                labels = []
                for ci, cand in enumerate(cands):
                    col255 = tuple(int(v * 255) for v in CAND_COLORS[ci % 3])
                    ax2, ay2, az2, e12, e22 = cand['params']
                    delta  = cand['loss'] - cur_loss
                    status = '\u2713 SWITCH' if cand['better'] else '\u2717 reject'
                    text   = (f"Cand {ci+1}: loss={cand['loss']:.5f}  "
                              f"\u0394={delta:+.5f}  {status}")
                    labels.append((36, HEIGHT - 200 + ci * 42, text, col255))

                switched_txt = ''
                if event['switched'] and t > 0.75:
                    switched_txt = '\u2192 Switching to better shape and restarting EM'

                f = overlay(
                    f,
                    f'Step 2 — EMS S-Step (Loop {loop_num}): Shape Candidate Evaluation',
                    body=(f'Current loss: {cur_loss:.5f}   {switched_txt}'),
                    labels=labels,
                )
                write(f)

            if event['switched']:
                for c in cands:
                    if c['better']:
                        cur_params = c['params']
                        break

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4b — Expand best-fit SQ → outscribed (contains all rock points)
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[4b/7] Outscribed expansion')
    ax_b, ay_b, az_b = final_params[0], final_params[1], final_params[2]
    ax_o, ay_o, az_o = outscribed_params[0], outscribed_params[1], outscribed_params[2]

    for i in range(N_OUTSCRIBE):
        t       = smooth(i / max(N_OUTSCRIBE - 1, 1))
        interp  = lerp_params(final_params, outscribed_params, t)
        sq_exp  = build_sq_mesh(interp, R, center, C_SQ)
        geoms   = [
            ('pc',  pcd,    m_pc),
            ('sq',  sq_exp, mkmat(C_SQ, alpha=0.85)),
        ]
        f = rf(geoms)
        f = overlay(
            f,
            'Step 2 — Expanding SQ to Outscribed (Circumscribed) Shape',
            body=(f'Scaling axes ×{1.0 + (scale - 1.0) * t:.3f}  '
                  f'→ ax={ax_b + (ax_o - ax_b)*t:.4f}  '
                  f'ay={ay_b + (ay_o - ay_b)*t:.4f}  '
                  f'az={az_b + (az_o - az_b)*t:.4f}  '
                  f'— all rock points now inside SQ'),
        )
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — DEM Grid (720×360)
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[5/7] DEM grid')
    final_sq = build_sq_mesh(outscribed_params, R, center, C_SQ)

    for i in range(N_GRID):
        t     = smooth(i / N_GRID)
        m_gf  = mkmat(C_GRID, alpha=t)
        m_sqf = mkmat(C_SQ,   alpha=0.35 + 0.15 * t)
        geoms = [
            ('pc',   pcd,      m_pc),
            ('sq',   final_sq, m_sqf),
            ('grid', dem_grid, m_gf),
        ]
        f = rf(geoms)
        n_shown = int(DEM_W * DEM_H * t)
        f = overlay(f,
                    'Step 3 — Gridding the Superquadric (720 \u00d7 360)',
                    body=f'{n_shown:,} / {DEM_W * DEM_H:,} surface cells  '
                         f'— each cell will cast a ray to the rock surface')
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Raycasting
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[6/7] Raycasting')
    n_valid = len(origins)

    for i in range(N_RAYCAST):
        t       = i / N_RAYCAST
        n_shown = int(n_valid * smooth(t))
        geoms   = [
            ('sq',   final_sq, mkmat(C_SQ,   alpha=0.25)),
            ('grid', dem_grid, mkmat(C_GRID,  alpha=0.4)),
        ]
        if n_shown > 0:
            ray_ls = make_ray_lineset(origins, hit_pts, n_shown)
            if ray_ls:
                geoms.append(('rays', ray_ls, m_ray))
            hp = o3d.geometry.PointCloud()
            hp.points = o3d.utility.Vector3dVector(hit_pts[:n_shown])
            hp.paint_uniform_color(C_HIT)
            geoms.append(('hits', hp, m_hit))

        f = rf(geoms)
        f = overlay(f,
                    'Step 4 — Raycasting: SQ Grid \u2192 Rock Surface',
                    body=f'{n_shown:,} / {n_valid:,} rays cast along surface normals')
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Reconstruction
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[7a/7] Reconstruction')
    for i in range(N_RECON):
        t = i / N_RECON
        if t < 0.5:
            t2      = smooth(t / 0.5)
            n_shown = int(len(np.asarray(recon_pcd.points)) * t2)
            sub_pcd = recon_pcd.select_by_index(list(range(n_shown)))
            geoms   = [('rpts', sub_pcd, m_recon_pc)]
            body    = f'Displaced points on rock surface: {n_shown:,}'
        else:
            t2    = smooth((t - 0.5) / 0.5)
            geoms = [
                ('rpts',  recon_pcd,      mkmat(C_RECON, pt_size=3.5, alpha=1.0 - t2)),
                ('pmesh', display_poisson, mkmat(C_RECON, alpha=t2)),
            ]
            body = 'Poisson surface reconstruction...'

        f = rf(geoms)
        f = overlay(f, 'Step 5 — Poisson Surface Reconstruction', body=body)
        write(f)

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 8 — Split Comparison
    # ─────────────────────────────────────────────────────────────────────────
    _pbar.set_description('[7b/7] Split screen')

    for i in range(N_COMPARE):
        combined = render_split_frame(
            [('orig',  display_rock,    m_rock)],
            [('recon', display_poisson, m_recon)],
            cam(), rock_center, cam_radius,
        )

        # Dividing line between panels
        img  = Image.fromarray(combined)
        draw = ImageDraw.Draw(img)
        draw.line([(WIDTH // 2, 80), (WIDTH // 2, HEIGHT - 58)],
                  fill=(255, 255, 255), width=2)
        combined = np.array(img)

        combined = overlay(
            combined,
            'Step 6 — Original vs Reconstructed (SQ + DEM 720\u00d7360)',
            labels=[
                (WIDTH // 4 - 130,     HEIGHT - 52,
                 'Original Mesh',          (230, 200, 160)),
                (3 * WIDTH // 4 - 160, HEIGHT - 52,
                 'SQ+DEM Reconstructed',   (160, 230, 160)),
            ],
        )
        write(combined)

    # ── Finalise ─────────────────────────────────────────────────────────────
    _pbar.close()
    _tty.close()
    writer.release()

    # Re-encode with H.264 for broad compatibility
    h264 = OUT_VIDEO.replace('.mp4', '_h264.mp4')
    os.system(f'ffmpeg -y -i "{OUT_VIDEO}" -vcodec libx264 -crf 18 '
              f'-pix_fmt yuv420p "{h264}" -loglevel error')
    if os.path.exists(h264):
        os.replace(h264, OUT_VIDEO)

    print(f'\n✓ Done! {frame_num[0]} frames written.')
    print(f'✓ Video saved to: {OUT_VIDEO}')


if __name__ == '__main__':
    main()
