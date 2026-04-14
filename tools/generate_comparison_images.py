#!/usr/bin/env python3
"""
generate_comparison_images.py
==============================
Generates a 4-panel comparison image:
  1. Draco best compression  (cl=10, qp=4  → 141 KB)
  2. Ours best compression   (180×90  DEM  →  63 KB)
  3. Draco best quality      (cl=10, qp=16 → ~1.4 MB)
  4. Ours best quality       (360×180 DEM  → 253 KB)

Both our grid sizes stay below the corresponding Draco budget while
achieving competitive reconstruction quality.

Output: /home/kanav/workspaces/pc_comp_recons/comparison_4panel.png
        (+ individual PNGs per panel)
"""

import os
import sys
import subprocess
import warnings

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from utils import sq_point_canonical, sq_normal_canonical, parse_sq_fit_txt

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
ROCK_DIR       = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1'
POISSON_PATH   = os.path.join(ROCK_DIR, 'poisson_Cluster_1.ply')
SQ_TXT         = os.path.join(ROCK_DIR, 'sq_fit_Cluster_1.txt')
DRACO_ENC      = '/home/kanav/workspaces/draco/build/draco_encoder'
DRACO_DEC      = '/home/kanav/workspaces/draco/build/draco_decoder'
ROCK_OBJ       = os.path.join(ROCK_DIR, 'rock.obj')
OUT_DIR        = '/home/kanav/workspaces/pc_comp_recons'
WORK_DIR       = '/tmp/draco_comparison'
os.makedirs(WORK_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# RENDER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
PANEL_W = 960
PANEL_H = 540

# Camera: eye offset relative to mesh center (will be scaled by extent)
# Rotated slightly to show depth
EYE_OFFSET = np.array([-1.2, -1.6, 0.9])   # tunable

C_SKIN  = [0.87, 0.72, 0.53, 1.0]   # warm skin tone — all panels
BG      = [0.07, 0.07, 0.12, 1.0]   # dark background

# ─────────────────────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────────────────────
_FONT_BOLD = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
]
_FONT_REG = [p.replace('-Bold', '').replace('Bold', '') for p in _FONT_BOLD]

def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE = _font(_FONT_BOLD, 38)
F_BODY  = _font(_FONT_REG,  24)
F_SIZE  = _font(_FONT_REG,  21)

# ─────────────────────────────────────────────────────────────────────────────
# DEM GENERATION  (parametric — takes explicit W, H)
# ─────────────────────────────────────────────────────────────────────────────
def build_dem(sq_params, sq_center, sq_R, poisson_path, W, H):
    """
    Circumscribed mode (matches Step_3 with EMS_CIRCUMSCRIBE=True):
    Shoot rays INWARD (-N_w) from the circumscribed SQ surface against the
    Poisson mesh.  Returns (D_hw, M_hw).
    """
    a      = np.array(sq_params[:3])
    e1, e2 = sq_params[3], sq_params[4]

    mesh_legacy = o3d.io.read_triangle_mesh(poisson_path)
    mesh_legacy.compute_triangle_normals()
    rock_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(rock_t)

    eta   = -np.pi     + (np.arange(W) + 0.5) * (2.0 * np.pi / W)
    omega = -0.5*np.pi + (np.arange(H) + 0.5) * (np.pi  / H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = sq_point_canonical(ETA, OMEGA, a, e1, e2)
    N_can = sq_normal_canonical(P_can, a, e1, e2)
    dots  = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[..., None] < 0.0, -N_can, N_can)

    P_w = (P_can @ sq_R.T) + sq_center
    N_w = N_can @ sq_R.T
    N_w = N_w / (np.linalg.norm(N_w, axis=-1, keepdims=True) + 1e-12)

    # Circumscribed: SQ is outside the rock → shoot inward (-N_w)
    origins = P_w.reshape(-1, 3).astype(np.float32)
    rays    = np.concatenate([origins, -N_w.reshape(-1, 3).astype(np.float32)], axis=1)
    t       = scene.cast_rays(
        o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))['t_hit'].numpy()
    hit = np.isfinite(t)
    D   = np.where(hit, t, 0.0).astype(np.float32).reshape(H, W)
    M   = hit.astype(np.uint8).reshape(H, W)
    print(f'    DEM {W}×{H}: {int(M.sum()):,}/{W*H:,} valid cells ({100*M.mean():.1f}%)')
    return D, M


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION  (parametric — takes explicit W, H, D, M)
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_from_dem(sq_params, sq_center, sq_R, D, M, W, H,
                         out_path, poisson_depth=9, normal_knn=30):
    """Back-project DEM cells and Poisson reconstruct."""
    a      = np.array(sq_params[:3])
    e1, e2 = sq_params[3], sq_params[4]

    eta   = -np.pi     + (np.arange(W) + 0.5) * (2.0 * np.pi / W)
    omega = -0.5*np.pi + (np.arange(H) + 0.5) * (np.pi  / H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = sq_point_canonical(ETA, OMEGA, a, e1, e2)
    N_can = sq_normal_canonical(P_can, a, e1, e2)
    dots  = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[..., None] < 0.0, -N_can, N_can)

    P_w = (P_can @ sq_R.T) + sq_center
    N_w = N_can @ sq_R.T
    N_w = N_w / (np.linalg.norm(N_w, axis=-1, keepdims=True) + 1e-12)

    valid   = M > 0
    P_recon = P_w - D[..., None] * N_w
    pts     = P_recon[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=normal_knn)

    print(f'    Poisson depth={poisson_depth} on {len(pts):,} pts ...')
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth)
    mesh.compute_vertex_normals()
    bbox = pcd.get_axis_aligned_bounding_box().scale(1.05, pcd.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_path, mesh)
    print(f'    Saved → {out_path}')
    return mesh


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING  —  single-shot Open3D OffscreenRenderer per panel
# ─────────────────────────────────────────────────────────────────────────────
def render_mesh_panel(mesh, color, scene_center,
                      title, size_label, size_bytes,
                      panel_w=PANEL_W, panel_h=PANEL_H):
    """
    Render one mesh with Open3D OffscreenRenderer (single shot — no loop crash).
    Returns HxWx3 uint8 RGB image with text overlay.
    """
    # Remove any stored vertex colors — they override mat.base_color and
    # the Poisson mesh always stores zeros (black), making the panel black.
    mesh.vertex_colors = o3d.utility.Vector3dVector([])
    mesh.orient_triangles()
    mesh.compute_vertex_normals()

    # Fresh renderer every call
    r = o3d.visualization.rendering.OffscreenRenderer(panel_w, panel_h)
    r.scene.set_background(BG)
    r.scene.scene.set_sun_light([0.4, -0.8, -0.5], [1.0, 0.97, 0.92], 120000)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.enable_indirect_light(True)
    r.scene.scene.set_indirect_light_intensity(55000)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader     = 'defaultLit'
    mat.base_color = color
    r.scene.add_geometry('mesh', mesh, mat)

    # Camera: orbit around scene center
    bb     = mesh.get_axis_aligned_bounding_box()
    center = np.array(bb.get_center())
    extent = float(np.max(bb.get_extent()))
    eye    = center + EYE_OFFSET * extent
    r.setup_camera(55.0, center.tolist(), eye.tolist(), [0.0, 0.0, 1.0])

    img = np.asarray(r.render_to_image())   # HxWx3 uint8 RGB
    del r   # release Filament resources immediately

    # Text overlay
    pil  = Image.fromarray(img)
    draw = ImageDraw.Draw(pil, 'RGBA')

    draw.rectangle([(0, 0), (panel_w, 60)], fill=(5, 5, 20, 210))
    draw.text((20, 10), title, font=F_TITLE, fill=(255, 215, 70, 255))

    draw.rectangle([(0, panel_h - 80), (panel_w, panel_h)], fill=(5, 5, 20, 200))
    draw.text((20, panel_h - 72), size_label, font=F_BODY,  fill=(200, 200, 215, 255))
    kb = size_bytes / 1024
    draw.text((20, panel_h - 44), f'{kb:.1f} KB  ({size_bytes:,} bytes)',
              font=F_SIZE, fill=(160, 220, 160, 255))

    return np.array(pil)[:, :, :3]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    DATA_180  = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1/Data_for_Claude/180_90'
    DATA_720  = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1/Data_for_Claude/720_360'

    # ── 1. Draco best compression ────────────────────────────────────────────
    print('[1/4] Draco best compression (cl=10, qp=4)...')
    drc_comp = os.path.join(WORK_DIR, 'draco_comp.drc')
    ply_comp = os.path.join(WORK_DIR, 'draco_comp.ply')
    subprocess.run([DRACO_ENC, '-i', ROCK_OBJ, '-o', drc_comp,
                    '-cl', '10', '-qp', '4'], capture_output=True, check=True)
    subprocess.run([DRACO_DEC, '-i', drc_comp, '-o', ply_comp],
                   capture_output=True, check=True)
    draco_comp_bytes = os.path.getsize(drc_comp)
    draco_comp_mesh  = o3d.io.read_triangle_mesh(ply_comp)
    print(f'  {draco_comp_bytes/1024:.1f} KB  |  '
          f'{len(draco_comp_mesh.vertices):,} verts')

    # ── 2. Ours best compression: 180×90 ────────────────────────────────────
    print('[2/4] Ours best compression (180×90)...')
    W_comp, H_comp = 180, 90
    our_comp_bytes  = 68 + W_comp * H_comp * 4   # SQ(68B) + DEM float32
    our_comp_mesh   = o3d.io.read_triangle_mesh(
        os.path.join(DATA_180, 'dem_recon_Cluster_1.ply'))
    print(f'  {our_comp_bytes/1024:.1f} KB  |  '
          f'{len(our_comp_mesh.vertices):,} verts')

    # ── 3. Draco best quality ────────────────────────────────────────────────
    print('[3/4] Draco best quality (cl=0, qp=16)...')
    drc_qual = os.path.join(WORK_DIR, 'draco_qual.drc')
    ply_qual = os.path.join(WORK_DIR, 'draco_qual.ply')
    subprocess.run([DRACO_ENC, '-i', ROCK_OBJ, '-o', drc_qual,
                    '-cl', '0', '-qp', '16'], capture_output=True, check=True)
    subprocess.run([DRACO_DEC, '-i', drc_qual, '-o', ply_qual],
                   capture_output=True, check=True)
    draco_qual_bytes = os.path.getsize(drc_qual)
    draco_qual_mesh  = o3d.io.read_triangle_mesh(ply_qual)
    print(f'  {draco_qual_bytes/1024:.1f} KB  |  '
          f'{len(draco_qual_mesh.vertices):,} verts')

    # ── 4. Ours best quality: 720×360 ───────────────────────────────────────
    print('[4/4] Ours best quality (720×360)...')
    W_qual, H_qual = 720, 360
    our_qual_bytes  = 68 + W_qual * H_qual * 4
    our_qual_mesh   = o3d.io.read_triangle_mesh(
        os.path.join(DATA_720, 'dem_recon_Cluster_1.ply'))
    print(f'  {our_qual_bytes/1024:.1f} KB  |  '
          f'{len(our_qual_mesh.vertices):,} verts')

    # ── Render 4 panels ──────────────────────────────────────────────────────
    bb = draco_comp_mesh.get_axis_aligned_bounding_box()
    scene_center = np.array(bb.get_center())

    print('\nRendering panels...')
    panels = [
        (draco_comp_mesh, C_SKIN, 'Draco  —  Best Compression',
         'cl=10  qp=4', draco_comp_bytes),
        (our_comp_mesh,   C_SKIN, 'Ours  —  Best Compression',
         f'DEM {W_comp}\u00d7{H_comp}', our_comp_bytes),
        (draco_qual_mesh, C_SKIN, 'Draco  —  Best Quality',
         'cl=0  qp=16', draco_qual_bytes),
        (our_qual_mesh,   C_SKIN, 'Ours  —  Best Quality',
         f'DEM {W_qual}\u00d7{H_qual}', our_qual_bytes),
    ]

    imgs = []
    for i, (mesh, color, title, label, nbytes) in enumerate(panels, 1):
        print(f'  Panel {i}/4: {title}...')
        img = render_mesh_panel(mesh, color, scene_center,
                                title, label, nbytes)
        path = os.path.join(OUT_DIR, f'comparison_panel_{i}.png')
        Image.fromarray(img).save(path)
        print(f'    → {path}')
        imgs.append(img)

    # ── Assemble 2×2 grid ────────────────────────────────────────────────────
    print('\nAssembling 2×2 comparison image...')
    row1 = np.hstack([imgs[0], imgs[1]])
    row2 = np.hstack([imgs[2], imgs[3]])
    grid = np.vstack([row1, row2])

    # Dividing lines
    pil  = Image.fromarray(grid)
    draw = ImageDraw.Draw(pil)
    # Vertical centre line
    draw.line([(PANEL_W, 0), (PANEL_W, PANEL_H*2)], fill=(200,200,200), width=3)
    # Horizontal centre line
    draw.line([(0, PANEL_H), (PANEL_W*2, PANEL_H)], fill=(200,200,200), width=3)
    # Top label bar
    draw.rectangle([(0,0),(PANEL_W*2, 52)], fill=(5,5,20,220))
    draw.text((PANEL_W//2 - 280, 8),
              'Draco Compression', font=F_TITLE, fill=(255,200,100,255))
    draw.text((PANEL_W + PANEL_W//2 - 220, 8),
              'Our Method (SQ+DEM)', font=F_TITLE, fill=(255,200,100,255))

    out_path = os.path.join(OUT_DIR, 'comparison_4panel.png')
    pil.save(out_path, dpi=(150, 150))
    print(f'\n✓ 4-panel image → {out_path}')

    # Summary table
    print('\n' + '='*65)
    print(f'  {"Method":<30} {"Size (KB)":>10}  {"Ratio":>8}')
    print('-'*65)
    print(f'  {"Draco best compression":<30} {draco_comp_bytes/1024:>10.1f}  {"1.00x":>8}')
    print(f'  {"Ours 180×90 DEM":<30} {our_comp_bytes/1024:>10.1f}  {our_comp_bytes/draco_comp_bytes:>7.2f}x')
    print(f'  {"Draco best quality":<30} {draco_qual_bytes/1024:>10.1f}  {"1.00x":>8}')
    print(f'  {"Ours 720×360 DEM":<30} {our_qual_bytes/1024:>10.1f}  {our_qual_bytes/draco_qual_bytes:>7.2f}x')
    print('='*65)


if __name__ == '__main__':
    main()
