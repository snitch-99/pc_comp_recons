#!/usr/bin/env python3
"""
generate_comparison_video.py
============================
10-second 360° turntable video of the 4-panel compression comparison.
Each mesh orbits a full revolution. Output: comparison_video.mp4
"""

import os
import sys
import numpy as np

import contextlib

# Suppress Open3D's verbose EGL/Filament noise
os.environ.setdefault('OPEN3D_LOGGING_LEVEL', 'error')

@contextlib.contextmanager
def suppress_filament():
    """Redirect both fd 1 (stdout) and fd 2 (stderr) to /dev/null.
    Filament prints to stdout; Open3D INFO goes to stderr. Kills both."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_out, old_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_out, 1);  os.close(old_out)
        os.dup2(old_err, 2);  os.close(old_err)
import open3d as o3d
import cv2
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_180   = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1/Data_for_Claude/180_90'
DATA_720   = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1/Data_for_Claude/720_360'
ROCK_OBJ   = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1/rock.obj'
DRACO_ENC  = '/home/kanav/workspaces/draco/build/draco_encoder'
DRACO_DEC  = '/home/kanav/workspaces/draco/build/draco_decoder'
WORK_DIR   = '/tmp/draco_comparison'
OUT_VIDEO  = '/home/kanav/workspaces/pc_comp_recons/comparison_video.mp4'
os.makedirs(WORK_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# VIDEO CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FPS        = 24
DURATION   = 10          # seconds
N_FRAMES   = FPS * DURATION   # 240
PANEL_W    = 720
PANEL_H    = 480
C_SKIN     = [0.87, 0.72, 0.53, 1.0]
BG         = [0.08, 0.08, 0.13, 1.0]

# Camera orbit
CAM_ELEV   = 20.0        # degrees above equator
CAM_DIST   = 2.4         # multiplier of mesh extent

# ─────────────────────────────────────────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────────────────────────────────────────
_BOLD = ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
         '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf']
_REG  = [p.replace('-Bold','').replace('Bold','') for p in _BOLD]

def _font(paths, size):
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

F_TITLE = _font(_BOLD, 28)
F_SIZE  = _font(_REG,  20)


# ─────────────────────────────────────────────────────────────────────────────
# RENDERER  — one per panel, geometry added once
# ─────────────────────────────────────────────────────────────────────────────
def render_panel(mesh, azim_deg, panel_w=PANEL_W, panel_h=PANEL_H):
    """Fresh renderer per call — avoids Filament multi-renderer crash."""
    bb     = mesh.get_axis_aligned_bounding_box()
    center = np.array(bb.get_center())
    extent = float(np.max(bb.get_extent()))
    dist   = extent * CAM_DIST

    az  = np.radians(azim_deg)
    el  = np.radians(CAM_ELEV)
    eye = center + dist * np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])

    r = o3d.visualization.rendering.OffscreenRenderer(panel_w, panel_h)
    r.scene.set_background(BG)
    r.scene.scene.set_sun_light([0.5, -0.8, -0.4], [1.0, 0.97, 0.92], 100000)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.enable_indirect_light(True)
    r.scene.scene.set_indirect_light_intensity(45000)

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader     = 'defaultLit'
    mat.base_color = C_SKIN
    r.scene.add_geometry('mesh', mesh, mat)
    r.setup_camera(50.0, center.tolist(), eye.tolist(), [0.0, 1.0, 0.0])

    img = np.asarray(r.render_to_image())
    del r
    return img


def add_overlay(img_arr, title, size_label, size_bytes):
    pil  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil, 'RGBA')
    w, h = pil.size

    draw.rectangle([(0, 0), (w, 48)], fill=(5, 5, 20, 210))
    draw.text((14, 8),  title,      font=F_TITLE, fill=(255, 215, 70, 255))

    draw.rectangle([(0, h - 58), (w, h)], fill=(5, 5, 20, 200))
    draw.text((14, h - 52), size_label,
              font=F_SIZE, fill=(200, 200, 215, 255))
    draw.text((14, h - 28), f'{size_bytes/1024:.1f} KB  ({size_bytes:,} B)',
              font=F_SIZE, fill=(160, 220, 160, 255))
    return np.array(pil)[:, :, :3]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import subprocess

    # ── Load / build meshes ──────────────────────────────────────────────────
    print('Loading meshes...')

    # Draco best compression
    drc_c = os.path.join(WORK_DIR, 'draco_comp.drc')
    ply_c = os.path.join(WORK_DIR, 'draco_comp.ply')
    if not os.path.exists(ply_c):
        subprocess.run([DRACO_ENC, '-i', ROCK_OBJ, '-o', drc_c,
                        '-cl', '10', '-qp', '4'], capture_output=True, check=True)
        subprocess.run([DRACO_DEC, '-i', drc_c, '-o', ply_c],
                       capture_output=True, check=True)
    m1 = o3d.io.read_triangle_mesh(ply_c)
    b1 = os.path.getsize(drc_c)

    # Ours best compression 180×90
    m2 = o3d.io.read_triangle_mesh(os.path.join(DATA_180, 'dem_recon_Cluster_1.ply'))
    b2 = 68 + 180 * 90 * 4

    # Draco best quality
    drc_q = os.path.join(WORK_DIR, 'draco_qual.drc')
    ply_q = os.path.join(WORK_DIR, 'draco_qual.ply')
    if not os.path.exists(ply_q):
        subprocess.run([DRACO_ENC, '-i', ROCK_OBJ, '-o', drc_q,
                        '-cl', '0', '-qp', '16'], capture_output=True, check=True)
        subprocess.run([DRACO_DEC, '-i', drc_q, '-o', ply_q],
                       capture_output=True, check=True)
    m3 = o3d.io.read_triangle_mesh(ply_q)
    b3 = os.path.getsize(drc_q)

    # Ours best quality 720×360
    m4 = o3d.io.read_triangle_mesh(os.path.join(DATA_720, 'dem_recon_Cluster_1.ply'))
    b4 = 68 + 720 * 360 * 4

    panels = [
        (m1, 'Draco  —  Best Compression', 'cl=10  qp=4',    b1),
        (m2, 'Ours  —  Best Compression',  'DEM 180×90',     b2),
        (m3, 'Draco  —  Best Quality',     'cl=0  qp=16',    b3),
        (m4, 'Ours  —  Best Quality',      'DEM 720×360',    b4),
    ]

    # Prep meshes once: clear colors, orient normals
    for m, _, _, _ in panels:
        m.vertex_colors = o3d.utility.Vector3dVector([])
        m.orient_triangles()
        m.compute_vertex_normals()

    # ── Video writer ─────────────────────────────────────────────────────────
    total_w = PANEL_W * 2
    total_h = PANEL_H * 2
    fourcc  = cv2.VideoWriter_fourcc(*'mp4v')
    writer  = cv2.VideoWriter(OUT_VIDEO, fourcc, FPS, (total_w, total_h))

    tty = open('/dev/tty', 'w')
    tty.write(f'Rendering {N_FRAMES} frames ({DURATION}s @ {FPS}fps)...\n')
    tty.flush()
    with tqdm(total=N_FRAMES, unit='frame', dynamic_ncols=True, file=tty,
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for f in range(N_FRAMES):
            azim = (f / N_FRAMES) * 360.0

            imgs = []
            for mesh, title, label, nb in panels:
                with suppress_filament():
                    raw = render_panel(mesh, azim)
                imgs.append(add_overlay(raw, title, label, nb))

            row1 = np.hstack([imgs[0], imgs[1]])
            row2 = np.hstack([imgs[2], imgs[3]])
            grid = np.vstack([row1, row2])

            grid[PANEL_H-1 : PANEL_H+1, :] = 40
            grid[:, PANEL_W-1 : PANEL_W+1] = 40

            bgr = cv2.cvtColor(grid.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            pbar.update(1)

    writer.release()

    # Re-encode with H.264 for broad compatibility
    h264 = OUT_VIDEO.replace('.mp4', '_h264.mp4')
    os.system(f'ffmpeg -y -i "{OUT_VIDEO}" -vcodec libx264 -crf 18 '
              f'-pix_fmt yuv420p "{h264}" -loglevel error')
    if os.path.exists(h264):
        os.replace(h264, OUT_VIDEO)

    print(f'\n✓ Video saved → {OUT_VIDEO}')


if __name__ == '__main__':
    main()
