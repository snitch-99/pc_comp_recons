"""
Step_4_reconstruction.py
========================
Re-projects each cluster's DEM back to a 3D point cloud and runs Poisson
reconstruction to produce a compact, faithful rock mesh.

Input:  sq_fit_Cluster_N.txt + dem_Cluster_N.npy + mask_Cluster_N.npy
Output: dem_recon_Cluster_N.ply
"""

import glob
import os
import sys

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils
from utils import sq_point_canonical, sq_normal_canonical, parse_sq_fit_txt

from config import (
    MAP_DIR               as IO_OP_DIR,
    DEM_W                 as W,
    DEM_H                 as H,
    DEM_RECON_POISSON_DEPTH as POISSON_DEPTH,
    DEM_RECON_NORMAL_KNN    as NORMAL_KNN,
)


# ---------------------------------------------------------------------------
# Per-cluster reconstruction
# ---------------------------------------------------------------------------

def reconstruct_from_dem(sq_params, sq_center, sq_R,
                         dem_path, mask_path, out_mesh_path):
    """
    Load DEM + mask, displace SQ grid points by their DEM depth along the
    inward normal, then run Poisson on the resulting point cloud.
    Returns the final mesh.
    """
    a      = np.array(sq_params[:3])
    e1, e2 = sq_params[3], sq_params[4]

    D = np.load(dem_path).astype(np.float64)
    M = np.load(mask_path).astype(np.uint8)
    valid   = M > 0
    n_valid = int(valid.sum())
    print(f"  [INFO] Valid DEM cells: {n_valid}/{H*W}")
    if n_valid < 500:
        print("  [WARN] Very few valid points — Poisson may be unstable.")

    eta   = -np.pi     + (np.arange(W) + 0.5) * (2*np.pi/W)
    omega = -0.5*np.pi + (np.arange(H) + 0.5) * (np.pi/H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = sq_point_canonical(ETA, OMEGA, a, e1, e2)
    N_can = sq_normal_canonical(P_can, a, e1, e2)
    dots  = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[..., None] < 0.0, -N_can, N_can)

    P_w = (P_can @ sq_R.T) + sq_center
    N_w = N_can @ sq_R.T
    N_w = N_w / (np.linalg.norm(N_w, axis=-1, keepdims=True) + 1e-12)

    P_recon = P_w - D[..., None] * N_w    # (H, W, 3)
    pts     = P_recon[valid]               # (N, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"  [INFO] DEM cloud size: {len(pts):,}")

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=NORMAL_KNN)

    print(f"  [INFO] Poisson depth={POISSON_DEPTH} ...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    mesh.compute_vertex_normals()

    bbox = pcd.get_axis_aligned_bounding_box().scale(1.05, pcd.get_center())
    mesh = mesh.crop(bbox)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(out_mesh_path, mesh):
        raise RuntimeError(f"Failed to write: {out_mesh_path}")
    print(f"  [INFO] Saved: {out_mesh_path}")
    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))
    if not cluster_files:
        print(f"[ERROR] No Cluster_*.ply in {IO_OP_DIR}"); sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s).\n")
    vis_data = []

    for i, _ in enumerate(cluster_files, start=1):
        name          = f"Cluster_{i}"
        txt_path      = os.path.join(IO_OP_DIR, f"sq_fit_{name}.txt")
        dem_path      = os.path.join(IO_OP_DIR, f"dem_{name}.npy")
        mask_path     = os.path.join(IO_OP_DIR, f"mask_{name}.npy")
        poisson_path  = os.path.join(IO_OP_DIR, f"poisson_{name}.ply")
        out_mesh_path = os.path.join(IO_OP_DIR, f"dem_recon_{name}.ply")

        print(f"{'═'*50}\n  [{i}/{len(cluster_files)}]  {name}\n{'═'*50}")

        missing = [p for p in [txt_path, dem_path, mask_path, poisson_path]
                   if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] Missing: {[os.path.basename(m) for m in missing]}\n"); continue

        sq = parse_sq_fit_txt(txt_path)
        try:
            dem_mesh = reconstruct_from_dem(
                sq_params=sq["params"], sq_center=sq["center"], sq_R=sq["R"],
                dem_path=dem_path, mask_path=mask_path,
                out_mesh_path=out_mesh_path,
            )
        except Exception as e:
            print(f"  [ERROR] {e}\n"); continue

        poisson_mesh = o3d.io.read_triangle_mesh(poisson_path)
        poisson_mesh.compute_vertex_normals()
        poisson_mesh.paint_uniform_color([0.9, 0.35, 0.2])
        dem_mesh.paint_uniform_color([0.2, 0.75, 0.3])
        vis_data.append((poisson_mesh, dem_mesh, name))
        print()

    if not vis_data:
        print("[ERROR] No meshes produced."); sys.exit(1)

    # ── Visualize ────────────────────────────────────────────────────────────
    all_extents = [r[0].get_axis_aligned_bounding_box().get_extent() for r in vis_data]
    GAP = max(float(np.max(e)) for e in all_extents) * 1.5
    WIN_W, WIN_H, PAD = 1500, 620, 25
    if not os.environ.get("PC_HEADLESS"):
        visualizers = []
        for idx, (poisson_mesh, dem_mesh, name) in enumerate(vis_data):
            top    = PAD + idx * (WIN_H + PAD*3)
            bb     = poisson_mesh.get_axis_aligned_bounding_box()
            bb_min = np.asarray(bb.get_min_bound())
            bb_max = np.asarray(bb.get_max_bound())
            extent = bb.get_extent()

            dem_bb   = dem_mesh.get_axis_aligned_bounding_box()
            dem_min  = np.asarray(dem_bb.get_min_bound())
            axis_sz  = float(np.max(extent)) * 0.15
            grid_sp  = max(0.05, round(float(np.max(extent)) / 6, 2))
            y_floor  = float(min(bb_min[1], dem_min[1]))
            dx       = [-GAP, 0.0, GAP]

            panel1 = [utils.viz_shift(poisson_mesh, dx[0])] + \
                     utils.viz_make_grid_panel(dx[0], bb_min, bb_max, y_floor, axis_sz, grid_sp)
            panel2 = [utils.viz_shift(dem_mesh, dx[1])] + \
                     utils.viz_make_grid_panel(dx[1], dem_min, np.asarray(dem_bb.get_max_bound()), y_floor, axis_sz, grid_sp)
            panel3 = [utils.viz_shift(poisson_mesh, dx[2]), utils.viz_shift(dem_mesh, dx[2])] + \
                     utils.viz_make_grid_panel(dx[2], bb_min, bb_max, y_floor, axis_sz, grid_sp)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{name}  |  Left: Poisson   Centre: DEM   Right: Overlay",
                width=WIN_W, height=WIN_H, left=PAD, top=top,
            )
            for g in panel1 + panel2 + panel3:
                vis.add_geometry(g)
            opt = vis.get_render_option(); opt.mesh_show_back_face = True
            vis.poll_events(); vis.update_renderer()
            visualizers.append(vis)

        utils.viz_event_loop(visualizers)
    print("\n✓ All DEM-based meshes saved!")
