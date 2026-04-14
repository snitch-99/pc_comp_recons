"""
Step_3_dem_generation.py
========================
Raycasts from the Superquadric surface to the Poisson mesh to build a
per-cell depth map (DEM) that compactly encodes the rock surface.

Input:  sq_fit_Cluster_N.txt + poisson_Cluster_N.ply
Output: dem_Cluster_N.npy | mask_Cluster_N.npy | recon_pts_Cluster_N.ply
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
    MAP_DIR              as IO_OP_DIR,
    DEM_W                as W,
    DEM_H                as H,
    USE_LOCAL_SUPPORT_FILTER,
    LOCAL_SUPPORT_K,
    SAVE_RECON_POINTS_PLY,
    EMS_CIRCUMSCRIBE,
)


# ---------------------------------------------------------------------------
# Local support mask filter
# ---------------------------------------------------------------------------

def filter_mask_by_local_support(mask_hw: np.ndarray, K: int = 3) -> np.ndarray:
    import scipy.signal
    M   = (mask_hw > 0).astype(np.uint8)
    out = M.copy()
    for j in range(mask_hw.shape[0]):
        for i in range(mask_hw.shape[1]):
            if M[j, i] == 0:
                continue
            W_ = mask_hw.shape[1]
            cnt = sum(
                M[jj, (i+di) % W_]
                for jj in (j-1, j, j+1) if 0 <= jj < mask_hw.shape[0]
                for di in (-1, 0, 1)
                if not (jj == j and di == 0)
            )
            if cnt < K:
                out[j, i] = 0
    return out


# ---------------------------------------------------------------------------
# DEM computation for one cluster
# ---------------------------------------------------------------------------

def compute_dem(rock_mesh_path, sq_params, sq_center, sq_R,
                out_dem_path, out_mask_path, out_recon_path=None):
    """
    Raycasts from SQ surface → rock Poisson mesh to build a perpendicular DEM.
    Returns: (D_hw, M_hw, sq_wireframe)
    """
    a  = np.array(sq_params[:3])
    e1, e2 = sq_params[3], sq_params[4]

    rock_legacy = o3d.io.read_triangle_mesh(rock_mesh_path)
    if rock_legacy.is_empty():
        raise RuntimeError(f"Rock mesh empty: {rock_mesh_path}")
    rock_legacy.compute_triangle_normals()
    rock_t = o3d.t.geometry.TriangleMesh.from_legacy(rock_legacy)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(rock_t)

    # SQ grid
    eta   = -np.pi     + (np.arange(W) + 0.5) * (2.0 * np.pi / W)
    omega = -0.5*np.pi + (np.arange(H) + 0.5) * (np.pi / H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = sq_point_canonical(ETA, OMEGA, a, e1, e2)
    N_can = sq_normal_canonical(P_can, a, e1, e2)
    dots  = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[..., None] < 0.0, -N_can, N_can)

    P_w = (P_can @ sq_R.T) + sq_center
    N_w = N_can @ sq_R.T
    N_w = N_w / (np.linalg.norm(N_w, axis=-1, keepdims=True) + 1e-12)

    origins  = P_w.reshape(-1, 3).astype(np.float32)
    N_w_flat = N_w.reshape(-1, 3).astype(np.float32)

    if EMS_CIRCUMSCRIBE:
        # Balloon mode: shoot inward
        rays = np.concatenate([origins, -N_w_flat], axis=1)
        t    = scene.cast_rays(
            o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
        hit  = np.isfinite(t)
        D    = np.where(hit, t, 0.0).astype(np.float32)
        M    = hit.astype(np.uint8)
        print(f"  [DEM] Circumscribed mode: {hit.sum():,}/{hit.size:,} cells "
              f"({100*hit.mean():.1f}%)")
    else:
        rays_pos = np.concatenate([origins,  N_w_flat], axis=1)
        rays_neg = np.concatenate([origins, -N_w_flat], axis=1)
        t_pos = scene.cast_rays(o3d.core.Tensor(rays_pos, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
        t_neg = scene.cast_rays(o3d.core.Tensor(rays_neg, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
        hit_p, hit_n = np.isfinite(t_pos), np.isfinite(t_neg)
        both  = hit_p & hit_n
        D = np.zeros_like(t_pos, dtype=np.float32)
        M = np.zeros_like(t_pos, dtype=np.uint8)
        pick_p = both & (np.abs(t_pos) >= np.abs(t_neg))
        D[pick_p]     = t_pos[pick_p]
        D[both & ~pick_p] = -t_neg[both & ~pick_p]
        D[hit_p & ~hit_n] = t_pos[hit_p & ~hit_n]
        D[hit_n & ~hit_p] = -t_neg[hit_n & ~hit_p]
        M[both | hit_p | hit_n] = 1

    D_hw = D.reshape(H, W)
    M_hw = M.reshape(H, W)

    if USE_LOCAL_SUPPORT_FILTER:
        M_hw = filter_mask_by_local_support(M_hw, K=LOCAL_SUPPORT_K)
        D_hw = np.where(M_hw > 0, D_hw, 0.0).astype(np.float32)

    np.save(out_dem_path,  D_hw)
    np.save(out_mask_path, M_hw)
    print(f"  [INFO] DEM: {out_dem_path}")
    print(f"  [INFO] Mask: {out_mask_path}  | valid: {int(M_hw.sum())}/{H*W}")

    # Reconstructed points
    if out_recon_path and SAVE_RECON_POINTS_PLY:
        D_f = D_hw.reshape(-1).astype(np.float64)
        M_f = M_hw.reshape(-1) > 0
        P_w_f = P_w.reshape(-1, 3).astype(np.float64)
        N_w_f = N_w.reshape(-1, 3).astype(np.float64)
        P_recon = (P_w_f - D_f[:, None] * N_w_f)[M_f]
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(P_recon)
        o3d.io.write_point_cloud(out_recon_path, pcd_out)
        print(f"  [INFO] Recon pts: {out_recon_path}")

    # SQ wireframe for visualization
    P_w_flat = P_w.reshape(-1, 3)
    sq_mesh  = o3d.geometry.TriangleMesh()
    sq_mesh.vertices = o3d.utility.Vector3dVector(P_w_flat.astype(np.float64))
    faces = []
    for j in range(H-1):
        for i in range(W):
            i2 = (i+1) % W
            v00 = j*W+i; v10 = j*W+i2; v01 = (j+1)*W+i; v11 = (j+1)*W+i2
            faces += [[v00, v10, v11], [v00, v11, v01]]
    sq_mesh.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(sq_mesh)
    wireframe.paint_uniform_color([0.6, 0.6, 0.6])

    return D_hw, M_hw, wireframe


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
        name       = f"Cluster_{i}"
        txt_path   = os.path.join(IO_OP_DIR, f"sq_fit_{name}.txt")
        mesh_path  = os.path.join(IO_OP_DIR, f"poisson_{name}.ply")
        dem_path   = os.path.join(IO_OP_DIR, f"dem_{name}.npy")
        mask_path  = os.path.join(IO_OP_DIR, f"mask_{name}.npy")
        recon_path = os.path.join(IO_OP_DIR, f"recon_pts_{name}.ply")

        print(f"{'═'*50}\n  [{i}/{len(cluster_files)}]  DEM for {name}\n{'═'*50}")

        missing = [p for p in [txt_path, mesh_path] if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] Missing: {missing}\n"); continue

        sq = parse_sq_fit_txt(txt_path)
        print(f"  SQ params: {[f'{v:.4f}' for v in sq['params']]}")

        pcd_path    = os.path.join(IO_OP_DIR, f"{name}.ply")
        cluster_pcd = o3d.io.read_point_cloud(pcd_path)
        cluster_pcd.paint_uniform_color([0.5, 0.5, 0.5])

        try:
            D_hw, M_hw, sq_wire = compute_dem(
                rock_mesh_path=mesh_path,
                sq_params=sq["params"], sq_center=sq["center"], sq_R=sq["R"],
                out_dem_path=dem_path, out_mask_path=mask_path,
                out_recon_path=recon_path,
            )
        except Exception as e:
            print(f"  [ERROR] {e}\n"); continue

        rock_mesh = o3d.io.read_triangle_mesh(mesh_path)
        rock_mesh.paint_uniform_color([0.9, 0.35, 0.2])
        recon_pcd = o3d.io.read_point_cloud(recon_path)
        recon_pcd.paint_uniform_color([0.2, 0.8, 0.2])
        vis_data.append((cluster_pcd, sq_wire, rock_mesh, recon_pcd, name))
        print()

    if not vis_data:
        print("[ERROR] No DEMs produced."); sys.exit(1)

    # ── Visualize ────────────────────────────────────────────────────────────
    all_extents = [r[0].get_axis_aligned_bounding_box().get_extent() for r in vis_data]
    GAP = max(float(np.max(e)) for e in all_extents) * 1.5
    WIN_W, WIN_H, PAD = 1500, 620, 25
    if not os.environ.get("PC_HEADLESS"):
        visualizers = []
        for idx, (cluster_pcd, sq_wire, rock_mesh, recon_pcd, name) in enumerate(vis_data):
            top    = PAD + idx * (WIN_H + PAD*3)
            bb     = cluster_pcd.get_axis_aligned_bounding_box()
            bb_min = np.asarray(bb.get_min_bound())
            bb_max = np.asarray(bb.get_max_bound())
            extent = bb.get_extent()

            wire_bb  = sq_wire.get_axis_aligned_bounding_box()
            wire_min = np.asarray(wire_bb.get_min_bound())
            axis_sz  = float(np.max(extent)) * 0.15
            grid_sp  = max(0.05, round(float(np.max(extent)) / 6, 2))
            y_floor  = float(min(bb_min[1], wire_min[1]))
            dx       = [-GAP, 0.0, GAP]

            panel1 = [utils.viz_shift(cluster_pcd, dx[0])] + \
                     utils.viz_make_grid_panel(dx[0], bb_min, bb_max, y_floor, axis_sz, grid_sp)
            panel2 = [utils.viz_shift(sq_wire, dx[1])] + \
                     utils.viz_make_grid_panel(dx[1], wire_min, np.asarray(wire_bb.get_max_bound()), y_floor, axis_sz, grid_sp)
            panel3 = [utils.viz_shift(rock_mesh, dx[2]), utils.viz_shift(recon_pcd, dx[2])] + \
                     utils.viz_make_grid_panel(dx[2], bb_min, bb_max, y_floor, axis_sz, grid_sp)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{name}  |  Left: Cluster   Centre: SQ grid   Right: Mesh+DEM",
                width=WIN_W, height=WIN_H, left=PAD, top=top,
            )
            for g in panel1 + panel2 + panel3:
                vis.add_geometry(g)
            opt = vis.get_render_option(); opt.mesh_show_back_face = True; opt.point_size = 4.0
            vis.poll_events(); vis.update_renderer()
            visualizers.append(vis)

        utils.viz_event_loop(visualizers)
    print("\n✓ All DEMs generated!")
