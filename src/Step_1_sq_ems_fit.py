"""
Step_1_sq_ems_fit.py
====================
Fits a Superquadric (EMS algorithm) to each pre-separated rock cluster.

Input  (Mode B): one or more .ply/.obj files already placed in MAP_DIR
                 (point clouds or meshes)
Output:          sq_fit_Cluster_N.txt  — SQ params + pose
                 Cluster_N.ply         — point cloud copy (for later steps)
"""

import glob
import os
import sys

import numpy as np
import open3d as o3d

# Local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ems_core import Superquadric, EMSFitter
import utils

# ---------------------------------------------------------------------------
# Config imports
# ---------------------------------------------------------------------------
from config import (
    MAP_DIR as IO_OP_DIR,
    EMS_MAX_ITERS  as MAX_ITERS,
    EMS_DOWNSAMPLE_N,
    EMS_CIRCUMSCRIBE,
    EMS_CIRCUM_TARGET,
    SINGLE_MESH_INPUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(filename: str) -> str:
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return os.path.abspath(filename)
    return os.path.join(IO_OP_DIR, filename)


def _save_fit_results_txt(out_path, params, center_world,
                           R_world_to_canonical, sigma_sq,
                           inliers_outliers, ply_path):
    num_in, num_out = inliers_outliers
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Superquadric EMS Fit Results\n==========================\n\n")
        f.write(f"Input point cloud: {ply_path}\n\n")
        f.write("Best-fit Superquadric parameters [ax, ay, az, e1, e2] (canonical frame):\n")
        f.write("  " + " ".join(f"{float(x):.10g}" for x in params) + "\n\n")
        f.write(f"sigma_sq: {float(sigma_sq):.10g}\n")
        f.write(f"Inliers: {int(num_in)}\nOutliers: {int(num_out)}\n\n")
        f.write("Pose metadata\n------------\n")
        f.write("Center (world):\n")
        f.write("  " + " ".join(f"{float(x):.10g}" for x in center_world) + "\n\n")
        f.write("Rotation R_init (world -> canonical):\n")
        for row in R_world_to_canonical:
            f.write("  " + " ".join(f"{float(x):.10g}" for x in row) + "\n")
        f.write("\nMapping note:\n")
        f.write("  canonical point p_c -> world point p_w = center + p_c @ R_init.T\n")


def circumscribe_sq(model: Superquadric, fitter: EMSFitter,
                    target: float = 1.0, n_bisect: int = 40):
    """Scale SQ axes outward until all cluster points satisfy F(p) <= target."""
    pts = fitter.points
    ax, ay, az = model.ax, model.ay, model.az
    e1, e2 = max(model.e1, 0.1), max(model.e2, 0.1)
    eps = 1e-12

    def max_F(s):
        X = np.abs(pts[:, 0] / (s*ax + eps))
        Y = np.abs(pts[:, 1] / (s*ay + eps))
        Z = np.abs(pts[:, 2] / (s*az + eps))
        inner = (X**(2/e2) + Y**(2/e2))**(e2/e1) + Z**(2/e1)
        return float(np.max(inner))

    if max_F(1.0) <= target:
        print(f"  [CIRCUM] Already satisfies F≤{target:.2f} at s=1.0 ✓")
        return model, 1.0

    lo, hi = 1.0, 5.0
    while max_F(hi) >= target:
        hi *= 2.0
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        (hi if max_F(mid) < target else lo).__class__  # dummy; real swap:
        if max_F(mid) < target:
            hi = mid
        else:
            lo = mid

    s    = hi * 1.01
    circ = Superquadric([s*model.ax, s*model.ay, s*model.az, model.e1, model.e2])
    print(f"  [CIRCUM] s={s:.4f} → ax={circ.ax:.4f} ay={circ.ay:.4f} az={circ.az:.4f}")
    return circ, s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ── Collect input files ──────────────────────────────────────────────────
    if len(sys.argv) > 1:
        # Explicit file passed on command line
        cluster_files = [_resolve_path(sys.argv[1])]
    elif SINGLE_MESH_INPUT:
        cluster_files = [_resolve_path(SINGLE_MESH_INPUT)]
    else:
        cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))

    if not cluster_files:
        print("[ERROR] No input files found. Add rock files to MAP_DIR or set SINGLE_MESH_INPUT.")
        sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s) to fit.\n")
    results = []   # (pcd_vis, sq_mesh, name)

    for i, ply_path in enumerate(cluster_files, start=1):
        name = f"Cluster_{i}" if SINGLE_MESH_INPUT or len(sys.argv) > 1 \
               else os.path.splitext(os.path.basename(ply_path))[0]

        print(f"{'═'*50}\n  [{i}/{len(cluster_files)}]  Fitting {name}\n{'═'*50}")

        # Load mesh or point cloud
        mesh = o3d.io.read_triangle_mesh(ply_path)
        if not mesh.is_empty() and len(mesh.triangles) > 0:
            print("  [INFO] Mesh detected — sampling 100 000-point proxy cloud.")
            pcd = mesh.sample_points_uniformly(number_of_points=100_000)
        else:
            pcd = o3d.io.read_point_cloud(ply_path)

        if pcd.is_empty():
            print(f"  [WARN] {name} is empty, skipping.")
            continue

        # Cache as Cluster_N.ply so downstream steps always find it
        pipeline_ply = os.path.join(IO_OP_DIR, f"{name}.ply")
        if ply_path != pipeline_ply:
            o3d.io.write_point_cloud(pipeline_ply, pcd)
        print(f"  {len(pcd.points):,} points (cached as {name}.ply).")

        # Downsample for EMS
        n_pts = len(pcd.points)
        if EMS_DOWNSAMPLE_N > 0 and n_pts > EMS_DOWNSAMPLE_N:
            idx     = np.random.choice(n_pts, EMS_DOWNSAMPLE_N, replace=False)
            pcd_fit = pcd.select_by_index(idx.tolist())
            print(f"  Downsampled {n_pts:,} → {EMS_DOWNSAMPLE_N:,} pts for EMS.")
        else:
            pcd_fit = pcd

        # EMS fit
        fitter = EMSFitter(pcd_fit)
        model  = fitter.fit(max_iters=MAX_ITERS)

        _, _, inlier_mask = utils.analyze_inliers(model, fitter.points)
        num_in  = int(inlier_mask.sum())
        num_out = len(fitter.points) - num_in
        sigma_sq = fitter.sigma_sq

        best_params = [float(x) for x in [model.ax, model.ay, model.az, model.e1, model.e2]]
        print(f"  SQ params [ax,ay,az,e1,e2]: {[f'{p:.4f}' for p in best_params]}")
        print(f"  Inliers: {num_in:,}  |  Outliers: {num_out:,}")

        # Circumscribe
        if EMS_CIRCUMSCRIBE:
            full_pts   = (np.asarray(pcd.points) - fitter.center) @ fitter.R_init
            mock_fitter = type('F', (), {'points': full_pts, 'extent': fitter.extent})()
            model, _scale = circumscribe_sq(model, mock_fitter, target=EMS_CIRCUM_TARGET)
            best_params = [float(x) for x in [model.ax, model.ay, model.az, model.e1, model.e2]]

        # Save txt
        out_txt = os.path.join(IO_OP_DIR, f"sq_fit_{name}.txt")
        _save_fit_results_txt(out_txt, best_params, fitter.center,
                              fitter.R_init, sigma_sq,
                              (num_in, num_out), ply_path)
        print(f"  Saved: {out_txt}\n")

        # Build SQ mesh in world frame
        V, F    = model.sample_surface(nu=200, nv=100)
        sq_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V),
            triangles=o3d.utility.Vector3iVector(F),
        )
        sq_mesh.compute_vertex_normals()
        T = np.eye(4); T[:3, :3] = fitter.R_init.T; T[:3, 3] = fitter.center
        sq_mesh.transform(T)

        # Prepare display clouds
        voxel_sz = max(1e-6, float(np.mean(fitter.extent)) / 300.0)
        try:
            pcd_vis = pcd.voxel_down_sample(voxel_size=voxel_sz)
        except Exception:
            pcd_vis = pcd
        pcd_vis.paint_uniform_color([0.2, 0.6, 1.0])   # blue
        sq_mesh.paint_uniform_color([1.0, 0.35, 0.2])  # orange-red

        results.append((pcd_vis, sq_mesh, name))

    if not results:
        print("[ERROR] No clusters fitted.")
        sys.exit(1)

    # ── Visualize ────────────────────────────────────────────────────────────
    all_extents = [r[0].get_axis_aligned_bounding_box().get_extent() for r in results]
    max_dim     = max(float(np.max(e)) for e in all_extents)
    GAP         = max_dim * 1.5
    WIN_W, WIN_H, PAD = 1500, 620, 25
    if not os.environ.get("PC_HEADLESS"):
        visualizers = []
        for idx, (pcd_vis, sq_mesh, name) in enumerate(results):
            top    = PAD + idx * (WIN_H + PAD * 3)
            bb     = pcd_vis.get_axis_aligned_bounding_box()
            bb_min = np.asarray(bb.get_min_bound())
            bb_max = np.asarray(bb.get_max_bound())
            extent = bb.get_extent()

            axis_sz    = float(np.max(extent)) * 0.15
            grid_sp    = max(0.05, round(float(np.max(extent)) / 6, 2))
            sq_bb      = sq_mesh.get_axis_aligned_bounding_box()
            sq_min     = np.asarray(sq_bb.get_min_bound())
            y_floor    = float(min(bb_min[1], sq_min[1]))
            dx         = [-GAP, 0.0, GAP]

            sq_wire = o3d.geometry.LineSet.create_from_triangle_mesh(sq_mesh)
            sq_wire.paint_uniform_color([1.0, 0.6, 0.1])

            panel1 = [utils.viz_shift(pcd_vis, dx[0])] + \
                     utils.viz_make_grid_panel(dx[0], bb_min, bb_max, y_floor, axis_sz, grid_sp)
            panel2 = [utils.viz_shift(sq_mesh, dx[1])] + \
                     utils.viz_make_grid_panel(dx[1], sq_min, np.asarray(sq_bb.get_max_bound()), y_floor, axis_sz, grid_sp)
            panel3 = [utils.viz_shift(pcd_vis, dx[2]), utils.viz_shift(sq_wire, dx[2])] + \
                     utils.viz_make_grid_panel(dx[2], bb_min, bb_max, y_floor, axis_sz, grid_sp)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{name}  |  Left: PC   Centre: SQ   Right: PC+Wire",
                width=WIN_W, height=WIN_H, left=PAD, top=top,
            )
            for g in panel1 + panel2 + panel3:
                vis.add_geometry(g)
            vis.poll_events(); vis.update_renderer()
            visualizers.append(vis)

        utils.viz_event_loop(visualizers)
    print("\n✓ All done!")
