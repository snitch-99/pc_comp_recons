"""
Step_2_poisson_recons.py
========================
Poisson surface reconstruction on each rock cluster point cloud.

Input:  Cluster_N.ply    (from Step 1)
Output: poisson_Cluster_N.ply
"""

import glob
import os
import sys

import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

from config import (
    MAP_DIR             as IO_OP_DIR,
    POISSON_DEPTH       as DEPTH,
    POISSON_NORMAL_KNN  as NORMAL_KNN,
    POISSON_DENSITY_QTLE as DENSITY_QUANTILE,
)


# ---------------------------------------------------------------------------
# Core reconstruction function
# ---------------------------------------------------------------------------

def poisson_reconstruct(
    ply_path:        str,
    out_mesh_path:   str,
    depth:           int   = 9,
    normal_knn:      int   = 50,
    density_quantile: float = 0.02,
    crop_to_bbox:    bool  = True,
) -> o3d.geometry.TriangleMesh:
    """Run Poisson reconstruction on a single cluster PLY. Returns the mesh."""
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {ply_path}")

    print(f"[INFO] {len(np.asarray(pcd.points)):,} pts  |  {ply_path}")

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=normal_knn)

    print(f"[INFO] Poisson depth={depth} ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    densities = np.asarray(densities)
    print(f"[INFO] Raw mesh: {len(mesh.vertices):,} verts  {len(mesh.triangles):,} tris")

    if crop_to_bbox:
        bbox = pcd.get_axis_aligned_bounding_box().scale(1.05, pcd.get_center())
        mesh = mesh.crop(bbox)
        print("[INFO] Cropped to expanded AABB.")

    if density_quantile and density_quantile > 0.0:
        if len(densities) == len(mesh.vertices):
            thr = np.quantile(densities, density_quantile)
            mesh.remove_vertices_by_mask(densities < thr)
            print(f"[INFO] Density filter: removed lowest {density_quantile*100:.1f}%")
        else:
            print("[WARN] Skipping density filter (size mismatch after crop).")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    if not o3d.io.write_triangle_mesh(out_mesh_path, mesh):
        raise RuntimeError(f"Failed to write: {out_mesh_path}")
    print(f"[INFO] Saved: {out_mesh_path}")
    return mesh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))
    if not cluster_files:
        print(f"[ERROR] No Cluster_*.ply in {IO_OP_DIR}")
        sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s).\n")
    vis_data = []

    for i, ply_path in enumerate(cluster_files, start=1):
        name     = os.path.splitext(os.path.basename(ply_path))[0]
        out_mesh = os.path.join(IO_OP_DIR, f"poisson_{name}.ply")
        print(f"{'═'*50}\n  [{i}/{len(cluster_files)}]  {name}\n{'═'*50}")

        try:
            mesh = poisson_reconstruct(
                ply_path=ply_path, out_mesh_path=out_mesh,
                depth=DEPTH, normal_knn=NORMAL_KNN,
                density_quantile=DENSITY_QUANTILE,
            )
        except Exception as e:
            print(f"  [ERROR] {e}\n"); continue

        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.paint_uniform_color([0.2, 0.6, 1.0])
        mesh.paint_uniform_color([0.9, 0.35, 0.2])
        vis_data.append((pcd, mesh, name))
        print()

    if not vis_data:
        print("[ERROR] No meshes produced."); sys.exit(1)

    # ── Visualize ────────────────────────────────────────────────────────────
    all_extents = [r[0].get_axis_aligned_bounding_box().get_extent() for r in vis_data]
    max_dim     = max(float(np.max(e)) for e in all_extents)
    GAP         = max_dim * 1.5
    WIN_W, WIN_H, PAD = 1500, 620, 25
    if not os.environ.get("PC_HEADLESS"):
        visualizers = []
        for idx, (pcd_vis, mesh, name) in enumerate(vis_data):
            top    = PAD + idx * (WIN_H + PAD * 3)
            bb     = pcd_vis.get_axis_aligned_bounding_box()
            bb_min = np.asarray(bb.get_min_bound())
            bb_max = np.asarray(bb.get_max_bound())
            extent = bb.get_extent()

            mesh_bb  = mesh.get_axis_aligned_bounding_box()
            mesh_min = np.asarray(mesh_bb.get_min_bound())
            axis_sz  = float(np.max(extent)) * 0.15
            grid_sp  = max(0.05, round(float(np.max(extent)) / 6, 2))
            y_floor  = float(min(bb_min[1], mesh_min[1]))
            dx       = [-GAP, 0.0, GAP]

            mesh_wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            mesh_wire.paint_uniform_color([1.0, 0.6, 0.1])

            panel1 = [utils.viz_shift(pcd_vis, dx[0])] + \
                     utils.viz_make_grid_panel(dx[0], bb_min, bb_max, y_floor, axis_sz, grid_sp)
            panel2 = [utils.viz_shift(mesh, dx[1])] + \
                     utils.viz_make_grid_panel(dx[1], mesh_min, np.asarray(mesh_bb.get_max_bound()), y_floor, axis_sz, grid_sp)
            panel3 = [utils.viz_shift(pcd_vis, dx[2]), utils.viz_shift(mesh_wire, dx[2])] + \
                     utils.viz_make_grid_panel(dx[2], bb_min, bb_max, y_floor, axis_sz, grid_sp)

            vis = o3d.visualization.Visualizer()
            vis.create_window(
                window_name=f"{name}  |  Left: PC   Centre: Poisson   Right: PC+Wire",
                width=WIN_W, height=WIN_H, left=PAD, top=top,
            )
            for g in panel1 + panel2 + panel3:
                vis.add_geometry(g)
            opt = vis.get_render_option(); opt.mesh_show_back_face = True
            vis.poll_events(); vis.update_renderer()
            visualizers.append(vis)

        utils.viz_event_loop(visualizers)
    print("\n✓ All Poisson meshes saved!")
