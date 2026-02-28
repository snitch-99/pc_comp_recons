import os
import numpy as np
import open3d as o3d


def poisson_reconstruct(
    ply_path="cluster_1.ply",
    out_mesh_path="rock_poisson_mesh.ply",
    depth=9,
    normal_knn=50,
    density_quantile=0.02,   # set 0.0 to disable
    crop_to_bbox=True,
    show_overlay=True,
    show_mesh_only=False,
):
    # 1) Load point cloud
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise RuntimeError(f"Loaded point cloud is empty: {ply_path}")

    pts = np.asarray(pcd.points)
    print(f"[INFO] Loaded: {ply_path}  |  #points={len(pts)}")

    # 2) Estimate + orient normals (required for Poisson)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=normal_knn)

    # 3) Poisson reconstruction
    print(f"[INFO] Poisson reconstruction: depth={depth} ...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    densities = np.asarray(densities)
    print(f"[INFO] Raw mesh: #verts={len(mesh.vertices)}, #tris={len(mesh.triangles)}")

    # 4) Crop to point cloud bounding box (helps avoid hallucinated outer shells)
    if crop_to_bbox:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox = bbox.scale(1.05, bbox.get_center())  # slight expansion
        mesh = mesh.crop(bbox)
        print("[INFO] Cropped mesh to expanded AABB of point cloud.")

    # 5) Optional density filtering (removes weakly-supported regions)
    # Note: after cropping, densities no longer align with vertices, so we only filter if sizes still match.
    if density_quantile is not None and density_quantile > 0.0:
        if len(densities) == len(mesh.vertices):
            thr = np.quantile(densities, density_quantile)
            mesh.remove_vertices_by_mask(densities < thr)
            print(f"[INFO] Density filter: removed lowest {density_quantile*100:.1f}% (thr={thr:.6f})")
        else:
            print("[WARN] Skipping density filter (densities don't match cropped mesh vertex count).")

    # 6) Cleanup + normals for visualization/export
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()

    # 7) Save mesh
    if not o3d.io.write_triangle_mesh(out_mesh_path, mesh):
        raise RuntimeError(f"Failed to write mesh: {out_mesh_path}")
    print(f"[INFO] Saved mesh to: {out_mesh_path}")

    # 8) Display
    if show_overlay:
        pcd_vis = pcd.paint_uniform_color([0.2, 0.6, 1.0])   # blue
        mesh_vis = mesh.paint_uniform_color([0.9, 0.3, 0.3]) # red
        o3d.visualization.draw_geometries(
            [pcd_vis, mesh_vis],
            window_name="Overlay: Point Cloud (blue) + Poisson Mesh (red)",
            mesh_show_back_face=True
        )

    if show_mesh_only:
        mesh_only = mesh.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries(
            [mesh_only],
            window_name="Poisson Mesh Only",
            mesh_show_back_face=True
        )

    return mesh


if __name__ == "__main__":
    import glob
    import sys
    from config import (MAP_DIR as IO_OP_DIR,
                        POISSON_DEPTH as DEPTH,
                        POISSON_NORMAL_KNN as NORMAL_KNN,
                        POISSON_DENSITY_QTLE as DENSITY_QUANTILE)

    # ── Find clusters ─────────────────────────────────────────────────────────
    cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))
    if not cluster_files:
        print(f"[ERROR] No Cluster_*.ply files found in:\n  {IO_OP_DIR}")
        sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s) to reconstruct.\n")

    # ── Run Poisson on each cluster ───────────────────────────────────────────
    vis_data = []   # (pcd_colored, mesh_colored, name)

    for i, ply_path in enumerate(cluster_files, start=1):
        name = os.path.splitext(os.path.basename(ply_path))[0]
        out_mesh = os.path.join(IO_OP_DIR, f"poisson_{name}.ply")

        print(f"{'═'*50}")
        print(f"  [{i}/{len(cluster_files)}]  Reconstructing {name}")
        print(f"{'═'*50}")

        try:
            mesh = poisson_reconstruct(
                ply_path=ply_path,
                out_mesh_path=out_mesh,
                depth=DEPTH,
                normal_knn=NORMAL_KNN,
                density_quantile=DENSITY_QUANTILE,
                crop_to_bbox=True,
                show_overlay=False,   # we'll show all at once below
                show_mesh_only=False,
            )
        except Exception as e:
            print(f"  [ERROR] {name}: {e}\n")
            continue

        # Prepare display objects
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd.paint_uniform_color([0.2, 0.6, 1.0])   # blue
        mesh.paint_uniform_color([0.9, 0.35, 0.2]) # orange-red
        vis_data.append((pcd, mesh, name))
        print()

    if not vis_data:
        print("[ERROR] No meshes were produced.")
        sys.exit(1)

    # ── Visualization helpers ──────────────────────────────────────────────────
    import copy as _copy

    def _shift(geom, dx):
        """Return a deep copy of geom translated by dx along X."""
        g = _copy.deepcopy(geom)
        g.translate([dx, 0, 0])
        return g

    def _make_grid_panel(center_x, bb_min, bb_max, y_floor, axis_size, grid_spacing):
        """Build XZ reference grid + coordinate frame for a panel centred at center_x.
        y_floor is shared across all panels so grids are coplanar."""
        geoms = []

        # ── Flat XZ grid at shared Y floor ─────────────────────────────────
        x0 = center_x + bb_min[0];  x1 = center_x + bb_max[0]
        z0 = bb_min[2];             z1 = bb_max[2]
        pts, lines, i = [], [], 0
        import numpy as _np
        for x in _np.arange(x0, x1 + grid_spacing, grid_spacing):
            pts += [[x, y_floor, z0], [x, y_floor, z1]]
            lines.append([i, i + 1]); i += 2
        for z in _np.arange(z0, z1 + grid_spacing, grid_spacing):
            pts += [[x0, y_floor, z], [x1, y_floor, z]]
            lines.append([i, i + 1]); i += 2
        grid = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(_np.array(pts, dtype=float)),
            lines=o3d.utility.Vector2iVector(_np.array(lines, dtype=_np.int32)),
        )
        grid.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5]] * len(lines))
        geoms.append(grid)

        # ── Coordinate frame at panel corner, sitting on the shared floor ───
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size,
            origin=[center_x + float(bb_min[0]),
                    y_floor,
                    float(bb_min[2])]
        )
        geoms.append(frame)
        return geoms

    # ── Fixed panel gap: use the largest cluster extent across all results ──
    all_extents = [r[0].get_axis_aligned_bounding_box().get_extent() for r in vis_data]
    max_dim     = max(float(np.max(e)) for e in all_extents)
    PANEL_GAP   = max_dim * 1.0          # fixed spacing — same for all windows

    # ── One window per cluster, 3 spatially-offset panels ─────────────────────
    # Panel 1 (left)   : cluster point cloud only
    # Panel 2 (centre) : poisson solid mesh only
    # Panel 3 (right)  : cluster point cloud + mesh wireframe overlaid
    print(f"  Panel gap (centre-to-centre): {PANEL_GAP:.3f} m")
    print(f"Opening {len(vis_data)} window(s) — 3 panels each (close all to exit).")

    WIN_W, WIN_H = 1500, 620
    WIN_PAD      = 25
    visualizers  = []

    for idx, (pcd_vis, mesh, name) in enumerate(vis_data):
        top = WIN_PAD + idx * (WIN_H + WIN_PAD * 3)

        bb      = pcd_vis.get_axis_aligned_bounding_box()
        bb_min  = np.asarray(bb.get_min_bound())
        bb_max  = np.asarray(bb.get_max_bound())
        extent  = bb.get_extent()

        axis_size    = float(np.max(extent)) * 0.15
        grid_spacing = max(0.05, round(float(np.max(extent)) / 6, 2))

        # Shared Y floor — lowest point between cluster and Poisson mesh
        mesh_bb  = mesh.get_axis_aligned_bounding_box()
        mesh_min = np.asarray(mesh_bb.get_min_bound())
        mesh_max = np.asarray(mesh_bb.get_max_bound())
        y_floor = float(min(bb_min[1], mesh_min[1]))

        # Panel X-centres: -PANEL_GAP, 0, +PANEL_GAP
        dx = [-PANEL_GAP, 0.0, PANEL_GAP]

        # ── Panel 1: cluster PC ───────────────────────────────────────────
        panel1 = [_shift(pcd_vis, dx[0])]
        panel1 += _make_grid_panel(dx[0], bb_min, bb_max, y_floor, axis_size, grid_spacing)

        # ── Panel 2: Poisson solid mesh ───────────────────────────────────
        panel2 = [_shift(mesh, dx[1])]
        panel2 += _make_grid_panel(dx[1], mesh_min, mesh_max, y_floor, axis_size, grid_spacing)

        # ── Panel 3: cluster PC + mesh wireframe ──────────────────────────
        mesh_wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_wire.paint_uniform_color([1.0, 0.6, 0.1])   # amber
        panel3  = [_shift(pcd_vis, dx[2]), _shift(mesh_wire, dx[2])]
        panel3 += _make_grid_panel(dx[2], bb_min, bb_max, y_floor, axis_size, grid_spacing)

        # ── Open window ───────────────────────────────────────────────────
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"{name}  |  Left: PC   Centre: Poisson   Right: PC+Wireframe",
            width=WIN_W, height=WIN_H,
            left=WIN_PAD, top=top,
        )
        for geom in panel1 + panel2 + panel3:
            vis.add_geometry(geom)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True
        vis.poll_events()
        vis.update_renderer()
        visualizers.append(vis)

    # Event loop — keep windows alive
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

    print("\n✓ All Poisson meshes saved and displayed!")
