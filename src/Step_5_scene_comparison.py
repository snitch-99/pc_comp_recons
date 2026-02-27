"""
Step 5 — Full Scene Reconstruction & Comparison
================================================
Assembles a complete reconstructed scene from pipeline outputs and shows it
side-by-side with the original point cloud.

Inputs (from MAP_DIR):
    ground_plane.txt          — RANSAC plane equation [a b c d]
    ground_plane.ply          — RANSAC inlier points (for bbox extent)
    dem_recon_Cluster_N.ply   — DEM-reconstructed rock meshes (world frame)
    <input point cloud>       — original scan

Outputs:
    scene_reconstruction.ply  — merged reconstructed scene mesh

Usage:
    python3 Step_5_scene_comparison.py
"""

import os
import sys
import glob
import numpy as np
import open3d as o3d

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MAP_DIR, INPUT_PLY


# =============================================================================
# HELPERS
# =============================================================================

def load_ground_plane(map_dir):
    """Load plane equation [a,b,c,d] from ground_plane.txt."""
    txt = os.path.join(map_dir, "ground_plane.txt")
    if not os.path.exists(txt):
        raise FileNotFoundError(
            f"[ERROR] ground_plane.txt not found in {map_dir}\n"
            "        Run Step_0_cluster_rocks.py first."
        )
    vals = list(map(float, open(txt).read().split()))
    return np.array(vals)   # [a, b, c, d]


def build_ground_mesh(ground_ply_path, plane_eq, color=(0.76, 0.70, 0.50)):
    """
    Build a finite flat ground mesh using the BBOX of the RANSAC inlier points
    projected onto the fitted plane.

    The mesh is a subdivided quad covering the exact footprint of the ground
    point cloud, lying on the RANSAC plane.
    """
    a, b, c, d = plane_eq
    normal = np.array([a, b, c])
    normal /= np.linalg.norm(normal)

    # Load ground inlier points
    gnd = o3d.io.read_point_cloud(ground_ply_path)
    if gnd.is_empty():
        raise RuntimeError(f"Ground point cloud is empty: {ground_ply_path}")
    pts = np.asarray(gnd.points)

    # Project every point onto the plane  p' = p - ((a·px+b·py+c·pz+d)/|n|²)·n
    n_sq = float(np.dot(normal, normal))
    dist = (pts @ normal + d) / n_sq
    pts_proj = pts - dist[:, None] * normal[None, :]

    # Build an ONB on the plane: u and v are orthogonal to the normal
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # 2-D coordinates in the plane
    coords_u = pts_proj @ u
    coords_v = pts_proj @ v
    u_min, u_max = coords_u.min(), coords_u.max()
    v_min, v_max = coords_v.min(), coords_v.max()

    # Build a grid of quads (N×M) covering the bbox
    N, M = 40, 40
    us = np.linspace(u_min, u_max, N)
    vs = np.linspace(v_min, v_max, M)
    UU, VV = np.meshgrid(us, vs)           # (M, N)

    # Origin of the plane: closest point on the plane to the origin
    origin = -d / n_sq * normal

    # 3-D vertices
    verts = []
    for mi in range(M):
        for ni in range(N):
            p3 = origin + UU[mi, ni] * u + VV[mi, ni] * v
            verts.append(p3)
    verts = np.array(verts)                # (M*N, 3)

    # Build triangles
    tris = []
    for mi in range(M - 1):
        for ni in range(N - 1):
            v00 = mi * N + ni
            v10 = mi * N + ni + 1
            v01 = (mi + 1) * N + ni
            v11 = (mi + 1) * N + ni + 1
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])
    tris = np.array(tris, dtype=np.int32)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(tris),
    )
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))
    return mesh

def make_axis_frame(size=0.5, origin=(0, 0, 0)):
    """Coordinate frame: red=X, green=Y, blue=Z."""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=list(origin)
    )


def make_world_grid(bounds_min, bounds_max, y_level, spacing=0.5):
    """
    Flat reference grid on the XZ plane at height y_level.
    bounds_min / bounds_max : (3,) arrays defining scene extent.
    spacing : distance between grid lines (metres).
    Returns an o3d.geometry.LineSet coloured light grey.
    """
    x0, x1 = bounds_min[0], bounds_max[0]
    z0, z1 = bounds_min[2], bounds_max[2]

    lines, pts = [], []
    pt_idx = 0

    # Lines parallel to Z (varying X)
    for x in np.arange(x0, x1 + spacing, spacing):
        pts.append([x, y_level, z0])
        pts.append([x, y_level, z1])
        lines.append([pt_idx, pt_idx + 1])
        pt_idx += 2

    # Lines parallel to X (varying Z)
    for z in np.arange(z0, z1 + spacing, spacing):
        pts.append([x0, y_level, z])
        pts.append([x1, y_level, z])
        lines.append([pt_idx, pt_idx + 1])
        pt_idx += 2

    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)),
    )
    grey = [[0.6, 0.6, 0.6]] * len(lines)
    grid.colors = o3d.utility.Vector3dVector(grey)
    return grid


# =============================================================================
# MAIN
# =============================================================================

def main():
    map_dir = MAP_DIR

    # ── 1. Ground plane ────────────────────────────────────────────────────────
    print("\n[1/4]  Loading ground plane ...")
    plane_eq   = load_ground_plane(map_dir)
    ground_ply = os.path.join(map_dir, "ground_plane.ply")
    if not os.path.exists(ground_ply):
        raise FileNotFoundError(
            f"[ERROR] ground_plane.ply not found in {map_dir}\n"
            "        Run Step_0_cluster_rocks.py first."
        )
    print(f"       Plane eq: {plane_eq[0]:.4f}x + {plane_eq[1]:.4f}y "
          f"+ {plane_eq[2]:.4f}z + {plane_eq[3]:.4f} = 0")

    ground_mesh = build_ground_mesh(ground_ply, plane_eq, color=(0.76, 0.70, 0.50))
    print(f"       Ground mesh: {len(ground_mesh.vertices)} verts")

    # ── 2. Rock meshes ─────────────────────────────────────────────────────────
    print("\n[2/4]  Loading DEM-reconstructed rock meshes ...")
    rock_files = sorted(glob.glob(os.path.join(map_dir, "dem_recon_Cluster_*.ply")))
    if not rock_files:
        print("[WARN] No dem_recon_Cluster_*.ply found — run Steps 1-4 first.")
    rock_meshes = []
    for rf in rock_files:
        mesh = o3d.io.read_triangle_mesh(rf)
        if mesh.is_empty():
            print(f"  [SKIP] {os.path.basename(rf)} is empty.")
            continue
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.25, 0.75, 0.35])   # green
        rock_meshes.append(mesh)
        print(f"  Loaded: {os.path.basename(rf)}  "
              f"({len(mesh.vertices):,} verts)")

    # ── 3. Merge and save reconstructed scene ──────────────────────────────────
    print("\n[3/4]  Merging into full scene mesh ...")
    all_meshes = [ground_mesh] + rock_meshes
    scene_mesh = all_meshes[0]
    for m in all_meshes[1:]:
        scene_mesh += m

    out_path = os.path.join(map_dir, "scene_reconstruction.ply")
    o3d.io.write_triangle_mesh(out_path, scene_mesh)
    print(f"       Saved: {out_path}")

    # ── 3b. Compression report (printed + saved before visualization) ──────────
    def fsize(path):
        return os.path.getsize(path) if os.path.exists(path) else 0

    def fmt(b):
        if b >= 1_048_576:
            return f"{b/1_048_576:.2f} MB"
        return f"{b/1_024:.1f} KB"

    cluster_names = sorted(
        {os.path.basename(f).replace("dem_recon_", "").replace(".ply", "")
         for f in rock_files}
    )
    total_poisson = 0
    total_compact = 0
    rows = []
    for cname in cluster_names:
        poisson_sz = fsize(os.path.join(map_dir, f"poisson_{cname}.ply"))
        dem_sz     = fsize(os.path.join(map_dir, f"dem_{cname}.npy"))
        mask_sz    = fsize(os.path.join(map_dir, f"mask_{cname}.npy"))
        sq_sz      = fsize(os.path.join(map_dir, f"sq_fit_{cname}.txt"))
        compact_sz = dem_sz + mask_sz + sq_sz
        ratio      = poisson_sz / compact_sz if compact_sz > 0 else float("inf")
        total_poisson += poisson_sz
        total_compact += compact_sz
        rows.append((cname, poisson_sz, compact_sz, ratio,
                     dem_sz, mask_sz, sq_sz))

    overall = total_poisson / total_compact if total_compact > 0 else float("inf")
    sep = "─" * 66

    lines = [
        "",
        sep,
        "  COMPRESSION REPORT  (Poisson PLY  vs  DEM + mask + SQ txt)",
        f"  Map: {map_dir}",
        sep,
        f"  {'Cluster':<12}  {'Poisson PLY':>12}  {'DEM npy':>9}  "
        f"{'Mask npy':>9}  {'SQ txt':>7}  {'Total':>10}  {'Ratio':>7}",
        sep,
    ]
    for cname, psz, csz, ratio, dsz, msz, ssz in rows:
        lines.append(
            f"  {cname:<12}  {fmt(psz):>12}  {fmt(dsz):>9}  "
            f"{fmt(msz):>9}  {fmt(ssz):>7}  {fmt(csz):>10}  {ratio:>5.1f}×"
        )
    lines += [
        sep,
        f"  {'TOTAL':<12}  {fmt(total_poisson):>12}  {'':>9}  "
        f"{'':>9}  {'':>7}  {fmt(total_compact):>10}  {overall:>5.1f}×",
        sep,
        "",
    ]

    report = "\n".join(lines)
    print(report)

    analysis_path = os.path.join(map_dir, "analysis.txt")
    with open(analysis_path, "w") as f:
        f.write(report + "\n")
    print(f"  Report saved → {analysis_path}\n")

    # ── 3c. Geometric accuracy — Hausdorff distance dem_recon vs poisson ────────
    N_SAMPLE = 50_000   # pts sampled from each reconstructed mesh
    print("Computing geometric accuracy (Hausdorff distances) ...")

    acc_rows = []
    for cname in cluster_names:
        recon_path   = os.path.join(map_dir, f"dem_recon_{cname}.ply")
        poisson_path = os.path.join(map_dir, f"poisson_{cname}.ply")
        if not os.path.exists(recon_path) or not os.path.exists(poisson_path):
            print(f"  [SKIP] {cname}: missing mesh file.")
            acc_rows.append((cname, None, None, None))
            continue

        recon_mesh   = o3d.io.read_triangle_mesh(recon_path)
        poisson_mesh = o3d.io.read_triangle_mesh(poisson_path)

        # Sample points from both meshes
        n_recon   = min(N_SAMPLE, len(recon_mesh.triangles)   * 3)
        n_poisson = min(N_SAMPLE, len(poisson_mesh.triangles) * 3)
        pcd_recon   = recon_mesh.sample_points_uniformly(n_recon)
        pcd_poisson = poisson_mesh.sample_points_uniformly(n_poisson)

        # Distances from recon → poisson (one-sided Hausdorff)
        dists = np.asarray(pcd_recon.compute_point_cloud_distance(pcd_poisson))
        mean_d = float(np.mean(dists))
        rms_d  = float(np.sqrt(np.mean(dists**2)))
        max_d  = float(np.max(dists))
        acc_rows.append((cname, mean_d, rms_d, max_d))
        print(f"  {cname}: mean={mean_d*100:.2f} cm  rms={rms_d*100:.2f} cm  "
              f"max={max_d*100:.2f} cm")

    sep2 = "─" * 62
    acc_lines = [
        "",
        sep2,
        "  GEOMETRIC ACCURACY  (dem_recon vs poisson ground truth)",
        f"  Sampled {N_SAMPLE:,} pts per mesh  |  distances in centimetres",
        sep2,
        f"  {'Cluster':<12}  {'Mean (cm)':>10}  {'RMS (cm)':>9}  {'Max (cm)':>9}",
        sep2,
    ]
    for cname, mean_d, rms_d, max_d in acc_rows:
        if mean_d is None:
            acc_lines.append(f"  {cname:<12}  {'N/A':>10}  {'N/A':>9}  {'N/A':>9}")
        else:
            acc_lines.append(
                f"  {cname:<12}  {mean_d*100:>10.3f}  {rms_d*100:>9.3f}  "
                f"{max_d*100:>9.3f}"
            )
    acc_lines += [sep2, ""]

    acc_report = "\n".join(acc_lines)
    print(acc_report)
    with open(analysis_path, "a") as f:
        f.write(acc_report + "\n")
    print(f"  Accuracy report appended → {analysis_path}\n")

    # ── 4. Load original point cloud ───────────────────────────────────────────
    print("\n[4/4]  Loading original point cloud ...")
    if not os.path.exists(INPUT_PLY):
        print(f"[WARN] Original cloud not found: {INPUT_PLY}")
        orig_pcd = None
    else:
        orig_pcd = o3d.io.read_point_cloud(INPUT_PLY)
        # Downsample for display
        orig_pcd = orig_pcd.voxel_down_sample(voxel_size=0.02)
        orig_pcd.paint_uniform_color([0.55, 0.70, 0.85])   # light blue
        print(f"       {len(orig_pcd.points):,} points (downsampled for display)")

    # ── 5. Visualize ───────────────────────────────────────────────────────────
    WIN_W, WIN_H = 1000, 700

    # Build shared axis frame and reference grid from scene bounds
    if orig_pcd:
        bb_min = orig_pcd.get_min_bound()
        bb_max = orig_pcd.get_max_bound()
    else:
        bb_min = ground_mesh.get_min_bound()
        bb_max = ground_mesh.get_max_bound()

    scene_size  = float(np.linalg.norm(bb_max - bb_min))
    axis_size   = scene_size * 0.10          # 10% of scene diagonal
    grid_y      = float(bb_min[1])           # floor level
    grid_spacing = scene_size / 20.0         # ~20 lines across scene

    axis_frame = make_axis_frame(size=axis_size, origin=bb_min)
    ref_grid   = make_world_grid(bb_min, bb_max, y_level=grid_y,
                                 spacing=grid_spacing)

    # Left window — original scan
    print("\nOpening comparison windows — close both to exit.\n"
          "  Left  : original point cloud (blue)\n"
          "  Right : reconstructed scene  (ground=tan, rocks=green)")

    vis_orig = o3d.visualization.Visualizer()
    vis_orig.create_window(window_name="Original Point Cloud",
                           width=WIN_W, height=WIN_H, left=0, top=40)
    if orig_pcd:
        vis_orig.add_geometry(orig_pcd)
    vis_orig.add_geometry(axis_frame)
    vis_orig.add_geometry(ref_grid)
    vis_orig.poll_events()
    vis_orig.update_renderer()

    # Right window — reconstructed scene
    vis_recon = o3d.visualization.Visualizer()
    vis_recon.create_window(window_name="Reconstructed Scene",
                            width=WIN_W, height=WIN_H, left=WIN_W + 20, top=40)
    vis_recon.add_geometry(ground_mesh)
    for m in rock_meshes:
        vis_recon.add_geometry(m)
    vis_recon.add_geometry(axis_frame)
    vis_recon.add_geometry(ref_grid)
    vis_recon.poll_events()
    vis_recon.update_renderer()

    # Event loop
    open_orig  = True
    open_recon = True
    while open_orig or open_recon:
        if open_orig:
            if not vis_orig.poll_events():
                open_orig = False
            else:
                vis_orig.update_renderer()
        if open_recon:
            if not vis_recon.poll_events():
                open_recon = False
            else:
                vis_recon.update_renderer()

    vis_orig.destroy_window()
    vis_recon.destroy_window()
    print("\n✓ Done!")



if __name__ == "__main__":
    main()
