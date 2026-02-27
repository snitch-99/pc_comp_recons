import os
import glob
import sys
import numpy as np
import open3d as o3d
from config import (
    MAP_DIR as IO_OP_DIR,
    DEM_W as W, DEM_H as H,
    DEM_RECON_POISSON_DEPTH as POISSON_DEPTH,
    DEM_RECON_NORMAL_KNN as NORMAL_KNN,
)



# =============================================================================
# SUPERQUADRIC HELPERS  (pure — no globals)
# =============================================================================

def signed_pow(x, e):
    return np.sign(x) * (np.abs(x) ** e)


def sq_point_canonical(eta, omega, a, e1, e2):
    ce = signed_pow(np.cos(eta), e2)
    se = signed_pow(np.sin(eta), e2)
    cw = signed_pow(np.cos(omega), e1)
    sw = signed_pow(np.sin(omega), e1)
    return np.stack([a[0]*cw*ce, a[1]*cw*se, a[2]*sw], axis=-1)


def sq_normal_canonical(p, a, e1, e2):
    x, y, z = p[...,0], p[...,1], p[...,2]
    a1, a2, a3 = a[0], a[1], a[2]
    eps = 1e-12
    X = np.abs(x/(a1+eps));  Y = np.abs(y/(a2+eps));  Z = np.abs(z/(a3+eps))
    p_xy = 2.0/(e2+eps);     p_z = 2.0/(e1+eps);      q = (e2+eps)/(e1+eps)
    A    = X**p_xy + Y**p_xy
    dF_dx = q*(A**(q-1)) * p_xy*(X**(p_xy-1))*(np.sign(x)/(a1+eps))
    dF_dy = q*(A**(q-1)) * p_xy*(Y**(p_xy-1))*(np.sign(y)/(a2+eps))
    dF_dz = p_z*(Z**(p_z-1))*(np.sign(z)/(a3+eps))
    n = np.stack([dF_dx, dF_dy, dF_dz], axis=-1)
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + eps)


# =============================================================================
# TXT PARSER  (reused from Step 3)
# =============================================================================

def parse_sq_fit_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    result = {}
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
            rows = [list(map(float, lines[i+k].strip().split())) for k in range(1,4)]
            result["R"] = np.array(rows, dtype=np.float64)
            i += 4; continue
        i += 1
    return result


# =============================================================================
# PER-CLUSTER RECONSTRUCTION
# =============================================================================

def reconstruct_from_dem(sq_params, sq_center, sq_R,
                         dem_path, mask_path,
                         out_mesh_path):
    """
    Loads DEM + mask, recreates the SQ surface grid, offsets each valid
    point by its DEM value along the outward normal, then runs Poisson
    reconstruction on those points.

    Returns the final mesh.
    """
    a  = np.array(sq_params[:3])
    e1, e2 = sq_params[3], sq_params[4]

    D = np.load(dem_path).astype(np.float64)
    M = np.load(mask_path).astype(np.uint8)
    valid   = M > 0
    n_valid = int(valid.sum())
    print(f"  [INFO] Valid DEM cells: {n_valid} / {H*W}")
    if n_valid < 500:
        print("  [WARN] Very few valid points — Poisson may be unstable.")

    # Recreate SQ grid (must match Step 3 exactly)
    eta   = -np.pi    + (np.arange(W)+0.5)*(2*np.pi/W)
    omega = -0.5*np.pi+ (np.arange(H)+0.5)*(np.pi/H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = sq_point_canonical(ETA, OMEGA, a, e1, e2)
    N_can = sq_normal_canonical(P_can, a, e1, e2)

    dots  = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[...,None] < 0., -N_can, N_can)

    P_w = (P_can @ sq_R.T) + sq_center          # (H,W,3)
    N_w = N_can @ sq_R.T                         # (H,W,3)
    N_w = N_w / (np.linalg.norm(N_w, axis=-1, keepdims=True) + 1e-12)

    # Displaced points
    P_recon = P_w + D[...,None] * N_w            # (H,W,3)
    pts = P_recon[valid]                          # (N,3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    print(f"  [INFO] DEM point cloud size: {len(pts):,}")

    # Estimate normals
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=NORMAL_KNN))
    pcd.normalize_normals()
    pcd.orient_normals_consistent_tangent_plane(k=NORMAL_KNN)

    # Poisson
    print(f"  [INFO] Poisson (depth={POISSON_DEPTH}) ...")
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH)
    mesh.compute_vertex_normals()

    # Crop + clean
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


# =============================================================================
# MAIN — loop all clusters
# =============================================================================

if __name__ == "__main__":

    cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))
    if not cluster_files:
        print(f"[ERROR] No Cluster_*.ply found in {IO_OP_DIR}")
        sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s).\n")
    vis_data = []   # (poisson_mesh, dem_mesh, name)

    for i, _ in enumerate(cluster_files, start=1):
        name = f"Cluster_{i}"
        print(f"{'═'*50}")
        print(f"  [{i}/{len(cluster_files)}]  Reconstructing {name}")
        print(f"{'═'*50}")

        txt_path      = os.path.join(IO_OP_DIR, f"sq_fit_{name}.txt")
        dem_path      = os.path.join(IO_OP_DIR, f"dem_{name}.npy")
        mask_path     = os.path.join(IO_OP_DIR, f"mask_{name}.npy")
        poisson_path  = os.path.join(IO_OP_DIR, f"poisson_{name}.ply")   # from Step 2
        out_mesh_path = os.path.join(IO_OP_DIR, f"dem_recon_{name}.ply")

        missing = [p for p in [txt_path, dem_path, mask_path, poisson_path]
                   if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] Missing: {[os.path.basename(m) for m in missing]}\n")
            continue

        sq = parse_sq_fit_txt(txt_path)

        try:
            dem_mesh = reconstruct_from_dem(
                sq_params=sq["params"],
                sq_center=sq["center"],
                sq_R=sq["R"],
                dem_path=dem_path,
                mask_path=mask_path,
                out_mesh_path=out_mesh_path,
            )
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            continue

        # Prepare display objects
        poisson_mesh = o3d.io.read_triangle_mesh(poisson_path)
        poisson_mesh.compute_vertex_normals()
        poisson_mesh.paint_uniform_color([0.9, 0.35, 0.2])  # orange-red = Step 2 mesh
        dem_mesh.paint_uniform_color([0.2, 0.75, 0.3])      # green      = DEM-based mesh

        vis_data.append((poisson_mesh, dem_mesh, name))
        print()

    if not vis_data:
        print("[ERROR] No meshes produced.")
        sys.exit(1)

    # ── Open one non-blocking window per cluster ──────────────────────────────
    print(f"Opening {len(vis_data)} window(s) — close all to exit.")
    print("  Red/orange = Poisson mesh (Step 2)")
    print("  Green      = DEM-reconstructed mesh (Step 4)")

    WIN_W, WIN_H = 900, 650
    PAD          = 20
    visualizers  = []

    for idx, (poisson_mesh, dem_mesh, name) in enumerate(vis_data):
        col = idx % 2;  row = idx // 2
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"DEM Recon — {name}",
                          width=WIN_W, height=WIN_H,
                          left=PAD + col*(WIN_W+PAD),
                          top=PAD  + row*(WIN_H+PAD))
        vis.add_geometry(poisson_mesh)
        vis.add_geometry(dem_mesh)
        vis.get_render_option().mesh_show_back_face = True
        vis.poll_events()
        vis.update_renderer()
        visualizers.append(vis)

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

    print("\n✓ All DEM-based meshes saved!")
