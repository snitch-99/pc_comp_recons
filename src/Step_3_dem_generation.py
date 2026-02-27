import os
import glob
import sys
import numpy as np
import open3d as o3d
from config import (
    MAP_DIR as IO_OP_DIR,
    DEM_W as W, DEM_H as H,
    USE_LOCAL_SUPPORT_FILTER, LOCAL_SUPPORT_K,
    SAVE_RECON_POINTS_PLY,
    EMS_CIRCUMSCRIBE,
)




# =============================================================================
# SUPERQUADRIC UTILITIES  (pure functions — no globals)
# =============================================================================

def signed_pow(x: np.ndarray, e: float) -> np.ndarray:
    return np.sign(x) * (np.abs(x) ** e)


def sq_point_canonical(eta, omega, a, e1, e2):
    ce = signed_pow(np.cos(eta), e2)
    se = signed_pow(np.sin(eta), e2)
    cw = signed_pow(np.cos(omega), e1)
    sw = signed_pow(np.sin(omega), e1)
    x = a[0] * cw * ce
    y = a[1] * cw * se
    z = a[2] * sw
    return np.stack([x, y, z], axis=-1)   # (..., 3)


def sq_normal_canonical(p, a, e1, e2):
    """Gradient of implicit SQ function, normalized."""
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    a1, a2, a3 = a[0], a[1], a[2]
    eps = 1e-12
    X = np.abs(x / (a1 + eps))
    Y = np.abs(y / (a2 + eps))
    Z = np.abs(z / (a3 + eps))
    e1 = max(e1, 0.1)   # clamp — 2/e blows up below this
    e2 = max(e2, 0.1)
    p_xy = 2.0 / e2
    p_z  = 2.0 / e1
    q    = e2 / e1
    A    = X**p_xy + Y**p_xy
    dA_dx = p_xy * (X**(p_xy - 1.0)) * (np.sign(x) / (a1 + eps))
    dA_dy = p_xy * (Y**(p_xy - 1.0)) * (np.sign(y) / (a2 + eps))
    dF_dx = q * (A**(q - 1.0)) * dA_dx
    dF_dy = q * (A**(q - 1.0)) * dA_dy
    dF_dz = p_z * (Z**(p_z - 1.0)) * (np.sign(z) / (a3 + eps))
    n   = np.stack([dF_dx, dF_dy, dF_dz], axis=-1)
    nn  = np.linalg.norm(n, axis=-1, keepdims=True) + eps
    return n / nn


def to_world(p_can, R, center):
    return (p_can @ R.T) + center


def n_to_world(n_can, R):
    n_w = n_can @ R.T
    return n_w / (np.linalg.norm(n_w, axis=-1, keepdims=True) + 1e-12)


# =============================================================================
# TXT PARSER  — reads sq_fit_Cluster_N.txt produced by Step 1
# =============================================================================

def parse_sq_fit_txt(txt_path: str):
    """
    Returns dict with keys:
        params   : [ax, ay, az, e1, e2]
        center   : np.array (3,)
        R        : np.array (3,3)   world -> canonical
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    result = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Best-fit Superquadric parameters"):
            vals = list(map(float, lines[i + 1].strip().split()))
            result["params"] = vals
            i += 2
            continue

        if line == "Center (world):":
            vals = list(map(float, lines[i + 1].strip().split()))
            result["center"] = np.array(vals, dtype=np.float64)
            i += 2
            continue

        if line.startswith("Rotation R_init"):
            rows = []
            for k in range(1, 4):
                rows.append(list(map(float, lines[i + k].strip().split())))
            result["R"] = np.array(rows, dtype=np.float64)
            i += 4
            continue

        i += 1

    return result


# =============================================================================
# LOCAL SUPPORT MASK FILTER
# =============================================================================

def filter_mask_by_local_support(mask_hw: np.ndarray, K: int = 3) -> np.ndarray:
    H_, W_ = mask_hw.shape
    M = (mask_hw > 0).astype(np.uint8)
    out = M.copy()
    for j in range(H_):
        for i in range(W_):
            if M[j, i] == 0:
                continue
            cnt = 0
            for jj in (j - 1, j, j + 1):
                if jj < 0 or jj >= H_:
                    continue
                for ii in ((i-1) % W_, i, (i+1) % W_):
                    if jj == j and ii == i:
                        continue
                    cnt += M[jj, ii]
            if cnt < K:
                out[j, i] = 0
    return out


# =============================================================================
# DEM COMPUTATION FOR ONE CLUSTER
# =============================================================================

def compute_dem(rock_mesh_path, sq_params, sq_center, sq_R,
                out_dem_path, out_mask_path, out_recon_path=None):
    """
    Raycasts from SQ surface → rock Poisson mesh to compute perpendicular DEM.

    Args:
        rock_mesh_path : path to poisson_Cluster_N.ply
        sq_params      : [ax, ay, az, e1, e2]
        sq_center      : (3,) world-space center
        sq_R           : (3,3) world → canonical rotation
        out_dem_path   : save dem as .npy
        out_mask_path  : save mask as .npy
        out_recon_path : if given, save reconstructed points .ply
    """
    a  = np.array(sq_params[:3])
    e1 = sq_params[3]
    e2 = sq_params[4]

    # Load rock mesh
    rock_legacy = o3d.io.read_triangle_mesh(rock_mesh_path)
    if rock_legacy.is_empty():
        raise RuntimeError(f"Rock mesh is empty: {rock_mesh_path}")
    rock_legacy.compute_triangle_normals()
    rock_t = o3d.t.geometry.TriangleMesh.from_legacy(rock_legacy)
    scene  = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(rock_t)

    # SQ grid
    eta   = -np.pi    + (np.arange(W) + 0.5) * (2.0 * np.pi / W)
    omega = -0.5*np.pi + (np.arange(H) + 0.5) * (np.pi / H)
    ETA, OMEGA = np.meshgrid(eta, omega)

    P_can = sq_point_canonical(ETA, OMEGA, a, e1, e2)        # (H,W,3)
    N_can = sq_normal_canonical(P_can, a, e1, e2)            # (H,W,3)

    # Ensure outward normals
    dots = np.sum(P_can * N_can, axis=-1)
    N_can = np.where(dots[..., None] < 0.0, -N_can, N_can)

    P_w = to_world(P_can, sq_R, sq_center).reshape(-1, 3)
    N_w = n_to_world(N_can, sq_R).reshape(-1, 3)

    origins = P_w.astype(np.float32)

    if EMS_CIRCUMSCRIBE:
        # ── Balloon mode: SQ wraps the rock from outside ──────────────────────
        # All rock points are guaranteed inside the SQ (verified by Step 1).
        # Shoot rays INWARD (-N) → they always hit the rock surface.
        # D = +t_neg  (positive depth from SQ surface inward to rock surface)
        rays_neg = np.concatenate([origins, -N_w.astype(np.float32)], axis=1)
        t_neg    = scene.cast_rays(
            o3d.core.Tensor(rays_neg, dtype=o3d.core.Dtype.Float32)
        )["t_hit"].numpy()

        hit = np.isfinite(t_neg)
        D   = np.where(hit, t_neg, 0.0).astype(np.float32)   # positive depth
        M   = hit.astype(np.uint8)
        print(f"  [DEM] Circumscribed mode: {hit.sum():,} / {hit.size:,} cells hit "
              f"({100*hit.mean():.1f}%)")
    else:
        # ── Bidirectional mode: SQ may be inside or partially outside rock ────
        rays_pos = np.concatenate([origins,  N_w.astype(np.float32)], axis=1)
        rays_neg = np.concatenate([origins, -N_w.astype(np.float32)], axis=1)

        t_pos = scene.cast_rays(o3d.core.Tensor(rays_pos, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()
        t_neg = scene.cast_rays(o3d.core.Tensor(rays_neg, dtype=o3d.core.Dtype.Float32))["t_hit"].numpy()

        hit_pos = np.isfinite(t_pos)
        hit_neg = np.isfinite(t_neg)
        d_pos   = t_pos
        d_neg   = -t_neg

        D = np.zeros_like(d_pos, dtype=np.float32)
        M = np.zeros_like(d_pos, dtype=np.uint8)

        both     = hit_pos & hit_neg
        only_pos = hit_pos & ~hit_neg
        only_neg = hit_neg & ~hit_pos
        pick_pos = both & (np.abs(d_pos) >= np.abs(d_neg))   # larger |t| = outer surface
        pick_neg = both & ~pick_pos

        D[pick_pos] = d_pos[pick_pos]
        D[pick_neg] = d_neg[pick_neg]
        D[only_pos] = d_pos[only_pos]
        D[only_neg] = d_neg[only_neg]
        M[both | only_pos | only_neg] = 1

    D_hw = D.reshape(H, W)
    M_hw = M.reshape(H, W)

    if USE_LOCAL_SUPPORT_FILTER:
        M_hw = filter_mask_by_local_support(M_hw, K=LOCAL_SUPPORT_K)
        D_hw = np.where(M_hw > 0, D_hw, 0.0).astype(np.float32)

    np.save(out_dem_path,  D_hw)
    np.save(out_mask_path, M_hw)
    print(f"  [INFO] DEM saved:  {out_dem_path}")
    print(f"  [INFO] Mask saved: {out_mask_path}")
    print(f"  [INFO] Valid cells: {int(M_hw.sum())} / {H*W}")

    # Reconstructed points
    if out_recon_path and SAVE_RECON_POINTS_PLY:
        D_flat  = D_hw.reshape(-1).astype(np.float64)
        M_flat  = M_hw.reshape(-1) > 0
        P_recon = P_w.astype(np.float64) + D_flat[:, None] * N_w.astype(np.float64)
        P_recon = P_recon[M_flat]
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(P_recon)
        o3d.io.write_point_cloud(out_recon_path, pcd_out)
        print(f"  [INFO] Recon pts: {out_recon_path}")

    return D_hw, M_hw


# =============================================================================
# MAIN — loop all clusters
# =============================================================================

if __name__ == "__main__":

    cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))
    if not cluster_files:
        print(f"[ERROR] No Cluster_*.ply found in {IO_OP_DIR}")
        sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s).\n")
    vis_data = []   # (rock_mesh, recon_pcd, name)

    for i, _ in enumerate(cluster_files, start=1):
        name = f"Cluster_{i}"
        print(f"{'═'*50}")
        print(f"  [{i}/{len(cluster_files)}]  DEM for {name}")
        print(f"{'═'*50}")

        txt_path   = os.path.join(IO_OP_DIR, f"sq_fit_{name}.txt")
        mesh_path  = os.path.join(IO_OP_DIR, f"poisson_{name}.ply")
        dem_path   = os.path.join(IO_OP_DIR, f"dem_{name}.npy")
        mask_path  = os.path.join(IO_OP_DIR, f"mask_{name}.npy")
        recon_path = os.path.join(IO_OP_DIR, f"recon_pts_{name}.ply")

        # Check files exist
        missing = [p for p in [txt_path, mesh_path] if not os.path.exists(p)]
        if missing:
            print(f"  [SKIP] Missing files: {missing}\n")
            continue

        # Parse SQ params from txt
        sq = parse_sq_fit_txt(txt_path)
        print(f"  SQ params: {[f'{v:.4f}' for v in sq['params']]}")
        print(f"  Center:    {sq['center'].round(4).tolist()}")

        try:
            D_hw, M_hw = compute_dem(
                rock_mesh_path=mesh_path,
                sq_params=sq["params"],
                sq_center=sq["center"],
                sq_R=sq["R"],
                out_dem_path=dem_path,
                out_mask_path=mask_path,
                out_recon_path=recon_path,
            )
        except Exception as e:
            print(f"  [ERROR] {e}\n")
            continue

        # Collect for visualization
        rock_mesh = o3d.io.read_triangle_mesh(mesh_path)
        rock_mesh.paint_uniform_color([0.9, 0.35, 0.2])   # orange-red

        recon_pcd = o3d.io.read_point_cloud(recon_path)
        recon_pcd.paint_uniform_color([0.2, 0.8, 0.2])    # green

        vis_data.append((rock_mesh, recon_pcd, name))
        print()

    if not vis_data:
        print("[ERROR] No DEMs were produced.")
        sys.exit(1)

    # ── Open one non-blocking window per cluster ──────────────────────────────
    print(f"Opening {len(vis_data)} window(s) — close all to exit.")

    WIN_W, WIN_H = 900, 650
    PAD          = 20
    visualizers  = []

    for idx, (rock_mesh, recon_pcd, name) in enumerate(vis_data):
        col  = idx % 2
        row  = idx // 2
        vis  = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"DEM — {name}",
                          width=WIN_W, height=WIN_H,
                          left=PAD + col*(WIN_W+PAD),
                          top=PAD  + row*(WIN_H+PAD))
        vis.add_geometry(rock_mesh)
        vis.add_geometry(recon_pcd)
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

    print("\n✓ All DEMs generated!")
