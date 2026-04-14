"""
run_experiments.py — Headless batch experiment runner
======================================================
Runs DEM+SQ at 3 resolutions and Draco at 2 configs on all Apollo rocks.
Saves rendered PNG images at each stage for thesis figures.

Usage:
    cd /home/kanav/workspaces/pc_comp_recons
    python3 src/run_experiments.py

Outputs:
    src/experiment_results.csv
    src/experiment_images/<rock_label>/<method>.png
"""

import csv
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import types
import importlib.util

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm

# Suppress Open3D warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
TOOLS_DIR      = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SRC   = os.path.join(TOOLS_DIR, "..", "src")
APOLLO_ROOT    = os.path.join(TOOLS_DIR, "..", "io_op", "IROS_DATASET", "MOON_SAMPLES")
OUTPUT_CSV     = os.path.join(TOOLS_DIR, "experiment_results.csv")
DRACO_ENCODER  = "/home/kanav/workspaces/draco/build/draco_encoder"
DRACO_DECODER  = "/home/kanav/workspaces/draco/build/draco_decoder"
STEP1_SCRIPT   = os.path.join(PIPELINE_SRC, "Step_1_sq_ems_fit.py")
STEP3_SCRIPT   = os.path.join(PIPELINE_SRC, "Step_3_dem_generation.py")
STEP4_SCRIPT   = os.path.join(PIPELINE_SRC, "Step_4_reconstruction.py")

DEM_RESOLUTIONS = [
    (180,   90),
    (360,  180),
    (720,  360),
    (1440, 720),
]

DRACO_CONFIGS = [
    {"name": "Draco Best Compression", "cl": 10, "qp": 4},
    {"name": "Draco Best Quality",     "cl":  0, "qp": 16},
]

IMAGES_DIR = os.path.join(TOOLS_DIR, "experiment_images")
IMG_W, IMG_H = 800, 600     # render resolution
SAMPLE_N = 100_000          # points per side for metric computation

# ─────────────────────────────────────────────────────────────────────────────
# MESH LOADING & METRICS
# ─────────────────────────────────────────────────────────────────────────────
def _build_rayscene(mesh):
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris  = np.asarray(mesh.triangles, dtype=np.int32)
    tm = o3d.t.geometry.TriangleMesh()
    tm.vertex["positions"] = o3d.core.Tensor(verts, dtype=o3d.core.Dtype.Float32)
    tm.triangle["indices"] = o3d.core.Tensor(tris,  dtype=o3d.core.Dtype.Int32)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tm)
    return scene, np.asarray(mesh.triangle_normals, dtype=np.float32)


def _load_geom(path):
    mesh = o3d.io.read_triangle_mesh(path)
    has_tris = not mesh.is_empty() and len(mesh.triangles) > 0
    if has_tris:
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        pcd = mesh.sample_points_uniformly(SAMPLE_N)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.normalize_normals()
        pts = np.asarray(pcd.points,  dtype=np.float32)
        nrm = np.asarray(pcd.normals, dtype=np.float32)
        scene, tri_n = _build_rayscene(mesh)
        return pts, nrm, tri_n, scene
    else:
        pcd = o3d.io.read_point_cloud(path)
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=20))
        pcd.normalize_normals()
        pts = np.asarray(pcd.points,  dtype=np.float32)
        nrm = np.asarray(pcd.normals, dtype=np.float32)
        if len(pts) > SAMPLE_N:
            idx = np.random.choice(len(pts), SAMPLE_N, replace=False)
            pts, nrm = pts[idx], nrm[idx]
        return pts, nrm, None, None


def compute_metrics(src_path, tgt_path):
    try:
        src_pts, src_nrm, _,       src_scene = _load_geom(src_path)
        tgt_pts, tgt_nrm, tgt_tri, tgt_scene = _load_geom(tgt_path)

        tgt_ref = tgt_scene if tgt_scene is not None else cKDTree(tgt_pts)
        src_ref = src_scene if src_scene is not None else cKDTree(src_pts)

        def _query(pts, ref, is_scene):
            if is_scene:
                res   = ref.compute_closest_points(
                    o3d.core.Tensor(pts, dtype=o3d.core.Dtype.Float32))
                dists = np.linalg.norm(pts - res["points"].numpy(), axis=1)
                pids  = res["primitive_ids"].numpy()
                return dists, pids
            else:
                d, idx = ref.query(pts)
                return d.astype(np.float32), idx

        fwd, tgt_pids = _query(src_pts, tgt_ref, tgt_scene is not None)
        bwd, _        = _query(tgt_pts, src_ref, src_scene is not None)
        bidir = np.concatenate([fwd, bwd])

        # Normal angle error
        s_n = src_nrm / (np.linalg.norm(src_nrm, axis=1, keepdims=True) + 1e-12)
        if tgt_scene is not None and tgt_tri is not None:
            t_n = tgt_tri[tgt_pids]
        else:
            _, idx_n = cKDTree(tgt_pts).query(src_pts)
            t_n = tgt_nrm[idx_n]
        t_n = t_n / (np.linalg.norm(t_n, axis=1, keepdims=True) + 1e-12)
        dot    = np.clip(np.abs(np.sum(s_n * t_n, axis=1)), 0.0, 1.0)
        angles = np.degrees(np.arccos(dot))

        return {
            "chamfer_mean": float(bidir.mean()),
            "chamfer_p95":  float(np.percentile(bidir, 95)),
            "chamfer_max":  float(bidir.max()),
            "hausdorff":    float(max(fwd.max(), bwd.max())),
            "na_mean":      float(angles.mean()),
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING
# ─────────────────────────────────────────────────────────────────────────────
def render_mesh(mesh_path, out_png, camera_params=None):
    """
    Render a mesh or point cloud to a PNG using Open3D offscreen rendering.
    Returns the camera_params used so subsequent renders use the same viewpoint.
    """
    try:
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        if mesh.is_empty() or len(mesh.triangles) == 0:
            pcd = o3d.io.read_point_cloud(mesh_path)
            geom = pcd
        else:
            mesh.orient_triangles()
            mesh.compute_vertex_normals()
            # If normals are pointing inward, flip them
            if not mesh.is_orientable():
                mesh.compute_vertex_normals()
            geom = mesh

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=IMG_W, height=IMG_H)
        vis.add_geometry(geom)

        # Apply stored camera params or use a consistent default view
        ctr = vis.get_view_control()
        if camera_params is not None:
            ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
        else:
            # Default: front-ish diagonal view
            ctr.set_zoom(0.7)
            ctr.set_front([0.5, -0.5, -0.7])
            ctr.set_lookat([0.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])

        # Render options
        opt = vis.get_render_option()
        opt.background_color = np.array([0.15, 0.15, 0.15])
        opt.mesh_show_back_face = True
        opt.light_on = True
        opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default

        vis.poll_events()
        vis.update_renderer()

        # Capture camera params from first render for reuse
        out_params = ctr.convert_to_pinhole_camera_parameters()

        vis.capture_screen_image(out_png, do_render=True)
        vis.destroy_window()

        return out_params

    except Exception as e:
        print(f"    ⚠ Render failed for {mesh_path}: {e}")
        return camera_params


# ─────────────────────────────────────────────────────────────────────────────
# POISSON RECONSTRUCTION (for point cloud PLY inputs)
# ─────────────────────────────────────────────────────────────────────────────
def poisson_reconstruct(ply_path, out_mesh_path, depth=9, normal_knn=30):
    """Reconstruct a mesh from a point cloud PLY via Poisson reconstruction."""
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"    Poisson: {len(pcd.points)} points → estimating normals...")
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=normal_knn))
    pcd.orient_normals_consistent_tangent_plane(k=normal_knn)
    print(f"    Poisson: running reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth)
    # Remove low-density vertices (outer shell artifacts)
    densities = np.asarray(densities)
    keep = densities > np.quantile(densities, 0.02)
    mesh.remove_vertices_by_mask(~keep)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    print(f"    Poisson: saved {len(mesh.triangles)} triangles → {out_mesh_path}")
    return out_mesh_path


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER (DEM+SQ)
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(obj_path, tmpdir, dem_w, dem_h):
    # Write a config override into tmpdir so step scripts use the right paths
    cfg_src = os.path.join(PIPELINE_SRC, "config.py")
    cfg_override = os.path.join(tmpdir, "config.py")
    with open(cfg_src) as f:
        cfg_text = f.read()
    with open(cfg_override, "w") as f:
        f.write(cfg_text)
        f.write(f"\nMAP_DIR = {repr(tmpdir)}\n")
        f.write(f"\nMAP_FOLDER = {repr(tmpdir)}\n")
        f.write(f"\nSINGLE_MESH_INPUT = {repr(obj_path)}\n")
        f.write(f"\nDEM_W = {dem_w}\n")
        f.write(f"\nDEM_H = {dem_h}\n")

    env = os.environ.copy()
    env["PC_HEADLESS"] = "1"
    env["PYTHONPATH"] = tmpdir + os.pathsep + PIPELINE_SRC + os.pathsep + env.get("PYTHONPATH", "")

    def _run(script):
        r = subprocess.run(
            [sys.executable, script],
            env=env, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            print(f"    ⚠ {os.path.basename(script)} failed:\n{(r.stdout+r.stderr)[-400:]}")
        return r.returncode == 0

    if not _run(STEP1_SCRIPT):
        return None, 0

    # Write source mesh as poisson_Cluster_1.ply (Step 3 expects it)
    poisson_ply = os.path.join(tmpdir, "poisson_Cluster_1.ply")
    if obj_path.lower().endswith(".ply"):
        # Input is a point cloud — run Poisson reconstruction
        poisson_reconstruct(obj_path, poisson_ply)
    else:
        src_mesh = o3d.io.read_triangle_mesh(obj_path)
        o3d.io.write_triangle_mesh(poisson_ply, src_mesh)

    if not _run(STEP3_SCRIPT):
        return None, 0
    if not _run(STEP4_SCRIPT):
        return None, 0

    sq_files    = glob.glob(os.path.join(tmpdir, "sq_fit_*.txt"))
    dem_files   = glob.glob(os.path.join(tmpdir, "dem_Cluster_*.npy"))
    msk_files   = glob.glob(os.path.join(tmpdir, "mask_Cluster_*.npy"))
    recon_plys  = glob.glob(os.path.join(tmpdir, "dem_recon_*.ply"))

    pip_size = (sum(os.path.getsize(f) for f in sq_files) +
                sum(os.path.getsize(f) for f in dem_files) +
                sum(os.path.getsize(f) for f in msk_files))
    recon_ply = recon_plys[0] if recon_plys else None
    return recon_ply, pip_size


# ─────────────────────────────────────────────────────────────────────────────
# DRACO RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_draco(obj_path, tmpdir, cl, qp):
    stem    = os.path.splitext(os.path.basename(obj_path))[0]
    drc_out = os.path.join(tmpdir, f"{stem}_cl{cl}_qp{qp}.drc")
    dec_out = os.path.join(tmpdir, f"{stem}_cl{cl}_qp{qp}_decoded.obj")

    # Draco needs a mesh — Poisson reconstruct if input is a point cloud PLY
    input_path = obj_path
    if obj_path.lower().endswith(".ply"):
        mesh_path = os.path.join(tmpdir, f"{stem}_poisson.ply")
        poisson_reconstruct(obj_path, mesh_path)
        input_path = mesh_path

    enc = subprocess.run(
        [DRACO_ENCODER, "-i", input_path, "-o", drc_out,
         "-cl", str(cl), "-qp", str(qp)],
        capture_output=True, text=True, timeout=300)
    if enc.returncode != 0 or not os.path.exists(drc_out):
        print(f"    ⚠ Draco encode failed: {enc.stderr[-200:]}")
        return None, 0

    drc_size = os.path.getsize(drc_out)

    dec = subprocess.run(
        [DRACO_DECODER, "-i", drc_out, "-o", dec_out],
        capture_output=True, text=True, timeout=120)
    if dec.returncode != 0 or not os.path.exists(dec_out):
        print(f"    ⚠ Draco decode failed: {dec.stderr[-200:]}")
        return None, drc_size

    return dec_out, drc_size


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def find_obj_files(root):
    result = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".obj"):
                result.append(os.path.join(dp, f))
    return sorted(result)


def rock_label(obj_path):
    """Extract a clean label from the Apollo path, e.g. 'Apollo 11 Rock 1'."""
    parts = obj_path.replace("\\", "/").split("/")
    try:
        apollo_idx = next(i for i, p in enumerate(parts) if p.startswith("APOLLO_"))
        mission = parts[apollo_idx].replace("_", " ").title()   # Apollo 11
        rock    = parts[apollo_idx + 1].replace("_", " ").title()  # Rock 1
        return f"{mission} {rock}"
    except Exception:
        return os.path.basename(os.path.dirname(obj_path))


def fmt_kb(n_bytes):
    return f"{n_bytes / 1024:.1f} KB"


def main():
    obj_files = find_obj_files(APOLLO_ROOT)
    print(f"Found {len(obj_files)} Apollo rock meshes\n")
    os.makedirs(IMAGES_DIR, exist_ok=True)

    rows = []

    rock_pbar = tqdm(obj_files, desc="Rocks", unit="rock", position=0)
    for obj_path in rock_pbar:
        orig_size = os.path.getsize(obj_path)
        label     = rock_label(obj_path)
        label_fs  = label.replace(" ", "_")   # filesystem-safe name
        rock_pbar.set_description(f"Rock: {label}")
        print(f"\n{'='*60}")
        print(f"Rock: {label}  ({orig_size/1024/1024:.1f} MB)")
        print(f"{'='*60}")

        rock_img_dir = os.path.join(IMAGES_DIR, label_fs)
        os.makedirs(rock_img_dir, exist_ok=True)

        row = {"Rock": label, "Original Size": fmt_kb(orig_size)}

        # For PLY point clouds, Poisson reconstruct once as shared ground truth
        if obj_path.lower().endswith(".ply"):
            gt_mesh_path = os.path.join(rock_img_dir, "poisson_gt.ply")
            print(f"  [Poisson GT] Reconstructing ground truth mesh...")
            poisson_reconstruct(obj_path, gt_mesh_path)
            gt_path = gt_mesh_path
        else:
            gt_path = obj_path

        # Render original mesh first — capture camera params for consistent viewpoint
        orig_png    = os.path.join(rock_img_dir, "original.png")
        print(f"  [Original] Rendering...")
        cam_params  = render_mesh(gt_path, orig_png)
        print(f"    Saved: {orig_png}")

        # ── DEM+SQ at 3 resolutions ───────────────────────────────────────────
        for dem_w, dem_h in tqdm(DEM_RESOLUTIONS, desc="  DEM+SQ", unit="res", position=1, leave=False):
            tag = f"{dem_w}x{dem_h}"
            print(f"  [DEM+SQ {tag}] Running pipeline...")
            tmpdir = tempfile.mkdtemp(prefix=f"dem_{dem_w}_")
            try:
                recon_ply, pip_size = run_pipeline(obj_path, tmpdir, dem_w, dem_h)
                if recon_ply and os.path.exists(recon_ply):
                    m = compute_metrics(gt_path, recon_ply)
                    ratio = orig_size / pip_size if pip_size > 0 else 0
                    print(f"    Size={fmt_kb(pip_size)}  Ratio={ratio:.1f}x  "
                          f"Chamfer={m.get('chamfer_mean', float('nan')):.6f}m  "
                          f"angle={m.get('na_mean', float('nan')):.2f}deg")
                    row[f"DEM+SQ {tag} Size"]      = fmt_kb(pip_size)
                    row[f"DEM+SQ {tag} Ratio"]     = f"{ratio:.1f}x"
                    row[f"DEM+SQ {tag} Chamfer"]   = f"{m.get('chamfer_mean', float('nan')):.6f}"
                    row[f"DEM+SQ {tag} p95"]       = f"{m.get('chamfer_p95',  float('nan')):.6f}"
                    row[f"DEM+SQ {tag} Hausdorff"] = f"{m.get('hausdorff',    float('nan')):.6f}"
                    row[f"DEM+SQ {tag} anorm"]     = f"{m.get('na_mean',      float('nan')):.2f}"

                    # Render reconstruction at same viewpoint
                    out_png = os.path.join(rock_img_dir, f"dem_sq_{tag}.png")
                    render_mesh(recon_ply, out_png, cam_params)
                    print(f"    Saved: {out_png}")
                else:
                    print(f"    Pipeline failed")
                    for k in ["Size", "Ratio", "Chamfer", "p95", "anorm"]:
                        row[f"DEM+SQ {tag} {k}"] = "FAILED"
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        # ── Draco ─────────────────────────────────────────────────────────────
        for cfg in tqdm(DRACO_CONFIGS, desc="  Draco", unit="cfg", position=1, leave=False):
            name, cl, qp = cfg["name"], cfg["cl"], cfg["qp"]
            tag = name.replace(" ", "_").lower()
            print(f"  [{name}] Running Draco cl={cl} qp={qp}...")
            tmpdir = tempfile.mkdtemp(prefix=f"draco_{cl}_")
            try:
                dec_out, drc_size = run_draco(obj_path, tmpdir, cl, qp)
                if dec_out and os.path.exists(dec_out):
                    m     = compute_metrics(gt_path, dec_out)
                    ratio = orig_size / drc_size if drc_size > 0 else 0
                    print(f"    Size={fmt_kb(drc_size)}  Ratio={ratio:.1f}x  "
                          f"Chamfer={m.get('chamfer_mean', float('nan')):.6f}m  "
                          f"angle={m.get('na_mean', float('nan')):.2f}deg")
                    row[f"{name} Size"]      = fmt_kb(drc_size)
                    row[f"{name} Ratio"]     = f"{ratio:.1f}x"
                    row[f"{name} Chamfer"]   = f"{m.get('chamfer_mean', float('nan')):.6f}"
                    row[f"{name} p95"]       = f"{m.get('chamfer_p95',  float('nan')):.6f}"
                    row[f"{name} Hausdorff"] = f"{m.get('hausdorff',    float('nan')):.6f}"
                    row[f"{name} anorm"]     = f"{m.get('na_mean',      float('nan')):.2f}"

                    # Render Draco reconstruction at same viewpoint
                    out_png = os.path.join(rock_img_dir, f"{tag}.png")
                    render_mesh(dec_out, out_png, cam_params)
                    print(f"    Saved: {out_png}")
                else:
                    print(f"    Draco failed")
                    for k in ["Size", "Ratio", "Chamfer", "p95", "anorm"]:
                        row[f"{name} {k}"] = "FAILED"
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        rows.append(row)
        print()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    if rows:
        fieldnames = list(rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
