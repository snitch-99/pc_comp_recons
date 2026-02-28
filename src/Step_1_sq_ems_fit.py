import numpy as np
import open3d as o3d
import scipy.optimize
import scipy.signal
import scipy.ndimage
import copy
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils


# --- CONFIGURATION PARAMETERS ---
SAFE_MIN_VAL          = 0.01       # Epsilon for ax,ay,az
SAFE_MIN_EXP          = 0.1        # Minimum for e1,e2 — below this 2/e blows up in normals
OUTLIER_RATIO_DEFAULT = 0.3
GUM_VOLUME_SCALE      = 1.5
W0_PRIOR_DEFAULT      = 0.5

# Optimization Bounds [ax, ay, az, e1, e2]
OPTIM_BOUNDS_LOWER = np.array([SAFE_MIN_VAL, SAFE_MIN_VAL, SAFE_MIN_VAL, SAFE_MIN_EXP, SAFE_MIN_EXP])
OPTIM_BOUNDS_UPPER = np.array([np.inf, np.inf, np.inf, 3.0, 3.0])

# EMS Loop Control
MAX_EMS_LOOPS    = 5       # Maximum number of restarts if S-step finds better candidates
CONVERGENCE_TOL  = 1e-4    # Termination threshold for parameter change
INLIER_THRESHOLD = 0.01    # Distance threshold (meters) for analyzing inliers

class Superquadric:
    """
    Represents a Superquadric surface.
    Parameters: [ax, ay, az, e1, e2]
    """
    def __init__(self, params=[1.0, 1.0, 1.0, 1.0, 1.0]):
        self.ax, self.ay, self.az, self.e1, self.e2 = params
        # Avoid numerical instability
        self.e1 = max(self.e1, SAFE_MIN_VAL)
        self.e2 = max(self.e2, SAFE_MIN_VAL)
        self.ax = max(self.ax, SAFE_MIN_VAL)
        self.ay = max(self.ay, SAFE_MIN_VAL)
        self.az = max(self.az, SAFE_MIN_VAL)

    def implicit_function(self, points):
        """
        F(x) = ((|x/ax|^(2/e2) + |y/ay|^(2/e2))^(e2/e1) + |z/az|^(2/e1))
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        term1_inner = (np.abs(x / self.ax)**(2 / self.e2)) + (np.abs(y / self.ay)**(2 / self.e2))
        term1 = term1_inner**(self.e2 / self.e1)
        term2 = np.abs(z / self.az)**(2 / self.e1)
        
        return term1 + term2

    def radial_distance_approximation(self, points):
        """
        Approximates the closest point on the surface using the radial intersection.
        mu_s = x * F(x)^(-e1/2)
        """
        F = self.implicit_function(points)
        factors = np.power(F, -self.e1 / 2.0)
        matched_points = points * factors[:, np.newaxis]
        return matched_points

    def inside_outside_function(self, points):
        return self.implicit_function(points)

    @staticmethod
    def _fexp(t: np.ndarray, e: float) -> np.ndarray:
        """Signed power: sign(t)*|t|^e."""
        return np.sign(t) * (np.abs(t) ** e)

    def sample_surface(self, nu: int = 200, nv: int = 100, include_poles: bool = True):
        """Sample the SQ surface with a regular (eta, omega) grid.

        Returns:
            vertices (V,3), faces (F,3) in canonical (OBB-aligned) frame.
        """
        nu = max(int(nu), 4)
        nv = max(int(nv), 3)

        # Use endpoint=False for eta to avoid seam duplication.
        etas = np.linspace(-np.pi, np.pi, nu, endpoint=False)
        if include_poles:
            omegas = np.linspace(-np.pi / 2.0, np.pi / 2.0, nv, endpoint=True)
        else:
            eps = 1e-6
            omegas = np.linspace(-np.pi / 2.0 + eps, np.pi / 2.0 - eps, nv, endpoint=True)

        ce = np.cos(etas)
        se = np.sin(etas)

        V = np.zeros((nu * nv, 3), dtype=np.float64)
        for j, om in enumerate(omegas):
            co = np.cos(om)
            so = np.sin(om)

            fx = self._fexp(co, self.e1)
            fz = self._fexp(so, self.e1)

            x = self.ax * fx * self._fexp(ce, self.e2)
            y = self.ay * fx * self._fexp(se, self.e2)
            z = self.az * fz * np.ones_like(x)

            start = j * nu
            V[start:start + nu, 0] = x
            V[start:start + nu, 1] = y
            V[start:start + nu, 2] = z

        faces = []
        for j in range(nv - 1):
            for i in range(nu):
                i2 = (i + 1) % nu
                v00 = j * nu + i
                v10 = j * nu + i2
                v01 = (j + 1) * nu + i
                v11 = (j + 1) * nu + i2
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])
        F = np.asarray(faces, dtype=np.int32)
        return V, F


class EMSFitter:
    def __init__(self, pcd, outlier_ratio=OUTLIER_RATIO_DEFAULT, init_type='BBOX',
                 w_o_prior=W0_PRIOR_DEFAULT):

        self.pcd_original = pcd
        self.points_original = np.asarray(pcd.points)
        
        # Pre-align using OBB
        obb = pcd.get_oriented_bounding_box()
        self.center = obb.center
        self.R_init = obb.R
        self.extent = obb.extent
        
        # Canonical points
        self.points = (self.points_original - self.center) @ self.R_init
        
        # Initial guess: OBB half-extents as starting axes, ellipsoid shape
        self.params = [self.extent[0]/2, self.extent[1]/2, self.extent[2]/2, 1.0, 1.0]
        self.sigma_sq = np.mean(self.extent)**2 * 0.1
        
        # GUM Model params
        self.w_0 = 0
        self.V = np.prod(self.extent * GUM_VOLUME_SCALE)
        self.p_outlier = 1.0 / self.V
        self.w_o_prior = w_o_prior

    def e_step(self):
        """Estimate posterior probability of each point being an inlier (z=1)."""
        sq = Superquadric(self.params)
        mu_s = sq.radial_distance_approximation(self.points)
        diff = self.points - mu_s
        sq_dist = np.sum(diff**2, axis=1)
        
        norm_factor = (2 * np.pi * self.sigma_sq) ** (-1.5)
        likelihood_inlier = norm_factor * np.exp(-sq_dist / (2 * self.sigma_sq))
        likelihood_outlier = self.p_outlier
        
        numerator = likelihood_inlier * (1 - self.w_o_prior)
        denominator = numerator + likelihood_outlier * self.w_o_prior
        z_prob = numerator / (denominator + 1e-12)
        return z_prob, mu_s

    def m_step(self, z_prob, mu_s):
        """Maximize Likelihood w.r.t params and sigma."""
        def loss(params_optim):
            sq = Superquadric(params_optim)
            mu_optim = sq.radial_distance_approximation(self.points)
            dists_sq = np.sum((self.points - mu_optim)**2, axis=1)
            term1 = np.sum(z_prob * dists_sq) / (2 * self.sigma_sq)
            return term1

        # Use Global Logic for bounds if needed, but least_squares requires direct passing
        lower = OPTIM_BOUNDS_LOWER
        upper = OPTIM_BOUNDS_UPPER
        
        self.params = np.clip(self.params, lower, upper)
        
        res = scipy.optimize.least_squares(
            lambda p: np.sqrt(z_prob) * np.linalg.norm(self.points - Superquadric(p).radial_distance_approximation(self.points), axis=1),
            self.params, bounds=(lower, upper), method='trf'
        )
        self.params = res.x
        
        sq_updated = Superquadric(self.params)
        mu_updated = sq_updated.radial_distance_approximation(self.points)
        dists_sq = np.sum((self.points - mu_updated)**2, axis=1)
        sum_z = np.sum(z_prob)
        self.sigma_sq = np.sum(z_prob * dists_sq) / (3 * sum_z)
        self.w_o_prior = 1.0 - (sum_z / len(self.points))
        return self.params, self.sigma_sq

    def s_step(self):
        """Generates candidate parameters based on geometric similarities."""
        current_loss = self.calculate_loss(self.params)
        best_params = copy.deepcopy(self.params)
        best_loss = current_loss
        found_better = False
        candidates = []

        ax, ay, az, e1, e2 = self.params
        # Swap Z <-> X
        candidates.append([az, ay, ax, e2, e1])
        # Swap Z <-> Y
        candidates.append([ax, az, ay, e2, e1])

        # Duality
        if e2 < 1.0: 
            e2_new = 2.0 - e2
            candidates.append([ax * np.sqrt(2), ay * np.sqrt(2), az, e1, e2_new])
        elif e2 > 1.0:
            e2_new = 2.0 - e2
            candidates.append([ax / np.sqrt(2), ay / np.sqrt(2), az, e1, e2_new])

        for cand in candidates:
            cand = [max(c, SAFE_MIN_VAL) for c in cand]
            loss = self.calculate_loss(cand)
            if loss < best_loss:
                best_loss = loss
                best_params = cand
                found_better = True
        
        if found_better:
            self.params = best_params
            return True
        return False

    def calculate_loss(self, params):
        sq = Superquadric(params)
        mu = sq.radial_distance_approximation(self.points)
        dists_sq = np.sum((self.points - mu)**2, axis=1)
        return np.sum(dists_sq)

    def fit(self, max_iters=100, external_pbar=None):
        """
        Executes the EMS (Expectation-Maximization-Switching) Loop.
        1. Run EM until convergence.
        2. Try Switching parameters.
        3. If Switch improves result, Restart EM.
        """
        ems_converged = False
        loop_count = 0
        
        while not ems_converged and loop_count < MAX_EMS_LOOPS:
            loop_count += 1
            
            # 1. EM Phase
            if external_pbar:
                external_pbar.reset(total=max_iters)
                external_pbar.set_description(f"EMS Loop {loop_count} (EM)")
            
            self._run_em_to_convergence(max_iters, external_pbar)

            # 2. S Phase (Switching)
            switched = self.s_step()
            
            if not switched:
                ems_converged = True
                print("Optimization Finished (No better switch found).")
            else:
                print("S-Step switched parameters. Restarting EM.")
        
        return Superquadric(self.params)

    def _run_em_to_convergence(self, max_iters, pbar):
        """Inner Loop: Runs EM until parameters stabilize or max_iters reached."""
        local_pbar = None
        if pbar is None:
             local_pbar = tqdm(range(max_iters), desc="EM Inner Loop", leave=False)
        
        prev_loss = float('inf')

        for i in range(max_iters):
            # E-Step
            z_prob, mu_s = self.e_step()
            
            # M-Step
            self.m_step(z_prob, mu_s)
            
            # Convergence Check (Loss Stagnation)
            curr_loss = self.calculate_loss(self.params)
            loss_change = abs(prev_loss - curr_loss)
            
            # Progress Update
            status = {'SigmaSq': f"{self.sigma_sq:.2g}", 'dLoss': f"{loss_change:.2g}"}
            if pbar:
                pbar.set_postfix(status)
                pbar.update(1)
            elif local_pbar:
                local_pbar.set_postfix(status)
                local_pbar.update(1)

            if loss_change < CONVERGENCE_TOL:
                break
                
            prev_loss = curr_loss
            
        if local_pbar:
            local_pbar.close()

    def execute(self, max_iters=100):
        """
        Main execution function to run the fitting process.
        Returns the fitted Superquadric model, sigma squared, and inlier/outlier counts.
        """
        print(f"Starting execution with {max_iters} iterations...")
        model = self.fit(max_iters=max_iters)
        # utils.analyze_inliers implementations vary; some return
        # (num_inliers, num_outliers) while others return additional outputs
        # like an inlier mask/indices or a ratio. Be tolerant.
        _inlier_res = utils.analyze_inliers(model, self.points)
        if isinstance(_inlier_res, (tuple, list)) and len(_inlier_res) >= 2:
            num_inliers, num_outliers = int(_inlier_res[0]), int(_inlier_res[1])
        elif isinstance(_inlier_res, dict):
            num_inliers = int(_inlier_res.get('num_inliers', _inlier_res.get('inliers', 0)))
            num_outliers = int(_inlier_res.get('num_outliers', _inlier_res.get('outliers', 0)))
        else:
            # Fallback: if it returns a single value, treat it as inliers count
            num_inliers = int(_inlier_res)
            num_outliers = max(0, len(self.points) - num_inliers)
        print(f"Execution finished: Inliers={num_inliers}, Outliers={num_outliers}")
        return model, self.sigma_sq, (num_inliers, num_outliers)

def _resolve_ply_path(filename: str) -> str:
    # Try current working directory first, then script directory
    if os.path.exists(filename):
        return filename
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, filename)
    return candidate

def _save_fit_results_txt(out_path: str,
                          params: list,
                          center_world: np.ndarray,
                          R_world_to_canonical: np.ndarray,
                          sigma_sq: float,
                          inliers_outliers: tuple,
                          ply_path: str):
    """Save fitted parameters and pose metadata to a text file."""
    num_inliers, num_outliers = inliers_outliers
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Superquadric EMS Fit Results\n")
        f.write("==========================\n\n")
        f.write(f"Input point cloud: {ply_path}\n\n")
        f.write("Best-fit Superquadric parameters [ax, ay, az, e1, e2] (canonical frame):\n")
        f.write("  " + " ".join([f"{float(x):.10g}" for x in params]) + "\n\n")
        f.write(f"sigma_sq: {float(sigma_sq):.10g}\n")
        f.write(f"Inliers: {int(num_inliers)}\n")
        f.write(f"Outliers: {int(num_outliers)}\n\n")

        f.write("Pose metadata\n")
        f.write("------------\n")
        f.write("Center (world):\n")
        f.write("  " + " ".join([f"{float(x):.10g}" for x in center_world.tolist()]) + "\n\n")

        f.write("Rotation R_init (world -> canonical):\n")
        for row in R_world_to_canonical:
            f.write("  " + " ".join([f"{float(x):.10g}" for x in row.tolist()]) + "\n")

        f.write("\nMapping note:\n")
        f.write("  canonical point p_c -> world point p_w = center + p_c @ R_init.T\n")


def circumscribe_sq(model, fitter, target=1.0, n_bisect=40):
    """
    Find the smallest scale factor s (>= 1) such that ALL cluster points
    satisfy F(p) <= target in canonical frame.

    target controls tightness:
        0.8  → loose SQ (bigger, all points well inside)
        1.0  → strict circumscription (default)
        1.2  → tight SQ (smaller, ~20% protrusion allowed)

    Uses the SQ implicit function:
        F(p) = ((|px/ax|^(2/e2) + |py/ay|^(2/e2))^(e2/e1) + |pz/az|^(2/e1))

    Returns:
        circumscribed_model : Superquadric with scaled axes [s*ax, s*ay, s*az]
        s                   : scale factor applied (>= 1)
    """
    pts = fitter.points   # (N, 3) canonical-frame cluster points
    ax, ay, az = model.ax, model.ay, model.az
    e1, e2 = max(model.e1, 0.1), max(model.e2, 0.1)
    eps = 1e-12

    def max_F(s):
        """Max implicit-function value over all points at scale s."""
        X = np.abs(pts[:, 0] / (s * ax + eps))
        Y = np.abs(pts[:, 1] / (s * ay + eps))
        Z = np.abs(pts[:, 2] / (s * az + eps))
        inner = (X ** (2.0 / e2) + Y ** (2.0 / e2)) ** (e2 / e1) + Z ** (2.0 / e1)
        return float(np.max(inner))

    # Check if EMS already satisfies the target
    if max_F(1.0) <= target:
        print(f"  [CIRCUM] SQ already satisfies F≤{target:.2f} at s=1.0 ✓")
        return model, 1.0

    # Binary search: find smallest s >= 1 such that max_F(s) <= target
    lo, hi = 1.0, 5.0
    while max_F(hi) > target:
        hi *= 2.0
    for _ in range(n_bisect):
        mid = (lo + hi) / 2.0
        if max_F(mid) <= target:
            hi = mid
        else:
            lo = mid

    s = hi
    circ = Superquadric([s * model.ax, s * model.ay, s * model.az,
                          model.e1, model.e2])
    print(f"  [CIRCUM] target={target:.2f}  s={s:.4f}  →  "
          f"ax={circ.ax:.4f}  ay={circ.ay:.4f}  az={circ.az:.4f}")
    return circ, s

if __name__ == "__main__":
    import glob
    from config import MAP_DIR as IO_OP_DIR, EMS_MAX_ITERS as MAX_ITERS, \
        EMS_DOWNSAMPLE_N, EMS_CIRCUMSCRIBE, EMS_CIRCUM_TARGET

    # ── Find all Cluster_*.ply files ─────────────────────────────────────────
    cluster_files = sorted(glob.glob(os.path.join(IO_OP_DIR, "Cluster_*.ply")))
    if not cluster_files:
        print(f"[ERROR] No Cluster_*.ply files found in:\n  {IO_OP_DIR}")
        sys.exit(1)

    print(f"Found {len(cluster_files)} cluster(s) to fit.\n")

    # ── Fit EMS to each cluster ───────────────────────────────────────────────
    results = []   # list of (pcd_vis, sq_mesh, cluster_name)

    for i, ply_path in enumerate(cluster_files, start=1):
        name = os.path.splitext(os.path.basename(ply_path))[0]
        print(f"{'═'*50}")
        print(f"  [{i}/{len(cluster_files)}]  Fitting {name}")
        print(f"{'═'*50}")

        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            print(f"  [WARNING] {name} is empty, skipping.\n")
            continue
        print(f"  {len(pcd.points):,} points loaded.")

        # --- Downsample for EMS fitting ---
        n_pts = len(pcd.points)
        if EMS_DOWNSAMPLE_N > 0 and n_pts > EMS_DOWNSAMPLE_N:
            idx = np.random.choice(n_pts, EMS_DOWNSAMPLE_N, replace=False)
            pcd_fit = pcd.select_by_index(idx.tolist())
            print(f"  Downsampled {n_pts:,} → {EMS_DOWNSAMPLE_N:,} pts for EMS fitting.")
        else:
            pcd_fit = pcd

        # --- EMS fit ---
        fitter = EMSFitter(pcd_fit)
        model, sigma_sq, (num_inliers, num_outliers) = fitter.execute(max_iters=MAX_ITERS)

        best_params = [float(x) for x in
                       [model.ax, model.ay, model.az, model.e1, model.e2]]
        print(f"  SQ params [ax,ay,az,e1,e2]: {[f'{p:.4f}' for p in best_params]}")
        print(f"  Inliers: {num_inliers:,}  |  Outliers: {num_outliers:,}")

        # --- Circumscribe: scale SQ out until it encloses all points ---
        if EMS_CIRCUMSCRIBE:
            # Use the FULL (not downsampled) point cloud for the containment check
            full_fitter = type('F', (), {'points': (np.asarray(pcd.points) - fitter.center) @ fitter.R_init,
                                         'extent': fitter.extent})()
            model, scale = circumscribe_sq(model, full_fitter, target=EMS_CIRCUM_TARGET)
            best_params = [float(x) for x in
                           [model.ax, model.ay, model.az, model.e1, model.e2]]

        # --- Save txt ---
        out_txt = os.path.join(IO_OP_DIR, f"sq_fit_{name}.txt")
        _save_fit_results_txt(out_txt, best_params, fitter.center,
                              fitter.R_init, sigma_sq,
                              (num_inliers, num_outliers), ply_path)
        print(f"  Saved: {out_txt}\n")

        # --- Build SQ mesh in world frame ---
        V, F = model.sample_surface(nu=200, nv=100, include_poles=True)
        sq_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(V),
            triangles=o3d.utility.Vector3iVector(F),
        )
        sq_mesh.compute_vertex_normals()
        T = np.eye(4)
        T[:3, :3] = fitter.R_init.T   # canonical → world rotation
        T[:3, 3]  = fitter.center
        sq_mesh.transform(T)

        # --- Prepare display cloud (downsampled) ---
        voxel_size = max(1e-6, np.mean(fitter.extent) / 300.0)
        try:
            pcd_vis = pcd.voxel_down_sample(voxel_size=voxel_size)
        except Exception:
            pcd_vis = pcd
        pcd_vis.paint_uniform_color([0.2, 0.6, 1.0])   # light blue
        sq_mesh.paint_uniform_color([1.0, 0.35, 0.2])  # orange-red

        results.append((pcd_vis, sq_mesh, name))

    if not results:
        print("[ERROR] No clusters were successfully fitted.")
        sys.exit(1)

    # ── One window per cluster, three side-by-side panels via X-offset ────────
    # Panel 1 (left)   : cluster point cloud only
    # Panel 2 (centre) : superquadric mesh only
    # Panel 3 (right)  : cluster point cloud  +  SQ wireframe overlaid
    print(f"Opening {len(results)} window(s) — 3 panels each (close all to exit).")

    WIN_W, WIN_H = 1500, 600
    PAD          = 20
    visualizers  = []

    for idx, (pcd_vis, sq_mesh, name) in enumerate(results):
        top  = PAD + idx * (WIN_H + PAD * 3)

        # ── Compute panel offset from the cluster bounding box ───────────────
        bb      = pcd_vis.get_axis_aligned_bounding_box()
        extent  = bb.get_extent()
        gap     = max(extent) * 1.4          # spacing between panels
        center  = bb.get_center()

        def _shift(geom, dx):
            """Return a copy of geom translated by dx along X."""
            import copy as _copy
            g = _copy.deepcopy(geom)
            g.translate([dx, 0, 0])
            return g

        # Panel 1 — cluster PC at original position (left)
        p1_pcd = _shift(pcd_vis, -gap)

        # Panel 2 — SQ solid mesh at centre
        p2_sq  = _shift(sq_mesh, 0)

        # Panel 3 — cluster PC + SQ wireframe at right
        sq_wire = o3d.geometry.LineSet.create_from_triangle_mesh(sq_mesh)
        sq_wire.paint_uniform_color([1.0, 0.6, 0.1])   # amber wireframe
        p3_pcd  = _shift(pcd_vis,  gap)
        p3_wire = _shift(sq_wire,  gap)

        # ── Build and open window ────────────────────────────────────────────
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"{name}  |  PC  ·  SQ  ·  PC+Wireframe",
            width=WIN_W, height=WIN_H,
            left=PAD, top=top,
        )
        for geom in [p1_pcd, p2_sq, p3_pcd, p3_wire]:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        visualizers.append(vis)

    # Event loop — keep all windows alive until all are closed
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

    print("\n✓ All done!")
