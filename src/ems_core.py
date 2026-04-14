"""
ems_core.py — Superquadric model + EMS fitter
==============================================
Single authoritative source for:
  - Superquadric  : implicit surface, surface sampling, radial projection
  - EMSFitter     : Expectation-Maximisation-Switching loop
"""

import numpy as np
import open3d as o3d
import scipy.optimize
import copy
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAFE_MIN_VAL          = 0.01
SAFE_MIN_EXP          = 0.1
OUTLIER_RATIO_DEFAULT = 0.3
GUM_VOLUME_SCALE      = 1.5
W0_PRIOR_DEFAULT      = 0.5

OPTIM_BOUNDS_LOWER = np.array([SAFE_MIN_VAL, SAFE_MIN_VAL, SAFE_MIN_VAL,
                                SAFE_MIN_EXP, SAFE_MIN_EXP])
OPTIM_BOUNDS_UPPER = np.array([np.inf, np.inf, np.inf, 3.0, 3.0])

MAX_EMS_LOOPS   = 5
CONVERGENCE_TOL = 1e-4
INLIER_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Superquadric
# ---------------------------------------------------------------------------
class Superquadric:
    """
    Represents a Superquadric surface.
    Parameters: [ax, ay, az, e1, e2]
    """
    def __init__(self, params=(1.0, 1.0, 1.0, 1.0, 1.0)):
        self.ax, self.ay, self.az, self.e1, self.e2 = params
        self.e1 = max(self.e1, SAFE_MIN_EXP)
        self.e2 = max(self.e2, SAFE_MIN_EXP)
        self.ax = max(self.ax, SAFE_MIN_VAL)
        self.ay = max(self.ay, SAFE_MIN_VAL)
        self.az = max(self.az, SAFE_MIN_VAL)

    # ------------------------------------------------------------------
    def implicit_function(self, points: np.ndarray) -> np.ndarray:
        """F(x,y,z) — 1 on surface, <1 inside, >1 outside."""
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        inner = (np.abs(x / self.ax) ** (2 / self.e2) +
                 np.abs(y / self.ay) ** (2 / self.e2))
        return inner ** (self.e2 / self.e1) + np.abs(z / self.az) ** (2 / self.e1)

    def radial_distance_approximation(self, points: np.ndarray) -> np.ndarray:
        """Project each point onto the SQ surface along the radial direction."""
        F = self.implicit_function(points)
        factors = np.power(np.maximum(F, 1e-12), -self.e1 / 2.0)
        return points * factors[:, np.newaxis]

    def inside_outside_function(self, points: np.ndarray) -> np.ndarray:
        return self.implicit_function(points)

    # ------------------------------------------------------------------
    @staticmethod
    def _fexp(t: np.ndarray, e: float) -> np.ndarray:
        """Signed power: sign(t)*|t|^e."""
        return np.sign(t) * (np.abs(t) ** e)

    def sample_surface(self, nu: int = 200, nv: int = 100,
                       include_poles: bool = True):
        """
        Sample the SQ surface on a regular (eta, omega) grid.
        Returns: vertices (V,3), faces (F,3) in canonical frame.
        """
        nu = max(int(nu), 4)
        nv = max(int(nv), 3)

        etas   = np.linspace(-np.pi, np.pi, nu, endpoint=False)
        omegas = (np.linspace(-np.pi / 2.0, np.pi / 2.0, nv, endpoint=True)
                  if include_poles else
                  np.linspace(-np.pi / 2.0 + 1e-6, np.pi / 2.0 - 1e-6,
                               nv, endpoint=True))

        ce, se = np.cos(etas), np.sin(etas)

        V = np.zeros((nu * nv, 3), dtype=np.float64)
        for j, om in enumerate(omegas):
            co, so = np.cos(om), np.sin(om)
            fx = self._fexp(co, self.e1)
            fz = self._fexp(so, self.e1)
            x = self.ax * fx * self._fexp(ce, self.e2)
            y = self.ay * fx * self._fexp(se, self.e2)
            z = self.az * fz * np.ones_like(x)
            start = j * nu
            V[start:start + nu] = np.column_stack([x, y, z])

        faces = []
        for j in range(nv - 1):
            for i in range(nu):
                i2  = (i + 1) % nu
                v00 = j * nu + i;   v10 = j * nu + i2
                v01 = (j+1)*nu + i; v11 = (j+1)*nu + i2
                faces.extend([[v00, v10, v11], [v00, v11, v01]])
        return V, np.asarray(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# EMSFitter
# ---------------------------------------------------------------------------
class EMSFitter:
    def __init__(self, pcd: o3d.geometry.PointCloud,
                 outlier_ratio: float = OUTLIER_RATIO_DEFAULT,
                 w_o_prior: float = W0_PRIOR_DEFAULT):
        self.pcd_original    = pcd
        self.points_original = np.asarray(pcd.points)

        # Pre-align using OBB
        obb = pcd.get_oriented_bounding_box()
        self.center  = obb.center
        self.R_init  = obb.R
        self.extent  = obb.extent

        # Canonical points
        self.points = (self.points_original - self.center) @ self.R_init

        # Initial guess: OBB half-extents
        self.params   = [self.extent[0]/2, self.extent[1]/2,
                         self.extent[2]/2, 1.0, 1.0]
        self.sigma_sq = np.mean(self.extent) ** 2 * 0.1

        # GUM model
        self.V         = np.prod(self.extent * GUM_VOLUME_SCALE)
        self.p_outlier = 1.0 / max(self.V, 1e-12)
        self.w_o_prior = w_o_prior

    # ------------------------------------------------------------------
    def e_step(self):
        sq    = Superquadric(self.params)
        mu_s  = sq.radial_distance_approximation(self.points)
        diff  = self.points - mu_s
        sq_dist = np.sum(diff ** 2, axis=1)

        norm_factor = (2 * np.pi * self.sigma_sq) ** (-1.5)
        l_in  = norm_factor * np.exp(-sq_dist / (2 * self.sigma_sq))
        l_out = self.p_outlier
        num   = l_in * (1 - self.w_o_prior)
        den   = num + l_out * self.w_o_prior
        return num / (den + 1e-12), mu_s

    def m_step(self, z_prob, mu_s):
        lower, upper = OPTIM_BOUNDS_LOWER, OPTIM_BOUNDS_UPPER
        self.params  = np.clip(self.params, lower, upper)

        res = scipy.optimize.least_squares(
            lambda p: (np.sqrt(z_prob) *
                       np.linalg.norm(self.points -
                                      Superquadric(p).radial_distance_approximation(self.points),
                                      axis=1)),
            self.params, bounds=(lower, upper), method='trf',
        )
        self.params = res.x

        sq_upd   = Superquadric(self.params)
        mu_upd   = sq_upd.radial_distance_approximation(self.points)
        dists_sq = np.sum((self.points - mu_upd) ** 2, axis=1)
        sum_z    = np.sum(z_prob)
        self.sigma_sq  = np.sum(z_prob * dists_sq) / (3 * max(sum_z, 1e-12))
        self.w_o_prior = 1.0 - (sum_z / len(self.points))
        return self.params, self.sigma_sq

    def s_step(self) -> bool:
        best_loss   = self._loss(self.params)
        best_params = copy.deepcopy(self.params)
        found       = False
        ax, ay, az, e1, e2 = self.params

        candidates = [
            [az, ay, ax, e2, e1],
            [ax, az, ay, e2, e1],
        ]
        if e2 < 1.0:
            candidates.append([ax*np.sqrt(2), ay*np.sqrt(2), az, e1, 2.0-e2])
        elif e2 > 1.0:
            candidates.append([ax/np.sqrt(2), ay/np.sqrt(2), az, e1, 2.0-e2])

        for cand in candidates:
            cand = [max(c, SAFE_MIN_VAL) for c in cand]
            loss = self._loss(cand)
            if loss < best_loss:
                best_loss, best_params, found = loss, cand, True

        if found:
            self.params = best_params
        return found

    def _loss(self, params) -> float:
        sq = Superquadric(params)
        mu = sq.radial_distance_approximation(self.points)
        return float(np.sum((self.points - mu) ** 2))

    # ------------------------------------------------------------------
    def fit(self, max_iters: int = 100,
            external_pbar=None) -> Superquadric:
        ems_converged = False
        loop_count    = 0

        while not ems_converged and loop_count < MAX_EMS_LOOPS:
            loop_count += 1
            if external_pbar:
                external_pbar.reset(total=max_iters)
                external_pbar.set_description(f"EMS Loop {loop_count}")
            self._run_em_to_convergence(max_iters, external_pbar)

            if not self.s_step():
                ems_converged = True
                print("Optimization finished (no better switch found).")
            else:
                print("S-Step switched parameters — restarting EM.")

        return Superquadric(self.params)

    def _run_em_to_convergence(self, max_iters: int, pbar):
        local_pbar = (tqdm(range(max_iters), desc="EM inner", leave=False)
                      if pbar is None else None)
        prev_loss  = float('inf')

        for _ in range(max_iters):
            z_prob, mu_s = self.e_step()
            self.m_step(z_prob, mu_s)

            curr_loss   = self._loss(self.params)
            loss_change = abs(prev_loss - curr_loss)
            status      = {'sigma': f'{self.sigma_sq:.2g}',
                           'dL':    f'{loss_change:.2g}'}

            if pbar:
                pbar.set_postfix(status); pbar.update(1)
            elif local_pbar:
                local_pbar.set_postfix(status); local_pbar.update(1)

            if loss_change < CONVERGENCE_TOL:
                break
            prev_loss = curr_loss

        if local_pbar:
            local_pbar.close()
