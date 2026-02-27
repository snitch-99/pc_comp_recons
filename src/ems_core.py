import numpy as np
import open3d as o3d
import scipy.optimize
import scipy.signal
import scipy.ndimage
import copy
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'utils'))
import utils

# --- CONFIGURATION PARAMETERS ---
SAFE_MIN_VAL          = 0.001       # Epsilon for avoiding division by zero or negative exponents
OUTLIER_RATIO_DEFAULT = 0.3
GUM_VOLUME_SCALE      = 1.5         # Scaling factor for the GUM volume (V)
W0_PRIOR_DEFAULT      = 0.5         # Initial outlier probability weight

# Optimization Bounds [ax, ay, az, e1, e2]
OPTIM_BOUNDS_LOWER = np.array([SAFE_MIN_VAL, SAFE_MIN_VAL, SAFE_MIN_VAL, 0.1, 0.1])
OPTIM_BOUNDS_UPPER = np.array([np.inf, np.inf, np.inf, 3.0, 3.0])

# EMS Loop Control
MAX_EMS_LOOPS    = 10       # Maximum number of restarts if S-step finds better candidates
CONVERGENCE_TOL  = 1e-6    # Termination threshold for parameter change
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


class EMSFitter:
    def __init__(self, pcd, outlier_ratio=OUTLIER_RATIO_DEFAULT, init_type='BBOX'):

        self.pcd_original = pcd
        self.points_original = np.asarray(pcd.points)
        
        # Pre-align using OBB
        obb = pcd.get_oriented_bounding_box()
        self.center = obb.center
        self.R_init = obb.R
        self.extent = obb.extent
        
        # Canonical points
        self.points = (self.points_original - self.center) @ self.R_init
        
        # Initial guess: tiny sphere at centroid → grows outward to fit the hollow shell.
        r0 = float(np.mean(self.extent)) * 0.10
        self.params = [r0, r0, r0, 1.0, 1.0]
        self.sigma_sq = np.mean(self.extent)**2 * 0.1
        
        # GUM Model params
        self.w_0 = 0 # Outlier probability (will be updated)
        self.V = np.prod(self.extent * GUM_VOLUME_SCALE) 
        self.p_outlier = 1.0 / self.V
        self.w_o_prior = W0_PRIOR_DEFAULT # Initial guess for outlier weight

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
        num_inliers, num_outliers = utils.analyze_inliers(model, self.points)
        print(f"Execution finished: Inliers={num_inliers}, Outliers={num_outliers}")
        return model, self.sigma_sq, (num_inliers, num_outliers)
