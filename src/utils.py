import numpy as np
import open3d as o3d
import scipy.signal
import scipy.ndimage

def world_to_parametric(model, points):
    """
    Converts 3D points (in canonical frame) to parametric angles (eta, omega).
    Approximate inverse mapping for Superquadric.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    def sgn_pow_inv(val, exponent):
        return np.sign(val) * (np.abs(val) ** (1.0/exponent))

    # 1. Calc Eta from Z
    z_norm = z / model.az
    # Clip to avoid numerical errors > 1
    sin_eta = sgn_pow_inv(z_norm, model.e1)
    sin_eta = np.clip(sin_eta, -1.0, 1.0)
    eta = np.arcsin(sin_eta)
    
    # 2. Calc Omega from X, Y
    term = (y / model.ay) / (x / model.ax + 1e-12) # avoid div0
    tan_omega = sgn_pow_inv(term, model.e2)
    omega = np.arctan(tan_omega)
    
    # Fix quadrant logic to match arctan2
    x_base = sgn_pow_inv(x / model.ax, model.e2)
    y_base = sgn_pow_inv(y / model.ay, model.e2)
    omega = np.arctan2(y_base, x_base)
    
    return eta, omega

def get_mesh(model, resolution=50):
    """Generates a triangle mesh for visualization."""
    eta = np.linspace(-np.pi/2, np.pi/2, resolution)
    omega = np.linspace(-np.pi, np.pi, resolution)
    
    ETA, OMEGA = np.meshgrid(eta, omega)
    
    def sgn_pow(val, exponent):
        return np.sign(val) * (np.abs(val) ** exponent)

    ce = np.cos(ETA)
    se = np.sin(ETA)
    co = np.cos(OMEGA)
    so = np.sin(OMEGA)
    
    x = model.ax * sgn_pow(ce, model.e1) * sgn_pow(co, model.e2)
    y = model.ay * sgn_pow(ce, model.e1) * sgn_pow(so, model.e2)
    z = model.az * sgn_pow(se, model.e1)
    
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx0 = i * resolution + j
            idx1 = i * resolution + (j + 1)
            idx2 = (i + 1) * resolution + (j + 1)
            idx3 = (i + 1) * resolution + j
            
            triangles.append([idx0, idx2, idx1])
            triangles.append([idx0, idx3, idx2])
            
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def get_occupied_mesh(model, original_points, pixel_size=0.01, min_neighbors=5):
    """
    Generates a mesh but ONLY triangles that are "occupied" by nearby original points.
    Vectorized implementation with Denoising.
    """
    # 1. Project points to Parametric Space
    p_eta, p_omega = world_to_parametric(model, original_points)
    
    # 2. Determine Resolution
    mean_scale_xy = (model.ax + model.ay) / 2.0
    circ_xy = 2 * np.pi * mean_scale_xy
    circ_z = np.pi * model.az
    
    res_omega = int(circ_xy / pixel_size)
    res_eta = int(circ_z / pixel_size)
    
    res_omega = max(res_omega, 10)
    res_eta = max(res_eta, 10)
    
    # 3. Parametric Grid
    eta_space = np.linspace(-np.pi/2, np.pi/2, res_eta)
    omega_space = np.linspace(-np.pi, np.pi, res_omega)
    
    # 4. Create Occupancy Grid
    eta_bins = np.digitize(p_eta, eta_space) - 1
    omega_bins = np.digitize(p_omega, omega_space) - 1
    
    occupancy_grid = np.zeros((res_omega, res_eta), dtype=bool)
    
    valid_mask = (eta_bins >= 0) & (eta_bins < res_eta-1) & \
                 (omega_bins >= 0) & (omega_bins < res_omega-1)
    
    valid_eta = eta_bins[valid_mask]
    valid_omega = omega_bins[valid_mask]
    
    occupancy_grid[valid_omega, valid_eta] = True
    
    raw_count = np.sum(occupancy_grid)
    
    # --- 4b. DENOISING ---
    if min_neighbors > 0:
        kernel = np.ones((3,3), dtype=int)
        neighbor_count = scipy.signal.convolve2d(occupancy_grid.astype(int), kernel, mode='same', boundary='fill')
        filtered_grid = occupancy_grid & (neighbor_count >= (min_neighbors))
        print(f"Denoising: Removed {raw_count - np.sum(filtered_grid)} isolated cells.")
        occupancy_grid = filtered_grid

    # Check coverage
    coverage = np.sum(occupancy_grid) / (res_omega * res_eta)
    print(f"Surface Coverage: {coverage*100:.2f}%")
    
    # 5. Determine Active Triangles
    mask = occupancy_grid[0:res_omega-1, 0:res_eta-1]
    active_quads = np.where(mask)
    rows = active_quads[0]
    cols = active_quads[1]
    
    if len(rows) == 0:
         return o3d.geometry.TriangleMesh()

    idx0 = rows * res_eta + cols
    idx1 = rows * res_eta + (cols + 1)
    idx2 = (rows + 1) * res_eta + (cols + 1)
    idx3 = (rows + 1) * res_eta + cols
    
    t1 = np.stack([idx0, idx2, idx1], axis=1)
    t2 = np.stack([idx0, idx3, idx2], axis=1)
    triangles = np.vstack([t1, t2])
    
    # 6. Generate Vertices
    ETA, OMEGA = np.meshgrid(eta_space, omega_space, indexing='xy') 
    
    def sgn_pow(val, exponent):
        return np.sign(val) * (np.abs(val) ** exponent)

    ce, se = np.cos(ETA), np.sin(ETA)
    co, so = np.cos(OMEGA), np.sin(OMEGA)
    
    x = model.ax * sgn_pow(ce, model.e1) * sgn_pow(co, model.e2)
    y = model.ay * sgn_pow(ce, model.e1) * sgn_pow(so, model.e2)
    z = model.az * sgn_pow(se, model.e1)
    
    vertices = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh

def analyze_inliers(model, points, distance_threshold=0.01):
    """
    Analyzes the quality of fit by counting inliers within a distance threshold.
    """
    mu = model.radial_distance_approximation(points)
    distances = np.linalg.norm(points - mu, axis=1)
    inlier_mask = distances <= distance_threshold
    num_inliers = np.sum(inlier_mask)
    num_outliers = len(points) - num_inliers
    return num_inliers, num_outliers, inlier_mask

def get_model_cloud(model, width=360, height=180):
    """
    Generates a point cloud from the Superquadric model.
    Migrated from DeviationPipeline.reconstruct_surface.
    """
    # 1. Generate Grid of Angles
    u = np.linspace(0, width-1, width)
    v = np.linspace(0, height-1, height)
    U, V = np.meshgrid(u, v)

    # Map U,V to Angles
    lon = (U / (width - 1)) * 2 * np.pi - np.pi   # -pi to pi
    lat = (V / (height - 1)) * np.pi - np.pi/2    # -pi/2 to pi/2

    # 2. Spherical -> Cartesian (Unit Sphere)
    cl = np.cos(lat)
    sl = np.sin(lat)
    clo = np.cos(lon)
    slo = np.sin(lon)
    
    def sgn_pow(val, exponent):
        return np.sign(val) * (np.abs(val) ** exponent)

    x = model.ax * sgn_pow(cl, model.e1) * sgn_pow(clo, model.e2)
    y = model.ay * sgn_pow(cl, model.e1) * sgn_pow(slo, model.e2)
    z = model.az * sgn_pow(sl, model.e1)

    points = np.dstack((x, y, z)).reshape(-1, 3)

    # Create PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    
    return pcd
