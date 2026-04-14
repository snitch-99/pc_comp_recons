"""
config.py  —  Central configuration for the Rock Reconstruction Pipeline
==========================================================================
CHANGE ONLY THIS FILE to switch between maps or tune parameters.

Map folders (relative to io_op/):
    Prior_Map   — original scene
    Changed_Map — scene after change

Usage in each script:
    from config import *
"""

import os

# ==============================================================================
# ★  CHANGE THIS ONE LINE to switch maps  ★
# ==============================================================================
MAP_FOLDER = '/home/kanav/workspaces/pc_comp_recons/src/APOLLO_15/rock_1'

# Optional: To bypass Step 0 and fit a single .obj or .ply mesh directly in Step 1,
# specify the filename here (assumes the file is placed inside your MAP_FOLDER).
# Example: "rock.obj". Set to None or "" to run normally on all clusters.
SINGLE_MESH_INPUT = None
# ==============================================================================

_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_IO_OP    = os.path.join(_ROOT, "io_op")
MAP_DIR   = MAP_FOLDER if os.path.isabs(MAP_FOLDER) else os.path.join(_IO_OP, MAP_FOLDER)

# --- Input point cloud (Step 0) ----------------------------------------------
# Automatically picks the right file based on MAP_FOLDER convention:
#   Prior_Map   → prior_point_cloud.ply
#   Changed_Map → Changed_point_cloud.ply
_PLY_NAMES = {
    "Prior_Map":   "prior_point_cloud.ply",
    "Changed_Map": "Changed_point_cloud.ply",
}
INPUT_PLY = os.path.join(MAP_DIR, _PLY_NAMES.get(MAP_FOLDER, "point_cloud.ply"))

# --- Step 0: RANSAC + DBSCAN -------------------------------------------------
RANSAC_DIST_THRESH = 0.05    # ground plane inlier distance (m)
RANSAC_N           = 3
RANSAC_ITERS       = 1000
RANSAC_MIN_INLIERS = 5000
MAX_PLANES         = 1       # only strip the ground plane

DBSCAN_EPS         = 0.05    # neighbourhood radius (m)
DBSCAN_MIN_POINTS  = 10

MIN_CLUSTER_RATIO  = 0.10    # keep clusters ≥ this fraction of the largest

# --- Step 1: EMS SQ fitting --------------------------------------------------
EMS_MAX_ITERS    = 200
EMS_DOWNSAMPLE_N = 0
# Circumscribed SQ: after EMS fit, scale UP axes until ALL cluster points lie
# inside the SQ  (F(p) ≤ 1). No mesh dependency — uses the implicit equation.
EMS_CIRCUMSCRIBE  = True
EMS_CIRCUM_TARGET = 1.0


# --- Step 2: Poisson reconstruction ------------------------------------------
POISSON_DEPTH        = 14
POISSON_NORMAL_KNN   = 51
POISSON_DENSITY_QTLE = 0.017

# --- Step 3: DEM generation --------------------------------------------------
DEM_W                    = 1440
DEM_H                    = 720
USE_LOCAL_SUPPORT_FILTER = True
LOCAL_SUPPORT_K          = 3
SAVE_RECON_POINTS_PLY    = True

# --- Step 4: DEM-based reconstruction ----------------------------------------
DEM_RECON_POISSON_DEPTH = 14
DEM_RECON_NORMAL_KNN    = 30
