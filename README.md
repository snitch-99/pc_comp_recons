# pc_comp_recons

**Point Cloud Compression & Scene Reconstruction Pipeline**

A modular Python pipeline that takes a raw photogrammetry point cloud, isolates individual rock clusters, fits compact superquadric + DEM representations, and reconstructs the full scene — enabling significant storage compression while preserving geometric fidelity.

---

## Overview

```
Raw Point Cloud (.ply)
        │
        ▼
 Step 0 — RANSAC ground removal + DBSCAN clustering
        │  → Cluster_N.ply, ground_plane.ply/txt
        ▼
 Step 1 — EMS Superquadric fitting per cluster
        │  → sq_fit_Cluster_N.txt
        ▼
 Step 2 — Poisson surface reconstruction (reference meshes)
        │  → poisson_Cluster_N.ply
        ▼
 Step 3 — DEM generation (spherical elevation map)
        │  → dem_Cluster_N.npy, mask_Cluster_N.npy
        ▼
 Step 4 — DEM-based mesh reconstruction
        │  → dem_recon_Cluster_N.ply
        ▼
 Step 5 — Full scene assembly + comparison + analysis
           → scene_reconstruction.ply, analysis.txt
```

---

## Repository Structure

```
pc_comp_recons/
├── src/
│   ├── config.py               # ★ Central config — change only this file
│   ├── ems_core.py             # EMS superquadric fitting core
│   ├── utils.py                # Shared utilities
│   ├── Step_0_cluster_rocks.py
│   ├── Step_1_sq_ems_fit.py
│   ├── Step_2_poisson_recons.py
│   ├── Step_3_dem_generation.py
│   ├── Step_4_reconstruction.py
│   └── Step_5_scene_comparison.py
└── io_op/
    ├── Prior_Map/              # Original scene inputs & outputs
    ├── Changed_Map/            # Modified scene inputs & outputs
    └── Old/                    # Archive
```

> **Note:** `.ply` files in `io_op/` are managed via [Git LFS](https://git-lfs.github.com/) due to their large size.

---

## Dependencies

- Python ≥ 3.10
- [Open3D](http://www.open3d.org/) — point cloud I/O, RANSAC, DBSCAN, Poisson reconstruction, visualisation
- NumPy
- SciPy

Install with:
```bash
pip install open3d numpy scipy
```

---

## Configuration

All parameters are centralised in `src/config.py`. **You only need to edit this one file** to switch maps or tune the pipeline.

**Key settings:**

| Parameter | Default | Description |
|---|---|---|
| `MAP_FOLDER` | `"Prior_Map"` | Switch between `"Prior_Map"` and `"Changed_Map"` |
| `RANSAC_DIST_THRESH` | `0.05` | Ground plane inlier distance (m) |
| `DBSCAN_EPS` | `0.05` | DBSCAN neighbourhood radius (m) |
| `MIN_CLUSTER_RATIO` | `0.10` | Minimum cluster size relative to largest |
| `EMS_MAX_ITERS` | `200` | Superquadric fitting iterations |
| `POISSON_DEPTH` | `9` | Poisson reconstruction tree depth |
| `DEM_W / DEM_H` | `720 / 360` | Spherical DEM resolution (longitude × latitude bins) |

---

## Usage

Run each step from the **repository root**, in order:

```bash
cd pc_comp_recons

python src/Step_0_cluster_rocks.py   # Segment ground + cluster rocks
python src/Step_1_sq_ems_fit.py      # Fit superquadrics to each cluster
python src/Step_2_poisson_recons.py  # Poisson reference meshes
python src/Step_3_dem_generation.py  # Generate spherical DEMs
python src/Step_4_reconstruction.py  # Reconstruct meshes from DEMs
python src/Step_5_scene_comparison.py # Assemble scene + analyse compression
```

All intermediate outputs are saved to `io_op/<MAP_FOLDER>/`.

---

## Pipeline Steps

### Step 0 — Rock Clustering
- Removes the ground plane iteratively using **RANSAC**
- Clusters remaining points with **DBSCAN**
- Saves each valid cluster as `Cluster_N.ply` and stores the ground plane equation in `ground_plane.txt`

### Step 1 — Superquadric Fitting
- Fits an **EMS (Expectation-Maximisation Superquadrics)** shape to each cluster
- Optionally circumscribes the SQ so all cluster points lie inside
- Saves shape parameters to `sq_fit_Cluster_N.txt`

### Step 2 — Poisson Reconstruction
- Estimates normals and runs **Poisson surface reconstruction** on each cluster
- Produces high-quality reference meshes (`poisson_Cluster_N.ply`) for accuracy comparison

### Step 3 — DEM Generation
- Projects each cluster onto a **spherical coordinate DEM** (elevation map)
- Outputs compact `dem_Cluster_N.npy` and `mask_Cluster_N.npy` arrays

### Step 4 — DEM-based Reconstruction
- Reconstructs a mesh from the DEM using the superquadric shape as a base
- Produces `dem_recon_Cluster_N.ply` in the original world frame

### Step 5 — Scene Comparison & Analysis
- Assembles the **full reconstructed scene** (ground mesh + all rock meshes)
- Displays the **original point cloud** and **reconstructed scene** side-by-side
- Generates `analysis.txt` with:
  - **Compression report**: Poisson PLY size vs DEM + mask + SQ (compact representation)
  - **Geometric accuracy**: Hausdorff distances between DEM reconstruction and Poisson reference

---

## Output Files (per cluster)

| File | Description |
|---|---|
| `Cluster_N.ply` | Raw segmented rock point cloud |
| `sq_fit_Cluster_N.txt` | Superquadric parameters |
| `poisson_Cluster_N.ply` | Poisson reference mesh |
| `dem_Cluster_N.npy` | Spherical elevation map |
| `mask_Cluster_N.npy` | Valid pixel mask for DEM |
| `dem_recon_Cluster_N.ply` | Final compact mesh reconstruction |
| `ground_plane.txt` | RANSAC plane equation `[a b c d]` |
| `ground_plane.ply` | Ground inlier points |
| `scene_reconstruction.ply` | Merged full scene mesh |
| `analysis.txt` | Compression + accuracy report |

---

## Git LFS

Large `.ply` point cloud files are stored using [Git LFS](https://git-lfs.github.com/). After cloning, run:

```bash
git lfs pull
```

to fetch the actual file contents.