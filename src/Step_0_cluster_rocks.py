"""
Step0_cluster_rocks.py
======================
Processes a photogrammetry scene point cloud to isolate individual rock clusters.

Pipeline:
  1. Load the input point cloud
  2. Iteratively remove dominant planes (ground, table, etc.) using RANSAC
  3. Cluster the remaining points using DBSCAN
  4. Save each valid cluster as Cluster_1.ply, Cluster_2.ply, ... in io_op/

Usage:
  python src/Step0_cluster_rocks.py [input.ply]

  If no argument is given, INPUT_PLY below is used as default.
"""

import os
import sys
import numpy as np
import open3d as o3d
from config import (
    INPUT_PLY, MAP_DIR as OUTPUT_DIR,
    RANSAC_DIST_THRESH, RANSAC_N, RANSAC_ITERS, RANSAC_MIN_INLIERS, MAX_PLANES,
    DBSCAN_EPS, DBSCAN_MIN_POINTS, MIN_CLUSTER_RATIO,
)


# ==============================================================================


def align_to_ground(pcd: o3d.geometry.PointCloud,
                    plane_model: list) -> tuple:
    """
    Rotate + translate the point cloud so that:
      - the RANSAC ground normal aligns with +Y (up)
      - the ground plane sits at Y = 0

    Returns:
        aligned_pcd : transformed PointCloud
        T           : 4×4 numpy transform (apply to any other cloud with
                      pcd.transform(T))
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal)

    # Ensure normal points *upward* (positive Y component)
    if normal[1] < 0:
        normal = -normal

    up = np.array([0.0, 1.0, 0.0])
    dot = float(np.clip(np.dot(normal, up), -1.0, 1.0))

    if abs(dot - 1.0) < 1e-6:          # already aligned
        R = np.eye(3)
    elif abs(dot + 1.0) < 1e-6:        # exactly anti-parallel
        R = np.diag([1.0, -1.0, -1.0])
    else:
        axis  = np.cross(normal, up)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        # Rodrigues' rotation formula
        K = np.array([
            [    0,  -axis[2],  axis[1]],
            [ axis[2],    0,   -axis[0]],
            [-axis[1],  axis[0],    0  ]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    # Rotate all points
    pts = np.asarray(pcd.points)
    pts_rot = (R @ pts.T).T

    # Translate so the ground sits at Y = 0
    # The rotated plane equation becomes Y + d' = 0, so shift by -min(Y_ground)
    # Use the mean Y of the rotated inlier points ≈ 0 target
    # Simple: shift all points down by the minimum Y in the rotated cloud
    y_min = pts_rot[:, 1].min()
    t = np.array([0.0, -y_min, 0.0])

    pts_aligned = pts_rot + t

    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(pts_aligned)
    if pcd.has_colors():
        aligned_pcd.colors = pcd.colors
    if pcd.has_normals():
        nrm = np.asarray(pcd.normals)
        aligned_pcd.normals = o3d.utility.Vector3dVector((R @ nrm.T).T)

    # Build 4×4 homogeneous transform  T = Translate(t) @ Rotate(R)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t + R @ np.zeros(3)   # t is applied after rotation
    # Correct: T maps x_orig → R·x + t
    T[:3, 3] = t

    print(f"  Ground normal: [{a:.4f}, {b:.4f}, {c:.4f}]  "
          f"→ aligned to Y-up, ground at Y=0")
    return aligned_pcd, T


def _resolve_ply_path(filename: str) -> str:
    """Try CWD first, then next to this script."""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    if os.path.exists(filename):
        return os.path.abspath(filename)
    candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    return candidate


def remove_planes_ransac(pcd: o3d.geometry.PointCloud):
    """
    Iteratively detects and removes dominant flat planes (ground, surfaces)
    from the point cloud using RANSAC.

    Returns:
        remaining  : PointCloud with planes stripped out
        plane_pcds : list of PointClouds, one per detected plane
    """
    remaining = pcd
    planes_removed = 0
    plane_pcds = []

    while planes_removed < MAX_PLANES:
        if len(remaining.points) < RANSAC_N:
            break

        plane_model, inliers = remaining.segment_plane(
            distance_threshold=RANSAC_DIST_THRESH,
            ransac_n=RANSAC_N,
            num_iterations=RANSAC_ITERS,
        )

        if len(inliers) < RANSAC_MIN_INLIERS:
            break

        # Collect plane points for visualization
        plane_pcd = remaining.select_by_index(inliers)
        plane_pcds.append(plane_pcd)

        # Keep the outlier (non-plane) points
        remaining = remaining.select_by_index(inliers, invert=True)
        planes_removed += 1

    print(f"Removed {planes_removed} plane(s). "
          f"{len(remaining.points)} points remaining for clustering.")
    return remaining, plane_pcds


def cluster_dbscan(pcd: o3d.geometry.PointCloud) -> list[o3d.geometry.PointCloud]:
    """
    Runs DBSCAN on the point cloud and returns a list of cluster point clouds,
    sorted by size (largest first), filtered by MIN_CLUSTER_POINTS.
    """
    labels = np.array(
        pcd.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=True,
        )
    )

    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 is noise in DBSCAN
    print(f"\nDBSCAN found {len(unique_labels)} cluster(s) "
          f"(plus {np.sum(labels == -1)} noise points).")

    clusters = []
    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0].tolist()
        cluster_pcd = pcd.select_by_index(indices)

        n_pts = len(cluster_pcd.points)
        if n_pts < MIN_CLUSTER_POINTS:
            print(f"  Cluster {label}: {n_pts} pts — too small, skipping.")
            continue

        clusters.append(cluster_pcd)
        print(f"  Cluster {label}: {n_pts} pts — kept.")

    # Sort largest → smallest
    clusters.sort(key=lambda c: len(c.points), reverse=True)
    return clusters


def save_clusters(clusters: list[o3d.geometry.PointCloud], output_dir: str):
    """Saves each cluster as Cluster_N.ply in the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    for i, cluster in enumerate(clusters, start=1):
        out_path = os.path.join(output_dir, f"Cluster_{i}.ply")
        o3d.io.write_point_cloud(out_path, cluster)
        print(f"  Saved: {out_path}  ({len(cluster.points)} pts)")


# Distinct bright colors for clusters (RGB, 0-1)
_CLUSTER_COLORS = [
    [0.96, 0.26, 0.21],  # red
    [0.13, 0.59, 0.95],  # blue
    [0.30, 0.69, 0.31],  # green
    [0.95, 0.08, 0.65],  # magenta
    [0.61, 0.15, 0.69],  # purple
    [0.00, 0.74, 0.83],  # cyan
    [1.00, 0.34, 0.13],  # orange
    [0.38, 0.49, 0.55],  # blue-grey
]


def _make_grid(bounds_min, bounds_max, y_level, spacing=0.5):
    """Flat reference grid on the XZ plane at height y_level."""
    x0, x1 = bounds_min[0], bounds_max[0]
    z0, z1 = bounds_min[2], bounds_max[2]
    lines, pts, idx = [], [], 0
    for x in np.arange(x0, x1 + spacing, spacing):
        pts += [[x, y_level, z0], [x, y_level, z1]]
        lines.append([idx, idx + 1]); idx += 2
    for z in np.arange(z0, z1 + spacing, spacing):
        pts += [[x0, y_level, z], [x1, y_level, z]]
        lines.append([idx, idx + 1]); idx += 2
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32)),
    )
    grid.colors = o3d.utility.Vector3dVector([[0.55, 0.55, 0.55]] * len(lines))
    return grid


def visualize(plane_pcds: list[o3d.geometry.PointCloud],
              clusters: list[o3d.geometry.PointCloud],
              noise_pcd: o3d.geometry.PointCloud | None = None):
    """
    Opens an Open3D window showing:
      - Each detected plane in a shade of grey
      - Each rock cluster in a distinct bright color
      - Noise points (if any) in dark grey
      - A world-origin coordinate frame (red=X, green=Y, blue=Z)
      - A flat XZ reference grid at floor level
    """
    geometries = []

    # ── Ground plane: red ──────────────────────────────────────────────────
    for plane in plane_pcds:
        colored = o3d.geometry.PointCloud(plane)
        colored.paint_uniform_color([0.85, 0.1, 0.1])   # red
        geometries.append(colored)

    # ── Clusters: distinct bright colors ───────────────────────────────────
    color_names = ["Blue", "Green", "Magenta", "Purple", "Cyan", "Orange", "Blue-grey"]
    print("\n  Clusters found:")
    for i, cluster in enumerate(clusters):
        colored = o3d.geometry.PointCloud(cluster)
        color = _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)]
        name  = color_names[i % len(color_names)]
        colored.paint_uniform_color(color)
        geometries.append(colored)

        # Draw oriented bounding box in the same colour
        obb = cluster.get_oriented_bounding_box()
        obb.color = color
        geometries.append(obb)

        print(f"    Cluster {i+1}: {len(cluster.points):,} pts  →  {name}")

    # ── Noise / fragments: dark grey ───────────────────────────────────────
    if noise_pcd is not None and len(noise_pcd.points) > 0:
        colored_noise = o3d.geometry.PointCloud(noise_pcd)
        colored_noise.paint_uniform_color([0.2, 0.2, 0.2])
        geometries.append(colored_noise)

    # ── Coordinate frame + reference grid ──────────────────────────────────
    all_pcds = list(plane_pcds) + list(clusters)
    if all_pcds:
        all_pts = np.vstack([np.asarray(p.points) for p in all_pcds])
        bb_min  = all_pts.min(axis=0)
        bb_max  = all_pts.max(axis=0)
        scene_size   = float(np.linalg.norm(bb_max - bb_min))
        grid_spacing = max(0.1, round(scene_size / 20, 1))   # ~20 lines across scene
        axis_size    = scene_size * 0.10

        axis_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=axis_size, origin=bb_min.tolist()
        )
        ref_grid = _make_grid(bb_min, bb_max,
                              y_level=float(bb_min[1]),
                              spacing=grid_spacing)
        geometries += [axis_frame, ref_grid]
        print(f"  Grid spacing: {grid_spacing:.2f} m  |  Axis size: {axis_size:.2f} m")

    print("\nOpening Open3D viewer — close the window to continue...")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Rock Clusters — RANSAC planes (red)  |  DBSCAN clusters (colour)",
        width=1280,
        height=720,
    )


def main():
    # Allow overriding input path from command line
    ply_name = sys.argv[1] if len(sys.argv) > 1 else INPUT_PLY
    ply_path = _resolve_ply_path(ply_name)

    if not os.path.exists(ply_path):
        print(f"[ERROR] Input point cloud not found.")
        print(f"        Tried: {ply_path}")
        print(f"        Either place your file as '{ply_name}' in CWD/src/ "
              f"or pass the path as an argument.")
        sys.exit(1)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading point cloud: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        print("[ERROR] Loaded point cloud is empty.")
        sys.exit(1)
    print(f"       {len(pcd.points)} points loaded.")

    # ── 2. RANSAC plane removal ─────────────────────────────────────────────────
    print(f"\n[2/6] RANSAC plane removal "
          f"(dist_thresh={RANSAC_DIST_THRESH}, min_inliers={RANSAC_MIN_INLIERS})")
    non_ground, plane_pcds = remove_planes_ransac(pcd)

    if len(non_ground.points) == 0:
        print("[ERROR] All points were removed during plane segmentation. "
              "Try lowering RANSAC_MIN_INLIERS.")
        sys.exit(1)

    # ── 2b. Align entire point cloud to ground plane ────────────────────────────
    print(f"\n[2b/6] Aligning point cloud to ground plane (Y-up) ...")
    plane_model_raw, _ = pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THRESH,
        ransac_n=RANSAC_N,
        num_iterations=RANSAC_ITERS,
    )
    pcd, align_T = align_to_ground(pcd, plane_model_raw)
    # Also transform the non-ground and plane clouds to the same aligned frame
    non_ground.transform(align_T)
    for p in plane_pcds:
        p.transform(align_T)
    # Save transform for downstream steps
    align_T_path = os.path.join(OUTPUT_DIR, "alignment_transform.npy")
    np.save(align_T_path, align_T)
    print(f"       Transform saved: {align_T_path}")

    # ── 3. DBSCAN clustering ────────────────────────────────────────────────────
    print(f"\n[3/6] DBSCAN clustering "
          f"(eps={DBSCAN_EPS}, min_points={DBSCAN_MIN_POINTS})")

    # Run DBSCAN and also keep noise points for visualization
    labels = np.array(
        non_ground.cluster_dbscan(
            eps=DBSCAN_EPS,
            min_points=DBSCAN_MIN_POINTS,
            print_progress=True,
        )
    )
    unique_labels = set(labels)
    unique_labels.discard(-1)
    print(f"\nDBSCAN found {len(unique_labels)} cluster(s) "
          f"(plus {np.sum(labels == -1)} noise points).")

    # Collect all clusters (unsorted first)
    all_clusters = []
    for label in sorted(unique_labels):
        indices = np.where(labels == label)[0].tolist()
        cluster_pcd = non_ground.select_by_index(indices)
        all_clusters.append(cluster_pcd)
    all_clusters.sort(key=lambda c: len(c.points), reverse=True)

    # Relative size filter: keep only clusters >= MIN_CLUSTER_RATIO * largest
    largest_count = len(all_clusters[0].points) if all_clusters else 1
    min_pts = int(largest_count * MIN_CLUSTER_RATIO)
    print(f"  Largest cluster: {largest_count} pts → keeping clusters with ≥ "
          f"{min_pts} pts ({MIN_CLUSTER_RATIO*100:.0f}% threshold)")

    clusters = []
    rejected_points = []
    for cluster_pcd in all_clusters:
        n_pts = len(cluster_pcd.points)
        if n_pts < min_pts:
            rejected_points.append(np.asarray(cluster_pcd.points))
        else:
            clusters.append(cluster_pcd)

    # Merge all rejected fragments + DBSCAN noise into one single grey cloud
    dbscan_noise_pts = np.asarray(non_ground.select_by_index(
        np.where(labels == -1)[0].tolist()).points)
    all_noise_pts = rejected_points + ([dbscan_noise_pts] if len(dbscan_noise_pts) > 0 else [])
    noise_pcd = o3d.geometry.PointCloud()
    if all_noise_pts:
        noise_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_noise_pts))

    if not clusters:
        print("[ERROR] No valid clusters found. "
              "Try adjusting DBSCAN_EPS, DBSCAN_MIN_POINTS, or MIN_CLUSTER_RATIO.")
        sys.exit(1)

    # ── 4. Save clusters ────────────────────────────────────────────────────────
    print(f"\n[4/6] Saving {len(clusters)} cluster(s) to: {os.path.abspath(OUTPUT_DIR)}")
    save_clusters(clusters, OUTPUT_DIR)

    # ── 5. Save ground plane (in aligned coordinates) ──────────────────────────
    if plane_pcds:
        # In the aligned frame the ground is Y=0, so plane eq is: 0x+1y+0z+0=0
        a, b, c, d = 0.0, 1.0, 0.0, 0.0
        plane_txt = os.path.join(OUTPUT_DIR, "ground_plane.txt")
        with open(plane_txt, "w") as f:
            f.write(f"{a:.10g} {b:.10g} {c:.10g} {d:.10g}\n")
        print(f"\n[5/6] Ground plane saved: {plane_txt}")
        print(f"      Equation: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

        ground_pts = np.vstack([np.asarray(p.points) for p in plane_pcds])
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_pts)
        ground_ply = os.path.join(OUTPUT_DIR, "ground_plane.ply")
        o3d.io.write_point_cloud(ground_ply, ground_pcd)
        print(f"      Ground points saved: {ground_ply}  ({len(ground_pts):,} pts)")

    print(f"\n✓ Done! {len(clusters)} cluster(s) saved to '{os.path.abspath(OUTPUT_DIR)}'")

    # ── 6. Visualize (optional — safe to Ctrl+C) ────────────────────────────────
    print(f"\n[6/6] Visualizing results...")
    visualize(plane_pcds, clusters, noise_pcd)


if __name__ == "__main__":
    main()
