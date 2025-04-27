import torch

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import clipperpy
import time
import pandas as pd
import matplotlib.pyplot as plt
import csv

import csv
import numpy as np
from scipy.spatial.transform import Rotation as R

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET
import ast
from sklearn.cluster import DBSCAN

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

np.random.seed(3)

from scipy.interpolate import griddata, RBFInterpolator
import numpy as np
from typing import Tuple, List, Optional

def interpolate_surface(boulder_points: np.ndarray, 
                         bounds: Optional[Tuple[float, float, float, float]] = None,
                         grid_spacing: float = 0.5,
                         method: str = 'rbf',
                         z_threshold: float = 2.5,
                         smooth_outliers: bool = True) -> np.ndarray:
    """
    Interpolate a surface from boulder points and sample it at regular intervals,
    with outlier filtering.
    
    Args:
        boulder_points: Nx3 array of boulder coordinates (x, y, z)
        bounds: Optional (xmin, xmax, ymin, ymax) to define the region of interest.
        grid_spacing: Distance between sampled points in meters
        method: Interpolation method ('rbf', 'linear', 'cubic', or 'nearest')
        z_threshold: Z-score threshold for outlier detection
        smooth_outliers: Whether to apply additional smoothing to filter outliers
    
    Returns:
        Mx3 array of sampled (x, y, z) points from the interpolated surface
    """
    if len(boulder_points) < 4:
        raise ValueError("Need at least 4 boulder points for reliable interpolation")
    
    # Extract x, y, z coordinates
    x = boulder_points[:, 0]
    y = boulder_points[:, 1]
    z = boulder_points[:, 2]
    
    # Determine bounds if not provided
    if bounds is None:
        padding = 2.0  # Add padding around the boulder points
        xmin, xmax = np.min(x) - padding, np.max(x) + padding
        ymin, ymax = np.min(y) - padding, np.max(y) + padding
    else:
        xmin, xmax, ymin, ymax = bounds
    
    # Create a regular grid for interpolation
    grid_x = np.arange(xmin, xmax, grid_spacing)
    grid_y = np.arange(ymin, ymax, grid_spacing)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    
    # Perform interpolation
    if method == 'rbf':
        # Radial basis function interpolation - often better for scattered data
        rbf = RBFInterpolator(boulder_points[:, :2], z, kernel='thin_plate_spline', smoothing=0.1)
        grid_points = np.column_stack((grid_xx.flatten(), grid_yy.flatten()))
        grid_z = rbf(grid_points).reshape(grid_xx.shape)
    else:
        # Use griddata for other methods
        grid_z = griddata((x, y), z, (grid_xx, grid_yy), method=method, fill_value=np.nan)
    
    # Apply spatial smoothing to reduce outliers
    if smooth_outliers:
        from scipy.ndimage import gaussian_filter
        grid_z = gaussian_filter(grid_z, sigma=1.0)
    
    # Create array of sampled points (x, y, z)
    sampled_points = np.column_stack((
        grid_xx.flatten(),
        grid_yy.flatten(),
        grid_z.flatten()
    ))
    
    # Remove points where interpolation failed (NaN values)
    valid_mask = ~np.isnan(sampled_points[:, 2])
    sampled_points = sampled_points[valid_mask]
    
    # Filter outliers based on z-score of height
    if len(sampled_points) > 0:
        z_values = sampled_points[:, 2]
        z_mean = np.mean(z_values)
        z_std = np.std(z_values)
        
        if z_std > 0:  # Avoid division by zero
            z_scores = np.abs((z_values - z_mean) / z_std)
            inlier_mask = z_scores < z_threshold
            sampled_points = sampled_points[inlier_mask]
    
    return sampled_points

def filter_surface_outliers(points: np.ndarray, 
                           neighborhood_radius: float = 0.5, 
                           height_threshold: float = 0.5) -> np.ndarray:
    """
    Filter outliers from surface points by checking neighborhood consistency.
    
    Args:
        points: Nx3 array of (x, y, z) points
        neighborhood_radius: Radius to consider as neighborhood
        height_threshold: Max allowed height difference from neighbors
        
    Returns:
        Filtered points with outliers removed
    """
    from scipy.spatial import cKDTree
    
    if len(points) < 10:  # Need enough points for meaningful neighborhoods
        return points
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(points[:, :2])  # Only use x,y for distance calculation
    
    # Check each point against its neighbors
    valid_points = []
    for i, point in enumerate(points):
        # Find neighbors within radius
        indices = tree.query_ball_point(point[:2], neighborhood_radius)
        if len(indices) > 1:  # At least one neighbor besides itself
            neighbor_heights = points[indices, 2]
            median_height = np.median(neighbor_heights)
            # Check if point's height is within threshold of median neighborhood height
            if abs(point[2] - median_height) <= height_threshold:
                valid_points.append(point)
    
    return np.array(valid_points)

def generate_alignment_points(mission_data, 
                             grid_spacing: float = 0.5, 
                             min_cluster_size: int = 3,
                             eps: float = 0.3,
                             filter_outliers: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate points for alignment by:
    1. Getting boulder points from mission data
    2. Clustering them to remove outliers
    3. Interpolating a surface
    4. Sampling points from the surface
    5. Filtering outliers from the interpolated surface
    
    Args:
        mission_data: List of (pose, boulders) from load_mission_data
        grid_spacing: Distance between sampled points
        min_cluster_size: Min points in a cluster
        eps: Max distance for clustering
        filter_outliers: Whether to apply additional outlier filtering
        
    Returns:
        Tuple of (boulder_points, interpolated_points)
    """
    from sklearn.cluster import DBSCAN
    
    # Get boulders from last frame
    boulders = mission_data[-1][1]
    
    # Apply clustering to filter outliers
    if len(boulders) >= min_cluster_size:
        clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(boulders)
        labels = clustering.labels_
        filtered_boulders = boulders[labels != -1]
    else:
        filtered_boulders = boulders
    
    # Check if we have enough points for interpolation
    if len(filtered_boulders) < 4:
        return filtered_boulders, np.empty((0, 3))
    
    # Interpolate surface and sample points
    try:
        interpolated_points = interpolate_surface(
            filtered_boulders,
            grid_spacing=grid_spacing,
            method='rbf',
            z_threshold=2.5,  # Filter z-score outliers
            smooth_outliers=True  # Apply spatial smoothing
        )
        
        # Additional neighborhood-based outlier filtering
        if filter_outliers and len(interpolated_points) > 0:
            interpolated_points = filter_surface_outliers(
                interpolated_points,
                neighborhood_radius=grid_spacing*2,  # Consider points within 2 grid cells
                height_threshold=0.3  # Max height difference from neighbors
            )
            
        return filtered_boulders, interpolated_points
    except Exception as e:
        print(f"Error in surface interpolation: {e}")
        return filtered_boulders, np.empty((0, 3))

# def interpolate_surface(boulder_points: np.ndarray, 
#                          bounds: Optional[Tuple[float, float, float, float]] = None,
#                          grid_spacing: float = 0.5,
#                          method: str = 'rbf') -> np.ndarray:
#     """
#     Interpolate a surface from boulder points and sample it at regular intervals.
    
#     Args:
#         boulder_points: Nx3 array of boulder coordinates (x, y, z)
#         bounds: Optional (xmin, xmax, ymin, ymax) to define the region of interest.
#                 If None, will be determined from boulder points with padding.
#         grid_spacing: Distance between sampled points in meters
#         method: Interpolation method ('rbf', 'linear', 'cubic', or 'nearest')
    
#     Returns:
#         Mx3 array of sampled (x, y, z) points from the interpolated surface
#     """
#     if len(boulder_points) < 4:
#         raise ValueError("Need at least 4 boulder points for reliable interpolation")
    
#     # Extract x, y, z coordinates
#     x = boulder_points[:, 0]
#     y = boulder_points[:, 1]
#     z = boulder_points[:, 2]
    
#     # Determine bounds if not provided
#     if bounds is None:
#         padding = 2.0  # Add padding around the boulder points
#         xmin, xmax = np.min(x) - padding, np.max(x) + padding
#         ymin, ymax = np.min(y) - padding, np.max(y) + padding
#     else:
#         xmin, xmax, ymin, ymax = bounds
    
#     # Create a regular grid for interpolation
#     grid_x = np.arange(xmin, xmax, grid_spacing)
#     grid_y = np.arange(ymin, ymax, grid_spacing)
#     grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    
#     # Perform interpolation
#     if method == 'rbf':
#         # Radial basis function interpolation - often better for scattered data
#         rbf = RBFInterpolator(boulder_points[:, :2], z, kernel='thin_plate_spline')
#         grid_points = np.column_stack((grid_xx.flatten(), grid_yy.flatten()))
#         grid_z = rbf(grid_points).reshape(grid_xx.shape)
#     else:
#         # Use griddata for other methods
#         grid_z = griddata((x, y), z, (grid_xx, grid_yy), method=method, fill_value=np.nan)
    
#     # Create array of sampled points (x, y, z)
#     sampled_points = np.column_stack((
#         grid_xx.flatten(),
#         grid_yy.flatten(),
#         grid_z.flatten()
#     ))
    
#     # Remove points where interpolation failed (NaN values)
#     valid_mask = ~np.isnan(sampled_points[:, 2])
#     sampled_points = sampled_points[valid_mask]
    
#     return sampled_points

# def generate_alignment_points(mission_data, 
#                              grid_spacing: float = 0.5, 
#                              min_cluster_size: int = 3,
#                              eps: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Generate points for alignment by:
#     1. Getting boulder points from mission data
#     2. Clustering them to remove outliers
#     3. Interpolating a surface
#     4. Sampling points from the surface
    
#     Args:
#         mission_data: List of (pose, boulders) from load_mission_data
#         grid_spacing: Distance between sampled points
#         min_cluster_size: Min points in a cluster
#         eps: Max distance for clustering
        
#     Returns:
#         Tuple of (boulder_points, interpolated_points)
#     """

    
#     # Get boulders from last frame
#     boulders = mission_data[-1][1]
    
#     # Apply clustering to filter outliers
#     if len(boulders) >= min_cluster_size:
#         clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(boulders)
#         labels = clustering.labels_
#         filtered_boulders = boulders[labels != -1]
#     else:
#         filtered_boulders = boulders
    
#     # Check if we have enough points for interpolation
#     if len(filtered_boulders) < 4:
#         return filtered_boulders, np.empty((0, 3))
    
#     # Interpolate surface and sample points
#     try:
#         interpolated_points = interpolate_surface(
#             filtered_boulders,
#             grid_spacing=grid_spacing,
#             method='rbf'
#         )
#         return filtered_boulders, interpolated_points
#     except Exception as e:
#         print(f"Error in surface interpolation: {e}")
#         return filtered_boulders, np.empty((0, 3))


def cluster_boulders(boulder_points, min_cluster_size=3, eps=0.3):
    """
    Apply DBSCAN clustering to filter out outlier boulders.
    
    Args:
        boulder_points (np.ndarray): Array of boulder coordinates
        min_cluster_size (int): Minimum number of points to form a cluster
        eps (float): Maximum distance between points in a cluster (in meters)
        
    Returns:
        np.ndarray: Filtered boulder points (only those in significant clusters)
    """

    
    if len(boulder_points) < min_cluster_size:
        return np.empty((0, 3))
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_cluster_size).fit(boulder_points)
    labels = clustering.labels_
    
    # Filter out noise points (labeled as -1)
    filtered_points = boulder_points[labels != -1]
    
    return filtered_points

def deduplicate_boulders(boulders, threshold=0.01):
    """
    Removes boulders that are within `threshold` meters of an already kept boulder.

    Args:
        boulders (np.ndarray): (N, 3) array of boulder positions
        threshold (float): Distance threshold for considering two boulders duplicates

    Returns:
        np.ndarray: Deduplicated boulder positions
    """
    if len(boulders) == 0:
        return boulders

    kept = [0]  # Always keep the first boulder

    for i in range(1, len(boulders)):
        dists = np.linalg.norm(boulders[i] - boulders[kept], axis=1)
        if np.all(dists > threshold):
            kept.append(i)

    return boulders[kept]


def extract_rock_locations(xml_file):
    """
    Parses an XML file and extracts x, y, and z coordinates of rocks.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        list: A list of tuples containing (x, y, z) coordinates of rocks.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    rock_positions = []

    for rock in root.findall(".//rocks/rock"):
        x = float(rock.get("x", 0))
        y = float(rock.get("y", 0))
        z = float(rock.get("z", 0))
        rock_positions.append((x, y, z))

    return rock_positions
def keep_one_per_duplicate(boulders, threshold=0.01):
    """
    Keeps only one boulder per group of duplicates (within `threshold` meters), 
    and removes isolated/noisy boulders.

    Args:
        boulders (np.ndarray): (N, 3) array of boulder positions
        threshold (float): Distance threshold for considering two boulders duplicates

    Returns:
        np.ndarray: Cleaned boulder positions (one per duplicate group)
    """
    if len(boulders) == 0:
        return boulders

    to_keep = []
    visited = np.zeros(len(boulders), dtype=bool)

    for i in range(len(boulders)):
        if visited[i]:
            continue

        # Compute distances from boulder i to all others
        dists = np.linalg.norm(boulders[i] - boulders, axis=1)

        # Find all others within threshold (excluding itself)
        close_indices = np.where((dists < threshold) & (dists > 0))[0]

        if len(close_indices) > 0:
            # If there are nearby duplicates, keep this one
            to_keep.append(i)
            # Mark all nearby duplicates as visited so we don't keep them again
            visited[close_indices] = True
            visited[i] = True

    return boulders[to_keep]

def keep_only_duplicates(boulders, threshold=0.01):
    """
    Keeps only boulders that have at least one other boulder within `threshold` meters.

    Args:
        boulders (np.ndarray): (N, 3) array of boulder positions
        threshold (float): Distance threshold for considering two boulders duplicates

    Returns:
        np.ndarray: Boulder positions that have duplicates
    """
    if len(boulders) == 0:
        return boulders

    keep = []

    for i in range(len(boulders)):
        # Compute distance to all others (excluding self)
        dists = np.linalg.norm(boulders[i] - boulders, axis=1)
        close = (dists < threshold) & (dists > 0)  # ignore self (distance 0)
        if np.any(close):
            keep.append(i)

    return boulders[keep]

def load_mission_data(filepath):
    """
    Loads mission CSV data and returns list of (pose, boulders_so_far) per frame.
    
    Args:
        filepath (str): path to the CSV file
    
    Returns:
        List of tuples: (4x4 pose matrix, numpy array of boulder coordinates)
    """
    df = pd.read_csv(filepath)

    outputs = []
    boulder_locations = {}  # Dictionary to store boulder coordinates
    
    for idx, row in df.iterrows():
        # Build 4x4 pose matrix
        x, y, z = row['x'], row['y'], row['z']
        roll, pitch, yaw = row['roll'], row['pitch'], row['yaw']

        rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = [x, y, z]

        # Process boulder information if present and valid
        boulder_info = row['boulders_world']
        if pd.notna(boulder_info) and isinstance(boulder_info, str):
            try:
                # Parse boulder coordinates from string like "[1.8434, -0.54672, -0.13613]"
                # Remove brackets and split by comma
                coords_str = boulder_info.strip('[]').split(',')
                if len(coords_str) >= 3:
                    boulder_id = idx  # Using row index as boulder ID if not provided
                    boulder_x = float(coords_str[0])
                    boulder_y = float(coords_str[1]) 
                    boulder_z = float(coords_str[2])
                    boulder_locations[boulder_id] = [boulder_x, boulder_y, boulder_z]
            except (ValueError, IndexError):
                # Skip if parsing fails
                pass
        
        # Create array of all boulder locations seen so far
        boulder_points = np.array(list(boulder_locations.values())) if boulder_locations else np.empty((0, 3))
        
        outputs.append((pose, boulder_points))

    return outputs

# def load_mission_data_just_boulders(filepath):
#     """
#     Loads mission CSV data and returns list of (pose, boulders_so_far) per frame.
    
#     Args:
#         filepath (str): path to the CSV file
    
#     Returns:
#         List of tuples: (4x4 pose matrix, list of boulders seen so far) for each frame
#     """
#     df = pd.read_csv(filepath)

#     outputs = []
#     seen_boulders = set()
#     for idx, row in df.iterrows():
#         # Build 4x4 pose matrix
#         x, y, z = row['x'], row['y'], row['z']
#         roll, pitch, yaw = row['roll'], row['pitch'], row['yaw']

#         rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()

#         pose = np.eye(4)
#         pose[:3, :3] = rotation
#         pose[:3, 3] = [x, y, z]

#         # SAFER: Update boulders seen so far
#         boulder_id = row['boulders_rover']
#         if pd.notna(boulder_id):
#             seen_boulders.add(boulder_id)

#         outputs.append((pose, sorted(seen_boulders)))

#     return outputs


def rigid_transform_3D(A, B):
    # this function is from https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def generateAssociationList(N1, N2):
    assocList = np.zeros((N1*N2,2),np.int32)

    i = 0

    for n1 in range(N1):
        for n2 in range(N2):
            assocList[i,0] = n1
            assocList[i,1] = n2
            i += 1

    return assocList