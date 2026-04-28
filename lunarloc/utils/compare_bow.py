from optimal_detections_loader import load_data, get_pose, get_transform, get_detections
import numpy as np 

import numpy as np
import matplotlib.pyplot as plt
import clipperpy
from sklearn.cluster import DBSCAN
from scipy.linalg import logm
import json


from optimal_detections_loader import load_data
from utils.datasets import extract_orbslam, extract_gt, extract_imu
from lac_data import FrameDataReader

def compute_relative_transform(T1, T2):
    """Compute relative transform T_rel = T2 * inv(T1)."""
    return T2 @ np.linalg.inv(T1)

def rotation_error(R_est, R_gt):
    """Compute geodesic rotation error (degrees)."""
    R_err = R_est.T @ R_gt
    cos_theta = (np.trace(R_err) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
    theta = np.degrees(np.arccos(cos_theta))
    return theta

def translation_error(t_est, t_gt):
    """Compute L2 translation error (meters)."""
    return np.linalg.norm(t_est - t_gt)


first_traverse = FrameDataReader("/home/annika/Downloads/datasets_with_imu_1/orbslam_straight_line_preset7_az100_alt2_20250917_165348.lac")
second_traverse = FrameDataReader("/home/annika/Downloads/datasets_with_imu_1/orbslam_straight_line_preset7_az210_alt10_20250917_114517.lac")

orbslam_estimates1, orbslam_frames1 = extract_imu(first_traverse)
gt_traj1, gt_frames1 = extract_gt(first_traverse)
gt_traj2, gt_frames2 = extract_gt(second_traverse)

import csv
import numpy as np

import csv
import numpy as np

# --- helpers ---
def inv_se3(T):
    R, t = T[:3, :3], T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv

def compute_relative_transform(T1, T2):
    """Relative transform from frame1 (T1) to frame2 (T2)."""
    return T2 @ inv_se3(T1)

def rotation_error_deg(R_est, R_gt):
    R_err = R_est.T @ R_gt
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def translation_error_m(t_est, t_gt):
    return np.linalg.norm(t_est - t_gt)

def pose_errors(T_est, T_gt):
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt,  t_gt  = T_gt[:3, :3],  T_gt[:3, 3]
    return rotation_error_deg(R_est, R_gt), translation_error_m(t_est, t_gt)


# --- load CSV loop closures ---
csv_file = "/home/annika/Downloads/netvlad_lc/LC__13__14.csv"
frames, poses = [], []

with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame1 = int(float(row["13_frame"]))
        frame2 = int(float(row["14_frame"]))
        frames.append((frame1, frame2))

        pose_vals = []
        for r in range(4):
            pose_vals.append([float(row[f"m{r}{c}"]) for c in range(4)])
        poses.append(np.array(pose_vals))


# --- fast frame->index maps ---
idx_map1 = {int(f): i for i, f in enumerate(np.asarray(gt_frames1).astype(int))}
idx_map2 = {int(f): i for i, f in enumerate(np.asarray(gt_frames2).astype(int))}


# --- compute errors ---
rot_errs, trans_errs = [], []

for (f1, f2), T_est in zip(frames, poses):
    if f1 not in idx_map1 or f2 not in idx_map2:
        continue

    T1, T2 = gt_traj1[idx_map1[f1]], gt_traj2[idx_map2[f2]]
    T_gt = compute_relative_transform(T1, T2)

    # direct and opposite
    rotA, transA = pose_errors(T_est, T_gt)
    rotB, transB = pose_errors(inv_se3(T_est), T_gt)

    # pick whichever is closer
    if (rotB + transB) < (rotA + transA):
        rot_errs.append(rotB)
        trans_errs.append(transB)
    else:
        rot_errs.append(rotA)
        trans_errs.append(transA)


# --- summary ---
if rot_errs:
    print(f"Mean rotation error:     {np.mean(rot_errs):.4f} deg")
    print(f"Median rotation error:   {np.median(rot_errs):.4f} deg")
    print(f"Mean translation error:  {np.mean(trans_errs):.4f} m")
    print(f"Median translation error:{np.median(trans_errs):.4f} m")
else:
    print("No valid loop closures found.")


# # Path to your CSV
# csv_file = "/home/annika/Downloads/bow_lc/LC__1__2.csv"

# frames = []
# poses = []

# with open(csv_file, "r") as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         # Frame 1 and Frame 2 as integers
#         frame1 = int(float(row["1_frame"]))
#         frame2 = int(float(row["2_frame"]))
#         frames.append((frame1, frame2))

#         # Build the 4x4 pose matrix
#         pose_vals = []
#         for r in range(4):
#             row_vals = []
#             for c in range(4):
#                 key = f"m{r}{c}"
#                 row_vals.append(float(row[key]))
#             pose_vals.append(row_vals)

#         pose = np.array(pose_vals)
#         poses.append(pose)

# # Print first 5 for checking
# for i in range(5):
#     print(f"Frames {frames[i]}:")
#     print(poses[i])
#     print()

# for i in range(5):  # just test first 5
#     f1, f2 = frames[i]
#     T_est = poses[i]  # from CSV

#     # Ground truth poses
#     idx1 = np.where(gt_frames1 == f1)[0][0]
#     idx2 = np.where(gt_frames2 == f2)[0][0]
#     T1 = gt_traj1[idx1]
#     T2 = gt_traj2[idx2]

#     # Ground truth relative transform
#     T_gt = compute_relative_transform(T1, T2)

#     # Split into rotation & translation
#     R_est, t_est = T_est[:3, :3], T_est[:3, 3]
#     R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

#     # Errors
#     rot_err = rotation_error(R_est, R_gt)
#     trans_err = translation_error(t_est, t_gt)

#     print(f"Match {i}: frames ({f1}, {f2})")
#     print(f"  Rotation error: {rot_err:.4f} deg")
#     print(f"  Translation error: {trans_err:.4f} m\n")

# import numpy as np

# # --- helpers ---
# def inv_se3(T):
#     """Fast inverse of a 4x4 rigid transform."""
#     R = T[:3, :3]
#     t = T[:3, 3]
#     Tinv = np.eye(4)
#     Tinv[:3, :3] = R.T
#     Tinv[:3, 3]  = -R.T @ t
#     return Tinv

# def compute_relative_transform(T1, T2):
#     """T_2<-1 : transform from frame1 to frame2 given world poses T1, T2."""
#     return T2 @ inv_se3(T1)

# def rotation_error_deg(R_est, R_gt):
#     R_err = R_est.T @ R_gt
#     # numerical safety
#     cos_theta = (np.trace(R_err) - 1.0) / 2.0
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     return np.degrees(np.arccos(cos_theta))

# def translation_error_m(t_est, t_gt):
#     return np.linalg.norm(t_est - t_gt)

# def pose_errors(T_est, T_gt):
#     R_est, t_est = T_est[:3, :3], T_est[:3, 3]
#     R_gt,  t_gt  = T_gt[:3, :3],  T_gt[:3, 3]
#     return rotation_error_deg(R_est, R_gt), translation_error_m(t_est, t_gt)

# # --- fast frame->index maps (avoid np.where in a loop) ---
# idx_map1 = {int(f): i for i, f in enumerate(np.asarray(gt_frames1).astype(int))}
# idx_map2 = {int(f): i for i, f in enumerate(np.asarray(gt_frames2).astype(int))}

# def frame_to_idx(map_, f):
#     try:
#         return map_[int(f)]
#     except KeyError:
#         raise KeyError(f"Frame {f} not found in GT frames")

# # --- compare both directions for the first N rows (and compute summary) ---
# N = min(20, len(frames))  # change as you like
# errs_direct = []  # T_est vs T_gt_2<-1
# errs_flip   = []  # inv(T_est) vs T_gt_2<-1
# errs_best   = []

# for i in range(N):
#     f1, f2 = frames[i]
#     T_est = poses[i]

#     i1 = frame_to_idx(idx_map1, f1)
#     i2 = frame_to_idx(idx_map2, f2)
#     T1, T2 = gt_traj1[i1], gt_traj2[i2]

#     # GT relative: from frame1 (traverse1) to frame2 (traverse2)
#     T_gt_2from1 = compute_relative_transform(T1, T2)

#     # Option A: compare estimate directly to GT
#     rotA, transA = pose_errors(T_est, T_gt_2from1)
#     # Option B: compare the inverse (opposite) of estimate to GT
#     rotB, transB = pose_errors(inv_se3(T_est), T_gt_2from1)

#     errs_direct.append((rotA, transA))
#     errs_flip.append((rotB, transB))

#     # Choose better by simple sum (you can weight these if you want)
#     sumA, sumB = rotA + transA, rotB + transB
#     if sumB < sumA:
#         choice = "OPPOSITE (inv est)"
#         rot_best, trans_best = rotB, transB
#     else:
#         choice = "DIRECT"
#         rot_best, trans_best = rotA, transA

#     errs_best.append((rot_best, trans_best))

#     if i < 10:  # print a few detailed lines
#         print(f"Row {i} frames ({f1}, {f2})")
#         print(f"  Direct : rot {rotA:.4f} deg, trans {transA:.4f} m")
#         print(f"  Opposite(inv est): rot {rotB:.4f} deg, trans {transB:.4f} m")
#         print(f"  -> Using {choice}\n")

# # quick summary
# def summarize(errs, tag):
#     if not errs: return
#     r = np.array([e[0] for e in errs])
#     t = np.array([e[1] for e in errs])
#     print(f"{tag}  mean rot {r.mean():.4f}°, median {np.median(r):.4f}° | "
#           f"mean trans {t.mean():.4f} m, median {np.median(t):.4f} m")

# summarize(errs_direct, "DIRECT ")
# summarize(errs_flip,   "OPPOSITE(inv est) ")
# summarize(errs_best,   "BEST-OF-TWO     ")

