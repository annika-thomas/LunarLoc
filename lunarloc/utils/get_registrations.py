#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import clipperpy
from sklearn.cluster import DBSCAN
from scipy.linalg import logm
import json

from optimal_detections_loader import load_data
from utils.datasets import extract_orbslam, extract_gt
from lac_data import FrameDataReader

# -----------------------------
# Pose utilities
# -----------------------------

def pose_error(R, t):
    """Compute rotation error (deg) and translation error (L2 norm)."""
    R_log = logm(R)
    rot_vec = np.array([R_log[2,1], R_log[0,2], R_log[1,0]])  # vee operator
    rot_error_rad_log = np.linalg.norm(rot_vec)
    rot_error_deg = np.degrees(rot_error_rad_log)
    trans_error = np.linalg.norm(t)
    return rot_error_deg, trans_error

def rigid_transform_3D(A, B):
    """Rigid transform from A->B using SVD."""
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)
    Am, Bm = A - centroid_A, B - centroid_B
    U, S, Vt = np.linalg.svd(Am @ Bm.T)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B
    return R, t

def get_all_detection_locs_global_override(data, frames_list, poses_list,
                                           camera=None, return_frames=False):
    """
    Transform detections into the global frame using external poses
    (e.g. ORB-SLAM estimates or ground truth).
    """
    pts_g, frames = [], []
    for i, det in enumerate(data["detections"]):
        if not isinstance(det, dict): 
            continue
        frame_id = int(data["frame"][i])
        T = pose_at_frame(frame_id, frames_list, poses_list)  # <-- ORB-SLAM or GT

        cam_items = det.items() if camera is None else [(camera, det.get(camera, {}))]
        for _, cam_dict in cam_items:
            if not isinstance(cam_dict, dict): 
                continue
            for d in cam_dict.values():
                loc = d.get("location")
                if loc is None: 
                    continue
                gp = (T @ np.r_[np.asarray(loc, float), 1.0])[:3]
                pts_g.append(gp)
                frames.append(frame_id)
    pts_g = np.asarray(pts_g, dtype=float)
    return (pts_g, np.asarray(frames)) if return_frames else pts_g


def rotation_error_deg(R):
    trace = np.trace(R)
    cos_theta = np.clip((trace - 1) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def translation_error(t):
    return np.linalg.norm(t)

def pose_at_frame(frame_id, frames_array, poses_array):
    """Return pose at the closest available frame in frames_array."""
    idx = np.argmin(np.abs(frames_array - frame_id))
    return poses_array[idx]


# -----------------------------
# Clustering
# -----------------------------

def reduce_clusters(points, frames, eps=0.01, min_samples=5, method="centroid"):
    """
    Cluster points and keep only clusters with at least `min_samples`.
    Returns reduced points and representative frame ids.
    """
    clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
    labels = clustering.labels_

    reduced, reduced_frames = [], []
    for lab in np.unique(labels):
        if lab == -1:
            continue
        cluster_pts = points[labels == lab]
        cluster_frames = frames[labels == lab]
        if len(cluster_pts) < min_samples:
            continue
        if method == "centroid":
            reduced_point = cluster_pts.mean(axis=0)
        elif method == "median":
            reduced_point = np.median(cluster_pts, axis=0)
        else:
            reduced_point = cluster_pts[0]
        reduced_frame = int(np.median(cluster_frames))
        reduced.append(reduced_point)
        reduced_frames.append(reduced_frame)
    return np.array(reduced), np.array(reduced_frames), labels

# -----------------------------
# Association / registration
# -----------------------------

def generateAssociationList(N1, N2):
    assocList = np.zeros((N1*N2,2),np.int32)
    i = 0
    for n1 in range(N1):
        for n2 in range(N2):
            assocList[i,0] = n1
            assocList[i,1] = n2
            i += 1
    return assocList

def findAssociationsWithClipper(lm1, lm2, epsilon=0.1):
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.epsilon = epsilon
    iparams.sigma = 0.5 * iparams.epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)
    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    clipper = clipperpy.CLIPPER(invariant, params)
    assocList = generateAssociationList(len(lm1), len(lm2))
    clipper.score_pairwise_consistency(lm1.T, lm2.T, assocList)
    A, C = clipper.get_affinity_matrix(), clipper.get_constraint_matrix()
    clipper.set_matrix_data(A,C)
    clipper.solve()
    return clipper.get_selected_associations()

def sliding_window_registration(locs1, locs2, frames1, frames2,
                                orbslam_frames, orbslam_poses,
                                gt_frames, gt_poses,
                                window_size=50, stride=10,
                                min_assoc=10, thresh=0.1):
    registrations = []
    n1, n2 = len(locs1), len(locs2)

    for start1 in range(0, n1 - window_size + 1, stride):
        sub1 = locs1[start1:start1 + window_size]
        frame1 = frames1[start1]

        for start2 in range(0, n2 - window_size + 1, stride):
            sub2 = locs2[start2:start2 + window_size]

            Ain = findAssociationsWithClipper(sub1, sub2, thresh)
            if Ain.shape[0] >= min_assoc:
                assoc1 = sub1[Ain[:, 0]]
                assoc2 = sub2[Ain[:, 1]]
                R, t = rigid_transform_3D(assoc1.T, assoc2.T)
                err_R = rotation_error_deg(R)
                err_t = translation_error(t)

                # ground truth relative at this frame
                T_orb = pose_at_frame(frame1, orbslam_frames, orbslam_poses)
                T_gt = pose_at_frame(frame1, gt_frames, gt_poses)
                T_rel_gt = T_gt @ np.linalg.inv(T_orb)

                # build estimated transform
                T_est = np.eye(4)
                T_est[:3,:3] = R
                T_est[:3, 3] = t.flatten()

                # error transform = how far off our estimate is from GT
                T_err = T_rel_gt @ np.linalg.inv(T_est)


                registrations.append({
                    "frame": int(frame1),

                    # loop closure estimate (to inject into PGO)
                    "R_est": R.tolist(),
                    "t_est": t.flatten().tolist(),

                    # ground truth relative correction at this frame
                    "T_rel_gt": T_rel_gt.tolist(),

                    # error of estimate vs ground truth
                    "rot_err_vs_gt": float(rotation_error_deg(T_err[:3,:3])),
                    "trans_err_vs_gt": float(np.linalg.norm(T_err[:3,3])),

                    # metadata
                    "num_assoc": int(Ain.shape[0])
                })

    return registrations

# -----------------------------
# Detection helpers
# -----------------------------

def get_all_detection_locs(data, camera=None, return_frames=False):
    pts, frames = [], []
    for i, det in enumerate(data["detections"]):
        if not isinstance(det, dict): 
            continue
        cam_items = det.items() if camera is None else [(camera, det.get(camera, {}))]
        for _, cam_dict in cam_items:
            if not isinstance(cam_dict, dict): 
                continue
            for d in cam_dict.values():
                loc = d.get("location")
                if loc is None: 
                    continue
                pts.append(loc)
                frames.append(int(data["frame"][i]))
    pts = np.asarray(pts, dtype=float)
    return (pts, np.asarray(frames)) if return_frames else pts

def get_all_detection_locs_global(data, camera=None, return_frames=False):
    pts_g, frames = [], []
    for i, det in enumerate(data["detections"]):
        if not isinstance(det, dict): 
            continue
        T = data["transform"][i]
        cam_items = det.items() if camera is None else [(camera, det.get(camera, {}))]
        for _, cam_dict in cam_items:
            if not isinstance(cam_dict, dict): 
                continue
            for d in cam_dict.values():
                loc = d.get("location")
                if loc is None: 
                    continue
                gp = (T @ np.r_[np.asarray(loc, float), 1.0])[:3]
                pts_g.append(gp)
                frames.append(int(data["frame"][i]))
    pts_g = np.asarray(pts_g, dtype=float)
    return (pts_g, np.asarray(frames)) if return_frames else pts_g

# -----------------------------
# Main script
# -----------------------------

if __name__ == "__main__":
    # load traverses
    second_traverse = FrameDataReader("/home/annika/Downloads/data/orbslam_straight_line_preset1_default_20250917_102024.lac")
    first_traverse = FrameDataReader("/home/annika/Downloads/data/orbslam_straight_line_preset1_default_20250917_103626.lac")

    orbslam_estimates1, orbslam_frames1 = extract_orbslam(first_traverse)
    gt_traj1, gt_frames1 = extract_gt(first_traverse)
    gt_traj2, gt_frames2 = extract_gt(second_traverse)

    data2 = load_data("/home/annika/Downloads/CSVs/CSVs/orbslam_straight_line_preset1_default_20250917_102024_detections.csv")
    data = load_data("/home/annika/Documents/updated_sets/updated_sets/orbslam_straight_line_preset1_default_20250926_182546_detections.csv")

    # detections → global frame
    # locs_global1_pre, frames_global1 = get_all_detection_locs_global(data, return_frames=True)
    locs_global1_pre, frames_global1 = get_all_detection_locs_global_override(
        data, orbslam_frames1, orbslam_estimates1, return_frames=True
    )
    locs_global2_pre, frames_global2 = get_all_detection_locs_global(data2, return_frames=True)

    # clustering (with frame tracking)
    locs_global1, frames_global1, _ = reduce_clusters(locs_global1_pre, frames_global1, eps=0.05, min_samples=1)
    locs_global2, frames_global2, _ = reduce_clusters(locs_global2_pre, frames_global2, eps=0.05, min_samples=1)

    # run sliding window registration
    registrations = sliding_window_registration(
        locs_global1, locs_global2,
        frames_global1, frames_global2,
        orbslam_frames1, orbslam_estimates1,
        gt_frames1, gt_traj1,
        window_size=50, stride=10, min_assoc=13, thresh=0.1
    )

    # save to json
    with open("registrations.json", "w") as f:
        json.dump(registrations, f, indent=2)
    print(f"✅ Saved {len(registrations)} registrations to registrations.json")

    # sanity check reload
    with open("registrations.json", "r") as f:
        loaded = json.load(f)

    print("\n--- Sanity check (first 5 registrations) ---")
    for r in loaded[:5]:
        print(
            f"Frame={r['frame']} | "
            f"LoopClosure Δ: R_est(3x3), t_est={np.round(r['t_est'],3)} | "
            f"GT err: rot={r['rot_err_vs_gt']:.3f}°, trans={r['trans_err_vs_gt']:.3f}m | "
            f"assoc={r['num_assoc']}"
        )
