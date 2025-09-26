from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pyt_t

import gtsam
from gtsam.symbol_shorthand import X
from utils.datasets import extract_orbslam, extract_gt, tf_at_frame

from lac_data import FrameDataReader


def tf_array_from_ids(result: gtsam.Values, ids):
    """Extract Pose3 matrices from result for a list of (symbol) ids, in order."""
    mats = []
    for i in ids:
        mats.append(result.atPose3(X(i)).matrix())
    return np.asarray(mats)


def align(model, data):
    """Horn alignment (closed form)."""
    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = model - model.mean(1)
    data_zerocentered = data - data.mean(1)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.T)
    S = np.eye(3)
    if (np.linalg.det(U) * np.linalg.det(Vh) < 0):
        S[2, 2] = -1
    rot = U @ S @ Vh
    trans = data.mean(1) - rot @ model.mean(1)

    model_aligned = rot @ model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]
    return rot, trans, trans_error


def align_trajectories(t_pred: np.ndarray, t_gt: np.ndarray):
    """
    Args:
        t_pred: (n, 3) translations
        t_gt:   (n, 3) translations
    Returns:
        t_align: (n, 3) aligned translations
    """
    t_align = np.matrix(t_pred).T
    R, t, _ = align(t_align, np.matrix(t_gt).T)
    t_align = (R @ t_align + t)
    return np.asarray(t_align).T


def pose_error(t_pred: np.ndarray, t_gt: np.ndarray):
    """
    Translation-only ATE stats.
    """
    n = t_pred.shape[0]
    trans_error = np.linalg.norm(t_pred - t_gt, axis=1)
    return {
        "compared_pose_pairs": n,
        "rmse": float(np.sqrt(np.dot(trans_error, trans_error) / n)),
        "mean": float(np.mean(trans_error)),
        "median": float(np.median(trans_error)),
        "std": float(np.std(trans_error)),
        "min": float(np.min(trans_error)),
        "max": float(np.max(trans_error)),
    }


# Noise is rotation-first (rad), then translation (m)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.001, 0.001, 0.001,   # rot (rad)  ~0.057 deg
              0.005, 0.005, 0.005],  # trans (m)  5 mm
             dtype=np.float64)
)

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.0005, 0.0005, 0.0005,  # rot (rad)
              0.001,  0.001,  0.001],  # trans (m)
             dtype=np.float64)
)

# “Soft” localization noise (GPS-like, still fairly strong but not a clamp)
LOCALIZATION_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.0001, 0.0001, 0.0001,  # rot (rad)
              0.02,   0.02,   0.02],   # trans (m) 2 cm
             dtype=np.float64)
)


def main(first_traverse: FrameDataReader, silent: bool = False):
    # --- Load ORB-SLAM (agent 1) and GT from the traverse ---
    orbslam_estimates1, orbslam_frames1 = extract_orbslam(first_traverse)
    gt_traj1, gt_frames1 = extract_gt(first_traverse)

    # Ensure same length if your extractors include trailing frame, etc.
    N = min(len(orbslam_estimates1), len(gt_traj1))
    orbslam_estimates1 = orbslam_estimates1[:N]
    orbslam_frames1 = orbslam_frames1[:N]
    gt_traj1 = gt_traj1[:N]

    # Quick pre-optimization metrics (translation only)
    gt_t1 = gt_traj1[:, :3, 3]
    estimated_t1 = orbslam_estimates1[:, :3, 3]
    estimated_t_aligned1 = align_trajectories(estimated_t1, gt_t1)
    ate1 = pose_error(estimated_t1, gt_t1)
    ate_aligned1 = pose_error(estimated_t_aligned1, gt_t1)

    print("Pre-opt ATE:", ate1)
    print("Pre-opt ATE (aligned):", ate_aligned1)

    # --- Build factor graph ---
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Between factors from ORB-SLAM relative motion
    for i in range(N - 1):
        T_w_cur = orbslam_estimates1[i]
        T_w_next = orbslam_estimates1[i + 1]
        T_cur_next = pyt_t.invert_transform(T_w_cur) @ T_w_next
        graph.add(gtsam.BetweenFactorPose3(X(i), X(i + 1), gtsam.Pose3(T_cur_next), ODOMETRY_NOISE))

    # Initial guesses = raw ORB-SLAM estimates
    for i, T_w in enumerate(orbslam_estimates1):
        initial.insert(X(i), gtsam.Pose3(T_w))

    # Strong(ish) anchor on the first pose to GT
    graph.add(gtsam.PriorFactorPose3(
        X(0),
        gtsam.Pose3(tf_at_frame(first_traverse[orbslam_frames1[0]])),
        PRIOR_NOISE,
    ))

    # --- Soft localization events modeled as BetweenFactors to dummy GT nodes ---
    dummy_offset = 10_000  # keep IDs separate
    # Choose localization frames by INDEX (every 50 frames here)
    localization_idx = set(range(0, N, 50))
    # (If you’d rather use only frames with GT present: intersection is already enforced by N above.)
    # --- Soft localization events (GPS-like updates) ---
    # rotation-first (rad), then translation (m)
    LOCALIZATION_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([0.01, 0.01, 0.01,   # rot ~0.57°
                0.10, 0.10, 0.10],  # trans 10 cm
                dtype=np.float64)
    )

    # Strong anchor to keep each dummy GT node fixed at GT (small sigmas)
    GT_ANCHOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-6, 1e-6, 1e-6,   # rot
                1e-6, 1e-6, 1e-6],  # trans
                dtype=np.float64)
    )

    dummy_offset = 10_000  # keep IDs separate from trajectory nodes

    # choose frames by INDEX (e.g., every 50th)
    localization_idx = set(range(0, N, 50))

    for i in range(N):
        if i in localization_idx:
            frame_id = orbslam_frames1[i]
            T_w_gt = tf_at_frame(first_traverse[frame_id])

            # Create a dummy node fixed (via strong prior) at the GT pose
            Y_i = X(dummy_offset + i)
            initial.insert(Y_i, gtsam.Pose3(T_w_gt))
            graph.add(gtsam.PriorFactorPose3(Y_i, gtsam.Pose3(T_w_gt), GT_ANCHOR_NOISE))

            # Measurement is identity: we want X(i) == Y_i up to LOCALIZATION_NOISE
            graph.add(gtsam.BetweenFactorPose3(
                X(i),
                Y_i,
                gtsam.Pose3(),          # identity transform
                LOCALIZATION_NOISE
            ))


    # --- Optimize ---
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    # --- Extract ONLY the agent-1 trajectory from the result (exclude dummy nodes) ---
    agent1_ids = list(range(N))
    agent1_final = tf_array_from_ids(result, agent1_ids)

    print("len(final agent1):", len(agent1_final))
    print("len(gt):", len(gt_t1))

    # --- Metrics after optimization ---
    final_t1 = agent1_final[:, :3, 3]
    final_t_aligned1 = align_trajectories(final_t1, gt_t1)
    ate_final1 = pose_error(final_t1, gt_t1)
    ate_final_aligned1 = pose_error(final_t_aligned1, gt_t1)

    print("Post-opt ATE:", ate_final1)
    print("Post-opt ATE (aligned):", ate_final_aligned1)

    if not silent:
        # Simple XY plot (GT vs. ORB-SLAM vs. Final)
        plt.figure()
        plt.plot(gt_t1[:, 0], gt_t1[:, 1], label="GT")
        plt.plot(estimated_t1[:, 0], estimated_t1[:, 1], label="ORB-SLAM (init)")
        plt.plot(final_t1[:, 0], final_t1[:, 1], label="Final (optimized)")
        plt.axis('equal'); plt.legend(); plt.title("Trajectories (XY)")
        plt.show()


if __name__ == "__main__":
    # Example usage: update to your paths
    first_traverse = FrameDataReader("/home/annika/Downloads/data/orbslam_straight_line_preset1_default_20250917_102024.lac")
    main(first_traverse)
