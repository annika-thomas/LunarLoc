"""
v3: front-camera registrations between two CSVs in the new format.

Reads `front_detections_global` directly (no per-frame transform), DBSCAN-
fuses points across frames so each physical landmark becomes one cluster,
then runs sliding-window CLIPPER + rigid registration between path1 and
path2.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import clipperpy
from sklearn.cluster import DBSCAN


# ---------- core helpers ----------
def generateAssociationList(N1, N2):
    L = np.zeros((N1 * N2, 2), np.int32)
    i = 0
    for n1 in range(N1):
        for n2 in range(N2):
            L[i, 0] = n1
            L[i, 1] = n2
            i += 1
    return L


def findAssociationsWithClipper(P1, P2, epsilon):
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.epsilon = epsilon
    iparams.sigma = 0.5 * iparams.epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)

    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    clipper = clipperpy.CLIPPER(invariant, params)

    n1, _ = P1.shape
    n2, _ = P2.shape
    print(f"num landmarks: {n1} vs {n2}")
    assoc = generateAssociationList(n1, n2)
    clipper.score_pairwise_consistency(P1.T, P2.T, assoc)

    A = clipper.get_affinity_matrix()
    C = clipper.get_constraint_matrix()
    clipper.set_matrix_data(A, C)
    clipper.solve()
    return clipper.get_selected_associations()


def reduce_clusters(points, frames, eps=0.05):
    """Fuse points within `eps` meters into one centroid per cluster."""
    if len(points) == 0:
        return points, frames
    labels = DBSCAN(eps=eps, min_samples=1).fit(points).labels_
    reduced, reduced_frames = [], []
    for lab in np.unique(labels):
        mask = labels == lab
        reduced.append(points[mask].mean(axis=0))
        reduced_frames.append(int(np.median(frames[mask])))
    # keep frame order so sliding windows step through the trajectory
    order = np.argsort(reduced_frames)
    return np.array(reduced)[order], np.array(reduced_frames)[order]


def rigid_transform_3D(A, B):
    cA = np.mean(A, axis=1).reshape(-1, 1)
    cB = np.mean(B, axis=1).reshape(-1, 1)
    H = (A - cA) @ (B - cB).T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = -R @ cA + cB
    return R, t


# ---------- new-CSV loader ----------
def _parse_detections(cell):
    if isinstance(cell, dict):
        return cell
    if pd.isna(cell):
        return {}
    s = str(cell).strip()
    if not s or s == "{}":
        return {}
    return json.loads(s)


def load_global(csv_path, cameras=("front_detections_global",)):
    """Return (points Nx3, frames N) for the given *_global columns."""
    df = pd.read_csv(csv_path)
    if "frame" not in df.columns:
        raise KeyError(f"missing 'frame' in {csv_path}")
    for c in cameras:
        if c not in df.columns:
            raise KeyError(f"missing '{c}' in {csv_path}")
    pts, fr = [], []
    for col in cameras:
        for f, cell in zip(df["frame"].tolist(), df[col].tolist()):
            det = _parse_detections(cell)
            for d in det.values():
                loc = d.get("location")
                if loc is None:
                    continue
                pts.append(loc)
                fr.append(int(f))
    return np.asarray(pts, dtype=float), np.asarray(fr)


# ---------- registration ----------
def sliding_window_registration(locs1, locs2, frames1,
                                window_size=50, stride=10,
                                min_assoc=10, thresh=0.1):
    out = []
    n1, n2 = len(locs1), len(locs2)
    for s1 in range(0, n1 - window_size + 1, stride):
        sub1 = locs1[s1:s1 + window_size]
        f1 = frames1[s1 + window_size // 2]
        for s2 in range(0, n2 - window_size + 1, stride):
            sub2 = locs2[s2:s2 + window_size]
            Ain = findAssociationsWithClipper(sub1, sub2, thresh)
            if Ain.shape[0] < min_assoc:
                continue
            R, t = rigid_transform_3D(sub1[Ain[:, 0]].T, sub2[Ain[:, 1]].T)
            out.append({
                "frame": int(f1),
                "R_est": R.tolist(),
                "t_est": t.flatten().tolist(),
                "num_assoc": int(Ain.shape[0]),
            })
    return out


if __name__ == "__main__":
    csv1 = "/home/annika/Downloads/optimal_detections/optimal_detection_output_20260426_113529/path1_optimal_detections.csv"
    csv2 = "/home/annika/Downloads/optimal_detections/optimal_detection_output_20260426_114251/path3-cross_optimal_detections.csv"

    cameras = ("front_detections_global", "back_detections_global")
    locs1_raw, fr1_raw = load_global(csv1, cameras)
    locs2_raw, fr2_raw = load_global(csv2, cameras)
    locs1, fr1 = reduce_clusters(locs1_raw, fr1_raw, eps=0.05)
    locs2, fr2 = reduce_clusters(locs2_raw, fr2_raw, eps=0.05)
    print(f"path1: {len(locs1_raw)} raw -> {len(locs1)} fused | "
          f"path2: {len(locs2_raw)} raw -> {len(locs2)} fused")

    plt.figure(figsize=(7, 6))
    if len(locs1):
        plt.scatter(locs1[:, 0], locs1[:, 1], s=8, c="red",  alpha=0.5, label="path1")
    if len(locs2):
        plt.scatter(locs2[:, 0], locs2[:, 1], s=8, c="blue", alpha=0.5, label="path2")
    plt.legend(); plt.axis("equal"); plt.title("front_detections_global (XY)")
    plt.show()

    registrations = sliding_window_registration(
        locs1, locs2, fr1,
        window_size=50, stride=10, min_assoc=8, thresh=0.1,
    )

    with open("registrations_v3.json", "w") as f:
        json.dump(registrations, f, indent=2)
    print(f"\nsaved {len(registrations)} registrations to registrations_v3.json")
    for r in registrations[:5]:
        print(f"  frame={r['frame']:5d} | assoc={r['num_assoc']:3d} | t_est={np.round(r['t_est'], 3)}")
