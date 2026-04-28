from optimal_detections_loader import load_data, get_pose, get_transform, get_detections
import numpy as np 

import numpy as np
import matplotlib.pyplot as plt
import clipperpy
from sklearn.cluster import DBSCAN
from scipy.linalg import logm


def pose_error(R, t):
    """
    Compute rotation error (deg) and translation error (L2 norm)
    compared to identity pose.
    """
    # --- Method 1: trace formula ---
    trace_val = np.trace(R)
    cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    rot_error_rad_trace = np.arccos(cos_theta)

    # --- Method 2: log map (more stable) ---
    R_log = logm(R)
    rot_vec = np.array([R_log[2,1], R_log[0,2], R_log[1,0]])  # vee operator
    rot_error_rad_log = np.linalg.norm(rot_vec)

    # Use the log map result (better for small angles)
    rot_error_deg = np.degrees(rot_error_rad_log)

    # Translation error
    trans_error = np.linalg.norm(t)

    return rot_error_deg, trans_error, np.degrees(rot_error_rad_trace)

def reduce_clusters(points, eps=0.05, min_samples=1, method="centroid"):
    """
    Cluster points and keep only clusters with at least `min_samples`.

    Args:
        points: (N, D) numpy array of points
        eps: float, distance threshold for clustering (tune to your scale)
        min_samples: minimum detections required to keep a cluster
        method: "centroid" (average), "median", or "first"

    Returns:
        reduced_points: (M, D) array with one representative per cluster
        labels: cluster labels for each original point (-1 = noise)
    """
    clustering = DBSCAN(eps=eps, min_samples=1).fit(points)
    labels = clustering.labels_

    reduced = []
    kept_labels = []
    for lab in np.unique(labels):
        if lab == -1:
            continue  # DBSCAN noise
        cluster_pts = points[labels == lab]
        if len(cluster_pts) < min_samples:
            continue  # drop small clusters

        if method == "centroid":
            reduced.append(cluster_pts.mean(axis=0))
        elif method == "median":
            reduced.append(np.median(cluster_pts, axis=0))
        else:  # "first"
            reduced.append(cluster_pts[0])

        kept_labels.append(lab)

    return np.array(reduced), labels

def generateAssociationList(N1, N2):
    assocList = np.zeros((N1*N2,2),np.int32)

    i = 0

    for n1 in range(N1):
        for n2 in range(N2):
            assocList[i,0] = n1
            assocList[i,1] = n2
            i += 1

    return assocList


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


def rotation_error_deg(R):
    """Compute angle error between R and identity in degrees."""
    # Clamp trace to [-1,3] range for numerical stability
    trace = np.trace(R)
    cos_theta = max(min((trace - 1) / 2, 1.0), -1.0)
    return np.degrees(np.arccos(cos_theta))

def translation_error(t):
    """Compute translation magnitude."""
    return np.linalg.norm(t)

def sliding_window_registration(locs1, locs2, window_size=50, stride=10, 
                                min_assoc=10, thresh=0.1):
    accepted_results = []

    n1, n2 = len(locs1), len(locs2)

    for start1 in range(0, n1 - window_size + 1, stride):
        sub1 = locs1[start1:start1 + window_size]

        for start2 in range(0, n2 - window_size + 1, stride):
            sub2 = locs2[start2:start2 + window_size]

            # Associations between the two subsets
            Ain = findAssociationsWithClipper(sub1, sub2, thresh)

            if Ain.shape[0] >= min_assoc:
                assoc1 = sub1[Ain[:, 0]]
                assoc2 = sub2[Ain[:, 1]]

                R, t = rigid_transform_3D(assoc1.T, assoc2.T)

                err_R = rotation_error_deg(R)
                err_t = translation_error(t)

                accepted_results.append({
                    "R": R,
                    "t": t,
                    "start1": start1, "end1": start1 + window_size,
                    "start2": start2, "end2": start2 + window_size,
                    "num_assoc": Ain.shape[0],
                    "rot_err_deg": err_R,
                    "trans_err": err_t
                })

    # Compute averages
    if accepted_results:
        avg_rot = np.mean([r["rot_err_deg"] for r in accepted_results])
        avg_trans = np.mean([r["trans_err"] for r in accepted_results])
    else:
        avg_rot, avg_trans = None, None

    return accepted_results, avg_rot, avg_trans


def findAssociationsWithClipper(landmarkPositions1, landmarkPositions2, epsilon):
    iparams = clipperpy.invariants.EuclideanDistanceParams()
    iparams.epsilon = epsilon
    iparams.sigma = 0.5 * iparams.epsilon
    invariant = clipperpy.invariants.EuclideanDistance(iparams)

    params = clipperpy.Params()
    params.rounding = clipperpy.Rounding.DSD_HEU
    clipper = clipperpy.CLIPPER(invariant, params)

    numLandmarks1, _ = landmarkPositions1.shape
    numLandmarks2, _ = landmarkPositions2.shape

    print("num landmarks in 1: ", numLandmarks1)
    print("num landmarks in 2: ", numLandmarks2)

    assocList = generateAssociationList(numLandmarks1, numLandmarks2)

    clipper.score_pairwise_consistency(landmarkPositions1.T, landmarkPositions2.T, assocList)

    A = clipper.get_affinity_matrix()
    C = clipper.get_constraint_matrix()

    # feed updated matrix back to Clipper and find associations
    clipper.set_matrix_data(A,C)
    clipper.solve()
    Ain = clipper.get_selected_associations()

    return Ain


def get_all_detection_locs(data, camera: str | None = None, return_frames: bool = False):
    """
    Collect all detection 'location' vectors from data['detections'].
    - camera: None for all cameras, or e.g. 'front_camera' / 'back_camera'
    - return_frames: also return the matching frame id for each point
    Returns: (N,3) array (and optionally (N,) frame ids)
    """
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

def get_all_detection_locs_global(data, camera: str | None = None, return_frames: bool = False):
    """
    Same as above, but each point is transformed into the GLOBAL frame using the frame's 4x4 T.
    Assumes T maps rover/local -> global (homogeneous transform).
    """
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
                gp = (T @ np.r_[np.asarray(loc, float), 1.0])[:3]  # one-liner transform
                pts_g.append(gp)
                frames.append(int(data["frame"][i]))
    pts_g = np.asarray(pts_g, dtype=float)
    return (pts_g, np.asarray(frames)) if return_frames else pts_g

def plot_detection_locs_xy(points, points2, title="Detections (XY)", xlabel="x", ylabel="y"):
    """Scatter plot (XY) for either local or global points."""
    if points.size == 0:
        raise ValueError("No points to plot.")
    plt.figure()
    plt.scatter(points[:,0],  points[:,1],  s=6, color='red',  label='set 1')
    plt.scatter(points2[:,0], points2[:,1], s=6, color='blue', label='set 2')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.show()

def plot_detection_locs_3d(points, title="Detections (3D)"):
    """3D scatter for either local or global points."""
    if points.size == 0:
        raise ValueError("No points to plot.")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    plt.show()

def plot_associations_xy(points1, points2, associations, 
                         title="Associations (XY)", xlabel="x", ylabel="y"):
    """
    Plot two sets of 2D points and draw lines between associated pairs.

    points1: (N1, 2) array
    points2: (N2, 2) array
    associations: (M, 2) array of indices [i, j] linking points1[i] <-> points2[j]
    """
    if points1.size == 0 or points2.size == 0:
        raise ValueError("No points to plot.")

    plt.figure()
    plt.scatter(points1[:, 0], points1[:, 1], s=6, color="red",  label="set 1")
    plt.scatter(points2[:, 0], points2[:, 1], s=6, color="blue", label="set 2")

    for (i, j) in associations:
        p1 = points1[i]
        p2 = points2[j]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linewidth=0.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()


# data = load_data("/home/annika/Downloads/Processed_LAC/orbslam_straight_line_preset1_default_20250917_102024_optimal_detections/orbslam_straight_line_preset1_default_20250917_102024_optimal_detections.csv")
data2 = load_data("/home/annika/Downloads/CSVs/CSVs/orbslam_straight_line_preset1_default_20250917_102024_detections.csv")
data = load_data("/home/annika/Documents/updated_sets/updated_sets/orbslam_straight_line_preset1_default_20250926_182546_detections.csv")
# data2 = data

# Local-frame detections, all cameras
locs_local1_pre = get_all_detection_locs(data)               # (N,3)
locs_local2_pre = get_all_detection_locs(data2)

# Global-frame detections
locs_global1_pre = get_all_detection_locs_global(data)       # (N,3)
locs_global2_pre = get_all_detection_locs_global(data2)

locs_global1, labels = reduce_clusters(locs_global1_pre, min_samples=1, eps=0.05)
locs_global2, labels = reduce_clusters(locs_global2_pre, min_samples=1, eps=0.05)

plot_detection_locs_xy(locs_global1, locs_global2, "Global detections (XY)")

Ain = findAssociationsWithClipper(locs_global1, locs_global2, 0.1)
associatedPointLocations_1 = locs_global1[Ain[:,0]]
associatedPointLocations_2 = locs_global2[Ain[:,1]]

R, t = rigid_transform_3D(associatedPointLocations_1.T, associatedPointLocations_2.T)

plot_associations_xy(locs_global1, locs_global2, Ain)


Rerr, Terr, deg = pose_error(R, t)
print("Rerr: ", Rerr)
print("rerr 2", deg )
print("Terr: ", Terr)
print("Number of associations: ", len(Ain))

# Example usage
results, avg_rot, avg_trans = sliding_window_registration(
    locs_global1, locs_global2,
    window_size=50, stride=10, min_assoc=13, thresh=0.1
)

for i, r in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(f"Window1: {r['start1']}–{r['end1']}  |  Window2: {r['start2']}–{r['end2']}")
    print(f"Associations: {r['num_assoc']}")
    print(f"Rotation error: {r['rot_err_deg']:.4f} deg")
    print(f"Translation error: {r['trans_err']:.4f} m")
    print("R:\n", r["R"])
    print("t.T:", r["t"].T, "\n")

# Print summary
if avg_rot is not None:
    print(f"Average rotation error: {avg_rot:.4f} deg")
    print(f"Average translation error: {avg_trans:.4f} m")
else:
    print("No valid registrations found.")