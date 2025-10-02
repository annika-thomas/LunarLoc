import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
from pathlib import Path

from utils.plot import plot_loop_closures
from lac_data import PlaybackAgent


N_ORB_FEATURES = 2000
RANSAC_MIN_INLIERS = 20

PLOT_ASSOCIATIONS = True


def match_orb_features(img1: np.ndarray, img2: np.ndarray):
    # Find ORB features on the images
    orb = cv2.ORB_create(nfeatures=N_ORB_FEATURES)
    kps1, des1 = orb.detectAndCompute(img1, None)
    kps2, des2 = orb.detectAndCompute(img2, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowes ratio
    good_matches = []
    for pair in matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return kps1, kps2, good_matches


def recover_transform(kps1: list, kps2: list, matches: list[cv2.DMatch], K: np.ndarray):
    # we need at least 8 matches
    if len(matches) < 8:
        return None

    # Find essential matrix
    pts1 = np.array([kps1[m.queryIdx].pt for m in matches], dtype=np.float32)
    pts2 = np.array([kps2[m.trainIdx].pt for m in matches], dtype=np.float32)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
    if E is None or mask is None:
        return None

    mask = mask.ravel().astype(bool)
    inliers = np.sum(mask)
    if inliers < RANSAC_MIN_INLIERS:
        return None

    # Get pose
    _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask.astype(np.uint8))
    mask_pose = mask_pose.ravel().astype(bool)
    inliers_pose = np.sum(mask_pose)
    if inliers_pose < RANSAC_MIN_INLIERS:
        return None

    return R, t.reshape(3, 1)


def recover_scale(
    agent_a: PlaybackAgent, agent_b: PlaybackAgent, agent_frames: tuple, t: np.ndarray
):
    i, j = agent_frames
    frame_i = agent_a._frame_data[int(i)]
    t_i = np.array([frame_i["x"], frame_i["y"], frame_i["z"]])
    frame_j = agent_b._frame_data[int(j)]
    t_j = np.array([frame_j["x"], frame_j["y"], frame_j["z"]])

    print("Agent A t:", t_i)
    print("Agent B t:", t_j)

    # True scale between frames i and j
    s = np.linalg.norm(t_j - t_i)
    est_t = (t / np.linalg.norm(t)) * s
    print("Estimate dt:", est_t.flatten())
    print("Error:", est_t.flatten() - (t_j - t_i))
    print("Error Norm:", np.linalg.norm(est_t.flatten() - (t_j - t_i)))
    print()
    return est_t


# NOTE: From original MAPLE repo
def camera_parameters(shape: tuple) -> tuple[float, float, float, float]:
    """Calculate the camera parameters.
    Args:
        shape: The shape of the input image
    Returns:
        Camera parameters [fx, fy, cx, cy]
    """
    height = shape[0]
    width = shape[1]

    fov = math.radians(70)  # 70 deg HFOV
    focal_length = width / (2 * math.tan(fov / 2))

    return (focal_length, focal_length, width / 2.0, height / 2.0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="First agent traverse", required=True)
    parser.add_argument("-t2", type=str, help="Second agent traverse", required=True)
    parser.add_argument("-lc", type=str, help="Loop closure csv", required=True)
    parser.add_argument("-s", help="If true, doesnt show plots", action="store_true")
    args = parser.parse_args()

    # Load LAC dataset
    lac_path = Path("data")
    agent_a = PlaybackAgent(str(lac_path / args.t1))
    print("Loaded agent A")
    agent_b = PlaybackAgent(str(lac_path / args.t2))
    print("Loaded agent B")

    output_path = Path("outputs")
    df = pd.read_csv(str(output_path / args.lc))
    headers = list(df.columns)
    loop_closures = list(df.itertuples(index=False, name=None))

    assert headers[0].removesuffix("_frame") == args.t1.removesuffix(".lac"), (
        "Column 1 name did not match the traverse 1 arg (-t1)"
    )
    assert headers[1].removesuffix("_frame") == args.t2.removesuffix(".lac"), (
        "Column 2 name did not match the traverse 2 arg (-t2)"
    )

    closures = []
    existing_closures = []
    for i, j in loop_closures:
        # HACK
        j += 2

        if i < 50 or j < 50:
            print(f"DISCARDED A[{i}] <-> B[{j}] drum was lifting")
            continue

        if i in existing_closures:
            print(f"DUPLICATE A[{i}] <-> B[{j}] loop closure rejected")
            continue

        img_i = agent_a._camera_data.get_image("FrontLeft", i)
        img_j = agent_b._camera_data.get_image("FrontLeft", j)

        fx, fy, cx, cy = camera_parameters(img_i.shape)
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        kps1, kps2, matches = match_orb_features(img_i, img_j)
        result = recover_transform(kps1, kps2, matches, K)

        if result is None:
            print(f"Loop closure A[{i}] <-> B[{j}] failed geometric verification")
            continue

        print(f"A[{i}] <-> B[{j}] SUCCESS")
        R, t = result
        t = recover_scale(agent_a, agent_b, (i, j), t)
        print()

        # Transform estimate
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        closures.append((i, j, T))

        existing_closures.append(i)

    rows = []
    for a_frame, b_frame, T in closures:
        flat_T = T.reshape(-1)
        row = [a_frame, b_frame] + flat_T.tolist()
        rows.append(row)

    columns = [
        f"{args.t1.removesuffix('.lac')}_frame",
        f"{args.t2.removesuffix('.lac')}_frame",
    ] + [f"m{i}{j}" for i in range(4) for j in range(4)]

    savepath = f"outputs/LC__{args.t1.removesuffix('.lac')}__{args.t2.removesuffix('.lac')}.csv"
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(savepath, index=False)
    print(f"Loop closures saved to: {savepath}")

    if not args.s:
        plot_loop_closures(
            agent_a._frame_data,
            agent_b._frame_data,
            closures,
            associations=PLOT_ASSOCIATIONS,
        )
        plt.show()
