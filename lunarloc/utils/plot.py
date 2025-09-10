import numpy as np

import gtsam
from gtsam.symbol_shorthand import X

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def poses_to_xyz(values: gtsam.Values, num_poses: int):
    xs, ys, zs = [], [], []
    for i in range(num_poses):
        pose = values.atPose3(X(i))
        t = pose.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])
    return xs, ys, zs


def plot_initial_final(
    dataset, initial: gtsam.Values, result: gtsam.Values, total_frames: int
):
    xs_i, ys_i, zs_i = poses_to_xyz(initial, total_frames)
    xs_o, ys_o, zs_o = poses_to_xyz(result, total_frames)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=90, azim=-90, roll=0)
    ax.plot(xs_i, ys_i, zs_i, "r-", alpha=0.5, label="Initial")
    ax.plot(xs_o, ys_o, zs_o, "b--", label="Optimized")
    plot_csv_dataset(dataset, ax=ax)
    ax.legend()
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_csv_dataset(dataset, ax: Axes | None = None):
    trajectory = []
    for frame_num in dataset.frames["frame"]:
        frame = dataset[frame_num]
        trajectory.append([frame["x"], frame["y"], frame["z"]])
    trajectory = np.array(trajectory).T

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=90, azim=-90, roll=0)
    ax.plot(trajectory[0], trajectory[1], trajectory[2], "black", label="Traverse")
    return ax


def plot_trajectory(
    trajectory: np.ndarray, highlights: list[bool] | None = None, ax: Axes | None = None
):
    # Extract the translation components (x, y, z)
    positions = trajectory[:, :3, 3]  # shape: (N, 3)

    x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]

    # Plot the trajectory in 3D
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=90, azim=-90, roll=0)
    ax.plot(x, y, z, label="Trajectory")

    if highlights is not None:
        mask = np.array(highlights, dtype=bool)
        ax.scatter(x[mask], y[mask], z[mask], color="red", label="Highlights")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    return ax
