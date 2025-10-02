import numpy as np

import gtsam
from gtsam.symbol_shorthand import X
from gtsam import symbolChr, symbolIndex

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def poses_to_xyz(values: gtsam.Values):
    data = {}  # dict: {letter: (xs, ys, zs)}
    for key in values.keys():
        if values.exists(key) and values.atPose3(key):
            letter = chr(symbolChr(key))  # e.g., 'a', 'b'
            idx = symbolIndex(key)
            pose = values.atPose3(key)
            t = pose.translation()
            xs, ys, zs = data.setdefault(letter, ([], [], []))
            xs.append(t[0])
            ys.append(t[1])
            zs.append(t[2])
    return data


def plot_initial_final(initial: gtsam.Values, result: gtsam.Values, total_frames: int):
    data_i = poses_to_xyz(initial)
    data_o = poses_to_xyz(result)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=90, azim=-90, roll=0)

    for key in data_i.keys():
        xs_i, ys_i, zs_i = data_i[key]
        xs_o, ys_o, zs_o = data_o[key]

        ax.plot(xs_i, ys_i, zs_i, "-", alpha=0.5, label=f"{key} Initial")
        ax.plot(xs_o, ys_o, zs_o, "--", label=f"{key} Optimized")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    return ax


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


def plot_loop_closures(
    dataset_a,
    dataset_b,
    loop_closures: list,
    associations: bool = True,
    ax: Axes | None = None,
):
    frames_a = [closure[0] for closure in loop_closures]
    closure_a = []
    frames_b = [closure[1] for closure in loop_closures]
    closure_b = []
    Ts = [closure[2] for closure in loop_closures]

    trajectory_a = []
    for frame_num in dataset_a.frames["frame"]:
        frame = dataset_a[frame_num]
        p = [frame["x"], frame["y"], frame["z"]]
        trajectory_a.append(p)
        if frame_num in frames_a:
            closure_a.append(p)
    trajectory_a = np.array(trajectory_a).T

    trajectory_b = []
    for frame_num in dataset_b.frames["frame"]:
        frame = dataset_b[frame_num]
        p = [frame["x"], frame["y"], frame["z"]]
        trajectory_b.append(p)
        if frame_num in frames_b:
            closure_b.append(p)
    trajectory_b = np.array(trajectory_b).T

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=90, azim=-90, roll=0)

    ax.plot(trajectory_a[0], trajectory_a[1], trajectory_a[2], "black", label="Agent A")

    ax.plot(trajectory_b[0], trajectory_b[1], trajectory_b[2], "blue", label="Agent B")

    if associations:
        for pa, pb in zip(closure_a, closure_b):
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], "r-")
            ax.scatter(pa[0], pa[1], pa[2], c="r")
            ax.scatter(pb[0], pb[1], pb[2], c="r")

    # I dont think this is correct?
    else:
        for pa, T in zip(closure_a, Ts):
            pb = T[:3, :3] @ pa + T[:3, 3]

            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], "r-")
            ax.scatter(pa[0], pa[1], pa[2], c="r")
            ax.scatter(pb[0], pb[1], pb[2], c="r")

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
