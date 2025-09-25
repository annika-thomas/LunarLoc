import os
import io
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np
import pytransform3d.rotations as pyt_r
import pytransform3d.transformations as pyt_t


def tf_at_frame(frame: dict) -> np.ndarray:
    try:
        return pyt_t.transform_from(
            R=pyt_r.matrix_from_euler(
                [frame["roll"], frame["pitch"], frame["yaw"]],
                i=0,
                j=1,
                k=2,
                extrinsic=True,
            ),
            p=[frame["x"], frame["y"], frame["z"]],
        )
    except KeyError:
        return pyt_t.transform_from(
            R=np.eye(3),
            p=[frame["x"], frame["y"], frame["z"]],
        )


def extract_orbslam(dataset):
    """Returns orbslam_estimates (Nx4x4) and orbslam_frames (N)"""

    # Read orbslam from the traverse
    return extract_trajectory(dataset, "orbslam")


def extract_gt(dataset):
    """Returns groundtruth_trajectory (Nx4x4) and groundtruth_frames (N)"""

    trajectory = []
    frames = []
    for frame_num in dataset.frames["frame"]:
        frame = dataset[frame_num]
        frames.append(frame_num)
        trajectory.append(tf_at_frame(frame))

    trajectory = np.array(trajectory)
    frames = np.array(frames)

    return trajectory, frames


def extract_trajectory(dataset, trajectory_name: str):
    """Extracts the trajectory and frames stored at custom/trajectory_name"""

    custom_df = dataset.custom_records[trajectory_name]
    frames = custom_df["frame"].to_numpy()
    estimates = custom_df.drop("frame", axis=1).to_numpy().reshape(-1, 4, 4)

    return estimates, frames


def store_trajectory(
    dataset_name: str,
    trajectory: np.ndarray,
    trajectory_frames: list,
    trajectory_name: str,
):
    """
    Stores a new trajectory into an existing .lac dataset.
    Example:
        store_trajectory("orbslam_circle_preset1_20250903_232138.lac", orbslam_estimate, orbslam_frames, "orbslam")
        This will store the Nx4x4 trajectory, orbslam_estimate, into the indicated dataset, at the custom field "orbslam"
    """
    lac_path: Path = Path("data") / dataset_name

    N = trajectory.shape[0]
    flat_trajectory = trajectory.reshape(N, 16)

    # Store in dataframe
    df = pd.DataFrame(
        flat_trajectory, columns=[f"m{i}{j}" for i in range(4) for j in range(4)]
    )
    df.insert(0, "frame", trajectory_frames)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Build new archive with everything + new trajectory csv
    tmp_path = lac_path.with_suffix(".tmp")
    with (
        tarfile.open(lac_path, "r:gz") as old_tar,
        tarfile.open(tmp_path, "w:gz") as new_tar,
    ):
        # Copy all existing members
        for member in old_tar.getmembers():
            f = old_tar.extractfile(member)
            if f is not None:
                new_tar.addfile(member, f)
            else:
                new_tar.addfile(member)

        # Add the orbslam data
        tarinfo = tarfile.TarInfo(name=f"custom/{trajectory_name}.csv")
        tarinfo.size = len(csv_bytes)
        new_tar.addfile(tarinfo, io.BytesIO(csv_bytes))

    # Replace original with the updated archive
    os.replace(tmp_path, lac_path)
