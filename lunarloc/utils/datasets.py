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
    orbslam_df = dataset.custom_records["orbslam"]
    orbslam_frames = orbslam_df["frame"].to_numpy()
    orbslam_estimates = orbslam_df.drop("frame", axis=1).to_numpy().reshape(-1, 4, 4)

    return orbslam_estimates, orbslam_frames


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
