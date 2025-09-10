import numpy as np
import pandas as pd
import tarfile
import io
import os
from pathlib import Path
import tqdm

from lac_data import PlaybackAgent, FrameDataReader

from maple.pose.stereoslam import SimpleStereoSLAM
from maple.utils import carla_to_pytransform


def correct_pose_orientation(pose):
    # Assuming pose is a 4x4 transformation matrix
    # Extract the rotation and translation components
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    # Create a rotation correction matrix
    # To get: x-forward, y-left, z-up
    import numpy as np

    correction = np.array(
        [
            [0, 0, 1],  # New x comes from old z (forward)
            [1, 0, 0],  # New y comes from old x (left)
            [0, 1, 0],  # New z comes from old y (up)
        ]
    )

    # Apply the correction to the rotation part only
    corrected_rotation = np.dot(correction, rotation)

    # Reconstruct the transformation matrix
    corrected_pose = np.eye(4)
    corrected_pose[:3, :3] = corrected_rotation
    corrected_pose[:3, 3] = translation

    # change just rotation

    return corrected_pose


def rotate_pose_in_place(pose_matrix, roll_deg=0, pitch_deg=0, yaw_deg=0):
    """
    Apply a local RPY rotation on the rotation part of the pose, keeping translation fixed.
    """
    import numpy as np

    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    # Compose rotation in local frame
    delta_R = Rz @ Ry @ Rx

    R_old = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]

    # Apply in local frame (right multiplication)
    R_new = R_old @ delta_R

    new_pose = np.eye(4)
    new_pose[:3, :3] = R_new
    new_pose[:3, 3] = t
    return new_pose


class OrbslamAgent(PlaybackAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        here = Path(__file__).resolve()
        resources_path = here.parent.parent.parent / "resources"

        # Orbslam setup
        self.orbslam = SimpleStereoSLAM(
            vocab_path=str(resources_path / "ORBvoc.txt"),
            settings_path=str(resources_path / "orbslam_config.yaml"),
        )

        self.init_pose = carla_to_pytransform(self.get_initial_position())
        self.T_orb_to_global = None
        self.prev_estimate_orbslamframe = None

    @property
    def frame(self) -> int:
        return self._frame

    def step(self, input_data: dict) -> np.ndarray | None:
        """Execute one step of navigation"""

        sensor_data_frontleft = input_data["Grayscale"]["FrontLeft"]
        sensor_data_frontright = input_data["Grayscale"]["FrontRight"]

        # Immediately return if we dont have images for this frame
        if sensor_data_frontleft is None or sensor_data_frontright is None:
            return None

        self.orbslam.process_frame(
            left_img=sensor_data_frontleft,
            right_img=sensor_data_frontright,
            timestamp=self.frame * 0.1,
        )
        estimate_orbslamframe = self.orbslam.get_current_pose()

        # Orbslam failed
        if estimate_orbslamframe is None:
            print(f"ORBSLAM: Failed to process frame {self.frame}")

            # Havent started yet
            if self.prev_estimate_orbslamframe is None:
                return None

            # Set estimate to previous
            estimate_orbslamframe = self.prev_estimate_orbslamframe

            # Update global in anticipation for a reset to identity
            orbslam_rotated = correct_pose_orientation(estimate_orbslamframe)
            self.T_orb_to_global = self.init_pose @ np.linalg.inv(orbslam_rotated)

        # First working frame
        if self.T_orb_to_global is None:
            orbslam_rotated = correct_pose_orientation(estimate_orbslamframe)
            self.T_orb_to_global = self.init_pose @ np.linalg.inv(orbslam_rotated)

            estimate = self.init_pose

        else:
            estimate = self.T_orb_to_global @ estimate_orbslamframe

        # Store previous estimate
        self.prev_estimate_orbslamframe = estimate

        estimate = rotate_pose_in_place(estimate, 90, 270, 0)
        return estimate


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="Agent traverse", required=True)
    args = parser.parse_args()

    lac_path: Path = Path("data") / args.t
    traverse_name = args.t.removesuffix(".lac")
    agent = OrbslamAgent(str(lac_path))

    pbar = tqdm.tqdm(
        total=len(agent._frame_data.frames["frame"]) - 1, desc="Processing frames"
    )

    # Collect orbslam vo estimates
    done = False
    frame = agent.frame
    prev_estimate = None
    estimates = []
    frames = []
    lost_tracking = []
    while not done:
        input_data = agent.input_data()
        estimate = agent.step(input_data)
        if estimate is not None:
            estimates.append(estimate)
            frames.append(frame)

        # Keep track of frames where we lost tracking to plot later
        lost_tracking.append(
            estimate is None
            or prev_estimate is None
            or np.array_equal(estimate, prev_estimate)
        )

        frame = agent.step_frame()
        prev_estimate = estimate
        done = agent.at_end()

        pbar.update(1)

    pbar.close()

    estimates = np.stack(estimates, axis=0)
    N = estimates.shape[0]
    flat_estimates = estimates.reshape(N, 16)

    # Store in dataframe
    df = pd.DataFrame(
        flat_estimates, columns=[f"m{i}{j}" for i in range(4) for j in range(4)]
    )
    df.insert(0, "frame", frames)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # Build new archive with everything + orbslam.csv
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
        tarinfo = tarfile.TarInfo(name="custom/orbslam.csv")
        tarinfo.size = len(csv_bytes)
        new_tar.addfile(tarinfo, io.BytesIO(csv_bytes))

    # Replace original with the updated archive
    os.replace(tmp_path, lac_path)
    print(f"Added custom/orbslam.csv to {traverse_name}.lac")

    # Plot
    import matplotlib.pyplot as plt
    from utils.plot import plot_csv_dataset, plot_trajectory

    traverse = FrameDataReader(str(lac_path))

    ax = plot_csv_dataset(traverse)
    ax = plot_trajectory(estimates, highlights=lost_tracking, ax=ax)
    savepath = (
        f"outputs/ORBSLAM_{traverse.metadata['description'].replace(' ', '_')}.png"
    )
    plt.savefig(savepath)
    print(f"Plot created at: {savepath}")
    plt.show()
