import numpy as np
import pytransform3d.transformations as pyt_t
import pytransform3d.rotations as pyt_r
from pathlib import Path
import tqdm

from lac_data import PlaybackAgent, FrameDataReader

try:
    from maple.pose.stereoslam import SimpleStereoSLAM
except Exception:
    print("MAPLE REPO NOT DETECTED, CANNOT USE ORBSLAM")

from utils.datasets import store_trajectory


def carla_to_pytransform(transform):
    """Convert a carla transform to a pytransform."""

    # Extract translation
    translation = [transform.location.x, transform.location.y, transform.location.z]

    # For ZYX convention
    euler = [transform.rotation.yaw, transform.rotation.pitch, transform.rotation.roll]
    rotation = pyt_r.matrix_from_euler(euler, 2, 1, 0, False)

    # Create 4x4 transformation matrix
    return pyt_t.transform_from(rotation, translation)


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

            return False

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


class DummyImuAgent(PlaybackAgent):
    TRANSLATION_NOISE = 0.005  # meters
    ROTATION_NOISE = 0.005  # radians

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_pose = carla_to_pytransform(self.get_initial_position())
        self.prev_frame_num = None
        self.t_noise_model = lambda: np.random.normal(
            loc=0.0, scale=self.TRANSLATION_NOISE, size=3
        )
        self.w_noise_model = lambda: np.random.normal(
            loc=0.0, scale=self.ROTATION_NOISE, size=3
        )

    @property
    def frame(self) -> int:
        return self._frame

    def step(self, input_data: dict) -> np.ndarray | None:
        """Execute one step of navigation"""
        if self.prev_frame_num is None:
            self.prev_frame_num = self.frame
            return self.prev_pose

        prev_frame = self._frame_data[self.prev_frame_num]
        this_frame = self._frame_data[self.frame]

        pos_prev = np.array([prev_frame["x"], prev_frame["y"], prev_frame["z"]])
        pos_cur = np.array([this_frame["x"], this_frame["y"], this_frame["z"]])
        dt_world = pos_cur - pos_prev
        dt_noise = dt_world + self.t_noise_model()

        rpy_prev = np.array(
            [prev_frame["roll"], prev_frame["pitch"], prev_frame["yaw"]]
        )
        rpy_cur = np.array([this_frame["roll"], this_frame["pitch"], this_frame["yaw"]])

        R_prev = pyt_r.matrix_from_euler(rpy_prev, i=0, j=1, k=2, extrinsic=True)
        R_cur = pyt_r.matrix_from_euler(rpy_cur, i=0, j=1, k=2, extrinsic=True)

        dt_rot = R_prev.T @ dt_noise

        R_delta = R_prev.T @ R_cur
        R_noise = pyt_r.matrix_from_euler(
            self.w_noise_model(), i=0, j=1, k=2, extrinsic=True
        )
        R_delta_noisy = R_delta @ R_noise

        T_odom = pyt_t.transform_from(R=R_delta_noisy, p=dt_rot)
        self.prev_pose = self.prev_pose @ T_odom

        self.prev_frame_num = self.frame
        return self.prev_pose


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="Agent traverse", required=True)
    parser.add_argument("-s", help="If true, doesnt show plots", action="store_true")
    args = parser.parse_args()

    lac_path: Path = Path("data") / args.t
    traverse_name = args.t.removesuffix(".lac")

    # CHANGE THESE TO PICK THE ODOM SOURCE
    # agent = OrbslamAgent(str(lac_path))
    agent = DummyImuAgent(str(lac_path))

    pbar = tqdm.tqdm(
        total=len(agent._frame_data.frames["frame"]) - 1, desc="Processing frames"
    )

    # Collect orbslam vo estimates
    done = False
    tracked_frames = 0
    frame = agent.frame
    prev_estimate = None
    estimates = []
    frames = []
    lost_tracking = []
    while not done:
        input_data = agent.input_data()
        estimate = agent.step(input_data)
        if estimate is not None and estimate is not False:
            estimates.append(estimate)
            frames.append(frame)

        # Keep track of frames where we lost tracking to plot later
        lost_tracking.append(
            estimate is None
            or prev_estimate is None
            or np.array_equal(estimate, prev_estimate)
        )

        if estimate is False:
            print("Aborting orbslam VO after unrecoverable tracking loss.")
            break

        frame = agent.step_frame()
        prev_estimate = estimate
        done = agent.at_end()

        pbar.update(1)

    pbar.close()

    estimates = np.stack(estimates, axis=0)
    store_trajectory(args.t, estimates, frames, "orbslam")
    print(f"Added custom/orbslam.csv to {traverse_name}.lac")

    # Plot
    import matplotlib.pyplot as plt
    from utils.plot import plot_csv_dataset, plot_trajectory

    traverse = FrameDataReader(str(lac_path))

    ax = plot_csv_dataset(traverse)
    ax = plot_trajectory(estimates, ax=ax)  # highlights=lost_tracking, ax=ax)
    savepath = (
        f"outputs/ORBSLAM_{traverse.metadata['description'].replace(' ', '_')}.png"
    )
    plt.savefig(savepath)
    print(f"Plot created at: {savepath}")
    if not args.s:
        plt.show()
