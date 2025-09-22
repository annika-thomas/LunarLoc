from pathlib import Path
import matplotlib.pyplot as plt
import copy

from evo.core import sync
from evo.core import metrics
from evo.tools import plot
from evo.core.trajectory import PoseTrajectory3D

from utils.datasets import extract_orbslam, extract_gt

from lac_data import FrameDataReader


def main(traverse: FrameDataReader, silent: bool = False):
    orbslam_estimates, orbslam_frames = extract_orbslam(traverse)
    gt_traj, gt_frames = extract_gt(traverse)

    traj_ref = PoseTrajectory3D(poses_se3=gt_traj, timestamps=gt_frames.astype(float))
    traj_est = PoseTrajectory3D(
        poses_se3=orbslam_estimates, timestamps=orbslam_frames.astype(float)
    )

    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    traj_est_aligned = copy.deepcopy(traj_est)
    traj_est_aligned.align(traj_ref, correct_scale=False, correct_only_scale=False)

    data = (traj_ref, traj_est_aligned)
    pose_relation = metrics.PoseRelation.translation_part
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stats = ape_metric.get_all_statistics()

    seconds_from_start = [t - traj_est.timestamps[0] for t in traj_est.timestamps]
    fig = plt.figure()
    plot.error_array(
        fig.gca(),
        ape_metric.error,
        x_array=seconds_from_start,
        statistics={s: v for s, v in ape_stats.items() if s != "sse"},
        name="APE",
        title="APE w.r.t. " + ape_metric.pose_relation.value,
        xlabel="$t$ (s)",
    )
    savepath = f"outputs/APE_{traverse.metadata['description'].replace(' ', '_')}.png"
    plt.savefig(savepath)
    print(f"Plot created at: {savepath}")
    if not silent:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="First agent traverse", required=True)
    parser.add_argument(
        "-s", type=bool, help="If true, doesnt show plots", action="store_true"
    )
    args = parser.parse_args()

    lac_path = Path("data")
    first_traverse = FrameDataReader(str(lac_path / args.t))
    assert "orbslam" in first_traverse.custom_records.keys()

    main(first_traverse, args.s)
