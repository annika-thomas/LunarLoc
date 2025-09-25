
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytransform3d.transformations as pyt_t

import gtsam
from gtsam.symbol_shorthand import X

from lac_data import FrameDataReader
# from utils.plot import plot_initial_final
from utils.datasets import tf_at_frame

# Noise is rotation-first, then translation
# ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
#     np.array([0.001, 0.001, 0.001, 0.005, 0.005, 0.005])
# )  # ~0.5 deg, 5cm
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.001, 0.001, 0.001, 0.005, 0.005, 0.005], dtype=np.float64)
)

# PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
#     np.array([0.0005, 0.0005, 0.0005, 0.001, 0.001, 0.001])
# )  # ~0.02 deg, 1cm
# LC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
#     np.array([0.000005, 0.000005, 0.000005, 0.00005, 0.00005, 0.00005])
# )  # ~3 deg, 50cm


def main(first_traverse: FrameDataReader, silent: bool = False):
    print("hmmm")
    # Read orbslam from the traverse
    orbslam_df = first_traverse.custom_records["orbslam"]
    orbslam_frames = orbslam_df["frame"].to_numpy()
    orbslam_estimates = orbslam_df.drop("frame", axis=1).to_numpy().reshape(-1, 4, 4)

    # Factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # # Between factors
    # total_frames = len(orbslam_frames)
    # for i in range(total_frames - 1):
    #     T_w_cur = orbslam_estimates[i]
    #     T_w_next = orbslam_estimates[i + 1]
    #     T_cur_next = pyt_t.invert_transform(T_w_cur) @ T_w_next

    #     graph.add(
    #         gtsam.BetweenFactorPose3(
    #             X(i), X(i + 1), gtsam.Pose3(T_cur_next), ODOMETRY_NOISE
    #         )
    #     )

    # # Initial estimate is raw orbslam estimate
    # for i, estimate in enumerate(orbslam_estimates):
    #     initial.insert(X(i), gtsam.Pose3(estimate))

    # # Set prior to GT first pose so they align when plotting
    # graph.add(
    #     gtsam.PriorFactorPose3(
    #         X(0),
    #         gtsam.Pose3(tf_at_frame(first_traverse[orbslam_frames[0]])),
    #         PRIOR_NOISE,
    #     )
    # )

    # # Add dummy loop closures
    # for i in range(1, total_frames - 1, 100):
    #     for j in range(1, total_frames - 1, 100):
    #         if i != j:
    #             T_w_i = tf_at_frame(first_traverse[orbslam_frames[i]])
    #             T_w_j = tf_at_frame(first_traverse[orbslam_frames[j]])
    #             T_i_j = pyt_t.invert_transform(T_w_i) @ T_w_j

    #             graph.add(
    #                 gtsam.BetweenFactorPose3(
    #                     X(i - 1), X(j - 1), gtsam.Pose3(T_i_j), LC_NOISE
    #                 )
    #             )

    # # Optimize
    # params = gtsam.LevenbergMarquardtParams()
    # optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    # result = optimizer.optimize()

    # print("Optimization complete!")
    # print("final graph error = ", graph.error(result))

    # plot_initial_final(first_traverse, initial, result, total_frames)

    # savepath = (
    #     f"outputs/PGO_{first_traverse.metadata['description'].replace(' ', '_')}.png"
    # )
    # plt.savefig(savepath)
    # print(f"Plot created at: {savepath}")
    # if not silent:
    #     plt.show()


if __name__ == "__main__":
    import argparse

    # parser = argparse.ArgumentParser()
    # # parser.add_argument("-t", type=str, help="First agent traverse", required=True)
    # parser.add_argument("-s", help="If true, doesnt show plots", action="store_true")
    # args = parser.parse_args()

    # lac_path = Path("data")
    # first_traverse = FrameDataReader(str(lac_path / args.t))
    first_traverse = FrameDataReader("/home/annika/Downloads/data/orbslam_straight_line_preset1_default_20250917_102024.lac")

    # assert "orbslam" in first_traverse.custom_records.keys()

    main(first_traverse)
