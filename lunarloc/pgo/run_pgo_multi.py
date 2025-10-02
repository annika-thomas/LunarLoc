from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytransform3d.transformations as pyt_t

import gtsam
from gtsam import symbolChr, symbolIndex

from lac_data import FrameDataReader
from pgo.run_ape import calc_ape
from utils.plot import plot_initial_final, plot_csv_dataset
from utils.datasets import extract_orbslam, tf_at_frame, extract_gt

# Odometry: ~0.5 deg, 5 cm
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.008726646, 0.008726646, 0.008726646, 0.05, 0.05, 0.05])
)

# Prior: ~0.02 deg, 1cm
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.000349066, 0.000349066, 0.000349066, 0.01, 0.01, 0.01])
)

# Loop closures: 3 deg, 50 cm
LC_NOISE = gtsam.noiseModel.Diagonal.Sigmas(
    np.array([0.052359877, 0.052359877, 0.052359877, 0.5, 0.5, 0.5])
)

# Robust loop closure noise
huber_k = 1.345
LC_NOISE_ROBUST = gtsam.noiseModel.Robust.Create(
    gtsam.noiseModel.mEstimator.Huber(huber_k), LC_NOISE
)


def A(i):
    return gtsam.symbol("a", int(i))


def B(j):
    return gtsam.symbol("b", int(j))


def main(
    agent_a: FrameDataReader, agent_b: FrameDataReader, closures, silent: bool = False
):
    # Read orbslam from the traverse
    agent_a_orbslam, agent_a_orbslam_frames = extract_orbslam(agent_a)
    agent_b_orbslam, agent_b_orbslam_frames = extract_orbslam(agent_b)

    # Factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    ###########
    # AGENT A
    ###########

    # Between factors
    total_frames = len(agent_a_orbslam_frames)
    for i in range(total_frames - 1):
        world_T_current = agent_a_orbslam[i]
        world_T_next = agent_a_orbslam[i + 1]
        current_T_next = pyt_t.invert_transform(world_T_current) @ world_T_next

        graph.add(
            gtsam.BetweenFactorPose3(
                A(agent_a_orbslam_frames[i]),
                A(agent_a_orbslam_frames[i + 1]),
                gtsam.Pose3(current_T_next),
                ODOMETRY_NOISE,
            )
        )

    # Initial estimate is raw orbslam estimate
    for i, estimate in enumerate(agent_a_orbslam):
        initial.insert(A(agent_a_orbslam_frames[i]), gtsam.Pose3(estimate))

    # Set prior to GT first pose so they align when plotting
    graph.add(
        gtsam.PriorFactorPose3(
            A(agent_a_orbslam_frames[0]),
            gtsam.Pose3(tf_at_frame(agent_a[agent_a_orbslam_frames[0]])),
            PRIOR_NOISE,
        )
    )
    # graph.add(
    #     gtsam.PriorFactorPose3(
    #         A(agent_a_orbslam_frames[-1]),
    #         gtsam.Pose3(tf_at_frame(agent_a[agent_a_orbslam_frames[-1]])),
    #         PRIOR_NOISE,
    #     )
    # )

    ###########
    # AGENT B
    ###########

    # Between factors
    total_frames = len(agent_b_orbslam_frames)
    for i in range(total_frames - 1):
        world_T_current = agent_b_orbslam[i]
        world_T_next = agent_b_orbslam[i + 1]
        current_T_next = pyt_t.invert_transform(world_T_current) @ world_T_next

        graph.add(
            gtsam.BetweenFactorPose3(
                B(agent_b_orbslam_frames[i]),
                B(agent_b_orbslam_frames[i + 1]),
                gtsam.Pose3(current_T_next),
                ODOMETRY_NOISE,
            )
        )

    # Initial estimate is raw orbslam estimate
    for i, estimate in enumerate(agent_b_orbslam):
        initial.insert(B(agent_b_orbslam_frames[i]), gtsam.Pose3(estimate))

    # Set prior to GT first pose so they align when plotting
    graph.add(
        gtsam.PriorFactorPose3(
            B(agent_b_orbslam_frames[0]),
            gtsam.Pose3(tf_at_frame(agent_b[agent_b_orbslam_frames[0]])),
            PRIOR_NOISE,
        )
    )
    # graph.add(
    #     gtsam.PriorFactorPose3(
    #         B(agent_b_orbslam_frames[-1]),
    #         gtsam.Pose3(tf_at_frame(agent_b[agent_b_orbslam_frames[-1]])),
    #         PRIOR_NOISE,
    #     )
    # )

    ################
    # LOOP CLOSURES
    ################

    # One loop closure at the very end
    # world_T_i = tf_at_frame(agent_a[agent_a_orbslam_frames[-1]])
    # world_T_j = tf_at_frame(agent_b[agent_b_orbslam_frames[-1]])
    # i_T_j = pyt_t.invert_transform(world_T_i) @ world_T_j
    # graph.add(
    #     gtsam.BetweenFactorPose3(
    #         A(agent_a_orbslam_frames[-1]),
    #         B(agent_b_orbslam_frames[-1]),
    #         gtsam.Pose3(i_T_j),
    #         LC_NOISE_ROBUST,
    #     )
    # )

    # Add loop closures
    for i, j, i_T_j in closures:
        # world_T_i = tf_at_frame(agent_a[i])
        # world_T_j = tf_at_frame(agent_a[j])
        # i_T_j = pyt_t.invert_transform(world_T_i) @ world_T_j

        graph.add(gtsam.BetweenFactorPose3(A(i), B(j), gtsam.Pose3(i_T_j), LC_NOISE))

    # Add dummy loop closures
    # total_frames = len(agent_a_orbslam_frames)
    # for i in range(0, total_frames, 100):
    #     for j in range(i + 100, total_frames, 100):
    #         world_T_i = tf_at_frame(agent_a[agent_a_orbslam_frames[i]])
    #         world_T_j = tf_at_frame(agent_a[agent_a_orbslam_frames[j]])
    #         i_T_j = pyt_t.invert_transform(world_T_i) @ world_T_j

    #         graph.add(
    #             gtsam.BetweenFactorPose3(
    #                 A(agent_a_orbslam_frames[i]),
    #                 A(agent_a_orbslam_frames[j]),
    #                 gtsam.Pose3(i_T_j),
    #                 LC_NOISE,
    #             )
    #         )

    # total_frames = len(agent_b_orbslam_frames)
    # for i in range(0, total_frames, 100):
    #     for j in range(i + 100, total_frames, 100):
    #         world_T_i = tf_at_frame(agent_b[agent_b_orbslam_frames[i]])
    #         world_T_j = tf_at_frame(agent_b[agent_b_orbslam_frames[j]])
    #         i_T_j = pyt_t.invert_transform(world_T_i) @ world_T_j

    #         graph.add(
    #             gtsam.BetweenFactorPose3(
    #                 B(agent_b_orbslam_frames[i]),
    #                 B(agent_b_orbslam_frames[j]),
    #                 gtsam.Pose3(i_T_j),
    #                 LC_NOISE,
    #             )
    #         )

    #############
    # OPTIMIZE
    #############

    # Optimize
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)

    print("error(initial):", graph.error(initial))
    result = optimizer.optimize()
    print("error(result):", graph.error(result))

    ax = plot_initial_final(initial, result, total_frames)
    ax = plot_csv_dataset(agent_a, ax=ax)
    ax = plot_csv_dataset(agent_b, ax=ax)

    traj_opt = gtvalues_to_trajectory(result)

    gt_traj, gt_frames = extract_gt(agent_a)
    ape_metric_a, _ = calc_ape(
        gt_traj, gt_frames, traj_opt["a"], agent_a_orbslam_frames
    )
    print("Agent A:")
    print(ape_metric_a.get_all_statistics())

    gt_traj, gt_frames = extract_gt(agent_b)
    ape_metric_b, _ = calc_ape(
        gt_traj, gt_frames, traj_opt["b"], agent_b_orbslam_frames
    )
    print("Agent B:")
    print(ape_metric_b.get_all_statistics())

    # savepath = (
    #     f"outputs/PGO_{first_traverse.metadata['description'].replace(' ', '_')}.png"
    # )
    # plt.savefig(savepath)
    # print(f"Plot created at: {savepath}")
    if not silent:
        plt.show()


def gtvalues_to_trajectory(values):
    data = defaultdict(list)
    for key in values.keys():
        if values.exists(key) and values.atPose3(key):
            letter = chr(symbolChr(key))
            pose = values.atPose3(key)
            T = pose.matrix()
            data[letter].append(T)
    out = {}
    for key in data.keys():
        out[key] = np.array(data[key])
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="First agent traverse", required=True)
    parser.add_argument("-t2", type=str, help="First agent traverse", required=True)
    parser.add_argument("-lc", type=str, help="Loop closure csv", required=True)
    parser.add_argument("-s", help="If true, doesnt show plots", action="store_true")
    args = parser.parse_args()

    assert args.lc.startswith("LC"), (
        "The loop closure csv must be geometrically verified first"
    )

    output_path = Path("outputs")
    df = pd.read_csv(str(output_path / args.lc))
    headers = list(df.columns)
    assert headers[0].removesuffix("_frame") == args.t1.removesuffix(".lac"), (
        "Column 1 name did not match the traverse 1 arg (-t1)"
    )
    assert headers[1].removesuffix("_frame") == args.t2.removesuffix(".lac"), (
        "Column 2 name did not match the traverse 2 arg (-t2)"
    )

    closures = []
    for _, row in df.iterrows():
        i = int(row[f"{args.t1.removesuffix('.lac')}_frame"])
        j = int(row[f"{args.t2.removesuffix('.lac')}_frame"])
        T = (
            row[[f"m{i}{j}" for i in range(4) for j in range(4)]]
            .to_numpy()
            .reshape(4, 4)
        )
        closures.append((i, j, T))

    lac_path = Path("data")
    first_traverse = FrameDataReader(str(lac_path / args.t1))
    second_traverse = FrameDataReader(str(lac_path / args.t2))
    assert "orbslam" in first_traverse.custom_records.keys()
    assert "orbslam" in second_traverse.custom_records.keys()

    main(first_traverse, second_traverse, closures, args.s)
