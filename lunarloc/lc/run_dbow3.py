from pyslam.config import Config

config = Config()

from pyslam.utilities.utils_sys import Printer

import cv2
import numpy as np
import csv

from pyslam.slam.frame import FeatureTrackerShared
from pyslam.local_features.feature_tracker import feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from pyslam.config_parameters import Parameters

Parameters.kLoopClosingDebugAndPrintToFile = False
Parameters.kLoopClosingDebugWithSimmetryMatrix = True
Parameters.kLoopClosingDebugWithLoopDetectionImages = True


### COMMENT FOR DEBUG INFO
Parameters.kVerbose = False

from pyslam.loop_closing.loop_detector_configs import (
    LoopDetectorConfigs,
    loop_detector_factory,
    SlamFeatureManagerInfo,
)
from pyslam.loop_closing.loop_detector_base import (
    LoopDetectorTask,
    LoopDetectorTaskType,
    LoopDetectKeyframeData,
)

from lac_data import PlaybackAgent
from pathlib import Path
import tqdm

# online loop closure detection by using DBoW3
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t1", type=str, help="First agent traverse", required=True)
    parser.add_argument("-t2", type=str, help="Second agent traverse", required=True)
    args = parser.parse_args()

    # Load LAC dataset
    lac_path = Path("data")
    agent_a = PlaybackAgent(str(lac_path / args.t1))
    agent_b = PlaybackAgent(str(lac_path / args.t2))

    pbar = tqdm.tqdm(
        total=len(agent_a._frame_data.frames["frame"]) - 1, desc="Agent A frames"
    )

    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config["num_features"] = 2000
    print("tracker_config: ", tracker_config)
    feature_tracker = feature_tracker_factory(**tracker_config)

    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared.
    FeatureTrackerShared.set_feature_tracker(feature_tracker)

    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing.
    # LoopDetectorConfigs: DBOW2, DBOW3, etc.
    loop_detection_config = LoopDetectorConfigs.DBOW3
    Printer.green("loop_detection_config: ", loop_detection_config)
    loop_detector = loop_detector_factory(
        **loop_detection_config,
        slam_info=SlamFeatureManagerInfo(
            feature_manager=feature_tracker.feature_manager
        ),
    )

    if not Parameters.kVerbose:
        loop_detector.print = staticmethod(lambda *args, **kwargs: None)

    # cv2.namedWindow("similarity matrix", cv2.WINDOW_NORMAL)  # to get a resizable window
    # cv2.namedWindow(
    #     "loop detection candidates", cv2.WINDOW_NORMAL
    # )  # to get a resizable window

    #########################
    # SEED THE DETECTOR WITH AGENT A
    ##############################

    img_id = agent_a._frame
    done = False
    while not done:
        img = None

        input_data = agent_a.input_data()
        sensor_data_frontleft = input_data["Grayscale"]["FrontLeft"]
        if sensor_data_frontleft is not None:
            img = sensor_data_frontleft[:, :, np.newaxis].repeat(3, axis=2)

        if img is not None:
            if Parameters.kVerbose:
                print("----------------------------------------")
                print(f"processing img {img_id} (Agent A)")

            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(
                img
            )  # with DL matchers this a null operation
            kps_ = [
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave)
                for kp in kps
            ]  # tuple_x_y_size_angle_response_octave

            task_type = LoopDetectorTaskType.LOOP_CLOSURE
            covisible_keyframes = []
            connected_keyframes = []
            keyframe = LoopDetectKeyframeData()
            keyframe.id = img_id
            keyframe.img = img
            keyframe.kps = kps_
            keyframe.des = des

            task = LoopDetectorTask(
                keyframe,
                img,
                task_type,
                covisible_keyframes=covisible_keyframes,
                connected_keyframes=connected_keyframes,
            )

            # check and compute if needed the local descriptors by using the independent local feature manager (if present).
            loop_detector.compute_local_des_if_needed(task)
            # run the loop detection task
            detection_output = loop_detector.run_task(task)

        img_id = agent_a.step_frame()
        done = agent_a.at_end()

        pbar.update(1)

    pbar.close()
    agent_a_length = img_id

    ##################################################
    # DETECT CLOSURES ON AGENT B AND IGNORE SELF-CLOSURES
    ################################################

    pbar = tqdm.tqdm(
        total=len(agent_b._frame_data.frames["frame"]) - 1, desc="Agent B frames"
    )

    img_id = agent_a_length
    prev_agent_frame = agent_b._frame
    done = False

    RESULTS = []

    while not done:
        img = None

        input_data = agent_b.input_data()
        sensor_data_frontleft: np.ndarray = input_data["Grayscale"]["FrontLeft"]
        if sensor_data_frontleft is not None:
            img = sensor_data_frontleft[:, :, np.newaxis].repeat(3, axis=2)

        if img is not None:
            if Parameters.kVerbose:
                print("----------------------------------------")
                print(f"processing img {img_id - agent_a_length} (Agent B)")

            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(
                img
            )  # with DL matchers this a null operation
            kps_ = [
                (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave)
                for kp in kps
            ]  # tuple_x_y_size_angle_response_octave

            task_type = LoopDetectorTaskType.LOOP_CLOSURE
            covisible_keyframes = []
            connected_keyframes = []
            keyframe = LoopDetectKeyframeData()
            keyframe.id = img_id
            keyframe.img = img
            keyframe.kps = kps_
            keyframe.des = des

            task = LoopDetectorTask(
                keyframe,
                img,
                task_type,
                covisible_keyframes=covisible_keyframes,
                connected_keyframes=connected_keyframes,
            )

            # check and compute if needed the local descriptors by using the independent local feature manager (if present).
            loop_detector.compute_local_des_if_needed(task)
            # run the loop detection task
            detection_output = loop_detector.run_task(task)

            if len(detection_output.candidate_idxs) > 0:
                # if detection_output.loop_detection_img_candidates is not None:
                #     # cv2.imshow(
                #     #     "loop detection candidates",
                #     #     detection_output.loop_detection_img_candidates,
                #     # )

                idxs_local = detection_output.candidate_idxs
                scores_local = detection_output.candidate_scores

                agent_b_frame_num = img_id - agent_a_length

                while len(idxs_local) != 0:
                    cur_score = scores_local.pop()
                    cur_idx = idxs_local.pop()

                    # if cur_idx is greater, this is a loop closure within agent b
                    if cur_idx < agent_a_length:
                        RESULTS.append((cur_idx, agent_b_frame_num))

            # cv2.waitKey(1)

        next_agent_frame = agent_b.step_frame()
        img_id += next_agent_frame - prev_agent_frame
        prev_agent_frame = next_agent_frame
        done = agent_b.at_end()

        pbar.update(1)

    pbar.close()

    print("RESULTS:")
    for entry in RESULTS:
        print(f"A[{int(entry[0])}] <-> B[{int(entry[1])}]")

    with open(
        f"outputs/{args.t1.removesuffix('.lac')}__{args.t2.removesuffix('.lac')}.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                f"{args.t1.removesuffix('.lac')}_frame",
                f"{args.t2.removesuffix('.lac')}_frame",
            ]
        )  # header
        writer.writerows(RESULTS)
