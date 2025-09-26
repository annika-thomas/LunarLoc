from pathlib import Path
import matplotlib.pyplot as plt
import copy
import matplotlib as mpl
mpl.use("Agg")  # set backend before importing pyplot/evo
# from evo.tools import plot

from evo.core import sync
from evo.core import metrics
from evo.tools import plot
from evo.core.trajectory import PoseTrajectory3D

from utils.datasets import extract_orbslam, extract_gt

from lac_data import FrameDataReader


def main(traverse: FrameDataReader, silent: bool = False):
    orbslam_estimates, orbslam_frames = extract_orbslam(traverse)
    gt_traj, gt_frames = extract_gt(traverse)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", type=str, help="First agent traverse", required=True)
    parser.add_argument("-s", help="If true, doesnt show plots", action="store_true")
    args = parser.parse_args()

    lac_path = Path("data")
    # first_traverse = FrameDataReader(str(lac_path / args.t))
    first_traverse = FrameDataReader("/home/annika/Downloads/data/orbslam_straight_line_preset1_default_20250917_102024.lac")
    assert "orbslam" in first_traverse.custom_records.keys()

    main(first_traverse, args.s)