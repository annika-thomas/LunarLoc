from pathlib import Path
import matplotlib.pyplot as plt

from lac_data import FrameDataReader
from utils.plot import plot_csv_dataset

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="lac name", required=True)
    args = parser.parse_args()

    lac_path = Path("data")
    first_traverse = FrameDataReader(str(lac_path / args.t))

    plot_csv_dataset(first_traverse)
    savepath = f"outputs/TRAVERSE_{first_traverse.metadata['description'].replace(' ', '_')}.png"
    plt.savefig(savepath)
    print(f"Saved plot to: {savepath}")
    plt.show()
