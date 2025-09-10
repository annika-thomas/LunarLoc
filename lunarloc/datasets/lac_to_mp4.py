import cv2
import re
import tqdm
from pathlib import Path

from lac_data import PlaybackAgent


def extract_number(filename):
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1


def create_video_from_images(traverse: PlaybackAgent, output_video_path, fps=10):
    # get dimensions
    first_image = None
    while first_image is None:
        first_image = traverse.input_data().get("Grayscale", None)
        traverse.step_frame()

    first_image = first_image["FrontLeft"]
    height, width = first_image.shape
    size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size, isColor=False)

    pbar = tqdm.tqdm(
        total=len(traverse._frame_data.frames["frame"]) - 1, desc="Processing frames"
    )

    done = False
    while not done:
        img = traverse.input_data().get("Grayscale", None)
        if img is not None:
            img = img["FrontLeft"]
            img_resized = cv2.resize(img, size)
            out.write(img_resized)

        traverse.step_frame()
        done = traverse.at_end()

        pbar.update(1)

    pbar.close()

    out.release()
    print(f"Video created at: {output_video_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=str, help="lac name", required=True)
    args = parser.parse_args()

    lac_path = Path("data")
    traverse = PlaybackAgent(str(lac_path / args.t))

    output_file = f"outputs/VIDEO_{args.t.removesuffix('.lac')}.mp4"
    frame_rate = 15

    create_video_from_images(traverse, output_file, frame_rate)
