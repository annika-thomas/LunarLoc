
import json, ast
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

def _parse_transform(cell: str) -> np.ndarray:
    """
    Parse a stringified 4x4 matrix (Python list format) into a (4,4) numpy array.
    """
    if isinstance(cell, (list, tuple)):
        return np.array(cell, dtype=float)
    if isinstance(cell, str):
        # The CSV column is a Python-list-looking string; ast.literal_eval is safe for this.
        mat = ast.literal_eval(cell)
        return np.array(mat, dtype=float)
    raise ValueError(f"Unsupported transform cell type: {type(cell)}")

def _parse_detections(cell: str) -> Dict[str, Any]:
    """
    Parse detections JSON stored as a string into a Python dict.
    """
    if isinstance(cell, dict):
        return cell
    if isinstance(cell, str):
        return json.loads(cell)
    raise ValueError(f"Unsupported detections cell type: {type(cell)}")

def load_data(csv_path: str) -> Dict[str, Any]:
    """
    Load the CSV and return a dict with top-level keys:
      - 'frame', 'x', 'y', 'z', 'r', 'p', 'yaw', 'transform', 'detections'
    Each key (except 'transform' and 'detections') maps to a NumPy array of shape (N,).
    'transform' maps to a list of (4,4) numpy arrays.
    'detections' maps to a list of dicts (parsed JSON), one per frame.
    """
    df = pd.read_csv(csv_path)

    # Column name assumptions (from the sample file):
    required = [
        "frame",
        "rover_x_global",
        "rover_y_global",
        "rover_z_global",
        "rover_roll",
        "rover_pitch",
        "rover_yaw",
        "rover_transform_matrix",
        "detections",
    ]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV (have {list(df.columns)})")

    # Build outputs
    out: Dict[str, Any] = {
        "frame": df["frame"].to_numpy(),
        "x": df["rover_x_global"].to_numpy(),
        "y": df["rover_y_global"].to_numpy(),
        "z": df["rover_z_global"].to_numpy(),
        "r": df["rover_roll"].to_numpy(),
        "p": df["rover_pitch"].to_numpy(),
        # NOTE: The user asked for 'y' in r,p,y (yaw). To avoid clashing with 'y' position,
        # we use 'yaw' here. If you *really* want it named 'y', you can alias it afterward.
        "yaw": df["rover_yaw"].to_numpy(),
        "transform": [ _parse_transform(cell) for cell in df["rover_transform_matrix"].tolist() ],
        "detections": [ _parse_detections(cell) for cell in df["detections"].tolist() ],
    }

    # Also provide an alias if the caller *really* wants r,p,y with 'y' meaning yaw:
    out["rpy_y_alias"] = out["yaw"]  # optional convenience

    return out

def index_of_frame(data: Dict[str, Any], frame: int) -> Optional[int]:
    """Return the index of a given frame, or None if not present."""
    frames = data["frame"]
    # Frames might not be consecutive; use exact match search.
    idx = np.where(frames == frame)[0]
    return int(idx[0]) if len(idx) else None

def get_pose(data: Dict[str, Any], frame: int) -> Dict[str, float]:
    """Return pose dict for a specific frame (x,y,z,r,p,yaw)."""
    i = index_of_frame(data, frame)
    if i is None:
        raise KeyError(f"Frame {frame} not found.")
    return {
        "x": float(data["x"][i]),
        "y": float(data["y"][i]),
        "z": float(data["z"][i]),
        "r": float(data["r"][i]),
        "p": float(data["p"][i]),
        "yaw": float(data["yaw"][i]),
    }

def get_transform(data: Dict[str, Any], frame: int) -> np.ndarray:
    """Return 4x4 transform matrix for a specific frame."""
    i = index_of_frame(data, frame)
    if i is None:
        raise KeyError(f"Frame {frame} not found.")
    return data["transform"][i]

def get_detections(data: Dict[str, Any], frame: int) -> Dict[str, Any]:
    """Return detections dict for a specific frame."""
    i = index_of_frame(data, frame)
    if i is None:
        raise KeyError(f"Frame {frame} not found.")
    return data["detections"][i]

def count_detections(det: Dict[str, Any]) -> int:
    """
    Return a simple count of detections across cameras.
    Assumes format like {'front_camera': {...}, 'back_camera': {...}, ...}
    where each camera dict holds detection_* entries.
    """
    total = 0
    for cam, cam_dict in det.items():
        if isinstance(cam_dict, dict):
            total += sum(1 for k in cam_dict.keys() if k.startswith("detection_"))
    return total

# ---------- Plot helpers (matplotlib; single plot per figure) ----------

def plot_xy_trajectory(data: Dict[str, Any], show: bool = True):
    """
    Plot the XY ground track (y vs x) for a quick 2D view.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data["x"], data["y"], marker='.', linestyle='-')
    plt.xlabel("x (global)")
    plt.ylabel("y (global)")
    plt.title("XY Trajectory")
    if show:
        plt.show()

def plot_z_over_time(data: Dict[str, Any], show: bool = True):
    """
    Plot z vs frame.
    """
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(data["frame"], data["z"], marker='.')
    plt.xlabel("frame")
    plt.ylabel("z (global)")
    plt.title("Z vs Frame")
    if show:
        plt.show()

def plot_detection_counts_over_time(data: Dict[str, Any], show: bool = True):
    """
    Plot total detection count per frame.
    """
    import matplotlib.pyplot as plt
    counts = [count_detections(d) for d in data["detections"]]
    plt.figure()
    plt.plot(data["frame"], counts, marker='.')
    plt.xlabel("frame")
    plt.ylabel("num detections")
    plt.title("Detections per Frame")
    if show:
        plt.show()
