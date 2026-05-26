"""
Lens distortion correction using OpenCV's fisheye model.

This camera uses a wide-angle fisheye lens. Parameters were determined
manually via tune_distortion.py by verifying that the field boundary line
becomes straight after undistortion.

Saved params (data/distortion_params.json):
    model : "fisheye"
    fx    : fisheye focal length in pixels (~1200 for this camera)
    cx/cy : principal point (image centre)
    k1-k4 : fisheye distortion coefficients (k1=-0.05 corrects this lens)
"""

import json
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def build_camera_matrix(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (K, D) from a params dict.

    K : 3×3 camera intrinsics matrix
    D : (4,) fisheye distortion coefficients [k1, k2, k3, k4]
    """
    K = np.array([
        [params["fx"], 0,            params["cx"]],
        [0,            params["fx"], params["cy"]],
        [0,            0,            1           ],
    ], dtype=np.float64)
    D = np.array(
        [params.get("k1", 0.0), params.get("k2", 0.0),
         params.get("k3", 0.0), params.get("k4", 0.0)],
        dtype=np.float64,
    )
    return K, D


def _output_camera_matrix(params: dict, out_fx: float | None = None) -> np.ndarray:
    """
    Build the output (rectilinear) camera matrix for undistortion.

    out_fx controls the output zoom level.  Checked in order:
      1. explicit out_fx argument
      2. params["out_fx"] if present
      3. params["fx"] (no zoom change at centre)
    Smaller out_fx = wider field of view.
    """
    fx = out_fx if out_fx is not None else params.get("out_fx", params["fx"])
    return np.array([
        [fx, 0,  params["cx"]],
        [0,  fx, params["cy"]],
        [0,  0,  1           ],
    ], dtype=np.float64)


def undistort_image(
    image: np.ndarray,
    params: dict,
    out_fx: float | None = None,
) -> np.ndarray:
    """
    Undistort an image using the fisheye model.

    out_fx : focal length of the output rectilinear camera.
             Smaller = wider field of view (more black border but keeps edges).
             Defaults to params["fx"] (no zoom change at centre).
    """
    K, D = build_camera_matrix(params)
    Knew = _output_camera_matrix(params, out_fx)
    return cv2.fisheye.undistortImage(image, K, D, Knew=Knew)


def undistort_points(
    pts: np.ndarray,
    params: dict,
    out_fx: float | None = None,
) -> np.ndarray:
    """
    Undistort an array of image points using the fisheye model.

    pts  : (N, 2) float array of [x, y] pixel coordinates (distorted).
    Returns (N, 2) undistorted pixel coordinates.
    """
    K, D = build_camera_matrix(params)
    Knew = _output_camera_matrix(params, out_fx)
    pts_f32 = pts.astype(np.float32).reshape(-1, 1, 2)
    undist = cv2.fisheye.undistortPoints(pts_f32, K, D, P=Knew)
    return undist.reshape(-1, 2)


def save_params(params: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Distortion params saved to {path}")


def load_params(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
