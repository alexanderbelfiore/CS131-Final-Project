"""
Manual fisheye distortion parameter tuning via visual sweep.

Generates a grid of undistorted images across a range of (k1, fx) values
so you can pick the best one visually, then saves the chosen parameters.

Usage:
    python src/tune_distortion.py <reference_image>

Then inspect data/distortion_sweep/ and run again with --k1 and --fx
to fine-tune and save.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lens_distortion import build_camera_matrix, _output_camera_matrix, undistort_image


def undistort_with_params(image: np.ndarray, k1: float, fx: float,
                          cx: float | None = None, cy: float | None = None,
                          out_fx: float | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0
    params = {"fx": fx, "cx": cx, "cy": cy, "k1": k1, "k2": 0.0, "k3": 0.0, "k4": 0.0}
    return undistort_image(image, params, out_fx=out_fx)


def run_sweep(image_path: str, output_dir: str = "data/distortion_sweep") -> None:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    h, w = img.shape[:2]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    k1_values = [-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1]
    fx_values = [700, 900, 1100, 1300, 1500]

    print(f"Generating sweep: {len(k1_values)} k1 × {len(fx_values)} fx values")
    print(f"Output: {out}/")

    for fx in fx_values:
        row_imgs = []
        for k1 in k1_values:
            undist = undistort_with_params(img, k1=k1, fx=fx, out_fx=600)
            thumb_h = 400
            scale = thumb_h / undist.shape[0]
            thumb_w = int(undist.shape[1] * scale)
            thumb = cv2.resize(undist, (thumb_w, thumb_h))
            label = f"fx={fx} k1={k1:.2f}"
            cv2.putText(thumb, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            row_imgs.append(thumb)

        row = np.hstack(row_imgs)
        fname = out / f"sweep_fx{fx}.jpg"
        cv2.imwrite(str(fname), row)
        print(f"  Saved {fname.name}")

    print("\nInspect the sweep images and pick (k1, fx) that makes field lines straightest.")
    print("Then run:  python src/tune_distortion.py <image> --k1 <val> --fx <val> --save")


def save_params(image_path: str, k1: float, fx: float,
                target_w: int, target_h: int,
                out_fx_ref: float = 600.0,
                out_path: str = "data/distortion_params.json") -> None:
    """
    Save distortion params scaled to the target video resolution.

    fx and out_fx are chosen visually on the reference image (image_path), but
    since focal length is in pixels it must scale with image width. This function
    converts both values from the reference image's pixel space to the target
    video frame's pixel space so they are ready for use on video frames.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")

    ref_h, ref_w = img.shape[:2]
    scale = target_w / ref_w
    fx_scaled = round(fx * scale, 4)
    out_fx_scaled = round(out_fx_ref * scale, 4)

    params = {
        "model": "fisheye",
        "k1": k1, "k2": 0.0, "k3": 0.0, "k4": 0.0,
        "fx": fx_scaled,
        "cx": target_w / 2.0, "cy": target_h / 2.0,
        "out_fx": out_fx_scaled,
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Reference image : {ref_w}x{ref_h}, fx={fx}, out_fx={out_fx_ref}")
    print(f"Target video    : {target_w}x{target_h}, scale={scale:.4f}")
    print(f"Scaled params   : fx={fx_scaled}, out_fx={out_fx_scaled}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual fisheye distortion tuning.")
    parser.add_argument("image", help="Reference image path")
    parser.add_argument("--k1", type=float, default=None, help="k1 to use (run sweep first if omitted)")
    parser.add_argument("--fx", type=int, default=None, help="fisheye focal length in pixels (in reference image space)")
    parser.add_argument("--target-w", type=int, required=False, default=1920, help="target video width in pixels")
    parser.add_argument("--target-h", type=int, required=False, default=1080, help="target video height in pixels")
    parser.add_argument("--out-fx", type=float, default=600.0, help="output focal length used during sweep (default 600)")
    parser.add_argument("--output", default="data/distortion_params.json", help="params output path")
    args = parser.parse_args()

    if args.k1 is not None and args.fx is not None:
        save_params(args.image, k1=args.k1, fx=args.fx,
                    target_w=args.target_w, target_h=args.target_h,
                    out_fx_ref=args.out_fx, out_path=args.output)
    else:
        run_sweep(args.image)
