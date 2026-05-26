"""
Video frame extraction with lens distortion correction.

Extracts frames from a video at a configurable step rate and optionally
applies lens undistortion using pre-computed distortion parameters.
"""

from pathlib import Path

import cv2
import numpy as np

from lens_distortion import load_params, undistort_image


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    step: int = 5,
    distortion_params: dict | None = None,
    max_frames: int | None = None,
    resize: float | None = None,
) -> list[dict]:
    """
    Extract frames from a video file.

    Args:
        video_path: Path to the input video.
        output_dir: Directory to write extracted frames.
        step: Extract every Nth frame (1 = every frame, 5 = every 5th, etc.).
        distortion_params: If provided, undistort each frame before saving.
        max_frames: Stop after extracting this many frames (useful for prototyping).
        resize: If provided, scale frames by this factor before saving (e.g. 0.5).

    Returns:
        List of dicts with keys: frame_index, video_frame_number, timestamp_s, path.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    print(f"Video: {video_path.name}")
    print(f"  FPS: {fps:.2f}, frames: {total_frames}, duration: {duration_s:.2f}s")
    print(f"  Extracting every {step} frame(s)")
    if distortion_params:
        print("  Lens undistortion: enabled")

    records = []
    video_frame_num = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if video_frame_num % step == 0:
            if distortion_params is not None:
                frame = undistort_image(frame, distortion_params)

            if resize is not None and resize != 1.0:
                new_w = int(frame.shape[1] * resize)
                new_h = int(frame.shape[0] * resize)
                frame = cv2.resize(frame, (new_w, new_h))

            timestamp = video_frame_num / fps if fps > 0 else 0.0
            filename = f"frame_{extracted_count:05d}.jpg"
            out_path = output_dir / filename
            cv2.imwrite(str(out_path), frame)

            records.append({
                "frame_index": extracted_count,
                "video_frame_number": video_frame_num,
                "timestamp_s": round(timestamp, 4),
                "path": str(out_path),
            })

            extracted_count += 1
            if max_frames is not None and extracted_count >= max_frames:
                break

        video_frame_num += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")
    return records


def save_frame_manifest(records: list[dict], path: str | Path) -> None:
    import json
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Frame manifest saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Extract frames from a video.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--output", default="data/frames", help="Output directory for frames")
    parser.add_argument("--step", type=int, default=5, help="Extract every Nth frame")
    parser.add_argument("--distortion", default=None, help="Path to distortion params JSON")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to extract")
    parser.add_argument("--resize", type=float, default=None, help="Scale factor (e.g. 0.5)")
    args = parser.parse_args()

    dist_params = load_params(args.distortion) if args.distortion else None

    records = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        step=args.step,
        distortion_params=dist_params,
        max_frames=args.max_frames,
        resize=args.resize,
    )

    manifest_path = Path(args.output) / "manifest.json"
    save_frame_manifest(records, manifest_path)
