"""
End-to-end pipeline: video -> frames -> detections -> tracks -> visualization.

Reproduces the result validated in notebooks/tracking_debug.ipynb. Reuses
existing extracted frames when data/frames/manifest.json already exists,
and otherwise extracts them from data/film.mp4 with the saved fisheye
distortion correction.

Run from the project root:

    python src/pipeline.py
"""

import json
import sys
from pathlib import Path

# Make both src/ and the project root importable so that:
#   - top-level imports like `import homography` work
#   - interpolate_tracks.py's `from src.tracker import Track` works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC          = _PROJECT_ROOT / "src"
for p in (_SRC, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cv2
import numpy as np

import homography as hg
from detect_players import detect_players
from extract_frames import extract_frames, save_frame_manifest
from export_tracks import export_tracks_csv
from interpolate_tracks import interpolate_tracks, stitch_tracks
from lens_distortion import load_params
from tracker import Tracker
from visualize import animate_tracks


# ---------------------------------------------------------------------------
# Defaults — match the parameter set validated in tracking_debug.ipynb
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    video_path        = str(_PROJECT_ROOT / "data/film.mp4"),
    frames_dir        = str(_PROJECT_ROOT / "data/frames"),
    output_dir        = str(_PROJECT_ROOT / "data/outputs"),
    csv_path          = str(_PROJECT_ROOT / "data/tracks/tracks_projected.csv"),
    distortion_params = str(_PROJECT_ROOT / "data/distortion_params.json"),
    yolo_weights      = "yolo11n.pt",

    frame_step = 5,

    # Detection
    detect_conf        = 0.25,
    detect_max_players = 13,
    detect_imgsz       = 1920,
    detect_iou         = 0.5,

    # Exclusion zones: (pixel_x, pixel_y, radius_yd) on the undistorted frame
    spectator_px = [(855, 466, 12)],

    # Tracker
    match_gate_base  = 2.0,
    match_gate_scale = 0.5,
    match_gate_max   = 7.0,
    max_gap_frames   = 15,
    sigma_a          = 4.0,
    sigma_z          = 0.5,

    # Stationary-track filter
    min_displacement_yd = 2.0,

    # Cross-track stitching
    stitch_max_gap   = 12,
    stitch_threshold = 6.0,

    # Within-track interpolation
    interp_max_linear_gap = 5,
    interp_hold_threshold = 3.5,

    # Visualization
    anim_fps    = 6.0,
    anim_output = "topdown_demo.mp4",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_exclusion_zones(spectator_px):
    """Project (pixel_x, pixel_y, radius_yd) entries to (field_x, field_y, radius_yd)."""
    out = []
    for px, py, r in spectator_px:
        x_yd, y_yd = hg.project_point((px, py))
        out.append((float(x_yd), float(y_yd), r))
    return out


def _total_displacement(track):
    real = [o for o in track.observations if not o["is_interpolated"]]
    if len(real) < 2:
        return 0.0
    xs = [o["x_yd"] for o in real]
    ys = [o["y_yd"] for o in real]
    return sum(
        float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]))
        for i in range(len(xs) - 1)
    )


def _load_or_extract_frames(video_path, frames_dir, distortion_path, step):
    manifest_path = frames_dir / "manifest.json"
    if manifest_path.exists():
        print(f"  Reusing existing frames at {frames_dir}")
        with open(manifest_path) as f:
            return json.load(f)

    print(f"  Extracting frames from {video_path}")
    dist_params = load_params(distortion_path) if distortion_path.exists() else None
    records = extract_frames(
        video_path=video_path,
        output_dir=frames_dir,
        step=step,
        distortion_params=dist_params,
    )
    save_frame_manifest(records, manifest_path)
    return records


def _resolve_frame_path(record, frames_dir):
    for key in ("path", "abs_path"):
        if key in record:
            p = Path(record[key])
            return p if p.is_absolute() else _PROJECT_ROOT / p
    return frames_dir / f"frame_{record['frame_index']:05d}.jpg"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config: dict | None = None) -> dict:
    """
    Execute every stage of the pipeline and write all outputs to disk.

    Returns a dict with the final tracks, CSV path, animation path, and
    frame manifest.
    """
    cfg = {**DEFAULTS, **(config or {})}

    video_path = Path(cfg["video_path"])
    frames_dir = Path(cfg["frames_dir"])
    output_dir = Path(cfg["output_dir"])
    csv_path   = Path(cfg["csv_path"])
    distortion = Path(cfg["distortion_params"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Frame extraction ────────────────────────────────────────────────
    print("\n[1/6] Frame extraction")
    manifest = _load_or_extract_frames(
        video_path, frames_dir, distortion, cfg["frame_step"]
    )
    print(f"  {len(manifest)} frames")

    # ── 2. Player detection (YOLO) ─────────────────────────────────────────
    print("\n[2/6] Player detection")
    raw_detections = []
    for record in manifest:
        bgr = cv2.imread(str(_resolve_frame_path(record, frames_dir)))
        if bgr is None:
            raise FileNotFoundError(record)
        _, dets = detect_players(
            bgr,
            model_path  = cfg["yolo_weights"],
            max_players = cfg["detect_max_players"],
            conf        = cfg["detect_conf"],
            iou         = cfg["detect_iou"],
            imgsz       = cfg["detect_imgsz"],
        )
        raw_detections.append({
            "frame_idx":   record["frame_index"],
            "timestamp_s": record["timestamp_s"],
            "detections":  dets,
        })
    n_raw = sum(len(d["detections"]) for d in raw_detections)
    print(f"  {n_raw} raw detections")

    # ── 3. Homography projection (+ exclusion zones) ───────────────────────
    print("\n[3/6] Homography projection")
    exclusion_zones = _build_exclusion_zones(cfg["spectator_px"])
    for ex in exclusion_zones:
        print(f"  exclusion zone: ({ex[0]:.1f}, {ex[1]:.1f}) yd, r={ex[2]} yd")

    projected_per_frame = []
    for frame in raw_detections:
        proj = hg.project_detections(
            frame["detections"],
            margin=5.0,
            exclusion_zones=exclusion_zones or None,
        )
        projected_per_frame.append({
            "frame_idx":   frame["frame_idx"],
            "timestamp_s": frame["timestamp_s"],
            "detections":  proj,
        })
    n_proj = sum(len(p["detections"]) for p in projected_per_frame)
    print(f"  {n_proj} on-field detections (from {n_raw} raw)")

    # ── 4. Tracking ────────────────────────────────────────────────────────
    print("\n[4/6] Tracking")
    trk = Tracker(
        match_gate_base  = cfg["match_gate_base"],
        match_gate_scale = cfg["match_gate_scale"],
        match_gate_max   = cfg["match_gate_max"],
        max_gap_frames   = cfg["max_gap_frames"],
        sigma_a          = cfg["sigma_a"],
        sigma_z          = cfg["sigma_z"],
    )
    for frame in projected_per_frame:
        trk.update(
            frame_idx   = frame["frame_idx"],
            timestamp_s = frame["timestamp_s"],
            detections  = frame["detections"],
        )
    tracks = trk.get_tracks()
    print(f"  {len(tracks)} raw tracks")

    pre = len(tracks)
    tracks = [t for t in tracks if _total_displacement(t) >= cfg["min_displacement_yd"]]
    print(f"  Stationary filter (>= {cfg['min_displacement_yd']} yd): "
          f"kept {len(tracks)} / removed {pre - len(tracks)}")

    # ── 5. Stitch + interpolate ────────────────────────────────────────────
    print("\n[5/6] Stitch + interpolate")
    pre = len(tracks)
    tracks = stitch_tracks(
        tracks,
        max_stitch_gap   = cfg["stitch_max_gap"],
        stitch_threshold = cfg["stitch_threshold"],
    )
    print(f"  Stitched {pre - len(tracks)} pair(s) -> {len(tracks)} tracks")

    interpolate_tracks(
        tracks,
        max_linear_gap = cfg["interp_max_linear_gap"],
        hold_threshold = cfg["interp_hold_threshold"],
    )
    n_interp = sum(sum(1 for o in t.observations if o["is_interpolated"]) for t in tracks)
    n_real   = sum(sum(1 for o in t.observations if not o["is_interpolated"]) for t in tracks)
    print(f"  Real: {n_real}, interpolated: {n_interp}")

    # ── 6. Export + animate ────────────────────────────────────────────────
    print("\n[6/6] Export + visualize")
    csv_out = export_tracks_csv(tracks, csv_path)
    print(f"  CSV: {csv_out}")

    anim_out = animate_tracks(
        tracks,
        frame_records = manifest,
        output_path   = output_dir / cfg["anim_output"],
        fps           = cfg["anim_fps"],
    )
    print(f"  Animation: {anim_out}")

    return {
        "tracks":    tracks,
        "csv":       csv_out,
        "animation": anim_out,
        "manifest":  manifest,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full ultimate-frisbee tracking pipeline."
    )
    parser.add_argument("--video",  default=DEFAULTS["video_path"])
    parser.add_argument("--frames", default=DEFAULTS["frames_dir"])
    parser.add_argument("--output", default=DEFAULTS["output_dir"])
    parser.add_argument("--csv",    default=DEFAULTS["csv_path"])
    args = parser.parse_args()

    run_pipeline({
        "video_path": args.video,
        "frames_dir": args.frames,
        "output_dir": args.output,
        "csv_path":   args.csv,
    })
