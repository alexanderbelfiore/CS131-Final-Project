"""
Top-down field visualization of player tracks.

Renders the ultimate field with uniformly-coloured player markers (dark
blue for real detections, a brighter blue for interpolated positions).
Supports both single-frame rendering and animation export to MP4 or GIF.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from homography import draw_top_down_field


# All players use the same dark-blue marker; interpolated positions get a
# brighter shade so they're visually distinct without being a different colour.
PLAYER_COLOR        = "#0b1d51"   # dark navy, real detections
PLAYER_COLOR_INTERP = "#4a7fc6"   # brighter blue, interpolated frames
PLAYER_EDGE         = "black"
PLAYER_SIZE         = 200


def _positions_at_frame(tracks, frame_idx):
    """Return [(x_yd, y_yd, is_interpolated), ...] for tracks active at frame_idx."""
    out = []
    for t in tracks:
        for o in t.observations:
            if o["frame_idx"] == frame_idx:
                out.append((o["x_yd"], o["y_yd"], o["is_interpolated"]))
                break
    return out


def render_frame(ax, tracks, frame_idx, timestamp_s=None, title=None):
    """Draw the top-down field with player markers for one frame onto `ax`."""
    if title is None:
        title = (
            f"Frame {frame_idx}  t={timestamp_s:.2f}s"
            if timestamp_s is not None else f"Frame {frame_idx}"
        )

    ax.clear()
    draw_top_down_field(ax, title=title)

    for x, y, is_interp in _positions_at_frame(tracks, frame_idx):
        color = PLAYER_COLOR_INTERP if is_interp else PLAYER_COLOR
        ax.scatter(
            [x], [y],
            s=PLAYER_SIZE, color=color,
            edgecolors=PLAYER_EDGE, linewidths=0.8,
            zorder=5,
        )


def animate_tracks(
    tracks,
    frame_records,
    output_path,
    fps: float = 6.0,
    figsize: tuple[float, float] = (11, 7),
) -> Path:
    """
    Save a top-down animation of player positions across `frame_records`.

    Args:
        tracks:        Player tracks (post-stitch, post-interpolation).
        frame_records: List of dicts with `frame_index` and `timestamp_s`,
                       typically the frame manifest from extract_frames.
        output_path:   Output path. Extension chooses the writer: .mp4
                       uses ffmpeg, .gif uses pillow.
        fps:           Output frame rate. Default 6.0 matches a ~30 fps
                       source extracted with step=5.
        figsize:       Matplotlib figure size in inches.

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    def update(record):
        render_frame(
            ax, tracks,
            frame_idx=record["frame_index"],
            timestamp_s=record["timestamp_s"],
        )

    anim = FuncAnimation(
        fig, update,
        frames=frame_records,
        interval=1000 / fps,
        repeat=True,
    )

    from matplotlib.animation import writers as _writers

    ext = output_path.suffix.lower()
    if ext in (".mp4", ".mov"):
        if _writers.is_available("ffmpeg"):
            anim.save(str(output_path), writer="ffmpeg", fps=fps, dpi=120)
        else:
            print("  ffmpeg not found — saving as .gif instead")
            output_path = output_path.with_suffix(".gif")
            anim.save(str(output_path), writer="pillow", fps=fps, dpi=100)
    elif ext == ".gif":
        anim.save(str(output_path), writer="pillow", fps=fps, dpi=100)
    else:
        raise ValueError(
            f"Unsupported animation extension: {ext} (use .mp4 or .gif)"
        )

    plt.close(fig)
    return output_path
