"""
Flatten interpolated tracks into a per-frame, per-player CSV.

Output schema (matches project plan section "Input and Output"):

    frame,timestamp,player_id,x_yards,y_yards,confidence,is_interpolated

- One row per (frame, player) for which the player has a real or interpolated
  observation. Frames where a player has no observation are omitted (we don't
  pad with NaNs or extrapolate beyond a track's window).
- player_id is rendered as "player_01", "player_02", ... using the track_id
  (zero-padded to width 2; widened automatically for more than 99 tracks).
- confidence is left empty for interpolated rows.
- is_interpolated is written as lowercase "true"/"false" to match the example.

Rows are sorted by (frame, player_id) for easy frame-by-frame consumption.
"""

import csv
from pathlib import Path

from src.tracker import Track


def _pid(track_id: int, width: int) -> str:
    return f"player_{track_id:0{width}d}"


def tracks_to_rows(tracks: list[Track]) -> list[dict]:
    """
    Flatten a list of tracks (already gap-filled by interpolate_tracks) into
    one dict per (frame, player) observation, sorted by (frame, player_id).
    """
    if not tracks:
        return []

    max_id = max(t.track_id for t in tracks)
    width = max(2, len(str(max_id)))

    rows: list[dict] = []
    for t in tracks:
        pid = _pid(t.track_id, width)
        for o in t.observations:
            rows.append({
                "frame": o["frame_idx"],
                "timestamp": round(float(o["timestamp_s"]), 3),
                "player_id": pid,
                "x_yards": round(float(o["x_yd"]), 2),
                "y_yards": round(float(o["y_yd"]), 2),
                "confidence": (
                    "" if o["conf"] is None else round(float(o["conf"]), 3)
                ),
                "is_interpolated": "true" if o["is_interpolated"] else "false",
            })

    rows.sort(key=lambda r: (r["frame"], r["player_id"]))
    return rows


def export_tracks_csv(
    tracks: list[Track],
    out_path: str | Path,
) -> Path:
    """Write the flattened rows to CSV. Returns the output path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = tracks_to_rows(tracks)
    fieldnames = [
        "frame", "timestamp", "player_id",
        "x_yards", "y_yards", "confidence", "is_interpolated",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return out_path
