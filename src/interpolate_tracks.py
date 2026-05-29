"""
Fill short gaps within each track's observations, and optionally stitch
fragmented tracks back together.

Within-track interpolation
--------------------------
For each gap between consecutive observations at frames f1 < f2:

    gap          = f2 - f1 - 1
    displacement = ||(x2,y2) - (x1,y1)||  (yards)

- gap == 0                                                  -> nothing to do
- gap <= max_linear_gap                                     -> linear interp
- gap >  max_linear_gap AND displacement < hold_threshold   -> linear interp
       (endpoints nearly identical → effectively a position-hold)
- gap >  max_linear_gap AND displacement >= hold_threshold  -> skip

Cross-track stitching  (stitch_tracks)
--------------------------------------
When a player's track dies and the same player reappears later as a new track,
stitch_tracks() merges them: it scans for pairs (dead track A, new track B)
where B starts within `max_stitch_gap` frames of A's last detection AND the
spatial displacement between A's last known position and B's first detection is
within `stitch_threshold` yards. Pairs are selected greedily by distance so
that a close nearby reappearance beats a distant one. The gap between A and B
is filled with the same linear/hold policy used within-track. B's observations
are appended to A and B is removed from the list.

Interpolated rows carry is_interpolated=True and conf=None.
"""

import numpy as np

from src.tracker import Track


def interpolate_track(
    track: Track,
    max_linear_gap: int = 5,
    hold_threshold: float = 2.0,
) -> None:
    """
    In-place gap filling for a single track. See module docstring for policy.
    """
    if len(track.observations) < 2:
        return

    obs = sorted(track.observations, key=lambda o: o["frame_idx"])
    out = [obs[0]]

    for prev, curr in zip(obs, obs[1:]):
        gap = curr["frame_idx"] - prev["frame_idx"] - 1
        if gap > 0:
            dx = curr["x_yd"] - prev["x_yd"]
            dy = curr["y_yd"] - prev["y_yd"]
            displacement = float(np.hypot(dx, dy))

            should_fill = (
                gap <= max_linear_gap
                or displacement < hold_threshold
            )
            if should_fill:
                for k in range(1, gap + 1):
                    alpha = k / (gap + 1)
                    out.append({
                        "frame_idx": prev["frame_idx"] + k,
                        "timestamp_s": (
                            prev["timestamp_s"]
                            + alpha * (curr["timestamp_s"] - prev["timestamp_s"])
                        ),
                        "x_yd": prev["x_yd"] + alpha * dx,
                        "y_yd": prev["y_yd"] + alpha * dy,
                        "conf": None,
                        "is_interpolated": True,
                    })
        out.append(curr)

    track.observations = out


def interpolate_tracks(
    tracks: list[Track],
    max_linear_gap: int = 5,
    hold_threshold: float = 3.5,
) -> list[Track]:
    """Apply interpolate_track to each track in place. Returns the same list."""
    for t in tracks:
        interpolate_track(
            t,
            max_linear_gap=max_linear_gap,
            hold_threshold=hold_threshold,
        )
    return tracks


# ---------------------------------------------------------------------------
# Cross-track stitching
# ---------------------------------------------------------------------------

def _real_obs(track: Track) -> list[dict]:
    """Observations that are real detections (not previously interpolated)."""
    return [o for o in track.observations if not o["is_interpolated"]]


def _gap_fill(o_prev: dict, o_next: dict, gap: int) -> list[dict]:
    """Linear interpolation between two observations across `gap` missing frames."""
    rows = []
    dx = o_next["x_yd"] - o_prev["x_yd"]
    dy = o_next["y_yd"] - o_prev["y_yd"]
    dt = o_next["timestamp_s"] - o_prev["timestamp_s"]
    for k in range(1, gap + 1):
        alpha = k / (gap + 1)
        rows.append({
            "frame_idx": o_prev["frame_idx"] + k,
            "timestamp_s": o_prev["timestamp_s"] + alpha * dt,
            "x_yd": o_prev["x_yd"] + alpha * dx,
            "y_yd": o_prev["y_yd"] + alpha * dy,
            "conf": None,
            "is_interpolated": True,
        })
    return rows


def stitch_tracks(
    tracks: list[Track],
    max_stitch_gap: int = 12,
    stitch_threshold: float = 6.0,
    max_linear_gap: int = 5,
    hold_threshold: float = 3.5,
) -> list[Track]:
    """
    Merge fragmented tracks that belong to the same player.

    For each pair (track A, track B) where:
      - A ends before B starts
      - B's first real detection is within `max_stitch_gap` frames of A's last
      - The spatial gap is within `stitch_threshold` yards

    ...B is merged into A: A keeps its track_id, the gap frames are filled with
    the same linear/hold policy as within-track interpolation, and B is dropped.

    Candidates are sorted by (gap, distance) so the closest match wins when
    multiple new tracks start near a dead one.

    Should be called BEFORE interpolate_tracks (so within-track gaps are filled
    after all stitches are finalised).
    """
    # Sort working list by start frame so we process in temporal order.
    working = sorted(tracks, key=lambda t: t.start_frame)

    # Pre-compute last real obs for every track.
    last_real: dict[int, dict] = {}
    for t in working:
        ro = _real_obs(t)
        if ro:
            last_real[t.track_id] = ro[-1]

    first_real: dict[int, dict] = {}
    for t in working:
        ro = _real_obs(t)
        if ro:
            first_real[t.track_id] = ro[0]

    # Build all candidate stitch pairs, then sort by distance.
    candidates: list[tuple[float, int, Track, Track]] = []
    track_by_id = {t.track_id: t for t in working}

    for b in working:
        if b.track_id not in first_real:
            continue
        b_first = first_real[b.track_id]

        for a in working:
            if a.track_id == b.track_id:
                continue
            if a.track_id not in last_real:
                continue
            a_last = last_real[a.track_id]

            gap = b_first["frame_idx"] - a_last["frame_idx"] - 1
            if gap < 0 or gap > max_stitch_gap:
                continue

            dist = float(np.hypot(
                b_first["x_yd"] - a_last["x_yd"],
                b_first["y_yd"] - a_last["y_yd"],
            ))
            if dist <= stitch_threshold:
                candidates.append((dist, gap, a, b))

    candidates.sort(key=lambda c: (c[0], c[1]))

    merged: set[int] = set()   # track IDs absorbed into another

    for dist, gap, a, b in candidates:
        if a.track_id in merged or b.track_id in merged:
            continue

        a_last = last_real[a.track_id]
        b_first = first_real[b.track_id]
        actual_gap = b_first["frame_idx"] - a_last["frame_idx"] - 1

        # Decide whether to interpolate the gap using the same policy.
        displacement = float(np.hypot(
            b_first["x_yd"] - a_last["x_yd"],
            b_first["y_yd"] - a_last["y_yd"],
        ))
        should_fill = (
            actual_gap <= max_linear_gap
            or displacement < hold_threshold
        )
        gap_rows = _gap_fill(a_last, b_first, actual_gap) if should_fill else []

        a.observations = sorted(
            a.observations + gap_rows + b.observations,
            key=lambda o: o["frame_idx"],
        )
        # Update a's last_real for potential downstream chains.
        b_ro = _real_obs(b)
        if b_ro:
            last_real[a.track_id] = b_ro[-1]

        merged.add(b.track_id)

    return [t for t in working if t.track_id not in merged]
