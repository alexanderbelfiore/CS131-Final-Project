"""
Player tracking in field coordinates.

Pipeline role
-------------
Receives projected (field-space) detections per frame from homography.py and
assigns persistent track IDs. Identity is maintained across frames using a
constant-velocity Kalman filter for short-horizon prediction plus Hungarian
assignment on field-space distance.

Design choices (matching the project plan, section 5)
-----------------------------------------------------
- All detections (on- and off-field) come in; homography pre-filters to on-field.
- Single-detection track creation (no minimum confirmation streak).
- Kalman uses the actual dt between frames, not a unit step.
- Gating widens linearly with consecutive lost frames so reactivation tolerates
  predicted-vs-actual drift after multi-frame occlusion.
- A track stays alive for `max_gap_frames` lost frames before dying; if the
  player reappears within that window, the same track ID is reused.
- A new max-tracks ceiling caps simultaneous live tracks (6v6 = 12 players).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Constant-velocity Kalman filter
# ---------------------------------------------------------------------------

class CVKalman:
    """
    Constant-velocity Kalman filter in field coordinates.

    State  : [x, y, vx, vy]   (yards, yards/s)
    Measurement : [x, y]      (yards)

    sigma_a : standard deviation of the acceleration disturbance (yd/s^2).
              Larger -> filter trusts measurements more, prediction drifts less.
              Players cut hard, so we set this fairly high.
    sigma_z : standard deviation of position measurement noise (yards).
              Reflects homography + bounding-box-foot precision.
    """

    def __init__(self, init_xy, sigma_a: float = 4.0, sigma_z: float = 0.5):
        self.x = np.array([init_xy[0], init_xy[1], 0.0, 0.0], dtype=np.float64)
        # Low initial position uncertainty (we just measured it).
        # High initial velocity uncertainty (we have no idea where they're going).
        self.P = np.diag([0.5 ** 2, 0.5 ** 2, 5.0 ** 2, 5.0 ** 2]).astype(np.float64)
        self.sigma_a = float(sigma_a)
        self.sigma_z = float(sigma_z)
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self.R = (self.sigma_z ** 2) * np.eye(2, dtype=np.float64)

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        return np.array([[1.0, 0.0, dt,  0.0],
                         [0.0, 1.0, 0.0, dt ],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    def _Q(self, dt: float) -> np.ndarray:
        # Discrete white-noise acceleration model.
        s2 = self.sigma_a ** 2
        return s2 * np.array([
            [dt ** 4 / 4, 0,           dt ** 3 / 2, 0          ],
            [0,           dt ** 4 / 4, 0,           dt ** 3 / 2],
            [dt ** 3 / 2, 0,           dt ** 2,     0          ],
            [0,           dt ** 3 / 2, 0,           dt ** 2    ],
        ], dtype=np.float64)

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update(self, z) -> None:
        z = np.asarray(z, dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------

@dataclass
class Track:
    track_id: int
    kalman: CVKalman
    start_frame: int
    last_detection_frame: int
    lost_frames: int = 0
    status: str = "live"   # "live" | "lost" | "dead"
    observations: list = field(default_factory=list)
    # Each observation is a dict:
    #   {frame_idx, timestamp_s, x_yd, y_yd, conf, is_interpolated}


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class Tracker:
    """
    Multi-target tracker that consumes per-frame projected detections and
    maintains persistent track IDs.

    Parameters
    ----------
    match_gate_base : yards. Gate around a Kalman prediction at lost=0.
    match_gate_scale: yards-per-lost-frame the gate widens by.
    match_gate_max  : yards. Hard cap on the gate.
    max_gap_frames  : how many consecutive missed frames a track is kept alive.
    sigma_a, sigma_z: passed through to each track's Kalman filter.

    Every unmatched on-field detection spawns a new track — no capacity cap.
    Spurious tracks (spectators, noise) are removed by the stationary filter
    applied after tracking.

    Defaults assume frames are extracted at step=5 from ~30 fps video
    (effective ~6 fps, dt ≈ 0.17 s). A player at ~7 yd/s moves ~1.2 yd/frame,
    so a base gate of 2 yd handles normal frame-to-frame motion comfortably.
    """

    def __init__(
        self,
        match_gate_base: float = 2.0,
        match_gate_scale: float = 0.5,
        match_gate_max: float = 5.0,
        max_gap_frames: int = 10,
        sigma_a: float = 4.0,
        sigma_z: float = 0.5,
    ):
        self.match_gate_base = match_gate_base
        self.match_gate_scale = match_gate_scale
        self.match_gate_max = match_gate_max
        self.max_gap_frames = max_gap_frames
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z

        self.tracks: list[Track] = []
        self._next_id: int = 1
        self._prev_ts: Optional[float] = None

    # -- helpers ------------------------------------------------------------

    def _gate(self, lost_frames: int) -> float:
        return min(
            self.match_gate_max,
            self.match_gate_base + self.match_gate_scale * lost_frames,
        )

    def _live_count(self) -> int:
        return sum(1 for t in self.tracks if t.status != "dead")

    # -- main entry point ---------------------------------------------------

    def update(
        self,
        frame_idx: int,
        timestamp_s: float,
        detections: list[dict],
    ) -> None:
        """
        Step the tracker forward by one frame.

        detections : list of dicts with at least:
            - 'field_xy' : (x_yd, y_yd)
            - 'conf'     : float
        Typically produced by homography.project_detections().
        """
        dt = 0.0 if self._prev_ts is None else (timestamp_s - self._prev_ts)

        # 1) Predict every still-alive track forward.
        active = [t for t in self.tracks if t.status != "dead"]
        for t in active:
            t.kalman.predict(dt)

        # 2) Build cost matrix (rows=tracks, cols=detections) with per-track
        #    gating; entries beyond the gate are marked as "infeasible".
        n_t, n_d = len(active), len(detections)
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        if n_t > 0 and n_d > 0:
            cost = np.full((n_t, n_d), np.inf, dtype=np.float64)
            for i, t in enumerate(active):
                px, py = t.kalman.position
                gate = self._gate(t.lost_frames)
                for j, det in enumerate(detections):
                    dx = det["field_xy"][0] - px
                    dy = det["field_xy"][1] - py
                    dist = float(np.hypot(dx, dy))
                    if dist <= gate:
                        cost[i, j] = dist

            # 3) Hungarian assignment. scipy needs finite costs, so substitute
            #    a large finite sentinel for infeasible pairs, then re-check
            #    against the original cost to reject those matches.
            big = 1e6
            cost_filled = np.where(np.isinf(cost), big, cost)
            row_ind, col_ind = linear_sum_assignment(cost_filled)

            for r, c in zip(row_ind, col_ind):
                if np.isinf(cost[r, c]):
                    continue   # gated out; not a real match
                t = active[r]
                det = detections[c]
                t.kalman.update(det["field_xy"])
                t.last_detection_frame = frame_idx
                t.lost_frames = 0
                t.status = "live"
                t.observations.append({
                    "frame_idx": frame_idx,
                    "timestamp_s": timestamp_s,
                    "x_yd": float(t.kalman.position[0]),
                    "y_yd": float(t.kalman.position[1]),
                    "conf": float(det["conf"]),
                    "is_interpolated": False,
                })
                matched_t.add(r)
                matched_d.add(c)

        # 4) Unmatched tracks: bump lost counter, possibly kill.
        for i, t in enumerate(active):
            if i in matched_t:
                continue
            t.lost_frames += 1
            if t.lost_frames > self.max_gap_frames:
                t.status = "dead"
            else:
                t.status = "lost"

        # 5) Unmatched detections: each spawns a new track unconditionally.
        for j, det in enumerate(detections):
            if j in matched_d:
                continue
            tr = Track(
                track_id=self._next_id,
                kalman=CVKalman(det["field_xy"],
                                sigma_a=self.sigma_a,
                                sigma_z=self.sigma_z),
                start_frame=frame_idx,
                last_detection_frame=frame_idx,
            )
            tr.observations.append({
                "frame_idx": frame_idx,
                "timestamp_s": timestamp_s,
                "x_yd": float(det["field_xy"][0]),
                "y_yd": float(det["field_xy"][1]),
                "conf": float(det["conf"]),
                "is_interpolated": False,
            })
            self.tracks.append(tr)
            self._next_id += 1

        self._prev_ts = timestamp_s

    # -- output -------------------------------------------------------------

    def get_tracks(self) -> list[Track]:
        """Return all tracks created by the tracker."""
        return list(self.tracks)


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def run_tracker(
    per_frame_detections: list[dict],
    **tracker_kwargs,
) -> list[Track]:
    """
    Convenience wrapper to run the tracker over a whole clip.

    per_frame_detections: list of dicts, ordered by frame, with keys:
        - 'frame_idx'    : int
        - 'timestamp_s'  : float
        - 'detections'   : list of projected detection dicts (with 'field_xy')

    Returns the final list of Track objects (not yet interpolated; pass to
    interpolate_tracks.interpolate_tracks for that).
    """
    trk = Tracker(**tracker_kwargs)
    for frame in per_frame_detections:
        trk.update(
            frame_idx=frame["frame_idx"],
            timestamp_s=frame["timestamp_s"],
            detections=frame["detections"],
        )
    return trk.get_tracks()
