"""
Homography: image pixel coordinates -> 2-D field coordinates (yards).

Field coordinate system
-----------------------
x : 0–70 yd  (left endline → right endline)
y : 0–40 yd  (near sideline → far sideline)

The four corners were annotated on the *undistorted* frame produced by
lens_distortion.undistort_image.  The near corners are off-frame and were
derived by intersecting the extrapolated endlines with the near sideline
(see the field-geometry section of the notebook).
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Field geometry constants (yards)
# ---------------------------------------------------------------------------

FIELD_LENGTH_YD = 70   # endline to endline (playing field only, no end zones)
FIELD_WIDTH_YD  = 40   # sideline to sideline

# ---------------------------------------------------------------------------
# Corner correspondences: undistorted pixel (x, y) → field (x_yd, y_yd)
# ---------------------------------------------------------------------------

_CORNER_PAIRS: list[tuple[tuple[float, float], tuple[float, float]]] = [
    ((  603,    464), ( 0, 40)),   # far-left  corner
    (( 1253,    477), (70, 40)),   # far-right corner
    ((-3753,    921), ( 0,  0)),   # near-left  corner (extrapolated off-frame)
    (( 4815,   1090), (70,  0)),   # near-right corner (extrapolated off-frame)
]

_pixel_pts = np.array([p for p, _ in _CORNER_PAIRS], dtype=np.float32)
_field_pts = np.array([f for _, f in _CORNER_PAIRS], dtype=np.float32)

# H maps undistorted image pixel coords → field coords (yards)
H, _mask = cv2.findHomography(_pixel_pts, _field_pts)

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_point(pixel_xy: tuple[float, float]) -> np.ndarray:
    """Project a single undistorted pixel (x, y) to field coords (x_yd, y_yd)."""
    pt = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    result = H @ pt
    return result[:2] / result[2]


def project_points(pixel_xys) -> np.ndarray:
    """
    Project an (N, 2) array of undistorted pixels to field coords.

    Returns an (N, 2) array of (x_yd, y_yd) values.
    """
    pts = np.array(pixel_xys, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    homog = np.hstack([pts, ones])
    result = (H @ homog.T).T
    return result[:, :2] / result[:, 2:3]


def project_detections(
    detections: list[dict],
    margin: float = 5.0,
) -> list[dict]:
    """
    Project each detection's foot pixel to field coords and discard
    detections that land more than `margin` yards outside the field boundary.

    Each returned dict contains all original detection keys plus:
        field_xy : (x_yd, y_yd) tuple
    """
    if not detections:
        return []

    feet = np.array([d["feet"] for d in detections], dtype=np.float64)
    field_coords = project_points(feet)

    out = []
    for det, fxy in zip(detections, field_coords):
        if is_on_field(fxy, margin=margin):
            out.append({**det, "field_xy": tuple(fxy.tolist())})
    return out


# ---------------------------------------------------------------------------
# Field bounds check
# ---------------------------------------------------------------------------

def is_on_field(
    field_xy,
    margin: float = 2.0,
) -> bool:
    """
    Return True if (x_yd, y_yd) is within the field boundary (plus margin).

    margin > 0 allows detections slightly outside the drawn boundary —
    useful for players near a sideline whose bounding-box foot lands just
    beyond the line.
    """
    x, y = field_xy
    return (
        -margin <= x <= FIELD_LENGTH_YD + margin
        and -margin <= y <= FIELD_WIDTH_YD + margin
    )


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def draw_top_down_field(ax, title: str = "Top-down field view") -> None:
    """
    Draw a regulation ultimate field on *ax* (matplotlib Axes).

    The field coordinate system matches the homography output:
        x-axis : 0–70 yd (endlines)
        y-axis : 0–40 yd (sidelines)
    End zones are shown as shaded areas at x < 0 and x > 70 if the caller
    wants to extend the axes, but the homography covers only 0–70 yd.
    """
    import matplotlib.patches as mpatches

    ax.set_xlim(-2, FIELD_LENGTH_YD + 2)
    ax.set_aspect("equal")
    ax.set_facecolor("#4a7c3f")   # grass green

    # Playing-field rectangle
    field_rect = mpatches.FancyBboxPatch(
        (0, 0), FIELD_LENGTH_YD, FIELD_WIDTH_YD,
        boxstyle="square,pad=0",
        linewidth=2, edgecolor="white", facecolor="#5a9e50",
    )
    ax.add_patch(field_rect)

    # Centre line
    ax.plot(
        [FIELD_LENGTH_YD / 2, FIELD_LENGTH_YD / 2], [0, FIELD_WIDTH_YD],
        "w--", linewidth=1, alpha=0.6,
    )

    # Yard markers every 10 yd
    for x in range(0, FIELD_LENGTH_YD + 1, 10):
        ax.plot([x, x], [0, FIELD_WIDTH_YD], "w-", linewidth=0.6, alpha=0.4)

    ax.set_ylim(-2, FIELD_WIDTH_YD + 2)
    ax.set_xlabel("x (yards)")
    ax.set_ylabel("y (yards)")
    ax.set_title(title)


def plot_players_on_field(
    ax,
    projected: list[dict],
    color: str = "yellow",
    label: str = "player",
    zorder: int = 5,
) -> None:
    """
    Scatter-plot projected player positions on a field axes already set up
    by draw_top_down_field.

    projected : list of dicts with a 'field_xy' key (x_yd, y_yd).
    """
    if not projected:
        return
    xs = [d["field_xy"][0] for d in projected]
    ys = [d["field_xy"][1] for d in projected]
    ax.scatter(xs, ys, s=120, color=color, edgecolors="black",
               linewidths=0.8, zorder=zorder, label=label)
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.text(x, y + 0.8, f"p{i+1}", ha="center", va="bottom",
                fontsize=7, color="white", zorder=zorder + 1)
