import cv2
import numpy as np

# Four field corners: image pixel (x, y) → field coordinate (x_yd, y_yd)
# Field system: x = 0–70 yd (endline to endline), y = 0–40 yd (far sideline to near sideline)
_CORNER_PAIRS = [
    ((  603,   464), ( 0,  0)),   # far-left  corner
    (( 1253,   477), (70,  0)),   # far-right corner
    ((-3753,   921), ( 0, 40)),   # near-left  corner (extrapolated off-frame)
    (( 4815,  1090), (70, 40)),   # near-right corner (extrapolated off-frame)
]

_pixel_pts = np.array([p for p, _ in _CORNER_PAIRS], dtype=np.float32)
_field_pts  = np.array([f for _, f in _CORNER_PAIRS], dtype=np.float32)

# H maps image pixel coords → field coords (yards)
H, _ = cv2.findHomography(_pixel_pts, _field_pts)


def project_point(pixel_xy):
    """Project a single image pixel (x, y) to field coordinates (x_yd, y_yd)."""
    pt = np.array([pixel_xy[0], pixel_xy[1], 1.0], dtype=np.float64)
    result = H @ pt
    return result[:2] / result[2]


def project_points(pixel_xys):
    """Project an array of image pixels, shape (N, 2), to field coords, shape (N, 2)."""
    pts = np.array(pixel_xys, dtype=np.float64)
    ones = np.ones((len(pts), 1))
    homog = np.hstack([pts, ones])
    result = (H @ homog.T).T
    return result[:, :2] / result[:, 2:3]
