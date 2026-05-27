"""
Player detection using YOLO.

Detects up to a configurable number of players (people) in a single frame.
Prefers false negatives over false positives: only high-confidence detections
are kept, and the cap (max_players) is a ceiling, not a target.
"""

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

_model: YOLO | None = None


def _get_model(model_path: str) -> YOLO:
    global _model
    if _model is None or _model.model_name != model_path:
        _model = YOLO(model_path)
    return _model


def detect_players(
    frame: np.ndarray,
    model_path: str = "yolo11n.pt",
    max_players: int = 12,
    conf: float = 0.4,
    iou: float = 0.8,
    imgsz: int = 1536,
) -> tuple[np.ndarray, list[dict]]:
    """
    Detect up to max_players people in a single BGR frame.

    A higher conf threshold keeps precision high at the cost of recall —
    i.e., we'd rather miss a player than hallucinate one.

    Args:
        frame:       BGR image as a NumPy array.
        model_path:  Path or name of the YOLO weights file.
        max_players: Hard ceiling on returned detections. Never returns more
                     than this many; may return fewer if confidence is low.
        conf:        Minimum confidence to keep a detection. Raise to reduce
                     false positives; lower to recover missed players.
        iou:         NMS IoU threshold passed to YOLO.
        imgsz:       Inference image size (longer side in pixels).

    Returns:
        feet_array:  (N, 2) float32 array of [x_center, y_bottom] foot coords,
                     where N <= max_players.
        detections:  List of dicts with keys: box (x1,y1,x2,y2), conf, feet (x,y).
    """
    model = _get_model(model_path)

    results = model.predict(
        frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=[0],  # person only
        verbose=False,
    )

    r = results[0]
    raw: list[tuple[np.ndarray, float]] = []

    if r.boxes:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for box, c in zip(boxes, confs):
            raw.append((box, float(c)))

    # Sort by confidence descending, then cap — never fill up to max_players
    # with low-confidence detections.
    raw.sort(key=lambda x: x[1], reverse=True)
    kept = raw[:max_players]

    detections: list[dict] = []
    feet_coords: list[list[float]] = []

    for box, c in kept:
        x1, y1, x2, y2 = map(int, box)
        x_center = int((x1 + x2) / 2)
        y_bottom = int(y2)
        feet_coords.append([x_center, y_bottom])
        detections.append({
            "box": (x1, y1, x2, y2),
            "conf": c,
            "feet": (x_center, y_bottom),
        })

    print(f"  Raw detections: {len(raw)}, kept (conf≥{conf}, cap={max_players}): {len(detections)}")

    feet_array = (
        np.array(feet_coords, dtype=np.float32)
        if feet_coords
        else np.zeros((0, 2), dtype=np.float32)
    )
    return feet_array, detections


def draw_detections(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    """
    Overlay bounding boxes, confidence labels, and foot markers on a copy of frame.
    Returns a BGR annotated image.
    """
    out = frame.copy()
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        fx, fy = det["feet"]
        c = det["conf"]

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(out, (fx, fy), 6, (0, 0, 255), -1)
        label = f"p{i + 1}  {c:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    cv2.putText(out, f"{len(detections)} player(s) detected", (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return out
