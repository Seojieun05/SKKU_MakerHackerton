# SPDX-License-Identifier: AGPL-3.0-only
import cv2
import numpy as np

COLORS = {"OCCUPIED": (40, 60, 230), "EMPTY": (70, 190, 70), "UNKNOWN": (140, 140, 140)}


def pixel_polygon(seat, width, height):
    return np.array([(round(x * (width - 1)), round(y * (height - 1))) for x, y in seat.polygon], dtype=np.int32)


def annotate(frame, seats, states=None, detections=()):
    view = frame.copy()
    h, w = view.shape[:2]
    for seat in seats:
        state = states[seat.id].state if states is not None else "UNKNOWN"
        polygon = pixel_polygon(seat, w, h)
        color = COLORS[state]
        cv2.polylines(view, [polygon], True, color, 2)
        x, y = polygon[0]
        cv2.putText(view, f"{seat.id}: {state}", (int(x), max(22, int(y) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    for det in detections:
        x1, y1, x2, y2 = det.box
        cv2.rectangle(view, (round(x1 * w), round(y1 * h)), (round(x2 * w), round(y2 * h)), (230, 170, 40), 1)
        ax, ay = det.anchor
        cv2.circle(view, (round(ax * w), round(ay * h)), 5, (0, 230, 255), -1)
    return view
