# SPDX-License-Identifier: AGPL-3.0-only
"""On-device mouse calibration. Only polygon coordinates are saved."""
from .camera import FrameSource
from .config import config_document, parse_config
from .core import Seat
from .status import write_json


def calibrate(args):
    import cv2
    import numpy as np
    from .display import annotate

    if args.config.exists() and not args.overwrite:
        raise FileExistsError(f"{args.config} already exists. Use --overwrite to replace it.")
    seats, points = [], []
    window = "Seat calibration: click corners | N next | U undo | S save | Q cancel"
    try:
        with FrameSource(args.source, args.width, args.height, args.camera_index) as camera:
            frame = camera.read()
            if frame is None:
                raise RuntimeError("Video has no frames.")
            # Freeze one frame in RAM so the calibration doesn't move under the mouse.
            h, w = frame.shape[:2]
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

            def click(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < w and 0 <= y < h:
                    points.append((x / max(1, w - 1), y / max(1, h - 1)))

            cv2.setMouseCallback(window, click)
            message = "Draw the area containing the CENTER of a seated person's detection box."
            while True:
                view = annotate(frame, seats)
                if points:
                    pixel_points = np.array([(round(x * (w - 1)), round(y * (h - 1))) for x, y in points], np.int32)
                    cv2.polylines(view, [pixel_points], len(points) >= 3, (0, 220, 255), 2)
                    for x, y in pixel_points:
                        cv2.circle(view, (int(x), int(y)), 4, (0, 220, 255), -1)
                cv2.putText(view, message, (12, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
                cv2.imshow(window, view)
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
                if key == ord("u"):
                    if points:
                        points.pop()
                    elif seats:
                        points.extend(seats.pop().polygon)
                if key in (ord("n"), ord("s")):
                    if points:
                        new_seats = seats + [Seat(f"A{len(seats) + 1:02d}", tuple(points))]
                        try:
                            parse_config(config_document(new_seats, (w, h)))
                        except ValueError:
                            message = "Invalid polygon. Click 3+ corners in order; U undoes a point."
                            continue
                        seats = new_seats
                        points.clear()
                        message = "Seat added. Draw another seat, or press S to save."
                    if key == ord("s") and seats:
                        write_json(args.config, config_document(seats, (w, h)))
                        print(f"Saved {len(seats)} seats to {args.config}")
                        return 0
    finally:
        cv2.destroyAllWindows()
