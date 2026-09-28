"""Opt-in real-model smoke test. Supply a local photo with at least one person.

python -m tests.smoke_model --image /path/to/person.jpg
No images or model weights are bundled. Frames are never written to disk.
"""
import argparse
import contextlib
import hashlib
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch

import cv2
import numpy as np

from seat_monitor.core import Seat, SeatState, assign_people
from seat_monitor.detector import DEFAULT_MODEL, PersonDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    image = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Cannot decode input image")
    empty = np.zeros_like(image)
    seat = Seat("A01", ((0, 0), (1, 0), (1, 1), (0, 1)))
    times = []
    # Block outbound socket connections while importing/loading/running YOLO.
    # This catches accidental model downloads or telemetry in this tested setup.
    with contextlib.ExitStack() as stack:
        for name in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, name, side_effect=AssertionError("Network access during inference")))
        detector = PersonDetector(args.model, confidence=0.4, imgsz=640, threads=4)
        for frame in (empty, image, empty):
            start = time.monotonic()
            detections = detector.detect(frame)
            times.append(round(time.monotonic() - start, 3))
            if frame is image:
                people = detections
                assert people, "No person detected in the supplied image"
            else:
                assert detections == [], "Blank control frame produced a person detection"
        occupied = assign_people([seat], people)
        state = SeatState()
        state.update(occupied[seat.id] > 0, 0)
        assert state.update(occupied[seat.id] > 0, 1) == "OCCUPIED"
        for second in range(2, 11):
            state.update(False, second)
        assert state.state == "EMPTY"
    print(json.dumps({
        "result": "passed", "people_detected": len(people),
        "inference_seconds_on_this_pc": times,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest() if args.model.is_file() else None,
        "checks": ["person image", "blank controls", "OCCUPIED to EMPTY", "network blocked"],
    }, indent=2))


if __name__ == "__main__":
    main()
