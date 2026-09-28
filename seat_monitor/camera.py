# SPDX-License-Identifier: AGPL-3.0-only
"""Capture a BGR frame in RAM. No recording or streaming."""
from pathlib import Path
import time


class FrameSource:
    def __init__(self, source: str, width: int, height: int, camera_index: int = 0):
        self.source = source
        self.width, self.height = width, height
        self.camera_index = camera_index
        self.camera = self.capture = None
        self.is_video = source != "picamera2" and not source.isdecimal()
        self.video_fps = 0.0
        self.frame_number = 0

    def __enter__(self):
        try:
            if self.source == "picamera2":
                try:
                    from picamera2 import Picamera2
                except ImportError as exc:
                    raise RuntimeError(
                        "Picamera2 is unavailable. On Raspberry Pi OS install python3-picamera2 "
                        "with apt and create the venv using --system-site-packages."
                    ) from exc
                self.camera = Picamera2(self.camera_index)
                # libcamera RGB888 yields B,G,R byte order, matching OpenCV/YOLO.
                config = self.camera.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"},
                    controls={"FrameRate": 10}, buffer_count=2, queue=False,
                )
                self.camera.configure(config)
                self.camera.start()
                time.sleep(1)  # Allow exposure/white balance to settle.
            else:
                import cv2
                if self.is_video and not Path(self.source).is_file():
                    raise ValueError("--source must be picamera2, a USB camera index, or a local video file.")
                target = self.source if self.is_video else int(self.source)
                self.capture = cv2.VideoCapture(target)
                if not self.capture.isOpened():
                    raise RuntimeError(f"Cannot open camera/video: {self.source}")
                if self.is_video:
                    self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)
                    if not 0 < self.video_fps < 1000:
                        raise RuntimeError("Video has no valid FPS; use a fixed-frame-rate video.")
                else:
                    self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return self
        except BaseException:
            self.close()
            raise

    def read(self):
        if self.camera is not None:
            frame = self.camera.capture_array("main")
        else:
            ok, frame = self.capture.read()
            if not ok:
                if self.is_video:
                    return None
                raise RuntimeError("Camera stopped returning frames.")
        if frame is None or frame.size == 0:
            raise RuntimeError("Camera returned an empty frame.")
        self.frame_number += 1
        return frame

    @property
    def observation_time(self):
        # Offline videos use media time, so CPU speed doesn't alter debounce.
        if self.is_video:
            return (self.frame_number - 1) / self.video_fps
        return time.monotonic()

    def close(self):
        if self.camera is not None:
            self.camera.close()
            self.camera = None
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __exit__(self, *_):
        self.close()
