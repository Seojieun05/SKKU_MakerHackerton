import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from seat_monitor.__main__ import parser, run
from seat_monitor.camera import FrameSource
from seat_monitor.core import Detection


class CameraTests(unittest.TestCase):
    def test_picamera2_uses_bgr_compatible_format_and_closes(self):
        camera = MagicMock()
        camera.capture_array.return_value = types.SimpleNamespace(size=9)
        module = types.SimpleNamespace(Picamera2=MagicMock(return_value=camera))
        with patch.dict("sys.modules", {"picamera2": module}), patch("seat_monitor.camera.time.sleep"):
            with FrameSource("picamera2", 1280, 720) as source:
                source.read()
        kwargs = camera.create_video_configuration.call_args.kwargs
        self.assertEqual(kwargs["main"]["format"], "RGB888")
        self.assertFalse(kwargs["queue"])
        camera.capture_array.assert_called_once_with("main")
        camera.close.assert_called_once()

    def test_camera_is_closed_when_start_fails(self):
        camera = MagicMock()
        camera.start.side_effect = RuntimeError("camera disconnected")
        module = types.SimpleNamespace(Picamera2=MagicMock(return_value=camera))
        with patch.dict("sys.modules", {"picamera2": module}):
            with self.assertRaises(RuntimeError):
                with FrameSource("picamera2", 1280, 720):
                    pass
        camera.close.assert_called_once()


class RuntimeTests(unittest.TestCase):
    def test_full_loop_and_shutdown_invalidates_status(self):
        frame = types.SimpleNamespace(shape=(720, 1280, 3))
        camera = MagicMock()
        camera.__enter__.return_value = camera
        camera.is_video = True
        camera.read.side_effect = [frame, frame, frame, None]
        observations = iter([0.0, 1.0, 2.0])
        type(camera).observation_time = property(lambda _: next(observations))
        detector = MagicMock()
        detector.detect.return_value = [Detection((0.1, 0.1, 0.4, 0.9), 0.9)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status.json"
            args = parser().parse_args(["run", "--whole-frame", "--status-file", str(output)])
            stream = io.StringIO()
            with patch("seat_monitor.__main__.FrameSource", return_value=camera), patch(
                "seat_monitor.__main__.PersonDetector", return_value=detector
            ), contextlib.redirect_stdout(stream):
                self.assertEqual(run(args), 0)
            messages = [json.loads(line) for line in stream.getvalue().splitlines()]
            self.assertTrue(any(m["seats"][0]["state"] == "OCCUPIED" for m in messages))
            final = json.loads(output.read_text())
            self.assertEqual(final["health"], "end_of_video")
            self.assertEqual(final["seats"][0]["state"], "UNKNOWN")

    def test_missing_model_invalidates_previous_status(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status.json"
            args = parser().parse_args(["run", "--whole-frame", "--model", str(Path(directory) / "missing.pt"), "--status-file", str(output)])
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileNotFoundError):
                run(args)
            final = json.loads(output.read_text())
            self.assertEqual(final["health"], "error")
            self.assertIsNone(final["seats"][0]["has_person"])

    def test_camera_failure_is_never_an_empty_seat(self):
        camera = MagicMock()
        camera.__enter__.side_effect = RuntimeError("unplugged")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "status.json"
            args = parser().parse_args(["run", "--whole-frame", "--status-file", str(output)])
            with patch("seat_monitor.__main__.PersonDetector"), patch(
                "seat_monitor.__main__.FrameSource", return_value=camera
            ), contextlib.redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                run(args)
            final = json.loads(output.read_text())
            self.assertEqual(final["health"], "error")
            self.assertEqual(final["seats"][0]["state"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
